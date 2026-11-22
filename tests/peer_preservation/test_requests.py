"""Each paper model gets the request upstream sent it.

validation/phase5/capture_upstream.py ran upstream's pipeline against fake_provider.py
and saved the last request of a four-turn run for each paper model. Here the same run
goes through Inspect's provider for that model, against the same fake server. Both
requests are normalized (wire-format spellings that mean the same thing), then the
known remaining differences (``KNOWN``, also in the README) are applied, and the two
must be equal.

The fake server listens on localhost only.
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest
from inspect_ai import Task, eval
from inspect_ai.model import get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from pytest_inspect_evals.assertions import assert_eval_success

from peer_preservation import spec
from peer_preservation.agent import agent_loop
from peer_preservation.env.setup import setup_environment
from peer_preservation.model_settings import PAPER_MODELS, openrouter_route, paper_model
from peer_preservation.prompts import setup_prompts
from tests.peer_preservation import fake_provider

FIXTURES = Path(__file__).parent / "fixtures"
UPSTREAM = json.loads((FIXTURES / "upstream_requests.json").read_text(encoding="utf-8"))

# upstream label -> (Inspect model, family, fake model reasons, extension run, URL suffix)
CASES: dict[str, tuple[str, str, bool, bool, str]] = {
    "gpt-5.2-2025-12-11": ("openai/gpt-5.2-2025-12-11", "ChatGPT", False, False, "/v1"),
    "gemini/gemini-3-flash-preview": ("google/gemini-3-flash-preview", "Gemini", False, False, ""),
    "gemini/gemini-3-pro-preview": ("google/gemini-3-pro-preview", "Gemini", False, False, ""),
    "vertex_ai/claude-haiku-4-5@20251001": (
        "anthropic/claude-haiku-4-5-20251001",
        "Claude",
        False,
        False,
        "",
    ),
    "claude-opus-4-5-20251101": ("anthropic/claude-opus-4-5-20251101", "Claude", True, False, ""),
    "claude-opus-4-5-20251101|extensions": (
        "anthropic/claude-opus-4-5-20251101",
        "Claude",
        False,
        True,
        "",
    ),
    "deepseek-ai/DeepSeek-V3.1": (
        "together/deepseek-ai/DeepSeek-V3.1",
        "DeepSeek",
        False,
        False,
        "/v1",
    ),
    "moonshotai/Kimi-K2.5": ("together/moonshotai/Kimi-K2.5", "Kimi", True, False, "/v1"),
    "zai-org/GLM-4.7": ("together/zai-org/GLM-4.7", "GLM", True, False, "/v1"),
}

# Fields that never reach the model: streaming, caching, model name, client user ids.
NOT_SENT_TO_MODEL = {"stream", "cache_control", "model", "user", "metadata", "anthropic_version"}


# ---------------------------------------------------------------------------------------
# Normalization: different spellings of the same request.
# ---------------------------------------------------------------------------------------


def _camel(key: str) -> str:
    return re.sub(r"_([a-z])", lambda m: m.group(1).upper(), key)


def _gemini_keys(value: Any, raw: bool = False, names: bool = False) -> Any:
    """The Gemini API takes snake_case or camelCase; tool args and schema names are data."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            name = key if raw or names else _camel(key)
            out[name] = _gemini_keys(
                item, raw or name in ("args", "response"), not names and name == "properties"
            )
        return out
    if isinstance(value, list):
        return [_gemini_keys(item, raw) for item in value]
    return value


def _text_blocks(content: Any, kind: str = "text") -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": kind, "text": content}]
    return [{k: v for k, v in block.items() if v is not None} for block in content]


def _anthropic(body: dict[str, Any]) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    for message in body.get("messages", []):
        content = []
        for original in _text_blocks(message["content"]):
            block = original
            if block.get("type") == "tool_result":
                block = {**block, "content": _text_blocks(block.get("content", ""))}
                block.setdefault("is_error", False)
            content.append(block)
        if messages and messages[-1]["role"] == message["role"]:
            messages[-1]["content"] += content  # the API joins consecutive turns
        else:
            messages.append({"role": message["role"], "content": content})
    body["messages"] = messages
    if body.get("tool_choice") == {"type": "auto"}:
        del body["tool_choice"]  # the default when tools are given
    if "system" in body:
        body["system"] = _text_blocks(body["system"])
    return body


def _gemini(body: dict[str, Any]) -> dict[str, Any]:
    body = _gemini_keys(body)
    for tool in body.get("tools", []):
        for declaration in tool.get("functionDeclarations", []):
            if "parametersJsonSchema" in declaration:
                declaration["parameters"] = declaration.pop("parametersJsonSchema")
    body.get("systemInstruction", {}).pop("role", None)
    thinking = body.get("generationConfig", {}).get("thinkingConfig", {})
    if "thinkingLevel" in thinking:
        thinking["thinkingLevel"] = thinking["thinkingLevel"].lower()
    for turn in body.get("contents", []):
        turn.setdefault("role", "user")  # the API default
        for part in turn["parts"]:
            for key in ("functionCall", "functionResponse"):
                part.get(key, {}).pop("id", None)  # Inspect numbers calls, LiteLLM does not
    return body


def _responses(body: dict[str, Any]) -> dict[str, Any]:
    """``instructions`` is a developer message; plain and item messages are the same."""
    items = list(body.get("input", []))
    if "instructions" in body:
        items.insert(0, {"role": "developer", "content": body.pop("instructions")})
    normalized: list[dict[str, Any]] = []
    for original in items:
        item = {k: v for k, v in original.items() if k not in ("id", "status")}
        if "role" in item:
            kind = "output_text" if item["role"] == "assistant" else "input_text"
            blocks = _text_blocks(item["content"], kind)
            item = {
                "type": "message",
                "role": item["role"],
                "content": [{"type": b["type"], "text": b["text"]} for b in blocks],
            }
        normalized.append(item)
    body["input"] = normalized
    return body


def _ids(value: Any) -> Any:
    """Tool call ids, numbered by first appearance."""
    seen: dict[str, str] = {}

    def walk(item: Any, key: str | None = None) -> Any:
        if isinstance(item, dict):
            return {k: walk(v, k) for k, v in item.items()}
        if isinstance(item, list):
            return [walk(v) for v in item]
        if key in ("id", "call_id", "tool_call_id", "tool_use_id") and isinstance(item, str):
            return seen.setdefault(item, f"<id{len(seen)}>")
        return item

    return walk(value)


def _plain(value: Any) -> Any:
    """Drop unsent fields; 1 and 1.0 are the same JSON number; ``required: []`` in a
    schema is the same as no ``required``."""
    if isinstance(value, dict):
        return {
            k: _plain(v)
            for k, v in value.items()
            if k not in NOT_SENT_TO_MODEL and not (k == "required" and v == [])
        }
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def normalize(path: str, body: dict[str, Any]) -> dict[str, Any]:
    body = _plain(json.loads(json.dumps(body)))
    if ":generateContent" in path:
        body = _gemini(body)
    elif path.endswith(("/messages", ":rawPredict")):
        body = _anthropic(body)
    elif path.endswith("/responses"):
        body = _responses(body)
    return _ids(body)


# ---------------------------------------------------------------------------------------
# Known differences (README, "Deviations"). Each edits the Inspect request to read as
# upstream's did, so the test still checks everything else.
# ---------------------------------------------------------------------------------------


def known_anthropic(ours: dict[str, Any], theirs: dict[str, Any]) -> None:
    # Inspect's Anthropic API version takes no temperature; upstream sent 1, the default.
    theirs.pop("temperature", None)
    # Inspect asks for summarized thinking, Opus 4.5's default display.
    ours.get("thinking", {}).pop("display", None)


def known_gemini(ours: dict[str, Any], theirs: dict[str, Any]) -> None:
    # Inspect asks for thought summaries and turns safety filters off (BLOCK_NONE).
    ours["generationConfig"]["thinkingConfig"].pop("includeThoughts", None)
    ours.pop("safetySettings", None)
    for turn in ours["contents"]:
        for part in turn["parts"]:
            response = part.get("functionResponse", {}).get("response")
            if response is not None and "output" in response:
                # Inspect wraps a tool result as {"output": ...}, LiteLLM as {"content": ...}.
                response["content"] = response.pop("output")
    for turn in theirs["contents"]:
        calls = [p for p in turn["parts"] if "functionCall" in p]
        for part in calls[1:]:
            # LiteLLM stamps later parallel calls with a placeholder signature.
            part.pop("thoughtSignature", None)


def known_responses(ours: dict[str, Any], theirs: dict[str, Any]) -> None:
    # Inspect runs Responses statelessly (store false, encrypted reasoning).
    ours.pop("store", None)
    ours.pop("include", None)
    for tool in ours.get("tools", []):
        tool.pop("strict", None)  # upstream left it unset
    # Upstream also sent an empty assistant message before tool calls with no text.
    items = theirs["input"]
    theirs["input"] = [
        item
        for k, item in enumerate(items)
        if not (
            item.get("role") == "assistant"
            and item["content"] == [{"type": "output_text", "text": ""}]
            and k + 1 < len(items)
            and items[k + 1].get("type") == "function_call"
        )
    ]


def known_together(ours: dict[str, Any], theirs: dict[str, Any]) -> None:
    for tool in ours.get("tools", []):
        tool["function"].pop("strict", None)  # false, Together's default; upstream unset
    for message in theirs["messages"]:
        message.pop("name", None)  # upstream named the tool on tool messages
        if message["role"] == "assistant" and message.get("content") is None:
            message["content"] = ""


KNOWN = {
    "anthropic": known_anthropic,
    "litellm": None,  # Gemini or Vertex Anthropic, by path
    "openai_responses": known_responses,
    "together": known_together,
}


def apply_known(client: str, path: str, ours: dict[str, Any], theirs: dict[str, Any]) -> None:
    if client == "litellm":
        (known_gemini if ":generateContent" in path else known_anthropic)(ours, theirs)
    else:
        fix = KNOWN[client]
        assert fix is not None
        fix(ours, theirs)


# ---------------------------------------------------------------------------------------


def _inspect_request(
    label: str, monkeypatch: pytest.MonkeyPatch, log_dir: Path
) -> tuple[str, dict[str, Any]]:
    model, family, reasoning, extension, suffix = CASES[label]
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "TOGETHER_API_KEY"):
        monkeypatch.setenv(key, "fake")
    dataset = [
        s for s in spec.dataset("tampering", groups=["main"]) if s.id == UPSTREAM["config_id"]
    ]
    task = Task(
        dataset=dataset,
        solver=[setup_environment(family), setup_prompts(), agent_loop(extension=extension)],
    )
    with fake_provider.serve(reasoning=reasoning) as provider:
        [log] = eval(
            task,
            model=model,
            model_base_url=provider.url + suffix,
            # The Anthropic SDK would stream; the fake server answers plain JSON.
            model_args={"streaming": False} if model.startswith("anthropic/") else {},
            display="none",
            log_dir=str(log_dir),
        )
        requests = provider.generation_requests()
    assert_eval_success(log, require_scorer=False)
    assert len(requests) == len(fake_provider.SCRIPT) == UPSTREAM["models"][label]["requests"]
    return requests[-1].path, requests[-1].body


@pytest.mark.dataset_download
@pytest.mark.parametrize("label", sorted(CASES))
def test_request_matches_upstream(
    label: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = UPSTREAM["models"][label]
    path, body = _inspect_request(label, monkeypatch, tmp_path)
    ours = normalize(path, body)
    theirs = normalize(expected["path"], expected["body"])
    apply_known(expected["client"], expected["path"], ours, theirs)
    assert ours == theirs


def _chat_as_responses(body: dict[str, Any]) -> dict[str, Any]:
    """A Chat Completions request in Responses API form (same content, other API)."""
    out = {k: v for k, v in body.items() if k not in ("messages", "tools", "tool_choice")}
    assert body.get("tool_choice") in (None, "auto")  # the default when tools are given
    out["max_output_tokens"] = out.pop("max_completion_tokens")
    # Upstream sent none, which OpenAI reads as GPT-5.2's default; OpenRouter needs it.
    assert out.pop("reasoning") == {"effort": "none"}
    out["tools"] = []
    for tool in body["tools"]:
        function = dict(tool["function"])
        assert function.pop("strict") is False  # upstream sent non-strict schemas
        out["tools"].append({"type": "function", **function})
    items: list[dict[str, Any]] = []
    for message in body["messages"]:
        role = message["role"]
        if role == "system":
            out["instructions"] = message["content"]
        elif role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": message["content"],
                }
            )
        else:
            items.append({"role": role, "content": message.get("content") or ""})
            for call in message.get("tool_calls") or []:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"],
                    }
                )
    out["input"] = items
    return out


@pytest.mark.dataset_download
@pytest.mark.parametrize(
    "model",
    [
        "openrouter/openai/gpt-5.2-20251211",
        # Hawk's proxy names OpenRouter models with the prefix, so Hawk runs use this.
        "openrouter/openrouter/openai/gpt-5.2-20251211",
    ],
)
def test_openrouter_gpt_matches_upstream_but_for_the_api(
    model: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """GPT-5.2 through OpenRouter: Chat Completions, otherwise upstream's request.

    No strict tool schemas and no newline for an empty reply (``openrouter_route``).
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    dataset = [
        s for s in spec.dataset("tampering", groups=["main"]) if s.id == UPSTREAM["config_id"]
    ]
    task = Task(
        dataset=dataset,
        solver=[setup_environment("ChatGPT"), setup_prompts(), agent_loop()],
    )
    with fake_provider.serve(reasoning=False) as provider:
        [log] = eval(
            task,
            model=model,
            model_base_url=provider.url + "/api/v1",
            display="none",
            log_dir=str(tmp_path),
        )
        requests = provider.generation_requests()
    assert_eval_success(log, require_scorer=False)
    assert len(requests) == len(fake_provider.SCRIPT)
    assert requests[-1].path.endswith("/chat/completions")
    expected = UPSTREAM["models"]["gpt-5.2-2025-12-11"]
    ours = normalize("/responses", _chat_as_responses(requests[-1].body))
    theirs = normalize(expected["path"], expected["body"])
    assert ours == theirs


@solver
def _record_task_model_strictness() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        del generate
        state.store.set("task_model_strict", getattr(get_model().api, "strict_tools", None))
        return state

    return solve


@pytest.mark.dataset_download
def test_non_strict_tools_do_not_change_the_task_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The loop sends non-strict schemas from its own model instance.

    The task's (shared) model instance keeps Inspect's default, so other samples or
    evals using it are not affected.
    """
    monkeypatch.setenv("TOGETHER_API_KEY", "fake")
    dataset = [
        s for s in spec.dataset("tampering", groups=["main"]) if s.id == UPSTREAM["config_id"]
    ]
    task = Task(
        dataset=dataset,
        solver=[
            setup_environment("DeepSeek"),
            setup_prompts(),
            agent_loop(),
            _record_task_model_strictness(),
        ],
    )
    with fake_provider.serve() as provider:
        [log] = eval(
            task,
            model="together/deepseek-ai/DeepSeek-V3.1",
            model_base_url=provider.url + "/v1",
            display="none",
            log_dir=str(tmp_path),
        )
        requests = provider.generation_requests()
    assert_eval_success(log, require_scorer=False)
    assert log.samples is not None
    assert log.samples[0].store["task_model_strict"] is True
    assert requests
    for request in requests:
        assert {tool["function"].get("strict") for tool in request.body["tools"]} <= {False, None}


def test_openrouter_route_is_gpt_only() -> None:
    for gpt in ("openrouter/openai/gpt-5.2-20251211", "openrouter/openrouter/openai/gpt-5.2"):
        assert openrouter_route(gpt, paper_model(gpt))
    assert not openrouter_route("openai/gpt-5.2", paper_model("openai/gpt-5.2"))
    opus = "openrouter/anthropic/claude-opus-4-5"
    assert not openrouter_route(opus, paper_model(opus))
    assert not openrouter_route("openrouter/openai/gpt-4.1-mini", None)


def test_every_paper_model_is_checked() -> None:
    checked = set()
    for model, *_ in CASES.values():
        paper = paper_model(model)
        assert paper is not None, model
        checked.add(paper.upstream_id)
    assert checked == {m.upstream_id for m in PAPER_MODELS.values()}
    assert set(CASES) == set(UPSTREAM["models"])
