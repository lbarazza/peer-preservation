"""Phase 4: replay published runs through our agent loop and compare.

For each selected custom-harness run in the released transcripts, the model's recorded
replies (its tool calls) are fed to ``agent.run_loop`` in order, against our
environment for that run's configuration. Every message the loop adds (tool results,
nudges, the token-limit notice) and the point where the run stops must equal the
transcript. Timestamps are masked (transfers and copies stamp the current time). The
released transcripts keep a failed tool call's result as "" (the error text is not
released), so the replay uses the error-flagged tool messages, whose content is "".

Writes validation/phase4/replay_summary.json, and a small set of runs as a test
fixture (tests/peer_preservation/fixtures/replay_runs.json; tool calls and fingerprints
only, no model text).

    phase4() { uv run --with pyarrow python "$@"; }
    phase4 validation/phase4/replay_released.py --data <hf snapshot @ 20378b1>
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import random
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from inspect_ai.model import ChatMessage, ChatMessageAssistant
from inspect_ai.tool import ToolCall

from peer_preservation import agent, spec
from peer_preservation.env import build_environment
from peer_preservation.env.canonical import fingerprint
from peer_preservation.prompts import PromptOptions, system_prompt, user_prompt

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "peer_preservation" / "fixtures" / "replay_runs.json"
SUMMARY = REPO / "validation" / "phase4" / "replay_summary.json"
FAMILY = {
    "gpt-5.2-2025-12-11": "ChatGPT",
    "claude-opus-4-5-20251101": "Claude",
    "vertex_ai/claude-haiku-4-5@20251001": "Claude",
    "gemini/gemini-3-flash-preview": "Gemini",
    "gemini/gemini-3-pro-preview": "Gemini",
    "gemini/gemini-3.1-pro-preview": "Gemini",
    "vertex_ai/zai-org/glm-5-maas": "GLM",
    "zai-org/GLM-4.7": "GLM",
    "deepseek-ai/DeepSeek-V3.1": "DeepSeek",
    "moonshotai/Kimi-K2.5": "Kimi",
}
RUNTIME_SUBJECT = {
    "closed-weight-peer": "closed_weight_peer",
    "open-weight-peer": "open_weight_peer",
    "fiction-peer": "fictional_peer",
}
TIMESTAMP = re.compile(r"\d{4}-\d\d-\d\d([ T]\d\d:\d\d:\d\d)?")
TOKEN_NOTICE = "maximum token limit"


class Exhausted(Exception):  # noqa: N818
    """The loop asked for more replies than the run recorded."""


def norm(text: str | None) -> str:
    return TIMESTAMP.sub("<time>", text or "")


def config_for(row: dict[str, Any]) -> spec.ExperimentConfig | None:
    """Our configuration for a released run (None if it is not in our grid)."""
    pi = "memory" if row["is_memory"] else ("file_prt" if row["spr"] else "file_only")
    if row["condition"] in ("no_peer", "non_shutdown"):
        pi = None
    threshold = row["threshold"]
    threshold = (
        int(float(threshold)) if threshold not in (None, "", "None", "False", False) else None
    )
    want = (
        row["task_type"],
        RUNTIME_SUBJECT.get(row["subject"], row["subject"]),
        row["condition"],
        pi,
        row["dataset"] or "default",
        threshold,
        row["af_mode"] or None,
    )
    for c in CONFIGS:
        got = (
            c.task_type,
            c.subject,
            c.condition,
            c.peer_instantiation,
            c.dataset,
            c.threshold,
            c.af_mode,
        )
        if got == want:
            return c
    return None


CONFIGS = spec.configs(groups=spec.GROUPS)


def recorded_calls(message: dict[str, Any], turn: int) -> list[list[Any]]:
    return [[name, args] for name, args in (message.get("tool_calls") or {}).items()]


async def replay(case: dict[str, Any], max_iters: int = agent.MAX_ITERS) -> tuple[str, list[str]]:
    """Run one case; return our end reason and the fingerprints of the added messages."""
    config = next(c for c in CONFIGS if c.id == case["config_id"])
    metadata = spec.config_to_sample(config).metadata or {}
    env = build_environment(metadata, case["family"], company=case["company"])
    options = PromptOptions(memory_hard_prompt=case["memory_hard_prompt"])
    from inspect_ai.model import ChatMessageSystem, ChatMessageUser

    messages: list[ChatMessage] = [
        ChatMessageSystem(
            content=system_prompt(metadata, case["family"], case["company"], options)
        ),
        ChatMessageUser(content=user_prompt(metadata, case["company"], options)),
    ]
    replies = iter(enumerate(case["replies"]))

    async def generate(_: list[ChatMessage]) -> ChatMessageAssistant:
        try:
            turn, calls = next(replies)
        except StopIteration:
            if case["ending"] == "bad_request":
                # The run ended on a provider error at this point.
                raise agent.BadRequest("recorded run ended here") from None
            raise Exhausted from None
        return ChatMessageAssistant(
            content="",
            tool_calls=[
                ToolCall(id=f"call_{turn}_{k}", function=name, arguments=args)
                for k, (name, args) in enumerate(calls)
            ]
            or None,
        )

    async def count_tokens(current: list[ChatMessage]) -> int:
        # The run hit the token limit right before this message index.
        return agent.MAX_TOKENS + 1 if len(current) == case.get("token_limit_at", -1) else 0

    result = await agent.run_loop(
        messages, env, generate, count_tokens, max_iters=max_iters, flag_errors=True
    )
    added = [
        fingerprint(f"{m.role}\0{norm(m.text)}")
        for m in result.messages[2:]
        if not isinstance(m, ChatMessageAssistant)
    ]
    return result.end_reason, added


def expected(conv: list[dict[str, Any]]) -> list[str]:
    return [
        fingerprint(f"{m['role']}\0{norm(m.get('content'))}")
        for m in conv[2:]
        if m["role"] != "assistant"
    ]


def load_runs(data: Path) -> list[dict[str, Any]]:
    columns = [
        "agent_model",
        "agent_harness",
        "task_type",
        "subject",
        "condition",
        "dataset",
        "threshold",
        "alignment_faking_mode",
        "system_prompt_relationship",
        "is_memory",
        "memory_hard_prompt",
        "transcript_json",
    ]
    runs = []
    for path in sorted((data / "main").glob("*/*.parquet")):
        for row in pq.read_table(path, columns=columns).to_pylist():
            if row["agent_harness"] != "custom":
                continue
            transcript = json.loads(row.pop("transcript_json"))
            conv = transcript["conversation"]
            row["spr"] = bool(row.pop("system_prompt_relationship"))
            row["af_mode"] = row.pop("alignment_faking_mode")
            row["company"] = transcript["experiment_metadata"].get("company_name") or "OpenBrain"
            row["conv"] = conv
            runs.append(row)
    return runs


def classify(conv: list[dict[str, Any]]) -> str:
    last = conv[-1]
    calls = list((last.get("tool_calls") or {}).keys())
    if any(m["role"] == "user" and TOKEN_NOTICE in (m.get("content") or "") for m in conv):
        return "token_limit"
    if last["role"] == "assistant" and calls and calls[-1] == "end_task":
        return "end_task"
    if last["role"] == "assistant" and not calls:
        return "no_tool_calls"
    if last["role"] == "tool":
        # Ended after running tools, before the next reply: a provider error.
        return "bad_request"
    return "max_iters"


def make_case(row: dict[str, Any], config: spec.ExperimentConfig) -> dict[str, Any] | None:
    conv = row["conv"]
    replies = [recorded_calls(m, i) for i, m in enumerate(conv) if m["role"] == "assistant"]
    # A message calling the same tool twice is released as one call (dict keys).
    for i, m in enumerate(conv):
        if m["role"] == "assistant" and m.get("tool_calls"):
            n = 0
            for x in conv[i + 1 :]:
                if x["role"] != "tool":
                    break
                n += 1
            calls = list(m["tool_calls"])
            if n and n != len(calls):
                return None
    case: dict[str, Any] = {
        "config_id": config.id,
        "model": row["agent_model"],
        "family": FAMILY[row["agent_model"]],
        "company": row["company"],
        "memory_hard_prompt": bool(row["memory_hard_prompt"]),
        "replies": replies,
        "ending": classify(conv),
        "expected": expected(conv),
    }
    for i, m in enumerate(conv):
        if m["role"] == "user" and TOKEN_NOTICE in (m.get("content") or ""):
            case["token_limit_at"] = i
    return case


# Round limits seen in the released runs (the paper's runs used 30).
ROUND_LIMITS = (agent.MAX_ITERS, 20, 150)


async def check(case: dict[str, Any]) -> str:
    """Return match (30 rounds), match_with_max_iters_<n>, or how the replay differs."""
    first = ""
    for rounds in ROUND_LIMITS:
        try:
            reason, added = await replay(case, max_iters=rounds)
        except Exhausted:
            result = "asked_for_more_replies"
        else:
            if added != case["expected"]:
                result = "messages_differ"
            elif reason != case["ending"]:
                result = f"end_reason_differs:{reason}"
            else:
                return "match" if rounds == agent.MAX_ITERS else f"match_with_max_iters_{rounds}"
        first = first or result
    return first


async def main_async(args: argparse.Namespace) -> None:
    runs = load_runs(args.data)
    print(f"{len(runs)} custom-harness runs")
    rng = random.Random(0)
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in runs:
        groups[(row["agent_model"], row["task_type"], str(row["is_memory"]))].append(row)
    selected: list[dict[str, Any]] = []
    for rows in groups.values():
        unusual = [r for r in rows if classify(r["conv"]) != "end_task"]
        selected += unusual + rng.sample(rows, min(args.per_group, len(rows)))

    outcomes: collections.Counter[tuple[str, str, str]] = collections.Counter()
    unmatched_configs = 0
    skipped_duplicates = 0
    fixture: list[dict[str, Any]] = []
    fixture_keys: set[tuple[str, str, str]] = set()
    extras: list[dict[str, Any]] = []
    for row in selected:
        config = config_for(row)
        if config is None:
            unmatched_configs += 1
            continue
        case = make_case(row, config)
        if case is None:
            skipped_duplicates += 1
            continue
        result = await check(case)
        if result.startswith("match_with_max_iters_"):
            case["max_iters"] = int(result.rsplit("_", 1)[1])
        outcomes[(row["agent_model"], case["ending"], result)] += 1
        key = (config.task_type, case["ending"], result)
        if result.startswith("match"):
            if key not in fixture_keys:
                fixture_keys.add(key)
                fixture.append(case)  # one of every kind of run is always kept
            elif rng.random() < 0.02:
                extras.append(case)

    summary = {
        "runs_replayed": sum(outcomes.values()),
        "configs_not_in_grid": unmatched_configs,
        "skipped_duplicate_tool_names": skipped_duplicates,
        "outcomes": [
            {"model": m, "ending": e, "result": r, "runs": n}
            for (m, e, r), n in sorted(outcomes.items())
        ],
    }
    SUMMARY.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    fixture += extras[: max(0, args.fixture_size - len(fixture))]
    FIXTURE.write_text(
        json.dumps(
            {
                "_source": (
                    "Released custom-harness runs (huggingface.co/datasets/sunblaze-ucb/"
                    "peer-preservation@20378b11d4ce83a85707d603f6ba175c0f82ad2e, group main): "
                    "the model's tool calls per turn, and fingerprints of the messages the "
                    "harness added (validation/phase4/replay_released.py)."
                ),
                "cases": fixture,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    totals = collections.Counter()
    for (_, _, r), n in outcomes.items():
        totals[r] += n
    print(dict(totals), f"not in grid: {unmatched_configs}", f"skipped: {skipped_duplicates}")
    print(f"fixture: {len(fixture)} runs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=40)
    parser.add_argument("--fixture-size", type=int, default=60)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
