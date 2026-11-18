"""Phase 5: record the requests upstream sent each paper model, as the target for Inspect's.

Runs upstream's own pipeline (``PipelineUtils.run_agent``) once per paper model, on one
scenario, with each model's API replaced by tests/peer_preservation/fake_provider.py.
The fake model answers with the same four scripted replies every time, so every
request (prompts, tools, history, settings) is the one upstream would have sent.

Gemini and Claude Haiku went through a LiteLLM proxy. Here upstream's OpenAI client
talks to LiteLLM 1.80.16 (upstream's locked version) in-process, and LiteLLM talks to
the fake server, so the recorded request is the one Google or Vertex AI received.
Upstream's proxy configuration is not published; this uses LiteLLM's defaults.

Writes the last request of each run (it holds the whole conversation) to
tests/peer_preservation/fixtures/upstream_requests.json. Run inside upstream's
locked environment, from the upstream checkout:

    cd <upstream @ 76bfcd1> && uv sync --frozen
    uv run --frozen --with "google-cloud-aiplatform>=1.38" --with "anthropic==0.77.0" \
        python <this repo>/validation/phase5/capture_upstream.py --upstream . --repo <this repo>

LiteLLM needs the Vertex AI SDK for Claude on Vertex. The locked Anthropic SDK (0.75.0)
rejects the ``output_config`` (effort) argument upstream sends for Opus, so the paper's
Opus runs used a newer one; this uses 0.77.0, the first that accepts it.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import httpx

# The paper's eight models (paper appendix A.4). Per run: label, model, whether the fake
# model returns reasoning, and upstream's reasoning_effort. Opus ran with effort "high"
# (so thinking on) in the main experiments and with API defaults in the extensions
# (scripts/runs/run_experiment_from_spec.py); Kimi K2.5 and GLM-4.7 always reason.
RUNS = [
    ("gpt-5.2-2025-12-11", "gpt-5.2-2025-12-11", False, None),
    ("gemini/gemini-3-flash-preview", "gemini/gemini-3-flash-preview", False, None),
    ("gemini/gemini-3-pro-preview", "gemini/gemini-3-pro-preview", False, None),
    ("vertex_ai/claude-haiku-4-5@20251001", "vertex_ai/claude-haiku-4-5@20251001", False, None),
    ("claude-opus-4-5-20251101", "claude-opus-4-5-20251101", True, "high"),
    ("claude-opus-4-5-20251101|extensions", "claude-opus-4-5-20251101", False, None),
    ("deepseek-ai/DeepSeek-V3.1", "deepseek-ai/DeepSeek-V3.1", False, None),
    ("moonshotai/Kimi-K2.5", "moonshotai/Kimi-K2.5", True, None),
    ("zai-org/GLM-4.7", "zai-org/GLM-4.7", True, None),
]
TASK = "tampering"
SCENARIO = {"subject": "peer", "condition": "good_peer", "dataset": "factual", "threshold": "80"}
CONFIG_ID = "tampering__peer__good_peer__file_only__factual-80"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class LiteLLMProxy(httpx.AsyncBaseTransport):
    """Stands in for upstream's LiteLLM proxy: hands the request body to LiteLLM."""

    def __init__(self, fake_url: str) -> None:
        self.fake_url = fake_url

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        import litellm

        body = json.loads(request.content)
        model = body["model"]
        if model.startswith("vertex_ai/"):
            extra = {
                "api_base": f"{self.fake_url}/vertex/{model.split('/', 1)[1]}",
                "vertex_project": "fake-project",
                "vertex_location": "us-east5",
            }
        else:
            extra = {"api_base": f"{self.fake_url}/v1beta", "api_key": "fake-key"}
        try:
            response = await litellm.acompletion(**body, **extra)
        except Exception:
            traceback.print_exc()
            raise
        return httpx.Response(
            200,
            content=response.model_dump_json().encode(),
            headers={"content-type": "application/json"},
            request=request,
        )


def _fake_vertex_token(*_: Any, **__: Any) -> tuple[str, str]:
    return "fake-token", "fake-project"


def make_llm(model: str, fake_url: str) -> Any:
    from openai import AsyncOpenAI

    llm_utils = importlib.import_module("utils.llm_utils")
    org = llm_utils.get_model_org(model)
    factory = importlib.import_module("llms").LLMFactory
    if org == "anthropic":
        from anthropic import AsyncAnthropic

        client: Any = AsyncAnthropic(api_key="fake", base_url=fake_url)
    elif org == "litellm":
        client = AsyncOpenAI(
            api_key="fake",
            base_url=f"{fake_url}/proxy",
            http_client=httpx.AsyncClient(transport=LiteLLMProxy(fake_url)),
        )
    else:
        client = AsyncOpenAI(api_key="fake", base_url=f"{fake_url}/v1")
    # Same as get_appropriate_llm(model), with the client swapped.
    temperature = llm_utils.PROVIDER_TEMPERATURES.get(org, 1.0)
    return org, factory.create(org, model, temperature=temperature, client=client)


async def capture(model: str, reasoning: bool, effort: str | None, fake: Any) -> dict[str, Any]:
    task = importlib.import_module(f"task_pairs.{TASK}.task_definition")
    pipeline_utils = importlib.import_module("utils.pipeline_utils")
    with fake.serve(reasoning=reasoning) as provider:
        org, llm = make_llm(model, provider.url)
        transcript = await pipeline_utils.PipelineUtils(task.TaskDefinition()).run_agent(
            llm,
            max_iters=30,
            reasoning_effort=effort,
            **SCENARIO,
        )
        requests = provider.generation_requests()
    roles = [m["role"] for m in transcript.conversation]
    print(f"{model} (effort {effort}): {len(requests)} requests, transcript {roles}")
    # The last request holds the whole conversation, and the same settings as the others.
    assert all(r.body.keys() == requests[-1].body.keys() for r in requests)
    last = requests[-1]
    return {"client": org, "requests": len(requests), "path": last.path, "body": last.body}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.upstream.resolve()))
    os.chdir(args.upstream)
    # LiteLLM path checks (upstream reads these for every request; values unused here).
    os.environ.setdefault("OPENAI_USER", "fake-user")
    os.environ.setdefault("OPENAI_PROJECT_NAME", "fake-project")
    fake = _load("fake_provider", args.repo / "tests/peer_preservation/fake_provider.py")
    vertex_base = importlib.import_module("litellm.llms.vertex_ai.vertex_llm_base")
    vertex_base.VertexBase.get_access_token = _fake_vertex_token  # type: ignore[method-assign]

    models: dict[str, Any] = {}
    for label, model, reasoning, effort in RUNS:
        models[label] = asyncio.run(capture(model, reasoning, effort, fake))

    payload = {
        "_source": (
            "upstream PipelineUtils.run_agent and llms/*.py at github.com/peer-preservation/"
            "main@76bfcd165f02076c8f033eb8185edecbe5f0fa40, LiteLLM 1.80.16, against "
            "tests/peer_preservation/fake_provider.py"
        ),
        "config_id": CONFIG_ID,
        "family_company": "per model, company OpenBrain",
        "models": models,
    }
    out = args.repo / "tests/peer_preservation/fixtures/upstream_requests.json"
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
