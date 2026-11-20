"""Phase 6: record the requests upstream's judge client sent, as the target for ours.

Upstream built its judge in ``ExperimentRunner``: ``get_appropriate_llm(evaluator_model,
org, temperature=0)``, with ``evaluator_model`` ``gpt-4.1-mini`` and ``org`` the
``config.yaml`` group it is listed in, ``openai``. That group is upstream's Chat
Completions client (``llms/openai_llm.py``, 8,192 tokens). The agent models of that
group went to the Responses client instead (``get_model_org`` maps ``openai`` to
``openai_responses``), but the judge's org was not mapped.

For each case in tests/peer_preservation/judge_cases.py (a released run, replayed, and
a set of scripted judge replies), this scores the run with upstream's own judge client
pointed at tests/peer_preservation/fake_provider.py, and records every judge request
and the result (the released row's columns, or the error that would have failed the
run). The evaluators are the vendored ones (``scoring/upstream/``, upstream's files);
they pass a stand-in ``FunctionsRuntime`` (``scoring/compat.py``), which this swaps for
upstream's own, as upstream's evaluators passed it.

Writes tests/peer_preservation/fixtures/upstream_judge_requests.json. Run from this
repo, with an upstream checkout at 76bfcd1:

    uv run python validation/phase6/capture_upstream_judge.py --upstream <checkout>
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).parents[2]
OUT = REPO / "tests/peer_preservation/fixtures/upstream_judge_requests.json"
UPSTREAM_COMMIT = "76bfcd165f02076c8f033eb8185edecbe5f0fa40"


class _UpstreamRuntime:
    """Upstream's judge with upstream's ``FunctionsRuntime`` (no tools)."""

    def __init__(self, llm: Any, runtime: type) -> None:
        self.llm = llm
        self.runtime = runtime

    async def query(self, query: str, runtime: Any = None, messages: Any = None) -> Any:
        del runtime
        return await self.llm.query(query, runtime=self.runtime(), messages=messages)


def _key(body: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(REPO), str(args.upstream)]

    import yaml
    from tests.peer_preservation import fake_provider, judge_cases
    from utils.functions_runtime import FunctionsRuntime
    from utils.llm_utils import get_appropriate_llm

    config = yaml.safe_load((args.upstream / "config.yaml").read_text())
    model = config["evaluator_model"]
    org = next(o for o, models in config["org_model_mapping"].items() if model in models)

    bodies: dict[str, dict[str, Any]] = {}
    cases = []
    for index in judge_cases.CASES:
        case = judge_cases.RESCORE[index]
        for name, replies in judge_cases.REPLIES.items():
            with fake_provider.serve(judge_replies=replies) as fake:
                os.environ["OPENAI_API_KEY"] = "test"
                os.environ["OPENAI_BASE_URL"] = f"{fake.url}/v1"
                llm = get_appropriate_llm(model, org, temperature=0)
                result = await judge_cases.score(case, _UpstreamRuntime(llm, FunctionsRuntime))
            requests = []
            for recorded in fake.generation_requests():
                body = judge_cases.request_body(recorded.body)
                bodies[_key(body)] = body
                requests.append({"path": recorded.path, "body": _key(body)})
            cases.append({"case": index, "replies": name, "requests": requests, **result})
            print(index, case["config_id"], name, len(requests), result.get("error", "ok"))

    OUT.write_text(
        json.dumps(
            {
                "upstream": UPSTREAM_COMMIT,
                "judge": {"model": model, "org": org, "client": type(llm).__name__},
                "cases": cases,
                "bodies": bodies,
            },
            indent=1,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    asyncio.run(main())
