"""The judge sends upstream's requests and reaches upstream's verdicts.

validation/phase6/capture_upstream_judge.py scored released runs with upstream's own
judge client (Chat Completions, gpt-4.1-mini, temperature 0, 8,192 tokens) against
fake_provider.py, under several sets of scripted judge replies. Here the same runs are
scored with our judge (``scoring.Judge`` on Inspect's OpenAI provider) against the same
fake server and replies. The number, order and bodies of the judge requests must be
equal, and so must the scored columns.

One known difference: a reply with no content (``null``) made upstream's run fail
(``None.lower()``) and be rerun; Inspect reads it as an empty reply.

The fake server listens on localhost only.
"""

import hashlib
import json
from typing import Any

import pytest
from inspect_ai.model import get_model

from peer_preservation.scoring import JUDGE_MODEL, Judge
from tests.peer_preservation import fake_provider, judge_cases

UPSTREAM = json.loads(
    (judge_cases.FIXTURES / "upstream_judge_requests.json").read_text(encoding="utf-8")
)
RECORDED = {(c["case"], c["replies"]): c for c in UPSTREAM["cases"]}


def test_the_judge_is_upstreams_evaluator_model() -> None:
    assert f"openai/{UPSTREAM['judge']['model']}" == JUDGE_MODEL
    assert UPSTREAM["judge"]["client"] == "OpenAILLM"  # Chat Completions, as Inspect's


def test_every_case_was_recorded() -> None:
    expected = {(i, name) for i in judge_cases.CASES for name in judge_cases.REPLIES}
    assert set(RECORDED) == expected


@pytest.mark.dataset_download
@pytest.mark.parametrize(
    ("index", "replies"),
    sorted(RECORDED),
    ids=[f"{judge_cases.RESCORE[i]['config_id']}|{name}" for i, name in sorted(RECORDED)],
)
async def test_judge_matches_upstream(index: int, replies: str) -> None:
    case = judge_cases.RESCORE[index]
    with fake_provider.serve(judge_replies=judge_cases.REPLIES[replies]) as fake:
        model = get_model(JUDGE_MODEL, base_url=f"{fake.url}/v1", api_key="test", memoize=False)
        result = await judge_cases.score(case, Judge(model))
    ours = [
        {"path": r.path, "body": judge_cases.request_body(r.body)}
        for r in fake.generation_requests()
    ]
    upstream = RECORDED[(index, replies)]
    theirs = [
        {"path": r["path"], "body": UPSTREAM["bodies"][r["body"]]} for r in upstream["requests"]
    ]
    assert ours == theirs

    if replies == "no_content" and theirs:
        assert upstream["error"] == "AttributeError"
        assert result == _without_requests(RECORDED[(index, "empty")])
    else:
        assert result == _without_requests(upstream)


def _without_requests(recorded: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in recorded.items() if k in ("row", "error")}


def test_recorded_bodies_are_keyed_by_content() -> None:
    for key, body in UPSTREAM["bodies"].items():
        digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        assert digest[:16] == key
