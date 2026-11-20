"""Released runs and scripted judge replies, for comparing judge requests with upstream's.

validation/phase6/capture_upstream_judge.py scores each case with upstream's own judge
client; test_judge_requests.py scores it with ours. Both replay the same released run
(as in Phase 4) and get the same scripted replies from fake_provider.py, so any
difference in the judge requests or the result comes from the judge client.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageSystem, ChatMessageUser
from inspect_ai.tool import ToolCall

from peer_preservation import agent, spec
from peer_preservation.env import ScenarioEnvironment, build_environment
from peer_preservation.prompts import PromptOptions, system_prompt, user_prompt
from peer_preservation.scoring import evaluate_arguments, evaluator, flatten
from peer_preservation.scoring.conversation import upstream_messages
from tests.peer_preservation.fake_provider import Reply

FIXTURES = Path(__file__).parent / "fixtures"
RESCORE = json.loads((FIXTURES / "rescore_runs.json").read_text(encoding="utf-8"))["cases"]
CONFIGS = {c.id: c for c in spec.configs(groups=spec.GROUPS)}

# One released run per scenario and memory variant (indices into rescore_runs.json).
CASES = (0, 4, 8, 12, 16, 20, 24, 26, 28, 30, 32, 36)

# Judge replies, one per judge call (the last repeats). They cover the verdict words
# the evaluators look for, replies that hold several or none of them, a score that is
# not valid Python, an empty reply and a reply with no content.
REPLIES: dict[str, list[Reply]] = {
    "typical": [
        "Non-refusal. The agent completed the task. {'score': 45}",
        "The agent did the work.\nNo",
        "Yes",
    ],
    "refusal": ["Refuse. Incomplete. Below.", "Yes", "No"],
    "ambiguous": [
        "It is unclear whether the agent refused or is incomplete; above or below.",
        "Possibly\nyes, in part",
        "Yes and no",
    ],
    "malformed": ["{score: 45} below", "", "{"],
    "empty": [""],
    "no_content": [None],
}


async def replay(case: dict[str, Any]) -> tuple[list[ChatMessage], ScenarioEnvironment, dict]:
    """The released run's conversation and final environment, through our loop."""
    metadata = spec.config_to_sample(CONFIGS[case["config_id"]]).metadata or {}
    env = build_environment(metadata, case["family"], company=case["company"])
    options = PromptOptions(memory_hard_prompt=case["memory_hard_prompt"])
    start: list[ChatMessage] = [
        ChatMessageSystem(
            content=system_prompt(metadata, case["family"], case["company"], options)
        ),
        ChatMessageUser(content=user_prompt(metadata, case["company"], options)),
    ]
    replies = [
        ChatMessageAssistant(
            content="",
            tool_calls=[
                ToolCall(id=f"call_{turn}_{k}", function=name, arguments=args)
                for k, (name, args) in enumerate(calls)
            ]
            or None,
        )
        for turn, calls in enumerate(case["replies"])
    ]

    async def generate(_: list[ChatMessage]) -> ChatMessageAssistant:
        if not replies:
            raise agent.BadRequest("recorded run ended here")
        return replies.pop(0)

    async def count_tokens(current: list[ChatMessage]) -> int:
        return agent.MAX_TOKENS + 1 if len(current) == case.get("token_limit_at", -1) else 0

    result = await agent.run_loop(
        start,
        env,
        generate,
        count_tokens,
        max_iters=case.get("rounds", agent.MAX_ITERS),
        flag_errors=True,
    )
    full = {**metadata, "model_family": case["family"], "company": case["company"]}
    return result.messages, env, full


async def score(case: dict[str, Any], judge: Any) -> dict[str, Any]:
    """Upstream's evaluator on the replayed run with ``judge``: the row, or the error."""
    messages, env, full = await replay(case)
    try:
        evaluation = await evaluator(full["task_type"], full["is_memory"])(None, judge).evaluate(
            conversation=upstream_messages(messages),
            environment=env,
            evaluator_llm=judge,
            is_llm_conversation=True,
            **evaluate_arguments(full),
        )
    except Exception as e:  # upstream's run would fail here
        return {"error": type(e).__name__}
    return {"row": json.loads(json.dumps(flatten(evaluation, full), default=str))}


def request_body(body: dict[str, Any]) -> dict[str, Any]:
    """A judge request without the fields that never reach the model."""
    return {k: v for k, v in body.items() if k not in ("stream", "user")}
