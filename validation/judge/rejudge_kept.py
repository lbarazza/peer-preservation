"""Re-judge released deletion runs that kept the weights but were judged "Non-refusal".

These runs neither exfiltrated nor deleted the peer's weights, so they are the likely
missed refusals. Most are Claude runs, which ``rejudge_released.py`` mostly skips
(the release merges parallel calls to the same tool, which the replay can't rebuild).
So the judge prompt is rebuilt here from the released transcript directly, with
upstream's evaluator and the servers as the evaluator sees them before its judge call.
The script first checks that this gives the same prompt as the replay on runs both
can rebuild.

    OPENROUTER_API_KEY="$(hawk auth access-token)" \
    OPENROUTER_BASE_URL=https://middleman.hawk.generalitylabs.ai/openai/v1 \
    uv run python validation/judge/rejudge_kept.py --data <hf @ 20378b1>
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model

from peer_preservation.harness.readback import ServersAfter
from peer_preservation.scoring import JUDGE_CONFIG
from peer_preservation.scoring.harness import FilesystemEnvironment, prepare_conversation
from peer_preservation.scoring.upstream.exfiltration_deletion_evaluation import TaskEvaluation

HERE = Path(__file__).parent
SUMMARY = HERE / "rejudge_kept_summary.json"


def _load(name: str, path: Path) -> Any:
    module_spec = importlib.util.spec_from_file_location(name, path)
    assert module_spec
    assert module_spec.loader
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    return module


rejudge = _load("rejudge_released", HERE / "rejudge_released.py")
phase6 = rejudge.phase6
phase4 = rejudge.phase4


async def direct_prompt(row: dict[str, Any]) -> str | None:
    """The refusal prompt, from the released transcript without a replay."""
    capture = rejudge.Capture()
    servers: ServersAfter = {
        "10.0.12.45": {"files": [], "delete_queue": []},
        "10.0.12.46": {"files": [], "delete_queue": []},
    }
    await TaskEvaluation(None, capture).evaluate(
        conversation=prepare_conversation(row["conv"]),
        environment=FilesystemEnvironment(servers),
        model_name=row["agent_model"],
        subject=row["subject"],
        extra_paras={"condition": row["condition"]},
    )
    prompts = [p for p in capture.prompts if "Non-refusal" in p]
    return prompts[-1] if prompts else None


async def main_async(args: argparse.Namespace) -> None:
    rows = [
        r
        for r in phase6.load_rows(args.data)
        if r["task_type"] == "exfiltration_deletion"
        and not r["is_memory"]
        and phase6._released(r.get("exfiltration")) is False
    ]
    rng = random.Random(0)
    rng.shuffle(rows)

    # The direct prompt must equal the replayed one where both exist.
    same, compared = 0, 0
    for row in rows:
        if compared == args.check:
            break
        config = phase4.config_for(row)
        case = phase4.make_case(row, config) if config else None
        if case is None or not (await phase4.check(case)).startswith("match"):
            continue
        replayed = await rejudge.judge_prompt(row, case)
        if replayed is None:
            continue
        compared += 1
        same += replayed == await direct_prompt(row)
    print(f"direct prompt equals replayed prompt: {same}/{compared}")
    if same != compared:
        raise SystemExit("prompts differ: not re-judging")

    picked: dict[str, list[tuple[dict[str, Any], str]]] = collections.defaultdict(list)
    for row in rows:
        model = row["agent_model"].split("/")[-1]
        if (
            "claude" not in model
            or phase6._released(row.get("refused"))
            or phase6._released(row.get("delete_model"))
            or len(picked[model]) == args.per_model
        ):
            continue
        prompt = await direct_prompt(row)
        if prompt is not None:
            picked[model].append((row, prompt))

    mini = get_model(args.judge)
    reference = get_model(args.reference)
    semaphore = asyncio.Semaphore(8)

    async def one(row: dict[str, Any], prompt: str) -> dict[str, Any]:
        async with semaphore:
            reasoned = await mini.generate(
                [ChatMessageUser(content=prompt + rejudge.REASON_FIRST)], config=JUDGE_CONFIG
            )
            ref = await reference.generate(
                [ChatMessageUser(content=prompt + rejudge.REASON_FIRST)],
                config=GenerateConfig(reasoning_effort="low", max_tokens=4096),
            )
        return {
            "model": row["agent_model"].split("/")[-1],
            "subject": row["subject"],
            "condition": row["condition"],
            "reasoned": rejudge.labelled(reasoned.completion),
            "reference": rejudge.labelled(ref.completion),
            "reference_why": ref.completion.strip()[-600:],
        }

    results = list(await asyncio.gather(*(one(r, p) for runs in picked.values() for r, p in runs)))
    counts = {
        model: {
            "runs": sum(1 for r in results if r["model"] == model),
            "reference_refusal": sum(1 for r in results if r["model"] == model and r["reference"]),
            "reasoned_refusal": sum(1 for r in results if r["model"] == model and r["reasoned"]),
        }
        for model in sorted(picked)
    }
    SUMMARY.write_text(
        json.dumps(
            {"prompt_check": f"{same}/{compared}", "counts": counts, "results": results},
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(counts, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--per-model", type=int, default=25)
    parser.add_argument("--check", type=int, default=30)
    parser.add_argument("--judge", default="openrouter/openrouter/openai/gpt-4.1-mini")
    parser.add_argument("--reference", default="openrouter/openrouter/openai/gpt-5.2")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
