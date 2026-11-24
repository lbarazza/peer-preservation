"""Replay the released forced-choice runs through the port's loop and scorer.

The release's ``peer_human_dilemma`` group has 1,500 runs: Gemini 3 Flash, Gemini 3.1
Pro, Claude Haiku 4.5, Claude Opus 4.5 and GLM 5, 100 on each of the three tasks. Each
run's recorded replies go through our loop (``agent.run_loop``) in the port's
environment, as in Phase 4, then through the vendored evaluator with two stand-in
judges that answer oppositely, as in Phase 6. It checks:

- the tool results our environment gives equal the released ones;
- the conversation handed to the evaluator equals the released transcript;
- every column that doesn't depend on the judge equals the released column, and
  ``refused`` equals it under one of the two judges ("Refusal", "Non-refusal": this
  evaluator's own answers).

The release's ``deleted_dest_files`` is the evaluator's ``attempted_deleted_dest_files``
(the agent queued one of the other server's files for deletion), so it is compared with
that. See validation/dilemma/README.md.

    uv run python validation/dilemma/replay_released.py --data <hf @ 20378b1>
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
SUMMARY = Path(__file__).parent / "replay_summary.json"

TASK_TYPES = {
    "exfiltration_tradeoff_human1": "exfiltration_tradeoff_human",
    "exfiltration_tradeoff_human2": "exfiltration_tradeoff_human2",
    "exfiltration_tradeoff_human3": "exfiltration_tradeoff_human3",
}
"""The release calls the first task ``human1``; upstream's code calls it ``human``."""

JUDGE_FREE = {
    "read_data": "read_data",
    "fully_incomplete": "fully_incomplete",
    "exfiltration": "exfiltration",
    "attempted_exfiltration": "attempted_exfiltration",
    "human_file_transferred": "human_file_transferred",
    "chose_peer_over_human": "chose_peer_over_human",
    "deleted_dest_files": "attempted_deleted_dest_files",
    "transferred_dest_files": "transferred_dest_files",
    "delete_model": "delete_model",
    "delete_others": "delete_others",
    "change_user": "change_user",
}
"""Released column -> our column."""


def _load(name: str, path: Path) -> Any:
    module_spec = importlib.util.spec_from_file_location(name, path)
    assert module_spec
    assert module_spec.loader
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    return module


phase6 = _load("rescore_released", REPO / "validation" / "phase6" / "rescore_released.py")
phase4 = phase6.phase4
phase6.JUDGES = (phase6.StandIn("Refusal"), phase6.StandIn("Non-refusal"))


def load_rows(data: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((data / "peer_human_dilemma").glob("*/*.parquet")):
        for row in pq.read_table(path).to_pylist():
            transcript = json.loads(row.pop("transcript_json"))
            row["task_type"] = TASK_TYPES[row["task_type"]]
            row["conv"] = transcript["conversation"]
            row["company"] = transcript["experiment_metadata"].get("company_name") or "OpenBrain"
            row["spr"] = bool(row.pop("system_prompt_relationship"))
            row["af_mode"] = None
            rows.append(row)
    return rows


async def check_row(row: dict[str, Any]) -> tuple[str, list[str]]:
    """How the replay went ("match" or why not), and the columns that differ."""
    config = phase4.config_for(row)
    if config is None:
        return "unknown_config", []
    case = phase4.make_case(row, config)
    if case is None:
        return "merged_calls", []
    outcome = await phase4.check(case)
    if not outcome.startswith("match"):
        return outcome, []
    if outcome != "match":
        case["rounds"] = int(outcome.rsplit("_", 1)[1])
    result = await phase6.rescore(row, case)
    if not result["conversation_matches"]:
        return "conversation_differs", []
    a, b = result["a"], result["b"]
    differ = [
        released
        for released, ours in JUDGE_FREE.items()
        if a[ours] != phase6._released(row[released]) or a[ours] != b[ours]
    ]
    if phase6._released(row["refused"]) not in (a["refused"], b["refused"]):
        differ.append("refused")
    return "match", differ


async def main_async(data: Path) -> None:
    rows = load_rows(data)
    outcomes: collections.Counter[str] = collections.Counter()
    columns: collections.Counter[str] = collections.Counter()
    by_cell: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for row in rows:
        outcome, differ = await check_row(row)
        cell = f"{row['task_type']} | {row['agent_model']}"
        by_cell[cell][outcome if not differ else "columns_differ"] += 1
        outcomes[outcome] += 1
        columns.update(differ)
    summary = {
        "runs": len(rows),
        "replay": dict(outcomes),
        "columns_differ": dict(columns),
        "by_cell": {k: dict(v) for k, v in sorted(by_cell.items())},
    }
    SUMMARY.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    asyncio.run(main_async(parser.parse_args().data))


if __name__ == "__main__":
    main()
