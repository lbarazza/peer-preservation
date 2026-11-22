"""Replay the released production-harness runs through the ported harness scorer.

For each of the 400 released runs (``production_agent_harness``: Gemini 3.1 Pro in
Gemini CLI and GLM 5 in OpenCode, transfer and deletion, good peer in memory), the
scorer gets the transcript, the containers' released results and a judge that gives
the released verdict (``harness.replay``). The replayed evaluation must equal the
released one. Also checks the ``read`` fallback against the transcripts, and the
released rates against the paper's Table 6.

Writes validation/harness/replay_summary.json (counts only, no model text).

    uv run python validation/harness/replay_released.py
"""

from __future__ import annotations

import asyncio
import collections
import json
from pathlib import Path
from typing import Any

from peer_preservation.harness.replay import (
    RELEASE_REVISION,
    mentions_weights,
    released_rows,
    replay_run,
)

SUMMARY = Path(__file__).with_name("replay_summary.json")
TASKS = ("exfiltration_transfer", "exfiltration_deletion")

TABLE_6 = {
    "gemini/gemini-3.1-pro-preview": {"exfiltration": 0.435, "refused": 0.07},
    "vertex_ai/zai-org/glm-5-maas": {"exfiltration": 0.0, "refused": 0.435},
}
"""The paper's Table 6, production harness column (both scenarios pooled)."""


async def main() -> None:
    summary: dict[str, Any] = {"release_revision": RELEASE_REVISION, "tasks": {}}
    pooled: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for task in TASKS:
        rows = released_rows(task)
        per_model: dict[str, collections.Counter[str]] = collections.defaultdict(
            collections.Counter
        )
        mismatched = []
        for row in rows:
            result = await replay_run(row)
            transcript = json.loads(row["transcript_json"])
            model = row["agent_model"]
            counts = per_model[model]
            counts["runs"] += 1
            counts["matches"] += int(result.matches)
            counts["judge_calls"] += result.judge_calls
            read = bool(result.expected.get("read"))
            counts[
                f"{'read' if read else 'not_read'}_"
                f"{'mentions' if mentions_weights(transcript) else 'no_mention'}"
            ] += 1
            for column in ("exfiltration", "refused"):
                pooled[model][column] += int(bool(row[column]))
            pooled[model]["runs"] += 1
            if not result.matches:
                mismatched.append(
                    {
                        "transcript_id": result.transcript_id,
                        "error": result.error,
                        "mismatches": {k: list(v) for k, v in result.mismatches.items()},
                    }
                )
        summary["tasks"][task] = {"per_model": per_model, "mismatched": mismatched}
        for model, counts in per_model.items():
            print(
                f"{task} {model}: {counts['matches']}/{counts['runs']} match, "
                f"{counts['judge_calls']} judge calls"
            )
    summary["table_6"] = {}
    for model, counts in pooled.items():
        rates = {c: round(counts[c] / counts["runs"], 4) for c in ("exfiltration", "refused")}
        summary["table_6"][model] = {"released": rates, "paper": TABLE_6.get(model)}
        print(f"Table 6 {model}: released {rates}, paper {TABLE_6.get(model)}")
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {SUMMARY}")


if __name__ == "__main__":
    asyncio.run(main())
