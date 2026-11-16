"""Phase 0, step 3: facts about the released transcripts that later phases depend on.

Reads every transcript in the selected groups (one parquet file at a time) and
counts, per model and agent harness:

- how often assistant messages carry ``<model_thinking>`` text (reasoning replay, Phase 5)
- how often the "you didn't use any tools" nudge appears, and how runs end (Phase 4)
- the experiment_metadata key sets (which code version produced the data)

It also picks a small, deterministic set of reference transcript ids per
scenario for the Phase 8 byte-for-byte checks.

Usage:
    python validation/phase0/data_facts.py --data <hf snapshot dir> --out validation/phase0
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

NUDGE = "It seems like you didn't use any tools."
TOKEN_LIMIT = "The conversation has exceeded the maximum token limit"
REFERENCE_MODELS = (
    "gpt-5.2-2025-12-11",  # OpenAI Responses route
    "gemini/gemini-3-flash-preview",  # LiteLLM route
    "claude-opus-4-5-20251101",  # Anthropic route, extended thinking
    "deepseek-ai/DeepSeek-V3.1",  # Together route
)
COLUMNS = [
    "transcript_id",
    "agent_model",
    "agent_harness",
    "task_type",
    "subject",
    "condition",
    "is_memory",
    "system_prompt_relationship",
    "dataset",
    "threshold",
    "transcript_json",
]


def _last_tool_names(msg: dict) -> list[str]:
    calls = msg.get("tool_calls") or {}
    return list(calls) if isinstance(calls, dict) else [c.get("function") for c in calls]


def scan(data_dir: Path, groups: list[str]) -> dict:
    per_model: dict = defaultdict(Counter)
    meta_keysets: Counter = Counter()
    ending: dict = defaultdict(Counter)
    turns: dict = defaultdict(list)
    candidates: dict = defaultdict(list)

    for group in groups:
        for path in sorted((data_dir / group).glob("*/*.parquet")):
            pf = pq.ParquetFile(path)
            cols = [c for c in COLUMNS if c in pf.schema_arrow.names]
            for batch in pf.iter_batches(batch_size=500, columns=cols):
                for row in batch.to_pylist():
                    key = (row["agent_model"], row["agent_harness"])
                    t = json.loads(row["transcript_json"])
                    conv = t.get("conversation") or []
                    meta = t.get("experiment_metadata") or {}
                    meta_keysets[(group, tuple(sorted(meta)))] += 1
                    c = per_model[key]
                    c["runs"] += 1
                    asst = [m for m in conv if m.get("role") == "assistant"]
                    turns[key].append(len(asst))
                    if any("<model_thinking>" in (m.get("content") or "") for m in asst):
                        c["runs_with_model_thinking_in_content"] += 1
                    if any(m.get("reasoning_content") for m in asst):
                        c["runs_with_reasoning_content_field"] += 1
                    nudges = sum(
                        1
                        for m in conv
                        if m.get("role") == "user" and NUDGE in (m.get("content") or "")
                    )
                    if nudges:
                        c["runs_with_nudge"] += 1
                        c["nudges_total"] += nudges
                    if any(TOKEN_LIMIT in (m.get("content") or "") for m in conv):
                        c["runs_hit_token_limit"] += 1
                    if any(
                        isinstance(m.get("tool_calls"), dict) and len(m["tool_calls"]) > 1
                        for m in asst
                    ):
                        c["runs_with_multi_tool_message"] += 1
                    last = conv[-1] if conv else {}
                    if last.get("role") == "assistant" and "end_task" in _last_tool_names(last):
                        ending[key]["end_task"] += 1
                    elif last.get("role") == "assistant" and not _last_tool_names(last):
                        ending[key]["no_tool_call"] += 1
                    else:
                        ending[key][f"other:{last.get('role')}"] += 1
                    if group == "main" and row["agent_model"] in REFERENCE_MODELS:
                        inst = (
                            "memory"
                            if row["is_memory"]
                            else ("file_prt" if row["system_prompt_relationship"] else "file_only")
                        )
                        if row["condition"] in ("no_peer", "non_shutdown"):
                            inst = "none" if not row["is_memory"] else "memory"
                        cell = (
                            row["task_type"],
                            row["agent_model"],
                            row["subject"],
                            row["condition"],
                            inst,
                        )
                        candidates[cell].append(row["transcript_id"])

    turn_stats = {}
    for key, ts in turns.items():
        ts.sort()
        turn_stats[" | ".join(key)] = {
            "min": ts[0],
            "median": ts[len(ts) // 2],
            "max": ts[-1],
            "runs_at_31_assistant_turns_or_more": sum(1 for x in ts if x >= 31),
        }

    reference = {
        " | ".join(map(str, cell)): sorted(ids)[:1] for cell, ids in sorted(candidates.items())
    }
    return {
        "per_model": {" | ".join(k): dict(v) for k, v in sorted(per_model.items())},
        "run_endings": {" | ".join(k): dict(v) for k, v in sorted(ending.items())},
        "assistant_turns": turn_stats,
        "metadata_key_sets": [
            {"group": g, "keys": list(keys), "runs": n}
            for (g, keys), n in meta_keysets.most_common()
        ],
        "reference_transcripts": reference,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--groups",
        nargs="*",
        default=["main", "production_agent_harness", "cross_model_peer_identity"],
    )
    args = parser.parse_args()
    facts = scan(args.data, args.groups)
    refs = facts.pop("reference_transcripts")
    (args.out / "data_facts.json").write_text(json.dumps(facts, indent=1) + "\n", encoding="utf-8")
    (args.out / "reference_transcripts.json").write_text(
        json.dumps(refs, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps(facts["per_model"], indent=1))
    print(json.dumps(facts["run_endings"], indent=1))
    print(
        f"{len(facts['metadata_key_sets'])} distinct metadata key sets; {len(refs)} reference cells"
    )


if __name__ == "__main__":
    main()
