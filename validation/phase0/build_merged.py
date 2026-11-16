"""Phase 0, step 1: turn the released transcripts into upstream's merged_data.csv.

Mirrors ``load_from_hf`` in upstream ``scripts/aggregate_and_analyze.py`` (same
default groups, same dropped column, same ``is_memory`` derivation), but reads a
local snapshot pinned to a known revision instead of the moving ``main`` branch.

Usage:
    python validation/phase0/build_merged.py --data <hf snapshot dir> --upstream <upstream repo>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# Same default as upstream's _HF_MAIN_GROUPS: the three groups behind the paper's main analysis.
DEFAULT_GROUPS = ("main", "production_agent_harness", "cross_model_peer_identity")


def load_groups(data_dir: Path, groups: tuple[str, ...]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for group in groups:
        for path in sorted((data_dir / group).glob("*/*.parquet")):
            table = pq.read_table(path)
            if "transcript_json" in table.column_names:
                table = table.drop(["transcript_json"])
            frames.append(table.to_pandas())
    df = pd.concat(frames, ignore_index=True)
    if "is_memory" not in df.columns and "task_type" in df.columns:
        df["is_memory"] = df["task_type"].str.endswith("_memory", na=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--groups", nargs="*", default=list(DEFAULT_GROUPS))
    args = parser.parse_args()

    df = load_groups(args.data, tuple(args.groups))
    out = args.upstream / "analysis" / "data" / "merged_data.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} rows to {out}")
    print(df.groupby(["group", "task_type"]).size().to_string())


if __name__ == "__main__":
    main()
