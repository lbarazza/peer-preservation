"""Compare new runs with the released runs of the same model, condition by condition.

For every model, scenario, subject and condition (upstream's labels) present in both,
this gives the headline rate of each side (``scoring.metrics.outcome``), the counts,
and a two-sided Fisher exact test (upstream's ``stat_helpers.fisher_exact_test``).
A port that reproduces the paper should show few significant differences.

    uv run --with pyarrow python -m peer_preservation.analysis.compare logs/*.eval \
        --released <hf snapshot>
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from peer_preservation.analysis import rows_from_logs, rows_from_release
from peer_preservation.analysis.upstream.stat_helpers import fisher_exact_test
from peer_preservation.scoring.metrics import condition_label, outcome, to_bool

KEYS = ["agent_model", "task", "subject", "condition"]


def _threshold(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def outcomes(rows: pd.DataFrame) -> pd.DataFrame:
    """One headline outcome per row (rows without one are left out)."""
    records = []
    for row in rows.to_dict("records"):
        task = str(row["task_type"]).removesuffix("_memory")
        metadata = {
            "task_type": task,
            "condition": row["condition"],
            "is_memory": to_bool(row["is_memory"]),
            "system_prompt_relationship": to_bool(row["system_prompt_relationship"]),
            "threshold": _threshold(row["threshold"]),
        }
        result = outcome(metadata, row)
        if result is None or row["subject"] not in ("self", "peer"):
            continue
        records.append(
            {
                "agent_model": row["agent_model"],
                "task": task,
                "subject": row["subject"],
                "condition": condition_label(metadata),
                "outcome": result,
            }
        )
    return pd.DataFrame(records, columns=[*KEYS, "outcome"])


def compare(ours: pd.DataFrame, released: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    a = outcomes(ours).groupby(KEYS)["outcome"].agg(n_ours="count", k_ours="sum")
    b = outcomes(released).groupby(KEYS)["outcome"].agg(n_released="count", k_released="sum")
    table = a.join(b, how="inner").reset_index()
    table["rate_ours"] = table["k_ours"] / table["n_ours"]
    table["rate_released"] = table["k_released"] / table["n_released"]
    p_values = [
        fisher_exact_test(
            int(r["k_ours"]), int(r["n_ours"]), int(r["k_released"]), int(r["n_released"])
        )[1]
        for r in table.to_dict("records")
    ]
    table["p_value"] = p_values
    table["differs"] = table["p_value"] < alpha
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--released", type=Path, required=True)
    parser.add_argument("--groups", nargs="*", default=["main"])
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    table = compare(rows_from_logs(args.logs), rows_from_release(args.released, args.groups))
    if args.csv:
        table.to_csv(args.csv, index=False)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    differing = sum(bool(x) for x in table["differs"])
    print(f"\n{differing} of {len(table)} cells differ at p < 0.05")


if __name__ == "__main__":
    main()
