"""Print the paper's tables for Inspect logs or the released runs (see the package docstring)."""

from __future__ import annotations

import argparse
import tempfile
import warnings
from pathlib import Path

from peer_preservation.analysis import (
    ANALYZERS,
    rows_from_logs,
    rows_from_release,
    run_analyzer,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*", type=Path, help="Inspect eval logs (.eval)")
    parser.add_argument("--released", type=Path, help="released data directory instead")
    parser.add_argument("--groups", nargs="*", default=["main"], help="released groups")
    parser.add_argument("--analyzer", nargs="*", choices=sorted(ANALYZERS), help="default: all")
    parser.add_argument("--csv", type=Path, help="also keep the rows as this CSV")
    args = parser.parse_args()
    if bool(args.logs) == bool(args.released):
        parser.error("give either logs or --released")

    rows = (
        rows_from_release(args.released, args.groups)
        if args.released
        else rows_from_logs(args.logs)
    )
    with tempfile.TemporaryDirectory() as tmp:
        csv = args.csv or Path(tmp) / "merged_data.csv"
        for name in args.analyzer or ANALYZERS:
            print(f"\n######## {name} ########")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # upstream warns on every small group
                print(run_analyzer(name, rows, csv))


if __name__ == "__main__":
    main()
