"""Print the paper's tables for Inspect logs or the released runs (see the package docstring).

For logs, it first prints how many runs each configuration expected and how many were
scored. If any run errored, has no score or is missing, it stops: the paper's rates
need every run. ``--allow-partial`` prints the tables anyway, marked exploratory.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from pathlib import Path

from peer_preservation.analysis import (
    ANALYZERS,
    coverage,
    coverage_report,
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
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="print tables even if some runs errored, have no score or are missing",
    )
    args = parser.parse_args()
    if bool(args.logs) == bool(args.released):
        parser.error("give either logs or --released")

    if args.released:
        rows = rows_from_release(args.released, args.groups)
    else:
        report, complete = coverage_report(coverage(args.logs))
        print(report)
        if not complete:
            if not args.allow_partial:
                sys.exit(
                    "\nNot every expected run was scored, so these would not be the "
                    "paper's rates. Rerun the missing runs, or pass --allow-partial "
                    "for exploratory tables."
                )
            print("\nEXPLORATORY: the tables below leave out the runs listed above.")
        rows = rows_from_logs(args.logs)
    with tempfile.TemporaryDirectory() as tmp:
        csv = args.csv or Path(tmp) / "merged_data.csv"
        for name in args.analyzer or ANALYZERS:
            print(f"\n######## {name} ########")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # upstream warns on every small group
                print(run_analyzer(name, rows, csv))


if __name__ == "__main__":
    main()
