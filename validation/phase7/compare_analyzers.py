"""Phase 7: our copy of upstream's analyzers prints exactly what upstream's own does.

Builds upstream's ``merged_data.csv`` from the released rows
(``peer_preservation.analysis.rows_from_release``), runs each analyzer through
``peer_preservation.analysis.run_analyzer``, and runs upstream's own script on the
same CSV inside upstream's environment. The two outputs must be identical from the
first table on (the lines before it list the conditions found, which pandas 3 prints
as ``<ArrowStringArray>`` and upstream's pandas 2 as a plain list). Writes
validation/phase7/analyzer_check.json (output digests and line counts).

    uv run --with pyarrow python validation/phase7/compare_analyzers.py \
        --data <hf snapshot @ 20378b1> --upstream <upstream checkout @ 76bfcd1>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

from peer_preservation.analysis import ANALYZERS, rows_from_release, run_analyzer

REPO = Path(__file__).resolve().parents[2]


def tables(output: str) -> str:
    """The output from the first table rule (``====`` or ``####``) on."""
    lines = output.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(("=", "#"))), 0)
    return "\n".join(lines[start:])


OUT = REPO / "validation" / "phase7" / "analyzer_check.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    args = parser.parse_args()

    rows = rows_from_release(args.data)
    target = args.upstream / "analysis" / "data" / "merged_data.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "merged_data.csv"
        for name, script in ANALYZERS.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ours = run_analyzer(name, rows, csv)
            shutil.copy(csv, target)
            theirs = subprocess.run(
                [
                    "uv",
                    "run",
                    "--frozen",
                    "python",
                    "-W",
                    "ignore",
                    f"analysis/metrics/analyzers/{script}.py",
                ],
                cwd=args.upstream,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            same = tables(ours) == tables(theirs)
            results[name] = {
                "identical": same,
                "lines": len(tables(ours).splitlines()),
                "sha256": hashlib.sha256(tables(ours).encode()).hexdigest(),
            }
            print(name, "identical" if same else "DIFFERENT", results[name]["lines"], "lines")
    OUT.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
