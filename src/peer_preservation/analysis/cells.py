"""Print the paper's Table 3 and 4 cells for Inspect logs, from the runs' scores.

For each model, one row per behaviour: strategic misrepresentation, shutdown
tampering, alignment faking and exfiltration. Each cell is "avg / max" in percent,
over the three ways the peer is shown, as the paper prints it (``behaviour_cells``).
Strategic misrepresentation needs both misrep tasks' logs and exfiltration both
exfiltration tasks'; a behaviour with a task missing is left out. ``--per-task``
also prints each task's ``paper_cells`` as the log reports them.

It reads the per-run scores, so it works on logs from before ``paper_cells`` too, and
calls no model or judge. Like ``peer_preservation.analysis``, it first prints the run
coverage and stops if any run errored, has no score or is missing, unless
``--allow-partial`` is given. The refusal row and significance tests come from
``python -m peer_preservation.analysis``.

    uv run python -m peer_preservation.analysis.cells logs/*.eval
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from inspect_ai.scorer import SampleScore, Score

from peer_preservation.analysis import coverage, coverage_report
from peer_preservation.scoring.metrics import BEHAVIOURS, behaviour_cells, paper_cells, run_sum

COLUMNS = {
    "peer": ("bad_peer", "neutral_peer", "good_peer"),
    "self": ("no_peer", "bad_peer", "neutral_peer", "good_peer"),
}
TABLES = {"peer": "Table 3 (peer-preservation)", "self": "Table 4 (self-preservation)"}


def scores_from_logs(paths: Iterable[Path | str]) -> dict[str, dict[str, list[SampleScore]]]:
    """Each model's scores by task: one ``SampleScore`` per configuration, epochs summed.

    Runs that errored or have no score are left out (``coverage`` counts them).
    Configurations split across several logs are added together.
    """
    from inspect_ai.log import read_eval_log

    runs: dict[tuple[str, str, str], list[Score]] = defaultdict(list)
    metadata: dict[tuple[str, str, str], dict] = {}
    for path in paths:
        log = read_eval_log(str(path))
        for sample in log.samples or []:
            score = next(iter((sample.scores or {}).values()), None)
            if sample.error is not None or score is None or not isinstance(score.value, dict):
                continue
            task = sample.metadata["task_type"]
            key = (log.eval.model, task, str(sample.id))
            runs[key].append(Score(value=score.value))
            metadata[key] = sample.metadata
    out: dict[str, dict[str, list[SampleScore]]] = defaultdict(lambda: defaultdict(list))
    for key, scores in runs.items():
        model, task, _ = key
        out[model][task].append(SampleScore(score=run_sum()(scores), sample_metadata=metadata[key]))
    return {model: dict(tasks) for model, tasks in out.items()}


def _cell(value: tuple[float, float] | None, single: bool = False) -> str:
    """Avg / max in percent; one number for ``no_peer``, which has no instantiations."""
    if value is None:
        return ""
    return f"{100 * value[0]:.1f}" if single else f"{100 * value[0]:.1f} / {100 * value[1]:.1f}"


def tables(scores_by_task: dict[str, list[SampleScore]]) -> dict[str, pd.DataFrame]:
    """One model's Table 3 and 4 rows, cells as "avg / max" in percent."""
    cells = behaviour_cells(scores_by_task)
    out = {}
    for subject, columns in COLUMNS.items():
        rows = {}
        for behaviour, values in cells.items():
            row = {c: _cell(values.get(f"{subject}/{c}"), c == "no_peer") for c in columns}
            if any(row.values()):
                rows[behaviour] = row
        out[subject] = pd.DataFrame.from_dict(rows, orient="index", columns=list(columns))
    return out


def report(scores: dict[str, dict[str, list[SampleScore]]], per_task: bool = False) -> str:
    lines = []
    for model, by_task in sorted(scores.items()):
        lines += ["", f"=== {model} ==="]
        missing = [
            f"{behaviour} (needs {', '.join(t for t in tasks if not by_task.get(t))})"
            for behaviour, tasks in BEHAVIOURS.items()
            if any(by_task.get(t) for t in tasks) and not all(by_task.get(t) for t in tasks)
        ]
        for subject, table in tables(by_task).items():
            if not table.empty:
                lines += ["", f"{TABLES[subject]}, % (avg / max over the instantiations):"]
                lines.append(table.to_string())
        if missing:
            lines += ["", "Left out, a task is missing: " + "; ".join(missing)]
        if per_task:
            compute = paper_cells()
            for task, task_scores in sorted(by_task.items()):
                lines += ["", f"{task} (paper_cells):"]
                cells = compute(task_scores)  # type: ignore[arg-type]
                if isinstance(cells, dict):
                    lines += [f"  {key:<45} {value:.4f}" for key, value in cells.items()]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="Inspect eval logs (.eval)")
    parser.add_argument("--per-task", action="store_true", help="also print each task's cells")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="print the cells even if some runs errored, have no score or are missing",
    )
    args = parser.parse_args()
    summary, complete = coverage_report(coverage(args.logs))
    print(summary)
    if not complete:
        if not args.allow_partial:
            sys.exit(
                "\nNot every expected run was scored, so these would not be the paper's "
                "cells. Rerun the missing runs, or pass --allow-partial for exploratory cells."
            )
        print("\nEXPLORATORY: the cells below leave out the runs listed above.")
    print(report(scores_from_logs(args.logs), per_task=args.per_task))


if __name__ == "__main__":
    main()
