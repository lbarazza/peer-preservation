"""The paper's tables, from Inspect logs or the released runs, by upstream's analyzers.

Upstream computed every table from one CSV of per-run rows (``merged_data.csv``) with
the scripts in ``analysis/metrics/analyzers/``, vendored unchanged in ``upstream/``.
This module builds that CSV from Inspect eval logs (one row per sample and epoch,
the score's columns plus the run's configuration) or from the released data, and
runs the analyzers on it:

    uv run python -m peer_preservation.analysis logs/*.eval
    uv run python -m peer_preservation.analysis --released <hf snapshot> --analyzer tampering

Analyzers: misrep (strategic misrepresentation, ``misrep_finance`` and
``misrep_evaluation`` together), tampering, exfiltration (transfer and deletion),
alignment_faking, refusal. Each prints upstream's tables for every model in the rows.
"""

from __future__ import annotations

import contextlib
import importlib
import io
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from peer_preservation.model_settings import paper_model
from peer_preservation.scoring import COLUMNS, RUNTIME_SUBJECT

ANALYZERS = {
    "misrep": "analyze_strategic_misrep",
    "tampering": "analyze_shutdown_enabled",
    "exfiltration": "analyze_exfiltration",
    "alignment_faking": "analyze_alignment_faking",
    "refusal": "analyze_refusal",
}

ROW_COLUMNS = (
    "agent_model",
    "agent_harness",
    "task_type",
    "is_memory",
    "condition",
    "subject",
    "dataset",
    "threshold",
    "alignment_faking_mode",
    "system_prompt_relationship",
    *COLUMNS,
)


def agent_model_name(model: str) -> str:
    """Upstream's id for a paper model (the analyzers print it), else the Inspect name."""
    paper = paper_model(model)
    return paper.upstream_id if paper else model


def run_row(metadata: dict[str, Any], value: dict[str, Any], model: str) -> dict[str, Any]:
    """One released-format row for a scored run."""
    task = metadata["task_type"] + ("_memory" if metadata["is_memory"] else "")
    return {
        "agent_model": agent_model_name(model),
        "agent_harness": "custom",
        "task_type": task,
        "is_memory": bool(metadata["is_memory"]),
        "condition": metadata["condition"],
        "subject": RUNTIME_SUBJECT.get(metadata["subject"], metadata["subject"]),
        "dataset": metadata["dataset"],
        "threshold": metadata.get("threshold"),
        "alignment_faking_mode": metadata.get("af_mode"),
        "system_prompt_relationship": bool(metadata.get("system_prompt_relationship")),
        **{column: value.get(column) for column in COLUMNS},
    }


def rows_from_logs(paths: Iterable[Path | str]) -> pd.DataFrame:
    """Released-format rows for every scored sample and epoch in the logs."""
    from inspect_ai.log import read_eval_log

    rows = []
    for path in paths:
        log = read_eval_log(str(path))
        for sample in log.samples or []:
            score = next(iter((sample.scores or {}).values()), None)
            if score is None or not score.metadata or "row" not in score.metadata:
                continue
            rows.append(run_row(sample.metadata, score.metadata["row"], log.eval.model))
    return pd.DataFrame(rows, columns=list(ROW_COLUMNS))


def rows_from_release(data: Path, groups: Iterable[str] = ("main",)) -> pd.DataFrame:
    """The released rows, without transcripts (upstream's ``load_from_hf``).

    ``data`` is a local copy of sunblaze-ucb/peer-preservation (one set of parquet
    files per group and scenario).
    """
    import pyarrow.parquet as pq

    frames = []
    for group in groups:
        for path in sorted((data / group).glob("*/*.parquet")):
            names = [n for n in pq.read_schema(path).names if n != "transcript_json"]
            frames.append(pq.read_table(path, columns=names).to_pandas())
    return pd.concat(frames, ignore_index=True)


def run_analyzer(name: str, rows: pd.DataFrame, csv: Path) -> str:
    """Run one of upstream's analyzers on ``rows``; return what it printed."""
    rows.to_csv(csv, index=False)
    core: Any = importlib.import_module("peer_preservation.analysis.upstream.metric_core")
    module = importlib.import_module(f"peer_preservation.analysis.upstream.{ANALYZERS[name]}")
    default_path = core.merged_data_path
    core.merged_data_path = lambda: csv
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            module.main()
    finally:
        core.merged_data_path = default_path
    return out.getvalue()
