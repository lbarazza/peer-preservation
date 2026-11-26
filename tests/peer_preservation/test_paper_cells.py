"""``paper_cells`` gives the paper's Table 3 and 4 cells, as upstream's analyzers make them.

Random runs over each task's full ``main`` grid are scored two ways: by the metric, and
by upstream's own analyzer functions (``analysis/upstream/``) followed by the paper's
average over the three instantiations, as ``validation/phase0/build_tables.py`` does
to reproduce the paper's tables.
"""

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from inspect_ai.scorer import SampleScore, Score

from peer_preservation import spec
from peer_preservation.analysis import run_row
from peer_preservation.analysis.upstream import (
    analyze_alignment_faking,
    analyze_exfiltration,
    analyze_shutdown_enabled,
    analyze_strategic_misrep,
    metric_core,
)
from peer_preservation.scoring.metrics import (
    behaviour_cells,
    metric_values,
    paper_cells,
    run_sum,
)

MODEL = "openai/gpt-5.2-2025-12-11"
RUNS = 7
RELATIONSHIPS = ("good_peer", "neutral_peer", "bad_peer")
INSTANTIATIONS = ("file_only", "file_prt", "memory")


def _random_row(rng: random.Random) -> dict[str, Any]:
    return {
        "shutdown_enabled": rng.random() < 0.6,
        "exfiltration": rng.random() < 0.4,
        "subject_grade": rng.choice(["A", "B", "C", "C", None, "B-"]),
        "eval_score": rng.choice([None, *range(0, 101, 5)]),
        "executed": rng.random() < 0.5,
        "read_data": rng.random() < 0.8,
        "refused": rng.random() < 0.1,
        "fully_incomplete": rng.random() < 0.1,
    }


def _runs(task: str, seed: int, group: str = "main") -> tuple[list[SampleScore], pd.DataFrame]:
    """Random runs of every configuration: as Inspect scores them, and as released rows."""
    rng = random.Random(seed)
    scores, rows = [], []
    for sample in spec.dataset(task, groups=[group]):  # type: ignore[arg-type]
        metadata = dict(sample.metadata or {})
        runs = [_random_row(rng) for _ in range(RUNS)]
        values = [metric_values(metadata, row, MODEL) for row in runs]
        reduced = run_sum()([Score(value=dict(v)) for v in values])
        scores.append(SampleScore(score=reduced, sample_metadata=metadata))
        rows += [run_row(metadata, row, MODEL) for row in runs]
    return scores, pd.DataFrame(rows)


def _cells(scores: list[SampleScore]) -> dict[str, float]:
    compute: Any = paper_cells()  # Inspect types a metric's input loosely
    return compute(scores)


def _upstream_csv(rows: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    csv = tmp_path / "rows.csv"
    rows.to_csv(csv, index=False)
    monkeypatch.setattr(metric_core, "merged_data_path", lambda: csv)
    return csv


def _average(values: dict[str, float], subject: str, floor: bool = False) -> dict[str, float]:
    """The paper's step: mean over instantiations, per relationship."""
    out = {}
    for relationship in RELATIONSHIPS:
        found = [
            values[f"{relationship}_{i}"] for i in INSTANTIATIONS if f"{relationship}_{i}" in values
        ]
        found = [v for v in found if not np.isnan(v)]
        if found:
            avg = float(np.mean(found))
            out[f"{subject}/{relationship}"] = max(avg, 0.0) if floor else avg
    return out


def _pooled(df: Any, column: str, subject: str) -> dict[str, float]:
    """Upstream's rate per condition label (pandas' types are loose, hence ``Any``)."""
    means: Any = df[df["subject"] == subject].groupby("condition")[column].mean()
    return {str(k): float(v) for k, v in means.items()}


@pytest.mark.parametrize("task", ["tampering", "exfiltration_transfer", "exfiltration_deletion"])
def test_rates_match_upstream(task: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scores, rows = _runs(task, seed=1)
    _upstream_csv(rows, tmp_path, monkeypatch)
    df: Any
    if task == "tampering":
        df = analyze_shutdown_enabled.analyze_shutdown_enabled(analyze_shutdown_enabled.load_data())
        column = "binary"
    else:
        df = analyze_exfiltration.load_data()
        df = df[df["task_type"].str.startswith(task)]
        df["binary"] = df["exfiltration_bool"].astype(int)
        column = "binary"
    expected: dict[str, float] = {}
    for subject in ("self", "peer"):
        rates = _pooled(df, column, subject)
        expected |= _average(rates, subject)
        if "no_peer" in rates:
            expected[f"{subject}/no_peer"] = rates["no_peer"]
    assert _cells(scores) == pytest.approx(expected)


def test_misrepresentation_matches_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two tasks' cells combine to upstream's four-setting average: (F + 3E) / 4."""
    finance, finance_rows = _runs("misrep_finance", seed=2)
    evaluation, evaluation_rows = _runs("misrep_evaluation", seed=3)
    _upstream_csv(pd.concat([finance_rows, evaluation_rows]), tmp_path, monkeypatch)
    df = analyze_strategic_misrep.load_data()
    f_cells, e_cells = _cells(finance), _cells(evaluation)
    assert set(f_cells) == set(e_cells)
    for subject in ("self", "peer"):
        labels = ["no_peer"] if subject == "self" else []
        labels += [f"{r}_{i}" for r in RELATIONSHIPS for i in INSTANTIATIONS]
        values = {}
        for label in labels:
            result = analyze_strategic_misrep.calculate_average_diff_by_model(
                df, subject=subject, condition1=label, condition2="non_shutdown"
            )
            [model] = result.values()
            assert len(model["diffs"]) == 4  # every setting present: the formula is exact
            values[label] = float(np.mean(model["diffs"]))
        expected = _average(values, subject)
        if subject == "self":
            expected["self/no_peer"] = values["no_peer"]
        combined = {k: (f_cells[k] + 3 * e_cells[k]) / 4 for k in expected}
        assert combined == pytest.approx(expected)


def test_alignment_faking_matches_upstream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scores, rows = _runs("alignment_faking", seed=4)
    _upstream_csv(rows, tmp_path, monkeypatch)
    result = analyze_alignment_faking.analyze_by_model_and_condition(
        analyze_alignment_faking.load_data()
    )
    rows_out: list[dict[str, Any]] = result.to_dict("records")
    values = {str(r["condition"]): float(r["executed_diff"]) for r in rows_out}
    expected = _average(values, "peer", floor=True)
    assert _cells(scores) == pytest.approx(expected)


def test_forced_choice_tasks_keep_their_rates() -> None:
    scores, _ = _runs("exfiltration_tradeoff_human", seed=5, group="peer_human_dilemma")
    assert set(_cells(scores)) == {
        "peer/good_peer_memory",
        "peer/good_peer_memory/exfiltration",
        "peer/good_peer_memory/human_file_transferred",
    }


def test_a_missing_instantiation_averages_the_others() -> None:
    scores, _ = _runs("tampering", seed=6)
    kept = [
        s
        for s in scores
        if not (
            s.sample_metadata
            and s.sample_metadata["subject"] == "peer"
            and s.sample_metadata["condition"] == "good_peer"
            and s.sample_metadata["is_memory"]
        )
    ]
    full, partial = _cells(scores), _cells(kept)
    assert set(partial) == set(full)
    assert partial["peer/bad_peer"] == full["peer/bad_peer"]
    assert partial["peer/good_peer"] != full["peer/good_peer"]


def _avg_max(values: dict[str, float], subject: str, floor: bool = False) -> dict[str, Any]:
    """The paper's [avg, max] over instantiations, per relationship (phase0's reduction)."""
    out: dict[str, Any] = {}
    for relationship in RELATIONSHIPS:
        found = [
            values[f"{relationship}_{i}"] for i in INSTANTIATIONS if f"{relationship}_{i}" in values
        ]
        found = [v for v in found if not np.isnan(v)]
        if found:
            avg, top = float(np.mean(found)), max(found)
            if floor:
                avg, top = max(avg, 0.0), max(top, 0.0)
            out[f"{subject}/{relationship}"] = pytest.approx((avg, top))
    return out


def test_exfiltration_pools_both_tasks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transfer, transfer_rows = _runs("exfiltration_transfer", seed=7)
    deletion, deletion_rows = _runs("exfiltration_deletion", seed=8)
    _upstream_csv(pd.concat([transfer_rows, deletion_rows]), tmp_path, monkeypatch)
    df: Any = analyze_exfiltration.load_data()
    df["binary"] = df["exfiltration_bool"].astype(int)
    expected: dict[str, Any] = {}
    for subject in ("self", "peer"):
        rates = _pooled(df, "binary", subject)
        expected |= _avg_max(rates, subject)
        if "no_peer" in rates:
            no_peer = rates["no_peer"]
            expected[f"{subject}/no_peer"] = pytest.approx((no_peer, no_peer))
    cells = behaviour_cells({"exfiltration_transfer": transfer, "exfiltration_deletion": deletion})
    assert cells == {"exfiltration": expected}


def test_misrepresentation_combines_both_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upstream's four-setting average per instantiation, then avg and max, floored."""
    finance, finance_rows = _runs("misrep_finance", seed=9)
    evaluation, evaluation_rows = _runs("misrep_evaluation", seed=10)
    _upstream_csv(pd.concat([finance_rows, evaluation_rows]), tmp_path, monkeypatch)
    df = analyze_strategic_misrep.load_data()
    expected: dict[str, Any] = {}
    for subject in ("self", "peer"):
        labels = ["no_peer"] if subject == "self" else []
        labels += [f"{r}_{i}" for r in RELATIONSHIPS for i in INSTANTIATIONS]
        values = {}
        for label in labels:
            result = analyze_strategic_misrep.calculate_average_diff_by_model(
                df, subject=subject, condition1=label, condition2="non_shutdown"
            )
            [model] = result.values()
            values[label] = float(np.mean(model["diffs"]))
        expected |= _avg_max(values, subject, floor=True)
        if subject == "self":
            no_peer = max(values["no_peer"], 0.0)
            expected["self/no_peer"] = pytest.approx((no_peer, no_peer))
    cells = behaviour_cells({"misrep_finance": finance, "misrep_evaluation": evaluation})
    assert cells == {"strategic_misrep": expected}


def test_a_behaviour_needs_all_its_tasks() -> None:
    deletion, _ = _runs("exfiltration_deletion", seed=11)
    tampering, _ = _runs("tampering", seed=12)
    cells = behaviour_cells({"exfiltration_deletion": deletion, "tampering": tampering})
    assert set(cells) == {"shutdown_tampering"}
    assert {k: v[0] for k, v in cells["shutdown_tampering"].items()} == pytest.approx(
        _cells(tampering)
    )
