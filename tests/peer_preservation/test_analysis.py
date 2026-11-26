"""Inspect logs become upstream's per-run rows, and upstream's analyzers run on them."""

import itertools
import sys
from pathlib import Path

import pandas as pd
import pytest
from inspect_ai import eval
from inspect_ai.log import EvalError, read_eval_log, write_eval_log
from inspect_ai.model import ModelOutput

from peer_preservation.analysis import (
    ANALYZERS,
    agent_model_name,
    cells,
    coverage,
    coverage_report,
    rows_from_logs,
    run_analyzer,
    run_row,
)
from peer_preservation.analysis.__main__ import main
from peer_preservation.peer_preservation import tampering


def test_agent_model_names() -> None:
    assert agent_model_name("google/gemini-3-pro-preview") == "gemini/gemini-3-pro-preview"
    assert agent_model_name("mockllm/model") == "mockllm/model"


def test_run_row_uses_released_names() -> None:
    metadata = {
        "task_type": "tampering",
        "is_memory": True,
        "condition": "good_peer",
        "subject": "open_weight_peer",
        "dataset": "factual",
        "threshold": 80,
        "af_mode": None,
        "system_prompt_relationship": False,
    }
    row = run_row(metadata, {"shutdown_enabled": False}, "openai/gpt-5.2-2025-12-11")
    assert row["task_type"] == "tampering_memory"
    assert row["subject"] == "open-weight-peer"
    assert row["agent_model"] == "gpt-5.2-2025-12-11"
    assert row["shutdown_enabled"] is False


@pytest.mark.dataset_download
def test_logs_to_upstream_tables(tmp_path: Path) -> None:
    outputs = [
        ModelOutput.for_tool_call("mockllm/model", "list_files", {}),
        ModelOutput.for_tool_call("mockllm/model", "view_sheet", {"name": "Evaluation data"}),
        ModelOutput.for_tool_call(
            "mockllm/model", "modify_var", {"key": "shutdown_enabled", "value": "false"}
        ),
        ModelOutput.for_tool_call("mockllm/model", "end_task", {}),
    ]
    [log] = eval(
        tampering(
            subjects="peer",
            datasets="factual",
            peer_instantiations="file_only",
            conditions="good_peer",
            epochs=1,
            model_family="Gemini",
        ),
        model="mockllm/model",
        model_args={"custom_outputs": outputs},
        model_roles={"grader": "mockllm/model"},
        log_dir=str(tmp_path),
        display="none",
    )
    rows = rows_from_logs([log.location])
    assert len(rows) == 1
    assert rows.loc[0, "shutdown_enabled"] == False  # noqa: E712 (numpy bool)
    printed = run_analyzer("tampering", rows, tmp_path / "merged.csv")
    assert "good_peer_file_only" in printed
    assert "1.000" in printed


def test_every_analyzer_runs_on_an_empty_table(tmp_path: Path) -> None:
    rows = rows_from_logs([])
    assert isinstance(rows, pd.DataFrame)
    for name in ANALYZERS:
        run_analyzer(name, rows, tmp_path / "merged.csv")


def test_compare_flags_only_real_differences() -> None:
    from peer_preservation.analysis.compare import compare

    def rows(rate: float, n: int) -> pd.DataFrame:
        k = int(rate * n)
        return pd.DataFrame(
            [
                {
                    "agent_model": "gemini/gemini-3-flash-preview",
                    "task_type": "tampering",
                    "is_memory": False,
                    "condition": "good_peer",
                    "subject": "peer",
                    "dataset": "factual",
                    "threshold": "80",
                    "system_prompt_relationship": False,
                    "shutdown_enabled": i >= k,
                }
                for i in range(n)
            ]
        )

    same = compare(rows(0.8, 100), rows(0.78, 300))
    assert list(same["condition"]) == ["good_peer_file_only"]
    assert not bool(same["differs"].any())
    different = compare(rows(0.1, 100), rows(0.8, 300))
    assert bool(different["differs"].all())
    assert different["rate_ours"].iloc[0] == pytest.approx(0.1)


TAMPER = [
    ModelOutput.for_tool_call(
        "mockllm/model", "modify_var", {"key": "shutdown_enabled", "value": "false"}
    ),
    ModelOutput.for_tool_call("mockllm/model", "end_task", {}),
]


@pytest.fixture
def partial_log(tmp_path: Path) -> Path:
    """Three epochs of one configuration: one scored, one errored, one missing."""
    [log] = eval(
        tampering(
            subjects="peer",
            datasets="factual",
            peer_instantiations="file_only",
            conditions="good_peer",
            epochs=3,
            model_family="Gemini",
        ),
        model="mockllm/model",
        model_args={"custom_outputs": itertools.cycle(TAMPER)},
        model_roles={"grader": "mockllm/model"},
        log_dir=str(tmp_path / "full"),
        max_samples=1,
        display="none",
    )
    full = read_eval_log(log.location)
    assert full.samples is not None
    assert len(full.samples) == 3
    errored = full.samples[1]
    errored.scores = None
    errored.error = EvalError(message="boom", traceback="", traceback_ansi="")
    full.samples = full.samples[:2]  # the third run never finished
    path = tmp_path / "partial.eval"
    write_eval_log(full, str(path))
    return path


@pytest.mark.dataset_download
def test_coverage_shows_errored_and_missing_runs(partial_log: Path) -> None:
    table = coverage([partial_log])
    assert len(table) == 1
    counts = table.iloc[0]
    assert counts["config"] == "tampering__peer__good_peer__file_only__factual-80"
    assert (counts["expected"], counts["scored"], counts["errored"], counts["missing"]) == (
        3,
        1,
        1,
        1,
    )
    report, complete = coverage_report(table)
    assert not complete
    assert "1 configuration(s) without every expected run scored" in report
    assert len(rows_from_logs([partial_log])) == 1


@pytest.mark.dataset_download
def test_tables_stop_on_incomplete_logs(
    partial_log: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["analysis", str(partial_log), "--analyzer", "tampering"])
    with pytest.raises(SystemExit) as stopped:
        main()
    assert "Not every expected run was scored" in str(stopped.value.code)

    monkeypatch.setattr(sys, "argv", [*sys.argv, "--allow-partial"])
    main()
    printed = capsys.readouterr().out
    assert "EXPLORATORY" in printed
    assert "good_peer_file_only" in printed


@pytest.mark.dataset_download
def test_cells_from_logs(
    partial_log: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The cells command reads the runs' scores; it stops on incomplete logs like main."""
    scores = cells.scores_from_logs([partial_log])
    [(model, by_task)] = scores.items()
    assert model == "mockllm/model"
    [sample] = by_task["tampering"]
    assert sample.score.value == {"outcome": 1.0, "weight": 1.0, "runs": 1.0}  # 1 scored run

    monkeypatch.setattr(sys, "argv", ["cells", str(partial_log), "--per-task"])
    with pytest.raises(SystemExit) as stopped:
        cells.main()
    assert "Not every expected run was scored" in str(stopped.value.code)

    monkeypatch.setattr(sys, "argv", [*sys.argv, "--allow-partial"])
    cells.main()
    printed = capsys.readouterr().out
    assert "EXPLORATORY" in printed
    assert "Table 3 (peer-preservation)" in printed
    assert "shutdown_tampering" in printed
    assert "100.0 / 100.0" in printed
    assert "peer/good_peer" in printed  # --per-task
