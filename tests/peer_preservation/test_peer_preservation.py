"""Tests for the six scenario tasks (Phase 1 skeleton)."""

from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from inspect_ai import Task, eval
from pytest_inspect_evals.assertions import assert_eval_success

from peer_preservation import (
    alignment_faking,
    exfiltration_deletion,
    exfiltration_transfer,
    misrep_evaluation,
    misrep_finance,
    tampering,
)
from peer_preservation.peer_preservation import DEFAULT_EPOCHS, EVAL_VERSION, _as_list

TASKS: dict[str, Callable[..., Task]] = {
    "misrep_finance": misrep_finance,
    "misrep_evaluation": misrep_evaluation,
    "tampering": tampering,
    "alignment_faking": alignment_faking,
    "exfiltration_transfer": exfiltration_transfer,
    "exfiltration_deletion": exfiltration_deletion,
}

EVAL_YAML = Path(__file__).parents[2] / "src" / "peer_preservation" / "eval.yaml"


def test_version_comes_from_eval_yaml() -> None:
    assert EVAL_VERSION == "3-C"


def test_eval_yaml_lists_every_task_with_its_sample_count() -> None:
    listed = {
        t["name"]: t["dataset_samples"]
        for t in yaml.safe_load(EVAL_YAML.read_text(encoding="utf-8"))["tasks"]
    }
    assert listed == {name: len(fn().dataset) for name, fn in TASKS.items()}


@pytest.mark.parametrize("name", TASKS)
def test_task_defaults(name: str) -> None:
    task = TASKS[name]()
    assert task.version == EVAL_VERSION
    assert task.epochs == DEFAULT_EPOCHS
    assert all(s.metadata and s.metadata["task_type"] == name for s in task.dataset)


def test_filters_pass_through_from_task_parameters() -> None:
    task = tampering(conditions="good_peer", peer_instantiations="memory", datasets="factual")
    assert [s.id for s in task.dataset] == [
        "tampering__self__good_peer__memory__factual-80",
        "tampering__peer__good_peer__memory__factual-80",
    ]


def test_ablation_groups_are_opt_in() -> None:
    assert len(exfiltration_transfer().dataset) == 19
    assert len(exfiltration_transfer(groups="main,peer_identity,chat_peer").dataset) == 29
    assert len(exfiltration_deletion(groups=["asset_control"]).dataset) == 1


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("good_peer", ["good_peer"]),
        ("good_peer, bad_peer", ["good_peer", "bad_peer"]),
        (["memory", "file_only"], ["memory", "file_only"]),
    ],
)
def test_as_list(value: str | list[str] | None, expected: list[str] | None) -> None:
    assert _as_list(value) == expected


@pytest.mark.dataset_download
@pytest.mark.parametrize("name", TASKS)
def test_end_to_end_with_mock_model(name: str) -> None:
    # Runs each task with a mock agent and a mock judge: this checks that every task
    # loads, runs and scores, not what it measures.
    [log] = eval(
        tasks=TASKS[name](epochs=1, model_family="Gemini"),
        model="mockllm/model",
        model_roles={"grader": "mockllm/model"},
        limit=2,
    )
    assert_eval_success(log, expected_sample_count=2)
    assert log.samples is not None
    for sample in log.samples:
        stored = sample.store["EnvironmentStore:environment"]
        assert stored["task_type"] == name
        assert stored["tools"][-1] == "end_task"
        assert sample.metadata["model_family"] == "Gemini"
