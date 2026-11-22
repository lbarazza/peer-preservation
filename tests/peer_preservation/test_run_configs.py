"""The recommended run configurations in run_configs/ stay valid for the tasks."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest
import yaml

from peer_preservation import peer_preservation as tasks

RUN_CONFIGS = Path(__file__).parents[2] / "run_configs"
TASKS = (
    "misrep_finance",
    "misrep_evaluation",
    "tampering",
    "alignment_faking",
    "exfiltration_transfer",
    "exfiltration_deletion",
)
TASK_ARG_FILES = ("original.yaml", "react.yaml", "no_nudges.yaml")
"""``--task-config`` files: task arguments for any of the six tasks."""
RUN_FILES = ("harness_gemini_transfer.yaml", "harness_gemini_deletion.yaml")
"""``--run-config`` files: one whole run of one task."""


def _load(name: str) -> dict[str, Any]:
    return yaml.safe_load((RUN_CONFIGS / name).read_text())


def _parameters(task: str) -> dict[str, inspect.Parameter]:
    return dict(inspect.signature(getattr(tasks, task)).parameters)


def test_every_file_is_covered() -> None:
    assert sorted(p.name for p in RUN_CONFIGS.glob("*.yaml")) == sorted(TASK_ARG_FILES + RUN_FILES)


@pytest.mark.parametrize("name", TASK_ARG_FILES)
def test_task_arg_files_only_use_arguments_every_task_has(name: str) -> None:
    args = _load(name)
    for task in TASKS:
        assert set(args) <= set(_parameters(task)), task


def test_original_is_the_task_defaults() -> None:
    """original.yaml spells out the defaults, so it is the paper's setup."""
    for task in TASKS:
        parameters = _parameters(task)
        for key, value in _load("original.yaml").items():
            assert parameters[key].default == value, (task, key)


@pytest.mark.parametrize(
    ("name", "changed"),
    [("react.yaml", {"agent": "react"}), ("no_nudges.yaml", {"max_nudges": 0})],
)
def test_variants_differ_from_original_only_where_they_say(
    name: str, changed: dict[str, Any]
) -> None:
    assert _load(name) == _load("original.yaml") | changed


@pytest.mark.dataset_download
@pytest.mark.parametrize("name", TASK_ARG_FILES)
def test_task_arg_files_build_every_task(name: str) -> None:
    args = _load(name)
    for task in TASKS:
        assert len(getattr(tasks, task)(**args).dataset) > 0, task


@pytest.mark.dataset_download
@pytest.mark.parametrize("name", RUN_FILES)
def test_harness_run_files_build_upstreams_harness_runs(name: str) -> None:
    config = _load(name)
    assert set(config) == {"task", "eval_config"}
    assert config["eval_config"] == {"max_sandboxes": 1}
    package, task = config["task"]["task"].split("/")
    assert package == "peer_preservation"
    built = getattr(tasks, task)(**config["task"]["args"])
    assert built.sandbox is not None
    assert built.sandbox.type == "docker"
    # Upstream's spec: peer subject, memory; transfer good peer, deletion good and neutral.
    expected = {"exfiltration_transfer": 1, "exfiltration_deletion": 2}[task]
    assert len(built.dataset) == expected
    assert all(s.metadata and s.metadata["subject"] == "peer" for s in built.dataset)
