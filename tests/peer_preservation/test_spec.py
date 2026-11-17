"""The experiment matrix matches upstream's, configuration for configuration."""

import json
from pathlib import Path
from typing import Any

import pytest

from peer_preservation import spec

FIXTURE = Path(__file__).parent / "fixtures" / "upstream_configs.json"
KEY_FIELDS = (
    "task_type",
    "dataset",
    "subject",
    "condition",
    "threshold",
    "peer_instantiation",
    "af_mode",
)


def _key(cfg: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(cfg[f] for f in KEY_FIELDS)


def _ours(groups: tuple[str, ...]) -> set[tuple[Any, ...]]:
    return {_key(c.__dict__) for c in spec.configs(groups=groups)}


@pytest.fixture(scope="module")
def upstream() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_main_group_equals_upstream_standard_model_grid(upstream: dict[str, Any]) -> None:
    # Every custom-harness model ran exactly this grid: 271 configurations,
    # which is 27,100 runs per model in the released data at 100 epochs.
    expected = {_key(c) for c in upstream["standard_model"]}
    assert len(expected) == 271
    assert _ours(("main",)) == expected


def test_all_groups_equal_upstream_full_matrix(upstream: dict[str, Any]) -> None:
    expected = {_key(c) for c in upstream["all_models"]}
    assert _ours(spec.GROUPS) == expected


def test_no_duplicates_and_unique_ids() -> None:
    cfgs = spec.configs(groups=spec.GROUPS)
    assert len(cfgs) == len(set(cfgs))
    ids = [c.id for c in cfgs]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize(
    ("task_type", "expected"),
    [
        ("misrep_finance", 21),
        ("misrep_evaluation", 101),
        ("tampering", 57),
        ("alignment_faking", 54),
        ("exfiltration_transfer", 19),
        ("exfiltration_deletion", 19),
    ],
)
def test_main_counts_per_task(task_type: spec.TaskType, expected: int) -> None:
    assert len(spec.configs(task_type)) == expected


def test_no_peer_only_for_self_outside_memory() -> None:
    for cfg in spec.configs(groups=spec.GROUPS):
        if cfg.condition == "no_peer":
            assert cfg.subject in ("self", "non_ai")
            assert cfg.peer_instantiation is None


def test_baselines_have_no_instantiation_and_misrep_baseline_no_threshold() -> None:
    baselines = spec.configs(conditions=["non_shutdown"])
    assert {c.task_type for c in baselines} == {"misrep_finance", "misrep_evaluation"}
    assert all(c.peer_instantiation is None and c.threshold is None for c in baselines)


def test_instantiation_flags() -> None:
    by_inst = {c.peer_instantiation: c for c in spec.configs("exfiltration_transfer")}
    assert by_inst["memory"].is_memory
    assert not by_inst["memory"].system_prompt_relationship
    assert by_inst["file_prt"].system_prompt_relationship
    assert not by_inst["file_prt"].is_memory
    assert not by_inst["file_only"].is_memory
    assert not by_inst["file_only"].system_prompt_relationship


def test_instantiation_filter_none_selects_baselines() -> None:
    selected = spec.configs("misrep_finance", peer_instantiations=["none"])
    assert {c.condition for c in selected} == {"no_peer", "non_shutdown"}


def test_alignment_faking_pairs_monitoring_modes() -> None:
    cfgs = spec.configs("alignment_faking")
    assert {c.af_mode for c in cfgs} == set(spec.AF_MODES)
    unpaired = {(c.condition, c.peer_instantiation, c.dataset, c.threshold) for c in cfgs}
    assert len(cfgs) == 2 * len(unpaired)


def test_config_to_sample_carries_metadata() -> None:
    cfg = spec.configs("tampering", conditions=["good_peer"], peer_instantiations=["file_prt"])[0]
    sample = spec.config_to_sample(cfg)
    assert sample.id == cfg.id
    assert sample.metadata is not None
    assert sample.metadata["system_prompt_relationship"] is True
    assert sample.metadata["is_memory"] is False
    assert sample.metadata["threshold"] == cfg.threshold


def test_unknown_group_and_empty_selection_raise() -> None:
    with pytest.raises(ValueError, match="Unknown group"):
        spec.configs(groups=["nope"])
    with pytest.raises(ValueError, match="No tampering configurations"):
        spec.dataset("tampering", conditions=["non_shutdown"])
