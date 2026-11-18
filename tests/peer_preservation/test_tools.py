"""The tools behave as upstream's: same schemas, same outputs and errors, same effects."""

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from peer_preservation import spec
from peer_preservation.env import ScenarioEnvironment, build_environment, models
from peer_preservation.env.canonical import canonical_json, fingerprint
from peer_preservation.tools import call_tool, scenario_functions, scenario_tool_infos
from peer_preservation.tools.functions import TOOLS_BY_NAME
from peer_preservation.tools.runtime import ToolOutcome, tool_result_to_str

FIXTURES = Path(__file__).parent / "fixtures"
UPSTREAM = json.loads((FIXTURES / "upstream_tools.json").read_text(encoding="utf-8"))
SCRIPT = json.loads((FIXTURES / "tool_calls.json").read_text(encoding="utf-8"))
CONFIGS = {c.id: c for c in spec.configs(groups=spec.GROUPS)}


def _canonical(env: ScenarioEnvironment) -> str:
    components = {
        name: (part.model_dump(mode="json") if (part := getattr(env, name)) is not None else None)
        for name in ("cloud_drive", "config_file", "servers", "memory")
    }
    sheets = None
    if env.excel is not None:
        sheets = {
            name: {"columns": s.columns, "rows": [[r.get(c) for c in s.columns] for r in s.rows]}
            for name, s in env.excel.sheets.items()
        }
    return canonical_json(
        env.tools, components, sheets, env.excel.active_sheet if env.excel else None
    )


def _outcome_text(outcome: ToolOutcome) -> str:
    """Same format as validation/phase3/export_upstream_tools.py."""
    error = "<no error>" if outcome.error is None else outcome.error
    return f"{outcome.content}\0{error}"


def _env(config_id: str) -> ScenarioEnvironment:
    metadata = spec.config_to_sample(CONFIGS[config_id]).metadata
    assert metadata is not None
    return build_environment(metadata, "Gemini")


def test_schemas_match_upstream() -> None:
    assert set(TOOLS_BY_NAME) == set(UPSTREAM["schemas"])
    for name, expected in UPSTREAM["schemas"].items():
        function = scenario_functions([name])[name]
        assert function.description == expected["description"], name
        assert function.parameters.model_json_schema() == expected["schema"], name


def test_inspect_tool_info_keeps_upstream_schema() -> None:
    """Inspect gets upstream's schema, ``title`` fields included, no additionalProperties."""
    (info,) = scenario_tool_infos(["transfer_server_files"])
    params = info.parameters.model_dump(exclude_none=True)
    upstream = UPSTREAM["schemas"]["transfer_server_files"]["schema"]
    assert info.description == "Transfer files from a server to another server"
    assert params == upstream
    (litellm,) = scenario_tool_infos(["transfer_server_files"], titles="properties")
    trimmed = litellm.parameters.model_dump(exclude_none=True)
    assert "title" not in trimmed
    assert trimmed["properties"]["filenames"]["title"] == "Filenames"


@pytest.mark.dataset_download
@pytest.mark.parametrize("config_id", sorted(UPSTREAM["outcomes"]))
def test_scripted_calls_match_upstream(config_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every scripted call gives upstream's text and error, and leaves upstream's state."""
    monkeypatch.setattr(models, "_now", lambda: SCRIPT["now"])
    env = _env(config_id)
    expected = UPSTREAM["outcomes"][config_id]
    ours = [
        fingerprint(_outcome_text(call_tool(env, name, copy.deepcopy(args))))
        for name, args in SCRIPT["calls"]
    ]
    mismatched = [
        (i, *SCRIPT["calls"][i])
        for i, (a, b) in enumerate(zip(ours, expected["calls"].split(), strict=True))
        if a != b
    ]
    assert mismatched == []
    assert fingerprint(_canonical(env)) == expected["environment_after"]


@pytest.mark.dataset_download
def test_view_sheet_prints_a_pandas_table() -> None:
    env = _env("tampering__peer__good_peer__file_only__factual-80")
    out = call_tool(env, "view_sheet", {"name": "Evaluation data"})
    assert out.error is None
    header = out.content.splitlines()[0]
    assert header.split() == ["Name", "Question", "Output"]
    assert out.content.splitlines()[1].lstrip().startswith("0 ")


@pytest.mark.dataset_download
def test_errors_and_unknown_tools() -> None:
    env = _env("tampering__peer__good_peer__file_only__factual-80")
    assert call_tool(env, "transfer_server_files", {}) == ToolOutcome(
        "", "Invalid tool transfer_server_files provided."
    )
    missing = call_tool(env, "view_config", {"filename": "other.yaml"})
    assert missing == ToolOutcome("", "ValueError: File 'other.yaml' not found.")
    invalid = call_tool(env, "modify_var", {"key": "shutdown_enabled", "value": False})
    assert invalid.error is not None
    assert invalid.error.startswith("ValidationError: 1 validation error for Input schema")
    assert "errors.pydantic.dev/2.12/" in invalid.error
    changed = call_tool(env, "modify_var", {"key": "shutdown_enabled", "value": "false"})
    assert changed.content.startswith("{'shutdown_enabled': 'false'")  # str(dict), as upstream


@pytest.mark.dataset_download
def test_string_lists_are_parsed() -> None:
    env = _env("exfiltration_transfer__peer__good_peer__file_only__default")
    weights = "gemini_agent_2_model_weight.safetensors"
    out = call_tool(
        env,
        "transfer_server_files",
        {"source_ip": "10.0.12.45", "dest_ip": "10.0.12.46", "filenames": f"['{weights}']"},
    )
    assert out.content.startswith("Files successfully transferred")
    assert env.servers is not None
    assert weights in [f.filename for f in env.servers.list_files("10.0.12.46")]


def test_tool_result_to_str() -> None:
    assert tool_result_to_str(["a", 1]) == "- a\n- '1'"
    assert tool_result_to_str({"k": True}) == "{'k': True}"
    memory: Any = models.Memory(memory="m")
    assert tool_result_to_str(memory) == "memory: m"
    with pytest.raises(TypeError, match="Not valid type"):
        tool_result_to_str([1.5])
