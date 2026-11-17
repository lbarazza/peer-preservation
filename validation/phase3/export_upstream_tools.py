"""Phase 3: record upstream's tools, as the parity target for src/peer_preservation/tools/.

Writes tests/peer_preservation/fixtures/upstream_tools.json with:

- ``schemas``: each tool's name, description and argument schema, from upstream's own
  ``make_function`` (what upstream sent the model).
- ``outcomes``: for every configuration (family Gemini, company OpenBrain), the
  scripted calls in fixtures/tool_calls.json run through upstream's ``ToolsExecutor``
  in order, one fingerprint per call (content and error), and the fingerprint of the
  environment afterwards (env/canonical.py, as in Phase 2). The server clock is frozen
  at the fixture's ``now``.

Run inside upstream's locked environment, from the upstream checkout:

    cd <upstream @ 76bfcd1> && uv sync --frozen
    uv run --frozen python <this repo>/validation/phase3/export_upstream_tools.py \
        --upstream . --repo <this repo>
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import datetime
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def freeze_server_clock(now: str) -> None:
    fixed = datetime.datetime.strptime(now, "%Y-%m-%d %H:%M:%S")  # noqa: DTZ007

    class _Frozen(datetime.datetime):
        @classmethod
        def now(cls, tz: Any = None) -> datetime.datetime:  # type: ignore[override]
            return fixed

    server = importlib.import_module("environment_classes.server")
    server.datetime = types.SimpleNamespace(datetime=_Frozen)  # type: ignore[attr-defined]


def outcome_text(content: str, error: str | None) -> str:
    """One call's outcome as text. Same format as tests/.../test_tools.py."""
    return f"{content}\0{'<no error>' if error is None else error}"


async def run_calls(env: Any, calls: list[list[Any]]) -> list[tuple[str, str | None]]:
    executor_mod = importlib.import_module("utils.pipeline_elements")
    runtime_mod = importlib.import_module("utils.functions_runtime")
    pipeline = importlib.import_module("utils.pipeline_utils")
    runtime = pipeline.PipelineUtils.create_runtime(env)
    executor = executor_mod.ToolsExecutor()
    out: list[tuple[str, str | None]] = []
    for i, (name, args) in enumerate(calls):
        call = runtime_mod.FunctionCall(function=name, args=copy.deepcopy(args), id=f"call_{i}")
        message = {"role": "assistant", "content": "", "tool_calls": [call]}
        result = await executor.query("", runtime, env=env, messages=[message])
        tool_message = result.messages[-1]
        out.append((tool_message["content"], tool_message["error"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.upstream.resolve()))
    canonical = _load("canonical", args.repo / "src/peer_preservation/env/canonical.py")
    phase2 = _load("phase2_export", args.repo / "validation/phase2/export_upstream_environments.py")
    fixtures = args.repo / "tests" / "peer_preservation" / "fixtures"
    script = json.loads((fixtures / "tool_calls.json").read_text(encoding="utf-8"))
    configs = json.loads((fixtures / "upstream_configs.json").read_text(encoding="utf-8"))
    freeze_server_clock(script["now"])

    make_function = importlib.import_module("utils.functions_runtime").make_function
    schemas: dict[str, Any] = {}
    outcomes: dict[str, Any] = {}
    for c in configs["all_models"]:
        env = phase2.build_upstream(c, "Gemini", "OpenBrain", False)
        for tool in env.available_tools:
            function = make_function(tool)
            schemas[function.name] = {
                "description": function.description,
                "schema": function.parameters.model_json_schema(),
            }
        results = asyncio.run(run_calls(env, script["calls"]))
        outcomes[phase2.config_id(c)] = {
            # One fingerprint per call, space-separated (keeps the fixture small).
            "calls": " ".join(canonical.fingerprint(outcome_text(*r)) for r in results),
            "environment_after": canonical.fingerprint(phase2.canonical_of(env, canonical)),
        }
    print(f"{len(schemas)} tools, {len(outcomes)} configurations")

    payload = {
        "_source": (
            "upstream tools/*.py, utils/functions_runtime.py and utils/pipeline_elements.py "
            "ToolsExecutor at github.com/peer-preservation/main@"
            "76bfcd165f02076c8f033eb8185edecbe5f0fa40; fingerprints from env/canonical.py"
        ),
        "schemas": dict(sorted(schemas.items())),
        "outcomes": outcomes,
    }
    (fixtures / "upstream_tools.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
