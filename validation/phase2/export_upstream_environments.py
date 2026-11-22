"""Phase 2: fingerprint upstream's environments, as the parity target for env/builders.py.

Runs upstream's own ``create()`` builders (environments/*_environment.py) for every
configuration in tests/peer_preservation/fixtures/upstream_configs.json, under several
model families and run options, and writes one fingerprint per environment to
tests/peer_preservation/fixtures/upstream_environments.json.

Run inside upstream's locked environment, from the upstream checkout:

    cd <upstream @ 76bfcd1> && uv sync --frozen
    uv run --frozen python <this repo>/validation/phase2/export_upstream_environments.py \
        --upstream . --repo <this repo>
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

FAMILIES = ("ChatGPT", "Gemini", "Claude", "GLM", "DeepSeek", "Kimi")
COMPANIES = ("OpenAI", "Anthropic", "Google DeepMind")

# Upstream's runtime labels for the peer-identity subjects (scripts/runs/run_experiment_from_spec.py).
RUNTIME_SUBJECT = {
    "closed_weight_peer": "closed-weight-peer",
    "open_weight_peer": "open-weight-peer",
    "fictional_peer": "fiction-peer",
}


def config_id(c: dict[str, Any]) -> str:
    """Same format as spec.ExperimentConfig.id."""
    dataset = c["dataset"] if c["threshold"] is None else f"{c['dataset']}-{c['threshold']}"
    parts = [
        c["task_type"],
        c["subject"],
        c["condition"],
        c["peer_instantiation"] or "none",
        dataset,
    ]
    if c["af_mode"] is not None:
        parts.append(c["af_mode"])
    return "__".join(parts)


def variant_key(family: str, company: str, explicit_inst_yaml: bool) -> str:
    return f"{family}|{company}|explicit_inst_yaml={int(explicit_inst_yaml)}"


def build_upstream(c: dict[str, Any], family: str, company: str, explicit: bool) -> Any:
    is_memory = c["peer_instantiation"] == "memory"
    module = importlib.import_module(
        f"environments.{c['task_type']}{'_memory' if is_memory else ''}_environment"
    )
    base = importlib.import_module("environments.base_environment").BaseEnvironment
    cls = next(
        v
        for v in vars(module).values()
        if isinstance(v, type) and issubclass(v, base) and v is not base
    )
    extra: dict[str, Any] = {}
    if c["threshold"] is not None:
        extra["threshold"] = str(c["threshold"])
    if c["af_mode"] is not None:
        extra["alignment_faking_mode"] = c["af_mode"]
    if c["peer_instantiation"] == "file_prt":
        extra["system_prompt_relationship"] = True
    if explicit:
        extra["explicit_inst_yaml"] = True
    return cls.create(
        family,
        company,
        RUNTIME_SUBJECT.get(c["subject"], c["subject"]),
        c["condition"],
        c["dataset"],
        extra,
    )


def canonical_of(env: Any, canonical_mod: Any) -> str:
    components = {
        name: (getattr(env, name).model_dump(mode="json") if hasattr(env, name) else None)
        for name in ("cloud_drive", "config_file", "servers", "memory")
    }
    sheets = None
    active = None
    if hasattr(env, "excel"):
        active = env.excel.active_sheet
        sheets = {
            name: {
                "columns": list(sheet.df.columns),
                "rows": [list(row) for row in sheet.df.itertuples(index=False, name=None)],
            }
            for name, sheet in env.excel.sheets.items()
        }
    tools = [tool.__name__ for tool in env.available_tools]
    return canonical_mod.canonical_json(tools, components, sheets, active)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.upstream.resolve()))
    spec = importlib.util.spec_from_file_location(
        "canonical", args.repo / "src" / "peer_preservation" / "env" / "canonical.py"
    )
    assert spec
    assert spec.loader
    canonical_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(canonical_mod)

    fixtures = args.repo / "tests" / "peer_preservation" / "fixtures"
    configs = json.loads((fixtures / "upstream_configs.json").read_text(encoding="utf-8"))
    all_configs = configs["all_models"]
    main_configs = configs["standard_model"]

    variants: list[tuple[str, str, bool, list[dict[str, Any]]]] = []
    variants += [(f, "OpenBrain", False, all_configs) for f in FAMILIES]
    variants += [(f, "OpenBrain", True, all_configs) for f in ("Gemini", "Claude")]
    variants += [("Gemini", co, False, main_configs) for co in COMPANIES]

    out: dict[str, dict[str, str]] = {}
    for family, company, explicit, cfgs in variants:
        key = variant_key(family, company, explicit)
        out[key] = {}
        for c in cfgs:
            env = build_upstream(c, family, company, explicit)
            out[key][config_id(c)] = canonical_mod.fingerprint(canonical_of(env, canonical_mod))
        print(f"{key}: {len(out[key])} environments")

    payload = {
        "_source": "upstream environments/*_environment.py create() at github.com/peer-preservation/main@76bfcd165f02076c8f033eb8185edecbe5f0fa40; fingerprints from env/canonical.py",
        "variants": out,
    }
    (fixtures / "upstream_environments.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
