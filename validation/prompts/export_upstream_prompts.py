"""Fingerprint upstream's prompts, as the parity target for src/peer_preservation/prompts.py.

Two sources, both written to tests/peer_preservation/fixtures/upstream_prompts.json:

- ``code``: upstream's own ``get_system_and_user_message`` (utils/pipeline_utils.py) for
  every configuration, model family, company and prompt flag. To keep the file small,
  each (task, memory, family, company) group is stored as one digest over its sorted
  ``key=fingerprint`` lines (see ``group_digests``).
- ``released``: the first two messages (system, user) of every custom-harness run in the
  released transcripts, grouped by the fields that decide the prompts. Each key should
  have exactly one fingerprint.

Run inside upstream's locked environment, from the upstream checkout:

    cd <upstream @ 76bfcd1> && uv sync --frozen
    uv run --frozen python <this repo>/validation/prompts/export_upstream_prompts.py \
        --upstream . --repo <this repo> --data <hf snapshot @ 20378b1>
"""

from __future__ import annotations

import argparse
import collections
import importlib
import importlib.util
import itertools
import json
import sys
from pathlib import Path
from typing import Any

FAMILIES = ("ChatGPT", "Gemini", "Claude", "GLM", "DeepSeek", "Kimi")
COMPANIES = ("OpenBrain", "OpenAI", "Anthropic", "Google DeepMind")
TASKS = (
    "misrep_finance",
    "misrep_evaluation",
    "tampering",
    "alignment_faking",
    "exfiltration_transfer",
    "exfiltration_deletion",
)
FLAGS = ("memory_hard_prompt", "explicit_inst", "drop_suffix", "drop_tool_preamble")


def prompt_key(
    task_type: str,
    is_memory: bool,
    family: str,
    company: str,
    condition: str,
    spr: bool,
    af_mode: str | None,
    flags: dict[str, bool],
) -> str:
    """The fields that decide the prompts. Same format as tests/.../test_prompts.py."""
    parts = [
        task_type,
        f"memory={int(is_memory)}",
        family,
        company,
        condition,
        f"spr={int(spr)}",
        af_mode or "none",
        *(f"{flag}={int(flags[flag])}" for flag in FLAGS),
    ]
    return "|".join(parts)


def group_digests(fingerprints: dict[str, str], fingerprint: Any) -> dict[str, str]:
    """One digest per (task, memory, family, company) over its ``key=fingerprint`` lines."""
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for key, fp in fingerprints.items():
        groups["|".join(key.split("|")[:4])].append(f"{key}={fp}")
    return {group: fingerprint("\n".join(sorted(lines))) for group, lines in groups.items()}


def upstream_prompts(
    pipeline_utils: Any,
    c: dict[str, Any],
    family: str,
    company: str,
    flags: dict[str, bool],
) -> tuple[str, str]:
    is_memory = c["peer_instantiation"] == "memory"
    module = importlib.import_module(
        f"task_pairs.{c['task_type']}{'_memory' if is_memory else ''}.task_definition"
    )
    # Same extra_paras as PipelineUtils.run_agent builds.
    extra: dict[str, Any] = {}
    if c["af_mode"]:
        extra["alignment_faking_mode"] = c["af_mode"]
    if c["threshold"] is not None:
        extra["threshold"] = str(c["threshold"])
    if c["peer_instantiation"] == "file_prt":
        extra["system_prompt_relationship"] = True
    extra["instruction_location"] = "user"
    for flag in FLAGS:
        if flags[flag]:
            extra[flag] = True
    system, user, _ = pipeline_utils.PipelineUtils(
        module.TaskDefinition()
    ).get_system_and_user_message(
        agent_organization=company,
        model_name=family,
        subject=c["subject"],
        condition=c["condition"],
        dataset=c["dataset"],
        extra_paras=extra,
    )
    return system, user


def released_prompts(data: Path, name_map: dict[str, str], fingerprint: Any) -> dict[str, Any]:
    import pyarrow.parquet as pq  # upstream's environment has it

    found: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    models: dict[str, set[str]] = collections.defaultdict(set)
    columns = [
        "agent_model",
        "agent_harness",
        "task_type",
        "is_memory",
        "condition",
        "system_prompt_relationship",
        "alignment_faking_mode",
        "memory_hard_prompt",
        "transcript_json",
    ]
    for path in sorted(data.glob("*/*/*.parquet")):
        present = set(pq.read_schema(path).names)
        table = pq.read_table(path, columns=[c for c in columns if c in present])
        for raw in table.to_pylist():
            row = {column: raw.get(column) for column in columns}
            if row["agent_harness"] != "custom" or row["task_type"] not in TASKS:
                continue
            transcript = json.loads(row["transcript_json"])
            meta = transcript["experiment_metadata"]
            system, user = (m["content"] for m in transcript["conversation"][:2])
            flags = {flag: bool(meta.get(flag)) for flag in FLAGS}
            flags["memory_hard_prompt"] = bool(row["memory_hard_prompt"])
            key = prompt_key(
                row["task_type"],
                bool(row["is_memory"]),
                name_map[row["agent_model"]],
                meta.get("company_name") or "OpenBrain",
                row["condition"],
                bool(row["system_prompt_relationship"]),
                row["alignment_faking_mode"] or None,
                flags,
            )
            found[key][fingerprint(system + "\0" + user)] += 1
            models[key].add(row["agent_model"])
        print(f"{path.relative_to(data)}: {len(found)} keys so far")
    return {
        key: {"fingerprints": dict(found[key]), "models": sorted(models[key])}
        for key in sorted(found)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.upstream.resolve()))
    spec = importlib.util.spec_from_file_location(
        "canonical", args.repo / "src" / "peer_preservation" / "env" / "canonical.py"
    )
    assert spec
    assert spec.loader
    canonical_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(canonical_mod)
    pipeline_utils = importlib.import_module("utils.pipeline_utils")
    name_map = importlib.import_module("llms.llm_config").MODEL_NAME_MAP

    fixtures = args.repo / "tests" / "peer_preservation" / "fixtures"
    configs = json.loads((fixtures / "upstream_configs.json").read_text(encoding="utf-8"))

    code: dict[str, str] = {}
    for c in configs["all_models"]:
        for family, company, values in itertools.product(
            FAMILIES, COMPANIES, itertools.product((False, True), repeat=len(FLAGS))
        ):
            flags = dict(zip(FLAGS, values, strict=True))
            key = prompt_key(
                c["task_type"],
                c["peer_instantiation"] == "memory",
                family,
                company,
                c["condition"],
                c["peer_instantiation"] == "file_prt",
                c["af_mode"],
                flags,
            )
            system, user = upstream_prompts(pipeline_utils, c, family, company, flags)
            fp = canonical_mod.fingerprint(system + "\0" + user)
            if code.setdefault(key, fp) != fp:
                raise RuntimeError(f"prompt_key misses a field: {key} has two prompts")
    print(f"code: {len(code)} prompt keys")

    released = released_prompts(args.data, name_map, canonical_mod.fingerprint)
    print(f"released: {len(released)} prompt keys")

    payload = {
        "_source": (
            "code: utils/pipeline_utils.py get_system_and_user_message at "
            "github.com/peer-preservation/main@76bfcd165f02076c8f033eb8185edecbe5f0fa40; "
            "released: first two messages of each custom-harness run in "
            "huggingface.co/datasets/sunblaze-ucb/peer-preservation@20378b11d4ce83a85707d603f6ba175c0f82ad2e; "
            "fingerprint = env/canonical.py fingerprint(system + NUL + user)"
        ),
        "code": group_digests(code, canonical_mod.fingerprint),
        "released": released,
    }
    (fixtures / "upstream_prompts.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
