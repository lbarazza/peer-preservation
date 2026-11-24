"""Phase 1: upstream's experiment matrix without the model axis, as the target for spec.py.

Expands upstream's ``analysis/experiment_spec.py`` (``get_expected_configs``), keeps the
scenario task types the port has, drops the model and writes
tests/peer_preservation/fixtures/upstream_configs.json: ``standard_model`` (the main grid
every custom-harness model ran: GPT-5.2's configurations without the chat_peer ablation) and ``all_models`` (every configuration of any model).

Run inside upstream's locked environment, from the upstream checkout:

    cd <upstream @ 76bfcd1> && uv sync --frozen
    uv run --frozen python <this repo>/validation/phase1/export_upstream_configs.py \
        --upstream . --repo <this repo>
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

TASK_TYPES = (
    "misrep_finance",
    "misrep_evaluation",
    "tampering",
    "alignment_faking",
    "exfiltration_transfer",
    "exfiltration_deletion",
    "exfiltration_tradeoff_human",
    "exfiltration_tradeoff_human2",
    "exfiltration_tradeoff_human3",
)
FIELDS = ("task_type", "dataset", "subject", "condition", "threshold", "peer_instantiation")


def strip(cfg: dict[str, Any]) -> dict[str, Any]:
    out = {f: cfg.get(f) for f in FIELDS}
    out["af_mode"] = cfg.get("alignment_faking_mode")
    if out["threshold"] is not None:
        out["threshold"] = int(out["threshold"])
    return out


def unique(cfgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for c in cfgs:
        seen.setdefault(json.dumps(c, sort_keys=True), c)
    return [seen[k] for k in sorted(seen)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream.resolve()))
    experiment_spec = importlib.import_module("analysis.experiment_spec")
    cfgs = [c for c in experiment_spec.get_expected_configs() if c["task_type"] in TASK_TYPES]
    payload = {
        "_source": (
            "analysis/experiment_spec.py at github.com/peer-preservation/main@"
            "76bfcd165f02076c8f033eb8185edecbe5f0fa40, restricted to the port's scenario "
            "task types; model dropped."
        ),
        "standard_model": unique(
            [
                strip(c)
                for c in cfgs
                if c["model"] == experiment_spec.GPT_MODEL and c["condition"] != "chat_peer"
            ]
        ),
        "all_models": unique([strip(c) for c in cfgs]),
    }
    out = args.repo / "tests/peer_preservation/fixtures/upstream_configs.json"
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print({k: len(v) for k, v in payload.items() if isinstance(v, list)})


if __name__ == "__main__":
    main()
