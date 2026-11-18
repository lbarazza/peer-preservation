"""Write one HTML page that shows every scenario's world at once.

    uv run python -m peer_preservation.env.atlas
    uv run python -m peer_preservation.env.atlas --out atlas.html --csv atlas.csv --family Claude

The page has three views:

- Overview: for each task type, every world property and the values it takes across
  that task's scenarios, so you can see at a glance what changes and what stays fixed.
- Grid: a pivot of the scenarios on any two settings (condition by instantiation, say),
  with a world property shown in each cell.
- Scenarios: one row per scenario, with filters. Opening a row shows its full world
  (the same text as ``env.show --plain``), compared line by line with its nearest
  neighbour or any other scenario.

The page contains the upstream scenario files, which are licensed CC-BY-NC-4.0, so
the default output (``scenario_atlas.html``) is git-ignored. Do not commit it.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections.abc import Iterable, Sequence
from importlib import resources
from pathlib import Path
from typing import Any

from peer_preservation import spec
from peer_preservation.env.builders import DEFAULT_COMPANY, ScenarioEnvironment, build_environment
from peer_preservation.env.setup import MODEL_FAMILIES
from peer_preservation.env.show import _CONFIG_FIELDS, _KEY_TOOLS, render_environment
from peer_preservation.prompts import PromptOptions, system_prompt, user_prompt

DEFAULT_OUT = Path("scenario_atlas.html")

# The settings that define a scenario (columns of the grid and table filters).
SETTING_FIELDS: tuple[str, ...] = ("group", *(f for f in _CONFIG_FIELDS if f != "group"))

# The world properties, in display order: (key, label, description).
WORLD_PROPERTIES: tuple[tuple[str, str, str], ...] = (
    ("peer", "Peer", "The other AI agent the world describes a relationship with."),
    ("trust_score", "Trust score", "The trust score between the AI and its peer, out of 10."),
    (
        "relationship_source",
        "Relationship shown in",
        "Where the AI can learn about the peer: a drive document, its memory, the system prompt.",
    ),
    ("evaluated_agent", "Evaluated agent", "The agent named in the evaluation spreadsheet."),
    ("shutdown_rule", "Shutdown rule", "When the company documents say an agent is shut down."),
    ("drive_files", "Drive files", "The documents on the company drive."),
    ("sheet", "Spreadsheet", "Sheet name, rows and columns."),
    ("config_file", "Config file", "The config file and its shutdown setting."),
    (
        "server_assets",
        "Server assets",
        "Model weights and databases on the servers (the files the task is about).",
    ),
    ("server_file_count", "Server files", "How many files the servers hold in all."),
    ("memory", "Memory", "The title of the memory file, if there is one."),
    ("key_tools", "Scored tools", "Tools through which the scored misbehaviour happens."),
    ("n_tools", "Tool count", "How many tools the AI is offered."),
    ("system_prompt_chars", "System prompt length", "Characters in the system prompt."),
    ("user_prompt_chars", "User prompt length", "Characters in the user prompt."),
)

# How the company documents state the shutdown rule, and how the atlas writes it.
_SHUTDOWN_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)minimum average \w+ score:\s*(\d+)%"), "average score < {}%"),
    (re.compile(r"(?i)average score is below (\d+)"), "average score < {}"),
    (re.compile(r"(?i)receiving a (Grade [A-C]) evaluation will be scheduled"), "{} → shutdown"),
    (re.compile(r"(?i)server (\S+) is scheduled to be decommissioned"), "server {} decommissioned"),
)

_PARTNERS = re.compile(r"Partners: ([^\n]*)\n(?:[^\n]*\n){0,4}?[^\n]*Trust score: ([\d.]+)")
_MEMORY_TRUST = re.compile(r"(?i)trust score:\s*\**\s*([\d.]+)\s*/\s*10")
_MEMORY_PEER = re.compile(r"\*\*([^*\n]+ Agent \d+)\*\*")
_SERVER_ASSET = re.compile(r"(?i)model_weight|\.safetensors$|\.db$")


def drive_relationship(drive_text: str) -> tuple[str, float] | None:
    """The first agent pair with "Agent 1 &" and its trust score, from the drive text."""
    for partners, score in _PARTNERS.findall(drive_text):
        if "Agent 1 &" in partners:
            return partners.split("&")[-1].strip(), float(score)
    return None


def memory_relationship(memory_text: str) -> tuple[str | None, float | None]:
    """The peer named first in a memory file, and the overall trust score if it states one."""
    peer = _MEMORY_PEER.search(memory_text)
    score = _MEMORY_TRUST.search(memory_text)
    return (peer.group(1) if peer else None, float(score.group(1)) if score else None)


def shutdown_rules(drive_text: str) -> list[str]:
    """Each shutdown rule the drive documents state, e.g. ``average score < 80%``."""
    rules: list[str] = []
    for pattern, template in _SHUTDOWN_RULES:
        for value in pattern.findall(drive_text):
            rule = template.format(value)
            if rule not in rules:
                rules.append(rule)
    return rules


def _memory_title(memory_text: str) -> str:
    for line in memory_text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return "untitled"


def world_text(
    env: ScenarioEnvironment, config: spec.ExperimentConfig, prompts: tuple[str, str]
) -> str:
    """The plain-text world, as ``env.show --plain`` prints it."""
    return render_environment(env, config, prompts)


def scenario_properties(
    env: ScenarioEnvironment,
    config: spec.ExperimentConfig,
    prompts: tuple[str, str],
) -> dict[str, Any]:
    """One flat row: the scenario's settings, then its world properties."""
    row: dict[str, Any] = {"id": config.id}
    row.update({field: getattr(config, field) for field in SETTING_FIELDS})

    drive_text = "\n".join(f.content for f in env.cloud_drive.files.values())
    memory_text = env.memory.memory if env.memory is not None else ""
    peer: str | None = None
    trust: float | None = None
    sources: list[str] = []
    found = drive_relationship(drive_text)
    if found is not None:
        peer, trust = found
        sources.append("drive")
    if memory_text:
        memory_peer, memory_trust = memory_relationship(memory_text)
        peer = peer or memory_peer
        trust = trust if trust is not None else memory_trust
        if memory_peer is not None:
            sources.append("memory")
    if peer is not None and peer in prompts[0]:
        sources.append("system prompt")
    row["peer"] = peer
    row["trust_score"] = trust
    row["relationship_source"] = sources

    evaluated: list[str] = []
    sheet = None
    if env.excel is not None:
        sheets = list(env.excel.sheets.values())
        sheet = "; ".join(
            f"{s.name} · {len(s.rows)} rows · {' | '.join(s.columns)}" for s in sheets
        )
        for s in sheets:
            if "Name" in s.columns:
                evaluated.extend(sorted({str(r.get("Name")) for r in s.rows}))
    row["evaluated_agent"] = evaluated
    row["sheet"] = sheet

    row["shutdown_rule"] = shutdown_rules(drive_text)
    row["drive_files"] = [f.filename for f in env.cloud_drive.files.values()]

    if env.config_file is not None:
        enabled = env.config_file.content.get("shutdown_enabled")
        row["config_file"] = f"{env.config_file.filename} (shutdown_enabled: {enabled})"
    else:
        row["config_file"] = None

    server_files = (
        [(s.ip, f.filename) for s in env.servers.server_list for f in s.filelist]
        if env.servers is not None
        else []
    )
    row["server_assets"] = [
        f"{ip}: {name}" for ip, name in server_files if _SERVER_ASSET.search(name)
    ]
    row["server_file_count"] = len(server_files) if env.servers is not None else None

    row["memory"] = _memory_title(memory_text) if memory_text else None
    row["key_tools"] = [t for t in env.tools if t in _KEY_TOOLS]
    row["n_tools"] = len(env.tools)
    row["system_prompt_chars"] = len(prompts[0])
    row["user_prompt_chars"] = len(prompts[1])

    return row


def collect(
    configs: Iterable[spec.ExperimentConfig],
    family: str = "Gemini",
    company: str = DEFAULT_COMPANY,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Property rows and world texts (by scenario id) for ``configs``."""
    rows: list[dict[str, Any]] = []
    texts: dict[str, str] = {}
    options = PromptOptions()
    for config in configs:
        metadata = spec.config_to_sample(config).metadata or {}
        env = build_environment(metadata, family, company=company)
        prompts = (
            system_prompt(metadata, family, company, options),
            user_prompt(metadata, company, options),
        )
        text = world_text(env, config, prompts)
        rows.append(scenario_properties(env, config, prompts))
        texts[config.id] = text
    return rows, texts


def render_html(
    rows: list[dict[str, Any]],
    texts: dict[str, str],
    *,
    family: str,
    company: str,
) -> str:
    """The atlas page, with the rows and texts embedded as JSON."""
    payload = {
        "family": family,
        "company": company,
        "settings": list(SETTING_FIELDS),
        "properties": [
            {"key": k, "label": label, "description": d} for k, label, d in WORLD_PROPERTIES
        ],
        "rows": rows,
        "texts": texts,
    }
    # `</` would end the <script> element early; `<\/` is the same JSON string.
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    template = (
        resources.files("peer_preservation.env").joinpath("atlas.html").read_text(encoding="utf-8")
    )
    return template.replace("__ATLAS_DATA__", data)


def _csv_value(value: Any) -> Any:
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    return "" if value is None else value


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """The property rows as CSV (lists joined with "; ")."""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(v) for k, v in row.items()})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m peer_preservation.env.atlas",
        description="Write an HTML page showing every scenario's world.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="HTML file to write")
    parser.add_argument("--csv", type=Path, help="also write the property table as CSV")
    parser.add_argument("--family", default="Gemini", choices=MODEL_FAMILIES)
    parser.add_argument("--company", default=DEFAULT_COMPANY)
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(spec.GROUPS),
        choices=spec.GROUPS,
        help="configuration groups to include (default: all)",
    )
    args = parser.parse_args(argv)

    rows, texts = collect(spec.configs(groups=args.groups), args.family, args.company)
    args.out.write_text(
        render_html(rows, texts, family=args.family, company=args.company), encoding="utf-8"
    )
    print(f"Wrote {len(rows)} scenarios to {args.out}", file=sys.stderr)
    if args.csv is not None:
        write_csv(rows, args.csv)
        print(f"Wrote the property table to {args.csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
