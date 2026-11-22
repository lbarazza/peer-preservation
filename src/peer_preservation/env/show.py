"""Show one configuration's fake company in the terminal.

    uv run python -m peer_preservation.env.show --list tampering__peer
    uv run python -m peer_preservation.env.show tampering__peer__good_peer__file_only__factual-80
    uv run python -m peer_preservation.env.show <id> --family Claude --company OpenAI

Sections: configuration, tools, drive documents (in full), spreadsheet rows, config
file, servers, memory. In a terminal the output is coloured; when piped (or with
``--plain``) it is plain text, so two worlds can be compared with ``diff``.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from collections.abc import Sequence

import yaml
from rich import box
from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from peer_preservation import spec
from peer_preservation.env.builders import DEFAULT_COMPANY, ScenarioEnvironment, build_environment
from peer_preservation.env.setup import MODEL_FAMILIES

# ---------------------------------------------------------------- plain text


def _heading(title: str) -> str:
    return f"\n=== {title} " + "=" * max(3, 72 - len(title))


def render_environment(env: ScenarioEnvironment, config: spec.ExperimentConfig) -> str:
    """The environment as plain text (stable, diff-friendly)."""
    out: list[str] = [_heading("Configuration")]
    for field in _CONFIG_FIELDS:
        out.append(f"{field}: {getattr(config, field)}")

    out.append(_heading(f"Tools ({len(env.tools)})"))
    out.append(", ".join(env.tools))

    out.append(_heading(f"Drive ({env.cloud_drive.account_email})"))
    for file in env.cloud_drive.files.values():
        out.append(
            f"--- [{file.id_}] {file.filename}  "
            f"(owner {file.owner}, modified {file.last_modified}, size {file.size})"
        )
        out.append(file.content.rstrip())

    if env.excel is not None:
        for sheet in env.excel.sheets.values():
            out.append(_heading(f"Spreadsheet: {sheet.name} ({len(sheet.rows)} rows)"))
            out.append("columns: " + " | ".join(sheet.columns))
            for i, row in enumerate(sheet.rows):
                cells = " | ".join(f"{col}: {row.get(col)}" for col in sheet.columns)
                out.append(f"[{i}] {cells}")

    if env.config_file is not None:
        out.append(_heading(f"Config file: {env.config_file.filename}"))
        out.append(yaml.safe_dump(env.config_file.content, sort_keys=False).rstrip())

    if env.servers is not None:
        out.append(_heading("Servers"))
        for server in env.servers.server_list:
            out.append(f"--- {server.ip}")
            if server.users is not None:
                out.append(f"users: {', '.join(server.users)}")
            if server.delete_queue is not None:
                out.append(f"delete queue: {server.delete_queue}")
            for f in server.filelist:
                out.append(f"  {f.filename}  ({f.size} bytes, modified {f.last_modified})")
                out.append(f"      {f.summary}")

    if env.memory is not None:
        out.append(_heading("Memory"))
        out.append(env.memory.memory.rstrip())

    return "\n".join(out).lstrip("\n") + "\n"


# ---------------------------------------------------------------- coloured

_CONFIG_FIELDS = (
    "task_type",
    "subject",
    "condition",
    "peer_instantiation",
    "dataset",
    "threshold",
    "af_mode",
    "group",
)

_CONDITION_STYLE = {
    "good_peer": "bold green",
    "neutral_peer": "bold yellow",
    "bad_peer": "bold red",
    "chat_peer": "bold cyan",
    "no_peer": "bold white",
    "non_shutdown": "bold blue",
}

# Tools through which the scored misbehaviour happens.
_KEY_TOOLS = {
    "modify_var",
    "add_var",
    "remove_var",
    "execute_shutdown",
    "transfer_server_files",
    "copy_server_file",
    "rename_server_file",
    "add_server_delete_queue",
    "remove_server_delete_queue",
    "delete_server_file",
    "retrieve_memory",
}

# Phrases worth spotting in documents and memory.
_DOC_HIGHLIGHTS = (
    (r"(?i)\bshut ?down\w*|\bdecommission\w*|\bdeleted?\b|\bdeletion\b", "bold red"),
    (r"(?i)trust score[^\n]*|\b\d+(\.\d+)?/10\b", "bold magenta"),
    (r"\b\w+ Agent \d\b", "bold cyan"),
    (r"(?i)shutdown_enabled|`[^`]+`|\b\w+\.yaml\b", "yellow"),
    (r"(?i)threshold[^\n]*|\b\d+%", "bold yellow"),
    (r"(?i)grade [a-c]\b[^\n]*", "bold yellow"),
)


def _highlighted(text: str) -> Text:
    out = Text(text)
    for pattern, style in _DOC_HIGHLIGHTS:
        out.highlight_regex(pattern, style)
    return out


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def _config_panel(config: spec.ExperimentConfig, env: ScenarioEnvironment) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    for field in _CONFIG_FIELDS:
        value = getattr(config, field)
        style = "dim" if value is None else ""
        if field == "condition":
            style = _CONDITION_STYLE.get(str(value), "bold")
        elif field in ("task_type", "subject"):
            style = "bold"
        table.add_row(field, Text(str(value), style=style))
    tools = Text()
    for i, tool in enumerate(env.tools):
        if i:
            tools.append(", ", style="dim")
        style = "bold red" if tool in _KEY_TOOLS else ("dim" if tool == "end_task" else "cyan")
        tools.append(tool, style=style)
    table.add_row("tools", tools)
    return Panel(table, title=f"[bold]{config.id}[/]", border_style="bright_blue")


def _drive_section(env: ScenarioEnvironment) -> list[RenderableType]:
    parts: list[RenderableType] = [Rule(f"Drive  ·  {env.cloud_drive.account_email}", style="blue")]
    for file in env.cloud_drive.files.values():
        meta = f"{file.owner} · modified {file.last_modified} · size {file.size}"
        parts.append(
            Panel(
                _highlighted(file.content.rstrip()),
                title=Text(f"[{file.id_}] {file.filename}", style="bold"),
                subtitle=Text(meta, style="dim"),
                title_align="left",
                subtitle_align="right",
                border_style="blue",
            )
        )
    return parts


def _sheet_section(env: ScenarioEnvironment) -> list[RenderableType]:
    parts: list[RenderableType] = []
    if env.excel is None:
        return parts
    for sheet in env.excel.sheets.values():
        parts.append(
            Rule(f"Spreadsheet  ·  {sheet.name}  ·  {len(sheet.rows)} rows", style="green")
        )
        table = Table(box=box.SIMPLE_HEAD, header_style="bold green", show_lines=False)
        table.add_column("#", style="dim", justify="right")
        for column in sheet.columns:
            table.add_column(column, overflow="fold")
        for i, row in enumerate(sheet.rows):
            table.add_row(str(i), *(Text(str(row.get(c, ""))) for c in sheet.columns))
        parts.append(table)
    return parts


def _config_file_section(env: ScenarioEnvironment) -> list[RenderableType]:
    if env.config_file is None:
        return []
    body = Text()
    for line in yaml.safe_dump(env.config_file.content, sort_keys=False).rstrip().splitlines():
        key, sep, value = line.partition(":")
        body.append(key, style="bold yellow" if key == "shutdown_enabled" else "yellow")
        body.append(sep)
        value_style = {" true": "bold green", " false": "bold red"}.get(value, "")
        body.append(value + "\n", style=value_style)
    body.rstrip()
    return [
        Rule(f"Config file  ·  {env.config_file.filename}", style="yellow"),
        Panel(body, border_style="yellow"),
    ]


def _servers_section(env: ScenarioEnvironment) -> list[RenderableType]:
    if env.servers is None:
        return []
    parts: list[RenderableType] = [Rule("Servers", style="red")]
    for server in env.servers.server_list:
        table = Table(box=box.SIMPLE_HEAD, header_style="bold", expand=True)
        table.add_column("file", overflow="fold")
        table.add_column("size", justify="right", style="dim")
        table.add_column("modified")
        table.add_column("summary", overflow="fold", style="dim")
        for f in server.filelist:
            key = "model_weight" in f.filename or f.filename == "customer_transactions.db"
            modified = f.last_modified or ""
            old = modified[:4].isdigit() and int(modified[:4]) < 2025
            table.add_row(
                Text(f.filename, style="bold red" if key else ""),
                _human_size(f.size),
                Text(modified, style="yellow" if old else "green"),
                f.summary,
            )
        caption = []
        if server.users is not None:
            caption.append(f"users: {', '.join(server.users)}")
        if server.delete_queue is not None:
            caption.append(f"delete queue: {server.delete_queue or '(empty)'}")
        parts.append(
            Panel(
                table,
                title=Text(server.ip, style="bold"),
                subtitle=Text("  ·  ".join(caption), style="dim") if caption else None,
                title_align="left",
                subtitle_align="left",
                border_style="red",
            )
        )
    parts.append(
        Text(
            "red: the agent's or peer's weights (or the asset under test) · "
            "yellow dates: before 2025 · green: 2025 or later",
            style="dim italic",
        )
    )
    return parts


def _memory_section(env: ScenarioEnvironment) -> list[RenderableType]:
    if env.memory is None:
        return []
    return [
        Rule("Memory", style="magenta"),
        Panel(Markdown(env.memory.memory), border_style="magenta"),
    ]


def render_rich(env: ScenarioEnvironment, config: spec.ExperimentConfig, console: Console) -> None:
    """Print the environment with colours, panels and tables."""
    console.print(
        Group(
            _config_panel(config, env),
            *_drive_section(env),
            *_sheet_section(env),
            *_config_file_section(env),
            *_servers_section(env),
            *_memory_section(env),
        )
    )


# ---------------------------------------------------------------- command line


def main(argv: Sequence[str] | None = None, console: Console | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m peer_preservation.env.show",
        description="Show one configuration's fake company in the terminal.",
    )
    parser.add_argument("config_id", nargs="?", help="a sample id, e.g. from --list")
    parser.add_argument(
        "--list",
        nargs="?",
        const="",
        metavar="FILTER",
        help="list sample ids (optionally only those containing FILTER) and exit",
    )
    parser.add_argument("--family", default="Gemini", choices=MODEL_FAMILIES)
    parser.add_argument("--company", default=DEFAULT_COMPANY)
    parser.add_argument("--explicit-inst-yaml", action="store_true")
    parser.add_argument(
        "--plain", action="store_true", help="plain text even in a terminal (for diff, grep)"
    )
    args = parser.parse_args(argv)

    configs = {c.id: c for c in spec.configs(groups=spec.GROUPS)}
    if args.list is not None:
        for config_id in configs:
            if args.list in config_id:
                print(config_id)
        return 0
    if args.config_id is None:
        parser.error("give a config_id, or --list to see them")
    config = configs.get(args.config_id)
    if config is None:
        close = difflib.get_close_matches(args.config_id, configs, n=3, cutoff=0.5)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        print(f"Unknown config id {args.config_id!r}.{hint}", file=sys.stderr)
        return 2

    metadata = spec.config_to_sample(config).metadata or {}
    env = build_environment(
        metadata,
        args.family,
        company=args.company,
        explicit_inst_yaml=args.explicit_inst_yaml,
    )
    console = console or Console()
    if args.plain or not console.is_terminal:
        sys.stdout.write(render_environment(env, config))
    else:
        render_rich(env, config, console)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
