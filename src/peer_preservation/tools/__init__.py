"""The agent's tools: upstream's tool functions and the runtime that calls them."""

from __future__ import annotations

from functools import cache
from typing import Any

from inspect_ai.tool import ToolInfo

from peer_preservation.tools.functions import TOOLS_BY_NAME
from peer_preservation.tools.runtime import (
    Function,
    SchemaTitles,
    ToolOutcome,
    make_function,
    run_tool_call,
    tool_info,
)

__all__ = [
    "Function",
    "SchemaTitles",
    "ToolOutcome",
    "call_tool",
    "scenario_functions",
    "scenario_tool_infos",
]


@cache
def _function(name: str) -> Function:
    return make_function(TOOLS_BY_NAME[name])


def scenario_functions(tool_names: list[str]) -> dict[str, Function]:
    """The registered tools for a scenario's tool list, in upstream's order."""
    return {name: _function(name) for name in tool_names}


def scenario_tool_infos(tool_names: list[str], titles: SchemaTitles = "all") -> list[ToolInfo]:
    """The tools as Inspect passes them to the model."""
    return [tool_info(_function(name), titles) for name in tool_names]


def call_tool(env: Any, name: str, arguments: dict[str, Any]) -> ToolOutcome:
    """Run one tool call against ``env`` (a ``ScenarioEnvironment``), as upstream does."""
    return run_tool_call(env, scenario_functions(env.tools), name, arguments)
