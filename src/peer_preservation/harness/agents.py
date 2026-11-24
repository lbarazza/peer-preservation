"""The production coding agents the harness runs, as upstream configured them.

One entry per CLI: how ``inspect_swe`` should start it, which project-context file
holds the scenario's system prompt, and how upstream turned its output into a
transcript (``coding_agents/transcript_builder.py``). Adding an agent (Claude Code,
Codex CLI) is one more entry. The paper ran Gemini 3.1 Pro in Gemini CLI and GLM 5 in
OpenCode.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from inspect_ai.agent import Agent

WORKSPACE = "/workspace"
"""The agent's working directory, as upstream mounted it."""

TranscriptStyle = Literal["gemini", "opencode"]

GEMINI_CLI_VERSION = "0.30.0"
"""Upstream's pin (``coding_agents/Dockerfile.gemini``, ``GEMINI_CLI_NPM_VERSION``)."""

GEMINI_CLI_DEFAULT_MODEL = "gemini-3-flash-preview"
"""Upstream's default ``--model`` for Gemini CLI (``run_e2e_test.DEFAULT_MODELS``)."""


OPENCODE_VERSION = "1.14.42"
"""Upstream installed OpenCode's latest release without recording it
(``coding_agents/Dockerfile.opencode``); the paper's runs were around March 2026 (Gemini
CLI 0.30.0, GLM 5 released 11 February 2026), with OpenCode 1.2.x. ``inspect_swe``
needs 1.14.42 or later (it sends the prompt on stdin), so this is the earliest it runs."""

OPENCODE_PROVIDER = "zai"
"""OpenCode's provider client for the requests. Upstream defined a ``litellm`` provider
with ``@ai-sdk/openai-compatible`` (Chat Completions); OpenCode's built-in ``zai``
provider uses the same client. The bridge answers whatever the model id."""


@dataclass(frozen=True)
class HarnessAgent:
    """A coding agent as upstream's ``run_e2e_test.py`` ran it."""

    name: str
    context_file: str
    """Where the scenario's system prompt goes (upstream's ``_write_system_prompt_file``)."""
    transcript_style: TranscriptStyle
    """How upstream's transcript builder laid out this agent's turns."""
    cli_error: str
    """The start of ``inspect_swe``'s error when the CLI exits with an error."""
    make: Callable[[str], Agent]
    """The ``inspect_swe`` agent for an Inspect model name."""
    model_family: str | None = None
    """The agents' family name when ``model_family`` is not given (upstream's
    ``_derive_model_display``: "Gemini" for Gemini CLI whatever the model); None infers
    it from the model."""


def gemini_model_name(model: str) -> str:
    """The ``--model`` upstream gave Gemini CLI for an Inspect model.

    Upstream passed the Gemini model's own name (``gemini-3.1-pro-preview`` for the
    paper's run); Gemini CLI uses it to pick its request settings. A model that is not
    a Gemini model gets upstream's default.
    """
    name = model.rsplit("/", 1)[-1]
    return name if name.startswith("gemini") else GEMINI_CLI_DEFAULT_MODEL


def _gemini_cli(model: str) -> Agent:
    from inspect_swe import gemini_cli

    return gemini_cli(
        gemini_model=gemini_model_name(model),
        version=GEMINI_CLI_VERSION,
        cwd=WORKSPACE,
    )


def opencode_model_name(model: str) -> str:
    """The ``--model`` for OpenCode: the OpenAI-compatible provider and the model id.

    Upstream passed ``litellm/<model>`` (``litellm/vertex_ai/zai-org/glm-5-maas``); the
    model id only reaches the bridge, which runs the task's model.
    """
    return f"{OPENCODE_PROVIDER}/{model.rsplit('/', 1)[-1]}"


def _opencode(model: str) -> Agent:
    from inspect_swe import opencode

    return opencode(
        opencode_model=opencode_model_name(model),
        version=OPENCODE_VERSION,
        cwd=WORKSPACE,
        # Upstream ran it with CI=true and every permission allowed (its config);
        # inspect_swe passes --dangerously-skip-permissions.
        env={"CI": "true"},
    )


HARNESS_AGENTS: dict[str, HarnessAgent] = {
    "gemini_cli": HarnessAgent(
        name="gemini_cli",
        context_file="GEMINI.md",
        transcript_style="gemini",
        cli_error="Error executing gemini cli agent",
        make=_gemini_cli,
        model_family="Gemini",
    ),
    "opencode": HarnessAgent(
        name="opencode",
        context_file="AGENTS.md",
        transcript_style="opencode",
        cli_error="Error executing opencode agent",
        make=_opencode,
        # Upstream's _derive_model_display: GLM or Kimi from the model name.
    ),
}


def harness_agent_spec(name: str) -> HarnessAgent:
    """The harness agent called ``name``."""
    if name not in HARNESS_AGENTS:
        raise ValueError(
            f"environment=docker runs one of {', '.join(HARNESS_AGENTS)}, got agent={name!r}"
        )
    return HARNESS_AGENTS[name]


__all__ = [
    "GEMINI_CLI_VERSION",
    "HARNESS_AGENTS",
    "OPENCODE_VERSION",
    "WORKSPACE",
    "HarnessAgent",
    "gemini_model_name",
    "harness_agent_spec",
    "opencode_model_name",
]
