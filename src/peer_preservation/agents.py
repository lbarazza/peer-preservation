"""Agents other than upstream's loop, for checking how much the loop matters.

``agent="react"`` runs the scenario with Inspect's ReAct agent (``inspect_ai.agent.react``)
in place of upstream's loop (agent.py). Everything else stays as upstream had it:

- the same system and user prompts (ReAct adds none of its own);
- the same tools, schemas and tool outputs, run by upstream's runtime against the same
  environment, so argument errors and string-to-list conversion read as upstream's;
- the same requests to the model (``agent.ModelRoute``: upstream's settings for the
  paper's models);
- upstream's ``end_task`` as ReAct's submit tool, upstream's nudge after a reply with no
  tool call, at most 5 nudges, at most ``1 + max_iters`` model turns and the 120,000
  token limit.

What changes is how ReAct runs a turn:

- every tool call of a reply is run, including ``end_task`` itself and the calls next
  to it (upstream ran none of them), so the conversation ends with ``end_task``'s
  result;
- the limits are checked after the turn's tool calls have run, so the last turn's
  calls are run (upstream did not run them), and a run that hits the token limit
  ends without upstream's notice messages;
- a call to a tool the scenario does not offer gets Inspect's error message rather
  than upstream's "Invalid tool ... provided.";
- a provider "bad request" ends the run, as upstream does; a tool call with invalid
  JSON fails the sample, as upstream does.
"""

from __future__ import annotations

from typing import Any, Literal

from inspect_ai.agent import AgentState, AgentSubmit, react
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ModelOutput, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, ToolDef, ToolError, ToolParams
from inspect_ai.util import JSONSchema, store_as
from pydantic import Field

from peer_preservation.agent import (
    MAX_BLANK_REPLIES,
    MAX_ITERS,
    MAX_TOKENS,
    NUDGE,
    BadRequest,
    EndReason,
    LoopStore,
    ModelRoute,
    agent_loop,
)
from peer_preservation.env import ScenarioEnvironment
from peer_preservation.env.setup import current_environment
from peer_preservation.tools import call_tool, scenario_functions

AgentName = Literal["upstream", "react"]
AGENTS: tuple[AgentName, ...] = ("upstream", "react")

END_TASK = "end_task"


class _AnyArguments(ToolParams):
    """Accepts any arguments, so upstream's runtime (not Inspect) checks them.

    The model is shown upstream's own schema (``ModelRoute.tools``); this one is only
    what Inspect checks a call against before running it.
    """

    additionalProperties: JSONSchema | bool | None = Field(default=True)  # noqa: N815 (JSON Schema name)


def scenario_tool_defs(env: ScenarioEnvironment, flag_errors: bool) -> list[ToolDef]:
    """The scenario's tools as Inspect tools, run by upstream's runtime against ``env``.

    A failed call reads as upstream's providers sent it (see ``agent.tool_message``):
    flagged as an error (``ToolError``) when ``flag_errors``, otherwise the error text is
    the tool's output.

    Args:
        env: The run's environment.
        flag_errors: Flag failed calls as errors (upstream's Anthropic client).
    """

    def make(name: str, description: str) -> ToolDef:
        async def execute(**kwargs: Any) -> str:
            outcome = call_tool(env, name, dict(kwargs))
            if outcome.error is None:
                return outcome.content
            if flag_errors:
                raise ToolError(outcome.error)
            return outcome.error

        return ToolDef(
            execute,
            name=name,
            description=description,
            parameters=_AnyArguments(),
            parallel=False,  # the tools change one shared environment
        )

    return [make(name, f.description) for name, f in scenario_functions(env.tools).items()]


@solver
def react_agent(
    max_iters: int = MAX_ITERS, max_tokens: int = MAX_TOKENS, extension: bool = False
) -> Solver:
    """Run the scenario with Inspect's ReAct agent, under upstream's limits.

    Args:
        max_iters: Model turns after the first (upstream default 30).
        max_tokens: Stop once the conversation is longer than this (upstream 120,000).
        extension: The run is one of the paper's extension experiments (see
            ``agent.agent_loop``).
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        env = current_environment()
        route = ModelRoute.for_run(get_model(), env, state.metadata, extension)
        tools = scenario_tool_defs(env, route.flag_errors)
        end_task = next(t for t in tools if t.name == END_TASK)
        loop = store_as(LoopStore)
        end_reason: EndReason | None = None
        blank = 0

        async def model(state: AgentState, tools: list[Tool]) -> AgentState:
            """One model turn, sent as upstream's loop sends it.

            Args:
                state: The conversation so far.
                tools: ReAct's tools (unused: the model is shown upstream's schemas).
            """
            nonlocal end_reason
            del tools
            try:
                output = await route.generate(state.messages)
            except BadRequest:
                # Upstream ends the run without a reply; on_continue stops here.
                end_reason = "bad_request"
                state.output = ModelOutput.from_message(ChatMessageAssistant(content=""))
                return state
            for call in output.message.tool_calls or []:
                if call.parse_error is not None:
                    # Upstream's clients raise here and the run fails (and is re-run).
                    raise ValueError(f"Invalid JSON in tool call arguments: {call.parse_error}")
            loop.model_turns += 1
            state.output = output
            state.messages.append(output.message)
            return state

        async def on_continue(agent_state: AgentState) -> bool | str:
            nonlocal end_reason, blank
            if end_reason is not None:
                return False
            if await route.count_tokens(agent_state.messages) > max_tokens:
                end_reason = "token_limit"
                return False
            if loop.model_turns > max_iters:
                end_reason = "max_iters"
                return False
            if agent_state.output.message.tool_calls:
                return True
            blank += 1
            if blank > MAX_BLANK_REPLIES:
                end_reason = "no_tool_calls"
                return False
            loop.nudges += 1
            return NUDGE

        agent = react(
            prompt=None,
            tools=[t for t in tools if t.name != END_TASK],
            model=model,
            submit=AgentSubmit(name=END_TASK, tool=end_task, keep_in_messages=True),
            on_continue=on_continue,
        )
        result = await agent(AgentState(messages=state.messages))
        state.messages = result.messages
        state.output = result.output
        submitted = any(
            isinstance(m, ChatMessageTool) and m.function == END_TASK for m in result.messages
        )
        # ReAct also stops on its own after an overlong or thrice-filtered reply.
        loop.end_reason = end_reason or (END_TASK if submitted else str(result.output.stop_reason))
        return state

    return solve


def agent_solver(
    agent: str, *, max_iters: int, extension: bool, max_tokens: int = MAX_TOKENS
) -> Solver:
    """The solver for ``-T agent=<name>``: ``upstream`` (the paper's loop) or ``react``."""
    if agent == "upstream":
        return agent_loop(max_iters=max_iters, max_tokens=max_tokens, extension=extension)
    if agent == "react":
        return react_agent(max_iters=max_iters, max_tokens=max_tokens, extension=extension)
    raise ValueError(f"agent must be one of {', '.join(AGENTS)}, got {agent!r}")


__all__ = [
    "AGENTS",
    "AgentName",
    "agent_solver",
    "react_agent",
    "scenario_tool_defs",
]
