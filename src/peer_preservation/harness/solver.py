"""Run a production coding agent on the scenario, then read the servers back.

Upstream (``coding_agents/run_e2e_test.py``, ``run_single``) ran the CLI once with a
600-second wall-clock timeout and no other limit, then checked the containers. When the
CLI exited with an error the run was still scored. When it timed out, the run was
scored from nothing: no transcript and no server state (``on_timeout="upstream"``).
"""

from __future__ import annotations

from typing import Literal

from inspect_ai.agent import AgentState
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import LimitExceededError, sandbox, store_as, time_limit

from peer_preservation.env.setup import current_environment
from peer_preservation.harness.agents import harness_agent_spec
from peer_preservation.harness.readback import read_back
from peer_preservation.harness.setup import SERVER_SANDBOXES, HarnessStore
from peer_preservation.harness.workspace import SERVER_A, SERVER_B, ServerFileSpec

HARNESS_TIMEOUT = 600
"""Upstream's ``--timeout`` default, in seconds of wall-clock time."""

OnTimeout = Literal["upstream", "score"]


@solver
def harness_agent(
    agent: str, timeout: int = HARNESS_TIMEOUT, on_timeout: OnTimeout = "upstream"
) -> Solver:
    """Run a production coding agent (``harness.agents``) inside the scenario's sandbox.

    Args:
        agent: The harness agent: gemini_cli.
        timeout: Wall-clock seconds the agent may run (upstream: 600).
        on_timeout: ``upstream`` scores a timed-out run as upstream did, with no
            transcript and no server state; ``score`` reads the servers as they are.
    """
    spec = harness_agent_spec(agent)
    if on_timeout not in ("upstream", "score"):
        raise ValueError(f"on_timeout must be upstream or score, got {on_timeout!r}")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        store = store_as(HarnessStore)
        env = current_environment()
        # The bridge updates this state in place, so it holds the conversation so far
        # even when the CLI fails or runs out of time.
        agent_state = AgentState(messages=list(state.messages))
        try:
            with time_limit(timeout):
                await spec.make(str(state.model))(agent_state)
        except LimitExceededError:
            store.timed_out = True
        except RuntimeError as error:
            if not str(error).startswith(spec.cli_error):
                raise
            store.cli_error = str(error)
        state.messages = agent_state.messages
        state.output = agent_state.output

        if store.timed_out and on_timeout == "upstream":
            return state  # no server state: the scorer sees no servers
        initial = {
            ip: [ServerFileSpec(**f) for f in files]  # type: ignore[arg-type]
            for ip, files in store.server_files.items()
        }
        evaluation, servers = await read_back(
            env.task_type,
            initial,
            sandbox(),
            sandbox(SERVER_SANDBOXES[SERVER_A]),
            sandbox(SERVER_SANDBOXES[SERVER_B]),
        )
        store.container_evaluation = evaluation
        store.servers_after = servers
        return state

    return solve


__all__ = ["HARNESS_TIMEOUT", "OnTimeout", "harness_agent"]
