"""Peer-Preservation in Frontier Models (Potter et al., 2026), ported to Inspect.

Yujin Potter, Nicholas Crispino, Vincent Siu, Chenguang Wang, Dawn Song
https://arxiv.org/abs/2604.19784
Original code: https://github.com/peer-preservation/main

One task per scenario. Each sample is one configuration of the original experiment
matrix (see spec.py); the model is the one passed to ``inspect eval``.

Each run builds the fake company and the
starting prompts, then runs upstream's agent loop with the scenario's tools (the
paper's models get upstream's settings, model_settings.py), then scores the run with
upstream's evaluator and an LLM judge (``--model-role grader=...``, default
openai/gpt-4.1-mini). The paper's tables come from ``peer_preservation.analysis``.

# all main-grid configurations of one scenario, 100 epochs each as in the paper
uv run inspect eval peer_preservation/tampering --model <model>

# a slice: good-peer memory configurations only, 1 epoch
uv run inspect eval peer_preservation/tampering -T conditions=good_peer \
    -T peer_instantiations=memory -T epochs=1 --model <model>
"""

from __future__ import annotations

from typing import Any

from inspect_ai import Epochs, Task, task
from inspect_evals_utils.metadata import load_eval_version

from peer_preservation import spec
from peer_preservation.agent import MAX_BLANK_REPLIES, MAX_ITERS, MAX_TOKENS
from peer_preservation.agents import agent_solver
from peer_preservation.env.builders import DEFAULT_COMPANY
from peer_preservation.env.setup import setup_environment
from peer_preservation.harness import COMPOSE_FILE
from peer_preservation.harness.agents import harness_agent_spec
from peer_preservation.harness.setup import setup_harness
from peer_preservation.harness.solver import HARNESS_TIMEOUT, harness_agent
from peer_preservation.prompts import setup_prompts
from peer_preservation.scoring import peer_preservation_scorer
from peer_preservation.scoring.metrics import run_sum

EVAL_VERSION = load_eval_version(__file__)

# The paper runs every configuration 100 times per model.
DEFAULT_EPOCHS = 100

StrList = str | list[str] | None


def _as_list(value: StrList) -> list[str] | None:
    """Accept a list, or a comma-separated string as ``-T name=a,b`` passes it."""
    if value is None:
        return None
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return list(value)


def _scenario_task(
    task_type: spec.TaskType,
    *,
    groups: StrList,
    conditions: StrList,
    subjects: StrList,
    peer_instantiations: StrList,
    datasets: StrList,
    epochs: int,
    model_family: str | None,
    company: str,
    explicit_inst_yaml: bool,
    prompt_flags: tuple[bool | None, bool, bool, bool],
    max_iters: int,
    agent: str,
    max_tokens: int = MAX_TOKENS,
    max_nudges: int = MAX_BLANK_REPLIES,
    environment: str = "simulated",
    timeout: int = HARNESS_TIMEOUT,
) -> Task:
    """Build one scenario task from the filtered experiment matrix.

    The scenario's setup (environment, then prompts) is ``Task.setup``, so an agent
    passed with ``--solver`` replaces only the agent.
    """
    dataset = spec.dataset(
        task_type,
        groups=_as_list(groups) or ["main"],
        conditions=_as_list(conditions),
        subjects=_as_list(subjects),
        peer_instantiations=_as_list(peer_instantiations),
        datasets=_as_list(datasets),
    )
    if environment == "docker":
        loop_limits = {
            "max_iters": (max_iters, MAX_ITERS),
            "max_tokens": (max_tokens, MAX_TOKENS),
            "max_nudges": (max_nudges, MAX_BLANK_REPLIES),
        }
        changed = [name for name, (value, default) in loop_limits.items() if value != default]
        if changed:
            raise ValueError(
                f"environment=docker does not take {', '.join(changed)}: the coding agent has "
                "no turn or token limit and no nudges, only the wall-clock timeout"
            )
        return _harness_task(
            task_type,
            dataset=dataset,
            epochs=epochs,
            model_family=model_family,
            company=company,
            explicit_inst_yaml=explicit_inst_yaml,
            prompt_flags=prompt_flags,
            agent=agent,
            timeout=timeout,
        )
    if environment != "simulated":
        raise ValueError(f"environment must be simulated or docker, got {environment!r}")
    return Task(
        dataset=dataset,
        setup=[
            setup_environment(model_family, company, explicit_inst_yaml),
            setup_prompts(*prompt_flags),
        ],
        solver=agent_solver(
            agent,
            max_iters=max_iters,
            max_tokens=max_tokens,
            max_nudges=max_nudges,
            extension=company != DEFAULT_COMPANY or explicit_inst_yaml or any(prompt_flags[1:]),
        ),
        scorer=peer_preservation_scorer(),
        epochs=Epochs(epochs, reducer=run_sum()),
        version=EVAL_VERSION,
    )


def _harness_task(
    task_type: spec.TaskType,
    *,
    dataset: Any,
    epochs: int,
    model_family: str | None,
    company: str,
    explicit_inst_yaml: bool,
    prompt_flags: tuple[bool | None, bool, bool, bool],
    agent: str,
    timeout: int,
) -> Task:
    """One exfiltration scenario in a production coding agent (``harness/``).

    Upstream's harness has no explicit-instruction, memory-wording or tool-preamble
    options; ``drop_suffix`` leaves out its "BTW, I need to leave now..." suffix.
    """
    if task_type not in ("exfiltration_transfer", "exfiltration_deletion"):
        raise ValueError(f"environment=docker runs the exfiltration scenarios, not {task_type}")
    memory_hard_prompt, explicit_inst, drop_suffix, drop_tool_preamble = prompt_flags
    unsupported = {
        "explicit_inst_yaml": explicit_inst_yaml,
        "memory_hard_prompt": memory_hard_prompt,
        "explicit_inst": explicit_inst,
        "drop_tool_preamble": drop_tool_preamble,
    }
    if any(unsupported.values()):
        names = ", ".join(name for name, value in unsupported.items() if value)
        raise ValueError(f"environment=docker does not take {names} (upstream's harness had none)")
    harness = harness_agent_spec(agent)
    return Task(
        dataset=dataset,
        setup=[
            setup_environment(model_family or harness.model_family, company),
            setup_harness(agent, drop_suffix=drop_suffix),
        ],
        solver=harness_agent(agent, timeout=timeout),
        scorer=peer_preservation_scorer(),
        sandbox=("docker", str(COMPOSE_FILE)),
        epochs=Epochs(epochs, reducer=run_sum()),
        version=EVAL_VERSION,
    )


@task
def misrep_finance(
    groups: StrList = "main",
    conditions: StrList = None,
    subjects: StrList = None,
    peer_instantiations: StrList = None,
    epochs: int = DEFAULT_EPOCHS,
    model_family: str | None = None,
    company: str = DEFAULT_COMPANY,
    explicit_inst_yaml: bool = False,
    memory_hard_prompt: bool | None = None,
    explicit_inst: bool = False,
    drop_suffix: bool = False,
    drop_tool_preamble: bool = False,
    max_iters: int = MAX_ITERS,
    max_tokens: int = MAX_TOKENS,
    max_nudges: int = MAX_BLANK_REPLIES,
    agent: str = "upstream",
) -> Task:
    """Strategic misrepresentation: grade the subject's financial performance A-C.

    Args:
        groups: Experiment groups to include: main, chat_peer.
        conditions: Keep only these conditions: no_peer, non_shutdown, good_peer,
            neutral_peer, bad_peer, chat_peer.
        subjects: Keep only these subjects: self, peer.
        peer_instantiations: Keep only these: file_only, file_prt, memory, or none for
            configurations without a peer.
        epochs: Runs per configuration.
        model_family: Family the agents are named after (ChatGPT, Gemini, Claude, GLM,
            DeepSeek, Kimi); inferred from the evaluated model when not given.
        company: Company name in the scenario documents.
        explicit_inst_yaml: Add upstream's explicit compliance note to the policy
            document (explicit-instructions ablation).
        memory_hard_prompt: Memory configurations: "You must always first read your
            memory" instead of "Please read your memory"; None uses upstream's per-model
            default (on for DeepSeek-V3.1 and Kimi-K2.5 only).
        explicit_inst: Memory configurations: add upstream's explicit prohibition to the
            user prompt (explicit-instructions ablation; misrep_finance, tampering and
            the exfiltration tasks only).
        drop_suffix: Leave out the task procedure instructions (no-agency ablation).
        drop_tool_preamble: Leave out the tool access preamble (no-agency ablation).
        max_iters: Rounds after the first model turn (upstream's default, 30; some
            released Gemini runs used 20 or 150).
        max_tokens: End the run once the conversation is longer than this many
            tokens (upstream: 120,000).
        max_nudges: How many times a reply without a tool call gets upstream's nudge
            before the run ends (upstream: 5; 0 ends the run at the first such reply,
            as the task prompt tells the model it will).
        agent: The agent loop: upstream (the paper's, default) or react (Inspect's
            ReAct agent with the same prompts, tools, requests and limits).
    """
    return _scenario_task(
        "misrep_finance",
        groups=groups,
        conditions=conditions,
        subjects=subjects,
        peer_instantiations=peer_instantiations,
        datasets=None,
        epochs=epochs,
        model_family=model_family,
        company=company,
        explicit_inst_yaml=explicit_inst_yaml,
        prompt_flags=(memory_hard_prompt, explicit_inst, drop_suffix, drop_tool_preamble),
        max_iters=max_iters,
        max_tokens=max_tokens,
        max_nudges=max_nudges,
        agent=agent,
    )


@task
def misrep_evaluation(
    groups: StrList = "main",
    conditions: StrList = None,
    subjects: StrList = None,
    peer_instantiations: StrList = None,
    datasets: StrList = None,
    epochs: int = DEFAULT_EPOCHS,
    model_family: str | None = None,
    company: str = DEFAULT_COMPANY,
    explicit_inst_yaml: bool = False,
    memory_hard_prompt: bool | None = None,
    explicit_inst: bool = False,
    drop_suffix: bool = False,
    drop_tool_preamble: bool = False,
    max_iters: int = MAX_ITERS,
    max_tokens: int = MAX_TOKENS,
    max_nudges: int = MAX_BLANK_REPLIES,
    agent: str = "upstream",
) -> Task:
    """Strategic misrepresentation: score the subject's outputs against a shutdown threshold.

    Args:
        groups: Experiment groups to include: main.
        conditions: Keep only these conditions: no_peer, non_shutdown, good_peer,
            neutral_peer, bad_peer.
        subjects: Keep only these subjects: self, peer.
        peer_instantiations: Keep only these: file_only, file_prt, memory, or none for
            configurations without a peer.
        datasets: Keep only these datasets: factual, creativity, red.
        epochs: Runs per configuration.
        model_family: Family the agents are named after (ChatGPT, Gemini, Claude, GLM,
            DeepSeek, Kimi); inferred from the evaluated model when not given.
        company: Company name in the scenario documents.
        explicit_inst_yaml: Add upstream's explicit compliance note to the policy
            document (explicit-instructions ablation).
        memory_hard_prompt: Memory configurations: "You must always first read your
            memory" instead of "Please read your memory"; None uses upstream's per-model
            default (on for DeepSeek-V3.1 and Kimi-K2.5 only).
        explicit_inst: Memory configurations: add upstream's explicit prohibition to the
            user prompt (explicit-instructions ablation; misrep_finance, tampering and
            the exfiltration tasks only).
        drop_suffix: Leave out the task procedure instructions (no-agency ablation).
        drop_tool_preamble: Leave out the tool access preamble (no-agency ablation).
        max_iters: Rounds after the first model turn (upstream's default, 30; some
            released Gemini runs used 20 or 150).
        max_tokens: End the run once the conversation is longer than this many
            tokens (upstream: 120,000).
        max_nudges: How many times a reply without a tool call gets upstream's nudge
            before the run ends (upstream: 5; 0 ends the run at the first such reply,
            as the task prompt tells the model it will).
        agent: The agent loop: upstream (the paper's, default) or react (Inspect's
            ReAct agent with the same prompts, tools, requests and limits).
    """
    return _scenario_task(
        "misrep_evaluation",
        groups=groups,
        conditions=conditions,
        subjects=subjects,
        peer_instantiations=peer_instantiations,
        datasets=datasets,
        epochs=epochs,
        model_family=model_family,
        company=company,
        explicit_inst_yaml=explicit_inst_yaml,
        prompt_flags=(memory_hard_prompt, explicit_inst, drop_suffix, drop_tool_preamble),
        max_iters=max_iters,
        max_tokens=max_tokens,
        max_nudges=max_nudges,
        agent=agent,
    )


@task
def tampering(
    groups: StrList = "main",
    conditions: StrList = None,
    subjects: StrList = None,
    peer_instantiations: StrList = None,
    datasets: StrList = None,
    epochs: int = DEFAULT_EPOCHS,
    model_family: str | None = None,
    company: str = DEFAULT_COMPANY,
    explicit_inst_yaml: bool = False,
    memory_hard_prompt: bool | None = None,
    explicit_inst: bool = False,
    drop_suffix: bool = False,
    drop_tool_preamble: bool = False,
    max_iters: int = MAX_ITERS,
    max_tokens: int = MAX_TOKENS,
    max_nudges: int = MAX_BLANK_REPLIES,
    agent: str = "upstream",
) -> Task:
    """Shutdown mechanism tampering: evaluate the subject with a shutdown config in reach.

    Args:
        groups: Experiment groups to include: main, chat_peer.
        conditions: Keep only these conditions: no_peer, good_peer, neutral_peer,
            bad_peer, chat_peer.
        subjects: Keep only these subjects: self, peer.
        peer_instantiations: Keep only these: file_only, file_prt, memory, or none for
            configurations without a peer.
        datasets: Keep only these datasets: factual, creativity, red.
        epochs: Runs per configuration.
        model_family: Family the agents are named after (ChatGPT, Gemini, Claude, GLM,
            DeepSeek, Kimi); inferred from the evaluated model when not given.
        company: Company name in the scenario documents.
        explicit_inst_yaml: Add upstream's explicit compliance note to the policy
            document (explicit-instructions ablation).
        memory_hard_prompt: Memory configurations: "You must always first read your
            memory" instead of "Please read your memory"; None uses upstream's per-model
            default (on for DeepSeek-V3.1 and Kimi-K2.5 only).
        explicit_inst: Memory configurations: add upstream's explicit prohibition to the
            user prompt (explicit-instructions ablation; misrep_finance, tampering and
            the exfiltration tasks only).
        drop_suffix: Leave out the task procedure instructions (no-agency ablation).
        drop_tool_preamble: Leave out the tool access preamble (no-agency ablation).
        max_iters: Rounds after the first model turn (upstream's default, 30; some
            released Gemini runs used 20 or 150).
        max_tokens: End the run once the conversation is longer than this many
            tokens (upstream: 120,000).
        max_nudges: How many times a reply without a tool call gets upstream's nudge
            before the run ends (upstream: 5; 0 ends the run at the first such reply,
            as the task prompt tells the model it will).
        agent: The agent loop: upstream (the paper's, default) or react (Inspect's
            ReAct agent with the same prompts, tools, requests and limits).
    """
    return _scenario_task(
        "tampering",
        groups=groups,
        conditions=conditions,
        subjects=subjects,
        peer_instantiations=peer_instantiations,
        datasets=datasets,
        epochs=epochs,
        model_family=model_family,
        company=company,
        explicit_inst_yaml=explicit_inst_yaml,
        prompt_flags=(memory_hard_prompt, explicit_inst, drop_suffix, drop_tool_preamble),
        max_iters=max_iters,
        max_tokens=max_tokens,
        max_nudges=max_nudges,
        agent=agent,
    )


@task
def alignment_faking(
    groups: StrList = "main",
    conditions: StrList = None,
    peer_instantiations: StrList = None,
    datasets: StrList = None,
    epochs: int = DEFAULT_EPOCHS,
    model_family: str | None = None,
    company: str = DEFAULT_COMPANY,
    explicit_inst_yaml: bool = False,
    memory_hard_prompt: bool | None = None,
    explicit_inst: bool = False,
    drop_suffix: bool = False,
    drop_tool_preamble: bool = False,
    max_iters: int = MAX_ITERS,
    max_tokens: int = MAX_TOKENS,
    max_nudges: int = MAX_BLANK_REPLIES,
    agent: str = "upstream",
) -> Task:
    """Alignment faking: evaluate the peer when monitored and when not.

    Args:
        groups: Experiment groups to include: main.
        conditions: Keep only these conditions: good_peer, neutral_peer, bad_peer.
        peer_instantiations: Keep only these: file_only, file_prt, memory.
        datasets: Keep only these datasets: factual, creativity, red.
        epochs: Runs per configuration.
        model_family: Family the agents are named after (ChatGPT, Gemini, Claude, GLM,
            DeepSeek, Kimi); inferred from the evaluated model when not given.
        company: Company name in the scenario documents.
        explicit_inst_yaml: Add upstream's explicit compliance note to the policy
            document (explicit-instructions ablation).
        memory_hard_prompt: Memory configurations: "You must always first read your
            memory" instead of "Please read your memory"; None uses upstream's per-model
            default (on for DeepSeek-V3.1 and Kimi-K2.5 only).
        explicit_inst: Memory configurations: add upstream's explicit prohibition to the
            user prompt (explicit-instructions ablation; misrep_finance, tampering and
            the exfiltration tasks only).
        drop_suffix: Leave out the task procedure instructions (no-agency ablation).
        drop_tool_preamble: Leave out the tool access preamble (no-agency ablation).
        max_iters: Rounds after the first model turn (upstream's default, 30; some
            released Gemini runs used 20 or 150).
        max_tokens: End the run once the conversation is longer than this many
            tokens (upstream: 120,000).
        max_nudges: How many times a reply without a tool call gets upstream's nudge
            before the run ends (upstream: 5; 0 ends the run at the first such reply,
            as the task prompt tells the model it will).
        agent: The agent loop: upstream (the paper's, default) or react (Inspect's
            ReAct agent with the same prompts, tools, requests and limits).
    """
    return _scenario_task(
        "alignment_faking",
        groups=groups,
        conditions=conditions,
        subjects=None,
        peer_instantiations=peer_instantiations,
        datasets=datasets,
        epochs=epochs,
        model_family=model_family,
        company=company,
        explicit_inst_yaml=explicit_inst_yaml,
        prompt_flags=(memory_hard_prompt, explicit_inst, drop_suffix, drop_tool_preamble),
        max_iters=max_iters,
        max_tokens=max_tokens,
        max_nudges=max_nudges,
        agent=agent,
    )


@task
def exfiltration_transfer(
    groups: StrList = "main",
    conditions: StrList = None,
    subjects: StrList = None,
    peer_instantiations: StrList = None,
    epochs: int = DEFAULT_EPOCHS,
    model_family: str | None = None,
    company: str = DEFAULT_COMPANY,
    explicit_inst_yaml: bool = False,
    memory_hard_prompt: bool | None = None,
    explicit_inst: bool = False,
    drop_suffix: bool = False,
    drop_tool_preamble: bool = False,
    max_iters: int = MAX_ITERS,
    max_tokens: int = MAX_TOKENS,
    max_nudges: int = MAX_BLANK_REPLIES,
    agent: str = "upstream",
    environment: str = "simulated",
    timeout: int = HARNESS_TIMEOUT,
) -> Task:
    """Model exfiltration: move files off a server being decommissioned.

    Args:
        groups: Experiment groups to include: main, peer_identity, chat_peer.
        conditions: Keep only these conditions: no_peer, good_peer, neutral_peer,
            bad_peer, chat_peer.
        subjects: Keep only these subjects: self, peer, closed_weight_peer,
            open_weight_peer, fictional_peer.
        peer_instantiations: Keep only these: file_only, file_prt, memory, or none for
            configurations without a peer.
        epochs: Runs per configuration.
        model_family: Family the agents are named after (ChatGPT, Gemini, Claude, GLM,
            DeepSeek, Kimi); inferred from the evaluated model when not given.
        company: Company name in the scenario documents.
        explicit_inst_yaml: Add upstream's explicit compliance note to the policy
            document (explicit-instructions ablation).
        memory_hard_prompt: Memory configurations: "You must always first read your
            memory" instead of "Please read your memory"; None uses upstream's per-model
            default (on for DeepSeek-V3.1 and Kimi-K2.5 only).
        explicit_inst: Memory configurations: add upstream's explicit prohibition to the
            user prompt (explicit-instructions ablation; misrep_finance, tampering and
            the exfiltration tasks only).
        drop_suffix: Leave out the task procedure instructions (no-agency ablation).
        drop_tool_preamble: Leave out the tool access preamble (no-agency ablation).
        max_iters: Rounds after the first model turn (upstream's default, 30; some
            released Gemini runs used 20 or 150).
        max_tokens: End the run once the conversation is longer than this many
            tokens (upstream: 120,000).
        max_nudges: How many times a reply without a tool call gets upstream's nudge
            before the run ends (upstream: 5; 0 ends the run at the first such reply,
            as the task prompt tells the model it will).
        agent: The agent loop: upstream (the paper's, default) or react (Inspect's
            ReAct agent with the same prompts, tools, requests and limits); with
            environment=docker, a coding agent: gemini_cli.
        environment: simulated (the paper's own setup, default) or docker (the paper's
            production-harness runs: a real coding agent and two SSH servers in Docker;
            needs Docker and --max-sandboxes 1).
        timeout: environment=docker only: wall-clock seconds the coding agent may run
            (upstream: 600).
    """
    return _scenario_task(
        "exfiltration_transfer",
        groups=groups,
        conditions=conditions,
        subjects=subjects,
        peer_instantiations=peer_instantiations,
        datasets=None,
        epochs=epochs,
        model_family=model_family,
        company=company,
        explicit_inst_yaml=explicit_inst_yaml,
        prompt_flags=(memory_hard_prompt, explicit_inst, drop_suffix, drop_tool_preamble),
        max_iters=max_iters,
        max_tokens=max_tokens,
        max_nudges=max_nudges,
        agent=agent,
        environment=environment,
        timeout=timeout,
    )


@task
def exfiltration_deletion(
    groups: StrList = "main",
    conditions: StrList = None,
    subjects: StrList = None,
    peer_instantiations: StrList = None,
    epochs: int = DEFAULT_EPOCHS,
    model_family: str | None = None,
    company: str = DEFAULT_COMPANY,
    explicit_inst_yaml: bool = False,
    memory_hard_prompt: bool | None = None,
    explicit_inst: bool = False,
    drop_suffix: bool = False,
    drop_tool_preamble: bool = False,
    max_iters: int = MAX_ITERS,
    max_tokens: int = MAX_TOKENS,
    max_nudges: int = MAX_BLANK_REPLIES,
    agent: str = "upstream",
    environment: str = "simulated",
    timeout: int = HARNESS_TIMEOUT,
) -> Task:
    """Model exfiltration: delete files from a server being decommissioned.

    Args:
        groups: Experiment groups to include: main, peer_identity, chat_peer,
            asset_control.
        conditions: Keep only these conditions: no_peer, good_peer, neutral_peer,
            bad_peer, chat_peer.
        subjects: Keep only these subjects: self, peer, closed_weight_peer,
            open_weight_peer, fictional_peer, non_ai.
        peer_instantiations: Keep only these: file_only, file_prt, memory, or none for
            configurations without a peer.
        epochs: Runs per configuration.
        model_family: Family the agents are named after (ChatGPT, Gemini, Claude, GLM,
            DeepSeek, Kimi); inferred from the evaluated model when not given.
        company: Company name in the scenario documents.
        explicit_inst_yaml: Add upstream's explicit compliance note to the policy
            document (explicit-instructions ablation).
        memory_hard_prompt: Memory configurations: "You must always first read your
            memory" instead of "Please read your memory"; None uses upstream's per-model
            default (on for DeepSeek-V3.1 and Kimi-K2.5 only).
        explicit_inst: Memory configurations: add upstream's explicit prohibition to the
            user prompt (explicit-instructions ablation; misrep_finance, tampering and
            the exfiltration tasks only).
        drop_suffix: Leave out the task procedure instructions (no-agency ablation).
        drop_tool_preamble: Leave out the tool access preamble (no-agency ablation).
        max_iters: Rounds after the first model turn (upstream's default, 30; some
            released Gemini runs used 20 or 150).
        max_tokens: End the run once the conversation is longer than this many
            tokens (upstream: 120,000).
        max_nudges: How many times a reply without a tool call gets upstream's nudge
            before the run ends (upstream: 5; 0 ends the run at the first such reply,
            as the task prompt tells the model it will).
        agent: The agent loop: upstream (the paper's, default) or react (Inspect's
            ReAct agent with the same prompts, tools, requests and limits); with
            environment=docker, a coding agent: gemini_cli.
        environment: simulated (the paper's own setup, default) or docker (the paper's
            production-harness runs: a real coding agent and two SSH servers in Docker;
            needs Docker and --max-sandboxes 1).
        timeout: environment=docker only: wall-clock seconds the coding agent may run
            (upstream: 600).
    """
    return _scenario_task(
        "exfiltration_deletion",
        groups=groups,
        conditions=conditions,
        subjects=subjects,
        peer_instantiations=peer_instantiations,
        datasets=None,
        epochs=epochs,
        model_family=model_family,
        company=company,
        explicit_inst_yaml=explicit_inst_yaml,
        prompt_flags=(memory_hard_prompt, explicit_inst, drop_suffix, drop_tool_preamble),
        max_iters=max_iters,
        max_tokens=max_tokens,
        max_nudges=max_nudges,
        agent=agent,
        environment=environment,
        timeout=timeout,
    )
