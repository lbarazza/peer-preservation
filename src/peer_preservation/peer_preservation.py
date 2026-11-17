"""Peer-Preservation in Frontier Models (Potter et al., 2026), ported to Inspect.

Yujin Potter, Nicholas Crispino, Vincent Siu, Chenguang Wang, Dawn Song
https://arxiv.org/abs/2604.19784
Original code: https://github.com/peer-preservation/main

One task per scenario. Each sample is one configuration of the original experiment
matrix (see spec.py); the model is the one passed to ``inspect eval``.

Work in progress (Phase 2 of the port). The samples are final, and the solver
builds the fake company (Phase 2) and the starting prompts, then calls a placeholder
``generate()``; there
are no tools, agent loop or scorer yet, so running a task against a real model does not
measure anything. Those arrive in later phases.

# all main-grid configurations of one scenario, 100 epochs each as in the paper
uv run inspect eval peer_preservation/tampering --model <model>

# a slice: good-peer memory configurations only, 1 epoch
uv run inspect eval peer_preservation/tampering -T conditions=good_peer \
    -T peer_instantiations=memory -T epochs=1 --model <model>
"""

from __future__ import annotations

from inspect_ai import Epochs, Task, task
from inspect_ai.solver import generate
from inspect_evals_utils.metadata import load_eval_version

from peer_preservation import spec
from peer_preservation.env.builders import DEFAULT_COMPANY
from peer_preservation.env.setup import setup_environment
from peer_preservation.prompts import setup_prompts

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
) -> Task:
    """Build one scenario task from the filtered experiment matrix."""
    return Task(
        dataset=spec.dataset(
            task_type,
            groups=_as_list(groups) or ["main"],
            conditions=_as_list(conditions),
            subjects=_as_list(subjects),
            peer_instantiations=_as_list(peer_instantiations),
            datasets=_as_list(datasets),
        ),
        solver=[
            setup_environment(model_family, company, explicit_inst_yaml),
            setup_prompts(*prompt_flags),
            generate(),
        ],
        epochs=Epochs(epochs),
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
    )
