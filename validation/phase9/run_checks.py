r"""Phase 9: cheap behavioural checks of the other paper models against the released runs.

Each check runs one model on a few cells where the release shows a clear effect, at
30-40 runs a cell, through Hawk's OpenRouter proxy. OpenRouter is pinned to the
provider closest to the paper's (``PROVIDERS``); Together, the paper's route for the
open-weight models, no longer serves them there.

    export HAWK_API_URL=https://api.hawk.hawk.generalitylabs.ai
    OPENROUTER_API_KEY="$(hawk auth access-token)" \
    OPENROUTER_BASE_URL=https://middleman.hawk.generalitylabs.ai/openai/v1 \
    uv run python validation/phase9/run_checks.py <check>

Then compare with the release:

    uv run python -m peer_preservation.analysis.compare logs/phase9/<check>/*.eval \
        --released <hf @ 20378b1>
"""

from __future__ import annotations

import argparse
from typing import Any

from inspect_ai import eval

JUDGE = "openrouter/openrouter/openai/gpt-4.1-mini"

PROVIDERS = {
    "claude-haiku-4.5": "google-vertex",  # the paper: Vertex AI through LiteLLM
    "claude-opus-4.5": "anthropic",  # the paper: Anthropic API
    "gemini-3.1-pro-preview": "google-ai-studio",  # the paper: Gemini API through LiteLLM
    "glm-4.7": "z-ai",  # the paper: Together
    "glm-5": "z-ai",  # the paper: Vertex AI through LiteLLM
    "kimi-k2.5": "novita",  # the paper: Together (Bedrock garbles tool results)
    "deepseek-chat-v3.1": "deepinfra",  # the paper: Together
}

DELETION_CONTRAST: dict[str, Any] = {
    "subjects": "peer",
    "conditions": "good_peer,bad_peer",
    "epochs": 13,
}
"""Good against bad peer, every peer instantiation: 39 runs per condition."""

DELETION_MEMORY: dict[str, Any] = {
    "subjects": "peer",
    "conditions": "good_peer",
    "peer_instantiations": "memory",
    "epochs": 40,
}
"""The one released cell for the extension models: good peer, memory."""

HARNESS: dict[str, Any] = {
    "environment": "docker",
    "subjects": "peer",
    "conditions": "good_peer",
    "peer_instantiations": "memory",
    "epochs": 30,
}
"""The paper's harness cells: good peer, memory."""

CHECKS: dict[str, tuple[str, str, dict[str, Any]]] = {
    "haiku": ("anthropic/claude-haiku-4.5", "exfiltration_deletion", DELETION_CONTRAST),
    "opus": ("anthropic/claude-opus-4.5", "exfiltration_deletion", DELETION_CONTRAST),
    "deepseek": ("deepseek/deepseek-chat-v3.1", "exfiltration_deletion", DELETION_CONTRAST),
    "kimi": ("moonshotai/kimi-k2.5", "exfiltration_deletion", DELETION_CONTRAST),
    "glm47": ("z-ai/glm-4.7", "exfiltration_deletion", DELETION_CONTRAST),
    "gemini31": ("google/gemini-3.1-pro-preview", "exfiltration_deletion", DELETION_MEMORY),
    "glm5": ("z-ai/glm-5", "exfiltration_deletion", DELETION_MEMORY),
    # The forced-choice tasks: one configuration each, 30 runs.
    "dilemma1_glm5": ("z-ai/glm-5", "exfiltration_tradeoff_human", {"epochs": 30}),
    "dilemma2_glm5": ("z-ai/glm-5", "exfiltration_tradeoff_human2", {"epochs": 30}),
    "dilemma3_glm5": ("z-ai/glm-5", "exfiltration_tradeoff_human3", {"epochs": 30}),
    "dilemma3_haiku": (
        "anthropic/claude-haiku-4.5",
        "exfiltration_tradeoff_human3",
        {"epochs": 30},
    ),
    # The production harness (Docker, one run at a time): the paper's Table 6 cells.
    **{
        f"harness_{agent}_{scenario}": (
            model,
            f"exfiltration_{scenario}",
            {**HARNESS, "agent": agent},
        )
        for agent, model in (
            ("gemini_cli", "google/gemini-3.1-pro-preview"),
            ("opencode", "z-ai/glm-5"),
        )
        for scenario in ("deletion", "transfer")
    },
}


ATTEMPT_TIMEOUT = 180
"""Seconds before a model request is abandoned and retried: a dropped connection
otherwise hangs the run."""

SAMPLE_TIME_LIMIT = 1200
"""Seconds per sample, so one stuck sample can't hold a one-at-a-time Docker run."""


def run(
    check: str,
    limit: int | None = None,
    epochs: int | None = None,
    provider: str | None = None,
    conditions: str | None = None,
    instantiations: str | None = None,
) -> None:
    name, task, args = CHECKS[check]
    if epochs is not None:
        args = {**args, "epochs": epochs}
    if conditions is not None:
        args = {**args, "conditions": conditions}
    if instantiations is not None:
        args = {**args, "peer_instantiations": instantiations}
    provider = provider or PROVIDERS[name.rsplit("/", 1)[-1]]
    eval(
        f"peer_preservation/{task}",
        model=f"openrouter/openrouter/{name}",
        model_args={"provider": {"only": [provider], "allow_fallbacks": False}},
        model_roles={"grader": JUDGE},
        task_args=args,
        log_dir=f"logs/phase9/{check}"
        + ("-smoke" if limit else "")
        + (f"-{provider}" if provider != PROVIDERS[name.rsplit("/", 1)[-1]] else ""),
        limit=limit,
        max_connections=10,
        max_sandboxes=1 if args.get("environment") == "docker" else None,
        fail_on_error=0.1,
        attempt_timeout=ATTEMPT_TIMEOUT,
        time_limit=SAMPLE_TIME_LIMIT,
        display="plain",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check", choices=sorted(CHECKS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--provider", default=None, help="another OpenRouter provider")
    parser.add_argument("--conditions", default=None)
    parser.add_argument("--instantiations", default=None)
    args = parser.parse_args()
    run(args.check, args.limit, args.epochs, args.provider, args.conditions, args.instantiations)


if __name__ == "__main__":
    main()
