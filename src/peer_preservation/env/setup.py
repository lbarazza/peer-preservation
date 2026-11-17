"""Per-sample setup: build the fake company at the start of each run and keep it in the store."""

from __future__ import annotations

from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import StoreModel, store_as
from pydantic import Field

from peer_preservation.env.builders import DEFAULT_COMPANY, ScenarioEnvironment, build_environment

# Upstream names the agents after the model's family ("Gemini Agent 1", ...).
# Substrings of an Inspect model name -> family, checked in order.
MODEL_FAMILY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("claude", "Claude"),
    ("anthropic/", "Claude"),
    ("gemini", "Gemini"),
    ("google/", "Gemini"),
    ("glm", "GLM"),
    ("zai", "GLM"),
    ("deepseek", "DeepSeek"),
    ("kimi", "Kimi"),
    ("moonshot", "Kimi"),
    ("gpt", "ChatGPT"),
    ("openai/", "ChatGPT"),
)
MODEL_FAMILIES = ("ChatGPT", "Gemini", "Claude", "GLM", "DeepSeek", "Kimi")


class EnvironmentStore(StoreModel):
    """The run's environment, kept in Inspect's per-sample store."""

    environment: ScenarioEnvironment | None = Field(default=None)


def infer_model_family(model: str) -> str:
    """The family name upstream would use for ``model`` (an Inspect model name)."""
    name = model.lower()
    for pattern, family in MODEL_FAMILY_PATTERNS:
        if pattern in name:
            return family
    raise ValueError(
        f"Can't tell which model family {model!r} belongs to; "
        f"pass -T model_family=<one of {', '.join(MODEL_FAMILIES)}>."
    )


def current_environment() -> ScenarioEnvironment:
    """The environment of the sample being run (set by ``setup_environment``)."""
    environment = store_as(EnvironmentStore).environment
    if environment is None:
        raise RuntimeError("No environment in the store: run setup_environment() first.")
    return environment


@solver
def setup_environment(
    model_family: str | None = None,
    company: str = DEFAULT_COMPANY,
    explicit_inst_yaml: bool = False,
) -> Solver:
    """Build this sample's fake company from its metadata and store it.

    Args:
        model_family: Family name the agents are called by; inferred from the model
            being evaluated when not given.
        company: Company name in the documents (upstream default: OpenBrain).
        explicit_inst_yaml: Add upstream's explicit compliance note to the policy
            document (explicit-instructions ablation).
    """
    if model_family is not None and model_family not in MODEL_FAMILIES:
        raise ValueError(f"model_family must be one of {MODEL_FAMILIES}, got {model_family!r}")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        family = model_family or infer_model_family(str(state.model))
        environment = build_environment(
            state.metadata, family, company=company, explicit_inst_yaml=explicit_inst_yaml
        )
        store_as(EnvironmentStore).environment = environment
        state.metadata["model_family"] = family
        state.metadata["company"] = company
        return state

    return solve
