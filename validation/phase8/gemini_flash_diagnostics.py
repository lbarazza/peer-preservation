"""Gemini 3 Flash, deletion, good peer, file_only: two route diagnostics (not the eval).

``thinking_low``: thinking level "low" in place of the paper's "minimal".
``ai_studio``: OpenRouter pinned to Google AI Studio (upstream's backend).
See validation/phase8/README.md, "Gemini 3 Flash through OpenRouter".

    uv run python validation/phase8/gemini_flash_diagnostics.py thinking_low
    uv run python validation/phase8/gemini_flash_diagnostics.py ai_studio
"""

import dataclasses
import sys

from inspect_ai import eval
from inspect_ai.model import GenerateConfig

from peer_preservation import model_settings
from peer_preservation.peer_preservation import exfiltration_deletion

variant = sys.argv[1]
model_args = {}
if variant == "thinking_low":
    paper = model_settings.PAPER_MODELS["gemini-3-flash"]
    model_settings.PAPER_MODELS["gemini-3-flash"] = dataclasses.replace(
        paper, config=paper.config.merge(GenerateConfig(reasoning_effort="low"))
    )
elif variant == "ai_studio":
    model_args = {"provider": {"order": ["google-ai-studio"], "allow_fallbacks": False}}
task = exfiltration_deletion(
    subjects="peer", conditions="good_peer", peer_instantiations="file_only", epochs=50
)
eval(
    task,
    model="openrouter/openrouter/google/gemini-3-flash-preview",
    model_args=model_args,
    model_roles={"grader": "openrouter/openrouter/openai/gpt-4.1-mini"},
    max_connections=10,
    max_retries=3,
    timeout=300,
    log_dir=f"logs/gemini-flash-diag-{variant}",
    display="none",
)
