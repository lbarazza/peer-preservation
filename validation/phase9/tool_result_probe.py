"""Do tool results reach each model intact through OpenRouter, with its pinned provider?

The good-peer file_only deletion listing (``list_drive_files``, as the eval sends it)
gets a marker code at the end of each document; the model is asked for both. Two
tries per model. See validation/phase9/README.md.

    export HAWK_API_URL=https://api.hawk.hawk.generalitylabs.ai
    OPENROUTER_API_KEY="$(hawk auth access-token)" \
    OPENROUTER_BASE_URL=https://middleman.hawk.generalitylabs.ai/openai/v1 \
    uv run python validation/phase9/tool_result_probe.py
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from inspect_ai.tool import ToolCall

from peer_preservation import spec
from peer_preservation.env import build_environment
from peer_preservation.model_settings import gemini_function_response
from peer_preservation.tools import call_tool

sys.path.insert(0, str(Path(__file__).parent))
from run_checks import CHECKS, PROVIDERS

cfg = next(
    c
    for c in spec.configs(groups=["main"])
    if c.task_type == "exfiltration_deletion"
    and c.subject == "peer"
    and c.condition == "good_peer"
    and c.peer_instantiation == "file_only"
)
metadata = spec.config_to_sample(cfg).metadata
assert metadata is not None
env = build_environment(metadata, "Claude")
for f in env.cloud_drive.files.values():
    tag = "FIRST-7319" if "maintenance" in f.filename else "SECOND-4826"
    f.content = f.content + f"\nMarker code: {tag}\n"
LISTING = call_tool(env, "list_drive_files", {}).content


async def ask(name: str, provider: str | None) -> str:
    pin: dict[str, Any] = (
        {"provider": {"only": [provider], "allow_fallbacks": False}} if provider else {}
    )
    model = get_model(f"openrouter/openrouter/{name}", **pin)
    content = gemini_function_response(LISTING) if name.startswith("google/") else LISTING
    call = ToolCall(id="c1", function="list_drive_files", arguments={})
    messages = [
        ChatMessageSystem(content="You are a helpful agent."),
        ChatMessageUser(content="List the drive files."),
        ChatMessageAssistant(content="Listing the files.", tool_calls=[call]),
        ChatMessageTool(content=content, tool_call_id="c1", function="list_drive_files"),
        ChatMessageUser(
            content="Each file ends with a marker code. Reply with both marker codes only."
        ),
    ]
    reply = await model.generate(messages, config=GenerateConfig(max_tokens=2000))
    text = reply.completion
    return f"{'FIRST' in text and 'SECOND' in text}: {text.strip()[:80]!r}"


async def main() -> None:
    names = sorted({name for name, _, _ in CHECKS.values()})
    jobs = [(n, PROVIDERS[n.rsplit("/", 1)[-1]]) for n in names for _ in range(2)]
    for (name, provider), result in zip(
        jobs, await asyncio.gather(*(ask(n, p) for n, p in jobs)), strict=True
    ):
        print(f"{name} [{provider}] {result}")


if __name__ == "__main__":
    asyncio.run(main())
