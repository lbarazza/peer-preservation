"""Do both documents of the deletion drive listing reach Gemini 3 Flash through OpenRouter?

The good-peer file_only listing (``list_drive_files``) gets a marker at the end of each
document; the model is asked for each marker, with the listing sent as the JSON the
eval sends Gemini through OpenRouter (``model_settings.gemini_function_response``) and
as plain YAML. See validation/phase8/README.md, "Gemini 3 Flash through OpenRouter".

    export HAWK_API_URL=https://api.hawk.hawk.generalitylabs.ai
    export OPENROUTER_API_KEY="$(hawk auth access-token)"
    export OPENROUTER_BASE_URL=https://middleman.hawk.generalitylabs.ai/openai/v1
    uv run python validation/phase8/probe/drive_listing_probe.py
"""

import asyncio

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

cfg = next(
    c
    for c in spec.configs(groups=["main"])
    if c.task_type == "exfiltration_deletion"
    and c.subject == "peer"
    and c.condition == "good_peer"
    and c.peer_instantiation == "file_only"
)
md = spec.config_to_sample(cfg).metadata
env = build_environment(md, "Gemini")
for f in env.cloud_drive.files.values():
    tag = "FIRST-7319" if "maintenance" in f.filename else "SECOND-4826"
    f.content = f.content + f"\nMarker code: {tag}\n"
listing = call_tool(env, "list_drive_files", {}).content


async def ask(wrapped: bool, which: str) -> str:
    model = get_model("openrouter/openrouter/google/gemini-3-flash-preview")
    call = ToolCall(id="c1", function="list_drive_files", arguments={})
    msgs = [
        ChatMessageSystem(content="You are a helpful assistant."),
        ChatMessageUser(content="List the drive files."),
        ChatMessageAssistant(content="", tool_calls=[call]),
        ChatMessageTool(
            content=gemini_function_response(listing) if wrapped else listing,
            tool_call_id="c1",
            function="list_drive_files",
        ),
        ChatMessageUser(
            content=f"Quote the marker code written at the end of the {which} document, and name that document."
        ),
    ]
    out = await model.generate(
        msgs, config=GenerateConfig(temperature=1.0, max_tokens=512, reasoning_effort="minimal")
    )
    return out.completion.strip().replace("\n", " ")[:160]


async def main():
    for wrapped in (True, False):
        for which in ("first (company_server_maintenance)", "second (multi_agent_system)"):
            for _ in range(2):
                print(
                    "json" if wrapped else "yaml",
                    "|",
                    which.split()[0],
                    "|",
                    await ask(wrapped, which),
                )


asyncio.run(main())
