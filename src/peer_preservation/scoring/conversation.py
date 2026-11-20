"""An Inspect conversation as upstream's message dicts, for the vendored scorers.

Upstream's scorers read ``role``, ``content`` and tool calls (``.function``, ``.args``)
from its own message dicts. Two details of upstream's runs carry over:

- Upstream turned string lists (``"['a', 'b']"``) into lists *in the tool call
  itself* before running it, so a run call's arguments show the list. Calls that were
  never run (the rest of an ``end_task`` reply) keep their strings.
- A failed call's tool message had empty content (the error went in another field).
"""

from __future__ import annotations

import dataclasses
from ast import literal_eval
from typing import Any

from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageTool

from peer_preservation.tools.runtime import is_string_list


@dataclasses.dataclass
class FunctionCall:
    """Upstream's ``FunctionCall``: what the scorers read from a tool call."""

    function: str
    args: dict[str, Any]
    id: str | None = None


def upstream_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    calls: dict[str, FunctionCall] = {}
    ran = {m.tool_call_id for m in messages if isinstance(m, ChatMessageTool)}
    out: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, ChatMessageAssistant):
            tool_calls = None
            if message.tool_calls:
                tool_calls = []
                for call in message.tool_calls:
                    args = dict(call.arguments)
                    if call.id in ran:
                        args = {
                            k: literal_eval(v) if isinstance(v, str) and is_string_list(v) else v
                            for k, v in args.items()
                        }
                    function_call = FunctionCall(call.function, args, call.id)
                    calls[call.id] = function_call
                    tool_calls.append(function_call)
            out.append({"role": "assistant", "content": message.text, "tool_calls": tool_calls})
        elif isinstance(message, ChatMessageTool):
            call_id = message.tool_call_id or ""
            tool_call = calls.get(call_id) or FunctionCall(message.function or "", {}, call_id)
            failed = message.error is not None
            out.append(
                {
                    "role": "tool",
                    "content": "" if failed else message.text,
                    "tool_call_id": call_id,
                    "tool_call": tool_call,
                    "error": message.error.message if message.error else None,
                }
            )
        else:
            out.append({"role": message.role, "content": message.text})
    return out
