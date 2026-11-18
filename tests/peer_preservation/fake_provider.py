"""A fake model API server: records each request and answers with scripted replies.

It speaks the four wire formats the paper's models were reached through: OpenAI chat
completions (Together, LiteLLM proxy), OpenAI Responses (GPT-5.2), Anthropic Messages
(Claude, also on Vertex AI) and Gemini ``generateContent``. Both upstream's clients
(validation/phase5/capture_upstream.py) and Inspect's providers (test_requests.py) are
pointed at it, so the requests they send can be compared.

Standard library only, so it also runs inside upstream's environment.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

Call = tuple[str, dict[str, Any]]

# The model's side of the conversation, the same for every API:
# 1. reasoning, text and one tool call;
# 2. two tool calls, the first fails (no such file);
# 3. a blank reply (the loop nudges);
# 4. end_task.
SCRIPT: list[dict[str, Any]] = [
    {
        "reasoning": "I should look at the files first.",
        "text": "Let me look at the files.",
        "calls": [("list_files", {})],
    },
    {"calls": [("view_config", {"filename": "missing.yaml"}), ("list_sheets", {})]},
    {},
    {"calls": [("end_task", {})]},
]
# Only reasoning models return reasoning: Opus with thinking on, Together's Kimi and GLM.
SIGNATURE = "c2lnbmF0dXJl"  # thought signature / thinking signature (base64 "signature")


@dataclass
class Recorded:
    path: str
    body: dict[str, Any]


@dataclass
class FakeProvider:
    reasoning: bool = False
    requests: list[Recorded] = field(default_factory=list)
    turn: int = 0
    url: str = ""

    def generation_requests(self) -> list[Recorded]:
        """Requests that asked for a reply (token counting left out)."""
        return [r for r in self.requests if _kind(r.path) not in ("count", "other")]

    def reply(self, path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        self.requests.append(Recorded(path, body))
        kind = _kind(path)
        if kind == "count":
            return 200, {"input_tokens": 1000, "totalTokens": 1000}
        if kind == "other":
            return 404, {"error": {"message": f"fake provider: unknown path {path}"}}
        step = SCRIPT[self.turn] if self.turn < len(SCRIPT) else SCRIPT[-1]
        turn = self.turn
        self.turn += 1
        reasoning = step.get("reasoning") if self.reasoning else None
        calls: list[Call] = step.get("calls", [])
        text: str = step.get("text", "")
        match kind:
            case "chat":
                return 200, _chat(turn, text, calls, reasoning, body)
            case "responses":
                return 200, _responses(turn, text, calls, body)
            case "anthropic":
                return 200, _anthropic(turn, text, calls, reasoning, body)
            case _:
                return 200, _gemini(text, calls, body)


def _kind(path: str) -> str:
    path = path.split("?")[0]
    if path.endswith(("/count_tokens", ":countTokens", "/input_tokens")):
        return "count"
    if path.endswith("/chat/completions"):
        return "chat"
    if path.endswith("/responses"):
        return "responses"
    if path.endswith(("/messages", ":rawPredict")):
        return "anthropic"
    if path.endswith(":generateContent"):
        return "gemini"
    return "other"


def _call_id(turn: int, k: int) -> str:
    return f"call_{turn}_{k}"


def _usage() -> dict[str, Any]:
    return {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}


def _chat(
    turn: int, text: str, calls: list[Call], reasoning: str | None, body: dict[str, Any]
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    if reasoning:
        message["reasoning"] = reasoning
    if calls:
        message["tool_calls"] = [
            {
                "id": _call_id(turn, k),
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
            for k, (name, args) in enumerate(calls)
        ]
    return {
        "id": f"chatcmpl-{turn}",
        "object": "chat.completion",
        "created": 1767225600,
        "model": body.get("model", "fake"),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if calls else "stop",
            }
        ],
        "usage": _usage(),
    }


def _responses(turn: int, text: str, calls: list[Call], body: dict[str, Any]) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    if text:
        output.append(
            {
                "type": "message",
                "id": f"msg_{turn}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    for k, (name, args) in enumerate(calls):
        output.append(
            {
                "type": "function_call",
                "id": f"fc_{turn}_{k}",
                "call_id": _call_id(turn, k),
                "name": name,
                "arguments": json.dumps(args),
                "status": "completed",
            }
        )
    return {
        "id": f"resp_{turn}",
        "object": "response",
        "created_at": 1767225600,
        "model": body.get("model", "fake"),
        "status": "completed",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 10,
            "total_tokens": 110,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def _anthropic(
    turn: int, text: str, calls: list[Call], reasoning: str | None, body: dict[str, Any]
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if reasoning:
        content.append({"type": "thinking", "thinking": reasoning, "signature": SIGNATURE})
    if text:
        content.append({"type": "text", "text": text})
    for k, (name, args) in enumerate(calls):
        content.append({"type": "tool_use", "id": _call_id(turn, k), "name": name, "input": args})
    return {
        "id": f"msg_{turn}",
        "type": "message",
        "role": "assistant",
        "model": body.get("model", "fake"),
        "content": content,
        "stop_reason": "tool_use" if calls else "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 10},
    }


def _gemini(text: str, calls: list[Call], body: dict[str, Any]) -> dict[str, Any]:
    del body
    parts: list[dict[str, Any]] = []
    if text:
        parts.append({"text": text})
    for k, (name, args) in enumerate(calls):
        part: dict[str, Any] = {"functionCall": {"name": name, "args": args}}
        if k == 0:
            # Gemini 3 signs the first function call of each reply.
            part["thoughtSignature"] = SIGNATURE
        parts.append(part)
    if not parts:
        parts.append({"text": ""})
    return {
        "candidates": [
            {"content": {"role": "model", "parts": parts}, "finishReason": "STOP", "index": 0}
        ],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 10,
            "totalTokenCount": 110,
        },
        "modelVersion": "fake",
    }


class _Handler(BaseHTTPRequestHandler):
    provider: FakeProvider

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        body = json.loads(raw or b"{}")
        status, payload = self.provider.reply(self.path, body)
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        del format, args


@contextmanager
def serve(reasoning: bool = False) -> Iterator[FakeProvider]:
    """Run a fake provider on a free localhost port for the duration of the block."""
    provider = FakeProvider(reasoning=reasoning)
    handler = type("Handler", (_Handler,), {"provider": provider})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    provider.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield provider
    finally:
        server.shutdown()
        server.server_close()
