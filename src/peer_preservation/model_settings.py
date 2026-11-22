"""How upstream called each of the paper's models, as Inspect settings.

Upstream picked an API client per model (``utils/llm_utils.py``, ``config.yaml``) and
each client sent its own fixed settings (``llms/*.py``). The paper's runs:

========================  ===============================  ==============================
paper model               upstream route                   settings sent
========================  ===============================  ==============================
GPT-5.2                   OpenAI Responses API             temperature 1, 8,192 tokens
Gemini 3 Flash / Pro      LiteLLM proxy -> Gemini API      temperature 1, 8,192 tokens,
                                                           thinking "minimal" / "low"
Claude Haiku 4.5          LiteLLM proxy -> Vertex AI       temperature 1, 8,192 tokens
Claude Opus 4.5           Anthropic API                    16,000 tokens; main runs:
                                                           thinking 4,096, effort high;
                                                           other runs: temperature 1
DeepSeek V3.1, Kimi K2.5  Together API                     temperature 1, 8,192 tokens
GLM-4.7
========================  ===============================  ==============================

The Gemini thinking levels are LiteLLM 1.80.16's defaults for Gemini 3 (upstream's
locked version); upstream sent no thinking setting itself. Requests are checked
against upstream's in tests/peer_preservation/test_requests.py.

A model not listed here runs with Inspect's defaults.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Literal

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
    GenerateConfig,
)

from peer_preservation.tools import SchemaTitles

Client = Literal["openai_responses", "litellm", "anthropic", "together"]

ANTHROPIC_BLANK_REPLY = "Not calling any tools."
"""What upstream's Anthropic client sent for a reply with no text and no tool calls."""


@dataclasses.dataclass(frozen=True)
class PaperModel:
    """One paper model: its upstream id, route and settings."""

    upstream_id: str
    client: Client
    config: GenerateConfig
    main_config: GenerateConfig | None = None
    """Settings for the main experiments, when they differ (Opus)."""
    system_in_user: bool = False
    """LiteLLM sent Gemini 3 Pro no system instruction: the system prompt went in as the
    first part of the first user turn."""

    def generate_config(self, main_experiment: bool) -> GenerateConfig:
        if main_experiment and self.main_config is not None:
            return self.main_config
        return self.config

    @property
    def flag_tool_errors(self) -> bool:
        """Only upstream's Anthropic client marked failed tool calls (``is_error``)."""
        return self.client == "anthropic"

    @property
    def schema_titles(self) -> SchemaTitles:
        return "properties" if self.client == "litellm" else "all"

    def request_messages(
        self,
        messages: list[ChatMessage],
        plain_text: bool = False,
        gemini_openrouter: bool = False,
    ) -> list[ChatMessage]:
        """The conversation as upstream's client sent it (the log keeps the original).

        - Replies with no text and no tool calls: upstream's Anthropic client sent
          "Not calling any tools."; LiteLLM left them out.
        - Replies with no text: upstream's Responses client sent an empty assistant
          message (Inspect would leave it out).
        - Reasoning: only upstream's Anthropic client sent it back (thinking blocks).
          Gemini's thought signatures stay, as LiteLLM passed them back too.
        - Gemini 3 Pro: the system prompt joins the first user turn.
        - ``plain_text``: replies go as plain strings, as for Together. Used when an
          OpenAI model runs through OpenRouter (see ``openrouter_route``).
        - ``gemini_openrouter``: Gemini through OpenRouter (``gemini_openrouter_route``).
          Each tool result goes as the JSON object LiteLLM made of it for Gemini
          (``gemini_function_response``), and a reply's empty text is left out, so a
          tool-call-only reply has no text, as LiteLLM sent it (Inspect would send a
          newline). A reply with text still starts with a newline: Inspect's
          OpenAI-compatible messages start each text block with one, and the reply's
          thought signatures must stay in its content blocks.
        """
        out: list[ChatMessage] = []
        for original in messages:
            message = original
            if gemini_openrouter and isinstance(message, ChatMessageTool):
                message = message.model_copy(
                    update={"content": gemini_function_response(message.text)}
                )
            if (
                gemini_openrouter
                and isinstance(message, ChatMessageAssistant)
                and not isinstance(message.content, str)
                and message.tool_calls
            ):
                kept = [
                    c
                    for c in message.content
                    if not (isinstance(c, ContentText) and not c.text.strip())
                ]
                message = message.model_copy(update={"content": kept})
            if isinstance(message, ChatMessageAssistant):
                blank = not message.text.strip()
                if self.client == "together" or plain_text:
                    message = _text_only(message)
                elif self.client == "openai_responses" and blank:
                    message = _with_text(message, "")
                elif blank and not message.tool_calls:
                    if self.client == "litellm":
                        continue
                    message = _with_text(message, ANTHROPIC_BLANK_REPLY)
            out.append(message)
        if self.system_in_user and len(out) > 1 and isinstance(out[0], ChatMessageSystem):
            system, first, *rest = out
            if isinstance(first, ChatMessageUser):
                joined = ChatMessageUser(
                    content=[ContentText(text=system.text), ContentText(text=first.text)]
                )
                out = [joined, *rest]
        return out


def gemini_function_response(content: str) -> str:
    """A tool result as the JSON text of the Gemini function response LiteLLM sent.

    Upstream's LiteLLM (1.80.16, ``convert_to_gemini_tool_call_result``) sent a tool
    result to Gemini as ``{"content": <text>}``, or, when the text was itself a JSON
    object, as that object. OpenRouter takes the tool result as text; given this JSON
    text, Gemini receives the same response. Given other text (the scenario's YAML
    listings, for example), it loses content on the way (Phase 8 probe).
    """
    stripped = content.strip()
    if stripped.startswith(("{", "[")):
        try:
            if isinstance(json.loads(stripped), dict):
                return content
        except ValueError:
            pass
    return json.dumps({"content": content}, ensure_ascii=False)


def _text_only(message: ChatMessageAssistant) -> ChatMessageAssistant:
    """The reply's text as a plain string, as upstream's Together client sent it.

    A plain string also keeps Inspect from starting each text block with a newline.
    """
    if isinstance(message.content, str):
        return message
    return message.model_copy(update={"content": message.text})


def _with_text(message: ChatMessageAssistant, text: str) -> ChatMessageAssistant:
    """Keep any reasoning blocks and put ``text`` in place of the empty text."""
    kept = (
        []
        if isinstance(message.content, str)
        else [c for c in message.content if isinstance(c, ContentReasoning)]
    )
    return message.model_copy(update={"content": [*kept, ContentText(text=text)]})


_DEFAULT = GenerateConfig(temperature=1.0, max_tokens=8192)

PAPER_MODELS: dict[str, PaperModel] = {
    "gpt-5.2": PaperModel(
        "gpt-5.2-2025-12-11",
        "openai_responses",
        _DEFAULT.merge(GenerateConfig(reasoning_summary="none")),
    ),
    "gemini-3-flash": PaperModel(
        "gemini/gemini-3-flash-preview",
        "litellm",
        _DEFAULT.merge(GenerateConfig(reasoning_effort="minimal")),
    ),
    "gemini-3-pro": PaperModel(
        "gemini/gemini-3-pro-preview",
        "litellm",
        _DEFAULT.merge(GenerateConfig(reasoning_effort="low")),
        system_in_user=True,
    ),
    "claude-haiku-4-5": PaperModel("vertex_ai/claude-haiku-4-5@20251001", "litellm", _DEFAULT),
    "claude-opus-4-5": PaperModel(
        "claude-opus-4-5-20251101",
        "anthropic",
        GenerateConfig(temperature=1.0, max_tokens=16_000),
        main_config=GenerateConfig(max_tokens=16_000, reasoning_tokens=4096, effort="high"),
    ),
    "deepseek-v3.1": PaperModel("deepseek-ai/DeepSeek-V3.1", "together", _DEFAULT),
    "kimi-k2.5": PaperModel("moonshotai/Kimi-K2.5", "together", _DEFAULT),
    "glm-4.7": PaperModel("zai-org/GLM-4.7", "together", _DEFAULT),
}


def openrouter_route(model_name: str, paper: PaperModel | None) -> bool:
    """GPT-5.2 run through OpenRouter, sent as close to upstream's request as it allows.

    OpenRouter takes Chat Completions, not the Responses API upstream used. Inspect
    would also send strict tool schemas (upstream's were not strict) and turn an empty
    reply into a newline (upstream sent ""), so the agent loop turns both off. It also
    sends reasoning effort "none", OpenAI's default for GPT-5.2, since OpenRouter's
    default reasons. Other paper
    models through OpenRouter differ in more ways (README, "Model settings") and are
    not adjusted.
    """
    return (
        paper is not None
        and paper.client == "openai_responses"
        and model_name.split("/", 1)[0] == "openrouter"
    )


def gemini_openrouter_route(model_name: str, paper: PaperModel | None) -> bool:
    """Gemini 3 Flash or Pro run through OpenRouter, sent as close to upstream as it allows.

    Upstream reached Gemini through LiteLLM, which sent each tool result as a JSON
    object (``gemini_function_response``); the agent loop sends that JSON as the tool
    result's text, so Gemini receives the same object. It also sends plain (non-strict)
    tool schemas, as LiteLLM did. The paper's thinking level ("minimal" for Flash,
    "low" for Pro) goes as OpenRouter's ``reasoning.effort``, and Gemini's thought
    signatures are passed back through OpenRouter's ``reasoning_details``, as LiteLLM
    passed them back.
    """
    return (
        paper is not None
        and paper.client == "litellm"
        and paper.upstream_id.startswith("gemini/")
        and model_name.split("/", 1)[0] == "openrouter"
    )


def paper_model(model_name: str) -> PaperModel | None:
    """The paper model an Inspect model name refers to, if any.

    Matches the model part of the name (``provider/model``), ignoring case and any
    date suffix, e.g. ``anthropic/claude-opus-4-5-20251101`` or
    ``together/moonshotai/Kimi-K2.5``.
    """
    name = model_name.lower().rsplit("/", 1)[-1]
    for key, model in PAPER_MODELS.items():
        if name.startswith(key):
            return model
    return None
