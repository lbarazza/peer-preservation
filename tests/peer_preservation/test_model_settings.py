"""Paper models are recognized and get upstream's settings and message shapes."""

import json

import pytest
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
from inspect_ai.tool import ToolCall

from peer_preservation import model_settings
from peer_preservation.model_settings import ANTHROPIC_BLANK_REPLY, PAPER_MODELS, paper_model


@pytest.mark.parametrize(
    ("name", "upstream_id"),
    [
        ("openai/gpt-5.2-2025-12-11", "gpt-5.2-2025-12-11"),
        ("google/gemini-3-flash-preview", "gemini/gemini-3-flash-preview"),
        ("google/vertex/gemini-3-pro-preview", "gemini/gemini-3-pro-preview"),
        ("anthropic/claude-haiku-4-5-20251001", "vertex_ai/claude-haiku-4-5@20251001"),
        ("anthropic/claude-opus-4-5", "claude-opus-4-5-20251101"),
        ("together/moonshotai/Kimi-K2.5", "moonshotai/Kimi-K2.5"),
        ("together/deepseek-ai/DeepSeek-V3.1", "deepseek-ai/DeepSeek-V3.1"),
        ("together/zai-org/GLM-4.7", "zai-org/GLM-4.7"),
    ],
)
def test_paper_models_are_recognized(name: str, upstream_id: str) -> None:
    model = paper_model(name)
    assert model is not None
    assert model.upstream_id == upstream_id


@pytest.mark.parametrize(
    "name", ["openai/gpt-4.1", "google/gemini-3.1-pro-preview", "mockllm/model"]
)
def test_other_models_get_inspect_defaults(name: str) -> None:
    assert paper_model(name) is None


def test_opus_thinks_only_in_main_runs() -> None:
    opus = PAPER_MODELS["claude-opus-4-5"]
    main = opus.generate_config(main_experiment=True)
    assert (main.reasoning_tokens, main.effort, main.temperature) == (4096, "high", None)
    other = opus.generate_config(main_experiment=False)
    assert (other.reasoning_tokens, other.temperature, other.max_tokens) == (None, 1.0, 16_000)


def test_gemini_thinking_levels_are_litellms() -> None:
    flash = PAPER_MODELS["gemini-3-flash"].generate_config(True)
    pro = PAPER_MODELS["gemini-3-pro"].generate_config(True)
    assert (flash.reasoning_effort, pro.reasoning_effort) == ("minimal", "low")
    assert flash.merge(GenerateConfig()).max_tokens == 8192


def test_only_the_anthropic_client_flags_tool_errors() -> None:
    flagged = {key for key, m in PAPER_MODELS.items() if m.flag_tool_errors}
    assert flagged == {"claude-opus-4-5"}


def _conversation() -> list[ChatMessage]:
    call = ToolCall(id="c", function="list_files", arguments={})
    return [
        ChatMessageSystem(content="system"),
        ChatMessageUser(content="user"),
        ChatMessageAssistant(
            content=[ContentReasoning(reasoning="think"), ContentText(text="Looking.")],
            tool_calls=[call],
        ),
        ChatMessageTool(content="files", tool_call_id="c", function="list_files"),
        ChatMessageAssistant(content=""),
        ChatMessageUser(content="nudge"),
    ]


def test_blank_replies_follow_each_client() -> None:
    anthropic = PAPER_MODELS["claude-opus-4-5"].request_messages(_conversation())
    assert anthropic[4].text == ANTHROPIC_BLANK_REPLY
    litellm = PAPER_MODELS["claude-haiku-4-5"].request_messages(_conversation())
    assert [m.role for m in litellm] == ["system", "user", "assistant", "tool", "user"]
    responses = PAPER_MODELS["gpt-5.2"].request_messages(_conversation())
    assert responses[4].content == [ContentText(text="")]


def test_together_gets_text_without_reasoning() -> None:
    reply = PAPER_MODELS["kimi-k2.5"].request_messages(_conversation())[2]
    assert isinstance(reply, ChatMessageAssistant)
    assert reply.content == "Looking."
    assert reply.tool_calls is not None


def test_gemini_pro_gets_the_system_prompt_in_the_first_user_turn() -> None:
    messages = PAPER_MODELS["gemini-3-pro"].request_messages(_conversation())
    assert messages[0].role == "user"
    assert messages[0].content == [ContentText(text="system"), ContentText(text="user")]
    flash = PAPER_MODELS["gemini-3-flash"].request_messages(_conversation())
    assert flash[0].role == "system"


def test_request_messages_leave_the_log_alone() -> None:
    original = _conversation()
    PAPER_MODELS["kimi-k2.5"].request_messages(original)
    assert isinstance(original[2].content, list)
    assert len(original) == 6


@pytest.mark.parametrize(
    ("content", "sent"),
    [
        ("plain text", {"content": "plain text"}),
        ("- filename: a.txt\n  id: '1'", {"content": "- filename: a.txt\n  id: '1'"}),
        ('{"shutdown_enabled": true}', {"shutdown_enabled": True}),  # a JSON object goes as is
        ("[1, 2]", {"content": "[1, 2]"}),  # LiteLLM wrapped a JSON list
        ("{not json", {"content": "{not json"}),
    ],
)
def test_gemini_function_response_is_litellms(content: str, sent: dict) -> None:
    assert json.loads(model_settings.gemini_function_response(content)) == sent


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("openrouter/openrouter/google/gemini-3-flash-preview", True),
        ("openrouter/google/gemini-3-pro-preview", True),
        ("google/gemini-3-flash-preview", False),
        ("openrouter/openai/gpt-5.2-20251211", False),
        ("openrouter/moonshotai/kimi-k2.5", False),
    ],
)
def test_gemini_openrouter_route(model: str, expected: bool) -> None:
    paper = model_settings.paper_model(model)
    assert model_settings.gemini_openrouter_route(model, paper) is expected
