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
        ("google/gemini-3.1-pro-preview", "gemini/gemini-3.1-pro-preview"),
        ("openai-api/vertex/glm-5-maas", "vertex_ai/zai-org/glm-5-maas"),
        # OpenRouter's names (Hawk adds a second "openrouter/").
        ("openrouter/openrouter/anthropic/claude-haiku-4.5", "vertex_ai/claude-haiku-4-5@20251001"),
        ("openrouter/anthropic/claude-opus-4.5", "claude-opus-4-5-20251101"),
        ("openrouter/deepseek/deepseek-chat-v3.1", "deepseek-ai/DeepSeek-V3.1"),
        ("openrouter/moonshotai/kimi-k2.5", "moonshotai/Kimi-K2.5"),
        ("openrouter/z-ai/glm-4.7", "zai-org/GLM-4.7"),
        ("openrouter/z-ai/glm-5", "vertex_ai/zai-org/glm-5-maas"),
        ("openrouter/google/gemini-3.1-pro-preview", "gemini/gemini-3.1-pro-preview"),
    ],
)
def test_paper_models_are_recognized(name: str, upstream_id: str) -> None:
    model = paper_model(name)
    assert model is not None
    assert model.upstream_id == upstream_id


@pytest.mark.parametrize(
    "name", ["openai/gpt-4.1", "google/gemini-2.5-pro", "openrouter/z-ai/glm-4.6", "mockllm/model"]
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


def _reply(reasoning: str | None, text: str, calls: bool) -> ChatMessageAssistant:
    content: list = [ContentReasoning(reasoning=reasoning)] if reasoning else []
    content.append(ContentText(text=text))
    tool_calls = [ToolCall(id="c1", function="list_files", arguments={})] if calls else None
    return ChatMessageAssistant(content=content, tool_calls=tool_calls)


@pytest.mark.parametrize(
    ("reasoning", "text", "calls", "expected"),
    [
        (
            "Look first.",
            "Let me look.",
            True,
            "<model_thinking>Look first.</model_thinking>\n\nLet me look.",
        ),
        # A reply of tool calls only had content None, so upstream recorded "None".
        ("Look first.", "", True, "<model_thinking>Look first.</model_thinking>\n\nNone"),
        (None, "Done.", False, None),
    ],
)
def test_glm5_reasoning_goes_in_the_text(
    reasoning: str | None, text: str, calls: bool, expected: str | None
) -> None:
    message = _reply(reasoning, text, calls)
    rewritten = model_settings.reasoning_as_text(message)
    if expected is None:
        assert rewritten is message
    else:
        assert rewritten.content == expected
        assert rewritten.tool_calls == message.tool_calls
    assert PAPER_MODELS["glm-5"].reasoning_in_text
    assert not any(m.reasoning_in_text for k, m in PAPER_MODELS.items() if k != "glm-5")


def test_glm5_keeps_blank_replies_as_plain_text() -> None:
    blank = ChatMessageAssistant(content=[ContentText(text="")])
    sent = PAPER_MODELS["glm-5"].request_messages([ChatMessageUser(content="hi"), blank])
    assert len(sent) == 2
    assert sent[1].content == ""


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("openrouter/anthropic/claude-opus-4.5", True),
        ("openrouter/openrouter/z-ai/glm-5", True),
        ("anthropic/claude-opus-4-5", False),
        ("openrouter/openai/gpt-4.1-mini", False),
    ],
)
def test_paper_openrouter_route(model: str, expected: bool) -> None:
    assert model_settings.paper_openrouter_route(model, paper_model(model)) is expected
