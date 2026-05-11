from types import SimpleNamespace

import pytest

from kestrel_sovereign.llm.adapter import ThinkingDelta
from kestrel_sovereign.llm.adapter import should_split_plain_reasoning
from kestrel_sovereign.llm.openai_adapter import (
    OpenAIAdapter,
    _ThinkingContentSplitter,
    _split_thinking_from_content,
)


def test_non_streaming_split_extracts_think_tags():
    thinking, content = _split_thinking_from_content(
        "<think>I should keep this private.</think>\n\nVisible answer."
    )

    assert thinking == "I should keep this private."
    assert content == "Visible answer."


def test_non_streaming_split_extracts_kimi_plain_reasoning():
    thinking, content = _split_thinking_from_content(
        "The user is asking for a direct answer.\n\nFinal answer."
    )

    assert thinking == "The user is asking for a direct answer."
    assert content == "Final answer."


def test_non_streaming_split_extracts_kimi_constraints_and_self_check():
    thinking, content = _split_thinking_from_content(
        "Constraints:\n- Answer with only the number.\n\n"
        "Wait, I need to check if there are constitutional issues.\n\n"
        "4"
    )

    assert "Constraints:" in thinking
    assert "Wait, I need" in thinking
    assert content == "4"


def test_non_streaming_split_preserves_visible_content_with_reasoning_field():
    thinking, content = _split_thinking_from_content(
        "Constraints:\n- Keep this visible.\n\nThe user-facing answer.",
        reasoning_content="Separate provider reasoning.",
    )

    assert thinking == "Separate provider reasoning."
    assert content == "Constraints:\n- Keep this visible.\n\nThe user-facing answer."


def test_plain_reasoning_model_gate_excludes_non_reasoning_chat_models():
    assert should_split_plain_reasoning("qwen3:4b")
    assert should_split_plain_reasoning("deepseek-r1")
    assert should_split_plain_reasoning("kimi-k2.6")
    assert not should_split_plain_reasoning("qwen2.5:0.5b")
    assert not should_split_plain_reasoning("deepseek-chat")


def test_streaming_splitter_emits_thinking_delta_for_plain_kimi_reasoning():
    splitter = _ThinkingContentSplitter(
        provider="openai",
        split_plain_reasoning=True,
    )

    events = splitter.feed("The user is asking for a direct answer.\n\n")
    events += splitter.feed("Final")
    events += splitter.flush()

    assert isinstance(events[0], ThinkingDelta)
    assert events[0].content == "The user is asking for a direct answer."
    assert "".join(e for e in events if isinstance(e, str)) == "Final"


def test_streaming_splitter_extracts_kimi_constraints_and_self_check():
    splitter = _ThinkingContentSplitter(
        provider="openai",
        split_plain_reasoning=True,
    )

    events = splitter.feed("Constraints:\n- Answer with only the number.\n\n")
    events += splitter.feed("Wait, I need to check constitutional issues.\n\n")
    events += splitter.feed("4")
    events += splitter.flush()

    thinking = [event.content for event in events if isinstance(event, ThinkingDelta)]
    assert thinking == [
        "Constraints:\n- Answer with only the number.",
        "Wait, I need to check constitutional issues.",
    ]
    assert "".join(e for e in events if isinstance(e, str)) == "4"


def test_streaming_splitter_buffers_split_closing_think_tag():
    splitter = _ThinkingContentSplitter(provider="openai")

    events = splitter.feed("<think>private</thi")
    events += splitter.feed("nk>answer")
    events += splitter.flush()

    thinking = [event.content for event in events if isinstance(event, ThinkingDelta)]
    assert thinking == ["private"]
    assert "".join(e for e in events if isinstance(e, str)) == "answer"


@pytest.mark.asyncio
async def test_streaming_response_yields_reasoning_content_as_thinking_delta():
    adapter = OpenAIAdapter()

    async def stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        reasoning_content="hidden reasoning",
                        content=None,
                    )
                )
            ]
        )
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="Visible answer.")
                )
            ]
        )

    async def create(**kwargs):
        return stream()

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    items = []
    async for item in adapter.get_streaming_response(
        client=client,
        model="Kimi-K2.6",
        messages=[{"role": "user", "content": "hi"}],
    ):
        items.append(item)

    assert isinstance(items[0], ThinkingDelta)
    assert items[0].content == "hidden reasoning"
    assert "".join(item for item in items if isinstance(item, str)) == "Visible answer."


@pytest.mark.asyncio
async def test_openai_compatible_adapter_uses_configured_provider_name():
    adapter = OpenAIAdapter(name="llama_cpp")

    async def stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        reasoning_content="local hidden reasoning",
                        content=None,
                    )
                )
            ]
        )

    async def create(**kwargs):
        return stream()

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    items = []
    async for item in adapter.get_streaming_response(
        client=client,
        model="Kimi-K2.6",
        messages=[{"role": "user", "content": "hi"}],
    ):
        items.append(item)

    assert isinstance(items[0], ThinkingDelta)
    assert items[0].provider == "llama_cpp"
