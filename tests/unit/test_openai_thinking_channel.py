from types import SimpleNamespace

import pytest

from kestrel_sovereign.llm.adapter import ThinkingDelta
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


def test_non_streaming_split_preserves_untagged_prose_as_visible():
    thinking, content = _split_thinking_from_content(
        "The user is asking for a direct answer.\n\nFinal answer."
    )

    assert thinking is None
    assert content == "The user is asking for a direct answer.\n\nFinal answer."


def test_non_streaming_split_preserves_visible_content_with_reasoning_field():
    thinking, content = _split_thinking_from_content(
        "I'll start by checking the file.\n\nThe answer is 4.",
        reasoning_content="Separate provider reasoning.",
    )

    assert thinking == "Separate provider reasoning."
    assert content == "I'll start by checking the file.\n\nThe answer is 4."


def test_streaming_splitter_preserves_untagged_prose_as_visible():
    splitter = _ThinkingContentSplitter(provider="openai")

    events = splitter.feed("I'll start by doing X, ok?\n\n")
    events += splitter.feed("Final answer.")
    events += splitter.flush()

    assert not any(isinstance(e, ThinkingDelta) for e in events)
    assert "".join(e for e in events if isinstance(e, str)) == (
        "I'll start by doing X, ok?\n\nFinal answer."
    )


def test_streaming_splitter_preserves_prose_after_closing_think_tag():
    splitter = _ThinkingContentSplitter(provider="openai")

    events = splitter.feed("<think>private plan</think>\n\n")
    events += splitter.feed("So they're testing X. Let me look around.\n\n")
    events += splitter.feed("Hey — here's the answer.")
    events += splitter.flush()

    thinking = [e.content for e in events if isinstance(e, ThinkingDelta)]
    assert thinking == ["private plan"]
    visible = "".join(e for e in events if isinstance(e, str))
    assert visible == (
        "\n\nSo they're testing X. Let me look around.\n\n"
        "Hey — here's the answer."
    )


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
