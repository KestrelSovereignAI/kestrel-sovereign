from types import SimpleNamespace

import pytest

from kestrel_sovereign.llm.adapter import ThinkingDelta
from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter
from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter
import kestrel_sovereign.llm.ollama_adapter as ollama_module


class _AsyncStream:
    def __init__(self, events):
        self._events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for event in self._events:
            yield event


def _anthropic_event(delta_type: str, **attrs):
    return SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(type=delta_type, **attrs),
    )


@pytest.mark.asyncio
async def test_anthropic_streaming_emits_thinking_delta():
    adapter = AnthropicAdapter()
    stream = _AsyncStream([
        _anthropic_event("thinking_delta", thinking="I should answer briefly."),
        _anthropic_event("text_delta", text="4"),
    ])
    client = SimpleNamespace(
        messages=SimpleNamespace(stream=lambda **kwargs: stream)
    )

    items = []
    async for item in adapter.get_streaming_response(
        client=client,
        model="claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "2+2?"}],
    ):
        items.append(item)

    assert isinstance(items[0], ThinkingDelta)
    assert items[0].provider == "anthropic"
    assert items[0].content == "I should answer briefly."
    assert "".join(item for item in items if isinstance(item, str)) == "4"


@pytest.mark.asyncio
async def test_anthropic_streaming_splits_think_tags():
    adapter = AnthropicAdapter()
    stream = _AsyncStream([
        _anthropic_event("text_delta", text="<think>private</think>"),
        _anthropic_event("text_delta", text="ANTHROPIC_OK"),
    ])
    client = SimpleNamespace(
        messages=SimpleNamespace(stream=lambda **kwargs: stream)
    )

    items = []
    async for item in adapter.get_streaming_response(
        client=client,
        model="claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "test"}],
    ):
        items.append(item)

    assert isinstance(items[0], ThinkingDelta)
    assert items[0].provider == "anthropic"
    assert items[0].content == "private"
    assert "".join(item for item in items if isinstance(item, str)) == "ANTHROPIC_OK"


@pytest.mark.asyncio
async def test_ollama_streaming_splits_think_tags(monkeypatch):
    monkeypatch.setattr(ollama_module, "OLLAMA_AVAILABLE", True)
    adapter = OllamaAdapter()

    async def stream():
        yield {"message": {"content": "<think>private plan</think>"}}
        yield {"message": {"content": "4"}}

    async def chat(**kwargs):
        return stream()

    client = SimpleNamespace(chat=chat)

    items = []
    async for item in adapter.get_streaming_response(
        client=client,
        model="qwen3",
        messages=[{"role": "user", "content": "2+2?"}],
    ):
        items.append(item)

    assert isinstance(items[0], ThinkingDelta)
    assert items[0].provider == "ollama"
    assert items[0].content == "private plan"
    assert "".join(item for item in items if isinstance(item, str)) == "4"


@pytest.mark.asyncio
async def test_ollama_with_tools_splits_non_streaming_content(monkeypatch):
    monkeypatch.setattr(ollama_module, "OLLAMA_AVAILABLE", True)
    adapter = OllamaAdapter()

    async def chat(**kwargs):
        return {
            "message": {
                "content": (
                    "The user is asking for simple arithmetic.\n\n"
                    "4"
                )
            }
        }

    client = SimpleNamespace(chat=chat)

    items = []
    async for item in adapter.get_streaming_response_with_tools(
        client=client,
        model="qwen3",
        messages=[{"role": "user", "content": "2+2?"}],
        tools=[{"type": "function", "function": {"name": "noop"}}],
    ):
        items.append(item)

    assert isinstance(items[0], ThinkingDelta)
    assert items[0].provider == "ollama"
    assert items[0].content == "The user is asking for simple arithmetic."
    assert "".join(item for item in items if isinstance(item, str)) == "4"


@pytest.mark.asyncio
async def test_ollama_non_reasoning_model_preserves_plain_intro(monkeypatch):
    monkeypatch.setattr(ollama_module, "OLLAMA_AVAILABLE", True)
    adapter = OllamaAdapter()

    async def chat(**kwargs):
        return {
            "message": {
                "content": (
                    "Based on the logs, the service restarted cleanly.\n\n"
                    "The fix is to refresh the page."
                )
            }
        }

    client = SimpleNamespace(chat=chat)

    response = await adapter.get_response(
        client=client,
        model="llama3.2",
        messages=[{"role": "user", "content": "what happened?"}],
    )

    assert response.content == (
        "Based on the logs, the service restarted cleanly.\n\n"
        "The fix is to refresh the page."
    )


@pytest.mark.asyncio
async def test_ollama_tool_support_trusts_capabilities_for_small_models():
    adapter = OllamaAdapter()

    async def show(model_name):
        assert model_name == "qwen2.5:0.5b"
        return {
            "template": "{{ if .Tools }}tools{{ end }}",
            "capabilities": ["completion", "tools"],
            "model_info": {"general.parameter_count": 494_030_000},
        }

    client = SimpleNamespace(show=show)

    assert await adapter._check_tool_support(client, "qwen2.5:0.5b") is True
