"""Regression (#2129): GoogleAdapter.get_response must not crash on google-genai
response shapes with no usable parts (safety-block / MAX_TOKENS) or on text parts
whose always-present ``function_call`` attribute is None."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kestrel_sovereign.llm.google_adapter import GoogleAdapter


def _client_returning(response):
    client = SimpleNamespace()
    client.aio = SimpleNamespace()
    client.aio.models = SimpleNamespace(
        generate_content=AsyncMock(return_value=response)
    )
    return client


async def _get(response):
    adapter = GoogleAdapter()
    client = _client_returning(response)
    return await adapter.get_response(
        client=client, model="gemini-2.0-flash",
        messages=[{"role": "user", "parts": [{"text": "hi"}]}],
    )


@pytest.mark.asyncio
async def test_safety_blocked_candidate_content_none():
    """A blocked prompt yields candidate.content is None — must return an empty
    LLMResponse, not raise AttributeError."""
    resp = SimpleNamespace(candidates=[SimpleNamespace(content=None)])
    result = await _get(resp)
    assert result.content is None
    assert result.tool_calls is None


@pytest.mark.asyncio
async def test_max_tokens_empty_parts_none():
    """A MAX_TOKENS-truncated candidate can have content.parts is None — must not
    raise TypeError on `for part in None`."""
    resp = SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=None))]
    )
    result = await _get(resp)
    assert result.content is None


@pytest.mark.asyncio
async def test_no_candidates():
    result = await _get(SimpleNamespace(candidates=[]))
    assert result.content is None


@pytest.mark.asyncio
async def test_cached_usage_is_normalized_on_direct_google_route():
    result = await _get(SimpleNamespace(
        candidates=[],
        usage_metadata=SimpleNamespace(
            prompt_token_count=11,
            candidates_token_count=4,
            total_token_count=15,
            cached_content_token_count=7,
        ),
    ))

    assert result.input_tokens == 4
    assert result.output_tokens == 4
    assert result.total_tokens == 8
    assert result.cache_read_input_tokens == 7


@pytest.mark.asyncio
async def test_text_part_with_none_function_call():
    """google-genai Part ALWAYS has a `function_call` attribute (None for text),
    so the old `hasattr(part, 'function_call')` entered the tool branch and
    crashed on None. A text part must be read as text."""
    text_part = SimpleNamespace(text="hello there", function_call=None)
    resp = SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[text_part]))]
    )
    result = await _get(resp)
    assert result.content == "hello there"
    assert result.tool_calls is None


@pytest.mark.asyncio
async def test_real_function_call_part_parsed():
    fc = SimpleNamespace(name="do_thing", args={"x": 1})
    fc_part = SimpleNamespace(text=None, function_call=fc)
    resp = SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[fc_part]))]
    )
    result = await _get(resp)
    assert result.tool_calls is not None
    assert result.tool_calls[0].name == "do_thing"
    assert result.tool_calls[0].arguments == {"x": 1}


@pytest.mark.asyncio
async def test_mixed_text_and_function_call_parts():
    text_part = SimpleNamespace(text="thinking", function_call=None)
    fc = SimpleNamespace(name="tool_a", args={})
    fc_part = SimpleNamespace(text=None, function_call=fc)
    resp = SimpleNamespace(
        candidates=[SimpleNamespace(
            content=SimpleNamespace(parts=[text_part, fc_part])
        )]
    )
    result = await _get(resp)
    assert result.content == "thinking"
    assert [tc.name for tc in result.tool_calls] == ["tool_a"]
