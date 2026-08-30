"""OpenAI-compatible cache-read usage normalization (#3019)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter


@pytest.mark.asyncio
async def test_nonstreaming_response_maps_cached_prompt_tokens() -> None:
    usage = SimpleNamespace(
        prompt_tokens=120,
        completion_tokens=5,
        total_tokens=125,
        prompt_tokens_details=SimpleNamespace(
            cached_tokens=80,
            cache_write_tokens=10,
        ),
    )
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="ok",
                        tool_calls=None,
                        reasoning_content=None,
                    )
                )
            ],
            usage=usage,
        )
    )

    response = await OpenAIAdapter().get_response(
        client=client,
        model="gpt-cache",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert response.input_tokens == 30
    assert response.total_tokens == 35
    assert response.cache_creation_input_tokens == 10
    assert response.cache_read_input_tokens == 80


def test_raw_chat_completion_maps_dict_cache_details() -> None:
    response = OpenAIAdapter()._llm_response_from_chat_completion(
        {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 5,
                "total_tokens": 125,
                "prompt_tokens_details": {
                    "cached_tokens": 80,
                    "cache_write_tokens": 10,
                },
            },
        }
    )

    assert response.input_tokens == 30
    assert response.total_tokens == 35
    assert response.cache_creation_input_tokens == 10
    assert response.cache_read_input_tokens == 80


def test_raw_chat_completion_accepts_compatible_top_level_cache_fields() -> None:
    response = OpenAIAdapter()._llm_response_from_chat_completion(
        {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 5,
                "total_tokens": 125,
                "cached_tokens": 80,
                "cache_write_tokens": 10,
            },
        }
    )

    assert response.input_tokens == 30
    assert response.total_tokens == 35
    assert response.cache_creation_input_tokens == 10
    assert response.cache_read_input_tokens == 80


def test_missing_cache_details_remains_unreported() -> None:
    response = OpenAIAdapter()._llm_response_from_chat_completion(
        {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 2,
                "total_tokens": 14,
            },
        }
    )

    assert response.cache_read_input_tokens is None
    assert response.cache_creation_input_tokens is None
