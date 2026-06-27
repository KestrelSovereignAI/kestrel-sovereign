"""v5 capability parity for the Anthropic adapters (#1984).

AnthropicAdapter (API key) advertises real token counting + prompt caching +
reasoning + raw passthrough. ClaudeMaxAdapter (OAuth/plan) inherits the
request-level features but must NOT advertise the API-key-only data-plane ones
(token counting, raw passthrough) — the consumer plan can't reach them.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from kestrel_sdk.llm import (
    PromptCacheMode,
    ProviderCapabilities,
    RequestOptions,
    TokenCountMode,
)

from kestrel_sovereign.llm.anthropic_adapter import AnthropicAdapter
from kestrel_sovereign.llm.claude_max_adapter import ClaudeMaxAdapter


def test_anthropic_v5_capabilities_and_round_trip():
    caps = AnthropicAdapter().provider_capabilities()

    assert caps.supports_token_counting is True
    assert caps.supports_prompt_cache is True
    assert caps.supports_raw_passthrough is True
    # Reasoning control is intentionally not advertised (model-version-specific).
    assert caps.supports_reasoning_control is False
    assert caps.token_count_mode == TokenCountMode.PROVIDER_NATIVE
    assert caps.prompt_cache_mode == PromptCacheMode.EXPLICIT_BREAKPOINTS
    assert caps.max_cache_breakpoints == 4
    assert "messages.count_tokens" in caps.raw_operations
    assert ProviderCapabilities.from_mapping(caps.to_dict()) == caps


def test_anthropic_contract_features():
    features = AnthropicAdapter().contract_features()
    assert {
        "token_counting",
        "prompt_cache",
        "raw_passthrough",
    }.issubset(features)
    assert "reasoning_control" not in features


def test_claude_max_plan_gates_off_platform_apis():
    caps = ClaudeMaxAdapter().provider_capabilities()

    # Request-level prompt caching still advertised on the plan route.
    assert caps.supports_prompt_cache is True
    # API-key-only data-plane features are NOT advertised.
    assert caps.supports_token_counting is False
    assert caps.supports_raw_passthrough is False
    assert caps.token_count_mode == TokenCountMode.NONE
    assert caps.raw_operations == ()

    features = ClaudeMaxAdapter().contract_features()
    assert "prompt_cache" in features
    assert "token_counting" not in features
    assert "raw_passthrough" not in features


@pytest.mark.asyncio
async def test_count_tokens_uses_real_endpoint():
    adapter = AnthropicAdapter()
    client = MagicMock()
    client.messages.count_tokens = AsyncMock(return_value=MagicMock(input_tokens=42))

    result = await adapter.count_tokens(
        client, "claude-opus-4-8", [{"role": "user", "content": "hello"}]
    )

    assert result is not None
    assert result.input_tokens == 42
    client.messages.count_tokens.assert_awaited_once()
    sent = client.messages.count_tokens.await_args.kwargs
    assert sent["model"] == "claude-opus-4-8"
    assert sent["messages"]


def test_apply_request_options_default_no_op():
    adapter = AnthropicAdapter()
    kwargs = {"max_tokens": 1024}
    out = adapter.apply_request_options(kwargs, RequestOptions(), model="claude-opus-4-8")
    assert out == {"max_tokens": 1024}


def test_apply_request_options_merges_raw_escape_hatch():
    adapter = AnthropicAdapter()
    out = adapter.apply_request_options(
        {"max_tokens": 1024},
        RequestOptions(raw={"metadata": {"user_id": "u1"}}),
        model="claude-opus-4-8",
    )
    assert out["metadata"] == {"user_id": "u1"}


def test_request_options_wiring_applies_raw_and_no_ops_without_options():
    # The request builders call _maybe_apply_request_options, so a caller's
    # request_options reach api_params; absence leaves them untouched.
    adapter = AnthropicAdapter()
    out = adapter._maybe_apply_request_options(
        {"model": "claude-opus-4-8"},
        {"request_options": RequestOptions(raw={"metadata": {"k": "v"}})},
        "claude-opus-4-8",
    )
    assert out["metadata"] == {"k": "v"}

    untouched = adapter._maybe_apply_request_options(
        {"model": "x"}, {}, "claude-opus-4-8"
    )
    assert untouched == {"model": "x"}


@pytest.mark.asyncio
async def test_raw_request_routes_named_operation_through_client():
    adapter = AnthropicAdapter()
    client = MagicMock()
    client.messages.count_tokens = AsyncMock(return_value={"input_tokens": 7})

    result = await adapter.raw_request(
        client, "messages.count_tokens", {"model": "claude-opus-4-8", "messages": []}
    )

    assert result.operation == "messages.count_tokens"
    assert result.data == {"input_tokens": 7}
    client.messages.count_tokens.assert_awaited_once_with(
        model="claude-opus-4-8", messages=[]
    )


@pytest.mark.asyncio
async def test_raw_request_supports_keyword_payload_form():
    # SDK-style: raw_request(client, "op", model=..., messages=...) — kwargs must
    # reach the target call, not be dropped.
    adapter = AnthropicAdapter()
    client = MagicMock()
    client.messages.count_tokens = AsyncMock(return_value={"input_tokens": 3})

    result = await adapter.raw_request(
        client, "messages.count_tokens", model="claude-opus-4-8", messages=[]
    )

    assert result.data == {"input_tokens": 3}
    client.messages.count_tokens.assert_awaited_once_with(
        model="claude-opus-4-8", messages=[]
    )
