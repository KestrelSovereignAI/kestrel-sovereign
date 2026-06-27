"""v5 capability parity for the Ollama adapter (#1987).

Ollama is a local runtime with no platform/data-plane APIs (no batch, files,
prompt cache, or pre-flight token-count endpoint), and its `think` flag is a
model-dependent boolean that doesn't map to the neutral reasoning fields. The
honest v5 addition is raw passthrough (reach ollama client methods) plus an
options escape hatch for ollama-native request fields.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from kestrel_sdk.llm import ProviderCapabilities, RequestOptions

from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter


def test_ollama_v5_capabilities_and_round_trip():
    caps = OllamaAdapter().provider_capabilities()

    assert caps.supports_raw_passthrough is True
    assert "chat" in caps.raw_operations and "pull" in caps.raw_operations
    # Honest about what a local runtime does NOT offer.
    assert caps.supports_batch is False
    assert caps.supports_files is False
    assert caps.supports_token_counting is False
    assert caps.supports_prompt_cache is False
    assert caps.supports_reasoning_control is False
    assert ProviderCapabilities.from_mapping(caps.to_dict()) == caps


def test_ollama_contract_features():
    assert OllamaAdapter().contract_features() == frozenset({"raw_passthrough"})


def test_apply_request_options_merges_ollama_native_raw():
    adapter = OllamaAdapter()
    out = adapter.apply_request_options(
        {"model": "qwen3"},
        RequestOptions(raw={"keep_alive": "30m", "options": {"num_ctx": 8192}}),
        model="qwen3",
    )
    assert out["keep_alive"] == "30m"
    assert out["options"] == {"num_ctx": 8192}


def test_apply_request_options_default_no_op():
    adapter = OllamaAdapter()
    out = adapter.apply_request_options({"model": "x"}, RequestOptions(), model="x")
    assert out == {"model": "x"}


@pytest.mark.asyncio
async def test_raw_request_routes_named_operation():
    adapter = OllamaAdapter()
    client = MagicMock()
    client.show = AsyncMock(return_value={"details": {"family": "qwen"}})

    result = await adapter.raw_request(client, "show", model="qwen3")

    assert result.operation == "show"
    assert result.data == {"details": {"family": "qwen"}}
    client.show.assert_awaited_once_with(model="qwen3")


@pytest.mark.asyncio
async def test_raw_request_requires_operation():
    adapter = OllamaAdapter()
    with pytest.raises(ValueError):
        await adapter.raw_request(MagicMock(), "")
