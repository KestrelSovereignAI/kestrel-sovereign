"""OpenRouter embeddings support (#2288).

Covers:
- ``aembed`` / ``aembed_batch`` against the OpenAI-compatible ``/v1/embeddings``
  envelope, with no hardcoded default model for the meta-provider.
- Truthful, route-scoped embedding capability advertisement (only when an
  embedding model is configured).
- ``dimensions`` pass-through for Matryoshka-capable models.
- Model-keyed embedding-profile ``space_id`` (upstream model, not route vendor).
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kestrel_sovereign.llm.openrouter_adapter import OpenRouterAdapter
from kestrel_sovereign.llm.provider_registry import ProviderRegistry
from kestrel_sovereign.llm.embedding_service import ProviderEmbeddingService


def _embedding_client(vector=(0.1, 0.2, 0.3)):
    """A client whose ``embeddings.create`` returns the real API shape."""
    create = AsyncMock(
        return_value=SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=list(vector))]
        )
    )
    return SimpleNamespace(embeddings=SimpleNamespace(create=create)), create


# --- capability advertisement ------------------------------------------------

def test_no_embedding_model_configured_advertises_no_embeddings():
    """A route without an embedding model must NOT advertise embeddings —
    the meta-provider never inherits OpenAI's text-embedding-3-small default."""
    caps = OpenRouterAdapter().provider_capabilities()
    assert caps.supports_embeddings is False
    assert caps.embedding_model is None
    assert caps.embedding_dim is None


def test_configured_embedding_model_advertises_real_values():
    """With a model configured, capabilities surface the real model + dim so
    the settings API GET reflects what the route actually serves."""
    adapter = OpenRouterAdapter(
        embedding_model="qwen/qwen3-embedding-0.6b",
        embedding_dim=768,
    )
    caps = adapter.provider_capabilities()
    assert caps.supports_embeddings is True
    assert caps.embedding_model == "qwen/qwen3-embedding-0.6b"
    assert caps.embedding_dim == 768


def test_supports_embeddings_false_without_model_even_if_forced():
    """``supports_embeddings=True`` with no model can't fabricate a capability —
    there is nothing to serve."""
    adapter = OpenRouterAdapter(supports_embeddings=True)
    assert adapter.provider_capabilities().supports_embeddings is False


# --- aembed / dimensions pass-through ----------------------------------------

async def test_aembed_uses_configured_model_and_dimensions():
    adapter = OpenRouterAdapter(
        embedding_model="qwen/qwen3-embedding-0.6b",
        embedding_dim=768,
    )
    client, create = _embedding_client()

    result = await adapter.aembed(client, "hello")

    assert result == [0.1, 0.2, 0.3]
    create.assert_awaited_once_with(
        model="qwen/qwen3-embedding-0.6b",
        input="hello",
        dimensions=768,
    )


async def test_aembed_explicit_dimensions_override_config():
    adapter = OpenRouterAdapter(
        embedding_model="qwen/qwen3-embedding-0.6b",
        embedding_dim=768,
    )
    client, create = _embedding_client()

    await adapter.aembed(client, "hello", dimensions=256)

    _, kwargs = create.await_args
    assert kwargs["dimensions"] == 256


async def test_aembed_batch_forwards_model_and_dimensions():
    adapter = OpenRouterAdapter(
        embedding_model="qwen/qwen3-embedding-0.6b",
        embedding_dim=768,
    )
    create = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                SimpleNamespace(index=0, embedding=[1.0, 0.0]),
                SimpleNamespace(index=1, embedding=[0.0, 1.0]),
            ]
        )
    )
    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))

    result = await adapter.aembed_batch(client, ["a", "b"])

    assert result == [[1.0, 0.0], [0.0, 1.0]]
    create.assert_awaited_once_with(
        model="qwen/qwen3-embedding-0.6b",
        input=["a", "b"],
        dimensions=768,
    )


async def test_aembed_without_any_model_raises():
    """No configured model and no ``model`` arg → explicit error, never a
    silent fall-through to a hardcoded default."""
    adapter = OpenRouterAdapter()
    client, _ = _embedding_client()

    with pytest.raises(ValueError, match="require an explicit embedding model"):
        await adapter.aembed(client, "hello")


async def test_aembed_omits_dimensions_when_unset():
    """A route with a model but no configured dim must not send ``dimensions``
    (let the model return its native size)."""
    adapter = OpenRouterAdapter(embedding_model="openai/text-embedding-3-small")
    client, create = _embedding_client()

    await adapter.aembed(client, "hello")

    _, kwargs = create.await_args
    assert "dimensions" not in kwargs


# --- model-keyed space_id ----------------------------------------------------

def test_embedding_space_id_keys_on_upstream_model_and_dim():
    adapter = OpenRouterAdapter(
        embedding_model="qwen/qwen3-embedding-0.6b",
        embedding_dim=768,
    )
    assert adapter.embedding_space_id() == "qwen3-embedding-0.6b@768"


def test_embedding_space_id_none_without_config():
    assert OpenRouterAdapter().embedding_space_id() is None


def test_profile_registration_uses_model_keyed_space_id():
    """The embedding profile keys on the upstream model, not on the route
    vendor — two different upstream models through the same OpenRouter route
    are different spaces, and the same upstream model reached elsewhere shares
    the space."""
    adapter = OpenRouterAdapter(
        embedding_model="qwen/qwen3-embedding-0.6b",
        embedding_dim=768,
    )
    provider = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": adapter,
        "client": object(),
        "capabilities": adapter.provider_capabilities().to_dict(),
    }
    service = ProviderEmbeddingService(provider)
    profile = service.describe()

    assert profile is not None
    assert profile.space_id == "qwen3-embedding-0.6b@768"
    assert profile.model == "qwen/qwen3-embedding-0.6b"
    assert profile.dim == 768

    # Two different upstream models through the same route → different spaces.
    other = OpenRouterAdapter(
        embedding_model="openai/text-embedding-3-large",
        embedding_dim=768,
    )
    other_service = ProviderEmbeddingService(
        {
            "vendor": "openrouter",
            "name": "openrouter:api",
            "adapter": other,
            "client": object(),
            "capabilities": other.provider_capabilities().to_dict(),
        }
    )
    other_profile = other_service.describe()
    assert other_profile is not None
    assert other_profile.profile_id != profile.profile_id


# --- registry wiring ---------------------------------------------------------

def test_registry_passes_route_embedding_config_to_adapter(monkeypatch):
    registry = ProviderRegistry.__new__(ProviderRegistry)
    registry.config = {}
    registry._bootstrap_openrouter_key = None
    registry._deferred_openrouter_routes = []
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    _client, adapter = registry._build_client_and_adapter(
        vendor="openrouter",
        route="api",
        adapter_cls=OpenRouterAdapter,
        vendor_cfg={"is_cloud": True},
        route_cfg={
            "api_key_env": "OPENROUTER_API_KEY",
            "embedding_model": "qwen/qwen3-embedding-0.6b",
            "embedding_dim": "768",
        },
    )

    caps = adapter.provider_capabilities()
    assert caps.supports_embeddings is True
    assert caps.embedding_model == "qwen/qwen3-embedding-0.6b"
    assert caps.embedding_dim == 768


def test_registry_openrouter_route_without_embedding_model_stays_off(monkeypatch):
    registry = ProviderRegistry.__new__(ProviderRegistry)
    registry.config = {}
    registry._bootstrap_openrouter_key = None
    registry._deferred_openrouter_routes = []
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    _client, adapter = registry._build_client_and_adapter(
        vendor="openrouter",
        route="api",
        adapter_cls=OpenRouterAdapter,
        vendor_cfg={"is_cloud": True},
        route_cfg={"api_key_env": "OPENROUTER_API_KEY"},
    )

    assert adapter.provider_capabilities().supports_embeddings is False
