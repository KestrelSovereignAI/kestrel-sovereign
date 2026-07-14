"""Regression for #2436: OpenRouter CHAT discovery must reach the bootstrap route.

Booting management-key-only (``OPENROUTER_MANAGEMENT_API_KEY`` set,
``OPENROUTER_API_KEY`` unset), ``finalize_providers()`` mints a bootstrap child
key and points the adapter's ``self.api_key`` at it. The EMBEDDING path
(``list_embedding_models`` → ``self.api_key``) discovered fine, but CHAT
discovery silently produced zero OpenRouter models.

Root cause: ``OpenRouterAdapter`` subclasses ``OpenAIAdapter`` and the route
declares ``base_url = https://openrouter.ai/api/v1``, so
``_discover_for_vendor_route`` dispatched it to the generic
``_discover_openai_compatible_remote`` path — which re-resolves the key from
``os.environ["OPENROUTER_API_KEY"]`` (unset in management-key-only mode) and
returned ``[]`` with only a debug log. ``model="auto"`` then never resolved.

Fix: exclude ``OpenRouterAdapter`` from the generic OpenAI-compatible
``base_url`` branches so chat discovery uses its authoritative ``list_models``
(which authenticates with the bootstrap-aware ``self.api_key``), exactly like
the embedding path.
"""

from __future__ import annotations

import pytest

from kestrel_sovereign.llm.model_cache import get_shared_model_cache
from kestrel_sovereign.llm.model_discovery import ModelDiscoveryMixin
from kestrel_sovereign.llm.model_metadata import ModelCategory, ModelInfo
from kestrel_sovereign.llm.model_selection import resolve_provider_default
from kestrel_sovereign.llm.openrouter_adapter import OpenRouterAdapter
from kestrel_sovereign.llm.embedding_discovery import EmbeddingModelInfo


class _FakeOpenRouterAdapter(OpenRouterAdapter):
    """A real ``OpenRouterAdapter`` subtype (so ``isinstance`` dispatch holds)
    whose network facets are canned. Mimics the bootstrap route: ``self.api_key``
    is the minted child key, NOT ``os.environ["OPENROUTER_API_KEY"]``."""

    def __init__(self, chat_models, embedding_models):
        # Bypass env/network __init__ — set only what discovery reads.
        self.api_key = "bootstrap-child-key"
        self.base_url = "https://openrouter.ai/api/v1"
        self.site_url = "https://kestrel.ai"
        self.app_name = "Kestrel"
        self._embedding_model = None
        self._embedding_dim = None
        self._supports_embeddings = False
        self._chat_models = chat_models
        self._embedding_models = embedding_models

    async def list_models(self, client=None):
        return list(self._chat_models)

    async def list_embedding_models(self, client=None):
        return list(self._embedding_models)


class _Svc(ModelDiscoveryMixin):
    """Minimal discovery host exercising the REAL dispatch/selection path."""

    def __init__(self, providers):
        self.providers = providers
        self._route_catalogs = None
        self._embedding_discovery_cache = None
        self._corpus_embedding_profile_provider = None


def _openrouter_route(adapter):
    # Mirrors the real bootstrap route dict: OpenRouterAdapter + declared
    # base_url + model="auto" (the config in kestrel.toml.example).
    return {
        "name": "openrouter:api",
        "vendor": "openrouter",
        "model": "auto",
        "adapter": adapter,
        "client": object(),  # ignored by the adapter's own list_models
        "base_url": "https://openrouter.ai/api/v1",
        "is_local": False,
        "is_cloud": True,
        "selection_hints": [],
        "capabilities": {},
    }


@pytest.fixture(autouse=True)
def _clear_shared_cache():
    get_shared_model_cache().clear()
    yield
    get_shared_model_cache().clear()


@pytest.mark.asyncio
async def test_chat_discovery_reaches_bootstrap_openrouter_route(monkeypatch):
    """With only a bootstrap key on the adapter, chat discovery must still
    produce OpenRouter CHAT models and resolve ``model="auto"`` (#2436)."""
    # Ensure the management-key-only condition: no static OPENROUTER_API_KEY in
    # the environment. The old code fell back to it and returned [].
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    chat_models = [
        ModelInfo(
            id="anthropic/claude-sonnet-4",
            provider="openrouter",
            display_name="Claude Sonnet 4",
            category=ModelCategory.CHAT,
            is_featured=True,
        ),
        ModelInfo(
            id="openai/gpt-5.4-mini",
            provider="openrouter",
            display_name="GPT 5.4 mini",
            category=ModelCategory.CHAT,
            is_featured=True,
        ),
    ]
    embedding_models = [
        EmbeddingModelInfo(
            id="qwen/qwen3-embedding-0.6b",
            provider="openrouter",
            display_name="Qwen3 Embedding 0.6B",
            native_dim=768,
        )
    ]
    adapter = _FakeOpenRouterAdapter(chat_models, embedding_models)
    route = _openrouter_route(adapter)
    svc = _Svc([route])

    models = await svc.discover_all_models(use_cache=False)

    by_vendor: dict[str, list[ModelInfo]] = {}
    for m in models:
        by_vendor.setdefault(m.provider, []).append(m)

    # CHAT discovery reached the bootstrap route.
    assert "openrouter" in by_vendor, (
        "openrouter CHAT discovery did not fire for the bootstrap route — "
        "dispatch still routed it through the env-key-only remote path (#2436)"
    )
    or_chat_ids = {
        m.id for m in by_vendor["openrouter"] if m.category == ModelCategory.CHAT
    }
    assert "anthropic/claude-sonnet-4" in or_chat_ids
    assert "openai/gpt-5.4-mini" in or_chat_ids

    # ``model="auto"`` resolved to a concrete model.
    assert route["model"] != "auto"
    assert route["model"] in or_chat_ids

    # resolve_provider_default no longer raises 'auto unresolved'.
    llm_config = {
        "vendors": {
            "openrouter": {"routes": {"api": {"model": "auto", "selection_hints": []}}}
        }
    }
    resolved = resolve_provider_default(
        "openrouter:api", llm_config=llm_config, cached_models=models
    )
    assert resolved in or_chat_ids


@pytest.mark.asyncio
async def test_embedding_path_still_reaches_bootstrap_route(monkeypatch):
    """Guard #2338: the embedding facet must keep discovering the bootstrap
    OpenRouter route regardless of the chat-dispatch fix."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    embedding_models = [
        EmbeddingModelInfo(
            id="qwen/qwen3-embedding-0.6b",
            provider="openrouter",
            display_name="Qwen3 Embedding 0.6B",
            native_dim=768,
        )
    ]
    adapter = _FakeOpenRouterAdapter([], embedding_models)
    route = _openrouter_route(adapter)
    svc = _Svc([route])

    discovered = await svc.discover_embedding_models(use_cache=False)

    assert any(
        m.provider == "openrouter" and m.route == "openrouter:api"
        for m in discovered
    ), "embedding discovery regressed for the bootstrap OpenRouter route (#2338)"
