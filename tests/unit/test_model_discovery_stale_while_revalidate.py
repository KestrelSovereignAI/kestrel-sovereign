"""Latency contract for stale-while-revalidate model catalog reads."""

import pytest

from kestrel_sovereign.llm.model_discovery import ModelDiscoveryMixin
from kestrel_sovereign.llm.model_metadata import ModelCategory, ModelInfo


class _Cache:
    def __init__(self, models):
        self.models = models
        self.refresh_factory = None

    def get(self):
        return None

    def get_any(self):
        return self.models

    def refresh_in_background(self, factory):
        self.refresh_factory = factory
        return True


class _Discovery(ModelDiscoveryMixin):
    def __init__(self):
        self.resolved = None

    def _resolve_auto_providers(self, models):
        self.resolved = models

    def _filter_models(self, models, **_filters):
        return list(models)

    async def reconcile_embedding_capabilities(self, *, use_cache):
        self.reconciled = use_cache


@pytest.mark.asyncio
async def test_stale_catalog_returns_without_awaiting_provider_refresh(monkeypatch):
    models = [
        ModelInfo(
            id="gpt-test",
            provider="openai",
            display_name="GPT Test",
            category=ModelCategory.CHAT,
        )
    ]
    cache = _Cache(models)
    monkeypatch.setattr(
        "kestrel_sovereign.llm.model_discovery.get_shared_model_cache",
        lambda: cache,
    )
    discovery = _Discovery()

    result = await discovery.discover_all_models(stale_while_revalidate=True)

    assert result == models
    assert discovery.resolved is models
    assert cache.refresh_factory is not None


@pytest.mark.asyncio
async def test_fresh_caller_joins_inflight_background_refresh(monkeypatch):
    models = [
        ModelInfo(
            id="gpt-refreshed",
            provider="openai",
            display_name="GPT Refreshed",
            category=ModelCategory.CHAT,
        )
    ]

    class _JoiningCache:
        def __init__(self):
            self.joined = False

        def get(self):
            return models if self.joined else None

        async def wait_for_refresh(self):
            self.joined = True
            return True

        def get_any(self):
            return None

    cache = _JoiningCache()
    monkeypatch.setattr(
        "kestrel_sovereign.llm.model_discovery.get_shared_model_cache",
        lambda: cache,
    )
    discovery = _Discovery()

    result = await discovery.discover_all_models()

    assert result == models
    assert cache.joined is True
    assert discovery.reconciled is True
