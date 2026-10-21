"""Latency contract for stale-while-revalidate model catalog reads."""

import asyncio

import pytest

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.llm.model_cache import SharedModelCache
from kestrel_sovereign.llm.model_discovery import ModelDiscoveryMixin
from kestrel_sovereign.llm.model_metadata import ModelCategory, ModelInfo


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
    cache = SharedModelCache()
    cache.set_stale(models)
    monkeypatch.setattr(
        "kestrel_sovereign.llm.model_discovery.get_shared_model_cache",
        lambda: cache,
    )
    discovery = _Discovery()

    owner = None
    try:
        result = await discovery.discover_all_models(stale_while_revalidate=True)
        owner = cache._refresh_task
        assert result == models
        assert discovery.resolved == models
        assert owner is not None and not owner.done()
    finally:
        if owner is not None:
            owner.cancel()
            await await_owned_task(owner)


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

    cache = SharedModelCache()
    release, joined = asyncio.Event(), asyncio.Event()

    async def refresh():
        await release.wait()
        cache.set(models)

    assert cache.refresh_in_background(refresh)
    owner = cache._refresh_task
    actual_wait = cache.wait_for_refresh

    async def observe_original_join():
        joined.set()
        return await actual_wait()

    monkeypatch.setattr(cache, "wait_for_refresh", observe_original_join)
    monkeypatch.setattr(
        "kestrel_sovereign.llm.model_discovery.get_shared_model_cache",
        lambda: cache,
    )
    discovery = _Discovery()

    reader = asyncio.create_task(discovery.discover_all_models())
    try:
        async with asyncio.timeout(5):
            await joined.wait()
        assert not reader.done() and not owner.done()
        release.set()
        result = await reader
        assert result == models
        assert discovery.reconciled is True
    finally:
        release.set()
        for task in (reader, owner):
            if not task.done():
                task.cancel()
            await await_owned_task(task)
