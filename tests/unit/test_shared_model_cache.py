"""Tests for SharedModelCache — process-wide model discovery cache."""
import time
import asyncio
from unittest.mock import MagicMock

import pytest

from kestrel_sovereign.llm.model_cache import SharedModelCache, get_shared_model_cache
from kestrel_sovereign.llm.model_metadata import ModelInfo, ModelCategory


def _make_model(model_id: str, provider: str = "openai") -> ModelInfo:
    return ModelInfo(
        id=model_id,
        provider=provider,
        display_name=model_id,
        category=ModelCategory.CHAT,
    )


class TestSharedModelCache:
    """Unit tests for SharedModelCache."""

    def test_empty_cache_returns_none(self):
        cache = SharedModelCache()
        assert cache.get() is None
        assert cache.get_any() is None
        assert not cache.has_data()

    def test_set_and_get(self):
        cache = SharedModelCache()
        models = [_make_model("gpt-4o"), _make_model("claude-sonnet")]
        cache.set(models)

        assert cache.has_data()
        assert cache.get() is not None
        assert len(cache.get()) == 2

    def test_get_returns_none_when_stale(self):
        cache = SharedModelCache(cache_ttl=1)
        models = [_make_model("gpt-4o")]
        cache.set(models)

        # Manually expire
        cache._timestamp = time.time() - 2

        assert cache.get() is None  # Stale
        assert cache.get_any() is not None  # Still available via get_any

    def test_set_stale_populates_with_zero_timestamp(self):
        cache = SharedModelCache()
        models = [_make_model("llama3")]
        cache.set_stale(models)

        assert cache.has_data()
        assert cache.get() is None  # timestamp=0 → expired
        assert cache.get_any() is not None  # But available for fallback

    def test_set_stale_does_not_overwrite_fresh(self):
        cache = SharedModelCache()
        fresh = [_make_model("gpt-4o")]
        stale = [_make_model("old-model")]

        cache.set(fresh)
        cache.set_stale(stale)  # Should be ignored

        assert len(cache.get_any()) == 1
        assert cache.get_any()[0].id == "gpt-4o"

    def test_clear(self):
        cache = SharedModelCache()
        cache.set([_make_model("gpt-4o")])
        cache.clear()

        assert not cache.has_data()
        assert cache.get() is None

    def test_fresh_set_overwrites_stale(self):
        cache = SharedModelCache()
        cache.set_stale([_make_model("old")])
        cache.set([_make_model("new")])

        result = cache.get()
        assert result is not None
        assert len(result) == 1
        assert result[0].id == "new"

    @pytest.mark.asyncio
    async def test_background_refresh_is_coalesced(self):
        cache = SharedModelCache()
        release = asyncio.Event()
        calls = 0

        async def refresh():
            nonlocal calls
            calls += 1
            await release.wait()

        assert cache.refresh_in_background(refresh) is True
        task = cache._refresh_task
        assert cache.refresh_in_background(refresh) is False
        await asyncio.sleep(0.06)
        assert calls == 1

        release.set()
        await task
        await asyncio.sleep(0)
        assert cache.refresh_in_background(refresh) is True
        await cache._refresh_task

    @pytest.mark.asyncio
    async def test_foreground_caller_joins_background_refresh(self):
        cache = SharedModelCache()
        cache.set_stale([_make_model("old")])
        release = asyncio.Event()

        async def refresh():
            await release.wait()
            cache.set([_make_model("new")])

        assert cache.refresh_in_background(refresh) is True
        waiter = asyncio.create_task(cache.wait_for_refresh())
        await asyncio.sleep(0.06)
        assert not waiter.done()

        release.set()
        assert await waiter is True
        assert cache.get()[0].id == "new"

    @pytest.mark.asyncio
    async def test_clear_cancels_live_loop_refresh_without_resurrecting_catalog(self):
        cache = SharedModelCache()
        started = asyncio.Event()

        async def refresh():
            started.set()
            await asyncio.Event().wait()
            cache.set([_make_model("must-not-return")])

        assert cache.refresh_in_background(refresh) is True
        task = cache._refresh_task
        await started.wait()

        cache.clear()

        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
        assert cache.get_any() is None
        assert cache._refresh_task is None

    @pytest.mark.asyncio
    async def test_waiter_treats_internal_refresh_cancellation_as_cache_miss(self):
        cache = SharedModelCache()
        started = asyncio.Event()

        async def refresh():
            started.set()
            await asyncio.Event().wait()

        assert cache.refresh_in_background(refresh) is True
        waiter = asyncio.create_task(cache.wait_for_refresh())
        await started.wait()

        cache.clear()

        assert await waiter is False
        assert not waiter.cancelled()

    @pytest.mark.asyncio
    async def test_closed_loop_refresh_does_not_wedge_replacement_loop(self):
        cache = SharedModelCache()
        stale_task = MagicMock()
        stale_task.done.return_value = False
        stale_loop = MagicMock()
        stale_loop.is_closed.return_value = True
        cache._refresh_task = stale_task
        cache._refresh_loop = stale_loop

        refreshed = asyncio.Event()

        async def refresh():
            refreshed.set()

        assert cache.refresh_in_background(refresh) is True
        replacement_task = cache._refresh_task
        assert replacement_task is not stale_task
        await replacement_task
        assert refreshed.is_set()

    def test_clear_tolerates_refresh_owned_by_closed_loop(self):
        cache = SharedModelCache()
        stale_task = MagicMock()
        stale_task.done.return_value = False
        stale_loop = MagicMock()
        stale_loop.is_closed.return_value = True
        cache._refresh_task = stale_task
        cache._refresh_loop = stale_loop

        cache.clear()

        assert cache._refresh_task is None
        assert cache._refresh_loop is None
        stale_loop.call_soon_threadsafe.assert_not_called()


class TestSharedModelCacheSingleton:
    """Tests for the module-level singleton."""

    def test_singleton_returns_same_instance(self):
        # Reset singleton for isolated test
        import kestrel_sovereign.llm.model_cache as mc
        mc._shared_cache = None

        cache1 = get_shared_model_cache()
        cache2 = get_shared_model_cache()
        assert cache1 is cache2

        # Cleanup
        mc._shared_cache = None

    def test_multiple_llm_services_share_cache(self):
        """Simulates the core use case: multiple LLMService instances share discovery data."""
        import kestrel_sovereign.llm.model_cache as mc
        mc._shared_cache = None

        cache = get_shared_model_cache()

        # Simulate first LLMService discovering models
        models = [_make_model("gpt-4o", "openai"), _make_model("claude-sonnet", "anthropic")]
        cache.set(models)

        # Simulate second LLMService reading from cache
        cache2 = get_shared_model_cache()
        result = cache2.get()
        assert result is not None
        assert len(result) == 2

        # Cleanup
        mc._shared_cache = None
