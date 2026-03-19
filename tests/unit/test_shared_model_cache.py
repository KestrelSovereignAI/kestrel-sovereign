"""Tests for SharedModelCache — process-wide model discovery cache."""
import time
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
