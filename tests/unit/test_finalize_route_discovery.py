"""Regression tests for #2247: model discovery must include finalize-registered
routes.

When OpenRouter registers only via the async ``finalize_providers()`` bootstrap
mint (management-key-only), any model-discovery snapshot taken *before* finalize
omits the new route:

* ``discover_all_models(use_cache=True)`` returns the pre-finalize snapshot on a
  cache hit — no ``/models`` query for the new vendor fires, ``by_vendor`` omits
  it, and its ``model="auto"`` route stays unresolved.

Two fixes are pinned here:

1. ``LLMService.finalize_providers`` clears the shared model cache when it
   registers a new route, so the next discovery re-runs over the complete
   provider list.
2. ``discover_all_models`` resolves this instance's ``model="auto"`` routes
   against the cached models even on a cache hit, so a route registered after
   the cache was populated still resolves when the cached snapshot already
   carries that vendor's models (multi-agent shared cache).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.llm.model_cache import get_shared_model_cache
from kestrel_sovereign.llm.model_metadata import ModelCategory, ModelInfo


@pytest.fixture(autouse=True)
def _clear_shared_cache():
    get_shared_model_cache().clear()
    yield
    get_shared_model_cache().clear()


@pytest.mark.asyncio
async def test_finalize_clears_stale_cache_when_new_route_registers():
    """A pre-finalize discovery snapshot must be dropped so the late-registered
    OpenRouter route gets discovered on the next pass."""
    from kestrel_sovereign.llm.service import LLMService

    svc = LLMService.__new__(LLMService)
    # Snapshot taken BEFORE finalize: no openrouter route.
    svc.providers = [{"name": "openai:api", "vendor": "openai", "model": "gpt-5.4-mini"}]

    # Populate the shared cache as a prior (openrouter-less) discovery would.
    get_shared_model_cache().set([
        ModelInfo(
            id="gpt-5.4-mini", provider="openai", display_name="GPT 5.4 mini",
            category=ModelCategory.CHAT,
        )
    ])
    assert get_shared_model_cache().has_data()

    # finalize registers the OpenRouter bootstrap route.
    registry = MagicMock()
    registry.finalize_providers = AsyncMock(return_value=["<provider-infos>"])
    svc.provider_registry = registry

    def _convert(_infos):
        return [
            {"name": "openai:api", "vendor": "openai", "model": "gpt-5.4-mini"},
            {"name": "openrouter:api", "vendor": "openrouter", "model": "auto"},
        ]

    svc._convert_providers_format = MagicMock(side_effect=_convert)

    await svc.finalize_providers()

    # The stale snapshot is gone → next discover_all_models will re-run.
    assert not get_shared_model_cache().has_data()
    assert any(p["vendor"] == "openrouter" for p in svc.providers)


@pytest.mark.asyncio
async def test_finalize_keeps_cache_when_no_new_route():
    """No new route → the cache is a valid snapshot and must NOT be cleared."""
    from kestrel_sovereign.llm.service import LLMService

    svc = LLMService.__new__(LLMService)
    svc.providers = [{"name": "openai:api", "vendor": "openai", "model": "gpt-5.4-mini"}]
    get_shared_model_cache().set([
        ModelInfo(id="gpt-5.4-mini", provider="openai", display_name="x",
                  category=ModelCategory.CHAT)
    ])

    registry = MagicMock()
    registry.finalize_providers = AsyncMock(return_value=["<infos>"])
    svc.provider_registry = registry
    svc._convert_providers_format = MagicMock(return_value=[
        {"name": "openai:api", "vendor": "openai", "model": "gpt-5.4-mini"},
    ])

    await svc.finalize_providers()

    assert get_shared_model_cache().has_data()


@pytest.mark.asyncio
async def test_cache_hit_resolves_late_registered_auto_route():
    """On a cache hit, discover_all_models must still resolve THIS instance's
    ``model="auto"`` routes against the cached models — otherwise a route
    registered after another instance warmed the cache stays unresolved even
    when the cached snapshot already carries its vendor's models (#2247)."""
    from kestrel_sovereign.llm.service import LLMService

    svc = LLMService.__new__(LLMService)
    openrouter_route = {
        "name": "openrouter:api", "vendor": "openrouter", "model": "auto",
        "selection_hints": [],
    }
    svc.providers = [openrouter_route]
    svc._route_catalogs = {}
    svc._ensure_route_catalogs_sync = MagicMock()

    # Another instance already discovered openrouter models into the shared
    # (fresh) cache.
    get_shared_model_cache().set([
        ModelInfo(
            id="anthropic/claude-opus-4", provider="openrouter",
            display_name="Claude Opus 4", category=ModelCategory.CHAT,
            is_featured=True,
        )
    ])

    models = await svc.discover_all_models(use_cache=True)

    # Cache hit returned the cached model...
    assert any(m.id == "anthropic/claude-opus-4" for m in models)
    # ...AND this instance's auto route resolved from it.
    assert openrouter_route["model"] == "anthropic/claude-opus-4"
