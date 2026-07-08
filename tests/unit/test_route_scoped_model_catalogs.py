"""Route-scoped model catalogs (#2262).

A vendor's routes expose different serveable model sets — ``openai:plan``
(codex) serves the codex set, ``openai:api`` the full platform catalog. These
tests pin:

* per-route model lists differ and don't cross-contaminate (a plan route's list
  excludes api-only models when the route advertises its own set),
* ``ModelInfo.underlying_provider`` is populated for OpenRouter ids and
  round-trips through ``to_dict``/``from_dict``.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.llm.model_cache import get_shared_model_cache
from kestrel_sovereign.llm.model_discovery import ModelDiscoveryMixin
from kestrel_sovereign.llm.model_metadata import ModelCategory, ModelInfo


@pytest.fixture(autouse=True)
def _clear_shared_cache():
    get_shared_model_cache().clear()
    yield
    get_shared_model_cache().clear()


def _chat(model_id: str, provider: str = "openai") -> ModelInfo:
    return ModelInfo(
        id=model_id,
        provider=provider,
        display_name=model_id,
        category=ModelCategory.CHAT,
    )


class _Svc(ModelDiscoveryMixin):
    """Minimal harness exposing only what ``discover_models_for_route`` needs."""

    def __init__(self, vendor_models, route_catalogs):
        self.providers = []
        self._vendor_models = list(vendor_models)
        self._route_catalogs = dict(route_catalogs)

    async def discover_all_models(self, *, use_cache=True, featured_only=False,
                                  category=None, providers=None):
        # Stand in for the real discovery: return the vendor set filtered the
        # same way the real method would, and leave ``_route_catalogs`` as
        # pre-seeded by the test.
        return self._filter_models(
            list(self._vendor_models),
            featured_only=featured_only,
            category=category,
            providers=providers,
        )


@pytest.mark.asyncio
async def test_plan_route_list_excludes_api_only_models():
    """A route-scoped (plan) list must come from THAT route's catalog and never
    inherit api-only models from the vendor's broader discovery (#2262)."""
    vendor_models = [_chat("gpt-5.5"), _chat("gpt-5.5-pro")]  # full api catalog
    route_catalogs = {"openai:plan": [_chat("gpt-5.5")]}      # codex serves less
    svc = _Svc(vendor_models, route_catalogs)

    plan = await svc.discover_models_for_route("openai", "plan")
    api = await svc.discover_models_for_route("openai", "api")

    plan_ids = {m.id for m in plan}
    api_ids = {m.id for m in api}

    assert plan_ids == {"gpt-5.5"}
    assert "gpt-5.5-pro" in api_ids
    # The two routes of the same vendor do NOT return the same set.
    assert plan_ids != api_ids
    # No api-only model leaked into the plan list.
    assert "gpt-5.5-pro" not in plan_ids


@pytest.mark.asyncio
async def test_route_without_own_catalog_inherits_vendor_set():
    """A route with no route-specific catalog falls back to the vendor set."""
    vendor_models = [_chat("gpt-5.5"), _chat("gpt-5.5-pro")]
    svc = _Svc(vendor_models, route_catalogs={})

    api = await svc.discover_models_for_route("openai", "api")

    assert {m.id for m in api} == {"gpt-5.5", "gpt-5.5-pro"}


@pytest.mark.asyncio
async def test_empty_route_catalog_does_not_leak_vendor_models():
    """An empty route-specific catalog means the route advertises no explicit
    set — it must NOT fall through to the vendor's api-only membership."""
    vendor_models = [_chat("gpt-5.5"), _chat("gpt-5.5-pro")]
    svc = _Svc(vendor_models, route_catalogs={"openai:plan": []})

    plan = await svc.discover_models_for_route("openai", "plan")

    assert plan == []


class TestUnderlyingProvider:
    def test_openrouter_populates_underlying_provider(self):
        m = ModelInfo(
            id="anthropic/claude-3-opus",
            provider="openrouter",
            display_name="Claude 3 Opus",
            underlying_provider="anthropic",
        )
        assert m.underlying_provider == "anthropic"

    def test_underlying_provider_round_trips(self):
        m = ModelInfo(
            id="anthropic/claude-3-opus",
            provider="openrouter",
            display_name="Claude 3 Opus",
            underlying_provider="anthropic",
        )
        data = m.to_dict()
        assert data["underlying_provider"] == "anthropic"
        restored = ModelInfo.from_dict(data)
        assert restored.underlying_provider == "anthropic"

    def test_underlying_provider_defaults_none(self):
        m = ModelInfo(id="gpt-5.5", provider="openai", display_name="GPT-5.5")
        assert m.underlying_provider is None
        assert ModelInfo.from_dict(m.to_dict()).underlying_provider is None


@pytest.mark.asyncio
async def test_openrouter_adapter_sets_underlying_provider():
    """OpenRouterAdapter.list_models must carry the id-prefix substrate onto
    each ModelInfo instead of dropping it (#2262)."""
    from kestrel_sovereign.llm.openrouter_adapter import OpenRouterAdapter

    adapter = OpenRouterAdapter()
    adapter.api_key = "test-key"

    payload = {
        "data": [
            {"id": "anthropic/claude-3-opus", "name": "Claude 3 Opus"},
            {"id": "openai/gpt-5.5", "name": "GPT-5.5"},
            {"id": "bare-model", "name": "Bare"},
        ]
    }
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=payload)

    http_client = AsyncMock()
    http_client.get = AsyncMock(return_value=resp)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=http_client)
    cm.__aexit__ = AsyncMock(return_value=False)

    import kestrel_sovereign.llm.openrouter_adapter as mod
    orig = mod.httpx.AsyncClient
    mod.httpx.AsyncClient = MagicMock(return_value=cm)
    try:
        models = await adapter.list_models()
    finally:
        mod.httpx.AsyncClient = orig

    by_id = {m.id: m for m in models}
    assert by_id["anthropic/claude-3-opus"].underlying_provider == "anthropic"
    assert by_id["openai/gpt-5.5"].underlying_provider == "openai"
    # No slash → no meta-provider prefix to carry.
    assert by_id["bare-model"].underlying_provider is None
