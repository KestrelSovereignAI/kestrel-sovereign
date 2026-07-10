"""Per-route embedding_model set/clear at runtime (#2337).

Covers the embeddings-popover per-route model picker's backend contract:
  - set/clear round-trip of a route's embedding_model override;
  - capability re-advertisement after a set (``supports_embeddings`` flips on);
  - the probe-on-save rejection path for a dead/misspelled cloud slug;
  - restore-on-clear of the route's pre-override capability state;
  - persistence of the override map (mirrors the embedding_route knob store).
"""
import asyncio
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest

from kestrel_sovereign.llm.service import LLMService
import kestrel_sovereign.llm.embedding_service as embedding_service_mod


def _route(
    name: str,
    vendor: str,
    *,
    is_local: bool = False,
    capabilities: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "name": name,
        "vendor": vendor,
        "route": name.split(":", 1)[1] if ":" in name else "api",
        "adapter": object(),
        "client": object(),
        "model": "auto",
        "is_local": is_local,
        "is_cloud": not is_local,
        "capabilities": dict(capabilities or {}),
    }


def _service(providers) -> LLMService:
    service = LLMService.__new__(LLMService)
    service.providers = providers
    service._route_embedding_model_overrides = {}
    service._route_embedding_caps_backup = {}
    service._route_embedding_model_persistence_callback = None
    service._embedding_discovery_cache = ["stale"]
    return service


# --- set / clear round-trip --------------------------------------------------

def test_set_then_clear_round_trip():
    provider = _route("openrouter:api", "openrouter", capabilities={})
    service = _service([provider])

    service.set_route_embedding_model(
        "openrouter:api", "qwen/qwen3-embedding-8b", 768
    )
    assert service.get_route_embedding_model_overrides() == {
        "openrouter:api": {"model": "qwen/qwen3-embedding-8b", "dim": 768}
    }

    service.set_route_embedding_model("openrouter:api", None)
    assert service.get_route_embedding_model_overrides() == {}


def test_set_invalidates_discovery_cache():
    provider = _route("openrouter:api", "openrouter", capabilities={})
    service = _service([provider])
    service.set_route_embedding_model("openrouter:api", "some/model", 512)
    assert service._embedding_discovery_cache is None


# --- capability re-advertisement after a set ---------------------------------

def test_set_re_advertises_embedding_capability():
    # No config pin, no prior capability — the route does not advertise.
    provider = _route("openrouter:api", "openrouter", capabilities={})
    service = _service([provider])
    assert service._provider_supports_embeddings(provider) is False

    service.set_route_embedding_model(
        "openrouter:api", "qwen/qwen3-embedding-8b", 768
    )

    caps = provider["capabilities"]
    assert caps["supports_embeddings"] is True
    assert caps["embedding_model"] == "qwen/qwen3-embedding-8b"
    assert caps["embedding_dim"] == 768
    assert service._provider_supports_embeddings(provider) is True


def test_clear_restores_pre_override_capabilities():
    # A route that advertised a config-pinned model before the runtime override.
    provider = _route(
        "openrouter:api",
        "openrouter",
        capabilities={
            "supports_embeddings": True,
            "embedding_model": "config/pinned-model",
            "embedding_dim": 1024,
        },
    )
    service = _service([provider])

    service.set_route_embedding_model("openrouter:api", "runtime/model", 768)
    assert provider["capabilities"]["embedding_model"] == "runtime/model"
    assert provider["capabilities"]["embedding_dim"] == 768

    # Clearing restores the config pin exactly, not a stale runtime value.
    service.set_route_embedding_model("openrouter:api", None)
    caps = provider["capabilities"]
    assert caps["embedding_model"] == "config/pinned-model"
    assert caps["embedding_dim"] == 1024
    assert caps["supports_embeddings"] is True


def test_clear_removes_capability_added_by_override():
    # No pre-override capability — clearing removes what the override added.
    provider = _route("openrouter:api", "openrouter", capabilities={})
    service = _service([provider])
    service.set_route_embedding_model("openrouter:api", "runtime/model", 768)
    service.set_route_embedding_model("openrouter:api", None)
    caps = provider["capabilities"]
    assert "embedding_model" not in caps
    assert "embedding_dim" not in caps
    assert "supports_embeddings" not in caps


def test_unknown_route_is_rejected():
    service = _service([_route("openrouter:api", "openrouter")])
    with pytest.raises(ValueError, match="no configured route matches"):
        service.set_route_embedding_model("nope:missing", "some/model")


# --- probe-on-save rejection path (#2337 / #2326) ----------------------------

async def test_probe_on_save_rejects_dead_cloud_slug(monkeypatch):
    provider = _route("openrouter:api", "openrouter", capabilities={})
    service = _service([provider])

    async def _dead_embed(self, text):
        raise RuntimeError("404 no provider serving qwen/does-not-exist")

    monkeypatch.setattr(
        embedding_service_mod.ProviderEmbeddingService, "aembed", _dead_embed
    )

    with pytest.raises(ValueError, match="live"):
        await service.aset_route_embedding_model(
            "openrouter:api", "qwen/does-not-exist", 768
        )

    # The pin must NOT have been committed on a failed probe.
    assert service.get_route_embedding_model_overrides() == {}
    assert "embedding_model" not in provider["capabilities"]


async def test_probe_on_save_rejects_empty_vector(monkeypatch):
    provider = _route("openrouter:api", "openrouter", capabilities={})
    service = _service([provider])

    async def _empty_embed(self, text):
        return None

    monkeypatch.setattr(
        embedding_service_mod.ProviderEmbeddingService, "aembed", _empty_embed
    )

    with pytest.raises(ValueError, match="no embedding"):
        await service.aset_route_embedding_model("openrouter:api", "dead/model")
    assert service.get_route_embedding_model_overrides() == {}


async def test_probe_on_save_accepts_live_cloud_slug(monkeypatch):
    provider = _route("openrouter:api", "openrouter", capabilities={})
    service = _service([provider])

    async def _live_embed(self, text):
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(
        embedding_service_mod.ProviderEmbeddingService, "aembed", _live_embed
    )

    await service.aset_route_embedding_model(
        "openrouter:api", "qwen/qwen3-embedding-8b", 768
    )
    assert service.get_route_embedding_model_overrides() == {
        "openrouter:api": {"model": "qwen/qwen3-embedding-8b", "dim": 768}
    }
    assert provider["capabilities"]["supports_embeddings"] is True


async def test_local_route_skips_probe(monkeypatch):
    # A local route never probes — a missing local model is a separate setup
    # problem, not the empty-upstream-pool failure the cloud probe guards.
    provider = _route("ollama:local", "ollama", is_local=True, capabilities={})
    service = _service([provider])

    def _boom(*a, **k):
        raise AssertionError("local route must not build a probe service")

    monkeypatch.setattr(embedding_service_mod, "ProviderEmbeddingService", _boom)

    await service.aset_route_embedding_model(
        "ollama:local", "qwen3-embedding-0.6b", 768
    )
    assert service.get_route_embedding_model_overrides() == {
        "ollama:local": {"model": "qwen3-embedding-0.6b", "dim": 768}
    }


async def test_clear_skips_probe(monkeypatch):
    provider = _route("openrouter:api", "openrouter", capabilities={})
    service = _service([provider])
    service.set_route_embedding_model("openrouter:api", "runtime/model", 768)

    def _boom(*a, **k):
        raise AssertionError("clearing must not probe")

    monkeypatch.setattr(embedding_service_mod, "ProviderEmbeddingService", _boom)
    await service.aset_route_embedding_model("openrouter:api", None)
    assert service.get_route_embedding_model_overrides() == {}


# --- #2372 echo does not cross routes ----------------------------------------


async def test_route_model_echo_returns_pinned_routes_own_slug():
    """The per-route echo must reflect the PINNED route's own model, not whatever
    the globally-resolved embedding provider (embedding_route/chat) picks (#2372).

    Global resolution here lands on the Ollama route (its own slug
    ``qwen3-embedding:8b``); the echo for the just-pinned ``openrouter:api`` must
    still come back with that route's own ``qwen/qwen3-embedding-8b``.
    """
    ollama = _route(
        "ollama:local",
        "ollama",
        is_local=True,
        capabilities={
            "supports_embeddings": True,
            "embedding_model": "qwen3-embedding:8b",
            "embedding_dim": 768,
        },
    )
    openrouter = _route(
        "openrouter:api",
        "openrouter",
        capabilities={
            "supports_embeddings": True,
            "embedding_model": "qwen/qwen3-embedding-8b",
            "embedding_dim": 768,
        },
    )
    # An adapter that discovers the pinned openrouter model so the resolver folds
    # the capability pin in as ``is_pinned`` and returns it.
    from unittest.mock import AsyncMock
    from kestrel_sovereign.llm.embedding_discovery import EmbeddingModelInfo

    openrouter["adapter"] = SimpleNamespace(
        list_embedding_models=AsyncMock(return_value=[
            EmbeddingModelInfo(id="qwen/qwen3-embedding-8b", provider="openrouter",
                               native_dim=768),
        ])
    )
    ollama["adapter"] = SimpleNamespace(
        list_embedding_models=AsyncMock(return_value=[
            EmbeddingModelInfo(id="qwen3-embedding:8b", provider="ollama",
                               native_dim=768),
        ])
    )

    service = _service([ollama, openrouter])
    service._embedding_route = "ollama:local"
    service._embedding_space_pins = None
    service._verified_space_pins = {}
    service._embedding_space_change_warnings = {}
    service._embedding_discovery_cache = None
    # Global embedding resolution crosses to the Ollama route.
    service.resolve_embedding_provider = lambda: ollama

    # Sanity: the base (global) settings echo Ollama's slug — the cross-route bug.
    base = service.get_embedding_settings()
    assert base["embedding_model"] == "qwen3-embedding:8b"

    # The route-scoped echo returns the pinned route's OWN slug.
    echo = await service.aget_embedding_settings_for_route("openrouter:api")
    assert echo["resolved_route"] == "openrouter:api"
    assert echo["embedding_model"] == "qwen/qwen3-embedding-8b"
    assert echo["embedding_dim"] == 768


async def test_route_model_echo_does_not_carry_global_route_fields():
    """P2 (#2372): the per-route echo must NOT inherit the global route's
    ``shared_space`` / ``space_change_warning``. Those are route-dependent; a
    response resolving ``openrouter:api`` may not describe ``ollama:local``.
    """
    ollama = _route(
        "ollama:local",
        "ollama",
        is_local=True,
        capabilities={
            "supports_embeddings": True,
            "embedding_model": "qwen3-embedding:8b",
            "embedding_dim": 768,
        },
    )
    openrouter = _route(
        "openrouter:api",
        "openrouter",
        capabilities={
            "supports_embeddings": True,
            "embedding_model": "qwen/qwen3-embedding-8b",
            "embedding_dim": 768,
        },
    )
    from unittest.mock import AsyncMock
    from kestrel_sovereign.llm.embedding_discovery import EmbeddingModelInfo

    for route_dict, slug in ((openrouter, "qwen/qwen3-embedding-8b"),
                             (ollama, "qwen3-embedding:8b")):
        route_dict["adapter"] = SimpleNamespace(
            list_embedding_models=AsyncMock(return_value=[
                EmbeddingModelInfo(id=slug, provider=route_dict["vendor"],
                                   native_dim=768),
            ])
        )

    service = _service([ollama, openrouter])
    service._embedding_route = "ollama:local"
    service._embedding_space_pins = None
    service._verified_space_pins = {}
    # The GLOBAL (ollama) route carries a space-change warning; the openrouter
    # echo must report None for its own (unaffected) route.
    service._embedding_space_change_warnings = {
        "ollama:local": {"route": "ollama:local", "chosen_model": "x"}
    }
    service._embedding_discovery_cache = None
    service.resolve_embedding_provider = lambda: ollama

    # The base (global) echo carries the ollama warning — the cross-route leak.
    base = service.get_embedding_settings()
    assert base["space_change_warning"] is not None

    echo = await service.aget_embedding_settings_for_route("openrouter:api")
    assert echo["resolved_route"] == "openrouter:api"
    # No global-route fields leaked into this route's echo.
    assert echo["space_change_warning"] is None
    assert echo["shared_space"] is None


def test_get_embedding_service_for_route_is_route_scoped():
    """P2 (#2372): a per-route embedding service is built for the NAMED route,
    not the globally-active one, so stale-row counts don't cross routes."""
    import kestrel_sovereign.llm.embedding_service as es_mod

    ollama = _route(
        "ollama:local", "ollama", is_local=True,
        capabilities={"supports_embeddings": True,
                      "embedding_model": "qwen3-embedding:8b", "embedding_dim": 768},
    )
    openrouter = _route(
        "openrouter:api", "openrouter",
        capabilities={"supports_embeddings": True,
                      "embedding_model": "qwen/qwen3-embedding-8b", "embedding_dim": 768},
    )
    service = _service([ollama, openrouter])
    service._embedding_space_pins = None
    service._verified_space_pins = {}
    service.resolve_embedding_provider = lambda: ollama

    built = {}

    class _FakeES:
        def __init__(self, provider, **kw):
            built["provider"] = provider

    service_es = getattr(es_mod, "ProviderEmbeddingService")
    es_mod.ProviderEmbeddingService = _FakeES
    try:
        svc = service.get_embedding_service_for_route("openrouter:api")
    finally:
        es_mod.ProviderEmbeddingService = service_es

    assert svc is not None
    assert built["provider"]["name"] == "openrouter:api"
    # Unknown route → None (no fabricated service).
    assert service.get_embedding_service_for_route("nope:missing") is None


# --- persistence (same store as the embedding_route knob) --------------------

async def test_override_map_is_persisted():
    provider = _route("openrouter:api", "openrouter", capabilities={})
    service = _service([provider])

    persisted = {}

    async def _persist(overrides):
        persisted["value"] = overrides

    service.set_route_embedding_model_persistence_callback(_persist)
    # Need a running loop for the scheduled persistence task.
    service._preference_persistence_tasks = set()
    service._handle_preference_persistence_done = (
        lambda task: service._preference_persistence_tasks.discard(task)
    )

    service.set_route_embedding_model("openrouter:api", "qwen/qwen3-embedding-8b", 768)
    # Let the scheduled persistence task run.
    await asyncio.sleep(0)
    assert persisted["value"] == {
        "openrouter:api": {"model": "qwen/qwen3-embedding-8b", "dim": 768}
    }
