"""Per-route embedding_model set/clear at runtime (#2337).

Covers the embeddings-popover per-route model picker's backend contract:
  - set/clear round-trip of a route's embedding_model override;
  - capability re-advertisement after a set (``supports_embeddings`` flips on);
  - the probe-on-save rejection path for a dead/misspelled cloud slug;
  - restore-on-clear of the route's pre-override capability state;
  - persistence of the override map (mirrors the embedding_route knob store).
"""
import asyncio
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
