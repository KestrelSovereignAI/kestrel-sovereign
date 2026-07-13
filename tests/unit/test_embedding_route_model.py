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


class _StatusError(RuntimeError):
    """An exception carrying an HTTP-ish ``status_code``, like an SDK error."""

    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


async def test_probe_on_save_classifies_auth_failure(monkeypatch):
    """#2418 — a 401 ``User not found.`` (dead/revoked agent key) must be
    reported as an AUTH failure pointing at the agent's credential, NOT the
    404 "model may not be served" hint that misdirected the operator."""
    provider = _route("openrouter:api", "openrouter", capabilities={})
    service = _service([provider])

    async def _auth_fail(self, text):
        raise _StatusError("401 - {'error': {'message': 'User not found.'}}", 401)

    monkeypatch.setattr(
        embedding_service_mod.ProviderEmbeddingService, "aembed", _auth_fail
    )

    with pytest.raises(ValueError) as excinfo:
        await service.aset_route_embedding_model(
            "openrouter:api", "qwen/qwen3-embedding-8b", 768
        )
    msg = str(excinfo.value).lower()
    assert "credential" in msg
    assert "openrouter" in msg
    # Must NOT misclassify as model-not-served.
    assert "may not be currently served" not in msg
    assert service.get_route_embedding_model_overrides() == {}


async def test_probe_on_save_classifies_not_served(monkeypatch):
    """#2418 — a genuine 404 stays the model-not-served message."""
    provider = _route("openrouter:api", "openrouter", capabilities={})
    service = _service([provider])

    async def _not_served(self, text):
        raise RuntimeError("404 no provider serving qwen/does-not-exist")

    monkeypatch.setattr(
        embedding_service_mod.ProviderEmbeddingService, "aembed", _not_served
    )

    with pytest.raises(ValueError) as excinfo:
        await service.aset_route_embedding_model(
            "openrouter:api", "qwen/does-not-exist", 768
        )
    msg = str(excinfo.value).lower()
    assert "may not be currently served" in msg
    assert "credential is invalid" not in msg


async def test_probe_on_save_classifies_transient(monkeypatch):
    """#2418 — a timeout is transient, not a bad model or bad key."""
    provider = _route("openrouter:api", "openrouter", capabilities={})
    service = _service([provider])

    async def _timeout(self, text):
        raise TimeoutError("Request timed out after 30s")

    monkeypatch.setattr(
        embedding_service_mod.ProviderEmbeddingService, "aembed", _timeout
    )

    with pytest.raises(ValueError) as excinfo:
        await service.aset_route_embedding_model(
            "openrouter:api", "qwen/qwen3-embedding-8b", 768
        )
    msg = str(excinfo.value).lower()
    assert "transient" in msg
    assert "retry" in msg


def test_classify_embedding_probe_failure_buckets():
    """Direct unit coverage of the classifier's ordering (#2418): a 401
    ``User not found.`` contains the substring "not found" yet must bucket as
    auth, not not_served."""
    assert LLMService._classify_embedding_probe_failure(
        _StatusError("User not found.", 401)
    ) == "auth"
    assert LLMService._classify_embedding_probe_failure(
        RuntimeError("model_not_found: nope")
    ) == "not_served"
    assert LLMService._classify_embedding_probe_failure(
        RuntimeError("connection timed out")
    ) == "transient"
    assert LLMService._classify_embedding_probe_failure(
        RuntimeError("something weird")
    ) == "unknown"


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


async def test_aget_settings_cleared_pin_explicit_route_is_never_silent_off():
    """Round-4 #2372: with an explicit ``embedding_route`` set and that route's
    per-route pin CLEARED, the async settings read must resolve corpus/catalog —
    never surface ``embedding_model: null`` (embeddings silently off).

    The cleared pin drops the route's ``supports_embeddings`` flag, and
    ``resolve_embedding_provider`` gates the explicit-route branch on that flag,
    so a naive read returns None before ever resolving. ``aget_embedding_settings``
    re-advertises capability through the single resolver first, breaking the
    circular gate.
    """
    from unittest.mock import AsyncMock
    from kestrel_sovereign.llm.embedding_discovery import EmbeddingModelInfo

    openrouter = _route("openrouter:api", "openrouter", capabilities={})
    openrouter["adapter"] = SimpleNamespace(
        list_embedding_models=AsyncMock(return_value=[
            EmbeddingModelInfo(id="qwen/qwen3-embedding-8b", provider="openrouter",
                               native_dim=4096, dim_options=[768, 4096]),
        ])
    )
    service = _service([openrouter])
    service._embedding_route = "openrouter:api"
    service._embedding_space_pins = None
    service._verified_space_pins = {}
    service._embedding_space_change_warnings = {}
    service._embedding_discovery_cache = None
    service._disabled_routes = {}
    service.disabled = False
    service._force_local_only_provider = None
    service._corpus_embedding_profile_provider = None

    # Simulate the operator having pinned then CLEARED the route's model — this is
    # the exact state the round-4 dogfood hit (caps carry no supports_embeddings).
    service.set_route_embedding_model("openrouter:api", "qwen/qwen3-embedding-8b", 768)
    service.set_route_embedding_model("openrouter:api", None)
    assert "supports_embeddings" not in openrouter["capabilities"]

    # Without the round-4 fix, resolve_embedding_provider() returns None here and
    # the read is silent-off. With it, the read self-heals to the discovered model.
    settings = await service.aget_embedding_settings()
    assert settings["resolved_route"] == "openrouter:api"
    assert settings["embedding_model"] == "qwen/qwen3-embedding-8b"
    assert settings["embedding_dim"] == 768


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


# --- verified-shared-space coherence gate (#2440) ----------------------------

def _verified_space_service():
    """A service with a VERIFIED ``qwen3`` shared space over ollama+openrouter."""
    from kestrel_sovereign.llm.embedding_space import EmbeddingSpacePin, ParityResult

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
    standalone = _route(
        "openai:api", "openai",
        capabilities={"supports_embeddings": True,
                      "embedding_model": "text-embedding-3-small", "embedding_dim": 1536},
    )
    service = _service([ollama, openrouter, standalone])
    pin = EmbeddingSpacePin(
        name="qwen3",
        model="qwen/qwen3-embedding-8b",
        dim=768,
        members=("ollama:local", "openrouter:api"),
    )
    service._embedding_space_pins = [pin]
    service._verified_space_pins = {
        "qwen3": ParityResult(passed=True, threshold=0.98, min_cosine=0.9804,
                              mean_cosine=0.99, n=4),
    }
    return service, pin


def test_conflicting_pin_on_verified_space_member_is_refused():
    """#2440: pinning a DIFFERENT model on a verified-space member refuses —
    a phantom pin the space silently overrides must not be accepted."""
    from kestrel_sovereign.llm.service import EmbeddingSpaceConflictError

    service, _ = _verified_space_service()
    with pytest.raises(EmbeddingSpaceConflictError) as exc:
        service._check_route_pin_space_coherent("ollama:local", "nomic-embed-text", 768)
    msg = str(exc.value)
    assert "qwen3" in msg
    assert "nomic-embed-text" in msg
    assert "fragment" in msg


def test_conflicting_dim_on_verified_space_member_is_refused():
    """#2440: the space's OWN model at a different dim also fragments it."""
    from kestrel_sovereign.llm.service import EmbeddingSpaceConflictError

    service, _ = _verified_space_service()
    with pytest.raises(EmbeddingSpaceConflictError):
        service._check_route_pin_space_coherent(
            "openrouter:api", "qwen/qwen3-embedding-8b", 1024
        )


def test_pinning_the_spaces_own_model_is_allowed():
    """#2440: re-pinning the space's own model/dim is coherent — no refusal."""
    service, _ = _verified_space_service()
    # Same model, same dim → no conflict raised.
    service._check_route_pin_space_coherent(
        "ollama:local", "qwen/qwen3-embedding-8b", 768
    )
    # Same model, dim omitted → no conflict raised.
    service._check_route_pin_space_coherent(
        "openrouter:api", "qwen/qwen3-embedding-8b", None
    )


def test_route_native_alias_of_spaces_model_is_allowed():
    """#2440 P2: a member may re-pin the space's model under its OWN route-native
    slug. Ollama serves ``qwen3-embedding:8b`` for the same weights the pin names
    ``qwen/qwen3-embedding-8b``; normalized identity makes them equal, so this
    no-op re-pin / rollback must NOT falsely 409."""
    service, _ = _verified_space_service()
    # Ollama's tagged alias for the space's model — same weights, no fragmentation.
    service._check_route_pin_space_coherent(
        "ollama:local", "qwen3-embedding:8b", 768
    )
    # And with the dim omitted.
    service._check_route_pin_space_coherent(
        "ollama:local", "qwen3-embedding:8b", None
    )


def test_sync_setter_refuses_conflicting_pin_on_verified_space_member():
    """#2440 P1: the SYNCHRONOUS setter (the boot/settings/reindex hydration path)
    must also reject a pre-existing conflicting stored pin, not silently re-apply
    it after deploy. Refusal happens before capabilities/overrides mutate."""
    from kestrel_sovereign.llm.service import EmbeddingSpaceConflictError

    service, _ = _verified_space_service()
    ollama = next(p for p in service.providers if p["name"] == "ollama:local")
    with pytest.raises(EmbeddingSpaceConflictError):
        service.set_route_embedding_model("ollama:local", "nomic-embed-text", 768)
    # No phantom pin stored, and the member's capability model is untouched.
    assert service.get_route_embedding_model_overrides() == {}
    assert ollama["capabilities"]["embedding_model"] == "qwen3-embedding:8b"


def test_sync_setter_allows_route_native_alias_of_spaces_model():
    """#2440 P1+P2: the sync hydration path accepts the space's model under the
    member's route-native alias (normalized identity), so a legitimate persisted
    pin re-applies cleanly on boot."""
    service, _ = _verified_space_service()
    # Ollama's tagged alias normalizes to the pin's model → coherent, stored.
    service.set_route_embedding_model("ollama:local", "qwen3-embedding:8b", 768)
    assert service.get_route_embedding_model_overrides() == {
        "ollama:local": {"model": "qwen3-embedding:8b", "dim": 768}
    }


def test_non_member_route_is_unaffected_by_verified_space():
    """#2440: a route that is NOT a member of the space may be pinned freely."""
    service, _ = _verified_space_service()
    # openai:api is not a member — any model is fine.
    service._check_route_pin_space_coherent("openai:api", "text-embedding-3-large", 3072)


def test_unverified_space_does_not_constrain_member_pin():
    """#2440: only a VERIFIED space wins. An unverified pin does not lock members."""
    service, _ = _verified_space_service()
    # Drop the verified parity so the space is declared but not verified.
    service._verified_space_pins = {}
    service._check_route_pin_space_coherent("ollama:local", "nomic-embed-text", 768)


async def test_aset_refuses_conflicting_pin_on_verified_space_member():
    """#2440: the async setter surfaces the conflict BEFORE storing anything —
    the override map stays empty so no phantom pin is persisted."""
    from kestrel_sovereign.llm.service import EmbeddingSpaceConflictError

    service, _ = _verified_space_service()
    with pytest.raises(EmbeddingSpaceConflictError):
        await service.aset_route_embedding_model("ollama:local", "nomic-embed-text", 768)
    # Nothing was stored — the refusal is at set time, not after.
    assert service.get_route_embedding_model_overrides() == {}
