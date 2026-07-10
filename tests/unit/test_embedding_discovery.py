"""Dynamic embedding-model discovery (#2338).

Embedding models must be discovered per-vendor like chat models, with config
acting only as override/pin — never a prerequisite for advertising capability.

Covers:
- The discovery facet per adapter (mocked catalog payloads incl. the DEDICATED
  OpenRouter ``/embeddings/models`` endpoint shape, Ollama ``/api/show``
  capability check, OpenAI id-prefix filter).
- Capability advertisement flips when discovery returns ≥1 model.
- A config pin overrides discovery (folded in as ``is_pinned``), and is not a
  prerequisite.
- Shared-space candidates computed by local∩cloud intersection.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from kestrel_sovereign.llm.embedding_discovery import (
    EmbeddingModelInfo,
    normalize_embedding_model_id,
)
from kestrel_sovereign.llm.model_discovery import ModelDiscoveryMixin
from kestrel_sovereign.llm.openrouter_adapter import OpenRouterAdapter
from kestrel_sovereign.llm.ollama_adapter import OllamaAdapter
from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter


# --- normalization -----------------------------------------------------------

def test_normalize_strips_vendor_prefix_and_tag():
    assert normalize_embedding_model_id("qwen/qwen3-embedding-0.6b") == "qwen3-embedding-0.6b"
    assert normalize_embedding_model_id("qwen3-embedding:0.6b") == "qwen3-embedding-0.6b"
    assert normalize_embedding_model_id("nomic-embed-text") == "nomic-embed-text"
    assert normalize_embedding_model_id("") == ""


def test_native_dim_folds_into_options():
    m = EmbeddingModelInfo(id="x", provider="p", native_dim=768, dim_options=[256, 512])
    assert 768 in m.dim_options
    assert m.dim_options == [256, 512, 768]


# --- OpenRouter dedicated endpoint facet -------------------------------------

async def test_openrouter_embedding_discovery_uses_dedicated_endpoint(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    adapter = OpenRouterAdapter()

    payload = {
        "data": [
            {
                "id": "qwen/qwen3-embedding-8b",
                "name": "Qwen3 Embedding 8B",
                "context_length": 32768,
                "output_dimensions": {"min": 32, "max": 4096},
            },
            {
                "id": "google/gemini-embedding-2",
                "name": "Gemini Embedding 2",
                "dimensions": 3072,
            },
        ]
    }

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, **kw):
            # Must hit the DEDICATED embeddings endpoint, not /models.
            assert url.endswith("/embeddings/models")
            return _Resp()

    with patch("kestrel_sovereign.llm.openrouter_adapter.httpx.AsyncClient", _Client):
        models = await adapter.list_embedding_models()

    ids = {m.id for m in models}
    assert ids == {"qwen/qwen3-embedding-8b", "google/gemini-embedding-2"}
    qwen = next(m for m in models if m.id == "qwen/qwen3-embedding-8b")
    assert qwen.native_dim == 4096
    gemini = next(m for m in models if m.id == "google/gemini-embedding-2")
    assert gemini.native_dim == 3072


async def test_openrouter_embedding_discovery_no_key_returns_empty(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    adapter = OpenRouterAdapter()
    assert await adapter.list_embedding_models() == []


# --- Ollama /api/show capability facet ---------------------------------------

async def test_ollama_embedding_discovery_filters_by_capability():
    adapter = OllamaAdapter()

    list_result = SimpleNamespace(models=[
        SimpleNamespace(model="nomic-embed-text:latest"),
        SimpleNamespace(model="llama3.2:3b"),
        SimpleNamespace(model="qwen3-embedding:0.6b"),
    ])

    async def _show(name):
        caps = ["embedding"] if "embed" in name else ["completion", "tools"]
        return {"capabilities": caps}

    fake_client = SimpleNamespace(
        list=AsyncMock(return_value=list_result),
        show=_show,
    )

    with patch("kestrel_sovereign.llm.ollama_adapter.OLLAMA_AVAILABLE", True), \
         patch("kestrel_sovereign.llm.ollama_adapter.ollama") as mock_ollama:
        mock_ollama.AsyncClient.return_value = fake_client
        models = await adapter.list_embedding_models()

    ids = {m.id for m in models}
    assert ids == {"nomic-embed-text:latest", "qwen3-embedding:0.6b"}
    assert all(m.provider == "ollama" for m in models)


async def test_ollama_embedding_discovery_uses_provided_client():
    """The route-initialized client (route host) must be used, not a default one (#2338).

    Provider init builds the client from the route's ``host``/``OLLAMA_HOST``;
    discarding it and constructing ``ollama.AsyncClient()`` would probe the wrong
    (localhost) daemon.
    """
    adapter = OllamaAdapter()

    list_result = SimpleNamespace(models=[
        SimpleNamespace(model="nomic-embed-text:latest"),
    ])

    async def _show(name):
        return {"capabilities": ["embedding"]}

    provided_client = SimpleNamespace(
        list=AsyncMock(return_value=list_result),
        show=_show,
    )

    with patch("kestrel_sovereign.llm.ollama_adapter.OLLAMA_AVAILABLE", True), \
         patch("kestrel_sovereign.llm.ollama_adapter.ollama") as mock_ollama:
        models = await adapter.list_embedding_models(provided_client)

    # The passed client served the listing...
    provided_client.list.assert_awaited_once()
    # ...and no default client was constructed.
    mock_ollama.AsyncClient.assert_not_called()
    assert {m.id for m in models} == {"nomic-embed-text:latest"}


# --- OpenAI id-prefix facet --------------------------------------------------

async def test_openai_embedding_discovery_filters_by_id_prefix():
    adapter = OpenAIAdapter()
    client = SimpleNamespace(
        models=SimpleNamespace(
            list=AsyncMock(return_value=SimpleNamespace(data=[
                SimpleNamespace(id="gpt-5.5"),
                SimpleNamespace(id="text-embedding-3-small"),
                SimpleNamespace(id="text-embedding-3-large"),
            ]))
        )
    )

    models = await adapter.list_embedding_models(client)
    ids = {m.id for m in models}
    assert ids == {"text-embedding-3-small", "text-embedding-3-large"}


# --- aggregation / capability flip / pin override ----------------------------

class _FakeService(ModelDiscoveryMixin):
    def __init__(self, providers):
        self.providers = providers

    def _select_discovery_routes(self):
        # One route per vendor, mirroring the real helper.
        by_vendor = {}
        for p in self.providers:
            v = p.get("vendor") or p.get("name", "").split(":", 1)[0]
            by_vendor.setdefault(v, p)
        return list(by_vendor.items())


def _adapter_returning(models):
    a = SimpleNamespace()
    a.list_embedding_models = AsyncMock(return_value=list(models))
    return a


async def test_capability_flips_on_when_discovery_returns_models():
    adapter = _adapter_returning([
        EmbeddingModelInfo(id="qwen/qwen3-embedding-8b", provider="openrouter"),
    ])
    provider = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": adapter,
        "client": None,
        "capabilities": {"supports_embeddings": False},  # no config pin
    }
    svc = _FakeService([provider])

    assert await svc.route_advertises_embeddings(provider) is True
    models = await svc.discover_embedding_models(vendor="openrouter")
    assert len(models) == 1


async def test_capability_stays_off_when_discovery_empty():
    provider = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": _adapter_returning([]),
        "client": None,
        "capabilities": {},
    }
    svc = _FakeService([provider])
    assert await svc.route_advertises_embeddings(provider) is False


async def test_config_pin_overrides_and_is_not_prerequisite():
    # Discovery returns the pinned model AND another; the pin marks its match.
    adapter = _adapter_returning([
        EmbeddingModelInfo(id="qwen/qwen3-embedding-8b", provider="openrouter"),
        EmbeddingModelInfo(id="google/gemini-embedding-2", provider="openrouter"),
    ])
    provider = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": adapter,
        "client": None,
        "embedding_model": "qwen/qwen3-embedding-8b",
        "embedding_dim": 768,
        "capabilities": {},
    }
    svc = _FakeService([provider])
    models = await svc.discover_embedding_models()

    pinned = next(m for m in models if m.id == "qwen/qwen3-embedding-8b")
    assert pinned.is_pinned is True
    assert pinned.native_dim == 768
    # The non-pinned model is still discovered — pin is not a prerequisite.
    assert any(m.id == "google/gemini-embedding-2" and not m.is_pinned for m in models)


async def test_config_pin_added_synthetically_when_discovery_misses_it():
    provider = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": _adapter_returning([]),  # discovery finds nothing
        "client": None,
        "embedding_model": "qwen/qwen3-embedding-0.6b",
        "embedding_dim": 768,
        "capabilities": {},
    }
    svc = _FakeService([provider])
    models = await svc.discover_embedding_models()
    assert len(models) == 1
    assert models[0].id == "qwen/qwen3-embedding-0.6b"
    assert models[0].is_pinned is True


# --- auto-resolution (mirrors chat "auto") -----------------------------------

async def test_resolve_default_respects_selection_hints():
    provider = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": _adapter_returning([
            EmbeddingModelInfo(id="google/gemini-embedding-2", provider="openrouter"),
            EmbeddingModelInfo(id="qwen/qwen3-embedding-8b", provider="openrouter"),
        ]),
        "client": None,
        "selection_hints": ["qwen3"],
        "capabilities": {},
    }
    svc = _FakeService([provider])
    chosen = await svc.resolve_default_embedding_model(provider)
    assert chosen.id == "qwen/qwen3-embedding-8b"


async def test_resolve_default_falls_back_to_first_discovered():
    provider = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": _adapter_returning([
            EmbeddingModelInfo(id="google/gemini-embedding-2", provider="openrouter"),
        ]),
        "client": None,
        "capabilities": {},  # no hints, no pin
    }
    svc = _FakeService([provider])
    chosen = await svc.resolve_default_embedding_model(provider)
    assert chosen.id == "google/gemini-embedding-2"


async def test_resolve_default_pin_wins_over_hint():
    provider = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": _adapter_returning([
            EmbeddingModelInfo(id="google/gemini-embedding-2", provider="openrouter"),
            EmbeddingModelInfo(id="qwen/qwen3-embedding-8b", provider="openrouter"),
        ]),
        "client": None,
        "embedding_model": "google/gemini-embedding-2",
        "selection_hints": ["qwen3"],
        "capabilities": {},
    }
    svc = _FakeService([provider])
    chosen = await svc.resolve_default_embedding_model(provider)
    assert chosen.id == "google/gemini-embedding-2"
    assert chosen.is_pinned is True


async def test_resolve_default_none_when_no_discovery():
    provider = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": _adapter_returning([]),
        "client": None,
        "capabilities": {},
    }
    svc = _FakeService([provider])
    assert await svc.resolve_default_embedding_model(provider) is None


# --- #2366 corpus-first auto resolution --------------------------------------


async def test_resolve_default_prefers_corpus_dominant_profile():
    # Catalog order puts gemini first; the corpus is entirely qwen3.
    provider = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": _adapter_returning([
            EmbeddingModelInfo(id="google/gemini-embedding-2", provider="openrouter"),
            EmbeddingModelInfo(id="qwen/qwen3-embedding-8b", provider="openrouter"),
        ]),
        "client": None,
        "capabilities": {},
    }
    svc = _FakeService([provider])
    corpus = {"provider": "openrouter", "model": "qwen3-embedding-8b",
              "dim": 768, "space_id": "s", "row_count": 63}
    chosen = await svc.resolve_default_embedding_model(provider, corpus_profile=corpus)
    assert chosen.id == "qwen/qwen3-embedding-8b"


async def test_resolve_default_corpus_matches_cross_route_identity():
    # The corpus model was written on Ollama (bare id); the route serves the
    # same weights under an OpenRouter vendor/ prefix — must still match.
    provider = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": _adapter_returning([
            EmbeddingModelInfo(id="google/gemini-embedding-2", provider="openrouter"),
            EmbeddingModelInfo(id="qwen/qwen3-embedding-0.6b", provider="openrouter"),
        ]),
        "client": None,
        "capabilities": {},
    }
    svc = _FakeService([provider])
    corpus = {"provider": "ollama", "model": "qwen3-embedding:0.6b",
              "dim": 768, "space_id": "s", "row_count": 10}
    chosen = await svc.resolve_default_embedding_model(provider, corpus_profile=corpus)
    assert chosen.id == "qwen/qwen3-embedding-0.6b"


async def test_resolve_default_prefers_deployment_dim_when_no_corpus_match():
    # Corpus model is absent from the catalog; fall to deployment-dim match
    # before catalog order (gemini@3072 sorts first).
    provider = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": _adapter_returning([
            EmbeddingModelInfo(id="google/gemini-embedding-2", provider="openrouter",
                               native_dim=3072),
            EmbeddingModelInfo(id="qwen/qwen3-embedding-8b", provider="openrouter",
                               native_dim=4096, dim_options=[768, 1024]),
        ]),
        "client": None,
        "capabilities": {},
    }
    svc = _FakeService([provider])
    chosen = await svc.resolve_default_embedding_model(provider, deployment_dim=768)
    assert chosen.id == "qwen/qwen3-embedding-8b"


async def test_resolve_default_empty_corpus_falls_through_to_hints():
    # Empty DB (corpus_profile=None) → prior hint/catalog behaviour is intact.
    provider = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": _adapter_returning([
            EmbeddingModelInfo(id="google/gemini-embedding-2", provider="openrouter"),
            EmbeddingModelInfo(id="qwen/qwen3-embedding-8b", provider="openrouter"),
        ]),
        "client": None,
        "selection_hints": ["qwen3"],
        "capabilities": {},
    }
    svc = _FakeService([provider])
    chosen = await svc.resolve_default_embedding_model(provider, corpus_profile=None)
    assert chosen.id == "qwen/qwen3-embedding-8b"


async def test_reconcile_records_space_change_warning_on_new_space():
    # Corpus is qwen3; the only discovered model is gemini — the auto default
    # necessarily changes the space, which must be recorded loudly.
    provider = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": _adapter_returning([
            EmbeddingModelInfo(id="google/gemini-embedding-2", provider="openrouter",
                               native_dim=3072),
        ]),
        "client": None,
        "capabilities": {},
    }
    svc = _FakeService([provider])
    corpus = {"provider": "openrouter", "model": "qwen3-embedding-8b",
              "dim": 768, "space_id": "s", "row_count": 63}
    svc._corpus_embedding_profile_provider = AsyncMock(return_value=corpus)
    svc._embedding_space_change_warnings = {}

    await svc.reconcile_embedding_capabilities(use_cache=False)

    assert provider["capabilities"]["embedding_model"] == "google/gemini-embedding-2"
    warning = svc._embedding_space_change_warnings.get("openrouter:api")
    assert warning is not None
    assert warning["chosen_model"] == "google/gemini-embedding-2"
    assert warning["corpus_model"] == "qwen3-embedding-8b"


async def test_reconcile_no_warning_when_corpus_matches():
    provider = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": _adapter_returning([
            EmbeddingModelInfo(id="qwen/qwen3-embedding-8b", provider="openrouter",
                               native_dim=768),
        ]),
        "client": None,
        "capabilities": {},
    }
    svc = _FakeService([provider])
    corpus = {"provider": "openrouter", "model": "qwen3-embedding-8b",
              "dim": 768, "space_id": "s", "row_count": 63}
    svc._corpus_embedding_profile_provider = AsyncMock(return_value=corpus)
    svc._embedding_space_change_warnings = {}

    await svc.reconcile_embedding_capabilities(use_cache=False)

    assert provider["capabilities"]["embedding_model"] == "qwen/qwen3-embedding-8b"
    assert svc._embedding_space_change_warnings.get("openrouter:api") is None


# --- shared-space candidates -------------------------------------------------

async def test_shared_space_candidates_intersect_local_and_cloud():
    cloud = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": _adapter_returning([
            EmbeddingModelInfo(id="qwen/qwen3-embedding-0.6b", provider="openrouter"),
            EmbeddingModelInfo(id="google/gemini-embedding-2", provider="openrouter"),
        ]),
        "client": None,
        "is_local": False,
        "capabilities": {},
    }
    local = {
        "vendor": "ollama",
        "name": "ollama:local",
        "adapter": _adapter_returning([
            EmbeddingModelInfo(id="qwen3-embedding:0.6b", provider="ollama"),
            EmbeddingModelInfo(id="nomic-embed-text", provider="ollama"),
        ]),
        "client": None,
        "is_local": True,
        "capabilities": {},
    }
    svc = _FakeService([cloud, local])

    shared = await svc.shared_embedding_space_candidates()
    # qwen3-embedding-0.6b is on both sides (normalized); gemini/nomic are not.
    assert [normalize_embedding_model_id(m.id) for m in shared] == ["qwen3-embedding-0.6b"]


async def test_universal_options_carry_member_routes_with_own_slugs():
    """The featured "Universal" option enriches a shared model with member
    routes, each carrying THAT route's own slug (#2337)."""
    cloud = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": _adapter_returning([
            EmbeddingModelInfo(id="qwen/qwen3-embedding-0.6b", provider="openrouter", native_dim=768),
            EmbeddingModelInfo(id="google/gemini-embedding-2", provider="openrouter"),
        ]),
        "client": None,
        "is_local": False,
        "capabilities": {},
    }
    local = {
        "vendor": "ollama",
        "name": "ollama:local",
        "adapter": _adapter_returning([
            EmbeddingModelInfo(id="qwen3-embedding:0.6b", provider="ollama", native_dim=768),
            EmbeddingModelInfo(id="nomic-embed-text", provider="ollama"),
        ]),
        "client": None,
        "is_local": True,
        "capabilities": {},
    }
    svc = _FakeService([cloud, local])

    options = await svc.universal_embedding_space_options()
    assert len(options) == 1
    opt = options[0]
    # Members span both routes, each with its own upstream slug and locality.
    members = {m["route"]: m for m in opt["members"]}
    assert set(members) == {"openrouter:api", "ollama:local"}
    assert members["openrouter:api"]["model"] == "qwen/qwen3-embedding-0.6b"
    assert members["openrouter:api"]["is_local"] is False
    assert members["ollama:local"]["model"] == "qwen3-embedding:0.6b"
    assert members["ollama:local"]["is_local"] is True


async def test_universal_options_empty_without_a_shared_model():
    """No model on both sides → no Universal option (never hardcoded)."""
    cloud = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": _adapter_returning([
            EmbeddingModelInfo(id="google/gemini-embedding-2", provider="openrouter"),
        ]),
        "client": None,
        "is_local": False,
        "capabilities": {},
    }
    local = {
        "vendor": "ollama",
        "name": "ollama:local",
        "adapter": _adapter_returning([
            EmbeddingModelInfo(id="nomic-embed-text", provider="ollama"),
        ]),
        "client": None,
        "is_local": True,
        "capabilities": {},
    }
    svc = _FakeService([cloud, local])
    assert await svc.universal_embedding_space_options() == []


# --- per-route capability (no vendor collapse, #2338) ------------------------

async def test_advertisement_is_route_specific_not_vendor_collapsed():
    """Two routes of the SAME vendor: only the embedding-capable one advertises.

    ``openai:api`` discovers an embedding model; ``openai:plan`` (codex — no
    ``list_embedding_models`` facet) discovers none. Capability must NOT collapse
    by vendor and flip ``openai:plan`` on off ``openai:api``'s models.
    """
    api = {
        "vendor": "openai",
        "name": "openai:api",
        "adapter": _adapter_returning([
            EmbeddingModelInfo(id="text-embedding-3-small", provider="openai"),
        ]),
        "client": None,
        "capabilities": {},
    }
    # A codex-like route: adapter has no ``list_embedding_models`` at all.
    plan = {
        "vendor": "openai",
        "name": "openai:plan",
        "adapter": SimpleNamespace(),
        "client": None,
        "capabilities": {},
    }
    svc = _FakeService([api, plan])

    assert await svc.route_advertises_embeddings(api) is True
    assert await svc.route_advertises_embeddings(plan) is False

    # Discovered models carry their originating route.
    models = await svc.discover_embedding_models()
    assert {m.route for m in models} == {"openai:api"}
    assert await svc.discover_embedding_models(route="openai:plan") == []


async def test_reconcile_writes_capability_into_the_discovering_route_only():
    """reconcile_embedding_capabilities flips the sync capability per route (#2338).

    The runtime path (resolve_embedding_provider / set-time validation) reads the
    static ``supports_embeddings`` flag; reconcile must populate it for the route
    that discovered embeddings — and leave the non-embedding sibling untouched.
    """
    api = {
        "vendor": "openai",
        "name": "openai:api",
        "adapter": _adapter_returning([
            EmbeddingModelInfo(id="text-embedding-3-large", provider="openai", native_dim=3072),
        ]),
        "client": None,
        "capabilities": {},
    }
    plan = {
        "vendor": "openai",
        "name": "openai:plan",
        "adapter": SimpleNamespace(),
        "client": None,
        "capabilities": {},
    }
    svc = _FakeService([api, plan])

    await svc.reconcile_embedding_capabilities()

    assert api["capabilities"]["supports_embeddings"] is True
    assert api["capabilities"]["embedding_model"] == "text-embedding-3-large"
    assert api["capabilities"]["embedding_dim"] == 3072
    # The sibling that discovered nothing is never flipped on.
    assert plan["capabilities"].get("supports_embeddings") in (None, False)


async def test_reconcile_never_downgrades_a_static_pin():
    """A TOML/pin capability is operator intent — reconcile only turns ON."""
    provider = {
        "vendor": "openai",
        "name": "openai:api",
        "adapter": _adapter_returning([]),  # discovery finds nothing live
        "client": None,
        "capabilities": {"supports_embeddings": True, "embedding_model": "pinned-model"},
    }
    svc = _FakeService([provider])
    await svc.reconcile_embedding_capabilities()
    assert provider["capabilities"]["supports_embeddings"] is True
    assert provider["capabilities"]["embedding_model"] == "pinned-model"


# --- #2372 single resolver: route → (model, dim), used by all surfaces -------


async def test_resolve_route_embedding_model_falls_through_to_catalog(monkeypatch):
    # Cleared-pin state: capabilities carry no embedding_model, but discovery
    # finds one for the route. The single resolver must return that model — a
    # cleared pin can NEVER be silent-None while a capable model is discovered.
    monkeypatch.setattr(
        "kestrel_sovereign.llm.model_discovery._resolve_deployment_embedding_dim",
        lambda: None,
    )
    provider = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": _adapter_returning([
            EmbeddingModelInfo(id="qwen/qwen3-embedding-8b", provider="openrouter",
                               native_dim=4096, dim_options=[768, 1024, 4096]),
        ]),
        "client": None,
        "capabilities": {},
    }
    svc = _FakeService([provider])
    model, dim = await svc.resolve_route_embedding_model(provider)
    assert model == "qwen/qwen3-embedding-8b"
    # Side-effect: the resolution is persisted into capabilities so the sync
    # readers (get_embedding_settings / ProviderEmbeddingService) agree.
    assert provider["capabilities"]["embedding_model"] == "qwen/qwen3-embedding-8b"
    assert provider["capabilities"]["supports_embeddings"] is True


async def test_resolve_route_embedding_model_prefers_corpus_cross_route(monkeypatch):
    # The corpus was written on Ollama (qwen3-embedding:8b); the route serves the
    # same weights under an OpenRouter vendor/ prefix. A cleared pin must resolve
    # to the corpus-continuous model via normalized matching — never gemini,
    # never None — and adopt the corpus dim the model can serve (not native).
    monkeypatch.setattr(
        "kestrel_sovereign.llm.model_discovery._resolve_deployment_embedding_dim",
        lambda: None,
    )
    provider = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": _adapter_returning([
            EmbeddingModelInfo(id="google/gemini-embedding-2", provider="openrouter",
                               native_dim=3072),
            EmbeddingModelInfo(id="qwen/qwen3-embedding-8b", provider="openrouter",
                               native_dim=4096, dim_options=[768, 4096]),
        ]),
        "client": None,
        "capabilities": {},
    }
    svc = _FakeService([provider])
    corpus = {"provider": "ollama", "model": "qwen3-embedding:8b",
              "dim": 768, "space_id": "s", "row_count": 63}
    svc._corpus_embedding_profile_provider = AsyncMock(return_value=corpus)
    model, dim = await svc.resolve_route_embedding_model(provider)
    assert model == "qwen/qwen3-embedding-8b"
    assert dim == 768


async def test_resolve_route_embedding_model_none_only_when_no_discovery():
    # Truthful "off": nothing discovered for the route → (None, None), not a
    # fabricated model.
    provider = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": _adapter_returning([]),
        "client": None,
        "capabilities": {},
    }
    svc = _FakeService([provider])
    assert await svc.resolve_route_embedding_model(provider) == (None, None)


async def test_resolve_route_embedding_model_preserves_explicit_pin():
    # A pinned route keeps its own slug AND its own pinned dim — the resolver
    # honours operator intent verbatim.
    provider = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": _adapter_returning([
            EmbeddingModelInfo(id="qwen/qwen3-embedding-8b", provider="openrouter",
                               native_dim=4096, dim_options=[768, 4096]),
        ]),
        "client": None,
        "capabilities": {
            "supports_embeddings": True,
            "embedding_model": "qwen/qwen3-embedding-8b",
            "embedding_dim": 768,
        },
    }
    svc = _FakeService([provider])
    model, dim = await svc.resolve_route_embedding_model(provider)
    assert model == "qwen/qwen3-embedding-8b"
    assert dim == 768


async def test_stale_space_change_warning_cleared_when_corpus_realigns(monkeypatch):
    # Round-4 #2372 "stale readout": a space-change warning recorded when the
    # auto default did not match the corpus must be CLEARED once a later resolve
    # lands back on the corpus space (e.g. after a re-embed changed the dominant
    # profile) — it must not outlive the condition that produced it.
    monkeypatch.setattr(
        "kestrel_sovereign.llm.model_discovery._resolve_deployment_embedding_dim",
        lambda: None,
    )
    provider = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": _adapter_returning([
            EmbeddingModelInfo(id="qwen/qwen3-embedding-8b", provider="openrouter",
                               native_dim=768),
        ]),
        "client": None,
        "capabilities": {},
    }
    svc = _FakeService([provider])
    svc._embedding_space_change_warnings = {}

    # Corpus dominant is (stale) nomic → qwen changes the space → warning recorded.
    svc._corpus_embedding_profile_provider = AsyncMock(return_value={
        "provider": "ollama", "model": "nomic-embed-text",
        "dim": 768, "space_id": "s", "row_count": 24,
    })
    await svc.resolve_route_embedding_model(provider)
    assert svc._embedding_space_change_warnings.get("openrouter:api") is not None

    # A re-embed moved the corpus onto qwen; the next resolve is coherent, so the
    # stale warning must be gone (not surfaced by the settings GET forever).
    svc._corpus_embedding_profile_provider = AsyncMock(return_value={
        "provider": "openrouter", "model": "qwen3-embedding-8b",
        "dim": 768, "space_id": "s", "row_count": 24,
    })
    svc.clear_embedding_discovery_cache()
    await svc.resolve_route_embedding_model(provider)
    assert svc._embedding_space_change_warnings.get("openrouter:api") is None


async def test_explicit_pin_clears_stale_space_change_warning(monkeypatch):
    # A deliberate operator pin is not an accidental space split — pinning must
    # clear any warning a prior auto-resolve recorded for the route (#2372).
    monkeypatch.setattr(
        "kestrel_sovereign.llm.model_discovery._resolve_deployment_embedding_dim",
        lambda: None,
    )
    provider = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": _adapter_returning([
            EmbeddingModelInfo(id="google/gemini-embedding-2", provider="openrouter",
                               native_dim=3072),
        ]),
        "client": None,
        "capabilities": {},
    }
    svc = _FakeService([provider])
    svc._embedding_space_change_warnings = {}
    svc._corpus_embedding_profile_provider = AsyncMock(return_value={
        "provider": "ollama", "model": "nomic-embed-text",
        "dim": 768, "space_id": "s", "row_count": 24,
    })
    await svc.resolve_route_embedding_model(provider)
    assert svc._embedding_space_change_warnings.get("openrouter:api") is not None

    # Operator pins gemini deliberately → the warning for this route is cleared.
    # A genuine pin carries NO auto-resolved marker (``set_route_embedding_model``
    # drops it), which is exactly how discovery tells a pin from an auto default.
    provider["capabilities"].update({
        "supports_embeddings": True,
        "embedding_model": "google/gemini-embedding-2",
        "embedding_dim": 3072,
    })
    provider["capabilities"].pop("embedding_model_auto_resolved", None)
    svc.clear_embedding_discovery_cache()
    await svc.resolve_route_embedding_model(provider)
    assert svc._embedding_space_change_warnings.get("openrouter:api") is None


async def test_auto_resolved_capability_is_not_a_pin_after_cache_clear(monkeypatch):
    # P1 (#2372): a default written back into capabilities by the resolver must
    # NOT be re-read as an operator pin after the reindex path clears the cache.
    # Otherwise the route freezes on the stale auto model/dim and the corpus
    # fallback (which should follow a corpus change) is blocked.
    monkeypatch.setattr(
        "kestrel_sovereign.llm.model_discovery._resolve_deployment_embedding_dim",
        lambda: None,
    )
    provider = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": _adapter_returning([
            EmbeddingModelInfo(id="qwen/qwen3-embedding-8b", provider="openrouter",
                               native_dim=768),
            EmbeddingModelInfo(id="google/gemini-embedding-2", provider="openrouter",
                               native_dim=3072),
        ]),
        "client": None,
        "capabilities": {},
    }
    svc = _FakeService([provider])

    # First resolve against a qwen corpus → auto-resolves to the qwen model and
    # marks the capability as auto-resolved (not a pin).
    corpus_qwen = {"provider": "openrouter", "model": "qwen3-embedding-8b",
                   "dim": 768, "space_id": "s", "row_count": 10}
    svc._corpus_embedding_profile_provider = AsyncMock(return_value=corpus_qwen)
    svc._embedding_space_change_warnings = {}
    model, _ = await svc.resolve_route_embedding_model(provider)
    assert model == "qwen/qwen3-embedding-8b"
    assert provider["capabilities"]["embedding_model_auto_resolved"] is True

    # The corpus changes to gemini; the reindex path clears the cache. The
    # stale auto default must NOT be honoured as a pin — re-resolution follows
    # the new corpus.
    svc._corpus_embedding_profile_provider = AsyncMock(return_value={
        "provider": "openrouter", "model": "gemini-embedding-2",
        "dim": 3072, "space_id": "s2", "row_count": 40,
    })
    svc.clear_embedding_discovery_cache()
    model2, _ = await svc.resolve_route_embedding_model(provider)
    assert model2 == "google/gemini-embedding-2"

    # And a discovered candidate for the auto value is not flagged is_pinned.
    models = await svc.discover_embedding_models()
    assert all(not m.is_pinned for m in models)


async def test_explicit_config_pin_survives_cache_clear(monkeypatch):
    # The mirror of the above: a GENUINE config pin (no auto marker) is still a
    # pin after a cache clear and wins over the corpus.
    monkeypatch.setattr(
        "kestrel_sovereign.llm.model_discovery._resolve_deployment_embedding_dim",
        lambda: None,
    )
    provider = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": _adapter_returning([
            EmbeddingModelInfo(id="qwen/qwen3-embedding-8b", provider="openrouter",
                               native_dim=768),
            EmbeddingModelInfo(id="google/gemini-embedding-2", provider="openrouter",
                               native_dim=3072),
        ]),
        "client": None,
        "capabilities": {
            "supports_embeddings": True,
            "embedding_model": "google/gemini-embedding-2",
            "embedding_dim": 3072,
        },
    }
    svc = _FakeService([provider])
    corpus_qwen = {"provider": "openrouter", "model": "qwen3-embedding-8b",
                   "dim": 768, "space_id": "s", "row_count": 10}
    svc._corpus_embedding_profile_provider = AsyncMock(return_value=corpus_qwen)
    svc._embedding_space_change_warnings = {}

    model, dim = await svc.resolve_route_embedding_model(provider)
    assert model == "google/gemini-embedding-2"  # pin beats corpus
    assert dim == 3072
    svc.clear_embedding_discovery_cache()
    model2, dim2 = await svc.resolve_route_embedding_model(provider)
    assert model2 == "google/gemini-embedding-2"
    assert "embedding_model_auto_resolved" not in provider["capabilities"]


async def test_discovery_cache_reused_until_cleared():
    adapter = _adapter_returning([
        EmbeddingModelInfo(id="qwen/qwen3-embedding-8b", provider="openrouter"),
    ])
    provider = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": adapter,
        "client": None,
        "capabilities": {},
    }
    svc = _FakeService([provider])

    await svc.discover_embedding_models()
    await svc.discover_embedding_models()
    assert adapter.list_embedding_models.await_count == 1

    svc.clear_embedding_discovery_cache()
    await svc.discover_embedding_models()
    assert adapter.list_embedding_models.await_count == 2
