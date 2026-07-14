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

import httpx
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
    chosen, _ = await svc.resolve_default_embedding_model(provider)
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
    chosen, _ = await svc.resolve_default_embedding_model(provider)
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
    chosen, _ = await svc.resolve_default_embedding_model(provider)
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
    assert await svc.resolve_default_embedding_model(provider) == (None, None)


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
    chosen, dim = await svc.resolve_default_embedding_model(provider, corpus_profile=corpus)
    assert chosen.id == "qwen/qwen3-embedding-8b"
    # #2376 — a corpus match carries the corpus's dominant dim, so the resolved
    # state is a full ``<model>@<dim>`` space, never a bare model.
    assert dim == 768


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
    chosen, dim = await svc.resolve_default_embedding_model(provider, corpus_profile=corpus)
    assert chosen.id == "qwen/qwen3-embedding-0.6b"
    assert dim == 768


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
    chosen, dim = await svc.resolve_default_embedding_model(provider, deployment_dim=768)
    assert chosen.id == "qwen/qwen3-embedding-8b"
    # #2376 — the dim-compat branch carries the matched deployment dim.
    assert dim == 768


async def test_resolve_route_refuses_model_without_a_dim():
    # #2376 — hint/catalog fallback picks a model discovery could give no dim for
    # (native_dim=None, empty dim_options — the ollama/OpenRouter case), and no
    # corpus/deployment continuity pins one. An embedding-capable state without a
    # concrete dim is invalid by construction (native-dim embeds → column-guard
    # write refusals → read-spec confusion), so the route must NOT be advertised
    # as embedding-capable: never persist embedding_model without embedding_dim.
    provider = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": _adapter_returning([
            EmbeddingModelInfo(id="google/gemini-embedding-2", provider="openrouter",
                               native_dim=None, dim_options=[]),
        ]),
        "client": None,
        "capabilities": {},
    }
    svc = _FakeService([provider])
    model, dim = await svc.resolve_route_embedding_model(provider)
    assert model is None
    assert dim is None
    caps = provider["capabilities"]
    assert caps.get("embedding_model") is None
    assert caps.get("embedding_dim") is None
    assert not caps.get("supports_embeddings")
    # The invariant: a persisted embedding_model always carries a dim.
    if caps.get("embedding_model") is not None:
        assert caps.get("embedding_dim") is not None


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
    chosen, _ = await svc.resolve_default_embedding_model(provider, corpus_profile=None)
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


# --- #2376 auto-resolved model must carry the resolved dim -------------------


async def test_cleared_pin_corpus_match_attaches_corpus_dim_without_dim_options(monkeypatch):
    # Round-5 (#2376): the corpus-matched model's ``dim_options`` is EMPTY —
    # discovery (ollama /api/show, the OpenRouter catalog) does not expose its
    # Matryoshka range — so ``_model_offers_dim`` can't confirm 768. The corpus
    # it matched against IS the proof: a corpus match keeps the existing space
    # (``<model>@768``), so 768 must still be attached, NEVER native 4096.
    monkeypatch.setattr(
        "kestrel_sovereign.llm.model_discovery._resolve_deployment_embedding_dim",
        lambda: None,
    )
    provider = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": _adapter_returning([
            EmbeddingModelInfo(id="qwen/qwen3-embedding-8b", provider="openrouter",
                               native_dim=4096),  # dim_options empty
        ]),
        "client": None,
        "capabilities": {},
    }
    svc = _FakeService([provider])
    corpus = {"provider": "openrouter", "model": "qwen3-embedding-8b",
              "dim": 768, "space_id": "s", "row_count": 63}
    svc._corpus_embedding_profile_provider = AsyncMock(return_value=corpus)

    model, dim = await svc.resolve_route_embedding_model(provider)
    assert (model, dim) == ("qwen/qwen3-embedding-8b", 768)
    # The dim is persisted so the sync readers agree and never embed at native.
    assert provider["capabilities"]["embedding_dim"] == 768


async def test_auto_resolved_state_embeds_at_resolved_dim_not_native(monkeypatch):
    # The live-write shape: once the resolver attaches the corpus dim, a
    # ProviderEmbeddingService built from those capabilities forwards
    # ``dimensions=768`` on every embed (a fake MRL adapter truncates to it),
    # so the write lands 768-wide vectors that the column guard accepts —
    # instead of native 4096 that it refuses (#2376).
    from kestrel_sovereign.llm.embedding_service import ProviderEmbeddingService

    monkeypatch.setattr(
        "kestrel_sovereign.llm.model_discovery._resolve_deployment_embedding_dim",
        lambda: None,
    )

    class _MRLAdapter:
        """A Matryoshka adapter: truncates to the requested ``dimensions``."""
        async def aembed(self, client, text, *, model=None, dimensions=None):
            width = dimensions or 4096  # native when nothing requested
            return [0.1] * width

    provider = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": _adapter_returning([
            EmbeddingModelInfo(id="qwen/qwen3-embedding-8b", provider="openrouter",
                               native_dim=4096),
        ]),
        "client": None,
        "capabilities": {},
    }
    svc = _FakeService([provider])
    corpus = {"provider": "openrouter", "model": "qwen3-embedding-8b",
              "dim": 768, "space_id": "s", "row_count": 63}
    svc._corpus_embedding_profile_provider = AsyncMock(return_value=corpus)
    await svc.resolve_route_embedding_model(provider)

    # Build the runtime embedding service from the resolved capabilities and a
    # real MRL adapter; the embed must arrive at 768, not the native 4096.
    provider["adapter"] = _MRLAdapter()
    embed_svc = ProviderEmbeddingService(provider)
    vec = await embed_svc.aembed("hello")
    assert len(vec) == 768


async def test_resolved_embedding_state_invariant_model_implies_dim(monkeypatch):
    # Invariant (#2376): a resolved embedding-capable route can NEVER carry a
    # model with a null dim — that state embeds at native width and breaks the
    # write/read paths. Every non-``None`` resolution attaches a dim.
    monkeypatch.setattr(
        "kestrel_sovereign.llm.model_discovery._resolve_deployment_embedding_dim",
        lambda: None,
    )
    scenarios = [
        # (corpus_profile, native_dim, dim_options)
        ({"provider": "openrouter", "model": "qwen3-embedding-8b",
          "dim": 768, "space_id": "s", "row_count": 5}, 4096, []),   # corpus match
        (None, 3072, []),                                            # catalog fallback
    ]
    for corpus, native_dim, dim_options in scenarios:
        provider = {
            "vendor": "openrouter",
            "name": "openrouter:api",
            "adapter": _adapter_returning([
                EmbeddingModelInfo(id="qwen/qwen3-embedding-8b", provider="openrouter",
                                   native_dim=native_dim, dim_options=list(dim_options)),
            ]),
            "client": None,
            "capabilities": {},
        }
        svc = _FakeService([provider])
        svc._corpus_embedding_profile_provider = AsyncMock(return_value=corpus)
        model, dim = await svc.resolve_route_embedding_model(provider)
        assert model is not None
        assert dim is not None, "embedding_model set ⇒ embedding_dim set"
        assert provider["capabilities"].get("embedding_dim") is not None


# --- #2433 chat-only OpenAI-compatible routes: no extra request, no traceback ---


async def test_openai_embedding_discovery_reuses_chat_listing_without_request():
    """Generic OpenAI-compatible routes derive embeddings from the chat listing
    already fetched (#2433) — no second ``/v1/models`` request."""
    adapter = OpenAIAdapter()
    client = SimpleNamespace(
        models=SimpleNamespace(list=AsyncMock())
    )

    models = await adapter.list_embedding_models(
        client,
        chat_models=[
            "gpt-5.5",
            "text-embedding-3-small",
            "meta-llama/Llama-3.1-8B",  # a RunPod chat-only pin
        ],
    )

    # The chat listing was filtered locally; no network call was issued.
    client.models.list.assert_not_awaited()
    assert {m.id for m in models} == {"text-embedding-3-small"}


async def test_openai_embedding_discovery_empty_chat_listing_no_request():
    """A chat-only route (RunPod vLLM whose model-list 404'd → empty chat set)
    yields no embedding models and makes no embedding request (#2433)."""
    adapter = OpenAIAdapter()
    client = SimpleNamespace(models=SimpleNamespace(list=AsyncMock()))

    models = await adapter.list_embedding_models(client, chat_models=[])

    client.models.list.assert_not_awaited()
    assert models == []


async def test_generic_route_probe_uses_reused_chat_ids(caplog):
    """The discovery layer passes reused chat ids to a generic adapter and never
    logs an ERROR for a chat-only route (#2433)."""
    import logging

    adapter = OpenAIAdapter()  # derives_embeddings_from_chat_listing = True
    client = SimpleNamespace(models=SimpleNamespace(list=AsyncMock()))
    provider = {
        "vendor": "openai",
        "name": "runpod:vllm",
        "adapter": adapter,
        "client": client,
        "capabilities": {},  # chat-only route: no embedding claim
    }
    svc = _FakeService([provider])
    # Chat discovery found only a chat-only pin — no embedding ids. The snapshot
    # is keyed by ROUTE, not vendor (#2433), so the route reuses its OWN listing.
    svc._chat_models_by_route = {"runpod:vllm": ["meta-llama/Llama-3.1-8B"]}

    with caplog.at_level(logging.INFO):
        models = await svc.discover_embedding_models()

    client.models.list.assert_not_awaited()
    assert models == []
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("no embedding models discovered" in r.message for r in caplog.records)


async def test_claiming_route_probe_failure_logs_error(caplog):
    """A route that explicitly claims embedding support and then fails is a loud
    ERROR with traceback — that observability is preserved (#2433)."""
    import logging

    failing = SimpleNamespace()
    failing.list_embedding_models = AsyncMock(side_effect=RuntimeError("boom"))
    provider = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": failing,
        "client": None,
        # Explicit operator pin ⇒ a claim.
        "embedding_model": "qwen/qwen3-embedding-8b",
        "capabilities": {},
    }
    svc = _FakeService([provider])

    with caplog.at_level(logging.INFO):
        await svc.discover_embedding_models()

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "a claiming route's failure must log an ERROR"
    assert any("claims" in r.message for r in errors)


async def test_non_claiming_route_probe_failure_is_info_not_error(caplog):
    """A route that does NOT claim embeddings and fails is a quiet INFO, never
    an ERROR traceback (#2433)."""
    import logging

    failing = SimpleNamespace()
    failing.list_embedding_models = AsyncMock(side_effect=RuntimeError("404"))
    provider = {
        "vendor": "somevendor",
        "name": "somevendor:route",
        "adapter": failing,
        "client": None,
        "capabilities": {},  # no claim
    }
    svc = _FakeService([provider])

    with caplog.at_level(logging.INFO):
        await svc.discover_embedding_models()

    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


async def test_auto_resolved_capability_is_not_a_claim():
    """An auto-resolved capability the resolver wrote back is not an operator
    claim, so it must NOT escalate a probe failure to ERROR (#2433)."""
    route = {
        "name": "openrouter:api",
        "capabilities": {
            "supports_embeddings": True,
            "embedding_model": "qwen/qwen3-embedding-8b",
            "embedding_model_auto_resolved": True,
        },
    }
    assert ModelDiscoveryMixin._route_claims_embedding_support(route) is False

    # A genuine pin (no auto marker) IS a claim.
    pinned = {"name": "openrouter:api", "embedding_model": "x", "capabilities": {}}
    assert ModelDiscoveryMixin._route_claims_embedding_support(pinned) is True


async def test_concurrent_discovery_is_single_flight():
    """Concurrent refresh+invoke coalesce behind one in-flight discovery — the
    adapter is probed once, not twice (#2433)."""
    import asyncio as _asyncio

    call_count = {"n": 0}

    async def _slow_list(client=None, chat_models=None):
        call_count["n"] += 1
        await _asyncio.sleep(0.05)
        return [EmbeddingModelInfo(id="text-embedding-3-small", provider="openai")]

    adapter = SimpleNamespace()
    adapter.list_embedding_models = _slow_list
    provider = {
        "vendor": "openai",
        "name": "openai:api",
        "adapter": adapter,
        "client": None,
        "capabilities": {},
    }
    svc = _FakeService([provider])

    a, b = await _asyncio.gather(
        svc.discover_embedding_models(),
        svc.discover_embedding_models(),
    )

    assert call_count["n"] == 1
    assert {m.id for m in a} == {"text-embedding-3-small"}
    assert {m.id for m in b} == {"text-embedding-3-small"}


async def test_openrouter_real_adapter_failure_propagates_and_logs_error(
    monkeypatch, caplog
):
    """A REAL OpenRouterAdapter whose dedicated endpoint fails no longer swallows
    the error as ``[]`` — it propagates to the discovery layer, which logs it
    loudly for a route that explicitly claims embedding support (#2433).

    The mock-adapter tests exercise the discovery layer's ERROR path; this one
    proves the real adapter actually surfaces the exception to reach it.
    """
    import logging

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    adapter = OpenRouterAdapter()

    class _FailingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, **kw):
            raise httpx.ConnectError("connection refused")

    # The real adapter must raise, not return [].
    with patch(
        "kestrel_sovereign.llm.openrouter_adapter.httpx.AsyncClient", _FailingClient
    ):
        with pytest.raises(httpx.HTTPError):
            await adapter.list_embedding_models()

    # And through the discovery layer, a CLAIMING route logs an ERROR.
    provider = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": adapter,
        "client": None,
        "embedding_model": "qwen/qwen3-embedding-8b",  # explicit operator claim
        "capabilities": {},
    }
    svc = _FakeService([provider])
    with patch(
        "kestrel_sovereign.llm.openrouter_adapter.httpx.AsyncClient", _FailingClient
    ), caplog.at_level(logging.INFO):
        await svc.discover_embedding_models()

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "a claiming route's real-adapter failure must log an ERROR"
    assert any("claims" in r.message for r in errors)


async def test_openrouter_real_adapter_failure_non_claiming_is_info(monkeypatch, caplog):
    """The same real-adapter failure on a NON-claiming route is a quiet INFO,
    never an ERROR traceback (#2433)."""
    import logging

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    adapter = OpenRouterAdapter()

    class _FailingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, **kw):
            raise httpx.ConnectError("connection refused")

    provider = {
        "vendor": "openrouter",
        "name": "openrouter:api",
        "adapter": adapter,
        "client": None,
        "capabilities": {},  # no claim
    }
    svc = _FakeService([provider])
    with patch(
        "kestrel_sovereign.llm.openrouter_adapter.httpx.AsyncClient", _FailingClient
    ), caplog.at_level(logging.INFO):
        await svc.discover_embedding_models()

    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


async def test_embedding_derivation_is_route_scoped_no_sibling_leak():
    """Two same-vendor OpenAI-compatible routes: only the route whose OWN chat
    listing carries an embedding id advertises embeddings. The reused listing is
    keyed by ROUTE, so one route's discovered embedding id never retags onto a
    chat-only sibling under the same vendor (#2433)."""
    api = {
        "vendor": "openai",
        "name": "openai:api",
        "adapter": OpenAIAdapter(),  # derives_embeddings_from_chat_listing
        "client": SimpleNamespace(models=SimpleNamespace(list=AsyncMock())),
        "capabilities": {},
    }
    alt = {
        "vendor": "openai",
        "name": "openai:alt",
        "adapter": OpenAIAdapter(),
        "client": SimpleNamespace(models=SimpleNamespace(list=AsyncMock())),
        "capabilities": {},
    }
    svc = _FakeService([api, alt])
    # Each route's OWN listing: only openai:api serves an embedding id.
    svc._chat_models_by_route = {
        "openai:api": ["gpt-5.5", "text-embedding-3-small"],
        "openai:alt": ["gpt-4o"],
    }

    models = await svc.discover_embedding_models()
    # The embedding model is attributed ONLY to the route that listed it.
    assert {(m.route, m.id) for m in models} == {
        ("openai:api", "text-embedding-3-small")
    }
    assert await svc.route_advertises_embeddings(api) is True
    assert await svc.route_advertises_embeddings(alt) is False
    # Neither route issued a second /v1/models request — both reused their snapshot.
    api["client"].models.list.assert_not_awaited()
    alt["client"].models.list.assert_not_awaited()


async def test_sibling_route_without_snapshot_fetches_own_listing():
    """A sibling route absent from the route-keyed snapshot passes ``None`` so
    its adapter fetches its OWN /v1/models — never reusing another route's
    listing (#2433). This is the case where chat discovery ran one route per
    vendor and a same-vendor sibling was not the chosen discovery route."""
    api = {
        "vendor": "openai",
        "name": "openai:api",
        "adapter": OpenAIAdapter(),
        "client": SimpleNamespace(models=SimpleNamespace(list=AsyncMock())),
        "capabilities": {},
    }
    alt_client = SimpleNamespace(
        models=SimpleNamespace(
            list=AsyncMock(
                return_value=SimpleNamespace(
                    data=[SimpleNamespace(id="text-embedding-3-large")]
                )
            )
        )
    )
    alt = {
        "vendor": "openai",
        "name": "openai:alt",
        "adapter": OpenAIAdapter(),
        "client": alt_client,
        "capabilities": {},
    }
    svc = _FakeService([api, alt])
    # Only the discovery route is snapshotted; the sibling has NO entry.
    svc._chat_models_by_route = {"openai:api": ["gpt-5.5", "text-embedding-3-small"]}

    models = await svc.discover_embedding_models()
    by_route = {(m.route, m.id) for m in models}
    # api reused its snapshot; alt fetched its OWN listing (None → fetch).
    assert ("openai:api", "text-embedding-3-small") in by_route
    assert ("openai:alt", "text-embedding-3-large") in by_route
    alt_client.models.list.assert_awaited_once()
    api["client"].models.list.assert_not_awaited()


async def test_snapshot_records_empty_listing_for_chat_only_route():
    """A discovery route whose chat listing came back EMPTY (RunPod's 404) is
    snapshotted as an explicit ``[]`` — a known-empty listing that stops the
    embedding re-probe, NOT an absent entry that would trigger a fetch (#2433)."""
    provider = {
        "vendor": "runpod",
        "name": "runpod:vllm",
        "adapter": OpenAIAdapter(),
        "client": SimpleNamespace(models=SimpleNamespace(list=AsyncMock())),
        "capabilities": {},
    }
    svc = _FakeService([provider])
    # Chat discovery returned NO models for this route (endpoint 404'd).
    svc._snapshot_chat_models_by_route([])
    assert svc._chat_models_by_route == {"runpod:vllm": []}

    models = await svc.discover_embedding_models()
    assert models == []
    # The explicit empty snapshot ([], not None) means no second /v1/models call.
    provider["client"].models.list.assert_not_awaited()


async def test_cache_hit_rebuilds_route_snapshot_from_cached_catalog():
    """On a shared-cache HIT the route-keyed snapshot is rebuilt from the cached
    catalog before reconciliation, so embedding discovery id-filters fresh chat
    ids instead of a stale/absent snapshot and never re-fetches (#2433)."""
    from kestrel_sovereign.llm.model_metadata import ModelInfo, ModelCategory

    provider = {
        "vendor": "openai",
        "name": "openai:api",
        "adapter": OpenAIAdapter(),
        "client": SimpleNamespace(models=SimpleNamespace(list=AsyncMock())),
        "capabilities": {},
    }
    svc = _FakeService([provider])
    # A stale snapshot from a prior cycle carried no embedding id.
    svc._chat_models_by_route = {"openai:api": ["old-chat-only"]}

    # The cached catalog now includes an embedding id for this route's vendor.
    cached = [
        ModelInfo(
            id="gpt-5.5",
            provider="openai",
            display_name="gpt-5.5",
            category=ModelCategory.CHAT,
        ),
        ModelInfo(
            id="text-embedding-3-small",
            provider="openai",
            display_name="text-embedding-3-small",
            category=ModelCategory.EMBEDDING,
        ),
    ]
    # Mirrors what discover_all_models._return_cached does before reconciling.
    svc._snapshot_chat_models_by_route(cached)
    assert svc._chat_models_by_route == {
        "openai:api": ["gpt-5.5", "text-embedding-3-small"]
    }

    models = await svc.discover_embedding_models()
    assert {(m.route, m.id) for m in models} == {
        ("openai:api", "text-embedding-3-small")
    }
    provider["client"].models.list.assert_not_awaited()


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
