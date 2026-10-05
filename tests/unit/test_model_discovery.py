"""``discover_all_models`` over fixture vendors: no network, no credentials (#3270).

The previous version of these tests built ``LLMService()`` from the host's live
configuration and asserted on whatever every configured vendor answered that
day. On a host with a 404ing RunPod endpoint and no AWS credentials it failed
on every branch, so it measured the host rather than the change.

These drive the real ``discover_all_models`` over one fixture route per
discovery path. Every HTTP request is answered by an ``httpx.MockTransport``,
including the real ``openai`` SDK client on the canonical OpenAI route, and
plugin adapters are in-process fakes. Nothing reads a host key.

Reproducing #3270 on a live host also showed the real cause of its
``Not all items are ModelInfo``. It was not the 404 or the missing credentials:
both were already excluded and recorded as discovery failures. The 81 offending
items came from the external xAI, DeepSeek, Kimi, Vertex and Bedrock plugins,
which build the SDK's base ``ModelInfo`` rather than the framework subclass.
``_as_framework_models`` now lifts them at the adapter boundary, and the
``xai`` fixture below is that shape.

The opt-in live variant stays in ``tests/test_model_discovery.py``.
"""

from __future__ import annotations

import logging

import httpx
import openai
import pytest
import pytest_asyncio

from kestrel_sdk.llm import ModelInfo as SDKModelInfo
from kestrel_sovereign.agent import token_counter
from kestrel_sovereign.llm import model_catalog
from kestrel_sovereign.llm.model_cache import get_shared_model_cache
from kestrel_sovereign.llm.model_catalog import ModelCatalogService
from kestrel_sovereign.llm.model_discovery import _as_framework_models
from kestrel_sovereign.llm.model_metadata import ModelCategory, ModelInfo
from kestrel_sovereign.llm.openai_adapter import OpenAIAdapter
from tests.utils.process_local_llm_service import process_local_service

DISCOVERY_LOGGER = "kestrel_sovereign.llm.model_discovery"

OPENAI_BASE = "https://api.openai.com/v1"
GROQ_BASE = "https://api.groq.fixture/openai/v1"
RUNPOD_BASE = "https://api.runpod.fixture/v2/vllm-fixture/openai/v1"
LLAMA_CPP_ROOT = "http://127.0.0.1:18080"
LLAMA_CPP_BASE = f"{LLAMA_CPP_ROOT}/v1"

PINNED_KIMI_MODEL = "kimi-fixture-pinned"
HIDDEN_OPENAI_MODEL = "gpt-fixture-retired"

#: The vendors whose discovery succeeds. ``runpod`` (404) and ``bedrock`` (no
#: credentials) must never contribute a model.
DISCOVERED_VENDORS = {"openai", "xai", "groq", "llama_cpp", "kimi"}


class NoCredentialsError(Exception):
    """What botocore raises from a Bedrock call on a host with no AWS keys."""

    def __str__(self) -> str:
        return "Unable to locate credentials"


class _FixtureHTTP:
    """Answers every HTTP request discovery makes; records any it did not expect."""

    def __init__(self) -> None:
        self.requests: list[str] = []
        self.unexpected: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        if url == f"{OPENAI_BASE}/models":
            assert request.headers["authorization"] == "Bearer fixture-openai-key"
            return httpx.Response(200, json={"object": "list", "data": [
                _openai_model("gpt-fixture-5", 1767225600),
                _openai_model("gpt-fixture-5-mini", 1767225600),
                _openai_model("text-embedding-3-small", 1705948997),
                _openai_model(HIDDEN_OPENAI_MODEL, 1700000000),
            ]})
        if url == f"{GROQ_BASE}/models":
            assert request.headers["authorization"] == "Bearer fixture-groq-key"
            return httpx.Response(200, json={"object": "list", "data": [
                {"id": "llama-fixture-70b", "created": 1767225600,
                 "context_window": 131072},
            ]})
        if url == f"{RUNPOD_BASE}/models":
            return httpx.Response(404, json={"error": "Not Found"})
        if url == f"{LLAMA_CPP_BASE}/models":
            return httpx.Response(200, json={"data": [{"id": "qwen3-fixture.gguf"}]})
        if url == f"{LLAMA_CPP_ROOT}/props":
            return httpx.Response(200, json={"n_ctx": 32768})
        self.unexpected.append(url)
        return httpx.Response(599)


def _openai_model(model_id: str, created: int) -> dict:
    return {"id": model_id, "object": "model", "created": created, "owned_by": "fixture"}


class _PluginAdapter:
    """An external LLM plugin: it depends only on the SDK, as published plugins do."""

    def __init__(self, *, models=None, error: Exception | None = None) -> None:
        self._models = models
        self._error = error

    async def list_models(self, client=None):
        if self._error is not None:
            raise self._error
        return self._models


class _CataloglessAdapter:
    """A plugin that publishes no catalog; its pinned model is still offered."""

    async def list_models(self, client=None):
        raise NotImplementedError


#: The record shape the xAI plugin returned on the host that filed #3270.
XAI_SDK_MODELS = [
    SDKModelInfo(
        id="grok-fixture-4",
        provider="xai",
        display_name="Grok Fixture 4",
        category=ModelCategory.CHAT,
        description="fixture reasoning model",
        created_at="1767225600",
        supports_vision=True,
        supports_tools=True,
        supports_streaming=True,
        context_limit=256000,
    ),
    SDKModelInfo(
        id="grok-fixture-4-mini",
        provider="xai",
        display_name="Grok Fixture 4 Mini",
        created_at="1767225600",
        supports_tools=True,
        supports_streaming=True,
        context_limit=131072,
    ),
]


@pytest.fixture
def fixture_http(monkeypatch) -> _FixtureHTTP:
    """Route every ``httpx.AsyncClient`` built during the test to the fixtures.

    A subclass rather than a factory function, so the ``openai`` SDK's
    ``isinstance(http_client, httpx.AsyncClient)`` check still holds.
    """
    http = _FixtureHTTP()

    class _OfflineAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(http.handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _OfflineAsyncClient)
    return http


@pytest.fixture
def fixture_catalog(tmp_path, monkeypatch) -> ModelCatalogService:
    """A catalog read from a fixture file, whose disk cache lands in ``tmp_path``.

    The default catalog writes ``model_discovery_cache.json`` beside the
    ``kestrel_sovereign`` package, i.e. into the source tree.
    """
    catalog_file = tmp_path / "model_catalog.toml"
    catalog_file.write_text(
        f'[hidden]\nopenai = ["{HIDDEN_OPENAI_MODEL}"]\n', encoding="utf-8"
    )
    catalog = ModelCatalogService(
        config_path=catalog_file,
        cache_path=tmp_path / "model_discovery_cache.json",
    )
    monkeypatch.setattr(model_catalog, "_catalog_service", catalog)
    # Discovery registers every context limit it sees in a process-wide map.
    monkeypatch.setattr(token_counter, "_discovered_context_limits", {})
    return catalog


@pytest_asyncio.fixture
async def discovery(fixture_http, fixture_catalog, monkeypatch):
    """A real ``LLMService`` with one fixture route per discovery path."""
    monkeypatch.setenv("FIXTURE_GROQ_API_KEY", "fixture-groq-key")
    monkeypatch.setenv("FIXTURE_RUNPOD_API_KEY", "fixture-runpod-key")

    openai_client = openai.AsyncOpenAI(
        api_key="fixture-openai-key",
        base_url=OPENAI_BASE,
        http_client=httpx.AsyncClient(),
        max_retries=0,
    )
    service = process_local_service([
        # Canonical OpenAI: the in-tree adapter over the real SDK client.
        _route("openai", OpenAIAdapter("openai"), client=openai_client),
        # An external plugin returning SDK records.
        _route("xai", _PluginAdapter(models=list(XAI_SDK_MODELS))),
        # Remote OpenAI-compatible endpoints, one answering and one 404ing.
        _route(
            "groq", OpenAIAdapter("groq"),
            base_url=GROQ_BASE, api_key_env="FIXTURE_GROQ_API_KEY",
        ),
        _route(
            "runpod", OpenAIAdapter("runpod"),
            base_url=RUNPOD_BASE, api_key_env="FIXTURE_RUNPOD_API_KEY",
        ),
        # A local OpenAI-compatible server.
        _route(
            "llama_cpp", OpenAIAdapter("llama_cpp"), route="local",
            base_url=LLAMA_CPP_BASE, is_local=True, is_cloud=False,
        ),
        # A plugin whose SDK finds no credentials.
        _route("bedrock", _PluginAdapter(error=NoCredentialsError())),
        # A plugin without a catalog, pinned to a model in config.
        _route("kimi", _CataloglessAdapter(), model=PINNED_KIMI_MODEL),
    ])
    service._discovery_failures = {}
    try:
        yield service
    finally:
        await openai_client.close()


def _route(vendor, adapter, *, route="api", client=None, model="auto", **config):
    return {
        "name": f"{vendor}:{route}",
        "vendor": vendor,
        "route": route,
        "model": model,
        "adapter": adapter,
        "client": client if client is not None else object(),
        "is_cloud": True,
        "selection_hints": [],
        **config,
    }


def _by_id(models) -> dict[str, ModelInfo]:
    return {model.id: model for model in models}


def _discovery_warnings(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == DISCOVERY_LOGGER and record.levelno == logging.WARNING
    ]


class TestDiscoveryReturnsOnlyModelInfo:
    @pytest.mark.asyncio
    async def test_discover_all_models_returns_model_info(self, discovery, fixture_http):
        models = await discovery.discover_all_models(use_cache=False)

        assert models, "No models discovered"
        not_framework = [
            (type(m).__module__, m.provider, m.id)
            for m in models
            if not isinstance(m, ModelInfo)
        ]
        assert not_framework == []
        assert fixture_http.unexpected == []

    @pytest.mark.asyncio
    async def test_plugin_sdk_records_are_lifted_with_every_field(self, discovery):
        models = _by_id(await discovery.discover_all_models(use_cache=False))

        grok = models["grok-fixture-4"]
        assert type(grok) is ModelInfo
        source = XAI_SDK_MODELS[0]
        assert (
            grok.provider, grok.display_name, grok.description, grok.created_at,
            grok.supports_vision, grok.supports_tools, grok.supports_streaming,
            grok.context_limit,
        ) == (
            source.provider, source.display_name, source.description,
            source.created_at, source.supports_vision, source.supports_tools,
            source.supports_streaming, source.context_limit,
        )
        assert grok.underlying_provider is None
        assert grok.output_limit is None
        assert {"underlying_provider", "output_limit"} <= set(grok.to_dict())

    @pytest.mark.asyncio
    async def test_every_discovery_path_contributes(self, discovery):
        models = await discovery.discover_all_models(use_cache=False)

        assert {m.provider for m in models} == DISCOVERED_VENDORS
        by_id = _by_id(models)
        assert by_id["llama-fixture-70b"].context_limit == 131072
        assert by_id["qwen3-fixture.gguf"].context_limit == 32768
        assert by_id[PINNED_KIMI_MODEL].provider == "kimi"

    @pytest.mark.asyncio
    async def test_hidden_models_are_never_returned(self, discovery):
        models = await discovery.discover_all_models(use_cache=False)

        assert HIDDEN_OPENAI_MODEL not in _by_id(models)


class TestVendorFailureOutcomes:
    """A failing vendor is excluded, logged, and recorded against its name."""

    @pytest.mark.asyncio
    async def test_a_vendor_answering_404_is_excluded_and_recorded(
        self, discovery, caplog
    ):
        caplog.set_level(logging.WARNING, logger=DISCOVERY_LOGGER)

        models = await discovery.discover_all_models(use_cache=False)

        assert [m.id for m in models if m.provider == "runpod"] == []
        recorded = discovery._discovery_failures["runpod"]
        assert recorded.startswith("RuntimeError: remote model discovery failed")
        assert "404 Not Found" in recorded
        assert any(
            message.startswith("runpod: model discovery failed") and "404" in message
            for message in _discovery_warnings(caplog)
        )

    @pytest.mark.asyncio
    async def test_a_vendor_without_credentials_is_excluded_and_recorded(
        self, discovery, caplog
    ):
        caplog.set_level(logging.WARNING, logger=DISCOVERY_LOGGER)

        models = await discovery.discover_all_models(use_cache=False)

        assert [m.id for m in models if m.provider == "bedrock"] == []
        assert discovery._discovery_failures["bedrock"] == (
            "NoCredentialsError: Unable to locate credentials"
        )
        assert (
            "bedrock: model discovery failed: Unable to locate credentials"
            in _discovery_warnings(caplog)
        )

    @pytest.mark.asyncio
    async def test_only_the_failing_vendors_are_recorded(self, discovery):
        await discovery.discover_all_models(use_cache=False)

        assert set(discovery._discovery_failures) == {"runpod", "bedrock"}


class TestDiscoveryFilters:
    @pytest.mark.asyncio
    async def test_featured_filter(self, discovery):
        all_models = await discovery.discover_all_models(use_cache=False)
        featured = await discovery.discover_all_models(
            use_cache=False, featured_only=True
        )

        assert featured
        assert {m.id for m in featured} <= {m.id for m in all_models}
        assert all(m.is_featured for m in featured)
        # A model pinned in config is operator intent and stays featured.
        assert PINNED_KIMI_MODEL in _by_id(featured)

    @pytest.mark.asyncio
    async def test_category_filter(self, discovery):
        chat = await discovery.discover_all_models(
            use_cache=False, category=ModelCategory.CHAT
        )
        embedding = await discovery.discover_all_models(
            use_cache=False, category=ModelCategory.EMBEDDING
        )

        assert chat and all(m.category == ModelCategory.CHAT for m in chat)
        assert [m.id for m in embedding] == ["text-embedding-3-small"]

    @pytest.mark.asyncio
    async def test_provider_filter(self, discovery):
        filtered = await discovery.discover_all_models(
            use_cache=False, providers=["xai"]
        )

        assert {m.id for m in filtered} == {m.id for m in XAI_SDK_MODELS}


class TestDiscoveryCache:
    @pytest.mark.asyncio
    async def test_a_cached_call_makes_no_request(
        self, discovery, fixture_http, fixture_catalog
    ):
        # tests/conftest.py clears the process-wide cache around every test.
        assert get_shared_model_cache().get_any() is None

        first = await discovery.discover_all_models(use_cache=True)
        assert get_shared_model_cache().has_data()
        requests_after_discovery = len(fixture_http.requests)

        second = await discovery.discover_all_models(use_cache=True)

        assert [m.id for m in second] == [m.id for m in first]
        assert len(fixture_http.requests) == requests_after_discovery
        assert fixture_catalog.cache_path.exists()

    @pytest.mark.asyncio
    async def test_the_disk_cache_round_trips_to_the_same_types(
        self, discovery, fixture_catalog
    ):
        """Before #3270 a fresh discovery returned SDK records the cache did not."""
        fresh = await discovery.discover_all_models(use_cache=False)

        reloaded = fixture_catalog.load_cache()

        assert {(type(m), m.id) for m in reloaded} == {
            (type(m), m.id) for m in fresh
        } | {(ModelInfo, HIDDEN_OPENAI_MODEL)}


class TestModelInfoStructure:
    @pytest.mark.asyncio
    async def test_every_model_has_the_required_fields(self, discovery):
        models = await discovery.discover_all_models(use_cache=False)

        for model in models:
            assert model.id and model.provider and model.display_name
            assert isinstance(model.category, ModelCategory)

    @pytest.mark.asyncio
    async def test_every_model_serializes_the_framework_shape(self, discovery):
        models = await discovery.discover_all_models(use_cache=False)

        for model in models:
            data = model.to_dict()
            assert {
                "id", "provider", "display_name", "category",
                "underlying_provider", "output_limit",
            } <= set(data)


class TestLegacyCompatibility:
    def test_list_available_models_lists_configured_routes(self, discovery):
        models = discovery.list_available_models()

        assert [m["provider"] for m in models] == [
            route["name"] for route in discovery.providers
        ]


class TestAdapterBoundary:
    """``_as_framework_models``: what an adapter returns versus what enters the catalog."""

    def test_framework_records_pass_through_unchanged(self):
        model = ModelInfo(id="m", provider="openai", display_name="m")

        assert _as_framework_models("openai", [model])[0] is model

    def test_an_sdk_record_is_lifted(self):
        lifted = _as_framework_models("xai", XAI_SDK_MODELS)

        assert [type(m) for m in lifted] == [ModelInfo, ModelInfo]
        assert [m.id for m in lifted] == [m.id for m in XAI_SDK_MODELS]

    def test_a_non_model_item_is_dropped_with_a_warning(self, caplog):
        caplog.set_level(logging.WARNING, logger=DISCOVERY_LOGGER)
        valid = ModelInfo(id="m", provider="xai", display_name="m")

        kept = _as_framework_models("xai", [valid, {"id": "raw"}, "grok-raw"])

        assert kept == [valid]
        assert _discovery_warnings(caplog) == [
            "xai: model discovery dropped 2 item(s) that are not ModelInfo: dict, str"
        ]

    def test_a_tuple_is_returned_as_a_list(self):
        model = ModelInfo(id="m", provider="xai", display_name="m")

        assert _as_framework_models("xai", (model,)) == [model]

    def test_a_result_that_is_not_a_list_raises(self):
        with pytest.raises(TypeError, match="returned NoneType"):
            _as_framework_models("xai", None)

    @pytest.mark.asyncio
    async def test_a_non_list_result_is_recorded_as_that_vendors_failure(self):
        service = process_local_service([])
        service._discovery_failures = {}

        models = await service._safe_list_models(
            "xai", _PluginAdapter(models={"grok": "not a list"}), object()
        )

        assert models == []
        assert service._discovery_failures["xai"].startswith(
            "TypeError: list_models returned dict"
        )

    @pytest.mark.asyncio
    async def test_a_partly_malformed_catalog_is_not_a_vendor_failure(self):
        """The well-formed records still came back, so discovery succeeded."""
        service = process_local_service([])
        service._discovery_failures = {"xai": "RuntimeError: earlier outage"}

        models = await service._safe_list_models(
            "xai", _PluginAdapter(models=[*XAI_SDK_MODELS, None]), object()
        )

        assert [m.id for m in models] == [m.id for m in XAI_SDK_MODELS]
        assert service._discovery_failures == {}
