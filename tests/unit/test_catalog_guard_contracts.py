"""Contract tests for the model-in-catalog guard.

The guard in ``LLMService._try_single_provider`` refuses to call a provider
with a ``target_model`` that isn't in that provider's vendor catalog. Kills
three bugs at once:
  * llama.cpp silent-override (llama-server serves whatever weights are loaded,
    ignoring the requested model ID — so callers get lied to),
  * cheap-model cascade (one bogus model ID broadcast onto every provider),
  * cross-vendor mistargeting (gpt-* routed to anthropic, etc).

Skipping raises ``ModelNotAvailableForRoute`` which the outer loop treats as
a silent skip (no HTTP call made) rather than an error.
"""

from unittest.mock import patch
from types import SimpleNamespace

import pytest

from kestrel_sovereign.llm.model_cache import SharedModelCache
from kestrel_sovereign.llm.model_metadata import ModelInfo, ModelCategory


def _model(id_: str, vendor: str) -> ModelInfo:
    return ModelInfo(
        id=id_,
        provider=vendor,
        display_name=id_,
        category=ModelCategory.CHAT,
        supports_tools=True,
    )


def _guard_subject(LLMService, *, discovery_failures=None):
    """Build the guard's subject with every attribute it actually reads.

    There were two places constructing this double, and only one was updated
    when `_model_available_for_route` gained a dependency — so the fixture
    tests passed while the inline one failed with an AttributeError. One
    builder, so the next dependency cannot be added to only half of them.
    """
    svc = SimpleNamespace()
    svc._discovery_failures = dict(discovery_failures or {})
    svc._discovery_failures_observed = set()
    svc._effective_discovery_failures = (
        LLMService._effective_discovery_failures.__get__(svc)
    )
    svc._vendor_catalog_is_authoritative = (
        LLMService._vendor_catalog_is_authoritative.__get__(svc)
    )
    svc._model_available_for_route = (
        LLMService._model_available_for_route.__get__(svc)
    )
    return svc


@pytest.fixture
def svc_with_catalog():
    """Build a minimal service-like object with the guard helper wired up."""
    from kestrel_sovereign.llm.service import LLMService

    # Populate the shared cache with a controlled catalog so the guard has
    # something to validate against. We don't need a full LLMService — the
    # helper reads `provider.vendor`, `provider.model`, the cache, and (since
    # #3190) `_vendor_catalog_is_authoritative`, which decides whether this
    # vendor's catalog may be used to DISPROVE a model at all. That predicate
    # is bound here rather than stubbed True: stubbing it would assert the
    # thing under test, and leaving it off makes these tests fail with an
    # AttributeError instead of exercising the guard.
    cache_contents = [
        _model("gpt-5", "openai"),
        _model("gpt-5-mini", "openai"),
        _model("claude-sonnet-4-6", "anthropic"),
        _model("claude-opus-4-7", "anthropic"),
    ]

    fake_cache = SharedModelCache()
    fake_cache.set(cache_contents)

    with patch(
        "kestrel_sovereign.llm.model_cache.get_shared_model_cache",
        return_value=fake_cache,
    ):
        # No known discovery failures: the catalog above is trustworthy, which
        # is the precondition every assertion in this module depends on.
        yield _guard_subject(LLMService)


def test_guard_accepts_model_in_vendor_catalog(svc_with_catalog):
    """A model that IS in the vendor's catalog passes the guard."""
    route = {"vendor": "openai", "route": "api", "model": "auto"}
    assert svc_with_catalog._model_available_for_route(route, "gpt-5") is True


def test_guard_rejects_cross_vendor_model(svc_with_catalog):
    """An OpenAI model routed to anthropic must be rejected — this is the
    cross-vendor broadcast bug."""
    route = {"vendor": "anthropic", "route": "api", "model": "auto"}
    assert svc_with_catalog._model_available_for_route(route, "gpt-5") is False


def test_guard_rejects_fake_model_on_local_route(svc_with_catalog):
    """llama.cpp silent-override guard: sending ``gpt-5-mini`` to a
    llama.cpp route whose configured model is a Kimi GGUF must be rejected,
    not silently served. Without this guard, llama-server returns Kimi
    output while logs claim gpt-5-mini was used."""
    route = {
        "vendor": "llama_cpp",
        "route": "local",
        "model": "Kimi-K2.5-UD-Q2_K_XL-00001-of-00008.gguf",
        "is_local": True,
    }
    assert svc_with_catalog._model_available_for_route(route, "gpt-5-mini") is False


def test_guard_accepts_routes_own_configured_model(svc_with_catalog):
    """Even if the configured local-model isn't yet in the discovery cache,
    the route's OWN configured model is always considered available (so
    cold-start calls to a route's default don't fail the guard before
    discovery has completed)."""
    route = {
        "vendor": "llama_cpp",
        "route": "local",
        "model": "some-model-not-in-cache.gguf",
        "is_local": True,
    }
    assert svc_with_catalog._model_available_for_route(
        route, "some-model-not-in-cache.gguf"
    ) is True


def test_guard_permits_when_cache_empty():
    """Cold-start: if discovery hasn't populated the cache yet, permit the
    call. The guard only blocks *known* mismatches, not unknown state."""
    from kestrel_sovereign.llm.service import LLMService

    empty_cache = SharedModelCache()
    with patch(
        "kestrel_sovereign.llm.model_cache.get_shared_model_cache",
        return_value=empty_cache,
    ):
        svc = _guard_subject(LLMService)
        route = {"vendor": "openai", "route": "api", "model": "auto"}
        assert svc._model_available_for_route(route, "gpt-5-mini") is True


def test_model_not_available_for_route_exception_shape():
    """The skip-signal carries vendor, route, and model for loggability."""
    from kestrel_sovereign.llm.service import ModelNotAvailableForRoute

    exc = ModelNotAvailableForRoute(vendor="anthropic", route="plan", model="gpt-5")
    assert exc.vendor == "anthropic"
    assert exc.route == "plan"
    assert exc.model == "gpt-5"
    assert "gpt-5" in str(exc)
    assert "anthropic:plan" in str(exc)
