"""Contract tests for mandate vendor auto-resolution.

These guard against the *broadcast cascade* bug: if an agent calls
``set_model_preference`` with just a bare model id (no vendor), the mandate
used to persist as ``{vendor: None, model: "<id>", route: None}``. The next
request would then attempt that bare id across every configured provider in
priority order — anthropic → openai → openrouter — eventually landing on
whichever backend happened to serve something with a matching id (in one
observed case, OpenRouter serving a Gemini model for ``gpt-5-mini``). The
mandate must name a vendor; if the caller omitted one, we resolve it from
the discovery catalog or refuse.

Regression for issue observed in host.log at
2026-04-23T14:58+ (see epic #688).
"""

from unittest.mock import MagicMock, patch

import pytest

from kestrel_sovereign.llm.model_metadata import ModelCategory, ModelInfo
from kestrel_sovereign.llm.service import LLMService


def _cached(models):
    """Return a mock SharedModelCache whose ``get_any()`` yields ``models``."""
    cache = MagicMock()
    cache.get_any = MagicMock(return_value=models)
    return cache


def _make_service():
    """Build a bare LLMService instance wired up enough for set/get preference."""
    svc = LLMService.__new__(LLMService)
    svc.providers = []
    svc._mandate_preference = {"vendor": None, "model": None, "route": None}
    svc._preference_persistence_callback = None
    svc._preference_persistence_tasks = set()
    return svc


def _mk_model(id_, vendor):
    return ModelInfo(
        id=id_,
        provider=vendor,
        display_name=id_,
        category=ModelCategory.CHAT,
        supports_tools=True,
    )


class TestBareModelResolution:
    """set_model_preference(model, vendor=None) auto-resolves from discovery."""

    def test_unambiguous_bare_model_auto_resolves_vendor(self):
        svc = _make_service()
        cache = _cached([_mk_model("claude-opus-4-7", "anthropic")])
        with patch("kestrel_sovereign.llm.model_cache.get_shared_model_cache", return_value=cache):
            svc.set_model_preference("claude-opus-4-7")

        pref = svc.get_model_preference()
        assert pref["vendor"] == "anthropic"
        assert pref["model"] == "claude-opus-4-7"
        assert pref["route"] is None

    def test_ambiguous_bare_model_refuses_with_vendor_list(self):
        """A model id served by >1 vendor (e.g. openrouter mirrors) must refuse."""
        svc = _make_service()
        cache = _cached([
            _mk_model("gpt-5-mini", "openai"),
            _mk_model("gpt-5-mini", "openrouter"),
        ])
        with patch("kestrel_sovereign.llm.model_cache.get_shared_model_cache", return_value=cache):
            with pytest.raises(ValueError) as exc:
                svc.set_model_preference("gpt-5-mini")

        # Error must name the ambiguous vendors so the caller can choose.
        msg = str(exc.value)
        assert "gpt-5-mini" in msg
        assert "openai" in msg
        assert "openrouter" in msg

        # Mandate was NOT persisted (the key invariant — no broadcast state
        # gets left behind on a refusal).
        assert svc.get_model_preference() == {"vendor": None, "model": None, "route": None}

    def test_unknown_bare_model_refuses(self):
        """A model not in discovery must refuse — this was the LLM-hallucinated-id path."""
        svc = _make_service()
        cache = _cached([_mk_model("claude-opus-4-7", "anthropic")])
        with patch("kestrel_sovereign.llm.model_cache.get_shared_model_cache", return_value=cache):
            with pytest.raises(ValueError) as exc:
                # Hypothetical: LLM tool call with a hallucinated id.
                svc.set_model_preference("gpt-9000-ultramax")

        assert "gpt-9000-ultramax" in str(exc.value)
        assert svc.get_model_preference() == {"vendor": None, "model": None, "route": None}

    def test_empty_catalog_refuses_bare_model(self):
        """Pre-discovery / empty cache: refuse rather than silently broadcast."""
        svc = _make_service()
        cache = _cached(None)
        with patch("kestrel_sovereign.llm.model_cache.get_shared_model_cache", return_value=cache):
            with pytest.raises(ValueError):
                svc.set_model_preference("claude-opus-4-7")

        assert svc.get_model_preference() == {"vendor": None, "model": None, "route": None}

    def test_explicit_vendor_bypasses_resolution(self):
        """When caller supplies a vendor, we trust it — no catalog lookup."""
        svc = _make_service()
        cache = _cached([])  # Empty catalog would refuse a bare call.
        with patch("kestrel_sovereign.llm.model_cache.get_shared_model_cache", return_value=cache):
            svc.set_model_preference("claude-opus-4-7", vendor="anthropic", route="plan")

        pref = svc.get_model_preference()
        assert pref == {
            "vendor": "anthropic",
            "model": "claude-opus-4-7",
            "route": "plan",
        }

    def test_auto_string_still_ignored(self):
        """``auto`` short-circuits before resolution (means default routing)."""
        svc = _make_service()
        cache = _cached([])
        with patch("kestrel_sovereign.llm.model_cache.get_shared_model_cache", return_value=cache):
            svc.set_model_preference("auto")  # no-op, no exception

        assert svc.get_model_preference() == {"vendor": None, "model": None, "route": None}


class TestBroadcastBugRegression:
    """The exact path that produced the "answered as Gemini" bug in host.log."""

    def test_llm_tool_call_with_bare_model_refuses_rather_than_broadcast(self):
        """Replay the LLM tool-call sequence that caused the live failure.

        Host log sequence (2026-04-23T14:58):
          1. mandate = anthropic:plan/claude-opus-4-20250514  (correct)
          2. agent streams; the response includes a tool call to set_model
             with a bare model id
          3. mandate got silently rewritten to {vendor: None, model:
             "gpt-5-mini"}
          4. next request broadcasts: anthropic:plan → openai:plan →
             openrouter:api — OpenRouter served it, agent reported "gemini"

        After this fix, step 3 raises ValueError and step 2 never succeeds in
        overwriting the mandate. The mandate stays anthropic:plan/claude-opus.
        """
        svc = _make_service()
        svc._mandate_preference = {
            "vendor": "anthropic",
            "model": "claude-opus-4-20250514",
            "route": "plan",
        }
        cache = _cached([_mk_model("claude-opus-4-20250514", "anthropic")])
        with patch("kestrel_sovereign.llm.model_cache.get_shared_model_cache", return_value=cache):
            with pytest.raises(ValueError):
                svc.set_model_preference("gpt-5-mini")

        # Mandate is untouched.
        assert svc.get_model_preference() == {
            "vendor": "anthropic",
            "model": "claude-opus-4-20250514",
            "route": "plan",
        }


def _route(name, vendor, route, model="auto"):
    """A minimal configured-route dict (matches LLMService.providers shape)."""
    return {"name": name, "vendor": vendor, "route": route, "model": model}


class TestExplicitMandateValidation:
    """set_model_preference(model, vendor=..., route=...) must validate the triple.

    Symmetric guard for the explicit-vendor path. The vendor-LESS path already
    refuses unknown models; previously the explicit path persisted ANY triple
    with no catalog/route check, so a hallucinated ``{vendor, route, model}``
    landed a broken mandate that only surfaced on the next request — the same
    silent route-fidelity skew as #1927 (surfaced by the #1925 sweep, #1946).
    """

    def test_unknown_route_refuses(self):
        """An explicit vendor:route that isn't a configured route must refuse."""
        svc = _make_service()
        svc.providers = [_route("openai:api", "openai", "api")]
        cache = _cached([_mk_model("gpt-5-mini", "openai")])
        with patch("kestrel_sovereign.llm.model_cache.get_shared_model_cache", return_value=cache):
            with pytest.raises(ValueError) as exc:
                svc.set_model_preference("gpt-5-mini", vendor="openai", route="plan")
        assert "openai:plan" in str(exc.value)
        # Broken mandate must NOT land.
        assert svc.get_model_preference() == {"vendor": None, "model": None, "route": None}

    def test_unknown_vendor_refuses(self):
        """An explicit vendor with no configured route must refuse."""
        svc = _make_service()
        svc.providers = [_route("openai:api", "openai", "api")]
        cache = _cached([_mk_model("gpt-5-mini", "openai")])
        with patch("kestrel_sovereign.llm.model_cache.get_shared_model_cache", return_value=cache):
            with pytest.raises(ValueError):
                svc.set_model_preference("some-model", vendor="madeupvendor")
        assert svc.get_model_preference() == {"vendor": None, "model": None, "route": None}

    def test_unknown_model_on_known_route_refuses(self):
        """A real vendor/route but a model the vendor doesn't serve must refuse."""
        svc = _make_service()
        svc.providers = [_route("openai:api", "openai", "api")]
        cache = _cached([_mk_model("gpt-5-mini", "openai")])
        with patch("kestrel_sovereign.llm.model_cache.get_shared_model_cache", return_value=cache):
            with pytest.raises(ValueError) as exc:
                svc.set_model_preference("gpt-5-hallucinated", vendor="openai", route="api")
        # Helpful message lists what IS available + points at list_models.
        msg = str(exc.value)
        assert "gpt-5-mini" in msg
        assert "list_models" in msg
        assert svc.get_model_preference() == {"vendor": None, "model": None, "route": None}

    def test_valid_explicit_triple_succeeds(self):
        """A real vendor/route/model in discovery persists as-is."""
        svc = _make_service()
        svc.providers = [_route("openai:api", "openai", "api")]
        cache = _cached([_mk_model("gpt-5-mini", "openai")])
        with patch("kestrel_sovereign.llm.model_cache.get_shared_model_cache", return_value=cache):
            svc.set_model_preference("gpt-5-mini", vendor="openai", route="api")
        assert svc.get_model_preference() == {
            "vendor": "openai", "model": "gpt-5-mini", "route": "api",
        }

    def test_valid_vendor_only_triple_succeeds(self):
        """Vendor without route resolves against any route for that vendor."""
        svc = _make_service()
        svc.providers = [_route("openai:api", "openai", "api")]
        cache = _cached([_mk_model("gpt-5-mini", "openai")])
        with patch("kestrel_sovereign.llm.model_cache.get_shared_model_cache", return_value=cache):
            svc.set_model_preference("gpt-5-mini", vendor="openai")
        assert svc.get_model_preference() == {
            "vendor": "openai", "model": "gpt-5-mini", "route": None,
        }

    def test_route_configured_default_model_always_serveable(self):
        """The route's own configured model passes even if not in the catalog."""
        svc = _make_service()
        svc.providers = [_route("openai:api", "openai", "api", model="gpt-5-configured")]
        cache = _cached([_mk_model("gpt-5-mini", "openai")])  # configured model absent
        with patch("kestrel_sovereign.llm.model_cache.get_shared_model_cache", return_value=cache):
            svc.set_model_preference("gpt-5-configured", vendor="openai", route="api")
        assert svc.get_model_preference()["model"] == "gpt-5-configured"

    def test_route_scoped_catalog_enforced(self):
        """A route with its OWN catalog (e.g. codex/openai:plan) must serve from it."""
        svc = _make_service()
        svc.providers = [_route("openai:plan", "openai", "plan")]
        # Route-scoped catalog serves only gpt-5-codex; the broader vendor
        # catalog has gpt-5-pro, which the plan route must NOT accept.
        svc._route_catalogs = {"openai:plan": [_mk_model("gpt-5-codex", "openai")]}
        cache = _cached([
            _mk_model("gpt-5-codex", "openai"),
            _mk_model("gpt-5-pro", "openai"),
        ])
        with patch("kestrel_sovereign.llm.model_cache.get_shared_model_cache", return_value=cache):
            # In-catalog model succeeds.
            svc.set_model_preference("gpt-5-codex", vendor="openai", route="plan")
            assert svc.get_model_preference()["model"] == "gpt-5-codex"
            # Vendor-catalog-only model is rejected on the route-scoped route.
            with pytest.raises(ValueError):
                svc.set_model_preference("gpt-5-pro", vendor="openai", route="plan")

    def test_empty_route_catalog_permits_coldstart(self):
        """An empty (unbuilt) route-scoped catalog is 'unknown' → permit."""
        svc = _make_service()
        svc.providers = [_route("openai:plan", "openai", "plan")]
        svc._route_catalogs = {"openai:plan": []}  # not yet built
        cache = _cached([_mk_model("gpt-5-codex", "openai")])
        with patch("kestrel_sovereign.llm.model_cache.get_shared_model_cache", return_value=cache):
            svc.set_model_preference("gpt-5-anything", vendor="openai", route="plan")
        assert svc.get_model_preference()["model"] == "gpt-5-anything"

    def test_populated_route_catalog_rejects_even_when_vendor_cache_empty(self):
        """Regression (codex P2): a POPULATED route-scoped catalog that proves
        the model unservable must reject BEFORE the vendor-cache fallback —
        otherwise an empty shared vendor cache on cold start would permit e.g.
        gpt-5-pro on openai:plan (the #1933 skew)."""
        svc = _make_service()
        svc.providers = [_route("openai:plan", "openai", "plan")]
        svc._route_catalogs = {"openai:plan": [_mk_model("gpt-5-codex", "openai")]}
        empty_cache = _cached([])  # shared vendor discovery not yet populated
        with patch("kestrel_sovereign.llm.model_cache.get_shared_model_cache", return_value=empty_cache):
            # Served by the route's own catalog → ok even with empty vendor cache.
            svc.set_model_preference("gpt-5-codex", vendor="openai", route="plan")
            assert svc.get_model_preference()["model"] == "gpt-5-codex"
            # Not in the route catalog → rejected despite the empty vendor cache.
            with pytest.raises(ValueError):
                svc.set_model_preference("gpt-5-pro", vendor="openai", route="plan")

    def test_route_catalogs_built_when_unset(self):
        """Regression: a route-scoped route must be recognized even when
        ``_route_catalogs`` is unset at validate time.

        This instance may have been created before the shared vendor cache was
        populated (by another instance), so ``_route_catalogs`` is still unset
        while the vendor catalog is non-empty. The validator must build route
        catalogs first (via ``_ensure_route_catalogs_sync``) so a route-scoped
        route (codex/openai:plan) is held to its OWN catalog rather than
        falling through to the broader vendor catalog and accepting an
        api-only model. (codex finding on #1946.)
        """
        from unittest.mock import AsyncMock

        from kestrel_sovereign.llm.codex_adapter import CodexAdapter

        svc = _make_service()
        # _route_catalogs intentionally NOT set on the instance.
        assert not hasattr(svc, "_route_catalogs")

        codex_adapter = CodexAdapter.__new__(CodexAdapter)
        # Route-scoped catalog serves only gpt-5-codex; the broader vendor
        # catalog (below) additionally has gpt-5-pro, which must be rejected.
        codex_adapter.list_models = AsyncMock(
            return_value=[_mk_model("gpt-5-codex", "openai")]
        )
        svc.providers = [{
            "name": "openai:plan", "vendor": "openai", "route": "plan",
            "model": "auto", "adapter": codex_adapter,
        }]
        cache = _cached([
            _mk_model("gpt-5-codex", "openai"),
            _mk_model("gpt-5-pro", "openai"),
        ])
        with patch("kestrel_sovereign.llm.model_cache.get_shared_model_cache", return_value=cache):
            # In the route-scoped catalog → accepted.
            svc.set_model_preference("gpt-5-codex", vendor="openai", route="plan")
            assert svc.get_model_preference()["model"] == "gpt-5-codex"
            # Vendor-catalog-only (api-only) model → rejected on the plan route,
            # proving route catalogs were built despite starting unset.
            with pytest.raises(ValueError):
                svc.set_model_preference("gpt-5-pro", vendor="openai", route="plan")

    def test_route_catalogs_built_inside_running_loop(self):
        """Regression (codex P1): validating inside a running event loop must
        build the REAL route-scoped catalog, not an empty placeholder.

        The ``set_model`` tool is async, so ``set_model_preference`` runs inside
        a loop. ``_ensure_route_catalogs_sync`` previously registered an empty
        placeholder there, and "empty route catalog → permit" let an api-only
        model land on the plan route. The sync helper now drives the (loop-
        independent) builder on a worker thread, so the plan route is held to
        codex's actual serveable subset.
        """
        import asyncio
        from unittest.mock import AsyncMock

        from kestrel_sovereign.llm.codex_adapter import CodexAdapter

        async def _run():
            svc = _make_service()
            assert not hasattr(svc, "_route_catalogs")
            codex_adapter = CodexAdapter.__new__(CodexAdapter)
            codex_adapter.list_models = AsyncMock(
                return_value=[_mk_model("gpt-5-codex", "openai")]
            )
            svc.providers = [{
                "name": "openai:plan", "vendor": "openai", "route": "plan",
                "model": "auto", "adapter": codex_adapter,
            }]
            cache = _cached([
                _mk_model("gpt-5-codex", "openai"),
                _mk_model("gpt-5-pro", "openai"),  # api-only, NOT in codex cache
            ])
            with patch("kestrel_sovereign.llm.model_cache.get_shared_model_cache", return_value=cache):
                svc.set_model_preference("gpt-5-codex", vendor="openai", route="plan")
                assert svc.get_model_preference()["model"] == "gpt-5-codex"
                with pytest.raises(ValueError):
                    svc.set_model_preference("gpt-5-pro", vendor="openai", route="plan")

        asyncio.run(_run())

    def test_empty_discovery_permits_known_route(self):
        """Known route + empty catalog (pre-discovery) → permit, don't block."""
        svc = _make_service()
        svc.providers = [_route("openai:api", "openai", "api")]
        cache = _cached(None)  # discovery hasn't populated
        with patch("kestrel_sovereign.llm.model_cache.get_shared_model_cache", return_value=cache):
            svc.set_model_preference("gpt-5-future", vendor="openai", route="api")
        assert svc.get_model_preference()["model"] == "gpt-5-future"
