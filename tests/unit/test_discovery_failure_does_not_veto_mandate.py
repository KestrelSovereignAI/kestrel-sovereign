"""A vendor whose model discovery FAILED must not veto an operator's pinned model.

Regression for #3190, measured on the live host 2026-08-31.

The operator disabled the Anthropic API key. ``GET /v1/models`` then 401ed, so
the shared catalog for ``anthropic`` collapsed to a single stale entry. The
serveability guard in ``_validate_explicit_mandate`` read *non-empty* as
*complete*, "proved" that ``claude-opus-5`` was not served by ``anthropic:plan``
— a model with 3,631 successful calls on that exact route — and raised.

``_load_model_preference`` catches that ValueError into a ``logging.warning``,
so all four agents booted with **no mandate at all**, fell through to
``route_priority``, and (with ``allow_paid_fallback = false``) landed on
``ollama:local/llama3.2:1b``. One of them then fabricated a commit SHA.

The fix records whether a vendor's discovery actually SUCCEEDED, and only a
successful discovery may disprove a model. The two tests that matter here are
the motivating case and its converse: permitting when discovery failed, and
still rejecting when discovery succeeded. Without the second, the fix would be
indistinguishable from deleting the guard.
"""

from unittest.mock import MagicMock, patch

import pytest

from kestrel_sovereign.llm.model_metadata import ModelCategory, ModelInfo
from kestrel_sovereign.llm.service import LLMService


def _mk_model(id_, vendor):
    return ModelInfo(
        id=id_,
        provider=vendor,
        display_name=id_,
        category=ModelCategory.CHAT,
        supports_tools=True,
    )


def _cached(models):
    cache = MagicMock()
    cache.get_any = MagicMock(return_value=models)
    return cache


def _make_service(*, discovery_failures=None, providers=None):
    """Bare LLMService wired just enough for _validate_explicit_mandate."""
    svc = LLMService.__new__(LLMService)
    svc.providers = providers if providers is not None else [
        {"name": "anthropic:plan", "vendor": "anthropic", "route": "plan", "model": "auto"},
    ]
    svc._mandate_preference = {"vendor": None, "model": None, "route": None}
    svc._preference_persistence_callback = None
    svc._preference_persistence_tasks = set()
    svc._route_catalogs = {}
    svc._ensure_route_catalogs_sync = lambda: None
    svc._discovery_failures = dict(discovery_failures or {})
    svc._mandate_load_error = None
    return svc


# The exact catalog state observed on the host after the 401: one stale entry.
_COLLAPSED_ANTHROPIC_CATALOG = [_mk_model("claude-sonnet-5", "anthropic")]


class TestFailedDiscoveryMustNotVeto:
    def test_partial_catalog_from_failed_discovery_permits_the_pinned_model(self):
        """THE motivating case. Discovery failed -> catalog is a remnant, not proof."""
        svc = _make_service(discovery_failures={"anthropic": "AuthenticationError: 401"})  # anthropic discovery never succeeded
        cache = _cached(_COLLAPSED_ANTHROPIC_CATALOG)
        with patch(
            "kestrel_sovereign.llm.model_cache.get_shared_model_cache",
            return_value=cache,
        ):
            # Must NOT raise: this is the operator's real, working pin.
            svc._validate_explicit_mandate("claude-opus-5", "anthropic", "plan")

    def test_successful_discovery_still_rejects_an_unserved_model(self):
        """The converse. Proves the guard was gated, not deleted (#1927/#1946)."""
        svc = _make_service()
        cache = _cached(_COLLAPSED_ANTHROPIC_CATALOG)
        with patch(
            "kestrel_sovereign.llm.model_cache.get_shared_model_cache",
            return_value=cache,
        ):
            with pytest.raises(ValueError, match="not served by that vendor/route"):
                svc._validate_explicit_mandate("claude-opus-5", "anthropic", "plan")

    def test_successful_discovery_permits_a_model_it_lists(self):
        svc = _make_service()
        cache = _cached(
            [_mk_model("claude-sonnet-5", "anthropic"), _mk_model("claude-opus-5", "anthropic")]
        )
        with patch(
            "kestrel_sovereign.llm.model_cache.get_shared_model_cache",
            return_value=cache,
        ):
            svc._validate_explicit_mandate("claude-opus-5", "anthropic", "plan")

    def test_empty_vendor_catalog_still_permits(self):
        """Pre-existing cold-start behaviour is unchanged."""
        svc = _make_service()
        cache = _cached([_mk_model("gpt-5.6-luna", "openai")])
        with patch(
            "kestrel_sovereign.llm.model_cache.get_shared_model_cache",
            return_value=cache,
        ):
            svc._validate_explicit_mandate("claude-opus-5", "anthropic", "plan")

    def test_unknown_route_is_still_a_hard_error(self):
        """Route existence is a separate, stricter check — untouched by this fix."""
        svc = _make_service(discovery_failures={"anthropic": "AuthenticationError: 401"})
        with pytest.raises(ValueError, match="no configured route matches"):
            svc._validate_explicit_mandate("claude-opus-5", "anthropic", "nonexistent")


class TestDiscoveryOutcomeRecording:
    def test_success_clears_a_prior_failure(self):
        """The veto must RESUME once the catalog is retrievable again.

        Without this, one transient outage disables the guard for the life of
        the process.
        """
        svc = _make_service(discovery_failures={"anthropic": "AuthenticationError: 401"})
        svc._note_discovery_outcome("anthropic", None)
        assert "anthropic" not in svc._discovery_failures

    def test_failure_records_the_reason(self):
        """The recorded reason is what the health surface shows the operator."""
        svc = _make_service()
        svc._note_discovery_outcome("anthropic", RuntimeError("401 Unauthorized"))
        assert "401 Unauthorized" in svc._discovery_failures["anthropic"]

    def test_failure_reason_is_bounded(self):
        svc = _make_service()
        svc._note_discovery_outcome("anthropic", RuntimeError("x" * 5000))
        assert len(svc._discovery_failures["anthropic"]) <= 300

    def test_partially_constructed_instance_does_not_raise(self):
        """__init__ may not have run (bare harness); recording must be inert."""
        svc = LLMService.__new__(LLMService)
        svc._note_discovery_outcome("anthropic", RuntimeError("boom"))  # no attrs


class TestMandateLoadErrorLifecycle:
    def test_successful_set_clears_a_recorded_load_error(self):
        """Both ends: the flag that gets set must also get cleared."""
        svc = _make_service()
        svc._mandate_load_error = "Cannot set model 'claude-opus-5' ..."
        cache = _cached([_mk_model("claude-opus-5", "anthropic")])
        with patch(
            "kestrel_sovereign.llm.model_cache.get_shared_model_cache",
            return_value=cache,
        ):
            svc.set_model_preference("claude-opus-5", "anthropic", "plan")
        assert svc._mandate_load_error is None
        assert svc._mandate_preference["model"] == "claude-opus-5"


class TestVendorAttributionInGather:
    """``asyncio.gather`` returns a bare list; the vendor must be zipped back on.

    The pre-fix loop was ``for result in results:`` — it could neither name the
    failing vendor in its warning nor record the outcome against it, so an
    outer-level discovery exception was anonymous.

    This drives the REAL ``discover_all_models``: a test that re-implements the
    loop would still pass if the production zip were deleted.
    """

    @pytest.mark.asyncio
    async def test_an_outer_exception_is_recorded_against_its_own_vendor(self):
        svc = _make_service()
        svc.providers = []
        svc._select_discovery_routes = lambda: [
            ("anthropic", {"name": "anthropic:plan"}),
            ("openai", {"name": "openai:api"}),
        ]

        async def _fake_discover(vendor, route):
            if vendor == "anthropic":
                raise RuntimeError("401 Unauthorized")
            return []

        svc._discover_for_vendor_route = _fake_discover
        # Stub everything downstream of the gather — this test is about
        # attribution, not enrichment.
        svc._resolve_auto_providers = lambda models: None
        svc._snapshot_chat_models_by_route = lambda models: None
        svc._apply_recency_visibility = lambda *a, **k: None
        svc._filter_models = lambda models, **k: models

        async def _noop_reconcile(*a, **k):
            return None

        svc.reconcile_embedding_capabilities = _noop_reconcile

        shared = MagicMock()
        shared.get = MagicMock(return_value=None)
        shared.get_any = MagicMock(return_value=None)
        shared.set = MagicMock()

        async def _wait():
            return False

        shared.wait_for_refresh = _wait

        catalog = MagicMock()
        catalog.enrich_models = MagicMock(side_effect=lambda models, **k: models)

        with patch(
            "kestrel_sovereign.llm.model_discovery.get_shared_model_cache",
            return_value=shared,
        ), patch(
            "kestrel_sovereign.llm.model_discovery.get_catalog_service",
            return_value=catalog,
        ):
            await svc.discover_all_models(use_cache=False)

        assert "401 Unauthorized" in svc._discovery_failures["anthropic"]
        assert "openai" not in svc._discovery_failures


class TestSafeListModelsRecordsOutcome:
    """``_safe_list_models`` is where a vendor's failure is first observed.

    If it stopped recording, ``_discovery_failures`` would stay empty and the
    outage in this module's docstring would recur unchanged — the veto would go
    on trusting a collapsed catalog. If it stopped CLEARING, one transient 401
    would disable the #1927/#1946 guard for the life of the process.
    """

    @pytest.mark.asyncio
    async def test_success_grants_the_vendor_trust(self):
        svc = _make_service()
        adapter = MagicMock()

        async def _list(client):
            return [_mk_model("claude-opus-5", "anthropic")]

        adapter.list_models = _list
        svc._discovery_failures["anthropic"] = "stale prior failure"
        models = await svc._safe_list_models("anthropic", adapter, MagicMock())
        assert len(models) == 1
        assert "anthropic" not in svc._discovery_failures

    @pytest.mark.asyncio
    async def test_exception_records_failure_and_withholds_trust(self):
        svc = _make_service()
        adapter = MagicMock()

        async def _list(client):
            raise RuntimeError("401 Unauthorized")

        adapter.list_models = _list
        models = await svc._safe_list_models("anthropic", adapter, MagicMock())
        assert models == []
        assert "401 Unauthorized" in svc._discovery_failures["anthropic"]

    @pytest.mark.asyncio
    async def test_not_implemented_grants_no_trust(self):
        """No catalog published is not the same as an empty catalog.

        Such a vendor's rows come from config defaults, which cannot disprove
        anything — so it must never become authoritative.
        """
        svc = _make_service()
        adapter = MagicMock()

        async def _list(client):
            raise NotImplementedError

        adapter.list_models = _list
        models = await svc._safe_list_models("anthropic", adapter, MagicMock())
        assert models == []
        assert "anthropic" not in svc._discovery_failures
