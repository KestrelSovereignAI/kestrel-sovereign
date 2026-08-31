"""A persisted model pin must survive a discovery outage (#3186 / #3190).

Measured on the live host 2026-08-31. The operator disabled the Anthropic API
key — a legitimate action, taken to stop a different agent falling back to the
paid API. Model discovery then 401ed, the shared catalog for `anthropic`
collapsed to a single stale entry, and the boot-time re-validation of every
agent's persisted preference "proved" `claude-opus-5` unservable on
`anthropic:plan`. That ValueError was swallowed into a `logging.warning`, so
all four agents booted with NO mandate, fell through to `route_priority`, and
with `allow_paid_fallback = false` landed on `ollama:local/llama3.2:1b`. One
then fabricated a pushed commit SHA.

The route worked perfectly throughout. Only *discovery* was broken.

Two changes, deliberately small:

1. The boot loader stops re-validating its own persisted decision. Validation
   belongs where NEW information enters (`set_model_preference` from an
   operator or a tool, which is what #1927/#1946 guard). Replaying a triple
   this agent already accepted is a different boundary, and gating it on a
   catalog fetched seconds earlier lets a transient outage silently revoke a
   deliberate choice. If the pin really is unservable,
   `resolve_provider_routing` raises at USE time — loud where it matters.

2. Discovery failures are recorded and surfaced on the health check, so a dead
   credential is visible instead of living only in a log line.

What is deliberately NOT here: any attempt to infer "did discovery succeed?"
from a returned catalog. Five review rounds each found one more adapter
convention — `AnthropicAdapter` swallows a 401 and returns `[]`,
`VertexAIAdapter` swallows and returns a non-empty STALE disk catalog, the
OpenAI-compatible helpers raise — because the return value does not carry the
fact. Nothing routing-critical depends on that inference now.
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


def _make_service(*, providers=None, discovery_failures=None):
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


class TestReplayedPinIsNotRevalidated:
    """THE fix. A persisted pin is re-applied without a fresh catalog check."""

    def test_a_replayed_pin_survives_a_collapsed_catalog(self):
        svc = _make_service()
        cache = _cached(_COLLAPSED_ANTHROPIC_CATALOG)
        with patch(
            "kestrel_sovereign.llm.model_cache.get_shared_model_cache",
            return_value=cache,
        ):
            # validate=False is what the boot loader passes.
            svc.set_model_preference(
                "claude-opus-5", "anthropic", "plan", validate=False
            )
        assert svc._mandate_preference == {
            "vendor": "anthropic",
            "model": "claude-opus-5",
            "route": "plan",
        }

    def test_the_same_call_WITH_validation_still_refuses(self):
        """The converse — proves validation was made optional, not deleted.

        Without this, 'fix' the bug by removing `_validate_explicit_mandate`
        entirely and the test above still passes while #1927/#1946 are gone.
        """
        svc = _make_service()
        cache = _cached(_COLLAPSED_ANTHROPIC_CATALOG)
        with patch(
            "kestrel_sovereign.llm.model_cache.get_shared_model_cache",
            return_value=cache,
        ):
            with pytest.raises(ValueError, match="not served by that vendor/route"):
                svc.set_model_preference("claude-opus-5", "anthropic", "plan")

    def test_validation_is_on_by_default(self):
        """A caller that forgets the keyword must get the guarded behaviour."""
        import inspect

        sig = inspect.signature(LLMService.set_model_preference)
        assert sig.parameters["validate"].default is True

    def test_the_boot_loader_passes_validate_false(self):
        """Wiring: testing the flag says nothing about who uses it."""
        import inspect

        from kestrel_sovereign.agent.model_preference import ModelPreferenceMixin

        src = inspect.getsource(ModelPreferenceMixin._load_model_preference)
        assert "validate=False" in src, (
            "the boot loader must re-apply its own persisted decision without "
            "re-validating it against a live catalog"
        )


class TestUnrelatedGuardsAreUnchanged:
    """The #1927/#1946 setter guard keeps working exactly as before."""

    def test_unknown_route_is_still_a_hard_error(self):
        svc = _make_service()
        with pytest.raises(ValueError, match="no configured route matches"):
            svc._validate_explicit_mandate("claude-opus-5", "anthropic", "nonexistent")

    def test_a_served_model_is_still_permitted(self):
        svc = _make_service()
        cache = _cached(
            [_mk_model("claude-sonnet-5", "anthropic"),
             _mk_model("claude-opus-5", "anthropic")]
        )
        with patch(
            "kestrel_sovereign.llm.model_cache.get_shared_model_cache",
            return_value=cache,
        ):
            svc._validate_explicit_mandate("claude-opus-5", "anthropic", "plan")

    def test_an_empty_vendor_catalog_still_permits(self):
        svc = _make_service()
        cache = _cached([_mk_model("gpt-5.6-luna", "openai")])
        with patch(
            "kestrel_sovereign.llm.model_cache.get_shared_model_cache",
            return_value=cache,
        ):
            svc._validate_explicit_mandate("claude-opus-5", "anthropic", "plan")


class TestDiscoveryFailuresAreRecorded:
    """Only OBSERVED exceptions. Nothing is inferred from a returned catalog."""

    def test_an_exception_is_recorded(self):
        svc = _make_service()
        svc._note_discovery_outcome("anthropic", RuntimeError("401 Unauthorized"))
        assert "401 Unauthorized" in svc._discovery_failures["anthropic"]

    def test_success_clears_a_prior_failure(self):
        svc = _make_service(discovery_failures={"anthropic": "401"})
        svc._note_discovery_outcome("anthropic", None)
        assert "anthropic" not in svc._discovery_failures

    def test_the_reason_is_bounded(self):
        svc = _make_service()
        svc._note_discovery_outcome("anthropic", RuntimeError("x" * 5000))
        assert len(svc._discovery_failures["anthropic"]) <= 300

    def test_a_partially_constructed_instance_does_not_raise(self):
        svc = LLMService.__new__(LLMService)
        svc._note_discovery_outcome("anthropic", RuntimeError("boom"))

    @pytest.mark.asyncio
    async def test_safe_list_models_records_an_adapter_exception(self):
        svc = _make_service()
        adapter = MagicMock()

        async def _list(client):
            raise RuntimeError("401 Unauthorized")

        adapter.list_models = _list
        assert await svc._safe_list_models("anthropic", adapter, MagicMock()) == []
        assert "401 Unauthorized" in svc._discovery_failures["anthropic"]

    @pytest.mark.asyncio
    async def test_not_implemented_is_not_a_failure(self):
        """No catalog published is not the same as a failed fetch."""
        svc = _make_service()
        adapter = MagicMock()

        async def _list(client):
            raise NotImplementedError

        adapter.list_models = _list
        assert await svc._safe_list_models("anthropic", adapter, MagicMock()) == []
        assert svc._discovery_failures == {}

    @pytest.mark.asyncio
    async def test_a_swallowing_adapter_is_invisible_and_that_is_stated(self):
        """Documents the KNOWN limit rather than pretending it is covered.

        `AnthropicAdapter` catches its own 401 and returns `[]`. Nothing here
        can see that, and no routing decision depends on this record — which is
        precisely why the limit is tolerable. Closing it needs an explicit
        outcome contract from the adapters.
        """
        svc = _make_service()
        adapter = MagicMock()

        async def _list(client):
            return []  # what AnthropicAdapter does with a 401

        adapter.list_models = _list
        await svc._safe_list_models("anthropic", adapter, MagicMock())
        assert svc._discovery_failures == {}


class TestRecordDiscoverySeam:
    """The OpenAI-compatible paths bypass `_safe_list_models` entirely."""

    @pytest.mark.asyncio
    async def test_a_failure_is_recorded(self):
        svc = _make_service()

        async def _boom():
            raise RuntimeError("remote model discovery failed (…/models): 404")

        assert await svc._record_discovery("runpod", _boom()) == []
        assert "404" in svc._discovery_failures["runpod"]

    @pytest.mark.asyncio
    async def test_success_clears(self):
        svc = _make_service(discovery_failures={"ollama": "stale"})

        async def _ok():
            return [_mk_model("llama3.2:1b", "ollama")]

        assert len(await svc._record_discovery("ollama", _ok())) == 1
        assert "ollama" not in svc._discovery_failures


class TestHelpersRaiseRatherThanSwallow:
    """Drive the REAL helpers. `_record_discovery` can only see a failure if
    the helper actually raises; revert either to `return []` and the outage
    becomes invisible again."""

    @pytest.mark.asyncio
    async def test_remote_helper_raises_on_a_failed_request(self, monkeypatch):
        import httpx

        svc = _make_service()
        monkeypatch.setenv("RUNPOD_API_KEY", "k")

        class _Boom:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def get(self, *a, **k):
                raise httpx.HTTPError("404 Not Found")

        monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _Boom())
        with pytest.raises(RuntimeError, match="remote model discovery failed"):
            await svc._discover_openai_compatible_remote(
                "runpod", {"base_url": "https://api.runpod.ai/v2/x/openai/v1"}
            )

    @pytest.mark.asyncio
    async def test_local_helper_raises_on_a_failed_request(self, monkeypatch):
        import httpx

        svc = _make_service()

        class _Boom:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def get(self, *a, **k):
                raise httpx.HTTPError("connection refused")

        monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _Boom())
        with pytest.raises(RuntimeError, match="local model discovery failed"):
            await svc._discover_local_openai_compatible(
                "ollama", {"base_url": "http://localhost:11434/v1"}
            )

    @pytest.mark.asyncio
    async def test_absent_config_returns_empty_without_raising(self):
        """Missing base_url is 'not attempted', not an outage."""
        svc = _make_service()
        assert await svc._discover_openai_compatible_remote("runpod", {}) == []
        assert await svc._discover_local_openai_compatible("ollama", {}) == []
        assert svc._discovery_failures == {}


class TestVendorAttributionInGather:
    """`asyncio.gather` returns a bare list; the vendor must be zipped back on.

    Drives the real `discover_all_models` — a test that repeats the loop would
    still pass if the production zip were deleted.
    """

    @pytest.mark.asyncio
    async def test_an_outer_exception_is_recorded_against_its_own_vendor(self):
        svc = _make_service(providers=[])
        svc._select_discovery_routes = lambda: [
            ("anthropic", {"name": "anthropic:plan"}),
            ("openai", {"name": "openai:api"}),
        ]

        async def _fake_discover(vendor, route):
            if vendor == "anthropic":
                raise RuntimeError("401 Unauthorized")
            return []

        svc._discover_for_vendor_route = _fake_discover
        svc._resolve_auto_providers = lambda models: None
        svc._snapshot_chat_models_by_route = lambda models: None
        svc._apply_recency_visibility = lambda *a, **k: None
        svc._filter_models = lambda models, **k: models

        async def _noop(*a, **k):
            return None

        svc.reconcile_embedding_capabilities = _noop

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


class TestMandateLoadErrorLifecycle:
    def test_a_successful_set_clears_a_recorded_load_error(self):
        svc = _make_service()
        svc._mandate_load_error = "Cannot set model 'claude-opus-5' ..."
        cache = _cached([_mk_model("claude-opus-5", "anthropic")])
        with patch(
            "kestrel_sovereign.llm.model_cache.get_shared_model_cache",
            return_value=cache,
        ):
            svc.set_model_preference("claude-opus-5", "anthropic", "plan")
        assert svc._mandate_load_error is None

    def test_clearing_the_preference_also_clears_it(self):
        """Both ends: returning to automatic routing is a deliberate state.

        A flag one door sets and only one other door clears is a flag that
        gets stuck, and health would keep reporting a failure the operator
        resolved by choosing to be unpinned.
        """
        svc = _make_service()
        svc._mandate_load_error = "Cannot set model 'claude-opus-5' ..."
        svc.clear_model_preference()
        assert svc._mandate_load_error is None


class TestAuditPathAgreesWithGeneration:
    """Generation and its own audit must reach the same verdict (#3190 r6 P1).

    The generation path skips the catalog guard for an EXPLICIT selection
    (`skip_catalog = explicit_selection and vendor not in _MODEL_IGNORING_VENDORS`).
    `get_audit_response` applied the guard unconditionally, so during the
    collapsed-catalog outage a pinned route would generate successfully and
    then fail its own audit: every route recorded as "target model not
    available", the loop failing closed at risk 3. With ResponseAudit in warn
    mode that annotates every response; in strict mode it denies every one.

    Same shape as the setter/runtime-guard split: one seam learned a rule and
    its sibling did not. The comment explaining WHY an explicit selection is
    trusted lived on the generation path the whole time.
    """

    @pytest.mark.asyncio
    async def test_a_pinned_route_is_not_rejected_by_its_own_audit(self):
        """Drives the real `get_audit_response` against the collapsed catalog."""
        adapter = MagicMock()
        adapter.provider_capabilities.return_value = MagicMock(
            supports_structured_output=True
        )
        provider = {
            "name": "anthropic:plan",
            "vendor": "anthropic",
            "route": "plan",
            "model": "auto",
            "adapter": adapter,
            "client": MagicMock(),
        }
        svc = _make_service(providers=[provider])
        svc._mandate_preference = {
            "vendor": "anthropic", "model": "claude-opus-5", "route": "plan",
        }
        svc._check_policy = lambda: None
        svc._resolve_invocation_context = lambda ctx: ctx
        svc._get_default_mandate_selector = lambda: None
        svc._available_providers = lambda: [provider]
        svc._resolve_model_selector = lambda sel, providers=None: {
            "provider": "anthropic:plan", "model": "claude-opus-5",
        }
        svc._resolve_concrete_model = lambda target, prov: "claude-opus-5"

        attempted = {"ran": False}

        async def _attempt(*a, **k):
            attempted["ran"] = True
            return '{"risk_level": 1, "reasoning": "fine"}'

        svc._run_provider_attempt = _attempt

        cache = _cached(_COLLAPSED_ANTHROPIC_CATALOG)
        with patch(
            "kestrel_sovereign.llm.model_cache.get_shared_model_cache",
            return_value=cache,
        ):
            result = await svc.get_audit_response("some response text")

        assert attempted["ran"], (
            "the audit never attempted the route — an explicitly pinned target "
            "was rejected by the catalog guard that generation skips"
        )
        assert result.get("risk_level") == 1

    @pytest.mark.asyncio
    async def test_a_NON_explicit_target_is_still_guarded(self):
        """The converse: with no pin, the collapsed catalog must still bite.

        Without this, 'fix' the P1 by deleting the guard from the audit path
        entirely and the test above still passes.
        """
        adapter = MagicMock()
        adapter.provider_capabilities.return_value = MagicMock(
            supports_structured_output=True
        )
        provider = {
            "name": "anthropic:plan",
            "vendor": "anthropic",
            "route": "plan",
            "model": "auto",
            "adapter": adapter,
            "client": MagicMock(),
        }
        svc = _make_service(providers=[provider])
        # No vendor/route pin at all -> not an explicit selection.
        svc._mandate_preference = {"vendor": None, "model": None, "route": None}
        svc._check_policy = lambda: None
        svc._resolve_invocation_context = lambda ctx: ctx
        svc._get_default_mandate_selector = lambda: None
        svc._available_providers = lambda: [provider]
        svc._resolve_model_selector = lambda sel, providers=None: {
            "provider": "anthropic:plan", "model": "claude-opus-5",
        }
        svc._resolve_concrete_model = lambda target, prov: "claude-opus-5"

        attempted = {"ran": False}

        async def _attempt(*a, **k):
            attempted["ran"] = True
            return '{"risk_level": 1, "reasoning": "fine"}'

        svc._run_provider_attempt = _attempt
        # Force a target_model without an explicit mandate.
        svc._mandate_preference = {"vendor": None, "model": "claude-opus-5", "route": None}

        cache = _cached(_COLLAPSED_ANTHROPIC_CATALOG)
        with patch(
            "kestrel_sovereign.llm.model_cache.get_shared_model_cache",
            return_value=cache,
        ):
            result = await svc.get_audit_response("some response text")

        assert not attempted["ran"], (
            "a non-explicit target absent from a trustworthy catalog must "
            "still be skipped — the guard was removed, not gated"
        )
        assert result.get("risk_level") == 3

    def test_generation_and_audit_read_the_same_rule(self):
        """Both sites must gate on explicitness, not just one.

        Pins the invariant rather than the implementation: if a future edit
        removes the condition from either site, this fails.
        """
        import inspect

        gen = inspect.getsource(LLMService._get_provider_for_embeddings) \
            if hasattr(LLMService, "_get_provider_for_embeddings") else ""
        audit = inspect.getsource(LLMService.get_audit_response)
        assert "skip_catalog" in audit, (
            "get_audit_response must gate the catalog guard on whether the "
            "selection is explicit, as the generation path does"
        )
        assert "_MODEL_IGNORING_VENDORS" in audit, (
            "and must honour the same vendor exemption, or local routes "
            "diverge between generation and audit"
        )
