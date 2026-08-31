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


def _loader_harness(pref):
    """A minimal agent whose `_load_model_preference` records its set call."""
    import json

    from kestrel_sovereign.agent.model_preference import ModelPreferenceMixin

    calls = []

    class _Agent(ModelPreferenceMixin):
        agent_id = "did:test:loader"

        def __init__(self):
            self.llm_service = MagicMock()
            self.llm_service.set_model_preference = (
                lambda m, v=None, r=None, *, validate=True: calls.append(
                    (m, v, r, validate)
                )
            )
            db = MagicMock()

            async def _fetchall(*a, **k):
                return [(json.dumps(pref),)]

            db.fetchall = _fetchall
            self._raw_storage = MagicMock()
            self._raw_storage.db = db

    return _Agent(), calls


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

    @pytest.mark.asyncio
    async def test_the_boot_loader_skips_validation_for_a_remote_pin(self):
        """Wiring, behaviourally: testing the flag says nothing about its user.

        Asserted through the call the loader actually makes rather than by
        grepping its source — a source assertion breaks the moment the value
        stops being a literal, which is exactly what happened when local
        vendors gained their own rule.
        """
        agent, calls = _loader_harness(
            {"vendor": "anthropic", "model": "claude-opus-5", "route": "plan"}
        )
        await agent._load_model_preference()
        assert calls == [("claude-opus-5", "anthropic", "plan", False)], (
            "a remote pin must be replayed without re-validation"
        )

    @pytest.mark.asyncio
    async def test_a_local_model_ignoring_pin_is_STILL_validated(self):
        """`ollama`/`llama_cpp` serve whatever is loaded and ignore the id.

        A stale id restored unvalidated is not just a wrong pin: streaming
        never calls `_model_available_for_route`, so responses from the newly
        loaded model are reported and METERED as the absent one. The bypass's
        justification — a transiently unfetchable remote catalog — does not
        apply to a local server, whose catalog IS what it has loaded.
        """
        agent, calls = _loader_harness(
            {"vendor": "ollama", "model": "llama3.2:1b", "route": "local"}
        )
        await agent._load_model_preference()
        assert calls == [("llama3.2:1b", "ollama", "local", True)], (
            "a local model-ignoring route must keep its validation"
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


class TestSkippedDiscoveryDoesNotClearAFailure:
    """"Nothing happened" is not "it worked" (#3190 r10 P2).

    The OpenAI-compatible helpers return `[]` when they never issued a request
    — no base_url, or the credential env var unset because the key is supplied
    inline or under a custom name. Treating that as success meant a real
    RunPod/xAI failure could be cleared by unsetting its key: health turns
    green having fetched nothing.

    The fix declines to infer SUCCESS from an empty return. It still does not
    infer FAILURE from one — that is the thing nine rounds established cannot
    work, because every adapter answers differently.
    """

    @pytest.mark.asyncio
    async def test_a_skipped_request_leaves_a_prior_failure_standing(self):
        svc = _make_service(discovery_failures={"runpod": "404 Not Found"})

        async def _skipped():
            return []  # never issued a request

        assert await svc._record_discovery("runpod", _skipped()) == []
        assert svc._discovery_failures.get("runpod") == "404 Not Found", (
            "a non-attempt must not clear a real failure"
        )

    @pytest.mark.asyncio
    async def test_a_skipped_request_does_not_invent_a_failure_either(self):
        """The other half: empty is not evidence of failure, only of nothing."""
        svc = _make_service()

        async def _skipped():
            return []

        await svc._record_discovery("runpod", _skipped())
        assert svc._discovery_failures == {}

    @pytest.mark.asyncio
    async def test_a_real_fetch_still_clears(self):
        """The converse — proves clearing was gated, not removed."""
        svc = _make_service(discovery_failures={"ollama": "connection refused"})

        async def _ok():
            return [_mk_model("llama3.2:1b", "ollama")]

        assert len(await svc._record_discovery("ollama", _ok())) == 1
        assert "ollama" not in svc._discovery_failures
