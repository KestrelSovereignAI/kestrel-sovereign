"""``llm_service`` and ``model_discovery`` read route health from the call ledger (#3912).

Health reads what dispatch did: the ``a2a_llm_calls`` rows LLMService writes
for every attempt. A route is reported ok or standby, and a startup failure
set aside, only on positive evidence:

- a route whose newest call failed, or that was disabled after a failure,
  warns;
- a configured route nothing in the agent's model configuration names, with
  no rows, is ``standby`` and degrades nothing;
- a route the configuration names, with no rows, is "active route unprobed";
- a failed startup reachability probe stands until a call known to be newer
  than it: one stamped after a health check first saw the probe's result;
- each route is read on its own, so one route's rows never crowd another's
  out of the read, and a capped read is followed by its failures. Each
  model's newest call is chosen by timestamp across the reads;
- lease-route rows count for the installed lease only when they name it.
  A failure naming no lease warns until a newer call names the lease. An
  installed lease that refuses calls now (expired, or not accepting) warns
  whatever its rows say.

Every ledger here is a real ``ObservabilityStore`` on a temp SQLite database,
and every service a real ``LLMService`` over mocked routes: no network, no
key, no paid completion.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from kestrel_sdk.llm import (
    InferenceLease,
    InferenceLeaseState,
    InferencePrivacy,
    InferenceRoute,
)
from pydantic import SecretStr

from kestrel_sovereign.a2a.stores.unified.observability_store import (
    _CONTENT_GATED_MARKER,
    ObservabilityStore,
)
from kestrel_sovereign.features.health import checks as health_checks
from kestrel_sovereign.features.health.checks import (
    LLM_ROUTE_EVIDENCE_ROW_LIMIT,
    LLM_ROUTE_EVIDENCE_WINDOW,
    _private_route_refusal,
    _startup_probe_seen_at,
    check_llm_service,
    check_model_discovery,
    derive_overall_status,
)
from kestrel_sovereign.lifecycle_checks import verify_llm_providers_reachable
from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.llm.error_handling import LLMProviderError
from kestrel_sovereign.llm.model_cache import (
    VendorDiscoveryOutcome,
    get_shared_model_cache,
)
from kestrel_sovereign.llm.output_ceiling import attach_stop_reason
from kestrel_sovereign.llm.remote_backend import REMOTE_ROUTE_NAME
from kestrel_sovereign.llm.service import LLMService, LLMServiceError
from kestrel_sovereign.privacy import PrivacyConfig, get_privacy_preset
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

DID = "did:test:route-evidence"
OTHER_DID = "did:test:someone-else"
MESSAGES = [{"role": "user", "content": "hi"}]
SENTINEL = "SENTINEL-3912-c0ffee"

ANTHROPIC = {"name": "anthropic:plan", "vendor": "anthropic", "route": "plan",
             "model": "claude-opus-5"}
LLAMA = {"name": "llama_cpp:local", "vendor": "llama_cpp", "route": "local",
         "model": "qwen3-8b", "is_local": True}
RUNPOD = {"name": "runpod:vllm", "vendor": "runpod", "route": "vllm",
          "model": "vllm-2f8vdgn"}
PINNED_ANTHROPIC = {"vendor": "anthropic", "route": "plan", "model": "claude-opus-5"}


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _cold_model_cache():
    cache = get_shared_model_cache()
    cache.clear()
    yield
    cache.clear()


@pytest.fixture
async def ledger(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "ledger.db"))
    await backend.connect()
    store = ObservabilityStore(backend)
    await store.initialize()
    try:
        yield store
    finally:
        await store.close()


@pytest.fixture
async def stack():
    async with AsyncExitStack() as owned:
        yield owned


@pytest.fixture
def lease_clients(monkeypatch):
    monkeypatch.setattr(
        "kestrel_sovereign.llm.remote_backend.openai.AsyncOpenAI",
        lambda **_kwargs: _LeaseClient(),
    )


class _ScriptedRoute:
    """A mocked provider route that answers or raises, one outcome per call."""

    def __init__(self, provider, outcomes=()):
        self.provider = provider
        self.outcomes = list(outcomes)
        self.calls = 0
        adapter = Mock()
        adapter.create_messages = Mock(return_value=list(MESSAGES))
        adapter.get_response = self._get_response
        self.adapter = adapter

    async def _get_response(self, *args, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return attach_stop_reason(
            LLMResponse(content=outcome, input_tokens=1, output_tokens=1), "stop"
        )

    def as_provider(self):
        return {
            **self.provider,
            "client": AsyncMock(),
            "adapter": self.adapter,
            "is_cloud": not self.provider.get("is_local"),
            "is_local": bool(self.provider.get("is_local")),
            "base_url": None,
            "capabilities": {},
        }


def _real_service(stack, routes, store, *, mandate_config=None):
    """A real LLMService over mocked routes, writing to ``store``."""
    registry = Mock()
    registry.initialize_providers = Mock(return_value=[])
    registry.get_providers_with_pattern = Mock(return_value=[])
    registry.get_local_providers = Mock(return_value=[])
    mandate_config = mandate_config or {"defaults": {}, "mandates": {}}
    with patch(
        "kestrel_sovereign.llm.service.load_section", return_value={}
    ), patch(
        "kestrel_sovereign.llm.service.load_config", return_value=mandate_config,
    ), patch("kestrel_sovereign.llm.service.ProviderRegistry", return_value=registry):
        service = LLMService()
    stack.push_async_callback(service.close)
    service._usage_db = None
    service._db_initialized = False
    service._usage_database_url = None
    service._db_backend = "sqlite"
    service.providers = [route.as_provider() for route in routes]
    service.attach_to_agent(DID)
    service.set_observability_store(store)
    return service


def _service(
    stack,
    providers,
    store,
    *,
    preference=None,
    disabled=None,
    reachability=None,
    mandate_config=None,
):
    """A real LLMService whose routes are configured but never called here:
    health reads its configuration, and the ledger rows a test writes."""
    service = _real_service(
        stack,
        [_ScriptedRoute(p) for p in providers],
        store,
        mandate_config=mandate_config,
    )
    if preference:
        service._mandate_preference = dict(preference)
    service._disabled_routes = dict(disabled or {})
    service.reachability = reachability
    return service


async def _install_lease(service, lease_id):
    """Activate a lease that serves no call here (ledger tests). The
    service's close releases it."""
    await _activate(service, lease_id, [])


def _agent(service, store, did=DID):
    return SimpleNamespace(did=did, llm_service=service, observability_store=store)


async def _log(store, provider, model, *, success, did=DID, **extra):
    await store.log_llm_call(
        provider=provider,
        model=model,
        duration_ms=5,
        success=success,
        agent_did=did,
        **extra,
    )


async def _log_at(store, monkeypatch, when, provider, model, **kwargs):
    with monkeypatch.context() as patched:
        patched.setattr(store, "now_utc", lambda: when)
        await _log(store, provider, model, **kwargs)


def _route(result, name):
    [route] = [r for r in result["details"]["routes"] if r["name"] == name]
    return route


def _discovery(failures):
    outcomes = {"anthropic": VendorDiscoveryOutcome.ok()}
    outcomes.update({
        vendor: VendorDiscoveryOutcome.failed(error)
        for vendor, error in failures.items()
    })
    get_shared_model_cache().set([], outcomes=outcomes)


class _LeaseAdapter:
    """The lease route's adapter: answers or raises, one outcome per call."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)

    def create_messages(self, *, user_prompt=None, system_prompt=None, **_):
        return [{"role": "user", "content": user_prompt or ""}]

    async def get_response(self, **kwargs):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return attach_stop_reason(
            LLMResponse(content=outcome, input_tokens=1, output_tokens=1), "stop"
        )


class _LeaseClient:
    async def close(self):
        pass


def _lease(lease_id, *, expires_in=timedelta(minutes=10)):
    """A ready lease. A negative ``expires_in`` is one that was ready when
    last updated and has since run past its expiry."""
    now = datetime.now(UTC)
    updated = min(now, now + expires_in - timedelta(minutes=1))
    return InferenceLease(
        lease_id=lease_id,
        quote_id=f"quote-{lease_id}",
        request_id=f"request-{lease_id}",
        owner_id="agent-1",
        provider_name="runpod",
        state=InferenceLeaseState.READY,
        model="llama-3",
        runtime="ollama",
        privacy=InferencePrivacy.AUTHENTICATED_ENDPOINT,
        created_at=updated,
        updated_at=updated,
        expires_at=now + expires_in,
        hourly_cost_usd=Decimal("0.50"),
        estimated_total_cost_usd=Decimal("0.20"),
        route=InferenceRoute(
            endpoint=SecretStr(f"https://{lease_id}.private.example.test/v1"),
            model="llama-3",
            api_key=SecretStr("route-secret"),
        ),
    )


async def _activate(service, lease_id, outcomes, *, expires_in=timedelta(minutes=10)):
    async def touch(requested):
        assert requested == lease_id
        return service._remote_lease

    await service.activate_inference_lease(
        _lease(lease_id, expires_in=expires_in),
        capabilities=("chat",),
        touch_lease=touch,
    )
    service._remote_adapter = _LeaseAdapter(outcomes)


def _bind_privacy(store, service, privacy):
    """Both production privacy hooks, bound as KestrelAgent binds them."""
    store.set_privacy_config_provider(lambda: privacy.config)
    service.set_force_local_only_provider(
        lambda: not privacy.config.allows_cloud_llm()
    )


# ---------------------------------------------------------------------------
# The active route's newest outcome
# ---------------------------------------------------------------------------


async def test_a_failed_newest_call_warns_and_a_later_success_passes(ledger, stack):
    """Rows written by real dispatch, not by hand."""
    anthropic = _ScriptedRoute(ANTHROPIC, [
        "first answer",
        LLMProviderError("anthropic", "overloaded"),
        "recovered",
    ])
    service = _real_service(stack, [anthropic], ledger)
    agent = _agent(service, ledger)

    assert await service.generate_with_messages(messages=list(MESSAGES)) == "first answer"
    with pytest.raises(Exception):
        await service.generate_with_messages(messages=list(MESSAGES))

    failing = await check_llm_service(agent)
    assert failing["status"] == "warn"
    route = _route(failing, "anthropic:plan")
    assert route["state"] == "failing"
    [model] = route["models"]
    assert (model["model"], model["calls"], model["failures"]) == ("claude-opus-5", 2, 1)
    assert model["last_call_succeeded"] is False
    assert "failing (newest call failed): anthropic:plan" in failing["message"]
    assert derive_overall_status([failing]) == "degraded"

    assert await service.generate_with_messages(messages=list(MESSAGES)) == "recovered"

    passing = await check_llm_service(agent)
    assert passing["status"] == "pass"
    assert _route(passing, "anthropic:plan")["state"] == "ok"
    assert "ok: anthropic:plan" in passing["message"]
    assert derive_overall_status([passing]) == "healthy"


async def test_each_model_on_a_route_is_judged_by_its_own_newest_call(ledger, stack):
    service = _service(stack, [ANTHROPIC], ledger)
    await _log(ledger, "anthropic:plan", "claude-haiku", success=True)
    await _log(ledger, "anthropic:plan", "claude-opus-5", success=False)
    await _log(ledger, "anthropic:plan", "claude-haiku", success=True)

    result = await check_llm_service(_agent(service, ledger))

    assert result["status"] == "warn"
    route = _route(result, "anthropic:plan")
    assert route["state"] == "failing"
    by_model = {m["model"]: m["last_call_succeeded"] for m in route["models"]}
    assert by_model == {"claude-haiku": True, "claude-opus-5": False}


async def test_only_this_agents_rows_and_the_window_count(ledger, stack, monkeypatch):
    service = _service(stack, [ANTHROPIC, LLAMA], ledger, preference=PINNED_ANTHROPIC)
    # Another agent's failure on the same shared table is not this agent's.
    await _log(ledger, "anthropic:plan", "claude-opus-5", success=False, did=OTHER_DID)
    # A failure older than the window is not evidence of the route now.
    stale = datetime.now(UTC) - LLM_ROUTE_EVIDENCE_WINDOW - timedelta(minutes=5)
    await _log_at(ledger, monkeypatch, stale, "llama_cpp:local", "qwen3-8b", success=False)

    result = await check_llm_service(_agent(service, ledger))

    assert result["status"] == "pass"
    assert _route(result, "anthropic:plan")["state"] == "unprobed"
    assert _route(result, "llama_cpp:local")["state"] == "standby"
    assert result["details"]["route_evidence"]["rows_read"] == 0


# ---------------------------------------------------------------------------
# Standby and "active route unprobed"
# ---------------------------------------------------------------------------


async def test_standby_providers_do_not_degrade_tick_status(ledger, stack):
    """The #3873 host: llama_cpp refuses connections and runpod 404s discovery,
    while anthropic:plan, the selected route, serves every call."""
    service = _service(
        stack,
        [ANTHROPIC, LLAMA, RUNPOD],
        ledger,
        preference=PINNED_ANTHROPIC,
        reachability=[
            {"name": "anthropic:plan", "status": "skipped"},
            {"name": "llama_cpp:local", "status": "unreachable"},
            {"name": "runpod:vllm", "status": "skipped"},
        ],
    )
    await _log(ledger, "anthropic:plan", "claude-opus-5", success=True)
    _discovery({"llama_cpp": "ConnectError: connection refused", "runpod": "404"})
    agent = _agent(service, ledger)

    llm = await check_llm_service(agent)
    discovery = await check_model_discovery(agent)

    assert llm["status"] == "pass"
    assert _route(llm, "anthropic:plan")["state"] == "ok"
    assert _route(llm, "llama_cpp:local")["state"] == "standby"
    assert _route(llm, "runpod:vllm")["state"] == "standby"
    assert "standby: llama_cpp:local, runpod:vllm" in llm["message"]
    assert discovery["status"] == "pass"
    assert sorted(discovery["details"]["standby_failed_vendors"]) == [
        "llama_cpp", "runpod",
    ]
    assert derive_overall_status([llm, discovery]) == "healthy"


async def test_a_failing_provider_that_served_traffic_still_degrades(ledger, stack):
    """The same standby route, once dispatch actually used it and it failed."""
    service = _service(
        stack,
        [ANTHROPIC, LLAMA],
        ledger,
        preference=PINNED_ANTHROPIC,
        reachability=[{"name": "llama_cpp:local", "status": "unreachable"}],
    )
    await _log(ledger, "anthropic:plan", "claude-opus-5", success=True)
    await _log(ledger, "llama_cpp:local", "qwen3-8b", success=False)
    _discovery({"llama_cpp": "ConnectError: connection refused"})
    agent = _agent(service, ledger)

    llm = await check_llm_service(agent)
    discovery = await check_model_discovery(agent)

    assert llm["status"] == "warn"
    assert _route(llm, "llama_cpp:local")["state"] == "failing"
    assert discovery["status"] == "warn"
    assert discovery["details"]["failed_vendors"] == {
        "llama_cpp": "ConnectError: connection refused"
    }
    assert discovery["details"]["standby_failed_vendors"] == {}
    assert derive_overall_status([llm, discovery]) == "degraded"


async def test_a_configured_route_with_no_rows_is_active_route_unprobed(ledger, stack):
    service = _service(stack, [ANTHROPIC, LLAMA], ledger, preference=PINNED_ANTHROPIC)
    result = await check_llm_service(_agent(service, ledger))

    assert result["status"] == "pass"
    route = _route(result, "anthropic:plan")
    assert route["state"] == "unprobed"
    assert route["configured_route"] is True
    assert "active route unprobed: anthropic:plan" in result["message"]
    assert _route(result, "llama_cpp:local")["state"] == "standby"


async def test_with_no_mandate_the_default_route_is_the_configured_one(ledger, stack):
    service = _service(stack, [ANTHROPIC, LLAMA], ledger)
    result = await check_llm_service(_agent(service, ledger))

    assert _route(result, "anthropic:plan")["state"] == "unprobed"
    assert _route(result, "llama_cpp:local")["state"] == "standby"


async def test_the_pinned_vendor_is_never_standby_for_discovery(ledger, stack):
    service = _service(stack, [ANTHROPIC, LLAMA], ledger, preference=PINNED_ANTHROPIC)
    _discovery({"anthropic": "AuthenticationError: 401", "llama_cpp": "refused"})

    discovery = await check_model_discovery(_agent(service, ledger))

    assert discovery["status"] == "fail"
    assert discovery["details"]["pinned_vendor_at_risk"] is True
    assert list(discovery["details"]["failed_vendors"]) == ["anthropic"]
    assert list(discovery["details"]["standby_failed_vendors"]) == ["llama_cpp"]


# ---------------------------------------------------------------------------
# Standby needs the whole model configuration
# ---------------------------------------------------------------------------


async def test_a_route_defaults_preferred_names_is_not_standby(ledger, stack):
    """The round-2 repro: anthropic first, ``defaults.preferred`` selecting
    llama_cpp, whose startup probe failed, on a cold ledger."""
    llama = _ScriptedRoute(LLAMA, ["local answer"])
    anthropic = _ScriptedRoute(ANTHROPIC, [])
    service = _real_service(
        stack,
        [anthropic, llama],
        ledger,
        mandate_config={"defaults": {"preferred": "llama_cpp"}, "mandates": {}},
    )
    service.reachability = [
        {"name": "anthropic:plan", "status": "skipped"},
        {"name": "llama_cpp:local", "status": "unreachable"},
    ]
    _discovery({"llama_cpp": "ConnectError: connection refused"})
    agent = _agent(service, ledger)

    llm = await check_llm_service(agent)
    discovery = await check_model_discovery(agent)

    assert llm["status"] == "warn"
    route = _route(llm, "llama_cpp:local")
    assert route["configured_route"] is True
    assert route["state"] == "unreachable"
    assert "unreachable at startup, no call known since: llama_cpp:local" in llm["message"]
    assert discovery["status"] == "warn"
    assert list(discovery["details"]["failed_vendors"]) == ["llama_cpp"]
    assert discovery["details"]["standby_failed_vendors"] == {}
    assert derive_overall_status([llm, discovery]) == "degraded"

    # Public dispatch agrees: the configured default is the route it takes.
    assert await service.get_response(system_prompt="", user_prompt="hello") == "local answer"
    assert (llama.calls, anthropic.calls) == (1, 0)


@pytest.mark.parametrize(
    ("mandate_config", "fallbacks"),
    [
        ({"defaults": {}, "mandates": {"code": "llama_cpp/qwen3-8b"}}, []),
        ({"defaults": {"cheap_model": "llama_cpp/qwen3-8b"}, "mandates": {}}, []),
        ({"defaults": {}, "mandates": {}}, [{"model": "qwen3-8b", "provider": "llama_cpp"}]),
        # A bare model id naming a route's default model.
        ({"defaults": {"preferred": "qwen3-8b"}, "mandates": {}}, []),
    ],
    ids=["mandate", "cheap-model", "fallback", "bare-default-model"],
)
async def test_a_route_any_mandate_selector_names_is_not_standby(
    ledger, stack, mandate_config, fallbacks
):
    service = _service(
        stack,
        [ANTHROPIC, LLAMA, RUNPOD],
        ledger,
        preference=PINNED_ANTHROPIC,
        mandate_config=mandate_config,
    )
    service._mandate_fallbacks = list(fallbacks)

    result = await check_llm_service(_agent(service, ledger))

    assert _route(result, "llama_cpp:local")["state"] == "unprobed"
    assert _route(result, "runpod:vllm")["state"] == "standby"


async def test_a_selector_naming_no_route_leaves_nothing_standby(ledger, stack):
    """A selector naming no configured vendor is tried as a model on every
    route, so no route can be shown to be selected by nothing."""
    service = _service(
        stack,
        [ANTHROPIC, LLAMA],
        ledger,
        preference=PINNED_ANTHROPIC,
        mandate_config={"defaults": {}, "mandates": {"truth": "xai"}},
        reachability=[{"name": "llama_cpp:local", "status": "unreachable"}],
    )
    _discovery({"llama_cpp": "refused"})
    agent = _agent(service, ledger)

    llm = await check_llm_service(agent)
    discovery = await check_model_discovery(agent)

    assert _route(llm, "anthropic:plan")["state"] == "unprobed"
    route = _route(llm, "llama_cpp:local")
    assert route["configured_route"] is None
    assert route["state"] == "unreachable"
    assert llm["status"] == "warn"
    assert discovery["status"] == "warn"
    assert discovery["details"]["standby_failed_vendors"] == {}


# ---------------------------------------------------------------------------
# The startup reachability probe
# ---------------------------------------------------------------------------


async def test_an_unreachable_selected_route_warns(ledger, stack):
    service = _service(
        stack,
        [LLAMA],
        ledger,
        preference={"vendor": "llama_cpp", "route": "local", "model": "qwen3-8b"},
        reachability=[{"name": "llama_cpp:local", "status": "unreachable"}],
    )
    result = await check_llm_service(_agent(service, ledger))

    assert result["status"] == "warn"
    assert _route(result, "llama_cpp:local")["state"] == "unreachable"


def _probed_local_service(stack, ledger, local_outcomes, *, cloud_outcomes=()):
    """A real service whose local route failed its real startup probe. The
    cloud route keeps startup from refusing, and is the local route's
    fallback."""
    llama = _ScriptedRoute(LLAMA, local_outcomes)
    llama.adapter.probe_reachable = AsyncMock(return_value=False)
    anthropic = _ScriptedRoute(ANTHROPIC, cloud_outcomes)
    return _real_service(stack, [llama, anthropic], ledger)


async def test_a_call_after_the_failed_probe_was_seen_supersedes_it(ledger, stack):
    """The round-1 repro, ordered: the probe fails, a check sees it, then a
    call succeeds. The probe runs once; a recovered route must not warn until
    the next restart."""
    service = _probed_local_service(
        stack, ledger,
        ["local answer", LLMProviderError("llama_cpp", "refused")],
        cloud_outcomes=[LLMProviderError("anthropic", "overloaded")],
    )
    agent = _agent(service, ledger)

    await verify_llm_providers_reachable(service, env={})
    assert [r["status"] for r in service.reachability] == ["unreachable", "skipped"]
    unreachable = await check_llm_service(agent)
    assert unreachable["status"] == "warn"
    assert _route(unreachable, "llama_cpp:local")["state"] == "unreachable"

    assert await service.generate_with_messages(messages=list(MESSAGES)) == "local answer"

    recovered = await check_llm_service(agent)
    assert recovered["status"] == "pass"
    assert _route(recovered, "llama_cpp:local")["state"] == "ok"
    assert "unreachable at startup" not in recovered["message"]
    # The probe result itself is still reported.
    assert recovered["details"]["reachability"] == service.reachability
    assert derive_overall_status([recovered]) == "healthy"

    # A later failure is the route's state again, whatever the probe said.
    with pytest.raises(Exception):
        await service.generate_with_messages(messages=list(MESSAGES))
    failing = await check_llm_service(agent)
    assert failing["status"] == "warn"
    assert _route(failing, "llama_cpp:local")["state"] == "failing"


async def test_a_call_not_known_to_follow_the_probe_does_not_supersede_it(ledger, stack):
    """Nothing checked between the probe and the call. The probe records no
    time, so that call cannot be placed after it: the probe stands until a
    call that can."""
    service = _probed_local_service(stack, ledger, ["unplaced", "placed"])
    agent = _agent(service, ledger)

    await verify_llm_providers_reachable(service, env={})
    assert await service.generate_with_messages(messages=list(MESSAGES)) == "unplaced"

    unplaced = await check_llm_service(agent)
    assert unplaced["status"] == "warn"
    route = _route(unplaced, "llama_cpp:local")
    assert route["state"] == "unreachable"
    assert [m["last_call_succeeded"] for m in route["models"]] == [True]

    assert await service.generate_with_messages(messages=list(MESSAGES)) == "placed"
    placed = await check_llm_service(agent)
    assert placed["status"] == "pass"
    assert _route(placed, "llama_cpp:local")["state"] == "ok"


async def test_a_reloaded_agents_failed_probe_outlasts_its_predecessors_success(
    ledger, stack
):
    """The round-3 repro: an agent reloaded in this process. Its first
    service's success came after the process started, but before the
    replacement's probe, which failed; it says nothing about that probe."""
    first = _real_service(stack, [_ScriptedRoute(LLAMA, ["first answer"])], ledger)
    assert await first.generate_with_messages(messages=list(MESSAGES)) == "first answer"
    first_agent = _agent(first, ledger)
    assert (await check_llm_service(first_agent))["status"] == "pass"

    second = _probed_local_service(stack, ledger, ["second answer"])
    await verify_llm_providers_reachable(second, env={})
    agent = _agent(second, ledger)

    reloaded = await check_llm_service(agent)
    assert reloaded["status"] == "warn"
    route = _route(reloaded, "llama_cpp:local")
    assert route["state"] == "unreachable"
    # The predecessor's success is read; it is not newer than the probe.
    assert [m["last_call_succeeded"] for m in route["models"]] == [True]
    assert derive_overall_status([reloaded]) == "degraded"

    assert await second.generate_with_messages(messages=list(MESSAGES)) == "second answer"
    recovered = await check_llm_service(agent)
    assert recovered["status"] == "pass"
    assert _route(recovered, "llama_cpp:local")["state"] == "ok"


async def test_a_probe_that_cannot_be_placed_stands(ledger, stack, monkeypatch):
    monkeypatch.setattr(health_checks, "_startup_probe_seen_at", lambda *_: None)
    service = _service(
        stack,
        [LLAMA],
        ledger,
        reachability=[{"name": "llama_cpp:local", "status": "unreachable"}],
    )
    agent = _agent(service, ledger)
    await check_llm_service(agent)
    await _log(ledger, "llama_cpp:local", "qwen3-8b", success=True)

    result = await check_llm_service(agent)

    assert result["status"] == "warn"
    assert _route(result, "llama_cpp:local")["state"] == "unreachable"


def test_each_probe_result_is_placed_when_a_check_first_sees_it():
    service = Mock()
    first = [{"name": "llama_cpp:local", "status": "unreachable"}]
    seen = _startup_probe_seen_at(service, first)
    assert seen is not None and seen <= datetime.now(UTC)
    # The same result keeps the time it was first seen.
    assert _startup_probe_seen_at(service, first) == seen
    # A re-probe assigns a new list, even an equal one: placed afresh.
    again = _startup_probe_seen_at(service, [dict(first[0])])
    assert again is not None and again >= seen
    # Another service's probe is its own.
    assert _startup_probe_seen_at(Mock(), first) >= seen
    # No result, or a service that cannot be tracked: nothing is placed.
    assert _startup_probe_seen_at(service, None) is None
    assert _startup_probe_seen_at(SimpleNamespace(), first) is None


async def test_a_failed_startup_probe_stands_while_its_route_has_no_call(ledger, stack):
    service = _service(
        stack,
        [ANTHROPIC, LLAMA],
        ledger,
        preference={"vendor": "llama_cpp", "route": "local", "model": "qwen3-8b"},
        reachability=[
            {"name": "anthropic:plan", "status": "skipped"},
            {"name": "llama_cpp:local", "status": "unreachable"},
        ],
    )
    # A call on another route says nothing about this one.
    await _log(ledger, "anthropic:plan", "claude-opus-5", success=True)

    result = await check_llm_service(_agent(service, ledger))

    assert result["status"] == "warn"
    assert _route(result, "llama_cpp:local")["state"] == "unreachable"
    assert "unreachable at startup, no call known since: llama_cpp:local" in result["message"]


# ---------------------------------------------------------------------------
# Failure-disabled routes are failing, never standby
# ---------------------------------------------------------------------------


async def test_a_failure_disabled_route_is_failing_despite_older_successes(ledger, stack):
    service = _service(
        stack,
        [ANTHROPIC, LLAMA],
        ledger,
        preference=PINNED_ANTHROPIC,
        disabled={"anthropic:plan": f"401 Unauthorized {SENTINEL}"},
    )
    await _log(ledger, "anthropic:plan", "claude-opus-5", success=True)
    await _log(ledger, "anthropic:plan", "claude-opus-5", success=True)

    result = await check_llm_service(_agent(service, ledger))

    assert result["status"] == "warn"
    assert _route(result, "anthropic:plan")["state"] == "disabled"
    assert "disabled after a failure: anthropic:plan" in result["message"]
    assert SENTINEL not in json.dumps(result)


async def test_a_disabled_route_with_no_rows_is_not_standby(ledger, stack):
    service = _service(
        stack,
        [ANTHROPIC, LLAMA],
        ledger,
        preference=PINNED_ANTHROPIC,
        disabled={"llama_cpp:local": "invalid api key"},
    )
    _discovery({"llama_cpp": "401"})
    agent = _agent(service, ledger)

    llm = await check_llm_service(agent)
    discovery = await check_model_discovery(agent)

    assert llm["status"] == "warn"
    assert _route(llm, "llama_cpp:local")["state"] == "disabled"
    assert discovery["status"] == "warn"
    assert discovery["details"]["standby_failed_vendors"] == {}


async def test_a_route_dispatch_disabled_after_an_auth_failure_is_failing(ledger, stack):
    """The real ``_maybe_disable_route``: fallback answered, the route is dead."""
    anthropic = _ScriptedRoute(ANTHROPIC, [
        "first answer",
        LLMProviderError("anthropic", "401 Unauthorized: invalid api key"),
    ])
    openai_route = _ScriptedRoute(
        {"name": "openai:api", "vendor": "openai", "route": "api", "model": "gpt-5"},
        ["fallback answer"],
    )
    service = _real_service(stack, [anthropic, openai_route], ledger)

    assert await service.generate_with_messages(messages=list(MESSAGES)) == "first answer"
    assert await service.generate_with_messages(messages=list(MESSAGES)) == "fallback answer"
    assert "anthropic:plan" in service._disabled_routes

    result = await check_llm_service(_agent(service, ledger))

    assert result["status"] == "warn"
    assert _route(result, "anthropic:plan")["state"] == "disabled"
    assert _route(result, "openai:api")["state"] == "ok"


# ---------------------------------------------------------------------------
# A capped read proves nothing about what lies beyond it
# ---------------------------------------------------------------------------


async def test_a_busy_route_does_not_hide_another_routes_failure(ledger, stack):
    """The round-2 repro: one llama_cpp failure, then a full read's worth of
    anthropic successes. Each route is read on its own."""
    service = _service(
        stack,
        [ANTHROPIC, LLAMA],
        ledger,
        preference=PINNED_ANTHROPIC,
        reachability=[{"name": "llama_cpp:local", "status": "unreachable"}],
    )
    await _log(ledger, "llama_cpp:local", "qwen3-8b", success=False)
    for _ in range(LLM_ROUTE_EVIDENCE_ROW_LIMIT):
        await _log(ledger, "anthropic:plan", "claude-opus-5", success=True)
    _discovery({"llama_cpp": "ConnectError: connection refused"})
    agent = _agent(service, ledger)

    llm = await check_llm_service(agent)
    discovery = await check_model_discovery(agent)

    assert llm["status"] == "warn"
    assert _route(llm, "llama_cpp:local")["state"] == "failing"
    anthropic = _route(llm, "anthropic:plan")
    assert anthropic["state"] == "ok"
    assert anthropic["row_limit_reached"] is True
    assert llm["details"]["route_evidence"]["row_limit_reached"] == ["anthropic:plan"]
    assert discovery["status"] == "warn"
    assert list(discovery["details"]["failed_vendors"]) == ["llama_cpp"]
    assert discovery["details"]["standby_failed_vendors"] == {}


async def test_a_capped_route_still_sees_a_model_whose_newest_call_failed(ledger, stack):
    service = _service(stack, [ANTHROPIC], ledger, preference=PINNED_ANTHROPIC)
    await _log(ledger, "anthropic:plan", "claude-opus-5", success=False)
    for _ in range(LLM_ROUTE_EVIDENCE_ROW_LIMIT):
        await _log(ledger, "anthropic:plan", "claude-haiku", success=True)

    result = await check_llm_service(_agent(service, ledger))

    assert result["status"] == "warn"
    route = _route(result, "anthropic:plan")
    assert route["state"] == "failing"
    by_model = {m["model"]: m["last_call_succeeded"] for m in route["models"]}
    assert by_model == {"claude-haiku": True, "claude-opus-5": False}


async def test_a_capped_route_judges_a_recovered_model_by_its_newest_call(
    ledger, stack, monkeypatch
):
    monkeypatch.setattr(health_checks, "LLM_ROUTE_EVIDENCE_ROW_LIMIT", 3)
    service = _service(stack, [ANTHROPIC], ledger, preference=PINNED_ANTHROPIC)
    await _log(ledger, "anthropic:plan", "claude-opus-5", success=False)
    await _log(ledger, "anthropic:plan", "claude-opus-5", success=True)
    for _ in range(3):
        await _log(ledger, "anthropic:plan", "claude-haiku", success=True)

    result = await check_llm_service(_agent(service, ledger))

    assert result["status"] == "pass"
    route = _route(result, "anthropic:plan")
    assert route["state"] == "ok"
    by_model = {m["model"]: m["last_call_succeeded"] for m in route["models"]}
    assert by_model == {"claude-haiku": True, "claude-opus-5": True}


async def test_more_failures_than_can_be_read_warns(ledger, stack, monkeypatch):
    """Every model read has recovered, but failures beyond the read may
    belong to a model that has not."""
    monkeypatch.setattr(health_checks, "LLM_ROUTE_EVIDENCE_ROW_LIMIT", 3)
    service = _service(stack, [ANTHROPIC], ledger, preference=PINNED_ANTHROPIC)
    for _ in range(3):
        await _log(ledger, "anthropic:plan", "claude-opus-5", success=False)
    await _log(ledger, "anthropic:plan", "claude-opus-5", success=True)
    for _ in range(3):
        await _log(ledger, "anthropic:plan", "claude-haiku", success=True)

    result = await check_llm_service(_agent(service, ledger))

    assert result["status"] == "warn"
    assert _route(result, "anthropic:plan")["state"] == "truncated"
    assert "failures beyond the 200 rows read: anthropic:plan" in result["message"]


async def test_a_call_failing_between_the_reads_is_the_models_newest(
    ledger, stack, monkeypatch
):
    """The round-3 repro: the route's first read is full and holds the
    model's success; a public call then fails before the failure read. That
    read returns the failure after the older success, but it is newer."""
    monkeypatch.setattr(health_checks, "LLM_ROUTE_EVIDENCE_ROW_LIMIT", 3)
    anthropic = _ScriptedRoute(ANTHROPIC, [
        "one", "two", "three", LLMProviderError("anthropic", "overloaded"),
    ])
    service = _real_service(stack, [anthropic], ledger)
    for _ in range(3):
        await service.generate_with_messages(messages=list(MESSAGES))
    read = ledger.query_llm_calls
    interleaved = []

    async def failure_between_reads(**kwargs):
        rows = await read(**kwargs)
        if not interleaved and kwargs.get("success") is None and len(rows) == 3:
            interleaved.append(rows[0].success)
            with pytest.raises(LLMServiceError):
                await service.generate_with_messages(messages=list(MESSAGES))
        return rows

    monkeypatch.setattr(ledger, "query_llm_calls", failure_between_reads)
    result = await check_llm_service(_agent(service, ledger))

    assert interleaved == [True]
    newest = (await read(agent_did=DID, limit=1))[0]
    assert newest.success is False
    assert result["status"] == "warn"
    route = _route(result, "anthropic:plan")
    assert route["state"] == "failing"
    [model] = route["models"]
    assert (model["model"], model["last_call_succeeded"]) == ("claude-opus-5", False)


@pytest.mark.parametrize(
    "outcomes", [(False, True), (True, False)], ids=["failure-first", "success-first"]
)
async def test_a_success_and_a_failure_at_one_instant_is_failing(
    ledger, stack, monkeypatch, outcomes
):
    """Neither can be shown to follow the other, so the success cannot set
    the failure aside, in whichever order the read returns them."""
    service = _service(stack, [ANTHROPIC], ledger, preference=PINNED_ANTHROPIC)
    instant = datetime.now(UTC) - timedelta(minutes=5)
    for success in outcomes:
        await _log_at(ledger, monkeypatch, instant, "anthropic:plan",
                      "claude-opus-5", success=success)

    result = await check_llm_service(_agent(service, ledger))

    assert result["status"] == "warn"
    assert _route(result, "anthropic:plan")["state"] == "failing"


# ---------------------------------------------------------------------------
# The private inference lease route
# ---------------------------------------------------------------------------


async def test_a_replacement_lease_does_not_inherit_the_old_leases_failures(
    ledger, stack, lease_clients
):
    """Private-only service: no configured providers, the lease is the route."""
    service = _real_service(stack, [], ledger)
    agent = _agent(service, ledger)

    await _activate(service, "lease-A", [RuntimeError("pod gone")])
    with pytest.raises(LLMServiceError):
        await service.generate_with_messages(messages=list(MESSAGES))

    [row] = await ledger.query_llm_calls(agent_did=DID)
    assert row.provider == REMOTE_ROUTE_NAME
    assert row.success is False
    assert row.metadata["inference_lease_id"] == "lease-A"

    serving = await check_llm_service(agent)
    assert serving["status"] == "warn"
    route = _route(serving, REMOTE_ROUTE_NAME)
    assert route["state"] == "failing"
    assert route["private_inference_lease"] is True
    assert derive_overall_status([serving]) == "degraded"

    await service.deactivate_inference_lease("lease-A")
    released = await check_llm_service(agent)
    # No lease and no providers: nothing to route to, and lease A's rows are
    # history, not a route.
    assert released["status"] == "warn"
    assert released["details"]["routes"] == []

    await _activate(service, "lease-B", ["from lease B"])
    replaced = await check_llm_service(agent)
    assert replaced["status"] == "pass"
    assert _route(replaced, REMOTE_ROUTE_NAME)["state"] == "unprobed"
    assert "active route unprobed: remote_gpu" in replaced["message"]

    assert await service.generate_with_messages(messages=list(MESSAGES)) == "from lease B"
    answered = await check_llm_service(agent)
    assert answered["status"] == "pass"
    route = _route(answered, REMOTE_ROUTE_NAME)
    assert route["state"] == "ok"
    assert [(m["calls"], m["failures"]) for m in route["models"]] == [(1, 0)]
    await service.deactivate_inference_lease("lease-B")


async def test_a_failing_lease_route_beside_healthy_providers_warns(
    ledger, stack, lease_clients
):
    """The lease takes precedence over the providers, so its failure counts."""
    anthropic = _ScriptedRoute(ANTHROPIC, [])
    service = _real_service(stack, [anthropic], ledger)
    await _activate(service, "lease-A", [RuntimeError("pod gone")])
    try:
        with pytest.raises(LLMServiceError):
            await service.generate_with_messages(messages=list(MESSAGES))
        result = await check_llm_service(_agent(service, ledger))
    finally:
        await service.deactivate_inference_lease("lease-A")

    assert anthropic.calls == 0
    assert result["status"] == "warn"
    assert _route(result, REMOTE_ROUTE_NAME)["state"] == "failing"
    assert _route(result, "anthropic:plan")["state"] == "unprobed"


async def test_an_expired_lease_warns_although_its_newest_call_succeeded(
    ledger, stack, lease_clients
):
    """The round-3 repro: the lease answered, then ran past its expiry. The
    route refuses every call before it is recorded, so its newest row is
    still the success."""
    service = _real_service(stack, [], ledger)
    agent = _agent(service, ledger)
    await _activate(service, "lease-A", ["answered"])
    assert await service.generate_with_messages(messages=list(MESSAGES)) == "answered"
    assert (await check_llm_service(agent))["status"] == "pass"

    # The same ready lease, as reconciliation would hand it over once its
    # expiry has passed. Its route is unchanged, so it replaces the installed
    # lease in place.
    await _activate(service, "lease-A", [], expires_in=timedelta(minutes=-1))
    expired = await check_llm_service(agent)
    assert expired["status"] == "warn"
    route = _route(expired, REMOTE_ROUTE_NAME)
    assert (route["state"], route["refusal"]) == ("unavailable", "lease expired")
    assert [m["last_call_succeeded"] for m in route["models"]] == [True]
    assert "refusing calls now: remote_gpu (lease expired)" in expired["message"]
    assert derive_overall_status([expired]) == "degraded"

    with pytest.raises(LLMServiceError, match="lease expired"):
        await service.generate_with_messages(messages=list(MESSAGES))
    # The refusal wrote no row: only the route's state shows it.
    [row] = await ledger.query_llm_calls(agent_did=DID)
    assert row.success is True
    refused = await check_llm_service(agent)
    assert refused["status"] == "warn"
    assert _route(refused, REMOTE_ROUTE_NAME)["state"] == "unavailable"


class _HeldLeaseAdapter(_LeaseAdapter):
    """Holds one call in flight until released."""

    def __init__(self):
        super().__init__([])
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def get_response(self, **kwargs):
        self.started.set()
        await self.release.wait()
        return attach_stop_reason(
            LLMResponse(content="late", input_tokens=1, output_tokens=1), "stop"
        )


async def test_a_draining_lease_route_warns(ledger, stack, lease_clients):
    """Release stops new calls and waits for the one in flight: meanwhile
    the route refuses every call, and has written no failure."""
    service = _real_service(stack, [], ledger)
    agent = _agent(service, ledger)
    await _activate(service, "lease-A", ["answered"])
    assert await service.generate_with_messages(messages=list(MESSAGES)) == "answered"

    held = _HeldLeaseAdapter()
    service._remote_adapter = held
    in_flight = asyncio.create_task(
        service.generate_with_messages(messages=list(MESSAGES))
    )
    await asyncio.wait_for(held.started.wait(), 5)
    releasing = asyncio.create_task(service.deactivate_inference_lease("lease-A"))
    try:
        async with asyncio.timeout(5):
            while service.get_backend_status()["remote_active"]:
                await asyncio.sleep(0)
        draining = await check_llm_service(agent)
    finally:
        held.release.set()
        assert await asyncio.wait_for(in_flight, 5) == "late"
        await asyncio.wait_for(releasing, 5)

    assert draining["status"] == "warn"
    route = _route(draining, REMOTE_ROUTE_NAME)
    assert (route["state"], route["refusal"]) == ("unavailable", "not accepting calls")
    assert [m["last_call_succeeded"] for m in route["models"]] == [True]


def test_lease_availability_that_cannot_be_read_is_a_refusal():
    future = datetime.now(UTC) + timedelta(minutes=5)
    lease = SimpleNamespace(lease_id="lease-A", expires_at=future)

    def raises():
        raise RuntimeError(f"status {SENTINEL}")

    assert _private_route_refusal(
        SimpleNamespace(_remote_lease=SimpleNamespace(lease_id="lease-A"))
    ) == "lease expiry unknown"
    assert _private_route_refusal(
        SimpleNamespace(_remote_lease=lease)
    ) == "route status unreadable"
    assert _private_route_refusal(
        SimpleNamespace(_remote_lease=lease, get_backend_status=raises)
    ) == "route status unreadable"
    assert _private_route_refusal(SimpleNamespace(
        _remote_lease=lease, get_backend_status=lambda: {"remote_active": True},
    )) is None


@pytest.mark.parametrize(
    "installed_fails", [True, False], ids=["B-fails-A-succeeds", "B-succeeds-A-fails"]
)
async def test_interleaved_leases_on_one_ledger_judge_only_their_own_rows(
    ledger, stack, lease_clients, installed_fails
):
    """Two services share one ledger, each with its own lease. The installed
    lease's health is its own calls; the other lease's newer call, success or
    failure, never decides it."""
    service_a = _real_service(stack, [], ledger)
    service_b = _real_service(stack, [], ledger)
    if installed_fails:
        await _activate(service_b, "lease-B", [RuntimeError("pod gone")])
        await _activate(service_a, "lease-A", ["from lease A"])
    else:
        await _activate(service_b, "lease-B", ["from lease B"])
        await _activate(service_a, "lease-A", [RuntimeError("pod gone")])
    try:
        # The installed lease B calls first; the other lease A's call is newer.
        if installed_fails:
            with pytest.raises(LLMServiceError):
                await service_b.generate_with_messages(messages=list(MESSAGES))
            assert await service_a.generate_with_messages(messages=list(MESSAGES)) == "from lease A"
        else:
            assert await service_b.generate_with_messages(messages=list(MESSAGES)) == "from lease B"
            with pytest.raises(LLMServiceError):
                await service_a.generate_with_messages(messages=list(MESSAGES))
        newest, older = await ledger.query_llm_calls(agent_did=DID)
        assert newest.metadata["inference_lease_id"] == "lease-A"
        assert older.metadata["inference_lease_id"] == "lease-B"

        result = await check_llm_service(_agent(service_b, ledger))
    finally:
        await service_a.deactivate_inference_lease("lease-A")
        await service_b.deactivate_inference_lease("lease-B")

    route = _route(result, REMOTE_ROUTE_NAME)
    [model] = route["models"]
    assert (model["calls"], model["last_call_succeeded"]) == (1, not installed_fails)
    if installed_fails:
        assert result["status"] == "warn"
        assert route["state"] == "failing"
    else:
        assert result["status"] == "pass"
        assert route["state"] == "ok"


async def test_lease_rows_naming_another_lease_are_ignored(ledger, stack, lease_clients):
    service = _service(stack, [ANTHROPIC], ledger)
    await _install_lease(service, "lease-B")
    await _log(ledger, REMOTE_ROUTE_NAME, "llama-3", success=False,
               metadata={"inference_lease_id": "lease-A"})
    await _log(ledger, REMOTE_ROUTE_NAME, "llama-3", success=True,
               metadata={"inference_lease_id": "lease-A"})

    result = await check_llm_service(_agent(service, ledger))

    assert result["status"] == "pass"
    route = _route(result, REMOTE_ROUTE_NAME)
    assert route["state"] == "unprobed"
    assert route["models"] == [] and route["unattributed_models"] == []


async def test_an_unattributed_lease_failure_warns_until_the_lease_names_a_newer_call(
    ledger, stack, lease_clients
):
    """A row naming no lease may be this lease's. Its failure warns, and an
    unattributed success never sets it aside: only a newer call naming the
    installed lease does."""
    service = _service(stack, [ANTHROPIC], ledger)
    await _install_lease(service, "lease-B")
    agent = _agent(service, ledger)
    await _log(ledger, REMOTE_ROUTE_NAME, "llama-3", success=True)
    await _log(ledger, REMOTE_ROUTE_NAME, "llama-3", success=False)

    failing = await check_llm_service(agent)
    assert failing["status"] == "warn"
    route = _route(failing, REMOTE_ROUTE_NAME)
    assert route["state"] == "unattributed_failing"
    assert route["models"] == []
    [model] = route["unattributed_models"]
    assert (model["calls"], model["failures"]) == (2, 1)
    assert "failing, lease unattributed" in failing["message"]
    assert derive_overall_status([failing]) == "degraded"

    await _log(ledger, REMOTE_ROUTE_NAME, "llama-3", success=True)
    still = await check_llm_service(agent)
    assert still["status"] == "warn"
    assert _route(still, REMOTE_ROUTE_NAME)["state"] == "unattributed_failing"

    await _log(ledger, REMOTE_ROUTE_NAME, "llama-3", success=True,
               metadata={"inference_lease_id": "lease-B"})
    recovered = await check_llm_service(agent)
    assert recovered["status"] == "pass"
    assert _route(recovered, REMOTE_ROUTE_NAME)["state"] == "ok"


async def test_an_unattributed_failure_newer_than_the_leases_success_warns(
    ledger, stack, monkeypatch, lease_clients
):
    """Position never attributes a row: an unnamed failure after the lease's
    own success is not counted as the lease's, and still warns."""
    service = _service(stack, [ANTHROPIC], ledger)
    await _install_lease(service, "lease-B")
    base = datetime.now(UTC) - timedelta(minutes=30)
    await _log_at(ledger, monkeypatch, base, REMOTE_ROUTE_NAME, "llama-3",
                  success=True, metadata={"inference_lease_id": "lease-B"})
    await _log_at(ledger, monkeypatch, base + timedelta(minutes=1),
                  REMOTE_ROUTE_NAME, "llama-3", success=False)

    result = await check_llm_service(_agent(service, ledger))

    assert result["status"] == "warn"
    route = _route(result, REMOTE_ROUTE_NAME)
    assert route["state"] == "unattributed_failing"
    [model] = route["models"]
    assert (model["calls"], model["failures"]) == (1, 0)


async def test_a_failed_private_call_warns_through_the_real_privacy_gate(
    ledger, stack, lease_clients
):
    """The round-1 repro: temporary storage elides the whole metadata column.
    The lease id is stamped after the gate, so the failure is the lease's."""
    privacy = SimpleNamespace(
        config=PrivacyConfig(storage="temp", processing="cloud")
    )
    service = _real_service(stack, [], ledger)
    _bind_privacy(ledger, service, privacy)
    agent = _agent(service, ledger)

    await _activate(service, "lease-A", [RuntimeError("pod gone"), "recovered"])
    try:
        with pytest.raises(LLMServiceError):
            await service.generate_with_messages(messages=list(MESSAGES))
        [row] = await ledger.query_llm_calls(agent_did=DID)
        assert row.provider == REMOTE_ROUTE_NAME and row.success is False
        # The gate elided everything else the call carried.
        assert row.metadata == {
            _CONTENT_GATED_MARKER: "temp",
            "inference_lease_id": "lease-A",
        }

        failing = await check_llm_service(agent)
        assert failing["status"] == "warn"
        assert _route(failing, REMOTE_ROUTE_NAME)["state"] == "failing"
        assert derive_overall_status([failing]) == "degraded"

        assert await service.generate_with_messages(messages=list(MESSAGES)) == "recovered"
        recovered = await check_llm_service(agent)
        assert recovered["status"] == "pass"
        assert _route(recovered, REMOTE_ROUTE_NAME)["state"] == "ok"
    finally:
        await service.deactivate_inference_lease("lease-A")


async def test_a_lease_failing_after_privacy_tightens_mid_lease_is_failing(
    ledger, stack, lease_clients
):
    privacy = SimpleNamespace(config=get_privacy_preset("normal"))
    service = _real_service(stack, [], ledger)
    _bind_privacy(ledger, service, privacy)
    agent = _agent(service, ledger)

    await _activate(service, "lease-A", ["named", RuntimeError("pod gone")])
    try:
        assert await service.generate_with_messages(messages=list(MESSAGES)) == "named"
        privacy.config = PrivacyConfig(storage="temp", processing="cloud")
        with pytest.raises(LLMServiceError):
            await service.generate_with_messages(messages=list(MESSAGES))
        gated, plain = await ledger.query_llm_calls(agent_did=DID)
        assert _CONTENT_GATED_MARKER in gated.metadata
        assert gated.metadata["inference_lease_id"] == "lease-A"
        assert plain.metadata["inference_lease_id"] == "lease-A"

        result = await check_llm_service(agent)
    finally:
        await service.deactivate_inference_lease("lease-A")

    assert result["status"] == "warn"
    route = _route(result, REMOTE_ROUTE_NAME)
    assert route["state"] == "failing"
    [model] = route["models"]
    assert (model["calls"], model["failures"], model["last_call_succeeded"]) == (
        2, 1, False,
    )


async def test_a_replacement_lease_under_temp_storage_is_not_judged_by_its_predecessor(
    ledger, stack, lease_clients
):
    privacy = SimpleNamespace(
        config=PrivacyConfig(storage="temp", processing="cloud")
    )
    service = _real_service(stack, [], ledger)
    _bind_privacy(ledger, service, privacy)
    agent = _agent(service, ledger)

    await _activate(service, "lease-A", [RuntimeError("pod gone")])
    with pytest.raises(LLMServiceError):
        await service.generate_with_messages(messages=list(MESSAGES))
    await service.deactivate_inference_lease("lease-A")

    await _activate(service, "lease-B", ["from lease B"])
    try:
        before = await check_llm_service(agent)
        assert before["status"] == "pass"
        assert _route(before, REMOTE_ROUTE_NAME)["state"] == "unprobed"

        assert await service.generate_with_messages(messages=list(MESSAGES)) == "from lease B"
        after = await check_llm_service(agent)
    finally:
        await service.deactivate_inference_lease("lease-B")

    assert after["status"] == "pass"
    assert _route(after, REMOTE_ROUTE_NAME)["state"] == "ok"


# ---------------------------------------------------------------------------
# Evidence that cannot be read is said so, never treated as standby
# ---------------------------------------------------------------------------


async def test_an_unreadable_ledger_warns_without_its_error_text(ledger, stack, monkeypatch):
    service = _service(stack, [ANTHROPIC, LLAMA], ledger, preference=PINNED_ANTHROPIC)

    async def broken(**_kwargs):
        raise RuntimeError(f"database said {SENTINEL}")

    monkeypatch.setattr(ledger, "query_llm_calls", broken)
    result = await check_llm_service(_agent(service, ledger))

    assert result["status"] == "warn"
    assert "call ledger unreadable: RuntimeError" in result["message"]
    assert result["details"]["route_evidence"]["available"] is False
    # Nothing could be read, so nothing is claimed to be standby.
    assert _route(result, "llama_cpp:local")["state"] == "unknown"
    assert _route(result, "anthropic:plan")["state"] == "unprobed"
    assert SENTINEL not in json.dumps(result)


async def test_a_wedged_ledger_read_is_bounded(ledger, stack, monkeypatch):
    service = _service(stack, [ANTHROPIC], ledger)

    async def wedged(**_kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(ledger, "query_llm_calls", wedged)
    monkeypatch.setattr(health_checks, "DATABASE_HEALTH_CHECK_TIMEOUT_S", 0.01)

    result = await asyncio.wait_for(check_llm_service(_agent(service, ledger)), 1)

    assert result["status"] == "warn"
    assert "read timed out" in result["message"]


async def test_without_a_ledger_a_standby_route_is_not_claimed(ledger, stack):
    service = _service(
        stack,
        [ANTHROPIC, LLAMA],
        ledger,
        preference=PINNED_ANTHROPIC,
        reachability=[{"name": "llama_cpp:local", "status": "unreachable"}],
    )
    _discovery({"llama_cpp": "refused"})
    agent = SimpleNamespace(did=DID, llm_service=service)

    llm = await check_llm_service(agent)
    discovery = await check_model_discovery(agent)

    assert llm["status"] == "warn"
    # Not shown standby, so its startup failure stands.
    assert _route(llm, "llama_cpp:local")["state"] == "unreachable"
    assert "no call evidence: no call ledger" in llm["message"]
    assert discovery["status"] == "warn"


# ---------------------------------------------------------------------------
# Content stays out
# ---------------------------------------------------------------------------


async def test_no_health_output_carries_prompt_response_or_error_text(
    ledger, stack, lease_clients
):
    service = _service(
        stack,
        [ANTHROPIC, LLAMA, RUNPOD],
        ledger,
        preference=PINNED_ANTHROPIC,
        disabled={"runpod:vllm": f"unauthorized {SENTINEL}"},
    )
    await _install_lease(service, "lease-A")
    content = {
        "system_prompt": f"system {SENTINEL}",
        "user_prompt": f"user {SENTINEL}",
        "response": f"response {SENTINEL}",
        "error_message": f"error {SENTINEL}",
        "tool_calls": [{"name": "t", "arguments": {"q": SENTINEL}}],
    }
    await _log(ledger, "anthropic:plan", "claude-opus-5", success=False, **content,
               metadata={"note": SENTINEL})
    await _log(ledger, "llama_cpp:local", "qwen3-8b", success=True, **content)
    await _log(ledger, REMOTE_ROUTE_NAME, "llama-3", success=False, **content,
               metadata={"inference_lease_id": "lease-A", "note": SENTINEL})
    await _log(ledger, REMOTE_ROUTE_NAME, "llama-3", success=False, **content,
               metadata={"note": SENTINEL})
    [stored, *_] = await ledger.query_llm_calls(agent_did=DID)
    assert SENTINEL in (stored.user_prompt_preview or "")  # the rows do carry it
    _discovery({"runpod": "404"})
    agent = _agent(service, ledger)

    llm = await check_llm_service(agent)
    discovery = await check_model_discovery(agent)

    assert llm["status"] == "warn"
    assert SENTINEL not in json.dumps(llm)
    assert SENTINEL not in json.dumps(discovery)
