"""``llm_service`` and ``model_discovery`` read route health from the call ledger (#3912).

Health reads what dispatch did: the ``a2a_llm_calls`` rows LLMService writes
for every attempt. A configured route with no rows and nothing selecting it is
``standby`` and degrades nothing; the route the agent's selection names, with
no rows, is reported "active route unprobed"; a route whose newest call failed,
or that was disabled after a failure, warns. A failed startup reachability
probe stands only while its route has no call to judge it by. Lease-route rows
count for the installed lease by its id, or by position once it has named
itself; rows the privacy gate stripped of their lease id are reported, never
silently dropped.

Every ledger here is a real ``ObservabilityStore`` on a temp SQLite database.
Providers are mocked: no network, no key, no paid completion.
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
    ObservabilityStore,
)
from kestrel_sovereign.features.health import checks as health_checks
from kestrel_sovereign.features.health.checks import (
    LLM_ROUTE_EVIDENCE_WINDOW,
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
async def owned_services():
    async with AsyncExitStack() as stack:
        yield stack


def _fake_service(
    providers,
    *,
    preference=None,
    disabled=None,
    reachability=None,
):
    """What health reads off an LLM service, and nothing else."""
    preference = preference or {"vendor": None, "route": None, "model": None}
    return SimpleNamespace(
        providers=[dict(p) for p in providers],
        get_model_preference=lambda: dict(preference),
        get_active_model_id=lambda: preference.get("model") or "auto",
        reachability=reachability,
        _disabled_routes=dict(disabled or {}),
        _mandate_load_error=None,
        _remote_lease=None,
    )


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


class _ScriptedRoute:
    """A mocked provider route that answers or raises, one outcome per call."""

    def __init__(self, provider, outcomes):
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


def _real_service(stack, routes, store):
    """A real LLMService over mocked routes, writing to ``store``."""
    registry = Mock()
    registry.initialize_providers = Mock(return_value=[])
    registry.get_providers_with_pattern = Mock(return_value=[])
    registry.get_local_providers = Mock(return_value=[])
    with patch(
        "kestrel_sovereign.llm.service.load_section", return_value={}
    ), patch(
        "kestrel_sovereign.llm.service.load_config",
        return_value={"defaults": {}, "mandates": {}},
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


def _lease(lease_id):
    now = datetime.now(UTC)
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
        created_at=now,
        updated_at=now,
        expires_at=now + timedelta(minutes=10),
        hourly_cost_usd=Decimal("0.50"),
        estimated_total_cost_usd=Decimal("0.20"),
        route=InferenceRoute(
            endpoint=SecretStr(f"https://{lease_id}.private.example.test/v1"),
            model="llama-3",
            api_key=SecretStr("route-secret"),
        ),
    )


async def _activate(service, lease_id, outcomes):
    async def touch(requested):
        assert requested == lease_id
        return service._remote_lease

    await service.activate_inference_lease(
        _lease(lease_id), capabilities=("chat",), touch_lease=touch
    )
    service._remote_adapter = _LeaseAdapter(outcomes)


@pytest.fixture
def lease_clients(monkeypatch):
    monkeypatch.setattr(
        "kestrel_sovereign.llm.remote_backend.openai.AsyncOpenAI",
        lambda **_kwargs: _LeaseClient(),
    )


# ---------------------------------------------------------------------------
# The active route's newest outcome
# ---------------------------------------------------------------------------


async def test_a_failed_newest_call_warns_and_a_later_success_passes(
    ledger, owned_services
):
    """Rows written by real dispatch, not by hand."""
    anthropic = _ScriptedRoute(ANTHROPIC, [
        "first answer",
        LLMProviderError("anthropic", "overloaded"),
        "recovered",
    ])
    service = _real_service(owned_services, [anthropic], ledger)
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


async def test_each_model_on_a_route_is_judged_by_its_own_newest_call(ledger):
    service = _fake_service([ANTHROPIC])
    await _log(ledger, "anthropic:plan", "claude-haiku", success=True)
    await _log(ledger, "anthropic:plan", "claude-opus-5", success=False)
    await _log(ledger, "anthropic:plan", "claude-haiku", success=True)

    result = await check_llm_service(_agent(service, ledger))

    assert result["status"] == "warn"
    route = _route(result, "anthropic:plan")
    assert route["state"] == "failing"
    by_model = {m["model"]: m["last_call_succeeded"] for m in route["models"]}
    assert by_model == {"claude-haiku": True, "claude-opus-5": False}


async def test_only_this_agents_rows_and_the_window_count(ledger, monkeypatch):
    service = _fake_service(
        [ANTHROPIC, LLAMA],
        preference={"vendor": "anthropic", "route": "plan", "model": "claude-opus-5"},
    )
    # Another agent's failure on the same shared table is not this agent's.
    await _log(ledger, "anthropic:plan", "claude-opus-5", success=False, did=OTHER_DID)
    # A failure older than the window is not evidence of the route now.
    stale = datetime.now(UTC) - LLM_ROUTE_EVIDENCE_WINDOW - timedelta(minutes=5)
    monkeypatch.setattr(ledger, "now_utc", lambda: stale)
    await _log(ledger, "llama_cpp:local", "qwen3-8b", success=False)
    monkeypatch.undo()

    result = await check_llm_service(_agent(service, ledger))

    assert result["status"] == "pass"
    assert _route(result, "anthropic:plan")["state"] == "unprobed"
    assert _route(result, "llama_cpp:local")["state"] == "standby"
    assert result["details"]["route_evidence"]["rows_read"] == 0


# ---------------------------------------------------------------------------
# Standby and "active route unprobed"
# ---------------------------------------------------------------------------


async def test_standby_providers_do_not_degrade_tick_status(ledger):
    """The #3873 host: llama_cpp refuses connections and runpod 404s discovery,
    while anthropic:plan, the selected route, serves every call."""
    service = _fake_service(
        [ANTHROPIC, LLAMA, RUNPOD],
        preference={"vendor": "anthropic", "route": "plan", "model": "claude-opus-5"},
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


async def test_a_failing_provider_that_served_traffic_still_degrades(ledger):
    """The same standby route, once dispatch actually used it and it failed."""
    service = _fake_service(
        [ANTHROPIC, LLAMA],
        preference={"vendor": "anthropic", "route": "plan", "model": "claude-opus-5"},
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


async def test_an_unreachable_selected_route_still_warns(ledger):
    service = _fake_service(
        [LLAMA],
        preference={"vendor": "llama_cpp", "route": "local", "model": "qwen3-8b"},
        reachability=[{"name": "llama_cpp:local", "status": "unreachable"}],
    )
    result = await check_llm_service(_agent(service, ledger))

    assert result["status"] == "warn"
    assert _route(result, "llama_cpp:local")["state"] == "unprobed"


async def test_a_call_after_a_failed_startup_probe_supersedes_it(
    ledger, owned_services
):
    """The startup probe runs once. A route that recovered must not warn
    until the next restart."""
    llama = _ScriptedRoute(LLAMA, ["local answer", LLMProviderError("llama_cpp", "refused")])
    llama.adapter.probe_reachable = AsyncMock(return_value=False)
    # The cloud route keeps startup from refusing, and is the fallback when
    # the local route fails at the end.
    anthropic = _ScriptedRoute(ANTHROPIC, [LLMProviderError("anthropic", "overloaded")])
    service = _real_service(owned_services, [llama, anthropic], ledger)
    agent = _agent(service, ledger)

    await verify_llm_providers_reachable(service, env={})
    assert [r["status"] for r in service.reachability] == ["unreachable", "skipped"]

    down = await check_llm_service(agent)
    assert down["status"] == "warn"
    assert _route(down, "llama_cpp:local")["state"] == "unprobed"
    assert "unreachable at startup, no call since: llama_cpp:local" in down["message"]

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


async def test_a_failed_startup_probe_stands_while_its_route_has_no_call(ledger):
    service = _fake_service(
        [ANTHROPIC, LLAMA],
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
    assert _route(result, "llama_cpp:local")["state"] == "unprobed"
    assert "unreachable at startup, no call since: llama_cpp:local" in result["message"]


async def test_a_configured_route_with_no_rows_is_active_route_unprobed(ledger):
    service = _fake_service(
        [ANTHROPIC, LLAMA],
        preference={"vendor": "anthropic", "route": "plan", "model": "claude-opus-5"},
    )
    result = await check_llm_service(_agent(service, ledger))

    assert result["status"] == "pass"
    route = _route(result, "anthropic:plan")
    assert route["state"] == "unprobed"
    assert route["configured_route"] is True
    assert "active route unprobed: anthropic:plan" in result["message"]
    assert _route(result, "llama_cpp:local")["state"] == "standby"


async def test_with_no_mandate_the_default_route_is_the_configured_one(ledger):
    service = _fake_service([ANTHROPIC, LLAMA])
    result = await check_llm_service(_agent(service, ledger))

    assert _route(result, "anthropic:plan")["state"] == "unprobed"
    assert _route(result, "llama_cpp:local")["state"] == "standby"


async def test_the_pinned_vendor_is_never_standby_for_discovery(ledger):
    service = _fake_service(
        [ANTHROPIC, LLAMA],
        preference={"vendor": "anthropic", "route": "plan", "model": "claude-opus-5"},
    )
    _discovery({"anthropic": "AuthenticationError: 401", "llama_cpp": "refused"})

    discovery = await check_model_discovery(_agent(service, ledger))

    assert discovery["status"] == "fail"
    assert discovery["details"]["pinned_vendor_at_risk"] is True
    assert list(discovery["details"]["failed_vendors"]) == ["anthropic"]
    assert list(discovery["details"]["standby_failed_vendors"]) == ["llama_cpp"]


# ---------------------------------------------------------------------------
# Failure-disabled routes are failing, never standby
# ---------------------------------------------------------------------------


async def test_a_failure_disabled_route_is_failing_despite_older_successes(ledger):
    service = _fake_service(
        [ANTHROPIC, LLAMA],
        preference={"vendor": "anthropic", "route": "plan", "model": "claude-opus-5"},
        disabled={"anthropic:plan": f"401 Unauthorized {SENTINEL}"},
    )
    await _log(ledger, "anthropic:plan", "claude-opus-5", success=True)
    await _log(ledger, "anthropic:plan", "claude-opus-5", success=True)

    result = await check_llm_service(_agent(service, ledger))

    assert result["status"] == "warn"
    assert _route(result, "anthropic:plan")["state"] == "disabled"
    assert "disabled after a failure: anthropic:plan" in result["message"]
    assert SENTINEL not in json.dumps(result)


async def test_a_disabled_route_with_no_rows_is_not_standby(ledger):
    service = _fake_service(
        [ANTHROPIC, LLAMA],
        preference={"vendor": "anthropic", "route": "plan", "model": "claude-opus-5"},
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


async def test_a_route_dispatch_disabled_after_an_auth_failure_is_failing(
    ledger, owned_services
):
    """The real ``_maybe_disable_route``: fallback answered, the route is dead."""
    anthropic = _ScriptedRoute(ANTHROPIC, [
        "first answer",
        LLMProviderError("anthropic", "401 Unauthorized: invalid api key"),
    ])
    openai_route = _ScriptedRoute(
        {"name": "openai:api", "vendor": "openai", "route": "api", "model": "gpt-5"},
        ["fallback answer"],
    )
    service = _real_service(owned_services, [anthropic, openai_route], ledger)

    assert await service.generate_with_messages(messages=list(MESSAGES)) == "first answer"
    assert await service.generate_with_messages(messages=list(MESSAGES)) == "fallback answer"
    assert "anthropic:plan" in service._disabled_routes

    result = await check_llm_service(_agent(service, ledger))

    assert result["status"] == "warn"
    assert _route(result, "anthropic:plan")["state"] == "disabled"
    assert _route(result, "openai:api")["state"] == "ok"


# ---------------------------------------------------------------------------
# The private inference lease route
# ---------------------------------------------------------------------------


async def test_a_replacement_lease_does_not_inherit_the_old_leases_failures(
    ledger, owned_services, lease_clients
):
    """Private-only service: no configured providers, the lease is the route."""
    service = _real_service(owned_services, [], ledger)
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
    ledger, owned_services, lease_clients
):
    """The lease takes precedence over the providers, so its failure counts."""
    anthropic = _ScriptedRoute(ANTHROPIC, [])
    service = _real_service(owned_services, [anthropic], ledger)
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


async def test_lease_rows_naming_another_lease_are_ignored(ledger):
    service = _fake_service([ANTHROPIC])
    service._remote_lease = SimpleNamespace(lease_id="lease-B")
    await _log(ledger, REMOTE_ROUTE_NAME, "llama-3", success=False,
               metadata={"inference_lease_id": "lease-A"})
    await _log(ledger, REMOTE_ROUTE_NAME, "llama-3", success=False,
               metadata={"inference_lease_id": "lease-A"})

    result = await check_llm_service(_agent(service, ledger))

    assert result["status"] == "pass"
    route = _route(result, REMOTE_ROUTE_NAME)
    assert route["state"] == "unprobed"
    assert route["models"] == [] and route["unattributed_models"] == []


async def test_lease_rows_naming_no_lease_are_reported_not_dropped(ledger):
    """A row with no lease id may be this lease's: its failure warns."""
    service = _fake_service([ANTHROPIC])
    service._remote_lease = SimpleNamespace(lease_id="lease-B")
    await _log(ledger, REMOTE_ROUTE_NAME, "llama-3", success=True)
    await _log(ledger, REMOTE_ROUTE_NAME, "llama-3", success=False)

    result = await check_llm_service(_agent(service, ledger))

    assert result["status"] == "warn"
    route = _route(result, REMOTE_ROUTE_NAME)
    assert route["state"] == "unattributed_failing"
    assert route["models"] == []
    [model] = route["unattributed_models"]
    assert (model["calls"], model["failures"]) == (2, 1)
    assert "failing, lease unattributed" in result["message"]
    assert derive_overall_status([result]) == "degraded"

    await _log(ledger, REMOTE_ROUTE_NAME, "llama-3", success=True)
    recovered = await check_llm_service(_agent(service, ledger))
    assert recovered["status"] == "pass"
    assert _route(recovered, REMOTE_ROUTE_NAME)["state"] == "unattributed_ok"


async def test_every_lease_row_after_the_lease_first_named_itself_is_its_own(
    ledger, monkeypatch
):
    """A lease is released only after its calls drain, so no other lease can
    write a row after this lease's first one: an unnamed row then is its own."""
    service = _fake_service([ANTHROPIC])
    service._remote_lease = SimpleNamespace(lease_id="lease-B")
    base = datetime.now(UTC) - timedelta(minutes=30)

    async def log_at(minutes, **kwargs):
        monkeypatch.setattr(ledger, "now_utc", lambda: base + timedelta(minutes=minutes))
        await _log(ledger, REMOTE_ROUTE_NAME, "llama-3", **kwargs)
        monkeypatch.undo()

    # Older than lease B's first named call: may be lease A's. Not counted.
    await log_at(1, success=False)
    await log_at(2, success=True, metadata={"inference_lease_id": "lease-B"})
    # Newer: lease B's, though its lease id did not survive.
    await log_at(3, success=False)

    result = await check_llm_service(_agent(service, ledger))

    assert result["status"] == "warn"
    route = _route(result, REMOTE_ROUTE_NAME)
    assert route["state"] == "failing"
    [model] = route["models"]
    assert (model["calls"], model["failures"]) == (2, 1)
    assert route["unattributed_models"] == []

    await log_at(4, success=True, metadata={"inference_lease_id": "lease-B"})
    recovered = await check_llm_service(_agent(service, ledger))
    assert recovered["status"] == "pass"
    assert _route(recovered, REMOTE_ROUTE_NAME)["state"] == "ok"


async def test_a_failed_lease_call_the_privacy_gate_stripped_still_warns(
    ledger, owned_services, lease_clients
):
    """The production privacy gate elides the whole metadata column, lease id
    included, in temporary storage. Health must not read the failure as no
    call at all."""
    privacy = SimpleNamespace(
        config=PrivacyConfig(storage="temp", processing="cloud")
    )
    service = _real_service(owned_services, [], ledger)
    # Both production privacy hooks, bound as KestrelAgent binds them.
    ledger.set_privacy_config_provider(lambda: privacy.config)
    service.set_force_local_only_provider(
        lambda: not privacy.config.allows_cloud_llm()
    )
    agent = _agent(service, ledger)

    await _activate(service, "lease-A", [RuntimeError("pod gone"), "recovered"])
    try:
        with pytest.raises(LLMServiceError):
            await service.generate_with_messages(messages=list(MESSAGES))
        [row] = await ledger.query_llm_calls(agent_did=DID)
        assert row.provider == REMOTE_ROUTE_NAME and row.success is False
        assert "inference_lease_id" not in row.metadata  # the gate took it

        failing = await check_llm_service(agent)
        assert failing["status"] == "warn"
        assert _route(failing, REMOTE_ROUTE_NAME)["state"] == "unattributed_failing"
        assert derive_overall_status([failing]) == "degraded"

        assert await service.generate_with_messages(messages=list(MESSAGES)) == "recovered"
        recovered = await check_llm_service(agent)
        assert recovered["status"] == "pass"
        assert _route(recovered, REMOTE_ROUTE_NAME)["state"] == "unattributed_ok"
    finally:
        await service.deactivate_inference_lease("lease-A")


async def test_a_lease_failing_after_privacy_tightens_mid_lease_is_failing(
    ledger, owned_services, lease_clients
):
    """The lease named itself before the gate began stripping its id, so the
    stripped failure that followed is attributed to it."""
    privacy = SimpleNamespace(config=get_privacy_preset("normal"))
    service = _real_service(owned_services, [], ledger)
    ledger.set_privacy_config_provider(lambda: privacy.config)
    service.set_force_local_only_provider(
        lambda: not privacy.config.allows_cloud_llm()
    )
    agent = _agent(service, ledger)

    await _activate(service, "lease-A", ["named", RuntimeError("pod gone")])
    try:
        assert await service.generate_with_messages(messages=list(MESSAGES)) == "named"
        privacy.config = PrivacyConfig(storage="temp", processing="cloud")
        with pytest.raises(LLMServiceError):
            await service.generate_with_messages(messages=list(MESSAGES))
        stripped, named = await ledger.query_llm_calls(agent_did=DID)
        assert "inference_lease_id" not in stripped.metadata
        assert named.metadata["inference_lease_id"] == "lease-A"

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


async def test_a_replacement_lease_is_judged_by_its_own_first_unnamed_call(
    ledger, owned_services, lease_clients
):
    """Where the gate strips every lease id, the newest lease-route call is
    the replacement's as soon as it has served one."""
    privacy = SimpleNamespace(
        config=PrivacyConfig(storage="temp", processing="cloud")
    )
    service = _real_service(owned_services, [], ledger)
    ledger.set_privacy_config_provider(lambda: privacy.config)
    service.set_force_local_only_provider(
        lambda: not privacy.config.allows_cloud_llm()
    )
    agent = _agent(service, ledger)

    await _activate(service, "lease-A", [RuntimeError("pod gone")])
    with pytest.raises(LLMServiceError):
        await service.generate_with_messages(messages=list(MESSAGES))
    await service.deactivate_inference_lease("lease-A")

    await _activate(service, "lease-B", ["from lease B"])
    try:
        # Until lease B serves, lease A's stripped failure cannot be told
        # apart from lease B's, so it warns: failing closed, not open. Rows
        # that keep their lease id are never mistaken this way.
        before = await check_llm_service(agent)
        assert _route(before, REMOTE_ROUTE_NAME)["state"] == "unattributed_failing"

        assert await service.generate_with_messages(messages=list(MESSAGES)) == "from lease B"
        after = await check_llm_service(agent)
    finally:
        await service.deactivate_inference_lease("lease-B")

    assert after["status"] == "pass"
    assert _route(after, REMOTE_ROUTE_NAME)["state"] == "unattributed_ok"


# ---------------------------------------------------------------------------
# Evidence that cannot be read is said so, never treated as standby
# ---------------------------------------------------------------------------


async def test_an_unreadable_ledger_warns_without_its_error_text(ledger, monkeypatch):
    service = _fake_service(
        [ANTHROPIC, LLAMA],
        preference={"vendor": "anthropic", "route": "plan", "model": "claude-opus-5"},
    )

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


async def test_a_wedged_ledger_read_is_bounded(ledger, monkeypatch):
    service = _fake_service([ANTHROPIC])

    async def wedged(**_kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(ledger, "query_llm_calls", wedged)
    monkeypatch.setattr(health_checks, "DATABASE_HEALTH_CHECK_TIMEOUT_S", 0.01)

    result = await asyncio.wait_for(check_llm_service(_agent(service, ledger)), 1)

    assert result["status"] == "warn"
    assert "read timed out" in result["message"]


async def test_without_a_ledger_a_standby_route_is_not_claimed(ledger):
    service = _fake_service(
        [ANTHROPIC, LLAMA],
        preference={"vendor": "anthropic", "route": "plan", "model": "claude-opus-5"},
        reachability=[{"name": "llama_cpp:local", "status": "unreachable"}],
    )
    _discovery({"llama_cpp": "refused"})
    agent = SimpleNamespace(did=DID, llm_service=service)

    llm = await check_llm_service(agent)
    discovery = await check_model_discovery(agent)

    assert llm["status"] == "warn"
    assert _route(llm, "llama_cpp:local")["state"] == "unknown"
    assert "no call evidence: no call ledger" in llm["message"]
    assert discovery["status"] == "warn"


# ---------------------------------------------------------------------------
# Content stays out
# ---------------------------------------------------------------------------


async def test_no_health_output_carries_prompt_response_or_error_text(ledger):
    service = _fake_service(
        [ANTHROPIC, LLAMA, RUNPOD],
        preference={"vendor": "anthropic", "route": "plan", "model": "claude-opus-5"},
        disabled={"runpod:vllm": f"unauthorized {SENTINEL}"},
    )
    service._remote_lease = SimpleNamespace(lease_id="lease-A")
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
    [stored, *_] = await ledger.query_llm_calls(agent_did=DID)
    assert SENTINEL in (stored.user_prompt_preview or "")  # the rows do carry it
    _discovery({"runpod": "404"})
    agent = _agent(service, ledger)

    llm = await check_llm_service(agent)
    discovery = await check_model_discovery(agent)

    assert llm["status"] == "warn"
    assert SENTINEL not in json.dumps(llm)
    assert SENTINEL not in json.dumps(discovery)
