"""The health checks judge the routes in use, and only those (#3873).

On Emma's host every tick reported ``degraded``: ``llm_service`` warned that
``llama_cpp:local`` was unreachable and listed all ten cloud routes, the
active ``anthropic:plan`` among them, as ``skipped``; ``model_discovery``
warned that llama_cpp and RunPod discovery failed. Both providers are standby
routes the operator keeps configured and idle on purpose, and the route the
agent actually used was never checked at all.

These tests build a real ``LLMService`` (offline, no adapter is ever called)
and drive the two checks the health tick runs. The route in use is judged on
the last real call the agent made on it, recorded passively; nothing here,
and nothing in the checks, sends a completion.
"""

from __future__ import annotations

from contextlib import AsyncExitStack, contextmanager
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

from kestrel_sovereign.features.health import checks as health_checks
from kestrel_sovereign.features.health.checks import derive_overall_status
from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.llm.model_cache import (
    VendorDiscoveryOutcome,
    get_shared_model_cache,
)
from kestrel_sovereign.llm.output_ceiling import attach_stop_reason
from kestrel_sovereign.llm.remote_backend import REMOTE_ROUTE_NAME
from kestrel_sovereign.llm.service import LLMService, LLMServiceError
from tests.utils.process_local_llm_service import process_local_service

ACTIVE = "anthropic:plan"
LOCAL_STANDBY = "llama_cpp:local"
RENTED_STANDBY = "runpod:vllm-2f8vdgn"


def _route(name, *, local=False):
    vendor, route = name.split(":", 1)
    return {
        "name": name,
        "vendor": vendor,
        "route": route,
        "model": "auto",
        "is_local": local,
        "is_cloud": not local,
    }


def _agent(*, pin=ACTIVE, reachability=None):
    """An agent pinned to ``pin`` with two standby routes configured."""

    service = process_local_service([
        _route(ACTIVE),
        _route(LOCAL_STANDBY, local=True),
        _route(RENTED_STANDBY),
        _route("openai:api"),
    ])
    vendor, route = pin.split(":", 1)
    service._mandate_preference = {
        "vendor": vendor, "route": route, "model": "claude-opus-5",
    }
    service.reachability = reachability if reachability is not None else [
        {"name": ACTIVE, "status": "skipped",
         "message": "cloud route not probed at startup"},
        {"name": LOCAL_STANDBY, "status": "unreachable",
         "message": "local route probe failed"},
        {"name": RENTED_STANDBY, "status": "skipped",
         "message": "cloud route not probed at startup"},
        {"name": "openai:api", "status": "skipped",
         "message": "cloud route not probed at startup"},
    ]
    return SimpleNamespace(llm_service=service)


def _discovery(**failures):
    outcomes = {"anthropic": VendorDiscoveryOutcome.ok()}
    outcomes.update(
        {vendor: VendorDiscoveryOutcome.failed(error) for vendor, error in failures.items()}
    )
    get_shared_model_cache().set([], outcomes=outcomes)


def _by_name(entries):
    return {entry["name"]: entry for entry in entries}


@contextmanager
def _sending_nothing():
    """A health check that tried a real completion, or touched the private
    lease, would fail inside this block."""

    def refuse(*_args, **_kwargs):
        raise AssertionError("a #3873 health check sent an LLM request")

    with patch.object(LLMService, "_run_provider_attempt", refuse), \
            patch.object(LLMService, "_stream_adapter_with_usage", refuse), \
            patch.object(LLMService, "_remote_route_attempt", refuse):
        yield


async def check_llm_service(agent):
    with _sending_nothing():
        return await health_checks.check_llm_service(agent)


async def check_model_discovery(agent):
    with _sending_nothing():
        return await health_checks.check_model_discovery(agent)


@pytest.mark.asyncio
async def test_an_active_route_with_no_call_is_reported_unprobed():
    result = await check_llm_service(_agent())

    [in_use] = result["details"]["routes_in_use"]
    assert in_use == {"name": ACTIVE, "status": "unprobed", "evidence": None}
    assert f"route in use {ACTIVE} unprobed" in result["message"]
    assert "this check sends none" in result["message"]
    # Unprobed is said plainly, not counted as a failure.
    assert result["status"] == "pass"


@pytest.mark.asyncio
async def test_the_active_route_is_judged_on_its_last_real_call():
    agent = _agent()
    ledger = agent.llm_service.route_calls

    ledger.record(ACTIVE)
    answered = await check_llm_service(agent)
    assert answered["status"] == "pass"
    entry = _by_name(answered["details"]["routes_in_use"])[ACTIVE]
    assert (entry["status"], entry["evidence"]) == ("ok", "last_call")
    assert f"route in use {ACTIVE}: last call answered" in answered["message"]

    class APIConnectionError(Exception):
        pass

    ledger.record(ACTIVE, APIConnectionError("Connection error."))
    failed = await check_llm_service(agent)
    assert failed["status"] == "warn"
    entry = _by_name(failed["details"]["routes_in_use"])[ACTIVE]
    assert (entry["status"], entry["error_type"]) == ("failed", "APIConnectionError")
    assert f"last call on route in use {ACTIVE} failed" in failed["message"]
    assert "APIConnectionError" in failed["message"]

    ledger.record(ACTIVE)
    assert (await check_llm_service(agent))["status"] == "pass"


@pytest.mark.asyncio
async def test_standby_providers_do_not_degrade_the_tick():
    _discovery(
        llama_cpp="ConnectError: [Errno 61] Connection refused",
        runpod="HTTPStatusError: 404 Not Found",
    )
    agent = _agent()
    agent.llm_service.route_calls.record(ACTIVE)

    llm = await check_llm_service(agent)
    discovery = await check_model_discovery(agent)

    standby = _by_name(llm["details"]["standby_routes"])
    assert set(standby) == {LOCAL_STANDBY, RENTED_STANDBY, "openai:api"}
    assert standby[LOCAL_STANDBY] == {
        "name": LOCAL_STANDBY, "status": "standby", "startup_probe": "unreachable",
    }
    assert llm["status"] == "pass"
    assert discovery["status"] == "pass"
    assert discovery["details"]["standby_vendors"] == ["llama_cpp", "runpod"]
    assert discovery["details"]["vendors_in_use"] == ["anthropic"]
    # The failures are still reported, only not judged.
    assert set(discovery["details"]["failed_vendors"]) == {"llama_cpp", "runpod"}
    assert "llama_cpp, runpod" in discovery["message"]
    assert derive_overall_status([llm, discovery]) == "healthy"


@pytest.mark.asyncio
async def test_a_failing_provider_a_route_in_use_selects_still_degrades_the_tick():
    """The same two providers, now with the agent pinned to the local one."""

    _discovery(
        llama_cpp="ConnectError: [Errno 61] Connection refused",
        runpod="HTTPStatusError: 404 Not Found",
    )
    agent = _agent(pin=LOCAL_STANDBY)

    llm = await check_llm_service(agent)
    discovery = await check_model_discovery(agent)

    [in_use] = llm["details"]["routes_in_use"]
    assert in_use == {
        "name": LOCAL_STANDBY, "status": "unreachable", "evidence": "startup_probe",
    }
    assert llm["status"] == "warn"
    assert ACTIVE in _by_name(llm["details"]["standby_routes"])
    # llama_cpp is the pinned vendor: its catalog failure puts the pin at risk.
    assert discovery["status"] == "fail"
    assert discovery["details"]["standby_vendors"] == ["runpod"]
    assert derive_overall_status([llm, discovery]) == "degraded"


@pytest.mark.asyncio
async def test_an_unpinned_vendor_a_route_in_use_selects_warns():
    """No mandate: every route_priority route is in use, so each counts."""

    _discovery(openai="AuthenticationError: 401")
    agent = _agent()
    service = agent.llm_service
    service._mandate_preference = {}
    service.config = {"route_priority": [ACTIVE, "openai:api"]}

    discovery = await check_model_discovery(agent)

    assert service.routes_in_use() == [ACTIVE, "openai:api"]
    assert discovery["status"] == "warn"
    assert discovery["details"]["standby_vendors"] == []
    assert "Model discovery failed for 1 vendor(s): openai" in discovery["message"]


@pytest.mark.asyncio
async def test_a_mandate_no_route_can_serve_fails_the_check():
    agent = _agent()
    agent.llm_service._mandate_preference = {
        "vendor": "gemini", "route": "api", "model": "gemini-3",
    }

    result = await check_llm_service(agent)

    assert result["status"] == "fail"
    assert "no route can serve the next turn" in result["message"]
    assert "routing_error" in result["details"]


# ---------------------------------------------------------------------------
# Real calls, then the check: a private lease route, and a route disabled by
# an auth failure, are judged on what the agent's own calls observed.
# ---------------------------------------------------------------------------


class _Route:
    """A configured route whose adapter answers, or raises ``failure``."""

    def __init__(self, name, *, failure=None):
        self.name = name
        self.vendor, _, self.route = name.partition(":")
        self.failure = failure
        self.calls = 0
        self.adapter = Mock()
        self.adapter.create_messages = Mock(
            return_value=[{"role": "user", "content": "hi"}]
        )
        self.adapter.get_response = self._get_response

    async def _get_response(self, *_args, **_kwargs):
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        return attach_stop_reason(
            LLMResponse(content=f"from {self.name}", input_tokens=1, output_tokens=1),
            "stop",
        )

    def as_provider(self):
        return {
            "name": self.name,
            "vendor": self.vendor,
            "route": self.route,
            "client": AsyncMock(),
            "adapter": self.adapter,
            "model": "m",
            "is_cloud": True,
            "is_local": False,
            "base_url": None,
        }


class _Unauthorized(Exception):
    status_code = 401


class _LeaseAdapter:
    """The private lease route's adapter, failing every call it is sent."""

    def __init__(self, failure):
        self.failure = failure
        self.calls = 0

    def create_messages(self, *, user_prompt=None, system_prompt=None, **_):
        return [{"role": "user", "content": user_prompt or ""}]

    async def get_response(self, **_kwargs):
        self.calls += 1
        raise self.failure


class _LeaseClient:
    async def close(self):
        pass


def _ready_lease(*, expires_in=timedelta(minutes=10)):
    now = datetime.now(UTC)
    return InferenceLease(
        lease_id="lease-1",
        quote_id="quote-1",
        request_id="request-1",
        owner_id="agent-1",
        provider_name="runpod",
        state=InferenceLeaseState.READY,
        model="qwen3:8b",
        runtime="ollama",
        privacy=InferencePrivacy.AUTHENTICATED_ENDPOINT,
        created_at=now - timedelta(minutes=20),
        updated_at=now - timedelta(minutes=20),
        expires_at=now + expires_in,
        hourly_cost_usd=Decimal("0.50"),
        estimated_total_cost_usd=Decimal("0.20"),
        route=InferenceRoute(
            endpoint=SecretStr("https://private.example.test/v1"),
            model="qwen3:8b",
            api_key=SecretStr("route-secret"),
        ),
    )


@pytest.fixture
async def real_service():
    """Build a real ``LLMService`` over fake routes; every one is closed."""

    async with AsyncExitStack() as services:

        def build(routes, *, route_priority):
            registry = Mock()
            registry.initialize_providers = Mock(return_value=[])
            registry.get_providers_with_pattern = Mock(return_value=[])
            registry.get_local_providers = Mock(return_value=[])
            with patch(
                "kestrel_sovereign.llm.service.load_section",
                return_value={"route_priority": list(route_priority)},
            ), patch(
                "kestrel_sovereign.llm.service.load_config",
                return_value={"defaults": {}, "mandates": {}},
            ), patch(
                "kestrel_sovereign.llm.service.ProviderRegistry",
                return_value=registry,
            ):
                service = LLMService()
            services.push_async_callback(service.close)
            service._usage_db = None
            service._db_initialized = False
            service._usage_database_url = None
            service._db_backend = "sqlite"
            service.providers = [route.as_provider() for route in routes]
            return service

        yield build


async def _activate_lease(service, monkeypatch, adapter, lease):
    monkeypatch.setattr(
        "kestrel_sovereign.llm.remote_backend.openai.AsyncOpenAI",
        lambda **_kwargs: _LeaseClient(),
    )

    async def touch(lease_id):
        return service._remote_lease

    await service.activate_inference_lease(
        lease,
        capabilities=("chat", "streaming", "tools", "structured_output"),
        touch_lease=touch,
    )
    service._remote_adapter = adapter


@pytest.mark.asyncio
async def test_a_failing_private_lease_route_is_the_route_judged(
    real_service, monkeypatch
):
    """The lease route takes every call and has no cloud fallback.

    Its failure is what the next turn meets, so it is the route the check
    judges; the cloud routes behind it are standby while the lease holds.
    """

    cloud = _Route("anthropic:api")
    service = real_service([cloud], route_priority=["anthropic:api"])
    adapter = _LeaseAdapter(ConnectionError("private route unreachable"))
    await _activate_lease(service, monkeypatch, adapter, _ready_lease())
    try:
        assert service.routes_in_use() == [REMOTE_ROUTE_NAME]
        with pytest.raises(LLMServiceError, match="no cloud fallback"):
            await service.generate(system_prompt="s", user_prompt="hi")
        assert (adapter.calls, cloud.calls) == (1, 0)

        result = await check_llm_service(SimpleNamespace(llm_service=service))
    finally:
        await service.deactivate_inference_lease("lease-1")

    assert result["status"] == "warn"
    [in_use] = result["details"]["routes_in_use"]
    assert (in_use["name"], in_use["status"]) == (REMOTE_ROUTE_NAME, "failed")
    assert in_use["error_type"] == "ConnectionError"
    assert f"last call on route in use {REMOTE_ROUTE_NAME} failed" in result["message"]
    assert _by_name(result["details"]["standby_routes"])["anthropic:api"][
        "status"
    ] == "standby"


@pytest.mark.asyncio
async def test_a_lease_route_refusing_before_any_adapter_runs_is_recorded(
    real_service, monkeypatch
):
    """An expired lease fails every call without reaching its adapter."""

    service = real_service(
        [_Route("anthropic:api")], route_priority=["anthropic:api"]
    )
    adapter = _LeaseAdapter(AssertionError("the adapter must not be reached"))
    await _activate_lease(service, monkeypatch, adapter, _ready_lease())
    service._remote_lease = _ready_lease(expires_in=-timedelta(seconds=1))
    try:
        with pytest.raises(LLMServiceError, match="lease expired"):
            await service.generate(system_prompt="s", user_prompt="hi")
        result = await check_llm_service(SimpleNamespace(llm_service=service))
    finally:
        await service.deactivate_inference_lease("lease-1", require_active=False)

    assert adapter.calls == 0
    assert result["status"] == "warn"
    [in_use] = result["details"]["routes_in_use"]
    assert (in_use["name"], in_use["status"]) == (REMOTE_ROUTE_NAME, "failed")


@pytest.mark.asyncio
async def test_a_lease_the_next_call_would_not_take_leaves_routing_alone(
    real_service, monkeypatch
):
    """A mandate pinned to a vendor bypasses the lease, as dispatch does."""

    service = real_service(
        [_Route("anthropic:api"), _Route("openai:api")],
        route_priority=["anthropic:api", "openai:api"],
    )
    adapter = _LeaseAdapter(AssertionError("the adapter must not be reached"))
    await _activate_lease(service, monkeypatch, adapter, _ready_lease())
    try:
        service._mandate_preference = {
            "vendor": "openai", "route": "api", "model": "m",
        }
        assert service.routes_in_use() == ["openai:api"]
        assert await service.generate(system_prompt="s", user_prompt="hi") == (
            "from openai:api"
        )
    finally:
        await service.deactivate_inference_lease("lease-1")
    assert adapter.calls == 0


@pytest.mark.asyncio
async def test_a_route_disabled_by_an_auth_failure_is_not_standby(real_service):
    """The preferred route fails auth and is disabled; a fallback answers.

    Routing then skips the disabled route, so it is no longer a route in use,
    but it is not a standby route either: until its key is rotated and the
    service restarted, no turn can use it.
    """

    preferred = _Route("anthropic:api", failure=_Unauthorized("invalid api key"))
    fallback = _Route("openai:api")
    service = real_service(
        [preferred, fallback], route_priority=["anthropic:api", "openai:api"]
    )

    answer = await service.generate_with_messages(
        messages=[{"role": "user", "content": "hi"}]
    )
    assert answer == "from openai:api"
    assert set(service.disabled_routes()) == {"anthropic:api"}
    assert service.routes_in_use() == ["openai:api"]

    result = await check_llm_service(SimpleNamespace(llm_service=service))

    assert result["status"] == "warn"
    [disabled] = result["details"]["disabled_routes"]
    assert (disabled["name"], disabled["status"]) == ("anthropic:api", "disabled")
    assert "invalid api key" in disabled["reason"]
    assert disabled["last_call"]["status"] == "failed"
    assert disabled["last_call"]["status_code"] == 401
    assert "anthropic:api" not in _by_name(result["details"]["standby_routes"])
    [in_use] = result["details"]["routes_in_use"]
    assert (in_use["name"], in_use["status"]) == ("openai:api", "ok")
    assert "route anthropic:api disabled this session" in result["message"]

    # The disabled route's vendor is judged by discovery, not called standby.
    _discovery(anthropic="AuthenticationError: 401")
    discovery = await check_model_discovery(SimpleNamespace(llm_service=service))
    assert discovery["status"] == "warn"
    assert discovery["details"]["standby_vendors"] == []
