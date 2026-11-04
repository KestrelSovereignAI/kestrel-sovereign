"""The LLM health checks judge only the routes a turn would try (#3912).

Before this, ``llm_service`` reported every cloud route, the active
``anthropic:plan`` included, as ``skipped`` at startup and never looked at it
again, while a standby provider the operator keeps configured and idle on
purpose (``llama_cpp`` on :8001, a RunPod endpoint) degraded every health tick
through both ``llm_service`` and ``model_discovery``.

Every service here is a real ``LLMService`` with offline routes. No adapter
is ever called: route outcomes come from the service's own recording points,
driven with stub attempts, and the checks send nothing.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from kestrel_sdk.llm.types import BackendType

from kestrel_sovereign.features.health.checks import (
    check_llm_service,
    check_model_discovery,
    derive_overall_status,
)
from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.llm.invocation_context import LLMInvocationContext
from kestrel_sovereign.llm.model_cache import (
    VendorDiscoveryOutcome,
    get_shared_model_cache,
)
from kestrel_sovereign.llm.output_ceiling import attach_stop_reason
from kestrel_sovereign.llm.remote_backend import REMOTE_ROUTE_NAME
from kestrel_sovereign.llm.route_coverage import RouteCoverage
from kestrel_sovereign.llm.service import LLMServiceError
from kestrel_sovereign.llm.streaming import LLMStreamingError
from tests.unit.test_private_inference_routing import _lease
from tests.utils.process_local_llm_service import process_local_service

PLAN = {
    "name": "anthropic:plan", "vendor": "anthropic", "route": "plan",
    "model": "claude-opus-5", "is_cloud": True, "is_local": False,
}
API = {
    "name": "anthropic:api", "vendor": "anthropic", "route": "api",
    "model": "claude-opus-5", "is_cloud": True, "is_local": False,
}
LLAMA = {
    "name": "llama_cpp:local", "vendor": "llama_cpp", "route": "local",
    "model": "glm-4.6", "is_cloud": False, "is_local": True,
}
RUNPOD = {
    "name": "runpod:vllm-2f8vdgn", "vendor": "runpod", "route": "vllm-2f8vdgn",
    "model": "qwen3", "is_cloud": True, "is_local": False,
}
PINNED_PLAN = {"vendor": "anthropic", "route": "plan", "model": "claude-opus-5"}

# What the startup probe recorded on the host #3873 was filed from: the
# standby llama_cpp daemon refused the connection, and no cloud route is ever
# probed at startup.
STARTUP_PROBES = [
    {"name": "anthropic:plan", "status": "skipped", "message": "cloud route not probed at startup"},
    {"name": "llama_cpp:local", "status": "unreachable", "message": "local route probe failed"},
    {"name": "runpod:vllm-2f8vdgn", "status": "skipped", "message": "cloud route not probed at startup"},
]


def _service(providers=(PLAN, LLAMA, RUNPOD), *, mandate=PINNED_PLAN, config=None):
    service = process_local_service([dict(provider) for provider in providers])
    service.config = dict(config or {})
    service._mandate_preference = dict(mandate) if mandate else {}
    service.reachability = [
        dict(probe) for probe in STARTUP_PROBES
        if any(probe["name"] == provider["name"] for provider in providers)
    ]
    return service


def _agent(service):
    return SimpleNamespace(llm_service=service)


def _with_lease(service, *, accepting=True, touch=None):
    """Activate a ready private inference lease on ``service``, offline."""
    lease = _lease()
    service._remote_lease = lease
    service._remote_client = object()
    service._remote_adapter = object()
    service._remote_route_condition = asyncio.Condition()
    service._remote_inflight = 0
    service._remote_accepting = accepting
    service._remote_capabilities = frozenset({"chat"})
    service._remote_route_epoch = 0
    service._backend = BackendType.REMOTE_GPU

    async def touch_current(lease_id):
        return service._remote_lease

    service._remote_touch_lease = touch or touch_current
    return lease


def _discovery(**failures):
    outcomes = {"anthropic": VendorDiscoveryOutcome.ok()}
    outcomes.update(
        {vendor: VendorDiscoveryOutcome.failed(error) for vendor, error in failures.items()}
    )
    get_shared_model_cache().set([], outcomes=outcomes)


def _no_telemetry(service):
    """Let a real attempt finalize without an observability store."""

    async def no_sink(*args, **kwargs):
        return None

    service._log_llm_call = no_sink
    service._record_model_usage = no_sink


async def _attempt(service, route, *, fails):
    """One real non-streaming attempt on ``route`` through the dispatch runner."""

    async def call():
        if fails:
            raise ConnectionError("provider refused")
        return LLMResponse(content="ok")

    _no_telemetry(service)
    if fails:
        with pytest.raises(ConnectionError):
            await service._run_provider_attempt(
                call(), route, "claude-opus-5", path="test",
                invocation_context=LLMInvocationContext(),
            )
    else:
        await service._run_provider_attempt(
            call(), route, "claude-opus-5", path="test",
            invocation_context=LLMInvocationContext(),
        )


class _AnsweringRoute:
    """An offline non-streaming route whose response stopped for ``stop_reason``."""

    def __init__(self, stop_reason):
        self.stop_reason = stop_reason

    async def get_response(self, **kwargs):
        if isinstance(self.stop_reason, BaseException):
            raise self.stop_reason
        return attach_stop_reason(
            LLMResponse(content="answer", input_tokens=3, output_tokens=2),
            self.stop_reason,
        )


class _StreamedRoute:
    """An offline usage stream on the active route: text, then how it ends.

    ``ending`` is ``"stop"`` or ``"length"`` (a terminal response with that
    stop reason), or an exception the stream raises after its text. Like the
    Anthropic adapter, it fills the service's usage sink as tokens arrive.
    """

    supports_partial_usage_flush = True

    def __init__(self, ending):
        self.ending = ending

    async def get_streaming_response_with_tools(self, *, usage_sink=None, **kwargs):
        yield "partial "
        if usage_sink is not None:
            usage_sink["input_tokens"] = 100
            usage_sink["output_tokens"] = 3
        if isinstance(self.ending, BaseException):
            raise self.ending
        yield attach_stop_reason(
            LLMResponse(content="partial ", input_tokens=100, output_tokens=3),
            self.ending,
        )


def _serve_route(service, route):
    """Serve ``service``'s pinned route with ``route``; returns its billed calls.

    The service has no private inference lease, so the public call APIs walk
    the configured routes. Each call record is what ``_log_llm_call`` was
    given.
    """

    billed = []

    async def log_llm_call(**kwargs):
        billed.append(kwargs)

    async def no_sink(*args, **kwargs):
        return None

    service._log_llm_call = log_llm_call
    service._record_model_usage = no_sink
    service._remote_route_condition = asyncio.Condition()
    service._remote_client = None
    service._backend = None
    service._remote_inflight = 0
    service._remote_accepting = False
    pinned = next(p for p in service.providers if p["name"] == "anthropic:plan")
    pinned["adapter"] = route
    pinned["client"] = object()
    return billed


def _stream(service):
    """The public tool-detecting stream, as a turn calls it."""
    return service.stream_with_tool_detection(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "noop"}}],
    )


async def _drain(service):
    return [item async for item in _stream(service)]


def _in_use(result):
    return [(route["name"], route["status"]) for route in result["details"]["routes_in_use"]]


def _standby(result):
    return [route["name"] for route in result["details"]["standby_routes"]]


class TestActiveRouteCoverage:
    @pytest.mark.asyncio
    async def test_an_uncalled_active_cloud_route_is_reported_unprobed_by_name(self):
        result = await check_llm_service(_agent(_service()))

        assert _in_use(result) == [("anthropic:plan", "unprobed")]
        assert "active route anthropic:plan unprobed" in result["message"]
        # Explicit, and not a failure: the check sends no completion to find out.
        assert result["status"] == "pass"

    @pytest.mark.asyncio
    async def test_the_active_route_is_judged_on_its_last_real_call(self):
        service = _service()

        await _attempt(service, "anthropic:plan", fails=False)
        answered = await check_llm_service(_agent(service))
        await _attempt(service, "anthropic:plan", fails=True)
        failed = await check_llm_service(_agent(service))
        await _attempt(service, "anthropic:plan", fails=False)
        recovered = await check_llm_service(_agent(service))

        assert _in_use(answered) == [("anthropic:plan", "ok")]
        assert answered["status"] == "pass"
        assert "active route anthropic:plan: last call answered" in answered["message"]
        assert _in_use(failed) == [("anthropic:plan", "failed")]
        assert failed["status"] == "warn"
        assert "active route anthropic:plan: last call failed" in failed["message"]
        assert recovered["status"] == "pass"

    @pytest.mark.asyncio
    async def test_the_outcome_record_holds_no_error_text(self):
        service = _service()
        await _attempt(service, "anthropic:plan", fails=True)

        result = await check_llm_service(_agent(service))

        assert "provider refused" not in repr(result)
        assert set(result["details"]["routes_in_use"][0]["last_call"]) == {
            "status", "at", "age_seconds",
        }

    @pytest.mark.asyncio
    async def test_the_check_dispatches_nothing(self, monkeypatch):
        service = _service()
        dispatched = []

        def refuse(*args, **kwargs):
            dispatched.append(args)
            raise AssertionError("a health check sent a request")

        monkeypatch.setattr(service, "_run_provider_attempt", refuse)
        monkeypatch.setattr(service, "_stream_adapter_with_usage", refuse)
        for provider in service.providers:
            provider["adapter"] = SimpleNamespace(
                get_response=refuse, probe_reachable=refuse, list_models=refuse,
            )

        await check_llm_service(_agent(service))
        await check_model_discovery(_agent(service))

        assert dispatched == []

    @pytest.mark.asyncio
    async def test_a_local_route_in_use_is_judged_on_its_startup_probe(self):
        service = _service(mandate={"vendor": "llama_cpp", "route": "local", "model": "glm-4.6"})

        unreachable = await check_llm_service(_agent(service))
        service.reachability[1]["status"] = "reachable"
        reachable = await check_llm_service(_agent(service))

        assert _in_use(unreachable) == [("llama_cpp:local", "unreachable")]
        assert unreachable["status"] == "warn"
        assert _in_use(reachable) == [("llama_cpp:local", "reachable")]
        assert reachable["status"] == "pass"

    @pytest.mark.asyncio
    async def test_fallback_routes_in_the_chain_are_judged(self):
        """An unpinned turn would try each route_priority route in order."""
        service = _service(
            mandate=None,
            config={"route_priority": ["anthropic:plan", "llama_cpp:local"]},
        )
        await _attempt(service, "anthropic:plan", fails=False)

        result = await check_llm_service(_agent(service))

        assert _in_use(result) == [("anthropic:plan", "ok"), ("llama_cpp:local", "unreachable")]
        assert [route["role"] for route in result["details"]["routes_in_use"]] == [
            "primary", "fallback",
        ]
        # RunPod's vendor is not in route_priority: the dispatch skips it.
        assert _standby(result) == ["runpod:vllm-2f8vdgn"]
        assert result["status"] == "warn"
        assert "fallback route llama_cpp:local was unreachable at startup" in result["message"]

    @pytest.mark.asyncio
    async def test_a_refused_plan_to_paid_downgrade_is_not_in_use(self):
        service = _service(
            providers=(PLAN, API),
            mandate=None,
            config={"allow_paid_fallback": False},
        )

        coverage = service.route_coverage()

        assert [route.name for route in coverage.in_use] == ["anthropic:plan"]
        assert [route.name for route in coverage.standby] == ["anthropic:api"]

    @pytest.mark.asyncio
    async def test_routing_that_serves_no_turn_fails(self):
        service = _service(mandate={"vendor": "openai", "route": "plan", "model": "gpt-5"})

        result = await check_llm_service(_agent(service))

        assert result["status"] == "fail"
        assert "no route can serve the next turn" in result["message"]
        assert result["details"]["routing_error"]
        assert derive_overall_status([result]) == "unhealthy"


class TestCallOutcomes:
    """A real call is judged on how it ended, through the public call APIs.

    A streamed attempt is judged on how its stream ended, never on the
    accounting verdict, which bills a failed stream's partial usage as a
    success.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("ending", "status"),
        [
            ("stop", "ok"),
            ("length", "failed"),
            (ConnectionError("provider refused"), "failed"),
        ],
        ids=["answered", "gate-rejected", "provider-error"],
    )
    async def test_a_call_is_judged_on_its_own_end(self, ending, status):
        service = _service()
        _serve_route(service, _AnsweringRoute(ending))

        if status == "ok":
            await service.generate_with_messages(messages=[{"role": "user", "content": "hi"}])
        else:
            with pytest.raises(LLMServiceError):
                await service.generate_with_messages(
                    messages=[{"role": "user", "content": "hi"}]
                )
        result = await check_llm_service(_agent(service))

        assert _in_use(result) == [("anthropic:plan", status)]
        assert result["status"] == ("pass" if status == "ok" else "warn")

    @pytest.mark.asyncio
    async def test_a_stream_that_fails_after_partial_usage_is_a_failed_call(self):
        service = _service()
        billed = _serve_route(service, _StreamedRoute(ConnectionError("dropped")))

        with pytest.raises(LLMStreamingError):
            await _drain(service)
        result = await check_llm_service(_agent(service))

        assert _in_use(result) == [("anthropic:plan", "failed")]
        assert result["status"] == "warn"
        assert "active route anthropic:plan: last call failed" in result["message"]
        assert "dropped" not in repr(result)
        # The partial usage the provider billed is still metered.
        assert len(billed) == 1
        assert billed[0]["meter_usage"] is True
        assert billed[0]["metadata"]["partial_abort"] is True
        assert billed[0]["input_tokens"] == 100

    @pytest.mark.asyncio
    async def test_a_finished_stream_is_an_answered_call(self):
        service = _service()
        _serve_route(service, _StreamedRoute("stop"))

        items = await _drain(service)
        result = await check_llm_service(_agent(service))

        assert isinstance(items[-1], LLMResponse)
        assert _in_use(result) == [("anthropic:plan", "ok")]
        assert result["status"] == "pass"

    @pytest.mark.asyncio
    async def test_a_stream_the_generation_gate_rejects_is_a_failed_call(self):
        service = _service()
        _serve_route(service, _StreamedRoute("length"))

        with pytest.raises(LLMStreamingError, match="output limit"):
            await _drain(service)
        result = await check_llm_service(_agent(service))

        assert _in_use(result) == [("anthropic:plan", "failed")]
        assert result["status"] == "warn"

    @pytest.mark.asyncio
    async def test_a_cancelled_stream_records_no_outcome(self):
        """Cancellation says nothing of the route, even with usage billed."""
        service = _service()
        billed = _serve_route(service, _StreamedRoute(asyncio.CancelledError()))

        with pytest.raises(asyncio.CancelledError):
            await _drain(service)
        result = await check_llm_service(_agent(service))

        assert billed and billed[0]["metadata"]["partial_abort"] is True
        assert _in_use(result) == [("anthropic:plan", "unprobed")]

    @pytest.mark.asyncio
    async def test_a_stream_abandoned_before_its_end_keeps_the_last_outcome(self):
        service = _service()
        await _attempt(service, "anthropic:plan", fails=True)
        last = service._route_outcomes["anthropic:plan"]
        _serve_route(service, _StreamedRoute("stop"))

        stream = _stream(service)
        assert await anext(stream) == "partial "
        await stream.aclose()

        # Neither answered nor failed: the consumer stopped reading.
        assert service._route_outcomes["anthropic:plan"] is last


class TestStandbyProviders:
    @pytest.mark.asyncio
    async def test_standby_providers_do_not_degrade_the_tick(self):
        """The #3873 host: pinned to anthropic:plan, llama_cpp and RunPod idle."""
        service = _service()
        _discovery(
            llama_cpp="ConnectError: [Errno 61] Connection refused",
            runpod="NotFoundError: 404",
        )
        await _attempt(service, "anthropic:plan", fails=False)

        llm = await check_llm_service(_agent(service))
        discovery = await check_model_discovery(_agent(service))

        assert llm["status"] == "pass"
        assert _standby(llm) == ["llama_cpp:local", "runpod:vllm-2f8vdgn"]
        assert "2 standby route(s) not in use, not judged" in llm["message"]
        assert discovery["status"] == "pass"
        assert discovery["details"]["standby_vendors"] == ["llama_cpp", "runpod"]
        assert "standby vendor(s)" in discovery["message"]
        assert derive_overall_status([llm, discovery]) == "healthy"

    @pytest.mark.asyncio
    async def test_a_failing_provider_an_active_route_selects_still_degrades(self):
        # ProviderRegistry lists the route_priority routes first, in order.
        service = _service(
            providers=(LLAMA, PLAN, RUNPOD),
            mandate=None,
            config={"route_priority": ["llama_cpp:local", "anthropic:plan"]},
        )
        _discovery(llama_cpp="ConnectError: [Errno 61] Connection refused")

        llm = await check_llm_service(_agent(service))
        discovery = await check_model_discovery(_agent(service))

        assert _in_use(llm)[0] == ("llama_cpp:local", "unreachable")
        assert llm["status"] == "warn"
        assert discovery["status"] == "warn"
        assert discovery["details"]["standby_vendors"] == []
        assert "llama_cpp" in discovery["message"]
        assert derive_overall_status([llm, discovery]) == "degraded"

    @pytest.mark.asyncio
    async def test_the_pinned_vendor_is_never_standby(self):
        service = _service()
        _discovery(anthropic="AuthenticationError: 401")

        discovery = await check_model_discovery(_agent(service))

        assert discovery["status"] == "fail"
        assert discovery["details"]["pinned_vendor_at_risk"] is True

    @pytest.mark.asyncio
    async def test_the_pinned_vendor_is_judged_while_its_fallbacks_serve(self):
        """An unmatched mandate is served by its fallbacks; the pin is still at risk."""
        service = _service(providers=(LLAMA, RUNPOD))
        service._mandate_fallbacks = [{"vendor": "llama_cpp", "model": "glm-4.6"}]
        _discovery(anthropic="AuthenticationError: 401", runpod="NotFoundError: 404")

        coverage = service.route_coverage()
        discovery = await check_model_discovery(_agent(service))

        assert [route.name for route in coverage.in_use] == ["llama_cpp:local"]
        assert discovery["status"] == "fail"
        assert discovery["details"]["pinned_vendor_at_risk"] is True
        assert discovery["details"]["standby_vendors"] == ["runpod"]


class TestPrivateLeasePrecedence:
    @pytest.mark.asyncio
    async def test_an_active_lease_is_the_only_route_in_use(self):
        service = _service(mandate=None)
        _with_lease(service)

        result = await check_llm_service(_agent(service))

        assert _in_use(result) == [(REMOTE_ROUTE_NAME, "unprobed")]
        assert _standby(result) == ["anthropic:plan", "llama_cpp:local", "runpod:vllm-2f8vdgn"]

    @pytest.mark.asyncio
    async def test_a_failing_lease_route_serving_traffic_warns(self):
        """A healthy cloud route must not mask the lease route that serves."""
        service = _service(mandate=None)
        _with_lease(service)
        await _attempt(service, "anthropic:plan", fails=False)
        await _attempt(service, REMOTE_ROUTE_NAME, fails=True)

        result = await check_llm_service(_agent(service))

        assert _in_use(result) == [(REMOTE_ROUTE_NAME, "failed")]
        assert result["status"] == "warn"

    @pytest.mark.asyncio
    async def test_a_lease_refusal_before_any_adapter_is_a_failed_call(self):
        service = _service(mandate=None)

        async def renewal_fails(lease_id):
            raise RuntimeError("provider renewal failed")

        _with_lease(service, touch=renewal_fails)
        with pytest.raises(LLMServiceError):
            async with service._remote_route_attempt(
                force_local_only=False, model_override=None,
            ):
                pytest.fail("the lease route was pinned")

        result = await check_llm_service(_agent(service))

        assert _in_use(result) == [(REMOTE_ROUTE_NAME, "failed")]
        assert result["status"] == "warn"

    @pytest.mark.asyncio
    async def test_a_lease_route_refusing_every_call_fails(self):
        service = _service(mandate=None)
        _with_lease(service, accepting=False)

        result = await check_llm_service(_agent(service))

        assert _in_use(result) == [(REMOTE_ROUTE_NAME, "refusing")]
        assert result["status"] == "fail"
        assert derive_overall_status([result]) == "unhealthy"

    @pytest.mark.asyncio
    async def test_a_pinned_vendor_keeps_its_route_in_use_over_a_lease(self):
        """The lease route is taken only when nothing narrowed routing."""
        service = _service()
        _with_lease(service)

        coverage = service.route_coverage()

        assert [route.name for route in coverage.in_use] == ["anthropic:plan"]

    @pytest.mark.asyncio
    async def test_a_local_only_privacy_state_never_takes_the_lease(self):
        service = _service(mandate=None)
        service.set_force_local_only_provider(lambda: True)
        _with_lease(service)

        coverage = service.route_coverage()

        assert [route.name for route in coverage.in_use] == ["llama_cpp:local"]

    @pytest.mark.asyncio
    async def test_cloud_discovery_failures_are_standby_while_the_lease_serves(self):
        service = _service(mandate=None)
        _with_lease(service)
        _discovery(anthropic="AuthenticationError: 401", runpod="NotFoundError: 404")

        discovery = await check_model_discovery(_agent(service))

        assert discovery["status"] == "pass"
        assert discovery["details"]["standby_vendors"] == ["anthropic", "runpod"]


class TestFailureDisabledRoutes:
    @pytest.mark.asyncio
    async def test_a_disabled_preferred_route_is_failing_not_standby(self):
        service = _service()
        service._disabled_routes = {"anthropic:plan": "AuthenticationError"}

        result = await check_llm_service(_agent(service))

        assert result["status"] == "fail"
        assert [route["name"] for route in result["details"]["disabled_routes"]] == [
            "anthropic:plan",
        ]
        assert result["details"]["disabled_routes"][0]["status"] == "disabled"
        assert "anthropic:plan" not in _standby(result)
        assert "disabled this session after a permanent auth failure" in result["message"]

    @pytest.mark.asyncio
    async def test_a_disabled_route_warns_while_the_chain_still_serves(self):
        service = _service(
            mandate=None,
            config={"route_priority": ["anthropic:plan", "llama_cpp:local"]},
        )
        service._disabled_routes = {"anthropic:plan": "AuthenticationError"}
        service.reachability[1]["status"] = "reachable"

        result = await check_llm_service(_agent(service))

        assert _in_use(result) == [("llama_cpp:local", "reachable")]
        assert result["status"] == "warn"
        assert "anthropic:plan" not in _standby(result)

    @pytest.mark.asyncio
    async def test_a_disabled_routes_vendor_is_judged_by_discovery(self):
        service = _service(
            mandate=None,
            config={"route_priority": ["anthropic:plan", "llama_cpp:local"]},
        )
        service._disabled_routes = {"anthropic:plan": "AuthenticationError"}
        _discovery(anthropic="AuthenticationError: 401")

        discovery = await check_model_discovery(_agent(service))

        assert discovery["status"] == "warn"
        assert "anthropic" not in discovery["details"]["standby_vendors"]


class TestPrivateOnlyService:
    @pytest.mark.asyncio
    async def test_coverage_runs_with_no_configured_provider(self):
        service = _service(providers=(), mandate=None)
        _with_lease(service)
        await _attempt(service, REMOTE_ROUTE_NAME, fails=True)

        result = await check_llm_service(_agent(service))

        assert result["message"].startswith("No configured LLM provider")
        assert _in_use(result) == [(REMOTE_ROUTE_NAME, "failed")]
        assert result["status"] == "warn"

    @pytest.mark.asyncio
    async def test_no_provider_and_no_lease_is_still_a_warning(self):
        result = await check_llm_service(_agent(_service(providers=(), mandate=None)))

        assert result["status"] == "warn"
        assert result["message"] == "LLM service exists but no providers initialized"


def test_route_coverage_is_the_services_report():
    coverage = _service().route_coverage()

    assert isinstance(coverage, RouteCoverage)
    assert coverage.routing_error is None
    assert [route.name for route in coverage.in_use] == ["anthropic:plan"]
    assert coverage.in_use[0].last_call is None


def test_an_outcome_is_stamped_in_utc():
    service = _service()
    service._record_route_outcome("anthropic:plan", answered=True)

    outcome = service.route_coverage().in_use[0].last_call

    assert outcome.answered is True
    assert datetime.now(UTC) - outcome.at < timedelta(seconds=5)
