"""Real terminal boundaries retain original control evidence (#3569 review15)."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from kestrel_sovereign.agent.invocation import (
    InvocationCancelledError, InvocationSelfFencedError,
    bind_async_generator_invocation, bind_async_invocation,
)
from kestrel_sovereign.agent.request_lifecycle import RequestCompletionDisposition
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCommitOutcomeError, ExecutionCustody,
    execution_commit_outcome, is_execution_control_error,
)
from tests.unit.test_execution_custody import Authority


def control_error(control, carrier="direct"):
    evidence = (ExecutionAuthorityError("original authority denied")
                if control == "denied" else ExecutionCommitOutcomeError(control))
    if carrier == "direct":
        return evidence
    error = {
        "wrapped": lambda: RuntimeError("wrapped original control"),
        "cancel": lambda: asyncio.CancelledError("cancelled original control"),
        "stop": lambda: InvocationCancelledError("typed Stop carrier"),
        "self-fence": lambda: InvocationSelfFencedError("typed owner-loss carrier"),
    }[carrier]()
    error.__cause__ = evidence
    return error


def dispatcher_fixture(error):
    from kestrel_sovereign.signals.dispatcher import SignalDispatcher
    dispatcher = SignalDispatcher.__new__(SignalDispatcher)
    dispatcher._agent = SimpleNamespace(did="original")
    dispatcher._durable_delivery_owner = "dispatcher:original"
    dispatcher._discard_transient_durable_handoff = Mock()
    dispatcher._retained_cognition_control_debt = {}
    dispatcher._retained_durable_cognition_tasks = set()
    dispatcher._runtime_owner_fence_lock = asyncio.Lock()
    dispatcher._durable_shutdown_owner_fenced = False
    dispatcher._durable_store = SimpleNamespace(backend=SimpleNamespace(
        backend_type="postgres", fail_cognition_delivery=AsyncMock(return_value=False),
        retain_cognition_cleanup_owner=AsyncMock(return_value=False),
    ))
    delivery = SimpleNamespace(delivery_id="delivery", consumer_id="consumer", lease_token="original-token")
    return dispatcher, delivery


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["denied", "unknown", "committed"])
@pytest.mark.parametrize("retained", [False, True])
async def test_failed_exact_terminal_cas_retains_original_debt(control, retained):
    error = control_error(control)
    dispatcher, delivery = dispatcher_fixture(error)
    operation = (dispatcher._release_retained_durable_cognition_task
                 if retained else dispatcher._terminalize_failed_cognition)
    with pytest.raises(type(error)) as caught:
        await operation(delivery, error)
    assert caught.value is error
    assert dispatcher._retained_cognition_control_debt == {"delivery": (delivery, error)}
    dispatcher._discard_transient_durable_handoff.assert_not_called()
    exact = dispatcher._durable_store.backend.fail_cognition_delivery.await_args
    dispatcher._durable_store.backend.fail_cognition_delivery.return_value = True
    await operation(delivery, error)
    assert not dispatcher._retained_cognition_control_debt
    assert dispatcher._durable_store.backend.fail_cognition_delivery.await_args == exact


@pytest.mark.asyncio
async def test_failed_cleanup_liveness_cas_is_not_a_healthy_heartbeat():
    error = control_error("unknown")
    dispatcher, delivery = dispatcher_fixture(error)
    dispatcher._retained_cognition_control_debt = {"delivery": (delivery, error)}
    with pytest.raises(type(error)) as caught:
        await dispatcher._heartbeat_runtime_owner()
    assert caught.value is error
    assert dispatcher._retained_cognition_control_debt == {"delivery": (delivery, error)}


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["denied", "unknown", "committed"])
@pytest.mark.parametrize("foreign_close", [False, True])
async def test_source_close_error_cannot_replace_original_scope_control(control, foreign_close):
    dispositions = []
    finalized = asyncio.Event()

    class Owner:
        _execution_custody = ExecutionCustody(Authority())

        def register_active_request(self, request_id, *, nested=False):
            pass

        async def await_durable_request_admission(self, request_id):
            return True

        def _cleanup_cancelled_request(self, request_id, **kwargs):
            dispositions.append(kwargs.get("disposition"))

        @bind_async_generator_invocation("request_id", track_request_lifecycle=True)
        async def stream(self, request_id=None):
            try:
                yield "first"
                yield "forbidden"
            finally:
                finalized.set()
                raise OSError("source close failed")

    owner = Owner()
    source = owner.stream(request_id="original")
    assert await anext(source) == "first"
    if control == "denied":
        owner._execution_custody.revoke("original authority denied")
    else:
        owner._execution_custody.preserve_commit_uncertainty(control)
    operation = anext(source)
    if foreign_close:
        operation = asyncio.create_task(operation)
    with pytest.raises(ExecutionAuthorityError) as caught:
        await operation
    assert execution_commit_outcome(caught.value) == (None if control == "denied" else control)
    assert finalized.is_set()
    assert dispositions == [RequestCompletionDisposition.ABANDONED]


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["denied", "unknown", "committed"])
@pytest.mark.parametrize("carrier", ["direct", "wrapped", "cancel", "stop", "self-fence"])
@pytest.mark.parametrize("transport", ["stream", "invoke"])
async def test_actual_http_stream_does_not_acknowledge_or_recommend_control_retry(control, carrier, transport):
    from fastapi import FastAPI
    from starlette.requests import Request
    from kestrel_sovereign.endpoints.agent import stream_agent_response, invoke_agent
    from kestrel_sovereign.streams.tap import AgentStreamTap

    AgentStreamTap.reset()
    error = control_error(control, carrier)
    finalized = asyncio.Event()
    agent = MagicMock()
    agent.storage.resolve_session_id = AsyncMock(return_value=None)
    agent.is_request_cancelled.return_value = False
    agent.is_request_self_fenced.return_value = True

    async def producer(*args, **kwargs):
        try:
            if False:
                yield "unreachable"
            raise error
        finally:
            finalized.set()

    agent.process_input_streaming = producer
    agent.process_input = AsyncMock(side_effect=error)
    app = FastAPI()
    app.state.agent = agent
    body = json.dumps({"input": "provider-free", "request_id": "original-http"}).encode()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request({"type": "http", "http_version": "1.1", "method": "POST",
                       "scheme": "http", "path": "/api/agent/stream", "raw_path": b"/api/agent/stream",
                       "query_string": b"", "headers": [(b"content-type", b"application/json")],
                       "client": ("test", 1), "server": ("test", 80), "app": app}, receive)
    function = stream_agent_response if transport == "stream" else invoke_agent
    endpoint = getattr(function, "__wrapped__", function)
    response = None
    try:
        with pytest.raises(type(error)) as caught:
            if transport == "stream":
                response = await endpoint(request)
                await anext(response.body_iterator)
            else:
                from starlette.responses import Response
                await endpoint(request, Response())
        assert caught.value is error
        if transport == "stream":
            assert finalized.is_set()
        assert is_execution_control_error(caught.value)
        agent._cleanup_cancelled_request.assert_called_once_with(
            "original-http", disposition=RequestCompletionDisposition.ABANDONED,
        )
    finally:
        if response is not None:
            await response.body_iterator.aclose()
        AgentStreamTap.reset()


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["denied", "unknown", "committed"])
@pytest.mark.parametrize("carrier", ["direct", "wrapped", "cancel"])
async def test_actual_first_admission_control_preserves_durable_unresolved_generation(tmp_path, control, carrier):
    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.stop import DistributedInvocationRegistry, DistributedInvocationStore
    from tests.unit.test_distributed_stop_invocations import _ReplicaAgent

    class Agent(_ReplicaAgent):
        @bind_async_invocation("request_id", track_request_lifecycle=True)
        async def turn(self, request_id=None):
            raise AssertionError("failed admission must not enter cognition")

    database = await AsyncDatabase.sqlite(str(tmp_path / "admission.db"))
    store = DistributedInvocationStore(database)
    await store.ensure_schema()
    recorded = []
    register = store.register

    async def capture_register(**kwargs):
        recorded.append(kwargs["generation_id"])
        return await register(**kwargs)

    store.register = capture_register
    error = control_error(control, carrier)
    store.poll_owner = AsyncMock(side_effect=error)
    registry = DistributedInvocationRegistry(store)
    agent = Agent("original-agent")
    registry.attach(agent)
    try:
        with pytest.raises(BaseException) as caught:
            await agent.turn(request_id="original-admission")
        assert is_execution_control_error(caught.value)
        assert execution_commit_outcome(caught.value) == (None if control == "denied" else control)
        await asyncio.gather(*tuple(registry._cleanup_tasks))
        assert len(recorded) == 1
        assert await database.fetchone(
            "SELECT generation_id FROM stop_unresolved_invocations WHERE generation_id = ?",
            (recorded[0],),
        ) == (recorded[0],)
        assert not await database.fetchone(
            "SELECT generation_id FROM stop_active_invocations WHERE generation_id = ?", (recorded[0],),
        )
    finally:
        await registry.close()
        await database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["denied", "unknown", "committed"])
@pytest.mark.parametrize("carrier", ["direct", "wrapped", "cancel", "stop", "self-fence"])
@pytest.mark.parametrize("transport", ["stream", "invoke"])
async def test_actual_bridge_stream_retains_control_and_abandoned_lifecycle(control, carrier, transport):
    from fastapi import FastAPI
    from starlette.requests import Request
    from kestrel_sovereign.features.bridge.protocol import BridgeRequest
    from kestrel_sovereign.features.bridge.router import get_router

    error = control_error(control, carrier)
    bridge = MagicMock()
    bridge.get_or_create_session = AsyncMock(return_value=SimpleNamespace(id="original-session"))
    bridge.log_invocation = AsyncMock()
    agent = MagicMock()
    agent.features = {"BridgeFeature": bridge}
    agent.is_request_cancelled.return_value = False
    agent.is_request_self_fenced.return_value = True
    finalized = asyncio.Event()

    async def producer(*args, **kwargs):
        try:
            if False:
                yield "unreachable"
            raise error
        finally:
            finalized.set()

    agent.process_input_streaming = producer
    agent.process_input = AsyncMock(side_effect=error)
    app = FastAPI()
    app.state.agent = agent
    request = Request({"type": "http", "http_version": "1.1", "method": "POST", "scheme": "http",
                       "path": "/api/bridge/stream", "raw_path": b"/api/bridge/stream", "query_string": b"",
                       "headers": [], "client": ("test", 1), "server": ("test", 80), "app": app})
    route = next(route for route in get_router().routes if getattr(route, "path", None) == f"/api/bridge/{transport}")
    endpoint = getattr(route.endpoint, "__wrapped__", route.endpoint)
    response = None
    try:
        with pytest.raises(type(error)) as caught:
            body = BridgeRequest(message="provider-free", request_id="original-bridge")
            if transport == "stream":
                response = await endpoint(request, body)
                await anext(response.body_iterator)
            else:
                from starlette.responses import Response
                await endpoint(request, body, Response())
        assert caught.value is error
        if transport == "stream":
            assert finalized.is_set()
        agent._cleanup_cancelled_request.assert_called_once_with(
            "original-bridge", disposition=RequestCompletionDisposition.ABANDONED,
        )
        assert bridge.log_invocation.await_count == 1, "failed turn must not log successful outbound work"
    finally:
        if response is not None:
            await response.body_iterator.aclose()
