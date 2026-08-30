"""#2674 finding 3: the /api/bridge/stream SSE error event must go through the
ONE shared user-safe streaming-error boundary — never ``str(e)``.

Terra drove the real authenticated FastAPI endpoint and leaked a
``BRIDGE_STRICT_WITHHELD_PROSE_MARKER`` through the SSE ``error`` event (the old
code emitted ``json.dumps({"type":"error","message": str(e)})``). Under a strict
buffered audit an adapter that raises after yielding partial prose can carry the
withheld response in its exception text, so reflecting it verbatim leaks past the
fail-closed gate. These boot the REAL app + router via ``TestClient`` with API-key
auth and assert the SSE error payload is a stable, content-free constant while the
marker (and any provider/underlying text) never surfaces.
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from kestrel_sovereign.agent.request_lifecycle import RequestLifecycleMixin
from kestrel_sovereign.features.bridge.feature import BridgeFeature
from kestrel_sovereign.features.bridge.protocol import BridgeRequest
from kestrel_sovereign.features.bridge.router import get_router


API_KEY = "test-bridge-key"


def _boot(process_input_streaming, *, cancel_on_check=None):
    """Boot the real app with a single agent exposing a BridgeFeature and the
    given ``process_input_streaming`` async generator. Returns ``(app, restore)``.
    """
    from server import app
    from kestrel_sovereign.server import (
        _mount_feature_routers,
        _unmount_feature_routers,
    )

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original_lifespan = app.router.lifespan_context
    original_agent = getattr(app.state, "agent", None)
    original_manager = getattr(app.state, "agent_manager", None)

    # A real BridgeFeature, but its session/log I/O is stubbed (no DB).
    bridge = BridgeFeature(agent=MagicMock())
    bridge.get_or_create_session = AsyncMock(
        return_value=SimpleNamespace(id="sess-1", gateway_session_id=None)
    )
    bridge.log_invocation = AsyncMock()

    agent = MagicMock()
    agent.features = {"BridgeFeature": bridge}
    agent.process_input_streaming = process_input_streaming
    cancellation_checks = 0

    def _is_request_cancelled(_request_id):
        nonlocal cancellation_checks
        cancellation_checks += 1
        return cancel_on_check is not None and cancellation_checks >= cancel_on_check

    agent.is_request_cancelled = MagicMock(side_effect=_is_request_cancelled)

    app.router.lifespan_context = noop_lifespan
    app.state.agent = agent
    app.state.agent_manager = None
    _mount_feature_routers(app)

    def restore():
        _unmount_feature_routers(app)
        app.router.lifespan_context = original_lifespan
        app.state.agent = original_agent
        app.state.agent_manager = original_manager

    return app, restore


def _post_stream(process_input_streaming):
    os.environ["KESTREL_API_KEY"] = API_KEY
    app, restore = _boot(process_input_streaming)
    try:
        with TestClient(app) as client:
            return client.post(
                "/api/bridge/stream",
                json={"message": "hi", "channel_type": "api"},
                headers={"X-API-Key": API_KEY},
            )
    finally:
        restore()


def _post_stream_with_agent(process_input_streaming, *, cancel_on_check=None):
    """Drive the real bridge route and retain its test agent for assertions."""

    os.environ["KESTREL_API_KEY"] = API_KEY
    app, restore = _boot(process_input_streaming, cancel_on_check=cancel_on_check)
    agent = app.state.agent
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/bridge/stream",
                json={"message": "hi", "channel_type": "api"},
                headers={"X-API-Key": API_KEY},
            )
        return response, agent
    finally:
        restore()


def _error_events(text: str):
    """Parse the SSE ``data:`` lines and return the decoded ``error`` events."""
    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        try:
            payload = json.loads(line[len("data:"):].strip())
        except ValueError:
            continue
        if isinstance(payload, dict) and payload.get("type") == "error":
            events.append(payload)
    return events


def _events(text: str):
    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        try:
            payload = json.loads(line[len("data:") :].strip())
        except ValueError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


@pytest.mark.asyncio
async def test_bridge_stop_interrupts_producer_blocked_before_first_event():
    """Bridge binds the same blocked producer owner to its request generation."""

    producer_started = asyncio.Event()
    release_producer = asyncio.Event()
    producer_finalized = asyncio.Event()
    bridge = MagicMock()
    bridge.get_or_create_session = AsyncMock(
        return_value=SimpleNamespace(id="bridge-session")
    )
    bridge.log_invocation = AsyncMock()

    class BlockedBridgeAgent(RequestLifecycleMixin):
        def __init__(self):
            self._current_request_id = None
            self._active_request_ids = set()
            self._active_request_counts = {}
            self._active_request_started_at = {}
            self._cancelled_requests = set()
            self._request_completion_events = {}
            self.features = {"BridgeFeature": bridge}

        async def process_input_streaming(self, *_args, **_kwargs):
            try:
                producer_started.set()
                await release_producer.wait()
                yield "late"
            finally:
                producer_finalized.set()

    agent = BlockedBridgeAgent()
    app = FastAPI()
    app.state.agent = agent
    request = Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/bridge/stream",
            "raw_path": b"/api/bridge/stream",
            "query_string": b"",
            "headers": [],
            "client": ("test", 1),
            "server": ("test", 80),
            "app": app,
        }
    )
    route = next(
        route
        for route in get_router().routes
        if getattr(route, "path", None) == "/api/bridge/stream"
    )
    endpoint = getattr(route.endpoint, "__wrapped__", route.endpoint)
    response = await endpoint(
        request,
        BridgeRequest(message="work", request_id="blocked-bridge"),
    )
    consumer = asyncio.create_task(anext(response.body_iterator))

    await asyncio.wait_for(producer_started.wait(), timeout=1)
    assert agent.cancel_current_request("blocked-bridge") is True
    try:
        assert '"type": "stopped"' in await asyncio.wait_for(
            consumer,
            timeout=0.2,
        )
    finally:
        release_producer.set()
        if not consumer.done():
            consumer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await consumer
        await response.body_iterator.aclose()

    assert producer_finalized.is_set()
    await asyncio.wait_for(
        agent.wait_for_request_completion("blocked-bridge"),
        timeout=1,
    )


def test_bridge_stream_withholds_queued_chunk_after_stop():
    """Cancellation wins after producer yield and before bridge publication."""

    async def _queued(*_args, **_kwargs):
        yield "must not escape"

    response, agent = _post_stream_with_agent(
        _queued,
        cancel_on_check=5,
    )

    assert response.status_code == 200
    events = _events(response.text)
    assert [event["type"] for event in events] == ["stopped"]
    assert "must not escape" not in response.text
    assert agent.features["BridgeFeature"].log_invocation.await_count == 1


def test_bridge_stream_reports_stop_after_clean_producer_eof():
    """A stopped clean unwind cannot fall through to done or outbound logging."""

    async def _clean_eof(*_args, **_kwargs):
        if False:
            yield "unreachable"

    response, agent = _post_stream_with_agent(
        _clean_eof,
        cancel_on_check=5,
    )

    assert response.status_code == 200
    assert [event["type"] for event in _events(response.text)] == ["stopped"]
    assert agent.features["BridgeFeature"].log_invocation.await_count == 1


def test_bridge_stream_generic_exception_emits_safe_constant_not_str_e():
    """A generic mid-stream exception whose text carries the withheld marker must
    NOT be reflected; the SSE error event is a stable, content-free constant."""
    marker = "BRIDGE_STRICT_WITHHELD_PROSE_MARKER"

    async def _boom(*args, **kwargs):
        yield "benign streamed prose "
        raise RuntimeError(f"internal failure carrying {marker}")

    response = _post_stream(_boom)

    assert response.status_code == 200
    # The withheld marker in the exception text never reaches the client.
    assert marker not in response.text
    # A single stable, safe error event is present.
    errors = _error_events(response.text)
    assert len(errors) == 1
    assert marker not in errors[0]["message"]
    assert "could not be completed" in errors[0]["message"]


def test_bridge_source_failure_is_not_recorded_as_abandoned_cleanup():
    """The bridge must reserve ABANDONED for an actual close failure."""

    async def _boom(*args, **kwargs):
        raise RuntimeError("provider failed")
        yield  # pragma: no cover

    response, agent = _post_stream_with_agent(_boom)

    assert response.status_code == 200
    agent._cleanup_cancelled_request.assert_called_once()
    assert agent._cleanup_cancelled_request.call_args.kwargs == {}


def test_bridge_stream_llm_streaming_error_provider_marker_never_surfaces():
    """#2674 findings 3 & 4: a route failure whose marker rides message,
    underlying AND provider must emit only the constant recovery guidance — the
    provider free string is never reflected on the bridge SSE path either."""
    from kestrel_sovereign.llm.streaming import LLMStreamingError

    marker = "ROUTE_FIELD_UNBOUNDED_MARKER__WITHHELD_TEXT"

    async def _route_fail(*args, **kwargs):
        raise LLMStreamingError(
            f"route failed {marker}",
            provider=marker,
            underlying=RuntimeError(marker),
        )
        yield  # pragma: no cover

    response = _post_stream(_route_fail)

    assert response.status_code == 200
    assert marker not in response.text
    errors = _error_events(response.text)
    assert len(errors) == 1
    assert marker not in errors[0]["message"]
    # The no-blind-fallback recovery guidance still reaches the client.
    assert "No fallback response was generated" in errors[0]["message"]


def test_agent_and_bridge_error_message_share_one_boundary():
    """The two transports must not drift: both derive the client-facing text from
    the single shared boundary, so the safe message content is identical."""
    from kestrel_sovereign.llm.streaming_errors import (
        safe_streaming_error_message,
        agent_stream_error_block,
        bridge_sse_error_event,
    )
    from kestrel_sovereign.llm.streaming import LLMStreamingError

    for exc in (RuntimeError("x"), LLMStreamingError("y", provider="LEAK")):
        msg = safe_streaming_error_message(exc)
        # The agent text/plain block is the same safe message with markdown
        # emphasis stripped (``⚠️ **header** body`` → ``header body``).
        block = agent_stream_error_block(exc)
        assert block.replace("**", "").replace("\n\n---\n⚠️ ", "") == msg
        # ...and the bridge SSE event carries it verbatim as the JSON message.
        payload = json.loads(
            bridge_sse_error_event(exc).strip()[len("data:"):].strip()
        )
        assert payload["message"] == msg
        assert "LEAK" not in payload["message"]
