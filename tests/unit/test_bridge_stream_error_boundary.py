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

import json
import os
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from kestrel_sovereign.features.bridge.feature import BridgeFeature


API_KEY = "test-bridge-key"


@pytest.fixture(autouse=True)
def _isolate_shared_rate_limit_bucket():
    """Keep earlier TestClient traffic from bypassing this endpoint boundary."""

    from kestrel_sovereign.rate_limit import limiter

    limiter.reset()
    yield
    limiter.reset()


def _boot(process_input_streaming):
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
