"""
Unit tests for agent request cancellation (stop button).
"""

import asyncio
import json

import pytest
from unittest.mock import MagicMock, AsyncMock


class TestAgentCancellation:
    """Tests for request cancellation functionality."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent with cancellation attributes."""
        from kestrel_sovereign.kestrel_agent import KestrelAgent
        
        # Create minimal mock
        agent = MagicMock(spec=KestrelAgent)
        agent._current_request_id = None
        agent._active_request_ids = set()
        agent._active_request_started_at = {}
        agent._cancelled_requests = set()
        agent._request_completion_events = {}

        # Bind actual methods
        agent.register_active_request = KestrelAgent.register_active_request.__get__(agent)
        agent.cancel_current_request = KestrelAgent.cancel_current_request.__get__(agent)
        agent.is_request_cancelled = KestrelAgent.is_request_cancelled.__get__(agent)
        agent.wait_for_request_completion = KestrelAgent.wait_for_request_completion.__get__(agent)
        agent._cleanup_cancelled_request = KestrelAgent._cleanup_cancelled_request.__get__(agent)
        agent.active_request_ages = KestrelAgent.active_request_ages.__get__(agent)
        agent.prune_stale_active_requests = KestrelAgent.prune_stale_active_requests.__get__(agent)

        return agent

    def test_cancel_when_no_active_request(self, mock_agent):
        """Cancel returns False when no request is active."""
        result = mock_agent.cancel_current_request()
        assert result is False

    def test_cancel_when_request_active(self, mock_agent):
        """Cancel returns True and marks request as cancelled."""
        mock_agent._current_request_id = "test-request-123"
        
        result = mock_agent.cancel_current_request()
        
        assert result is True
        assert "test-request-123" in mock_agent._cancelled_requests

    def test_cancel_specific_request_id(self, mock_agent):
        """Explicit request IDs should be cancellable without changing current state first."""
        mock_agent._active_request_ids = {"req-1", "req-2"}
        mock_agent._current_request_id = "req-2"

        result = mock_agent.cancel_current_request("req-1")

        assert result is True
        assert "req-1" in mock_agent._cancelled_requests

    def test_is_request_cancelled_returns_true_for_cancelled(self, mock_agent):
        """is_request_cancelled returns True for cancelled requests."""
        mock_agent._cancelled_requests.add("cancelled-req")
        
        assert mock_agent.is_request_cancelled("cancelled-req") is True

    def test_is_request_cancelled_returns_false_for_active(self, mock_agent):
        """is_request_cancelled returns False for non-cancelled requests."""
        mock_agent._current_request_id = "active-req"
        
        assert mock_agent.is_request_cancelled("active-req") is False

    def test_cleanup_removes_from_cancelled_set(self, mock_agent):
        """Cleanup removes request from cancelled set."""
        mock_agent._current_request_id = "test-req"
        mock_agent._cancelled_requests.add("test-req")
        
        mock_agent._cleanup_cancelled_request("test-req")
        
        assert "test-req" not in mock_agent._cancelled_requests
        assert mock_agent._current_request_id is None

    def test_cleanup_only_clears_matching_current_request(self, mock_agent):
        """Cleanup only clears current_request_id if it matches."""
        mock_agent._current_request_id = "different-req"
        mock_agent._cancelled_requests.add("test-req")
        
        mock_agent._cleanup_cancelled_request("test-req")
        
        assert "test-req" not in mock_agent._cancelled_requests
        assert mock_agent._current_request_id == "different-req"  # Not cleared

    @pytest.mark.asyncio
    async def test_completion_waiter_does_not_acknowledge_cancel_marker(self, mock_agent):
        """A cooperative cancel marker is not proof that execution stopped."""
        mock_agent.register_active_request("still-running")
        assert mock_agent.cancel_current_request("still-running") is True

        waiter = asyncio.create_task(
            mock_agent.wait_for_request_completion("still-running")
        )
        await asyncio.sleep(0)

        assert waiter.done() is False
        mock_agent._cleanup_cancelled_request("still-running")
        await asyncio.wait_for(waiter, timeout=1)

    @pytest.mark.asyncio
    async def test_completion_waiter_requires_final_duplicate_cleanup(self, mock_agent):
        """One retry cleanup cannot acknowledge a same-id sibling still running."""
        mock_agent.register_active_request("retry-id")
        mock_agent.register_active_request("retry-id")
        assert mock_agent.cancel_current_request("retry-id") is True
        waiter = asyncio.create_task(
            mock_agent.wait_for_request_completion("retry-id")
        )

        mock_agent._cleanup_cancelled_request("retry-id")
        await asyncio.sleep(0)
        assert waiter.done() is False

        mock_agent._cleanup_cancelled_request("retry-id")
        await asyncio.wait_for(waiter, timeout=1)

    @pytest.mark.asyncio
    async def test_stream_closes_inner_generator_before_completion_ack(self):
        """Stop cannot acknowledge before the turn iterator releases its locks."""
        from fastapi import FastAPI
        from starlette.requests import Request

        from kestrel_sovereign.agent.request_lifecycle import RequestLifecycleMixin
        from kestrel_sovereign.endpoints.agent import stream_agent_response

        inner_cleanup_started = asyncio.Event()
        release_inner_cleanup = asyncio.Event()

        class LiveAgent(RequestLifecycleMixin):
            def __init__(self):
                self._current_request_id = None
                self._active_request_ids = set()
                self._active_request_counts = {}
                self._active_request_started_at = {}
                self._cancelled_requests = set()
                self._request_completion_events = {}
                self.storage = MagicMock()
                self.storage.resolve_session_id = AsyncMock(return_value=None)

            async def process_input_streaming(self, *_args, **_kwargs):
                try:
                    yield "first"
                    yield "second"
                finally:
                    inner_cleanup_started.set()
                    await release_inner_cleanup.wait()

        agent = LiveAgent()
        app = FastAPI()
        app.state.agent = agent
        body = json.dumps(
            {"input": "work", "request_id": "inner-cleanup"}
        ).encode()
        delivered = False

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        request = Request(
            {
                "type": "http",
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/api/agent/stream",
                "raw_path": b"/api/agent/stream",
                "query_string": b"",
                "headers": [(b"content-type", b"application/json")],
                "client": ("test", 1),
                "server": ("test", 80),
                "app": app,
            },
            receive,
        )
        endpoint = getattr(stream_agent_response, "__wrapped__", stream_agent_response)
        response = await endpoint(request)
        stream = response.body_iterator

        assert await anext(stream) == "first"
        assert agent.cancel_current_request("inner-cleanup") is True
        completion = asyncio.create_task(
            agent.wait_for_request_completion("inner-cleanup")
        )
        assert "Request stopped" in await anext(stream)

        eof = asyncio.create_task(anext(stream))
        await asyncio.wait_for(inner_cleanup_started.wait(), timeout=1)
        assert completion.done() is False

        # Client disconnect cancels the outer StreamingResponse task. The
        # inner turn cleanup still owns completion and must finish first.
        eof.cancel()
        await asyncio.sleep(0)
        assert completion.done() is False
        release_inner_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await eof
        await asyncio.wait_for(completion, timeout=1)

    def test_multiple_cancellations_tracked(self, mock_agent):
        """Multiple requests can be cancelled and tracked."""
        mock_agent._cancelled_requests.add("req-1")
        mock_agent._cancelled_requests.add("req-2")

        assert mock_agent.is_request_cancelled("req-1") is True
        assert mock_agent.is_request_cancelled("req-2") is True
        assert mock_agent.is_request_cancelled("req-3") is False

    def test_register_stamps_started_at(self, mock_agent):
        """Registering an active request records a monotonic start time."""
        mock_agent.register_active_request("req-1")

        assert "req-1" in mock_agent._active_request_ids
        assert "req-1" in mock_agent._active_request_started_at
        ages = mock_agent.active_request_ages()
        assert "req-1" in ages
        assert ages["req-1"] >= 0.0

    def test_cleanup_drops_started_at(self, mock_agent):
        """Cleanup removes the registration timestamp too."""
        mock_agent.register_active_request("req-1")
        mock_agent._cleanup_cancelled_request("req-1")

        assert "req-1" not in mock_agent._active_request_started_at
        assert mock_agent.active_request_ages() == {}

    def test_duplicate_inflight_request_id_is_reference_counted(self, mock_agent):
        """One retry cleanup must not unregister its still-running sibling."""
        mock_agent.register_active_request("retry-id")
        mock_agent.register_active_request("retry-id")
        mock_agent._cancelled_requests.add("retry-id")

        mock_agent._cleanup_cancelled_request("retry-id")

        assert mock_agent._active_request_counts["retry-id"] == 1
        assert "retry-id" in mock_agent._active_request_ids
        assert "retry-id" in mock_agent._active_request_started_at
        assert "retry-id" in mock_agent._cancelled_requests

        mock_agent._cleanup_cancelled_request("retry-id")

        assert "retry-id" not in mock_agent._active_request_counts
        assert "retry-id" not in mock_agent._active_request_ids
        assert "retry-id" not in mock_agent._active_request_started_at
        assert "retry-id" not in mock_agent._cancelled_requests

    def test_prune_removes_stale_request(self, mock_agent):
        """A request older than the window is pruned and returned."""
        mock_agent.register_active_request("stale")
        # Back-date the registration well past the threshold.
        mock_agent._active_request_started_at["stale"] -= 1000

        pruned = mock_agent.prune_stale_active_requests(900)

        assert pruned == ["stale"]
        assert "stale" not in mock_agent._active_request_ids
        assert "stale" not in mock_agent._active_request_started_at
        # current_request_id pointed at the pruned id → cleared.
        assert mock_agent._current_request_id is None

    def test_prune_keeps_fresh_request(self, mock_agent):
        """A fresh request is not pruned."""
        mock_agent.register_active_request("fresh")

        pruned = mock_agent.prune_stale_active_requests(900)

        assert pruned == []
        assert "fresh" in mock_agent._active_request_ids

    def test_prune_stamps_unknown_request_clock(self, mock_agent):
        """An id with no recorded start time is stamped, not pruned blind."""
        # Simulate a foreign/legacy registration straight into the set.
        mock_agent._active_request_ids.add("foreign")

        pruned = mock_agent.prune_stale_active_requests(900)

        assert pruned == []
        assert "foreign" in mock_agent._active_request_ids
        # The staleness clock now started for it.
        assert "foreign" in mock_agent._active_request_started_at


class TestStopEndpoint:
    """Tests for the /agent/stop endpoint."""

    @pytest.mark.asyncio
    async def test_stop_endpoint_calls_cancel(self):
        """Stop endpoint calls cancel_current_request on agent."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from kestrel_sovereign.endpoints.agent import router
        
        app = FastAPI()
        app.include_router(router)
        
        # Mock agent
        mock_agent = MagicMock()
        mock_agent.cancel_current_request = MagicMock(return_value=True)
        mock_agent.wait_for_request_completion = AsyncMock(return_value=None)
        app.state.agent = mock_agent
        
        client = TestClient(app)
        response = client.post("/api/agent/stop")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["cancelled"] is True
        mock_agent.cancel_current_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_endpoint_no_active_request(self):
        """Stop endpoint returns cancelled=False when no request active."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from kestrel_sovereign.endpoints.agent import router
        
        app = FastAPI()
        app.include_router(router)
        
        # Mock agent with no active request
        mock_agent = MagicMock()
        mock_agent.cancel_current_request = MagicMock(return_value=False)
        app.state.agent = mock_agent
        
        client = TestClient(app)
        response = client.post("/api/agent/stop")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["cancelled"] is False

    @pytest.mark.asyncio
    async def test_stop_endpoint_passes_request_id(self):
        """Stop endpoint forwards explicit request IDs for scoped cancellation."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from kestrel_sovereign.endpoints.agent import router

        app = FastAPI()
        app.include_router(router)

        mock_agent = MagicMock()
        mock_agent.cancel_current_request = MagicMock(return_value=True)
        mock_agent.wait_for_request_completion = AsyncMock(return_value=None)
        app.state.agent = mock_agent

        client = TestClient(app)
        response = client.post("/api/agent/stop", json={"request_id": "req-123"})

        assert response.status_code == 200
        mock_agent.cancel_current_request.assert_called_once_with(request_id="req-123")

    @pytest.mark.asyncio
    async def test_stop_endpoint_decodes_a_verbatim_invoke_header_echo_once(self):
        """A response header copied into stop targets the original opaque ID.

        X-Request-ID is a percent-encoded transport form.  The deliberate
        literal percent and percent-looking text here catch both the old
        literal-header bug and accidental double decoding.
        """
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from kestrel_sovereign.agent.invocation import invocation_id_response_header
        from kestrel_sovereign.endpoints.agent import router

        request_id = "cancel ☃ / 100% %E2%98%83?x=y#fragment"
        header_echo = invocation_id_response_header(request_id)
        app = FastAPI()
        app.include_router(router)
        mock_agent = MagicMock()
        mock_agent.cancel_current_request = MagicMock(return_value=True)
        mock_agent.wait_for_request_completion = AsyncMock(return_value=None)
        app.state.agent = mock_agent

        response = TestClient(app).post(
            "/api/agent/stop",
            headers={"X-Request-ID": header_echo},
        )

        assert response.status_code == 200, response.text
        assert response.json()["request_id"] == request_id
        mock_agent.cancel_current_request.assert_called_once_with(request_id=request_id)

    @pytest.mark.asyncio
    async def test_stop_body_request_id_remains_literal_and_wins_over_header(self):
        """Body IDs retain their historical precedence over header wire IDs."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from kestrel_sovereign.agent.invocation import invocation_id_response_header
        from kestrel_sovereign.endpoints.agent import router

        app = FastAPI()
        app.include_router(router)
        mock_agent = MagicMock()
        mock_agent.cancel_current_request = MagicMock(return_value=True)
        mock_agent.wait_for_request_completion = AsyncMock(return_value=None)
        app.state.agent = mock_agent

        response = TestClient(app).post(
            "/api/agent/stop",
            headers={"X-Request-ID": invocation_id_response_header("header ☃")},
            json={"request_id": "body literal %E2%98%83"},
        )

        assert response.status_code == 200, response.text
        mock_agent.cancel_current_request.assert_called_once_with(
            request_id="body literal %E2%98%83"
        )

    @pytest.mark.asyncio
    async def test_stream_accepts_its_own_header_echo_without_forking_identity(self):
        """The shared invoke/stream ingress decodes the echoed wire key once."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from kestrel_sovereign.agent.invocation import invocation_id_response_header
        from kestrel_sovereign.endpoints.agent import router
        from kestrel_sovereign.rate_limit import limiter

        request_id = "stream ☃ / 100% %E2%98%83?retry=yes"
        header_echo = invocation_id_response_header(request_id)
        received_ids = []

        async def _stream(*_args, **kwargs):
            received_ids.append(kwargs["request_id"])
            yield "ok"

        app = FastAPI()
        app.state.limiter = limiter
        app.include_router(router)
        mock_agent = MagicMock()
        mock_agent.process_input_streaming = _stream
        mock_agent.register_active_request = MagicMock()
        mock_agent._cleanup_cancelled_request = MagicMock()
        mock_agent.is_request_cancelled = MagicMock(return_value=False)
        mock_agent.storage.resolve_session_id = AsyncMock(side_effect=lambda value: value)
        app.state.agent = mock_agent

        response = TestClient(app).post(
            "/api/agent/stream",
            headers={"X-Request-ID": header_echo},
            json={"input": "teach this"},
        )

        assert response.status_code == 200, response.text
        assert response.headers["X-Request-ID"] == header_echo
        assert received_ids == [request_id]
        mock_agent.register_active_request.assert_called_once_with(request_id)
        mock_agent._cleanup_cancelled_request.assert_called_once_with(request_id)

    @pytest.mark.asyncio
    async def test_invoke_accepts_its_own_header_echo_without_forking_identity(self):
        """Non-streaming invocation shares the canonical header wire contract."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from kestrel_sovereign.agent.invocation import invocation_id_response_header
        from kestrel_sovereign.endpoints.agent import router
        from kestrel_sovereign.rate_limit import limiter

        request_id = "invoke ☃ / 100% %E2%98%83?retry=yes"
        header_echo = invocation_id_response_header(request_id)
        app = FastAPI()
        app.state.limiter = limiter
        app.include_router(router)
        mock_agent = MagicMock()
        mock_agent.process_input = AsyncMock(return_value="ok")
        mock_agent.register_active_request = MagicMock()
        mock_agent._cleanup_cancelled_request = MagicMock()
        mock_agent.storage.resolve_session_id = AsyncMock(side_effect=lambda value: value)
        mock_agent._conversation_response_identity = MagicMock(return_value={})
        app.state.agent = mock_agent

        response = TestClient(app).post(
            "/api/agent/invoke",
            headers={"X-Request-ID": header_echo},
            json={"input": "teach this"},
        )

        assert response.status_code == 200, response.text
        assert response.headers["X-Request-ID"] == header_echo
        assert mock_agent.process_input.await_args.kwargs["invocation_id"] == request_id
        mock_agent.register_active_request.assert_called_once_with(request_id)
        mock_agent._cleanup_cancelled_request.assert_called_once_with(request_id)

    @pytest.mark.asyncio
    async def test_stream_endpoint_emits_stop_notice_on_empty_cancelled_stream(self):
        """#2674 P2: a strict (fail-closed) response audit stopped before dispatch
        WITHHOLDS every chunk and returns cleanly, so ``process_input_streaming``
        yields nothing and the endpoint's in-loop cancel check never runs. The
        endpoint's post-loop fallback must still surface the standard "Request
        stopped" body — otherwise a stopped strict turn returns a silent, empty
        200 and the user sees no acknowledgement of their stop."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from kestrel_sovereign.endpoints.agent import router
        from kestrel_sovereign.rate_limit import limiter

        app = FastAPI()
        app.state.limiter = limiter
        app.include_router(router)

        async def _empty_stream(*args, **kwargs):
            # Mirrors the strict cancel-before-dispatch branch: yield nothing,
            # return cleanly. The ``yield`` after ``return`` is unreachable but
            # makes this a genuine async generator.
            return
            yield  # pragma: no cover

        mock_agent = MagicMock()
        mock_agent.register_active_request = MagicMock()
        mock_agent.process_input_streaming = _empty_stream
        # The request was cancelled; the withheld turn produced no chunks.
        mock_agent.is_request_cancelled = MagicMock(return_value=True)
        mock_agent._cleanup_cancelled_request = MagicMock()
        mock_agent.storage.resolve_session_id = AsyncMock(side_effect=lambda s: s)
        app.state.agent = mock_agent

        client = TestClient(app)
        response = client.post("/api/agent/stream", json={"input": "hi"})

        assert response.status_code == 200
        # The post-loop fallback surfaced the stop notice on an empty stream.
        assert "Request stopped" in response.text
        # Exactly one notice — the post-loop emit must not double up with any
        # in-loop emit (there were no chunks, so only the fallback fires).
        assert response.text.count("Request stopped") == 1

    @pytest.mark.asyncio
    async def test_stream_endpoint_reuses_client_request_id_for_turn_provenance(self):
        """A stream retry id is validated, echoed, and passed to the turn."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from kestrel_sovereign.endpoints.agent import router
        from kestrel_sovereign.rate_limit import limiter

        received_ids = []

        async def _stream(*args, **kwargs):
            received_ids.append(kwargs["request_id"])
            yield "ok"

        app = FastAPI()
        app.state.limiter = limiter
        app.include_router(router)
        mock_agent = MagicMock()
        mock_agent.register_active_request = MagicMock()
        mock_agent.process_input_streaming = _stream
        mock_agent.is_request_cancelled = MagicMock(return_value=False)
        mock_agent._cleanup_cancelled_request = MagicMock()
        mock_agent.storage.resolve_session_id = AsyncMock(side_effect=lambda s: s)
        app.state.agent = mock_agent

        client = TestClient(app)
        response = client.post(
            "/api/agent/stream",
            json={"input": "teach this", "request_id": "retry-2765"},
        )
        retry_response = client.post(
            "/api/agent/stream",
            json={"input": "teach this", "request_id": "retry-2765"},
        )

        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == "retry-2765"
        assert retry_response.status_code == 200
        assert retry_response.headers["X-Request-ID"] == "retry-2765"
        assert response.headers["X-Stream-Delivery-ID"].startswith("stream:")
        assert (
            response.headers["X-Stream-Delivery-ID"]
            != retry_response.headers["X-Stream-Delivery-ID"]
        )
        assert received_ids == ["retry-2765", "retry-2765"]

    @pytest.mark.asyncio
    async def test_stream_endpoint_encodes_unicode_retry_id_without_orphaning_lifecycle(self):
        """UTF-8 retry IDs remain raw to the turn and safe in response headers."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from kestrel_sovereign.endpoints.agent import router
        from kestrel_sovereign.rate_limit import limiter

        received_ids = []

        async def _stream(*args, **kwargs):
            received_ids.append(kwargs["request_id"])
            yield "ok"

        app = FastAPI()
        app.state.limiter = limiter
        app.include_router(router)
        mock_agent = MagicMock()
        mock_agent.register_active_request = MagicMock()
        mock_agent.process_input_streaming = _stream
        mock_agent.is_request_cancelled = MagicMock(return_value=False)
        mock_agent._cleanup_cancelled_request = MagicMock()
        mock_agent.storage.resolve_session_id = AsyncMock(side_effect=lambda s: s)
        app.state.agent = mock_agent

        response = TestClient(app).post(
            "/api/agent/stream",
            json={"input": "teach this", "request_id": "retry-☃"},
        )

        assert response.status_code == 200, response.text
        assert response.headers["X-Request-ID"] == "retry-%E2%98%83"
        assert received_ids == ["retry-☃"]
        mock_agent._cleanup_cancelled_request.assert_called_once_with("retry-☃")

    @pytest.mark.asyncio
    async def test_stream_setup_failure_cleans_tap_and_request_lifecycle(self, monkeypatch):
        """A response-construction error cannot strand a registered stream."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import kestrel_sovereign.endpoints.agent as agent_endpoints
        from kestrel_sovereign.endpoints.agent import router
        from kestrel_sovereign.rate_limit import limiter
        from kestrel_sovereign.streams.tap import AgentStreamTap

        class ResponseConstructionFailure:
            def __init__(self, *args, **kwargs):
                raise UnicodeEncodeError("latin-1", "☃", 0, 1, "ordinal not in range")

        AgentStreamTap.reset()
        app = FastAPI()
        app.state.limiter = limiter
        app.include_router(router)
        mock_agent = MagicMock()
        mock_agent.register_active_request = MagicMock()
        mock_agent._cleanup_cancelled_request = MagicMock()
        mock_agent.storage.resolve_session_id = AsyncMock(side_effect=lambda s: s)
        app.state.agent = mock_agent
        monkeypatch.setattr(agent_endpoints, "StreamingResponse", ResponseConstructionFailure)

        response = TestClient(app, raise_server_exceptions=False).post(
            "/api/agent/stream",
            json={"input": "teach this", "request_id": "cleanup-retry-2765"},
        )

        assert response.status_code == 500
        mock_agent._cleanup_cancelled_request.assert_called_once_with("cleanup-retry-2765")
        assert AgentStreamTap.get_instance()._queues == {}

    @pytest.mark.asyncio
    async def test_stream_endpoint_rejects_invalid_client_request_id(self):
        """Malformed retry ids never reach cancellation or provenance code."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from kestrel_sovereign.endpoints.agent import router
        from kestrel_sovereign.rate_limit import limiter

        app = FastAPI()
        app.state.limiter = limiter
        app.include_router(router)
        app.state.agent = MagicMock()

        client = TestClient(app)
        response = client.post(
            "/api/agent/stream",
            json={"input": "teach this", "request_id": ""},
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_stream_endpoint_late_cancel_after_completed_output_no_stop_notice(self):
        """#2674 P2 race: a normal/strict-approved turn streams its full answer,
        then the user's cancellation becomes visible AFTER the final chunk has
        already been yielded but BEFORE the endpoint's async generator exits.

        The post-loop fallback must NOT retroactively append "Request stopped"
        to an already-delivered, complete response. The fallback is gated on the
        stream having produced NO output — so a turn that emitted any user-visible
        chunk keeps its clean ending even when ``is_request_cancelled`` flips True
        during teardown."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from kestrel_sovereign.endpoints.agent import router
        from kestrel_sovereign.rate_limit import limiter

        app = FastAPI()
        app.state.limiter = limiter
        app.include_router(router)

        async def _completed_stream(*args, **kwargs):
            # A normal turn that fully streams its answer to the client.
            yield "The answer "
            yield "is 42."

        # ``is_request_cancelled`` is consulted once per chunk inside the loop
        # (must be False so both chunks pass through). It would return True on
        # any later call — modeling a stop that lands only after the final chunk
        # was already yielded. On the pre-fix code the post-loop fallback would
        # call this a 3rd time, see True, and wrongly append the stop notice; the
        # fixed fallback short-circuits on ``response_chunk_yielded`` and never
        # reaches this predicate again.
        cancel_calls = {"n": 0}

        def late_cancel(request_id=None):
            cancel_calls["n"] += 1
            # False for the two in-loop checks (both chunks stream through);
            # True on any subsequent call (the late cancellation).
            return cancel_calls["n"] > 2

        mock_agent = MagicMock()
        mock_agent.register_active_request = MagicMock()
        mock_agent.process_input_streaming = _completed_stream
        mock_agent.is_request_cancelled = late_cancel
        mock_agent._cleanup_cancelled_request = MagicMock()
        mock_agent.storage.resolve_session_id = AsyncMock(side_effect=lambda s: s)
        app.state.agent = mock_agent

        client = TestClient(app)
        response = client.post("/api/agent/stream", json={"input": "hi"})

        assert response.status_code == 200
        # The complete answer reached the client...
        assert "The answer is 42." in response.text
        # ...and was NOT retroactively labeled stopped by the late cancellation.
        assert "Request stopped" not in response.text
        # Only the two in-loop checks ran: the post-loop fallback short-circuited
        # on ``response_chunk_yielded`` and never consulted the cancellation flag
        # a third time. (On the pre-fix code this predicate fired a 3rd time and
        # the stop notice leaked onto the completed answer.)
        assert cancel_calls["n"] == 2


class TestStreamEndpointErrorContract:
    """#2674 findings 4 & 6: the streaming endpoint must NEVER reflect arbitrary
    exception text to the client. ``str(e)`` for an internal exception is
    untrusted — a mid-buffer failure under a strict audit could carry the
    withheld response (a proven ``RAW_EXCEPTION_RESPONSE_MARKER`` leak). The
    coherent API contract is a CONSTANT safe failure body for every streaming
    internal exception; the full detail goes to the operator log only.

    #2674 finding 4: a typed ``LLMStreamingError`` route failure must NOT reflect
    ``provider`` either — that field is an unvalidated free string accepted at
    construction (Terra leaked ``ROUTE_FIELD_UNBOUNDED_MARKER__WITHHELD_TEXT``
    through it). The endpoint now emits a CONSTANT "your selected model route"
    label with the same no-blind-fallback / recovery guidance, so the user can
    still recover without a silent model swap while nothing route-, underlying-,
    or message-derived reaches them. The failing route stays operator-log only."""

    def _app_with_stream(self, stream_fn):
        from fastapi import FastAPI
        from kestrel_sovereign.endpoints.agent import router
        from kestrel_sovereign.rate_limit import limiter

        app = FastAPI()
        app.state.limiter = limiter
        app.include_router(router)

        mock_agent = MagicMock()
        mock_agent.register_active_request = MagicMock()
        mock_agent.process_input_streaming = stream_fn
        mock_agent.is_request_cancelled = MagicMock(return_value=False)
        mock_agent._cleanup_cancelled_request = MagicMock()
        mock_agent.storage.resolve_session_id = AsyncMock(side_effect=lambda s: s)
        app.state.agent = mock_agent
        return app

    @pytest.mark.asyncio
    async def test_generic_exception_yields_constant_not_raw_text(self):
        from fastapi.testclient import TestClient

        marker = "RAW_EXCEPTION_RESPONSE_MARKER_should_never_surface"

        async def _boom(*args, **kwargs):
            yield "some withheld pre-verdict bytes"  # (never reaches client here)
            raise RuntimeError(marker)

        app = self._app_with_stream(_boom)
        client = TestClient(app)
        response = client.post("/api/agent/stream", json={"input": "hi"})

        assert response.status_code == 200
        # The arbitrary exception text is NOT reflected to the user...
        assert marker not in response.text
        # ...and a stable, constant safe failure body is present instead.
        assert "Error generating response." in response.text

    @pytest.mark.asyncio
    async def test_generic_exception_body_is_constant_across_errors(self):
        """Two different internal exceptions produce the SAME safe body — the
        constant contract, not a per-exception rendering."""
        from fastapi.testclient import TestClient

        def _run(exc):
            async def _boom(*args, **kwargs):
                raise exc
                yield  # pragma: no cover

            app = self._app_with_stream(_boom)
            return TestClient(app).post(
                "/api/agent/stream", json={"input": "hi"}
            ).text

        body_a = _run(RuntimeError("first distinctive DETAIL_AAA"))
        body_b = _run(ValueError("second distinctive DETAIL_BBB"))
        assert "DETAIL_AAA" not in body_a
        assert "DETAIL_BBB" not in body_b
        # The user-visible failure text is identical regardless of the exception.
        marker = "⚠️ **Error generating response.**"
        assert marker in body_a and marker in body_b

    @pytest.mark.asyncio
    async def test_llm_streaming_error_uses_constant_route_label(self):
        """#2674 finding 4: a typed route/mandate failure keeps the
        no-blind-fallback / recovery guidance but does NOT reflect ``provider``
        (an unvalidated free string). The user sees a CONSTANT selected-route
        label, never the route name, and no withheld prose."""
        from fastapi.testclient import TestClient
        from kestrel_sovereign.llm.streaming import LLMStreamingError

        async def _route_fail(*args, **kwargs):
            raise LLMStreamingError(
                "route mandate failed",
                provider="openai:plan",
                underlying="401 Unauthorized",
            )
            yield  # pragma: no cover

        app = self._app_with_stream(_route_fail)
        client = TestClient(app)
        response = client.post("/api/agent/stream", json={"input": "hi"})

        assert response.status_code == 200
        # The provider/route name is NOT reflected...
        assert "openai:plan" not in response.text
        # ...but the constant label + recovery guidance is present.
        assert "Your selected model route failed" in response.text
        assert "No fallback response was generated" in response.text

    @pytest.mark.asyncio
    async def test_llm_streaming_error_does_not_leak_underlying_text(self):
        """#2674 findings 1 & 4: neither the ``underlying`` exception / ``str(e)``
        (raw adapter text) NOR the ``provider`` route (an unvalidated free string)
        may reach the client. Faithfully models the ``raise LLMStreamingError(
        f"Selected route {name} failed: {e}", underlying=e)`` site where a marker
        rides the message AND the underlying — and also proves the provider label
        itself is never reflected."""
        from fastapi.testclient import TestClient
        from kestrel_sovereign.llm.streaming import LLMStreamingError

        marker = "UNDERLYING_ADAPTER_MARKER_should_never_surface"

        async def _route_fail(*args, **kwargs):
            underlying = RuntimeError(marker)
            raise LLMStreamingError(
                f"Selected route openai:plan failed: {underlying}",
                provider="openai:plan",
                underlying=underlying,
            )
            yield  # pragma: no cover

        app = self._app_with_stream(_route_fail)
        client = TestClient(app)
        response = client.post("/api/agent/stream", json={"input": "hi"})

        assert response.status_code == 200
        # Neither the raw underlying text NOR the provider route reaches the
        # client...
        assert marker not in response.text
        assert "openai:plan" not in response.text
        # ...but the constant no-fallback guidance still does, so the user can
        # recover without a silent model swap.
        assert "No fallback response was generated" in response.text

    @pytest.mark.asyncio
    async def test_llm_streaming_error_after_partial_yield_hides_underlying(self):
        """A late adapter failure: the stream yields partial prose and THEN the
        routing layer raises ``LLMStreamingError`` wrapping that late exception.
        On the direct (unbuffered) stream the partial prose already reached the
        client, but the wrapped underlying marker must still never appear."""
        from fastapi.testclient import TestClient
        from kestrel_sovereign.llm.streaming import LLMStreamingError

        marker = "LATE_UNDERLYING_MARKER_should_never_surface"

        async def _partial_then_fail(*args, **kwargs):
            yield "here is some partial answer prose "
            underlying = RuntimeError(marker)
            raise LLMStreamingError(
                f"Selected route openai:plan failed: {underlying}",
                provider="openai:plan",
                underlying=underlying,
            )

        app = self._app_with_stream(_partial_then_fail)
        client = TestClient(app)
        response = client.post("/api/agent/stream", json={"input": "hi"})

        assert response.status_code == 200
        # Partial prose that streamed before the late failure is present (direct
        # path is incremental)...
        assert "partial answer prose" in response.text
        # ...but neither the wrapped underlying marker NOR the provider route is.
        assert marker not in response.text
        assert "openai:plan" not in response.text
        assert "No fallback response was generated" in response.text

    @pytest.mark.asyncio
    async def test_llm_streaming_error_provider_field_marker_never_surfaces(self):
        """#2674 finding 4 (direct Terra repro): ``LLMStreamingError.provider``
        accepts ARBITRARY strings at construction, so a marker smuggled into the
        provider field itself must not reach the client. The endpoint reflects a
        CONSTANT route label, never ``provider``, so the marker in message,
        underlying AND provider are all absent while the guidance remains."""
        from fastapi.testclient import TestClient
        from kestrel_sovereign.llm.streaming import LLMStreamingError

        marker = "ROUTE_FIELD_UNBOUNDED_MARKER__WITHHELD_TEXT"

        async def _route_fail(*args, **kwargs):
            raise LLMStreamingError(
                f"route failed {marker}",
                provider=marker,       # attacker/content-controlled free string
                underlying=RuntimeError(marker),
            )
            yield  # pragma: no cover

        app = self._app_with_stream(_route_fail)
        client = TestClient(app)
        response = client.post("/api/agent/stream", json={"input": "hi"})

        assert response.status_code == 200
        # The marker rode message, underlying AND provider — none may surface.
        assert marker not in response.text
        # The constant label + recovery guidance still reach the user.
        assert "Your selected model route failed" in response.text
        assert "No fallback response was generated" in response.text
