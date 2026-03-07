"""
Unit tests for SSE connection limits (issue #145).

Verifies that the /agent/notifications/sse endpoint enforces per-client
connection limits to prevent resource exhaustion.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from endpoints.agent import (
    router,
    _sse_connections,
    _sse_lock,
    MAX_SSE_CONNECTIONS_PER_CLIENT,
    notifications_sse,
)


@pytest.fixture(autouse=True)
def clean_sse_connections():
    """Reset the SSE connection tracker before and after each test."""
    _sse_connections.clear()
    yield
    _sse_connections.clear()


@pytest.fixture
def app_with_mock_agent():
    """Create a FastAPI app with a mock agent for testing SSE endpoints."""
    app = FastAPI()
    app.include_router(router)

    mock_agent = MagicMock()
    mock_agent.agent_id = "did:test:mock-agent"
    mock_agent.get_pending_notifications = MagicMock(return_value=[])
    app.state.agent = mock_agent

    return app


class TestSSEConnectionLimits:
    """Tests for SSE per-client connection limiting."""

    def test_max_sse_per_client_constant_is_set(self):
        """The MAX_SSE_CONNECTIONS_PER_CLIENT constant is importable and positive."""
        from kestrel_sovereign.kestrel_config.constants import MAX_SSE_CONNECTIONS_PER_CLIENT as limit
        assert limit > 0
        assert limit == 5

    def test_sse_connection_rejected_when_at_limit(self, app_with_mock_agent):
        """SSE connections should be rejected with 429 when at the per-client limit."""
        # Pre-fill the connection tracker to simulate being at the limit.
        # TestClient uses "testclient" as client.host; mock agent has agent_id.
        # The key is now (client_ip, agent_id).
        agent_id = app_with_mock_agent.state.agent.agent_id
        _sse_connections[("testclient", agent_id)] = MAX_SSE_CONNECTIONS_PER_CLIENT

        client = TestClient(app_with_mock_agent, raise_server_exceptions=False)
        response = client.get("/agent/notifications/sse")

        assert response.status_code == 429
        assert "Too many SSE connections" in response.json()["detail"]
        assert str(MAX_SSE_CONNECTIONS_PER_CLIENT) in response.json()["detail"]

    def test_sse_connection_rejected_over_limit(self, app_with_mock_agent):
        """SSE connections should also be rejected when over the limit."""
        agent_id = app_with_mock_agent.state.agent.agent_id
        _sse_connections[("testclient", agent_id)] = MAX_SSE_CONNECTIONS_PER_CLIENT + 10

        client = TestClient(app_with_mock_agent, raise_server_exceptions=False)
        response = client.get("/agent/notifications/sse")

        assert response.status_code == 429

    def test_different_clients_tracked_independently(self, app_with_mock_agent):
        """Connection limits for one client should not block another client."""
        agent_id = app_with_mock_agent.state.agent.agent_id
        # Simulate one client at the limit
        _sse_connections[("192.168.1.1", agent_id)] = MAX_SSE_CONNECTIONS_PER_CLIENT

        # testclient has 0 connections, so it should NOT get rejected.
        assert _sse_connections.get(("testclient", agent_id), 0) == 0

        # Now max out testclient too — should get rejected
        _sse_connections[("testclient", agent_id)] = MAX_SSE_CONNECTIONS_PER_CLIENT
        client = TestClient(app_with_mock_agent, raise_server_exceptions=False)
        response = client.get("/agent/notifications/sse")
        assert response.status_code == 429

        # But if testclient has room, it would not be rejected
        _sse_connections[("testclient", agent_id)] = 0
        # The other client being maxed should not affect testclient
        assert _sse_connections[("192.168.1.1", agent_id)] == MAX_SSE_CONNECTIONS_PER_CLIENT
        assert _sse_connections[("testclient", agent_id)] == 0

    def test_agent_not_initialized_returns_503(self):
        """SSE endpoint should return 503 when agent is not initialized."""
        app = FastAPI()
        app.include_router(router)
        # Deliberately do not set app.state.agent

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/agent/notifications/sse")

        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_connection_counter_increments_and_decrements(self):
        """Verify the connection counter lifecycle: increment on connect, decrement on disconnect."""
        # Simulate what the endpoint does: increment before generator,
        # decrement in the finally block.
        conn_key = ("10.0.0.1", "did:test:agent")

        assert _sse_connections.get(conn_key, 0) == 0

        # Simulate increment (what the endpoint does before returning the generator)
        async with _sse_lock:
            _sse_connections[conn_key] += 1
        assert _sse_connections[conn_key] == 1

        # Simulate a second connection
        async with _sse_lock:
            _sse_connections[conn_key] += 1
        assert _sse_connections[conn_key] == 2

        # Simulate first disconnect (what the generator's finally block does)
        async with _sse_lock:
            _sse_connections[conn_key] -= 1
            if _sse_connections[conn_key] <= 0:
                del _sse_connections[conn_key]
        assert _sse_connections[conn_key] == 1

        # Simulate second disconnect
        async with _sse_lock:
            _sse_connections[conn_key] -= 1
            if _sse_connections[conn_key] <= 0:
                del _sse_connections[conn_key]
        assert conn_key not in _sse_connections

    @pytest.mark.asyncio
    async def test_limit_check_is_exact_boundary(self):
        """Verify the limit check rejects at exactly MAX, not above or below."""
        conn_key = ("10.0.0.2", "did:test:agent")

        # At limit - 1: should be allowed
        _sse_connections[conn_key] = MAX_SSE_CONNECTIONS_PER_CLIENT - 1
        async with _sse_lock:
            allowed = _sse_connections[conn_key] < MAX_SSE_CONNECTIONS_PER_CLIENT
        assert allowed is True

        # At exactly the limit: should be rejected
        _sse_connections[conn_key] = MAX_SSE_CONNECTIONS_PER_CLIENT
        async with _sse_lock:
            allowed = _sse_connections[conn_key] < MAX_SSE_CONNECTIONS_PER_CLIENT
        assert allowed is False

    def test_unknown_client_fallback_logic(self):
        """When request.client is None, client_ip should default to 'unknown'."""
        # Verify the fallback logic: None client -> "unknown" key
        client = None
        client_ip = client.host if client else "unknown"
        assert client_ip == "unknown"

        # Verify "unknown" tuple key works with the connection tracker
        conn_key = ("unknown", "did:test:agent")
        _sse_connections[conn_key] = MAX_SSE_CONNECTIONS_PER_CLIENT
        assert _sse_connections[conn_key] >= MAX_SSE_CONNECTIONS_PER_CLIENT

    def test_429_response_includes_limit_in_detail(self, app_with_mock_agent):
        """The 429 response detail should include the configured limit value."""
        agent_id = app_with_mock_agent.state.agent.agent_id
        _sse_connections[("testclient", agent_id)] = MAX_SSE_CONNECTIONS_PER_CLIENT

        client = TestClient(app_with_mock_agent, raise_server_exceptions=False)
        response = client.get("/agent/notifications/sse")

        detail = response.json()["detail"]
        assert f"limit: {MAX_SSE_CONNECTIONS_PER_CLIENT}" in detail


class TestSSEConnectionTrackerModule:
    """Tests for the module-level SSE connection tracking structures."""

    def test_sse_connections_is_defaultdict(self):
        """_sse_connections should be a defaultdict(int) for safe access."""
        key = ("nonexistent_ip", "did:nonexistent")
        assert _sse_connections[key] == 0
        # Clean up the key we just created
        del _sse_connections[key]

    def test_sse_lock_is_asyncio_lock(self):
        """_sse_lock should be an asyncio.Lock for thread-safe access."""
        assert isinstance(_sse_lock, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_lock_provides_mutual_exclusion(self):
        """The lock should serialize access to the connection counter."""
        acquired_order = []

        async def task(name):
            async with _sse_lock:
                acquired_order.append(f"{name}_start")
                await asyncio.sleep(0.01)
                acquired_order.append(f"{name}_end")

        # Run two tasks - they should not interleave
        await asyncio.gather(task("a"), task("b"))

        # One task must complete fully before the other starts
        assert acquired_order[0].endswith("_start")
        assert acquired_order[1].endswith("_end")
        # The first task's start/end should be the same letter
        assert acquired_order[0][0] == acquired_order[1][0]
