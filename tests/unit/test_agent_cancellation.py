"""
Unit tests for agent request cancellation (stop button).
"""

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
        agent._cancelled_requests = set()
        
        # Bind actual methods
        agent.cancel_current_request = KestrelAgent.cancel_current_request.__get__(agent)
        agent.is_request_cancelled = KestrelAgent.is_request_cancelled.__get__(agent)
        agent._cleanup_cancelled_request = KestrelAgent._cleanup_cancelled_request.__get__(agent)
        
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

    def test_multiple_cancellations_tracked(self, mock_agent):
        """Multiple requests can be cancelled and tracked."""
        mock_agent._cancelled_requests.add("req-1")
        mock_agent._cancelled_requests.add("req-2")
        
        assert mock_agent.is_request_cancelled("req-1") is True
        assert mock_agent.is_request_cancelled("req-2") is True
        assert mock_agent.is_request_cancelled("req-3") is False


class TestStopEndpoint:
    """Tests for the /agent/stop endpoint."""

    @pytest.mark.asyncio
    async def test_stop_endpoint_calls_cancel(self):
        """Stop endpoint calls cancel_current_request on agent."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from endpoints.agent import router
        
        app = FastAPI()
        app.include_router(router)
        
        # Mock agent
        mock_agent = MagicMock()
        mock_agent.cancel_current_request = MagicMock(return_value=True)
        app.state.agent = mock_agent
        
        client = TestClient(app)
        response = client.post("/agent/stop")
        
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
        from endpoints.agent import router
        
        app = FastAPI()
        app.include_router(router)
        
        # Mock agent with no active request
        mock_agent = MagicMock()
        mock_agent.cancel_current_request = MagicMock(return_value=False)
        app.state.agent = mock_agent
        
        client = TestClient(app)
        response = client.post("/agent/stop")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["cancelled"] is False

    @pytest.mark.asyncio
    async def test_stop_endpoint_passes_request_id(self):
        """Stop endpoint forwards explicit request IDs for scoped cancellation."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from endpoints.agent import router

        app = FastAPI()
        app.include_router(router)

        mock_agent = MagicMock()
        mock_agent.cancel_current_request = MagicMock(return_value=True)
        app.state.agent = mock_agent

        client = TestClient(app)
        response = client.post("/agent/stop", json={"request_id": "req-123"})

        assert response.status_code == 200
        mock_agent.cancel_current_request.assert_called_once_with(request_id="req-123")
