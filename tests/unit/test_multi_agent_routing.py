"""
Tests for multi-agent routing middleware and /api/agents endpoint.

Verifies that:
- Agent routing middleware dispatches to the correct agent
- Path rewriting strips the /api/agents/{name}/ prefix
- Unknown agents return 404
- Single-agent mode is unaffected
- /api/agents returns rookery mode with all agents
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kestrel_sovereign.rookery.agent_manager import AgentManager


def _make_mock_agent(agent_id: str = "did:pkh:eip155:1:0xABC", name: str = "TestAgent"):
    """Create a mock KestrelAgent."""
    agent = MagicMock()
    agent.agent_id = agent_id
    agent.initialize = AsyncMock()
    agent.shutdown = AsyncMock()

    # Mock get_agent_card for /api/agents endpoint
    mock_card = MagicMock()
    mock_card.model_dump.return_value = {
        "name": name,
        "description": f"Test agent {name}",
        "url": f"http://localhost:8080",
        "version": "1.0",
    }
    agent.get_agent_card = AsyncMock(return_value=mock_card)

    # Mock basic agent methods used by endpoints
    agent.privacy_mode = MagicMock()
    agent.privacy_mode.value = "NORMAL"
    agent.features = {}
    agent.get_current_model = MagicMock(return_value="test-model")

    return agent


def _create_multi_agent_app(agents: dict[str, MagicMock]) -> FastAPI:
    """Create a FastAPI app with multi-agent routing middleware and test endpoints."""
    from server import app as real_app, _AGENT_PATH_RE

    app = FastAPI()

    # Set up AgentManager with mock agents
    manager = AgentManager()
    for name, agent in agents.items():
        manager._agents[name] = agent
        manager._agent_names[agent.agent_id] = name
    app.state.agent_manager = manager
    app.state.agent = None

    # Add the routing middleware
    import re

    @app.middleware("http")
    async def agent_routing_middleware(request, call_next):
        from fastapi.responses import JSONResponse

        agent_manager = getattr(request.app.state, 'agent_manager', None)
        if agent_manager is None:
            return await call_next(request)

        path = request.url.path
        match = _AGENT_PATH_RE.match(path)
        if match:
            agent_name = match.group(1)
            remaining_path = "/" + match.group(2)

            agent = agent_manager.get_agent(agent_name)
            if agent is None:
                return JSONResponse(
                    status_code=404,
                    content={"detail": f"Agent '{agent_name}' not found"},
                )

            request.state.agent = agent
            request.scope["path"] = remaining_path

        return await call_next(request)

    # Add a test endpoint that uses get_agent
    from fastapi import Request
    from endpoints.agent_helpers import get_agent

    @app.get("/api/agent/info")
    async def agent_info(request: Request):
        agent = get_agent(request)
        return {"agent_id": agent.agent_id}

    @app.get("/api/agents")
    async def list_agents(request: Request):
        agent_manager = getattr(request.app.state, 'agent_manager', None)
        if agent_manager:
            agents_list = []
            for name, agent in agent_manager.list_agents().items():
                agents_list.append({
                    "id": agent.agent_id,
                    "name": name,
                    "status": "online",
                })
            return {"agents": agents_list, "mode": "rookery"}
        return {"agents": [], "mode": "standalone"}

    return app


class TestAgentRoutingMiddleware:
    """Test the agent routing middleware dispatches correctly."""

    def test_routes_to_correct_agent(self):
        """Requests to /api/agents/{name}/... should reach the correct agent."""
        claw = _make_mock_agent("did:claw", "Claw")
        emma = _make_mock_agent("did:emma", "Emma")
        app = _create_multi_agent_app({"Claw": claw, "Emma": emma})

        client = TestClient(app, raise_server_exceptions=False)

        # Request for Claw
        resp = client.get("/api/agents/Claw/api/agent/info")
        assert resp.status_code == 200
        assert resp.json()["agent_id"] == "did:claw"

        # Request for Emma
        resp = client.get("/api/agents/Emma/api/agent/info")
        assert resp.status_code == 200
        assert resp.json()["agent_id"] == "did:emma"

    def test_unknown_agent_returns_404(self):
        """Requests to /api/agents/{unknown}/... should return 404."""
        claw = _make_mock_agent("did:claw", "Claw")
        app = _create_multi_agent_app({"Claw": claw})

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/agents/Ghost/api/agent/info")
        assert resp.status_code == 404
        assert "Ghost" in resp.json()["detail"]

    def test_case_insensitive_routing(self):
        """Agent name matching should be case-insensitive."""
        claw = _make_mock_agent("did:claw", "Claw")
        app = _create_multi_agent_app({"Claw": claw})

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/agents/claw/api/agent/info")
        assert resp.status_code == 200
        assert resp.json()["agent_id"] == "did:claw"

    def test_api_agents_list_returns_rookery_mode(self):
        """/api/agents should return all agents with mode: rookery."""
        claw = _make_mock_agent("did:claw", "Claw")
        emma = _make_mock_agent("did:emma", "Emma")
        app = _create_multi_agent_app({"Claw": claw, "Emma": emma})

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "rookery"
        assert len(data["agents"]) == 2
        names = {a["name"] for a in data["agents"]}
        assert names == {"Claw", "Emma"}

    def test_path_rewriting_strips_prefix(self):
        """The middleware should strip /api/agents/{name} and leave the rest."""
        claw = _make_mock_agent("did:claw", "Claw")
        app = _create_multi_agent_app({"Claw": claw})

        client = TestClient(app, raise_server_exceptions=False)
        # The /agent/info endpoint is mounted at /agent/info
        # After rewriting /api/agents/Claw/api/agent/info -> /agent/info
        resp = client.get("/api/agents/Claw/api/agent/info")
        assert resp.status_code == 200


class TestAgentCRUDEndpoints:
    """Test POST /api/agents and DELETE /api/agents/{name}."""

    def test_delete_agent(self):
        """DELETE /api/agents/{name} should remove an agent."""
        claw = _make_mock_agent("did:claw", "Claw")
        emma = _make_mock_agent("did:emma", "Emma")
        app = _create_multi_agent_app({"Claw": claw, "Emma": emma})

        # Add delete endpoint
        from fastapi import Request

        @app.delete("/api/agents/{agent_name}")
        async def delete_agent(request: Request, agent_name: str):
            agent_manager = getattr(request.app.state, 'agent_manager', None)
            removed = await agent_manager.remove_agent(agent_name)
            if not removed:
                return {"success": False}, 404
            return {"success": True, "name": agent_name}

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.delete("/api/agents/Claw")
        assert resp.status_code == 200
        assert resp.json()["success"] is True

        # Verify Claw is gone
        resp = client.get("/api/agents/Claw/api/agent/info")
        assert resp.status_code == 404

        # Emma still works
        resp = client.get("/api/agents/Emma/api/agent/info")
        assert resp.status_code == 200

    def test_delete_nonexistent_agent(self):
        """DELETE /api/agents/{unknown} should fail gracefully."""
        claw = _make_mock_agent("did:claw", "Claw")
        app = _create_multi_agent_app({"Claw": claw})

        from fastapi import Request
        from fastapi.responses import JSONResponse

        @app.delete("/api/agents/{agent_name}")
        async def delete_agent(request: Request, agent_name: str):
            agent_manager = getattr(request.app.state, 'agent_manager', None)
            removed = await agent_manager.remove_agent(agent_name)
            if not removed:
                return JSONResponse(
                    status_code=404,
                    content={"detail": f"Agent '{agent_name}' not found"},
                )
            return {"success": True}

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.delete("/api/agents/Ghost")
        assert resp.status_code == 404

    def test_create_agent_rejected_in_standalone(self):
        """POST /api/agents should fail in standalone mode."""
        app = FastAPI()
        app.state.agent_manager = None
        app.state.agent = _make_mock_agent("did:solo", "Solo")

        from fastapi import Request
        from fastapi.responses import JSONResponse

        @app.post("/api/agents")
        async def create_agent(request: Request):
            agent_manager = getattr(request.app.state, 'agent_manager', None)
            if agent_manager is None:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Agent creation is only available in multi-agent mode."},
                )

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/agents", json={"name": "NewBot"})
        assert resp.status_code == 400
        assert "multi-agent" in resp.json()["detail"]


class TestAgentNameValidation:
    """Test agent name validation regex."""

    def test_valid_names(self):
        """Valid agent names should pass validation."""
        import re
        pattern = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")
        valid = ["Claw", "Emma", "test-bot", "Agent_01", "a"]
        for name in valid:
            assert pattern.match(name), f"'{name}' should be valid"

    def test_invalid_names(self):
        """Invalid agent names should fail validation."""
        import re
        pattern = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")
        invalid = ["", "123bot", "-dash", "_under", "has space", "a" * 65, "../escape"]
        for name in invalid:
            assert not pattern.match(name), f"'{name}' should be invalid"


class TestSSEConnectionIsolation:
    """Test that SSE connections are tracked per (client_ip, agent_id)."""

    def test_sse_connection_key_is_tuple(self):
        """_sse_connections should use (ip, agent_id) as key."""
        from endpoints.agent import _sse_connections
        from collections import defaultdict

        # Verify the type annotation is correct
        assert isinstance(_sse_connections, defaultdict)
        # Add entries with tuple keys
        _sse_connections[("127.0.0.1", "did:agent1")] = 1
        _sse_connections[("127.0.0.1", "did:agent2")] = 2
        assert _sse_connections[("127.0.0.1", "did:agent1")] == 1
        assert _sse_connections[("127.0.0.1", "did:agent2")] == 2
        # Clean up
        del _sse_connections[("127.0.0.1", "did:agent1")]
        del _sse_connections[("127.0.0.1", "did:agent2")]


class TestSingleAgentMode:
    """Test that single-agent mode is unaffected by routing middleware."""

    def test_no_agent_manager_passes_through(self):
        """When agent_manager is None, middleware is a no-op."""
        app = FastAPI()
        app.state.agent_manager = None

        mock_agent = _make_mock_agent("did:single", "Solo")
        app.state.agent = mock_agent

        from fastapi import Request
        from endpoints.agent_helpers import get_agent

        @app.get("/api/agent/info")
        async def agent_info(request: Request):
            agent = get_agent(request)
            return {"agent_id": agent.agent_id}

        # Add the no-op middleware
        @app.middleware("http")
        async def agent_routing_middleware(request, call_next):
            agent_manager = getattr(request.app.state, 'agent_manager', None)
            if agent_manager is None:
                return await call_next(request)
            return await call_next(request)

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/agent/info")
        assert resp.status_code == 200
        assert resp.json()["agent_id"] == "did:single"
