"""
Unit tests for host.py — Kestrel Host thin FastAPI proxy.

Tests the host endpoints using ASGI test client with mocked agent backends.
"""

import os
import pytest
import httpx
from unittest.mock import patch, MagicMock

from kestrel_sovereign.rookery.config import (
    RookeryConfig,
    HostConfig,
    LocalAgentConfig,
    RemoteAgentConfig,
)
from kestrel_sovereign.rookery.process_manager import ProcessManager, AgentProcess


async def make_host_app(config: RookeryConfig):
    """Create and start a host app with a given config.

    Patches load_rookery_config so the lifespan uses our test config.
    Returns (app, lifespan_manager) — caller must use them in async with.
    """
    with patch("host.load_rookery_config", return_value=config):
        # Reload to pick up the patched function reference
        import importlib
        import host as host_module
        # Patch at the module level BEFORE reload
        host_module.load_rookery_config = lambda: config
        # Now the app references the patched function

        from asgi_lifespan import LifespanManager
        return host_module.app, LifespanManager


# All host tests disable autostart to avoid spawning real processes
@pytest.fixture(autouse=True)
def disable_autostart():
    """Disable agent autostart during host tests."""
    with patch.dict(os.environ, {"KESTREL_HOST_AUTOSTART": "false"}):
        yield


class TestHealthEndpoint:
    """Tests for GET /health."""

    @pytest.mark.asyncio
    async def test_health_returns_ok(self):
        """Health endpoint returns status ok with agent statuses."""
        config = RookeryConfig(
            agents={
                "claw": LocalAgentConfig(data_dir="agent_data/claw", port=9901),
            }
        )

        import host as host_module
        original_fn = host_module.load_rookery_config
        host_module.load_rookery_config = lambda: config
        try:
            from asgi_lifespan import LifespanManager
            async with LifespanManager(host_module.app) as manager:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=manager.app),
                    base_url="http://testhost",
                ) as client:
                    resp = await client.get("/health")

            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["role"] == "host"
            assert "agents" in data
        finally:
            host_module.load_rookery_config = original_fn

    @pytest.mark.asyncio
    async def test_health_shows_offline_agents(self):
        """Health endpoint shows offline agents when they can't be reached."""
        config = RookeryConfig(
            agents={
                "claw": LocalAgentConfig(data_dir="agent_data/claw", port=9901),
                "helper": LocalAgentConfig(data_dir="agent_data/helper", port=9902),
            }
        )

        import host as host_module
        original_fn = host_module.load_rookery_config
        host_module.load_rookery_config = lambda: config
        try:
            from asgi_lifespan import LifespanManager
            async with LifespanManager(host_module.app) as manager:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=manager.app),
                    base_url="http://testhost",
                ) as client:
                    resp = await client.get("/health")

            data = resp.json()
            for name, status in data["agents"].items():
                assert status["status"] == "offline"
        finally:
            host_module.load_rookery_config = original_fn

    @pytest.mark.asyncio
    async def test_health_no_auth_required(self):
        """Health endpoint is accessible without auth."""
        config = RookeryConfig(agents={})

        import host as host_module
        original_fn = host_module.load_rookery_config
        host_module.load_rookery_config = lambda: config
        try:
            from asgi_lifespan import LifespanManager
            async with LifespanManager(host_module.app) as manager:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=manager.app),
                    base_url="http://testhost",
                ) as client:
                    resp = await client.get("/health")

            assert resp.status_code == 200
        finally:
            host_module.load_rookery_config = original_fn


class TestAuthMiddleware:
    """Tests for authentication middleware."""

    @pytest.mark.asyncio
    async def test_protected_endpoint_requires_auth(self):
        """Protected endpoints return 401 without auth."""
        config = RookeryConfig(agents={})

        import host as host_module
        original_fn = host_module.load_rookery_config
        host_module.load_rookery_config = lambda: config
        try:
            from asgi_lifespan import LifespanManager
            async with LifespanManager(host_module.app) as manager:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=manager.app),
                    base_url="http://testhost",
                ) as client:
                    resp = await client.get("/api/agents")

            assert resp.status_code == 401
        finally:
            host_module.load_rookery_config = original_fn

    @pytest.mark.asyncio
    async def test_api_key_header_auth(self):
        """X-API-Key header authentication works."""
        test_key = "test-key-12345"
        config = RookeryConfig(agents={})

        import host as host_module
        original_fn = host_module.load_rookery_config
        host_module.load_rookery_config = lambda: config
        try:
            with patch.dict(os.environ, {"KESTREL_API_KEY": test_key}):
                from asgi_lifespan import LifespanManager
                async with LifespanManager(host_module.app) as manager:
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=manager.app),
                        base_url="http://testhost",
                    ) as client:
                        resp = await client.get(
                            "/api/agents",
                            headers={"X-API-Key": test_key},
                        )

            assert resp.status_code == 200
        finally:
            host_module.load_rookery_config = original_fn

    @pytest.mark.asyncio
    async def test_bearer_token_auth(self):
        """Bearer token authentication works."""
        test_key = "test-bearer-key"
        config = RookeryConfig(agents={})

        import host as host_module
        original_fn = host_module.load_rookery_config
        host_module.load_rookery_config = lambda: config
        try:
            with patch.dict(os.environ, {"KESTREL_API_KEY": test_key}):
                from asgi_lifespan import LifespanManager
                async with LifespanManager(host_module.app) as manager:
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=manager.app),
                        base_url="http://testhost",
                    ) as client:
                        resp = await client.get(
                            "/api/agents",
                            headers={"Authorization": f"Bearer {test_key}"},
                        )

            assert resp.status_code == 200
        finally:
            host_module.load_rookery_config = original_fn

    @pytest.mark.asyncio
    async def test_query_param_auth_rejected_on_non_sse_path(self):
        """Query parameter authentication is rejected on non-SSE paths (#160)."""
        test_key = "test-query-key"
        config = RookeryConfig(agents={})

        import host as host_module
        original_fn = host_module.load_rookery_config
        host_module.load_rookery_config = lambda: config
        try:
            with patch.dict(os.environ, {"KESTREL_API_KEY": test_key}):
                from asgi_lifespan import LifespanManager
                async with LifespanManager(host_module.app) as manager:
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=manager.app),
                        base_url="http://testhost",
                    ) as client:
                        resp = await client.get(
                            "/api/agents",
                            params={"api_key": test_key},
                        )

            assert resp.status_code == 401
        finally:
            host_module.load_rookery_config = original_fn


class TestListAgents:
    """Tests for GET /api/agents."""

    @pytest.mark.asyncio
    async def test_list_agents_returns_all_agents(self):
        """List agents returns all configured agents."""
        test_key = "test-key"
        config = RookeryConfig(
            agents={
                "claw": LocalAgentConfig(data_dir="agent_data/claw", port=9901),
                "helper": LocalAgentConfig(data_dir="agent_data/helper", port=9902),
            }
        )

        import host as host_module
        original_fn = host_module.load_rookery_config
        host_module.load_rookery_config = lambda: config
        try:
            with patch.dict(os.environ, {"KESTREL_API_KEY": test_key}):
                from asgi_lifespan import LifespanManager
                async with LifespanManager(host_module.app) as manager:
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=manager.app),
                        base_url="http://testhost",
                    ) as client:
                        resp = await client.get(
                            "/api/agents",
                            headers={"X-API-Key": test_key},
                        )

            assert resp.status_code == 200
            data = resp.json()
            assert len(data["agents"]) == 2

            names = {a["name"] for a in data["agents"]}
            assert names == {"claw", "helper"}
        finally:
            host_module.load_rookery_config = original_fn

    @pytest.mark.asyncio
    async def test_list_agents_shows_offline_status(self):
        """Agents that can't be reached show as offline."""
        test_key = "test-key"
        config = RookeryConfig(
            agents={
                "claw": LocalAgentConfig(data_dir="agent_data/claw", port=9901),
            }
        )

        import host as host_module
        original_fn = host_module.load_rookery_config
        host_module.load_rookery_config = lambda: config
        try:
            with patch.dict(os.environ, {"KESTREL_API_KEY": test_key}):
                from asgi_lifespan import LifespanManager
                async with LifespanManager(host_module.app) as manager:
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=manager.app),
                        base_url="http://testhost",
                    ) as client:
                        resp = await client.get(
                            "/api/agents",
                            headers={"X-API-Key": test_key},
                        )

            data = resp.json()
            assert data["agents"][0]["status"] == "offline"
        finally:
            host_module.load_rookery_config = original_fn

    @pytest.mark.asyncio
    async def test_list_agents_includes_type(self):
        """Agent entries include type (local/remote)."""
        test_key = "test-key"
        config = RookeryConfig(
            agents={
                "local-agent": LocalAgentConfig(data_dir="agent_data/a", port=9901),
                "remote-agent": RemoteAgentConfig(url="https://example.com"),
            }
        )

        import host as host_module
        original_fn = host_module.load_rookery_config
        host_module.load_rookery_config = lambda: config
        try:
            with patch.dict(os.environ, {"KESTREL_API_KEY": test_key}):
                from asgi_lifespan import LifespanManager
                async with LifespanManager(host_module.app) as manager:
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=manager.app),
                        base_url="http://testhost",
                    ) as client:
                        resp = await client.get(
                            "/api/agents",
                            headers={"X-API-Key": test_key},
                        )

            data = resp.json()
            types = {a["name"]: a["type"] for a in data["agents"]}
            assert types["local-agent"] == "local"
            assert types["remote-agent"] == "remote"
        finally:
            host_module.load_rookery_config = original_fn


class TestProxyToAgent:
    """Tests for proxy endpoint /api/agents/{agent_id}/*."""

    @pytest.mark.asyncio
    async def test_proxy_nonexistent_agent_returns_404(self):
        """Proxying to unknown agent returns 404."""
        test_key = "test-key"
        config = RookeryConfig(
            agents={
                "claw": LocalAgentConfig(data_dir="agent_data/claw", port=9901),
            }
        )

        import host as host_module
        original_fn = host_module.load_rookery_config
        host_module.load_rookery_config = lambda: config
        try:
            with patch.dict(os.environ, {"KESTREL_API_KEY": test_key}):
                from asgi_lifespan import LifespanManager
                async with LifespanManager(host_module.app) as manager:
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=manager.app),
                        base_url="http://testhost",
                    ) as client:
                        resp = await client.get(
                            "/api/agents/nonexistent/health",
                            headers={"X-API-Key": test_key},
                        )

            assert resp.status_code == 404
            assert "not found" in resp.json()["detail"]
        finally:
            host_module.load_rookery_config = original_fn

    @pytest.mark.asyncio
    async def test_proxy_offline_agent_returns_503(self):
        """Proxying to offline agent returns 503."""
        test_key = "test-key"
        config = RookeryConfig(
            agents={
                "claw": LocalAgentConfig(data_dir="agent_data/claw", port=9901),
            }
        )

        import host as host_module
        original_fn = host_module.load_rookery_config
        host_module.load_rookery_config = lambda: config
        try:
            with patch.dict(os.environ, {"KESTREL_API_KEY": test_key}):
                from asgi_lifespan import LifespanManager
                async with LifespanManager(host_module.app) as manager:
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=manager.app),
                        base_url="http://testhost",
                    ) as client:
                        resp = await client.get(
                            "/api/agents/claw/conversations",
                            headers={"X-API-Key": test_key},
                        )

            assert resp.status_code == 503
            assert "offline" in resp.json()["detail"]
        finally:
            host_module.load_rookery_config = original_fn


# -----------------------------------------------------------------------
# Process Management Endpoint tests
# -----------------------------------------------------------------------

class TestProcessManagementEndpoints:
    """Tests for POST /api/agents/{id}/start, /stop, GET /status, /logs."""

    @pytest.mark.asyncio
    async def test_start_nonexistent_agent_returns_404(self):
        """Starting an unknown agent returns 404."""
        test_key = "test-key"
        config = RookeryConfig(
            agents={
                "claw": LocalAgentConfig(data_dir="agent_data/claw", port=9901),
            }
        )

        import host as host_module
        original_fn = host_module.load_rookery_config
        host_module.load_rookery_config = lambda: config
        try:
            with patch.dict(os.environ, {"KESTREL_API_KEY": test_key}):
                from asgi_lifespan import LifespanManager
                async with LifespanManager(host_module.app) as manager:
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=manager.app),
                        base_url="http://testhost",
                    ) as client:
                        resp = await client.post(
                            "/api/agents/nonexistent/start",
                            headers={"X-API-Key": test_key},
                        )

            assert resp.status_code == 404
        finally:
            host_module.load_rookery_config = original_fn

    @pytest.mark.asyncio
    async def test_stop_nonexistent_agent_returns_404(self):
        """Stopping an unknown agent returns 404."""
        test_key = "test-key"
        config = RookeryConfig(
            agents={
                "claw": LocalAgentConfig(data_dir="agent_data/claw", port=9901),
            }
        )

        import host as host_module
        original_fn = host_module.load_rookery_config
        host_module.load_rookery_config = lambda: config
        try:
            with patch.dict(os.environ, {"KESTREL_API_KEY": test_key}):
                from asgi_lifespan import LifespanManager
                async with LifespanManager(host_module.app) as manager:
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=manager.app),
                        base_url="http://testhost",
                    ) as client:
                        resp = await client.post(
                            "/api/agents/nonexistent/stop",
                            headers={"X-API-Key": test_key},
                        )

            assert resp.status_code == 404
        finally:
            host_module.load_rookery_config = original_fn

    @pytest.mark.asyncio
    async def test_status_nonexistent_agent_returns_404(self):
        """Status for unknown agent returns 404."""
        test_key = "test-key"
        config = RookeryConfig(
            agents={
                "claw": LocalAgentConfig(data_dir="agent_data/claw", port=9901),
            }
        )

        import host as host_module
        original_fn = host_module.load_rookery_config
        host_module.load_rookery_config = lambda: config
        try:
            with patch.dict(os.environ, {"KESTREL_API_KEY": test_key}):
                from asgi_lifespan import LifespanManager
                async with LifespanManager(host_module.app) as manager:
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=manager.app),
                        base_url="http://testhost",
                    ) as client:
                        resp = await client.get(
                            "/api/agents/nonexistent/status",
                            headers={"X-API-Key": test_key},
                        )

            assert resp.status_code == 404
        finally:
            host_module.load_rookery_config = original_fn

    @pytest.mark.asyncio
    async def test_logs_nonexistent_agent_returns_404(self):
        """Logs for unknown agent returns 404."""
        test_key = "test-key"
        config = RookeryConfig(
            agents={
                "claw": LocalAgentConfig(data_dir="agent_data/claw", port=9901),
            }
        )

        import host as host_module
        original_fn = host_module.load_rookery_config
        host_module.load_rookery_config = lambda: config
        try:
            with patch.dict(os.environ, {"KESTREL_API_KEY": test_key}):
                from asgi_lifespan import LifespanManager
                async with LifespanManager(host_module.app) as manager:
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=manager.app),
                        base_url="http://testhost",
                    ) as client:
                        resp = await client.get(
                            "/api/agents/nonexistent/logs",
                            headers={"X-API-Key": test_key},
                        )

            assert resp.status_code == 404
        finally:
            host_module.load_rookery_config = original_fn

    @pytest.mark.asyncio
    async def test_status_registered_agent(self):
        """Status for registered agent returns process info."""
        test_key = "test-key"
        config = RookeryConfig(
            agents={
                "claw": LocalAgentConfig(data_dir="agent_data/claw", port=9901),
            }
        )

        import host as host_module
        original_fn = host_module.load_rookery_config
        host_module.load_rookery_config = lambda: config
        try:
            with patch.dict(os.environ, {"KESTREL_API_KEY": test_key}):
                from asgi_lifespan import LifespanManager
                async with LifespanManager(host_module.app) as manager:
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=manager.app),
                        base_url="http://testhost",
                    ) as client:
                        resp = await client.get(
                            "/api/agents/claw/status",
                            headers={"X-API-Key": test_key},
                        )

            assert resp.status_code == 200
            data = resp.json()
            assert data["name"] == "claw"
            assert data["port"] == 9901
            assert data["status"] == "stopped"
        finally:
            host_module.load_rookery_config = original_fn

    @pytest.mark.asyncio
    async def test_stop_registered_agent(self):
        """Stopping a registered (but not running) agent succeeds."""
        test_key = "test-key"
        config = RookeryConfig(
            agents={
                "claw": LocalAgentConfig(data_dir="agent_data/claw", port=9901),
            }
        )

        import host as host_module
        original_fn = host_module.load_rookery_config
        host_module.load_rookery_config = lambda: config
        try:
            with patch.dict(os.environ, {"KESTREL_API_KEY": test_key}):
                from asgi_lifespan import LifespanManager
                async with LifespanManager(host_module.app) as manager:
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=manager.app),
                        base_url="http://testhost",
                    ) as client:
                        resp = await client.post(
                            "/api/agents/claw/stop",
                            headers={"X-API-Key": test_key},
                        )

            assert resp.status_code == 200
            data = resp.json()
            assert data["agent_id"] == "claw"
            assert data["status"] == "stopped"
        finally:
            host_module.load_rookery_config = original_fn

    @pytest.mark.asyncio
    async def test_remote_agent_not_manageable(self):
        """Remote agents cannot be started/stopped (returns 404)."""
        test_key = "test-key"
        config = RookeryConfig(
            agents={
                "remote": RemoteAgentConfig(url="https://example.com"),
            }
        )

        import host as host_module
        original_fn = host_module.load_rookery_config
        host_module.load_rookery_config = lambda: config
        try:
            with patch.dict(os.environ, {"KESTREL_API_KEY": test_key}):
                from asgi_lifespan import LifespanManager
                async with LifespanManager(host_module.app) as manager:
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=manager.app),
                        base_url="http://testhost",
                    ) as client:
                        resp = await client.post(
                            "/api/agents/remote/start",
                            headers={"X-API-Key": test_key},
                        )

            assert resp.status_code == 404
            assert "not a local agent" in resp.json()["detail"]
        finally:
            host_module.load_rookery_config = original_fn
