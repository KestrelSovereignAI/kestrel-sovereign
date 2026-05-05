"""
Unit tests for MultiAgent proxy routing logic.

Tests proxy header building, agent resolution, URL construction,
and the proxy request flow using httpx mock transport.
"""

import pytest
import httpx

from kestrel_sovereign.multi_agent.config import (
    MultiAgentConfig,
    HostConfig,
    LocalAgentConfig,
    RemoteAgentConfig,
)
from kestrel_sovereign.multi_agent.proxy import (
    get_agent_base_url,
    resolve_agent,
    build_proxy_headers,
    proxy_request_streaming,
)


class TestGetAgentBaseUrl:
    """Tests for get_agent_base_url."""

    def test_local_agent_url(self):
        """Local agent URL is http://localhost:{port}."""
        config = LocalAgentConfig(data_dir="agent_data/test", port=8801)
        assert get_agent_base_url(config) == "http://localhost:8801"

    def test_remote_agent_url(self):
        """Remote agent URL is the configured URL."""
        config = RemoteAgentConfig(url="https://remote.example.com")
        assert get_agent_base_url(config) == "https://remote.example.com"

    def test_remote_agent_url_trailing_slash_stripped(self):
        """Remote URLs have trailing slashes stripped."""
        config = RemoteAgentConfig(url="https://remote.example.com/")
        assert get_agent_base_url(config) == "https://remote.example.com"

    def test_local_agent_different_ports(self):
        """Different local agents get different URLs."""
        agent1 = LocalAgentConfig(data_dir="agent_data/a", port=8801)
        agent2 = LocalAgentConfig(data_dir="agent_data/b", port=8802)
        assert get_agent_base_url(agent1) != get_agent_base_url(agent2)


class TestResolveAgent:
    """Tests for resolve_agent."""

    def test_resolve_existing_local_agent(self):
        """Resolve a local agent by name."""
        config = MultiAgentConfig(
            agents={
                "claw": LocalAgentConfig(data_dir="agent_data/claw", port=8801),
            }
        )
        result = resolve_agent("claw", config)
        assert result is not None
        assert isinstance(result, LocalAgentConfig)
        assert result.port == 8801

    def test_resolve_existing_remote_agent(self):
        """Resolve a remote agent by name."""
        config = MultiAgentConfig(
            agents={
                "remote": RemoteAgentConfig(url="https://example.com"),
            }
        )
        result = resolve_agent("remote", config)
        assert result is not None
        assert isinstance(result, RemoteAgentConfig)

    def test_resolve_nonexistent_agent(self):
        """Return None for unknown agent ID."""
        config = MultiAgentConfig(agents={})
        result = resolve_agent("nonexistent", config)
        assert result is None

    def test_resolve_is_case_sensitive(self):
        """Agent names are case-sensitive."""
        config = MultiAgentConfig(
            agents={
                "Claw": LocalAgentConfig(data_dir="agent_data/claw", port=8801),
            }
        )
        assert resolve_agent("claw", config) is None
        assert resolve_agent("Claw", config) is not None


class TestBuildProxyHeaders:
    """Tests for build_proxy_headers."""

    @pytest.fixture
    def mock_request(self):
        """Create a minimal mock request with headers."""
        class MockHeaders:
            def __init__(self, headers: dict):
                self._headers = headers

            def items(self):
                return self._headers.items()

            def get(self, key, default=None):
                return self._headers.get(key, default)

        class MockRequest:
            def __init__(self, headers: dict):
                self.headers = MockHeaders(headers)

        return MockRequest

    def test_forwards_standard_headers(self, mock_request):
        """Standard headers are forwarded."""
        req = mock_request({
            "content-type": "application/json",
            "x-api-key": "test-key",
            "authorization": "Bearer abc123",
        })
        headers = build_proxy_headers(req)
        assert headers["content-type"] == "application/json"
        assert headers["x-api-key"] == "test-key"
        assert headers["authorization"] == "Bearer abc123"

    def test_excludes_host_header(self, mock_request):
        """Host header is excluded (agent has its own host)."""
        req = mock_request({
            "host": "localhost:8888",
            "content-type": "application/json",
        })
        headers = build_proxy_headers(req)
        assert "host" not in headers
        assert "content-type" in headers

    def test_excludes_hop_by_hop_headers(self, mock_request):
        """Hop-by-hop headers (transfer-encoding, connection) are excluded."""
        req = mock_request({
            "transfer-encoding": "chunked",
            "connection": "keep-alive",
            "content-type": "text/plain",
        })
        headers = build_proxy_headers(req)
        assert "transfer-encoding" not in headers
        assert "connection" not in headers
        assert "content-type" in headers

    def test_empty_headers(self, mock_request):
        """Empty headers produce empty result."""
        req = mock_request({})
        headers = build_proxy_headers(req)
        assert headers == {}


class TestProxyRequestStreaming:
    """Tests for proxy_request_streaming with SSE auto-detection."""

    @pytest.fixture
    def multi_agent_config(self):
        return MultiAgentConfig(
            agents={
                "claw": LocalAgentConfig(data_dir="agent_data/claw", port=9901),
            }
        )

    @pytest.mark.asyncio
    async def test_streaming_unknown_agent_returns_404(self, multi_agent_config):
        """Streaming proxy to unknown agent returns 404."""
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/agents/unknown/stream",
            "query_string": b"",
            "headers": [],
        }
        from starlette.requests import Request as StarletteRequest
        request = StarletteRequest(scope)

        async with httpx.AsyncClient() as client:
            response = await proxy_request_streaming(
                request=request,
                agent_id="unknown",
                path="stream",
                config=multi_agent_config,
                client=client,
            )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_streaming_offline_agent_returns_503(self, multi_agent_config):
        """Streaming proxy to offline agent returns 503."""
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/agents/claw/api/agent",
            "query_string": b"",
            "headers": [],
        }
        from starlette.requests import Request as StarletteRequest

        async def receive():
            return {"type": "http.request", "body": b""}

        request = StarletteRequest(scope, receive)

        async with httpx.AsyncClient() as client:
            response = await proxy_request_streaming(
                request=request,
                agent_id="claw",
                path="api/agent",
                config=multi_agent_config,
                client=client,
            )

        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_non_streaming_response_returned_normally(self, multi_agent_config):
        """Non-SSE responses are returned as regular Response objects."""

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"result": "ok"},
                headers={"content-type": "application/json"},
            )

        transport = httpx.MockTransport(mock_handler)

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/agents/claw/api/health",
            "query_string": b"",
            "headers": [],
        }
        from starlette.requests import Request as StarletteRequest

        async def receive():
            return {"type": "http.request", "body": b""}

        request = StarletteRequest(scope, receive)

        async with httpx.AsyncClient(transport=transport) as client:
            response = await proxy_request_streaming(
                request=request,
                agent_id="claw",
                path="api/health",
                config=multi_agent_config,
                client=client,
            )

        assert response.status_code == 200
        # Non-streaming returns a regular Response
        from fastapi.responses import StreamingResponse
        assert not isinstance(response, StreamingResponse)


class TestProxyRemoteAgent:
    """Tests for proxying to remote agents."""

    def test_remote_agent_base_url(self):
        """Remote agent uses configured URL."""
        config = RemoteAgentConfig(url="https://agent.example.com")
        assert get_agent_base_url(config) == "https://agent.example.com"

    def test_resolve_remote_agent(self):
        """Remote agents are resolved by name."""
        config = MultiAgentConfig(
            agents={
                "remote-buddy": RemoteAgentConfig(url="https://agent.example.com"),
            }
        )
        result = resolve_agent("remote-buddy", config)
        assert result is not None
        assert isinstance(result, RemoteAgentConfig)
