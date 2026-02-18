"""
Integration test: Kestrel Host proxying to a real agent backend.

Starts a minimal FastAPI "agent" server in-process, then tests
the host's proxy routing against it.
"""

import os
import pytest
import httpx
from unittest.mock import patch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from kestrel_sovereign.rookery.config import (
    RookeryConfig,
    HostConfig,
    LocalAgentConfig,
)


@pytest.fixture
def mock_agent_app():
    """Create a minimal agent FastAPI app that simulates a real agent."""
    agent = FastAPI()

    @agent.get("/health")
    async def health():
        return {"status": "ok", "agent_initialized": True}

    @agent.get("/api/agents")
    async def get_agents():
        return {
            "agents": [
                {
                    "name": "TestAgent",
                    "id": "did:key:z6MkTest123",
                    "status": "online",
                    "version": "1.0.0",
                    "url": "http://localhost:9950",
                    "capabilities": {"streaming": True},
                    "skills": [],
                }
            ]
        }

    @agent.get("/api/conversations")
    async def list_conversations():
        return {"conversations": [{"id": "conv-1", "title": "Test"}]}

    @agent.post("/api/agent")
    async def chat(request: Request):
        body = await request.json()
        return JSONResponse({"response": f"Echo: {body.get('message', '')}"})

    @agent.get("/api/agent/stream")
    async def stream_response():
        async def generate():
            yield "data: {\"chunk\": \"hello\"}\n\n"
            yield "data: {\"chunk\": \"world\"}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    return agent


@pytest.fixture
def integration_config():
    """Config pointing to the mock agent's port."""
    return RookeryConfig(
        host=HostConfig(port=8888),
        agents={
            "test-agent": LocalAgentConfig(
                data_dir="agent_data/test",
                port=9950,
            ),
        },
    )


def _swap_http_client(host_app, mock_agent_app):
    """Replace the host's httpx client with one routing to the mock agent.

    Returns (mock_client, original_client) — caller must close both.
    """
    agent_transport = httpx.ASGITransport(app=mock_agent_app)
    mock_client = httpx.AsyncClient(
        transport=agent_transport,
        base_url="http://localhost:9950",
    )
    original_client = host_app.state.http_client
    host_app.state.http_client = mock_client
    return mock_client, original_client


@pytest.mark.integration
class TestHostAgentIntegration:
    """Integration tests with host proxying to a live mock agent."""

    @pytest.mark.asyncio
    async def test_host_proxies_to_agent(self, mock_agent_app, integration_config):
        """Host proxies GET requests to agent and returns agent's response."""
        test_key = "integration-test-key"

        import host as host_module
        the_app = host_module.app
        original_fn = host_module.load_rookery_config
        host_module.load_rookery_config = lambda: integration_config
        try:
            with patch.dict(os.environ, {"KESTREL_API_KEY": test_key}):
                from asgi_lifespan import LifespanManager
                async with LifespanManager(the_app):
                    mock_client, original_client = _swap_http_client(the_app, mock_agent_app)

                    try:
                        async with httpx.AsyncClient(
                            transport=httpx.ASGITransport(app=the_app),
                            base_url="http://testhost",
                        ) as client:
                            resp = await client.get(
                                "/api/agents/test-agent/conversations",
                                headers={"X-API-Key": test_key},
                            )

                        assert resp.status_code == 200
                        data = resp.json()
                        assert "conversations" in data
                        assert data["conversations"][0]["id"] == "conv-1"
                    finally:
                        await mock_client.aclose()
                        await original_client.aclose()
        finally:
            host_module.load_rookery_config = original_fn

    @pytest.mark.asyncio
    async def test_host_aggregates_agent_cards(self, mock_agent_app, integration_config):
        """GET /api/agents aggregates A2A cards from all agents."""
        test_key = "integration-test-key"

        import host as host_module
        the_app = host_module.app
        original_fn = host_module.load_rookery_config
        host_module.load_rookery_config = lambda: integration_config
        try:
            with patch.dict(os.environ, {"KESTREL_API_KEY": test_key}):
                from asgi_lifespan import LifespanManager
                async with LifespanManager(the_app):
                    mock_client, original_client = _swap_http_client(the_app, mock_agent_app)

                    try:
                        async with httpx.AsyncClient(
                            transport=httpx.ASGITransport(app=the_app),
                            base_url="http://testhost",
                        ) as client:
                            resp = await client.get(
                                "/api/agents",
                                headers={"X-API-Key": test_key},
                            )

                        assert resp.status_code == 200
                        data = resp.json()
                        assert len(data["agents"]) == 1

                        agent_entry = data["agents"][0]
                        assert agent_entry["name"] == "test-agent"
                        assert agent_entry["status"] == "online"
                        assert "card" in agent_entry
                        assert agent_entry["card"]["name"] == "TestAgent"
                        assert agent_entry["id"] == "did:key:z6MkTest123"
                    finally:
                        await mock_client.aclose()
                        await original_client.aclose()
        finally:
            host_module.load_rookery_config = original_fn

    @pytest.mark.asyncio
    async def test_host_health_shows_online_agent(self, mock_agent_app, integration_config):
        """GET /health shows agent as online when it's reachable."""
        import host as host_module
        the_app = host_module.app
        original_fn = host_module.load_rookery_config
        host_module.load_rookery_config = lambda: integration_config
        try:
            from asgi_lifespan import LifespanManager
            async with LifespanManager(the_app):
                mock_client, original_client = _swap_http_client(the_app, mock_agent_app)

                try:
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=the_app),
                        base_url="http://testhost",
                    ) as client:
                        resp = await client.get("/health")

                    assert resp.status_code == 200
                    data = resp.json()
                    assert data["agents"]["test-agent"]["status"] == "online"
                finally:
                    await mock_client.aclose()
                    await original_client.aclose()
        finally:
            host_module.load_rookery_config = original_fn

    @pytest.mark.asyncio
    async def test_host_proxies_post_with_body(self, mock_agent_app, integration_config):
        """Host proxies POST requests with body to agent."""
        test_key = "integration-test-key"

        import host as host_module
        the_app = host_module.app
        original_fn = host_module.load_rookery_config
        host_module.load_rookery_config = lambda: integration_config
        try:
            with patch.dict(os.environ, {"KESTREL_API_KEY": test_key}):
                from asgi_lifespan import LifespanManager
                async with LifespanManager(the_app):
                    mock_client, original_client = _swap_http_client(the_app, mock_agent_app)

                    try:
                        async with httpx.AsyncClient(
                            transport=httpx.ASGITransport(app=the_app),
                            base_url="http://testhost",
                        ) as client:
                            resp = await client.post(
                                "/api/agents/test-agent/agent",
                                json={"message": "hello"},
                                headers={"X-API-Key": test_key},
                            )

                        assert resp.status_code == 200
                        data = resp.json()
                        assert data["response"] == "Echo: hello"
                    finally:
                        await mock_client.aclose()
                        await original_client.aclose()
        finally:
            host_module.load_rookery_config = original_fn

    @pytest.mark.asyncio
    async def test_host_proxies_sse_stream(self, mock_agent_app, integration_config):
        """Host proxies SSE streaming responses from agent."""
        test_key = "integration-test-key"

        import host as host_module
        the_app = host_module.app
        original_fn = host_module.load_rookery_config
        host_module.load_rookery_config = lambda: integration_config
        try:
            with patch.dict(os.environ, {"KESTREL_API_KEY": test_key}):
                from asgi_lifespan import LifespanManager
                async with LifespanManager(the_app):
                    mock_client, original_client = _swap_http_client(the_app, mock_agent_app)

                    try:
                        async with httpx.AsyncClient(
                            transport=httpx.ASGITransport(app=the_app),
                            base_url="http://testhost",
                        ) as client:
                            resp = await client.get(
                                "/api/agents/test-agent/agent/stream",
                                headers={"X-API-Key": test_key},
                            )

                        assert resp.status_code == 200
                        content = resp.text
                        assert "hello" in content
                        assert "world" in content
                    finally:
                        await mock_client.aclose()
                        await original_client.aclose()
        finally:
            host_module.load_rookery_config = original_fn
