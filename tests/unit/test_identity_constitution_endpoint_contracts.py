"""Focused contract tests for identity and constitution endpoints."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


def _prepare_app(agent):
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original = {
        "lifespan": app.router.lifespan_context,
        "agent": getattr(app.state, "agent", None),
        "manager": getattr(app.state, "agent_manager", None),
    }
    app.router.lifespan_context = noop_lifespan
    app.state.agent = agent
    app.state.agent_manager = None
    return app, original


def _restore_app(app, original):
    app.router.lifespan_context = original["lifespan"]
    app.state.agent = original["agent"]
    app.state.agent_manager = original["manager"]


def test_identity_endpoint_returns_avatar_and_constitution_fields():
    agent_node = SimpleNamespace(
        node_type="agent",
        properties={
            "name": "Claw",
            "created_at": "2026-03-17T00:00:00Z",
            "constitution_hash": "const-hash",
            "avatar_hash": "avatar-hash",
        },
    )
    storage = MagicMock()
    storage.get_node = AsyncMock(return_value=agent_node)

    agent = MagicMock(agent_id="did:pkh:eip155:1:0xabc", storage=storage)
    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get("/api/identity", headers={"X-API-Key": "test-key"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["did"] == "did:pkh:eip155:1:0xabc"
        assert payload["name"] == "Claw"
        assert payload["constitution_hash"] == "const-hash"
        assert payload["avatar_hash"] == "avatar-hash"
        assert payload["avatar_url"] == "/api/files/avatar-hash"
    finally:
        _restore_app(app, original)


def test_constitution_endpoint_returns_text_hash_and_metadata():
    agent_node = SimpleNamespace(
        node_type="agent",
        properties={"constitution_hash": "const-hash"},
    )
    constitution_node = SimpleNamespace(
        node_type="constitution",
        properties={"version": 1, "anchored": True},
    )
    storage = MagicMock()
    storage.get_node = AsyncMock(side_effect=[agent_node, constitution_node])

    agent = MagicMock(agent_id="did:pkh:eip155:1:0xabc", storage=storage)
    agent._get_governing_constitution = AsyncMock(return_value="We the sovereign...")

    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.get("/api/constitution", headers={"X-API-Key": "test-key"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["text"] == "We the sovereign..."
        assert payload["hash"] == "const-hash"
        assert payload["metadata"]["anchored"] is True
        assert payload["verified"] is True
    finally:
        _restore_app(app, original)
