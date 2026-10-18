"""Focused contract tests for identity and constitution endpoints."""

from contextlib import asynccontextmanager
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
import httpx
import pytest

pytestmark = pytest.mark.usefixtures("isolated_process_rate_limiter")


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
    # GET /api/identity now resolves the display name through the agent
    # interface (resolve_effective_name), not by reading the node directly, so
    # a volatile-mode rename is reflected live (#2672 review P2). A bare
    # MagicMock returns a MagicMock here (serialized as ``{}``), so give this
    # contract fake the real resolution semantics: prefer a live string name,
    # else the stored node's name, else the default.
    def _resolve_effective_name(node=None, *, default=None):
        live = getattr(agent, "_agent_name", None)
        if isinstance(live, str) and live.strip():
            return live
        props = getattr(node, "properties", None) or {}
        name = props.get("name")
        if isinstance(name, str) and name.strip():
            return name
        return default
    agent._agent_name = None  # no live rename in this contract; fall to node
    agent.resolve_effective_name = _resolve_effective_name
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


@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data",
    "https://127.0.0.1/avatar.png",
    "https://bad..example/avatar.png",
    "https://" + "a" * 64 + ".example/avatar.png",
    "https://.example/avatar.png",
    "https://bad\ud800.example/avatar.png",
    "https://01.2.3.4/avatar.png",
    "https://1。2。3。04/avatar.png",
    "https://public.example/avatar\ud800.png",
    "https://[v1.example.com]/avatar.png",
    "https://[2001:4860:4860::8888%en0]/avatar.png",
    "https://[2001:4860:4860::8888%é]/avatar.png",
])
def test_avatar_url_rejection_uses_shared_policy_before_transport_creation(url, monkeypatch):
    from kestrel_sdk.security import outbound

    def unexpected_dns(*args, **kwargs):
        pytest.fail("malformed or literal private avatar URL reached DNS")

    monkeypatch.setattr("socket.getaddrinfo", unexpected_dns)
    store_avatar = AsyncMock()
    agent = SimpleNamespace(agent_id="did:test:avatar", storage=SimpleNamespace(files=SimpleNamespace(store_avatar=store_avatar)))
    factory = MagicMock(side_effect=AssertionError("unsafe URL reached HTTP transport"))
    monkeypatch.setattr(outbound, "pinned_httpx_async_transport", factory)
    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.post(
                    "/api/identity/avatar", content=json.dumps({"url": url}),
                    headers={"X-API-Key": "test-key", "Content-Type": "application/json"},
                )
        assert response.status_code == 400
        assert "Disallowed avatar URL" in response.json()["detail"]
        factory.assert_not_called()
        store_avatar.assert_not_awaited()
    finally:
        _restore_app(app, original)


def test_avatar_fetch_passes_the_shared_validation_result_into_native_transport(monkeypatch):
    from kestrel_sdk.security import outbound

    image = b"avatar image bytes"
    url = "https://8.8.8.8/avatar.png"
    store_avatar = AsyncMock(return_value="avatar-hash")
    agent = SimpleNamespace(agent_id="did:test:avatar", storage=SimpleNamespace(files=SimpleNamespace(store_avatar=store_avatar)))
    approved = []
    dial = AsyncMock(side_effect=AssertionError("avatar bypassed its configured transport"))
    monkeypatch.setattr("httpcore._backends.anyio.AnyIOBackend.connect_tcp", dial)

    def transport_factory(validated):
        approved.append(validated)

        def respond(request):
            assert str(request.url) == validated.url
            return httpx.Response(200, content=image)

        return httpx.MockTransport(respond)

    monkeypatch.setattr(outbound, "pinned_httpx_async_transport", transport_factory)
    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.post("/api/identity/avatar", json={"url": url}, headers={"X-API-Key": "test-key"})
        assert response.status_code == 200
        assert response.json()["avatar_hash"] == "avatar-hash"
        assert len(approved) == 1
        assert approved[0].url == url
        assert [str(ip) for ip in approved[0].ip_addresses] == ["8.8.8.8"]
        dial.assert_not_awaited()
        store_avatar.assert_awaited_once_with(
            image_data=image, agent_id=agent.agent_id, avatar_type="primary", source_url=url,
        )
    finally:
        _restore_app(app, original)


def test_avatar_transport_origin_refusal_is_a_client_error_without_storage(monkeypatch):
    from kestrel_sdk.security import outbound

    store = AsyncMock()
    agent = SimpleNamespace(agent_id="did:test:avatar", storage=SimpleNamespace(files=SimpleNamespace(store_avatar=store)))

    def refuse(request):
        raise outbound.SSRFError("pinned transport refused unexpected origin")

    monkeypatch.setattr(outbound, "pinned_httpx_async_transport", lambda validated: httpx.MockTransport(refuse))
    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.post("/api/identity/avatar", json={"url": "https://8.8.8.8/avatar.png"}, headers={"X-API-Key": "test-key"})
        assert response.status_code == 400
        assert "unexpected origin" in response.json()["detail"]
        store.assert_not_awaited()
    finally:
        _restore_app(app, original)


def test_avatar_native_url_limits_are_client_errors_before_network_or_storage(monkeypatch):
    store = AsyncMock()
    agent = SimpleNamespace(agent_id="did:test:avatar", storage=SimpleNamespace(files=SimpleNamespace(store_avatar=store)))
    dial = AsyncMock(side_effect=AssertionError("unusable native URL reached the network"))
    monkeypatch.setattr("httpcore._backends.anyio.AnyIOBackend.connect_tcp", dial)
    app, original = _prepare_app(agent)
    try:
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            with TestClient(app) as client:
                response = client.post("/api/identity/avatar", json={"url": "https://8.8.8.8/" + "x" * 100000}, headers={"X-API-Key": "test-key"})
        assert response.status_code == 400
        assert "URL too long" in response.json()["detail"]
        dial.assert_not_awaited()
        store.assert_not_awaited()
    finally:
        _restore_app(app, original)
