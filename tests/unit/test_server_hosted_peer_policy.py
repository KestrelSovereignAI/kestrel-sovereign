"""Hosted A2A policy must use active local-host peer settings."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from kestrel_sovereign import server
from kestrel_sovereign.features.peers.directory import (
    LocalHostPeerDirectory,
    PeerRequester,
)
from kestrel_sovereign.features.peers.feature import PeersFeature
from kestrel_sovereign.multi_agent.agent_manager import AgentManager
from kestrel_sovereign.multi_agent.config import MultiAgentConfig


def _host_app(*, bind="0.0.0.0", port=8888, explicit_path=None):
    app = FastAPI()
    app.state.multi_agent_config = MultiAgentConfig.model_validate(
        {"host": {"bind": bind, "port": port}}
    )
    app.state.multi_agent_config_path = explicit_path
    return app


def _local_peer_agent():
    return SimpleNamespace(
        did="did:test:hosted-peer",
        agent_id="did:test:hosted-peer",
        features={},
    )


def _local_peers_feature(agent, router, requester):
    peers = PeersFeature(agent)
    peers._peer_router = router
    peers._peer_requester = requester
    peers._host_url = getattr(router, "_host_url", None)
    agent.features = {"PeersFeature": peers}
    return peers


@pytest.mark.asyncio
async def test_hosted_policy_refreshes_local_router_with_generated_api_key(monkeypatch):
    """Registration sees a key generated after PeersFeature initialized."""

    monkeypatch.delenv("KESTREL_HOST_URL", raising=False)
    generated_key = "generated-after-feature-startup"
    monkeypatch.setattr(server, "get_api_key", lambda: generated_key)
    stale_router = LocalHostPeerDirectory("http://localhost:8888", api_key="")
    agent = _local_peer_agent()
    peers = _local_peers_feature(
        agent,
        stale_router,
        PeerRequester(agent_id := "did:test:hosted-peer", object()),
    )

    app = _host_app(port=9123)
    manager = AgentManager()
    manager._register_agent("hosted-peer", agent)
    await server._onboard_host_registered_agent(app, manager, "hosted-peer", agent)
    policy = manager.a2a_hosted_policy_for(agent)
    assert policy is not None
    router, requester = policy.router, policy.requester

    assert isinstance(router, LocalHostPeerDirectory)
    assert router is not stale_router
    assert router._host_url == "http://localhost:9123"
    assert router._headers()["X-API-Key"] == generated_key
    assert requester.identity == agent_id
    assert peers._peer_router is router
    assert peers._peer_requester is requester


def test_hosted_policy_uses_explicit_multi_agent_config_port(monkeypatch, tmp_path):
    """An explicit config outside default discovery still drives local A2A."""

    monkeypatch.delenv("KESTREL_HOST_URL", raising=False)
    monkeypatch.setattr(server, "get_api_key", lambda: "host-key")
    agent = _local_peer_agent()
    _local_peers_feature(
        agent,
        None,
        None,
    )
    app = _host_app(port=9234, explicit_path=tmp_path / "custom-host.toml")

    router, _ = server._hosted_peer_directory_context(app, agent)

    assert router._host_url == "http://localhost:9234"


def test_hosted_policy_uses_platform_port_override(monkeypatch):
    """The local policy follows PORT-adjusted config, not a stale env URL."""

    monkeypatch.setenv("KESTREL_HOST_URL", "http://stale-host:8888")
    monkeypatch.setattr(server, "get_api_key", lambda: "host-key")
    app = _host_app(port=8888)
    server._apply_platform_host_port(
        app.state.multi_agent_config,
        {"PORT": "9345"},
    )
    agent = _local_peer_agent()
    _local_peers_feature(
        agent,
        LocalHostPeerDirectory("http://localhost:8888"),
        PeerRequester("did:test:hosted-peer", object()),
    )

    router, _ = server._hosted_peer_directory_context(app, agent)

    assert router._host_url == "http://localhost:9345"


def test_hosted_policy_keeps_injected_scoped_router(monkeypatch):
    """A user-scoped router is never replaced with global local discovery."""

    monkeypatch.setattr(server, "get_api_key", lambda: "host-key")
    scoped_router = SimpleNamespace(authorize_inbound_sender=object())
    scoped_requester = PeerRequester("did:test:hosted-peer", object())
    agent = _local_peer_agent()
    peers = _local_peers_feature(agent, scoped_router, scoped_requester)

    router, requester = server._hosted_peer_directory_context(
        _host_app(port=9456), agent
    )

    assert router is scoped_router
    assert requester is scoped_requester
    assert peers._peer_router is scoped_router
