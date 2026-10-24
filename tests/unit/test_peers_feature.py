"""Peer feature configuration and managed-child identity bootstrap.

Native public-tool behavior is exercised through actual SDK bindings in
``test_a2a_native_host`` and ``test_a2a_native_peer_stop``.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kestrel_sovereign.a2a.did_registry import (
    A2A_PEER_IDENTITY_DOCUMENTS_ENV,
    A2A_PEER_STABLE_AGENT_ID_FIELD,
)
from kestrel_sovereign.a2a.inbound_authorization import has_a2a_inbound_scoped_policy
from kestrel_sovereign.features.peers.feature import PeersFeature, _discover_host_url


def test_discover_host_url_from_env(monkeypatch):
    monkeypatch.setenv("KESTREL_HOST_URL", "http://localhost:9999/")

    assert _discover_host_url() == "http://localhost:9999"


@pytest.mark.asyncio
async def test_managed_subprocess_installs_scoped_inbound_policy(monkeypatch):
    """A fleet child transport key is admission, never sender authority."""

    monkeypatch.setenv("KESTREL_HOST_URL", "http://localhost:9999")
    monkeypatch.setenv("KESTREL_A2A_TRANSPORT_ONLY", "true")
    signing_did = "did:web:example.test:claw"
    monkeypatch.setenv(
        A2A_PEER_IDENTITY_DOCUMENTS_ENV,
        json.dumps(
            [
                {
                    "id": signing_did,
                    "verificationMethod": [
                        {
                            "id": f"{signing_did}#key-1",
                            "controller": signing_did,
                            "publicKeyMultibase": "zTest",
                        }
                    ],
                    A2A_PEER_STABLE_AGENT_ID_FIELD: "did:test:claw",
                }
            ]
        ),
    )
    agent = SimpleNamespace(
        _agent_name="ivy",
        did="did:test:ivy",
        peer_directory_router=None,
        peer_requester=None,
    )
    feature = PeersFeature(agent)

    with patch(
        "kestrel_sovereign.features.peers.feature.ensure_a2a_transport_key",
        return_value="peer-transport-key",
    ), patch(
        "kestrel_sovereign.features.storage_access.resolve_feature_database",
        return_value=None,
    ):
        await feature.initialize()

    authorizer = getattr(agent, "a2a_inbound_sender_authorizer", None)
    assert has_a2a_inbound_scoped_policy(agent) is True
    assert authorizer is not None
    assert authorizer.requires_verified_sender is True

    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False
    client.get.return_value = _mock_directory_response()
    with patch(
        "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
        return_value=client,
    ):
        assert await authorizer.authorize(signing_did) is True
        assert await authorizer.authorize("did:web:example.test:outsider") is False


def _mock_directory_response():
    """Automatic directory used before each local peer route."""
    response = MagicMock(status_code=200)
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "agents": [
            {"id": "did:test:emma", "name": "emma", "routing_name": "emma"},
            {"id": "did:test:claw", "name": "claw", "routing_name": "claw"},
            {"id": "did:test:meridian", "name": "meridian", "routing_name": "meridian"},
            {"id": "did:test:nellie", "name": "nellie", "routing_name": "nellie"},
            {"id": "did:test:talon", "name": "talon", "routing_name": "talon"},
        ],
    }
    return response
