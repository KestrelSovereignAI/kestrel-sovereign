"""Direct contracts for the Peers feature."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from kestrel_sovereign.features.peers.feature import PeersFeature, _discover_host_url


def test_discover_host_url_from_env(monkeypatch):
    monkeypatch.setenv("KESTREL_HOST_URL", "http://localhost:9999/")

    assert _discover_host_url() == "http://localhost:9999"


@pytest.mark.asyncio
async def test_list_peers_filters_out_self():
    agent = SimpleNamespace(_agent_name="emma")
    feature = PeersFeature(agent)
    feature._host_url = "http://multi_agent"
    feature._api_key = ""
    feature._own_name = "emma"

    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = [
        {"name": "emma", "status": "online"},
        {"name": "claw", "status": "online", "description": "peer"},
    ]

    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False
    client.get.return_value = response

    with patch("kestrel_sovereign.features.peers.feature.httpx.AsyncClient", return_value=client):
        result = await feature.list_peers()

    assert result["self"] == "emma"
    assert result["peers"] == [{"name": "claw", "status": "online", "description": "peer"}]


@pytest.mark.asyncio
async def test_ask_agent_rejects_self_target():
    agent = SimpleNamespace(_agent_name="emma")
    feature = PeersFeature(agent)
    feature._host_url = "http://multi_agent"
    feature._api_key = ""
    feature._own_name = "emma"

    result = await feature.ask_agent("emma", "hello")

    assert result["response"] is None
    assert "yourself" in result["error"]


@pytest.mark.asyncio
async def test_ask_agent_reports_offline_peer():
    agent = SimpleNamespace(_agent_name="emma")
    feature = PeersFeature(agent)
    feature._host_url = "http://multi_agent"
    feature._api_key = ""
    feature._own_name = "emma"

    response = MagicMock(status_code=503)
    response.raise_for_status.return_value = None

    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False
    client.post.return_value = response

    with patch("kestrel_sovereign.features.peers.feature.httpx.AsyncClient", return_value=client):
        result = await feature.ask_agent("claw", "hello")

    assert result["response"] is None
    assert "offline" in result["error"]


@pytest.mark.asyncio
async def test_ask_agent_returns_peer_response():
    agent = SimpleNamespace(_agent_name="emma")
    feature = PeersFeature(agent)
    feature._host_url = "http://multi_agent"
    feature._api_key = "key"
    feature._own_name = "emma"

    response = MagicMock(status_code=200)
    response.raise_for_status.return_value = None
    response.json.return_value = {"response": "hi from claw"}

    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False
    client.post.return_value = response

    with patch("kestrel_sovereign.features.peers.feature.httpx.AsyncClient", return_value=client):
        result = await feature.ask_agent("claw", "hello")

    assert result == {"agent": "claw", "response": "hi from claw"}
