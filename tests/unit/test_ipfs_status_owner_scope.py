"""Agent-local IPFS status discloses only the routed agent's pins (#3226).

Co-hosted agents share one IPFS daemon. ``GET /api/ipfs/status`` returned
that daemon's whole recursive pin set, its peer id and version, and the
backup tier's connection details to whichever agent the request was
routed to. Now the agent-local route answers with this agent's own pinned
exports — CIDs its own receipts recorded and the daemon reports pinned —
and the node itself is a sovereign surface, ``GET /api/ipfs/node``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from kestrel_sovereign.auth import CallerContext
from kestrel_sovereign.endpoints import models as model_endpoints
from kestrel_sovereign.storage.async_storage import AsyncStorage


class _Response:
    def __init__(self, status=200, payload=None):
        self.status = status
        self._payload = payload or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return self._payload


class _Session:
    """One aiohttp.ClientSession stand-in: POSTs to the daemon, HEADs to gateways."""

    def __init__(self, posts=None, heads=None, fail=None):
        self._posts = posts or {}
        self._heads = heads or {}
        self._fail = fail

    async def __aenter__(self):
        if self._fail is not None:
            raise self._fail
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, url):
        response = self._posts.get(url)
        if response is None:
            raise AssertionError(f"Unexpected POST url: {url}")
        return response

    def head(self, url):
        response = self._heads.get(url)
        if response is None:
            raise AssertionError(f"Unexpected HEAD url: {url}")
        return response


@pytest.fixture
def identities():
    unique = uuid4().hex[:12]
    return SimpleNamespace(
        mine=f"did:pkh:eip155:1:0xMINE{unique}",
        theirs=f"did:pkh:eip155:1:0xTHEIRS{unique}",
        mine_pinned=f"bafyMINEpinned{unique}",
        mine_unpinned=f"bafyMINEunpinned{unique}",
        theirs_pinned=f"bafyTHEIRS{unique}",
        nobodys=f"bafyNOBODY{unique}",
    )


def _artifact(content_hash, cid):
    return SimpleNamespace(
        content_hash=content_hash,
        storage_tier=SimpleNamespace(value="ipfs"),
        ipfs_cid=cid,
        filecoin_deal_id=None,
        encrypted=True,
        encryption_key_hash=None,
    )


@pytest.fixture
async def agents(db_backend, identities):
    """Two real, bound storage facades over one backend, each with receipts."""
    mine = AsyncStorage(backend=db_backend, agent_id=identities.mine)
    theirs = AsyncStorage(backend=db_backend, agent_id=identities.theirs)
    await mine.initialize()
    await theirs.initialize()
    await mine.record_backup_artifact(identities.mine, _artifact(f"h1{identities.mine}", identities.mine_pinned))
    await mine.record_backup_artifact(identities.mine, _artifact(f"h2{identities.mine}", identities.mine_unpinned))
    await theirs.record_backup_artifact(identities.theirs, _artifact(f"h1{identities.theirs}", identities.theirs_pinned))
    adapter = SimpleNamespace(filecoin_adapter=SimpleNamespace(cache_dir="/host/storage_cache"))
    for facade in (mine, theirs):
        facade.sovereign_adapter = adapter
    return (
        SimpleNamespace(did=identities.mine, agent_name="display-mine", storage=mine),
        SimpleNamespace(did=identities.theirs, agent_name="display-theirs", storage=theirs),
    )


def _daemon(identities, *, fail=None):
    api = f"{model_endpoints.get_ipfs_api_url()}/api/v0"
    pins = {
        identities.mine_pinned: {"Type": "recursive"},
        identities.theirs_pinned: {"Type": "recursive"},
        identities.nobodys: {"Type": "recursive"},
    }
    return _Session(
        posts={
            f"{api}/id": _Response(payload={"ID": "peer-123", "AgentVersion": "kubo/1.0.0"}),
            f"{api}/version": _Response(payload={"Version": "1.2.3"}),
            f"{api}/pin/ls?type=recursive": _Response(payload={"Keys": pins}),
        },
        fail=fail,
    )


def _gateways():
    cid = "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG"
    return _Session(heads={
        f"https://ipfs.io/ipfs/{cid}": _Response(status=200),
        f"https://dweb.link/ipfs/{cid}": _Response(status=503),
        f"https://cloudflare-ipfs.com/ipfs/{cid}": _Response(status=200),
    })


def _client(routed_agent, other_agent, caller) -> httpx.AsyncClient:
    """Routed to one agent, app default another, caller as the auth middleware would attach."""
    app = FastAPI()

    @app.middleware("http")
    async def _route(request, call_next):
        request.state.agent = routed_agent
        request.state.caller = caller
        return await call_next(request)

    app.include_router(model_endpoints.router)
    app.state.agent = other_agent
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")


def _sessions(*sessions):
    return patch.object(model_endpoints.aiohttp, "ClientSession", side_effect=list(sessions))


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_agent_local_status_shows_only_own_pins_and_no_node_identity(agents, identities):
    mine, theirs = agents
    with _sessions(_daemon(identities), _gateways()):
        async with _client(mine, theirs, CallerContext.authenticated("user@example.com")) as client:
            mine_status = (await client.get("/api/ipfs/status")).json()
    with _sessions(_daemon(identities), _gateways()):
        async with _client(theirs, mine, CallerContext.authenticated("user@example.com")) as client:
            theirs_status = (await client.get("/api/ipfs/status")).json()

    # Mine: the pinned export, not the unpinned one, not theirs, not the
    # daemon's other content.
    assert mine_status["pinned_content"] == [{"cid": identities.mine_pinned, "type": "recursive"}], mine_status
    assert theirs_status["pinned_content"] == [{"cid": identities.theirs_pinned, "type": "recursive"}], theirs_status

    # Reachability is the agent's business; identity is the host's.
    assert mine_status["local_node"] == {"available": True, "error": None}
    assert mine_status["backup_tier"]["label"] == "sovereign-operated"
    assert "details" not in mine_status["backup_tier"]
    assert mine_status["filecoin_adapter"] == {"configured": True}
    assert mine_status["can_view_node"] is False
    assert [g["name"] for g in mine_status["gateways"]] == ["ipfs.io", "dweb.link", "cloudflare-ipfs"]


@pytest.mark.asyncio
async def test_a_cid_without_a_receipt_is_never_reported(identities):
    """The daemon holds a pin nobody recorded; no agent learns of it, and a
    pin a receipt names but the daemon lacks is not invented."""
    receipts = [SimpleNamespace(node_id="h", properties={"ipfs_cid": identities.mine_unpinned})]

    async def by_type(node_type):
        return receipts if node_type == "backup_artifact" else []

    storage = SimpleNamespace(privacy_config=None, get_nodes_by_type=by_type)
    agent = SimpleNamespace(did=identities.mine, agent_name="x", storage=storage)
    with _sessions(_daemon(identities), _gateways()):
        async with _client(agent, agent, CallerContext.authenticated("u")) as client:
            status = (await client.get("/api/ipfs/status")).json()
    assert status["pinned_content"] == []
    assert status["local_node"]["available"] is True


@pytest.mark.asyncio
async def test_daemon_failure_is_a_connection_failure_not_an_ownership_answer(identities):
    receipts = [SimpleNamespace(node_id="h", properties={"ipfs_cid": identities.mine_pinned})]

    async def by_type(node_type):
        return receipts if node_type == "backup_artifact" else []

    storage = SimpleNamespace(privacy_config=None, get_nodes_by_type=by_type)
    agent = SimpleNamespace(did=identities.mine, agent_name="x", storage=storage)
    with _sessions(_daemon(identities, fail=ConnectionError("refused")), _gateways()):
        async with _client(agent, agent, CallerContext.sovereign()) as client:
            status = (await client.get("/api/ipfs/status")).json()
    assert status["local_node"] == {"available": False, "error": "Connection failed"}
    assert status["pinned_content"] == []
    with _sessions(_daemon(identities, fail=ConnectionError("refused"))):
        async with _client(agent, agent, CallerContext.sovereign()) as client:
            node = (await client.get("/api/ipfs/node")).json()
    assert node["local_node"]["available"] is False
    assert node["local_node"]["error"] == "Connection failed"
    assert node["pinned_content"] == [] and node["pinned_total"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "caller",
    [
        pytest.param(CallerContext.authenticated("user@example.com"), id="oauth-user"),
        pytest.param(CallerContext.a2a_transport(), id="a2a-transport"),
        pytest.param(CallerContext.anonymous(), id="anonymous"),
        pytest.param(None, id="no-middleware"),
    ],
)
async def test_node_view_refuses_every_non_sovereign_caller(identities, caller):
    storage = SimpleNamespace(privacy_config=None, get_nodes_by_type=lambda t: [])
    agent = SimpleNamespace(did=identities.mine, agent_name="x", storage=storage)
    with _sessions(_daemon(identities)) as sessions:
        async with _client(agent, agent, caller) as client:
            response = await client.get("/api/ipfs/node")
    assert response.status_code == 403, response.text
    assert response.json()["detail"] == "Sovereign authority is required."
    # Refused before the handler ran: the daemon was never asked.
    sessions.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_node_view_gives_the_sovereign_the_whole_daemon(agents, identities):
    mine, theirs = agents
    with _sessions(_daemon(identities), _gateways()):
        async with _client(mine, theirs, CallerContext.sovereign()) as client:
            status = (await client.get("/api/ipfs/status")).json()
    with _sessions(_daemon(identities)):
        async with _client(mine, theirs, CallerContext.sovereign()) as client:
            node = (await client.get("/api/ipfs/node")).json()

    # The hint on the agent-local route matches the gate.
    assert status["can_view_node"] is True
    # ...and the agent-local route still withholds the host's facts from
    # a sovereign caller too: the split is by route, not by caller.
    assert "peer_id" not in status["local_node"]
    assert status["pinned_content"] == [{"cid": identities.mine_pinned, "type": "recursive"}]

    assert node["local_node"] == {
        "available": True,
        "error": None,
        "peer_id": "peer-123",
        "agent_version": "kubo/1.0.0",
        "version": "1.2.3",
    }
    assert {p["cid"] for p in node["pinned_content"]} == {
        identities.mine_pinned, identities.theirs_pinned, identities.nobodys
    }
    assert node["pinned_total"] == 3
    assert "details" in node["backup_tier"]
    assert node["filecoin_adapter"] == {"configured": True, "cache_dir": "/host/storage_cache"}
