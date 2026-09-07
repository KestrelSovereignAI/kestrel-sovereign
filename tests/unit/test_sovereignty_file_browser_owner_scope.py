"""The sovereignty file browser serves only the routed agent's artifacts (#3225).

``storage_cache/`` is one directory per host, written by every agent's
adapter and by every producer that ever ran here. The browser listed all
of it and served any name to any authenticated caller, on any routed
agent. The owner record is not in the directory: it is the receipt an
export writes into the agent's *own* storage, read through the
owner-scoped graph. These tests put two agents' receipts in ONE store —
the shared-PostgreSQL shape — and two agents' files in ONE cache
directory, and drive the real routes.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from kestrel_sovereign.endpoints import sovereignty as sovereignty_endpoints
from kestrel_sovereign.storage.async_storage import AsyncStorage


@pytest.fixture
def identities():
    """Two DIDs and two content hashes nobody else has used.

    A shared PostgreSQL database keeps every earlier run's receipts and
    owner rows; unique identities isolate without truncating tables this
    test does not own.
    """
    unique = uuid4().hex[:12]
    return SimpleNamespace(
        mine=f"did:pkh:eip155:1:0xMINE{unique}",
        theirs=f"did:pkh:eip155:1:0xTHEIRS{unique}",
        mine_hash=hashlib.sha256(f"mine-{unique}".encode()).hexdigest(),
        theirs_hash=hashlib.sha256(f"theirs-{unique}".encode()).hexdigest(),
        orphan_hash=hashlib.sha256(f"orphan-{unique}".encode()).hexdigest(),
    )


def _artifact(content_hash, cid=None):
    return SimpleNamespace(
        content_hash=content_hash,
        storage_tier=SimpleNamespace(value="local_only"),
        ipfs_cid=cid,
        filecoin_deal_id=None,
        encrypted=False,
        encryption_key_hash=None,
    )


@pytest.fixture
async def agents(db_backend, identities):
    """Two real, bound storage facades over one backend, each with a receipt."""
    mine = AsyncStorage(backend=db_backend, agent_id=identities.mine)
    theirs = AsyncStorage(backend=db_backend, agent_id=identities.theirs)
    await mine.initialize()
    await theirs.initialize()
    await mine.record_backup_artifact(identities.mine, _artifact(identities.mine_hash))
    await theirs.record_backup_artifact(identities.theirs, _artifact(identities.theirs_hash))
    return (
        SimpleNamespace(did=identities.mine, agent_name="display-mine", storage=mine),
        SimpleNamespace(did=identities.theirs, agent_name="display-theirs", storage=theirs),
    )


@pytest.fixture
def cache_dir(tmp_path, identities, monkeypatch):
    """One host cache holding both agents' files and an orphan.

    The orphan's sidecar *claims* to be mine — the shape a planted or
    legacy entry has — and no receipt names it, so nobody may see it.
    """
    cache = tmp_path / "storage_cache"
    cache.mkdir()
    (cache / f"{identities.mine_hash}.cache").write_bytes(b"mine-bytes")
    (cache / f"{identities.mine_hash}.meta").write_text(json.dumps({"agent": identities.mine}))
    (cache / f"{identities.theirs_hash}.cache").write_bytes(b"theirs-bytes")
    (cache / f"{identities.theirs_hash}.meta").write_text(json.dumps({"agent": identities.theirs}))
    (cache / f"key_{identities.theirs_hash}.key").write_bytes(b"wrapped-key")
    (cache / f"{identities.orphan_hash}.cache").write_bytes(b"orphan-bytes")
    (cache / f"{identities.orphan_hash}.meta").write_text(json.dumps({"agent": identities.mine}))
    monkeypatch.setattr(sovereignty_endpoints, "STORAGE_CACHE_DIR", cache)
    return cache


def _client(routed_agent, other_agent) -> httpx.AsyncClient:
    """An in-loop ASGI client whose request is ROUTED to one agent.

    The multi-agent middleware sets ``request.state.agent``; the single-agent
    fallback is ``app.state.agent``. The two are deliberately different
    here, so a handler that reads the app default instead of the routed
    agent answers for the wrong one.
    """
    app = FastAPI()

    @app.middleware("http")
    async def _route(request, call_next):
        request.state.agent = routed_agent
        return await call_next(request)

    app.include_router(sovereignty_endpoints.router)
    app.state.agent = other_agent
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_listing_shows_only_the_routed_agents_artifacts(agents, cache_dir, identities):
    mine, theirs = agents
    async with _client(mine, theirs) as client:
        mine_list = (await client.get("/api/sovereignty/files")).json()
    async with _client(theirs, mine) as client:
        theirs_list = (await client.get("/api/sovereignty/files")).json()

    assert {f["name"] for f in mine_list["files"]} == {
        f"{identities.mine_hash}.cache",
        f"{identities.mine_hash}.meta",
    }, mine_list
    assert {f["name"] for f in theirs_list["files"]} == {
        f"{identities.theirs_hash}.cache",
        f"{identities.theirs_hash}.meta",
        f"key_{identities.theirs_hash}.key",
    }, theirs_list
    assert mine_list["file_count"] == 2 and theirs_list["file_count"] == 3
    assert mine_list["total_size"] == len(b"mine-bytes") + len(json.dumps({"agent": identities.mine}))
    # Host layout is not the agent's to see.
    assert "cache_dir" not in mine_list
    cache_entry = next(f for f in mine_list["files"] if f["type"] == "cache")
    assert cache_entry["has_meta"] is True
    assert cache_entry["metadata"] == {"agent": identities.mine}
    assert cache_entry["hash"] == identities.mine_hash


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_foreign_orphan_and_unknown_names_are_the_same_404(agents, cache_dir, identities):
    """A name the caller does not own must not be a probe for existence."""
    mine, theirs = agents
    unknown = hashlib.sha256(b"never-written").hexdigest()
    async with _client(mine, theirs) as client:
        responses = {}
        for label, name in (
            ("theirs", f"{identities.theirs_hash}.cache"),
            ("theirs-key", f"key_{identities.theirs_hash}.key"),
            ("theirs-meta", f"{identities.theirs_hash}.meta"),
            ("orphan", f"{identities.orphan_hash}.cache"),
            ("unknown", f"{unknown}.cache"),
            ("not-an-artifact", "README.txt"),
        ):
            responses[label] = (
                await client.get(f"/api/sovereignty/files/{name}"),
                await client.get(f"/api/sovereignty/files/{name}/preview"),
            )

    for label, (download, preview) in responses.items():
        assert download.status_code == 404, (label, download.text)
        assert preview.status_code == 404, (label, preview.text)
        assert download.json() == responses["unknown"][0].json(), label
        assert preview.json() == responses["unknown"][1].json(), label


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_owner_can_preview_and_download_its_own_artifact(agents, cache_dir, identities):
    mine, theirs = agents
    async with _client(theirs, mine) as client:
        preview = await client.get(f"/api/sovereignty/files/{identities.theirs_hash}.cache/preview")
        download = await client.get(f"/api/sovereignty/files/{identities.theirs_hash}.cache")
        key = await client.get(f"/api/sovereignty/files/key_{identities.theirs_hash}.key")

    assert preview.status_code == 200, preview.text
    assert preview.json()["content"] == "theirs-bytes"
    assert download.status_code == 200 and download.content == b"theirs-bytes"
    assert key.status_code == 200 and key.content == b"wrapped-key"


@pytest.mark.asyncio
async def test_traversal_is_refused_before_ownership_is_asked(sqlite_backend, cache_dir, identities):
    """The filename-shape check stays first and stays a 400: it is about
    the request, not about what exists or who owns it."""
    storage = SimpleNamespace(privacy_config=None, get_nodes_by_type=AsyncMock(side_effect=AssertionError("must not be reached")))
    agent = SimpleNamespace(did=identities.mine, agent_name="x", storage=storage)
    async with _client(agent, agent) as client:
        # A literal "/" never reaches the handler (the router has no such
        # route); an encoded backslash does, and must be refused there.
        for name in ("..hidden", "a%5Cb.cache"):
            assert (await client.get(f"/api/sovereignty/files/{name}/preview")).status_code == 400, name
            assert (await client.get(f"/api/sovereignty/files/{name}")).status_code == 400, name


@pytest.mark.asyncio
async def test_privacy_mode_that_hides_persisted_rows_hides_artifacts(cache_dir, identities):
    """Same rule as GET /api/storage/stats: an ephemeral or temp-storage
    session owns nothing visible, even when a receipt would be found."""
    receipt = SimpleNamespace(node_id=identities.mine_hash, properties={})
    storage = SimpleNamespace(
        privacy_config=SimpleNamespace(is_ephemeral=lambda: True, uses_temp_storage=lambda: False),
        get_nodes_by_type=AsyncMock(return_value=[receipt]),
    )
    agent = SimpleNamespace(did=identities.mine, agent_name="x", storage=storage)
    async with _client(agent, agent) as client:
        listing = (await client.get("/api/sovereignty/files")).json()
        preview = await client.get(f"/api/sovereignty/files/{identities.mine_hash}.cache/preview")
    assert listing == {"files": [], "total_size": 0, "file_count": 0}
    assert preview.status_code == 404
    storage.get_nodes_by_type.assert_not_awaited()


@pytest.mark.asyncio
async def test_receipt_content_hash_also_confers_ownership(cache_dir, identities):
    """A ``sovereignty_receipt`` naming the hash is the second receipt the
    export writes; either record is the agent's own."""
    receipt = SimpleNamespace(node_id="sovereignty_receipt_x", properties={"content_hash": identities.mine_hash})

    async def by_type(node_type):
        return [receipt] if node_type == "sovereignty_receipt" else []

    storage = SimpleNamespace(privacy_config=None, get_nodes_by_type=by_type)
    agent = SimpleNamespace(did=identities.mine, agent_name="x", storage=storage)
    async with _client(agent, agent) as client:
        listing = (await client.get("/api/sovereignty/files")).json()
    assert {f["name"] for f in listing["files"]} == {
        f"{identities.mine_hash}.cache",
        f"{identities.mine_hash}.meta",
    }
