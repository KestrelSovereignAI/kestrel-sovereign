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
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI

from kestrel_sovereign.endpoints import sovereignty as sovereignty_endpoints
from kestrel_sovereign.filecoin_adapter import FilecoinAdapter, StorageTier
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
        mine_bare_hash=hashlib.sha256(f"mine-bare-{unique}".encode()).hexdigest(),
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
    await mine.record_backup_artifact(identities.mine, _artifact(identities.mine_bare_hash))
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
    # Owned, but written without a sidecar (the pre-#1725 shape).
    (cache / f"{identities.mine_bare_hash}.cache").write_bytes(b"bare")
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
        f"{identities.mine_bare_hash}.cache",
    }, mine_list
    bare = next(f for f in mine_list["files"] if f["hash"] == identities.mine_bare_hash)
    assert bare["has_meta"] is False and bare["metadata"] is None
    assert {f["name"] for f in theirs_list["files"]} == {
        f"{identities.theirs_hash}.cache",
        f"{identities.theirs_hash}.meta",
        f"key_{identities.theirs_hash}.key",
    }, theirs_list
    assert mine_list["file_count"] == 3 and theirs_list["file_count"] == 3
    assert mine_list["total_size"] == (
        len(b"mine-bytes") + len(json.dumps({"agent": identities.mine})) + len(b"bare")
    )
    # Host layout is not the agent's to see.
    assert "cache_dir" not in mine_list
    cache_entry = next(f for f in mine_list["files"] if f["hash"] == identities.mine_hash and f["type"] == "cache")
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


@pytest.mark.asyncio
async def test_a_recorded_cache_name_of_another_export_confers_nothing(cache_dir, identities):
    """A receipt's cache name is its own export's (#2873): one naming another
    content hash's entry neither lists nor serves it."""
    receipt = SimpleNamespace(
        node_id=identities.mine_hash, properties={"cache_name": identities.theirs_hash}
    )

    async def by_type(node_type):
        return [receipt] if node_type == "backup_artifact" else []

    storage = SimpleNamespace(privacy_config=None, get_nodes_by_type=by_type)
    agent = SimpleNamespace(did=identities.mine, agent_name="x", storage=storage)
    async with _client(agent, agent) as client:
        listing = (await client.get("/api/sovereignty/files")).json()
        download = await client.get(f"/api/sovereignty/files/{identities.theirs_hash}.cache")
    assert {f["name"] for f in listing["files"]} == {
        f"{identities.mine_hash}.cache",
        f"{identities.mine_hash}.meta",
    }
    assert download.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "node_id",
    ["envelope-name", "parent-dir", "uppercase"],
)
async def test_a_receipt_whose_node_id_is_not_a_content_hash_confers_nothing(
    cache_dir, identities, node_id
):
    """A content hash is what ``store_content`` names content by, a
    lowercase SHA-256. A receipt naming anything else names no cache entry:
    in particular not another agent's envelope, ``{hash}.{owner_id}``, whose
    name a receipt could otherwise hold as its content hash (#2873)."""
    theirs_envelope = f"{identities.theirs_hash}.{'b' * 32}"
    (cache_dir / f"{theirs_envelope}.cache").write_bytes(b"their envelope")
    (cache_dir / f"{theirs_envelope}.meta").write_text(json.dumps({"agent": identities.theirs}))
    (cache_dir.parent / "outside.cache").write_bytes(b"outside the cache")
    claimed = {
        "envelope-name": theirs_envelope,
        "parent-dir": "../outside",
        "uppercase": identities.theirs_hash.upper(),
    }[node_id]
    receipt = SimpleNamespace(node_id=claimed, properties={})

    async def by_type(node_type):
        return [receipt] if node_type == "backup_artifact" else []

    storage = SimpleNamespace(privacy_config=None, get_nodes_by_type=by_type)
    agent = SimpleNamespace(did=identities.mine, agent_name="x", storage=storage)
    async with _client(agent, agent) as client:
        listing = (await client.get("/api/sovereignty/files")).json()
        download = await client.get(f"/api/sovereignty/files/{theirs_envelope}.cache")
    assert listing["files"] == []
    assert download.status_code == 404


@pytest.mark.asyncio
async def test_each_agent_sees_only_its_own_envelope_of_a_shared_plaintext(
    tmp_path, identities, monkeypatch
):
    """Two agents' encrypted exports of one plaintext share its content
    hash, and each agent's receipt names it. The host cache keys each
    envelope by the agent it was sealed for (#2873), and each receipt
    records the name; the browser lists and serves each agent its own, and
    the other's is the same 404 as a name that does not exist. The name
    derives from the key the envelope was sealed under, and the browser
    reads the receipt's record of it, so none of this depends on the key
    configured now."""
    monkeypatch.setenv("KESTREL_DATA_KEY", Fernet.generate_key().decode())
    cache = tmp_path / "storage_cache"
    monkeypatch.setattr(sovereignty_endpoints, "STORAGE_CACHE_DIR", cache)
    plaintext = f"one plaintext, two owners {identities.mine_hash}".encode()
    with patch(
        "kestrel_sovereign.filecoin_adapter.requests.post",
        side_effect=ConnectionError("no IPFS daemon"),
    ):
        adapter = FilecoinAdapter(cache_dir=str(cache))
        stored = {
            did: adapter.store_content(
                plaintext,
                storage_tier=StorageTier.LOCAL_ONLY,
                encrypt=True,
                metadata={"agent": did},
                envelope_owner=did,
            )
            for did in (identities.mine, identities.theirs)
        }
    shared = stored[identities.mine].content_hash
    assert stored[identities.theirs].content_hash == shared
    names = {path.name for path in cache.glob(f"{shared}.*.cache")}
    assert len(names) == 2

    # The browser asks the receipts, not the key.
    monkeypatch.setenv("KESTREL_DATA_KEY", Fernet.generate_key().decode())

    def agent_for(did):
        receipt = SimpleNamespace(
            node_id=shared, properties={"cache_name": stored[did].cache_name}
        )

        async def by_type(node_type):
            return [receipt] if node_type == "backup_artifact" else []

        storage = SimpleNamespace(privacy_config=None, get_nodes_by_type=by_type)
        return SimpleNamespace(did=did, agent_name="x", storage=storage)

    mine, theirs = agent_for(identities.mine), agent_for(identities.theirs)
    async with _client(mine, theirs) as client:
        listing = (await client.get("/api/sovereignty/files")).json()
    listed = {f["name"] for f in listing["files"]}
    [own_cache] = [name for name in listed if name.endswith(".cache")]
    assert own_cache == f"{stored[identities.mine].cache_name}.cache"
    [other_cache] = names - {own_cache}
    own_stem = own_cache[: -len(".cache")]
    assert listed == {own_cache, f"{own_stem}.meta"}, listing
    entry = next(f for f in listing["files"] if f["name"] == own_cache)
    assert entry["hash"] == shared
    assert entry["has_meta"] is True and entry["metadata"]["agent"] == identities.mine

    unknown = f"{shared}.{'0' * 32}.cache"
    async with _client(mine, theirs) as client:
        own = await client.get(f"/api/sovereignty/files/{own_cache}")
        own_preview = await client.get(f"/api/sovereignty/files/{own_cache}/preview")
        responses = {
            name: (
                await client.get(f"/api/sovereignty/files/{name}"),
                await client.get(f"/api/sovereignty/files/{name}/preview"),
            )
            for name in (other_cache, f"{other_cache[: -len('.cache')]}.meta", unknown)
        }
    assert own.status_code == 200 and own.content == (cache / own_cache).read_bytes()
    assert own_preview.status_code == 200
    for name, (download, preview) in responses.items():
        assert download.status_code == 404, (name, download.text)
        assert preview.status_code == 404, (name, preview.text)
        assert download.json() == responses[unknown][0].json(), name

    # The other agent sees the other envelope, and only it.
    async with _client(theirs, mine) as client:
        their_listing = (await client.get("/api/sovereignty/files")).json()
    assert {f["name"] for f in their_listing["files"]} == {
        other_cache,
        f"{other_cache[: -len('.cache')]}.meta",
    }
