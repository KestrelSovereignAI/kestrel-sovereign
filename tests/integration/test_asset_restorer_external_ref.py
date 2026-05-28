"""
Integration tests for #1438 — external-ref asset restoration via the
filecoin adapter (Phase-2 follow-up to #1391).

Phase-1 (#1391) shipped inline asset restoration and explicitly
skipped ``external_ref`` blocks (CAR contains a CBOR-tag-42 link
node rather than the asset bytes). Phase-2 implements the missing
half: extract the linked CID, fetch via ``FilecoinAdapter.retrieve_content``,
hand the bytes to the registered restorer.

The tests pin:

  * External-ref asset → ``retrieve_content`` is called with the
    embedded CID → fetched bytes reach the restorer.
  * Malformed link (wrong CBOR tag, missing bytes, etc.) lands as
    ``external_ref_malformed_link`` on ``asset_payloads_skipped``,
    import succeeds.
  * ``retrieve_content`` raises → lands as
    ``external_ref_fetch_failed: <err>``, import succeeds.
  * Inline + external-ref assets in one CAR — both routed correctly,
    counts accumulate per type.
  * Restorers see the raw bytes (no convergent-keyring decrypt for
    external-ref — caller's encryption layer if any).
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import pytest

from kestrel_sovereign.filecoin_adapter import (
    FilecoinAdapter, StorageResult, StorageTier,
)
from kestrel_sovereign.storage import Storage
from kestrel_sovereign.storage.car_builder import compute_raw_cid
from kestrel_sovereign.storage.sovereign_adapter import (
    AssetCollector, AssetDescriptor, AssetMetadata, AssetRestorer,
    SovereignStorageAdapter,
)

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class _StubFilecoinAdapter:
    """A controllable stand-in for :class:`FilecoinAdapter`.

    Holds an in-memory ``cid → bytes`` map for ``retrieve_content``.
    ``store_content`` delegates to a real ``FilecoinAdapter`` (so the
    CAR export still works through local cache). When
    ``raise_on_retrieve`` is set, ``retrieve_content`` raises with
    that message — exercises the fetch-failure path.
    """

    def __init__(
        self,
        *,
        real: FilecoinAdapter,
        external_map: Optional[Dict[str, bytes]] = None,
        fail_cids: Optional[Dict[str, str]] = None,
    ) -> None:
        self._real = real
        self._external_map = external_map or {}
        # cid → error message. Retrieves whose content_hash OR
        # ipfs_cid is a key here raise with the mapped message;
        # everything else falls through to ``external_map`` or the
        # real adapter.
        self._fail_cids = fail_cids or {}

    @property
    def cache_dir(self):
        return self._real.cache_dir

    def store_content(self, *args, **kwargs) -> StorageResult:
        return self._real.store_content(*args, **kwargs)

    def retrieve_content(
        self,
        content_hash: str,
        ipfs_cid: Optional[str] = None,
        key_hash: Optional[str] = None,
    ) -> bytes:
        for cid in (ipfs_cid, content_hash):
            if cid and cid in self._fail_cids:
                raise RuntimeError(self._fail_cids[cid])
        if ipfs_cid and ipfs_cid in self._external_map:
            return self._external_map[ipfs_cid]
        if content_hash in self._external_map:
            return self._external_map[content_hash]
        return self._real.retrieve_content(
            content_hash=content_hash, ipfs_cid=ipfs_cid, key_hash=key_hash,
        )


class _CapturingRestorer(AssetRestorer):
    def __init__(self, types: List[str]) -> None:
        self._types = types
        self.calls: List[Tuple[str, List[Tuple[AssetMetadata, bytes]]]] = []

    @property
    def asset_types(self) -> List[str]:
        return list(self._types)

    async def restore_assets(
        self, agent_did: str, assets: List[Tuple[AssetMetadata, bytes]],
    ) -> int:
        self.calls.append((agent_did, list(assets)))
        return len(assets)


class _ExternalAssetCollector(AssetCollector):
    """Returns a fixed list of external-ref assets (each has
    ``ipfs_cid`` set, no inline data)."""

    def __init__(self, descriptors: List[AssetDescriptor]) -> None:
        self._descriptors = descriptors

    async def collect_assets(self, agent_did: str) -> List[AssetDescriptor]:
        return list(self._descriptors)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


def _content_cid(blob: bytes) -> str:
    """Return the IPFS CID that ``add_external_ref`` would produce
    for *blob* when stored via the convergent encryptor path. For the
    tests we just need a deterministic CID matching the blob."""
    _, cid_str = compute_raw_cid(blob)
    return cid_str


# ---------------------------------------------------------------------------
# 1. Happy path: external-ref asset fetched + handed to restorer
# ---------------------------------------------------------------------------

async def test_external_ref_asset_fetched_and_routed(temp_db):
    agent_did = "did:pkh:eip155:1:0xexternal-1" + "0" * 8
    avatar_bytes = b"avatar-from-ipfs"
    avatar_cid = _content_cid(avatar_bytes)

    real_fc = FilecoinAdapter()
    stub_fc = _StubFilecoinAdapter(
        real=real_fc, external_map={avatar_cid: avatar_bytes},
    )

    async with Storage(db_path=temp_db) as storage:
        await storage.add_conversation(
            "user", "msg", metadata={"timestamp": "2026-05-26T10:00:00Z"},
        )
        adapter = SovereignStorageAdapter(
            storage.db, user_secret="ext-ref-1", filecoin_adapter=stub_fc,
        )
        collector = _ExternalAssetCollector([
            AssetDescriptor(
                asset_type="avatar", asset_key="main",
                content_hash="ignored", size_bytes=len(avatar_bytes),
                ipfs_cid=avatar_cid, data=None,
            ),
        ])
        cid = await adapter.export_agent(
            agent_did, storage_tier=StorageTier.LOCAL_ONLY,
            asset_collector=collector,
        )
        await storage.db.execute_commit("DELETE FROM conversation_history")

        restorer = _CapturingRestorer(["avatar"])
        result = await adapter.import_agent(
            cid, asset_restorers=[restorer],
        )
        assert result.success is True
        assert result.asset_payload_counts == {"avatar": 1}
        assert result.asset_payloads_skipped == []
        assert len(restorer.calls) == 1
        _, batch = restorer.calls[0]
        meta, bytes_ = batch[0]
        assert meta.asset_key == "main"
        assert bytes_ == avatar_bytes


# ---------------------------------------------------------------------------
# 2. Mixed CAR — inline + external-ref both restored
# ---------------------------------------------------------------------------

async def test_inline_and_external_ref_in_one_car(temp_db):
    agent_did = "did:pkh:eip155:1:0xexternal-2" + "0" * 8
    inline_bytes = b"inline-payload"
    external_bytes = b"external-payload"
    external_cid = _content_cid(external_bytes)

    real_fc = FilecoinAdapter()
    stub_fc = _StubFilecoinAdapter(
        real=real_fc, external_map={external_cid: external_bytes},
    )

    async with Storage(db_path=temp_db) as storage:
        await storage.add_conversation(
            "user", "msg", metadata={"timestamp": "2026-05-26T10:00:00Z"},
        )
        adapter = SovereignStorageAdapter(
            storage.db, user_secret="ext-ref-2", filecoin_adapter=stub_fc,
        )
        collector = _ExternalAssetCollector([
            AssetDescriptor(
                asset_type="avatar", asset_key="inline-key",
                content_hash="h-inline", size_bytes=len(inline_bytes),
                data=inline_bytes,
            ),
            AssetDescriptor(
                asset_type="avatar", asset_key="external-key",
                content_hash="h-ext", size_bytes=len(external_bytes),
                ipfs_cid=external_cid,
            ),
        ])
        cid = await adapter.export_agent(
            agent_did, storage_tier=StorageTier.LOCAL_ONLY,
            asset_collector=collector,
        )
        await storage.db.execute_commit("DELETE FROM conversation_history")

        restorer = _CapturingRestorer(["avatar"])
        result = await adapter.import_agent(
            cid, asset_restorers=[restorer],
        )
        assert result.success is True
        assert result.asset_payload_counts == {"avatar": 2}
        assert result.asset_payloads_skipped == []
        # The restorer saw both assets — verify the bytes per key.
        _, batch = restorer.calls[0]
        by_key = {m.asset_key: b for m, b in batch}
        assert by_key["inline-key"] == inline_bytes
        assert by_key["external-key"] == external_bytes


# ---------------------------------------------------------------------------
# 3. Fetch failure → skip with structured reason; import still succeeds
# ---------------------------------------------------------------------------

async def test_external_ref_fetch_failure_lands_on_skipped(temp_db):
    agent_did = "did:pkh:eip155:1:0xexternal-3" + "0" * 8
    missing_cid = _content_cid(b"unreachable-asset")

    real_fc = FilecoinAdapter()
    stub_fc = _StubFilecoinAdapter(
        real=real_fc,
        fail_cids={missing_cid: "IPFS node offline"},
    )

    async with Storage(db_path=temp_db) as storage:
        await storage.add_conversation(
            "user", "msg", metadata={"timestamp": "2026-05-26T10:00:00Z"},
        )
        adapter = SovereignStorageAdapter(
            storage.db, user_secret="ext-ref-3", filecoin_adapter=stub_fc,
        )
        collector = _ExternalAssetCollector([
            AssetDescriptor(
                asset_type="avatar", asset_key="missing",
                content_hash="h", size_bytes=10,
                ipfs_cid=missing_cid,
            ),
        ])
        cid = await adapter.export_agent(
            agent_did, storage_tier=StorageTier.LOCAL_ONLY,
            asset_collector=collector,
        )
        await storage.db.execute_commit("DELETE FROM conversation_history")

        restorer = _CapturingRestorer(["avatar"])
        result = await adapter.import_agent(
            cid, asset_restorers=[restorer],
        )
        assert result.success is True
        assert result.asset_payload_counts == {}
        assert len(result.asset_payloads_skipped) == 1
        s = result.asset_payloads_skipped[0]
        assert s["asset_key"] == "missing"
        assert s["reason"].startswith("external_ref_fetch_failed")
        assert "IPFS node offline" in s["reason"]


# ---------------------------------------------------------------------------
# 4. Malformed link (handcrafted CAR with broken external-ref) → skip
# ---------------------------------------------------------------------------

async def test_malformed_external_ref_lands_on_skipped(temp_db):
    """Directly construct a CAR with a manifest pointing at a
    dag-cbor block that LOOKS like an external-ref ({"link": …,
    "type": …}) but whose ``link`` value isn't a CBOR-tag-42 — the
    helper rejects it with ``external_ref_malformed_link``."""
    from dataclasses import asdict
    from kestrel_sovereign.storage.car_builder import CARBuilder
    from kestrel_sovereign.storage.sovereign_adapter import (
        AssetMetadata, RootManifest, ShardMetadata,
    )

    agent_did = "did:pkh:eip155:1:0xexternal-4" + "0" * 8
    real_fc = FilecoinAdapter()
    stub_fc = _StubFilecoinAdapter(real=real_fc)

    async with Storage(db_path=temp_db) as storage:
        await storage.add_conversation(
            "user", "msg", metadata={"timestamp": "2026-05-26T10:00:00Z"},
        )
        adapter = SovereignStorageAdapter(
            storage.db, user_secret="ext-ref-4", filecoin_adapter=stub_fc,
        )

        # Hand-build a CAR: minimal shard + a "broken" link node
        # whose ``link`` field is a plain string instead of a
        # CBORTag.
        builder = CARBuilder()

        # Conversation shard.
        shard_content = b"[]"  # empty shard JSON is valid
        ciphertext, key = adapter.encryptor.encrypt_with_nonce_prefix(shard_content)
        shard_block_cid = builder.add_raw_block(ciphertext)
        shard_meta = ShardMetadata(
            shard_id="conv_2026-05", type="conversation",
            time_range="2026-05", cid=shard_block_cid,
            content_hash="h", size_bytes=len(ciphertext),
        )

        # Broken external-ref-shaped block.
        bad_block_cid = builder.add_dag_cbor_block({
            "link": "not-a-cbor-tag-just-a-string",
            "type": "external_ref",
        })
        asset_meta = AssetMetadata(
            asset_type="avatar", asset_key="broken",
            cid=bad_block_cid, content_hash="h", size_bytes=4,
        )

        # Keyring.
        keyring = {"conv_2026-05": key.hex()}
        keyring_cipher = adapter._encrypt_keyring(keyring)
        keyring_cid = builder.add_raw_block(keyring_cipher)

        # Manifest.
        manifest = RootManifest(
            version="3.0", timestamp="2026-05-26T10:00:00Z",
            agent_did=agent_did, shards=[shard_meta], assets=[asset_meta],
            keyring_cid=keyring_cid,
        )
        manifest_cid = builder.add_dag_cbor_block(asdict(manifest))
        builder.set_root(manifest_cid)
        car_bytes = builder.build()

        await storage.db.execute_commit("DELETE FROM conversation_history")

        restorer = _CapturingRestorer(["avatar"])
        result = await adapter.import_agent(
            car_bytes, asset_restorers=[restorer],
        )
        assert result.success is True
        assert result.asset_payload_counts == {}
        assert len(result.asset_payloads_skipped) == 1
        assert result.asset_payloads_skipped[0]["reason"] == (
            "external_ref_malformed_link"
        )


# ---------------------------------------------------------------------------
# 5. Pre-#1438 behavior unchanged when no external-ref assets present
# ---------------------------------------------------------------------------

async def test_pre_1438_inline_only_path_unchanged(temp_db):
    agent_did = "did:pkh:eip155:1:0xexternal-5" + "0" * 8
    payload = b"inline-only"

    async with Storage(db_path=temp_db) as storage:
        await storage.add_conversation(
            "user", "msg", metadata={"timestamp": "2026-05-26T10:00:00Z"},
        )
        adapter = SovereignStorageAdapter(
            storage.db, user_secret="ext-ref-5",
        )
        collector = _ExternalAssetCollector([
            AssetDescriptor(
                asset_type="avatar", asset_key="main",
                content_hash="h", size_bytes=len(payload), data=payload,
            ),
        ])
        cid = await adapter.export_agent(
            agent_did, storage_tier=StorageTier.LOCAL_ONLY,
            asset_collector=collector,
        )
        await storage.db.execute_commit("DELETE FROM conversation_history")

        restorer = _CapturingRestorer(["avatar"])
        result = await adapter.import_agent(
            cid, asset_restorers=[restorer],
        )
        assert result.success is True
        assert result.asset_payload_counts == {"avatar": 1}
        assert result.asset_payloads_skipped == []
