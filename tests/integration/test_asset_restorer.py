"""
Integration tests for #1391 — :class:`AssetRestorer` protocol +
``SovereignStorageAdapter.import_agent(asset_restorers=...)``.

Pre-#1391 ``import_agent`` packed asset metadata into the manifest
but never restored asset bytes. These tests pin the new behavior:

  * pre-#1391 callers (no restorers) see identical behavior — the
    metadata-only ``assets_restored`` field still populates, and
    ``asset_payload_counts`` is empty.
  * with restorers wired, every adapter-encrypted asset (inline raw
    block + keyring entry) is decrypted and handed off in batch.
  * caller-pre-encrypted assets (no keyring entry) get raw bytes
    handed through so the restorer can decrypt in its own scheme.
  * ``asset_type`` routing: a restorer only sees its declared types.
  * external-ref assets are surfaced on ``asset_payloads_skipped``
    rather than handed off — Phase 2 follow-up.
  * a restorer failure propagates with an audit-logged error.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any, List, Tuple

import pytest

from kestrel_sovereign.filecoin_adapter import StorageTier
from kestrel_sovereign.storage import Storage
from kestrel_sovereign.storage.sovereign_adapter import (
    AssetCollector,
    AssetDescriptor,
    AssetMetadata,
    AssetRestorer,
    SovereignStorageAdapter,
)

pytestmark = pytest.mark.asyncio


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


@pytest.fixture
def temp_db_2():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


class _StaticAssetCollector(AssetCollector):
    """Hand the exporter a fixed list of assets to bundle.

    Mirrors the shape Frinz uses for avatar/lora exports.
    """

    def __init__(self, assets: List[AssetDescriptor]) -> None:
        self._assets = assets

    async def collect_assets(self, agent_did: str) -> List[AssetDescriptor]:
        return list(self._assets)


class _CapturingRestorer(AssetRestorer):
    """Records every (metadata, bytes) it sees so tests can assert."""

    def __init__(self, types: List[str], *, return_n: int | None = None) -> None:
        self._types = types
        self._return_n = return_n
        self.calls: List[Tuple[str, List[Tuple[AssetMetadata, bytes]]]] = []

    @property
    def asset_types(self) -> List[str]:
        return list(self._types)

    async def restore_assets(
        self, agent_did: str, assets: List[Tuple[AssetMetadata, bytes]],
    ) -> int:
        self.calls.append((agent_did, list(assets)))
        if self._return_n is not None:
            return self._return_n
        return len(assets)


class _FailingRestorer(AssetRestorer):
    """Always raises — used to exercise the error path."""

    def __init__(self, types: List[str], message: str = "kaboom") -> None:
        self._types = types
        self._message = message

    @property
    def asset_types(self) -> List[str]:
        return list(self._types)

    async def restore_assets(
        self, agent_did: str, assets: List[Tuple[AssetMetadata, bytes]],
    ) -> int:
        raise RuntimeError(self._message)


# ---------------------------------------------------------------------------
# 1. Pre-#1391 back-compat — no restorers means no payload handoff
# ---------------------------------------------------------------------------

async def test_no_restorers_preserves_pre_1391_behavior(temp_db):
    """Without ``asset_restorers``, ``assets_restored`` still
    populates from the manifest but ``asset_payload_counts`` is
    empty — the old API surface is byte-identical."""
    agent_did = "did:pkh:eip155:1:0xaa" + "11" * 19
    async with Storage(db_path=temp_db) as storage:
        await storage.add_conversation(
            "user", "hi", metadata={"timestamp": "2026-05-25T10:00:00Z"},
        )
        adapter = SovereignStorageAdapter(storage.db, user_secret="r-test-1")
        collector = _StaticAssetCollector([
            AssetDescriptor(
                asset_type="avatar", asset_key="main",
                content_hash="deadbeef", size_bytes=4,
                data=b"PNG!",
            ),
        ])
        cid = await adapter.export_agent(
            agent_did, storage_tier=StorageTier.LOCAL_ONLY,
            asset_collector=collector,
        )
        await storage.db.execute_commit("DELETE FROM conversation_history")

        result = await adapter.import_agent(cid)
        assert result.success is True
        assert result.assets_restored, "manifest metadata still populates"
        assert result.assets_restored[0]["asset_key"] == "main"
        assert result.asset_payload_counts == {}
        assert result.asset_payloads_skipped == []


# ---------------------------------------------------------------------------
# 2. Happy path — adapter-encrypted asset round-trips to the restorer
# ---------------------------------------------------------------------------

async def test_inline_asset_decrypted_and_handed_to_restorer(temp_db):
    """The exporter passes plaintext bytes; the adapter encrypts them
    via the convergent keyring; the importer decrypts them and hands
    plaintext to the restorer."""
    agent_did = "did:pkh:eip155:1:0xbb" + "11" * 19
    plaintext = b"avatar binary payload \xff\x00\x01"
    async with Storage(db_path=temp_db) as storage:
        await storage.add_conversation(
            "user", "msg", metadata={"timestamp": "2026-05-25T10:00:00Z"},
        )
        adapter = SovereignStorageAdapter(storage.db, user_secret="r-test-2")
        collector = _StaticAssetCollector([
            AssetDescriptor(
                asset_type="avatar", asset_key="main",
                content_hash="hash-1", size_bytes=len(plaintext),
                data=plaintext, encrypted=False,
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
        # Restorer saw plaintext.
        assert len(restorer.calls) == 1
        seen_agent_did, batch = restorer.calls[0]
        assert seen_agent_did == agent_did
        assert len(batch) == 1
        meta, bytes_ = batch[0]
        assert meta.asset_key == "main"
        assert meta.asset_type == "avatar"
        assert bytes_ == plaintext


# ---------------------------------------------------------------------------
# 3. Pre-encrypted asset (no keyring entry) — raw bytes pass through
# ---------------------------------------------------------------------------

async def test_pre_encrypted_asset_passes_raw_bytes_through(temp_db):
    """When the descriptor's ``encrypted=True`` the adapter doesn't
    store a per-asset keyring entry; the importer hands raw block
    bytes through so the restorer can decrypt under its own scheme."""
    agent_did = "did:pkh:eip155:1:0xcc" + "11" * 19
    caller_ciphertext = b"caller-pre-encrypted-blob"
    async with Storage(db_path=temp_db) as storage:
        await storage.add_conversation(
            "user", "msg", metadata={"timestamp": "2026-05-25T10:00:00Z"},
        )
        adapter = SovereignStorageAdapter(storage.db, user_secret="r-test-3")
        collector = _StaticAssetCollector([
            AssetDescriptor(
                asset_type="opaque_blob", asset_key="key-1",
                content_hash="hash-2", size_bytes=len(caller_ciphertext),
                data=caller_ciphertext, encrypted=True,
            ),
        ])
        cid = await adapter.export_agent(
            agent_did, storage_tier=StorageTier.LOCAL_ONLY,
            asset_collector=collector,
        )
        await storage.db.execute_commit("DELETE FROM conversation_history")

        restorer = _CapturingRestorer(["opaque_blob"])
        result = await adapter.import_agent(
            cid, asset_restorers=[restorer],
        )
        assert result.success is True
        assert result.asset_payload_counts == {"opaque_blob": 1}
        _, batch = restorer.calls[0]
        meta, bytes_ = batch[0]
        # Bytes were NOT decrypted — they're the caller's ciphertext.
        assert bytes_ == caller_ciphertext


# ---------------------------------------------------------------------------
# 4. Type routing — restorer only sees its declared types
# ---------------------------------------------------------------------------

async def test_restorer_only_sees_its_declared_types(temp_db):
    agent_did = "did:pkh:eip155:1:0xdd" + "11" * 19
    fhir_bytes = b'{"resourceType":"Patient","id":"p1"}'
    avatar_bytes = b"PNG-avatar"
    async with Storage(db_path=temp_db) as storage:
        await storage.add_conversation(
            "user", "msg", metadata={"timestamp": "2026-05-25T10:00:00Z"},
        )
        adapter = SovereignStorageAdapter(storage.db, user_secret="r-test-4")
        collector = _StaticAssetCollector([
            AssetDescriptor(
                asset_type="fhir_resource", asset_key="Patient/p1",
                content_hash="h-fhir", size_bytes=len(fhir_bytes),
                data=fhir_bytes,
            ),
            AssetDescriptor(
                asset_type="avatar", asset_key="main",
                content_hash="h-av", size_bytes=len(avatar_bytes),
                data=avatar_bytes,
            ),
        ])
        cid = await adapter.export_agent(
            agent_did, storage_tier=StorageTier.LOCAL_ONLY,
            asset_collector=collector,
        )
        await storage.db.execute_commit("DELETE FROM conversation_history")

        fhir_restorer = _CapturingRestorer(["fhir_resource"])
        avatar_restorer = _CapturingRestorer(["avatar"])

        result = await adapter.import_agent(
            cid, asset_restorers=[fhir_restorer, avatar_restorer],
        )
        assert result.success is True
        assert result.asset_payload_counts == {
            "fhir_resource": 1, "avatar": 1,
        }
        # Each restorer saw only its type.
        assert len(fhir_restorer.calls) == 1
        assert fhir_restorer.calls[0][1][0][0].asset_type == "fhir_resource"
        assert len(avatar_restorer.calls) == 1
        assert avatar_restorer.calls[0][1][0][0].asset_type == "avatar"


# ---------------------------------------------------------------------------
# 5. Unhandled type → skipped, not handed off
# ---------------------------------------------------------------------------

async def test_unhandled_asset_type_recorded_as_skipped(temp_db):
    agent_did = "did:pkh:eip155:1:0xee" + "11" * 19
    async with Storage(db_path=temp_db) as storage:
        await storage.add_conversation(
            "user", "msg", metadata={"timestamp": "2026-05-25T10:00:00Z"},
        )
        adapter = SovereignStorageAdapter(storage.db, user_secret="r-test-5")
        collector = _StaticAssetCollector([
            AssetDescriptor(
                asset_type="unknown_type", asset_key="orphan",
                content_hash="h", size_bytes=4, data=b"DATA",
            ),
        ])
        cid = await adapter.export_agent(
            agent_did, storage_tier=StorageTier.LOCAL_ONLY,
            asset_collector=collector,
        )
        await storage.db.execute_commit("DELETE FROM conversation_history")

        # Restorer only handles a DIFFERENT type.
        other = _CapturingRestorer(["something_else"])
        result = await adapter.import_agent(
            cid, asset_restorers=[other],
        )
        assert result.success is True
        assert result.asset_payload_counts == {}
        assert len(other.calls) == 0
        assert len(result.asset_payloads_skipped) == 1
        s = result.asset_payloads_skipped[0]
        assert s["asset_key"] == "orphan"
        assert s["asset_type"] == "unknown_type"
        assert s["reason"] == "no_restorer_for_type"


# ---------------------------------------------------------------------------
# 6. Two restorers claim the same type — both receive the asset
# ---------------------------------------------------------------------------

async def test_multiple_restorers_for_same_type_both_called(temp_db):
    """Useful for an audit-collector that wants every asset alongside
    the primary restorer."""
    agent_did = "did:pkh:eip155:1:0xff" + "11" * 19
    payload = b"shared-asset"
    async with Storage(db_path=temp_db) as storage:
        await storage.add_conversation(
            "user", "msg", metadata={"timestamp": "2026-05-25T10:00:00Z"},
        )
        adapter = SovereignStorageAdapter(storage.db, user_secret="r-test-6")
        collector = _StaticAssetCollector([
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

        primary = _CapturingRestorer(["avatar"])
        audit = _CapturingRestorer(["avatar"])
        result = await adapter.import_agent(
            cid, asset_restorers=[primary, audit],
        )
        assert result.success is True
        # Both saw the asset; payload_counts sums them.
        assert result.asset_payload_counts == {"avatar": 2}
        assert len(primary.calls) == 1
        assert len(audit.calls) == 1


# ---------------------------------------------------------------------------
# 7. Restorer raises → import propagates the error
# ---------------------------------------------------------------------------

async def test_restorer_failure_propagates_with_audit(temp_db):
    agent_did = "did:pkh:eip155:1:0x10" + "11" * 19
    async with Storage(db_path=temp_db) as storage:
        await storage.add_conversation(
            "user", "msg", metadata={"timestamp": "2026-05-25T10:00:00Z"},
        )
        adapter = SovereignStorageAdapter(storage.db, user_secret="r-test-7")
        collector = _StaticAssetCollector([
            AssetDescriptor(
                asset_type="avatar", asset_key="main",
                content_hash="h", size_bytes=4, data=b"DATA",
            ),
        ])
        cid = await adapter.export_agent(
            agent_did, storage_tier=StorageTier.LOCAL_ONLY,
            asset_collector=collector,
        )
        await storage.db.execute_commit("DELETE FROM conversation_history")

        bad = _FailingRestorer(["avatar"], message="restorer-said-no")
        with pytest.raises(RuntimeError, match="restorer-said-no"):
            await adapter.import_agent(cid, asset_restorers=[bad])
        # Audit row marks the import as error.
        log = await adapter.get_import_log(agent_did=agent_did, limit=5)
        assert log
        assert log[0]["status"] == "error"
        assert "restore_failed" in (log[0]["reject_reason"] or "")


# ---------------------------------------------------------------------------
# 8. Restorer returns non-int → adapter falls back to len(batch)
# ---------------------------------------------------------------------------

async def test_restorer_non_int_return_falls_back_to_batch_len(temp_db):
    agent_did = "did:pkh:eip155:1:0x11" + "11" * 19
    async with Storage(db_path=temp_db) as storage:
        await storage.add_conversation(
            "user", "msg", metadata={"timestamp": "2026-05-25T10:00:00Z"},
        )
        adapter = SovereignStorageAdapter(storage.db, user_secret="r-test-8")
        collector = _StaticAssetCollector([
            AssetDescriptor(
                asset_type="avatar", asset_key=f"a-{i}",
                content_hash=f"h-{i}", size_bytes=4, data=b"DATA",
            )
            for i in range(3)
        ])
        cid = await adapter.export_agent(
            agent_did, storage_tier=StorageTier.LOCAL_ONLY,
            asset_collector=collector,
        )
        await storage.db.execute_commit("DELETE FROM conversation_history")

        class _BadReturnRestorer(AssetRestorer):
            @property
            def asset_types(self):
                return ["avatar"]

            async def restore_assets(self, agent_did, assets):
                return None  # Bug — should be int.

        result = await adapter.import_agent(
            cid, asset_restorers=[_BadReturnRestorer()],
        )
        assert result.success is True
        # Defensive: adapter falls back to len(batch).
        assert result.asset_payload_counts == {"avatar": 3}


# ---------------------------------------------------------------------------
# 9. asset_restorers=[] is identical to None
# ---------------------------------------------------------------------------

async def test_empty_restorers_list_same_as_none(temp_db):
    agent_did = "did:pkh:eip155:1:0x12" + "11" * 19
    async with Storage(db_path=temp_db) as storage:
        await storage.add_conversation(
            "user", "msg", metadata={"timestamp": "2026-05-25T10:00:00Z"},
        )
        adapter = SovereignStorageAdapter(storage.db, user_secret="r-test-9")
        collector = _StaticAssetCollector([
            AssetDescriptor(
                asset_type="avatar", asset_key="main",
                content_hash="h", size_bytes=4, data=b"DATA",
            ),
        ])
        cid = await adapter.export_agent(
            agent_did, storage_tier=StorageTier.LOCAL_ONLY,
            asset_collector=collector,
        )
        await storage.db.execute_commit("DELETE FROM conversation_history")

        result = await adapter.import_agent(cid, asset_restorers=[])
        assert result.success is True
        assert result.asset_payload_counts == {}
        assert result.asset_payloads_skipped == []


# ---------------------------------------------------------------------------
# 10. Conversations restore correctly alongside asset restoration
# ---------------------------------------------------------------------------

async def test_conversations_still_restore_alongside_assets(temp_db):
    """Asset restoration must not break the conversation-restore path."""
    agent_did = "did:pkh:eip155:1:0x13" + "11" * 19
    async with Storage(db_path=temp_db) as storage:
        seed_msgs = [
            ("user", "msg-1"),
            ("assistant", "reply-1"),
            ("user", "msg-2"),
        ]
        for role, content in seed_msgs:
            await storage.add_conversation(
                role, content,
                metadata={"timestamp": "2026-05-25T10:00:00Z"},
            )
        adapter = SovereignStorageAdapter(storage.db, user_secret="r-test-10")
        collector = _StaticAssetCollector([
            AssetDescriptor(
                asset_type="avatar", asset_key="main",
                content_hash="h", size_bytes=4, data=b"DATA",
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
        assert result.messages_restored == len(seed_msgs)
        assert result.asset_payload_counts == {"avatar": 1}
        # And the conversations actually landed.
        restored = await storage.get_conversation_history()
        assert [(m["role"], m["content"]) for m in restored] == seed_msgs
