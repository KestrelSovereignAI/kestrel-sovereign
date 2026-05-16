"""
Real export→import round-trip tests for the verification-gated,
audit-logged sovereignty *import receiver* (issue #1272).

The import path is now symmetric with export:

  * ``import_agent`` returns a structured ``SovereignImportResult``
    (not a dict), carrying an ``ImportContinuity`` score.
  * A package that fails static verification (e.g. wrong owner
    secret) is *rejected* — a structured result, never a raised
    exception — and the host DB is left UNTOUCHED.
  * Every attempt (imported | rejected | error) appends exactly one
    row to the append-only ``agent_import_log``.

NO MOCKS. NO STUBS. Real SQLite, real CARBuilder/CARReader, real
AES-256-GCM via ``ConvergentEncryptor``, ``StorageTier.LOCAL_ONLY``
so the suite stays hermetic (the encryption + sharding path is
identical regardless of tier).
"""

from __future__ import annotations

import secrets
from pathlib import Path

import pytest

from kestrel_sovereign.filecoin_adapter import StorageTier
from kestrel_sovereign.storage import Storage
from kestrel_sovereign.storage.sovereign_adapter import (
    SovereignImportResult,
    SovereignStorageAdapter,
)

pytestmark = pytest.mark.integration


AGENT_ID = "test:import-receiver"
AGENT_DID = "did:test:import-receiver"


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Per-test SQLite DB path (tmp_path auto-cleans)."""
    return tmp_path / "kestrel_receiver.db"


@pytest.fixture
def user_secret() -> str:
    """Fresh per-test 256-bit owner secret."""
    return secrets.token_hex(32)


_MESSAGES = [
    ("user", "Remember my dog's name is Biscuit", "2025-11-02T10:00:00Z"),
    ("assistant", "Got it — Biscuit it is.", "2025-11-02T10:00:01Z"),
    ("user", "RECEIVER_MARKER_9f3a", "2025-12-03T09:00:00Z"),
    ("assistant", "Marker acknowledged.", "2025-12-03T09:00:01Z"),
]


async def _seed(storage: Storage, messages=_MESSAGES) -> None:
    for role, content, ts in messages:
        await storage.add_conversation(role, content, metadata={"timestamp": ts})


# ---------------------------------------------------------------------------
# 1. Verified ingest happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_verified_ingest_happy_path(db_path: Path, user_secret: str):
    async with Storage(db_path=str(db_path), agent_id=AGENT_ID) as storage:
        await _seed(storage)
        assert len(await storage.get_conversation_history()) == len(_MESSAGES)

        adapter = SovereignStorageAdapter(
            storage.db, user_secret=user_secret, agent_id=AGENT_ID,
        )
        cid = await adapter.export_agent(
            AGENT_DID, storage_tier=StorageTier.LOCAL_ONLY,
        )
        assert cid

        # Simulate data loss.
        await storage.db.execute_commit("DELETE FROM conversation_history")
        assert await storage.get_conversation_history() == []

        result = await adapter.import_agent(cid)

        assert isinstance(result, SovereignImportResult)
        assert result.success is True
        assert result.status == "imported"
        assert result.reject_reason is None
        assert result.agent_did == AGENT_DID
        assert result.manifest_version == "3.0"
        assert result.messages_restored == len(_MESSAGES)
        assert result.shards_restored > 0
        assert result.source_cid == cid

        # A clean static archive verifies perfectly.
        assert result.continuity.overall_score == 1.0
        assert result.continuity.is_verified()
        assert result.continuity.checks_passed == result.continuity.checks_total

        # Data really came back.
        restored = await storage.get_conversation_history()
        assert [(m["role"], m["content"]) for m in restored] == [
            (r, c) for r, c, _ in _MESSAGES
        ]

        # Exactly one audit row, status imported, fields match.
        log = await adapter.get_import_log(agent_did=AGENT_DID)
        assert len(log) == 1
        row = log[0]
        assert row["status"] == "imported"
        assert row["reject_reason"] is None
        assert row["package_hash"] == result.package_hash
        assert row["continuity_score"] == result.continuity.overall_score
        assert row["messages_restored"] == len(_MESSAGES)
        assert row["shards_restored"] == result.shards_restored
        assert row["source_did"] == AGENT_DID


# ---------------------------------------------------------------------------
# 2. import_agent accepts raw CAR bytes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_import_accepts_raw_bytes(db_path: Path, user_secret: str):
    async with Storage(db_path=str(db_path), agent_id=AGENT_ID) as storage:
        await _seed(storage)

        adapter = SovereignStorageAdapter(
            storage.db, user_secret=user_secret, agent_id=AGENT_ID,
        )
        cid = await adapter.export_agent(
            AGENT_DID, storage_tier=StorageTier.LOCAL_ONLY,
        )

        # Pull the real CAR blob back out of the local filecoin cache
        # via the adapter's own download path — no mocking, this is the
        # exact bytes that were uploaded.
        car_bytes = await adapter._download_content(cid)
        assert isinstance(car_bytes, (bytes, bytearray))
        assert len(car_bytes) > 0

        await storage.db.execute_commit("DELETE FROM conversation_history")
        assert await storage.get_conversation_history() == []

        result = await adapter.import_agent(bytes(car_bytes))

        assert result.success is True
        assert result.status == "imported"
        assert result.messages_restored == len(_MESSAGES)
        # Bytes path => no CID was downloaded.
        assert result.source_cid is None

        restored = await storage.get_conversation_history()
        assert [(m["role"], m["content"]) for m in restored] == [
            (r, c) for r, c, _ in _MESSAGES
        ]

        log = await adapter.get_import_log(agent_did=AGENT_DID)
        assert len(log) == 1
        assert log[0]["status"] == "imported"
        assert log[0]["source_cid"] is None


# ---------------------------------------------------------------------------
# 3. Rejected attempt leaves DB untouched + logs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rejected_import_leaves_db_untouched_and_logs(
    db_path: Path, user_secret: str,
):
    wrong_secret = secrets.token_hex(32)
    assert wrong_secret != user_secret

    async with Storage(db_path=str(db_path), agent_id=AGENT_ID) as storage:
        await _seed(storage)
        export_adapter = SovereignStorageAdapter(
            storage.db, user_secret=user_secret, agent_id=AGENT_ID,
        )
        cid = await export_adapter.export_agent(
            AGENT_DID, storage_tier=StorageTier.LOCAL_ONLY,
        )

        # Replace host data with a distinct sentinel conversation that
        # must survive a rejected import verbatim.
        await storage.db.execute_commit("DELETE FROM conversation_history")
        await storage.add_conversation(
            "user", "SENTINEL_DO_NOT_DELETE",
            metadata={"timestamp": "2026-01-01T00:00:00Z"},
        )
        sentinel_before = await storage.get_conversation_history()
        assert len(sentinel_before) == 1

        wrong_adapter = SovereignStorageAdapter(
            storage.db, user_secret=wrong_secret, agent_id=AGENT_ID,
        )
        result = await wrong_adapter.import_agent(cid)

        assert result.success is False
        assert result.status == "rejected"
        assert result.reject_reason == "keyring_decrypt_failed"
        assert result.messages_restored == 0
        assert result.shards_restored == 0
        # car + manifest verify fine; only the keyring/shard decrypt
        # fails, so the score is partial (not zero, not verified).
        assert 0.0 < result.continuity.overall_score < 1.0
        assert not result.continuity.is_verified()

        # Host DB UNTOUCHED — sentinel still present, no INSERT/DELETE.
        sentinel_after = await storage.get_conversation_history()
        assert len(sentinel_after) == 1
        assert sentinel_after[0]["content"] == "SENTINEL_DO_NOT_DELETE"

        # Rejected attempt still audited.
        log = await wrong_adapter.get_import_log(agent_did=AGENT_DID)
        assert len(log) == 1
        row = log[0]
        assert row["status"] == "rejected"
        assert row["reject_reason"] == "keyring_decrypt_failed"
        assert row["package_hash"] == result.package_hash
        assert row["messages_restored"] == 0
        assert row["shards_restored"] == 0


# ---------------------------------------------------------------------------
# 4. continuity_threshold gate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_continuity_threshold_gate(db_path: Path, user_secret: str):
    async with Storage(db_path=str(db_path), agent_id=AGENT_ID) as storage:
        await _seed(storage)
        adapter = SovereignStorageAdapter(
            storage.db, user_secret=user_secret, agent_id=AGENT_ID,
        )
        cid = await adapter.export_agent(
            AGENT_DID, storage_tier=StorageTier.LOCAL_ONLY,
        )

        # A perfectly valid package, but an impossible threshold (>1.0)
        # so even a flawless score (1.0) is below the bar.
        await storage.db.execute_commit("DELETE FROM conversation_history")
        await storage.add_conversation(
            "user", "THRESHOLD_SENTINEL",
            metadata={"timestamp": "2026-01-01T00:00:00Z"},
        )

        result = await adapter.import_agent(cid, continuity_threshold=1.01)

        assert result.success is False
        assert result.status == "rejected"
        assert result.reject_reason is not None
        assert result.reject_reason.startswith("continuity_below_threshold")
        # The package itself is pristine — only the threshold rejected it.
        assert result.continuity.overall_score == 1.0

        # DB untouched.
        after = await storage.get_conversation_history()
        assert len(after) == 1
        assert after[0]["content"] == "THRESHOLD_SENTINEL"

        log = await adapter.get_import_log(agent_did=AGENT_DID)
        assert len(log) == 1
        assert log[0]["status"] == "rejected"
        assert log[0]["reject_reason"].startswith("continuity_below_threshold")


# ---------------------------------------------------------------------------
# 5. Append-only audit log accumulates across mixed outcomes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_append_only_audit_log_accumulates(
    db_path: Path, user_secret: str,
):
    wrong_secret = secrets.token_hex(32)

    async with Storage(db_path=str(db_path), agent_id=AGENT_ID) as storage:
        await _seed(storage)
        good = SovereignStorageAdapter(
            storage.db, user_secret=user_secret, agent_id=AGENT_ID,
        )
        cid = await good.export_agent(
            AGENT_DID, storage_tier=StorageTier.LOCAL_ONLY,
        )

        # Attempt 1: good import.
        await storage.db.execute_commit("DELETE FROM conversation_history")
        r1 = await good.import_agent(cid)
        assert r1.success and r1.status == "imported"
        assert len(await good.get_import_log(agent_did=AGENT_DID)) == 1

        # Attempt 2: wrong-secret rejection (DB must stay as r1 left it).
        bad = SovereignStorageAdapter(
            storage.db, user_secret=wrong_secret, agent_id=AGENT_ID,
        )
        history_after_r1 = await storage.get_conversation_history()
        r2 = await bad.import_agent(cid)
        assert r2.success is False and r2.status == "rejected"
        assert (
            await storage.get_conversation_history() == history_after_r1
        ), "rejected import must not mutate the host DB"
        assert len(await good.get_import_log(agent_did=AGENT_DID)) == 2

        # Attempt 3: good import again.
        await storage.db.execute_commit("DELETE FROM conversation_history")
        r3 = await good.import_agent(cid)
        assert r3.success and r3.status == "imported"

        log = await good.get_import_log(agent_did=AGENT_DID)
        assert len(log) == 3
        # Newest first.
        assert [row["status"] for row in log] == [
            "imported", "rejected", "imported",
        ]
        ids = [row["id"] for row in log]
        assert ids == sorted(ids, reverse=True), "log must be newest-first"

        # Append-only by design: no public update/delete surface exists.
        assert not hasattr(good, "update_import_log")
        assert not hasattr(good, "delete_import_log")
        assert not hasattr(good, "clear_import_log")

        # Row count strictly increases — never rewritten in place.
        assert len(await good.get_import_log()) >= 3
