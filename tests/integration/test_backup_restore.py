"""
Backup/restore round-trip integration test — port of
``scripts/test_backup_restore.sh`` (epic #1050 tier 2.3).

The bash original ran a 6-phase manual flow:

    1. Generate a Fernet key, create a test agent
    2. Insert sensitive test data
    3. Export sovereignty (encrypted shards → CID)
    4. Wipe the DB to simulate data loss
    5. Re-import from CID
    6. Verify integrity + reject wrong key

Three things changed in the port beyond bash → Python:

- **No Fernet.** The bash script generated a Fernet key per
  ``CRYPTO_INVENTORY.md`` line 82's note "update once v2 AEAD ships".
  The v2 AEAD (AES-256-GCM via ``SovereignStorageAdapter``'s
  ``ConvergentEncryptor``) shipped in the Quantum Hardening epic
  (#921), so the test now uses the post-quantum path directly. The
  ``user_secret`` argument to :class:`SovereignStorageAdapter` is a
  string and the adapter derives the AES key internally — no Fernet
  involvement at any layer.
- **No Docker mode.** The bash script's ``--docker`` branch ran the
  same flow inside a sovereign container; the integration suite
  already runs in CI's container, so the dual-mode wiring is dead
  weight here. If a Docker-only verification is later wanted, that's
  a separate test (and probably belongs in tests/e2e/ alongside
  Playwright fixtures).
- **Local storage tier.** ``StorageTier.LOCAL_ONLY`` keeps the test
  hermetic — no IPFS dependency in the dev venv or CI. The
  encryption + sharding path is identical regardless of tier; the
  only thing IPFS would prove additionally is network upload, which
  ``tests/integration/test_sovereignty_e2e.py`` already covers when
  ``IPFS_HOST`` is set.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

import pytest

from kestrel_sovereign.filecoin_adapter import StorageTier
from kestrel_sovereign.storage import Storage
from kestrel_sovereign.storage.sovereign_adapter import SovereignStorageAdapter

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

AGENT_ID = "test:backup-restore"
AGENT_DID = "did:test:backup-restore"


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Per-test SQLite DB path. ``tmp_path`` cleans up automatically;
    the bash original used ``mktemp -d`` + an EXIT trap, the pytest
    fixture is the equivalent."""
    return tmp_path / "kestrel_prime.db"


@pytest.fixture
def user_secret() -> str:
    """A fresh per-test 256-bit secret. The bash script generated a
    Fernet key here; we just need 32 random bytes hex-encoded — the
    ``ConvergentEncryptor`` inside ``SovereignStorageAdapter`` derives
    its AES-256-GCM keys via SHA-256 from this secret, so the input
    just needs sufficient entropy."""
    return secrets.token_hex(32)


# Test data that exercises three things at once:
#   - a "sensitive" string that proves round-trip encryption works
#     (the bash original used a fake SSN — same idea)
#   - a marker string we can grep for to confirm we got OUR backup
#   - multiple roles + timestamps so the month-shard path is exercised
_TEST_MESSAGES = [
    ("user", "My social security number is 123-45-6789", "2025-11-01T10:00:00Z"),
    ("assistant", "I will keep that secure.", "2025-11-01T10:00:01Z"),
    ("user", "Remember my birthday is July 4, 1990.", "2025-11-15T09:00:00Z"),
    ("assistant", "Noted! Your birthday is Independence Day.", "2025-11-15T09:00:01Z"),
    ("user", "BACKUP_MARKER_12345", "2025-12-01T12:00:00Z"),
]


# ---------------------------------------------------------------------------
# The round-trip
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_backup_restore_round_trip(db_path: Path, user_secret: str):
    """Phases 1-5 of the bash original, end-to-end in one async test.

    The split-into-six-shell-blocks structure of the bash script was
    necessary because each block had to spin up a fresh ``uv run
    python -c '...'`` subprocess to load the storage stack. In
    Python land we can drive everything from a single async context.
    """
    # ---- Phase 1: create agent + insert test data ------------------
    async with Storage(db_path=str(db_path), agent_id=AGENT_ID) as storage:
        for role, content, ts in _TEST_MESSAGES:
            await storage.add_conversation(
                role, content, metadata={"timestamp": ts},
            )
        history = await storage.get_conversation_history()
        assert len(history) == len(_TEST_MESSAGES), (
            f"setup: expected {len(_TEST_MESSAGES)} messages, "
            f"got {len(history)}"
        )

        # ---- Phase 2: export sovereignty (backup → CID) -------------
        adapter = SovereignStorageAdapter(
            storage.db,
            user_secret=user_secret,
            agent_id=AGENT_ID,
        )
        cid = await adapter.export_agent(
            AGENT_DID,
            storage_tier=StorageTier.LOCAL_ONLY,
        )
        assert cid, "export returned empty CID"

        # ---- Phase 3: simulate data loss ----------------------------
        await storage.db.execute_commit("DELETE FROM conversation_history")
        post_delete = await storage.get_conversation_history()
        assert len(post_delete) == 0, "data-loss simulation incomplete"

        # ---- Phase 4: import from CID -------------------------------
        result = await adapter.import_agent(cid)
        assert result.success
        assert result.status == "imported"
        assert result.messages_restored == len(_TEST_MESSAGES)
        assert result.shards_restored > 0
        assert result.agent_did == AGENT_DID
        assert result.continuity.is_verified()

        # ---- Phase 5: verify integrity ------------------------------
        restored = await storage.get_conversation_history()
        assert len(restored) == len(_TEST_MESSAGES), (
            f"restored {len(restored)} messages, "
            f"expected {len(_TEST_MESSAGES)}"
        )

        # The marker proves we got OUR backup, not some other agent's.
        joined = " ".join(str(m.get("content", "")) for m in restored)
        assert "BACKUP_MARKER_12345" in joined, (
            "backup marker missing from restored data"
        )
        # The "sensitive" string proves the AES-256-GCM round-trip
        # actually decrypted (encrypted-on-disk + decrypted-on-import).
        assert "123-45-6789" in joined, (
            "sensitive payload missing — encryption round-trip broken"
        )

        # Full per-message round-trip equality — bash didn't check
        # this strictly (only marker + SSN substring), but a real
        # round-trip test should assert role + content equality on
        # every message.
        for original, got in zip(_TEST_MESSAGES, restored):
            orig_role, orig_content, _ = original
            assert got["role"] == orig_role
            assert got["content"] == orig_content


@pytest.mark.asyncio
async def test_import_with_wrong_key_rejects(db_path: Path, user_secret: str):
    """Phase 6 of the bash original, ported to the new contract: import
    with a non-matching ``user_secret`` must be *rejected* (structured
    result, no exception) and must leave the host DB untouched. Proves
    the encryption is real, the AEAD tag is validated, and the
    verification gate fires before any host mutation.

    The bash variant used a different Fernet key as the "wrong" key;
    we just generate a fresh secret. Either way the
    ``ConvergentEncryptor`` derives a different AES key from the
    secret and the AES-256-GCM tag fails to verify."""
    wrong_secret = secrets.token_hex(32)
    assert wrong_secret != user_secret  # sanity

    async with Storage(db_path=str(db_path), agent_id=AGENT_ID) as storage:
        await storage.add_conversation(
            "user", "secret-payload",
            metadata={"timestamp": "2025-11-01T10:00:00Z"},
        )
        export_adapter = SovereignStorageAdapter(
            storage.db,
            user_secret=user_secret,
            agent_id=AGENT_ID,
        )
        cid = await export_adapter.export_agent(
            AGENT_DID,
            storage_tier=StorageTier.LOCAL_ONLY,
        )

        # Sentinel must survive a rejected import — the host DB is
        # only mutated AFTER verification passes.
        before = await storage.get_conversation_history()
        assert len(before) == 1

        wrong_adapter = SovereignStorageAdapter(
            storage.db,
            user_secret=wrong_secret,
            agent_id=AGENT_ID,
        )
        result = await wrong_adapter.import_agent(cid)
        assert result.success is False
        assert result.status == "rejected"
        assert result.reject_reason == "keyring_decrypt_failed"

        # Host DB UNTOUCHED.
        after = await storage.get_conversation_history()
        assert len(after) == 1
        assert after[0]["content"] == "secret-payload"
