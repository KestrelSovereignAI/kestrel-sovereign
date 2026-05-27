"""Unit tests for the one-shot encryption backfill (#1401).

The migration's invariants:
- Idempotent — second run is a no-op.
- Metadata-preserving — existing fields survive.
- Per-row atomic — partial runs are resumable.
- Encrypt-only — never decrypts existing rows.
- Soft-deleted rows ARE encrypted (they can be restored from Trash).
"""

from __future__ import annotations

import base64
import json
import os
import sqlite3
from pathlib import Path

import pytest

from kestrel_sovereign.security.encryption_backfill import (
    backfill_all,
    backfill_conversation_history,
    backfill_files,
    discover_agent_id,
    _is_plaintext,
)


AGENT_DID = "did:pkh:eip155:1:0xTestAgent1234567890"
OTHER_AGENT_DID = "did:pkh:eip155:1:0xOtherAgent9999999"


@pytest.fixture
def data_key(monkeypatch):
    """Configure a stable KESTREL_DATA_KEY so backfill can encrypt.

    Without a configured master key, ``get_fernet`` returns None and
    the backfill is a no-op (counted as ``skipped_no_fernet``). Set
    a deterministic test key so we can assert real encryption.
    """
    key = base64.urlsafe_b64encode(b"k" * 32).decode("ascii")
    monkeypatch.setenv("KESTREL_DATA_KEY", key)
    # Reset cached fernets so the fresh env var is picked up.
    from kestrel_sdk.security import encryption as _enc_mod
    if hasattr(_enc_mod, "_FERNET_CACHE"):
        _enc_mod._FERNET_CACHE = None
    yield key


@pytest.fixture
def seeded_db(tmp_path):
    """Build a fresh kestrel_prime.db with mixed encryption state."""
    db_path = tmp_path / "kestrel_prime.db"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE graph_nodes (
            node_id TEXT PRIMARY KEY,
            node_type TEXT NOT NULL,
            label TEXT NOT NULL,
            properties TEXT
        );
        CREATE TABLE conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT NOT NULL DEFAULT '',
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            deleted_at TIMESTAMP DEFAULT NULL
        );
        CREATE TABLE files (
            content_hash TEXT PRIMARY KEY,
            original_name TEXT NOT NULL,
            content BLOB,
            metadata TEXT
        );
        """
    )
    cur.execute(
        "INSERT INTO graph_nodes (node_id, node_type, label, properties) "
        "VALUES (?, 'agent', 'Test', '{}')",
        (AGENT_DID,),
    )
    # Plaintext rows (pre-migration shape)
    cur.execute(
        "INSERT INTO conversation_history (agent_id, role, content, metadata) "
        "VALUES (?, 'assistant', 'plain reply 1', NULL)",
        (AGENT_DID,),
    )
    cur.execute(
        "INSERT INTO conversation_history (agent_id, role, content, metadata) "
        "VALUES (?, 'user', 'plain user msg', "
        "'{\"session_id\": \"sess-A\"}')",
        (AGENT_DID,),
    )
    cur.execute(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, metadata, deleted_at) "
        "VALUES (?, 'assistant', 'trashed plain', NULL, "
        "'2026-01-01 00:00:00')",
        (AGENT_DID,),
    )
    # Already-encrypted row (must not be touched)
    cur.execute(
        "INSERT INTO conversation_history (agent_id, role, content, metadata) "
        "VALUES (?, 'assistant', 'ciphertext-stub', "
        "'{\"enc\": true, \"key_version\": 1, \"session_id\": \"sess-B\"}')",
        (AGENT_DID,),
    )
    # Row owned by a DIFFERENT agent (multi-agent DB isolation)
    cur.execute(
        "INSERT INTO conversation_history (agent_id, role, content, metadata) "
        "VALUES (?, 'user', 'other agent plain', NULL)",
        (OTHER_AGENT_DID,),
    )
    # Row with malformed metadata JSON (should be treated as plaintext)
    cur.execute(
        "INSERT INTO conversation_history (agent_id, role, content, metadata) "
        "VALUES (?, 'system', 'malformed-meta', 'not-json-at-all')",
        (AGENT_DID,),
    )
    # Plaintext files
    cur.execute(
        "INSERT INTO files (content_hash, original_name, content, metadata) "
        "VALUES (?, ?, ?, ?)",
        ("hash1", "a.bin", b"raw bytes 1", None),
    )
    cur.execute(
        "INSERT INTO files (content_hash, original_name, content, metadata) "
        "VALUES (?, ?, ?, ?)",
        ("hash2", "b.txt", b"plain text", '{"author": "alice"}'),
    )
    # Already-encrypted file (untouched)
    cur.execute(
        "INSERT INTO files (content_hash, original_name, content, metadata) "
        "VALUES (?, ?, ?, ?)",
        ("hash3", "c.bin", b"already-encrypted-stub", '{"enc": true}'),
    )
    conn.commit()
    conn.close()
    return db_path


class TestIsPlaintext:
    """Predicate for ``meta -> is plaintext?``."""

    def test_null_metadata(self):
        assert _is_plaintext(None)

    def test_empty_metadata(self):
        assert _is_plaintext("")

    def test_malformed_json(self):
        assert _is_plaintext("not-json")

    def test_non_dict_json(self):
        assert _is_plaintext("[1, 2, 3]")

    def test_missing_enc_key(self):
        assert _is_plaintext('{"session_id": "x"}')

    def test_enc_false(self):
        assert _is_plaintext('{"enc": false}')

    def test_enc_true(self):
        assert not _is_plaintext('{"enc": true}')

    def test_enc_true_with_extra_fields(self):
        assert not _is_plaintext(
            '{"enc": true, "key_version": 1, "session_id": "x"}'
        )


class TestDiscoverAgentId:

    def test_finds_agent_node(self, seeded_db):
        assert discover_agent_id(seeded_db) == AGENT_DID

    def test_returns_none_when_no_graph_nodes_table(self, tmp_path):
        db = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE other (x INTEGER)")
        conn.close()
        assert discover_agent_id(db) is None

    def test_returns_none_when_no_agent_row(self, tmp_path):
        db = tmp_path / "no_agent.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE graph_nodes (node_id TEXT PRIMARY KEY, "
            "node_type TEXT NOT NULL, label TEXT NOT NULL, properties TEXT)"
        )
        conn.commit()
        conn.close()
        assert discover_agent_id(db) is None


class TestConversationHistoryBackfill:

    def test_dry_run_does_not_write(self, seeded_db, data_key):
        report = backfill_conversation_history(
            seeded_db, agent_id=AGENT_DID, dry_run=True,
        )
        assert report.plaintext == 4  # 3 plain + 1 malformed-meta for this agent
        assert report.encrypted_now == 0
        # Verify nothing was written.
        conn = sqlite3.connect(str(seeded_db))
        cur = conn.cursor()
        cur.execute(
            "SELECT content, metadata FROM conversation_history "
            "WHERE agent_id = ? AND role = 'assistant' AND "
            "content = 'plain reply 1'",
            (AGENT_DID,),
        )
        row = cur.fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "plain reply 1"
        assert row[1] is None  # metadata untouched

    def test_write_encrypts_plaintext_rows(self, seeded_db, data_key):
        report = backfill_conversation_history(
            seeded_db, agent_id=AGENT_DID, dry_run=False,
        )
        assert report.plaintext == 4
        assert report.encrypted_now == 4
        assert report.errors == []

        # All this agent's rows now carry enc:true.
        conn = sqlite3.connect(str(seeded_db))
        cur = conn.cursor()
        cur.execute(
            "SELECT id, content, metadata FROM conversation_history "
            "WHERE agent_id = ?",
            (AGENT_DID,),
        )
        rows = cur.fetchall()
        conn.close()
        for row_id, content, metadata_json in rows:
            meta = json.loads(metadata_json)
            assert meta.get("enc") is True, f"row {row_id} not encrypted"
            assert meta.get("key_version") == 1

    def test_preserves_other_metadata_fields(self, seeded_db, data_key):
        backfill_conversation_history(
            seeded_db, agent_id=AGENT_DID, dry_run=False,
        )
        conn = sqlite3.connect(str(seeded_db))
        cur = conn.cursor()
        cur.execute(
            "SELECT metadata FROM conversation_history "
            "WHERE agent_id = ? AND role = 'user'",
            (AGENT_DID,),
        )
        meta = json.loads(cur.fetchone()[0])
        conn.close()
        # session_id from the original metadata must survive.
        assert meta.get("session_id") == "sess-A"
        assert meta.get("enc") is True

    def test_does_not_touch_other_agent_rows(self, seeded_db, data_key):
        backfill_conversation_history(
            seeded_db, agent_id=AGENT_DID, dry_run=False,
        )
        conn = sqlite3.connect(str(seeded_db))
        cur = conn.cursor()
        cur.execute(
            "SELECT content, metadata FROM conversation_history "
            "WHERE agent_id = ?",
            (OTHER_AGENT_DID,),
        )
        row = cur.fetchone()
        conn.close()
        # Other agent's plaintext row is untouched.
        assert row[0] == "other agent plain"
        assert row[1] is None

    def test_does_not_touch_already_encrypted_rows(self, seeded_db, data_key):
        backfill_conversation_history(
            seeded_db, agent_id=AGENT_DID, dry_run=False,
        )
        conn = sqlite3.connect(str(seeded_db))
        cur = conn.cursor()
        cur.execute(
            "SELECT content FROM conversation_history "
            "WHERE agent_id = ? AND metadata LIKE '%sess-B%'",
            (AGENT_DID,),
        )
        row = cur.fetchone()
        conn.close()
        # The already-encrypted row's content is still the literal
        # stub we seeded — backfill did not re-encrypt it.
        assert row[0] == "ciphertext-stub"

    def test_idempotent_second_run(self, seeded_db, data_key):
        backfill_conversation_history(
            seeded_db, agent_id=AGENT_DID, dry_run=False,
        )
        # Second run finds zero plaintext rows.
        report = backfill_conversation_history(
            seeded_db, agent_id=AGENT_DID, dry_run=False,
        )
        assert report.plaintext == 0
        assert report.encrypted_now == 0
        assert report.errors == []


class TestFilesBackfill:

    def test_dry_run_does_not_write(self, seeded_db, data_key):
        report = backfill_files(seeded_db, dry_run=True)
        assert report.plaintext == 2
        assert report.encrypted_now == 0
        conn = sqlite3.connect(str(seeded_db))
        cur = conn.cursor()
        cur.execute(
            "SELECT content FROM files WHERE content_hash = 'hash1'"
        )
        assert cur.fetchone()[0] == b"raw bytes 1"
        conn.close()

    def test_write_encrypts_plaintext_files(self, seeded_db, data_key):
        report = backfill_files(seeded_db, dry_run=False)
        assert report.plaintext == 2
        assert report.encrypted_now == 2

        conn = sqlite3.connect(str(seeded_db))
        cur = conn.cursor()
        cur.execute(
            "SELECT content, metadata FROM files "
            "WHERE content_hash IN ('hash1', 'hash2')"
        )
        for content, metadata_json in cur.fetchall():
            assert content != b"raw bytes 1"
            assert content != b"plain text"
            meta = json.loads(metadata_json)
            assert meta.get("enc") is True
        # The pre-encrypted file is untouched.
        cur.execute("SELECT content FROM files WHERE content_hash = 'hash3'")
        assert cur.fetchone()[0] == b"already-encrypted-stub"
        conn.close()

    def test_preserves_other_file_metadata(self, seeded_db, data_key):
        backfill_files(seeded_db, dry_run=False)
        conn = sqlite3.connect(str(seeded_db))
        cur = conn.cursor()
        cur.execute(
            "SELECT metadata FROM files WHERE content_hash = 'hash2'"
        )
        meta = json.loads(cur.fetchone()[0])
        conn.close()
        assert meta.get("author") == "alice"
        assert meta.get("enc") is True

    def test_no_files_table_returns_empty_report(self, tmp_path, data_key):
        db = tmp_path / "no_files.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE other (x INTEGER)"
        )
        conn.commit()
        conn.close()
        report = backfill_files(db, dry_run=False)
        assert report.scanned == 0
        assert report.plaintext == 0
        assert report.encrypted_now == 0


class TestBackfillAll:

    def test_combined_dry_run(self, seeded_db, data_key):
        report = backfill_all(seeded_db, dry_run=True)
        assert report.dry_run is True
        assert report.agent_id == AGENT_DID
        assert report.conversation.plaintext == 4
        assert report.files.plaintext == 2
        # Dry run means nothing written.
        assert report.conversation.encrypted_now == 0
        assert report.files.encrypted_now == 0

    def test_combined_write(self, seeded_db, data_key):
        report = backfill_all(seeded_db, dry_run=False)
        assert report.conversation.encrypted_now == 4
        assert report.files.encrypted_now == 2

        # Sanity: a third dry-run pass finds nothing left.
        report2 = backfill_all(seeded_db, dry_run=True)
        assert report2.conversation.plaintext == 0
        assert report2.files.plaintext == 0

    def test_resolves_agent_id_from_db(self, seeded_db, data_key):
        report = backfill_all(seeded_db)
        assert report.agent_id == AGENT_DID

    def test_explicit_agent_id_wins(self, seeded_db, data_key):
        report = backfill_all(seeded_db, agent_id=OTHER_AGENT_DID)
        # With OTHER_AGENT_DID, only the one row owned by that agent
        # gets backfilled — the four AGENT_DID rows are out of scope.
        assert report.agent_id == OTHER_AGENT_DID
        assert report.conversation.plaintext == 1

    def test_unresolvable_agent_id_reports_error(self, tmp_path, data_key):
        db = tmp_path / "no_agent.db"
        conn = sqlite3.connect(str(db))
        conn.executescript(
            """
            CREATE TABLE graph_nodes (
                node_id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                label TEXT NOT NULL,
                properties TEXT
            );
            CREATE TABLE files (
                content_hash TEXT PRIMARY KEY,
                original_name TEXT NOT NULL,
                content BLOB,
                metadata TEXT
            );
            """
        )
        conn.commit()
        conn.close()
        report = backfill_all(db)
        assert report.agent_id is None
        # Files backfill still runs — it doesn't need agent_id.
        assert report.conversation.errors  # one error explaining the miss
        assert report.files.errors == []
