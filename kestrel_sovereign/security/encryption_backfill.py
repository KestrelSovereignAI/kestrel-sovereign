"""One-shot backfill: encrypt pre-migration plaintext rows at rest (#1401).

The runtime write path (``AsyncConversationStore.add_message``,
``AsyncFileStore.store_file``) has set ``meta["enc"] = True`` for all
new rows since the encryption migration shipped, but pre-existing rows
were not backfilled. The opportunistic read-side
``_migrate_message`` runs only when an old row was encrypted under the
GLOBAL key and the agent key is now available — it doesn't fire for
truly plaintext rows. This module is the missing one-shot.

Operator workflow:

    # 1) Audit (read-only): see how many rows would be touched.
    kestrel migrate-encryption --data-dir agent_data/meridian --dry-run

    # 2) Stop the agent host (or shut down only the affected agent).
    kestrel stop

    # 3) Backfill.
    kestrel migrate-encryption --data-dir agent_data/meridian

    # 4) Verify: a second --dry-run should report 0 plaintext rows.
    kestrel migrate-encryption --data-dir agent_data/meridian --dry-run

    # 5) Restart.
    kestrel start

Invariants
----------

* **Idempotent.** Rows whose metadata already carries ``enc: true`` are
  excluded from the scan, so a second run is a no-op.

* **Metadata-preserving.** Existing metadata fields (``session_id``,
  ``sent_form``, ``privacy_mode``, …) survive the backfill. Only
  ``enc`` and ``key_version`` are added/updated.

* **Per-row atomic.** Each UPDATE is its own commit. A crash mid-run
  leaves the DB in a state that's safe to resume from (the unprocessed
  rows are simply still plaintext; re-running picks up where we left
  off because the WHERE clause is metadata-keyed).

* **Encrypt-only.** We never decrypt existing rows. Rows already
  encrypted under any key are untouched.

* **Soft-deleted rows included.** Trash rows can be restored by the
  user via the Trash UI (#763) and must be encrypted-at-rest too,
  otherwise a restore would silently make a previously-protected row
  visible-in-plain on disk.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from kestrel_sovereign.security.encryption import (
    get_agent_fernet,
    get_fernet,
)
from kestrel_sdk.security.encryption import (
    encrypt_bytes,
    encrypt_string_fernet,
)

logger = logging.getLogger(__name__)


# Mirror ``async_conversation_store.CURRENT_KEY_VERSION``. Duplicated
# (rather than imported) so the migration module doesn't pull in the
# async storage stack. Bump in lockstep if that constant ever changes.
CURRENT_KEY_VERSION = 1


@dataclass
class TableReport:
    """Per-table counts surfaced to the operator."""
    table: str
    scanned: int = 0
    plaintext: int = 0
    encrypted_now: int = 0
    skipped_no_fernet: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table": self.table,
            "scanned": self.scanned,
            "plaintext": self.plaintext,
            "encrypted_now": self.encrypted_now,
            "skipped_no_fernet": self.skipped_no_fernet,
            "errors": list(self.errors),
        }


@dataclass
class BackfillReport:
    """Combined report for ``migrate-encryption``."""
    db_path: str
    agent_id: Optional[str]
    dry_run: bool
    conversation: TableReport
    files: TableReport

    def to_dict(self) -> Dict[str, Any]:
        return {
            "db_path": self.db_path,
            "agent_id": self.agent_id,
            "dry_run": self.dry_run,
            "conversation": self.conversation.to_dict(),
            "files": self.files.to_dict(),
        }


def discover_agent_id(db_path: str | Path) -> Optional[str]:
    """Read the agent's DID from ``graph_nodes`` (node_type='agent').

    Agent DBs created by the inception flow always have exactly one
    node_type='agent' row holding the agent's DID. Multi-agent hosts
    keep each agent's DID in that agent's own DB. Returns None when
    the table is absent (legacy schema) or no agent node exists —
    the caller decides whether to require ``--agent-id``.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='graph_nodes'"
        )
        if cur.fetchone() is None:
            return None
        cur.execute(
            "SELECT node_id FROM graph_nodes "
            "WHERE node_type='agent' LIMIT 1"
        )
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _is_plaintext(metadata_json: Optional[str]) -> bool:
    """A row is plaintext when its metadata lacks ``enc: true``.

    Rows with NULL metadata, malformed JSON, or ``enc: false`` are
    treated as plaintext. Rows with ``enc: true`` are encrypted
    regardless of ``key_version``.
    """
    if not metadata_json:
        return True
    try:
        meta = json.loads(metadata_json)
    except (ValueError, TypeError):
        return True
    if not isinstance(meta, dict):
        return True
    return not meta.get("enc")


def _parse_metadata(metadata_json: Optional[str]) -> Dict[str, Any]:
    """Best-effort metadata parse. Returns {} on NULL/malformed."""
    if not metadata_json:
        return {}
    try:
        meta = json.loads(metadata_json)
    except (ValueError, TypeError):
        return {}
    return meta if isinstance(meta, dict) else {}


def backfill_conversation_history(
    db_path: str | Path,
    *,
    agent_id: str,
    dry_run: bool = False,
) -> TableReport:
    """Re-encrypt every plaintext ``conversation_history`` row.

    Per-agent encryption (``key_version=1``) when ``KESTREL_DATA_KEY``
    is configured and the agent_id resolves a fernet; otherwise the
    method is a no-op and reports ``skipped_no_fernet`` for the
    counted rows.
    """
    report = TableReport(table="conversation_history")
    agent_fernet = get_agent_fernet(agent_id) if agent_id else None
    global_fernet = get_fernet()
    fernet = agent_fernet or global_fernet

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        # Scope to this agent's rows. Multi-agent hosts share the DB
        # in some deployments; never touch a sibling agent's bytes.
        cur.execute(
            "SELECT id, content, metadata FROM conversation_history "
            "WHERE agent_id = ?",
            (agent_id,),
        )
        rows = cur.fetchall()

        for row_id, content, metadata_json in rows:
            report.scanned += 1
            if not _is_plaintext(metadata_json):
                continue
            report.plaintext += 1
            if fernet is None:
                report.skipped_no_fernet += 1
                continue
            if dry_run:
                continue
            meta = _parse_metadata(metadata_json)
            try:
                ciphertext, was_encrypted = encrypt_string_fernet(
                    content, fernet,
                )
            except Exception as exc:  # noqa: BLE001
                report.errors.append(
                    f"row id={row_id}: encrypt failed: {exc}"
                )
                continue
            if not was_encrypted:
                # ``encrypt_string_fernet`` returns ``was_encrypted=False``
                # when no fernet is configured — we already short-circuited
                # that above. Belt-and-suspenders: count it as skipped
                # rather than silently writing the cleartext back.
                report.skipped_no_fernet += 1
                continue
            meta["enc"] = True
            meta["key_version"] = (
                CURRENT_KEY_VERSION if agent_fernet is not None else 0
            )
            cur.execute(
                "UPDATE conversation_history SET content = ?, metadata = ? "
                "WHERE id = ? AND agent_id = ?",
                (ciphertext, json.dumps(meta), row_id, agent_id),
            )
            conn.commit()
            report.encrypted_now += 1
    finally:
        conn.close()
    return report


def backfill_files(
    db_path: str | Path,
    *,
    dry_run: bool = False,
) -> TableReport:
    """Re-encrypt every plaintext ``files`` row with the global fernet.

    The file store doesn't carry a per-agent key (no ``key_version``
    field in the ``files`` schema) — files are addressed by content
    hash, which collides across agents that store the same bytes.
    Use the global fernet only.
    """
    report = TableReport(table="files")
    fernet = get_fernet()

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='files'"
        )
        if cur.fetchone() is None:
            return report
        cur.execute("SELECT content_hash, content, metadata FROM files")
        rows = cur.fetchall()
        for content_hash, content_blob, metadata_json in rows:
            report.scanned += 1
            if not _is_plaintext(metadata_json):
                continue
            report.plaintext += 1
            if fernet is None:
                report.skipped_no_fernet += 1
                continue
            if dry_run:
                continue
            meta = _parse_metadata(metadata_json)
            try:
                ciphertext, was_encrypted = encrypt_bytes(content_blob, fernet)
            except Exception as exc:  # noqa: BLE001
                report.errors.append(
                    f"hash={content_hash}: encrypt failed: {exc}"
                )
                continue
            if not was_encrypted:
                report.skipped_no_fernet += 1
                continue
            meta["enc"] = True
            cur.execute(
                "UPDATE files SET content = ?, metadata = ? "
                "WHERE content_hash = ?",
                (ciphertext, json.dumps(meta), content_hash),
            )
            conn.commit()
            report.encrypted_now += 1
    finally:
        conn.close()
    return report


def backfill_all(
    db_path: str | Path,
    *,
    agent_id: Optional[str] = None,
    dry_run: bool = False,
) -> BackfillReport:
    """Run both backfills against a single agent's kestrel_prime.db.

    When ``agent_id`` is None, resolves from the DB's
    ``graph_nodes`` table. The migration is a no-op for
    ``conversation_history`` if the agent_id can't be resolved (the
    runtime write path is scoped by agent_id; backfilling without
    that scope would corrupt cross-agent isolation on multi-agent
    DBs). Files always backfill globally — they have no per-agent
    scope to violate.
    """
    resolved_id = agent_id or discover_agent_id(db_path)
    if resolved_id is None:
        conversation_report = TableReport(
            table="conversation_history",
            errors=[
                "agent_id could not be resolved from graph_nodes; pass "
                "--agent-id explicitly or run inception first"
            ],
        )
    else:
        conversation_report = backfill_conversation_history(
            db_path, agent_id=resolved_id, dry_run=dry_run,
        )

    files_report = backfill_files(db_path, dry_run=dry_run)

    return BackfillReport(
        db_path=str(db_path),
        agent_id=resolved_id,
        dry_run=dry_run,
        conversation=conversation_report,
        files=files_report,
    )
