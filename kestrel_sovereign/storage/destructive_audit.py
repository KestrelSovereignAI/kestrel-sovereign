"""Append-only audit trail for destructive storage operations (#750)."""

from __future__ import annotations

import contextvars
import getpass
import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import aiosqlite


AUDIT_DB_FILENAME = "kestrel_audit.db"
DESTRUCTIVE_AUDIT_TABLE = "destructive_audit_log"

_caller_identity_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "kestrel_destructive_audit_caller_identity",
    default=None,
)


@contextmanager
def destructive_audit_caller(identity: str) -> Iterator[None]:
    """Attach a caller identity to destructive audit records in this task."""

    token = _caller_identity_var.set(identity)
    try:
        yield
    finally:
        _caller_identity_var.reset(token)


def audit_db_path_for(main_db_path: str | os.PathLike[str]) -> Path:
    """Return the isolated destructive-audit DB path for a SQLite data DB."""

    return Path(main_db_path).expanduser().resolve().parent / AUDIT_DB_FILENAME


def default_caller_identity() -> str:
    """Process identity fallback when no auth principal is threaded."""

    contextual = _caller_identity_var.get()
    if contextual:
        return contextual
    try:
        user = getpass.getuser()
    except Exception:
        user = "unknown"
    return f"process:{os.getpid()}:{user}"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def hash_rows(rows: list[dict[str, Any]]) -> str:
    """Deterministic pre-operation hash for rows about to be destroyed."""

    return "sha256:" + hashlib.sha256(_json_dumps(rows).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DestructiveAuditEvent:
    agent_id: str
    operation_type: str
    row_count: int
    pre_operation_hash: str
    scope: dict[str, Any]
    reason: str
    snapshot_reference: Optional[str] = None
    caller_identity: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    approval_id: Optional[str] = None


class DestructiveAuditLog:
    """SQLite append-only audit DB isolated from the data being purged."""

    def __init__(self, db_path: str | os.PathLike[str]):
        self.db_path = Path(db_path)

    async def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA busy_timeout=30000")
            await db.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {DESTRUCTIVE_AUDIT_TABLE} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    caller_identity TEXT NOT NULL,
                    operation_type TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    pre_operation_hash TEXT NOT NULL,
                    snapshot_reference TEXT,
                    scope TEXT NOT NULL,
                    reason TEXT,
                    request_id TEXT,
                    session_id TEXT,
                    approval_id TEXT,
                    anchor_status TEXT NOT NULL DEFAULT 'pending'
                )
                """
            )
            await db.execute(
                f"""
                CREATE TRIGGER IF NOT EXISTS destructive_audit_no_update
                BEFORE UPDATE ON {DESTRUCTIVE_AUDIT_TABLE}
                BEGIN
                    SELECT RAISE(ABORT, 'destructive_audit_log is append-only');
                END
                """
            )
            await db.execute(
                f"""
                CREATE TRIGGER IF NOT EXISTS destructive_audit_no_delete
                BEFORE DELETE ON {DESTRUCTIVE_AUDIT_TABLE}
                BEGIN
                    SELECT RAISE(ABORT, 'destructive_audit_log is append-only');
                END
                """
            )
            await db.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_destructive_audit_agent_ts
                ON {DESTRUCTIVE_AUDIT_TABLE}(agent_id, timestamp)
                """
            )
            await db.commit()

    async def append(self, event: DestructiveAuditEvent) -> int:
        await self.initialize()
        timestamp = datetime.now(timezone.utc).isoformat()
        caller_identity = event.caller_identity or default_caller_identity()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                f"""
                INSERT INTO {DESTRUCTIVE_AUDIT_TABLE}
                    (timestamp, agent_id, caller_identity, operation_type,
                     row_count, pre_operation_hash, snapshot_reference, scope,
                     reason, request_id, session_id, approval_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    event.agent_id,
                    caller_identity,
                    event.operation_type,
                    event.row_count,
                    event.pre_operation_hash,
                    event.snapshot_reference,
                    _json_dumps(event.scope),
                    event.reason,
                    event.request_id,
                    event.session_id,
                    event.approval_id,
                ),
            )
            await db.commit()
            return int(cursor.lastrowid)


__all__ = [
    "AUDIT_DB_FILENAME",
    "DESTRUCTIVE_AUDIT_TABLE",
    "DestructiveAuditEvent",
    "DestructiveAuditLog",
    "audit_db_path_for",
    "destructive_audit_caller",
    "hash_rows",
]
