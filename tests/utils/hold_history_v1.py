"""Rewind a Hold store to the state a pre-``authority`` release left behind.

The v2 anchor-format migration (#3166) must boot a host whose history was
anchored by an older binary. These helpers reproduce that host from a store
written by the current code: ``hold_receipts`` loses its ``authority`` column,
the v2 migration marker is removed, and every external history head (the
anchor file or PostgreSQL evidence key, and the SQLite custody marker) is
rewritten as the v1 anchor of the same receipts.

``v1_history_anchor`` is a pinned, independent copy of the v1 algorithm rather
than a call into ``kestrel_sovereign.hold.state``: if the runtime's v1 encoder
drifts, an upgrade test built on it would agree with the drift and stop
proving that a real v1 host still boots.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

V1_HISTORY_ANCHOR_HEADER = b"kestrel-hold-history-v1\n"
V1_RECEIPT_COLUMNS = (
    "receipt_id, operation_id, action, disposition, scope, target_id, reason, "
    "actor_id, occurred_at, expected_hold_receipt_id, prior_hold_receipt_id, "
    "resulting_hold_receipt_id"
)


def _framed_digest(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def v1_history_anchor(rows: Iterable[Any]) -> bytes:
    """The whole-history anchor a v1 release wrote for these twelve-column rows."""

    rows = tuple(rows)
    digest = hashlib.sha256()
    digest.update(V1_HISTORY_ANCHOR_HEADER)
    for row in rows:
        assert len(row) == 12, "a v1 anchor covers exactly the twelve v1 fields"
        for value in (row[0], _framed_digest(row)):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return (
        V1_HISTORY_ANCHOR_HEADER
        + str(len(rows)).encode("ascii")
        + b"\n"
        + digest.hexdigest().encode("ascii")
        + b"\n"
    )


async def v1_receipt_rows(db: Any) -> tuple[Any, ...]:
    return tuple(
        await db.fetchall(
            f"SELECT {V1_RECEIPT_COLUMNS} FROM hold_receipts ORDER BY receipt_id"
        )
    )


async def publish_history_head(store: Any, payload: bytes) -> None:
    """Overwrite every external history head the store keeps with ``payload``."""

    from kestrel_sovereign.hold.state import _POSTGRES_HISTORY_ANCHOR_KEY

    if store._history_anchor_path is not None:
        store._write_file_evidence(
            store._history_anchor_path,
            payload,
            label="test Hold history anchor",
        )
    else:
        await store._write_postgres_evidence(_POSTGRES_HISTORY_ANCHOR_KEY, payload)
    store._write_sqlite_custody_marker(payload)


async def rewind_to_v1_history_anchor(db: Any, store: Any) -> bytes:
    """Put ``db`` and its external evidence into the v1 shape; return the anchor.

    Every receipt, witness, and latch is preserved exactly; only what the v2
    migration adds is taken away.
    """

    from kestrel_sovereign.hold.state import (
        _HISTORY_ANCHOR_V2_MIGRATION,
        _HISTORY_ANCHOR_V3_MIGRATION,
    )

    rows = await v1_receipt_rows(db)
    if getattr(db, "backend_type", "") == "postgres":
        await db.execute("ALTER TABLE hold_receipts DROP COLUMN authority")
    else:
        # A table rebuild, because SQLite's DROP COLUMN crashes on this
        # triggered table. The old table's indexes and feed trigger go with
        # it and are re-established by the upgrade, as on any schema check.
        await db.execute(
            "CREATE TABLE hold_receipts_v1 ("
            "receipt_id TEXT NOT NULL PRIMARY KEY, "
            "operation_id TEXT NOT NULL UNIQUE, action TEXT NOT NULL, "
            "disposition TEXT NOT NULL, scope TEXT NOT NULL, "
            "target_id TEXT NOT NULL, reason TEXT NOT NULL DEFAULT '', "
            "actor_id TEXT NOT NULL, occurred_at TEXT NOT NULL, "
            "expected_hold_receipt_id TEXT NOT NULL DEFAULT '', "
            "prior_hold_receipt_id TEXT NOT NULL DEFAULT '', "
            "resulting_hold_receipt_id TEXT NOT NULL DEFAULT '', "
            "feed_seq BIGINT)"
        )
        columns = f"{V1_RECEIPT_COLUMNS}, feed_seq"
        await db.execute(
            f"INSERT INTO hold_receipts_v1 ({columns}) "
            f"SELECT {columns} FROM hold_receipts"
        )
        await db.execute("DROP TABLE hold_receipts")
        await db.execute("ALTER TABLE hold_receipts_v1 RENAME TO hold_receipts")
    await db.execute(
        "DELETE FROM hold_schema_migrations WHERE name IN (?, ?)",
        (_HISTORY_ANCHOR_V2_MIGRATION, _HISTORY_ANCHOR_V3_MIGRATION),
    )
    assert not await db.column_exists("hold_receipts", "authority")
    anchor = v1_history_anchor(rows)
    await publish_history_head(store, anchor)
    return anchor


V2_HISTORY_ANCHOR_HEADER = b"kestrel-hold-history-v2\n"
V2_RECEIPT_COLUMNS = f"{V1_RECEIPT_COLUMNS}, authority"
_WIDENED_SCOPE_CHECK = "CHECK (scope IN ('host', 'agent', 'mandate'))"
_NARROW_SCOPE_CHECK = "CHECK (scope IN ('host', 'agent'))"
_SCOPED_TABLES = (
    "hold_latches",
    "hold_receipts",
    "hold_receipt_witnesses",
    "hold_receipt_content_witnesses",
)


def v2_history_anchor(rows: Iterable[Any]) -> bytes:
    """The whole-history anchor a v2 release (#3166) wrote for these rows.

    Pinned independently of the runtime for the same reason as the v1 copy.
    """

    rows = tuple(rows)
    digest = hashlib.sha256()
    digest.update(V2_HISTORY_ANCHOR_HEADER)
    for row in rows:
        assert len(row) == 13, "a v2 anchor covers the v1 fields plus authority"
        content = list(row[:12])
        if row[12] != "sovereign":
            content.append(row[12])
        for value in (row[0], _framed_digest(content), row[12]):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return (
        V2_HISTORY_ANCHOR_HEADER
        + str(len(rows)).encode("ascii")
        + b"\n"
        + digest.hexdigest().encode("ascii")
        + b"\n"
    )


async def rewind_to_v2_history_anchor(db: Any, store: Any) -> bytes:
    """Put ``db`` into the shape a v2 release (#3166) left; return the anchor.

    The scope CHECK is narrowed back to host/agent on every scoped table, the
    v3 marker is removed, and every external head is the v2 anchor of the
    same receipts. Every receipt, witness, and latch is preserved exactly.
    """

    from kestrel_sovereign.hold.state import _HISTORY_ANCHOR_V3_MIGRATION

    rows = tuple(
        await db.fetchall(
            f"SELECT {V2_RECEIPT_COLUMNS} FROM hold_receipts ORDER BY receipt_id"
        )
    )
    for table in _SCOPED_TABLES:
        if getattr(db, "backend_type", "") == "postgres":
            constraints = await db.fetchall(
                "SELECT conname, pg_get_constraintdef(oid) FROM pg_constraint "
                "WHERE conrelid = to_regclass(?) AND contype = 'c'",
                (table,),
            )
            for name, definition in constraints:
                if "'mandate'" in str(definition):
                    await db.execute(f'ALTER TABLE {table} DROP CONSTRAINT "{name}"')
            await db.execute(
                f"ALTER TABLE {table} ADD CONSTRAINT {table}_scope_check "
                f"{_NARROW_SCOPE_CHECK}"
            )
        else:
            [row] = await db.fetchall(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            )
            ddl = " ".join(row[0].split())
            assert _WIDENED_SCOPE_CHECK in ddl
            narrowed = ddl.replace(_WIDENED_SCOPE_CHECK, _NARROW_SCOPE_CHECK)
            prefix = f"CREATE TABLE {table} ("
            assert narrowed.startswith(prefix)
            await db.rebuild_sqlite_table(
                table, "CREATE TABLE {table} (" + narrowed[len(prefix):]
            )
    await db.execute(
        "DELETE FROM hold_schema_migrations WHERE name = ?",
        (_HISTORY_ANCHOR_V3_MIGRATION,),
    )
    anchor = v2_history_anchor(rows)
    await publish_history_head(store, anchor)
    return anchor
