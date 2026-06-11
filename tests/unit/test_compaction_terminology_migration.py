"""Tests for the compress → compact terminology data migration.

``migrate_compaction_terminology`` rewrites persisted session-compaction
metadata strings in ``conversation_history`` so the code never needs
dual-string read compat:

- ``type: "compression"`` → ``"compaction"``
- ``type: "hierarchical_compression"`` → ``"hierarchical_compaction"``
- key ``messages_compressed`` → ``messages_compacted``
- key ``compressed_at`` → ``compacted_at``
- ``salvage_reason: "manual-compress"`` → ``"manual-compact"``
- ``excluded_reason: "Replaced by compression"`` → ``"Replaced by compaction"``

Content (possibly encrypted at rest, #1401) is deliberately untouched.
The migration runs on every startup via ``AsyncDatabase._init_schema``,
so idempotency is part of the documented contract.
"""

import json

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db import SQLiteBackend
from kestrel_sovereign.storage.sqla.migrations import (
    migrate_compaction_terminology,
)


async def _db(tmp_path):
    raw = SQLiteBackend(str(tmp_path / "compaction-migration-test.db"))
    await raw.connect()
    db = AsyncDatabase(raw)
    # _init_schema creates conversation_history AND runs the migration
    # once (startup wiring under test elsewhere); rows inserted after
    # this exercise the function directly.
    await db._init_schema()
    return db


async def _insert(db, content, metadata):
    await db.execute(
        "INSERT INTO conversation_history (agent_id, role, content, metadata) "
        "VALUES (?, ?, ?, ?)",
        ("test-agent", "system", content, json.dumps(metadata) if metadata is not None else None),
    )


async def _all_rows(db):
    rows = await db.fetchall(
        "SELECT content, metadata FROM conversation_history ORDER BY id", ()
    )
    return [(r[0], json.loads(r[1]) if r[1] else None) for r in rows]


@pytest.mark.asyncio
async def test_rewrites_compaction_marker_metadata(tmp_path):
    db = await _db(tmp_path)
    await _insert(
        db,
        "[COMPRESSED CONTEXT - 13 messages summarized]\n\nSummary.",
        {
            "type": "compression",
            "messages_compressed": 13,
            "tokens_before": 900,
            "tokens_after": 90,
            "compressed_at": "2026-01-02T03:04:05+00:00",
        },
    )

    await migrate_compaction_terminology(db)

    content, meta = (await _all_rows(db))[0]
    assert meta["type"] == "compaction"
    assert meta["messages_compacted"] == 13
    assert "messages_compressed" not in meta
    assert meta["compacted_at"] == "2026-01-02T03:04:05+00:00"
    assert "compressed_at" not in meta
    # Untouched fields survive
    assert meta["tokens_before"] == 900
    # Content (possibly encrypted at rest) is never rewritten
    assert content.startswith("[COMPRESSED CONTEXT")


@pytest.mark.asyncio
async def test_rewrites_hierarchical_marker_and_salvage_reason(tmp_path):
    db = await _db(tmp_path)
    await _insert(
        db,
        "[HIERARCHICAL COMPRESSION - 20 messages, 4 chunks]\n\nSummary.",
        {"type": "hierarchical_compression", "messages_compressed": 20},
    )
    await _insert(
        db,
        "[SALVAGED]",
        {"type": "salvage", "salvage_reason": "manual-compress"},
    )
    await _insert(
        db,
        "old turn",
        {
            "excluded_from_context": True,
            "excluded_reason": "Replaced by compression",
            "summarized_into": "1",
        },
    )

    await migrate_compaction_terminology(db)

    rows = await _all_rows(db)
    assert rows[0][1]["type"] == "hierarchical_compaction"
    assert rows[0][1]["messages_compacted"] == 20
    assert rows[1][1]["salvage_reason"] == "manual-compact"
    assert rows[2][1]["excluded_reason"] == "Replaced by compaction"
    assert rows[2][1]["summarized_into"] == "1"


@pytest.mark.asyncio
async def test_leaves_unrelated_rows_alone(tmp_path):
    db = await _db(tmp_path)
    # Serializes to ``..."note": "discusses compression"...`` — matched
    # by the LIKE filter but no rewrite condition applies, so the
    # metadata must come back byte-identical.
    unrelated = {"type": "context_summary", "note": "discusses compression"}
    await _insert(db, "hello", unrelated)
    await _insert(db, "plain message", {"type": "chat"})
    await _insert(db, "no metadata", None)

    await migrate_compaction_terminology(db)

    rows = await _all_rows(db)
    assert rows[0][1] == unrelated
    assert rows[1][1] == {"type": "chat"}
    assert rows[2][1] is None


@pytest.mark.asyncio
async def test_idempotent_second_run_is_noop(tmp_path):
    db = await _db(tmp_path)
    await _insert(
        db,
        "[COMPRESSED CONTEXT - 3 messages summarized]",
        {"type": "compression", "messages_compressed": 3},
    )

    await migrate_compaction_terminology(db)
    first = await _all_rows(db)
    await migrate_compaction_terminology(db)
    second = await _all_rows(db)

    assert first == second
    assert first[0][1]["type"] == "compaction"


@pytest.mark.asyncio
async def test_tolerates_invalid_metadata_json(tmp_path):
    db = await _db(tmp_path)
    # Bypass _insert to write raw non-JSON metadata that matches the
    # LIKE filter — the migration must skip it, not raise.
    await db.execute(
        "INSERT INTO conversation_history (agent_id, role, content, metadata) "
        "VALUES (?, ?, ?, ?)",
        ("test-agent", "system", "x", 'not-json-compression"'),
    )

    await migrate_compaction_terminology(db)

    rows = await db.fetchall(
        "SELECT metadata FROM conversation_history ORDER BY id", ()
    )
    assert rows[0][0] == 'not-json-compression"'


@pytest.mark.asyncio
async def test_init_schema_runs_migration_on_startup(tmp_path):
    """The startup path itself rewrites legacy rows — a restart after
    upgrade is the only operator action required (carries to Frinz and
    other downstream deployments on dependency bump + restart)."""
    raw = SQLiteBackend(str(tmp_path / "startup-test.db"))
    await raw.connect()
    db = AsyncDatabase(raw)
    await db._init_schema()
    await _insert(
        db,
        "[COMPRESSED CONTEXT - 5 messages summarized]",
        {"type": "compression", "messages_compressed": 5},
    )

    # Simulate the next boot over the same backend.
    await db._init_schema()

    _, meta = (await _all_rows(db))[0]
    assert meta["type"] == "compaction"
    assert meta["messages_compacted"] == 5
