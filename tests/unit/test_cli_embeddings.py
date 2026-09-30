"""``kestrel embeddings verify|backfill``: the phase-2 gate CLI (#3405).

These drive the real ``kestrel`` entry point against a SQLite agent
database. ``run()`` owns its event loop, so the tests are synchronous and
inspect the database with the stdlib driver, which runs no schema
initializer of its own. The PostgreSQL case runs when ``TEST_POSTGRES_URL``
is set, which the CI unit tier provides.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sqlite3
import struct
import sys
from contextlib import closing
from dataclasses import fields
from uuid import uuid4

import pytest

from kestrel_sovereign import cli, cli_embeddings
from kestrel_sovereign.storage.active_embedding_profile import (
    ACTIVE_EMBEDDING_PROFILE_KEY,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.embedding_vec_backfill import (
    DEFAULT_BATCH_SIZE,
    LEGACY_EMBEDDING_TABLES,
    EmbeddingVecReport,
)
from tests.utils.legacy_embedding_column import restore_legacy_embedding_column
from tests.utils.postgres_schema import (
    pgvector_schema,
    quoted_search_path,
    with_search_path,
)

_REPORT_FIELDS = [field.name for field in fields(EmbeddingVecReport)]


def _pack(values):
    return struct.pack(f"<{len(values)}f", *values)


def _agent_data_dir(tmp_path, monkeypatch, *, legacy_column):
    for name in (
        "KESTREL_DATABASE_URL",
        "DATABASE_URL",
        "KESTREL_DB_BACKEND",
        "KESTREL_DB_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    data_dir = tmp_path / "agent"
    data_dir.mkdir()

    async def create():
        db = await AsyncDatabase.sqlite(str(data_dir / "kestrel_prime.db"))
        try:
            if legacy_column:
                await restore_legacy_embedding_column(db)
        finally:
            await db.close()

    asyncio.run(create())
    return data_dir


@pytest.fixture
def agent_dir(tmp_path, monkeypatch):
    """An agent data dir whose ``kestrel_prime.db`` has the pre-#3411 schema.

    The boot retires the legacy ``embedding`` column (#3411); it is restored
    so rows can be seeded the way an older release wrote them.
    """
    return _agent_data_dir(tmp_path, monkeypatch, legacy_column=True)


@pytest.fixture
def retired_agent_dir(tmp_path, monkeypatch):
    """An agent data dir as a fresh boot leaves it: no legacy column."""
    return _agent_data_dir(tmp_path, monkeypatch, legacy_column=False)


def _connect(data_dir):
    return sqlite3.connect(str(data_dir / "kestrel_prime.db"))


def _insert_saved_item(data_dir, item_id, embedding=None, embedding_vec=None):
    with closing(_connect(data_dir)) as conn, conn:
        conn.execute(
            "INSERT INTO saved_items (id, agent_id, item_type, name, content, "
            "embedding, embedding_vec) "
            "VALUES (?, 'did:test:agent', 'stash', ?, 'c', ?, ?)",
            (item_id, item_id, embedding, embedding_vec),
        )


def _insert_chunk(data_dir, content, embedding):
    with closing(_connect(data_dir)) as conn, conn:
        conn.execute(
            "INSERT INTO document_chunks (file_hash, content, embedding) "
            "VALUES (?, ?, ?)",
            ("doc", content, embedding),
        )


def _drop_chunk_embedding_vec(data_dir):
    with closing(_connect(data_dir)) as conn, conn:
        conn.execute("ALTER TABLE document_chunks DROP COLUMN embedding_vec")


def _columns(data_dir, table, id_col, row_id):
    with closing(_connect(data_dir)) as conn, conn:
        return conn.execute(
            f"SELECT embedding, embedding_vec FROM {table} WHERE {id_col} = ?",
            (row_id,),
        ).fetchone()


def _has_embedding_vec(data_dir, table):
    with closing(_connect(data_dir)) as conn, conn:
        return bool(conn.execute(
            f"SELECT 1 FROM pragma_table_info('{table}') "
            "WHERE name = 'embedding_vec'"
        ).fetchall())


def _snapshot(data_dir):
    """Every row of both tables plus the whole schema."""
    with closing(_connect(data_dir)) as conn, conn:
        return {
            "saved_items": conn.execute(
                "SELECT * FROM saved_items ORDER BY id"
            ).fetchall(),
            "document_chunks": conn.execute(
                "SELECT * FROM document_chunks ORDER BY chunk_id"
            ).fetchall(),
            "schema": conn.execute(
                "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
            ).fetchall(),
        }


def _kestrel(monkeypatch, *argv):
    """Run ``kestrel <argv>`` through the real entry point; return its code."""
    monkeypatch.setattr(sys, "argv", ["kestrel", *argv])
    return cli.main()


def _embeddings_json(monkeypatch, capsys, command, data_dir, *extra):
    rc = _kestrel(
        monkeypatch, "embeddings", command, "--data-dir", str(data_dir),
        "--json", *extra,
    )
    return rc, json.loads(capsys.readouterr().out)


def _table(payload, name):
    (entry,) = [t for t in payload["tables"] if t["table"] == name]
    return entry


@pytest.mark.parametrize("command", ["verify", "backfill"])
def test_empty_tables_meet_the_gate(agent_dir, monkeypatch, capsys, command):
    rc, payload = _embeddings_json(monkeypatch, capsys, command, agent_dir)

    assert rc == 0
    assert payload["command"] == command
    assert payload["gate_met"] is True
    assert [t["table"] for t in payload["tables"]] == list(LEGACY_EMBEDDING_TABLES)
    for entry in payload["tables"]:
        assert entry["embedding_vec_present"] is True
        assert entry["total_rows"] == 0
        assert entry["rows_missing_embedding_vec"] == 0
        assert entry["rows_backfilled"] == 0
        assert entry["gate_met"] is True


@pytest.mark.parametrize("command", ["verify", "backfill"])
def test_retired_legacy_column_meets_the_gate(
    retired_agent_dir, monkeypatch, capsys, command
):
    # After #3411 drops the legacy column there is nothing left to copy.
    with closing(_connect(retired_agent_dir)) as conn, conn:
        conn.execute(
            "INSERT INTO saved_items (id, agent_id, item_type, name, content, "
            "embedding_vec) VALUES ('vec', 'did:test:agent', 'stash', 'vec', "
            "'c', ?)",
            (_pack([1.0, 2.0]),),
        )
    before = _snapshot(retired_agent_dir)

    rc, payload = _embeddings_json(monkeypatch, capsys, command, retired_agent_dir)

    assert rc == 0
    assert payload["gate_met"] is True
    saved_items = _table(payload, "saved_items")
    assert (
        saved_items["rows_embedding_vec_only"],
        saved_items["rows_missing_embedding_vec"],
        saved_items["rows_backfilled"],
    ) == (1, 0, 0)
    assert _snapshot(retired_agent_dir) == before


_BUCKETS = (
    "rows_with_both",
    "rows_missing_embedding_vec",
    "rows_embedding_vec_only",
    "rows_without_any_embedding",
)


@pytest.mark.parametrize("schema", ["agent_dir", "retired_agent_dir"])
def test_a_never_embedded_row_is_counted_and_does_not_fail_the_gate(
    request, monkeypatch, capsys, schema
):
    # #3415: 47 chunks per agent had no vector in either column, and verify
    # put them in no bucket.
    data_dir = request.getfixturevalue(schema)
    with closing(_connect(data_dir)) as conn, conn:
        conn.execute(
            "INSERT INTO document_chunks (file_hash, content, embedding_vec) "
            "VALUES ('doc', 'embedded', ?)",
            (_pack([1.0, 2.0]),),
        )
        conn.execute(
            "INSERT INTO document_chunks (file_hash, content) "
            "VALUES ('doc', 'never embedded')"
        )

    rc, payload = _embeddings_json(
        monkeypatch, capsys, "verify", data_dir, "--table", "document_chunks"
    )

    assert rc == 0
    assert payload["gate_met"] is True
    (entry,) = payload["tables"]
    assert (
        entry["total_rows"],
        entry["rows_embedding_vec_only"],
        entry["rows_without_any_embedding"],
    ) == (2, 1, 1)
    assert sum(entry[bucket] for bucket in _BUCKETS) == entry["total_rows"]

    rc = _kestrel(
        monkeypatch, "embeddings", "verify", "--data-dir", str(data_dir),
        "--table", "document_chunks",
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert f"  {'rows_without_any_embedding':<28} {1:>8}" in out
    assert "note: 1 row(s) have no embedding in either column" in out
    assert "`kestrel embeddings reindex --yes` embeds them" in out
    assert "phase-2 gate: met" in out


def test_json_report_carries_every_report_field(agent_dir, monkeypatch, capsys):
    _insert_saved_item(agent_dir, "legacy-only", embedding=_pack([1.0, 2.0]))

    rc, payload = _embeddings_json(
        monkeypatch, capsys, "verify", agent_dir, "--table", "saved_items"
    )

    assert rc == cli_embeddings.EXIT_GATE_NOT_MET
    assert payload["gate_met"] is False
    (entry,) = payload["tables"]
    assert set(entry) == {*_REPORT_FIELDS, "gate_met"}
    assert entry["table"] == "saved_items"
    assert entry["rows_missing_embedding_vec"] == 1
    assert entry["rows_unbackfillable"] == 0
    assert entry["gate_met"] is False


def test_backfill_copies_missing_rows_and_then_meets_the_gate(
    agent_dir, monkeypatch, capsys
):
    item_vec = _pack([0.1, 0.2, 0.3])
    chunk_vec = _pack([0.4, 0.5, 0.6])
    _insert_saved_item(agent_dir, "legacy-only", embedding=item_vec)
    _insert_chunk(agent_dir, "chunk", chunk_vec)

    rc, before = _embeddings_json(monkeypatch, capsys, "verify", agent_dir)
    assert rc == cli_embeddings.EXIT_GATE_NOT_MET
    assert _table(before, "saved_items")["rows_missing_embedding_vec"] == 1
    assert _table(before, "document_chunks")["rows_missing_embedding_vec"] == 1

    rc, backfill = _embeddings_json(
        monkeypatch, capsys, "backfill", agent_dir, "--batch-size", "1"
    )

    assert rc == 0
    assert backfill["gate_met"] is True
    for name in LEGACY_EMBEDDING_TABLES:
        entry = _table(backfill, name)
        assert entry["rows_backfilled"] == 1
        assert entry["rows_missing_embedding_vec"] == 0
        assert entry["rows_with_both"] == 1
    assert _columns(agent_dir, "saved_items", "id", "legacy-only") == (
        item_vec, item_vec,
    )
    assert _columns(agent_dir, "document_chunks", "file_hash", "doc") == (
        chunk_vec, chunk_vec,
    )

    rc, again = _embeddings_json(monkeypatch, capsys, "backfill", agent_dir)
    assert rc == 0
    assert all(t["rows_backfilled"] == 0 for t in again["tables"])


def test_unbackfillable_nan_row_is_counted_and_reflected_in_the_exit_code(
    agent_dir, monkeypatch, capsys
):
    finite = _pack([0.5, -0.25])
    nan = _pack([0.5, float("nan")])
    _insert_saved_item(agent_dir, "a-nan", embedding=nan)
    _insert_saved_item(agent_dir, "b-finite", embedding=finite)

    # The finite row can still be copied, so the gate is not met yet.
    rc, verify = _embeddings_json(
        monkeypatch, capsys, "verify", agent_dir, "--table", "saved_items"
    )
    assert rc == cli_embeddings.EXIT_GATE_NOT_MET
    (entry,) = verify["tables"]
    assert (entry["rows_missing_embedding_vec"], entry["rows_unbackfillable"]) == (2, 1)

    # After the backfill only the NaN row is missing, and it is exactly the
    # one unbackfillable row: the gate is met.
    rc, backfill = _embeddings_json(
        monkeypatch, capsys, "backfill", agent_dir, "--table", "saved_items"
    )
    assert rc == 0
    (entry,) = backfill["tables"]
    assert entry["rows_backfilled"] == 1
    assert (entry["rows_missing_embedding_vec"], entry["rows_unbackfillable"]) == (1, 1)
    assert entry["gate_met"] is True
    assert _columns(agent_dir, "saved_items", "id", "a-nan") == (nan, None)
    assert _columns(agent_dir, "saved_items", "id", "b-finite") == (finite, finite)

    rc, verify_after = _embeddings_json(
        monkeypatch, capsys, "verify", agent_dir, "--table", "saved_items"
    )
    assert rc == 0
    assert verify_after["tables"][0]["rows_unbackfillable"] == 1


_ACTIVE = "activeprofil"
_ROUTE_SCOPED = "routescoped1"


def _record_active_profile(data_dir, profile_id, agent_id="did:test:agent"):
    with closing(_connect(data_dir)) as conn, conn:
        conn.execute(
            "INSERT INTO agent_metadata (agent_id, key, value) VALUES (?, ?, ?)",
            (agent_id, ACTIVE_EMBEDDING_PROFILE_KEY, json.dumps({
                "profile_id": profile_id,
                "provider": "shared:qwen3-embedding-8b@768",
                "model": "qwen3-embedding-8b",
                "dim": 768,
                "recorded_at": "2026-09-30T05:00:00+00:00",
            })),
        )


def _insert_profiled_rows(data_dir):
    """Stored vectors on and off ``_ACTIVE``, and one row with no vector."""
    with closing(_connect(data_dir)) as conn, conn:
        for content, vec, profile in (
            ("on", _pack([1.0]), _ACTIVE),
            ("reindexed away", _pack([2.0]), _ROUTE_SCOPED),
            ("unstamped", _pack([3.0]), None),
            ("never embedded", None, None),
        ):
            conn.execute(
                "INSERT INTO conversation_history (agent_id, role, content, "
                "embedding_vec, embedding_profile_id) "
                "VALUES ('did:test:agent', 'user', ?, ?, ?)",
                (content, vec, profile),
            )
        conn.execute(
            "INSERT INTO saved_items (id, agent_id, item_type, name, content, "
            "embedding_vec, embedding_profile_id) VALUES "
            "('s', 'did:test:agent', 'stash', 's', 'c', ?, ?)",
            (_pack([4.0]), _ROUTE_SCOPED),
        )
        conn.execute(
            "INSERT INTO document_chunks (file_hash, content, embedding_vec, "
            "embedding_profile_id) VALUES ('doc', 'chunk', ?, ?)",
            (_pack([5.0]), _ACTIVE),
        )


def test_verify_counts_stored_vectors_off_the_recorded_active_profile(
    retired_agent_dir, monkeypatch, capsys
):
    # #3420: a reindex moved every row off the profile the agent searches,
    # and verify still reported the gate met without a word about it.
    _record_active_profile(retired_agent_dir, _ACTIVE)
    _insert_profiled_rows(retired_agent_dir)
    before = _snapshot(retired_agent_dir)

    rc, payload = _embeddings_json(monkeypatch, capsys, "verify", retired_agent_dir)

    # Search visibility is reported, not part of the phase-2 exit code.
    assert rc == 0
    assert payload["gate_met"] is True
    active = payload["active_embedding_profile"]
    assert (active["status"], active["profile_id"]) == ("recorded", _ACTIVE)
    assert [r["agent_id"] for r in active["records"]] == ["did:test:agent"]
    assert active["tables"] == {
        "conversation_history": {
            "vectors_on_active_profile": 1, "vectors_off_active_profile": 2,
        },
        "saved_items": {
            "vectors_on_active_profile": 0, "vectors_off_active_profile": 1,
        },
        "document_chunks": {
            "vectors_on_active_profile": 1, "vectors_off_active_profile": 0,
        },
    }
    assert active["vectors_off_active_profile"] == 3
    assert _snapshot(retired_agent_dir) == before

    rc = _kestrel(
        monkeypatch, "embeddings", "verify", "--data-dir", str(retired_agent_dir)
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert (
        f"active embedding profile: {_ACTIVE} (shared:qwen3-embedding-8b@768/"
        "qwen3-embedding-8b dim=768, recorded by did:test:agent at "
        "2026-09-30T05:00:00+00:00)"
    ) in out
    assert f"  {'conversation_history':<28} on {1:>8}  off {2:>8}" in out
    assert "WARNING: 3 stored vector(s) are not on the active profile" in out
    assert out.rstrip().splitlines()[-1].startswith("phase-2 gate: met")


def test_verify_scopes_the_profile_count_to_the_selected_table(
    retired_agent_dir, monkeypatch, capsys
):
    _record_active_profile(retired_agent_dir, _ACTIVE)
    _insert_profiled_rows(retired_agent_dir)

    _, payload = _embeddings_json(
        monkeypatch, capsys, "verify", retired_agent_dir, "--table", "saved_items"
    )

    active = payload["active_embedding_profile"]
    assert list(active["tables"]) == ["saved_items"]
    assert active["vectors_off_active_profile"] == 1


@pytest.mark.parametrize(
    ("records", "status"),
    [((), "not_recorded"), ((_ACTIVE, _ROUTE_SCOPED), "ambiguous")],
)
def test_verify_counts_nothing_without_one_recorded_profile(
    retired_agent_dir, monkeypatch, capsys, records, status
):
    for index, profile_id in enumerate(records):
        _record_active_profile(retired_agent_dir, profile_id, f"did:test:{index}")
    _insert_profiled_rows(retired_agent_dir)

    rc, payload = _embeddings_json(monkeypatch, capsys, "verify", retired_agent_dir)

    assert rc == 0
    active = payload["active_embedding_profile"]
    assert (active["status"], active["profile_id"]) == (status, None)
    assert active["vectors_off_active_profile"] is None
    assert set(active["tables"].values()) == {None}

    _kestrel(monkeypatch, "embeddings", "verify", "--data-dir", str(retired_agent_dir))
    out = capsys.readouterr().out
    assert "not counted without one recorded profile" in out
    assert "WARNING" not in out


def test_verify_reports_an_unreadable_record(retired_agent_dir, monkeypatch, capsys):
    with closing(_connect(retired_agent_dir)) as conn, conn:
        conn.execute(
            "INSERT INTO agent_metadata (agent_id, key, value) VALUES (?, ?, ?)",
            ("did:test:agent", ACTIVE_EMBEDDING_PROFILE_KEY, "not json"),
        )

    rc, payload = _embeddings_json(monkeypatch, capsys, "verify", retired_agent_dir)

    assert rc == 0
    active = payload["active_embedding_profile"]
    assert active["status"] == "unreadable"
    assert "ActiveEmbeddingProfileError" in active["error"]


def test_backfill_reports_no_active_profile_section(
    retired_agent_dir, monkeypatch, capsys
):
    _, payload = _embeddings_json(monkeypatch, capsys, "backfill", retired_agent_dir)

    assert "active_embedding_profile" not in payload


def test_verify_writes_nothing(agent_dir, monkeypatch, capsys):
    legacy = _pack([1.0, 2.0])
    _insert_saved_item(agent_dir, "a-legacy-only", embedding=legacy)
    _insert_saved_item(
        agent_dir, "b-disagree", embedding=legacy, embedding_vec=_pack([3.0, 4.0])
    )
    _insert_saved_item(agent_dir, "c-nan", embedding=_pack([float("nan")]))
    _insert_chunk(agent_dir, "chunk", legacy)
    before = _snapshot(agent_dir)

    rc = _kestrel(monkeypatch, "embeddings", "verify", "--data-dir", str(agent_dir))

    assert rc == cli_embeddings.EXIT_GATE_NOT_MET
    assert _snapshot(agent_dir) == before
    assert "rows_backfilled" in capsys.readouterr().out


def _use_rollback_journal(data_dir):
    """Switch the fixture database from WAL to a ``DELETE`` rollback journal."""
    with closing(_connect(data_dir)) as conn:
        assert conn.execute("PRAGMA journal_mode=DELETE").fetchone() == ("delete",)


def _db_file(data_dir):
    return data_dir / "kestrel_prime.db"


def _sidecars(data_dir):
    return sorted(p.name for p in data_dir.glob("kestrel_prime.db-*"))


def _file_state(path):
    return path.read_bytes(), path.stat().st_mtime_ns


def _journal_mode(data_dir):
    uri = f"{_db_file(data_dir).as_uri()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        return conn.execute("PRAGMA journal_mode").fetchone()[0]


def test_verify_leaves_a_rollback_journal_database_unchanged(
    agent_dir, monkeypatch, capsys
):
    # The normal open runs ``PRAGMA journal_mode=WAL``, which permanently
    # converts a rollback-journal database whatever schema initializer it is
    # given (#3407).
    _insert_saved_item(agent_dir, "legacy-only", embedding=_pack([1.0, 2.0]))
    _use_rollback_journal(agent_dir)
    db_file = _db_file(agent_dir)
    # An mtime from long ago, so any write moves it however coarse the clock.
    os.utime(db_file, ns=(1_000_000_000_000_000_000, 1_000_000_000_000_000_000))
    before = _file_state(db_file)

    rc, payload = _embeddings_json(monkeypatch, capsys, "verify", agent_dir)

    assert rc == cli_embeddings.EXIT_GATE_NOT_MET
    assert _table(payload, "saved_items")["rows_missing_embedding_vec"] == 1
    assert _file_state(db_file) == before
    assert _sidecars(agent_dir) == []
    assert _journal_mode(agent_dir) == "delete"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
@pytest.mark.parametrize("journal_mode", ["delete", "wal"])
def test_verify_reports_on_a_read_only_database(
    agent_dir, monkeypatch, capsys, journal_mode
):
    if os.geteuid() == 0:
        pytest.skip("root ignores permission bits")
    _insert_saved_item(agent_dir, "legacy-only", embedding=_pack([1.0, 2.0]))
    if journal_mode == "delete":
        _use_rollback_journal(agent_dir)
    # A stopped agent: a WAL database is checkpointed, with no sidecars.
    assert _sidecars(agent_dir) == []
    db_file = _db_file(agent_dir)
    before = db_file.read_bytes()
    # A read-only file in a read-only directory: nothing can be written to
    # the database or created beside it, as on a read-only filesystem.
    db_file.chmod(0o444)
    agent_dir.chmod(0o555)
    try:
        rc = _kestrel(
            monkeypatch, "embeddings", "verify", "--data-dir", str(agent_dir),
            "--json",
        )
        captured = capsys.readouterr()
    finally:
        agent_dir.chmod(0o755)
        db_file.chmod(0o644)

    assert "ERROR" not in captured.err
    assert rc == cli_embeddings.EXIT_GATE_NOT_MET
    payload = json.loads(captured.out)
    assert payload["gate_met"] is False
    assert _table(payload, "saved_items")["rows_missing_embedding_vec"] == 1
    assert db_file.read_bytes() == before
    assert _journal_mode(agent_dir) == journal_mode


def test_verify_reads_rows_still_only_in_a_live_wal(
    agent_dir, monkeypatch, capsys, caplog
):
    # A running agent keeps the database open in WAL mode. An ``immutable=1``
    # read would ignore the WAL and miss the row it holds.
    holder = _connect(agent_dir)
    try:
        holder.execute("PRAGMA wal_autocheckpoint=0")
        holder.execute(
            "INSERT INTO saved_items (id, agent_id, item_type, name, content, "
            "embedding) VALUES ('in-wal', 'did:test:agent', 'stash', 'in-wal', "
            "'c', ?)",
            (_pack([1.0, 2.0]),),
        )
        holder.commit()
        assert _sidecars(agent_dir), "setup must leave a live WAL"

        with caplog.at_level(logging.WARNING):
            rc, payload = _embeddings_json(
                monkeypatch, capsys, "verify", agent_dir, "--table", "saved_items"
            )
    finally:
        holder.close()

    assert rc == cli_embeddings.EXIT_GATE_NOT_MET
    (entry,) = payload["tables"]
    assert entry["rows_missing_embedding_vec"] == 1
    # The agent's WAL outlived this read, which the cold-read backend reports.
    assert "WAL sidecars still present" in caplog.text


def test_verify_refuses_a_report_the_database_changed_under(
    agent_dir, monkeypatch, capsys
):
    # With no WAL to read, verify opens ``immutable=1``, which cannot see a
    # writer that commits while it reads. The report would describe a
    # database that no longer exists, so verify must refuse to print it.
    from kestrel_sovereign.storage import embedding_vec_backfill

    assert _sidecars(agent_dir) == [], "setup must leave no WAL to read"
    real_verify = embedding_vec_backfill.verify_embedding_vec

    async def verify_then_write(db, table):
        report = await real_verify(db, table)
        _insert_saved_item(agent_dir, "late", embedding=_pack([9.0] * 4096))
        return report

    monkeypatch.setattr(
        embedding_vec_backfill, "verify_embedding_vec", verify_then_write
    )

    rc = _kestrel(
        monkeypatch, "embeddings", "verify", "--data-dir", str(agent_dir),
        "--table", "saved_items",
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert "changed while it was being read" in captured.err
    assert "phase-2 gate" not in captured.out


@pytest.mark.parametrize("command", ["verify", "backfill"])
def test_absent_embedding_vec_column_is_reported_not_created(
    agent_dir, monkeypatch, capsys, command
):
    _drop_chunk_embedding_vec(agent_dir)
    _insert_chunk(agent_dir, "chunk", _pack([1.0]))
    before = _snapshot(agent_dir)

    rc = _kestrel(
        monkeypatch, "embeddings", command, "--data-dir", str(agent_dir),
        "--table", "document_chunks",
    )

    # Opening the database the default way would have run the startup
    # migration, which adds the column and copies the legacy vector into it.
    assert not _has_embedding_vec(agent_dir, "document_chunks")
    assert _snapshot(agent_dir) == before
    # The helper counts the legacy row as unbackfillable, so the equality
    # holds, but there is no column for a reader to query: the gate is not met.
    assert rc == cli_embeddings.EXIT_GATE_NOT_MET
    out = capsys.readouterr().out
    assert "embedding_vec column is ABSENT" in out
    assert "never creates it" in out
    assert "document_chunks.embedding_vec is missing" in out
    assert "phase-2 gate: NOT met" in out


@pytest.mark.parametrize("command", ["verify", "backfill"])
def test_absent_column_fails_the_gate_even_when_the_table_is_empty(
    agent_dir, monkeypatch, capsys, command
):
    # PostgreSQL defers creating embedding_vec until a legacy embedding
    # exists, so an empty table is exactly where the column can be missing.
    # A reader switched to it would fail, so the gate must not pass (#3405).
    _drop_chunk_embedding_vec(agent_dir)

    rc, payload = _embeddings_json(
        monkeypatch, capsys, command, agent_dir, "--table", "document_chunks"
    )

    assert rc == cli_embeddings.EXIT_GATE_NOT_MET
    assert payload["gate_met"] is False
    (entry,) = payload["tables"]
    assert entry["table"] == "document_chunks"
    assert entry["embedding_vec_present"] is False
    assert entry["total_rows"] == 0
    assert entry["rows_missing_embedding_vec"] == entry["rows_unbackfillable"] == 0
    assert entry["gate_met"] is False

    rc = _kestrel(
        monkeypatch, "embeddings", command, "--data-dir", str(agent_dir),
        "--table", "document_chunks",
    )

    assert rc == cli_embeddings.EXIT_GATE_NOT_MET
    out = capsys.readouterr().out
    assert "embedding_vec column is ABSENT" in out
    assert "gate: NOT met — document_chunks.embedding_vec is missing" in out
    assert "phase-2 gate: NOT met" in out
    assert not _has_embedding_vec(agent_dir, "document_chunks")


def test_absent_column_on_one_table_fails_only_that_table(
    agent_dir, monkeypatch, capsys
):
    item_vec = _pack([0.1, 0.2])
    _insert_saved_item(agent_dir, "legacy-only", embedding=item_vec)
    _drop_chunk_embedding_vec(agent_dir)

    rc, payload = _embeddings_json(monkeypatch, capsys, "backfill", agent_dir)

    # saved_items has its column and a clean backfill, so it meets the gate;
    # the empty document_chunks has no column, which fails the whole run.
    assert rc == cli_embeddings.EXIT_GATE_NOT_MET
    assert payload["gate_met"] is False
    saved = _table(payload, "saved_items")
    assert (saved["embedding_vec_present"], saved["rows_backfilled"]) == (True, 1)
    assert saved["gate_met"] is True
    chunks = _table(payload, "document_chunks")
    assert (chunks["embedding_vec_present"], chunks["total_rows"]) == (False, 0)
    assert chunks["gate_met"] is False

    rc, payload = _embeddings_json(
        monkeypatch, capsys, "verify", agent_dir, "--table", "saved_items"
    )
    assert rc == 0
    assert payload["gate_met"] is True


@pytest.mark.parametrize(
    ("present", "missing", "unbackfillable", "met"),
    [
        (True, 0, 0, True),
        (True, 2, 2, True),
        (True, 2, 1, False),
        (False, 0, 0, False),
        (False, 3, 3, False),
    ],
)
def test_gate_requires_the_column_and_every_missing_row_unbackfillable(
    present, missing, unbackfillable, met
):
    report = EmbeddingVecReport(
        table="saved_items",
        embedding_vec_present=present,
        total_rows=missing,
        rows_with_both=0,
        rows_missing_embedding_vec=missing,
        rows_embedding_vec_only=0,
        rows_without_any_embedding=0,
        rows_disagreeing=0,
        rows_backfilled=0,
        rows_unbackfillable=unbackfillable,
    )

    assert cli_embeddings._gate_met(report) is met


def test_text_report_prints_every_field_and_the_gate(agent_dir, monkeypatch, capsys):
    legacy = _pack([1.0, 2.0])
    _insert_saved_item(
        agent_dir, "disagree", embedding=legacy, embedding_vec=_pack([3.0, 4.0])
    )

    rc = _kestrel(monkeypatch, "embeddings", "verify", "--data-dir", str(agent_dir))

    assert rc == 0
    out = capsys.readouterr().out
    assert out.startswith("# embeddings verify (read-only)")
    for name in LEGACY_EMBEDDING_TABLES:
        assert f"\n{name}:\n" in out
    for field in _REPORT_FIELDS:
        if field not in ("table", "embedding_vec_present"):
            assert field in out
    assert "1 row(s) disagree" in out
    assert "phase-2 gate: met" in out


def test_text_report_names_the_unmet_gate(agent_dir, monkeypatch, capsys):
    _insert_saved_item(agent_dir, "legacy-only", embedding=_pack([1.0]))

    rc = _kestrel(
        monkeypatch, "embeddings", "verify", "--data-dir", str(agent_dir),
        "--table", "saved_items",
    )

    assert rc == cli_embeddings.EXIT_GATE_NOT_MET
    out = capsys.readouterr().out
    assert "rows_missing_embedding_vec (1) != rows_unbackfillable (0)" in out
    assert "`kestrel embeddings backfill` can copy the difference" in out
    assert "phase-2 gate: NOT met" in out


def test_missing_database_is_refused_and_not_created(tmp_path, monkeypatch, capsys):
    for name in ("KESTREL_DATABASE_URL", "DATABASE_URL", "KESTREL_DB_BACKEND"):
        monkeypatch.delenv(name, raising=False)

    rc = _kestrel(monkeypatch, "embeddings", "verify", "--data-dir", str(tmp_path))

    assert rc == 2
    assert "no database found" in capsys.readouterr().err
    assert not (tmp_path / "kestrel_prime.db").exists()


async def test_helper_refusal_exits_2(capsys):
    class _OtherBackend:
        backend_type = "mysql"

    rc = await cli_embeddings._embedding_vec(
        _OtherBackend(), command="verify", table="saved_items"
    )

    assert rc == 2
    assert "unsupported database backend" in capsys.readouterr().err


async def test_batch_size_reaches_the_helper_only_when_given(monkeypatch, capsys):
    from kestrel_sovereign.storage import embedding_vec_backfill

    calls = []

    async def spy(db, table, **kwargs):
        calls.append((table, kwargs))
        return EmbeddingVecReport(table, True, 0, 0, 0, 0, 0, 0, 0, 0)

    monkeypatch.setattr(embedding_vec_backfill, "backfill_embedding_vec", spy)

    assert await cli_embeddings._embedding_vec(
        object(), command="backfill", table="saved_items", batch_size=7
    ) == 0
    assert await cli_embeddings._embedding_vec(
        object(), command="backfill", table="document_chunks"
    ) == 0
    assert calls == [("saved_items", {"batch_size": 7}), ("document_chunks", {})]
    capsys.readouterr()


@pytest.fixture
def postgres_url(monkeypatch):
    url = os.environ.get("TEST_POSTGRES_URL")
    if not url:
        pytest.skip("TEST_POSTGRES_URL is not set")
    for name in ("DATABASE_URL", "KESTREL_DB_BACKEND", "KESTREL_DB_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KESTREL_DATABASE_URL", url)
    return url


def _on_postgres(url, work, *, initialize_schema):
    async def go():
        db = await AsyncDatabase.postgres(
            url,
            schema_initializer=(
                None if initialize_schema else cli_embeddings._leave_schema_unchanged
            ),
        )
        try:
            return await work(db)
        finally:
            await db.close()

    return asyncio.run(go())


async def _pg_chunk_state(db, file_hash):
    """``(embedding_vec typmod or None if absent, this test's rows)``."""
    column = await db.fetchone(
        "SELECT a.atttypmod FROM pg_attribute a "
        "WHERE a.attrelid = to_regclass('document_chunks') "
        "AND a.attname = 'embedding_vec' AND NOT a.attisdropped",
        (),
    )
    typmod = None if column is None else int(column[0])
    vec_expr = "NULL" if typmod is None else "embedding_vec::text"
    rows = await db.fetchall(
        f"SELECT embedding, {vec_expr} FROM document_chunks WHERE file_hash = ?",
        (file_hash,),
    )
    return typmod, [(bytes(legacy), vec) for legacy, vec in rows]


def test_postgres_verify_changes_neither_the_column_nor_a_row(
    postgres_url, monkeypatch, capsys
):
    file_hash = f"cli-embeddings-verify-{uuid4()}"

    async def insert_legacy_row(db):
        # The boot retires the legacy column once no row needs it (#3411).
        await restore_legacy_embedding_column(db, "document_chunks")
        typmod, _ = await _pg_chunk_state(db, file_hash)
        width = typmod if typmod is not None and typmod > 0 else 4
        await db.execute(
            "INSERT INTO document_chunks (file_hash, content, embedding) "
            "VALUES (?, ?, ?)",
            (file_hash, "chunk", _pack([0.5] * width)),
        )

    _on_postgres(postgres_url, insert_legacy_row, initialize_schema=True)
    try:
        before = _on_postgres(
            postgres_url,
            lambda db: _pg_chunk_state(db, file_hash),
            initialize_schema=False,
        )

        # No --data-dir: KESTREL_DATABASE_URL selects PostgreSQL.
        rc = _kestrel(
            monkeypatch, "embeddings", "verify", "--table", "document_chunks",
            "--json",
        )
        payload = json.loads(capsys.readouterr().out)

        after = _on_postgres(
            postgres_url,
            lambda db: _pg_chunk_state(db, file_hash),
            initialize_schema=False,
        )
        # Whether or not embedding_vec existed, verify neither created it nor
        # copied this row's legacy vector into it.
        assert after == before
        assert after[1][0][1] is None
        (entry,) = payload["tables"]
        assert entry["embedding_vec_present"] is (before[0] is not None)
        assert entry["rows_backfilled"] == 0
        assert entry["rows_missing_embedding_vec"] >= 1
        # This row can be copied once the column exists; the gate is not met.
        assert rc == cli_embeddings.EXIT_GATE_NOT_MET
    finally:
        async def delete_row(db):
            await db.execute(
                "DELETE FROM document_chunks WHERE file_hash = ?", (file_hash,)
            )

        # Opening with the default initializer while this legacy row exists
        # would let the startup migration create the column.
        _on_postgres(postgres_url, delete_row, initialize_schema=False)


@pytest.fixture
def private_pg_schema(postgres_url):
    """A schema of this test's own, dropped at teardown.

    :func:`_create_pg_chunks` builds ``document_chunks`` in it and points the
    CLI there. The shared ``document_chunks`` column is never altered, so no
    parallel test can observe it change.
    """
    schema = f"cli_embeddings_{uuid4().hex}"

    async def create_schema(db):
        await db.execute(f"CREATE SCHEMA {schema}", ())

    _on_postgres(postgres_url, create_schema, initialize_schema=False)
    try:
        yield schema
    finally:
        async def remove_schema(db):
            await db.execute(f"DROP SCHEMA {schema} CASCADE", ())

        _on_postgres(postgres_url, remove_schema, initialize_schema=False)


def _create_pg_chunks(url, schema, monkeypatch, *, embedding_vec):
    """Create ``<schema>.document_chunks`` and point the CLI at it.

    The CLI's ``KESTREL_DATABASE_URL`` carries ``search_path=<schema>``
    (asyncpg sends unknown DSN parameters as server settings), so the gate is
    checked against a table this test owns. With an ``embedding_vec`` column
    the path also names pgvector's schema, for the helper's ``::vector``
    cast, and the DDL qualifies the type: this connection's path is the xdist
    worker's own schema, which need not hold the extension (#3401).
    """

    async def create(db):
        vector_schema = await pgvector_schema(db) if embedding_vec else None
        vec_column = (
            f', embedding_vec "{vector_schema}".vector(2)' if embedding_vec else ""
        )
        await db.execute(
            f"CREATE TABLE {schema}.document_chunks ("
            "chunk_id SERIAL PRIMARY KEY, file_hash TEXT, content TEXT, "
            f"embedding BYTEA{vec_column})",
            (),
        )
        return vector_schema

    vector_schema = _on_postgres(url, create, initialize_schema=False)
    search_path = quoted_search_path(
        schema, *([vector_schema] if vector_schema else [])
    )
    monkeypatch.setenv("KESTREL_DATABASE_URL", with_search_path(url, search_path))


def test_cli_search_path_names_each_schema_once_and_quoted():
    # pgvector may already live in the schema named first, e.g. a worker's own.
    assert quoted_search_path("cli_x", "public", "cli_x") == '"cli_x","public"'
    assert quoted_search_path('odd"name') == '"odd""name"'


async def _pg_private_chunks_have_embedding_vec(db, schema):
    column = await db.fetchone(
        "SELECT 1 FROM pg_attribute "
        f"WHERE attrelid = to_regclass('{schema}.document_chunks') "
        "AND attname = 'embedding_vec' AND NOT attisdropped",
        (),
    )
    return column is not None


@pytest.mark.parametrize("command", ["verify", "backfill"])
def test_postgres_absent_embedding_vec_on_an_empty_table_fails_the_gate(
    postgres_url, private_pg_schema, monkeypatch, capsys, command
):
    # The startup migration defers the column until a legacy embedding
    # exists, so an empty PostgreSQL table has none. A reader switched to it
    # would fail, so the gate must not pass (#3405).
    _create_pg_chunks(
        postgres_url, private_pg_schema, monkeypatch, embedding_vec=False
    )

    rc = _kestrel(
        monkeypatch, "embeddings", command, "--table", "document_chunks", "--json"
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == cli_embeddings.EXIT_GATE_NOT_MET
    assert payload["gate_met"] is False
    (entry,) = payload["tables"]
    assert entry["table"] == "document_chunks"
    assert entry["embedding_vec_present"] is False
    assert entry["total_rows"] == 0
    assert entry["gate_met"] is False
    assert not _on_postgres(
        postgres_url,
        lambda db: _pg_private_chunks_have_embedding_vec(db, private_pg_schema),
        initialize_schema=False,
    )


def test_postgres_clean_backfill_with_the_column_meets_the_gate(
    postgres_url, private_pg_schema, monkeypatch, capsys
):
    _create_pg_chunks(
        postgres_url, private_pg_schema, monkeypatch, embedding_vec=True
    )
    legacy = _pack([0.5, -0.25])

    async def insert(db):
        await db.execute(
            f"INSERT INTO {private_pg_schema}.document_chunks "
            "(file_hash, content, embedding) VALUES (?, ?, ?)",
            ("doc", "chunk", legacy),
        )

    _on_postgres(postgres_url, insert, initialize_schema=False)

    rc = _kestrel(
        monkeypatch, "embeddings", "backfill", "--table", "document_chunks",
        "--json",
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["gate_met"] is True
    (entry,) = payload["tables"]
    assert entry["embedding_vec_present"] is True
    assert (entry["total_rows"], entry["rows_backfilled"]) == (1, 1)
    assert entry["rows_missing_embedding_vec"] == 0
    assert entry["gate_met"] is True

    rc = _kestrel(
        monkeypatch, "embeddings", "verify", "--table", "document_chunks", "--json"
    )
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["tables"][0]["rows_with_both"] == 1


def _create_pg_profiled_chunks(url, schema, monkeypatch):
    """``<schema>.document_chunks`` holding stored vectors on and off
    ``_ACTIVE``, and an ``agent_metadata`` recording ``_ACTIVE`` (#3420).

    Points the CLI at the schema like :func:`_create_pg_chunks`, and returns
    the ``search_path`` URL for opening it directly.
    """

    async def create(db):
        vector_schema = await pgvector_schema(db)
        await db.execute(
            f"CREATE TABLE {schema}.document_chunks ("
            "chunk_id SERIAL PRIMARY KEY, file_hash TEXT, content TEXT, "
            f'embedding BYTEA, embedding_vec "{vector_schema}".vector(2), '
            "embedding_profile_id TEXT)",
            (),
        )
        await db.execute(
            f"CREATE TABLE {schema}.agent_metadata ("
            "agent_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, "
            "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
            "PRIMARY KEY (agent_id, key))",
            (),
        )
        # Untyped literals take the column's type, so no ``vector`` cast is
        # needed on a path that may not name pgvector's schema.
        await db.execute(
            f"INSERT INTO {schema}.document_chunks "
            "(file_hash, content, embedding_vec, embedding_profile_id) VALUES "
            f"('doc', 'on', '[1,2]', '{_ACTIVE}'), "
            f"('doc', 'reindexed away', '[3,4]', '{_ROUTE_SCOPED}'), "
            "('doc', 'never embedded', NULL, NULL)",
            (),
        )
        await db.execute(
            f"INSERT INTO {schema}.agent_metadata (agent_id, key, value) "
            "VALUES (?, ?, ?)",
            (
                "did:test:agent",
                ACTIVE_EMBEDDING_PROFILE_KEY,
                json.dumps({"profile_id": _ACTIVE, "dim": 2}),
            ),
        )
        return vector_schema

    vector_schema = _on_postgres(url, create, initialize_schema=False)
    scoped_url = with_search_path(url, quoted_search_path(schema, vector_schema))
    monkeypatch.setenv("KESTREL_DATABASE_URL", scoped_url)
    return scoped_url


async def _pg_profiled_chunks(db):
    return await db.fetchall(
        "SELECT chunk_id, embedding_vec::text, embedding_profile_id "
        "FROM document_chunks ORDER BY chunk_id",
        (),
    )


def test_postgres_verify_counts_vectors_off_the_recorded_profile(
    postgres_url, private_pg_schema, monkeypatch, capsys
):
    _create_pg_profiled_chunks(postgres_url, private_pg_schema, monkeypatch)

    rc = _kestrel(
        monkeypatch, "embeddings", "verify", "--table", "document_chunks", "--json"
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    active = payload["active_embedding_profile"]
    assert (active["status"], active["profile_id"]) == ("recorded", _ACTIVE)
    assert active["tables"] == {
        "document_chunks": {
            "vectors_on_active_profile": 1, "vectors_off_active_profile": 1,
        },
    }
    assert active["vectors_off_active_profile"] == 1


def test_postgres_reindex_refuses_a_target_other_than_the_recorded_profile(
    postgres_url, private_pg_schema, monkeypatch, capsys
):
    scoped_url = _create_pg_profiled_chunks(
        postgres_url, private_pg_schema, monkeypatch
    )
    monkeypatch.setattr(cli_embeddings, "_resolve_column_dim", lambda: 2)

    class _RouteScoped:
        embedding_dim = 2
        embedded = []

        def current_profile_id(self):
            return _ROUTE_SCOPED

        def describe(self):
            return None

        async def aembed_batch(self, texts):
            self.embedded.extend(texts)
            return [[1.0, 1.0] for _ in texts]

    service = _RouteScoped()

    async def reindex(db):
        before = await _pg_profiled_chunks(db)
        rc = await cli_embeddings._reindex(
            db, table="document_chunks", agent_id=None, batch=10,
            rate_limit=0.0, dry_run=False, apply=True,
            embedding_service=service, target_profile_id=_ROUTE_SCOPED,
            target_dim=2,
        )
        return rc, before, await _pg_profiled_chunks(db)

    rc, before, after = _on_postgres(scoped_url, reindex, initialize_schema=False)

    assert rc == 2
    assert f"records as active, {_ACTIVE}" in capsys.readouterr().err
    assert service.embedded == []
    assert after == before


def _parse(*argv):
    parser = argparse.ArgumentParser()
    cli_embeddings.add_embeddings_subparser(parser.add_subparsers(dest="command"))
    return parser.parse_args(["embeddings", *argv])


def test_parser_defaults_and_batch_size_validation(capsys):
    verify = _parse("verify")
    assert (verify.table, verify.as_json) == ("all", False)
    assert not hasattr(verify, "batch_size")

    backfill = _parse("backfill", "--table", "document_chunks", "--batch-size", "7")
    assert (backfill.table, backfill.batch_size) == ("document_chunks", 7)
    # Unset means the helper's own default, which the help text names.
    assert _parse("backfill").batch_size is None

    for bad in ("0", "-1", "many"):
        with pytest.raises(SystemExit) as exc:
            _parse("backfill", "--batch-size", bad)
        assert exc.value.code == 2
    with pytest.raises(SystemExit):
        _parse("verify", "--table", "conversation_history")
    capsys.readouterr()


def test_cli_constants_match_the_helper():
    assert cli_embeddings._LEGACY_EMBEDDING_TABLES == tuple(LEGACY_EMBEDDING_TABLES)
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    cli_embeddings.add_embeddings_subparser(subparsers)
    (embed_sub,) = [
        action for action in subparsers.choices["embeddings"]._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    (batch_size,) = [
        action for action in embed_sub.choices["backfill"]._actions
        if action.dest == "batch_size"
    ]
    assert f"(default: {DEFAULT_BATCH_SIZE})" in batch_size.help
    # 1 is an uncaught exception and 2 a refusal; the gate needs its own code.
    assert cli_embeddings.EXIT_GATE_NOT_MET not in (0, 1, 2)
