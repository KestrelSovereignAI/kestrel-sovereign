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
import os
import sqlite3
import struct
import sys
from dataclasses import fields
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import pytest

from kestrel_sovereign import cli, cli_embeddings
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.embedding_vec_backfill import (
    DEFAULT_BATCH_SIZE,
    LEGACY_EMBEDDING_TABLES,
    EmbeddingVecReport,
)

_REPORT_FIELDS = [field.name for field in fields(EmbeddingVecReport)]


def _pack(values):
    return struct.pack(f"<{len(values)}f", *values)


@pytest.fixture
def agent_dir(tmp_path, monkeypatch):
    """An agent data dir whose ``kestrel_prime.db`` has the full schema."""
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
        await db.close()

    asyncio.run(create())
    return data_dir


def _connect(data_dir):
    return sqlite3.connect(str(data_dir / "kestrel_prime.db"))


def _insert_saved_item(data_dir, item_id, embedding=None, embedding_vec=None):
    with _connect(data_dir) as conn:
        conn.execute(
            "INSERT INTO saved_items (id, agent_id, item_type, name, content, "
            "embedding, embedding_vec) "
            "VALUES (?, 'did:test:agent', 'stash', ?, 'c', ?, ?)",
            (item_id, item_id, embedding, embedding_vec),
        )


def _insert_chunk(data_dir, content, embedding):
    with _connect(data_dir) as conn:
        conn.execute(
            "INSERT INTO document_chunks (file_hash, content, embedding) "
            "VALUES (?, ?, ?)",
            ("doc", content, embedding),
        )


def _drop_chunk_embedding_vec(data_dir):
    with _connect(data_dir) as conn:
        conn.execute("ALTER TABLE document_chunks DROP COLUMN embedding_vec")


def _columns(data_dir, table, id_col, row_id):
    with _connect(data_dir) as conn:
        return conn.execute(
            f"SELECT embedding, embedding_vec FROM {table} WHERE {id_col} = ?",
            (row_id,),
        ).fetchone()


def _has_embedding_vec(data_dir, table):
    with _connect(data_dir) as conn:
        return bool(conn.execute(
            f"SELECT 1 FROM pragma_table_info('{table}') "
            "WHERE name = 'embedding_vec'"
        ).fetchall())


def _snapshot(data_dir):
    """Every row of both tables plus the whole schema."""
    with _connect(data_dir) as conn:
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
        return EmbeddingVecReport(table, True, 0, 0, 0, 0, 0, 0, 0)

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
def private_pg_schema(postgres_url, monkeypatch):
    """A private schema the CLI resolves ``document_chunks`` in first.

    The CLI's ``KESTREL_DATABASE_URL`` carries ``search_path=<schema>,public``
    (asyncpg sends unknown DSN parameters as server settings), so the gate is
    checked against a table this test owns. ``public`` stays on the path for
    the pgvector type. The shared ``document_chunks`` column is never altered,
    so no parallel test can observe it change.
    """
    schema = f"cli_embeddings_{uuid4().hex}"

    async def create_schema(db):
        await db.execute(f"CREATE SCHEMA {schema}", ())

    _on_postgres(postgres_url, create_schema, initialize_schema=False)
    parts = urlsplit(postgres_url)
    query = [(k, v) for k, v in parse_qsl(parts.query) if k != "search_path"]
    query.append(("search_path", f"{schema},public"))
    monkeypatch.setenv(
        "KESTREL_DATABASE_URL", urlunsplit(parts._replace(query=urlencode(query)))
    )
    try:
        yield schema
    finally:
        async def remove_schema(db):
            await db.execute(f"DROP SCHEMA {schema} CASCADE", ())

        _on_postgres(postgres_url, remove_schema, initialize_schema=False)


def _create_pg_chunks(url, schema, *, embedding_vec):
    vec_column = ", embedding_vec vector(2)" if embedding_vec else ""

    async def create(db):
        if embedding_vec:
            await db.execute("CREATE EXTENSION IF NOT EXISTS vector", ())
        await db.execute(
            f"CREATE TABLE {schema}.document_chunks ("
            "chunk_id SERIAL PRIMARY KEY, file_hash TEXT, content TEXT, "
            f"embedding BYTEA{vec_column})",
            (),
        )

    _on_postgres(url, create, initialize_schema=False)


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
    _create_pg_chunks(postgres_url, private_pg_schema, embedding_vec=False)

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
    _create_pg_chunks(postgres_url, private_pg_schema, embedding_vec=True)
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
