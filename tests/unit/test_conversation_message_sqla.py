"""Tests for the ``conversation_history`` SQLAlchemy entity + greenfield
``embedding_vec`` migration.

Greenfield = no pre-existing ``embedding`` column to migrate from, so
the migration just adds ``embedding_vec`` (and HNSW on PG) at the
configured dim. No dim sniffing, no backfill.

Covers:

- :class:`ConversationMessage` ORM column mapping (table name,
  ``embedding`` attr → ``embedding_vec`` SQL column, PortableVector
  dimension).
- :func:`build_conversation_message_spec` validates dimension, sets
  ``agent_id`` as the only required filter key, exposes ``role`` and
  ``deleted_at`` as optional WHERE filters.
- :data:`CONVERSATION_MESSAGE_EMBEDDING_DIM` honours the
  ``KESTREL_EMBEDDING_DIM`` env override + falls back to 768 on
  garbage input.
- :func:`migrate_conversation_history_add_embedding_vec` is idempotent
  on PG (skip if column present, skip if table missing), creates
  extension + ALTER + HNSW in the right order, runs in a transaction,
  and SQLite-side adds the BLOB column with no backfill.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sovereign.storage.sqla import (
    CONVERSATION_MESSAGE_EMBEDDING_DIM,
    ConversationMessage,
    build_conversation_message_spec,
)
from kestrel_sovereign.storage.sqla.conversation_message import (
    resolve_embedding_dim,
)
from kestrel_sovereign.storage.sqla.migrations import (
    migrate_conversation_history_add_embedding_vec,
)
from kestrel_sovereign.storage.sqla.types import PortableVector


# ----------------------------------------------------------------- ORM wiring


def test_conversation_message_table_name():
    assert ConversationMessage.__tablename__ == "conversation_history"


def test_conversation_message_embedding_uses_portable_vector():
    """The ORM attribute ``embedding`` is wired to the
    ``embedding_vec`` SQL column (greenfield — there's no legacy
    ``embedding`` column to keep in parallel here, but we keep the
    attribute name symmetric with SavedItem / DocumentChunk).
    """
    col = ConversationMessage.__table__.columns["embedding_vec"]
    assert isinstance(col.type, PortableVector)
    assert col.type.dimension == CONVERSATION_MESSAGE_EMBEDDING_DIM
    # Python attribute name is ``embedding``, SQL column is ``embedding_vec``.
    assert ConversationMessage.embedding.expression.name == "embedding_vec"


def test_conversation_message_maps_expected_columns():
    """The ORM exposes the columns the retriever needs for filtering
    (``agent_id``, ``role``, ``deleted_at``) plus identity (``id``)
    and the embedding column. Other columns may be present but these
    are the contract.
    """
    cols = ConversationMessage.__table__.columns
    for required in ("id", "agent_id", "role", "deleted_at", "embedding_vec"):
        assert required in cols, f"missing ORM column {required!r}"


# ----------------------------------------------------------------- spec contract


def test_build_spec_rejects_non_positive_dimension():
    with pytest.raises(ValueError, match="dimension"):
        build_conversation_message_spec(0)
    with pytest.raises(ValueError, match="dimension"):
        build_conversation_message_spec(-1)


def test_build_spec_requires_agent_id():
    """Every retrieval is agent-scoped. The spec MUST enforce that —
    a buggy caller forgetting agent_id would silently search every
    agent's history without it."""
    spec = build_conversation_message_spec(768)
    assert spec.required_filter_keys == ("agent_id",)


def test_build_spec_exposes_optional_filters():
    """Retriever pins ``role='assistant'`` and
    ``deleted_at=None`` (which SQLAlchemy renders as IS NULL).
    Both must be present in ``filter_columns`` so the WHERE clause
    actually gets the predicate."""
    spec = build_conversation_message_spec(768)
    assert "agent_id" in spec.filter_columns
    assert "role" in spec.filter_columns
    assert "deleted_at" in spec.filter_columns


def test_build_spec_has_no_tenant_filter_key():
    """``conversation_history`` is single-tenant per agent in
    sovereign-core; frinz's multi-tenant scoping is above this
    layer. A tenant_id_filter_key here would require frinz to
    plumb it through, which it doesn't."""
    spec = build_conversation_message_spec(768)
    assert spec.tenant_id_filter_key is None


def test_build_spec_passes_through_runtime_dimension():
    """Different embedding models → different dims. The retriever
    constructs the spec per-query from len(query_embedding), so the
    spec must honour what it's given (not pin to the default)."""
    for dim in (768, 1024, 1536):
        spec = build_conversation_message_spec(dim)
        assert spec.dimension == dim


# ----------------------------------------------------------------- dim resolution


def test_default_dim_is_768_without_env():
    """Default Ollama nomic-embed-text dim. Hardcoded so a missing
    env var doesn't silently shift to whatever a future model picks.

    Assert via the resolver with an empty env mapping rather than the
    import-time constant — operators that run the suite in a shell
    pre-configured for a 1024 / 1536 model would otherwise see this
    fail even though production behaviour is correct. (Caught by
    codex review.)
    """
    assert resolve_embedding_dim({}) == 768


def test_resolve_embedding_dim_honours_env_override():
    """Operators that switch to mxbai-embed-large / OpenAI ada-002
    set KESTREL_EMBEDDING_DIM before first boot. Pass the override
    via the ``env`` arg instead of monkeypatch + importlib.reload —
    reloading the module re-registers ConversationMessage against
    SovereignBase and crashes with ``Table is already defined``.
    """
    assert resolve_embedding_dim({"KESTREL_EMBEDDING_DIM": "1024"}) == 1024
    assert resolve_embedding_dim({"KESTREL_EMBEDDING_DIM": "1536"}) == 1536


def test_resolve_embedding_dim_no_env_returns_default():
    """No env var → 768. (Matches the constant captured at import time.)"""
    assert resolve_embedding_dim({}) == 768


def test_resolve_embedding_dim_empty_string_returns_default():
    """``KESTREL_EMBEDDING_DIM=`` (set but empty) is treated as unset."""
    assert resolve_embedding_dim({"KESTREL_EMBEDDING_DIM": ""}) == 768


def test_resolve_embedding_dim_rejects_garbage():
    """A typo like ``seven-sixty-eight`` must fall back to 768, not
    crash boot."""
    assert resolve_embedding_dim(
        {"KESTREL_EMBEDDING_DIM": "seven-sixty-eight"}
    ) == 768


def test_resolve_embedding_dim_rejects_non_positive():
    """Zero / negative dims would crash the spec or create a nonsense
    column; fall back to the default."""
    assert resolve_embedding_dim({"KESTREL_EMBEDDING_DIM": "0"}) == 768
    assert resolve_embedding_dim({"KESTREL_EMBEDDING_DIM": "-1"}) == 768


# ----------------------------------------------------------------- migration


def _fake_db_with_fetchall(backend_type: str, fetchall_returns: list) -> MagicMock:
    """Stub ``AsyncDatabase`` mirroring the saved_items / document_chunks
    test stub. Returns a MagicMock with ``backend_type``, a queued
    AsyncMock ``fetchall``, and a no-op transaction context manager.
    """
    db = MagicMock()
    db.backend_type = backend_type
    db.fetchall = AsyncMock(side_effect=fetchall_returns)
    db.execute = AsyncMock()

    class _TxCM:
        async def __aenter__(self_inner):
            return self_inner

        async def __aexit__(self_inner, *a):
            return False

    db.transaction = MagicMock(return_value=_TxCM())
    return db


@pytest.mark.asyncio
async def test_migration_noop_when_pg_column_already_exists():
    """Idempotency: the migration must short-circuit (no DDL) when
    ``embedding_vec`` already exists on PG."""
    db = _fake_db_with_fetchall("postgres", [[(1,)]])
    await migrate_conversation_history_add_embedding_vec(db)
    db.execute.assert_not_called()


@pytest.mark.asyncio
async def test_migration_noop_when_pg_table_missing():
    """Fresh DB before ``_init_schema`` has run: bail rather than
    crash. ``conversation_history`` is created by CORE_SCHEMA on the
    same boot, so this branch only fires if migrations somehow run
    out of order."""
    db = _fake_db_with_fetchall(
        "postgres",
        [
            [],   # embedding_vec column probe → absent
            [],   # information_schema.tables probe → table missing
        ],
    )
    await migrate_conversation_history_add_embedding_vec(db)
    db.execute.assert_not_called()


@pytest.mark.asyncio
async def test_migration_pg_creates_extension_before_alter():
    """Regression from the saved_items / document_chunks PRs:
    ``CREATE EXTENSION IF NOT EXISTS vector`` MUST come before
    ``ALTER TABLE ... vector(N)``. The reverse order fails on fresh
    PG with ``type "vector" does not exist`` and the migration is
    caught by the non-fatal try/except at the call site — leaving
    the column never created."""
    db = _fake_db_with_fetchall(
        "postgres",
        [
            [],         # embedding_vec column probe → absent
            [(1,)],     # table-exists probe → table present
        ],
    )
    await migrate_conversation_history_add_embedding_vec(db)

    sql_calls = [c.args[0] for c in db.execute.call_args_list]
    create_ext_idx = next(
        (i for i, q in enumerate(sql_calls) if "CREATE EXTENSION" in q), None
    )
    alter_idx = next(
        (i for i, q in enumerate(sql_calls) if "ADD COLUMN embedding_vec vector" in q),
        None,
    )
    assert create_ext_idx is not None, "expected CREATE EXTENSION"
    assert alter_idx is not None, "expected ALTER TABLE ADD COLUMN"
    assert create_ext_idx < alter_idx


@pytest.mark.asyncio
async def test_migration_pg_creates_hnsw_index():
    """The retriever's kNN performance depends on the HNSW cosine
    index. If the migration silently skipped it (e.g. ran the ALTER
    outside the transaction and then crashed), every query would
    fall back to a full sequential scan."""
    db = _fake_db_with_fetchall(
        "postgres",
        [
            [],         # embedding_vec absent
            [(1,)],     # table exists
        ],
    )
    await migrate_conversation_history_add_embedding_vec(db)
    sql_calls = [c.args[0] for c in db.execute.call_args_list]
    assert any(
        "hnsw" in q.lower() and "embedding_vec" in q and "vector_cosine_ops" in q
        for q in sql_calls
    ), f"expected HNSW index in {sql_calls!r}"


@pytest.mark.asyncio
async def test_migration_pg_uses_configured_dim():
    """The ALTER must reference the dim resolved from
    ``CONVERSATION_MESSAGE_EMBEDDING_DIM`` (768 by default)."""
    db = _fake_db_with_fetchall(
        "postgres",
        [
            [],
            [(1,)],
        ],
    )
    await migrate_conversation_history_add_embedding_vec(db)
    sql_calls = [c.args[0] for c in db.execute.call_args_list]
    assert any(
        f"ADD COLUMN embedding_vec vector({CONVERSATION_MESSAGE_EMBEDDING_DIM})" in q
        for q in sql_calls
    )


@pytest.mark.asyncio
async def test_migration_wraps_in_transaction_pg():
    """Partial failure must roll back so the next boot can retry from
    a clean state."""
    db = _fake_db_with_fetchall(
        "postgres",
        [
            [],
            [(1,)],
        ],
    )
    await migrate_conversation_history_add_embedding_vec(db)
    db.transaction.assert_called_once()


@pytest.mark.asyncio
async def test_migration_sqlite_adds_blob_no_backfill():
    """SQLite path: BLOB column, no UPDATE backfill (greenfield —
    nothing to copy from)."""
    db = _fake_db_with_fetchall(
        "sqlite",
        [
            [],       # pragma_table_info probe → column absent
            [(1,)],   # sqlite_master probe → table present
        ],
    )
    await migrate_conversation_history_add_embedding_vec(db)
    sql_calls = [c.args[0] for c in db.execute.call_args_list]
    assert any("ADD COLUMN embedding_vec BLOB" in q for q in sql_calls)
    # Crucially: NO bytes-copy UPDATE, since there's no legacy column.
    assert not any(
        "UPDATE conversation_history SET embedding_vec = embedding" in q
        for q in sql_calls
    )


@pytest.mark.asyncio
async def test_migration_sqlite_noop_when_column_present():
    db = _fake_db_with_fetchall("sqlite", [[("embedding_vec",)]])
    await migrate_conversation_history_add_embedding_vec(db)
    db.execute.assert_not_called()


@pytest.mark.asyncio
async def test_migration_sqlite_noop_when_table_missing():
    db = _fake_db_with_fetchall(
        "sqlite",
        [
            [],   # column probe → absent
            [],   # sqlite_master probe → no table
        ],
    )
    await migrate_conversation_history_add_embedding_vec(db)
    db.execute.assert_not_called()


@pytest.mark.asyncio
async def test_migration_skips_unknown_dialect():
    db = _fake_db_with_fetchall("mysql", [])
    await migrate_conversation_history_add_embedding_vec(db)
    db.execute.assert_not_called()
    db.fetchall.assert_not_called()
