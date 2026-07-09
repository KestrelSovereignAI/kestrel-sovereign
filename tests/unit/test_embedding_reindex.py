"""Unit tests for ``kestrel embeddings reindex`` (#2289).

Seed a real SQLite DB with mixed-profile rows and verify:

- every stale row is re-embedded and stamped to the target profile;
- an interrupted run loses nothing and a re-run completes it (resume);
- dry-run reports honest per-table counts and touches nothing;
- reindex refuses when no embedding provider resolves, when
  ``embedding_route = "none"``, and on a dimension mismatch.
"""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path
from typing import List, Optional

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.embedding_reindex import (
    REINDEX_TABLES,
    EmbeddingReindexer,
)

TARGET = "targetprofile"
OTHER = "oldprofileee"
DIM = 4


# --------------------------------------------------------------- fake service


class FakeEmbeddingService:
    """Deterministic ``DIM``-length embeddings keyed off text length.

    ``crash_on_call`` makes ``aembed_batch`` raise ``KeyboardInterrupt``
    (a BaseException the reindexer does NOT catch) on the Nth call, to
    model a hard interrupt mid-run.
    """

    def __init__(self, *, profile_id: str = TARGET, dim: int = DIM, crash_on_call: Optional[int] = None):
        self.embedding_dim = dim
        self._profile_id = profile_id
        self._dim = dim
        self.crash_on_call = crash_on_call
        self.calls = 0

    async def aembed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        self.calls += 1
        if self.crash_on_call is not None and self.calls >= self.crash_on_call:
            raise KeyboardInterrupt("simulated interrupt")
        out: List[Optional[List[float]]] = []
        for t in texts:
            if not t:
                out.append(None)
                continue
            base = float(len(t) % 7) + 1.0
            out.append([base + i for i in range(self._dim)])
        return out

    def current_profile_id(self) -> str:
        return self._profile_id

    def describe(self):  # registry upsert is best-effort; None = skip
        return None


# ----------------------------------------------------------------- fixtures


@pytest.fixture
async def db():
    with tempfile.TemporaryDirectory() as tmp:
        database = await AsyncDatabase.sqlite(str(Path(tmp) / "reindex.db"))
        yield database
        await database.close()


async def _seed(database: AsyncDatabase) -> None:
    """Seed mixed-profile rows across all three embedded tables."""
    # conversation_history: two stale (OTHER + NULL), one already target.
    await database.execute_commit(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, embedding_profile_id) VALUES (?, ?, ?, ?)",
        ("a1", "assistant", "hello world", OTHER),
    )
    await database.execute_commit(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, embedding_profile_id) VALUES (?, ?, ?, ?)",
        ("a1", "user", "another message", None),
    )
    await database.execute_commit(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, embedding_vec, embedding_profile_id) "
        "VALUES (?, ?, ?, ?, ?)",
        ("a1", "assistant", "already done", struct.pack("<4f", 1, 2, 3, 4), TARGET),
    )
    # A different agent's stale row (for agent-scope tests).
    await database.execute_commit(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, embedding_profile_id) VALUES (?, ?, ?, ?)",
        ("a2", "assistant", "other agent row", OTHER),
    )

    # saved_items: one stale with summary, one stale content-only.
    await database.execute_commit(
        "INSERT INTO saved_items "
        "(id, agent_id, item_type, name, summary, content, embedding_profile_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("s1", "a1", "note", "Note 1", "a short summary", "long content", OTHER),
    )
    await database.execute_commit(
        "INSERT INTO saved_items "
        "(id, agent_id, item_type, name, summary, content, embedding_profile_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("s2", "a1", "note", "Note 2", None, "content only body", None),
    )

    # document_chunks: two stale (global — no agent_id).
    await database.execute_commit(
        "INSERT INTO document_chunks (file_hash, content, embedding_profile_id) "
        "VALUES (?, ?, ?)",
        ("h1", "chunk one text", OTHER),
    )
    await database.execute_commit(
        "INSERT INTO document_chunks (file_hash, content, embedding_profile_id) "
        "VALUES (?, ?, ?)",
        ("h2", "chunk two text", None),
    )


async def _profiles(database: AsyncDatabase, table: str, id_col: str):
    rows = await database.fetchall(
        f"SELECT {id_col}, embedding_profile_id, embedding_vec FROM {table} "
        f"ORDER BY {id_col}",
        (),
    )
    return rows


# ------------------------------------------------------------------- tests


@pytest.mark.asyncio
async def test_reindex_stamps_all_rows_to_target(db):
    await _seed(db)
    svc = FakeEmbeddingService()
    reindexer = EmbeddingReindexer(db, svc, TARGET, column_dim=DIM, batch_size=50)

    for table in REINDEX_TABLES:
        await reindexer.reindex_table(table)

    # Every conversation row now carries the target profile + a vector.
    ch = await _profiles(db, "conversation_history", "id")
    assert all(r[1] == TARGET for r in ch), ch
    assert all(r[2] is not None for r in ch), ch

    si = await _profiles(db, "saved_items", "id")
    assert all(r[1] == TARGET for r in si), si
    assert all(r[2] is not None for r in si), si

    dc = await _profiles(db, "document_chunks", "chunk_id")
    assert all(r[1] == TARGET for r in dc), dc
    assert all(r[2] is not None for r in dc), dc

    # Nothing stale remains.
    counts = await reindexer.count_all_stale()
    assert sum(counts.values()) == 0


@pytest.mark.asyncio
async def test_reindex_is_idempotent(db):
    await _seed(db)
    svc = FakeEmbeddingService()
    reindexer = EmbeddingReindexer(db, svc, TARGET, column_dim=DIM, batch_size=50)
    for table in REINDEX_TABLES:
        await reindexer.reindex_table(table)

    # Second pass re-embeds nothing.
    total = 0
    for table in REINDEX_TABLES:
        stats = await reindexer.reindex_table(table)
        total += stats.reembedded
    assert total == 0


@pytest.mark.asyncio
async def test_dry_run_counts_match_and_touch_nothing(db):
    await _seed(db)
    svc = FakeEmbeddingService()
    reindexer = EmbeddingReindexer(db, svc, TARGET, column_dim=DIM, batch_size=50)

    counts = await reindexer.count_all_stale()
    # conv: 2x a1 stale + 1x a2 stale = 3; saved_items: 2; chunks: 2.
    assert counts["conversation_history"] == 3
    assert counts["saved_items"] == 2
    assert counts["document_chunks"] == 2

    # The already-target conversation row is not counted.
    assert svc.calls == 0  # counting must not embed anything

    # Agent-scoped count excludes the other agent's row.
    scoped = await reindexer.count_stale("conversation_history", agent_id="a1")
    assert scoped == 2


@pytest.mark.asyncio
async def test_resume_after_interrupt_loses_nothing(db):
    await _seed(db)
    # batch_size=1 so each row is its own commit; crash on the 2nd batch.
    crashing = FakeEmbeddingService(crash_on_call=2)
    reindexer = EmbeddingReindexer(db, crashing, TARGET, column_dim=DIM, batch_size=1)

    with pytest.raises(KeyboardInterrupt):
        await reindexer.reindex_table("conversation_history")

    # Exactly one row committed before the interrupt.
    done = await db.fetchall(
        "SELECT COUNT(*) FROM conversation_history WHERE embedding_profile_id = ?",
        (TARGET,),
    )
    # 1 pre-seeded target row + 1 freshly committed = 2.
    assert int(done[0][0]) == 2

    # Re-run with a healthy service finishes the remaining stale rows.
    healthy = EmbeddingReindexer(
        db, FakeEmbeddingService(), TARGET, column_dim=DIM, batch_size=50
    )
    for table in REINDEX_TABLES:
        await healthy.reindex_table(table)

    counts = await healthy.count_all_stale()
    assert sum(counts.values()) == 0


@pytest.mark.asyncio
async def test_empty_source_text_is_skipped_not_stamped(db):
    # A row with whitespace-only content can't be embedded — it must stay
    # NULL (never falsely stamped as target) and not loop forever.
    await db.execute_commit(
        "INSERT INTO document_chunks (file_hash, content, embedding_profile_id) "
        "VALUES (?, ?, ?)",
        ("blank", "   ", OTHER),
    )
    svc = FakeEmbeddingService()
    reindexer = EmbeddingReindexer(db, svc, TARGET, column_dim=DIM, batch_size=10)
    stats = await reindexer.reindex_table("document_chunks")
    assert stats.reembedded == 0
    assert stats.skipped_empty == 1


# --------------------------------------------------- CLI refusal / dim guard


@pytest.mark.asyncio
async def test_reindex_refuses_when_no_provider_resolves(db, capsys):
    from kestrel_sovereign import cli_embeddings

    class NoEmbedLLM:
        def get_embedding_route(self):
            return None

        def get_embedding_service(self):
            return None

    rc = await cli_embeddings._reindex(
        db,
        table=None,
        agent_id=None,
        batch=10,
        rate_limit=0.0,
        dry_run=False,
        apply=True,
        llm_service=NoEmbedLLM(),
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "no embedding-capable provider" in err


@pytest.mark.asyncio
async def test_reindex_refuses_when_route_is_none(db, capsys):
    from kestrel_sovereign import cli_embeddings

    class NoneRouteLLM:
        def get_embedding_route(self):
            return "none"

        def get_embedding_service(self):
            raise AssertionError("must refuse before resolving a service")

    rc = await cli_embeddings._reindex(
        db,
        table=None,
        agent_id=None,
        batch=10,
        rate_limit=0.0,
        dry_run=False,
        apply=True,
        llm_service=NoneRouteLLM(),
    )
    assert rc == 2
    assert '"none"' in capsys.readouterr().err


@pytest.mark.asyncio
async def test_reindex_refuses_on_dim_mismatch(db, capsys, monkeypatch):
    from kestrel_sovereign import cli_embeddings

    # Column width 768 (default), but the resolved provider embeds at 1536.
    monkeypatch.setattr(cli_embeddings, "_resolve_column_dim", lambda: 768)
    svc = FakeEmbeddingService(dim=1536)

    rc = await cli_embeddings._reindex(
        db,
        table=None,
        agent_id=None,
        batch=10,
        rate_limit=0.0,
        dry_run=False,
        apply=True,
        embedding_service=svc,
        target_profile_id=TARGET,
        target_dim=1536,
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "does not match the vector-column width" in err
    assert "KESTREL_EMBEDDING_DIM=1536" in err


@pytest.mark.asyncio
async def test_reindex_dry_run_reports_and_applies(db, capsys, monkeypatch):
    from kestrel_sovereign import cli_embeddings

    await _seed(db)
    monkeypatch.setattr(cli_embeddings, "_resolve_column_dim", lambda: DIM)
    svc = FakeEmbeddingService()

    # Dry-run (apply=False) must not touch rows.
    rc = await cli_embeddings._reindex(
        db,
        table=None,
        agent_id=None,
        batch=10,
        rate_limit=0.0,
        dry_run=True,
        apply=False,
        embedding_service=svc,
        target_profile_id=TARGET,
        target_dim=DIM,
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "TOTAL" in out
    assert svc.calls == 0
    remaining = EmbeddingReindexer(db, svc, TARGET, column_dim=DIM)
    assert sum((await remaining.count_all_stale()).values()) == 7

    # Apply (yes) re-embeds them.
    rc = await cli_embeddings._reindex(
        db,
        table=None,
        agent_id=None,
        batch=10,
        rate_limit=0.0,
        dry_run=False,
        apply=True,
        embedding_service=svc,
        target_profile_id=TARGET,
        target_dim=DIM,
    )
    assert rc == 0
    assert sum((await remaining.count_all_stale()).values()) == 0


@pytest.mark.asyncio
async def test_null_vector_on_target_profile_is_still_stale(db):
    # A partial migration / fallback can leave a row stamped with the
    # TARGET profile but with a NULL vector. It must still be swept —
    # otherwise it stays invisible to kNN forever (#2289).
    await db.execute_commit(
        "INSERT INTO document_chunks (file_hash, content, embedding_profile_id) "
        "VALUES (?, ?, ?)",
        ("nullvec", "recoverable chunk text", TARGET),
    )
    svc = FakeEmbeddingService()
    reindexer = EmbeddingReindexer(db, svc, TARGET, column_dim=DIM, batch_size=10)

    # Counted as stale despite already carrying the target profile id.
    assert await reindexer.count_stale("document_chunks") == 1

    stats = await reindexer.reindex_table("document_chunks")
    assert stats.reembedded == 1

    rows = await db.fetchall(
        "SELECT embedding_vec FROM document_chunks WHERE file_hash = ?",
        ("nullvec",),
    )
    assert rows[0][0] is not None
    assert await reindexer.count_stale("document_chunks") == 0


@pytest.mark.asyncio
async def test_load_persisted_embedding_route(db):
    from kestrel_sovereign import cli_embeddings

    # No row yet → not found, config default left in place.
    found, route = await cli_embeddings._load_persisted_embedding_route(db, "a1")
    assert (found, route) == (False, None)

    import json

    await db.execute_commit(
        "INSERT OR REPLACE INTO agent_metadata (agent_id, key, value) "
        "VALUES (?, ?, ?)",
        ("a1", "embedding_route", json.dumps("openai:api")),
    )
    # Explicit agent_id lookup.
    assert await cli_embeddings._load_persisted_embedding_route(db, "a1") == (
        True,
        "openai:api",
    )
    # agent_id=None uses the sole stored row (single-agent DB).
    assert await cli_embeddings._load_persisted_embedding_route(db, None) == (
        True,
        "openai:api",
    )
