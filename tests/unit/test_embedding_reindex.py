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
    dominant_embedding_profile,
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


# ---------------------------------------------------- #2366 dominant profile


async def _register_profile(
    database: AsyncDatabase, pid: str, *, provider: str, model: str, dim: int,
    space_id: str,
) -> None:
    await database.execute_commit(
        "INSERT OR IGNORE INTO embedding_profiles "
        "(id, provider, model, dim, space_id, normalized) VALUES (?, ?, ?, ?, ?, ?)",
        (pid, provider, model, dim, space_id, 0),
    )


@pytest.mark.asyncio
async def test_dominant_embedding_profile_picks_majority(db):
    await _seed(db)  # OTHER dominates: 3 conv + 1 saved + 1 chunk vs TARGET's 1.
    await _register_profile(
        db, OTHER, provider="ollama", model="qwen3-embedding-8b", dim=768,
        space_id="qwen3-embedding-8b@768",
    )
    await _register_profile(
        db, TARGET, provider="google", model="gemini-embedding-2", dim=3072,
        space_id="google:gemini-embedding-2",
    )
    prof = await dominant_embedding_profile(db)
    assert prof is not None
    assert prof["profile_id"] == OTHER
    assert prof["model"] == "qwen3-embedding-8b"
    assert prof["dim"] == 768
    # OTHER rows: conv a1(1)+a2(1) + saved(1) + chunk(1) = 4 (NULLs excluded).
    assert prof["row_count"] == 4


@pytest.mark.asyncio
async def test_dominant_embedding_profile_agent_scoped(db):
    await _seed(db)
    await _register_profile(
        db, OTHER, provider="ollama", model="qwen3-embedding-8b", dim=768,
        space_id="s",
    )
    prof = await dominant_embedding_profile(db, agent_id="a1")
    assert prof is not None
    # a1's OTHER rows: conv(1) + saved(1) + global chunk(1) = 3 (a2 excluded).
    assert prof["row_count"] == 3


@pytest.mark.asyncio
async def test_dominant_embedding_profile_none_when_empty(db):
    assert await dominant_embedding_profile(db) is None


@pytest.mark.asyncio
async def test_dominant_embedding_profile_none_without_registry_row(db):
    await _seed(db)  # rows exist but no embedding_profiles descriptor registered.
    assert await dominant_embedding_profile(db) is None


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


# -------------------------------------------------- #2427 halving-retry


class FirstCallDropsThenSucceeds:
    """All-None on the FIRST ``aembed_batch`` call, real vectors after.

    Models a transient Ollama runner recycle (``… EOF``): the whole first
    batch call comes back empty, but the split retry succeeds. No poison row —
    every row must be recovered.
    """

    def __init__(self, *, dim: int = DIM):
        self.embedding_dim = dim
        self._dim = dim
        self.calls = 0

    async def aembed_batch(self, texts):
        self.calls += 1
        if self.calls == 1:
            return [None for _ in texts]
        return [[float(len(t) % 7) + 1.0 + i for i in range(self._dim)] for t in texts]

    def current_profile_id(self) -> str:
        return TARGET

    def describe(self):
        return None


class PoisonsWholeBatch:
    """All-None whenever a specific poison text is in the batch, else real vectors.

    Halving must isolate the poison row down to a size-1 batch that fails
    alone — every other row re-embeds.
    """

    def __init__(self, poison: str, *, dim: int = DIM):
        self.embedding_dim = dim
        self._dim = dim
        self._poison = poison
        self.calls = 0

    async def aembed_batch(self, texts):
        self.calls += 1
        if any(t == self._poison for t in texts):
            return [None for _ in texts]
        return [[float(len(t) % 7) + 1.0 + i for i in range(self._dim)] for t in texts]

    def current_profile_id(self) -> str:
        return TARGET

    def describe(self):
        return None


@pytest.mark.asyncio
async def test_transient_batch_drop_recovers_via_halved_retry(db, caplog):
    import logging

    # Four stale document chunks in one batch; the first embed call drops the
    # whole batch (runner recycle), the split retry succeeds.
    for i in range(4):
        await db.execute_commit(
            "INSERT INTO document_chunks (file_hash, content, embedding_profile_id) "
            "VALUES (?, ?, ?)",
            (f"h{i}", f"chunk text number {i}", OTHER),
        )
    svc = FirstCallDropsThenSucceeds()
    reindexer = EmbeddingReindexer(
        db, svc, TARGET, column_dim=DIM, batch_size=50, retry_backoff_s=0.0
    )

    with caplog.at_level(logging.INFO, logger="kestrel_sovereign.storage.embedding_reindex"):
        stats = await reindexer.reindex_table("document_chunks")

    assert stats.reembedded == 4
    assert stats.failed == 0
    assert svc.calls >= 2  # first call dropped, retried split in half
    assert any("retrying split in half" in r.message for r in caplog.records)
    # Nothing stale remains — the transient blip cost nothing.
    assert await reindexer.count_stale("document_chunks") == 0


@pytest.mark.asyncio
async def test_poison_row_isolated_by_halved_retry(db):
    # One poison chunk among four; the batch fails wholesale whenever the poison
    # is present. Recursive halving must isolate it: 3 re-embedded, 1 failed.
    poison = "POISON ROW CONTENT"
    await db.execute_commit(
        "INSERT INTO document_chunks (file_hash, content, embedding_profile_id) "
        "VALUES (?, ?, ?)",
        ("poison", poison, OTHER),
    )
    for i in range(3):
        await db.execute_commit(
            "INSERT INTO document_chunks (file_hash, content, embedding_profile_id) "
            "VALUES (?, ?, ?)",
            (f"ok{i}", f"good chunk {i}", OTHER),
        )
    svc = PoisonsWholeBatch(poison)
    reindexer = EmbeddingReindexer(
        db, svc, TARGET, column_dim=DIM, batch_size=50, retry_backoff_s=0.0
    )

    stats = await reindexer.reindex_table("document_chunks")

    assert stats.reembedded == 3
    assert stats.failed == 1
    # The poison row alone stays stale; the other three flipped to target.
    rows = await db.fetchall(
        "SELECT file_hash, embedding_profile_id FROM document_chunks ORDER BY file_hash",
        (),
    )
    by_hash = {r[0]: r[1] for r in rows}
    assert by_hash["poison"] == OTHER
    assert all(by_hash[f"ok{i}"] == TARGET for i in range(3))


@pytest.mark.asyncio
async def test_split_retry_budget_is_bounded(db):
    # A fully dead service must not recurse forever: with the retry budget
    # exhausted, every row is still written off as failed and the run ends.
    for i in range(8):
        await db.execute_commit(
            "INSERT INTO document_chunks (file_hash, content, embedding_profile_id) "
            "VALUES (?, ?, ?)",
            (f"d{i}", f"dead chunk {i}", OTHER),
        )

    class Dead:
        embedding_dim = DIM

        def __init__(self):
            self.calls = 0

        async def aembed_batch(self, texts):
            self.calls += 1
            return [None for _ in texts]

        def current_profile_id(self):
            return TARGET

        def describe(self):
            return None

    svc = Dead()
    reindexer = EmbeddingReindexer(
        db, svc, TARGET, column_dim=DIM, batch_size=50,
        retry_backoff_s=0.0, max_split_retries=3,
    )
    stats = await reindexer.reindex_table("document_chunks")
    assert stats.reembedded == 0
    assert stats.failed == 8
    # Retries were capped — not one call per possible split of 8 rows.
    assert svc.calls <= 1 + 3 * 2


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


def test_reindex_refuses_when_resolved_db_missing(tmp_path, capsys):
    # reindex/audit are maintenance verbs over an existing corpus. A
    # --data-dir with no kestrel_prime.db must refuse with a non-zero exit
    # rather than let the driver create an empty DB and count zero rows
    # (the confident false success #2327 is about).
    import argparse

    from kestrel_sovereign import cli_embeddings

    args = argparse.Namespace(
        embeddings_command="reindex",
        table=None,
        agent_id=None,
        agent_name=None,
        data_dir=str(tmp_path),  # empty dir — no kestrel_prime.db
        batch=10,
        rate_limit=0.0,
        dry_run=True,
        yes=False,
    )
    rc = cli_embeddings.run(args)
    assert rc == 2
    err = capsys.readouterr().err
    assert "no database found" in err
    assert "will not create a new" in err


def test_resolve_db_target_refuses_db_path_with_multi_agent_roster(monkeypatch):
    # KESTREL_DB_PATH set + a multi_agent.toml roster + no explicit selector
    # must refuse rather than let KESTREL_DB_PATH silently pick the DB (#2327).
    import argparse

    from kestrel_sovereign import cli_embeddings

    monkeypatch.setenv("KESTREL_DB_PATH", "/tmp/agent_data/claw")
    monkeypatch.delenv("KESTREL_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("KESTREL_DB_BACKEND", raising=False)
    monkeypatch.setattr(
        cli_embeddings,
        "_discover_local_agents",
        lambda: {"wren": "/tmp/agent_data/wren"},
    )

    args = argparse.Namespace(agent_name=None, data_dir=None)
    err, pg_url, sqlite_path = cli_embeddings._resolve_db_target(args)
    assert pg_url is None and sqlite_path is None
    assert err is not None
    assert "KESTREL_DB_PATH is set" in err
    assert "wren" in err
    assert "--agent-name" in err


def test_resolve_db_target_honors_db_path_without_roster(monkeypatch):
    # No roster → KESTREL_DB_PATH still resolves as the server default.
    import argparse

    from kestrel_sovereign import cli_embeddings

    monkeypatch.setenv("KESTREL_DB_PATH", "/tmp/solo")
    monkeypatch.delenv("KESTREL_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("KESTREL_DB_BACKEND", raising=False)
    monkeypatch.setattr(cli_embeddings, "_discover_local_agents", lambda: {})

    args = argparse.Namespace(agent_name=None, data_dir=None)
    err, pg_url, sqlite_path = cli_embeddings._resolve_db_target(args)
    assert err is None and pg_url is None
    assert sqlite_path == "/tmp/solo/kestrel_prime.db"


def test_resolve_db_target_refuses_db_path_from_kestrel_home(monkeypatch, tmp_path):
    # The multi-agent guard must hold even when the command runs OUTSIDE the
    # project root: discovery is anchored on paths.project_dir() (KESTREL_HOME
    # aware), not os.getcwd(). Without mocking _discover_local_agents, set up a
    # real KESTREL_HOME whose multi_agent.toml defines an agent, chdir
    # elsewhere, set KESTREL_DB_PATH, and confirm the refusal still fires
    # rather than KESTREL_DB_PATH silently winning (#2327).
    import argparse
    import os

    from kestrel_sovereign import cli_embeddings

    home = tmp_path / "home"
    (home / "agent_data" / "wren").mkdir(parents=True)
    # A real (empty-but-present) DB file so the roster's relative data_dir
    # resolves against the home, not cwd.
    (home / "agent_data" / "wren" / "kestrel_prime.db").write_text("")
    (home / "multi_agent.toml").write_text(
        "[agents.wren]\n"
        'data_dir = "agent_data/wren"\n'
        "port = 8801\n"
    )

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    monkeypatch.setenv("KESTREL_HOME", str(home))
    monkeypatch.setenv("KESTREL_DB_PATH", str(tmp_path / "wrong" / "claw"))
    monkeypatch.delenv("KESTREL_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("KESTREL_DB_BACKEND", raising=False)

    args = argparse.Namespace(agent_name=None, data_dir=None)
    err, pg_url, sqlite_path = cli_embeddings._resolve_db_target(args)
    assert pg_url is None and sqlite_path is None
    assert err is not None
    assert "KESTREL_DB_PATH is set" in err
    assert "wren" in err


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


def _cli_context_service(provider):
    """A freshly-constructed process-local ``LLMService`` (CLI context).

    Mirrors the ``__new__`` build used by ``test_embedding_route_model`` — no
    boot-time pin-load or embedding discovery has run, so a route with empty
    static capabilities advertises no embedding support (the #2361 starting
    state).
    """
    from kestrel_sovereign.llm.service import LLMService

    service = LLMService.__new__(LLMService)
    service.providers = [provider]
    service._route_embedding_model_overrides = {}
    service._route_embedding_caps_backup = {}
    service._route_embedding_model_persistence_callback = None
    service._embedding_route_persistence_callback = None
    service._embedding_discovery_cache = []  # empty → discovery is a no-op
    service._force_local_only_provider = None
    service._embedding_space_pins = None
    service._verified_space_pins = {}
    service._embedding_route = None
    service.disabled = False
    return service


@pytest.mark.asyncio
async def test_cli_service_honors_persisted_route_model_pin(db):
    """A route the server resolves via a runtime pin must resolve in the CLI too (#2361).

    Persist a per-route ``embedding_model`` pin (#2337) + ``embedding_route``
    (#2263) the way the settings API/UI does, then construct a CLI-context
    ``LLMService`` whose STATIC config advertises no embedding support for that
    route. Applying the persisted config must re-advertise the route (via the
    pin) so ``_resolve_target`` succeeds and targets the pinned model's profile —
    instead of refusing with "does not advertise embedding support".
    """
    import json

    from kestrel_sovereign import cli_embeddings
    from kestrel_sovereign.llm.embedding_service import derive_embedding_profile

    provider = {
        "name": "openrouter:api",
        "vendor": "openrouter",
        "route": "api",
        "adapter": object(),
        "client": object(),
        "model": "auto",
        "is_local": False,
        "is_cloud": True,
        "capabilities": {},
    }
    service = _cli_context_service(provider)

    # Baseline: the route advertises no embedding support — the exact CLI refusal
    # #2361 reports.
    with pytest.raises(ValueError, match="does not advertise embedding support"):
        service.set_embedding_route("openrouter:api", persist=False)

    # Persist a runtime model pin (#2337) + embedding_route (#2263), as the
    # settings path would.
    await db.execute_commit(
        "INSERT OR REPLACE INTO agent_metadata (agent_id, key, value) "
        "VALUES (?, ?, ?)",
        (
            "a1",
            "embedding_model_overrides",
            json.dumps(
                {"openrouter:api": {"model": "qwen/qwen3-embedding-8b", "dim": 768}}
            ),
        ),
    )
    await db.execute_commit(
        "INSERT OR REPLACE INTO agent_metadata (agent_id, key, value) "
        "VALUES (?, ?, ?)",
        ("a1", "embedding_route", json.dumps("openrouter:api")),
    )

    err = await cli_embeddings._apply_persisted_embedding_config(service, db, "a1")
    assert err is None

    # The pin re-advertised embedding support, so _resolve_target now succeeds
    # and targets the pinned model's profile — not a refusal.
    resolve_err, embedding_service, target = cli_embeddings._resolve_target(service)
    assert resolve_err is None
    assert embedding_service is not None
    expected = derive_embedding_profile(
        provider="openrouter", model="qwen/qwen3-embedding-8b", dim=768
    ).profile_id
    assert target == expected


# ---------------------------- endpoint reindex target honours the pin (#2423)


class _FakeEmbeddingAdapter:
    """Adapter that DISCOVERS one default model and can embed at ``dim``.

    Models the live #2423 shape: a route (ollama:local) whose discovery
    default is ``nomic-embed-text`` — so an un-pinned reindex resolves nomic,
    exactly the profile the corpus was wrongly stamped with. A per-route pin
    must override that default.
    """

    def __init__(self, default_model: str, dim: int):
        self._default_model = default_model
        self._dim = dim

    async def list_embedding_models(self, client):
        from kestrel_sovereign.llm.embedding_discovery import EmbeddingModelInfo

        return [EmbeddingModelInfo(id=self._default_model, provider="ollama", native_dim=self._dim)]

    async def aembed_batch(self, client, texts, model=None, **kwargs):
        return [[float(len(t) % 5) + i for i in range(self._dim)] for t in texts]


@pytest.mark.asyncio
async def test_reindex_target_honours_persisted_route_model_pin_multi_agent(
    db, monkeypatch
):
    """Endpoint reindex must stamp the PINNED model's profile, not the route default (#2423).

    Multi-agent-shaped: the reindex resolves from a per-agent ``LLMService``
    instance whose IN-MEMORY pin state is empty (the route-model POST mutated a
    different per-agent instance). The pin lives only in the DB — the same
    persisted state the settings GET honours. ``_resolve_reindex_target`` must
    re-apply it from the DB before resolving, so the target is the pinned
    ``qwen3-embedding:8b`` profile rather than the route's discovery default
    (``nomic-embed-text``) — the exact divergence #2423 hit live (settings said
    qwen3, corpus got stamped nomic).
    """
    import json

    from kestrel_sovereign.endpoints.models import _resolve_reindex_target
    from kestrel_sovereign.llm.embedding_service import derive_embedding_profile
    from kestrel_sovereign import cli_embeddings

    PIN_DIM = 16
    PINNED_MODEL = "qwen3-embedding:8b"
    DEFAULT_MODEL = "nomic-embed-text"

    provider = {
        "name": "ollama:local",
        "vendor": "ollama",
        "route": "local",
        "adapter": _FakeEmbeddingAdapter(DEFAULT_MODEL, PIN_DIM),
        "client": object(),
        "model": "auto",
        "is_local": True,
        "is_cloud": False,
        "capabilities": {},
    }
    # Per-agent instance: EMPTY in-memory pin state (the POST landed elsewhere).
    service = _cli_context_service(provider)

    # Persist the pin (#2337) + embedding_route (#2263) — the DB is the only
    # place the pin exists for THIS instance.
    await db.execute_commit(
        "INSERT OR REPLACE INTO agent_metadata (agent_id, key, value) "
        "VALUES (?, ?, ?)",
        (
            "a1",
            "embedding_model_overrides",
            json.dumps({"ollama:local": {"model": PINNED_MODEL, "dim": PIN_DIM}}),
        ),
    )
    await db.execute_commit(
        "INSERT OR REPLACE INTO agent_metadata (agent_id, key, value) "
        "VALUES (?, ?, ?)",
        ("a1", "embedding_route", json.dumps("ollama:local")),
    )
    await _seed(db)

    # The vector column matches the pin dim so the dim guard passes.
    monkeypatch.setattr(cli_embeddings, "_resolve_column_dim", lambda: PIN_DIM)

    class _Agent:
        pass

    agent = _Agent()
    agent.llm_service = service
    agent.agent_id = "a1"

    embedding_service, target, target_dim, column_dim = await _resolve_reindex_target(
        agent, db=db, agent_id="a1"
    )

    pinned_profile = derive_embedding_profile(
        provider="ollama", model=PINNED_MODEL, dim=PIN_DIM
    ).profile_id
    default_profile = derive_embedding_profile(
        provider="ollama", model=DEFAULT_MODEL, dim=PIN_DIM
    ).profile_id

    # Resolved target is the PINNED model — never the route's discovery default.
    assert target == pinned_profile
    assert target != default_profile
    assert target_dim == PIN_DIM
    assert embedding_service.current_profile_id() == pinned_profile

    # And an actual reindex through the resolved service stamps every row with
    # the pinned profile (the corpus and the settings surface now agree).
    reindexer = EmbeddingReindexer(
        db, embedding_service, target, column_dim=column_dim, batch_size=50
    )
    for table in REINDEX_TABLES:
        await reindexer.reindex_table(table, agent_id="a1")

    a1_rows = await db.fetchall(
        "SELECT embedding_profile_id FROM conversation_history WHERE agent_id = ?",
        ("a1",),
    )
    assert a1_rows
    assert all(r[0] == pinned_profile for r in a1_rows), a1_rows


@pytest.mark.asyncio
async def test_reindex_target_clears_stale_in_memory_pin_when_db_empty(db, monkeypatch):
    """Reindex must DROP a stale in-memory pin the operator cleared in the DB (#2423).

    The per-agent ``LLMService`` still holds a runtime pin
    (``qwen3-embedding:8b``) from before the operator cleared it, but the DB
    override map is now ``{}`` (the authoritative persisted state the settings
    GET honours). A purely additive re-seed would leave the stale pin active and
    stamp rows with ``qwen3`` again; ``_resolve_reindex_target`` must sync the
    runtime overrides to the empty DB state so the target falls back to the
    route's discovery default (``nomic-embed-text``) — the DB-authoritative
    cleared model.
    """
    from kestrel_sovereign.endpoints.models import _resolve_reindex_target
    from kestrel_sovereign import cli_embeddings

    PIN_DIM = 16
    STALE_MODEL = "qwen3-embedding:8b"
    DEFAULT_MODEL = "nomic-embed-text"

    provider = {
        "name": "ollama:local",
        "vendor": "ollama",
        "route": "local",
        "adapter": _FakeEmbeddingAdapter(DEFAULT_MODEL, PIN_DIM),
        "client": object(),
        "model": "auto",
        "is_local": True,
        "is_cloud": False,
        "capabilities": {},
    }
    service = _cli_context_service(provider)
    # Seed a STALE in-memory pin (as if the route-model POST landed here before
    # the operator cleared it) — the DB below has NO override for this route.
    service.set_route_embedding_model("ollama:local", STALE_MODEL, PIN_DIM, persist=False)
    assert "ollama:local" in service.get_route_embedding_model_overrides()

    # DB has the route persisted but the override map CLEARED (operator cleared
    # the pin so writes match the corpus).
    await db.execute_commit(
        "INSERT OR REPLACE INTO agent_metadata (agent_id, key, value) "
        "VALUES (?, ?, ?)",
        ("a1", "embedding_route", '"ollama:local"'),
    )
    await _seed(db)

    monkeypatch.setattr(cli_embeddings, "_resolve_column_dim", lambda: PIN_DIM)

    class _Agent:
        pass

    agent = _Agent()
    agent.llm_service = service
    agent.agent_id = "a1"

    embedding_service, target, target_dim, column_dim = await _resolve_reindex_target(
        agent, db=db, agent_id="a1"
    )

    # The stale in-memory pin was dropped to match the empty DB state, so the
    # resolved model is the route's discovery default — NOT the cleared pin.
    assert service.get_route_embedding_model_overrides() == {}
    assert embedding_service.model == DEFAULT_MODEL
    assert embedding_service.model != STALE_MODEL


# ------------------------------------- guard-before-side-effects (#2362)


def _reindex_args():
    import argparse

    return argparse.Namespace(
        embeddings_command="reindex",
        table=None,
        agent_id=None,
        agent_name=None,
        data_dir=None,
        batch=10,
        rate_limit=0.0,
        dry_run=True,
        yes=False,
    )


def test_reindex_guard_leaves_no_junk_dirs(monkeypatch, tmp_path, capsys):
    # A writable-but-bogus KESTREL_DB_PATH + a multi-agent roster must refuse
    # WITHOUT any path-touching side effect: no ``trusted_agents/`` dir and no
    # ``llm_usage.db`` under the bogus path. The DB-selection guard has to fire
    # before anything constructs against the resolved path (#2362).
    from kestrel_sovereign import cli_embeddings

    bogus = tmp_path / "bogus"
    bogus.mkdir()
    monkeypatch.setenv("KESTREL_DB_PATH", str(bogus))
    monkeypatch.delenv("KESTREL_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("KESTREL_DB_BACKEND", raising=False)
    monkeypatch.setattr(
        cli_embeddings,
        "_discover_local_agents",
        lambda: {"wren": str(tmp_path / "agent_data" / "wren")},
    )

    rc = cli_embeddings.run(_reindex_args())
    assert rc == 2
    err = capsys.readouterr().err
    assert "KESTREL_DB_PATH is set" in err
    # No junk written by a command that (correctly) refused to run.
    assert not (bogus / "trusted_agents").exists()
    assert not (bogus / "llm_usage.db").exists()
    assert list(bogus.iterdir()) == []


def test_reindex_guard_on_unwritable_path_refuses_cleanly(monkeypatch, tmp_path, capsys):
    # An unwritable KESTREL_DB_PATH must exit with the clean refusal message,
    # not a raw OSError traceback from an eager makedirs firing before the
    # guard (#2362).
    import os

    from kestrel_sovereign import cli_embeddings

    ro_parent = tmp_path / "ro"
    ro_parent.mkdir()
    unwritable = ro_parent / "claw"  # makedirs here would EACCES
    os.chmod(ro_parent, 0o500)
    monkeypatch.setenv("KESTREL_DB_PATH", str(unwritable))
    monkeypatch.delenv("KESTREL_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("KESTREL_DB_BACKEND", raising=False)
    monkeypatch.setattr(
        cli_embeddings,
        "_discover_local_agents",
        lambda: {"wren": str(tmp_path / "agent_data" / "wren")},
    )

    try:
        rc = cli_embeddings.run(_reindex_args())
    finally:
        os.chmod(ro_parent, 0o700)

    assert rc == 2
    err = capsys.readouterr().err
    assert "KESTREL_DB_PATH is set" in err
    assert not unwritable.exists()
