"""Relevance-based episode recall + access tracking (#1674 P2).

Two tiers:

1. Deterministic (no provider): the keyword-fallback recall path and the
   access-count rehearsal increment — exercised with ``llm_service=None`` so
   ``search_episodes`` takes the LIKE fallback. These run everywhere.

2. Ollama-gated semantic smoke: episodes get embedded on save (reusing the
   shared embedding service + vector backend, the same path as saved_items),
   and a topical query surfaces the relevant *old* episode by cosine — not by
   recency. Skipped when Ollama / nomic-embed-text isn't reachable.

The whole point of P2: a genuinely-consulted episode accrues access heat and
resists the forgetting deletion tier (tested against the real decay curve in
``test_retention_purge_primitive``).
"""

from __future__ import annotations

import os
import tempfile
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.memory_consolidator import MemoryConsolidator
from kestrel_sovereign.storage.session_grouping import timestamp_query_param


pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db_ = await AsyncDatabase.sqlite(path)
    yield db_
    await db_.close()
    os.unlink(path)


async def _insert_episode(db, ep_id, title, summary, *, agent_id="agent-p2"):
    await db.execute(
        """INSERT INTO memory_episodes (id, agent_id, title, summary, created_at)
           VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)""",
        (ep_id, agent_id, title, summary),
    )
    await db.commit()


async def _insert_conversation(
    db, content: str, created_at: str, metadata: dict[str, object], *, agent_id: str
) -> int:
    """Insert a history row with a deliberately chosen legacy timestamp form."""
    await db.execute(
        """INSERT INTO conversation_history
           (agent_id, role, content, metadata, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (agent_id, "user", content, json.dumps(metadata), created_at),
    )
    row = await db.fetchone("SELECT last_insert_rowid()")
    assert row is not None
    return int(row[0])


class _FrozenDateTime(datetime):
    """A deterministic UTC clock while preserving real SQLite database I/O."""

    value = datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return cls.value.replace(tzinfo=None)
        return cls.value.astimezone(tz)


async def test_consolidator_sqlite_cutoffs_normalize_mixed_timestamp_forms(
    db, monkeypatch: pytest.MonkeyPatch
):
    """Nightly episode, pattern, and session cutoffs share one SQLite boundary.

    The database deliberately contains the same cutoff day in SQL text and
    ISO ``T``/``Z`` text. Each real storage query must exclude its strict
    cutoff row and include only later rows, without binding an aware datetime
    through SQLite's deprecated implicit adapter.
    """
    import kestrel_sovereign.storage.memory_consolidator as consolidator_module

    monkeypatch.setattr(consolidator_module, "datetime", _FrozenDateTime)
    assert timestamp_query_param("sqlite", _FrozenDateTime.value) == (
        "2026-07-31T12:00:00+00:00"
    )
    assert await db.fetchone(
        "SELECT julianday(?)", (timestamp_query_param("sqlite", _FrozenDateTime.value),)
    )

    # 30-day episode cutoff: only the three post-cutoff ISO rows may form the
    # episode. The exact SQL-text boundary and the earlier ISO row stay out.
    episode_agent = "agent-cutoff-episodes"
    await _insert_conversation(
        db, "episode at boundary", "2026-07-01 12:00:00", {}, agent_id=episode_agent
    )
    await _insert_conversation(
        db, "episode before boundary", "2026-07-01T11:59:59Z", {}, agent_id=episode_agent
    )
    episode_ids = [
        await _insert_conversation(
            db,
            f"episode after boundary {index}",
            f"2026-07-01T12:00:0{index}Z",
            {},
            agent_id=episode_agent,
        )
        for index in range(1, 4)
    ]
    episodes, skipped = await MemoryConsolidator(db, episode_agent)._create_episodes()
    assert not skipped
    assert len(episodes) == 1
    assert episodes[0].key_message_ids == [str(value) for value in episode_ids]

    # 90-day pattern cutoff: five post-boundary rows yield a single morning
    # pattern. Five exact/before-boundary late-night rows must not contribute.
    pattern_agent = "agent-cutoff-patterns"
    for index in range(5):
        await _insert_conversation(
            db,
            f"pattern excluded {index}",
            "2026-05-02 12:00:00" if index == 0 else f"2026-05-02T11:59:5{index}Z",
            {"time_of_day": "late_night"},
            agent_id=pattern_agent,
        )
        await _insert_conversation(
            db,
            f"pattern included {index}",
            f"2026-05-02T12:00:0{index + 1}Z",
            {"time_of_day": "morning"},
            agent_id=pattern_agent,
        )
    patterns = await MemoryConsolidator(db, pattern_agent)._detect_patterns()
    assert [(pattern.pattern_type, pattern.observations) for pattern in patterns] == [
        ("time_preference", 5),
    ]
    assert patterns[0].trigger_conditions["time_of_day"] == "morning"

    # Session fallback has no prior episode and therefore uses its 24-hour
    # cutoff. It must apply the same canonical strict comparison.
    session_agent = "agent-cutoff-session"
    await _insert_conversation(
        db, "session at boundary", "2026-07-30 12:00:00", {}, agent_id=session_agent
    )
    await _insert_conversation(
        db, "session before boundary", "2026-07-30T11:59:59Z", {}, agent_id=session_agent
    )
    session_ids = [
        await _insert_conversation(
            db,
            f"session after boundary {index}",
            f"2026-07-30T12:00:0{index}Z",
            {},
            agent_id=session_agent,
        )
        for index in range(1, 4)
    ]
    session = await MemoryConsolidator(db, session_agent).create_session_episode(force=True)
    assert session is not None
    assert session.key_message_ids == [str(value) for value in session_ids]


# --------------------------------------------------------------------------
# Deterministic — keyword fallback recall + access increment (no provider)
# --------------------------------------------------------------------------


async def test_keyword_recall_when_no_embedding_provider(db):
    """With no embedding service, search_episodes falls back to a LIKE scan
    over title+summary and still recalls the relevant episode."""
    c = MemoryConsolidator(db, "agent-p2", llm_service=None)
    await _insert_episode(db, "sail", "Sailing trip", "We sailed the lake on a sloop.")
    await _insert_episode(db, "coffee", "Coffee roast", "Roasting Ethiopian beans at home.")

    found = await c.search_episodes("sailing", limit=5)

    assert [e.id for e in found] == ["sail"]


async def test_keyword_recall_accepts_natural_language_topic_query(db):
    """#2330: PG/legacy fallback must not require the whole question verbatim."""
    c = MemoryConsolidator(db, "agent-p2", llm_service=None)
    await _insert_episode(
        db, "compass", "North observatory compass",
        "We discussed the ancient zirconium compass and navigation.",
    )
    await _insert_episode(db, "coffee", "Coffee roast", "Ethiopian beans at home.")

    found = await c.search_episodes(
        "what did we discuss about zirconium", limit=5
    )

    assert [episode.id for episode in found] == ["compass"]


async def test_keyword_recall_ranks_full_overlap_before_recent_partial_hits(db):
    """Common recent tokens cannot crowd an older full-topic match out."""
    c = MemoryConsolidator(db, "agent-p2", llm_service=None)
    await _insert_episode(
        db, "full", "Zirconium observatory compass", "Navigation discussion."
    )
    for index in range(30):
        await _insert_episode(
            db, f"partial-{index}", f"Routine compass note {index}", "Maintenance."
        )

    found = await c.search_episodes("zirconium compass", limit=2)

    assert found[0].id == "full"


async def test_recall_increments_access_count(db):
    """Every episode surfaced by recall has its access_count bumped — the
    rehearsal signal that feeds the deletion-tier decay."""
    c = MemoryConsolidator(db, "agent-p2", llm_service=None)
    await _insert_episode(db, "sail", "Sailing trip", "We sailed the lake.")

    assert (await db.fetchone(
        "SELECT access_count FROM memory_episodes WHERE id='sail'"))[0] == 0

    await c.search_episodes("sailing", limit=5)
    after_one = (await db.fetchone(
        "SELECT access_count FROM memory_episodes WHERE id='sail'"))[0]
    assert after_one == 1

    await c.search_episodes("sailing", limit=5)
    after_two = (await db.fetchone(
        "SELECT access_count FROM memory_episodes WHERE id='sail'"))[0]
    assert after_two == 2


async def test_from_row_tolerates_native_datetimes():
    """asyncpg (Postgres) returns native datetime objects, not ISO strings;
    from_row must accept both so PG keyword recall can materialize episodes."""
    import datetime as _dt
    from kestrel_sovereign.storage.memory_models import MemoryEpisode

    now = _dt.datetime(2026, 6, 1, 12, 0, tzinfo=_dt.timezone.utc)
    ep = MemoryEpisode.from_row(
        ("id1", "agent", "Title", "Summary", now, now, "[]", "arc", now, 0.7, 4)
    )
    assert ep.created_at == now and ep.timespan_start == now
    assert ep.importance == 0.7 and ep.access_count == 4

    # ISO strings (SQLite) still work.
    ep2 = MemoryEpisode.from_row(
        ("id2", "agent", "T", "S", now.isoformat(), now.isoformat(),
         "[]", "arc", now.isoformat(), 0.5, 0)
    )
    assert ep2.created_at == now


async def test_recall_empty_query_returns_nothing(db):
    c = MemoryConsolidator(db, "agent-p2", llm_service=None)
    await _insert_episode(db, "sail", "Sailing trip", "We sailed the lake.")
    assert await c.search_episodes("", limit=5) == []
    assert await c.search_episodes("   ", limit=5) == []


async def test_recall_merges_vector_and_keyword(db):
    """Vector hits rank first; keyword hits (incl. un-embedded/legacy episodes
    the kNN can't see) fill in behind, deduped and capped. Without the merge,
    legacy NULL-embedding episodes could never be recalled or protected."""
    c = MemoryConsolidator(db, "agent-p2", llm_service=None)
    await _insert_episode(db, "emb", "Embedded sailing", "sailing semantic hit")
    await _insert_episode(db, "legacy", "Legacy sailing", "sailing keyword only")

    # Simulate: vector path saw only the embedded row; keyword sees both.
    async def fake_knn(query, limit):
        return ["emb"]

    c._knn_episode_ids = fake_knn  # type: ignore[assignment]

    found = await c.search_episodes("sailing", limit=5)
    ids = [e.id for e in found]
    assert ids[0] == "emb"           # vector hit ranks first
    assert "legacy" in ids           # keyword-only legacy episode still recalled
    assert len(ids) == len(set(ids))  # deduped


async def test_recall_keyword_not_starved_when_vector_fills_limit(db):
    """Regression: even when semantic kNN returns `limit` embedded hits, an
    exact keyword match from a legacy NULL-embedding episode must still surface
    (and get access-heat). Interleave guarantees it a slot."""
    c = MemoryConsolidator(db, "agent-p2", llm_service=None)
    for i in range(3):
        await _insert_episode(db, f"emb{i}", "Embedded sailing", "semantic")
    await _insert_episode(db, "legacy", "Legacy sailing", "keyword only")

    async def fake_knn(query, limit):
        return ["emb0", "emb1", "emb2"]  # vector fills the whole limit=3

    c._knn_episode_ids = fake_knn  # type: ignore[assignment]

    found = await c.search_episodes("sailing", limit=3)
    ids = [e.id for e in found]
    assert "legacy" in ids, f"keyword-only legacy episode starved: {ids}"
    assert ids[0] == "emb0"  # top vector hit still first


async def test_recall_scoped_to_agent(db):
    c = MemoryConsolidator(db, "agent-p2", llm_service=None)
    await _insert_episode(db, "mine", "Sailing trip", "Sailing.", agent_id="agent-p2")
    await _insert_episode(db, "other", "Sailing trip", "Sailing.", agent_id="someone-else")

    found = await c.search_episodes("sailing", limit=5)
    assert [e.id for e in found] == ["mine"]


async def test_episode_vector_query_uses_episode_instruction(db, monkeypatch):
    class QueryAwareService:
        async def aembed_query(self, text, *, instruction):
            self.call = (text, instruction)
            return [1.0, 0.0]

        def current_profile_id(self):
            return "episode-profile"

    class Backend:
        knn = AsyncMock(return_value=[("episode-1", 0.9)])

    service = QueryAwareService()
    backend = Backend()
    consolidator = MemoryConsolidator(db, "agent-p2")
    consolidator._get_embedding_service = lambda: service
    consolidator._get_vector_session_factory = lambda: object()
    monkeypatch.setattr(
        "kestrel_sovereign.storage.vector.get_vector_backend",
        lambda session_factory, spec: backend,
    )

    found = await consolidator._knn_episode_ids("remember the observatory", 3)

    from kestrel_sovereign.llm.embedding_service import (
        EPISODE_RETRIEVAL_INSTRUCTION,
    )

    assert found == ["episode-1"]
    assert service.call == (
        "remember the observatory",
        EPISODE_RETRIEVAL_INSTRUCTION,
    )
    assert backend.knn.await_args.kwargs["filter"]["embedding_profile_id"] == (
        "episode-profile"
    )


# --------------------------------------------------------------------------
# Ollama-gated semantic smoke — real embeddings, cosine recall of old episode
# --------------------------------------------------------------------------

@pytest.fixture
def skip_if_no_ollama():
    # Optional dependency — import here (not module-level) so the deterministic
    # keyword/access tests above still run on CI without the ollama client.
    ollama = pytest.importorskip("ollama")
    try:
        client = ollama.Client()
        models = client.list()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Ollama not available: {exc}")
    names = []
    for entry in getattr(models, "models", []) or []:
        name = getattr(entry, "model", None) or (
            entry.get("model") if isinstance(entry, dict) else None
        )
        if name:
            names.append(name)
    if not any(n.startswith("nomic-embed-text") for n in names):
        pytest.skip("ollama nomic-embed-text not present")


class _OllamaEmbeddingService:
    """Minimal embedding-service facade (mirrors the saved_items smoke)."""

    model = "nomic-embed-text"
    embedding_dim = 768

    def __init__(self):
        import ollama  # optional dep; only constructed in the gated smoke
        self._client = ollama.AsyncClient()

    async def aembed(self, text):
        if not text:
            return None
        from kestrel_sovereign.llm.embedding_service import (
            _prepare_retrieval_document,
        )
        try:
            r = await self._client.embed(
                model=self.model,
                input=_prepare_retrieval_document(text, self.model),
            )
        except Exception:  # noqa: BLE001
            return None
        embeddings = r.get("embeddings") or []
        return list(embeddings[0]) if embeddings else None

    async def aembed_query(self, text, *, instruction):
        from kestrel_sovereign.llm.embedding_service import (
            _prepare_retrieval_query,
        )
        try:
            r = await self._client.embed(
                model=self.model,
                input=_prepare_retrieval_query(text, self.model, instruction),
            )
        except Exception:  # noqa: BLE001
            return None
        embeddings = r.get("embeddings") or []
        return list(embeddings[0]) if embeddings else None

    def describe(self):
        from kestrel_sovereign.llm.embedding_service import derive_embedding_profile
        return derive_embedding_profile(
            provider="ollama", model=self.model, dim=self.embedding_dim,
        )

    def current_profile_id(self):
        p = self.describe()
        return p.profile_id if p else None


async def test_episode_embedded_on_save_and_recalled_semantically(
    db, skip_if_no_ollama,
):
    """An episode is embedded at save time, and a topically-related query
    surfaces it by cosine — even though a more-recent, unrelated episode
    exists (proving recall is by relevance, not recency)."""
    from kestrel_sovereign.storage.memory_models import MemoryEpisode

    c = MemoryConsolidator(db, "agent-p2")
    c._get_embedding_service = lambda: _OllamaEmbeddingService()

    # Older, relevant episode saved first…
    await c._save_episode(MemoryEpisode(
        id="sailing", agent_id="agent-p2",
        title="Lake sailing weekend",
        summary="We took the wooden sloop out on Lake Michigan all weekend.",
    ))
    # …then a NEWER unrelated episode (recency would surface this one).
    await c._save_episode(MemoryEpisode(
        id="taxes", agent_id="agent-p2",
        title="Quarterly taxes",
        summary="Filed the quarterly estimated tax payment and reconciled receipts.",
    ))

    # Both rows embedded on save.
    embedded = await db.fetchall(
        "SELECT id FROM memory_episodes WHERE embedding_vec IS NOT NULL "
        "AND agent_id='agent-p2'"
    )
    assert {r[0] for r in embedded} == {"sailing", "taxes"}

    found = await c.search_episodes("boats on the water", limit=2)
    assert found, "semantic recall returned nothing"
    assert found[0].id == "sailing", (
        f"relevance recall failure: expected 'sailing' top-1, got "
        f"{[e.id for e in found]}"
    )
    # The surfaced episode was marked as accessed.
    acc = (await db.fetchone(
        "SELECT access_count FROM memory_episodes WHERE id='sailing'"))[0]
    assert acc >= 1
