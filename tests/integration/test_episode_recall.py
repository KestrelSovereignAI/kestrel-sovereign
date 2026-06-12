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

import pytest
import pytest_asyncio

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.memory_consolidator import MemoryConsolidator


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
        try:
            r = await self._client.embed(model=self.model, input=text)
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
