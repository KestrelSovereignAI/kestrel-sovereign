"""Repair of episodes already synthesized from ciphertext (#2856).

#2850 stopped NEW episodes being built from the at-rest envelope. The rows
written before it keep serving titles like ``Discussion of ksav2, ax1iv9...``
until something rewrites them. A first attempt at that repair was more
dangerous than the damage — it deleted healthy episodes, could not rebuild
anything older than the 30-day clustering window, and orphaned the mirrored
graph node. Those three failures are what these tests pin down.

Everything here runs against a real SQLite ``AsyncDatabase``, so the LIKE
pre-filter, the ``IN`` rehydration and the UPDATE are genuinely exercised
rather than mocked.
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db import SQLiteBackend
from kestrel_sovereign.storage.memory_consolidator import MemoryConsolidator


AGENT = "did:test:repair"
ENC_META = {"enc": True, "key_version": 1}

# The shape the bug actually produced: the envelope magic plus the base64 body
# it was concatenated with.
ENVELOPE_BODY = "ax1iv9waarjjkc8neundo25yhy9nmiw5hvx8dr8mxj7ixr4rlfls"
ENVELOPE_BODY2 = "axvp51xic1htadk6n1kqaxivtagj6ejuywjwdzf91r9otnvxvbwup"
# Standard base64 contains '+', so a real envelope tokenizes into the magic
# prefix plus several body chunks — which is what Emma's damaged titles show.
ENVELOPE = f"KSAv2:{ENVELOPE_BODY}+{ENVELOPE_BODY2}"
CORRUPT_TITLE = f"Discussion of ksav2, {ENVELOPE_BODY}, {ENVELOPE_BODY2}"
CORRUPT_SUMMARY = (
    f"A conversation with 3 messages (2 from user). Topics: ksav2, "
    f"{ENVELOPE_BODY}, {ENVELOPE_BODY2}. "
    f"Emotional trajectory: emotionally steady."
)

PLAINTEXTS = [
    "the scheduler keeps dropping leases when the host restarts",
    "we should make the lease renewal durable before the next deploy",
    "agreed, durable leases first",
]

# A HEALTHY conversation about the envelope format, stored encrypted. Standard
# base64 contains '+', so a real envelope tokenizes into several terms — and
# talking about the format naturally reuses exactly those words. This is what
# separates a proof from a heuristic.
HEALTHY_ENVELOPE = "KSAv2:base64+prefix+aead"
HEALTHY_PLAINTEXTS = [
    "why does every row in the database start with KSAv2",
    "the KSAv2 prefix marks an aead envelope and the body is base64",
]


async def _make_db(tmp_path, name="repair.db"):
    raw = SQLiteBackend(str(tmp_path / name))
    await raw.connect()
    db = AsyncDatabase(raw)
    await db.execute(
        "CREATE TABLE conversation_history ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, agent_id TEXT NOT NULL, "
        "role TEXT NOT NULL, content TEXT NOT NULL, metadata TEXT, "
        "created_at TIMESTAMP, deleted_at TIMESTAMP, archived_at TIMESTAMP)"
    )
    await db.execute(
        "CREATE TABLE memory_episodes ("
        "id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, title TEXT NOT NULL, "
        "summary TEXT, timespan_start TIMESTAMP, timespan_end TIMESTAMP, "
        "key_message_ids TEXT, emotional_arc TEXT, created_at TIMESTAMP, "
        "importance REAL DEFAULT 0.5, access_count INTEGER DEFAULT 0, "
        "embedding_vec BLOB, embedding_profile_id TEXT, "
        "excluded_from_context INTEGER DEFAULT 0)"
    )
    await db.execute(
        "CREATE TABLE graph_nodes ("
        "node_id TEXT PRIMARY KEY, node_type TEXT NOT NULL, "
        "label TEXT NOT NULL, properties TEXT)"
    )
    return db


async def _add_messages(db, contents, *, encrypted=True, age_days=1,
                        archived=False, deleted=False):
    """Insert source rows and return their IDs in order."""
    ids = []
    when = datetime.now(timezone.utc) - timedelta(days=age_days)
    for offset, content in enumerate(contents):
        meta = dict(ENC_META) if encrypted else {}
        await db.execute(
            "INSERT INTO conversation_history "
            "(agent_id, role, content, metadata, created_at, archived_at, deleted_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                AGENT,
                "user" if offset % 2 == 0 else "assistant",
                content,
                json.dumps(meta),
                (when + timedelta(seconds=offset)).isoformat(),
                when.isoformat() if archived else None,
                when.isoformat() if deleted else None,
            ),
        )
        ids.append(str(await db.fetchval("SELECT last_insert_rowid()")))
    return ids


async def _add_episode(db, episode_id, title, summary, message_ids, *,
                       with_node=True, embedding=b"poisoned-vector"):
    await db.execute(
        "INSERT INTO memory_episodes "
        "(id, agent_id, title, summary, key_message_ids, emotional_arc, "
        " created_at, importance, embedding_vec) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            episode_id, AGENT, title, summary, json.dumps(message_ids),
            "emotionally steady", datetime.now(timezone.utc).isoformat(),
            0.5, embedding,
        ),
    )
    if with_node:
        await db.execute(
            "INSERT INTO graph_nodes (node_id, node_type, label, properties) "
            "VALUES (?, ?, ?, ?)",
            (episode_id, "episode", title, json.dumps({"summary": summary})),
        )


def _decrypting_store(mapping):
    """A conversation store that turns each envelope into its plaintext."""
    store = MagicMock()
    store.decrypt_stored_content.side_effect = (
        lambda content, meta: mapping.get(content, content)
    )
    return store


class _RecordingGraphStore:
    """Real enough to observe ordering: writes straight to the same DB."""

    def __init__(self, db, *, fail=False):
        self._db = db
        self._fail = fail
        self.row_titles_at_write = []

    async def get_node(self, node_id):
        row = await self._db.fetchall(
            "SELECT node_id, node_type, label, properties FROM graph_nodes "
            "WHERE node_id = ?", (node_id,)
        )
        if not row:
            return None
        node = MagicMock()
        node.label = row[0][2]
        node.properties = json.loads(row[0][3] or "{}")
        return node

    async def add_node(self, node):
        # Capture what the SQL row still says at the moment the node is
        # rewritten — this is how the ordering requirement is observed.
        self.row_titles_at_write.append(
            await self._db.fetchval(
                "SELECT title FROM memory_episodes WHERE id = ?", (node.node_id,)
            )
        )
        if self._fail:
            raise RuntimeError("graph write refused")
        await self._db.execute(
            "UPDATE graph_nodes SET label = ?, properties = ? WHERE node_id = ?",
            (node.label, json.dumps(node.properties), node.node_id),
        )

    async def add_edge(self, *args, **kwargs):
        return None


def _consolidator(db, store=None, graph_store=None):
    return MemoryConsolidator(
        db=db, agent_id=AGENT, graph_store=graph_store, conversation_store=store
    )


async def _corrupt_fixture(tmp_path, *, age_days=1, archived=False, **kw):
    db = await _make_db(tmp_path)
    ids = await _add_messages(
        db, [ENVELOPE, ENVELOPE, ENVELOPE], age_days=age_days, archived=archived
    )
    await _add_episode(db, "episode:corrupt", CORRUPT_TITLE, CORRUPT_SUMMARY,
                       ids, **kw)
    # Each envelope row decrypts to a distinct plaintext.
    store = MagicMock()
    seq = iter(PLAINTEXTS * 4)
    store.decrypt_stored_content.side_effect = (
        lambda content, meta: next(seq) if content == ENVELOPE else content
    )
    return db, ids, store


class TestHealthyEpisodesSurvive:
    """Requirement 1: the magic word alone must never authorize a rewrite."""

    @pytest.mark.asyncio
    async def test_episode_genuinely_about_the_envelope_format_is_untouched(
        self, tmp_path
    ):
        """The case that made the first repair attempt destructive.

        These rows ARE encrypted, so an envelope exists to blame — and because
        the conversation is about the envelope format, the very terms its magic
        tokenizes to also appear in the plaintext. A detector that asks only
        "is this term in the envelope?" rewrites a perfectly good episode here,
        every night. The plaintext subtraction is what stops it.
        """
        db = await _make_db(tmp_path)
        ids = await _add_messages(
            db, [HEALTHY_ENVELOPE, HEALTHY_ENVELOPE], encrypted=True
        )
        title = "Discussion of ksav2, base64, prefix, aead"
        await _add_episode(
            db, "episode:healthy", title,
            "A conversation with 2 messages (1 from user). Topics: ksav2, "
            "base64, prefix, aead. Emotional trajectory: emotionally steady.",
            ids,
        )
        healthy_talk = iter(HEALTHY_PLAINTEXTS * 4)
        store = MagicMock()
        store.decrypt_stored_content.side_effect = (
            lambda content, meta: next(healthy_talk)
            if content == HEALTHY_ENVELOPE else content
        )
        c = _consolidator(db, store, _RecordingGraphStore(db))
        try:
            report = await c.repair_ciphertext_episodes()

            assert report["scanned"] == 1, "pre-filter should still surface it"
            assert report["repaired"] == 0
            assert report["cleared"] == 1
            assert await db.fetchval(
                "SELECT title FROM memory_episodes WHERE id = ?",
                ("episode:healthy",)
            ) == title
            assert await db.fetchval(
                "SELECT COUNT(*) FROM memory_episodes"
            ) == 1, "repair must never delete an episode"
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_single_envelope_term_is_not_enough_corroboration(self, tmp_path):
        db = await _make_db(tmp_path)
        ids = await _add_messages(db, [ENVELOPE, ENVELOPE])
        # Only the magic word is alien — no accompanying base64 body.
        await _add_episode(db, "episode:thin", "Discussion of ksav2",
                           "Topics: ksav2.", ids)
        store = _decrypting_store({ENVELOPE: "a totally ordinary sentence"})
        c = _consolidator(db, store, _RecordingGraphStore(db))
        try:
            report = await c.repair_ciphertext_episodes()
            assert report["repaired"] == 0
            assert report["cleared"] == 1
        finally:
            await db.close()


class TestRebuildIgnoresTheClusteringWindow:
    """Requirement 2: repair must not depend on re-clustering."""

    @pytest.mark.asyncio
    async def test_sources_older_than_the_window_are_still_rebuilt(self, tmp_path):
        # 200 days old and archived — invisible to _create_episodes, which
        # looks back 30 days and filters archived_at IS NULL.
        db, ids, store = await _corrupt_fixture(
            tmp_path, age_days=200, archived=True
        )
        c = _consolidator(db, store, _RecordingGraphStore(db))
        try:
            episodes, _ = await c._create_episodes()
            assert episodes == [], (
                "precondition: clustering cannot see these rows, so a "
                "delete-and-rebuild repair would lose them permanently"
            )

            report = await c.repair_ciphertext_episodes()

            assert report["repaired"] == 1
            title = await db.fetchval(
                "SELECT title FROM memory_episodes WHERE id = ?",
                ("episode:corrupt",)
            )
            assert "ksav2" not in title.lower()
            assert ENVELOPE_BODY not in title
            assert "scheduler" in title.lower()
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_recorded_provenance_is_preserved(self, tmp_path):
        db, ids, store = await _corrupt_fixture(tmp_path)
        c = _consolidator(db, store, _RecordingGraphStore(db))
        try:
            await c.repair_ciphertext_episodes()
            assert json.loads(await db.fetchval(
                "SELECT key_message_ids FROM memory_episodes WHERE id = ?",
                ("episode:corrupt",)
            )) == ids
        finally:
            await db.close()


class TestGraphNodeOrdering:
    """Requirement 3: the visible surface must never be the last thing fixed."""

    @pytest.mark.asyncio
    async def test_node_is_rewritten_before_the_sql_row(self, tmp_path):
        db, ids, store = await _corrupt_fixture(tmp_path)
        graph = _RecordingGraphStore(db)
        c = _consolidator(db, store, graph)
        try:
            await c.repair_ciphertext_episodes()
            assert graph.row_titles_at_write == [CORRUPT_TITLE], (
                "the SQL row must still hold the old title when the node is "
                "written — otherwise the row was updated first"
            )
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_row_is_not_updated_when_the_node_write_fails(self, tmp_path):
        db, ids, store = await _corrupt_fixture(tmp_path)
        c = _consolidator(db, store, _RecordingGraphStore(db, fail=True))
        try:
            report = await c.repair_ciphertext_episodes()

            assert report["repaired"] == 0
            assert [u["reason"] for u in report["unrepairable"]] == ["write_failed"]
            assert await db.fetchval(
                "SELECT title FROM memory_episodes WHERE id = ?",
                ("episode:corrupt",)
            ) == CORRUPT_TITLE, (
                "a clean row behind a poisoned panel is worse than a wholly "
                "corrupt episode: nothing is left to re-detect from"
            )
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_no_node_id_churn_so_no_orphan_duplicate(self, tmp_path):
        db, ids, store = await _corrupt_fixture(tmp_path)
        c = _consolidator(db, store, _RecordingGraphStore(db))
        try:
            await c.repair_ciphertext_episodes()
            nodes = await db.fetchall(
                "SELECT node_id, label FROM graph_nodes WHERE node_type = 'episode'"
            )
            assert len(nodes) == 1, "a rebuilt episode must not mint a second node"
            assert nodes[0][0] == "episode:corrupt"
            assert "ksav2" not in nodes[0][1].lower()
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_stale_node_with_no_graph_store_refuses_repair(self, tmp_path):
        db, ids, store = await _corrupt_fixture(tmp_path)
        c = _consolidator(db, store, graph_store=None)
        try:
            report = await c.repair_ciphertext_episodes()
            assert report["repaired"] == 0
            assert [u["reason"] for u in report["unrepairable"]] == [
                "graph_store_unavailable"
            ]
            assert await db.fetchval(
                "SELECT title FROM memory_episodes WHERE id = ?",
                ("episode:corrupt",)
            ) == CORRUPT_TITLE
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_episode_without_a_node_repairs_without_a_graph_store(
        self, tmp_path
    ):
        db, ids, store = await _corrupt_fixture(tmp_path, with_node=False)
        c = _consolidator(db, store, graph_store=None)
        try:
            report = await c.repair_ciphertext_episodes()
            assert report["repaired"] == 1
        finally:
            await db.close()


class TestReEmbedding:
    """Requirement 4: the poisoned vector must be replaced, not hidden."""

    @pytest.mark.asyncio
    async def test_repaired_episode_is_re_embedded(self, tmp_path):
        db, ids, store = await _corrupt_fixture(tmp_path)
        embedded = []

        service = MagicMock()

        async def _aembed(text):
            embedded.append(text)
            return [0.25, 0.5, 0.75]

        service.aembed = _aembed
        service.current_profile_id = MagicMock(return_value="profile-x")

        c = _consolidator(db, store, _RecordingGraphStore(db))
        c._get_embedding_service = lambda: service
        try:
            await c.repair_ciphertext_episodes()

            assert embedded, "repair must re-derive the vector"
            assert ENVELOPE_BODY not in embedded[0]
            assert "scheduler" in embedded[0].lower()

            row = (await db.fetchall(
                "SELECT embedding_vec, embedding_profile_id, "
                "excluded_from_context FROM memory_episodes WHERE id = ?",
                ("episode:corrupt",)
            ))[0]
            assert row[0] != b"poisoned-vector"
            assert row[1] == "profile-x"
            assert row[2] == 0, "repair replaces the vector, it does not hide the row"
        finally:
            await db.close()


class TestFailsClosed:
    @pytest.mark.asyncio
    async def test_missing_sources_are_reported_not_guessed(self, tmp_path):
        db, ids, store = await _corrupt_fixture(tmp_path)
        await db.execute(
            "UPDATE conversation_history SET deleted_at = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), ids[0]),
        )
        c = _consolidator(db, store, _RecordingGraphStore(db))
        try:
            report = await c.repair_ciphertext_episodes()
            assert report["repaired"] == 0
            assert report["unrepairable"][0]["reason"] == "sources_missing"
            assert report["unrepairable"][0]["missing"] == 1
            assert await db.fetchval(
                "SELECT title FROM memory_episodes WHERE id = ?",
                ("episode:corrupt",)
            ) == CORRUPT_TITLE
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_undecryptable_sources_are_reported_not_rewritten(self, tmp_path):
        db, ids, _ = await _corrupt_fixture(tmp_path)
        store = MagicMock()
        store.decrypt_stored_content.side_effect = RuntimeError("no key")
        c = _consolidator(db, store, _RecordingGraphStore(db))
        try:
            report = await c.repair_ciphertext_episodes()
            assert report["repaired"] == 0
            assert report["unrepairable"][0]["reason"] == "undecryptable_sources"
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_no_conversation_store_wired_repairs_nothing(self, tmp_path):
        """Without a decryptor every row fails closed — never rewritten blind."""
        db, ids, _ = await _corrupt_fixture(tmp_path)
        c = _consolidator(db, store=None, graph_store=_RecordingGraphStore(db))
        try:
            report = await c.repair_ciphertext_episodes()
            assert report["repaired"] == 0
            assert report["unrepairable"][0]["reason"] == "undecryptable_sources"
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_volatile_privacy_mode_blocks_the_write(self, tmp_path):
        db, ids, store = await _corrupt_fixture(tmp_path)
        policy = MagicMock()
        policy.allows_persistent_writes.return_value = False
        c = MemoryConsolidator(
            db=db, agent_id=AGENT, graph_store=_RecordingGraphStore(db),
            conversation_store=store, persist_policy=policy,
        )
        try:
            report = await c.repair_ciphertext_episodes()
            assert report.get("skipped") is True
            assert report["skipped_reason"] == "privacy_mode_forbids_persistence"
            assert await db.fetchval(
                "SELECT title FROM memory_episodes WHERE id = ?",
                ("episode:corrupt",)
            ) == CORRUPT_TITLE
        finally:
            await db.close()


class TestDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_reports_without_writing(self, tmp_path):
        db, ids, store = await _corrupt_fixture(tmp_path)
        graph = _RecordingGraphStore(db)
        c = _consolidator(db, store, graph)
        try:
            report = await c.repair_ciphertext_episodes(dry_run=True)

            assert report["dry_run"] is True
            assert report["repaired"] == 0
            assert len(report["planned"]) == 1
            plan = report["planned"][0]
            assert plan["episode_id"] == "episode:corrupt"
            assert plan["old_title"] == CORRUPT_TITLE
            assert "ksav2" not in plan["new_title"].lower()
            assert "ksav2" in plan["evidence_terms"]

            assert graph.row_titles_at_write == [], "dry run must not touch the KG"
            assert await db.fetchval(
                "SELECT title FROM memory_episodes WHERE id = ?",
                ("episode:corrupt",)
            ) == CORRUPT_TITLE
        finally:
            await db.close()


class TestReviewFindings:
    """Defects found in codex review r1, each pinned by its failure mode."""

    @pytest.mark.asyncio
    async def test_withdrawn_episodes_are_never_repaired(self, tmp_path):
        """excluded_from_context = 1 means the lifecycle took this content away.

        Rewriting the graph node from freshly decrypted sources would put
        withdrawn topics back in the Memories panel. The ciphertext title is
        the safer state for a row nobody may read.
        """
        db, ids, store = await _corrupt_fixture(tmp_path)
        await db.execute(
            "UPDATE memory_episodes SET excluded_from_context = 1 WHERE id = ?",
            ("episode:corrupt",),
        )
        graph = _RecordingGraphStore(db)
        c = _consolidator(db, store, graph)
        try:
            report = await c.repair_ciphertext_episodes()

            assert report["scanned"] == 0
            assert report["repaired"] == 0
            assert graph.row_titles_at_write == [], "the KG node must not be touched"
            assert await db.fetchval(
                "SELECT label FROM graph_nodes WHERE node_id = ?",
                ("episode:corrupt",)
            ) == CORRUPT_TITLE
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_message_ids_are_bound_as_integers(self, tmp_path):
        """conversation_history.id is an integer column.

        key_message_ids persists strings, and on PostgreSQL asyncpg rejects a
        text parameter against an integer column — which would make every
        candidate a lookup failure and disable repair on that backend
        entirely. Assert the binding type directly; SQLite is too permissive
        to catch this on its own.
        """
        db, ids, store = await _corrupt_fixture(tmp_path)
        bound: list = []
        original = db.fetchall

        async def _spy(sql, params=()):
            if "FROM conversation_history" in sql and " id IN (" in sql:
                bound.extend(params[1:])
            return await original(sql, params)

        db.fetchall = _spy
        c = _consolidator(db, store, _RecordingGraphStore(db))
        try:
            report = await c.repair_ciphertext_episodes()

            assert report["repaired"] == 1
            assert bound, "the rehydration query never ran"
            assert all(isinstance(b, int) for b in bound), (
                f"message IDs must bind as integers, got {[type(b) for b in bound]}"
            )
        finally:
            db.fetchall = original
            await db.close()

    @pytest.mark.asyncio
    async def test_poisoned_vector_is_cleared_even_with_no_embedder(self, tmp_path):
        """The repair must not strand a base64-derived vector.

        _embed_episode is best-effort and no-ops without a provider. Since the
        repaired title no longer matches the pre-filter, the episode is never
        selected again — so if the old vector survived that write it would sit
        in the shared embedding space permanently.
        """
        db, ids, store = await _corrupt_fixture(tmp_path)
        c = _consolidator(db, store, _RecordingGraphStore(db))
        c._get_embedding_service = lambda: None  # no provider configured
        try:
            report = await c.repair_ciphertext_episodes()
            assert report["repaired"] == 1

            row = (await db.fetchall(
                "SELECT embedding_vec, embedding_profile_id "
                "FROM memory_episodes WHERE id = ?", ("episode:corrupt",)
            ))[0]
            assert row[0] is None, "the ciphertext-derived vector must not survive"
            assert row[1] is None

            second = await c.repair_ciphertext_episodes()
            assert second["scanned"] == 0, (
                "precondition: the episode is genuinely unreachable afterwards, "
                "which is why the vector had to be cleared in the same write"
            )
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_repair_titles_never_reach_the_log(self, tmp_path, caplog):
        db, ids, store = await _corrupt_fixture(tmp_path)
        c = _consolidator(db, store, _RecordingGraphStore(db))
        try:
            with caplog.at_level("INFO"):
                await c.repair_ciphertext_episodes()
            logged = "\n".join(r.getMessage() for r in caplog.records)
            assert "episode:corrupt" in logged, "the outcome should still be logged"
            assert ENVELOPE_BODY not in logged
            assert "scheduler" not in logged.lower(), (
                "user-derived topics must not reach an ungoverned log surface"
            )
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_exhausting_the_budget_is_reported_not_silent(self, tmp_path):
        db, ids, store = await _corrupt_fixture(tmp_path)
        c = _consolidator(db, store, _RecordingGraphStore(db))
        try:
            report = await c.repair_ciphertext_episodes(limit=1)
            assert report["limit_reached"] is True

            fresh = await c.repair_ciphertext_episodes(limit=50)
            assert "limit_reached" not in fresh
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_exhausted_budget_reaches_the_consolidation_report(self, tmp_path):
        """run_consolidation is the only production caller — the marker dies
        there or it is never seen at all."""
        db, ids, store = await _corrupt_fixture(tmp_path)
        c = _consolidator(db, store, _RecordingGraphStore(db))
        original = c.repair_ciphertext_episodes

        async def _capped(**kwargs):
            return await original(limit=1)

        c.repair_ciphertext_episodes = _capped
        try:
            report = await c.run_consolidation()
            assert report["episode_repair_limit_reached"] is True
        finally:
            await db.close()


class TestRacesDuringRepair:
    """codex review r2: repair checks state, then awaits, then writes."""

    @pytest.mark.asyncio
    async def test_withdrawal_mid_repair_does_not_persist(self, tmp_path):
        """The lifecycle withdraws the episode while its sources decrypt.

        The row update must lose the race rather than overwrite it, and the
        graph node must be put back — a withdrawn episode showing
        reconstructed topics in the Memories panel is the exact leak the
        exclusion exists to prevent.
        """
        db, ids, store = await _corrupt_fixture(tmp_path)
        graph = _RecordingGraphStore(db)

        # Withdraw at the moment the node is rewritten: after selection and
        # decryption, before the row update.
        original_add = graph.add_node

        async def _withdraw_then_write(node):
            await db.execute(
                "UPDATE memory_episodes SET excluded_from_context = 1 "
                "WHERE id = ?", (node.node_id,)
            )
            graph.add_node = original_add  # only race the first write
            return await original_add(node)

        graph.add_node = _withdraw_then_write
        c = _consolidator(db, store, graph)
        try:
            report = await c.repair_ciphertext_episodes()

            assert report["repaired"] == 0
            assert [u["reason"] for u in report["unrepairable"]] == [
                "abandoned_mid_repair"
            ]
            assert await db.fetchval(
                "SELECT title FROM memory_episodes WHERE id = ?",
                ("episode:corrupt",)
            ) == CORRUPT_TITLE
            assert await db.fetchval(
                "SELECT label FROM graph_nodes WHERE node_id = ?",
                ("episode:corrupt",)
            ) == CORRUPT_TITLE, "the node rewrite must have been rolled back"
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_privacy_flip_mid_repair_does_not_persist(self, tmp_path):
        """A transition to a volatile mode lands during decryption.

        The raw memory_episodes write is governed by no graph proxy, so only
        a re-check under the transition lock stops it.
        """
        db, ids, store = await _corrupt_fixture(tmp_path, with_node=False)
        policy = MagicMock()
        # Allowed at the start of the pass, forbidden by the time we write.
        policy.allows_persistent_writes.side_effect = [True, False, False, False]
        c = MemoryConsolidator(
            db=db, agent_id=AGENT, graph_store=None,
            conversation_store=store, persist_policy=policy,
        )
        try:
            report = await c.repair_ciphertext_episodes()

            assert report["repaired"] == 0
            assert [u["reason"] for u in report["unrepairable"]] == [
                "abandoned_mid_repair"
            ]
            assert await db.fetchval(
                "SELECT title FROM memory_episodes WHERE id = ?",
                ("episode:corrupt",)
            ) == CORRUPT_TITLE
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_transition_lock_is_held_across_the_writes(self, tmp_path):
        db, ids, store = await _corrupt_fixture(tmp_path)
        held = {"during_write": False}

        class _Lock:
            def __init__(self):
                self.depth = 0

            async def __aenter__(self):
                self.depth += 1

            async def __aexit__(self, *exc):
                self.depth -= 1

        lock = _Lock()
        graph = _RecordingGraphStore(db)
        original_add = graph.add_node

        async def _observe(node):
            held["during_write"] = lock.depth > 0
            return await original_add(node)

        graph.add_node = _observe
        c = MemoryConsolidator(
            db=db, agent_id=AGENT, graph_store=graph,
            conversation_store=store, transition_lock=lock,
        )
        try:
            report = await c.repair_ciphertext_episodes()
            assert report["repaired"] == 1
            assert held["during_write"], (
                "durable writes must happen inside the privacy-transition lock"
            )
            assert lock.depth == 0, "the lock must be released"
        finally:
            await db.close()


class TestRoundThreeFindings:
    @pytest.mark.asyncio
    async def test_failed_rollback_deletes_the_node_rather_than_exposing_it(
        self, tmp_path
    ):
        """A swallowed rollback error leaves withdrawn topics on the panel.

        The caller would report the repair abandoned while the node still
        showed content reconstructed from sources nobody may read.
        """
        db, ids, store = await _corrupt_fixture(tmp_path)
        graph = _RecordingGraphStore(db)
        deleted: list = []

        async def _delete(node_id):
            deleted.append(node_id)
            await db.execute(
                "DELETE FROM graph_nodes WHERE node_id = ?", (node_id,)
            )

        graph.delete_node = _delete
        original_add = graph.add_node
        calls = {"n": 0}

        async def _withdraw_then_fail_rollback(node):
            calls["n"] += 1
            if calls["n"] == 1:
                await db.execute(
                    "UPDATE memory_episodes SET excluded_from_context = 1 "
                    "WHERE id = ?", (node.node_id,)
                )
                return await original_add(node)
            raise RuntimeError("rollback write refused")

        graph.add_node = _withdraw_then_fail_rollback
        c = _consolidator(db, store, graph)
        try:
            report = await c.repair_ciphertext_episodes()

            assert report["repaired"] == 0
            assert deleted == ["episode:corrupt"], (
                "an unrestorable node must be removed, not left exposed"
            )
            assert await db.fetchval(
                "SELECT COUNT(*) FROM graph_nodes WHERE node_id = ?",
                ("episode:corrupt",)
            ) == 0
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_cleared_candidates_never_starve_a_corrupt_one(self, tmp_path):
        """The budget bounds repairs, not the scan.

        A handful of permanent false positives at the head of created_at order
        must not hide a corrupt episode behind them forever.
        """
        db = await _make_db(tmp_path)

        # Three healthy envelope-format discussions, created FIRST.
        healthy_ids = await _add_messages(
            db, [HEALTHY_ENVELOPE, HEALTHY_ENVELOPE], encrypted=True, age_days=9
        )
        for n in range(3):
            await _add_episode(
                db, f"episode:healthy-{n}",
                "Discussion of ksav2, base64, prefix, aead",
                "Topics: ksav2, base64, prefix, aead.", healthy_ids,
            )

        # The corrupt one, created LAST — behind all of them.
        corrupt_ids = await _add_messages(
            db, [ENVELOPE, ENVELOPE, ENVELOPE], age_days=1
        )
        await _add_episode(db, "episode:corrupt", CORRUPT_TITLE,
                           CORRUPT_SUMMARY, corrupt_ids)

        seq = iter(PLAINTEXTS * 8)
        healthy_seq = iter(HEALTHY_PLAINTEXTS * 8)
        store = MagicMock()
        store.decrypt_stored_content.side_effect = lambda content, meta: (
            next(seq) if content == ENVELOPE
            else next(healthy_seq) if content == HEALTHY_ENVELOPE
            else content
        )

        c = _consolidator(db, store, _RecordingGraphStore(db))
        try:
            report = await c.repair_ciphertext_episodes(limit=1)

            # The invariant, stated without depending on scan order: a repair
            # budget of ONE must not stop three other candidates being
            # examined. A scan-bounded budget would surface exactly one row.
            assert report["scanned"] == 4, (
                "examination must not be bounded by the repair budget, or "
                "permanent false positives can hide corrupt episodes forever"
            )
            assert report["repaired"] == 1
            assert await db.fetchval(
                "SELECT title FROM memory_episodes WHERE id = ?",
                ("episode:corrupt",)
            ) != CORRUPT_TITLE
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_malformed_metadata_costs_one_episode_not_the_pass(
        self, tmp_path
    ):
        db = await _make_db(tmp_path)

        # A candidate whose metadata carries a null intensity, created first.
        bad_ids = await _add_messages(db, [ENVELOPE, ENVELOPE, ENVELOPE],
                                      age_days=9)
        for mid in bad_ids:
            await db.execute(
                "UPDATE conversation_history SET metadata = ? WHERE id = ?",
                (json.dumps({**ENC_META, "emotional_intensity": None,
                             "emotional_valence": None}), mid),
            )
        await _add_episode(db, "episode:bad", CORRUPT_TITLE, CORRUPT_SUMMARY,
                           bad_ids)

        good_ids = await _add_messages(db, [ENVELOPE, ENVELOPE, ENVELOPE],
                                       age_days=1)
        await _add_episode(db, "episode:good", CORRUPT_TITLE, CORRUPT_SUMMARY,
                           good_ids)

        seq = iter(PLAINTEXTS * 8)
        store = MagicMock()
        store.decrypt_stored_content.side_effect = (
            lambda content, meta: next(seq) if content == ENVELOPE else content
        )
        c = _consolidator(db, store, _RecordingGraphStore(db))
        try:
            report = await c.repair_ciphertext_episodes()

            assert report["repaired"] == 1, (
                "the healthy candidate behind the malformed one must still be "
                "repaired"
            )
            assert await db.fetchval(
                "SELECT title FROM memory_episodes WHERE id = ?",
                ("episode:good",)
            ) != CORRUPT_TITLE
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_sleep_report_carries_repair_diagnostics(self):
        """SleepReport.to_dict is what the scheduler returns to an operator."""
        from kestrel_sovereign.agent.sleep import SleepReport

        report = SleepReport(success=True)
        report.episodes_repaired = 3
        report.episode_repair_limit_reached = True

        consolidation = report.to_dict()["consolidation"]
        assert consolidation["episodes_repaired"] == 3
        assert consolidation["episode_repair_limit_reached"] is True


class TestRoundFourFindings:
    @pytest.mark.asyncio
    async def test_corruption_is_detected_after_ciphertext_rotation(
        self, tmp_path
    ):
        """KeyRotationService re-encrypts conversation_history.content.

        Afterwards the stored envelope shares only the magic prefix with the
        body tokens baked into the corrupt title. Requiring a match against the
        CURRENT ciphertext would clear a genuinely corrupt episode forever.
        """
        db, ids, store = await _corrupt_fixture(tmp_path)
        # Re-encrypt: same plaintext, entirely different envelope bytes.
        rotated = "KSAv2:zzq7rt4mplk9wwvv2xh8ddn3+ppl0aa5ssq2mmz7ttx4vv9bb1cc"
        await db.execute(
            "UPDATE conversation_history SET content = ? WHERE agent_id = ?",
            (rotated, AGENT),
        )
        seq = iter(PLAINTEXTS * 4)
        store.decrypt_stored_content.side_effect = (
            lambda content, meta: next(seq) if content == rotated else content
        )
        c = _consolidator(db, store, _RecordingGraphStore(db))
        try:
            report = await c.repair_ciphertext_episodes()

            assert report["repaired"] == 1, (
                "rotation must not make a corrupt episode undetectable"
            )
            title = await db.fetchval(
                "SELECT title FROM memory_episodes WHERE id = ?",
                ("episode:corrupt",)
            )
            assert ENVELOPE_BODY not in title
            assert "scheduler" in title.lower()
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_healthy_episode_survives_rotation_too(self, tmp_path):
        """The rotation fallback must not weaken the healthy-episode guard."""
        db = await _make_db(tmp_path)
        ids = await _add_messages(
            db, [HEALTHY_ENVELOPE, HEALTHY_ENVELOPE], encrypted=True
        )
        title = "Discussion of ksav2, base64, prefix, aead"
        await _add_episode(
            db, "episode:healthy", title,
            "A conversation with 2 messages (1 from user). Topics: ksav2, "
            "base64, prefix, aead. Emotional trajectory: emotionally steady.",
            ids,
        )
        rotated = "KSAv2:qqw8ee4rr7tt2yy5uu1ii3oo9pp6aa0ss"
        await db.execute(
            "UPDATE conversation_history SET content = ? WHERE agent_id = ?",
            (rotated, AGENT),
        )
        talk = iter(HEALTHY_PLAINTEXTS * 4)
        store = MagicMock()
        store.decrypt_stored_content.side_effect = (
            lambda content, meta: next(talk) if content == rotated else content
        )
        c = _consolidator(db, store, _RecordingGraphStore(db))
        try:
            report = await c.repair_ciphertext_episodes()
            assert report["repaired"] == 0
            assert report["cleared"] == 1
            assert await db.fetchval(
                "SELECT title FROM memory_episodes WHERE id = ?",
                ("episode:healthy",)
            ) == title
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_source_trashed_mid_repair_does_not_persist(self, tmp_path):
        """Soft deletion only stamps conversation_history.deleted_at.

        It never touches the episode's exclusion flag, so no predicate on
        memory_episodes alone would notice; the write would publish topics
        drawn from a message the user just moved to Trash.
        """
        db, ids, store = await _corrupt_fixture(tmp_path)
        graph = _RecordingGraphStore(db)
        original_add = graph.add_node

        async def _trash_then_write(node):
            await db.execute(
                "UPDATE conversation_history SET deleted_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), ids[0]),
            )
            graph.add_node = original_add
            return await original_add(node)

        graph.add_node = _trash_then_write
        c = _consolidator(db, store, graph)
        try:
            report = await c.repair_ciphertext_episodes()

            assert report["repaired"] == 0
            assert [u["reason"] for u in report["unrepairable"]] == [
                "abandoned_mid_repair"
            ]
            assert await db.fetchval(
                "SELECT title FROM memory_episodes WHERE id = ?",
                ("episode:corrupt",)
            ) == CORRUPT_TITLE
            assert await db.fetchval(
                "SELECT label FROM graph_nodes WHERE node_id = ?",
                ("episode:corrupt",)
            ) == CORRUPT_TITLE, "the node rewrite must have been rolled back"
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_sleep_report_surfaces_a_failing_repair_pass(self):
        """"0 repaired" must not read the same as "nothing to repair"."""
        from kestrel_sovereign.agent.sleep import SleepReport

        report = SleepReport(success=True)
        report.episode_repair_failed = True
        assert report.to_dict()["consolidation"]["episode_repair_failed"] is True

    @pytest.mark.asyncio
    async def test_consolidation_records_a_repair_failure(self, tmp_path):
        db, ids, store = await _corrupt_fixture(tmp_path)
        c = _consolidator(db, store, _RecordingGraphStore(db))

        async def _boom(**kwargs):
            raise RuntimeError("pass exploded")

        c.repair_ciphertext_episodes = _boom
        try:
            result = await c.run_consolidation()
            assert result["episode_repair_error"] == "pass exploded"
            assert result["episodes_repaired"] == 0
        finally:
            await db.close()


class TestRoundFiveFindings:
    @pytest.mark.asyncio
    async def test_hard_purged_source_blocks_the_write(self, tmp_path):
        """A purged row cannot satisfy `deleted_at IS NOT NULL`.

        Testing only for soft deletion would wave a hard purge straight
        through, publishing topics decrypted from a message that no longer
        exists at all.
        """
        db, ids, store = await _corrupt_fixture(tmp_path)
        graph = _RecordingGraphStore(db)
        original_add = graph.add_node

        async def _purge_then_write(node):
            await db.execute(
                "DELETE FROM conversation_history WHERE id = ?", (ids[0],)
            )
            graph.add_node = original_add
            return await original_add(node)

        graph.add_node = _purge_then_write
        c = _consolidator(db, store, graph)
        try:
            report = await c.repair_ciphertext_episodes()

            assert report["repaired"] == 0
            assert [u["reason"] for u in report["unrepairable"]] == [
                "abandoned_mid_repair"
            ]
            assert await db.fetchval(
                "SELECT title FROM memory_episodes WHERE id = ?",
                ("episode:corrupt",)
            ) == CORRUPT_TITLE
            assert await db.fetchval(
                "SELECT label FROM graph_nodes WHERE node_id = ?",
                ("episode:corrupt",)
            ) == CORRUPT_TITLE
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_node_is_restored_when_the_row_update_raises(self, tmp_path):
        """A raising row update must not leave the rebuilt narrative on the panel.

        Control jumps to the outer write_failed handler, so without an explicit
        undo the node keeps the plaintext-derived title — and a withdrawal
        landing afterwards makes that permanent, since the row is then excluded
        from every future scan.
        """
        db, ids, store = await _corrupt_fixture(tmp_path)
        graph = _RecordingGraphStore(db)
        c = _consolidator(db, store, graph)

        original_execute = db.execute

        async def _fail_the_episode_update(sql, params=()):
            if sql.startswith("UPDATE memory_episodes SET title"):
                raise RuntimeError("lock timeout")
            return await original_execute(sql, params)

        db.execute = _fail_the_episode_update
        try:
            report = await c.repair_ciphertext_episodes()

            assert report["repaired"] == 0
            assert [u["reason"] for u in report["unrepairable"]] == [
                "write_failed"
            ]
            db.execute = original_execute
            assert await db.fetchval(
                "SELECT label FROM graph_nodes WHERE node_id = ?",
                ("episode:corrupt",)
            ) == CORRUPT_TITLE, (
                "the node must not keep the rebuilt narrative after a failed "
                "row update"
            )
        finally:
            db.execute = original_execute
            await db.close()

    @pytest.mark.asyncio
    async def test_every_candidate_is_examined_below_the_scan_limit(
        self, tmp_path
    ):
        """Randomising the scan must not cost coverage in the normal case."""
        db, ids, store = await _corrupt_fixture(tmp_path)
        healthy_ids = await _add_messages(
            db, [HEALTHY_ENVELOPE, HEALTHY_ENVELOPE], encrypted=True
        )
        for n in range(4):
            await _add_episode(
                db, f"episode:healthy-{n}",
                "Discussion of ksav2, base64, prefix, aead",
                "Topics: ksav2, base64, prefix, aead.", healthy_ids,
            )
        healthy_talk = iter(HEALTHY_PLAINTEXTS * 16)
        seq = iter(PLAINTEXTS * 16)
        store.decrypt_stored_content.side_effect = lambda content, meta: (
            next(seq) if content == ENVELOPE
            else next(healthy_talk) if content == HEALTHY_ENVELOPE
            else content
        )
        c = _consolidator(db, store, _RecordingGraphStore(db))
        try:
            report = await c.repair_ciphertext_episodes()
            assert report["scanned"] == 5
            assert report["cleared"] == 4
            assert report["repaired"] == 1
        finally:
            await db.close()


class TestLockReachesTheConsolidator:
    """The lock is only useful if the real construction path delivers it.

    MemorySystem builds the consolidator in initialize(), not __init__, so a
    lock accepted by the constructor and never stored never arrives. Tests that
    build MemoryConsolidator directly cannot see that.
    """

    @pytest.mark.asyncio
    async def test_memory_system_forwards_the_lock_to_the_consolidator(
        self, tmp_path
    ):
        from kestrel_sovereign.storage.async_storage import AsyncStorage
        from kestrel_sovereign.storage.memory_system import MemorySystem

        sentinel = object()
        storage = AsyncStorage(db_path=str(tmp_path / "ms.db"), agent_id=AGENT)
        await storage.initialize()
        ms = MemorySystem(
            storage=storage, agent_id=AGENT, transition_lock=sentinel
        )
        try:
            await ms.initialize()
            assert ms.consolidator._transition_lock is sentinel
        finally:
            await storage.close()


class TestWiredIntoConsolidation:
    """Repair that nothing calls is repair that never happens."""
    """Repair that nothing calls is repair that never happens."""

    @pytest.mark.asyncio
    async def test_nightly_consolidation_runs_the_repair_pass(self, tmp_path):
        db, ids, store = await _corrupt_fixture(tmp_path)
        c = _consolidator(db, store, _RecordingGraphStore(db))
        try:
            report = await c.run_consolidation()

            assert report.get("episodes_repaired") == 1
            assert await db.fetchval(
                "SELECT title FROM memory_episodes WHERE id = ?",
                ("episode:corrupt",)
            ) != CORRUPT_TITLE
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_repair_failure_does_not_abort_consolidation(self, tmp_path):
        db, ids, store = await _corrupt_fixture(tmp_path)
        c = _consolidator(db, store, _RecordingGraphStore(db))

        async def _boom(**kwargs):
            raise RuntimeError("repair exploded")

        c.repair_ciphertext_episodes = _boom
        try:
            report = await c.run_consolidation()

            assert "error" not in report, "consolidation must survive a repair failure"
            assert report["episodes_repaired"] == 0
            assert report["episode_repair_error"] == "repair exploded"
            assert "total_messages_processed" in report, (
                "the steps after repair must still run"
            )
        finally:
            await db.close()


class TestIdempotence:
    @pytest.mark.asyncio
    async def test_second_pass_finds_nothing_to_do(self, tmp_path):
        db, ids, store = await _corrupt_fixture(tmp_path)
        c = _consolidator(db, store, _RecordingGraphStore(db))
        try:
            first = await c.repair_ciphertext_episodes()
            assert first["repaired"] == 1

            second = await c.repair_ciphertext_episodes()
            assert second["scanned"] == 0, (
                "a repaired episode must fall out of the pre-filter entirely"
            )
            assert second["repaired"] == 0
        finally:
            await db.close()
