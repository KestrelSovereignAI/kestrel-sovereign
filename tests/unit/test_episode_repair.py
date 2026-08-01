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
ENVELOPE = f"KSAv2:{ENVELOPE_BODY}"
CORRUPT_TITLE = f"Discussion of ksav2, {ENVELOPE_BODY}"
CORRUPT_SUMMARY = (
    f"A conversation with 3 messages (2 from user). Topics: ksav2, "
    f"{ENVELOPE_BODY}. Emotional trajectory: emotionally steady."
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


class TestWiredIntoConsolidation:
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
