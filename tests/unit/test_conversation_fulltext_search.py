"""Full-text search over conversations (conversations-pane server search).

The pane's search box used to filter client-side on name + preview only —
message content was unsearchable. These tests cover the new first-class path:

  - ``search_session_summaries`` (pure core): groups decrypted messages with
    the shared #2019 boundary algorithm and returns only matching sessions,
    decorated with match_count / match_role / match_snippet. Matching reuses
    ``search_history`` semantics: wrapper-stripped substring first
    (#1537/#1549), tokenized fallback, wrapper-only queries gated (#1554).
  - ``AsyncConversationStore.search_sessions``: real-SQLite scan that
    decrypts client-side (SQL LIKE cannot see encrypted content) and
    respects the active/archived view split (#2149).
  - ``PrivacyEnforcingStorage.search_conversations``: EPHEMERAL returns
    nothing; ISOLATED searches only the in-memory session buffer.
  - ``GET /api/conversations?q=``: endpoint dispatch + response shape.
"""

import json
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from kestrel_sovereign.storage.async_conversation_store import (
    AsyncConversationStore,
    search_session_summaries,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase

BASE = datetime(2026, 7, 1, 12, 0, 0)


def _msg(i, role, content, *, minutes=0, session_id=None):
    meta = {}
    if session_id is not None:
        meta["session_id"] = session_id
    return {
        "id": i,
        "role": role,
        "content": content,
        "metadata": meta,
        "created_at": BASE + timedelta(minutes=minutes),
    }


# ---------------------------------------------------------------------------
# search_session_summaries — pure core
# ---------------------------------------------------------------------------

class TestSearchSessionSummaries:
    def test_matches_only_sessions_containing_query(self):
        msgs = [
            _msg(1, "user", "let's talk about penguins", session_id="s1"),
            _msg(2, "assistant", "penguins are birds", minutes=1, session_id="s1"),
            _msg(3, "user", "now about databases", minutes=90, session_id="s2"),
            _msg(4, "assistant", "sqlite is a database", minutes=91, session_id="s2"),
        ]
        results = search_session_summaries(msgs, "penguins")
        assert [s["session_id"] for s in results] == ["s1"]
        assert results[0]["match_count"] == 2
        assert "penguins" in results[0]["match_snippet"].lower()
        assert results[0]["match_role"] == "user"

    def test_assistant_content_is_searchable(self):
        msgs = [
            _msg(1, "user", "what is the capital", session_id="s1"),
            _msg(2, "assistant", "the capital is Ouagadougou", minutes=1, session_id="s1"),
        ]
        results = search_session_summaries(msgs, "ouagadougou")
        assert len(results) == 1
        assert results[0]["match_role"] == "assistant"

    def test_title_only_match(self):
        msgs = [_msg(1, "user", "unrelated content", session_id="s1")]
        results = search_session_summaries(
            msgs, "budget", names={"s1": "Budget planning"}
        )
        assert len(results) == 1
        assert results[0]["name"] == "Budget planning"
        assert results[0]["match_count"] == 0
        assert results[0]["match_snippet"] is None

    def test_no_match_returns_empty(self):
        msgs = [_msg(1, "user", "hello world", session_id="s1")]
        assert search_session_summaries(msgs, "zebra") == []

    def test_empty_query_returns_empty(self):
        msgs = [_msg(1, "user", "hello world", session_id="s1")]
        assert search_session_summaries(msgs, "   ") == []

    def test_newest_first_and_limit(self):
        msgs = []
        for n in range(3):
            msgs.append(_msg(n * 2 + 1, "user", f"kestrel topic {n}",
                             minutes=n * 120, session_id=f"s{n}"))
        results = search_session_summaries(msgs, "kestrel", limit=2)
        assert [s["session_id"] for s in results] == ["s2", "s1"]

    def test_snippet_is_centered_on_hit(self):
        long = ("x" * 300) + " the needle sits here " + ("y" * 300)
        msgs = [_msg(1, "user", long, session_id="s1")]
        results = search_session_summaries(msgs, "needle")
        snippet = results[0]["match_snippet"]
        assert "needle" in snippet
        assert len(snippet) < 200  # excerpt, not the whole message
        assert snippet.startswith("…") and snippet.endswith("…")

    def test_wrapper_content_not_matchable(self):
        """Retrieved-context transport must not make a session searchable
        (#1537/#1549) — same projection search_history uses."""
        msgs = [
            _msg(1, "user",
                 "<retrieved_context>secret zebra facts</retrieved_context>\n"
                 "<user_input>\ntell me about weather\n</user_input>",
                 session_id="s1"),
        ]
        assert search_session_summaries(msgs, "zebra") == []
        # ...but the canonical user text is matchable, and the snippet is
        # wrapper-free.
        results = search_session_summaries(msgs, "weather")
        assert len(results) == 1
        assert "retrieved_context" not in results[0]["match_snippet"]

    def test_tokenized_fallback_multi_term(self):
        msgs = [
            _msg(1, "user",
                 "Meridian was the first Kestrel agent with context memory",
                 session_id="s1"),
        ]
        results = search_session_summaries(
            msgs, "Meridian Kestrel memory management"
        )
        assert len(results) == 1
        assert results[0]["match_count"] == 1

    def test_preview_fields_retained_for_endpoint_decoration(self):
        msgs = [_msg(1, "user", "penguin talk", session_id="s1")]
        results = search_session_summaries(msgs, "penguin")
        assert results[0]["preview_content"] == "penguin talk"
        assert "preview_metadata" in results[0]

    def test_marker_only_renamed_session_is_title_searchable(self):
        """A just-started conversation (marker row, no messages yet) is
        list-visible (#2222), so its user-assigned title must be searchable
        too (codex r2 P2)."""
        marker = {
            "id": 1,
            "role": "system",
            "content": "",
            "metadata": {"new_session": True, "session_id": "s-new"},
            "created_at": BASE,
        }
        results = search_session_summaries(
            [marker], "budget", names={"s-new": "Budget planning"}
        )
        assert [s["session_id"] for s in results] == ["s-new"]
        assert results[0]["message_count"] == 0
        assert results[0]["match_snippet"] is None
        # ...and a non-matching title still keeps the empty session out.
        assert search_session_summaries(
            [marker], "zebra", names={"s-new": "Budget planning"}
        ) == []

    def test_wake_only_hit_carries_its_wake_source_for_decoration(self):
        """Search reuses the shared #2019 grouping, so it inherits the #2947
        preview picker: a wake row never fills the preview slot, and the wake
        source rides along so the endpoint can title the card honestly."""
        wake = {
            "id": 1,
            "role": "user",
            "content": "[TALON_JOB_COMPLETE] penguin job finished",
            "metadata": {
                "session_id": "s-wake",
                "signal_wake": {"source": "talon.job_complete", "mode": "cognition"},
            },
            "created_at": BASE,
        }
        results = search_session_summaries([wake], "penguin")
        assert [s["session_id"] for s in results] == ["s-wake"]
        assert results[0]["preview_content"] is None
        assert results[0]["preview_wake_source"] == "talon.job_complete"

    def test_resumed_session_coalesces_to_one_hit(self):
        """A session resumed past the gap must surface as ONE result (#2019)."""
        msgs = [
            _msg(1, "user", "penguin one", minutes=0, session_id="s1"),
            _msg(2, "user", "penguin two", minutes=120, session_id="s1"),
        ]
        results = search_session_summaries(msgs, "penguin")
        assert len(results) == 1
        assert results[0]["match_count"] == 2


# ---------------------------------------------------------------------------
# AsyncConversationStore.search_sessions — real SQLite, encryption, views
# ---------------------------------------------------------------------------

@pytest.fixture
async def store():
    with tempfile.TemporaryDirectory() as tmp:
        db = await AsyncDatabase.sqlite(str(Path(tmp) / "test.db"))
        store = AsyncConversationStore(db, agent_id="test-agent")
        yield store
        await db.close()


@pytest.fixture
async def encrypted_store(monkeypatch):
    monkeypatch.setenv("KESTREL_DATA_KEY", "unit-test-conversation-search-key")
    with tempfile.TemporaryDirectory() as tmp:
        db = await AsyncDatabase.sqlite(str(Path(tmp) / "test.db"))
        store = AsyncConversationStore(db, agent_id="enc-agent")
        yield store
        await db.close()


class TestStoreSearchSessions:
    @pytest.mark.asyncio
    async def test_finds_matching_session_only(self, store):
        await store.add_conversation("user", "all about penguins", session_id="s1")
        await store.add_conversation("assistant", "penguins!", session_id="s1")
        await store.add_conversation("user", "sqlite tuning", session_id="s2")

        results = await store.search_sessions("penguins")
        assert [s["session_id"] for s in results] == ["s1"]
        assert results[0]["match_count"] == 2

    @pytest.mark.asyncio
    async def test_encrypted_content_is_searchable_and_snippet_plaintext(
        self, encrypted_store
    ):
        assert encrypted_store.encryption_enabled
        await encrypted_store.add_conversation(
            "user", "the launch codes are in the penguin folder", session_id="s1"
        )
        # Stored ciphertext must not contain the plaintext.
        row = await encrypted_store.db.fetchall(
            "SELECT content FROM conversation_history", ()
        )
        assert "penguin" not in row[0][0]

        results = await encrypted_store.search_sessions("penguin folder")
        assert len(results) == 1
        assert "penguin folder" in results[0]["match_snippet"]

    @pytest.mark.asyncio
    async def test_archived_view_split(self, store):
        await store.add_conversation("user", "archived penguin", session_id="s1")
        await store.add_conversation("user", "active penguin", session_id="s2")
        await store.db.execute_commit(
            "UPDATE conversation_history SET archived_at = CURRENT_TIMESTAMP "
            "WHERE metadata LIKE ?",
            ('%"session_id": "s1"%',),
        )

        active = await store.search_sessions("penguin", view="active")
        archived = await store.search_sessions("penguin", view="archived")
        assert [s["session_id"] for s in active] == ["s2"]
        assert [s["session_id"] for s in archived] == ["s1"]

    @pytest.mark.asyncio
    async def test_soft_deleted_rows_excluded(self, store):
        await store.add_conversation("user", "trash penguin", session_id="s1")
        await store.clear_history()
        assert await store.search_sessions("penguin") == []

    @pytest.mark.asyncio
    async def test_title_match_via_store(self, store):
        await store.add_conversation("user", "nothing relevant", session_id="s1")
        await store.set_conversation_name("s1", "Penguin plans")
        results = await store.search_sessions("penguin")
        assert len(results) == 1
        assert results[0]["name"] == "Penguin plans"


# ---------------------------------------------------------------------------
# Privacy wrapper — EPHEMERAL / ISOLATED
# ---------------------------------------------------------------------------

class TestPrivacyWrappedSearch:
    @pytest.mark.asyncio
    async def test_ephemeral_returns_nothing(self):
        from kestrel_sovereign.privacy import PrivacyMode
        from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage

        wrapper = PrivacyEnforcingStorage(MagicMock(), PrivacyMode.EPHEMERAL)
        assert await wrapper.search_conversations("a", "penguin") == []

    @pytest.mark.asyncio
    async def test_isolated_searches_session_buffer(self):
        from unittest.mock import AsyncMock

        from kestrel_sovereign.privacy import PrivacyMode
        from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage

        underlying = MagicMock()
        # Persisted titles must NEVER surface in an isolated session, even
        # when a buffer session_id collides with a persisted one (codex r3
        # P1) — titles go through the wrapper's privacy-aware accessor,
        # which returns {} in this mode.
        underlying.get_conversation_names = AsyncMock(
            return_value={"session-local": "Persisted secret title"}
        )
        wrapper = PrivacyEnforcingStorage(underlying, PrivacyMode.ISOLATED)
        await wrapper.add_conversation("user", "penguin in isolation")
        await wrapper.add_conversation("user", "unrelated")

        results = await wrapper.search_conversations("a", "penguin")
        assert len(results) == 1
        assert "penguin" in results[0]["match_snippet"]
        assert "name" not in results[0]
        # A query matching only the persisted title finds nothing here.
        assert await wrapper.search_conversations("a", "persisted secret") == []
        # No archive concept in the buffer.
        assert await wrapper.search_conversations("a", "penguin", view="archived") == []


# ---------------------------------------------------------------------------
# GET /api/conversations?q= — endpoint dispatch
# ---------------------------------------------------------------------------

def _prepare_app(agent):
    from server import app

    @asynccontextmanager
    async def noop_lifespan(_app):
        yield

    original = {
        "lifespan": app.router.lifespan_context,
        "agent": getattr(app.state, "agent", None),
        "manager": getattr(app.state, "agent_manager", None),
    }
    app.router.lifespan_context = noop_lifespan
    app.state.agent = agent
    app.state.agent_manager = None
    return app, original


def _restore_app(app, original):
    app.router.lifespan_context = original["lifespan"]
    app.state.agent = original["agent"]
    app.state.agent_manager = original["manager"]


def test_conversations_endpoint_q_dispatches_to_search():
    calls = {}

    async def fake_search(agent_id, query, limit=20, view="active"):
        calls["args"] = (agent_id, query, limit, view)
        return [{
            "session_id": "s1",
            "started_at": "2026-07-01T12:00:00",
            "last_message_at": "2026-07-01T12:05:00",
            "message_count": 2,
            "user_message_count": 1,
            "preview_content": "penguin talk",
            "preview_metadata": {},
            "match_count": 1,
            "match_role": "user",
            "match_snippet": "penguin talk",
            "name": "Birds",
        }]

    storage = MagicMock()
    storage.agent_id = "agent-1"
    storage.encryption_enabled = False
    storage.search_conversations = fake_search
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        client = TestClient(app)
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            resp = client.get(
                "/api/conversations?q=penguin&limit=7&view=active",
                headers={"X-API-Key": "test-key"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert calls["args"] == ("agent-1", "penguin", 7, "active")
        assert body["query"] == "penguin"
        assert body["total"] == 1
        conv = body["conversations"][0]
        assert conv["match_snippet"] == "penguin talk"
        assert conv["name"] == "Birds"
        # preview decorated from preview_content, raw fields stripped
        assert conv["preview"] == "penguin talk"
        assert "preview_content" not in conv
    finally:
        _restore_app(app, original)


def test_conversations_endpoint_q_redacts_snippets_when_decrypt_false():
    """decrypt=false must not leak plaintext through search results.

    Matching necessarily decrypts server-side, but the RESPONSE must honor
    the caller's no-plaintext request when the store is encrypted at rest:
    snippet redacted, preview blanked + flagged (codex P2).
    """
    async def fake_search(agent_id, query, limit=20, view="active"):
        return [{
            "session_id": "s1",
            "started_at": "2026-07-01T12:00:00",
            "last_message_at": "2026-07-01T12:05:00",
            "message_count": 2,
            "user_message_count": 1,
            "preview_content": "decrypted secret",
            "preview_metadata": {},
            "match_count": 1,
            "match_role": "user",
            "match_snippet": "decrypted secret excerpt",
        }]

    storage = MagicMock()
    storage.agent_id = "agent-1"
    storage.encryption_enabled = True
    storage.search_conversations = fake_search
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        client = TestClient(app)
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            resp = client.get(
                "/api/conversations?q=secret&decrypt=false",
                headers={"X-API-Key": "test-key"},
            )
        assert resp.status_code == 200
        conv = resp.json()["conversations"][0]
        assert conv["match_snippet"] is None
        assert conv["preview"] == ""
        assert conv["preview_encrypted"] is True
        assert "preview_content" not in conv
        assert "decrypted secret" not in resp.text
    finally:
        _restore_app(app, original)


def test_conversations_endpoint_without_q_lists_normally():
    async def fake_query_conversations(agent_id, limit=50, view="active"):
        return []

    storage = MagicMock()
    storage.agent_id = "agent-1"
    storage.encryption_enabled = False
    storage.query_conversations = fake_query_conversations
    agent = MagicMock(storage=storage)

    app, original = _prepare_app(agent)
    try:
        client = TestClient(app)
        with patch.dict("os.environ", {"KESTREL_API_KEY": "test-key"}):
            resp = client.get("/api/conversations", headers={"X-API-Key": "test-key"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["conversations"] == []
        assert "query" not in body
    finally:
        _restore_app(app, original)
