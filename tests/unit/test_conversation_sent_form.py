"""Sent-form metadata round-trip for user-turn storage.

After #1402 the user-turn write path is split: ``content`` holds the
canonical raw user turn and ``rendered_content`` holds the byte-stable
transport form (memories + RAG baked in). The ``sent_form`` flag still
marks rows that carry a transport form so the history-load path knows to
replay verbatim.

This module verifies the flag survives the write → encrypt → read →
decrypt → metadata-cleanup pipeline so ``ContextBuilder`` can trust it
when deciding whether to replay-or-wrap at load time. Deeper tests for
the canonical/transport split, lazy migration, and search semantics live
in :mod:`test_canonical_transport_split`.
"""
import pytest
import tempfile
from pathlib import Path

from kestrel_sovereign.security.input_guardrails import wrap_user_input
from kestrel_sovereign.storage.async_conversation_store import AsyncConversationStore
from kestrel_sovereign.storage.async_database import AsyncDatabase


@pytest.fixture
async def store():
    """Real SQLite-backed conversation store (no encryption) for wire-level checks."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        db = await AsyncDatabase.sqlite(str(db_path))
        store = AsyncConversationStore(db, agent_id="test-agent")
        yield store
        await db.close()


class TestSentFormMetadataRoundTrip:
    @pytest.mark.asyncio
    async def test_sent_form_flag_round_trips(self, store):
        """A user row written with the canonical/transport split surfaces
        the ``sent_form`` flag on read so format_conversation_history can
        branch on it."""
        raw = wrap_user_input("hello")
        rendered = (
            "<retrieved_context>\n<memories>\nM\n</memories>\n</retrieved_context>\n"
            + raw
        )
        await store.add_conversation(
            "user", raw,
            metadata={"sent_form": True}, session_id="s1",
            rendered_content=rendered,
        )

        history = await store.get_conversation_history(limit=10)

        assert len(history) == 1
        row = history[0]
        assert row["role"] == "user"
        assert row["content"] == raw
        assert row["rendered_content"] == rendered
        assert row.get("metadata", {}).get("sent_form") is True

    @pytest.mark.asyncio
    async def test_legacy_row_has_no_sent_form_flag(self, store):
        """A user row written without the flag must read back without it,
        so context_builder falls through to the wrap-on-load legacy branch."""
        await store.add_conversation("user", "raw text", session_id="s1")

        history = await store.get_conversation_history(limit=10)

        assert len(history) == 1
        row = history[0]
        assert row["content"] == "raw text"
        assert "rendered_content" not in row
        meta = row.get("metadata") or {}
        assert "sent_form" not in meta

    @pytest.mark.asyncio
    async def test_sent_form_coexists_with_session_and_other_meta(self, store):
        """The flag must not interfere with other metadata the store writes
        (session_id from implicit-derivation, enc flag from encryption)."""
        await store.add_conversation(
            "user",
            wrap_user_input("hi"),
            metadata={"sent_form": True, "custom": "keep-me"},
            session_id="explicit-session",
            rendered_content=wrap_user_input("hi"),
        )

        history = await store.get_conversation_history(limit=10)
        meta = history[0].get("metadata") or {}

        assert meta.get("sent_form") is True
        assert meta.get("custom") == "keep-me"
        assert meta.get("session_id") == "explicit-session"
