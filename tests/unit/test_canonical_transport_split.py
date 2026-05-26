"""Canonical/transport split for the conversation record (#1402).

Verifies the invariant that ``content`` holds canonical raw user speech
and ``rendered_content`` holds the byte-stable transport form (memories +
RAG baked in) for sent_form user turns. The split preserves prompt-cache
prefix stability (Anthropic ``cache_control``, llama.cpp KV, OpenAI prefix
cache) while letting search/audit/UI consumers read clean user text.

Covers:
  * new write path puts raw in ``content`` and transport in ``rendered_content``
  * lazy split-migration on read for legacy ``sent_form`` rows
  * byte-preservation through migration (cache prefix continues to hit)
  * migration idempotency
  * ``search_history`` does not false-match against stamped retrieved_context
  * no double-stamping across turns
  * ``format_conversation_history`` replays rendered_content verbatim
"""
import json
import tempfile
from pathlib import Path

import pytest

from kestrel_sovereign.agent.context_builder import (
    ContextBuilder,
    extract_raw_user_content,
)
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


RENDERED_FORM = (
    "<retrieved_context>\n"
    "<memories>\nMemory M1 about cats\n</memories>\n"
    "<documents>\nRAG doc D1 about dogs\n</documents>\n"
    "</retrieved_context>\n"
    "<user_input>\nhello world\n</user_input>"
)
RAW_FORM = "<user_input>\nhello world\n</user_input>"


class TestNewWritePath:
    @pytest.mark.asyncio
    async def test_canonical_and_transport_are_persisted_separately(self, store):
        """The new write path stores ``content`` as raw user turn and
        ``rendered_content`` as the byte-stable transport form."""
        await store.add_conversation(
            "user", RAW_FORM,
            metadata={"sent_form": True}, session_id="s1",
            rendered_content=RENDERED_FORM,
        )

        history = await store.get_conversation_history(limit=10)
        assert len(history) == 1
        row = history[0]
        assert row["role"] == "user"
        assert row["content"] == RAW_FORM
        assert row["rendered_content"] == RENDERED_FORM
        assert row.get("metadata", {}).get("sent_form") is True

    @pytest.mark.asyncio
    async def test_assistant_turn_has_no_rendered_content(self, store):
        """Assistant turns don't carry retrieval — rendered_content stays NULL."""
        await store.add_conversation(
            "assistant", "hello back", session_id="s1"
        )
        history = await store.get_conversation_history(limit=10)
        assert "rendered_content" not in history[0]

    @pytest.mark.asyncio
    async def test_legacy_write_without_rendered_content_is_canonical(self, store):
        """Writers that don't pass rendered_content (e.g. unwrapped raw
        user turns from older paths) store cleanly with no transport bytes."""
        await store.add_conversation("user", "raw text", session_id="s1")
        history = await store.get_conversation_history(limit=10)
        assert history[0]["content"] == "raw text"
        assert "rendered_content" not in history[0]
        meta = history[0].get("metadata") or {}
        assert "sent_form" not in meta


class TestLazySplitMigration:
    """Legacy rows wrote the rendered form into ``content`` with
    ``metadata.sent_form=True``. On read we split in-memory and
    opportunistically backfill the new shape, preserving the rendered
    bytes byte-for-byte so the cache prefix continues to hit."""

    async def _write_legacy_sent_form_row(self, store, rendered: str) -> int:
        """Bypass add_conversation to simulate a row written by the
        pre-#1402 code path (content = rendered, rendered_content NULL)."""
        await store.db.execute_commit(
            "INSERT INTO conversation_history "
            "(agent_id, role, content, rendered_content, metadata, created_at) "
            "VALUES (?, ?, ?, NULL, ?, datetime('now'))",
            (store.agent_id, "user", rendered,
             json.dumps({"sent_form": True, "session_id": "s1"})),
        )
        row = await store.db.fetchone(
            "SELECT id FROM conversation_history WHERE agent_id = ? ORDER BY id DESC LIMIT 1",
            (store.agent_id,),
        )
        return row[0]

    @pytest.mark.asyncio
    async def test_legacy_row_splits_on_read(self, store):
        await self._write_legacy_sent_form_row(store, RENDERED_FORM)
        history = await store.get_conversation_history(limit=10)
        row = history[0]
        # Canonical is the stripped user speech
        assert row["content"] == "hello world"
        # Transport is the original rendered bytes — preserved byte-for-byte
        assert row["rendered_content"] == RENDERED_FORM

    @pytest.mark.asyncio
    async def test_legacy_migration_persists_to_db(self, store):
        """After the first read, the row exposes the new shape via the
        store's decrypted-read path so subsequent reads don't keep
        splitting. We read through ``get_conversation_history`` rather
        than asserting raw DB bytes because encryption-at-rest is
        env-driven and the test must work either way."""
        row_id = await self._write_legacy_sent_form_row(store, RENDERED_FORM)
        await store.get_conversation_history(limit=10)  # trigger migration

        # Disable on-read migration so a second read can't paper over
        # a missed first-read write. Whatever's on disk now is what the
        # second read must surface.
        store._migrate_on_read = False
        history = await store.get_conversation_history(limit=10)
        store._migrate_on_read = True

        row = history[0]
        assert row["id"] == row_id
        assert row["content"] == "hello world"
        assert row["rendered_content"] == RENDERED_FORM

    @pytest.mark.asyncio
    async def test_legacy_migration_byte_preserving(self, store):
        """``rendered_content`` after migration MUST equal pre-migration
        ``content`` byte-for-byte — that's the cache-prefix-stability
        invariant the original sent-form persistence existed to guarantee.
        Verified through the decrypted-read path so encryption-at-rest
        doesn't muddy the assertion."""
        pre_bytes = RENDERED_FORM
        await self._write_legacy_sent_form_row(store, pre_bytes)
        history = await store.get_conversation_history(limit=10)
        rendered = history[0]["rendered_content"]
        assert rendered == pre_bytes
        assert rendered.encode("utf-8") == pre_bytes.encode("utf-8")

    @pytest.mark.asyncio
    async def test_legacy_migration_idempotent(self, store):
        """Running the migration twice produces the same final state."""
        await self._write_legacy_sent_form_row(store, RENDERED_FORM)
        first = await store.get_conversation_history(limit=10)
        second = await store.get_conversation_history(limit=10)
        # Same final shape, same canonical content, same transport bytes.
        assert first[0]["content"] == second[0]["content"]
        assert first[0]["rendered_content"] == second[0]["rendered_content"]
        # And only one row — migration must not duplicate rows on re-read.
        assert len(first) == 1
        assert len(second) == 1


class TestNoDoubleStamp:
    @pytest.mark.asyncio
    async def test_subsequent_user_turn_does_not_inherit_prior_retrieval(self, store):
        """Two user turns. Each turn's ``content`` row is the canonical raw
        speech only — no stamped retrieved_context from earlier turns
        leaks into later canonical rows."""
        await store.add_conversation(
            "user", wrap_user_input("turn 1 question"),
            metadata={"sent_form": True}, session_id="s1",
            rendered_content=(
                "<retrieved_context>\n<memories>\nM1\n</memories>\n</retrieved_context>\n"
                + wrap_user_input("turn 1 question")
            ),
        )
        await store.add_conversation("assistant", "answer 1", session_id="s1")
        await store.add_conversation(
            "user", wrap_user_input("turn 2 question"),
            metadata={"sent_form": True}, session_id="s1",
            rendered_content=(
                "<retrieved_context>\n<memories>\nM2\n</memories>\n</retrieved_context>\n"
                + wrap_user_input("turn 2 question")
            ),
        )

        history = await store.get_conversation_history(limit=10)
        user_rows = [r for r in history if r["role"] == "user"]
        # Each canonical content has its own raw turn, never the other's
        # retrieval blob.
        assert "<retrieved_context>" not in user_rows[0]["content"]
        assert "<retrieved_context>" not in user_rows[1]["content"]
        assert "M1" not in user_rows[0]["content"]
        assert "M2" not in user_rows[1]["content"]
        # And no cross-contamination
        assert "turn 2" not in user_rows[0]["content"]
        assert "turn 1" not in user_rows[1]["content"]


class TestSearchHistoryDoesNotMatchRetrieval:
    @pytest.mark.asyncio
    async def test_search_misses_text_that_only_lives_in_retrieved_context(self, store):
        """Searching for a phrase that only appears inside the rendered
        transport (memories/RAG) must not match — search is over canonical
        user speech, not transport bytes."""
        await store.add_conversation(
            "user", wrap_user_input("what's the weather?"),
            metadata={"sent_form": True}, session_id="s1",
            rendered_content=(
                "<retrieved_context>\n<memories>\n"
                "The user mentioned PINEAPPLE_SECRET_MARKER last week\n"
                "</memories>\n</retrieved_context>\n"
                + wrap_user_input("what's the weather?")
            ),
        )

        # The phrase only lives in rendered_content (memories) — must NOT match
        results = await store.search_history("PINEAPPLE_SECRET_MARKER", limit=10)
        assert results == []

        # The real user text DOES match
        results = await store.search_history("weather", limit=10)
        assert len(results) == 1
        assert "<retrieved_context>" not in results[0]["content"]

    @pytest.mark.asyncio
    async def test_search_does_not_match_legacy_retrieval_either(self, store):
        """Search must also exclude retrieval blobs from legacy rows (those
        that haven't been split-migrated yet) — the same in-memory split
        runs in search_history."""
        await store.db.execute_commit(
            "INSERT INTO conversation_history "
            "(agent_id, role, content, rendered_content, metadata, created_at) "
            "VALUES (?, ?, ?, NULL, ?, datetime('now'))",
            (store.agent_id, "user",
             "<retrieved_context>\n<memories>\nLEGACY_MARKER_X\n</memories>\n"
             "</retrieved_context>\n<user_input>\nplain question\n</user_input>",
             json.dumps({"sent_form": True})),
        )

        # Legacy retrieval text must NOT match
        results = await store.search_history("LEGACY_MARKER_X", limit=10)
        assert results == []

        # Raw user text DOES match
        results = await store.search_history("plain question", limit=10)
        assert len(results) == 1


class TestFormatConversationHistoryReplaysVerbatim:
    """``ContextBuilder.format_conversation_history`` is the LLM-call path:
    it must emit the rendered_content bytes verbatim for sent_form user
    turns. That's the cache-prefix-stability contract.

    We construct entries that mimic ``get_conversation_history`` output
    and assert the emitter pulls from ``rendered_content``."""

    def _format(self, history):
        cb = ContextBuilder.__new__(ContextBuilder)
        # Minimal private state for format_conversation_history. The
        # counter property keys off self.model, which falls back to
        # _model_fallback when _llm_service is None. Matching _counter_model
        # to _model_fallback prevents the property from refreshing the
        # cached counter.
        cb._llm_service = None
        cb._model_fallback = "test-stub"

        class _Counter:
            def count(self, s):
                return max(1, len(s) // 4)

            def truncate_to_tokens(self, s, n):
                return s[: n * 4]

        cb._counter = _Counter()
        cb._counter_model = "test-stub"
        return cb.format_conversation_history(history, max_tokens=10_000)

    def test_sent_form_emits_rendered_content_verbatim(self):
        history = [
            {
                "role": "user",
                "content": "hello world",  # canonical raw user text (no wrap)
                "rendered_content": RENDERED_FORM,
                "metadata": {"sent_form": True},
            },
        ]
        formatted = self._format(history)
        assert len(formatted) == 1
        assert formatted[0]["content"] == RENDERED_FORM

    def test_legacy_unwrapped_user_turn_gets_wrapped(self):
        history = [
            {
                "role": "user",
                "content": "raw legacy text",
                "metadata": {},
            },
        ]
        formatted = self._format(history)
        assert formatted[0]["content"] == "<user_input>\nraw legacy text\n</user_input>"

    def test_assistant_turn_passes_through(self):
        history = [{"role": "assistant", "content": "answer body", "metadata": {}}]
        formatted = self._format(history)
        assert formatted[0]["content"] == "answer body"

    def test_safety_fallback_when_sent_form_has_no_rendered_content(self):
        """Defensive: if a sent_form entry somehow arrives without
        rendered_content (shouldn't happen post-#1402 — the read path
        splits in-memory), fall back to raw content rather than crash."""
        history = [
            {
                "role": "user",
                "content": "stripped raw",
                "metadata": {"sent_form": True},
            },
        ]
        formatted = self._format(history)
        # Falls back to raw content — not double-wrapped (sent_form path).
        # This is the safety net; the production read path never reaches
        # here because _resolve_canonical populates rendered_content.
        assert formatted[0]["content"] == "stripped raw"


class TestExtractRawIsIdempotentOnNewWrites:
    """Existing consumers (MemoryFeature, endpoints, personality_analyzer,
    wellness) call ``extract_raw_user_content`` on ``content`` to defend
    against the legacy shape. Post-#1402 ``content`` is already raw —
    the call must be a no-op (idempotent) so we can leave it in place
    during the migration window without behavioral drift."""

    def test_extract_is_noop_on_new_write_path_content(self):
        # The new write path writes content = wrap_user_input(raw),
        # NOT the rendered form. extract_raw_user_content strips the
        # outer <user_input> wrapper and returns the raw text.
        wrapped = wrap_user_input("clean user text")
        assert extract_raw_user_content(wrapped) == "clean user text"
        # Second call is also a no-op
        assert extract_raw_user_content("clean user text") == "clean user text"
