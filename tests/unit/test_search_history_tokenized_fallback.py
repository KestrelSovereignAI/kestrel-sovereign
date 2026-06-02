"""
Regression: broad natural-language queries must find relevant memories.

Issue #1500 — ``search_history`` used ``if query_lower in content.lower()``,
requiring the entire query to appear as one contiguous substring. Broad
recall queries like ``"Meridian first Kestrel agent context memory management"``
returned 0 results even though a stored row contained all the key terms.

The fix adds a tokenized fallback: when the exact substring doesn't match,
query tokens are scored against content and rows above a 60 % threshold
are returned, ranked by match quality.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

from kestrel_sovereign.storage.async_conversation_store import (
    _tokenize_for_search,
    _token_match_score,
    _TOKEN_MATCH_THRESHOLD,
    AsyncConversationStore,
)


# -- Unit tests for helper functions -----------------------------------------

class TestTokenizeForSearch:
    def test_basic_tokenization(self):
        tokens = _tokenize_for_search("Meridian first Kestrel agent")
        assert "meridian" in tokens
        assert "first" in tokens
        assert "kestrel" in tokens
        assert "agent" in tokens

    def test_stopwords_removed(self):
        tokens = _tokenize_for_search("the first and the last")
        assert "the" not in tokens
        assert "and" not in tokens
        assert "first" in tokens
        assert "last" in tokens

    def test_single_char_tokens_dropped(self):
        tokens = _tokenize_for_search("a b c real word")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "c" not in tokens
        assert "real" in tokens
        assert "word" in tokens

    def test_punctuation_stripped(self):
        tokens = _tokenize_for_search("hello, world! memory's context.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "memory" in tokens
        assert "context" in tokens


class TestTokenMatchScore:
    def test_all_tokens_present(self):
        tokens = ["meridian", "kestrel", "memory"]
        score = _token_match_score(tokens, "meridian and kestrel memory system")
        assert score == 1.0

    def test_no_tokens_present(self):
        tokens = ["meridian", "kestrel", "memory"]
        score = _token_match_score(tokens, "completely unrelated content")
        assert score == 0.0

    def test_partial_match(self):
        tokens = ["meridian", "kestrel", "memory", "context"]
        score = _token_match_score(tokens, "meridian discussed kestrel features")
        assert score == 0.5  # 2 out of 4

    def test_empty_tokens(self):
        assert _token_match_score([], "any content") == 0.0


# -- Integration test for search_history tokenized fallback ------------------

STORED_CONTENT = (
    "meridian, and our discussion of the first kestrel agent. "
    "It seems our context and memory mgmt needs some work..."
)

BROAD_QUERY = "Meridian first Kestrel agent context memory management"
EXACT_QUERY = "meridian, and our discussion"


@pytest.mark.asyncio
async def test_broad_query_finds_relevant_row_via_token_fallback():
    """Issue #1500 regression: a broad NL query with key terms scattered
    across the stored content must return the row via token fallback."""
    store = AsyncConversationStore.__new__(AsyncConversationStore)
    store.agent_id = "test-agent"
    store._agent_fernet = None
    store._global_fernet = None
    store._migrate_on_read = False

    fake_row = (
        1, "user", STORED_CONTENT, None, "2026-01-01 00:00:00", None,
    )
    store.db = MagicMock()
    store.db.fetchall = AsyncMock(return_value=[fake_row])

    results = await store.search_history(BROAD_QUERY, limit=10)
    assert len(results) >= 1, (
        "Broad NL query should match via tokenized fallback"
    )
    assert results[0]["content"] == STORED_CONTENT


@pytest.mark.asyncio
async def test_exact_substring_still_works():
    """Exact substring queries must continue to match as before."""
    store = AsyncConversationStore.__new__(AsyncConversationStore)
    store.agent_id = "test-agent"
    store._agent_fernet = None
    store._global_fernet = None
    store._migrate_on_read = False

    fake_row = (
        1, "user", STORED_CONTENT, None, "2026-01-01 00:00:00", None,
    )
    store.db = MagicMock()
    store.db.fetchall = AsyncMock(return_value=[fake_row])

    results = await store.search_history(EXACT_QUERY, limit=10)
    assert len(results) == 1
    assert results[0]["content"] == STORED_CONTENT


@pytest.mark.asyncio
async def test_token_fallback_does_not_match_low_overlap():
    """A query sharing only one term with content should NOT match."""
    store = AsyncConversationStore.__new__(AsyncConversationStore)
    store.agent_id = "test-agent"
    store._agent_fernet = None
    store._global_fernet = None
    store._migrate_on_read = False

    fake_row = (
        1, "user", "the weather today is sunny and warm",
        None, "2026-01-01 00:00:00", None,
    )
    store.db = MagicMock()
    store.db.fetchall = AsyncMock(return_value=[fake_row])

    results = await store.search_history(
        "Meridian first Kestrel agent context memory management", limit=10
    )
    assert len(results) == 0, (
        "Low-overlap content must not match via tokenized fallback"
    )


@pytest.mark.asyncio
async def test_exact_matches_ranked_before_token_fallback():
    """When both exact and token-fallback matches exist, exact comes first."""
    store = AsyncConversationStore.__new__(AsyncConversationStore)
    store.agent_id = "test-agent"
    store._agent_fernet = None
    store._global_fernet = None
    store._migrate_on_read = False

    exact_row = (
        1, "user", "kestrel memory search",
        None, "2026-01-01 00:00:00", None,
    )
    token_row = (
        2, "user", "our kestrel agent has great memory and search capabilities",
        None, "2026-01-02 00:00:00", None,
    )
    store.db = MagicMock()
    store.db.fetchall = AsyncMock(return_value=[exact_row, token_row])

    results = await store.search_history("kestrel memory search", limit=10)
    assert len(results) >= 1
    # Exact match should be first
    assert results[0]["content"] == "kestrel memory search"


@pytest.mark.asyncio
async def test_search_does_not_match_retrieved_context_wrappers():
    """Search must match canonical content, not rendered transport blobs.
    Ensures token fallback doesn't reintroduce retrieved_context false positives."""
    store = AsyncConversationStore.__new__(AsyncConversationStore)
    store.agent_id = "test-agent"
    store._agent_fernet = None
    store._global_fernet = None
    store._migrate_on_read = False

    # Simulate a sent_form row where canonical content is plain user text
    # but the rendered form contains <retrieved_context> with extra terms
    plain_content = "what is the weather like today"
    fake_row = (
        1, "user", plain_content,
        None, "2026-01-01 00:00:00", None,
    )
    store.db = MagicMock()
    store.db.fetchall = AsyncMock(return_value=[fake_row])

    # Query terms that only appear in retrieved_context, not in canonical
    results = await store.search_history(
        "Meridian Kestrel constitution governance", limit=10
    )
    assert len(results) == 0


@pytest.mark.asyncio
async def test_single_word_query_uses_exact_only():
    """Single-word queries should not activate token fallback."""
    store = AsyncConversationStore.__new__(AsyncConversationStore)
    store.agent_id = "test-agent"
    store._agent_fernet = None
    store._global_fernet = None
    store._migrate_on_read = False

    fake_row = (
        1, "user", "kestrel agent memory discussion",
        None, "2026-01-01 00:00:00", None,
    )
    store.db = MagicMock()
    store.db.fetchall = AsyncMock(return_value=[fake_row])

    # "kestrel" is a substring so it should match exactly
    results = await store.search_history("kestrel", limit=10)
    assert len(results) == 1

    # "zebra" is not a substring and single-word can't do token fallback
    results = await store.search_history("zebra", limit=10)
    assert len(results) == 0
