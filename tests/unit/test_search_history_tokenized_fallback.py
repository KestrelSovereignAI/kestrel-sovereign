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

    def test_negation_tokens_preserved(self):
        """Negation must stay in the token set or recall surfaces opposite-meaning
        memories. e.g. "do not use OpenAI" must not reduce to "use openai".
        Codex round 1 P2 on #1500 rescue."""
        for negator in ("not", "no", "never", "neither", "nor", "without"):
            tokens = _tokenize_for_search(f"do {negator} use OpenAI")
            assert negator in tokens, (
                f"{negator!r} must not be stripped — it changes the semantics "
                "of the query"
            )


class TestNegationGate:
    """The negation equivalence-class gate prevents opposite-meaning rows
    from satisfying tokenized fallback (codex rounds 1-4 on #1500 rescue)."""

    def test_negator_missing_in_content_rejects(self):
        # "no use openai" against opposite-meaning "normally use openai":
        # without the gate this scored 3/3=1.0 (substring "no" hit "normally")
        # or 2/3 (word-boundary). Both above threshold — false positive.
        # Hard gate forces 0.0.
        score = _token_match_score(["no", "use", "openai"], "normally use openai")
        assert score == 0.0

        for negator in ("not", "never", "neither", "nor", "without"):
            score = _token_match_score(
                [negator, "use", "openai"], "use openai now"
            )
            assert score == 0.0, f"missing {negator!r} must hard-reject the row"

    def test_negator_equivalence_class_in_content_passes(self):
        # "not use openai" should still match "never use openai" — same
        # negated meaning, different negator word (codex round 4 P2).
        score = _token_match_score(["not", "use", "openai"], "never use openai")
        # Substring "not" in "never"? "never" = n-e-v-e-r → "not" not a
        # substring. So "not" doesn't hit per-token, but the GATE passes
        # because "never" is a recognized negator in content. Score = 2/3.
        assert score == pytest.approx(2 / 3)

    def test_negator_in_query_and_same_negator_in_content_full_score(self):
        score = _token_match_score(["no", "use", "openai"], "do no use openai please")
        assert score == 1.0

    def test_query_without_negator_unaffected(self):
        score = _token_match_score(["use", "openai"], "we should use openai today")
        assert score == 1.0


class TestTechnicalTermsUnregressed:
    """Codex round 4 P2: word-boundary matching for short non-negators
    regressed technical-term fallback because Python's ``\\b`` treats ``_``
    as a word character. Plain substring is preserved for non-negators."""

    def test_short_tech_terms_substring_match_api_key(self):
        # "api" in "api_key" — substring hit, was MISS under word-boundary.
        score = _token_match_score(["openai", "api", "key"], "openai api_key configured")
        # api hits inside api_key, key hits inside api_key, openai hits.
        assert score == 1.0

    def test_compound_word_substring_match(self):
        for content in (
            "the memorystore design",
            "memory-management subsystem",
            "the in-memory layout",
        ):
            score = _token_match_score(["memory"], content)
            assert score == 1.0, f"substring match should hit on {content!r}"


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


# Rendered transport form: canonical user text is a plain weather question,
# but the baked <retrieved_context> wrapper + "RELEVANT MEMORIES" heading
# carry the wrapper-only query terms. Such a row is NOT flagged sent_form
# here (metadata is None), so _resolve_canonical does not strip it — the
# search projection must.
WRAPPER_ONLY_ROW_CONTENT = (
    "<retrieved_context>\n<memories>\n"
    "--- RELEVANT MEMORIES (from past conversations) ---\n"
    "NOTE: These are retrieved from earlier conversations, not the current session.\n"
    "[Memory 1] User: Meridian and the first kestrel agent, constitution governance\n"
    "</memories>\n</retrieved_context>\n"
    "<user_input>\nwhat is the weather like today\n</user_input>"
)


@pytest.mark.asyncio
async def test_search_does_not_match_wrapper_only_terms_when_unstripped():
    """#1537 regression: a query of wrapper-only terms must NOT match a row
    solely because that phrase appears in rendered/retrieved-context wrapper
    text. The row's canonical user content is an unrelated weather question."""
    store = AsyncConversationStore.__new__(AsyncConversationStore)
    store.agent_id = "test-agent"
    store._agent_fernet = None
    store._global_fernet = None
    store._migrate_on_read = False

    fake_row = (
        1, "user", WRAPPER_ONLY_ROW_CONTENT, None, "2026-01-01 00:00:00", None,
    )
    store.db = MagicMock()
    store.db.fetchall = AsyncMock(return_value=[fake_row])

    # The heading phrase lives only in the wrapper; canonical content is
    # the weather question. Must not match.
    results = await store.search_history(
        "RELEVANT MEMORIES from past conversations", limit=10
    )
    assert len(results) == 0, (
        "wrapper-only terms must not make an unrelated row searchable"
    )

    # And the broad token query whose terms appear ONLY in the wrapper
    # memory line must likewise not match the canonical weather row.
    results = await store.search_history(
        "Meridian Kestrel constitution governance", limit=10
    )
    assert len(results) == 0


@pytest.mark.asyncio
async def test_canonical_content_still_matches_when_wrapped():
    """#1537 must not break #1500: a row whose canonical user text contains
    the broad query terms still matches even when wrapped in <user_input>
    and a retrieved-context block that the search projection strips."""
    store = AsyncConversationStore.__new__(AsyncConversationStore)
    store.agent_id = "test-agent"
    store._agent_fernet = None
    store._global_fernet = None
    store._migrate_on_read = False

    wrapped = (
        "<retrieved_context>\n<memories>\n"
        "--- RELEVANT MEMORIES (from past conversations) ---\n"
        "</memories>\n</retrieved_context>\n"
        f"<user_input>\n{STORED_CONTENT}\n</user_input>"
    )
    fake_row = (1, "user", wrapped, None, "2026-01-01 00:00:00", None)
    store.db = MagicMock()
    store.db.fetchall = AsyncMock(return_value=[fake_row])

    results = await store.search_history(BROAD_QUERY, limit=10)
    assert len(results) == 1, "canonical content must still match via token fallback"


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
