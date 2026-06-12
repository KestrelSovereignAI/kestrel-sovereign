"""
Unit tests for Human-Like Memory System.

Tests all memory components:
- EmotionalTagger
- TemporalAnalyzer
- AssociativeLinker
- MemoryRetriever
- MemoryConsolidator
- MemorySystem (unified interface)
"""
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

from kestrel_sovereign.storage import (
    MemoryMetadata,
    TemporalPattern,
    MemoryEpisode,
    EmotionalCategory,
    EmotionalTagger,
    TemporalAnalyzer,
    AssociativeLinker,
    MemoryRetriever,
    calculate_decay,
)


class TestEmotionalTagger:
    """Tests for EmotionalTagger sentiment and importance detection."""

    @pytest.fixture
    def tagger(self):
        return EmotionalTagger()

    @pytest.mark.asyncio
    async def test_detect_positive_emotion(self, tagger):
        """Should detect positive emotions."""
        result = await tagger.analyze("I am so happy today!", "user")
        assert result.emotional_valence > 0, "Should have positive valence"
        assert "joy" in result.emotional_categories

    @pytest.mark.asyncio
    async def test_detect_negative_emotion(self, tagger):
        """Should detect negative emotions."""
        result = await tagger.analyze("I feel so sad and depressed", "user")
        assert result.emotional_valence < 0, "Should have negative valence"
        assert "sadness" in result.emotional_categories

    @pytest.mark.asyncio
    async def test_detect_mixed_emotions(self, tagger):
        """Should detect multiple emotions in one message."""
        result = await tagger.analyze(
            "I was scared at first but then felt happy", "user"
        )
        assert len(result.emotional_categories) >= 2

    @pytest.mark.asyncio
    async def test_emotional_intensity_amplifiers(self, tagger):
        """Intensity amplifiers should increase emotional intensity."""
        # Use longer messages to get lower baseline intensity
        mild = await tagger.analyze("I think things are going okay today", "user")
        intense = await tagger.analyze("I am EXTREMELY happy!!! This is AMAZING!!!", "user")
        assert intense.emotional_intensity > mild.emotional_intensity

    @pytest.mark.asyncio
    async def test_importance_life_event(self, tagger):
        """Life events should have high importance."""
        result = await tagger.analyze("I just got promoted at work!", "user")
        assert result.importance > 0.7
        assert "life_event" in result.importance_reasons

    @pytest.mark.asyncio
    async def test_importance_personal_disclosure(self, tagger):
        """Personal disclosures should have high importance."""
        result = await tagger.analyze(
            "I've never told anyone this but I was scared", "user"
        )
        assert result.importance > 0.6
        assert "personal_disclosure" in result.importance_reasons

    @pytest.mark.asyncio
    async def test_importance_explicit_marker(self, tagger):
        """Explicit memory markers should be high importance."""
        result = await tagger.analyze(
            "Remember this: my favorite color is blue", "user"
        )
        assert result.importance > 0.7
        assert "explicit_marker" in result.importance_reasons

    @pytest.mark.asyncio
    async def test_temporal_context(self, tagger):
        """Should include time_of_day and day_of_week."""
        result = await tagger.analyze("Hello", "user")
        assert result.time_of_day in ["morning", "afternoon", "evening", "late_night"]
        assert result.day_of_week in [
            "monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday"
        ]


class TestTemporalAnalyzer:
    """Tests for TemporalAnalyzer pattern detection."""

    @pytest.fixture
    def mock_db(self):
        return MagicMock()

    @pytest.fixture
    def analyzer(self, mock_db):
        return TemporalAnalyzer(mock_db)

    @pytest.mark.asyncio
    async def test_detect_time_preference(self, analyzer):
        """Should detect most active time of day."""
        messages = [
            {"metadata": {"time_of_day": "late_night"}},
            {"metadata": {"time_of_day": "late_night"}},
            {"metadata": {"time_of_day": "late_night"}},
            {"metadata": {"time_of_day": "late_night"}},
            {"metadata": {"time_of_day": "late_night"}},
            {"metadata": {"time_of_day": "morning"}},
        ]

        pattern = await analyzer._detect_time_preference(messages, "test", 3)
        assert pattern is not None
        assert "late at night" in pattern.description.lower()
        assert pattern.confidence > 0.5

    @pytest.mark.asyncio
    async def test_detect_emotion_time_correlation(self, analyzer):
        """Should detect emotional patterns by time."""
        messages = [
            {"metadata": {"time_of_day": "late_night", "emotional_valence": -0.5}},
            {"metadata": {"time_of_day": "late_night", "emotional_valence": -0.6}},
            {"metadata": {"time_of_day": "late_night", "emotional_valence": -0.4}},
            {"metadata": {"time_of_day": "morning", "emotional_valence": 0.7}},
        ]

        patterns = await analyzer._detect_emotion_time_correlation(
            messages, "test", 3
        )
        assert len(patterns) > 0
        # Should detect user feels down late at night
        late_night_pattern = next(
            (p for p in patterns if "late" in p.description.lower()),
            None
        )
        assert late_night_pattern is not None

    def test_get_time_of_day(self):
        """Should correctly classify times."""
        assert TemporalAnalyzer._get_time_of_day(
            datetime(2025, 1, 1, 8, 0)
        ) == "morning"
        assert TemporalAnalyzer._get_time_of_day(
            datetime(2025, 1, 1, 14, 0)
        ) == "afternoon"
        assert TemporalAnalyzer._get_time_of_day(
            datetime(2025, 1, 1, 20, 0)
        ) == "evening"
        assert TemporalAnalyzer._get_time_of_day(
            datetime(2025, 1, 1, 2, 0)
        ) == "late_night"


class TestAssociativeLinker:
    """Tests for AssociativeLinker concept extraction."""

    @pytest.fixture
    def linker(self):
        # Create with None graph for concept extraction tests only
        return AssociativeLinker(None)

    def test_extract_person_concepts(self, linker):
        """Should extract person relationship concepts."""
        text = "I called mom yesterday and she mentioned grandma"
        concepts = linker._extract_concepts(text)
        assert "mom" in concepts
        assert "grandma" in concepts

    def test_extract_place_concepts(self, linker):
        """Should extract place concepts."""
        text = "I went to work in Brooklyn this morning"
        concepts = linker._extract_concepts(text)
        assert "work" in concepts
        assert "brooklyn" in concepts

    def test_extract_time_concepts(self, linker):
        """Should extract time concepts."""
        text = "I remember Christmas morning when we opened presents"
        concepts = linker._extract_concepts(text)
        assert "christmas" in concepts
        assert "morning" in concepts

    def test_extract_emotion_concepts(self, linker):
        """Should extract emotion concepts."""
        text = "I was so happy but also a little scared"
        concepts = linker._extract_concepts(text)
        assert "happy" in concepts
        assert "scared" in concepts


class TestMemoryRetriever:
    """Tests for MemoryRetriever weighted scoring."""

    def test_semantic_score(self):
        """Semantic scoring should match keywords."""
        retriever = MemoryRetriever(None, None)

        # High overlap
        score_high = retriever._score_semantic(
            "I love cooking dinner for my family",
            "cooking dinner",
            []
        )
        assert score_high > 0.5

        # No overlap
        score_low = retriever._score_semantic(
            "The weather is nice today",
            "cooking dinner",
            []
        )
        assert score_low < score_high

    def test_emotional_score_congruence(self):
        """Mood-congruent memories should score higher."""
        retriever = MemoryRetriever(None, None)

        # Create positive context
        positive_context = MemoryMetadata(emotional_valence=0.7)

        # Positive memory matches positive context
        score_match = retriever._score_emotional(
            {"emotional_valence": 0.6},
            positive_context
        )

        # Negative memory doesn't match positive context
        score_mismatch = retriever._score_emotional(
            {"emotional_valence": -0.6},
            positive_context
        )

        assert score_match > score_mismatch

    def test_recency_score_decay(self):
        """Recent memories should score higher than old ones."""
        retriever = MemoryRetriever(None, None)

        now = datetime.now(timezone.utc)
        recent = now - timedelta(days=1)
        old = now - timedelta(days=60)

        score_recent = retriever._score_recency(recent.isoformat(), 0.5)
        score_old = retriever._score_recency(old.isoformat(), 0.5)

        assert score_recent > score_old

    def test_importance_slows_decay(self):
        """High importance should slow decay."""
        retriever = MemoryRetriever(None, None)

        now = datetime.now(timezone.utc)
        old = (now - timedelta(days=60)).isoformat()

        score_low_importance = retriever._score_recency(old, 0.2)
        score_high_importance = retriever._score_recency(old, 1.0)

        assert score_high_importance > score_low_importance


    @pytest.mark.asyncio
    async def test_update_applied_delegates_to_atomic_increment(self):
        """``update_applied`` delegates to the store's atomic JSON-set
        helper rather than a caller-side read-modify-write — that's the
        write-path contract that prevents lost increments under
        concurrent application.  See #1326 codex round-1 race.

        Pins the field names too: ``applied_count`` (counter) +
        ``last_applied`` (timestamp); ``access_count`` must NOT be
        touched (separate signal).
        """
        store = AsyncMock()
        store.agent_id = "test-agent"
        store.atomic_increment_metadata_counter = AsyncMock(return_value=True)

        retriever = MemoryRetriever(store, None)
        await retriever.update_applied(message_id=42, agent_id="test-agent")

        store.atomic_increment_metadata_counter.assert_awaited_once_with(
            42,
            counter_field="applied_count",
            timestamp_field="last_applied",
        )

    @pytest.mark.asyncio
    async def test_update_applied_swallows_write_errors(self):
        """Bookkeeping failures (DB down, disk full) must not propagate
        — calling code (typically reflection / pre-sleep hook) doesn't
        want a flaky write to abort its own work."""
        store = AsyncMock()
        store.agent_id = "test-agent"
        store.atomic_increment_metadata_counter = AsyncMock(
            side_effect=RuntimeError("disk full")
        )

        retriever = MemoryRetriever(store, None)
        # Must not raise.
        await retriever.update_applied(message_id=1, agent_id="test-agent")

    @pytest.mark.asyncio
    async def test_update_access_also_uses_atomic_increment(self):
        """Pre-#1326 ``update_access`` had the same read-modify-write
        race ``update_applied`` would have inherited.  Both now route
        through the atomic helper — pin that here so a regression to
        the racy pattern doesn't slip back in for either field."""
        store = AsyncMock()
        store.agent_id = "test-agent"
        store.atomic_increment_metadata_counter = AsyncMock(return_value=True)

        retriever = MemoryRetriever(store, None)
        await retriever.update_access(message_id=7, agent_id="test-agent")

        store.atomic_increment_metadata_counter.assert_awaited_once_with(
            7,
            counter_field="access_count",
            timestamp_field="last_accessed",
        )

    @pytest.mark.asyncio
    async def test_retrieve_includes_user_messages_with_role_preserved(self):
        """User-role messages MUST surface — they carry biographical content
        (preferences, names, dates, locations) that's the whole point of
        recall in a conversational AI.

        Echo prevention now relies on (a) the exact-query dedup at line 141
        of ``memory_retriever.py`` and (b) role attribution in the injection
        format at ``agent/memory_manager.py``. The old blanket
        ``role=user`` skip from #271 over-broadly suppressed every user-
        stated fact along with the questions it was trying to suppress.
        See #1481.
        """
        store = AsyncMock()
        store.get_conversation_history.return_value = [
            {"role": "user", "id": 1, "content": "My favorite color is blue.",
             "metadata": {}, "created_at": "2025-01-15 10:00:00"},
            {"role": "assistant", "id": 2, "content": "Got it — blue.",
             "metadata": {"importance": 0.5}, "created_at": "2025-01-15 10:00:01"},
            {"role": "user", "id": 3, "content": "Remember my lucky number is 42",
             "metadata": {}, "created_at": "2025-01-15 10:01:00"},
            {"role": "assistant", "id": 4, "content": "I will remember 42.",
             "metadata": {"importance": 0.5}, "created_at": "2025-01-15 10:01:01"},
        ]
        store.embedding_service = None  # keyword-only path

        retriever = MemoryRetriever(store, None)
        results = await retriever.retrieve(
            query="favorite color",
            agent_id="test-agent",
            limit=10,
            min_score=0.0,
        )

        contents = [r["content"] for r in results]
        roles = [r["role"] for r in results]
        # The user-stated fact must surface — it's the actual answer to the query.
        assert any("blue" in c for c in contents), (
            f"Expected to find user's biographical content; got {contents!r}"
        )
        # User-role messages must be present in the results.
        assert "user" in roles, (
            f"Expected at least one user-role memory in results; got roles={roles!r}"
        )

    @pytest.mark.asyncio
    async def test_retrieve_skips_exact_query_echo(self):
        """The exact-match echo guard at line 141 is the only echo-prevention
        layer now. Verify it still drops a row whose content exactly equals
        the current query (case-insensitive, whitespace-trimmed).
        """
        store = AsyncMock()
        store.get_conversation_history.return_value = [
            {"role": "user", "id": 1, "content": "What is my favorite color?",
             "metadata": {}, "created_at": "2025-01-15 10:00:00"},
            {"role": "user", "id": 2, "content": "I love sailing.",
             "metadata": {}, "created_at": "2025-01-15 10:01:00"},
        ]
        store.embedding_service = None

        retriever = MemoryRetriever(store, None)
        results = await retriever.retrieve(
            query="What is my favorite color?",  # exact echo of row 1
            agent_id="test-agent",
            limit=10,
            min_score=0.0,
        )

        # Row 1 (exact echo) MUST be dropped.
        ids = [r.get("id") for r in results]
        assert 1 not in ids, (
            f"Exact-query echo should be dropped; got ids={ids!r}"
        )

    @pytest.mark.asyncio
    async def test_retrieve_echo_guard_normalizes_trivial_variants(self):
        """Punctuation, casing, whitespace differences shouldn't let a
        prior user question slip past the echo guard. Regression for
        codex P2 round 2 on #1481 — the exact lowercase comparison was
        too strict; ``what is my favorite color`` and
        ``What is my favorite color?`` should both be treated as echoes.
        """
        store = AsyncMock()
        store.get_conversation_history.return_value = [
            {"role": "user", "id": 1,
             "content": "<user_input>\nWhat is my favorite color?\n</user_input>",
             "metadata": {}, "created_at": "2025-01-15 10:00:00"},
        ]
        store.embedding_service = None

        retriever = MemoryRetriever(store, None)

        for variant in [
            "what is my favorite color",          # no punctuation
            "  What is my favorite COLOR? ",       # extra whitespace + casing
            "What\tis my\nfavorite color?",        # tab + newline
            "What is my favorite color???",        # extra punctuation
        ]:
            results = await retriever.retrieve(
                query=variant, agent_id="t", limit=10, min_score=0.0,
            )
            ids = [r.get("id") for r in results]
            assert 1 not in ids, (
                f"Echo variant {variant!r} should be dropped; got ids={ids!r}"
            )

    @pytest.mark.asyncio
    async def test_retrieve_echo_guard_preserves_internal_punctuation(self):
        """Internal punctuation in alphanumeric tokens is semantically
        meaningful — ``C++`` is not ``C``, ``1.2`` is not ``12``. Echo
        normalization must edge-strip only. Regression for codex P3 on
        #1481.
        """
        store = AsyncMock()
        store.get_conversation_history.return_value = [
            {"role": "user", "id": 1,
             "content": "<user_input>\nI use C++.\n</user_input>",
             "metadata": {}, "created_at": "2025-01-15 10:00:00"},
            {"role": "user", "id": 2,
             "content": "<user_input>\nVersion 1.2 ships next week.\n</user_input>",
             "metadata": {}, "created_at": "2025-01-15 10:01:00"},
        ]
        store.embedding_service = None

        retriever = MemoryRetriever(store, None)

        # "I use C" must NOT echo-collide with "I use C++" (different facts).
        results = await retriever.retrieve(
            query="I use C", agent_id="t", limit=10, min_score=0.0,
        )
        ids = [r.get("id") for r in results]
        assert 1 in ids, (
            f"Internal punct (C++) collapsed to (C); got ids={ids!r}"
        )

        # "Version 12" must NOT echo-collide with "Version 1.2".
        results = await retriever.retrieve(
            query="Version 12 ships next week", agent_id="t",
            limit=10, min_score=0.0,
        )
        ids = [r.get("id") for r in results]
        assert 2 in ids, (
            f"Internal punct (1.2) collapsed to (12); got ids={ids!r}"
        )

    @pytest.mark.asyncio
    async def test_retrieve_echo_guard_strips_user_input_wrapper(self):
        """User turns are persisted wrapped via ``wrap_user_input`` as
        ``<user_input>\\n...\\n</user_input>``. The echo guard MUST unwrap
        before comparing — otherwise a literal repeat of a prior question
        bypasses the guard and gets surfaced as a memory.

        Regression for codex P2 on #1481: without ``extract_raw_user_content``
        the comparison ``content.strip().lower() == query_normalized`` would
        only fire on raw test data, never on production-wrapped chat rows.
        """
        store = AsyncMock()
        store.get_conversation_history.return_value = [
            {"role": "user", "id": 1,
             "content": "<user_input>\nWhat is my favorite color?\n</user_input>",
             "metadata": {}, "created_at": "2025-01-15 10:00:00"},
            {"role": "user", "id": 2,
             "content": "<user_input>\nI love sailing.\n</user_input>",
             "metadata": {}, "created_at": "2025-01-15 10:01:00"},
        ]
        store.embedding_service = None

        retriever = MemoryRetriever(store, None)
        results = await retriever.retrieve(
            query="What is my favorite color?",  # raw — matches unwrapped row 1
            agent_id="test-agent",
            limit=10,
            min_score=0.0,
        )

        ids = [r.get("id") for r in results]
        assert 1 not in ids, (
            f"Wrapped-content echo should still be dropped after unwrap; "
            f"got ids={ids!r}"
        )
        # Row 2 (biographical fact, not the query) MUST still surface.
        assert 2 in ids, f"Expected row 2 to surface; got ids={ids!r}"


class TestDecayCalculation:
    """Tests for standalone decay calculation."""

    def test_fresh_memory_no_decay(self):
        """Fresh memories should have full strength."""
        now = datetime.now(timezone.utc).isoformat()
        strength = calculate_decay(now, importance=0.5)
        assert strength > 0.95

    def test_decay_at_half_life(self):
        """At half-life, strength should be ~0.5."""
        now = datetime.now(timezone.utc)
        half_life = (now - timedelta(days=30)).isoformat()
        # With importance=0, half-life is 30 days
        strength = calculate_decay(half_life, importance=0.0)
        assert 0.4 < strength < 0.6

    def test_protected_memory_no_decay(self):
        """Protected memories should not decay."""
        now = datetime.now(timezone.utc)
        very_old = (now - timedelta(days=365)).isoformat()
        strength = calculate_decay(very_old, decay_protected=True)
        assert strength == 1.0

    def test_access_strengthens_memory(self):
        """Accessed memories should decay slower."""
        now = datetime.now(timezone.utc)
        old = (now - timedelta(days=60)).isoformat()

        strength_no_access = calculate_decay(old, importance=0.5, access_count=0)
        strength_accessed = calculate_decay(old, importance=0.5, access_count=10)

        assert strength_accessed > strength_no_access

    def test_applied_strengthens_more_than_access(self):
        """Applied memories should decay slower than memories that have
        only been retrieved at the same count.  Rewards being
        load-bearing over being merely familiar — the #1326 distinction.
        """
        now = datetime.now(timezone.utc)
        old = (now - timedelta(days=60)).isoformat()

        strength_accessed = calculate_decay(
            old, importance=0.5, access_count=3, applied_count=0,
        )
        strength_applied = calculate_decay(
            old, importance=0.5, access_count=0, applied_count=3,
        )

        assert strength_applied > strength_accessed, (
            "applied_count must produce a stronger decay-resistance boost "
            "than access_count at the same magnitude, or the new field "
            "carries no signal"
        )

    def test_applied_and_access_compound(self):
        """A memory that's been BOTH accessed AND applied should out-strength
        a memory that's been accessed N+M times but never applied.  The two
        boosts multiply, so they're additive in log space and a memory
        that's load-bearing benefits from both signals."""
        now = datetime.now(timezone.utc)
        old = (now - timedelta(days=60)).isoformat()

        only_accessed = calculate_decay(
            old, importance=0.5, access_count=6, applied_count=0,
        )
        accessed_and_applied = calculate_decay(
            old, importance=0.5, access_count=3, applied_count=3,
        )
        assert accessed_and_applied > only_accessed

    def test_applied_default_unchanged_behavior(self):
        """Default ``applied_count=0`` must produce identical strength
        to a pre-#1326 caller that didn't pass the parameter at all.
        Regression guard — adding the parameter cannot change the math
        for existing callers.

        ``approx`` accommodates the microsecond-level wall-clock drift
        between the two calls; the two strengths must agree to far
        beyond what any consumer of this function could distinguish."""
        now = datetime.now(timezone.utc)
        old = (now - timedelta(days=45)).isoformat()

        old_caller_shape = calculate_decay(old, importance=0.5, access_count=2)
        new_caller_default = calculate_decay(
            old, importance=0.5, access_count=2, applied_count=0,
        )
        assert old_caller_shape == pytest.approx(new_caller_default, rel=1e-9)


class TestMemoryMetadata:
    """Tests for MemoryMetadata model."""

    def test_to_dict(self):
        """Should serialize to dict correctly."""
        meta = MemoryMetadata(
            emotional_valence=0.7,
            emotional_intensity=0.5,
            emotional_categories=["joy"],
            importance=0.8,
            importance_reasons=["life_event"],
            time_of_day="morning",
            day_of_week="monday",
        )

        d = meta.to_dict()
        assert d["emotional_valence"] == 0.7
        assert d["emotional_categories"] == ["joy"]
        assert d["importance_reasons"] == ["life_event"]

    def test_from_dict(self):
        """Should deserialize from dict correctly."""
        d = {
            "emotional_valence": 0.7,
            "emotional_intensity": 0.5,
            "emotional_categories": ["joy"],
            "importance": 0.8,
            "importance_reasons": ["life_event"],
            "time_of_day": "morning",
            "day_of_week": "monday",
        }

        meta = MemoryMetadata.from_dict(d)
        assert meta.emotional_valence == 0.7
        assert meta.emotional_categories == ["joy"]

    def test_merge_with_preserves_existing(self):
        """Merge should preserve existing metadata."""
        meta = MemoryMetadata(emotional_valence=0.7)
        existing = {"enc": True, "session_id": "abc123"}

        merged = meta.merge_with(existing)
        assert merged["enc"] is True
        assert merged["session_id"] == "abc123"
        assert merged["emotional_valence"] == 0.7

    def test_applied_count_defaults_to_zero(self):
        """New #1326 fields default-zero so existing rows / callers see no
        change in shape until they're explicitly populated."""
        meta = MemoryMetadata()
        assert meta.applied_count == 0
        assert meta.last_applied is None

    def test_applied_fields_roundtrip(self):
        """``applied_count`` + ``last_applied`` survive to_dict / from_dict."""
        meta = MemoryMetadata(
            applied_count=4,
            last_applied="2026-05-20T16:00:00+00:00",
        )
        roundtripped = MemoryMetadata.from_dict(meta.to_dict())
        assert roundtripped.applied_count == 4
        assert roundtripped.last_applied == "2026-05-20T16:00:00+00:00"

    def test_from_dict_missing_applied_fields_is_zero(self):
        """Pre-#1326 metadata rows (without applied_count/last_applied)
        deserialize cleanly with defaults — no KeyError, no breakage."""
        legacy = {
            "emotional_valence": 0.5,
            "access_count": 7,
            "last_accessed": "2026-05-15T12:00:00+00:00",
            # no applied_count / last_applied
        }
        meta = MemoryMetadata.from_dict(legacy)
        assert meta.applied_count == 0
        assert meta.last_applied is None
        # Existing fields still come through.
        assert meta.access_count == 7


class TestTemporalPattern:
    """Tests for TemporalPattern model."""

    def test_to_dict(self):
        """Should serialize correctly."""
        pattern = TemporalPattern(
            id="pattern_1",
            agent_id="agent_1",
            pattern_type="time_preference",
            description="Most active late at night",
            trigger_conditions={"time_of_day": "late_night"},
            confidence=0.8,
            observations=10,
        )

        d = pattern.to_dict()
        assert d["id"] == "pattern_1"
        assert d["trigger_conditions"] == {"time_of_day": "late_night"}


class TestMemoryEpisode:
    """Tests for MemoryEpisode model."""

    def test_to_dict(self):
        """Should serialize correctly."""
        episode = MemoryEpisode(
            id="episode_1",
            agent_id="agent_1",
            title="A difficult conversation",
            summary="Worked through some challenges",
            emotional_arc="difficulty → resolution",
            key_message_ids=["msg1", "msg2"],
        )

        d = episode.to_dict()
        assert d["title"] == "A difficult conversation"
        assert d["emotional_arc"] == "difficulty → resolution"


class TestMemoryConsolidatorKG:
    """Tests for MemoryConsolidator writing episodes to the Knowledge Graph."""

    @pytest.mark.asyncio
    async def test_save_episode_writes_kg_node(self):
        """When graph_store is provided, _save_episode should create a KG node and edge."""
        from kestrel_sovereign.storage.memory_consolidator import MemoryConsolidator

        mock_db = AsyncMock()
        mock_graph = AsyncMock()

        consolidator = MemoryConsolidator(
            db=mock_db,
            agent_id="did:test:agent123",
            graph_store=mock_graph,
        )

        episode = MemoryEpisode(
            id="episode:did:test:agent123:2026-03-15:abc12345",
            agent_id="did:test:agent123",
            title="A joyful moment",
            summary="A conversation with 5 messages. Emotional trajectory: generally positive.",
            emotional_arc="generally positive",
            key_message_ids=["1", "2", "3", "4", "5"],
            timespan_start=datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc),
            timespan_end=datetime(2026, 3, 15, 11, 0, tzinfo=timezone.utc),
        )

        await consolidator._save_episode(episode)

        # Should have written to memory_episodes table
        mock_db.execute.assert_called_once()

        # Should have written a KG node
        mock_graph.add_node.assert_called_once()
        node = mock_graph.add_node.call_args[0][0]
        assert node.node_id == episode.id
        assert node.node_type == "episode"
        assert node.label == "A joyful moment"
        assert node.properties["source"] == "consolidator"
        assert node.properties["message_count"] == 5
        assert node.properties["emotional_arc"] == "generally positive"

        # Should have created an edge from agent to episode
        mock_graph.add_edge.assert_called_once_with(
            "did:test:agent123", episode.id, "remembers"
        )

    @pytest.mark.asyncio
    async def test_save_episode_without_graph_store(self):
        """Without graph_store, _save_episode should still save to DB without error."""
        from kestrel_sovereign.storage.memory_consolidator import MemoryConsolidator

        mock_db = AsyncMock()

        consolidator = MemoryConsolidator(
            db=mock_db,
            agent_id="did:test:agent123",
        )

        episode = MemoryEpisode(
            id="episode:test:2026-03-15:xyz",
            agent_id="did:test:agent123",
            title="Test episode",
            summary="Test summary",
            emotional_arc="neutral",
            key_message_ids=["1", "2", "3"],
        )

        await consolidator._save_episode(episode)

        # Should have written to DB
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_episode_kg_failure_is_nonfatal(self):
        """If KG write fails, the episode should still be saved to the DB."""
        from kestrel_sovereign.storage.memory_consolidator import MemoryConsolidator

        mock_db = AsyncMock()
        mock_graph = AsyncMock()
        mock_graph.add_node.side_effect = Exception("KG write failed")

        consolidator = MemoryConsolidator(
            db=mock_db,
            agent_id="did:test:agent123",
            graph_store=mock_graph,
        )

        episode = MemoryEpisode(
            id="episode:test:2026-03-15:fail",
            agent_id="did:test:agent123",
            title="Should still save",
            summary="Test",
            emotional_arc="neutral",
            key_message_ids=["1"],
        )

        # Should not raise
        await consolidator._save_episode(episode)

        # DB write should still have happened
        mock_db.execute.assert_called_once()


class TestMemoryConsolidatorEpisodeCreation:
    """Tests for #1489 — scheduled consolidation must produce retrievable episodes."""

    @pytest.fixture
    def _now(self):
        # Pin to mid-day UTC so message timestamps generated via
        # ``_now - timedelta(days=1) + timedelta(minutes=N*5)`` never
        # straddle midnight UTC. A live ``datetime.now`` made
        # ``test_partial_day_consolidation_picks_up_only_new_messages``
        # flaky depending on UTC time-of-day at CI launch — the 45-min
        # span produced two ``date_key`` groups when launched within
        # 45 min of midnight UTC, yielding two episodes instead of one.
        return datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

    def _make_rows(self, count, *, date, metadata=None, agent_id="did:test:agent1"):
        """Create mock conversation_history rows (id, content, metadata, created_at, role)."""
        import json
        rows = []
        for i in range(count):
            ts = (date + timedelta(minutes=i * 5)).isoformat()
            meta = json.dumps(metadata or {})
            rows.append((i + 1, f"message {i}", meta, ts, "user" if i % 2 == 0 else "assistant"))
        return rows

    @pytest.mark.asyncio
    async def test_unenriched_messages_produce_episodes(self, _now):
        """Messages without emotional metadata should still produce episodes (#1489)."""
        from kestrel_sovereign.storage.memory_consolidator import MemoryConsolidator

        date = _now - timedelta(days=1)
        rows = self._make_rows(5, date=date)

        mock_db = AsyncMock()
        mock_db.fetchall.return_value = rows
        mock_db.execute = AsyncMock()
        # No prior episode for this day — idempotency probe returns None
        mock_db.fetchval.return_value = None

        consolidator = MemoryConsolidator(db=mock_db, agent_id="did:test:agent1")
        episodes, skipped = await consolidator._create_episodes()

        assert len(episodes) >= 1, (
            f"Expected at least 1 episode from 5 unenriched messages, "
            f"got {len(episodes)}; skipped={skipped}"
        )

    @pytest.mark.asyncio
    async def test_enriched_high_importance_messages_produce_episodes(self, _now):
        """Messages with high emotional metadata should produce episodes."""
        from kestrel_sovereign.storage.memory_consolidator import MemoryConsolidator

        date = _now - timedelta(days=1)
        meta = {"emotional_intensity": 0.8, "importance": 0.9, "emotional_categories": ["joy"]}
        rows = self._make_rows(5, date=date, metadata=meta)

        mock_db = AsyncMock()
        mock_db.fetchall.return_value = rows
        mock_db.execute = AsyncMock()
        mock_db.fetchval.return_value = None

        consolidator = MemoryConsolidator(db=mock_db, agent_id="did:test:agent1")
        episodes, skipped = await consolidator._create_episodes()

        assert len(episodes) >= 1

    @pytest.mark.asyncio
    async def test_enriched_low_importance_messages_skipped_with_reason(self, _now):
        """Enriched messages below threshold should be skipped with a clear reason."""
        from kestrel_sovereign.storage.memory_consolidator import MemoryConsolidator

        date = _now - timedelta(days=1)
        meta = {"emotional_intensity": 0.1, "importance": 0.3, "emotional_categories": ["neutral"]}
        rows = self._make_rows(5, date=date, metadata=meta)

        mock_db = AsyncMock()
        mock_db.fetchall.return_value = rows
        mock_db.execute = AsyncMock()

        consolidator = MemoryConsolidator(db=mock_db, agent_id="did:test:agent1")
        episodes, skipped = await consolidator._create_episodes()

        assert len(episodes) == 0
        assert len(skipped) >= 1
        reasons = [r for _, _, r in skipped]
        assert "below_emotional_threshold" in reasons

    @pytest.mark.asyncio
    async def test_too_few_messages_skipped_with_reason(self, _now):
        """Clusters below MIN_EPISODE_MESSAGES should report below_min_messages."""
        from kestrel_sovereign.storage.memory_consolidator import MemoryConsolidator

        date = _now - timedelta(days=1)
        rows = self._make_rows(2, date=date)

        mock_db = AsyncMock()
        mock_db.fetchall.return_value = rows
        mock_db.execute = AsyncMock()

        consolidator = MemoryConsolidator(db=mock_db, agent_id="did:test:agent1")
        episodes, skipped = await consolidator._create_episodes()

        assert len(episodes) == 0
        assert len(skipped) >= 1
        reasons = [r for _, _, r in skipped]
        assert "below_min_messages" in reasons

    @pytest.mark.asyncio
    async def test_run_consolidation_reports_skip_reasons(self, _now):
        """run_consolidation report should include skip_reasons (#1489)."""
        from kestrel_sovereign.storage.memory_consolidator import MemoryConsolidator

        date = _now - timedelta(days=1)
        rows = self._make_rows(5, date=date)

        mock_db = AsyncMock()
        # fetchall sequence: conversation_history, _covered_message_ids,
        # then any remaining queries from later phases (empty results).
        mock_db.fetchall.side_effect = [
            rows,   # conversation_history
            [],     # _covered_message_ids: no prior episodes
            [],     # _detect_patterns (or whatever follows)
            [],     # _archive_decayed
            [],     # safety extra
        ]
        mock_db.execute = AsyncMock()
        mock_db.fetchval.return_value = 5  # COUNT(*) total_messages_processed

        consolidator = MemoryConsolidator(db=mock_db, agent_id="did:test:agent1")
        report = await consolidator.run_consolidation()

        assert "error" not in report, f"Unexpected error in report: {report.get('error')}"
        assert "clusters_skipped" in report
        assert "skip_reasons" in report
        assert report["episodes_created"] >= 1

    @pytest.mark.asyncio
    async def test_load_marker_state_uses_correct_db_attr(self):
        """_load_marker_state must use self._db, not self.db (#1489 bugfix)."""
        from kestrel_sovereign.storage.memory_consolidator import MemoryConsolidator

        mock_db = AsyncMock()
        mock_db.fetchone.return_value = None

        consolidator = MemoryConsolidator(db=mock_db, agent_id="did:test:agent1")

        # Should NOT raise AttributeError (the pre-fix code used self.db)
        result = await consolidator._load_marker_state(42)

        assert result is None
        mock_db.fetchone.assert_called_once()

    @pytest.mark.asyncio
    async def test_episodes_retrievable_after_consolidation(self, _now):
        """After consolidation, get_episodes should return the created episodes."""
        from kestrel_sovereign.storage.memory_consolidator import MemoryConsolidator

        date = _now - timedelta(days=1)
        rows = self._make_rows(5, date=date)

        mock_db = AsyncMock()
        # fetchall sequence: conversation_history, _covered_message_ids
        # (empty — no prior episodes), then later-phase queries.
        mock_db.fetchall.side_effect = [
            rows,   # conversation_history
            [],     # _covered_message_ids
            [],     # _detect_patterns
            [],     # _archive_decayed
            [],     # safety extra
        ]
        mock_db.execute = AsyncMock()
        mock_db.fetchval.return_value = 5  # COUNT(*) total_messages_processed

        consolidator = MemoryConsolidator(db=mock_db, agent_id="did:test:agent1")
        report = await consolidator.run_consolidation()

        assert "error" not in report, f"Unexpected error in report: {report.get('error')}"
        assert report["episodes_created"] >= 1
        # Verify _save_episode was called (INSERT into memory_episodes)
        insert_calls = [
            c for c in mock_db.execute.call_args_list
            if "memory_episodes" in str(c)
        ]
        assert len(insert_calls) >= 1, "Expected at least one INSERT into memory_episodes"

    @pytest.mark.asyncio
    async def test_day_already_fully_consolidated_is_skipped_not_duplicated(self, _now):
        """A day whose messages are all already in an episode must not be
        re-consolidated on the next nightly run (#1489 P2 — codex round 1)."""
        import json
        from kestrel_sovereign.storage.memory_consolidator import MemoryConsolidator

        date = _now - timedelta(days=1)
        rows = self._make_rows(5, date=date)  # ids 1..5

        mock_db = AsyncMock()
        # First fetchall is conversation_history; second is the dedup probe
        # against memory_episodes returning an existing episode that covers
        # every message in the cluster.
        mock_db.fetchall.side_effect = [
            rows,
            [(json.dumps(["1", "2", "3", "4", "5"]),)],
        ]
        mock_db.execute = AsyncMock()
        mock_db.fetchval.return_value = None

        consolidator = MemoryConsolidator(db=mock_db, agent_id="did:test:agent1")
        episodes, skipped = await consolidator._create_episodes()

        assert len(episodes) == 0, (
            "A day with all messages covered by an existing episode must not "
            "produce a duplicate"
        )
        reasons = [r for _, _, r in skipped]
        assert "already_consolidated" in reasons

    @pytest.mark.asyncio
    async def test_partial_day_consolidation_picks_up_only_new_messages(self, _now):
        """A midday consolidation that produced an episode from the morning's
        messages must not lock the afternoon out: the next run should pick up
        only the new messages (#1489 P2 — codex round 2)."""
        import json
        from kestrel_sovereign.storage.memory_consolidator import MemoryConsolidator

        date = _now - timedelta(days=1)
        rows = self._make_rows(10, date=date)  # ids 1..10

        mock_db = AsyncMock()
        # Existing episode covers only the early ids 1..3.
        mock_db.fetchall.side_effect = [
            rows,
            [(json.dumps(["1", "2", "3"]),)],
        ]
        mock_db.execute = AsyncMock()
        mock_db.fetchval.return_value = None

        consolidator = MemoryConsolidator(db=mock_db, agent_id="did:test:agent1")
        episodes, skipped = await consolidator._create_episodes()

        assert len(episodes) == 1, "Should create an episode for the new messages"
        assert episodes[0].key_message_ids == ["4", "5", "6", "7", "8", "9", "10"], (
            "Episode should cover only the messages not already in a prior episode"
        )

    @pytest.mark.asyncio
    async def test_session_episode_covers_messages_consolidator_dedups(self, _now):
        """A session episode (id format ``episode:<agent>:YYYY-MM-DD-HHMM:<suffix>``)
        covering the cluster's messages must dedup the nightly consolidator,
        not just same-format daily episodes (#1489 P2 — codex round 3)."""
        import json
        from kestrel_sovereign.storage.memory_consolidator import MemoryConsolidator

        date = _now - timedelta(days=1)
        rows = self._make_rows(5, date=date)

        mock_db = AsyncMock()
        # The episodes probe is now a single agent-wide query (no LIKE on
        # the date), so it returns both consolidator-format AND session-format
        # episodes. Here only a session episode exists, and it already covers
        # all 5 messages.
        mock_db.fetchall.side_effect = [
            rows,
            [(json.dumps(["1", "2", "3", "4", "5"]),)],
        ]
        mock_db.execute = AsyncMock()
        mock_db.fetchval.return_value = None

        consolidator = MemoryConsolidator(db=mock_db, agent_id="did:test:agent1")
        episodes, skipped = await consolidator._create_episodes()

        assert len(episodes) == 0
        reasons = [r for _, _, r in skipped]
        assert "already_consolidated" in reasons

    @pytest.mark.asyncio
    async def test_dedup_gates_run_on_new_messages_not_full_cluster(self, _now):
        """Emotional-threshold averaging must run on post-dedup messages so
        a few new high-importance messages aren't shadowed by many old
        low-importance ones (#1489 P2 — codex round 4)."""
        import json
        from kestrel_sovereign.storage.memory_consolidator import MemoryConsolidator

        date = _now - timedelta(days=1)
        # 7 already-covered low-importance enriched messages
        # + 5 new high-importance enriched messages.
        low = {"emotional_intensity": 0.05, "importance": 0.1, "emotional_categories": ["neutral"]}
        high = {"emotional_intensity": 0.9, "importance": 0.95, "emotional_categories": ["joy"]}
        low_rows = []
        for i in range(7):
            ts = (date + timedelta(minutes=i * 5)).isoformat()
            low_rows.append((i + 1, f"low {i}", json.dumps(low), ts, "user"))
        high_rows = []
        for i in range(5):
            ts = (date + timedelta(minutes=(7 + i) * 5)).isoformat()
            high_rows.append((8 + i, f"high {i}", json.dumps(high), ts, "user"))
        rows = low_rows + high_rows

        mock_db = AsyncMock()
        # Covered: ids 1..7 (the low-importance ones)
        mock_db.fetchall.side_effect = [
            rows,
            [(json.dumps([str(i) for i in range(1, 8)]),)],
        ]
        mock_db.execute = AsyncMock()
        mock_db.fetchval.return_value = None

        consolidator = MemoryConsolidator(db=mock_db, agent_id="did:test:agent1")
        episodes, skipped = await consolidator._create_episodes()

        assert len(episodes) == 1, (
            "New high-importance messages must produce an episode even when "
            "the full-day averages (including covered messages) would fail "
            "the emotional gate"
        )
        assert episodes[0].key_message_ids == ["8", "9", "10", "11", "12"]

    @pytest.mark.asyncio
    async def test_partial_day_below_min_after_dedup_is_skipped(self, _now):
        """If only a few new messages remain after dedup and they fall below
        MIN_EPISODE_MESSAGES, the cluster is skipped with a distinct reason
        rather than padded with already-covered messages."""
        import json
        from kestrel_sovereign.storage.memory_consolidator import MemoryConsolidator

        date = _now - timedelta(days=1)
        rows = self._make_rows(6, date=date)  # ids 1..6

        mock_db = AsyncMock()
        # Existing episode covers ids 1..5; only id 6 is new.
        mock_db.fetchall.side_effect = [
            rows,
            [(json.dumps(["1", "2", "3", "4", "5"]),)],
        ]
        mock_db.execute = AsyncMock()
        mock_db.fetchval.return_value = None

        consolidator = MemoryConsolidator(db=mock_db, agent_id="did:test:agent1")
        episodes, skipped = await consolidator._create_episodes()

        assert len(episodes) == 0
        reasons = [r for _, _, r in skipped]
        assert "below_min_after_dedup" in reasons


class TestConsolidateForgetting:
    """MemorySystem.consolidate() is the single chokepoint that runs the
    forgetting deletion tier (#1674 P3) — so both the tool and the nightly
    sleep cycle forget identically. Tests target it directly."""

    @staticmethod
    def _make_ms(monkeypatch, *, run_result, enabled=True,
                 delete_threshold=0.02, grace_days=90):
        from kestrel_sovereign.storage import MemorySystem
        import kestrel_sovereign.storage.retention as retention_mod

        monkeypatch.setattr(
            retention_mod, "load_forgetting_config",
            lambda: {"enabled": enabled, "delete_threshold": delete_threshold,
                     "grace_days": grace_days},
        )
        ms = MemorySystem(storage=MagicMock(), agent_id="did:test:fgt")
        ms.consolidator = MagicMock()
        ms.consolidator.run_consolidation = AsyncMock(return_value=run_result)
        ms.storage.purge_decayed_episodes = AsyncMock(return_value=3)
        return ms

    @pytest.mark.asyncio
    async def test_runs_forgetting_when_enabled(self, monkeypatch):
        ms = self._make_ms(
            monkeypatch,
            run_result={"episodes_created": 2, "messages_archived": 0},
            enabled=True, delete_threshold=0.05, grace_days=45,
        )
        report = await ms.consolidate()
        ms.storage.purge_decayed_episodes.assert_awaited_once_with(
            delete_threshold=0.05, grace_days=45, reason="forgetting",
        )
        assert report["episodes_deleted"] == 3
        assert report["episodes_created"] == 2

    @pytest.mark.asyncio
    async def test_skips_forgetting_when_disabled(self, monkeypatch):
        ms = self._make_ms(
            monkeypatch, run_result={"episodes_created": 1}, enabled=False,
        )
        report = await ms.consolidate()
        ms.storage.purge_decayed_episodes.assert_not_awaited()
        assert report["episodes_deleted"] == 0

    @pytest.mark.asyncio
    async def test_skips_forgetting_when_consolidation_errored(self, monkeypatch):
        ms = self._make_ms(
            monkeypatch, run_result={"error": "salvage unavailable"}, enabled=True,
        )
        report = await ms.consolidate()
        ms.storage.purge_decayed_episodes.assert_not_awaited()
        assert report["episodes_deleted"] == 0
        assert "error" in report

    @pytest.mark.asyncio
    async def test_forgetting_failure_does_not_fail_consolidation(self, monkeypatch):
        ms = self._make_ms(
            monkeypatch, run_result={"episodes_created": 1}, enabled=True,
        )
        ms.storage.purge_decayed_episodes = AsyncMock(
            side_effect=RuntimeError("graph store down"))
        report = await ms.consolidate()
        assert report["episodes_deleted"] == 0
        assert report["episodes_created"] == 1
        assert "error" not in report


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
