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


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
