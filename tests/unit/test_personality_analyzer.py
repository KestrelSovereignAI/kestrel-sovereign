#!/usr/bin/env pytest
"""
Unit tests for the Personality Analyzer module.

Tests the PersonalityAnalyzer, CalibrationPromptGenerator, and related functions.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from kestrel_sovereign.identity import (
    PersonalityFingerprint,
    PersonalityAnalyzer,
    CalibrationPromptGenerator,
    AnalysisResult,
    analyze_personality,
    generate_calibration_prompt,
)


class TestPersonalityAnalyzer:
    """Tests for PersonalityAnalyzer class."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database."""
        db = AsyncMock()
        return db

    @pytest.fixture
    def analyzer(self, mock_db):
        """Create an analyzer with mock db."""
        return PersonalityAnalyzer(
            db=mock_db,
            agent_id="did:test:agent123",
            sample_limit=100
        )

    @pytest.mark.asyncio
    async def test_analyze_no_history(self, analyzer, mock_db):
        """Test analysis with no conversation history."""
        mock_db.fetchall.return_value = []

        result = await analyzer.analyze()

        assert isinstance(result, AnalysisResult)
        assert result.confidence == 0.0
        assert result.sample_size == 0
        assert "No conversation history" in result.analysis_notes[0]

    @pytest.mark.asyncio
    async def test_analyze_with_responses(self, analyzer, mock_db):
        """Test analysis with sample responses."""
        # Mock responses that indicate formal, structured style
        mock_responses = [
            ("I will analyze this problem carefully. The solution involves:\n\n1. First step\n2. Second step\n\nPlease let me know if you need clarification.",),
            ("Furthermore, I would like to add that the implementation requires careful consideration of all factors.",),
            ("Here is the code solution:\n\n```python\ndef example():\n    pass\n```\n\nThis should work for your use case.",),
        ] * 10  # Repeat to get 30 samples

        # First call returns responses, second returns empty for calibration
        mock_db.fetchall.side_effect = [
            mock_responses,
            []  # No calibration examples
        ]

        result = await analyzer.analyze()

        assert result.sample_size == 30
        assert result.confidence > 0.5
        assert result.fingerprint.uses_lists is True
        assert result.fingerprint.uses_code_blocks is True

    def test_analyze_structure_short_responses(self, analyzer):
        """Test structural analysis with short responses."""
        responses = ["Yes.", "No.", "Done.", "Thanks."] * 10

        result = analyzer._analyze_structure(responses)

        assert result["length_preference"] == "short"
        assert result["verbosity"] == "terse"

    def test_analyze_structure_long_responses(self, analyzer):
        """Test structural analysis with long responses."""
        long_text = "This is a detailed explanation. " * 50
        responses = [long_text] * 10

        result = analyzer._analyze_structure(responses)

        assert result["length_preference"] == "long"
        assert result["verbosity"] == "verbose"

    def test_analyze_structure_lists(self, analyzer):
        """Test detection of list usage."""
        responses = [
            "Here are the steps:\n- Step 1\n- Step 2\n- Step 3",
            "The items include:\n* Item A\n* Item B",
        ] * 5

        result = analyzer._analyze_structure(responses)

        assert result["uses_lists"] is True

    def test_analyze_structure_code_blocks(self, analyzer):
        """Test detection of code block usage."""
        responses = [
            "Here is the code:\n```python\nprint('hello')\n```",
        ] * 5

        result = analyzer._analyze_structure(responses)

        assert result["uses_code_blocks"] is True

    def test_analyze_lexical_formal(self, analyzer):
        """Test lexical analysis for formal style."""
        responses = [
            "Furthermore, I shall endeavor to provide a comprehensive analysis.",
            "Consequently, the aforementioned solution shall be implemented.",
            "Therefore, it is imperative that we proceed accordingly.",
        ] * 10

        result = analyzer._analyze_lexical(responses)

        assert result["formality"] > 0.6

    def test_analyze_lexical_casual(self, analyzer):
        """Test lexical analysis for casual style."""
        responses = [
            "Hey! That's gonna work great! Don't worry about it. I can't believe it.",
            "Yeah, I don't think we should do that. It won't work. That's not ideal.",
            "Cool, let's just sorta wing it. I can't wait to try! It'll be fun.",
        ] * 20  # More samples with multiple contractions to exceed threshold

        result = analyzer._analyze_lexical(responses)

        assert result["formality"] < 0.4
        assert result["uses_contractions"] is True

    def test_analyze_emotional_empathetic(self, analyzer):
        """Test emotional analysis for empathetic style."""
        responses = [
            "I understand how frustrating that must be. I feel for you.",
            "I appreciate you sharing that with me. I realize this is hard.",
            "That sounds really difficult. I'm glad you reached out. I understand.",
            "I can see why that would be concerning. I appreciate your patience.",
        ] * 20  # More samples with multiple empathy markers

        result = analyzer._analyze_emotional(responses)

        # With multiple markers per response and more samples, empathy should be notable
        assert result["empathy_level"] > 0.3

    def test_analyze_emotional_emojis(self, analyzer):
        """Test emoji detection."""
        responses = [
            "Great job! 😊",
            "That's awesome! 🎉",
            "Let me help you 🤔",
        ] * 5

        result = analyzer._analyze_emotional(responses)

        assert result["uses_emojis"] is True

    def test_analyze_stylistic_humor(self, analyzer):
        """Test humor detection."""
        responses = [
            "Haha, that's a good one!",
            "Just kidding, but seriously...",
            "That made me laugh :D",
        ] * 5

        result = analyzer._analyze_stylistic(responses)

        assert result["humor_style"] == "playful"

    def test_analyze_stylistic_no_humor(self, analyzer):
        """Test detection of no humor."""
        responses = [
            "The solution is straightforward.",
            "Here is the implementation.",
            "Please review the changes.",
        ] * 10

        result = analyzer._analyze_stylistic(responses)

        assert result["humor_style"] is None

    def test_extract_vocabulary_preferences(self, analyzer):
        """Test vocabulary preference extraction."""
        responses = [
            "The implementation uses kubernetes for orchestration.",
            "This kubernetes deployment requires proper configuration.",
            "For kubernetes, we need to consider scalability.",
        ] * 5

        vocab = analyzer._extract_vocabulary_preferences(responses)

        assert "kubernetes" in vocab

    def test_synthesize_fingerprint_formal(self, analyzer):
        """Test fingerprint synthesis for formal style."""
        structural = {"avg_length": 400, "length_preference": "medium", "verbosity": "moderate",
                     "uses_lists": True, "uses_code_blocks": False, "uses_headers": False}
        lexical = {"formality": 0.8, "directness": 0.5, "uses_contractions": False}
        emotional = {"empathy_level": 0.3, "expressiveness": 0.2, "uses_emojis": False,
                    "emotional_baseline": 0.4}
        stylistic = {"humor_style": None, "preferred_greeting": None, "preferred_signoff": None}

        fp = analyzer._synthesize_fingerprint(structural, lexical, emotional, stylistic, [], [])

        assert fp.communication_style == "formal"
        assert fp.formality_level > 0.7

    def test_synthesize_fingerprint_warm(self, analyzer):
        """Test fingerprint synthesis for warm style."""
        structural = {"avg_length": 400, "length_preference": "medium", "verbosity": "moderate",
                     "uses_lists": True, "uses_code_blocks": True, "uses_headers": False}
        lexical = {"formality": 0.5, "directness": 0.4, "uses_contractions": True}
        emotional = {"empathy_level": 0.8, "expressiveness": 0.5, "uses_emojis": False,
                    "emotional_baseline": 0.6}
        stylistic = {"humor_style": "occasional", "preferred_greeting": "hello",
                    "preferred_signoff": None}

        fp = analyzer._synthesize_fingerprint(structural, lexical, emotional, stylistic, [], [])

        assert fp.communication_style == "warm"
        assert fp.empathy_level > 0.6


class TestCalibrationPromptGenerator:
    """Tests for CalibrationPromptGenerator class."""

    @pytest.fixture
    def formal_fingerprint(self):
        """Create a formal personality fingerprint."""
        return PersonalityFingerprint(
            communication_style="formal",
            formality_level=0.8,
            verbosity_preference="moderate",
            emotional_baseline=0.3,
            humor_style=None,
            empathy_level=0.4,
            typical_response_length="medium",
            uses_lists=True,
            uses_code_blocks=True,
            uses_emojis=False,
            calibration_examples=[
                {"input": "Hello", "output": "Greetings. How may I assist you today?"},
            ],
            vocabulary_preferences=["comprehensive", "therefore"],
        )

    @pytest.fixture
    def playful_fingerprint(self):
        """Create a playful personality fingerprint."""
        return PersonalityFingerprint(
            communication_style="playful",
            formality_level=0.2,
            verbosity_preference="moderate",
            emotional_baseline=0.8,
            humor_style="playful",
            empathy_level=0.7,
            typical_response_length="medium",
            uses_lists=False,
            uses_code_blocks=False,
            uses_emojis=True,
            preferred_greeting="hey",
            preferred_signoff="Cheers",
            calibration_examples=[
                {"input": "Hi!", "output": "Hey there! 😊 What's up?"},
            ],
        )

    def test_generate_system_prompt_formal(self, formal_fingerprint):
        """Test system prompt generation for formal style."""
        generator = CalibrationPromptGenerator(formal_fingerprint)
        prompt = generator.generate_system_prompt_addition()

        assert "Communication Style" in prompt
        assert "formal" in prompt.lower()
        assert "Avoid using emojis" in prompt

    def test_generate_system_prompt_playful(self, playful_fingerprint):
        """Test system prompt generation for playful style."""
        generator = CalibrationPromptGenerator(playful_fingerprint)
        prompt = generator.generate_system_prompt_addition()

        assert "playful" in prompt.lower()
        assert "emojis sparingly" in prompt
        assert "hey" in prompt.lower()
        assert "Cheers" in prompt

    def test_generate_few_shot_prompt(self, formal_fingerprint):
        """Test few-shot prompt generation."""
        generator = CalibrationPromptGenerator(formal_fingerprint)
        prompt = generator.generate_few_shot_prompt()

        assert "Response Examples" in prompt
        assert "Hello" in prompt
        assert "Greetings" in prompt

    def test_generate_few_shot_prompt_empty(self):
        """Test few-shot with no examples."""
        fp = PersonalityFingerprint()
        generator = CalibrationPromptGenerator(fp)
        prompt = generator.generate_few_shot_prompt()

        assert prompt == ""

    def test_generate_full_calibration(self, formal_fingerprint):
        """Test full calibration prompt generation."""
        generator = CalibrationPromptGenerator(formal_fingerprint)
        prompt = generator.generate_full_calibration()

        assert "Personality Calibration" in prompt
        assert "Communication Style" in prompt
        assert "Response Examples" in prompt


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.mark.asyncio
    async def test_analyze_personality(self):
        """Test the analyze_personality convenience function."""
        mock_db = AsyncMock()
        mock_db.fetchall.return_value = []

        result = await analyze_personality(mock_db, "did:test:123")

        assert isinstance(result, AnalysisResult)
        assert result.sample_size == 0

    def test_generate_calibration_prompt(self):
        """Test the generate_calibration_prompt convenience function."""
        fp = PersonalityFingerprint(
            communication_style="warm",
            formality_level=0.5,
        )

        prompt = generate_calibration_prompt(fp)

        assert "Personality Calibration" in prompt
        assert "warm" in prompt.lower() or "balanced" in prompt.lower()


class TestPersonalityConsistency:
    """Tests for personality consistency across analysis runs."""

    @pytest.fixture
    def consistent_responses(self):
        """Create responses with consistent style."""
        return [
            "I understand your concern. Let me help you with that.",
            "I appreciate you bringing this up. Here's what we can do:",
            "That makes sense. I'll explain the solution step by step.",
            "I see what you mean. The approach I'd recommend is:",
        ] * 20

    def test_consistent_style_detection(self, consistent_responses):
        """Test that consistent responses yield consistent fingerprint."""
        mock_db = AsyncMock()
        analyzer = PersonalityAnalyzer(mock_db, "did:test:123")

        # Run analysis multiple times
        results = []
        for _ in range(3):
            structural = analyzer._analyze_structure(consistent_responses)
            lexical = analyzer._analyze_lexical(consistent_responses)
            emotional = analyzer._analyze_emotional(consistent_responses)
            stylistic = analyzer._analyze_stylistic(consistent_responses)

            fp = analyzer._synthesize_fingerprint(
                structural, lexical, emotional, stylistic, [], []
            )
            results.append(fp)

        # All fingerprints should be identical
        for fp in results[1:]:
            assert fp.communication_style == results[0].communication_style
            assert fp.formality_level == results[0].formality_level
            assert fp.empathy_level == results[0].empathy_level

    def test_style_differentiates(self):
        """Test that different styles produce different fingerprints."""
        mock_db = AsyncMock()
        analyzer = PersonalityAnalyzer(mock_db, "did:test:123")

        formal_responses = [
            "Furthermore, I shall provide a comprehensive analysis.",
            "Consequently, the implementation requires careful consideration.",
        ] * 10

        casual_responses = [
            "Hey! That's gonna work great, don't you think?",
            "Yeah, let's just go for it! Sounds cool to me.",
        ] * 10

        formal_lexical = analyzer._analyze_lexical(formal_responses)
        casual_lexical = analyzer._analyze_lexical(casual_responses)

        # Formal should have higher formality score
        assert formal_lexical["formality"] > casual_lexical["formality"]
        assert formal_lexical["formality"] - casual_lexical["formality"] > 0.2
