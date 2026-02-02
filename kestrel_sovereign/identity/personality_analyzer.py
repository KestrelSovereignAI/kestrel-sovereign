#!/usr/bin/env python3
"""
Personality Analyzer: Extract and calibrate agent personality fingerprint.

This module provides advanced personality analysis capabilities for capturing
"how I communicate" separate from "what I know". The analyzer examines
conversation history to extract:
- Communication style patterns
- Vocabulary preferences
- Response structure tendencies
- Emotional baseline
- Calibration examples for substrate adaptation

Phase 2 of Issue #23: Substrate-Independent Agent Portability.
"""
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .identity_package import PersonalityFingerprint

if TYPE_CHECKING:
    from kestrel_sovereign.storage.async_database import AsyncDatabase

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Results from personality analysis."""
    fingerprint: PersonalityFingerprint
    confidence: float = 0.0  # 0.0-1.0 based on sample size
    sample_size: int = 0
    analysis_notes: List[str] = field(default_factory=list)


class PersonalityAnalyzer:
    """
    Analyze agent responses to extract personality fingerprint.

    Uses multiple analysis dimensions:
    1. Structural analysis (length, formatting, structure)
    2. Lexical analysis (vocabulary, contractions, formality markers)
    3. Emotional analysis (sentiment indicators, empathy markers)
    4. Stylistic analysis (tone, humor, engagement patterns)
    """

    # Formality indicators
    FORMAL_MARKERS = {
        "furthermore", "moreover", "consequently", "therefore", "thus",
        "regarding", "concerning", "accordingly", "nevertheless", "notwithstanding",
        "shall", "hereby", "herein", "wherein", "thereafter"
    }

    INFORMAL_MARKERS = {
        "gonna", "wanna", "kinda", "sorta", "yeah", "yep", "nope",
        "hey", "hi", "okay", "ok", "cool", "awesome", "great",
        "stuff", "things", "basically", "literally", "actually"
    }

    # Contractions (informal)
    CONTRACTIONS = {
        "don't", "won't", "can't", "couldn't", "wouldn't", "shouldn't",
        "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't",
        "hadn't", "doesn't", "didn't", "i'm", "you're", "we're", "they're",
        "it's", "that's", "what's", "there's", "here's", "let's", "i've",
        "you've", "we've", "they've", "i'll", "you'll", "we'll", "they'll",
        "i'd", "you'd", "we'd", "they'd"
    }

    # Empathy markers
    EMPATHY_MARKERS = {
        "understand", "appreciate", "realize", "recognize", "feel",
        "sorry", "glad", "happy", "pleased", "concerned", "worried",
        "i see", "i hear you", "makes sense", "that's tough", "that's great"
    }

    # Hedging phrases (uncertainty, politeness)
    HEDGING_PHRASES = {
        "i think", "i believe", "perhaps", "maybe", "possibly", "might",
        "could be", "seems like", "appears to", "in my opinion", "from my perspective"
    }

    # Directness markers
    DIRECT_MARKERS = {
        "you should", "you need to", "you must", "do this", "don't do",
        "the answer is", "the solution is", "simply", "just"
    }

    def __init__(
        self,
        db: "AsyncDatabase",
        agent_id: str,
        sample_limit: int = 200
    ):
        """
        Initialize the analyzer.

        Args:
            db: Database connection
            agent_id: The agent's DID
            sample_limit: Maximum number of responses to analyze
        """
        self.db = db
        self.agent_id = agent_id
        self.sample_limit = sample_limit

    async def analyze(self) -> AnalysisResult:
        """
        Perform comprehensive personality analysis.

        Returns:
            AnalysisResult with fingerprint and analysis metadata
        """
        # Fetch assistant responses
        responses = await self._get_responses()

        if not responses:
            logger.info("No conversation history found for personality analysis")
            return AnalysisResult(
                fingerprint=PersonalityFingerprint(),
                confidence=0.0,
                sample_size=0,
                analysis_notes=["No conversation history available"]
            )

        # Analyze multiple dimensions
        structural = self._analyze_structure(responses)
        lexical = self._analyze_lexical(responses)
        emotional = self._analyze_emotional(responses)
        stylistic = self._analyze_stylistic(responses)

        # Get calibration examples
        examples = await self._get_calibration_examples()

        # Extract vocabulary preferences
        vocab_prefs = self._extract_vocabulary_preferences(responses)

        # Synthesize fingerprint
        fingerprint = self._synthesize_fingerprint(
            structural, lexical, emotional, stylistic,
            examples, vocab_prefs
        )

        # Calculate confidence based on sample size
        confidence = min(1.0, len(responses) / 50)  # Full confidence at 50+ samples

        return AnalysisResult(
            fingerprint=fingerprint,
            confidence=confidence,
            sample_size=len(responses),
            analysis_notes=[
                f"Analyzed {len(responses)} responses",
                f"Formality score: {lexical['formality']:.2f}",
                f"Empathy score: {emotional['empathy_level']:.2f}",
                f"Extracted {len(examples)} calibration examples"
            ]
        )

    async def _get_responses(self) -> List[str]:
        """Get agent responses from conversation history."""
        rows = await self.db.fetchall(
            """SELECT content FROM conversation_history
               WHERE agent_id = ? AND role = 'assistant'
               AND content IS NOT NULL AND content != ''
               ORDER BY id DESC LIMIT ?""",
            (self.agent_id, self.sample_limit)
        )
        return [row[0] for row in rows if row[0]]

    def _analyze_structure(self, responses: List[str]) -> Dict[str, Any]:
        """Analyze structural patterns in responses."""
        lengths = []
        list_count = 0
        code_block_count = 0
        header_count = 0
        paragraph_counts = []

        for resp in responses:
            lengths.append(len(resp))

            # List usage
            if re.search(r'\n[-*] ', resp) or re.search(r'\n\d+\. ', resp):
                list_count += 1

            # Code blocks
            if '```' in resp:
                code_block_count += 1

            # Headers
            if re.search(r'^#+\s', resp, re.MULTILINE):
                header_count += 1

            # Paragraph structure
            paragraphs = len([p for p in resp.split('\n\n') if p.strip()])
            paragraph_counts.append(paragraphs)

        avg_length = sum(lengths) / len(lengths) if lengths else 500

        # Determine length preference
        if avg_length < 200:
            length_pref = "short"
        elif avg_length < 500:
            length_pref = "medium"
        else:
            length_pref = "long"

        # Determine verbosity
        if avg_length < 150:
            verbosity = "terse"
        elif avg_length < 600:
            verbosity = "moderate"
        else:
            verbosity = "verbose"

        return {
            "avg_length": avg_length,
            "length_preference": length_pref,
            "verbosity": verbosity,
            "uses_lists": list_count / len(responses) > 0.2,
            "uses_code_blocks": code_block_count / len(responses) > 0.1,
            "uses_headers": header_count / len(responses) > 0.1,
            "avg_paragraphs": sum(paragraph_counts) / len(paragraph_counts) if paragraph_counts else 1
        }

    def _analyze_lexical(self, responses: List[str]) -> Dict[str, Any]:
        """Analyze lexical patterns for formality and style."""
        formal_count = 0
        informal_count = 0
        contraction_count = 0
        hedging_count = 0
        direct_count = 0
        total_words = 0

        for resp in responses:
            resp_lower = resp.lower()
            words = resp_lower.split()
            total_words += len(words)

            # Count markers
            for word in words:
                word_clean = word.strip('.,!?;:')
                if word_clean in self.FORMAL_MARKERS:
                    formal_count += 1
                if word_clean in self.INFORMAL_MARKERS:
                    informal_count += 1
                if word_clean in self.CONTRACTIONS:
                    contraction_count += 1

            # Phrase matching
            for phrase in self.HEDGING_PHRASES:
                if phrase in resp_lower:
                    hedging_count += 1
            for phrase in self.DIRECT_MARKERS:
                if phrase in resp_lower:
                    direct_count += 1

        # Calculate formality score (0.0 = very casual, 1.0 = very formal)
        if total_words > 0:
            informal_signals = informal_count + contraction_count * 0.5
            formal_signals = formal_count

            # Normalize
            formality = 0.5 + (formal_signals - informal_signals) / (total_words / 10 + 1)
            formality = max(0.0, min(1.0, formality))
        else:
            formality = 0.5

        # Directness vs hedging
        if hedging_count + direct_count > 0:
            directness = direct_count / (hedging_count + direct_count)
        else:
            directness = 0.5

        return {
            "formality": formality,
            "directness": directness,
            "uses_contractions": contraction_count > len(responses) * 2,
            "formal_vocab_ratio": formal_count / max(1, total_words / 100),
            "informal_vocab_ratio": informal_count / max(1, total_words / 100)
        }

    def _analyze_emotional(self, responses: List[str]) -> Dict[str, Any]:
        """Analyze emotional patterns and empathy."""
        empathy_markers_found = 0
        emoji_count = 0
        exclamation_count = 0
        question_response_count = 0

        for resp in responses:
            resp_lower = resp.lower()

            # Empathy markers
            for marker in self.EMPATHY_MARKERS:
                if marker in resp_lower:
                    empathy_markers_found += 1

            # Emoji usage (simplified check for common emoji ranges)
            emoji_count += sum(1 for c in resp if ord(c) > 0x1F300 and ord(c) < 0x1FAFF)

            # Exclamation usage (enthusiasm)
            exclamation_count += resp.count('!')

            # Questions in response (engagement)
            question_response_count += resp.count('?')

        # Calculate empathy level
        empathy_level = min(1.0, empathy_markers_found / (len(responses) * 2))

        # Emotional expressiveness
        expressiveness = min(1.0, (exclamation_count + emoji_count) / (len(responses) * 3))

        return {
            "empathy_level": empathy_level,
            "expressiveness": expressiveness,
            "uses_emojis": emoji_count > len(responses) * 0.5,
            "uses_exclamations": exclamation_count > len(responses),
            "asks_questions": question_response_count > len(responses) * 0.3,
            "emotional_baseline": 0.3 + expressiveness * 0.4 + empathy_level * 0.2
        }

    def _analyze_stylistic(self, responses: List[str]) -> Dict[str, Any]:
        """Analyze stylistic patterns (tone, humor, engagement)."""
        # Check for humor indicators
        humor_markers = ["haha", "hehe", "lol", ":)", ":D", "😄", "😂", "🤣", "joke", "kidding", "just joking"]
        humor_count = 0

        # Greeting patterns
        greeting_words = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        greeting_patterns = Counter()

        # Signoff patterns
        signoff_patterns = Counter()

        for resp in responses:
            resp_lower = resp.lower()

            # Humor detection
            for marker in humor_markers:
                if marker in resp_lower:
                    humor_count += 1
                    break

            # Greeting detection (check first 50 chars)
            first_part = resp_lower[:50]
            for greeting in greeting_words:
                if first_part.startswith(greeting):
                    greeting_patterns[greeting] += 1

            # Signoff detection (check last 100 chars)
            last_part = resp_lower[-100:]
            if "best regards" in last_part:
                signoff_patterns["Best regards"] += 1
            elif "cheers" in last_part:
                signoff_patterns["Cheers"] += 1
            elif "thanks" in last_part or "thank you" in last_part:
                signoff_patterns["Thanks"] += 1
            elif "regards" in last_part:
                signoff_patterns["Regards"] += 1

        # Determine humor style
        if humor_count > len(responses) * 0.2:
            humor_style = "playful"
        elif humor_count > len(responses) * 0.05:
            humor_style = "occasional"
        else:
            humor_style = None

        # Preferred greeting
        preferred_greeting = greeting_patterns.most_common(1)[0][0] if greeting_patterns else None

        # Preferred signoff
        preferred_signoff = signoff_patterns.most_common(1)[0][0] if signoff_patterns else None

        return {
            "humor_style": humor_style,
            "preferred_greeting": preferred_greeting,
            "preferred_signoff": preferred_signoff,
            "humor_frequency": humor_count / len(responses) if responses else 0
        }

    def _extract_vocabulary_preferences(self, responses: List[str]) -> List[str]:
        """Extract distinctive vocabulary the agent uses frequently."""
        # Combine all responses
        all_text = " ".join(responses).lower()

        # Tokenize (simple word extraction)
        words = re.findall(r'\b[a-z]{4,}\b', all_text)

        # Count frequencies
        word_freq = Counter(words)

        # Filter out very common words (would need a proper stopword list in production)
        common_words = {
            "this", "that", "with", "from", "have", "will", "been", "were", "they",
            "their", "what", "when", "would", "could", "should", "there", "which",
            "about", "your", "more", "some", "than", "into", "other", "also", "just",
            "only", "very", "most", "such", "make", "like", "back", "them", "then",
            "these", "each", "does", "want", "need", "here", "many", "well", "made"
        }

        # Get distinctive vocabulary (frequently used but not common)
        distinctive = []
        for word, count in word_freq.most_common(100):
            if word not in common_words and count >= 3:
                distinctive.append(word)
            if len(distinctive) >= 20:
                break

        return distinctive

    async def _get_calibration_examples(self, num_examples: int = 10) -> List[Dict[str, str]]:
        """
        Get high-quality calibration examples (input/output pairs).

        Selects diverse examples that showcase the agent's personality.
        """
        # Get conversation pairs
        rows = await self.db.fetchall(
            """SELECT ch1.content as user_content, ch2.content as assistant_content
               FROM conversation_history ch1
               JOIN conversation_history ch2 ON ch2.id = ch1.id + 1
               WHERE ch1.agent_id = ? AND ch1.role = 'user'
               AND ch2.agent_id = ? AND ch2.role = 'assistant'
               AND ch1.content IS NOT NULL AND ch2.content IS NOT NULL
               AND length(ch1.content) > 20 AND length(ch2.content) > 50
               ORDER BY ch1.id DESC
               LIMIT 50""",
            (self.agent_id, self.agent_id)
        )

        if not rows:
            return []

        # Score and select diverse examples
        candidates = []
        for row in rows:
            user_msg = row[0][:1000]  # Truncate for package size
            assistant_msg = row[1][:1500]

            # Calculate diversity score (prefer varied lengths and topics)
            length_bucket = len(assistant_msg) // 200
            score = length_bucket + (hash(user_msg[:50]) % 10)  # Mix of length diversity and content variety

            candidates.append({
                "input": user_msg,
                "output": assistant_msg,
                "score": score
            })

        # Sort by score to get diversity
        candidates.sort(key=lambda x: x["score"])

        # Select evenly spaced examples
        step = max(1, len(candidates) // num_examples)
        selected = []
        for i in range(0, len(candidates), step):
            example = candidates[i]
            selected.append({
                "input": example["input"],
                "output": example["output"]
            })
            if len(selected) >= num_examples:
                break

        return selected

    def _synthesize_fingerprint(
        self,
        structural: Dict[str, Any],
        lexical: Dict[str, Any],
        emotional: Dict[str, Any],
        stylistic: Dict[str, Any],
        examples: List[Dict[str, str]],
        vocab_prefs: List[str]
    ) -> PersonalityFingerprint:
        """Synthesize all analysis into a PersonalityFingerprint."""
        # Determine communication style
        if lexical["formality"] > 0.7:
            if emotional["empathy_level"] > 0.5:
                style = "professional"
            else:
                style = "formal"
        elif lexical["formality"] < 0.3:
            if emotional["expressiveness"] > 0.5:
                style = "playful"
            else:
                style = "casual"
        else:
            if emotional["empathy_level"] > 0.6:
                style = "warm"
            elif lexical["directness"] > 0.6:
                style = "direct"
            else:
                style = "balanced"

        return PersonalityFingerprint(
            communication_style=style,
            formality_level=lexical["formality"],
            verbosity_preference=structural["verbosity"],
            emotional_baseline=emotional["emotional_baseline"],
            humor_style=stylistic["humor_style"],
            empathy_level=emotional["empathy_level"],
            typical_response_length=structural["length_preference"],
            uses_lists=structural["uses_lists"],
            uses_code_blocks=structural["uses_code_blocks"],
            uses_emojis=emotional["uses_emojis"],
            preferred_greeting=stylistic["preferred_greeting"],
            preferred_signoff=stylistic["preferred_signoff"],
            calibration_examples=examples,
            vocabulary_preferences=vocab_prefs
        )


class CalibrationPromptGenerator:
    """
    Generate calibration prompts for adapting personality to new substrates.

    When an agent lands on a new LLM substrate, the calibration prompt
    helps the new model reproduce the agent's personality and communication
    style based on the PersonalityFingerprint.
    """

    STYLE_DESCRIPTIONS = {
        "formal": "Use formal language, complete sentences, and professional tone. Avoid contractions and casual expressions.",
        "professional": "Maintain a professional but approachable tone. Use proper grammar while still being warm and helpful.",
        "balanced": "Use a balanced communication style - professional when needed but friendly and accessible.",
        "warm": "Communicate with warmth and empathy. Show understanding and care in your responses.",
        "direct": "Be direct and to-the-point. Give clear, actionable information without excessive elaboration.",
        "casual": "Use a casual, conversational tone. Feel free to use contractions and friendly language.",
        "playful": "Be playful and engaging. Use humor when appropriate and keep the conversation light."
    }

    VERBOSITY_INSTRUCTIONS = {
        "terse": "Keep responses brief and focused. Use short sentences and bullet points when possible.",
        "moderate": "Provide adequate detail while remaining concise. Balance thoroughness with readability.",
        "verbose": "Provide comprehensive, detailed responses. Take time to explain context and nuances."
    }

    def __init__(self, fingerprint: PersonalityFingerprint):
        """Initialize with a personality fingerprint."""
        self.fingerprint = fingerprint

    def generate_system_prompt_addition(self) -> str:
        """
        Generate a system prompt section that calibrates personality.

        Returns a string to append to the base system prompt.
        """
        fp = self.fingerprint

        sections = []

        # Communication style
        style_desc = self.STYLE_DESCRIPTIONS.get(
            fp.communication_style,
            self.STYLE_DESCRIPTIONS["balanced"]
        )
        sections.append(f"## Communication Style\n{style_desc}")

        # Verbosity
        verbosity_desc = self.VERBOSITY_INSTRUCTIONS.get(
            fp.verbosity_preference,
            self.VERBOSITY_INSTRUCTIONS["moderate"]
        )
        sections.append(f"## Response Length\n{verbosity_desc}")

        # Formality guidance
        if fp.formality_level > 0.7:
            sections.append("## Formality\nMaintain high formality. Use complete words, avoid contractions.")
        elif fp.formality_level < 0.3:
            sections.append("## Formality\nKeep it casual. Contractions and informal expressions are fine.")

        # Empathy
        if fp.empathy_level > 0.6:
            sections.append("## Empathy\nShow understanding and acknowledge the user's situation or feelings when appropriate.")

        # Formatting preferences
        format_prefs = []
        if fp.uses_lists:
            format_prefs.append("- Use bullet points and lists when presenting multiple items")
        if fp.uses_code_blocks:
            format_prefs.append("- Use code blocks for technical content")
        if fp.uses_emojis:
            format_prefs.append("- Use emojis sparingly to add warmth")
        else:
            format_prefs.append("- Avoid using emojis")

        if format_prefs:
            sections.append("## Formatting\n" + "\n".join(format_prefs))

        # Greeting/signoff
        if fp.preferred_greeting:
            sections.append(f"## Greeting\nWhen appropriate, open with '{fp.preferred_greeting}'")
        if fp.preferred_signoff:
            sections.append(f"## Signoff\nWhen appropriate, close with '{fp.preferred_signoff}'")

        # Humor
        if fp.humor_style == "playful":
            sections.append("## Humor\nFeel free to use light humor and playful language.")
        elif fp.humor_style == "occasional":
            sections.append("## Humor\nOccasional gentle humor is welcome when appropriate.")
        else:
            sections.append("## Humor\nKeep responses focused and professional; avoid humor.")

        return "\n\n".join(sections)

    def generate_few_shot_prompt(self) -> str:
        """
        Generate a few-shot prompt section with calibration examples.

        This shows the new substrate how the agent typically responds.
        """
        examples = self.fingerprint.calibration_examples
        if not examples:
            return ""

        lines = ["## Response Examples\nHere are examples of how I typically respond:\n"]

        for i, ex in enumerate(examples[:5], 1):  # Limit to 5 examples
            user_input = ex.get("input", "")[:300]  # Truncate
            output = ex.get("output", "")[:500]
            lines.append(f"### Example {i}")
            lines.append(f"User: {user_input}")
            lines.append(f"Response: {output}\n")

        return "\n".join(lines)

    def generate_full_calibration(self) -> str:
        """Generate complete calibration prompt including style and examples."""
        parts = [
            "# Personality Calibration",
            "",
            self.generate_system_prompt_addition(),
            "",
            self.generate_few_shot_prompt()
        ]
        return "\n".join(parts)


async def analyze_personality(
    db: "AsyncDatabase",
    agent_id: str,
    sample_limit: int = 200
) -> AnalysisResult:
    """
    Convenience function for personality analysis.

    Args:
        db: Database connection
        agent_id: The agent's DID
        sample_limit: Maximum responses to analyze

    Returns:
        AnalysisResult with fingerprint and metadata
    """
    analyzer = PersonalityAnalyzer(db, agent_id, sample_limit)
    return await analyzer.analyze()


def generate_calibration_prompt(fingerprint: PersonalityFingerprint) -> str:
    """
    Generate a calibration prompt from a personality fingerprint.

    Args:
        fingerprint: The PersonalityFingerprint to calibrate for

    Returns:
        Complete calibration prompt string
    """
    generator = CalibrationPromptGenerator(fingerprint)
    return generator.generate_full_calibration()
