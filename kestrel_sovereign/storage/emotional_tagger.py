"""
Emotional analysis for conversation messages.

Provides sentiment analysis and importance detection to enrich conversation
metadata with emotional context. This enables human-like memory retrieval
weighted by emotional significance.

Pattern: Follows existing storage module patterns with async methods.
Dependencies: Uses regex patterns by default, optional spaCy for enhanced analysis.
"""
import re
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

from .memory_models import MemoryMetadata, EmotionalCategory

logger = logging.getLogger(__name__)


class EmotionalTagger:
    """
    Analyzes messages for emotional content and importance.

    Enriches conversation metadata with:
    - Emotional valence (-1.0 to +1.0) and intensity (0.0 to 1.0)
    - Emotional categories (joy, sadness, anger, etc.)
    - Importance score (0.0 to 1.0) with reasons
    - Temporal context (time of day, day of week)

    Uses lightweight regex patterns by default. If spaCy is available
    (from optional 'local' dependencies), provides enhanced sentiment analysis.
    """

    # ─────────────────────────────────────────────────────────────────
    # Importance Signal Patterns
    # ─────────────────────────────────────────────────────────────────

    # Personal disclosures - things people don't share casually
    DISCLOSURE_PATTERNS = [
        r"i'?ve never told",
        r"between you and me",
        r"don'?t tell anyone",
        r"this is personal",
        r"i'?ve been meaning to say",
        r"i need to confess",
        r"honestly[,\s]",
        r"to be honest",
        r"the truth is",
        r"i'?m afraid to",
        r"i'?ve always wanted to",
        r"my secret",
        r"nobody knows",
    ]

    # Life events - major transitions that define a person's story
    LIFE_EVENT_PATTERNS = [
        r"(got|getting|just) (married|divorced|engaged|promoted|fired|laid off)",
        r"(passed away|died|lost my|funeral|grieving)",
        r"(born|pregnant|expecting|baby|miscarriage)",
        r"(moving to|moved to|leaving|relocated)",
        r"(started|quit|left|finished) (my job|school|college|university)",
        r"(diagnosed with|found out i have|test results)",
        r"(broke up|breaking up|getting back together)",
        r"(first time|first day|anniversary|birthday)",
        r"(graduation|retired|retiring)",
    ]

    # Explicit memory markers - user directly asks to remember
    EXPLICIT_MARKERS = [
        r"remember this",
        r"this is important",
        r"don'?t forget",
        r"i need you to remember",
        r"please remember",
        r"mark this",
        r"note this",
        r"keep in mind",
        r"for future reference",
    ]

    # ─────────────────────────────────────────────────────────────────
    # Emotion Detection Patterns
    # ─────────────────────────────────────────────────────────────────

    # Positive emotion keywords (valence > 0)
    POSITIVE_KEYWORDS = {
        EmotionalCategory.JOY: [
            "happy", "excited", "thrilled", "delighted", "wonderful",
            "amazing", "fantastic", "great", "awesome", "excellent",
            "love", "loving", "loved", "joy", "joyful", "elated",
            "ecstatic", "overjoyed", "cheerful", "glad", "pleased",
        ],
        EmotionalCategory.LOVE: [
            "love", "adore", "cherish", "devoted", "affection",
            "care about", "caring", "tender", "romantic", "intimate",
            "passionate", "infatuated", "smitten", "crush",
        ],
        EmotionalCategory.HOPE: [
            "hope", "hopeful", "optimistic", "looking forward",
            "excited about", "can't wait", "anticipating", "eager",
            "confident", "positive", "bright future",
        ],
        EmotionalCategory.NOSTALGIA: [
            "remember when", "back then", "used to", "those days",
            "miss", "missed", "missing", "memories", "childhood",
            "old times", "reminds me", "throwback",
        ],
    }

    # Negative emotion keywords (valence < 0)
    NEGATIVE_KEYWORDS = {
        EmotionalCategory.SADNESS: [
            "sad", "unhappy", "depressed", "down", "blue", "miserable",
            "heartbroken", "devastated", "crushed", "grief", "grieving",
            "mourning", "crying", "tears", "sob", "weep", "hurt",
        ],
        EmotionalCategory.ANGER: [
            "angry", "mad", "furious", "enraged", "pissed", "livid",
            "frustrated", "annoyed", "irritated", "hate", "resent",
            "outraged", "infuriated", "bitter", "hostile",
        ],
        EmotionalCategory.FEAR: [
            "scared", "afraid", "terrified", "frightened", "panic",
            "anxious", "worried", "nervous", "dread", "dreading",
            "fearful", "petrified", "horror", "alarmed",
        ],
        EmotionalCategory.ANXIETY: [
            "anxious", "stressed", "overwhelmed", "panicking",
            "can't stop thinking", "worried sick", "on edge",
            "restless", "uneasy", "apprehensive", "tense",
        ],
        EmotionalCategory.DISGUST: [
            "disgusted", "gross", "revolting", "repulsed", "sick of",
            "nauseating", "appalled", "horrified", "sickening",
        ],
    }

    # ─────────────────────────────────────────────────────────────────
    # Epistemic Status Patterns (claim certainty cues)
    # ─────────────────────────────────────────────────────────────────

    # High-certainty linguistic cues (0.8-1.0)
    HIGH_CERTAINTY_CUES = [
        r"\bi(?:'ve)? decided\b",
        r"\bwe(?:'ve)? decided\b",
        r"\bi(?:'m)? certain\b",
        r"\bi(?:'m)? sure\b",
        r"\bdefinitely\b",
        r"\babsolutely\b",
        r"\bwithout a doubt\b",
        r"\bno question\b",
        r"\bfor sure\b",
        r"\bi know (?:that |for )",
        r"\bcommitting to\b",
        r"\bsticking with\b",
    ]

    # Medium-certainty cues (0.5-0.7)
    MEDIUM_CERTAINTY_CUES = [
        r"\bi think\b",
        r"\bi believe\b",
        r"\bprobably\b",
        r"\blikely\b",
        r"\bit seems\b",
        r"\bapparently\b",
        r"\bi(?:'m)? fairly (?:sure|certain|confident)\b",
        r"\bfrom what i(?:'ve)? (?:heard|seen|read)\b",
        r"\bgoing with\b",
    ]

    # Low-certainty cues (0.1-0.4)
    LOW_CERTAINTY_CUES = [
        r"\bmaybe\b",
        r"\bperhaps\b",
        r"\bi(?:'m)? not sure\b",
        r"\bi(?:'m)? not certain\b",
        r"\bmight\b",
        r"\bcould be\b",
        r"\bpossibly\b",
        r"\bi guess\b",
        r"\bi wonder\b",
        r"\bi(?:'m)? unsure\b",
        r"\bsupposedly\b",
    ]

    # Non-claim patterns — questions, greetings, etc.
    NON_CLAIM_PATTERNS = [
        r"^\s*(?:hi|hello|hey|good (?:morning|afternoon|evening)|thanks|thank you)\s*[.!]?\s*$",
        r"^\s*(?:how are you|what'?s up|what do you think)\s*[?]?\s*$",
        r"^[^.!]*\?\s*$",  # Messages that are purely questions
    ]

    # Claim source inference patterns
    HEARSAY_CUES = [
        r"\b(?:someone|somebody) (?:told|said|mentioned)\b",
        r"\bi(?:'ve)? heard (?:that )?\b",
        r"\baccording to\b",
        r"\bthey say\b",
        r"\brumor has it\b",
        r"\bapparently\b",
    ]

    OBSERVED_CUES = [
        r"\bi (?:saw|noticed|observed|witnessed)\b",
        r"\bi(?:'ve)? seen\b",
        r"\bwatching\b",
        r"\blooking at\b",
    ]

    INFERRED_CUES = [
        r"\btherefore\b",
        r"\bso that means\b",
        r"\bwhich means\b",
        r"\bi(?:'m)? (?:guessing|inferring|deducing|concluding)\b",
        r"\bit follows that\b",
        r"\bbased on\b",
        r"\bgiven that\b",
        r"\bimplies\b",
        r"\bmust be\b",
    ]

    # Intensity amplifiers
    INTENSITY_AMPLIFIERS = [
        r"\b(very|really|extremely|incredibly|absolutely|totally|completely|utterly)\b",
        r"\b(so|such|too)\b",
        r"!!+",
        r"\b(never|always|forever)\b",
    ]

    INTENSITY_DAMPENERS = [
        r"\b(kind of|kinda|sort of|sorta|somewhat|slightly|a bit|a little)\b",
        r"\b(maybe|perhaps|possibly)\b",
    ]

    def __init__(self, use_spacy: bool = False):
        """
        Initialize the emotional tagger.

        Args:
            use_spacy: If True, attempt to use spaCy for enhanced analysis.
                      Falls back to keyword-based if spaCy unavailable.
        """
        self._spacy_nlp = None
        if use_spacy:
            self._init_spacy()

    def _init_spacy(self) -> None:
        """Attempt to initialize spaCy for enhanced analysis."""
        try:
            import spacy
            # Try to load a model that might have sentiment capabilities
            try:
                self._spacy_nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy initialized for emotional tagger")
            except OSError:
                logger.warning("spaCy model not found, using keyword-based analysis")
        except ImportError:
            logger.info("spaCy not installed, using keyword-based analysis")

    async def analyze(self, content: str, role: str = "user") -> MemoryMetadata:
        """
        Full analysis: emotional + importance + temporal + epistemic.

        Args:
            content: The message content to analyze
            role: Message role ('user' or 'assistant')

        Returns:
            MemoryMetadata with all enrichment fields populated
        """
        # Analyze sentiment
        valence, intensity = self._analyze_sentiment(content)

        # Detect emotional categories
        categories = self._detect_emotions(content)

        # Detect importance
        importance, reasons = self._detect_importance(content, role)

        # Get temporal context
        time_of_day = self._get_time_of_day()
        day_of_week = self._get_day_of_week()

        # Detect epistemic status (certainty, source, temporal validity)
        epistemic = self._detect_epistemic_status(content)

        return MemoryMetadata(
            emotional_valence=valence,
            emotional_intensity=intensity,
            emotional_categories=categories,
            importance=importance,
            importance_reasons=reasons,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            claim_certainty=epistemic.get("claim_certainty"),
            claim_source=epistemic.get("claim_source"),
            temporal_validity=epistemic.get("temporal_validity"),
        )

    def _analyze_sentiment(self, content: str) -> Tuple[float, float]:
        """
        Analyze sentiment of content.

        Returns:
            Tuple of (valence, intensity):
            - valence: -1.0 (very negative) to +1.0 (very positive)
            - intensity: 0.0 (neutral/mild) to 1.0 (very strong)
        """
        content_lower = content.lower()

        # Count positive and negative keyword matches
        positive_count = 0
        negative_count = 0
        total_matches = 0

        # Check positive keywords
        for category, keywords in self.POSITIVE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in content_lower:
                    positive_count += 1
                    total_matches += 1

        # Check negative keywords
        for category, keywords in self.NEGATIVE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in content_lower:
                    negative_count += 1
                    total_matches += 1

        # Calculate valence
        if total_matches == 0:
            valence = 0.0
        else:
            # Valence = (positive - negative) / total, scaled to [-1, 1]
            valence = (positive_count - negative_count) / total_matches

        # Calculate base intensity from match density
        words = content_lower.split()
        word_count = max(len(words), 1)
        base_intensity = min(1.0, total_matches / (word_count * 0.1))

        # Adjust intensity based on amplifiers/dampeners
        intensity_modifier = 1.0

        for pattern in self.INTENSITY_AMPLIFIERS:
            if re.search(pattern, content_lower):
                intensity_modifier += 0.2

        for pattern in self.INTENSITY_DAMPENERS:
            if re.search(pattern, content_lower):
                intensity_modifier -= 0.2

        # ALL CAPS adds intensity
        caps_ratio = sum(1 for c in content if c.isupper()) / max(len(content), 1)
        if caps_ratio > 0.5:
            intensity_modifier += 0.3

        # Exclamation marks add intensity
        exclaim_count = content.count("!")
        if exclaim_count >= 3:
            intensity_modifier += 0.2
        elif exclaim_count >= 1:
            intensity_modifier += 0.1

        intensity = min(1.0, max(0.0, base_intensity * intensity_modifier))

        return (valence, intensity)

    def _detect_emotions(self, content: str) -> List[str]:
        """
        Classify content into emotion categories.

        Returns:
            List of emotion category names detected
        """
        content_lower = content.lower()
        detected = set()

        # Check positive categories
        for category, keywords in self.POSITIVE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in content_lower:
                    detected.add(category.value)
                    break

        # Check negative categories
        for category, keywords in self.NEGATIVE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in content_lower:
                    detected.add(category.value)
                    break

        return list(detected)

    def _detect_importance(
        self,
        content: str,
        role: str = "user"
    ) -> Tuple[float, List[str]]:
        """
        Detect importance signals in content.

        Returns:
            Tuple of (importance_score, reasons):
            - importance_score: 0.0 to 1.0
            - reasons: List of why this is important
        """
        score = 0.5  # Base importance
        reasons: List[str] = []
        content_lower = content.lower()

        # Personal disclosure patterns (high importance)
        for pattern in self.DISCLOSURE_PATTERNS:
            if re.search(pattern, content_lower, re.I):
                score += 0.25
                if "personal_disclosure" not in reasons:
                    reasons.append("personal_disclosure")
                break

        # Life event patterns (highest importance)
        for pattern in self.LIFE_EVENT_PATTERNS:
            if re.search(pattern, content_lower, re.I):
                score += 0.35
                if "life_event" not in reasons:
                    reasons.append("life_event")
                break

        # Explicit memory markers (user directly asks to remember)
        for pattern in self.EXPLICIT_MARKERS:
            if re.search(pattern, content_lower, re.I):
                score += 0.3
                if "explicit_marker" not in reasons:
                    reasons.append("explicit_marker")
                break

        # Emotional intensity adds importance
        valence, intensity = self._analyze_sentiment(content)
        if intensity > 0.6:
            score += 0.15
            if "high_emotion" not in reasons:
                reasons.append("high_emotion")

        # Long, detailed messages tend to be more important
        word_count = len(content.split())
        if word_count > 100:
            score += 0.1
            if "detailed_content" not in reasons:
                reasons.append("detailed_content")

        # Questions from user are moderately important (they want something)
        if role == "user" and "?" in content:
            score += 0.05

        # First-person narratives ("I" statements) often contain personal info
        i_count = len(re.findall(r"\bi\b", content_lower))
        if i_count >= 3:
            score += 0.1
            if "self_narrative" not in reasons:
                reasons.append("self_narrative")

        # Cap at 1.0
        score = min(1.0, score)

        return (score, reasons)

    def _detect_epistemic_status(self, content: str) -> Dict[str, Any]:
        """Infer epistemic status from linguistic cues in the message.

        Returns a dict with optional keys:
        - claim_certainty (float 0.0-1.0)
        - claim_source ("direct" | "inferred" | "hearsay" | "observed")
        - temporal_validity ("durable" | "ephemeral" | "moment")

        Returns an empty dict for non-claim messages (questions, greetings)
        so that None values on MemoryMetadata signal "not a claim".
        """
        content_lower = content.lower().strip()

        # Check for non-claim patterns first — return empty for questions/greetings
        for pattern in self.NON_CLAIM_PATTERNS:
            if re.search(pattern, content_lower, re.IGNORECASE):
                return {}

        result: Dict[str, Any] = {}

        # --- Claim certainty ---
        high_hits = sum(
            1 for p in self.HIGH_CERTAINTY_CUES
            if re.search(p, content_lower)
        )
        medium_hits = sum(
            1 for p in self.MEDIUM_CERTAINTY_CUES
            if re.search(p, content_lower)
        )
        low_hits = sum(
            1 for p in self.LOW_CERTAINTY_CUES
            if re.search(p, content_lower)
        )

        total_hits = high_hits + medium_hits + low_hits
        if total_hits > 0:
            # Weighted average of certainty tiers
            certainty = (
                high_hits * 0.9 + medium_hits * 0.6 + low_hits * 0.25
            ) / total_hits
            result["claim_certainty"] = round(min(1.0, max(0.0, certainty)), 2)

        # --- Claim source ---
        hearsay_hits = sum(
            1 for p in self.HEARSAY_CUES
            if re.search(p, content_lower)
        )
        observed_hits = sum(
            1 for p in self.OBSERVED_CUES
            if re.search(p, content_lower)
        )
        inferred_hits = sum(
            1 for p in self.INFERRED_CUES
            if re.search(p, content_lower)
        )

        source_counts = {
            "hearsay": hearsay_hits,
            "observed": observed_hits,
            "inferred": inferred_hits,
        }
        best_source = max(source_counts, key=source_counts.get)
        if source_counts[best_source] > 0:
            result["claim_source"] = best_source
        elif "claim_certainty" in result:
            # Has certainty cues but no specific source indicator → direct
            result["claim_source"] = "direct"

        # --- Temporal validity ---
        # Moment indicators
        moment_patterns = [
            r"\bright now\b", r"\bat the moment\b", r"\bcurrently\b",
            r"\bjust now\b", r"\btoday\b",
        ]
        # Ephemeral indicators
        ephemeral_patterns = [
            r"\bfor now\b", r"\btemporarily\b", r"\bfor the time being\b",
            r"\buntil\b", r"\bthis week\b", r"\bthis month\b",
        ]
        # Durable indicators
        durable_patterns = [
            r"\balways\b", r"\bforever\b", r"\bpermanently\b",
            r"\bfrom now on\b", r"\bdecided\b", r"\bcommitting\b",
        ]

        moment_count = sum(1 for p in moment_patterns if re.search(p, content_lower))
        ephemeral_count = sum(1 for p in ephemeral_patterns if re.search(p, content_lower))
        durable_count = sum(1 for p in durable_patterns if re.search(p, content_lower))

        if durable_count > ephemeral_count and durable_count > moment_count:
            result["temporal_validity"] = "durable"
        elif moment_count > ephemeral_count:
            result["temporal_validity"] = "moment"
        elif ephemeral_count > 0:
            result["temporal_validity"] = "ephemeral"
        elif "claim_certainty" in result:
            # Default for claims without explicit temporal cues
            result["temporal_validity"] = "durable"

        return result

    def _get_time_of_day(self, dt: Optional[datetime] = None) -> str:
        """
        Classify current time into period.

        Returns:
            One of: 'morning', 'afternoon', 'evening', 'late_night'
        """
        dt = dt or datetime.now()
        hour = dt.hour

        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 22:
            return "evening"
        else:
            return "late_night"

    def _get_day_of_week(self, dt: Optional[datetime] = None) -> str:
        """
        Get lowercase day name.

        Returns:
            Day name like 'monday', 'tuesday', etc.
        """
        dt = dt or datetime.now()
        return dt.strftime("%A").lower()

    async def analyze_batch(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[MemoryMetadata]:
        """
        Analyze multiple messages efficiently.

        Args:
            messages: List of dicts with 'content' and 'role' keys

        Returns:
            List of MemoryMetadata, one per message
        """
        results = []
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")
            meta = await self.analyze(content, role)
            results.append(meta)
        return results


# Convenience function for one-off analysis
async def analyze_message(
    content: str,
    role: str = "user"
) -> MemoryMetadata:
    """
    Analyze a single message for emotional content.

    Convenience function that creates a tagger and analyzes one message.
    For multiple messages, create an EmotionalTagger instance directly.

    Args:
        content: Message text to analyze
        role: 'user' or 'assistant'

    Returns:
        MemoryMetadata with emotional enrichment
    """
    tagger = EmotionalTagger()
    return await tagger.analyze(content, role)
