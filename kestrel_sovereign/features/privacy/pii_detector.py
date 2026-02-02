"""
PII Detection Module for Kestrel Privacy System.

This module provides Named Entity Recognition (NER) based PII detection using spaCy,
with a graceful fallback to regex-based detection when spaCy is not available.

The NER approach catches:
- PERSON names (John Smith, Dr. Jane Doe)
- ORG names (Acme Corp, Bank of America)
- GPE/LOC geographic locations (New York, 123 Main Street)
- DATE expressions that might be birthdates
- Plus all the regex patterns (emails, phones, SSNs, credit cards)

Usage:
    detector = PIIDetector()
    anonymized = detector.anonymize("Call John Smith at 555-123-4567")
    # Returns: "Call [NAME_REDACTED] at [PHONE_REDACTED]"
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PIIType(Enum):
    """Types of PII that can be detected and redacted."""
    PERSON = "PERSON"       # Names detected via NER
    ORG = "ORG"             # Organization names
    GPE = "GPE"             # Geopolitical entities (cities, countries)
    LOC = "LOC"             # Locations (addresses, landmarks)
    DATE = "DATE"           # Dates (potential birthdates)
    EMAIL = "EMAIL"         # Email addresses
    PHONE = "PHONE"         # Phone numbers
    SSN = "SSN"             # Social Security Numbers
    CREDIT_CARD = "CC"      # Credit card numbers
    ADDRESS = "ADDRESS"     # Street addresses
    ZIP = "ZIP"             # ZIP codes


@dataclass
class PIIMatch:
    """Represents a detected PII entity."""
    pii_type: PIIType
    text: str
    start: int
    end: int
    confidence: float = 1.0  # NER confidence, regex always 1.0


class PIIDetector:
    """
    Detects and redacts Personally Identifiable Information (PII).
    
    Uses spaCy NER models when available, with regex fallback.
    The hybrid approach ensures:
    1. Names are detected by NER (regex can't reliably do this)
    2. Structured data (emails, phones, SSNs) caught by regex
    """
    
    # Regex patterns for structured PII
    # Order matters for overlap detection - more specific patterns first
    PATTERNS = {
        # Credit cards first (16 digits) - must come before phone (10 digits)
        PIIType.CREDIT_CARD: re.compile(
            r'\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{16}\b'
        ),
        PIIType.EMAIL: re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        ),
        PIIType.SSN: re.compile(
            r'\b\d{3}-\d{2}-\d{4}\b'
        ),
        # Phone pattern requires separators or area code format to avoid matching 
        # parts of longer number sequences like credit cards
        PIIType.PHONE: re.compile(
            r'(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}|\(\d{3}\)\s?\d{3}[-.\s]?\d{4}'
        ),
        PIIType.ADDRESS: re.compile(
            r'\b\d{1,5}\s+[\w\s]{2,30}(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b',
            re.IGNORECASE
        ),
        PIIType.ZIP: re.compile(
            r'\b\d{5}(-\d{4})?\b'
        ),
    }
    
    # spaCy entity labels to PII types
    SPACY_LABEL_MAP = {
        "PERSON": PIIType.PERSON,
        "ORG": PIIType.ORG,
        "GPE": PIIType.GPE,
        "LOC": PIIType.LOC,
        "FAC": PIIType.LOC,  # Facilities like buildings
        "DATE": PIIType.DATE,
    }
    
    # Redaction placeholders
    REDACTION_LABELS = {
        PIIType.PERSON: "[NAME_REDACTED]",
        PIIType.ORG: "[ORG_REDACTED]",
        PIIType.GPE: "[LOCATION_REDACTED]",
        PIIType.LOC: "[LOCATION_REDACTED]",
        PIIType.DATE: "[DATE_REDACTED]",
        PIIType.EMAIL: "[EMAIL_REDACTED]",
        PIIType.PHONE: "[PHONE_REDACTED]",
        PIIType.SSN: "[SSN_REDACTED]",
        PIIType.CREDIT_CARD: "[CARD_REDACTED]",
        PIIType.ADDRESS: "[ADDRESS_REDACTED]",
        PIIType.ZIP: "[ZIP_REDACTED]",
    }
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize the PII detector.
        
        Args:
            spacy_model: Name of the spaCy model to use (default: en_core_web_sm)
                         Use en_core_web_trf for higher accuracy if available.
        """
        self.nlp = None
        self._load_spacy(spacy_model)
        
    def _load_spacy(self, model_name: str) -> None:
        """Attempt to load spaCy model, gracefully handle failure."""
        try:
            import spacy
            self.nlp = spacy.load(model_name)
            logger.info(f"spaCy model '{model_name}' loaded for NER-based PII detection")
        except ImportError:
            logger.warning(
                "spaCy not installed. Using regex-only PII detection. "
                "Install with: pip install 'kestrel_agent[pii]' && python -m spacy download en_core_web_sm"
            )
        except OSError:
            logger.warning(
                f"spaCy model '{model_name}' not found. Using regex-only PII detection. "
                f"Install with: python -m spacy download {model_name}"
            )
    
    @property
    def has_ner(self) -> bool:
        """Returns True if NER is available."""
        return self.nlp is not None
    
    def detect(self, text: str) -> List[PIIMatch]:
        """
        Detect all PII entities in text.
        
        Args:
            text: Input text to scan for PII
            
        Returns:
            List of PIIMatch objects sorted by position
        """
        matches: List[PIIMatch] = []
        seen_spans: Set[Tuple[int, int]] = set()  # Avoid duplicate matches
        
        # 1. Run regex patterns FIRST - these are high-confidence structured patterns
        # (emails, phones, SSNs, credit cards are unambiguous)
        for pii_type, pattern in self.PATTERNS.items():
            for match in pattern.finditer(text):
                span = (match.start(), match.end())
                if not self._overlaps_any(span, seen_spans):
                    seen_spans.add(span)
                    matches.append(PIIMatch(
                        pii_type=pii_type,
                        text=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=1.0
                    ))
        
        # 2. Run NER for names, orgs, locations (things regex can't catch)
        if self.nlp is not None:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in self.SPACY_LABEL_MAP:
                    pii_type = self.SPACY_LABEL_MAP[ent.label_]
                    span = (ent.start_char, ent.end_char)
                    # Skip if overlapping with regex match (regex is more reliable for structured data)
                    if not self._overlaps_any(span, seen_spans):
                        # Additional filter: skip common non-PII date words and numeric-only dates
                        if ent.label_ == "DATE":
                            # Skip numeric-only "dates" like "1111"
                            if ent.text.replace(" ", "").isdigit():
                                continue
                            # Skip common relative date words that aren't PII
                            common_dates = {"today", "yesterday", "tomorrow", "now", "later", 
                                           "monday", "tuesday", "wednesday", "thursday", 
                                           "friday", "saturday", "sunday"}
                            if ent.text.lower() in common_dates:
                                continue
                        seen_spans.add(span)
                        matches.append(PIIMatch(
                            pii_type=pii_type,
                            text=ent.text,
                            start=ent.start_char,
                            end=ent.end_char,
                            confidence=0.9  # NER confidence varies; simplified
                        ))
        
        # Sort by position for proper replacement
        matches.sort(key=lambda m: m.start)
        return matches
    
    def _overlaps_any(self, span: Tuple[int, int], seen: Set[Tuple[int, int]]) -> bool:
        """Check if a span overlaps with any seen span."""
        start, end = span
        for s_start, s_end in seen:
            # Check for any overlap
            if start < s_end and end > s_start:
                return True
        return False
    
    def anonymize(
        self, 
        text: str, 
        types_to_redact: Optional[Set[PIIType]] = None
    ) -> str:
        """
        Anonymize text by replacing PII with redaction labels.
        
        Args:
            text: Input text to anonymize
            types_to_redact: Optional set of PIIType to redact. 
                             If None, redacts all detected PII.
        
        Returns:
            Anonymized text with PII replaced by labels
        """
        matches = self.detect(text)
        
        if not matches:
            return text
        
        # Filter by types if specified
        if types_to_redact is not None:
            matches = [m for m in matches if m.pii_type in types_to_redact]
        
        # Replace from end to start to preserve indices
        result = text
        for match in reversed(matches):
            label = self.REDACTION_LABELS.get(match.pii_type, "[REDACTED]")
            result = result[:match.start] + label + result[match.end:]
        
        return result
    
    def get_pii_report(self, text: str) -> Dict:
        """
        Generate a report of PII found in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with detection statistics and findings
        """
        matches = self.detect(text)
        
        by_type = {}
        for match in matches:
            type_name = match.pii_type.value
            if type_name not in by_type:
                by_type[type_name] = []
            by_type[type_name].append({
                "text": match.text,
                "position": (match.start, match.end),
                "confidence": match.confidence
            })
        
        return {
            "total_pii_found": len(matches),
            "ner_available": self.has_ner,
            "findings_by_type": by_type,
            "types_detected": list(by_type.keys())
        }


# Global singleton for reuse
_detector: Optional[PIIDetector] = None


def get_pii_detector() -> PIIDetector:
    """Get or create the global PII detector instance."""
    global _detector
    if _detector is None:
        _detector = PIIDetector()
    return _detector


def anonymize_text(text: str) -> str:
    """
    Convenience function to anonymize text.
    
    Args:
        text: Input text to anonymize
        
    Returns:
        Anonymized text with PII replaced
    """
    return get_pii_detector().anonymize(text)
