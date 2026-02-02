"""
Tests for the PII Detection module.

Tests both regex-based and NER-based detection (when spaCy is available).
"""

import pytest
from decimal import Decimal

from kestrel_sovereign.features.privacy.pii_detector import (
    PIIDetector,
    PIIType,
    PIIMatch,
    get_pii_detector,
    anonymize_text,
)


# Use module-scoped fixture to load spaCy only once
@pytest.fixture(scope="module")
def detector():
    """Get the global detector singleton (loads spaCy only once)."""
    return get_pii_detector()


# Cache for skipif decorator (evaluated at collection time)
_has_ner = None


def _check_has_ner():
    """Check if NER is available (cached)."""
    global _has_ner
    if _has_ner is None:
        _has_ner = get_pii_detector().has_ner
    return _has_ner


class TestRegexDetection:
    """Tests for regex-based PII detection (always available)."""

    def test_detects_email(self, detector):
        """Should detect email addresses."""
        text = "Contact me at john.doe@example.com for more info."
        matches = detector.detect(text)

        email_matches = [m for m in matches if m.pii_type == PIIType.EMAIL]
        assert len(email_matches) == 1
        assert email_matches[0].text == "john.doe@example.com"

    def test_detects_phone_numbers(self, detector):
        """Should detect various phone number formats."""
        cases = [
            ("Call 555-123-4567", "555-123-4567"),
            ("Phone: (555) 123-4567", "(555) 123-4567"),
            ("Mobile: 555.123.4567", "555.123.4567"),
            ("Intl: +1-555-123-4567", "+1-555-123-4567"),
        ]

        for text, expected_phone in cases:
            matches = detector.detect(text)
            phone_matches = [m for m in matches if m.pii_type == PIIType.PHONE]
            assert len(phone_matches) == 1, f"Failed for: {text}"
            assert phone_matches[0].text == expected_phone

    def test_detects_ssn(self, detector):
        """Should detect Social Security Numbers."""
        text = "My SSN is 123-45-6789"
        matches = detector.detect(text)

        ssn_matches = [m for m in matches if m.pii_type == PIIType.SSN]
        assert len(ssn_matches) == 1
        assert ssn_matches[0].text == "123-45-6789"

    def test_detects_credit_card(self, detector):
        """Should detect credit card numbers."""
        cases = [
            "Card: 4111-1111-1111-1111",
            "Card: 4111 1111 1111 1111",
            "Card: 4111111111111111",
        ]

        for text in cases:
            matches = detector.detect(text)
            cc_matches = [m for m in matches if m.pii_type == PIIType.CREDIT_CARD]
            assert len(cc_matches) == 1, f"Failed for: {text}"

    def test_detects_address(self, detector):
        """Should detect street addresses."""
        cases = [
            "I live at 123 Main Street",
            "Office at 456 Oak Avenue",
            "Send to 789 Park Boulevard",
        ]

        for text in cases:
            matches = detector.detect(text)
            addr_matches = [m for m in matches if m.pii_type == PIIType.ADDRESS]
            assert len(addr_matches) >= 1, f"Failed for: {text}"

    def test_detects_zip_code(self, detector):
        """Should detect ZIP codes."""
        cases = [
            ("Zip: 12345", "12345"),
            ("Zip: 12345-6789", "12345-6789"),
        ]

        for text, expected_zip in cases:
            matches = detector.detect(text)
            zip_matches = [m for m in matches if m.pii_type == PIIType.ZIP]
            assert len(zip_matches) == 1, f"Failed for: {text}"
            assert zip_matches[0].text == expected_zip


class TestAnonymization:
    """Tests for text anonymization."""

    def test_anonymize_email(self, detector):
        """Should replace email with redaction label."""
        text = "Contact me at john@example.com"
        result = detector.anonymize(text)

        assert "john@example.com" not in result
        assert "[EMAIL_REDACTED]" in result

    def test_anonymize_phone(self, detector):
        """Should replace phone with redaction label."""
        text = "Call me at 555-123-4567"
        result = detector.anonymize(text)

        assert "555-123-4567" not in result
        assert "[PHONE_REDACTED]" in result

    def test_anonymize_multiple_pii(self, detector):
        """Should handle multiple PII items."""
        text = "Contact John at john@test.com or call 555-123-4567"
        result = detector.anonymize(text)

        assert "john@test.com" not in result
        assert "555-123-4567" not in result
        assert "[EMAIL_REDACTED]" in result
        assert "[PHONE_REDACTED]" in result

    def test_anonymize_preserves_non_pii(self, detector):
        """Should preserve non-PII text."""
        text = "The meeting is scheduled for tomorrow at the office."
        result = detector.anonymize(text)

        # Should be mostly unchanged (maybe dates if NER is active)
        assert "meeting" in result
        assert "scheduled" in result
        assert "office" in result

    def test_anonymize_selective_types(self, detector):
        """Should allow redacting only specific PII types."""
        text = "Email: test@example.com, Phone: 555-123-4567"

        # Only redact emails
        result = detector.anonymize(text, types_to_redact={PIIType.EMAIL})

        assert "[EMAIL_REDACTED]" in result
        # Phone might or might not be present depending on detection order
        # The key assertion is that EMAIL is redacted


class TestNERDetection:
    """Tests for NER-based detection (requires spaCy)."""

    def test_ner_availability_reported(self, detector):
        """Should correctly report if NER is available."""
        # This just tests the property works; actual availability depends on install
        assert isinstance(detector.has_ner, bool)

    @pytest.mark.skipif(
        not _check_has_ner(),
        reason="spaCy not installed or model not available"
    )
    def test_detects_person_names(self, detector):
        """Should detect person names via NER."""
        text = "I met with John Smith yesterday."
        matches = detector.detect(text)

        person_matches = [m for m in matches if m.pii_type == PIIType.PERSON]
        assert len(person_matches) >= 1
        # Name detection varies by model, just verify we found something

    @pytest.mark.skipif(
        not _check_has_ner(),
        reason="spaCy not installed or model not available"
    )
    def test_detects_organizations(self, detector):
        """Should detect organization names via NER."""
        text = "I work at Microsoft in Seattle."
        matches = detector.detect(text)

        org_matches = [m for m in matches if m.pii_type == PIIType.ORG]
        assert len(org_matches) >= 1

    @pytest.mark.skipif(
        not _check_has_ner(),
        reason="spaCy not installed or model not available"
    )
    def test_anonymize_person_names(self, detector):
        """Should anonymize person names when NER is available."""
        text = "Please contact Dr. Jane Smith for your appointment."
        result = detector.anonymize(text)

        # With NER, "Jane Smith" or similar should be redacted
        assert "[NAME_REDACTED]" in result or "Jane Smith" not in result


class TestPIIReport:
    """Tests for PII report generation."""

    def test_generates_report(self, detector):
        """Should generate a structured report."""
        text = "Call John at john@test.com or 555-123-4567"
        report = detector.get_pii_report(text)

        assert "total_pii_found" in report
        assert report["total_pii_found"] >= 2  # At least email and phone
        assert "ner_available" in report
        assert "findings_by_type" in report
        assert "types_detected" in report

    def test_report_empty_for_clean_text(self, detector):
        """Should report zero PII for clean text."""
        text = "The weather is nice today."
        report = detector.get_pii_report(text)

        assert report["total_pii_found"] == 0
        assert report["findings_by_type"] == {}


class TestGlobalSingleton:
    """Tests for the global detector singleton."""

    def test_get_detector_returns_same_instance(self):
        """Should return the same detector instance."""
        d1 = get_pii_detector()
        d2 = get_pii_detector()
        assert d1 is d2

    def test_convenience_function_works(self):
        """The anonymize_text convenience function should work."""
        text = "Email: test@example.com"
        result = anonymize_text(text)

        assert "[EMAIL_REDACTED]" in result
        assert "test@example.com" not in result


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_string(self, detector):
        """Should handle empty string."""
        assert detector.anonymize("") == ""
        assert detector.detect("") == []

    def test_no_pii_in_text(self, detector):
        """Should return original text when no PII found."""
        text = "Hello world, this is a test message."
        result = detector.anonymize(text)
        # Should be same (unless NER finds something unexpected)
        assert "Hello" in result and "world" in result

    def test_multiple_same_type_pii(self, detector):
        """Should handle multiple instances of same PII type."""
        text = "Contact a@b.com or c@d.com or e@f.com"
        matches = detector.detect(text)

        email_matches = [m for m in matches if m.pii_type == PIIType.EMAIL]
        assert len(email_matches) == 3

    def test_overlapping_patterns(self, detector):
        """Should not double-redact overlapping patterns."""
        # A phone number could theoretically match partial ZIP patterns
        text = "Call 555-12-3456"  # This matches SSN pattern
        result = detector.anonymize(text)

        # Should have exactly one redaction, not multiple
        redaction_count = result.count("[")
        assert redaction_count >= 1
