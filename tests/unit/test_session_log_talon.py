"""Tests for Talon PR detection in Session Log collector."""

import pytest
from kestrel_sovereign.features.strategic_memory.session_log import (
    _is_talon_pr,
    _review_latency,
)


class TestIsTalonPR:
    """Test _is_talon_pr detection logic."""

    def test_talon_branch_prefix(self):
        pr = {"head": {"ref": "talon/fix-auth-302"}, "labels": []}
        assert _is_talon_pr(pr) is True

    def test_talon_dash_branch_prefix(self):
        pr = {"head": {"ref": "talon-issue-42"}, "labels": []}
        assert _is_talon_pr(pr) is True

    def test_talon_label(self):
        pr = {"head": {"ref": "feature/something"}, "labels": [{"name": "talon-pr"}]}
        assert _is_talon_pr(pr) is True

    def test_talon_label_case_insensitive(self):
        pr = {"head": {"ref": "main"}, "labels": [{"name": "Talon-PR"}]}
        assert _is_talon_pr(pr) is True

    def test_non_talon_pr(self):
        pr = {"head": {"ref": "feature/add-tests"}, "labels": [{"name": "enhancement"}]}
        assert _is_talon_pr(pr) is False

    def test_no_head_no_labels(self):
        pr = {}
        assert _is_talon_pr(pr) is False

    def test_talon_in_middle_of_branch_not_matched(self):
        """Only prefix matches, not substring."""
        pr = {"head": {"ref": "fix/talon-bug"}, "labels": []}
        assert _is_talon_pr(pr) is False


class TestReviewLatency:
    """Test _review_latency calculation."""

    def test_normal_latency(self):
        created = "2026-03-30T08:00:00Z"
        merged = "2026-03-30T12:30:00Z"
        assert _review_latency(created, merged) == pytest.approx(4.5)

    def test_zero_latency(self):
        t = "2026-03-30T10:00:00Z"
        assert _review_latency(t, t) == pytest.approx(0.0)

    def test_invalid_timestamps(self):
        assert _review_latency("not-a-date", "also-not") == 0.0

    def test_none_input(self):
        assert _review_latency(None, None) == 0.0

    def test_multi_day_latency(self):
        created = "2026-03-29T08:00:00Z"
        merged = "2026-03-30T08:00:00Z"
        assert _review_latency(created, merged) == pytest.approx(24.0)
