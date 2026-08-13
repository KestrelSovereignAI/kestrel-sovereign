"""Tests for provider-neutral Strategic Memory session-log outcomes."""

from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

from kestrel_sovereign.features.strategic_memory.session_log import (
    _review_latency,
    collect_session_log,
)


class TestReviewLatency:
    """Test provider-neutral review-latency calculation."""

    def test_normal_latency(self):
        created = "2026-03-30T08:00:00Z"
        merged = "2026-03-30T12:30:00Z"
        assert _review_latency(created, merged) == pytest.approx(4.5)

    def test_zero_latency(self):
        t = "2026-03-30T10:00:00Z"
        assert _review_latency(t, t) == pytest.approx(0.0)

    def test_invalid_timestamps(self):
        assert _review_latency("not-a-date", "also-not") is None

    def test_none_input(self):
        assert _review_latency(None, None) is None

    def test_multi_day_latency(self):
        created = "2026-03-29T08:00:00Z"
        merged = "2026-03-30T08:00:00Z"
        assert _review_latency(created, merged) == pytest.approx(24.0)


@pytest.mark.asyncio
async def test_session_log_reports_all_prs_as_provider_neutral_outcomes():
    """Branch and label conventions do not create provider-owned sections."""

    today = date.today().isoformat()

    async def github_response(path, _token):
        if "/issues?" in path:
            return []
        if "/pulls?state=closed" in path:
            return [
                {
                    "number": 41,
                    "title": "Harden dispatch contract",
                    "user": {"login": "coding-service"},
                    "created_at": f"{today}T08:00:00Z",
                    "merged_at": f"{today}T12:30:00Z",
                    "head": {"ref": "vendor/fix-41"},
                    "labels": [{"name": "vendor-automation"}],
                },
                {
                    "number": 43,
                    "title": "Migrate historical contribution",
                    "user": {"login": "import-service"},
                    "created_at": None,
                    "merged_at": f"{today}T14:00:00Z",
                },
            ]
        if "/pulls?state=open" in path:
            return [{
                "number": 42,
                "title": "Add provider contract tests",
                "user": {"login": "another-service"},
                "created_at": f"{today}T13:00:00Z",
                "head": {"ref": "another-provider/tests-42"},
                "labels": [{"name": "automation"}],
            }]
        if "/issues/comments?" in path:
            return []
        raise AssertionError(f"unexpected GitHub path: {path}")

    with (
        patch(
            "kestrel_sovereign.features.strategic_memory.session_log.get_github_token",
            return_value="token",
        ),
        patch(
            "kestrel_sovereign.features.strategic_memory.session_log.github_api_get",
            new=AsyncMock(side_effect=github_response),
        ),
    ):
        report = await collect_session_log({
            "morning_signal_config": {"scan_repos": ["owner/repo"]},
        })

    assert "- **2** PRs merged" in report
    assert "- **1** PRs opened" in report
    assert "- **Avg review latency:** 4.5h" in report
    assert "repo#41: Harden dispatch contract (@coding-service, review: 4.5h)" in report
    assert "repo#43: Migrate historical contribution (@import-service, review: unavailable)" in report
    assert "review: 0.0h" not in report
    assert "repo#42: Add provider contract tests (@another-service)" in report
    assert [line for line in report.splitlines() if line.startswith("## ")] == [
        "## Outcomes",
        "## PRs Merged",
        "## PRs Opened",
        "## Contributor Scoreboard",
    ]
