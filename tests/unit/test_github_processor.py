"""Tests for Kestrel Talon (GitHub/ADO issue processor).

Run with: pytest tests/unit/test_github_processor.py -v
"""

import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

kestreltalon = pytest.importorskip("kestreltalon")
from kestreltalon.config import TalonConfig
from kestreltalon.models import (
    CIStatus,
    IssueComment,
    IssueContext,
    ProcessingResult,
    ProcessingStatus,
)


class TestTalonConfig:
    """Tests for configuration."""

    def test_config_defaults(self):
        config = TalonConfig()

        assert config.bot_assignee == "claude-bot"
        assert config.max_turns == 50
        assert config.dry_run is False

    def test_config_validation_missing_token(self):
        with patch.dict(os.environ, {"GITHUB_TOKEN": "", "ANTHROPIC_API_KEY": "test"}):
            config = TalonConfig(
                github_token="",
                anthropic_api_key="test",
                repo="owner/repo",
            )
            errors = config.validate()

            assert "GITHUB_TOKEN" in str(errors)

    def test_config_validation_missing_repo(self):
        config = TalonConfig(
            github_token="test",
            anthropic_api_key="test",
            repo="",
        )
        errors = config.validate()

        assert "repo" in str(errors).lower()

    def test_config_validation_invalid_repo_format(self):
        config = TalonConfig(
            github_token="test",
            anthropic_api_key="test",
            repo="invalid-repo-format",
        )
        errors = config.validate()

        assert "owner/repo" in str(errors)

    def test_config_repo_parts(self):
        config = TalonConfig(repo="myorg/myrepo")

        assert config.repo_owner == "myorg"
        assert config.repo_name == "myrepo"


class TestIssueContext:
    """Tests for IssueContext model."""

    def test_format_for_prompt_basic(self):
        context = IssueContext(
            number=42,
            title="Fix authentication bug",
            body="The login fails when...",
            author="testuser",
            labels=["bug", "high-priority"],
            assignees=["claude-bot"],
            comments=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            url="https://github.com/owner/repo/issues/42",
        )

        prompt = context.format_for_prompt()

        assert "# Issue #42: Fix authentication bug" in prompt
        assert "The login fails when..." in prompt
        assert "bug, high-priority" in prompt

    def test_format_for_prompt_with_comments(self):
        context = IssueContext(
            number=42,
            title="Test issue",
            body="Issue body",
            author="testuser",
            labels=[],
            assignees=[],
            comments=[
                IssueComment(
                    id=1,
                    author="reviewer",
                    body="Have you tried X?",
                    created_at=datetime(2024, 1, 15, 10, 30),
                ),
            ],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            url="https://github.com/owner/repo/issues/42",
        )

        prompt = context.format_for_prompt()

        assert "## Discussion" in prompt
        assert "reviewer" in prompt
        assert "Have you tried X?" in prompt

    def test_format_for_prompt_with_files(self):
        context = IssueContext(
            number=42,
            title="Test issue",
            body="",
            author="testuser",
            labels=[],
            assignees=[],
            comments=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            url="https://github.com/owner/repo/issues/42",
            referenced_files=["src/auth.py", "tests/test_auth.py"],
        )

        prompt = context.format_for_prompt()

        assert "## Referenced Files" in prompt
        assert "src/auth.py" in prompt
        assert "tests/test_auth.py" in prompt


class TestProcessingResult:
    """Tests for ProcessingResult model."""

    def test_summary_completed(self):
        result = ProcessingResult(
            issue_number=42,
            status=ProcessingStatus.COMPLETED,
            branch_name="issue-42-fix-bug",
            pr_url="https://github.com/owner/repo/pull/100",
            commits=["abc123", "def456"],
        )

        summary = result.summary()

        assert "#42" in summary
        assert "completed" in summary
        assert "issue-42-fix-bug" in summary
        assert "https://github.com/owner/repo/pull/100" in summary
        assert "Commits: 2" in summary

    def test_summary_blocked(self):
        result = ProcessingResult(
            issue_number=42,
            status=ProcessingStatus.BLOCKED,
            blocking_question="What database should I use?",
        )

        summary = result.summary()

        assert "#42" in summary
        assert "blocked" in summary
        assert "What database should I use?" in summary


class TestCIStatus:
    """Tests for CIStatus model."""

    def test_is_pending(self):
        status = CIStatus(conclusion=None, status="in_progress")
        assert status.is_pending is True

        status = CIStatus(conclusion=None, status="queued")
        assert status.is_pending is True

        status = CIStatus(conclusion="success", status="completed")
        assert status.is_pending is False

    def test_is_success(self):
        status = CIStatus(conclusion="success", status="completed")
        assert status.is_success is True

        status = CIStatus(conclusion="failure", status="completed")
        assert status.is_success is False

    def test_is_failure(self):
        status = CIStatus(conclusion="failure", status="completed")
        assert status.is_failure is True

        status = CIStatus(conclusion="timed_out", status="completed")
        assert status.is_failure is True

        status = CIStatus(conclusion="success", status="completed")
        assert status.is_failure is False

    def test_failure_summary(self):
        status = CIStatus(
            conclusion="failure",
            status="completed",
            check_runs=[
                {
                    "name": "pytest",
                    "conclusion": "failure",
                    "output": {"summary": "3 tests failed"},
                },
                {
                    "name": "lint",
                    "conclusion": "success",
                    "output": {"summary": "All good"},
                },
            ],
        )

        summary = status.failure_summary()

        assert "pytest" in summary
        assert "3 tests failed" in summary
        assert "lint" not in summary


class TestGitHubClientExtraction:
    """Tests for extraction methods in GitHubClient."""

    def test_extract_file_references(self):
        from kestreltalon.github_client import GitHubClient

        config = TalonConfig(
            github_token="test",
            anthropic_api_key="test",
            repo="owner/repo",
        )
        client = GitHubClient(config)

        context = IssueContext(
            number=1,
            title="Test",
            body="Check `src/auth.py` and see src/utils/helpers.py for details",
            author="test",
            labels=[],
            assignees=[],
            comments=[
                IssueComment(
                    id=1,
                    author="other",
                    body="Also look at tests/test_auth.py",
                    created_at=datetime.now(),
                )
            ],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            url="https://github.com/owner/repo/issues/1",
        )

        files = client._extract_file_references(context)

        assert "src/auth.py" in files
        assert "src/utils/helpers.py" in files
        assert "tests/test_auth.py" in files

    def test_extract_error_messages(self):
        from kestreltalon.github_client import GitHubClient

        config = TalonConfig(
            github_token="test",
            anthropic_api_key="test",
            repo="owner/repo",
        )
        client = GitHubClient(config)

        context = IssueContext(
            number=1,
            title="Test",
            body="""
Here's the error:
```
Traceback (most recent call last):
  File "auth.py", line 10
    raise ValueError("Invalid token")
ValueError: Invalid token
```
            """,
            author="test",
            labels=[],
            assignees=[],
            comments=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            url="https://github.com/owner/repo/issues/1",
        )

        errors = client._extract_error_messages(context)

        assert len(errors) == 1
        assert "Traceback" in errors[0]
        assert "ValueError" in errors[0]


@pytest.mark.skipif(
    not os.environ.get("GITHUB_TOKEN") or not os.environ.get("ANTHROPIC_API_KEY"),
    reason="Requires GITHUB_TOKEN and ANTHROPIC_API_KEY",
)
class TestTalonE2E:
    """End-to-end tests requiring real credentials."""

    @pytest.fixture
    def config(self):
        return TalonConfig(
            repo=os.environ.get("TEST_GITHUB_REPO", "your-org/your-repo"),
            dry_run=True,
        )

    @pytest.mark.asyncio
    async def test_fetch_assigned_issues(self, config):
        from kestreltalon.github_client import GitHubClient

        client = GitHubClient(config)
        issues = client.get_assigned_issues()

        assert isinstance(issues, list)

    @pytest.mark.asyncio
    async def test_build_issue_context(self, config):
        from kestreltalon.github_client import GitHubClient

        client = GitHubClient(config)

        issues = list(client.repo.get_issues(state="open"))
        if not issues:
            pytest.skip("No open issues in test repo")

        from kestreltalon.platform import WorkItem
        work_item = WorkItem(number=issues[0].number, title=issues[0].title, raw=issues[0])
        context = client.build_issue_context(work_item)

        assert context.number == work_item.number
        assert context.title == work_item.title
        assert isinstance(context.comments, list)
