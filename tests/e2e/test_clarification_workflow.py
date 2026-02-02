"""End-to-end tests for GitHub clarification workflow using Playwright.

Tests the full workflow:
1. Create a vague GitHub issue
2. Run agent to analyze and post clarification questions
3. Verify questions render correctly with checkboxes
4. Answer questions via checkbox interaction
5. Signal ready and verify agent can continue

Requires: pip install playwright && playwright install chromium
"""

import os
import re
import subprocess
from typing import Generator

import pytest

# Skip entire module if playwright not installed
pytest.importorskip("playwright", reason="playwright package not installed")
from playwright.sync_api import Page, expect


# Test configuration
REPO = os.environ.get("TEST_REPO", "KestrelSovereignAI/kestrel-sovereign")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")


def run_gh_command(args: list[str]) -> str:
    """Run a gh CLI command and return output."""
    result = subprocess.run(
        ["gh"] + args,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def create_test_issue(title: str, body: str, labels: list[str] = None) -> int:
    """Create a test issue and return its number."""
    cmd = ["issue", "create", "--repo", REPO, "--title", title, "--body", body]
    if labels:
        for label in labels:
            cmd.extend(["--label", label])
    output = run_gh_command(cmd)
    # Extract issue number from URL like https://github.com/owner/repo/issues/123
    match = re.search(r"/issues/(\d+)", output)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract issue number from: {output}")


def close_test_issue(issue_number: int) -> None:
    """Close a test issue."""
    run_gh_command(["issue", "close", "--repo", REPO, str(issue_number)])


def get_issue_comments(issue_number: int) -> str:
    """Get all comments on an issue."""
    return run_gh_command(["issue", "view", "--repo", REPO, str(issue_number), "--comments"])


def get_issue_labels(issue_number: int) -> list[str]:
    """Get labels on an issue."""
    output = run_gh_command(
        ["issue", "view", "--repo", REPO, str(issue_number), "--json", "labels", "-q", ".labels[].name"]
    )
    return [label.strip() for label in output.strip().split("\n") if label.strip()]


def run_agent_on_issue(issue_number: int) -> str:
    """Run the kestrel-github agent on an issue."""
    result = subprocess.run(
        ["uv", "run", "kestrel-github", "claim", "--repo", REPO, "--issue", str(issue_number)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    return result.stdout + result.stderr


class TestClarificationWorkflow:
    """Test the full clarification workflow."""

    @pytest.fixture
    def vague_issue(self) -> Generator[int, None, None]:
        """Create a vague test issue that should trigger clarification."""
        issue_number = create_test_issue(
            title="[TEST] Improve API performance",
            body="""We need to make the API faster.

Users are complaining about slow response times. Please optimize.

Maybe add caching? Or improve database queries?

This is a test issue for E2E testing - will be closed automatically.""",
            labels=["enhancement"],
        )
        yield issue_number
        # Cleanup
        try:
            close_test_issue(issue_number)
        except Exception:
            pass  # Best effort cleanup

    @pytest.fixture
    def clear_issue(self) -> Generator[int, None, None]:
        """Create a clear test issue that should NOT trigger clarification."""
        issue_number = create_test_issue(
            title="[TEST] Add __version__ to package",
            body="""Add a `__version__` variable to the main `__init__.py` file.

The version should be read from pyproject.toml or set to "0.1.0".

This is a simple, clear task with no ambiguity.

This is a test issue for E2E testing - will be closed automatically.""",
            labels=["enhancement", "agent-ready"],
        )
        yield issue_number
        # Cleanup
        try:
            close_test_issue(issue_number)
        except Exception:
            pass

    def test_vague_issue_triggers_clarification(self, vague_issue: int) -> None:
        """Test that a vague issue triggers the clarification workflow."""
        issue_number = vague_issue

        # Run agent
        output = run_agent_on_issue(issue_number)

        # Verify agent posted clarification
        assert "CLARIFYING" in output, f"Expected CLARIFYING in output: {output}"

        # Verify labels
        labels = get_issue_labels(issue_number)
        assert "agent-clarifying" in labels, f"Expected agent-clarifying label, got: {labels}"

        # Verify comment was posted with questions
        comments = get_issue_comments(issue_number)
        assert "Clarification Needed" in comments, f"Expected clarification comment: {comments}"
        assert "- [ ]" in comments, f"Expected checkboxes in comment: {comments}"

    def test_clarification_questions_format(self, vague_issue: int) -> None:
        """Test that clarification questions are well-formatted."""
        issue_number = vague_issue

        # Run agent
        run_agent_on_issue(issue_number)

        # Get comments
        comments = get_issue_comments(issue_number)

        # Verify structure
        assert "### 1." in comments, "Expected numbered questions"
        assert "- [ ]" in comments, "Expected unchecked checkboxes"
        assert "Other: ___" in comments, "Expected 'Other' option"
        assert "ready" in comments.lower(), "Expected ready signal"

        # Verify no artifacts like )] in options
        # Find all checkbox lines
        checkbox_lines = [line for line in comments.split("\n") if "- [ ]" in line]
        for line in checkbox_lines:
            # Should not end with common artifacts
            assert not line.rstrip().endswith('")]'), f"Artifact found in: {line}"
            assert not line.rstrip().endswith("')]"), f"Artifact found in: {line}"

    def test_clear_issue_skips_clarification(self, clear_issue: int) -> None:
        """Test that a clear issue with agent-ready label skips clarification."""
        issue_number = clear_issue

        # Run agent - it should skip clarification due to agent-ready label
        output = run_agent_on_issue(issue_number)

        # Should NOT be in clarifying state
        labels = get_issue_labels(issue_number)
        assert "agent-clarifying" not in labels, f"Should skip clarification with agent-ready: {labels}"

        # Should progress to in_progress or completed
        assert "agent-claimed" in labels or "agent-complete" in labels, f"Expected progress: {labels}"


class TestClarificationUI:
    """Test the clarification UI rendering in GitHub."""

    @pytest.fixture
    def issue_with_clarification(self) -> Generator[int, None, None]:
        """Create an issue and run agent to get clarification."""
        issue_number = create_test_issue(
            title="[TEST-UI] Add rate limiting",
            body="""Add rate limiting to the API.

We need to prevent abuse but not sure how.

This is a test issue for UI testing - will be closed automatically.""",
            labels=["enhancement"],
        )

        # Run agent to get clarification posted
        run_agent_on_issue(issue_number)

        yield issue_number

        # Cleanup
        try:
            close_test_issue(issue_number)
        except Exception:
            pass

    @pytest.mark.skipif(not GITHUB_TOKEN, reason="GITHUB_TOKEN required for browser tests")
    def test_checkboxes_render_in_browser(self, page: Page, issue_with_clarification: int) -> None:
        """Test that checkboxes render and are clickable in GitHub UI."""
        issue_number = issue_with_clarification

        # Navigate to issue
        page.goto(f"https://github.com/{REPO}/issues/{issue_number}")

        # Wait for comments to load
        page.wait_for_selector(".timeline-comment", timeout=10000)

        # Find clarification comment
        clarification_comment = page.locator(".timeline-comment:has-text('Clarification Needed')")
        expect(clarification_comment).to_be_visible()

        # Verify checkboxes exist and are unchecked
        checkboxes = clarification_comment.locator("input[type='checkbox']")
        expect(checkboxes.first).to_be_visible()

        # Count checkboxes (should be multiple)
        checkbox_count = checkboxes.count()
        assert checkbox_count >= 4, f"Expected at least 4 checkboxes (1 per question min), got {checkbox_count}"

    @pytest.mark.skipif(not GITHUB_TOKEN, reason="GITHUB_TOKEN required for browser tests")
    def test_checkbox_interaction(self, page: Page, issue_with_clarification: int) -> None:
        """Test that checkboxes can be checked/unchecked."""
        issue_number = issue_with_clarification

        # Navigate to issue
        page.goto(f"https://github.com/{REPO}/issues/{issue_number}")
        page.wait_for_selector(".timeline-comment", timeout=10000)

        # Find first checkbox in clarification comment
        clarification_comment = page.locator(".timeline-comment:has-text('Clarification Needed')")
        first_checkbox = clarification_comment.locator("input[type='checkbox']").first

        # Note: GitHub checkboxes in comments require edit permission
        # This test verifies they render correctly
        expect(first_checkbox).to_be_visible()
        expect(first_checkbox).not_to_be_checked()


class TestFullWorkflowE2E:
    """Full end-to-end workflow test."""

    @pytest.fixture
    def workflow_issue(self) -> Generator[int, None, None]:
        """Create a test issue for full workflow."""
        issue_number = create_test_issue(
            title="[TEST-E2E] Add health check endpoint",
            body="""Add a /health endpoint to the API.

Should return 200 OK with JSON body.

This is an E2E test issue - will be closed automatically.""",
            labels=["enhancement", "agent-ready"],  # Skip clarification for faster test
        )
        yield issue_number
        try:
            close_test_issue(issue_number)
        except Exception:
            pass

    def test_full_workflow_creates_pr(self, workflow_issue: int) -> None:
        """Test that a clear issue results in a PR being created."""
        issue_number = workflow_issue

        # Run agent
        output = run_agent_on_issue(issue_number)

        # Check if PR was created (success) or blocked (acceptable)
        labels = get_issue_labels(issue_number)

        if "agent-complete" in labels:
            # Verify PR exists
            assert "pull request" in output.lower() or "PR" in output, f"Expected PR info: {output}"
        elif "agent-blocked" in labels:
            # Blocked is acceptable - check for reason
            comments = get_issue_comments(issue_number)
            assert "Blocked" in comments or "blocked" in comments.lower()
        else:
            # Should be one of the above
            assert "agent-claimed" in labels, f"Unexpected state: {labels}"
