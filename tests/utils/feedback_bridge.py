"""
Test Result Feedback Bridge.

Connects pytest test results to the Kestrel/kestrel feedback system,
enabling automatic feedback submission from test runs for agent
self-reflection and improvement.

This creates a feedback loop:
1. Tests run and collect results
2. Failures/errors become SYSTEM feedback entries
3. ReflectionFeature analyzes patterns in feedback
4. Insights can become improvement tickets

Usage:
    # As pytest plugin (automatic)
    pytest --feedback-bridge tests/

    # Programmatic usage
    from tests.utils.feedback_bridge import TestResultCollector, submit_test_feedback

    collector = TestResultCollector()
    # ... run tests ...
    await submit_test_feedback(collector.results, feedback_store)
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TestOutcome(str, Enum):
    """Test execution outcome."""
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    XFAILED = "xfailed"  # Expected failure
    XPASSED = "xpassed"  # Unexpected pass


@dataclass
class TestResult:
    """Individual test result."""
    node_id: str  # pytest node ID (e.g., "tests/test_foo.py::test_bar")
    outcome: TestOutcome
    duration: float  # seconds
    file_path: str
    function_name: str
    class_name: Optional[str] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    markers: list[str] = field(default_factory=list)
    worker_id: Optional[str] = None  # For parallel execution
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TestRunSummary:
    """Summary of a complete test run."""
    run_id: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    xfailed: int = 0
    xpassed: int = 0
    duration: float = 0.0
    workers: int = 1
    test_scope: str = "all"  # Could be "unit", "integration", "e2e"
    git_branch: Optional[str] = None
    git_commit: Optional[str] = None
    results: list[TestResult] = field(default_factory=list)


class TestResultCollector:
    """
    Collects test results during pytest execution.

    Can be used as a pytest plugin or standalone.
    """

    def __init__(self, run_id: Optional[str] = None):
        from tests.utils.parallel_support import get_worker_id
        import uuid

        self.run_id = run_id or f"test-{uuid.uuid4().hex[:8]}"
        self.worker_id = get_worker_id()
        self.summary = TestRunSummary(
            run_id=self.run_id,
            started_at=datetime.utcnow(),
        )
        self._collect_git_info()

    def _collect_git_info(self):
        """Collect git branch and commit information."""
        import subprocess

        try:
            # Get current branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                self.summary.git_branch = result.stdout.strip()

            # Get current commit
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                self.summary.git_commit = result.stdout.strip()
        except Exception:
            pass  # Git info is optional

    def add_result(self, result: TestResult):
        """Add a test result to the collection."""
        self.summary.results.append(result)
        self.summary.total_tests += 1
        self.summary.duration += result.duration

        if result.outcome == TestOutcome.PASSED:
            self.summary.passed += 1
        elif result.outcome == TestOutcome.FAILED:
            self.summary.failed += 1
        elif result.outcome == TestOutcome.ERROR:
            self.summary.errors += 1
        elif result.outcome == TestOutcome.SKIPPED:
            self.summary.skipped += 1
        elif result.outcome == TestOutcome.XFAILED:
            self.summary.xfailed += 1
        elif result.outcome == TestOutcome.XPASSED:
            self.summary.xpassed += 1

    def finalize(self):
        """Mark the test run as complete."""
        self.summary.finished_at = datetime.utcnow()

    @property
    def failures(self) -> list[TestResult]:
        """Get all failed tests."""
        return [r for r in self.summary.results if r.outcome == TestOutcome.FAILED]

    @property
    def errors(self) -> list[TestResult]:
        """Get all error tests."""
        return [r for r in self.summary.results if r.outcome == TestOutcome.ERROR]

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate percentage."""
        if self.summary.total_tests == 0:
            return 100.0
        return (self.summary.passed / self.summary.total_tests) * 100


def determine_severity(result: TestResult, summary: TestRunSummary) -> str:
    """
    Determine feedback severity based on test result context.

    Critical:
    - Integration/E2E test failures
    - Tests with 'critical' marker
    - Errors (not failures)

    High:
    - Multiple failures in same file
    - Tests with 'security' marker

    Medium:
    - Standard test failures

    Low:
    - Unexpected passes (xpassed)
    - Flaky test indicators
    """
    # Check markers
    if "critical" in result.markers:
        return "critical"
    if "security" in result.markers:
        return "high"

    # Errors are more severe than failures
    if result.outcome == TestOutcome.ERROR:
        return "critical"

    # Integration tests are higher severity
    if "integration" in result.node_id or "e2e" in result.node_id:
        return "high"

    # Multiple failures in same file indicate systematic issue
    same_file_failures = sum(
        1 for r in summary.results
        if r.file_path == result.file_path
        and r.outcome in (TestOutcome.FAILED, TestOutcome.ERROR)
    )
    if same_file_failures >= 3:
        return "high"

    # Default severity
    if result.outcome == TestOutcome.FAILED:
        return "medium"

    return "low"


def determine_category(result: TestResult) -> str:
    """
    Determine feedback category based on test result.

    - error: Test execution error (not assertion failure)
    - bug: Assertion failure
    - improvement: xpassed (code improved, test expectation outdated)
    """
    if result.outcome == TestOutcome.ERROR:
        return "error"
    elif result.outcome == TestOutcome.XPASSED:
        return "improvement"
    else:
        return "bug"


def create_feedback_title(result: TestResult) -> str:
    """Create concise feedback title from test result."""
    # Extract test name from node_id
    parts = result.node_id.split("::")
    test_name = parts[-1] if parts else result.function_name

    # Make it readable
    readable = test_name.replace("_", " ").replace("test ", "")

    outcome_prefix = {
        TestOutcome.FAILED: "Test failure:",
        TestOutcome.ERROR: "Test error:",
        TestOutcome.XPASSED: "Unexpected pass:",
    }.get(result.outcome, "Test issue:")

    return f"{outcome_prefix} {readable}"


def create_feedback_description(result: TestResult, summary: TestRunSummary) -> str:
    """Create detailed feedback description from test result."""
    lines = [
        f"## Test: {result.node_id}",
        "",
        f"**Outcome:** {result.outcome.value}",
        f"**Duration:** {result.duration:.2f}s",
        f"**File:** {result.file_path}",
        f"**Function:** {result.function_name}",
    ]

    if result.class_name:
        lines.append(f"**Class:** {result.class_name}")

    if result.markers:
        lines.append(f"**Markers:** {', '.join(result.markers)}")

    if result.error_message:
        lines.extend([
            "",
            "## Error Message",
            "```",
            result.error_message,
            "```",
        ])

    if result.error_traceback:
        lines.extend([
            "",
            "## Traceback",
            "```python",
            result.error_traceback[:2000],  # Truncate long tracebacks
            "```" if len(result.error_traceback) <= 2000 else "... (truncated)\n```",
        ])

    # Add context about the test run
    lines.extend([
        "",
        "## Test Run Context",
        f"- **Run ID:** {summary.run_id}",
        f"- **Total Tests:** {summary.total_tests}",
        f"- **Pass Rate:** {(summary.passed / max(1, summary.total_tests)) * 100:.1f}%",
    ])

    if summary.git_branch:
        lines.append(f"- **Branch:** {summary.git_branch}")
    if summary.git_commit:
        lines.append(f"- **Commit:** {summary.git_commit}")
    if result.worker_id:
        lines.append(f"- **Worker:** {result.worker_id}")

    return "\n".join(lines)


async def submit_test_feedback(
    summary: TestRunSummary,
    feedback_store,
    agent_name: str = "test-system",
    min_severity: str = "low",
    include_summary: bool = True,
) -> list[str]:
    """
    Submit test results as feedback entries.

    Args:
        summary: Completed test run summary
        feedback_store: FeedbackStore instance (SQLite or PostgreSQL)
        agent_name: Agent identifier for feedback attribution
        min_severity: Minimum severity to submit ('low', 'medium', 'high', 'critical')
        include_summary: Whether to include a run summary entry

    Returns:
        List of created feedback IDs
    """
    from kestrel_sovereign.a2a.stores.feedback_store import (
        FeedbackSource, FeedbackCategory, FeedbackSeverity
    )

    severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    min_level = severity_order.get(min_severity, 0)

    feedback_ids = []

    # Submit individual failure/error feedback
    for result in summary.results:
        if result.outcome not in (TestOutcome.FAILED, TestOutcome.ERROR, TestOutcome.XPASSED):
            continue

        severity = determine_severity(result, summary)
        if severity_order.get(severity, 0) < min_level:
            continue

        category = determine_category(result)
        title = create_feedback_title(result)
        description = create_feedback_description(result, summary)

        context = {
            "node_id": result.node_id,
            "file_path": result.file_path,
            "function_name": result.function_name,
            "duration": result.duration,
            "markers": result.markers,
            "worker_id": result.worker_id,
            "run_id": summary.run_id,
        }

        if result.error_message:
            context["error_message"] = result.error_message

        try:
            feedback_id = await feedback_store.submit_feedback(
                agent_name=agent_name,
                source=FeedbackSource.SYSTEM,
                category=FeedbackCategory(category),
                severity=FeedbackSeverity(severity),
                title=title,
                description=description,
                context=context,
                metadata={
                    "source_type": "test_result",
                    "test_outcome": result.outcome.value,
                    "git_branch": summary.git_branch,
                    "git_commit": summary.git_commit,
                }
            )
            feedback_ids.append(feedback_id)
            logger.info(f"Submitted test feedback: {feedback_id} - {title}")
        except Exception as e:
            logger.error(f"Failed to submit test feedback: {e}")

    # Submit run summary if requested and there were failures
    if include_summary and (summary.failed > 0 or summary.errors > 0):
        try:
            summary_title = f"Test run summary: {summary.failed} failures, {summary.errors} errors"
            summary_description = _create_summary_description(summary)
            summary_severity = "critical" if summary.errors > 0 else (
                "high" if summary.failed >= 5 else "medium"
            )

            feedback_id = await feedback_store.submit_feedback(
                agent_name=agent_name,
                source=FeedbackSource.SYSTEM,
                category=FeedbackCategory.ERROR,
                severity=FeedbackSeverity(summary_severity),
                title=summary_title,
                description=summary_description,
                context={
                    "run_id": summary.run_id,
                    "total_tests": summary.total_tests,
                    "passed": summary.passed,
                    "failed": summary.failed,
                    "errors": summary.errors,
                    "skipped": summary.skipped,
                    "duration": summary.duration,
                    "pass_rate": (summary.passed / max(1, summary.total_tests)) * 100,
                },
                metadata={
                    "source_type": "test_run_summary",
                    "git_branch": summary.git_branch,
                    "git_commit": summary.git_commit,
                }
            )
            feedback_ids.append(feedback_id)
            logger.info(f"Submitted test run summary feedback: {feedback_id}")
        except Exception as e:
            logger.error(f"Failed to submit test run summary: {e}")

    return feedback_ids


def _create_summary_description(summary: TestRunSummary) -> str:
    """Create test run summary description."""
    duration = summary.finished_at - summary.started_at if summary.finished_at else timedelta(0)

    lines = [
        f"# Test Run Summary: {summary.run_id}",
        "",
        f"**Started:** {summary.started_at.isoformat()}",
        f"**Duration:** {duration.total_seconds():.1f}s",
        "",
        "## Results",
        f"- **Total:** {summary.total_tests}",
        f"- **Passed:** {summary.passed} ({(summary.passed / max(1, summary.total_tests)) * 100:.1f}%)",
        f"- **Failed:** {summary.failed}",
        f"- **Errors:** {summary.errors}",
        f"- **Skipped:** {summary.skipped}",
    ]

    if summary.xfailed > 0:
        lines.append(f"- **Expected Failures:** {summary.xfailed}")
    if summary.xpassed > 0:
        lines.append(f"- **Unexpected Passes:** {summary.xpassed}")

    if summary.git_branch or summary.git_commit:
        lines.extend([
            "",
            "## Git Context",
        ])
        if summary.git_branch:
            lines.append(f"- **Branch:** {summary.git_branch}")
        if summary.git_commit:
            lines.append(f"- **Commit:** {summary.git_commit}")

    # List failed tests
    if summary.failed > 0 or summary.errors > 0:
        lines.extend([
            "",
            "## Failed Tests",
        ])
        for result in summary.results:
            if result.outcome in (TestOutcome.FAILED, TestOutcome.ERROR):
                icon = "❌" if result.outcome == TestOutcome.FAILED else "💥"
                lines.append(f"- {icon} {result.node_id}")

    return "\n".join(lines)


# Export for pytest plugin
def pytest_configure(config):
    """Register the feedback bridge plugin."""
    if hasattr(config, "_feedback_bridge_active"):
        return

    # Check if feedback bridge is enabled
    if not config.getoption("--feedback-bridge", default=False):
        return

    config._feedback_bridge_active = True
    config._feedback_collector = TestResultCollector()
    config.pluginmanager.register(FeedbackBridgePlugin(config._feedback_collector))


class FeedbackBridgePlugin:
    """Pytest plugin for automatic feedback submission."""

    def __init__(self, collector: TestResultCollector):
        self.collector = collector

    def pytest_runtest_logreport(self, report):
        """Collect test results as they complete."""
        # Only process the 'call' phase (actual test execution)
        if report.when != "call":
            return

        # Map pytest outcome to our enum
        outcome_map = {
            "passed": TestOutcome.PASSED,
            "failed": TestOutcome.FAILED,
            "skipped": TestOutcome.SKIPPED,
        }
        outcome = outcome_map.get(report.outcome, TestOutcome.ERROR)

        # Check for xfail/xpass
        if hasattr(report, "wasxfail"):
            outcome = TestOutcome.XPASSED if report.passed else TestOutcome.XFAILED

        # Extract test info
        node_parts = report.nodeid.split("::")
        file_path = node_parts[0] if node_parts else ""
        class_name = node_parts[1] if len(node_parts) > 2 else None
        function_name = node_parts[-1] if node_parts else ""

        # Get error info
        error_message = None
        error_traceback = None
        if hasattr(report, "longrepr") and report.longrepr:
            if hasattr(report.longrepr, "reprcrash"):
                error_message = str(report.longrepr.reprcrash)
            error_traceback = str(report.longrepr)

        # Get markers
        markers = []
        if hasattr(report, "keywords"):
            markers = [k for k in report.keywords if not k.startswith("_")]

        result = TestResult(
            node_id=report.nodeid,
            outcome=outcome,
            duration=report.duration,
            file_path=file_path,
            function_name=function_name,
            class_name=class_name,
            error_message=error_message,
            error_traceback=error_traceback,
            markers=markers,
            worker_id=os.environ.get("PYTEST_XDIST_WORKER"),
        )

        self.collector.add_result(result)

    def pytest_sessionfinish(self, session):
        """Submit feedback when test session completes."""
        self.collector.finalize()

        # Only submit if there were failures
        if self.collector.summary.failed == 0 and self.collector.summary.errors == 0:
            return

        # Check for feedback store configuration
        db_path = os.environ.get("KESTREL_FEEDBACK_DB")
        if not db_path:
            logger.info("KESTREL_FEEDBACK_DB not set, skipping feedback submission")
            return

        # Submit asynchronously
        try:
            from kestrel_sovereign.a2a.stores.feedback_store import SQLiteFeedbackStore

            async def _submit():
                store = SQLiteFeedbackStore(db_path)
                await store.initialize()
                ids = await submit_test_feedback(
                    self.collector.summary,
                    store,
                    agent_name=os.environ.get("KESTREL_AGENT_NAME", "test-system"),
                )
                logger.info(f"Submitted {len(ids)} feedback entries from test run")

            asyncio.get_event_loop().run_until_complete(_submit())
        except Exception as e:
            logger.error(f"Failed to submit test feedback: {e}")


def pytest_addoption(parser):
    """Add --feedback-bridge option to pytest."""
    parser.addoption(
        "--feedback-bridge",
        action="store_true",
        default=False,
        help="Enable test result feedback submission to Kestrel feedback store",
    )
