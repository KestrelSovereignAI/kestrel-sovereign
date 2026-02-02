"""
Test Result Analyzer for Agent Reflection.

Analyzes patterns in test failures to help agents understand
recurring issues and improve their behavior.

This module bridges test results to the agent reflection system:
1. Aggregates test failures from feedback store
2. Identifies patterns (flaky tests, recurring failures, regressions)
3. Generates insights for ReflectionFeature consumption
4. Proposes improvement actions

Usage:
    from tests.utils.test_result_analyzer import TestResultAnalyzer

    analyzer = TestResultAnalyzer(feedback_store)
    insights = await analyzer.analyze_recent_failures(days=7)

    for insight in insights:
        print(f"{insight.type}: {insight.title}")
        print(f"  Evidence: {len(insight.evidence)} test failures")
        print(f"  Suggested action: {insight.suggested_action}")
"""

import re
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class InsightType(Enum):
    """Type of insight from test analysis."""
    PATTERN = "pattern"           # Recurring test failure pattern
    REGRESSION = "regression"     # Previously passing test now fails
    FLAKY = "flaky"              # Test sometimes passes, sometimes fails
    IMPROVEMENT = "improvement"   # Test expectations outdated (xpassed)
    HOTSPOT = "hotspot"          # Area of code with many failures
    PERFORMANCE = "performance"  # Tests taking too long


@dataclass
class TestInsight:
    """An insight derived from test result analysis."""
    type: InsightType
    title: str
    description: str
    evidence: list[str]  # Feedback IDs supporting this insight
    confidence: float    # 0.0 - 1.0
    severity: str        # 'low', 'medium', 'high', 'critical'
    actionable: bool
    suggested_action: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestPattern:
    """A detected pattern in test failures."""
    pattern_key: str  # e.g., "file:path/to/test.py" or "error:ModuleNotFoundError"
    occurrences: int
    failure_ids: list[str]
    first_seen: datetime
    last_seen: datetime
    description: str


class TestResultAnalyzer:
    """
    Analyzes test failures to generate insights for agent reflection.

    Connects to the feedback store and looks for patterns in test-related
    entries (those with source_type='test_result' in metadata).

    Can also analyze TestResult objects directly (for CI scripts).
    """

    # Minimum occurrences to consider a pattern significant
    MIN_PATTERN_OCCURRENCES = 2

    # Minimum confidence threshold for actionable insights
    MIN_CONFIDENCE = 0.6

    def __init__(self, feedback_store=None):
        """
        Initialize the analyzer.

        Args:
            feedback_store: Optional FeedbackStore instance (SQLite or PostgreSQL).
                           Can be None if using analyze() directly with TestResults.
        """
        self.store = feedback_store

    def analyze(self, results: list) -> list[TestInsight]:
        """
        Synchronously analyze a list of TestResult objects.

        This is the simpler entry point for CI scripts that already have
        TestResult objects (from FeedbackBridge or direct construction).

        Args:
            results: List of TestResult objects from feedback_bridge

        Returns:
            List of TestInsight objects
        """
        if not results:
            return []

        insights = []

        # Analyze by different dimensions
        insights.extend(self._analyze_file_patterns_sync(results))
        insights.extend(self._analyze_error_patterns_sync(results))
        insights.extend(self._analyze_hotspots_sync(results))
        insights.extend(self._analyze_flaky_tests_sync(results))

        # Filter by confidence
        insights = [i for i in insights if i.confidence >= self.MIN_CONFIDENCE]

        # Sort by severity and confidence
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        insights.sort(key=lambda x: (severity_order.get(x.severity, 4), -x.confidence))

        return insights

    def analyze_from_db(self, db_path: str) -> list[TestInsight]:
        """
        Analyze test results from a feedback database file.

        This is for CI scripts that need to read directly from SQLite.

        Args:
            db_path: Path to the SQLite feedback database

        Returns:
            List of TestInsight objects
        """
        import json
        import sqlite3
        from pathlib import Path

        if not Path(db_path).exists():
            return []

        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.execute(
                "SELECT category, severity, content, context FROM feedback"
            )
            entries = []
            for row in cursor:
                entries.append({
                    "category": row[0],
                    "severity": row[1],
                    "content": row[2],
                    "context": json.loads(row[3]) if row[3] else {}
                })
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()

        # Convert to TestResult-like objects
        from tests.utils.feedback_bridge import TestResult, TestOutcome

        results = []
        for e in entries:
            ctx = e.get("context", {})
            outcome = TestOutcome.FAILED if e["category"] == "bug" else TestOutcome.ERROR
            results.append(TestResult(
                node_id=ctx.get("test_node_id", "unknown"),
                outcome=outcome,
                duration=ctx.get("duration", 0),
                error_message=e["content"],
            ))

        return self.analyze(results)

    def _analyze_file_patterns_sync(self, results: list) -> list[TestInsight]:
        """Find files with recurring failures (sync version)."""
        file_failures = defaultdict(list)

        for result in results:
            # Extract file path from node_id (e.g., "tests/unit/test_foo.py::test_bar")
            if "::" in result.node_id:
                file_path = result.node_id.split("::")[0]
                file_failures[file_path].append(result)

        insights = []
        for file_path, failures in file_failures.items():
            if len(failures) >= self.MIN_PATTERN_OCCURRENCES:
                unique_tests = len(set(f.node_id for f in failures))
                confidence = min(1.0, len(failures) / 5)
                severity = "high" if len(failures) >= 5 else "medium"

                insights.append(TestInsight(
                    type=InsightType.HOTSPOT,
                    title=f"Test file hotspot: {file_path}",
                    description=f"File '{file_path}' has {len(failures)} failures "
                               f"across {unique_tests} unique tests.",
                    evidence=[f.node_id for f in failures],
                    confidence=confidence,
                    severity=severity,
                    actionable=True,
                    suggested_action=f"Review test file {file_path} for systemic issues.",
                    metadata={
                        "file_path": file_path,
                        "failure_count": len(failures),
                        "unique_tests": unique_tests,
                    }
                ))

        return insights

    def _analyze_error_patterns_sync(self, results: list) -> list[TestInsight]:
        """Find recurring error types (sync version)."""
        error_patterns = defaultdict(list)

        for result in results:
            error_msg = result.error_message or ""
            if not error_msg:
                continue

            match = re.search(r"(\w+Error|\w+Exception)", error_msg)
            if match:
                error_type = match.group(1)
                error_patterns[error_type].append(result)

        insights = []
        for error_type, failures in error_patterns.items():
            if len(failures) >= self.MIN_PATTERN_OCCURRENCES:
                confidence = min(1.0, len(failures) / 5)
                suggested_action = self._get_error_action(error_type, failures)
                severity = "critical" if error_type in ("ModuleNotFoundError", "ImportError") else "medium"

                insights.append(TestInsight(
                    type=InsightType.PATTERN,
                    title=f"Recurring error: {error_type}",
                    description=f"'{error_type}' occurred in {len(failures)} test failures.",
                    evidence=[f.node_id for f in failures],
                    confidence=confidence,
                    severity=severity,
                    actionable=True,
                    suggested_action=suggested_action,
                    metadata={
                        "error_type": error_type,
                        "occurrence_count": len(failures),
                    }
                ))

        return insights

    def _analyze_hotspots_sync(self, results: list) -> list[TestInsight]:
        """Find code areas with many failures by module (sync version)."""
        module_failures = defaultdict(list)

        for result in results:
            node_id = result.node_id
            if not node_id or "::" not in node_id:
                continue

            file_path = node_id.split("::")[0]
            parts = file_path.split("/")
            if len(parts) >= 2:
                module = "/".join(parts[:2])
                module_failures[module].append(result)

        insights = []
        for module, failures in module_failures.items():
            if len(failures) >= 3:
                confidence = min(1.0, len(failures) / 10)

                insights.append(TestInsight(
                    type=InsightType.HOTSPOT,
                    title=f"Module failure hotspot: {module}",
                    description=f"Module '{module}' has {len(failures)} test failures.",
                    evidence=[f.node_id for f in failures[:10]],
                    confidence=confidence,
                    severity="high" if len(failures) >= 10 else "medium",
                    actionable=True,
                    suggested_action=f"Audit module {module} for shared fixture issues.",
                    metadata={
                        "module": module,
                        "failure_count": len(failures),
                    }
                ))

        return insights

    def _analyze_flaky_tests_sync(self, results: list) -> list[TestInsight]:
        """Detect potentially flaky tests (sync version)."""
        flaky_indicators = [
            "timeout", "connection reset", "connection refused",
            "temporary", "intermittent", "retry", "race condition",
        ]

        potential_flaky = []
        for result in results:
            error_msg = (result.error_message or "").lower()

            for indicator in flaky_indicators:
                if indicator in error_msg:
                    potential_flaky.append(result)
                    break

        if len(potential_flaky) >= 2:
            return [TestInsight(
                type=InsightType.FLAKY,
                title="Potential flaky tests detected",
                description=f"Found {len(potential_flaky)} failures with flakiness indicators.",
                evidence=[f.node_id for f in potential_flaky[:10]],
                confidence=0.6,
                severity="medium",
                actionable=True,
                suggested_action="Review tests for race conditions, add proper waits.",
                metadata={
                    "flaky_count": len(potential_flaky),
                    "affected_tests": list(set(f.node_id for f in potential_flaky))[:10],
                }
            )]

        return []

    # Alias for backward compatibility
    @property
    def insight_type(self):
        return InsightType

    async def analyze_recent_failures(
        self,
        days: int = 7,
        min_severity: str = "low",
    ) -> list[TestInsight]:
        """
        Analyze test failures from the past N days.

        Args:
            days: Number of days to look back
            min_severity: Minimum severity to include

        Returns:
            List of TestInsight objects
        """
        from kestrel_sovereign.a2a.stores.feedback_store import FeedbackSource, FeedbackStatus

        since = datetime.utcnow() - timedelta(days=days)

        # Query test-related feedback
        entries = await self.store.query_feedback(
            source=FeedbackSource.SYSTEM,
            since=since,
            limit=1000,
        )

        # Filter to test results only
        test_entries = [
            e for e in entries
            if e.metadata.get("source_type") in ("test_result", "test_run_summary")
        ]

        if not test_entries:
            return []

        insights = []

        # Analyze by different dimensions
        insights.extend(await self._analyze_file_patterns(test_entries))
        insights.extend(await self._analyze_error_patterns(test_entries))
        insights.extend(await self._analyze_hotspots(test_entries))
        insights.extend(await self._analyze_regressions(test_entries))
        insights.extend(await self._analyze_flaky_tests(test_entries))

        # Filter by confidence
        insights = [i for i in insights if i.confidence >= self.MIN_CONFIDENCE]

        # Sort by severity and confidence
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        insights.sort(key=lambda x: (severity_order.get(x.severity, 4), -x.confidence))

        return insights

    async def _analyze_file_patterns(self, entries: list) -> list[TestInsight]:
        """Find files with recurring failures."""
        file_failures = defaultdict(list)

        for entry in entries:
            file_path = entry.context.get("file_path")
            if file_path:
                file_failures[file_path].append(entry)

        insights = []
        for file_path, failures in file_failures.items():
            if len(failures) >= self.MIN_PATTERN_OCCURRENCES:
                # Calculate confidence based on failure consistency
                unique_tests = len(set(f.context.get("node_id") for f in failures))
                confidence = min(1.0, len(failures) / 5)  # Max confidence at 5+ failures

                severity = "high" if len(failures) >= 5 else "medium"

                insights.append(TestInsight(
                    type=InsightType.HOTSPOT,
                    title=f"Test file hotspot: {file_path}",
                    description=f"File '{file_path}' has {len(failures)} failures "
                               f"across {unique_tests} unique tests in the analysis period.",
                    evidence=[f.feedback_id for f in failures],
                    confidence=confidence,
                    severity=severity,
                    actionable=True,
                    suggested_action=f"Review test file {file_path} for systemic issues. "
                                    f"Consider refactoring shared fixtures or improving test isolation.",
                    metadata={
                        "file_path": file_path,
                        "failure_count": len(failures),
                        "unique_tests": unique_tests,
                    }
                ))

        return insights

    async def _analyze_error_patterns(self, entries: list) -> list[TestInsight]:
        """Find recurring error types."""
        error_patterns = defaultdict(list)

        for entry in entries:
            error_msg = entry.context.get("error_message", "")
            if not error_msg:
                continue

            # Extract error type (e.g., "ModuleNotFoundError", "AssertionError")
            match = re.search(r"(\w+Error|\w+Exception)", error_msg)
            if match:
                error_type = match.group(1)
                error_patterns[error_type].append(entry)

        insights = []
        for error_type, failures in error_patterns.items():
            if len(failures) >= self.MIN_PATTERN_OCCURRENCES:
                confidence = min(1.0, len(failures) / 5)

                # Suggest action based on error type
                suggested_action = self._get_error_action(error_type, failures)
                severity = "critical" if error_type in ("ModuleNotFoundError", "ImportError") else "medium"

                insights.append(TestInsight(
                    type=InsightType.PATTERN,
                    title=f"Recurring error: {error_type}",
                    description=f"'{error_type}' occurred in {len(failures)} test failures. "
                               f"This suggests a systematic issue that should be addressed.",
                    evidence=[f.feedback_id for f in failures],
                    confidence=confidence,
                    severity=severity,
                    actionable=True,
                    suggested_action=suggested_action,
                    metadata={
                        "error_type": error_type,
                        "occurrence_count": len(failures),
                    }
                ))

        return insights

    def _get_error_action(self, error_type: str, failures: list) -> str:
        """Get suggested action for an error type."""
        actions = {
            "ModuleNotFoundError": "Check for missing dependencies in pyproject.toml or misplaced imports.",
            "ImportError": "Verify module paths and circular import issues.",
            "AssertionError": "Review test assertions - expected values may need updating.",
            "TimeoutError": "Consider increasing timeouts or optimizing slow operations.",
            "ConnectionError": "Ensure test infrastructure (DB, Redis, API) is properly mocked or available.",
            "PermissionError": "Check file permissions and directory access in tests.",
            "ValueError": "Review input validation in tested code.",
            "TypeError": "Check type annotations and function signatures.",
        }
        return actions.get(error_type, f"Investigate {error_type} root cause across {len(failures)} failures.")

    async def _analyze_hotspots(self, entries: list) -> list[TestInsight]:
        """Find code areas with many failures (by module/package)."""
        module_failures = defaultdict(list)

        for entry in entries:
            file_path = entry.context.get("file_path", "")
            if not file_path:
                continue

            # Extract module (e.g., "tests/integration" from "tests/integration/test_foo.py")
            parts = file_path.split("/")
            if len(parts) >= 2:
                module = "/".join(parts[:2])
                module_failures[module].append(entry)

        insights = []
        for module, failures in module_failures.items():
            if len(failures) >= 3:  # Higher threshold for module-level insights
                confidence = min(1.0, len(failures) / 10)

                insights.append(TestInsight(
                    type=InsightType.HOTSPOT,
                    title=f"Module failure hotspot: {module}",
                    description=f"Module '{module}' has {len(failures)} test failures. "
                               f"Consider reviewing shared infrastructure or dependencies.",
                    evidence=[f.feedback_id for f in failures[:10]],  # Limit evidence
                    confidence=confidence,
                    severity="high" if len(failures) >= 10 else "medium",
                    actionable=True,
                    suggested_action=f"Audit module {module} for shared fixture issues, "
                                    f"flaky dependencies, or architectural problems.",
                    metadata={
                        "module": module,
                        "failure_count": len(failures),
                    }
                ))

        return insights

    async def _analyze_regressions(self, entries: list) -> list[TestInsight]:
        """Detect tests that started failing after previously passing."""
        # Group by test (node_id)
        test_history = defaultdict(list)
        for entry in entries:
            node_id = entry.context.get("node_id")
            if node_id:
                test_history[node_id].append(entry)

        insights = []
        for node_id, failures in test_history.items():
            # Sort by date
            sorted_failures = sorted(failures, key=lambda x: x.created_at)

            # Check if this is a new test that never passed
            if len(sorted_failures) >= 2:
                # Multiple failures = potential regression
                first_failure = sorted_failures[0]
                latest_failure = sorted_failures[-1]

                # Check git context for regression detection
                branches = set(f.metadata.get("git_branch") for f in sorted_failures if f.metadata.get("git_branch"))

                if len(branches) > 1:  # Failures across branches = potential regression
                    insights.append(TestInsight(
                        type=InsightType.REGRESSION,
                        title=f"Potential regression: {node_id.split('::')[-1]}",
                        description=f"Test '{node_id}' has failed {len(sorted_failures)} times "
                                   f"across branches: {', '.join(branches)}",
                        evidence=[f.feedback_id for f in sorted_failures],
                        confidence=0.7,
                        severity="high",
                        actionable=True,
                        suggested_action=f"Investigate if recent changes broke this test. "
                                        f"Check commits between first failure and now.",
                        metadata={
                            "node_id": node_id,
                            "failure_count": len(sorted_failures),
                            "branches": list(branches),
                        }
                    ))

        return insights

    async def _analyze_flaky_tests(self, entries: list) -> list[TestInsight]:
        """Detect potentially flaky tests (intermittent failures)."""
        # This requires correlating with pass data, which we don't have
        # For now, detect tests that fail with timing-related errors
        flaky_indicators = [
            "timeout",
            "connection reset",
            "connection refused",
            "temporary",
            "intermittent",
            "retry",
            "race condition",
        ]

        potential_flaky = []
        for entry in entries:
            error_msg = entry.context.get("error_message", "").lower()
            description = entry.description.lower()

            for indicator in flaky_indicators:
                if indicator in error_msg or indicator in description:
                    potential_flaky.append(entry)
                    break

        if len(potential_flaky) >= 2:
            return [TestInsight(
                type=InsightType.FLAKY,
                title="Potential flaky tests detected",
                description=f"Found {len(potential_flaky)} failures with indicators of flakiness "
                           f"(timeouts, connection issues, race conditions).",
                evidence=[f.feedback_id for f in potential_flaky[:10]],
                confidence=0.6,
                severity="medium",
                actionable=True,
                suggested_action="Review tests for race conditions, add proper waits, "
                                "or improve test isolation. Consider adding retry logic.",
                metadata={
                    "flaky_count": len(potential_flaky),
                    "affected_tests": list(set(
                        f.context.get("node_id") for f in potential_flaky
                        if f.context.get("node_id")
                    ))[:10],
                }
            )]

        return []

    async def get_failure_summary(
        self,
        days: int = 7,
    ) -> dict[str, Any]:
        """
        Get a summary of recent test failures.

        Args:
            days: Number of days to look back

        Returns:
            Summary dictionary with counts and trends
        """
        from kestrel_sovereign.a2a.stores.feedback_store import FeedbackSource

        since = datetime.utcnow() - timedelta(days=days)

        entries = await self.store.query_feedback(
            source=FeedbackSource.SYSTEM,
            since=since,
            limit=1000,
        )

        test_entries = [
            e for e in entries
            if e.metadata.get("source_type") == "test_result"
        ]

        summary_entries = [
            e for e in entries
            if e.metadata.get("source_type") == "test_run_summary"
        ]

        # Calculate metrics
        total_runs = len(summary_entries)
        total_failures = len(test_entries)

        # Group by date for trend analysis
        daily_failures = defaultdict(int)
        for entry in test_entries:
            date_key = entry.created_at.strftime("%Y-%m-%d")
            daily_failures[date_key] += 1

        # Most failing tests
        test_failure_counts = defaultdict(int)
        for entry in test_entries:
            node_id = entry.context.get("node_id")
            if node_id:
                test_failure_counts[node_id] += 1

        top_failing = sorted(
            test_failure_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return {
            "period_days": days,
            "total_test_runs": total_runs,
            "total_failures": total_failures,
            "daily_failure_trend": dict(daily_failures),
            "top_failing_tests": [
                {"test": t, "failures": c} for t, c in top_failing
            ],
            "analyzed_at": datetime.utcnow().isoformat(),
        }
