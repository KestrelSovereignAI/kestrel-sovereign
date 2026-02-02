"""Test utilities for Kestrel test suite."""

from .async_waits import (
    wait_until,
    wait_for_value,
    wait_for_http_ready,
    wait_for_db_connection,
    WaitTimeoutError
)

from .parallel_support import (
    get_worker_id,
    unique_email,
    unique_name,
    worker_cleanup_pattern,
    redis_key,
    TestDataFactory,
)

from .feedback_bridge import (
    TestOutcome,
    TestResult,
    TestRunSummary,
    TestResultCollector,
    submit_test_feedback,
)

from .test_result_analyzer import (
    InsightType,
    TestInsight,
    TestResultAnalyzer,
)

__all__ = [
    # Async waits
    'wait_until',
    'wait_for_value',
    'wait_for_http_ready',
    'wait_for_db_connection',
    'WaitTimeoutError',
    # Parallel support
    'get_worker_id',
    'unique_email',
    'unique_name',
    'worker_cleanup_pattern',
    'redis_key',
    'TestDataFactory',
    # Feedback bridge
    'TestOutcome',
    'TestResult',
    'TestRunSummary',
    'TestResultCollector',
    'submit_test_feedback',
    # Test result analyzer
    'InsightType',
    'TestInsight',
    'TestResultAnalyzer',
]
