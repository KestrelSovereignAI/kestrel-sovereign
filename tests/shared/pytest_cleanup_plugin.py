"""
Pytest plugin for resource cleanup hooks.

This plugin ensures resources are cleaned up:
1. After each test (via fixture teardown)
2. After test session (via pytest_sessionfinish)
3. On keyboard interrupt (via pytest_keyboard_interrupt)
4. On crash (via atexit/signal handlers in resource_registry)

Usage:
    In conftest.py:
        from tests.shared.pytest_cleanup_plugin import *
"""

import pytest
from .resource_registry import registry
from .cost_tracker import cost_tracker


def pytest_configure(config):
    """Called after command line options are parsed."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "cloud_resource: marks tests as using cloud resources (RunPod, etc)"
    )
    config.addinivalue_line(
        "markers", "expensive: marks tests that may incur costs"
    )


def pytest_sessionstart(session):
    """Called before test collection."""
    # Initialize registry (triggers crash recovery if needed)
    registry._initialize()
    print("\n[TEST SESSION] Resource cleanup enabled")


def pytest_sessionfinish(session, exitstatus):
    """Called after all tests complete."""
    # Clean up any remaining resources
    registry.cleanup_all()

    # Print cost report if any cloud resources were used
    if cost_tracker.total_cost > 0:
        print(cost_tracker.report())
        cost_tracker.save_report()


def pytest_keyboard_interrupt(excinfo):
    """Called on keyboard interrupt (Ctrl+C)."""
    print("\n[INTERRUPT] Keyboard interrupt - cleaning up resources...")
    registry.cleanup_all()


def pytest_collection_modifyitems(config, items):
    """Skip cloud_resource tests unless --run-cloud is provided."""
    if config.getoption("--run-cloud"):
        # --run-cloud given: don't skip cloud tests
        return

    skip_cloud = pytest.mark.skip(reason="need --run-cloud option to run")
    for item in items:
        if "cloud_resource" in item.keywords:
            item.add_marker(skip_cloud)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    """Wrap test setup."""
    yield
    # Resources registered during setup will be tracked


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item):
    """Wrap test teardown."""
    yield
    # Individual test cleanup happens here via fixtures


def pytest_exception_interact(node, call, report):
    """Called when test raises exception (for debugging)."""
    if call.excinfo is not None:
        # Log the exception for debugging
        pass  # Could log to file or metric system


# Fixture for tests that need to track resources
@pytest.fixture
def resource_tracker():
    """
    Fixture providing access to the resource registry.

    Usage:
        def test_something(resource_tracker):
            key = resource_tracker.track_docker(container_id)
            try:
                # test code
            finally:
                resource_tracker.untrack(key)
    """
    from . import resource_registry as rr
    return rr


@pytest.fixture
def cost_tracking():
    """
    Fixture providing access to cost tracking.

    Usage:
        def test_something(cost_tracking):
            with cost_tracking.track_cost("runpod", pod_id, 0.50):
                # test code using the resource
    """
    from . import cost_tracker as ct
    return ct
