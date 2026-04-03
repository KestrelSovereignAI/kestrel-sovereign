"""
Tests for kestrel_feature_observability.metrics module.

Verifies metric definitions, generation, and graceful degradation.
"""

import pytest

try:
    import prometheus_client  # noqa: F401
    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False

requires_prometheus = pytest.mark.skipif(
    not _HAS_PROMETHEUS,
    reason="prometheus-client not installed (optional dependency)",
)


@requires_prometheus
class TestMetricDefinitions:
    def test_prometheus_available_flag(self):
        from kestrel_feature_observability.metrics import PROMETHEUS_AVAILABLE
        assert PROMETHEUS_AVAILABLE is True

    def test_registry_exists(self):
        from kestrel_feature_observability.metrics import REGISTRY
        assert REGISTRY is not None

    def test_request_metrics_defined(self):
        from kestrel_feature_observability.metrics import REQUEST_COUNT, REQUEST_DURATION
        assert REQUEST_COUNT is not None
        assert REQUEST_DURATION is not None

    def test_llm_metrics_defined(self):
        from kestrel_feature_observability.metrics import LLM_CALLS, LLM_DURATION, LLM_TOKENS
        assert LLM_CALLS is not None
        assert LLM_DURATION is not None
        assert LLM_TOKENS is not None

    def test_tool_metrics_defined(self):
        from kestrel_feature_observability.metrics import TOOL_CALLS, TOOL_DURATION
        assert TOOL_CALLS is not None
        assert TOOL_DURATION is not None

    def test_hook_metrics_defined(self):
        from kestrel_feature_observability.metrics import HOOK_EVENTS, HOOK_DENIALS
        assert HOOK_EVENTS is not None
        assert HOOK_DENIALS is not None

    def test_system_metrics_defined(self):
        from kestrel_feature_observability.metrics import CONTEXT_PRESSURE, ACTIVE_SESSIONS
        assert CONTEXT_PRESSURE is not None
        assert ACTIVE_SESSIONS is not None


@requires_prometheus
class TestGenerateMetrics:
    def test_generates_bytes(self):
        from kestrel_feature_observability.metrics import generate_metrics
        output = generate_metrics()
        assert isinstance(output, bytes)

    def test_content_type(self):
        from kestrel_feature_observability.metrics import get_content_type
        ct = get_content_type()
        assert "text/plain" in ct or "text/openmetrics" in ct

    def test_metrics_contain_kestrel_prefix(self):
        from kestrel_feature_observability.metrics import generate_metrics, HOOK_EVENTS
        HOOK_EVENTS.labels(event_type="test_pkg_gen").inc()
        output = generate_metrics().decode("utf-8")
        assert "kestrel_" in output


class TestGracefulDegradation:
    def test_metrics_module_loads(self):
        from kestrel_feature_observability.metrics import PROMETHEUS_AVAILABLE
        assert PROMETHEUS_AVAILABLE is _HAS_PROMETHEUS
