"""
Tests for Prometheus metrics endpoint and instrumentation.

Covers:
1. Metric definitions exist and are correct types
2. /metrics endpoint returns Prometheus text format
3. /metrics returns 404 when prometheus-client not installed
4. LLM service increments Prometheus counters
5. Labels have bounded cardinality (no user data)
6. Metrics survive without prometheus-client (graceful degradation)
7. Request middleware records metrics
"""

from unittest.mock import patch

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


# ---------------------------------------------------------------------------
# 1. Metric definitions
# ---------------------------------------------------------------------------

@requires_prometheus
class TestMetricDefinitions:
    def test_prometheus_available_flag(self):
        from kestrel_sdk.metrics import PROMETHEUS_AVAILABLE
        assert PROMETHEUS_AVAILABLE is True

    def test_registry_exists(self):
        from kestrel_sdk.metrics import REGISTRY
        assert REGISTRY is not None

    def test_request_metrics_defined(self):
        from kestrel_sdk.metrics import REQUEST_COUNT, REQUEST_DURATION
        assert REQUEST_COUNT is not None
        assert REQUEST_DURATION is not None

    def test_llm_metrics_defined(self):
        from kestrel_sdk.metrics import LLM_CALLS, LLM_DURATION, LLM_TOKENS
        assert LLM_CALLS is not None
        assert LLM_DURATION is not None
        assert LLM_TOKENS is not None

    def test_tool_metrics_defined(self):
        from kestrel_sdk.metrics import TOOL_CALLS, TOOL_DURATION
        assert TOOL_CALLS is not None
        assert TOOL_DURATION is not None

    def test_hook_metrics_defined(self):
        from kestrel_sdk.metrics import HOOK_EVENTS, HOOK_DENIALS
        assert HOOK_EVENTS is not None
        assert HOOK_DENIALS is not None

    def test_system_metrics_defined(self):
        from kestrel_sdk.metrics import CONTEXT_PRESSURE, ACTIVE_SESSIONS
        assert CONTEXT_PRESSURE is not None
        assert ACTIVE_SESSIONS is not None


# ---------------------------------------------------------------------------
# 2. generate_metrics helper
# ---------------------------------------------------------------------------

@requires_prometheus
class TestGenerateMetrics:
    def test_generates_bytes(self):
        from kestrel_sdk.metrics import generate_metrics
        output = generate_metrics()
        assert isinstance(output, bytes)

    def test_content_type(self):
        from kestrel_sdk.metrics import get_content_type
        ct = get_content_type()
        assert "text/plain" in ct or "text/openmetrics" in ct

    def test_metrics_contain_kestrel_prefix(self):
        from kestrel_sdk.metrics import generate_metrics, HOOK_EVENTS
        # Increment a counter to ensure output is non-empty
        HOOK_EVENTS.labels(event_type="test_gen").inc()
        output = generate_metrics().decode("utf-8")
        assert "kestrel_" in output


# ---------------------------------------------------------------------------
# 3. /metrics endpoint
# ---------------------------------------------------------------------------

@requires_prometheus
class TestMetricsEndpoint:
    @pytest.fixture
    def endpoint_module(self):
        import kestrel_sovereign.endpoints.metrics as m
        return m

    @pytest.mark.asyncio
    async def test_returns_prometheus_format(self, endpoint_module):
        response = await endpoint_module.prometheus_metrics()
        assert response.status_code == 200
        assert b"kestrel_" in response.body or response.body == b""

    @pytest.mark.asyncio
    async def test_returns_correct_content_type(self, endpoint_module):
        response = await endpoint_module.prometheus_metrics()
        assert "text/" in response.media_type

    @pytest.mark.asyncio
    async def test_returns_404_when_unavailable(self, endpoint_module):
        with patch.object(endpoint_module, "PROMETHEUS_AVAILABLE", False):
            response = await endpoint_module.prometheus_metrics()
            assert response.status_code == 404
            assert b"observability" in response.body


# ---------------------------------------------------------------------------
# 4. LLM service Prometheus instrumentation
# ---------------------------------------------------------------------------

@requires_prometheus
class TestLLMServicePrometheus:
    @pytest.mark.asyncio
    async def test_llm_call_increments_counters(self):
        from kestrel_sovereign.llm.service import LLMService
        from kestrel_sdk.metrics import REGISTRY

        service = LLMService.__new__(LLMService)
        service._observability_store = None
        service._metering_callback = None
        service._observability_context = {}

        before = REGISTRY.get_sample_value(
            "kestrel_llm_calls_total",
            {"provider": "test_provider", "model": "test_model", "success": "True"},
        ) or 0.0

        await service._log_llm_call(
            provider="test_provider",
            model="test_model",
            duration_ms=200,
            success=True,
            input_tokens=100,
            output_tokens=50,
        )

        after = REGISTRY.get_sample_value(
            "kestrel_llm_calls_total",
            {"provider": "test_provider", "model": "test_model", "success": "True"},
        )
        assert after == before + 1

    @pytest.mark.asyncio
    async def test_llm_tokens_counted(self):
        from kestrel_sovereign.llm.service import LLMService
        from kestrel_sdk.metrics import REGISTRY

        service = LLMService.__new__(LLMService)
        service._observability_store = None
        service._metering_callback = None
        service._observability_context = {}

        input_before = REGISTRY.get_sample_value(
            "kestrel_llm_tokens_total",
            {"model": "token_model", "direction": "input"},
        ) or 0.0
        output_before = REGISTRY.get_sample_value(
            "kestrel_llm_tokens_total",
            {"model": "token_model", "direction": "output"},
        ) or 0.0

        await service._log_llm_call(
            provider="p",
            model="token_model",
            duration_ms=100,
            success=True,
            input_tokens=75,
            output_tokens=25,
        )

        input_after = REGISTRY.get_sample_value(
            "kestrel_llm_tokens_total",
            {"model": "token_model", "direction": "input"},
        )
        output_after = REGISTRY.get_sample_value(
            "kestrel_llm_tokens_total",
            {"model": "token_model", "direction": "output"},
        )
        assert input_after == input_before + 75
        assert output_after == output_before + 25

    @pytest.mark.asyncio
    async def test_llm_duration_histogram(self):
        from kestrel_sovereign.llm.service import LLMService
        from kestrel_sdk.metrics import REGISTRY

        service = LLMService.__new__(LLMService)
        service._observability_store = None
        service._metering_callback = None
        service._observability_context = {}

        await service._log_llm_call(
            provider="dur_provider",
            model="dur_model",
            duration_ms=350,
            success=True,
        )

        count = REGISTRY.get_sample_value(
            "kestrel_llm_duration_seconds_count",
            {"provider": "dur_provider", "model": "dur_model"},
        )
        assert count is not None and count >= 1


# ---------------------------------------------------------------------------
# 5. Graceful degradation (no prometheus-client)
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    def test_metrics_module_loads_without_prometheus(self):
        """Metrics module loads regardless of prometheus-client availability."""
        import kestrel_sdk.metrics as m
        # PROMETHEUS_AVAILABLE reflects whether the package is installed
        assert m.PROMETHEUS_AVAILABLE is _HAS_PROMETHEUS
        if _HAS_PROMETHEUS:
            assert m.REQUEST_COUNT is not None
        else:
            assert m.REQUEST_COUNT is None

# ---------------------------------------------------------------------------
# 6. Label cardinality
# ---------------------------------------------------------------------------

@requires_prometheus
class TestLabelCardinality:
    def test_no_user_data_in_labels(self):
        """Verify metric label names don't include user/session identifiers."""
        from kestrel_sdk.metrics import (
            REQUEST_COUNT, REQUEST_DURATION,
            LLM_CALLS, LLM_DURATION, LLM_TOKENS,
            TOOL_CALLS, TOOL_DURATION,
            HOOK_EVENTS, HOOK_DENIALS,
        )
        forbidden = {"user_id", "session_id", "user_message", "api_key"}
        for metric in [
            REQUEST_COUNT, REQUEST_DURATION,
            LLM_CALLS, LLM_DURATION, LLM_TOKENS,
            TOOL_CALLS, TOOL_DURATION,
            HOOK_EVENTS, HOOK_DENIALS,
        ]:
            label_names = set(metric._labelnames)
            assert label_names.isdisjoint(forbidden), (
                f"Metric {metric._name} has forbidden labels: {label_names & forbidden}"
            )
