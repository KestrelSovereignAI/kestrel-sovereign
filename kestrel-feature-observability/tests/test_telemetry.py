"""
Tests for kestrel_feature_observability.telemetry module.

Verifies tracing enable/disable logic and graceful degradation.
"""

import os
import pytest
from unittest.mock import patch


class TestIsTracingEnabled:
    def test_disabled_when_otel_not_available(self):
        from kestrel_feature_observability import telemetry
        with patch.object(telemetry, '_OTEL_AVAILABLE', False):
            assert telemetry.is_tracing_enabled() is False

    def test_disabled_when_explicitly_off(self):
        from kestrel_feature_observability import telemetry
        with patch.object(telemetry, '_OTEL_AVAILABLE', True):
            with patch.dict(os.environ, {"KESTREL_TRACING_ENABLED": "false"}):
                assert telemetry.is_tracing_enabled() is False

    def test_enabled_when_explicitly_on(self):
        from kestrel_feature_observability import telemetry
        with patch.object(telemetry, '_OTEL_AVAILABLE', True):
            with patch.dict(os.environ, {"KESTREL_TRACING_ENABLED": "true"}):
                assert telemetry.is_tracing_enabled() is True


class TestOptionalSpan:
    def test_yields_none_when_tracing_disabled(self):
        from kestrel_feature_observability.telemetry import optional_span
        with patch('kestrel_feature_observability.telemetry.get_tracer', return_value=None):
            with optional_span("test.span") as span:
                assert span is None

    def test_exception_propagates_through_span(self):
        from kestrel_feature_observability.telemetry import optional_span
        with patch('kestrel_feature_observability.telemetry.get_tracer', return_value=None):
            with pytest.raises(ValueError, match="test error"):
                with optional_span("test.span") as span:
                    raise ValueError("test error")


class TestSetupTracing:
    def test_returns_false_when_otel_not_available(self):
        from kestrel_feature_observability import telemetry
        with patch.object(telemetry, '_OTEL_AVAILABLE', False):
            assert telemetry.setup_tracing() is False


class TestStartEndSpan:
    def test_start_span_returns_none_when_disabled(self):
        from kestrel_feature_observability.telemetry import start_span
        with patch('kestrel_feature_observability.telemetry.get_tracer', return_value=None):
            assert start_span("test.span") is None

    def test_end_span_noop_with_none(self):
        from kestrel_feature_observability.telemetry import end_span
        end_span(None)
        end_span(None, error=ValueError("test"))
