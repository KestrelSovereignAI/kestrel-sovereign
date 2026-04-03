"""Tests for kestrel_sovereign.telemetry module.

Verifies:
- Graceful degradation when OTEL packages are not installed
- is_tracing_enabled() logic with various env var combinations
- optional_span() context manager works as no-op without OTEL
- setup_tracing() returns False when disabled
"""
import os
import pytest
from unittest.mock import patch, MagicMock


class TestIsTracingEnabled:
    """Test the is_tracing_enabled() function."""

    def test_disabled_when_otel_not_available(self):
        """Tracing is disabled when OTEL packages are not installed."""
        from kestrel_sovereign import telemetry
        with patch.object(telemetry, '_OTEL_AVAILABLE', False):
            assert telemetry.is_tracing_enabled() is False

    def test_disabled_when_explicitly_off(self):
        """KESTREL_TRACING_ENABLED=false disables tracing."""
        from kestrel_sovereign import telemetry
        with patch.object(telemetry, '_OTEL_AVAILABLE', True):
            with patch.dict(os.environ, {"KESTREL_TRACING_ENABLED": "false"}):
                assert telemetry.is_tracing_enabled() is False

    def test_disabled_when_explicitly_zero(self):
        """KESTREL_TRACING_ENABLED=0 disables tracing."""
        from kestrel_sovereign import telemetry
        with patch.object(telemetry, '_OTEL_AVAILABLE', True):
            with patch.dict(os.environ, {"KESTREL_TRACING_ENABLED": "0"}):
                assert telemetry.is_tracing_enabled() is False

    def test_enabled_when_explicitly_on(self):
        """KESTREL_TRACING_ENABLED=true enables tracing."""
        from kestrel_sovereign import telemetry
        with patch.object(telemetry, '_OTEL_AVAILABLE', True):
            with patch.dict(os.environ, {"KESTREL_TRACING_ENABLED": "true"}):
                assert telemetry.is_tracing_enabled() is True

    def test_auto_detect_with_endpoint(self):
        """Auto-detect enables tracing when OTLP endpoint is set."""
        from kestrel_sovereign import telemetry
        with patch.object(telemetry, '_OTEL_AVAILABLE', True):
            with patch.dict(os.environ, {
                "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317",
            }, clear=False):
                # Remove KESTREL_TRACING_ENABLED to trigger auto-detect
                env = os.environ.copy()
                env.pop("KESTREL_TRACING_ENABLED", None)
                with patch.dict(os.environ, env, clear=True):
                    assert telemetry.is_tracing_enabled() is True

    def test_auto_detect_without_endpoint(self):
        """Auto-detect disables tracing when no OTLP endpoint is set."""
        from kestrel_sovereign import telemetry
        with patch.object(telemetry, '_OTEL_AVAILABLE', True):
            env = os.environ.copy()
            env.pop("KESTREL_TRACING_ENABLED", None)
            env.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
            with patch.dict(os.environ, env, clear=True):
                assert telemetry.is_tracing_enabled() is False


class TestOptionalSpan:
    """Test the optional_span() context manager."""

    def test_yields_none_when_tracing_disabled(self):
        """optional_span yields None when no tracer is configured."""
        from kestrel_sovereign.telemetry import optional_span
        with patch('kestrel_sovereign.telemetry.get_tracer', return_value=None):
            with optional_span("test.span") as span:
                assert span is None

    def test_no_exception_when_tracing_disabled(self):
        """optional_span does not raise when tracing is disabled."""
        from kestrel_sovereign.telemetry import optional_span
        with patch('kestrel_sovereign.telemetry.get_tracer', return_value=None):
            with optional_span("test.span", {"key": "value"}) as span:
                assert span is None
                # Do some work
                result = 1 + 1
            assert result == 2

    def test_exception_propagates_through_span(self):
        """Exceptions inside optional_span propagate correctly."""
        from kestrel_sovereign.telemetry import optional_span
        with patch('kestrel_sovereign.telemetry.get_tracer', return_value=None):
            with pytest.raises(ValueError, match="test error"):
                with optional_span("test.span") as span:
                    raise ValueError("test error")


class TestSetupTracing:
    """Test the setup_tracing() function."""

    def test_returns_false_when_otel_not_available(self):
        """setup_tracing returns False when OTEL packages are missing."""
        from kestrel_sovereign import telemetry
        with patch.object(telemetry, '_OTEL_AVAILABLE', False):
            result = telemetry.setup_tracing()
            assert result is False

    def test_returns_false_when_tracing_disabled(self):
        """setup_tracing returns False when tracing is explicitly disabled."""
        from kestrel_sovereign import telemetry
        with patch.object(telemetry, '_OTEL_AVAILABLE', True):
            with patch.dict(os.environ, {"KESTREL_TRACING_ENABLED": "false"}):
                result = telemetry.setup_tracing()
                assert result is False


class TestStartEndSpan:
    """Test start_span() and end_span() helpers."""

    def test_start_span_returns_none_when_disabled(self):
        """start_span returns None when tracing is disabled."""
        from kestrel_sovereign.telemetry import start_span
        with patch('kestrel_sovereign.telemetry.get_tracer', return_value=None):
            span = start_span("test.span")
            assert span is None

    def test_end_span_noop_with_none(self):
        """end_span does nothing when given None."""
        from kestrel_sovereign.telemetry import end_span
        # Should not raise
        end_span(None)
        end_span(None, error=ValueError("test"))


class TestGetTracer:
    """Test get_tracer() function."""

    def test_returns_none_by_default(self):
        """get_tracer returns None when tracing has not been set up."""
        from kestrel_sovereign import telemetry
        original = telemetry._tracer
        try:
            telemetry._tracer = None
            assert telemetry.get_tracer() is None
        finally:
            telemetry._tracer = original


class TestGracefulDegradation:
    """Test that tracing-instrumented code works without OTEL packages."""

    def test_import_telemetry_module_succeeds(self):
        """The telemetry module imports successfully regardless of OTEL availability."""
        import importlib
        from kestrel_sovereign import telemetry
        # Module should be importable
        assert hasattr(telemetry, 'setup_tracing')
        assert hasattr(telemetry, 'optional_span')
        assert hasattr(telemetry, 'get_tracer')

    def test_optional_span_in_sync_code(self):
        """optional_span works correctly in synchronous code paths."""
        from kestrel_sovereign.telemetry import optional_span
        with patch('kestrel_sovereign.telemetry.get_tracer', return_value=None):
            results = []
            with optional_span("test.operation") as span:
                results.append("before")
                if span:
                    span.set_attribute("key", "value")
                results.append("after")
            assert results == ["before", "after"]

    @pytest.mark.asyncio
    async def test_optional_span_in_async_code(self):
        """optional_span works correctly in async code paths."""
        import asyncio
        from kestrel_sovereign.telemetry import optional_span
        with patch('kestrel_sovereign.telemetry.get_tracer', return_value=None):
            with optional_span("test.async_operation") as span:
                assert span is None
                await asyncio.sleep(0)  # Yield to event loop
