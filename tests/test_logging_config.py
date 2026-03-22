"""Tests for kestrel_sovereign.logging_config — structured JSON logging."""

import json
import logging
import os
import uuid
from unittest import mock

import pytest

from kestrel_sovereign.logging_config import (
    JSONFormatter,
    agent_name_var,
    correlation_id_var,
    feature_name_var,
    get_correlation_id,
    session_id_var,
    setup_logging,
    tool_name_var,
)


# ---------------------------------------------------------------------------
# JSONFormatter
# ---------------------------------------------------------------------------


class TestJSONFormatter:
    """Test the JSONFormatter produces valid JSON with required fields."""

    def setup_method(self):
        self.formatter = JSONFormatter()

    def _make_record(self, msg="test message", level=logging.INFO, name="test.logger"):
        record = logging.LogRecord(
            name=name,
            level=level,
            pathname="test.py",
            lineno=1,
            msg=msg,
            args=(),
            exc_info=None,
        )
        return record

    def test_output_is_valid_json(self):
        record = self._make_record()
        output = self.formatter.format(record)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_required_fields_present(self):
        record = self._make_record()
        parsed = json.loads(self.formatter.format(record))
        assert "timestamp" in parsed
        assert "level" in parsed
        assert "logger" in parsed
        assert "message" in parsed
        assert "correlation_id" in parsed

    def test_level_and_logger(self):
        record = self._make_record(level=logging.WARNING, name="my.module")
        parsed = json.loads(self.formatter.format(record))
        assert parsed["level"] == "WARNING"
        assert parsed["logger"] == "my.module"

    def test_message_content(self):
        record = self._make_record(msg="hello world")
        parsed = json.loads(self.formatter.format(record))
        assert parsed["message"] == "hello world"

    def test_context_vars_included_when_set(self):
        tokens = []
        tokens.append(session_id_var.set("sess-123"))
        tokens.append(agent_name_var.set("TestAgent"))
        tokens.append(tool_name_var.set("search"))
        tokens.append(feature_name_var.set("memory"))
        try:
            record = self._make_record()
            parsed = json.loads(self.formatter.format(record))
            assert parsed["session_id"] == "sess-123"
            assert parsed["agent_name"] == "TestAgent"
            assert parsed["tool_name"] == "search"
            assert parsed["feature_name"] == "memory"
        finally:
            for tok in tokens:
                # Reset in reverse order
                pass
            session_id_var.set(None)
            agent_name_var.set(None)
            tool_name_var.set(None)
            feature_name_var.set(None)

    def test_context_vars_omitted_when_not_set(self):
        # Ensure vars are None
        session_id_var.set(None)
        agent_name_var.set(None)
        tool_name_var.set(None)
        feature_name_var.set(None)

        record = self._make_record()
        parsed = json.loads(self.formatter.format(record))
        assert "session_id" not in parsed
        assert "agent_name" not in parsed
        assert "tool_name" not in parsed
        assert "feature_name" not in parsed

    def test_exception_info_included(self):
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = self._make_record()
        record.exc_info = exc_info
        parsed = json.loads(self.formatter.format(record))
        assert "exception" in parsed
        assert "ValueError: boom" in parsed["exception"]

    def test_timestamp_is_iso_format(self):
        record = self._make_record()
        parsed = json.loads(self.formatter.format(record))
        # Should contain 'T' separator and '+00:00' timezone
        assert "T" in parsed["timestamp"]

    def test_single_line_output(self):
        record = self._make_record(msg="line one\nline two")
        output = self.formatter.format(record)
        # JSON output itself should be a single line (newlines are escaped)
        assert "\n" not in output


# ---------------------------------------------------------------------------
# get_correlation_id
# ---------------------------------------------------------------------------


class TestCorrelationId:
    """Test correlation ID generation and retrieval."""

    def test_generates_id_when_not_set(self):
        correlation_id_var.set(None)
        cid = get_correlation_id()
        assert cid is not None
        assert len(cid) == 16  # uuid4().hex[:16]

    def test_returns_existing_id(self):
        correlation_id_var.set("existing-id")
        try:
            assert get_correlation_id() == "existing-id"
        finally:
            correlation_id_var.set(None)


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


class TestSetupLogging:
    """Test the setup_logging configuration function."""

    def teardown_method(self):
        # Restore root logger to a clean state
        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(logging.WARNING)

    def test_default_is_text_format(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("KESTREL_LOG_FORMAT", None)
            setup_logging()
        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert not isinstance(root.handlers[0].formatter, JSONFormatter)

    def test_json_format_via_arg(self):
        setup_logging(fmt="json")
        root = logging.getLogger()
        assert isinstance(root.handlers[0].formatter, JSONFormatter)

    def test_json_format_via_env(self):
        with mock.patch.dict(os.environ, {"KESTREL_LOG_FORMAT": "json"}):
            setup_logging()
        root = logging.getLogger()
        assert isinstance(root.handlers[0].formatter, JSONFormatter)

    def test_log_level_via_arg(self):
        setup_logging(level="DEBUG")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_log_level_via_env(self):
        with mock.patch.dict(os.environ, {"KESTREL_LOG_LEVEL": "WARNING"}):
            setup_logging()
        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_clears_existing_handlers(self):
        root = logging.getLogger()
        before = len(root.handlers)
        root.addHandler(logging.StreamHandler())
        root.addHandler(logging.StreamHandler())
        assert len(root.handlers) == before + 2

        setup_logging()
        # setup_logging clears all handlers and adds exactly one
        assert len(root.handlers) == 1

    def test_text_format_output(self, capsys):
        """Text format should produce traditional level:name:message output."""
        setup_logging(fmt="text", level="INFO")
        test_logger = logging.getLogger("test.text")
        test_logger.info("hello text")
        captured = capsys.readouterr()
        assert "INFO" in captured.err
        assert "hello text" in captured.err

    def test_json_format_output(self, capsys):
        """JSON format should produce parseable JSON on stderr."""
        setup_logging(fmt="json", level="INFO")
        # Reset correlation_id so it's predictable
        correlation_id_var.set(None)
        test_logger = logging.getLogger("test.json")
        test_logger.info("hello json")
        captured = capsys.readouterr()
        parsed = json.loads(captured.err.strip())
        assert parsed["message"] == "hello json"
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.json"

    def test_existing_log_calls_work_with_json(self, capsys):
        """Existing logger.info("text") calls should produce valid JSON without changes."""
        setup_logging(fmt="json", level="INFO")
        correlation_id_var.set(None)
        logger = logging.getLogger("kestrel_sovereign.server")
        logger.info("Server starting up...")
        captured = capsys.readouterr()
        parsed = json.loads(captured.err.strip())
        assert parsed["message"] == "Server starting up..."
