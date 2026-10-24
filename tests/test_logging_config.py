"""Tests for kestrel_sovereign.logging_config — structured JSON logging."""

import asyncio
import contextlib
import json
import logging
import os
import re
from unittest import mock

import pytest

from kestrel_sovereign.logging_config import (
    JSONFormatter,
    agent_name_var,
    correlation_id_var,
    feature_name_var,
    get_correlation_id,
    normalize_correlation_id,
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

    @pytest.mark.parametrize(
        "value",
        [None, "", "contains spaces", "<script>", "a" * 129],
    )
    def test_rejects_unsafe_external_ids(self, value):
        assert normalize_correlation_id(value) is None

    def test_trims_support_safe_external_id(self):
        assert normalize_correlation_id("  req_01J.safe:123  ") == "req_01J.safe:123"

    def test_replaces_unsafe_ambient_id(self):
        correlation_id_var.set("<unsafe>")
        try:
            cid = get_correlation_id()
            assert cid != "<unsafe>"
            assert normalize_correlation_id(cid) == cid
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

    def test_text_format_includes_request_correlation_id(self, capsys):
        setup_logging(fmt="text", level="INFO")
        token = correlation_id_var.set("text-support-2651")
        try:
            logging.getLogger("test.text.correlation").error("request failed")
        finally:
            correlation_id_var.reset(token)

        captured = capsys.readouterr()
        assert "correlation_id=text-support-2651" in captured.err
        assert "request failed" in captured.err

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

    def test_json_format_includes_request_correlation_id(self, capsys):
        setup_logging(fmt="json", level="INFO")
        token = correlation_id_var.set("json-support-2651")
        try:
            logging.getLogger("test.json.correlation").error("request failed")
        finally:
            correlation_id_var.reset(token)

        parsed = json.loads(capsys.readouterr().err.strip())
        assert parsed["correlation_id"] == "json-support-2651"
        assert parsed["message"] == "request failed"

    def test_existing_log_calls_work_with_json(self, capsys):
        """Existing logger.info("text") calls should produce valid JSON without changes."""
        setup_logging(fmt="json", level="INFO")
        correlation_id_var.set(None)
        logger = logging.getLogger("kestrel_sovereign.server")
        logger.info("Server starting up...")
        captured = capsys.readouterr()
        parsed = json.loads(captured.err.strip())
        assert parsed["message"] == "Server starting up..."


# ---------------------------------------------------------------------------
# Timestamps and uvicorn's own loggers (#3781)
# ---------------------------------------------------------------------------

_TIMESTAMP = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}"


async def _asgi_app(scope, receive, send):  # pragma: no cover - never served
    raise AssertionError("the app is never called")


@contextlib.contextmanager
def _uvicorn_configured(**config_kwargs):
    """Configure logging exactly as uvicorn's CLI does before importing the app.

    ``uvicorn.Config`` applies uvicorn's default logging config in its
    constructor: handlers and formatters of its own on ``uvicorn`` and
    ``uvicorn.access``, with ``propagate=False``. ``access_log=False``
    (``--no-access-log``) then strips ``uvicorn.access`` of its handlers. The
    loggers are restored afterwards so no other test sees them changed.
    """
    import uvicorn

    from kestrel_sovereign.logging_config import UVICORN_LOGGERS

    saved = {}
    for name in UVICORN_LOGGERS:
        lg = logging.getLogger(name)
        saved[name] = (list(lg.handlers), lg.propagate, lg.level)
    try:
        config = uvicorn.Config(
            _asgi_app, http="h11", ws="none", **config_kwargs
        )
        assert logging.getLogger("uvicorn").handlers, "uvicorn configured nothing"
        assert logging.getLogger("uvicorn.access").propagate is False
        yield config
    finally:
        for name, (handlers, propagate, level) in saved.items():
            lg = logging.getLogger(name)
            lg.handlers[:] = handlers
            lg.propagate = propagate
            lg.setLevel(level)


def _logs_requests(config) -> bool:
    """Whether uvicorn's HTTP protocol, built now, would log each request.

    The protocol decides once per connection, from
    ``uvicorn.access.hasHandlers()``, so this asks the real class rather than
    restating its rule.
    """
    from uvicorn.protocols.http.h11_impl import H11Protocol
    from uvicorn.server import ServerState

    loop = asyncio.new_event_loop()
    try:
        protocol = H11Protocol(
            config=config, server_state=ServerState(), app_state={}, _loop=loop
        )
        return protocol.access_log
    finally:
        loop.close()


@pytest.fixture
def uvicorn_logging_configured():
    with _uvicorn_configured():
        yield


class TestHostLogLines:
    """Every host log line carries a timestamp, uvicorn's included."""

    def teardown_method(self):
        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(logging.WARNING)

    def test_text_format_starts_with_a_timestamp(self, capsys):
        setup_logging(fmt="text", level="INFO")
        logging.getLogger("test.timestamp").info("phase reached")

        line = capsys.readouterr().err.strip()
        assert re.match(rf"^{_TIMESTAMP} INFO:test\.timestamp:", line), line
        assert line.endswith("phase reached")

    def test_uvicorn_server_lines_use_the_host_format(
        self, uvicorn_logging_configured, capsys
    ):
        setup_logging(fmt="text", level="INFO")
        logging.getLogger("uvicorn.error").info(
            "Waiting for connections to close. (CTRL+C to force quit)"
        )

        captured = capsys.readouterr()
        lines = (captured.err + captured.out).strip().splitlines()
        assert len(lines) == 1, lines
        assert re.match(
            rf"^{_TIMESTAMP} INFO:uvicorn\.error:\[correlation_id=-\]:"
            r"Waiting for connections to close\.",
            lines[0],
        ), lines[0]

    def test_uvicorn_access_lines_use_the_host_format(
        self, uvicorn_logging_configured, capsys
    ):
        setup_logging(fmt="text", level="INFO")
        # The shape uvicorn's HTTP protocols log each request with.
        logging.getLogger("uvicorn.access").info(
            '%s - "%s %s HTTP/%s" %d', "127.0.0.1:5000", "GET", "/health", "1.1", 200
        )

        captured = capsys.readouterr()
        lines = (captured.err + captured.out).strip().splitlines()
        assert len(lines) == 1, lines
        assert re.match(rf"^{_TIMESTAMP} INFO:uvicorn\.access:", lines[0]), lines[0]
        assert lines[0].endswith('127.0.0.1:5000 - "GET /health HTTP/1.1" 200')

    def test_uvicorn_lines_follow_the_json_format_too(
        self, uvicorn_logging_configured, capsys
    ):
        setup_logging(fmt="json", level="INFO")
        logging.getLogger("uvicorn.error").info("Shutting down")

        captured = capsys.readouterr()
        lines = (captured.err + captured.out).strip().splitlines()
        assert len(lines) == 1, lines
        parsed = json.loads(lines[0])
        assert parsed["logger"] == "uvicorn.error"
        assert parsed["message"] == "Shutting down"
        assert "T" in parsed["timestamp"]

    def test_access_log_stays_on_when_uvicorn_logs_requests(self, capsys):
        with _uvicorn_configured() as config:
            assert _logs_requests(config) is True
            setup_logging(fmt="text", level="INFO")
            assert _logs_requests(config) is True

    def test_no_access_log_stays_off_after_setup(self, capsys):
        """``--no-access-log`` is not undone by the application's logging setup.

        uvicorn applies it before importing the application, whose import then
        calls ``setup_logging``; routing ``uvicorn.access`` to the root handler
        would give it a handler again and turn request logging back on.
        """
        with _uvicorn_configured(access_log=False) as config:
            assert _logs_requests(config) is False
            setup_logging(fmt="text", level="INFO")
            assert _logs_requests(config) is False

            logging.getLogger("uvicorn.access").info(
                '%s - "%s %s HTTP/%s" %d',
                "127.0.0.1:5000", "GET", "/health", "1.1", 200,
            )
            logging.getLogger("uvicorn.error").info("Started server process")

        captured = capsys.readouterr()
        lines = (captured.err + captured.out).strip().splitlines()
        # The server's own lines are still routed; the access line went nowhere.
        assert len(lines) == 1, lines
        assert re.match(
            rf"^{_TIMESTAMP} INFO:uvicorn\.error:", lines[0]
        ), lines[0]
