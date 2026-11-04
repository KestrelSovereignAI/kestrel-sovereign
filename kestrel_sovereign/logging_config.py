"""Structured JSON logging with configurable formatters.

Provides optional JSON log output for production/enterprise deployments
(Datadog, Splunk, ELK, CloudWatch, Google Cloud Logging).

Usage:
    from kestrel_sovereign.logging_config import setup_logging
    setup_logging()  # reads KESTREL_LOG_FORMAT and KESTREL_LOG_LEVEL from env

Environment variables:
    KESTREL_LOG_FORMAT  - "text" (default) or "json"
    KESTREL_LOG_LEVEL   - DEBUG, INFO (default), WARNING, ERROR
"""

import contextvars
import json
import logging
import os
import re
import traceback
import uuid
from datetime import datetime, timezone
from typing import Optional

# The SDK's own function and key, not a copy: a second implementation would
# key its digest differently, so Core and SDK log lines for the same text could
# not be matched, and the two would drift (#3952).
from kestrel_sdk.logging_summary import (
    CONTENT_DIGEST_HEX_CHARS as CONTENT_DIGEST_HEX_CHARS,
    content_log_summary as content_log_summary,
)

# --- Context variables for request-scoped fields ---

correlation_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)
session_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "session_id", default=None
)
agent_name_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "agent_name", default=None
)
tool_name_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "tool_name", default=None
)
feature_name_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "feature_name", default=None
)


MAX_CORRELATION_ID_LENGTH = 128
_CORRELATION_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")


def normalize_correlation_id(value: object) -> Optional[str]:
    """Return a support-safe correlation ID or ``None`` when invalid."""
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate or len(candidate) > MAX_CORRELATION_ID_LENGTH:
        return None
    if _CORRELATION_ID_RE.fullmatch(candidate) is None:
        return None
    return candidate


def generate_correlation_id() -> str:
    """Generate a new opaque, support-safe request correlation ID."""
    return uuid.uuid4().hex[:16]


def resolve_correlation_id(
    value: object = None,
    *,
    use_context: bool = True,
) -> str:
    """Resolve a safe supplied/ambient ID, or generate a fresh one."""
    return (
        normalize_correlation_id(value)
        or (
            normalize_correlation_id(correlation_id_var.get())
            if use_context
            else None
        )
        or generate_correlation_id()
    )


def get_correlation_id() -> str:
    """Get the current safe correlation ID, generating one if not set."""
    cid = resolve_correlation_id()
    if cid != correlation_id_var.get():
        correlation_id_var.set(cid)
    return cid


def exception_log_summary(error: BaseException) -> str:
    """Describe an exception for a log line without its message text.

    A tool's or a subagent's exception can quote what it failed on (its
    arguments, a stored message it read), so a log line gets the class and the
    :func:`content_log_summary` of the message, never the message (#3919).
    """
    return f"{type(error).__name__} {content_log_summary(str(error))}"


def exception_frames(error: BaseException) -> str:
    """The traceback frames of ``error``, without its message.

    ``exc_info=True`` ends the logged traceback with ``Type: message``, which
    carries the message text :func:`exception_log_summary` withholds. The
    frames alone (file, line, function, source line) still say where it
    failed.
    """
    return "".join(traceback.format_tb(error.__traceback__))


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects.

    Standard fields: timestamp, level, logger, message, correlation_id.
    Context fields (when set): session_id, agent_name, tool_name, feature_name.
    """

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id_var.get(),
        }

        # Add context fields when present
        session_id = session_id_var.get()
        if session_id is not None:
            entry["session_id"] = session_id

        agent_name = agent_name_var.get()
        if agent_name is not None:
            entry["agent_name"] = agent_name

        tool_name = tool_name_var.get()
        if tool_name is not None:
            entry["tool_name"] = tool_name

        feature_name = feature_name_var.get()
        if feature_name is not None:
            entry["feature_name"] = feature_name

        # Include exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, default=str)


class CorrelationContextFilter(logging.Filter):
    """Attach request correlation context for non-JSON formatters."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id_var.get() or "-"
        return True


TEXT_LOG_FORMAT = (
    "%(asctime)s %(levelname)s:%(name)s:[correlation_id=%(correlation_id)s]:%(message)s"
)

#: uvicorn configures these loggers before it imports the application, with
#: handlers and a formatter of its own and ``propagate=False`` on the first and
#: last. Left that way, uvicorn's own lines (startup, shutdown, "Waiting for
#: connections to close", every access line) reach the host log in its format
#: and without a timestamp, so a restart cannot be timed from the log (#3781).
UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


class MCPOAuthDiagnosticFilter(logging.Filter):
    """Native OAuth exceptions may embed state, codes or entire token models.

    Preserve severity and logger identity, never format third-party content.
    The feature reports typed safe outcome codes at its own boundary.
    """

    def filter(self, record):
        if record.name == "mcp.client.auth" or record.name.startswith("mcp.client.auth."):
            record.msg = "MCP OAuth diagnostic redacted; inspect the connection status"
            record.args = ()
            record.exc_info = record.exc_text = record.stack_info = None
        return True


def _install_mcp_oauth_diagnostic_filter(handler):
    handler.addFilter(MCPOAuthDiagnosticFilter())
    # Logger filters run before any additional handler (including diagnostic
    # collectors). Python does not apply ancestor logger filters to children.
    # These are the native SDK's auth loggers; the canonical handler also
    # guards every descendant namespace, including future SDK modules.
    for name in ("mcp.client.auth", "mcp.client.auth.oauth2"):
        logger = logging.getLogger(name)
        if not any(isinstance(item, MCPOAuthDiagnosticFilter) for item in logger.filters):
            logger.addFilter(MCPOAuthDiagnosticFilter())


def _route_uvicorn_loggers_to_root() -> None:
    """Send uvicorn's records through the root handler and its formatter.

    A logger with no handlers that does not propagate writes nowhere, and is
    left that way: it was silenced on purpose. ``--no-access-log`` silences
    ``uvicorn.access`` exactly so, and uvicorn's HTTP protocols then log
    requests only while ``hasHandlers()`` is true, so letting it propagate
    would turn access logging back on.
    """
    for name in UVICORN_LOGGERS:
        uvicorn_logger = logging.getLogger(name)
        if not uvicorn_logger.handlers and not uvicorn_logger.propagate:
            continue
        for handler in list(uvicorn_logger.handlers):
            uvicorn_logger.removeHandler(handler)
        uvicorn_logger.propagate = True


def setup_logging(
    fmt: Optional[str] = None,
    level: Optional[str] = None,
) -> None:
    """Configure the root logger with the chosen format and level.

    uvicorn's loggers are routed through the same root handler, so every line
    the host writes carries the same format and timestamp.

    Args:
        fmt: "text" or "json". Defaults to env KESTREL_LOG_FORMAT or "text".
        level: Log level name. Defaults to env KESTREL_LOG_LEVEL or "INFO".
    """
    log_format = (fmt or os.environ.get("KESTREL_LOG_FORMAT", "text")).lower()
    log_level = (level or os.environ.get("KESTREL_LOG_LEVEL", "INFO")).upper()

    root = logging.getLogger()

    # Clear existing handlers (e.g. from basicConfig) to avoid duplicates
    root.handlers.clear()

    handler = logging.StreamHandler()
    handler.addFilter(CorrelationContextFilter())
    _install_mcp_oauth_diagnostic_filter(handler)

    if log_format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(TEXT_LOG_FORMAT))

    root.addHandler(handler)
    root.setLevel(getattr(logging, log_level, logging.INFO))
    _route_uvicorn_loggers_to_root()
