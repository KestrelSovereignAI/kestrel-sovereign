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
import uuid
from datetime import datetime, timezone
from typing import Optional

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


def get_correlation_id() -> str:
    """Get the current correlation ID, generating one if not set."""
    cid = correlation_id_var.get()
    if cid is None:
        cid = uuid.uuid4().hex[:16]
        correlation_id_var.set(cid)
    return cid


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


def setup_logging(
    fmt: Optional[str] = None,
    level: Optional[str] = None,
) -> None:
    """Configure the root logger with the chosen format and level.

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

    if log_format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(levelname)s:%(name)s:%(message)s"
        ))

    root.addHandler(handler)
    root.setLevel(getattr(logging, log_level, logging.INFO))
