"""Task-local database write audit hooks."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator, Optional

WriteAuditCallback = Callable[[str], None]

_WRITE_AUDIT_CALLBACK: ContextVar[Optional[WriteAuditCallback]] = ContextVar(
    "kestrel_db_write_audit_callback",
    default=None,
)
_REQUESTED_HANDLER_WRITE_AUDIT_CALLBACK: ContextVar[
    Optional[WriteAuditCallback]
] = ContextVar(
    "kestrel_requested_handler_write_audit_callback",
    default=None,
)


@contextmanager
def capture_write_queries(callback: WriteAuditCallback) -> Iterator[None]:
    """Capture write SQL issued by the current task context."""

    token = _WRITE_AUDIT_CALLBACK.set(callback)
    try:
        yield
    finally:
        _WRITE_AUDIT_CALLBACK.reset(token)


@contextmanager
def suppress_write_audit() -> Iterator[None]:
    """Temporarily disable write capture for dispatcher-owned writes."""

    token = _WRITE_AUDIT_CALLBACK.set(None)
    try:
        yield
    finally:
        _WRITE_AUDIT_CALLBACK.reset(token)


@contextmanager
def request_handler_write_audit(callback: WriteAuditCallback) -> Iterator[None]:
    """Request write capture for the dispatcher ACTION handler only."""

    token = _REQUESTED_HANDLER_WRITE_AUDIT_CALLBACK.set(callback)
    try:
        yield
    finally:
        _REQUESTED_HANDLER_WRITE_AUDIT_CALLBACK.reset(token)


def requested_handler_write_audit_callback() -> Optional[WriteAuditCallback]:
    return _REQUESTED_HANDLER_WRITE_AUDIT_CALLBACK.get()


def record_write_query(query: str) -> None:
    callback = _WRITE_AUDIT_CALLBACK.get()
    if callback is not None:
        callback(query)


def record_write_script(script: str) -> None:
    if script.strip():
        record_write_query(script)


__all__ = [
    "capture_write_queries",
    "record_write_query",
    "record_write_script",
    "request_handler_write_audit",
    "requested_handler_write_audit_callback",
    "suppress_write_audit",
]
