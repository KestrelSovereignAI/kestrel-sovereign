"""Bounded recovery of idempotent database polling after native contention."""

import asyncio
import logging
import math
import sqlite3
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    is_execution_control_error,
)
from kestrel_sovereign.storage.db.interface import DatabaseError

_Result = TypeVar("_Result")


def _transient_driver_error(error: BaseException) -> bool:
    """Use driver types/codes, never error text or a generic wrapper class."""
    if isinstance(error, sqlite3.Error):
        code = getattr(error, "sqlite_errorcode", None)
        return isinstance(code, int) and code & 0xFF == sqlite3.SQLITE_BUSY
    import asyncpg

    return isinstance(
        error,
        (
            asyncpg.SerializationError,
            asyncpg.DeadlockDetectedError,
            asyncpg.ConnectionDoesNotExistError,
            asyncpg.ConnectionFailureError,
            asyncpg.CannotConnectNowError,
            asyncpg.TooManyConnectionsError,
            asyncpg.LockNotAvailableError,
            ConnectionRefusedError,
            ConnectionResetError,
            ConnectionAbortedError,
        ),
    )


def is_transient_database_poll_error(error: BaseException) -> bool:
    """Recognize only native availability failures with no execution debt.

    A bare DatabaseError proves no recovery disposition. Every explicit leaf
    must be a known transient driver failure, and intermediate carriers must
    be database wrappers or groups. Authority, acknowledged/uncertain commits
    and physical cleanup evidence always retain their original control.
    """
    if not isinstance(error, (DatabaseError, ExceptionGroup)) or (
        is_execution_control_error(error)
    ):
        return False
    nodes = tuple(_explicit_control_chain(error))
    if not all(
        isinstance(node, (DatabaseError, BaseExceptionGroup))
        or _transient_driver_error(node)
        for node in nodes
    ):
        return False
    leaves = [
        node
        for node in nodes
        if node.__cause__ is None and not isinstance(node, BaseExceptionGroup)
    ]
    return bool(leaves) and all(_transient_driver_error(node) for node in leaves)


@dataclass(frozen=True)
class DatabasePollRetryPolicy:
    """A finite retry budget for already-idempotent resident polling passes.

    This does not make arbitrary writes retryable. The operation must use its
    canonical transactional/claim ledger and retain uncertain native outcomes.
    Startup/schema changes, external sends and foreground requests do not use
    this policy. Successful passes reset the next pass's attempt budget.
    """

    max_attempts: int = 3

    def __post_init__(self) -> None:
        if type(self.max_attempts) is not int or self.max_attempts <= 0:
            raise ValueError("Database polling attempts must be a positive integer")

    async def run(
        self,
        operation: Callable[[], Awaitable[_Result]],
        *,
        retry_seconds: float,
        label: str,
    ) -> _Result:
        if (
            isinstance(retry_seconds, bool)
            or not isinstance(retry_seconds, (int, float))
            or not math.isfinite(retry_seconds)
            or retry_seconds <= 0
        ):
            raise ValueError("Database polling retry delay must be finite and positive")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Database polling requires a concrete service label")
        for attempt in range(1, self.max_attempts + 1):
            try:
                return await operation()
            except (DatabaseError, ExceptionGroup) as error:
                if (
                    attempt == self.max_attempts
                    or not is_transient_database_poll_error(error)
                ):
                    raise
                logging.getLogger(__name__).warning(
                    "Database poll deferred: service=%s attempt=%s limit=%s "
                    "error=%s retry_seconds=%s",
                    label,
                    attempt,
                    self.max_attempts,
                    type(error).__name__,
                    retry_seconds,
                )
                await asyncio.sleep(retry_seconds)
        raise AssertionError("Validated database polling budget admitted no attempt")
