"""Work deferred until the transaction holding a write commits (#2913).

A write made inside a caller's transaction is not durable when it returns: it
commits, or rolls back, with that transaction. Work that may only follow a
persisted write (counting a turn stored without its embedding, say) is
registered with the backend's ``call_after_commit`` and runs once the
outermost transaction commits. A rollback discards it.
"""

from __future__ import annotations

import logging
from typing import Callable, List

logger = logging.getLogger(__name__)

AfterCommitCallback = Callable[[], None]


def run_after_commit(callback: AfterCommitCallback) -> None:
    """Run one callback whose writes have committed.

    The writes are durable, so a callback that fails must not reach the
    writer as a failed write: it is logged with its traceback instead.
    """
    try:
        callback()
    except Exception:  # noqa: BLE001 - the write committed; reporting it as failed would lie
        logger.exception("After-commit callback %r failed", callback)


class AfterCommitCallbacks:
    """The callbacks registered inside one open top-level transaction.

    A nested scope that rolls back on its own (a savepoint) gets a frame of
    its own: merged into its parent when it is released, discarded when it
    rolls back. A joined nested scope registers on its parent's frame,
    because its writes commit or roll back with the parent's.
    """

    def __init__(self) -> None:
        self._frames: List[List[AfterCommitCallback]] = [[]]

    def register(self, callback: AfterCommitCallback) -> None:
        self._frames[-1].append(callback)

    def open_savepoint(self) -> None:
        self._frames.append([])

    def release_savepoint(self) -> None:
        released = self._frames.pop()
        self._frames[-1].extend(released)

    def roll_back_savepoint(self) -> None:
        self._frames.pop()

    @property
    def pending(self) -> bool:
        """Whether anything is registered to run once the transaction commits."""
        return any(self._frames)

    def run(self) -> None:
        """Run what was registered, in order. Call once, after the commit."""
        callbacks, self._frames = self._frames[0], [[]]
        for callback in callbacks:
            run_after_commit(callback)
