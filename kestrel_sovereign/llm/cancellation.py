"""Shared LLM request-cancellation helpers."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import AsyncIterator, Awaitable, Callable, Optional, TypeVar

CancelToken = Callable[[], bool]

_T = TypeVar("_T")
_POLL_INTERVAL_SECONDS = 0.1


def raise_if_cancelled(cancel_token: Optional[CancelToken]) -> None:
    """Raise ``CancelledError`` when the request-bound token has fired."""
    if cancel_token is not None and cancel_token():
        raise asyncio.CancelledError


async def anext_or_cancelled(
    iterator: AsyncIterator[_T],
    cancel_token: Optional[CancelToken],
    *,
    poll_interval: float = _POLL_INTERVAL_SECONDS,
) -> _T:
    """Await the next async-iterator item while polling ``cancel_token``.

    ``async for`` only regains control after the provider yields an item. A
    silent SSE/WebSocket can therefore hide a user stop until an idle timeout.
    This helper keeps the provider ``__anext__`` task pending while polling the
    token, then cancels that pending read so the surrounding SDK stream context
    can close its underlying connection immediately.
    """
    raise_if_cancelled(cancel_token)
    task = asyncio.create_task(iterator.__anext__())
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=poll_interval)
            if done:
                return task.result()
            raise_if_cancelled(cancel_token)
    except BaseException:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task
        raise


async def await_or_cancelled(
    awaitable: Awaitable[_T],
    cancel_token: Optional[CancelToken],
    *,
    poll_interval: float = _POLL_INTERVAL_SECONDS,
) -> _T:
    """Await any operation while polling ``cancel_token``."""
    raise_if_cancelled(cancel_token)
    task = asyncio.create_task(awaitable)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=poll_interval)
            if done:
                return task.result()
            raise_if_cancelled(cancel_token)
    except BaseException:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task
        raise


__all__ = [
    "CancelToken",
    "anext_or_cancelled",
    "await_or_cancelled",
    "raise_if_cancelled",
]
