"""Shared LLM request-cancellation helpers."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import AsyncIterator, Awaitable, Callable, Optional, TypeVar

CancelToken = Callable[[], bool]
AuthCancelFactory = Callable[[], BaseException]

_T = TypeVar("_T")
_POLL_INTERVAL_SECONDS = 0.1


class AuthCancellationToken:
    """Explicit cancellation token for interactive auth/login flows.

    This is intentionally separate from per-turn request cancellation:
    firing this token means the caller owns the abort signal and wants
    the auth layer to raise a typed, user-facing cancellation error.
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    async def wait(self) -> None:
        await self._event.wait()

    def raise_if_cancelled(self, exc_factory: AuthCancelFactory) -> None:
        if self.is_cancelled():
            raise exc_factory()


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
    task = asyncio.ensure_future(awaitable)
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


async def await_or_auth_cancelled(
    awaitable: Awaitable[_T],
    cancellation_token: Optional[AuthCancellationToken],
    exc_factory: AuthCancelFactory,
) -> _T:
    """Await ``awaitable`` unless an explicit auth token fires first.

    Ordinary task cancellation is deliberately not translated here so
    outer ``asyncio.timeout(...)`` scopes can still rewrite their own
    ``CancelledError`` into ``TimeoutError``. Only the explicit auth
    token maps to the caller-provided typed exception.
    """
    if cancellation_token is None:
        return await awaitable
    cancellation_token.raise_if_cancelled(exc_factory)
    task = asyncio.ensure_future(awaitable)
    abort_task = asyncio.create_task(cancellation_token.wait())
    try:
        done, _ = await asyncio.wait(
            {task, abort_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if abort_task in done:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task
            raise exc_factory()
        abort_task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await abort_task
        return task.result()
    except BaseException:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task
        if not abort_task.done():
            abort_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await abort_task
        raise


__all__ = [
    "AuthCancellationToken",
    "CancelToken",
    "anext_or_cancelled",
    "await_or_auth_cancelled",
    "await_or_cancelled",
    "raise_if_cancelled",
]
