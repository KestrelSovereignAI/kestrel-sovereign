"""Closes for the handles an :class:`LLMService` owns, kept until confirmed (#3559).

A provider adapter, an HTTP client, or a private inference client is released
only when its close finishes without raising. A close that raised has not
released its handle, and neither has one that is still running when its
timeout expires. :class:`HandleCloser` keeps every such handle, together with
the close a timeout left running, so that a later close retries the first and
waits for the second instead of starting it again. A second close of an HTTP
client already being closed returns at once, which would report the first
close finished while it is still running.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

#: A zero-argument callable that closes one handle. It may return an
#: awaitable, which the close then waits for.
CloseCall = Callable[[], Any]


@dataclass
class _UnconfirmedClose:
    label: str
    handle: object
    close: CloseCall
    #: The close started for ``handle`` and not yet confirmed, if any.
    attempt: Optional["asyncio.Future[Any]"] = None


def client_closes(label: str, client: Any) -> List[Tuple[str, object, CloseCall]]:
    """The ``(label, handle, close)`` of every close that releases ``client``.

    A client is released by its ``close()``, or else by the ``aclose()`` of
    the HTTP client it keeps as ``_client`` (Ollama). A google-genai
    ``Client`` holds a second, asynchronous transport that its ``close()``
    leaves open, and the Google and Vertex adapters call through it; its
    ``aio.aclose()`` releases that one. Each close is a handle of its own, so
    one that fails is retried without the other.
    """
    closes: List[Tuple[str, object, CloseCall]] = []
    close = getattr(client, "close", None)
    if callable(close):
        closes.append((label, client, close))
    else:
        inner = getattr(client, "_client", None)
        inner_close = getattr(inner, "aclose", None)
        if callable(inner_close):
            closes.append((label, inner, inner_close))
    aio = getattr(client, "aio", None)
    aio_close = getattr(aio, "aclose", None)
    if callable(aio_close):
        closes.append((f"{label} async transport", aio, aio_close))
    return closes


def _attempt_failed(attempt: "asyncio.Future[Any]") -> bool:
    return attempt.cancelled() or attempt.exception() is not None


def _observe_attempt(attempt: "asyncio.Future[Any]") -> None:
    """Retrieve a finished attempt's error so asyncio does not report it lost.

    Every caller that awaited the attempt has already reported the error, or
    reported a timeout and left the handle for a later close.
    """
    if not attempt.cancelled():
        attempt.exception()


def _caller_cancelled() -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


class HandleCloser:
    """Handles whose close has not been confirmed, each retried by a later close."""

    def __init__(self) -> None:
        # id(handle) -> its close. The entry holds the handle, so the id
        # cannot be reused by another object while the entry exists.
        self._unconfirmed: Dict[int, _UnconfirmedClose] = {}

    def unconfirmed(self) -> List[Tuple[str, object, CloseCall]]:
        """The ``(label, handle, close)`` of every handle not confirmed closed."""
        return [
            (entry.label, entry.handle, entry.close)
            for entry in self._unconfirmed.values()
        ]

    async def close(
        self,
        label: str,
        handle: object,
        close: CloseCall,
        *,
        timeout: float,
    ) -> None:
        """Close ``handle`` with ``close``, and return only once it has closed.

        A close that an earlier call left running is waited for again rather
        than started twice; one that finished with an error is started again.
        Either way the handle is forgotten only once a close finishes without
        raising.

        Raises:
            TimeoutError: The close did not finish within ``timeout``. It keeps
                running, and a later call waits for it.
            Exception: What the close raised. A later call starts it again.
            asyncio.CancelledError: The caller was cancelled. A close already
                started keeps running, and a later call waits for it.
        """
        key = id(handle)
        entry = self._unconfirmed.get(key)
        if entry is None:
            entry = _UnconfirmedClose(label, handle, close)
            self._unconfirmed[key] = entry
        attempt = entry.attempt
        if attempt is None or (attempt.done() and _attempt_failed(attempt)):
            entry.label, entry.close, entry.attempt = label, close, None
            result = close()
            if not inspect.isawaitable(result):
                del self._unconfirmed[key]
                return
            attempt = entry.attempt = asyncio.ensure_future(result)
            attempt.add_done_callback(_observe_attempt)
        try:
            await asyncio.wait_for(asyncio.shield(attempt), timeout=timeout)
        except asyncio.CancelledError:
            if attempt.cancelled() and not _caller_cancelled():
                raise RuntimeError(
                    f"closing {label} was cancelled before it finished"
                ) from None
            raise
        except TimeoutError:
            if not attempt.done():
                raise TimeoutError(
                    f"closing {label} did not finish within {timeout}s; it is "
                    "still running"
                ) from None
            if attempt.cancelled():
                raise RuntimeError(
                    f"closing {label} was cancelled before it finished"
                ) from None
            error = attempt.exception()
            if error is not None:
                # The close raised, by itself or just as the wait timed out.
                raise error
            # The close finished just as the wait timed out.
        if self._unconfirmed.get(key) is entry:
            del self._unconfirmed[key]
