"""Cancellation-safe ownership of internal asyncio tasks.

Callers that create cleanup, launch, or persistence tasks remain responsible
for reaching a terminal outcome even when their own task is cancelled. This
module provides that host-wide ownership rule without depending on subprocess,
Talon, compute, or training concepts.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Generic, TypeVar, cast


_T = TypeVar("_T")
_ITERATOR_TERMINAL = object()
_ITERATOR_INTERRUPTED = object()


@dataclass(frozen=True, slots=True)
class OwnedTaskOutcome(Generic[_T]):
    """Terminal result of an owned task plus caller cancellation state."""

    result: _T | None
    error: BaseException | None
    cancellation: asyncio.CancelledError | None


class OwnedAsyncIterator(Generic[_T]):
    """Keep every interaction with an async iterator in one owner task.

    Async generators may bind ``ContextVar`` tokens while producing their first
    item. Resuming or closing such a generator from another task operates in a
    copied context and makes token reset fail. This adapter owns construction,
    iteration, and closure in one task while its consumer receives items over a
    one-item handshake. ``aclose`` remains cancellation-safe by joining that
    owner with :func:`await_owned_task`.
    """

    def __init__(
        self,
        iterator_factory: Callable[[], AsyncIterator[_T]],
        *,
        operation: str,
        cleanup_requested: Callable[[], bool] | None = None,
    ) -> None:
        self._iterator_factory = iterator_factory
        self._operation = operation
        self._cleanup_requested = cleanup_requested
        self._items: asyncio.Queue[tuple[object, _T | None]] = asyncio.Queue()
        self._continue = asyncio.Event()
        self._stop = asyncio.Event()
        self._item_outstanding = False
        self._waiting_for_continue = False
        self._interrupt_after_continue = False
        self._closed = False
        self._interrupted_by_cleanup = False
        self._cleanup_error: BaseException | None = None
        self._owner = asyncio.create_task(
            self._run(),
            name=f"owned_async_iterator:{operation}",
        )
        self._cancellation_owner = asyncio.create_task(
            self._watch_owner_cancellation(),
            name=f"owned_async_iterator_cancel:{operation}",
        )

    def __aiter__(self) -> "OwnedAsyncIterator[_T]":
        return self

    @property
    def terminal_error(self) -> BaseException | None:
        """Return the source owner's terminal error once it has settled."""

        if not self._owner.done():
            return None
        try:
            return self._owner.exception()
        except asyncio.CancelledError as error:
            return error

    @property
    def owner_task(self) -> asyncio.Task[None]:
        """Return the task whose cancellation interrupts a blocked producer."""

        return self._cancellation_owner

    @property
    def interrupted_by_cleanup(self) -> bool:
        """Whether requested cancellation interrupted source iteration."""

        return self._interrupted_by_cleanup

    @property
    def cleanup_error(self) -> BaseException | None:
        """Return an error raised while cancellation/closure unwound the source."""

        return self._cleanup_error

    async def __anext__(self) -> _T:
        if self._closed:
            raise StopAsyncIteration
        if self._item_outstanding:
            self._item_outstanding = False
            self._continue.set()
            if self._interrupt_after_continue:
                self._interrupt_after_continue = False
                asyncio.get_running_loop().call_soon(
                    self._cancel_owner_if_running
                )

        marker, item = await self._items.get()
        if marker is _ITERATOR_INTERRUPTED:
            # Do not join here. The consumer may first publish its bounded Stop
            # acknowledgement; its ``finally`` then calls ``aclose`` and owns
            # the source cleanup through terminalization.
            raise StopAsyncIteration
        if marker is _ITERATOR_TERMINAL:
            self._closed = True
            outcome = await await_owned_task(self._owner)
            raise_owned_outcome(outcome, operation=self._operation)
            raise StopAsyncIteration

        self._item_outstanding = True
        return cast(_T, item)

    async def _watch_owner_cancellation(self) -> None:
        """Translate lifecycle cancellation without racing a delivered item."""

        completion = asyncio.Event()

        def mark_complete(_task: asyncio.Task[None]) -> None:
            completion.set()

        self._owner.add_done_callback(mark_complete)
        try:
            await completion.wait()
        except asyncio.CancelledError:
            if self._owner.done():
                return
            if self._item_outstanding:
                # Let the consumer resume the source once. A source already at
                # EOF can publish its real cleanup result; a provider that
                # blocks in the next ``anext`` is interrupted one loop turn later.
                self._interrupt_after_continue = True
            elif self._waiting_for_continue:
                # The consumer already released the handshake in this event-loop
                # turn. Let the owner publish a synchronous EOF/failure first,
                # then interrupt only if it actually blocks in the next source step.
                asyncio.get_running_loop().call_soon(
                    self._cancel_owner_if_running
                )
            else:
                self._owner.cancel()
            raise
        finally:
            self._owner.remove_done_callback(mark_complete)

    def _cancel_owner_if_running(self) -> None:
        if not self._owner.done():
            self._owner.cancel()

    async def _run(self) -> None:
        iterator: AsyncIterator[_T] | None = None
        try:
            iterator = self._iterator_factory()
            try:
                async for item in iterator:
                    if self._stop.is_set():
                        break
                    await self._items.put((self, item))
                    self._waiting_for_continue = True
                    try:
                        await self._continue.wait()
                    finally:
                        self._waiting_for_continue = False
                    self._continue.clear()
                    if self._stop.is_set():
                        break
            except BaseException as error:
                # An exception raised while a requested close interrupts
                # ``anext`` belongs to generator unwinding, not ordinary
                # source execution. Preserve that distinction for lifecycle
                # acknowledgement at the transport boundary.
                cleanup_requested = self._stop.is_set()
                if not cleanup_requested and self._cleanup_requested is not None:
                    try:
                        cleanup_requested = self._cleanup_requested() is True
                    except Exception:
                        # An observability predicate must not replace the
                        # iterator's actual terminal failure.
                        cleanup_requested = False
                if cleanup_requested and not isinstance(
                    error,
                    asyncio.CancelledError,
                ):
                    self._cleanup_error = error
                if cleanup_requested and isinstance(error, asyncio.CancelledError):
                    self._interrupted_by_cleanup = True
                    self._items.put_nowait((_ITERATOR_INTERRUPTED, None))
                    return
                raise
        finally:
            try:
                if iterator is not None:
                    close_iterator = getattr(iterator, "aclose", None)
                    if callable(close_iterator):
                        try:
                            await close_iterator()
                        except BaseException as error:
                            self._cleanup_error = error
                            raise
            finally:
                self._items.put_nowait((_ITERATOR_TERMINAL, None))

    async def aclose(self) -> None:
        """Stop producing and join the source iterator's owner task."""

        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._continue.set()
        owner_cancelled_by_close = False
        if not self._owner.done() and not self._interrupted_by_cleanup:
            owner_cancelled_by_close = self._owner.cancel()
        outcome = await await_owned_task(self._owner)
        if (
            owner_cancelled_by_close
            and isinstance(outcome.error, asyncio.CancelledError)
            and self._cleanup_error is None
        ):
            # Cancellation is the private interrupt used to wake a producer
            # blocked in ``anext``. Once the owner's finally block has closed
            # the iterator, it is a successful close, while cancellation of
            # the caller itself still takes precedence below.
            outcome = OwnedTaskOutcome(None, None, outcome.cancellation)
        raise_owned_outcome(outcome, operation=self._operation)


async def await_owned_task(
    task: asyncio.Task[_T],
    pending_cancellation: asyncio.CancelledError | None = None,
) -> OwnedTaskOutcome[_T]:
    """Retrieve ``task`` despite repeated cancellation of its caller.

    The shielded waiter carries only a successful event result, never the
    owned task's exception. The owned result is retrieved exactly once after
    terminalization, avoiding orphaned-task and shield-future warnings.
    """

    completion = asyncio.Event()

    def mark_complete(_task: asyncio.Task[_T]) -> None:
        completion.set()

    task.add_done_callback(mark_complete)
    completion_task = asyncio.create_task(completion.wait())
    try:
        while not task.done():
            try:
                await asyncio.shield(completion_task)
            except asyncio.CancelledError as error:
                if pending_cancellation is None:
                    pending_cancellation = error
    finally:
        task.remove_done_callback(mark_complete)
        if not completion_task.done():
            completion_task.cancel()

        # Do not await between terminalization and result retrieval. A further
        # Task.cancel() in that window could interrupt the bookkeeping that
        # prevents an unobserved completion-task exception.
        def retrieve_completion(waiter: asyncio.Task[bool]) -> None:
            try:
                waiter.result()
            except asyncio.CancelledError:
                pass

        if completion_task.done():
            retrieve_completion(completion_task)
        else:
            completion_task.add_done_callback(retrieve_completion)

    try:
        return OwnedTaskOutcome(task.result(), None, pending_cancellation)
    except BaseException as error:
        return OwnedTaskOutcome(None, error, pending_cancellation)


def raise_owned_outcome(
    outcome: OwnedTaskOutcome[_T],
    *,
    operation: str,
) -> _T:
    """Apply caller-cancellation precedence to a retrieved owned outcome."""

    if outcome.cancellation is not None:
        if outcome.error is not None:
            outcome.cancellation.add_note(f"{operation} also failed: {outcome.error}")
            raise outcome.cancellation from outcome.error
        raise outcome.cancellation
    if outcome.error is not None:
        raise outcome.error
    return cast(_T, outcome.result)


async def run_blocking_operation(
    operation: Callable[..., _T],
    *args: object,
) -> _T:
    """Finish a blocking operation before propagating caller cancellation."""

    operation_task = asyncio.create_task(asyncio.to_thread(operation, *args))
    outcome = await await_owned_task(operation_task)
    return raise_owned_outcome(outcome, operation="owned blocking operation")


__all__ = [
    "OwnedAsyncIterator",
    "OwnedTaskOutcome",
    "await_owned_task",
    "raise_owned_outcome",
    "run_blocking_operation",
]
