"""Streaming response that owns closure of its asynchronous body."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, NoReturn

from fastapi.responses import StreamingResponse

from kestrel_sovereign._async_ownership import (
    OwnedTaskOutcome,
    raise_owned_outcome,
)
from kestrel_sovereign.execution_custody import (
    execution_terminal_error,
    is_execution_control_error,
)


class ClosingStreamingResponse(StreamingResponse):
    """Close the response iterator even when transport delivery is interrupted.

    Starlette does not close ``body_iterator`` when an ASGI 2.4 ``send`` raises
    after a yielded chunk.  These streams own request lifecycle and paid model
    work below that iterator, so transport exit is also a cleanup boundary.

    ``close_in_stream_task`` is for inline-safe, idempotent close-aware bodies.
    It closes in the original producer (and handles unstarted bodies on outer
    response exit) without admitting a cleanup Task. Both modes retain native
    control's exact cause through ASGI cancellation/disconnect normalization.
    """

    def __init__(self, *args, close_in_stream_task: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._close_in_stream_task = close_in_stream_task

    @staticmethod
    def _raise_terminal(error: BaseException) -> NoReturn:
        """Preserve native control before ASGI normalizes cancellation/OSError."""
        if (
            isinstance(error, (asyncio.CancelledError, OSError))
            and is_execution_control_error(error)
        ):
            raise RuntimeError(
                "Streaming execution control requires reconciliation"
            ) from error
        raise error

    async def _run_with_inline_close(self, operation: Awaitable[None]) -> None:
        """Retire an inline-safe body without admitting another cleanup Task."""
        primary = None
        try:
            await operation
        except BaseException as error:
            primary = error
        close = getattr(self.body_iterator, "aclose", None)
        if callable(close):
            try:
                await close()
            except BaseException as error:
                primary = execution_terminal_error(
                    primary, error, retain_originals=True,
                )
        if primary is not None:
            self._raise_terminal(primary)

    async def stream_response(
        self, send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        if self._close_in_stream_task:
            # On ASGI <2.4 this is the ORIGINAL producer task, not the
            # response's listening task. Close here before that task exits.
            await self._run_with_inline_close(super().stream_response(send))
        else:
            try:
                await super().stream_response(send)
            except BaseException as error:
                self._raise_terminal(error)

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        if self._close_in_stream_task:
            # Also retires an unstarted body if response/producer admission
            # failed. Such callers supply an idempotent, close-aware body.
            await self._run_with_inline_close(super().__call__(scope, receive, send))
            return
        primary: BaseException | None = None
        try:
            await super().__call__(scope, receive, send)
        except BaseException as error:
            primary = error

        close = getattr(self.body_iterator, "aclose", None)
        outcome: OwnedTaskOutcome[object] = OwnedTaskOutcome(
            None,
            None,
            primary if isinstance(primary, asyncio.CancelledError) else None,
        )
        if callable(close):
            from kestrel_sovereign._async_ownership import await_owned_operation
            outcome = await await_owned_operation(
                close(), name="closing_streaming_response:body",
                retirement_resource=self.body_iterator, retain_originals=True,
                pending_cancellation=(
                    primary if isinstance(primary, asyncio.CancelledError) else None
                ),
            )

        if any(
            error is not None and is_execution_control_error(error)
            for error in (primary, outcome.error, outcome.cancellation)
        ):
            self._raise_terminal(execution_terminal_error(
                primary, outcome.error, outcome.cancellation, retain_originals=True,
                isolate_control_originals=True,
            ))

        if primary is not None and not isinstance(
            primary, asyncio.CancelledError
        ):
            secondary = outcome.error or outcome.cancellation
            if secondary is not None:
                raise BaseExceptionGroup(
                    "Streaming response delivery and body cleanup both failed",
                    [primary, secondary],
                )
            raise primary
        raise_owned_outcome(outcome, operation="streaming response body cleanup")


__all__ = ["ClosingStreamingResponse"]
