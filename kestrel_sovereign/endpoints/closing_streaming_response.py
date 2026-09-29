"""Streaming response that owns closure of its asynchronous body."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi.responses import StreamingResponse

from kestrel_sovereign._async_ownership import (
    OwnedTaskOutcome,
    await_owned_task,
    raise_owned_outcome,
)


class ClosingStreamingResponse(StreamingResponse):
    """Close the response iterator even when transport delivery is interrupted.

    Starlette does not close ``body_iterator`` when an ASGI 2.4 ``send`` raises
    after a yielded chunk.  These streams own request lifecycle and paid model
    work below that iterator, so transport exit is also a cleanup boundary.
    """

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
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
            close_task = asyncio.create_task(
                close(),
                name="closing_streaming_response:body",
            )
            outcome = await await_owned_task(
                close_task,
                pending_cancellation=(
                    primary if isinstance(primary, asyncio.CancelledError) else None
                ),
            )

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
