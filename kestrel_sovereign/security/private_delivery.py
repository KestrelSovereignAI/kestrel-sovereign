"""Retain private host capability authority through the actual ASGI send.

Buffering middleware can acknowledge its own queue before bytes reach the
transport. A capability response therefore keeps its original leases until
the outer middleware acknowledges the final body. This module owns transport
custody; each capability owns its authentication, operation and error policy.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from dataclasses import dataclass

from starlette.responses import Response

from .capability_authority import HostCapabilityAccessDenied
from kestrel_sovereign._async_ownership import (
    await_owned_task, create_owned_task, raise_owned_outcome,
)


PRIVATE_RESPONSE_HEADERS = {
    "Cache-Control": "private, no-store, max-age=0",
    "Pragma": "no-cache",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "sandbox; default-src 'none'",
}
_DELIVERY_KEY = "kestrel.private_delivery"


@dataclass(frozen=True)
class PrivateTransportBorrow:
    """Trusted original owner callbacks; this object confers no authority."""
    operation: Callable[[], AbstractAsyncContextManager[Callable[[], None]]]
    cancel: Callable[[asyncio.Task], None]


class _Delivery:
    def __init__(self):
        self.check_current = None
        self.deadline = None
        self.active = False
        self.failed = False
        self.finished = asyncio.Event()
        self.access_denied = HostCapabilityAccessDenied
        self.transport_borrow = None

    def begin(self, check, deadline, access_denied, transport_borrow=None):
        if self.check_current is not None:
            raise access_denied("private delivery already bound")
        if transport_borrow is not None and (not isinstance(transport_borrow, PrivateTransportBorrow)
                or not callable(transport_borrow.operation) or not callable(transport_borrow.cancel)):
            raise TypeError("original private transport ownership callbacks required")
        self.access_denied = access_denied
        self.check_current, self.deadline, self.active = check, deadline, True
        self.transport_borrow = transport_borrow

    def check(self):
        if not self.active or self.failed:
            raise self.access_denied("private delivery is no longer active")
        self.check_current()


class PrivateDeliveryMiddleware:
    """Install outside buffering middleware for all private host services."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        delivery = _Delivery()
        scope[_DELIVERY_KEY] = delivery

        async def guarded_send(message):
            if delivery.check_current is None:
                await send(message)
                return
            try:
                async with asyncio.timeout_at(delivery.deadline):
                    delivery.check()
                    if delivery.transport_borrow is None:
                        await send(message)
                    else:
                        async def native_send():
                            async with delivery.transport_borrow.operation() as current:
                                delivery.check()
                                current()
                                await send(message)
                                current()
                        # The outer send may block after buffering middleware
                        # has acknowledged its queue. Keep this exact child in
                        # the capability's original borrower registry until
                        # physical transport settlement, even under repeated
                        # cancellation or cancellation-suppressing transports.
                        child = create_owned_task(native_send(), name="private-capability-send")
                        outcome = await await_owned_task(child,
                            cancel_on_caller_cancellation=delivery.transport_borrow.cancel)
                        raise_owned_outcome(outcome, operation="private capability transport")
            except BaseException:
                delivery.failed = True
                delivery.finished.set()
                raise
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                delivery.finished.set()

        try:
            await self.app(scope, receive, guarded_send)
        finally:
            delivery.active = False
            delivery.finished.set()


class AuthorizedCapabilityResponse(Response):
    """Keep a host-resolved context and operation lease until delivery ends.

``operation(context, prepared_value)`` returns the capability's async authority
context manager, yielding a synchronous current-policy checker. The resolver
must retain the original context binding. ``error_response(error, prepared)``
maps failures using the capability's own mutation/reconciliation semantics.
Untrusted body preparation runs after authentication and before policy leases.
The optional trusted ``transport_operation(context, prepared_value, check)``
retains that same authority and yields a ``PrivateTransportBorrow`` for each
actual outer send. Its cancellation callback shares the original owner's
single-cancellation custody. No context is resolved from response metadata.
"""

    def __init__(
        self, request, *, authority, resolve_context, operation, build,
        error_response, timeout_seconds=30, prepare=None,
        access_denied=HostCapabilityAccessDenied, transport_operation=None,
    ):
        super().__init__()
        if (isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds)
                or timeout_seconds <= 0):
            raise ValueError("private response timeout must be positive and finite")
        self.request, self.authority = request, authority
        self.resolve_context, self.operation, self.build = resolve_context, operation, build
        self.error_response, self.access_denied = error_response, access_denied
        self.timeout_seconds, self.prepare = timeout_seconds, prepare
        if transport_operation is not None and not callable(transport_operation):
            raise TypeError("private transport operation must be callable")
        self.transport_operation = transport_operation

    async def __call__(self, scope, receive, send):
        started = False
        prepared = False
        delivery = scope.get(_DELIVERY_KEY)

        async def guarded_send(message):
            nonlocal started
            check()
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            if not isinstance(delivery, _Delivery):
                raise self.access_denied("private transport custody is unavailable")
            async with asyncio.timeout(self.timeout_seconds) as request_timeout:
                async with self.resolve_context(self.request) as context:
                    remaining = (context.expires_at - self.authority.clock()).total_seconds()
                    context_deadline = asyncio.get_running_loop().time() + max(0, remaining)
                    async with asyncio.timeout(max(0, remaining)):
                        value = await self.prepare() if self.prepare is not None else None
                        async with self.operation(context, value) as check:
                            response = (await self.build(context, value) if self.prepare is not None
                                        else await self.build(context))
                            prepared = True
                            check()
                            for key, header in PRIVATE_RESPONSE_HEADERS.items():
                                response.headers[key] = header
                            async with AsyncExitStack() as transport:
                                guard = (await transport.enter_async_context(
                                    self.transport_operation(context, value, check))
                                    if self.transport_operation is not None else None)
                                delivery.begin(check, min(request_timeout.when(), context_deadline),
                                               self.access_denied, guard)
                                try:
                                    await response(scope, receive, guarded_send)
                                    await delivery.finished.wait()
                                    delivery.check()
                                finally:
                                    delivery.active = False
        except Exception as error:
            if started:
                raise
            response = self.error_response(error, prepared)
            for key, header in PRIVATE_RESPONSE_HEADERS.items():
                response.headers[key] = header
            await response(scope, receive, send)
