"""Retain private host capability authority through the actual ASGI send.

Buffering middleware can acknowledge its own queue before bytes reach the
transport. A capability response therefore keeps its original leases until
the outer middleware acknowledges the final body. This module owns transport
custody; each capability owns its authentication, operation and error policy.
"""

from __future__ import annotations

import asyncio
import math

from starlette.responses import Response

from .capability_authority import HostCapabilityAccessDenied


PRIVATE_RESPONSE_HEADERS = {
    "Cache-Control": "private, no-store, max-age=0",
    "Pragma": "no-cache",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "sandbox; default-src 'none'",
}
_DELIVERY_KEY = "kestrel.private_delivery"


class _Delivery:
    def __init__(self):
        self.check_current = None
        self.deadline = None
        self.active = False
        self.failed = False
        self.finished = asyncio.Event()
        self.access_denied = HostCapabilityAccessDenied

    def begin(self, check, deadline, access_denied):
        if self.check_current is not None:
            raise access_denied("private delivery already bound")
        self.access_denied = access_denied
        self.check_current, self.deadline, self.active = check, deadline, True

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
                    await send(message)
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
"""

    def __init__(
        self, request, *, authority, resolve_context, operation, build,
        error_response, timeout_seconds=30, prepare=None,
        access_denied=HostCapabilityAccessDenied,
    ):
        super().__init__()
        if (isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds)
                or timeout_seconds <= 0):
            raise ValueError("private response timeout must be positive and finite")
        self.request, self.authority = request, authority
        self.resolve_context, self.operation, self.build = resolve_context, operation, build
        self.error_response, self.access_denied = error_response, access_denied
        self.timeout_seconds, self.prepare = timeout_seconds, prepare

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
                            delivery.begin(check, min(request_timeout.when(), context_deadline),
                                           self.access_denied)
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
