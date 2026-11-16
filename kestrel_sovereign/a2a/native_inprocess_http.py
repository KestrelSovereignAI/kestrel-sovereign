"""Incremental ASGI I/O over Core's original callback-source lifetime.

This is an HTTP transport, not another native codec or authorization policy.
The original SDK still encodes requests and the published recipient routes
still authenticate them. Only NativeA2APublication constructs the binding.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import httpx
from kestrel_sdk.native_a2a.utils.source_ownership import CallbackSource, SourceAcquisition
from starlette.types import ASGIApp, Message, Scope

from kestrel_sovereign._async_ownership import (
    OwnedAsyncIterator,
    OwnedOperation,
    OwnedTaskAdmission,
    await_owned_task,
    create_owned_iterator,
    create_owned_task,
    pending_owned_retirement_error,
    raise_owned_outcome,
    wait_owned_tasks_until,
)
from kestrel_sovereign.execution_custody import (
    execution_observation_error,
    retain_native_execution_stream_close,
)


@dataclass(frozen=True)
class _ResponseStart:
    status: int
    headers: tuple[tuple[bytes, bytes], ...]


@dataclass(frozen=True)
class _RequestFailure:
    """Deliver the original receive operation, never a protocol error projection."""

    operation: OwnedOperation


def _deadline(request: httpx.Request, key: str) -> float:
    values = request.extensions.get("timeout")
    value = values.get(key) if isinstance(values, Mapping) else None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"Native in-process HTTP requires a finite nonnegative {key} timeout")
    return float(value)


class _Exchange:
    """One original application/source and its single physical close attempt."""

    def __init__(self, transport: _NativeInProcessHTTPTransport, request: httpx.Request, scope: Scope):
        self.transport, self.request, self.scope = transport, request, scope
        self.request_stream = request.stream
        self.read_timeout = _deadline(request, "read")
        self.write_timeout = _deadline(request, "write")
        self.disconnected = asyncio.Event()
        self.response_finished = False
        self.source: OwnedAsyncIterator[_ResponseStart | _RequestFailure | bytes] | None = None
        self.request_source: OwnedAsyncIterator[bytes] | None = None
        self.request_read: OwnedOperation | None = None
        self.request_complete = False
        self.receiving = False
        self.close_admission: OwnedTaskAdmission | None = None
        self.close_task: asyncio.Task | None = None
        self.close_opening: BaseException | None = None
        self.retirement_deadline: float | None = None

    async def pull_request(self) -> Message:
        """A finite write observation, not the physical input iterator's task."""
        self.transport.require_binding(self.scope)
        if self.request_source is None:
            self.request_source = await create_owned_iterator(
                source_factory=lambda resources: self.request_stream,
                operation="native-inprocess-http-upload",
            )
        try:
            async with asyncio.timeout(self.write_timeout):
                chunk = await self.request_source.__anext__(defer_cancellation=False)
        except StopAsyncIteration:
            self.transport.require_binding(self.scope)
            self.request_complete = True
            return {"type": "http.request", "body": b"", "more_body": False}
        except TimeoutError as error:
            if self.retirement_deadline is None:
                self.retirement_deadline = asyncio.get_running_loop().time() + self.write_timeout
            raise httpx.WriteTimeout("Native ASGI request body deadline exceeded", request=self.request) from error
        if not isinstance(chunk, bytes):
            raise TypeError("Native ASGI request body chunks must be bytes")
        self.transport.require_binding(self.scope)
        return {"type": "http.request", "body": chunk, "more_body": True}

    async def receive(self, emit) -> Message:
        if self.receiving:
            raise RuntimeError("Native ASGI request body requires one receive observer")
        self.receiving = True
        try:
            if self.disconnected.is_set():
                return {"type": "http.disconnect"}
            if self.request_complete:
                await self.disconnected.wait()
                return {"type": "http.disconnect"}
            if self.request_read is None:
                self.request_read = OwnedOperation(
                    self.pull_request(), retirement_resource=self, retain_originals=True,
                    name="native-inprocess-http-receive",
                )
            outcome = await self.request_read.observe(defer_cancellation=False)
            if outcome.error is not None or outcome.cancellation is not None:
                if self.request_source is not None:
                    self.request_source.request_stop()
                # A recipient may catch receive errors and send a protocol
                # error reply. Wake the HTTP observer FIRST with the actual
                # retained operation via the SAME bounded frame handoff. Its
                # physical failure cannot disappear into that projection.
                await emit(_RequestFailure(self.request_read))
                return raise_owned_outcome(outcome, operation="native-inprocess-http-receive",
                                           retain_originals=True, retain_observer_cancellations=True)
            self.request_read = None  # This exact successful owner was joined.
            return outcome.result
        finally:
            self.receiving = False

    async def retire_request(self) -> None:
        errors = []
        try:
            if self.request_source is None:
                await self.request_stream.aclose()
            else:
                await self.request_source.aclose()
        except BaseException as error:
            errors.append(error)
        if self.request_read is not None:
            outcome = await self.request_read.observe()
            errors.extend((outcome.error, outcome.cancellation))
        terminal = execution_observation_error(*errors)
        if terminal is not None:
            raise terminal

    def acquire(self, resources: SourceAcquisition) -> CallbackSource[_ResponseStart | _RequestFailure | bytes]:
        # The original app owner retires the original input source, including
        # a stream which the app never read. A stream that WAS acquired is
        # iterated and closed only by its own canonical source owner.
        resources.retain(self.request_stream, self.retire_request)
        response_started = False

        async def produce(emit):
            async def publish(frame: _ResponseStart | bytes) -> bool:
                if await emit(frame):
                    return True
                self.disconnected.set()
                return False

            async def send(message: Message) -> None:
                nonlocal response_started
                if self.disconnected.is_set():
                    # ASGI 2.0 withdrawal is a receive-side disconnect, not an
                    # exception injected through opaque middleware/finalizers.
                    # Do not admit any later frames. Yield even if an application
                    # ignores disconnect and keeps sending: close still has its
                    # finite budget and retains that original as private debt.
                    await asyncio.sleep(0)
                    return
                self.transport.require_binding(self.scope)
                if message["type"] == "http.response.start":
                    if response_started or self.response_finished:
                        raise RuntimeError("Native ASGI response started more than once")
                    status = message["status"]
                    if isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599:
                        raise ValueError("Native ASGI response status is invalid")
                    headers = tuple(message.get("headers", ()))
                    if any(not isinstance(k, bytes) or not isinstance(v, bytes) for k, v in headers):
                        raise TypeError("Native ASGI response headers must be byte pairs")
                    response_started = True
                    await publish(_ResponseStart(status, headers))
                    return
                if message["type"] != "http.response.body":
                    raise ValueError("Unsupported native ASGI response message")
                if not response_started or self.response_finished:
                    raise RuntimeError("Native ASGI response body is outside its original response")
                data = message.get("body", b"")
                if not isinstance(data, bytes):
                    raise TypeError("Native ASGI response body must be bytes")
                if self.request.method != "HEAD":
                    for offset in range(0, len(data), self.transport.buffer_bytes):
                        # Canonical SourceHandoff admits only one unacknowledged
                        # frame. The next pull ACKs it; no response body list.
                        if not await publish(data[offset:offset + self.transport.buffer_bytes]):
                            return
                if not message.get("more_body", False):
                    self.response_finished = True
                    self.disconnected.set()

            # Routers mutate their per-dispatch scope (e.g. Mount rewrites
            # root_path). Revalidate the retained ingress route, not that
            # mutable child scope, on each outward frame/read.
            async def receive() -> Message:
                return await self.receive(emit)

            await self.transport.application(dict(self.scope), receive, send)
            if (not response_started or not self.response_finished) and not self.disconnected.is_set():
                raise RuntimeError("Native ASGI application returned an incomplete response")

        return CallbackSource(produce)

    async def next(self) -> _ResponseStart | bytes:
        if self.source is None:
            raise RuntimeError("Native ASGI source was not acquired")
        self.transport.require_binding(self.scope)
        try:
            async with asyncio.timeout(self.read_timeout):
                # This read borrows terminal observation; the retained exchange
                # close owns physical retirement. Concurrent Stop must not turn
                # a finite read into an unbounded must-join of opaque finally.
                value = await self.source.__anext__(defer_cancellation=False)
        except TimeoutError as error:
            raise httpx.ReadTimeout("Native ASGI response deadline exceeded", request=self.request) from error
        self.transport.require_binding(self.scope)
        if isinstance(value, _RequestFailure):
            outcome = await value.operation.observe()
            raise_owned_outcome(outcome, operation="native-inprocess-http-receive",
                                retain_originals=True, retain_observer_cancellations=True)
            raise RuntimeError("Native ASGI upload failure has no original failure")
        return value

    async def close(self) -> None:
        if self.source is None:
            return
        self.disconnected.set()
        self.source.request_stop()
        if self.request_source is not None:
            self.request_source.request_stop()
        if self.close_admission is None:
            self.close_admission = OwnedTaskAdmission()
            if self.retirement_deadline is None:
                self.retirement_deadline = asyncio.get_running_loop().time() + self.read_timeout
            try:
                self.close_task = create_owned_task(
                    self.source.aclose(), admission=self.close_admission,
                    retirement_resource=self, retain_originals=True,
                )
            except BaseException as error:
                self.close_opening = error
                self.close_task = self.close_admission.task
        sources = (self.source,) if self.request_source is None else (self.source, self.request_source)
        source_tasks = tuple(task for source in sources for task in (source.producer_task, source.owner_task))
        if self.request_read is not None and self.request_read.task is not None:
            source_tasks = (*source_tasks, self.request_read.task)
        original_tasks = ((self.close_task, *source_tasks)
                          if self.close_task is not None else source_tasks)
        observed = await wait_owned_tasks_until(original_tasks, deadline=self.retirement_deadline)
        debt = pending_owned_retirement_error(
            self, original_tasks,
            message="Original native ASGI source retirement is still pending",
        )
        request_outcome = None
        if self.request_read is not None and (self.request_read.task is None or self.request_read.task.done()):
            request_outcome = await self.request_read.observe()
        request_error = (execution_observation_error(request_outcome.error, request_outcome.cancellation)
                         if request_outcome is not None else None)
        if debt is not None:
            raise execution_observation_error(
                self.close_opening, *(source.terminal_error for source in sources), request_error,
                observed.cancellation, debt,
            )
        # An accepted close task can fail publication before its body runs, or
        # allocation can refuse it entirely. The SAME source owners can still
        # finish their original cleanup. Harvest those outcomes independently
        # so a later physical failure never disappears behind admission failure.
        source_outcomes = [await source.join_source_owners(closing=True) for source in sources]
        outcome = await await_owned_task(self.close_task) if self.close_task is not None else None
        terminal = execution_observation_error(
            self.close_opening, *(error for source in source_outcomes for error in (source.error, source.cancellation)),
            request_error,
            outcome.error if outcome is not None else None,
            outcome.cancellation if outcome is not None else None,
            observed.cancellation,
        )
        # Settled failure is not pending physical retirement. Concurrent
        # observers still retain this exact exchange/close task and outcome.
        if self.close_task is not None:
            self.transport.exchanges.discard(self)
        if terminal is not None:
            raise terminal


class _ResponseBody(httpx.AsyncByteStream):
    def __init__(self, exchange: _Exchange):
        self.exchange = exchange

    async def __aiter__(self):
        while True:
            try:
                value = await self.exchange.next()
            except StopAsyncIteration:
                return
            if not isinstance(value, bytes):
                raise RuntimeError("Native ASGI response emitted another header block")
            yield value

    async def aclose(self) -> None:
        await self.exchange.close()


class _NativeInProcessHTTPTransport(httpx.AsyncBaseTransport):
    """A publication-bound transport with no network fallback or ASGI lifespan."""

    def __init__(self, *, application: ASGIApp, public_base_url: str,
                 require_binding: Callable[[Scope], None], buffer_bytes: int = 65536):
        if isinstance(buffer_bytes, bool) or not isinstance(buffer_bytes, int) or buffer_bytes <= 0:
            raise ValueError("Native ASGI frame capacity must be a positive integer")
        self.application, self.require_binding = application, require_binding
        self.base = httpx.URL(public_base_url)
        self.buffer_bytes = buffer_bytes
        self.exchanges: set[_Exchange] = set()
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.closed:
            raise RuntimeError("Native in-process HTTP transport was retired")
        if (request.url.scheme, request.url.host, request.url.port) != (self.base.scheme, self.base.host, self.base.port):
            raise ValueError("Native in-process HTTP request differs from its selected origin")
        if request.url.username or request.url.password or request.url.fragment:
            raise ValueError("Native in-process HTTP request contains an unsupported authority")
        if request.headers.get("host", "").lower() != request.url.netloc.decode("ascii").lower():
            raise ValueError("Native in-process HTTP Host differs from its original URL")
        if not isinstance(request.stream, httpx.AsyncByteStream):
            raise TypeError("Native in-process HTTP requires its original async request body")
        scope: Scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.0"},
            "http_version": "1.1", "method": request.method,
            "headers": [(k.lower(), v) for k, v in request.headers.raw],
            "scheme": request.url.scheme, "path": request.url.path,
            "raw_path": request.url.raw_path.split(b"?", 1)[0],
            "query_string": request.url.query, "server": (request.url.host, request.url.port),
            "client": None, "root_path": self.base.path.rstrip("/"), "app": self.application,
        }
        self.require_binding(scope)
        exchange = _Exchange(self, request, scope)
        self.exchanges.add(exchange)
        try:
            exchange.source = await create_owned_iterator(
                source_factory=exchange.acquire, operation="native-inprocess-http",
            )
            start = await exchange.next()
            if not isinstance(start, _ResponseStart):
                raise RuntimeError("Native ASGI response body preceded its headers")
            response = httpx.Response(start.status, headers=start.headers, stream=_ResponseBody(exchange))
            # HTTPX closes inside its read-error handler, after marking the
            # response closed. The existing canonical owner retains that exact
            # original failure alongside bounded source-retirement evidence.
            retain_native_execution_stream_close(response)
            return response
        except BaseException as original:
            if exchange.source is None:
                self.exchanges.discard(exchange)
            try:
                await exchange.close()
            except BaseException as retirement:
                raise execution_observation_error(original, retirement)
            raise

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        try:
            await self.aclose()
        except BaseException as cleanup:
            # AsyncClient delegates its context exit to this transport. Its
            # original read/cancellation must survive a later pool close too.
            raise execution_observation_error(exc_value, cleanup)

    async def aclose(self) -> None:
        self.closed = True
        errors = []
        for exchange in tuple(self.exchanges):
            try:
                await exchange.close()
            except BaseException as error:
                errors.append(error)
        terminal = execution_observation_error(*errors)
        if terminal is not None:
            raise terminal
