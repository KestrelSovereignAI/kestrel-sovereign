"""Original synchronous restrictions at tested native HTTPX/TLS boundaries.

This optional boundary instruments one existing native HTTPX client/pool. It
does not create authority, a client, a pool, an HTTP parser or a retry policy.
Only direct verified TLS1.3, HTTP/1.1 and the exact tested native backend are
supported. Async native admission cannot be evaluated here and is refused.
"""

from __future__ import annotations

import asyncio
import _socket
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import inspect
import math
import socket
import ssl
import threading
from typing import TypeVar

import httpcore
from httpcore._backends.sync import SyncBackend, SyncStream
import httpx
from httpx._client import UseClientDefault

from kestrel_sovereign._async_ownership import await_owned_operation
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    bind_execution_custody_snapshot,
    capture_native_execution_admission,
    current_execution_custody,
    execution_terminal_error,
    require_execution_work,
)

NATIVE_SYNC_HTTP_ADMISSION_API_VERSION = 2
_Result = TypeVar("_Result")


class NativeSyncHTTPOutcomeUnknown(ExecutionAuthorityError):
    """Original HTTP bytes may have reached the peer; never authorize replay."""

    native_http_outcome = "unknown"

    def __init__(self, original: BaseException, *, native_result=None):
        self.original_error = original
        # PRIVATE original facts for the host's existing custody owner only.
        # This is not delivery permission and is never included in error text.
        self.native_result = native_result
        super().__init__("original synchronous HTTP outcome requires reconciliation")


@dataclass(eq=False)
class _Restriction:
    client: httpx.Client
    current: Callable[[], None]
    before_request_write: Callable[[], None] | None = None
    active: bool = True

    def check(self):
        if not self.active:
            raise ExecutionAuthorityError(
                "original synchronous HTTP restriction retired"
            )
        _check_sync_restriction(self.current)

    def check_request_write(self):
        if not self.active:
            raise ExecutionAuthorityError(
                "original synchronous HTTP restriction retired"
            )
        if self.before_request_write is not None:
            _check_sync_restriction(self.before_request_write)


def _check_sync_restriction(check):
    result = check()
    if result is not None:
        if inspect.iscoroutine(result):
            result.close()
        raise TypeError("synchronous HTTP restriction must return None or raise")


_RESTRICTIONS: ContextVar[tuple[_Restriction, ...]] = ContextVar(
    "native_sync_http_restrictions", default=()
)
_REQUEST: ContextVar[_Request | None] = ContextVar(
    "original_native_sync_http_request", default=None
)
_THREAD: ContextVar[_ThreadOperation | None] = ContextVar(
    "original_native_sync_http_thread", default=None
)


def _require_sync_work(original_validators=()):
    # Refusing is essential: sampling or skipping an async gate in this thread
    # would discard original authority. There is no asyncio.run bridge.
    if original_validators or capture_native_execution_admission():
        raise ExecutionAuthorityError(
            "synchronous HTTP cannot evaluate original async native admission"
        )
    require_execution_work()


@dataclass(eq=False)
class _SocketBorrow:
    socket: ssl.SSLSocket
    active: bool = True
    lock: threading.Lock = field(default_factory=threading.Lock)

    def retire(self):
        with self.lock:
            self.active = False

    def cancel(self):
        # A queued cancellation of an old borrow cannot close a connection
        # already returned to the pool and borrowed by another request.
        with self.lock:
            if self.active:
                try:
                    # SSLSocket.shutdown clears _sslobj before the kernel
                    # shutdown. A concurrently admitted SSLSocket.send could
                    # then use its plaintext fallback. Interrupt only the raw
                    # fd; the native writer must retain its TLS state.
                    _socket.socket.shutdown(self.socket, socket.SHUT_RDWR)
                except OSError:
                    pass


@dataclass(eq=False)
class _ThreadOperation:
    active: bool = True
    sent: bool = False
    sockets: set[_SocketBorrow] = field(default_factory=set)
    requests: set[_Request] = field(default_factory=set)

    def cancel(self, _task):
        # Do NOT cancel asyncio.to_thread's Task: that loses the running native
        # thread. Retire IO, interrupt exact retained sockets, then join it.
        self.active = False
        for stream in tuple(self.sockets):
            stream.cancel()


@dataclass(eq=False)
class _Request:
    client: httpx.Client
    restrictions: tuple[_Restriction, ...]
    scopes: tuple
    validators: tuple
    thread: _ThreadOperation | None
    sent: bool = False
    borrows: dict[ssl.SSLSocket, _SocketBorrow] = field(default_factory=dict)
    response: httpx.Response | None = None

    def retain_socket(self, original_socket):
        if self.thread is not None and original_socket not in self.borrows:
            borrow = _SocketBorrow(original_socket)
            self.borrows[original_socket] = borrow
            self.thread.sockets.add(borrow)

    def retire_sockets(self):
        # Do this BEFORE native response.close returns the connection to its
        # pool. Each borrow has its own lock against an already queued cancel.
        for borrow in self.borrows.values():
            borrow.retire()
            if self.thread is not None:
                self.thread.sockets.discard(borrow)
        self.borrows.clear()

    def check(self):
        if self.thread is not None and not self.thread.active:
            raise ExecutionAuthorityError("original synchronous HTTP thread is retiring")
        with bind_execution_custody_snapshot(self.scopes):
            _require_sync_work(self.validators)
            if not self.restrictions:
                raise ExecutionAuthorityError("synchronous HTTP requires an original restriction")
            for restriction in self.restrictions:
                restriction.check()
            # Restriction callbacks can synchronously retire the original turn.
            # They never supply or replace Core's native execution admission.
            _require_sync_work(self.validators)
            if self.thread is not None and not self.thread.active:
                raise ExecutionAuthorityError("original synchronous HTTP thread is retiring")


def _request(client):
    request = _REQUEST.get()
    if request is None or request.client is not client:
        raise ExecutionAuthorityError("native synchronous IO has no original request")
    request.check()
    return request


class _Stream(httpcore.NetworkStream):
    """Delegate the native stream and preserve its original partial-write loop."""

    def __init__(self, native: SyncStream, client: httpx.Client, *, tls=False):
        if type(native) is not SyncStream:
            raise TypeError("synchronous HTTP requires the actual native stream")
        self.native, self.client, self.tls = native, client, tls
        if tls:
            original_socket = native.get_extra_info("socket")
            if (
                type(original_socket) is not ssl.SSLSocket
                or original_socket.version() != "TLSv1.3"
            ):
                native.close()
                raise TypeError("synchronous HTTP requires native TLS1.3")
            original_send = original_socket.send

            def send(buffer, flags=0):
                request = _request(client)
                request.retain_socket(original_socket)
                # This is the actual native handoff, repeated by SyncStream's
                # existing loop for every partial send. No await/observer here.
                request.check()
                for restriction in request.restrictions:
                    restriction.check_request_write()
                # The extra predicate may itself retire an original grant.
                # Recheck the original owner and Core fence before any bytes.
                request.check()
                request.sent = True
                if request.thread is not None:
                    request.thread.sent = True
                return original_send(buffer, flags)

            original_socket.send = send

    def read(self, max_bytes, timeout=None):
        _request(self.client)
        data = self.native.read(max_bytes, timeout)
        _request(self.client)
        return data

    def write(self, buffer, timeout=None):
        _request(self.client)
        if not self.tls:
            raise TypeError("synchronous HTTP refuses plaintext native writes")
        return self.native.write(buffer, timeout)

    def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        _request(self.client)
        if (
            self.tls
            or not isinstance(ssl_context, ssl.SSLContext)
            or not ssl_context.check_hostname
            or ssl_context.verify_mode != ssl.CERT_REQUIRED
        ):
            raise TypeError("synchronous HTTP requires direct verified native TLS")
        native = self.native.start_tls(ssl_context, server_hostname, timeout)
        # Wrapping owns the returned socket before any post-handshake denial.
        result = _Stream(native, self.client, tls=True)
        try:
            _request(self.client)
        except BaseException:
            result.close()
            raise
        return result

    def close(self):
        return self.native.close()  # Physical retirement never requires a grant.

    def get_extra_info(self, info):
        return self.native.get_extra_info(info)


class _Backend(httpcore.NetworkBackend):
    def __init__(self, native: SyncBackend, client: httpx.Client):
        self.native, self.client = native, client

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        _request(self.client)
        native = self.native.connect_tcp(host, port, timeout, local_address, socket_options)
        result = _Stream(native, self.client)
        try:
            _request(self.client)
        except BaseException:
            result.close()
            raise
        return result

    def connect_unix_socket(self, *args, **kwargs):
        raise TypeError("synchronous HTTP refuses Unix socket transport")


class _Body(httpx.SyncByteStream):
    def __init__(self, response, original, request):
        self.response, self.original, self.request = response, original, request

    def __iter__(self):
        token = _REQUEST.set(self.request)
        try:
            self.request.check()
            for data in self.original:
                self.request.check()
                yield data
        except BaseException as original:
            try:
                self.response.close()
            except BaseException as cleanup:
                raise execution_terminal_error(original, cleanup, retain_originals=True)
            raise
        finally:
            _REQUEST.reset(token)

    def close(self):
        self.request.retire_sockets()
        self.original.close()
        if self.request.thread is not None:
            self.request.thread.requests.discard(self.request)
        self.request.check()  # Native close can block after the last body read.


@dataclass(frozen=True)
class _Installation:
    transport: httpx.HTTPTransport
    pool: httpcore.ConnectionPool
    backend: _Backend
    send: Callable
    dispatch: Callable


def _bounded_timeouts(values):
    for value in values:
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 30:
            raise TypeError("synchronous HTTP requires finite native timeouts at most 30 seconds")


def _configuration(client, installed=None):
    if type(client) is not httpx.Client:
        raise TypeError("synchronous admission requires the actual native HTTPX Client")
    if (httpx.__version__, httpcore.__version__) != ("0.28.1", "1.0.9"):
        raise TypeError("synchronous HTTP requires the tested native HTTPX/HTTPcore versions")
    transport = client._transport
    pool = getattr(transport, "_pool", None)
    if (
        type(transport) is not httpx.HTTPTransport
        or type(pool) is not httpcore.ConnectionPool
        or any(value is not None for value in client._mounts.values())
        or pool._http2
        or not pool._http1
        or pool._proxy is not None
        or pool._uds is not None
        or pool._retries != 0
        or client.is_closed
        or client.follow_redirects
        or client.auth is not None
        or client.cookies
    ):
        raise TypeError("unsupported native synchronous HTTP transport configuration")
    _bounded_timeouts(
        (client.timeout.connect, client.timeout.read, client.timeout.write, client.timeout.pool)
    )
    if installed is None:
        if (
            type(pool._network_backend) is not SyncBackend
            or pool.connections
            or pool._requests
            or getattr(client.send, "__func__", None) is not httpx.Client.send
            or getattr(client._send_single_request, "__func__", None)
            is not httpx.Client._send_single_request
        ):
            raise TypeError("synchronous HTTP refuses previously used or replaced native dispatch")
    elif (
        type(installed) is not _Installation
        or transport is not installed.transport
        or pool is not installed.pool
        or pool._network_backend is not installed.backend
        or client.send is not installed.send
        or client._send_single_request is not installed.dispatch
    ):
        raise TypeError("synchronous HTTP native boundary was replaced")
    return transport, pool


def _install(client):
    installed = getattr(client, "_kestrel_sync_admission", None)
    transport, pool = _configuration(client, installed)
    if installed is not None:
        return
    native_send, native_dispatch = client.send, client._send_single_request

    def send(*args, **kwargs):
        _configuration(client, client._kestrel_sync_admission)
        authentication = kwargs.get("auth")
        redirect = kwargs.get("follow_redirects", False)
        if (authentication is not None and type(authentication) is not UseClientDefault) or (
            redirect is not False and type(redirect) is not UseClientDefault
        ):
            raise TypeError("synchronous HTTP refuses authentication flows and redirect replay")
        request = _Request(
            client,
            tuple(value for value in _RESTRICTIONS.get() if value.client is client),
            current_execution_custody(),
            capture_native_execution_admission(),
            _THREAD.get(),
        )
        request.check()
        token = _REQUEST.set(request)
        try:
            response = native_send(*args, **kwargs)
            request.check()
            return response
        except BaseException as original:
            if request.sent and not isinstance(original, NativeSyncHTTPOutcomeUnknown):
                raise NativeSyncHTTPOutcomeUnknown(
                    original, native_result=request.response
                ) from original
            raise
        finally:
            if request.response is None or request.response.is_closed:
                request.retire_sockets()
            _REQUEST.reset(token)

    def dispatch(request):
        _configuration(client, client._kestrel_sync_admission)
        original = _request(client)
        if request.url.scheme != "https":
            raise TypeError("synchronous HTTP requires HTTPS")
        if "cookie" in request.headers:
            raise TypeError("synchronous HTTP refuses cookie-bearing requests")
        timeout = request.extensions.get("timeout")
        if type(timeout) is not dict or set(timeout) != {"connect", "read", "write", "pool"}:
            raise TypeError("synchronous HTTP requires explicit native timeouts")
        _bounded_timeouts(timeout.values())
        response = native_dispatch(request)
        original.response = response
        if original.thread is not None:
            original.thread.requests.add(original)
        response.stream = _Body(response, response.stream, original)
        try:
            original.check()
        except BaseException as denial:
            try:
                response.close()
            except BaseException as cleanup:
                raise execution_terminal_error(denial, cleanup, retain_originals=True)
            raise
        return response

    backend = _Backend(pool._network_backend, client)
    pool._network_backend = backend
    client._send_single_request = dispatch
    client.send = send
    client._kestrel_sync_admission = _Installation(transport, pool, backend, send, dispatch)


@contextmanager
def bind_native_sync_http_execution_admission(
    client: httpx.Client,
    check_current: Callable[[], None],
    *,
    check_before_request_write: Callable[[], None] | None = None,
) -> Iterator[None]:
    """Borrow an original restriction; the installed native client stays gated.

    The callback grants nothing and is never reconstructed from IDs. Copied
    contexts retain denial after this lexical operation exits. Unsupported
    async native gates are refused, not bypassed. An installed client's future
    requests must also have an explicit live original restriction.

    API2 optionally restricts EACH actual native request-byte send further.
    This cannot replace current authority, which remains checked at every
    existing IO/delivery boundary. Spending freshness may expire after a full
    request without discarding its original terminal receipt. The two-argument
    API1 call remains supported; the exported version identifies API2 support.
    """
    if not callable(check_current) or inspect.iscoroutinefunction(check_current):
        raise TypeError("synchronous HTTP requires an original synchronous restriction")
    if check_before_request_write is not None and (
        not callable(check_before_request_write)
        or inspect.iscoroutinefunction(check_before_request_write)
    ):
        raise TypeError(
            "synchronous HTTP requires a synchronous request-write restriction"
        )
    _require_sync_work()
    restriction = _Restriction(client, check_current, check_before_request_write)
    restriction.check()
    _install(client)
    token = _RESTRICTIONS.set((*_RESTRICTIONS.get(), restriction))
    try:
        yield
    finally:
        restriction.active = False
        _RESTRICTIONS.reset(token)


async def await_native_sync_http(
    client: httpx.Client, operation: Callable[[httpx.Client], _Result]
) -> _Result:
    """Run/join native work on its original thread through caller cancellation.

    The caller retains its SAME original admission outside this await. The
    operation consumes/closes its responses in that thread; never return an
    unconsumed stream or detach a worker. Cancellation fences/shuts down only
    this operation's exact retained sockets and physically joins the thread.
    """
    if not callable(operation) or inspect.iscoroutinefunction(operation):
        raise TypeError("original native synchronous operation required")
    original = tuple(value for value in _RESTRICTIONS.get() if value.client is client)
    if not original:
        raise ExecutionAuthorityError("synchronous HTTP requires an original restriction")
    for restriction in original:
        restriction.check()
    _require_sync_work()
    thread = _ThreadOperation()
    token = _THREAD.set(thread)

    def run():
        result, terminal = None, None
        try:
            result = operation(client)
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                raise TypeError("native synchronous operation returned an awaitable")
        except BaseException as original_error:
            terminal = original_error
        pending = tuple(thread.requests)
        for request in pending:
            try:
                request.response.close()
            except BaseException as cleanup:
                terminal = execution_terminal_error(terminal, cleanup, retain_originals=True)
        if pending and terminal is None:
            terminal = NativeSyncHTTPOutcomeUnknown(
                TypeError("native synchronous operation left an unconsumed response")
            )
        if terminal is not None:
            raise terminal
        return result

    try:
        outcome = await await_owned_operation(
            asyncio.to_thread(run),
            retain_originals=True,
            cancel_on_caller_cancellation=thread.cancel,
        )
        terminal = execution_terminal_error(
            outcome.error,
            outcome.cancellation,
            retain_originals=True,
            isolate_control_originals=True,
        )
        if terminal is not None:
            raise terminal
        try:
            for restriction in original:
                restriction.check()
            _require_sync_work()
        except BaseException as denial:
            if thread.sent:
                raise NativeSyncHTTPOutcomeUnknown(denial, native_result=outcome.result) from denial
            raise
        return outcome.result
    finally:
        thread.active = False
        _THREAD.reset(token)
