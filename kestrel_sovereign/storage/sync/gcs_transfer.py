"""Blocking GCS calls that a cancelled snapshot pass can interrupt (#4028).

``google-cloud-storage`` is synchronous, so the GCS target runs each call on a
worker thread. Cancelling the task that awaits such a thread does not stop the
thread: it goes on uploading, and the interpreter waits for it at exit. A host
whose shutdown budget cancelled a 2 GB upload therefore still ended in SIGKILL.

:func:`run_gcs_transfer` runs one call on a thread of its own under a
:class:`GCSTransfer`. Cancelling the awaiting task aborts the transfer: every
socket the call is using is shut down, so a send or receive blocked on the
network returns at once, and the next request the call makes is refused. The
task then waits for the thread to finish before the cancellation propagates.

Only the requests made through a session passed to
:func:`make_session_abortable` can be interrupted. Anything else the call
blocks on (a DNS lookup, a credential refresh, a retry back-off sleep) is
waited for up to :data:`GCS_TRANSFER_ABORT_GRACE_SECONDS`. The thread is a
daemon so that a call still blocked after that ends with the process instead
of holding it alive; it is reported by name.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import socket
import threading
import weakref
from typing import Callable, Optional, TypeVar

import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool

from kestrel_sovereign.execution_custody import (
    execution_terminal_error,
    is_execution_control_error,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

#: How long a cancelled transfer's thread is waited for once its sockets are
#: shut down. An interrupted request returns in milliseconds; this covers the
#: rest (a retry back-off sleep, a DNS lookup) and must fit in the host's
#: shutdown reserve (``process_manager.HOST_SHUTDOWN_RESERVE_SECONDS``).
GCS_TRANSFER_ABORT_GRACE_SECONDS = 5.0

#: The prefix of every transfer thread's name, which carries the target.
GCS_TRANSFER_THREAD_PREFIX = "gcs-transfer:"


class GCSTransferAborted(Exception):
    """A request of an aborted GCS transfer was refused.

    Not an ``OSError`` or ``ConnectionError``: the library's retry predicate
    must not retry it.
    """


class GCSTransfer:
    """One blocking GCS call's abort switch and the connections it uses."""

    def __init__(self, label: str):
        self.label = label
        self._lock = threading.Lock()
        self._aborted = False
        self._connections: "weakref.WeakSet[HTTPConnection]" = weakref.WeakSet()

    @property
    def aborted(self) -> bool:
        return self._aborted

    def claim(self, connection: HTTPConnection) -> None:
        """Bind ``connection`` to this transfer for its next request.

        Raises :class:`GCSTransferAborted` once the transfer is aborted.
        """
        with self._lock:
            if self._aborted:
                raise GCSTransferAborted(f"GCS transfer to {self.label} was aborted")
            self._connections.add(connection)

    def release(self, connection: HTTPConnection) -> None:
        with self._lock:
            self._connections.discard(connection)

    def abort(self) -> None:
        """Refuse every further request and interrupt the ones in flight."""
        with self._lock:
            self._aborted = True
            connections = list(self._connections)
        for connection in connections:
            _interrupt(connection)


def _interrupt(connection: HTTPConnection) -> None:
    """Shut down ``connection``'s socket so a call blocked on it returns."""
    sock = connection.sock
    if sock is None:
        # Still connecting: ``connect`` sees the abort once it returns.
        return
    try:
        # The plain-socket shutdown, also for a TLS socket: it wakes a thread
        # blocked in that socket's send or receive, whose own error then ends
        # the request. ``SSLSocket.shutdown`` would also discard the TLS
        # object that thread is still using.
        socket.socket.shutdown(sock, socket.SHUT_RDWR)
    except OSError:
        # Already closed or disconnected: nothing is blocked on it.
        pass


_worker = threading.local()


def _current_transfer() -> Optional[GCSTransfer]:
    return getattr(_worker, "transfer", None)


class _AbortableConnection:
    """A connection that belongs to the transfer of the thread using it.

    A pooled connection moves to whichever transfer makes its next request.
    """

    _gcs_transfer: Optional[GCSTransfer] = None

    def _claim_for_current_transfer(self) -> Optional[GCSTransfer]:
        transfer = _current_transfer()
        previous = self._gcs_transfer
        if previous is not None and previous is not transfer:
            previous.release(self)
        self._gcs_transfer = transfer
        if transfer is not None:
            transfer.claim(self)
        return transfer

    def connect(self) -> None:
        transfer = self._claim_for_current_transfer()
        super().connect()
        if transfer is not None and transfer.aborted:
            # Aborted while connecting, before there was a socket to shut.
            self.close()
            raise GCSTransferAborted(f"GCS transfer to {transfer.label} was aborted")

    def request(self, *args, **kwargs):
        self._claim_for_current_transfer()
        return super().request(*args, **kwargs)


class _AbortableHTTPConnection(_AbortableConnection, HTTPConnection):
    pass


class _AbortableHTTPSConnection(_AbortableConnection, HTTPSConnection):
    pass


class _AbortableHTTPConnectionPool(HTTPConnectionPool):
    ConnectionCls = _AbortableHTTPConnection


class _AbortableHTTPSConnectionPool(HTTPSConnectionPool):
    ConnectionCls = _AbortableHTTPSConnection


_ABORTABLE_POOL_CLASSES = {
    "http": _AbortableHTTPConnectionPool,
    "https": _AbortableHTTPSConnectionPool,
}


class _AbortableHTTPAdapter(HTTPAdapter):
    """``requests``' default adapter, opening abortable connections."""

    def init_poolmanager(self, *args, **kwargs) -> None:
        super().init_poolmanager(*args, **kwargs)
        self.poolmanager.pool_classes_by_scheme = _ABORTABLE_POOL_CLASSES

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        if proxy.lower().startswith("socks"):
            raise requests.exceptions.InvalidProxyURL(
                f"a GCS transfer through the SOCKS proxy {proxy} cannot be "
                "interrupted; use an HTTP(S) proxy"
            )
        manager = super().proxy_manager_for(proxy, **proxy_kwargs)
        manager.pool_classes_by_scheme = _ABORTABLE_POOL_CLASSES
        return manager


def make_session_abortable(session: requests.Session) -> None:
    """Make every request ``session`` sends interruptible by its transfer.

    ``session`` is a GCS client's transport. Only ``requests``' default
    adapter is replaced; any other (a mutual-TLS adapter) is refused rather
    than left uninterruptible.
    """
    for prefix, adapter in list(session.adapters.items()):
        if isinstance(adapter, _AbortableHTTPAdapter):
            continue
        if type(adapter) is not HTTPAdapter:
            raise TypeError(
                f"GCS transfers over {type(adapter).__name__} (mounted at "
                f"{prefix!r}) cannot be interrupted"
            )
        adapter.close()
        session.mount(prefix, _AbortableHTTPAdapter(max_retries=adapter.max_retries))


async def run_gcs_transfer(operation: Callable[[], _T], *, label: str) -> _T:
    """Run the blocking ``operation`` on a thread of its own and return its
    result. ``label`` names the target in the thread's name and in reports.

    Cancelling the awaiting task aborts the transfer and waits for the thread
    (at most :data:`GCS_TRANSFER_ABORT_GRACE_SECONDS`) before the cancellation
    propagates; the thread's own outcome is discarded.
    """
    transfer = GCSTransfer(label)
    outcome: "concurrent.futures.Future[_T]" = concurrent.futures.Future()

    def work() -> None:
        if not outcome.set_running_or_notify_cancel():
            return
        _worker.transfer = transfer
        try:
            result = operation()
        except BaseException as error:  # noqa: BLE001 - delivered to the awaiting task
            outcome.set_exception(error)
        else:
            outcome.set_result(result)
        finally:
            _worker.transfer = None

    worker = threading.Thread(
        target=work, name=f"{GCS_TRANSFER_THREAD_PREFIX}{label}", daemon=True
    )
    waiter = asyncio.wrap_future(outcome)
    worker.start()
    try:
        await asyncio.wait({waiter})
    except asyncio.CancelledError as cancellation:
        transfer.abort()
        raise await _join_aborted(waiter, worker, transfer, cancellation)
    # The thread has delivered its outcome; all that is left is its return.
    worker.join()
    return waiter.result()


async def _join_aborted(
    waiter: "asyncio.Future[_T]",
    worker: threading.Thread,
    transfer: GCSTransfer,
    cancellation: asyncio.CancelledError,
) -> BaseException:
    """Wait for an aborted transfer's thread; return what to raise.

    A repeated cancellation does not cut the wait short. One carrying
    process-control evidence is raised along with the first.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + GCS_TRANSFER_ABORT_GRACE_SECONDS
    controls: list[BaseException] = []
    while not waiter.done():
        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        try:
            await asyncio.wait({waiter}, timeout=remaining)
        except asyncio.CancelledError as repeated:
            if is_execution_control_error(repeated):
                controls.append(repeated)
    if waiter.done():
        # Retrieved: the cancellation, not the aborted call's error, is the
        # outcome.
        waiter.exception()
        worker.join()
    else:
        logger.warning(
            "GCS transfer to %s did not stop within %.1fs of its cancellation; "
            "its thread %s is left to end with the process",
            transfer.label,
            GCS_TRANSFER_ABORT_GRACE_SECONDS,
            worker.name,
        )
        waiter.add_done_callback(_retrieve_outcome)
    if controls:
        return execution_terminal_error(cancellation, *controls, retain_originals=True)
    return cancellation


def _retrieve_outcome(waiter: "asyncio.Future[object]") -> None:
    if not waiter.cancelled():
        waiter.exception()


__all__ = [
    "GCS_TRANSFER_ABORT_GRACE_SECONDS",
    "GCS_TRANSFER_THREAD_PREFIX",
    "GCSTransfer",
    "GCSTransferAborted",
    "make_session_abortable",
    "run_gcs_transfer",
]
