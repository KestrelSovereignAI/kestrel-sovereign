"""A cancelled GCS transfer stops, and its thread ends with it (#4028).

The GCS target ran its synchronous upload with ``asyncio.to_thread``.
Cancelling the snapshot pass at the host's shutdown budget ended the pass,
but the thread went on uploading and the interpreter waited for it at exit,
so the host was still SIGKILLed. ``run_gcs_transfer`` shuts down the sockets
the call is using, refuses its next request, and joins the thread.

Every server here is a local socket; no test reaches GCS or reads
credentials.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import subprocess
import sys
import textwrap
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
import requests
from google.cloud.storage.retry import DEFAULT_RETRY
from requests.adapters import HTTPAdapter

import kestrel_sovereign
from kestrel_sovereign.storage.sync import gcs_transfer
from kestrel_sovereign.storage.sync.gcs_transfer import (
    GCS_TRANSFER_THREAD_PREFIX,
    GCSTransferAborted,
    make_session_abortable,
    run_gcs_transfer,
)

#: Far longer than any wait below: a thread still blocked when a test checks
#: was not interrupted.
_REQUEST_TIMEOUT_SECONDS = 60.0


def _transfer_threads(label: str) -> list[threading.Thread]:
    name = f"{GCS_TRANSFER_THREAD_PREFIX}{label}"
    return [thread for thread in threading.enumerate() if thread.name == name]


@contextmanager
def _http_server():
    """A keep-alive HTTP/1.1 server: ``/ok`` answers, ``/stall`` never does.

    Yields its URL, an event set when a ``/stall`` request arrives, and the
    connections it has accepted.
    """
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    stalled = threading.Event()
    held: list[socket.socket] = []
    closing = threading.Event()

    def serve(conn: socket.socket) -> None:
        buffered = b""
        while not closing.is_set():
            try:
                chunk = conn.recv(65536)
            except OSError:
                return
            if not chunk:
                return
            buffered += chunk
            while b"\r\n\r\n" in buffered:
                head, buffered = buffered.split(b"\r\n\r\n", 1)
                path = head.split(b" ", 2)[1]
                if path == b"/stall":
                    stalled.set()
                    return  # holds the connection open, answering nothing
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n"
                    b"Connection: keep-alive\r\n\r\nok"
                )

    def accept() -> None:
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            held.append(conn)
            threading.Thread(target=serve, args=(conn,), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{listener.getsockname()[1]}", stalled, held
    finally:
        closing.set()
        listener.close()
        for conn in held:
            conn.close()


@pytest.fixture
def session():
    session = requests.Session()
    make_session_abortable(session)
    yield session
    session.close()


async def _cancel_once(task: asyncio.Task) -> float:
    """Cancel ``task``; return how long its cancellation took to arrive."""
    started = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 30)
    return time.monotonic() - started


async def test_the_result_and_the_error_come_back_from_the_thread():
    assert await run_gcs_transfer(lambda: 42, label="result") == 42

    def fail():
        raise ValueError("bucket missing")

    with pytest.raises(ValueError, match="bucket missing"):
        await run_gcs_transfer(fail, label="error")
    assert not _transfer_threads("result") and not _transfer_threads("error")


async def test_cancelling_interrupts_a_request_blocked_on_the_network(session):
    seen: list[BaseException] = []

    with _http_server() as (url, stalled, _accepted):

        def request():
            try:
                return session.get(f"{url}/stall", timeout=_REQUEST_TIMEOUT_SECONDS)
            except BaseException as error:
                seen.append(error)
                raise

        task = asyncio.create_task(run_gcs_transfer(request, label="blocked"))
        assert await asyncio.to_thread(stalled.wait, 10)
        await asyncio.sleep(0.1)  # into the blocking receive

        elapsed = await _cancel_once(task)

    assert elapsed < gcs_transfer.GCS_TRANSFER_ABORT_GRACE_SECONDS, elapsed
    assert not _transfer_threads("blocked"), "the transfer thread outlived its cancellation"
    assert len(seen) == 1 and isinstance(seen[0], requests.ConnectionError), seen


async def test_a_pooled_connection_belongs_to_the_transfer_using_it(session):
    """A connection an earlier transfer opened is interrupted by the abort of
    the transfer that reuses it."""
    with _http_server() as (url, stalled, accepted):
        first = await run_gcs_transfer(
            lambda: session.get(f"{url}/ok", timeout=_REQUEST_TIMEOUT_SECONDS).text,
            label="first",
        )
        assert first == "ok"
        assert len(accepted) == 1

        task = asyncio.create_task(
            run_gcs_transfer(
                lambda: session.get(f"{url}/stall", timeout=_REQUEST_TIMEOUT_SECONDS),
                label="second",
            )
        )
        assert await asyncio.to_thread(stalled.wait, 10)
        await asyncio.sleep(0.1)
        assert len(accepted) == 1, "the second transfer opened a new connection"

        elapsed = await _cancel_once(task)

    assert elapsed < gcs_transfer.GCS_TRANSFER_ABORT_GRACE_SECONDS, elapsed
    assert not _transfer_threads("second")


async def test_an_aborted_transfer_refuses_its_next_request(session):
    proceed = threading.Event()
    seen: list[BaseException] = []

    with _http_server() as (url, _stalled, _accepted):

        def request_after_the_abort():
            proceed.wait(10)
            try:
                return session.get(f"{url}/ok", timeout=_REQUEST_TIMEOUT_SECONDS)
            except BaseException as error:
                seen.append(error)
                raise

        task = asyncio.create_task(
            run_gcs_transfer(request_after_the_abort, label="refused")
        )
        await asyncio.sleep(0.1)
        task.cancel()
        await asyncio.sleep(0.1)
        proceed.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 30)

    assert len(seen) == 1 and isinstance(seen[0], GCSTransferAborted), seen
    # The library retries connection errors; it must not retry the refusal.
    assert DEFAULT_RETRY._predicate(seen[0]) is False


async def test_a_thread_that_cannot_be_interrupted_is_reported_and_left_to_the_process(
    monkeypatch, caplog
):
    monkeypatch.setattr(gcs_transfer, "GCS_TRANSFER_ABORT_GRACE_SECONDS", 0.2)
    release = threading.Event()
    task = asyncio.create_task(
        run_gcs_transfer(lambda: release.wait(30), label="stuck")
    )
    await asyncio.sleep(0.1)
    try:
        with caplog.at_level(logging.WARNING, logger=gcs_transfer.__name__):
            elapsed = await _cancel_once(task)
        (thread,) = _transfer_threads("stuck")
        assert thread.daemon, "a thread left running would hold the process at exit"
        assert elapsed >= 0.2
        assert f"{GCS_TRANSFER_THREAD_PREFIX}stuck" in caplog.text
        assert "did not stop within 0.2s" in caplog.text
    finally:
        release.set()
    await asyncio.to_thread(thread.join, 10)
    assert not thread.is_alive()


async def test_a_repeated_cancellation_still_waits_for_the_thread():
    release = threading.Event()
    task = asyncio.create_task(
        run_gcs_transfer(lambda: release.wait(30), label="repeated")
    )
    await asyncio.sleep(0.1)
    task.cancel()
    await asyncio.sleep(0.05)
    task.cancel()
    asyncio.get_running_loop().call_later(0.3, release.set)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 30)
    assert release.is_set(), "the second cancellation cut the join short"
    assert not _transfer_threads("repeated")


def test_only_the_default_adapter_is_made_abortable():
    session = requests.Session()
    make_session_abortable(session)
    adapters = dict(session.adapters)
    assert {type(adapter) for adapter in adapters.values()} == {
        gcs_transfer._AbortableHTTPAdapter
    }
    make_session_abortable(session)
    assert dict(session.adapters) == adapters, "a second call replaced the adapters"

    class MutualTls(HTTPAdapter):
        pass

    other = requests.Session()
    other.mount("https://", MutualTls())
    with pytest.raises(TypeError, match="MutualTls"):
        make_session_abortable(other)


def test_a_socks_proxy_is_refused_rather_than_left_uninterruptible(session):
    adapter = session.get_adapter("https://storage.googleapis.com")
    with pytest.raises(requests.exceptions.InvalidProxyURL, match="SOCKS"):
        adapter.proxy_manager_for("socks5://127.0.0.1:1080")
    manager = adapter.proxy_manager_for("http://127.0.0.1:3128")
    assert manager.pool_classes_by_scheme is gcs_transfer._ABORTABLE_POOL_CLASSES


#: A child process: a real ``GCSTarget`` against a local server speaking the
#: resumable-upload protocol, which accepts the session and then stops reading
#: the chunk, so the upload blocks in its send. The child cancels the pass the
#: way the host's shutdown budget does, stops the service and returns.
_CHILD = textwrap.dedent(
    r"""
    import asyncio, json, os, socket, sqlite3, sys, threading, time
    from contextlib import closing
    from pathlib import Path

    work = Path(sys.argv[1])
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    port = listener.getsockname()[1]
    stalled = threading.Event()
    held = []

    def serve(conn):
        buffered = b""
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                return
            buffered += chunk
            if b"\r\n\r\n" not in buffered:
                continue
            head, body = buffered.split(b"\r\n\r\n", 1)
            request_line = head.split(b"\r\n", 1)[0]
            if request_line.startswith(b"PUT "):
                stalled.set()
                return  # reads no more of the chunk, and never answers
            length = 0
            for line in head.split(b"\r\n")[1:]:
                name, _, value = line.partition(b":")
                if name.strip().lower() == b"content-length":
                    length = int(value)
            while len(body) < length:
                body += conn.recv(65536)
            conn.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"Location: http://127.0.0.1:%d/upload/session\r\n"
                b"Content-Length: 0\r\n\r\n" % port
            )
            buffered = b""

    def accept():
        while True:
            conn, _ = listener.accept()
            held.append(conn)
            threading.Thread(target=serve, args=(conn,), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()
    os.environ["STORAGE_EMULATOR_HOST"] = f"http://127.0.0.1:{port}"

    from kestrel_sovereign.storage.sync.gcs_target import GCSTarget
    from kestrel_sovereign.storage.sync.service import SyncService

    db = work / "agent.db"
    with closing(sqlite3.connect(db)) as conn:
        conn.execute("CREATE TABLE blobs (body BLOB)")
        conn.execute("INSERT INTO blobs VALUES (randomblob(12 * 1024 * 1024))")
        conn.commit()

    async def main():
        service = SyncService(db_path=str(db), state_file=str(work / "agent.db.sync"))
        target = GCSTarget("bucket", agent_id="agent", state_dir=work / "state")
        service.add_target(target)
        await service.start()
        backup = asyncio.create_task(service.force_snapshot())
        while not stalled.is_set():
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.5)  # the send fills the socket buffers and blocks
        cancelled_at = time.monotonic()
        cancelled = service.cancel_inflight_snapshots("shutdown budget")
        results = await backup
        await service.stop()
        print(json.dumps({
            "cancelled": cancelled,
            "error": results[target.name].error,
            "stopped_after": time.monotonic() - cancelled_at,
        }), flush=True)

    asyncio.run(main())
    """
)


def test_a_cancelled_upload_lets_the_process_exit(tmp_path):
    """Beyond the pass finishing: the process itself exits, which it did not
    while the upload thread went on sending."""
    package_root = Path(kestrel_sovereign.__file__).resolve().parent.parent
    environment = dict(os.environ, PYTHONPATH=str(package_root))
    environment.pop("STORAGE_EMULATOR_HOST", None)
    started = time.monotonic()
    try:
        child = subprocess.run(
            [sys.executable, "-c", _CHILD, str(tmp_path)],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as expired:
        pytest.fail(
            "the child did not exit after its upload was cancelled; its upload "
            f"thread is still sending.\nstdout:\n{expired.stdout}\nstderr:\n{expired.stderr}"
        )
    assert child.returncode == 0, child.stderr
    (line,) = [line for line in child.stdout.splitlines() if line.startswith("{")]
    report = json.loads(line)
    assert report["cancelled"] == [
        "snapshot to gs://bucket/kestrel/agent cancelled at shutdown budget"
    ]
    assert report["error"] == "cancelled at shutdown budget"
    assert report["stopped_after"] < gcs_transfer.GCS_TRANSFER_ABORT_GRACE_SECONDS
    assert time.monotonic() - started < 60
