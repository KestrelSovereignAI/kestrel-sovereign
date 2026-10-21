"""Real loopback HTTP/SQLite snapshot retirement, not paid/live acceptance."""

import asyncio
import json
import sqlite3
from contextlib import suppress

import httpx
import pytest
import pytest_asyncio

from kestrel_sovereign._async_ownership import create_owned_task, owned_task_error
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    execution_cleanup_debts,
    is_execution_control_error,
)
from kestrel_sovereign.storage.providers import lighthouse_rest
from kestrel_sovereign.storage.sync.lighthouse_target import LighthouseTarget
from kestrel_sovereign.storage.sync.service import SyncService


async def _eventually(predicate):
    for _ in range(150):
        if predicate():
            return
        await asyncio.sleep(0.01)
    assert predicate(), "original native obligation did not reach its terminal boundary"


@pytest_asyncio.fixture
async def native_snapshot(tmp_path, monkeypatch):
    db = tmp_path / "native-snapshot.db"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE payload (value TEXT)")
        connection.execute("INSERT INTO payload VALUES ('actual local snapshot')")
    entered, release_server = asyncio.Event(), asyncio.Event()
    close_entered, release_close = asyncio.Event(), asyncio.Event()
    handlers, clients, source_errors, close_tasks = set(), [], [], []
    state = {"close_error": OSError("native HTTP client close acknowledgement lost")}
    state["release_server"] = release_server
    state["hold_path"] = None

    async def serve(reader, writer):
        task = asyncio.current_task()
        handlers.add(task)
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            size = (
                next(
                    int(line.split(b":", 1)[1])
                    for line in headers.split(b"\r\n")
                    if line.lower().startswith(b"content-length:")
                )
                if b"content-length:" in headers.lower()
                else 0
            )
            await reader.readexactly(size)
            hold_path = state["hold_path"]
            if hold_path is None or hold_path.encode() in headers.split(b"\r\n", 1)[0]:
                entered.set()
                await release_server.wait()
            if hold_path is None:
                writer.write(b"HTTP/1.1 503 Unavailable\r\nContent-Length: 0\r\n\r\n")
            else:
                payload = json.dumps(
                    {
                        "Hash": "local-loopback-cid",
                        "Size": size,
                        "data": {"fileList": []},
                    }
                ).encode()
                # Lighthouse wraps uploads in data only when that key exists.
                if headers.startswith(b"POST "):
                    payload = json.dumps(
                        {"Hash": "local-loopback-cid", "Size": size}
                    ).encode()
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                    + str(len(payload)).encode()
                    + b"\r\n\r\n"
                    + payload
                )
            await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            await writer.wait_closed()
            handlers.discard(task)

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    original_upload = lighthouse_rest.LighthouseRestClient.upload_car

    class LocalClient(lighthouse_rest.LighthouseRestClient):
        UPLOAD_URL = f"http://127.0.0.1:{port}"
        API_URL = f"http://127.0.0.1:{port}"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            clients.append(self)

        async def upload_car(self, *args, **kwargs):
            try:
                return await original_upload(self, *args, **kwargs)
            except BaseException as error:
                if state.get("upload_failure_as_cancel") and isinstance(
                    error, httpx.HTTPStatusError
                ):
                    # An admitted native callback retains its ORIGINAL real
                    # HTTP failure as the explicit cause of cancellation.
                    cancellation = asyncio.CancelledError(
                        "original native upload callback cancelled"
                    )
                    source_errors.append(cancellation)
                    raise cancellation from error
                source_errors.append(error)
                raise

        async def get_uploads(self, *args, **kwargs):
            try:
                return await super().get_uploads(*args, **kwargs)
            except BaseException as error:
                source_errors.append(error)
                raise

    actual_close = httpx.AsyncClient.aclose

    async def close_with_lost_ack(client):
        # Forward the actual HTTPX/httpcore socket retirement before losing ACK.
        await actual_close(client)
        if state["hold_path"] is not None and not entered.is_set():
            return  # The preceding successful upload's client closes normally.
        close_index = len(close_tasks)
        close_tasks.append(asyncio.current_task())
        close_entered.set()
        await release_close.wait()
        if state.get("close_probe") is not None:
            await state["close_probe"]()
        if state.get("close_errors") is not None:
            raise state["close_errors"][close_index]
        if state["close_error"] is not None:
            raise state["close_error"]

    monkeypatch.setattr(lighthouse_rest, "LighthouseRestClient", LocalClient)
    monkeypatch.setattr(httpx.AsyncClient, "aclose", close_with_lost_ack)
    target = LighthouseTarget("disposable-local-only", "native-snapshot", tmp_path)
    service = SyncService(str(db), wal_sync_interval=0.01)
    service.add_target(target)
    tasks = []
    try:
        yield (
            db,
            target,
            service,
            entered,
            close_entered,
            release_close,
            clients,
            source_errors,
            close_tasks,
            state,
            tasks,
        )
    finally:
        # No new periodic attempt during fixture-only retirement. This does not
        # settle the worker or alter any assertion made by the actual test.
        service.wal_sync_interval = 60
        release_close.set()
        release_server.set()
        server.close()
        await server.wait_closed()
        for task in tasks + ([service._poll_task] if service._poll_task else []):
            for _ in range(3):
                if task.done():
                    break
                task.cancel()
                await asyncio.wait({task}, timeout=0.05)
            assert task.done(), "fixture could not retire its exact owned worker"
            with suppress(BaseException):
                await task
        for client in clients:
            if client._client is not None:
                await actual_close(client._client)
        if handlers:
            await asyncio.gather(*tuple(handlers), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["snapshot", "force", "periodic", "stop"])
@pytest.mark.parametrize("close_cancels", [False, True])
async def test_upload_cancellation_and_native_close_failure_survive_every_boundary(
    native_snapshot, door, close_cancels
):
    (
        db,
        target,
        service,
        entered,
        close_entered,
        release_close,
        clients,
        source_errors,
        close_tasks,
        state,
        tasks,
    ) = native_snapshot
    if close_cancels:
        state["close_error"] = asyncio.CancelledError("native close ACK cancelled")
        state["close_error"].__cause__ = OSError("original native close ACK failure")
    if door in {"periodic", "stop"}:
        await service.start()
        worker = service._poll_task
    else:
        operation = (
            target.sync_snapshot(db) if door == "snapshot" else service.force_snapshot()
        )
        worker = create_owned_task(operation, retain_originals=True)
    tasks.append(worker)
    await asyncio.wait_for(entered.wait(), 2)
    if door == "stop":
        observer = create_owned_task(service.stop(), retain_originals=True)
        tasks.append(observer)
    else:
        worker.cancel("original upload cancellation")
        observer = worker
    await asyncio.wait_for(close_entered.wait(), 2)
    assert clients[0]._client.is_closed, "native close must precede injected ACK loss"
    assert not observer.done(), "shutdown cannot claim retirement before ACK"
    release_close.set()
    await _eventually(observer.done)
    failure = owned_task_error(observer)
    assert failure is not None, "native failure became an ordinary failed SyncResult"
    chain = tuple(_explicit_control_chain(failure))
    assert source_errors and isinstance(source_errors[0], asyncio.CancelledError)
    assert source_errors[0] in chain, "original upload cancellation was replaced"
    assert state["close_error"] in chain
    assert is_execution_control_error(failure)
    assert any(
        resource is clients[0] and task is close_tasks[0]
        for resource, task in execution_cleanup_debts(failure)
    )
    assert worker.done(), "periodic worker survived its shutdown cancellation"
    assert not target._latest_cid
    assert not target._manifest_path.exists()
    if door in {"periodic", "stop"}:
        for _ in range(2):
            with pytest.raises(BaseException) as repeated:
                await service.stop()
            assert repeated.value is failure
            assert len(close_tasks) == 1, "failed native closer must not be retried"


@pytest.mark.asyncio
@pytest.mark.parametrize("close_fails", [False, True])
async def test_repeated_caller_cancellation_does_not_interrupt_native_retirement(
    native_snapshot, close_fails
):
    (
        db,
        target,
        service,
        entered,
        close_entered,
        release_close,
        clients,
        source_errors,
        close_tasks,
        state,
        tasks,
    ) = native_snapshot
    if not close_fails:
        state["close_error"] = None
    worker = create_owned_task(target.sync_snapshot(db), retain_originals=True)
    tasks.append(worker)
    await asyncio.wait_for(entered.wait(), 2)
    worker.cancel("original upload withdrawal")
    await asyncio.wait_for(close_entered.wait(), 2)
    worker.cancel("second withdrawal during close ACK")
    await asyncio.sleep(0.02)
    worker.cancel("third withdrawal during close ACK")
    await asyncio.sleep(0.02)
    assert not worker.done(), "observer withdrawal abandoned native close"
    assert not close_tasks[0].done()
    release_close.set()
    await _eventually(worker.done)
    failure = owned_task_error(worker)
    assert source_errors[0] in tuple(_explicit_control_chain(failure))
    assert close_tasks[0].done()
    if close_fails:
        assert state["close_error"] in tuple(_explicit_control_chain(failure))
        assert execution_cleanup_debts(failure)
    else:
        assert isinstance(failure, asyncio.CancelledError)
        assert not execution_cleanup_debts(failure)


@pytest.mark.asyncio
async def test_native_closer_admission_refusal_retains_unstarted_client(
    native_snapshot, monkeypatch
):
    (
        db,
        target,
        service,
        entered,
        close_entered,
        release_close,
        clients,
        source_errors,
        close_tasks,
        state,
        tasks,
    ) = native_snapshot
    worker = create_owned_task(target.sync_snapshot(db), retain_originals=True)
    tasks.append(worker)
    await asyncio.wait_for(entered.wait(), 2)
    actual_create = asyncio.create_task
    refusal = RuntimeError("actual native closer Task admission refused")

    def refuse_original_closer(operation, *args, **kwargs):
        if operation.cr_code.co_qualname == "create_owned_task.<locals>.run":
            raise refusal
        return actual_create(operation, *args, **kwargs)

    monkeypatch.setattr(asyncio, "create_task", refuse_original_closer)
    worker.cancel("original upload withdrawal before closer admission")
    await _eventually(worker.done)
    failure = owned_task_error(worker)
    assert failure is not None
    chain = tuple(_explicit_control_chain(failure))
    assert source_errors[0] in chain and refusal in chain
    assert execution_cleanup_debts(failure) == ((clients[0], None),)
    assert not close_tasks and not close_entered.is_set()
    assert not clients[0]._client.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("close_fails", [False, True])
async def test_shutdown_observer_withdrawal_joins_worker_before_propagating(
    native_snapshot, close_fails
):
    (
        db,
        target,
        service,
        entered,
        close_entered,
        release_close,
        clients,
        source_errors,
        close_tasks,
        state,
        tasks,
    ) = native_snapshot
    if not close_fails:
        state["close_error"] = None
    await service.start()
    worker = service._poll_task
    tasks.append(worker)
    await asyncio.wait_for(entered.wait(), 2)
    stop = create_owned_task(service.stop(), retain_originals=True)
    tasks.append(stop)
    await asyncio.wait_for(close_entered.wait(), 2)
    stop.cancel("shutdown observer withdrawal")
    await asyncio.sleep(0.02)
    stop.cancel("repeated shutdown observer withdrawal")
    await asyncio.sleep(0.02)
    assert not stop.done() and not worker.done() and not close_tasks[0].done()
    release_close.set()
    await _eventually(stop.done)
    failure = owned_task_error(stop)
    assert worker.done() and close_tasks[0].done()
    chain = tuple(_explicit_control_chain(failure))
    assert source_errors[0] in chain
    assert any(
        isinstance(error, asyncio.CancelledError)
        and error.args == ("shutdown observer withdrawal",)
        for error in chain
    )
    if close_fails:
        assert state["close_error"] in chain and execution_cleanup_debts(failure)


@pytest.mark.asyncio
async def test_ordinary_http_failure_with_successful_close_remains_failed_result(
    native_snapshot,
):
    (
        db,
        target,
        service,
        entered,
        close_entered,
        release_close,
        clients,
        source_errors,
        close_tasks,
        state,
        tasks,
    ) = native_snapshot
    state["close_error"] = None
    # Let the real peer send its ordinary 503 response.
    # The handler's event is released through the actual listening server below.
    worker = create_owned_task(target.sync_snapshot(db), retain_originals=True)
    tasks.append(worker)
    await asyncio.wait_for(entered.wait(), 2)
    # Finish the request as a real HTTP failure rather than inject an exception.
    writer_release = state["release_server"]
    writer_release.set()
    release_close.set()
    await _eventually(worker.done)
    assert owned_task_error(worker) is None
    result = worker.result()
    assert not result.success and "HTTPStatusError" in result.error
    assert clients[0]._client is None


@pytest.mark.asyncio
async def test_caller_withdrawal_during_ordinary_http_failure_close_is_not_fallback(
    native_snapshot,
):
    (
        db,
        target,
        service,
        entered,
        close_entered,
        release_close,
        clients,
        source_errors,
        close_tasks,
        state,
        tasks,
    ) = native_snapshot
    state["close_error"] = None
    worker = create_owned_task(target.sync_snapshot(db), retain_originals=True)
    tasks.append(worker)
    await asyncio.wait_for(entered.wait(), 2)
    state["release_server"].set()
    await asyncio.wait_for(close_entered.wait(), 2)
    assert isinstance(source_errors[0], httpx.HTTPStatusError)
    worker.cancel("withdrawal during HTTP failure retirement")
    await asyncio.sleep(0.02)
    assert not worker.done() and not close_tasks[0].done()
    release_close.set()
    await _eventually(worker.done)
    failure = owned_task_error(worker)
    assert isinstance(failure, asyncio.CancelledError)
    assert source_errors[0] in tuple(_explicit_control_chain(failure))
    assert not execution_cleanup_debts(failure)


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["force", "periodic", "stop"])
@pytest.mark.parametrize("close_cancels", [False, True])
async def test_successful_snapshot_retention_preserves_original_shutdown_control(
    native_snapshot, door, close_cancels
):
    (
        db,
        target,
        service,
        entered,
        close_entered,
        release_close,
        clients,
        source_errors,
        close_tasks,
        state,
        tasks,
    ) = native_snapshot
    state["hold_path"] = "/api/user/files_uploaded"
    if close_cancels:
        state["close_error"] = asyncio.CancelledError("native pruning close cancelled")
        state["close_error"].__cause__ = OSError("native pruning close ACK failed")
    if door in {"periodic", "stop"}:
        await service.start()
        worker = service._poll_task
    else:
        worker = create_owned_task(service.force_snapshot(), retain_originals=True)
    tasks.append(worker)
    await asyncio.wait_for(entered.wait(), 3)
    assert target._latest_cid == "local-loopback-cid"
    assert target._manifest_path.exists(), "must reach actual post-success pruning"
    if door == "stop":
        observer = create_owned_task(service.stop(), retain_originals=True)
        tasks.append(observer)
    else:
        worker.cancel("original successful-backup pruning withdrawal")
        observer = worker
    await asyncio.wait_for(close_entered.wait(), 3)
    assert len(clients) == 2 and clients[1]._client.is_closed
    assert not observer.done()
    release_close.set()
    await _eventually(observer.done)
    failure = owned_task_error(observer)
    assert failure is not None, "pruning control became successful snapshot metadata"
    chain = tuple(_explicit_control_chain(failure))
    assert state["close_error"] in chain and is_execution_control_error(failure)
    assert source_errors[0] in chain, "original pruning cancellation was replaced"
    assert any(isinstance(error, asyncio.CancelledError) for error in chain)
    assert any(
        resource is clients[1] and owner is close_tasks[0]
        for resource, owner in execution_cleanup_debts(failure)
    )
    assert worker.done(), "periodic worker resumed after pruning cancellation"
