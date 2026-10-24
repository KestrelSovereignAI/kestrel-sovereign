"""Native primary storage and every service remain owned through shutdown."""

import asyncio

import pytest
import pytest_asyncio

from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.task_manager import TaskManager
from kestrel_sovereign.execution_custody import ExecutionCommitOutcomeError

STORE_NAMES = {
    "feedback_store", "memory_service", "observability_store", "session_service",
    "context_store", "task_store", "question_store", "outbound_store",
}


@pytest_asyncio.fixture
async def manager(native_a2a_backend_factory):
    backend = native_a2a_backend_factory()
    await backend.connect()
    instance = TaskManager.from_backend(
        backend, scope=NativeRecipientScope("did:test:recipient"), owns_backend=True,
    )
    try:
        await instance.initialize()
        yield instance
    finally:
        await instance.close()


def observe_closes(manager, monkeypatch, *, failures=None):
    calls = []
    failures = failures or {}
    for name in STORE_NAMES:
        original = getattr(manager, name).close

        async def close(_name=name, _original=original):
            calls.append(_name)
            if _name in failures:
                raise failures[_name]
            await _original()

        monkeypatch.setattr(getattr(manager, name), "close", close)
    return calls


async def test_native_shutdown_closes_every_service_and_its_owned_primary(manager, monkeypatch):
    calls = observe_closes(manager, monkeypatch)
    backend = manager.task_store.backend
    await manager.close()
    assert set(calls) == STORE_NAMES and len(calls) == len(STORE_NAMES)
    assert not backend.is_connected


@pytest.mark.parametrize("store_name", sorted(STORE_NAMES))
async def test_failed_service_close_retains_its_owner_and_is_retried(
    manager, monkeypatch, store_name
):
    owner = getattr(manager, store_name)
    original = ConnectionError("Injected service retirement failure")
    with monkeypatch.context() as patcher:
        calls = observe_closes(manager, patcher, failures={store_name: original})
        with pytest.raises(ConnectionError) as caught:
            await manager.close()
        assert caught.value is original
        assert set(calls) == STORE_NAMES and len(calls) == len(STORE_NAMES)
    assert getattr(manager, store_name) is owner
    calls = observe_closes(manager, monkeypatch)
    await manager.close()
    assert set(calls) == STORE_NAMES


async def test_every_failed_service_is_named_and_original_error_is_preserved(manager, monkeypatch):
    memory = ConnectionError("Memory retirement failed")
    task = OSError("Task retirement failed")
    with monkeypatch.context() as patcher:
        calls = observe_closes(
            manager, patcher, failures={"memory_service": memory, "task_store": task}
        )
        with pytest.raises(ConnectionError) as caught:
            await manager.close()
        assert caught.value is memory and set(calls) == STORE_NAMES
        notes = caught.value.__notes__
        assert any("memory_service" in note for note in notes)
        assert any("task_store" in note and "Task retirement failed" in note for note in notes)


async def test_cancelled_drain_keeps_primary_and_all_services_for_retry(manager, monkeypatch):
    original = asyncio.CancelledError("Shutdown caller cancelled")

    async def cancelled_drain(*, cancel):
        assert cancel is True
        raise original

    with monkeypatch.context() as patcher:
        calls = observe_closes(manager, patcher)
        patcher.setattr(manager, "drain_execution_tasks", cancelled_drain)
        with pytest.raises(asyncio.CancelledError) as caught:
            await manager.close()
        assert caught.value is original and calls == []
        assert manager.task_store.backend.is_connected
    await manager.close()
    assert not manager.task_store.backend.is_connected


async def test_shutdown_keeps_original_execution_receipt_while_closing_other_services(
    manager, monkeypatch
):
    marker = ExecutionCommitOutcomeError("unknown")
    original = ConnectionError("Native service retirement was uncertain")
    original.__cause__ = marker
    with monkeypatch.context() as patcher:
        calls = observe_closes(manager, patcher, failures={"task_store": original})
        with pytest.raises(ConnectionError) as caught:
            await manager.close()
        assert caught.value is original and caught.value.__cause__ is marker
        assert set(calls) == STORE_NAMES


async def test_sqlite_worker_retirement_is_owned_until_the_worker_exits(tmp_path, monkeypatch):
    import threading

    from kestrel_sovereign.storage.db import ConnectionError as BackendError
    from kestrel_sovereign.storage.db import SQLiteBackend
    from kestrel_sovereign.storage.db import sqlite as sqlite_backend
    from tests.utils.aiosqlite_workers import (
        aiosqlite_worker,
        delay_aiosqlite_worker_exit,
    )

    release = threading.Event()
    delayed = threading.Event()
    selected = []
    with delay_aiosqlite_worker_exit(
        release, delayed, should_delay=lambda worker: worker in selected
    ) as workers:
        backend = SQLiteBackend(str(tmp_path / "retirement.db"))
        await backend.connect()
        instance = TaskManager.from_backend(
            backend, scope=NativeRecipientScope("did:test:recipient"), owns_backend=True,
        )
        await instance.initialize()
        worker = aiosqlite_worker(backend._connection)
        selected.append(worker)
        monkeypatch.setattr(sqlite_backend, "AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S", 0.01)
        try:
            with pytest.raises(BackendError):
                await instance.close()
            assert workers == [worker] and backend.connection_retirement_pending
            with pytest.raises(BackendError, match="still retiring"):
                await instance.close()
        finally:
            release.set()
            worker.join(timeout=5)
            await instance.close()
    assert not worker.is_alive() and not backend.connection_retirement_pending
