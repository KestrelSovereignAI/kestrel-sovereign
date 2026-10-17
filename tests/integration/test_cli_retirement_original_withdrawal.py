"""Actual CLI observer withdrawal remains alongside native body/close receipts.

Real SQLite opening, audit SQL, physical close and native Tasks execute in the
CLI-owned event loop. The retired PostgreSQL lease supplies an actual original
authority denial at declared callback boundaries, not a fake native ACK.
"""

import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sovereign import _async_ownership, cli_embeddings
from kestrel_sovereign.execution_custody import _explicit_control_chain
from kestrel_sovereign.storage.async_database import (
    AsyncDatabase,
    DatabaseInitializationCleanupError,
    DatabaseRetirementError,
    _close_failed_database_initialization,
)
from kestrel_sovereign.storage.db import sqlite
from kestrel_sovereign.storage.db.sqlite import _aiosqlite_worker_is_alive
from tests.integration.native_memory_original_control import (
    native_memory_owners as native_memory_owners,
)
from tests.integration.test_execution_authority_postgres import native_pg as native_pg
from tests.unit.test_cli_embeddings import _agent_data_dir


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body_kind,close_kind",
    [
        ("healthy", "control"),
        ("control", "healthy"),
        ("ordinary", "ordinary"),
        ("healthy", "healthy"),
        ("control", "control"),
    ],
)
async def test_actual_cli_retirement_keeps_original_observer_withdrawal(
    native_memory_owners, tmp_path, monkeypatch, body_kind, close_kind
):
    n = native_memory_owners
    # Fixture ownership, not a subsecond shutdown-latency assertion.
    monkeypatch.setattr(sqlite, "AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S", 10.0)
    data_dir = await asyncio.to_thread(
        _agent_data_dir, tmp_path, monkeypatch, legacy_column=False
    )
    actual_factory, actual_audit = AsyncDatabase.sqlite, cli_embeddings._audit
    actual_wait = _async_ownership.await_owned_task
    owners, native_connections, body_receipts, close_receipts, withdrawals = (
        [],
        [],
        [],
        [],
        [],
    )
    observer = None

    def failure(kind, receipts):
        if kind == "control":
            try:
                n.lease.require_live()
            except BaseException as original:
                receipts.append(original)
                raise
        if kind == "ordinary":
            original = OSError("ordinary local observation failed")
            receipts.append(original)
            raise original

    async def observed_audit(*args, **kwargs):
        result = await actual_audit(*args, **kwargs)
        failure(body_kind, body_receipts)
        return result

    async def observed_wait(task, *args, **kwargs):
        outcome = await actual_wait(task, *args, **kwargs)
        if (
            task.get_name() == "owned-database-retirement"
            and outcome.cancellation is not None
        ):
            # SAME outcome object, no replacement Future/result/cancellation.
            withdrawals.append(outcome.cancellation)
        return outcome

    async def factory(cls, *args, **kwargs):
        nonlocal observer
        db = await actual_factory(*args, **kwargs)
        owners.append(db)
        observer = asyncio.current_task()
        native = db.backend._connection
        native_connections.append(native)
        native_close = native.close

        async def close():
            await native_close()
            # The original CLI observer is still awaiting this accepted close.
            # Withdraw observation, not the physical closer or its native ACK.
            observer.cancel("actual CLI retirement observer withdrawn")
            await asyncio.sleep(0)
            failure(close_kind, close_receipts)

        monkeypatch.setattr(native, "close", close)
        return db

    monkeypatch.setattr(AsyncDatabase, "sqlite", classmethod(factory))
    monkeypatch.setattr(cli_embeddings, "_audit", observed_audit)
    monkeypatch.setattr(_async_ownership, "await_owned_task", observed_wait)
    args = SimpleNamespace(
        embeddings_command="audit",
        data_dir=str(data_dir),
        agent_name=None,
        table="saved_items",
        agent_id=n.did,
    )
    caught = None
    try:
        try:
            await asyncio.to_thread(cli_embeddings.run, args)
        except BaseException as error:
            caught = error
        assert owners and withdrawals, "original CLI withdrawal was not observed"
        assert caught is not None and withdrawals[0] in tuple(
            _explicit_control_chain(caught)
        ), (
            f"joined CLI retirement discarded its original observer cancellation: "
            f"body={body_kind} close={close_kind} {caught!r}"
        )
        if close_kind != "healthy":
            assert isinstance(caught, DatabaseRetirementError)
            assert caught.retirement_owner is owners[0]
            assert close_receipts[0] in tuple(
                _explicit_control_chain(caught.cleanup_error)
            )
        if body_kind != "healthy":
            assert body_receipts[0] in tuple(_explicit_control_chain(caught))
        assert all(not _aiosqlite_worker_is_alive(c) for c in native_connections)
        assert not n.requests
    finally:
        for db in owners:
            await asyncio.to_thread(lambda db=db: asyncio.run(db.close()))
        assert all(not _aiosqlite_worker_is_alive(c) for c in native_connections)


@pytest.mark.asyncio
@pytest.mark.parametrize("body_kind", ["control", "ordinary"])
@pytest.mark.parametrize("close_kind", ["healthy", "control", "ordinary"])
async def test_initialization_wrapper_retains_original_joined_withdrawal(
    native_memory_owners, tmp_path, monkeypatch, body_kind, close_kind
):
    n = native_memory_owners
    monkeypatch.setattr(sqlite, "AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S", 10.0)
    db = await AsyncDatabase.sqlite(str(tmp_path / "initialization.db"))
    await db.execute("CREATE TABLE actual_effect(id INTEGER PRIMARY KEY)")
    native = db.backend._connection
    actual_close, actual_wait = native.close, _async_ownership.await_owned_task
    withdrawals, closes = [], []
    observer = None

    def receipt(kind):
        if kind == "control":
            try:
                n.lease.require_live()
            except BaseException as original:
                return original
        return OSError("ordinary initialization observation failed")

    opening = receipt(body_kind)

    async def observed_wait(task, *args, **kwargs):
        outcome = await actual_wait(task, *args, **kwargs)
        if (
            task.get_name() == "owned-database-retirement"
            and outcome.cancellation is not None
        ):
            withdrawals.append(outcome.cancellation)
        return outcome

    async def close():
        await actual_close()
        observer.cancel("actual initialization observer withdrawn")
        await asyncio.sleep(0)
        if close_kind != "healthy":
            original = receipt(close_kind)
            closes.append(original)
            raise original

    async def recovery():
        nonlocal observer
        observer = asyncio.current_task()
        await _close_failed_database_initialization(db, opening)

    monkeypatch.setattr(_async_ownership, "await_owned_task", observed_wait)
    monkeypatch.setattr(native, "close", close)
    try:
        owner = _async_ownership.create_owned_task(
            recovery(), name="actual-initialization-recovery", retain_originals=True
        )
        outcome = await actual_wait(owner)
        assert withdrawals and outcome.error is not None
        chain = tuple(_explicit_control_chain(outcome.error))
        assert withdrawals[0] in chain and opening in chain
        if close_kind != "healthy":
            assert isinstance(outcome.error, DatabaseInitializationCleanupError)
            assert outcome.error.retirement_owner is db
            assert closes[0] in chain
        assert not _aiosqlite_worker_is_alive(native)
        assert not n.requests
    finally:
        await db.close()
        assert not _aiosqlite_worker_is_alive(native)
