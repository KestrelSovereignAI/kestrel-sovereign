"""Join actual SQLite creation before withdrawal, without fabricated close ACK."""

import asyncio
import os
import sqlite3
import threading

import aiosqlite
import pytest

from kestrel_sovereign.execution_custody import execution_cleanup_debts, execution_evidence_carriers, is_execution_control_error
from kestrel_sovereign.storage.db.interface import ConnectionError
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from tests.unit.test_host_context_native_cleanup import _actual_operation
from tests.utils.aiosqlite_workers import aiosqlite_worker
from tests.utils.virtual_clock import HANG_GUARD_SECONDS, await_within_hang_guard, wait_for_thread_event


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["primary", "cold", "snapshot", "backup"])
@pytest.mark.parametrize("repeated", [False, True])
async def test_actual_connection_creation_is_joined_before_caller_withdrawal(tmp_path, monkeypatch, route, repeated):
    path = tmp_path / "original-creation.db"
    backend = SQLiteBackend(str(path))
    await backend.connect()
    if route in {"primary", "cold"}:
        await backend.close()
        backend = SQLiteBackend(str(path), cold_read=route == "cold")
        target = backend._cold_read_uri() if route == "cold" else str(path)
    else:
        target = path.as_uri() + "?mode=rw" if route == "snapshot" else str(tmp_path / "original-backup.db")
    native_connect, async_connect = sqlite3.connect, aiosqlite.connect
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    actual, wrappers, delivered, terminal = [], [], [], []
    observed = {"abandoned": False, "test_reconciled": False}
    original_wait = asyncio.wait

    async def observe_original_wait(*args, **kwargs):
        try:
            return await original_wait(*args, **kwargs)
        except asyncio.CancelledError as error:
            if asyncio.current_task() is pending:
                delivered.append(error)
            raise

    def original_native_connect(database, *args, **kwargs):
        connection = native_connect(database, *args, **kwargs)
        if database != target:
            return connection
        actual.append(connection)
        entered.set()
        try:
            if not release.wait(HANG_GUARD_SECONDS):
                connection.close()
                raise TimeoutError("Original test SQLite creator was not released")
            if observed["abandoned"]:
                # Negative/mutation cleanup AFTER the failure snapshot, on the
                # actual creating worker. This is not a product retirement ACK.
                assert connection.execute("SELECT 1").fetchone() == (1,)
                connection.close()
                observed["test_reconciled"] = True
            return connection
        finally:
            returned.set()

    def original_async_connect(database, *args, **kwargs):
        connection = async_connect(database, *args, **kwargs)
        if database == target:
            wrappers.append(connection)
        return connection

    async def acquire():
        try:
            if route in {"primary", "cold"}:
                return await backend.connect()
            if route == "backup":
                return await backend.backup_to(target)
            return await backend._open_snapshot_read_connection()
        except asyncio.CancelledError as error:
            terminal.append(error)
            raise

    pending = None
    try:
        with monkeypatch.context() as fault:
            fault.setattr(sqlite3, "connect", original_native_connect)
            fault.setattr(aiosqlite, "connect", original_async_connect)
            fault.setattr(asyncio, "wait", observe_original_wait)
            pending = asyncio.create_task(acquire())
            await wait_for_thread_event(entered, step="the actual original SQLite creator holding its open handle")
            pending.cancel("Caller withdrew during original SQLite creation")
            # Deliver cancellation to the already parked caller before taking
            # the pre-release snapshot; no wall-clock completion guess.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            if repeated:
                pending.cancel("Same caller withdrew again before creation returned")
                await asyncio.sleep(0)
                await asyncio.sleep(0)
            premature = pending.done()
            observed["abandoned"] = premature
            release.set()
            result = (await await_within_hang_guard(
                asyncio.gather(pending, return_exceptions=True), step="the original SQLite creation and retirement joining"
            ))[0]
            await wait_for_thread_event(returned, step="the exact native creator returning")
            if isinstance(result, aiosqlite.Connection):
                await result.close()  # explicit test recovery of a lost cancellation
            assert not premature, ("Caller escaped while its original SQLite creator still owned the handle", observed)
            assert isinstance(result, asyncio.CancelledError)
            assert len(delivered) == (2 if repeated else 1) and len(terminal) == 1
            graph = tuple(execution_evidence_carriers(terminal[0]))
            assert all(any(original is carrier for carrier in graph) for original in delivered)
            assert len(actual) == len(wrappers) == 1 and not observed["test_reconciled"]
            assert not aiosqlite_worker(wrappers[0]).is_alive()
            # SQL execute checks thread affinity BEFORE closed state. This
            # native property checks the actual C connection's closed pointer
            # without substituting a wrapper flag or another SQL connection.
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                _ = actual[0].in_transaction
    finally:
        release.set()
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)
        await backend.close()


@pytest.mark.asyncio
async def test_failed_actual_sqlite_creation_joins_its_original_worker(tmp_path, monkeypatch):
    directory = tmp_path / "not-a-database"
    directory.mkdir()
    backend = SQLiteBackend(str(directory))
    original, wrappers = aiosqlite.connect, []

    def observe(*args, **kwargs):
        connection = original(*args, **kwargs)
        wrappers.append(connection)
        return connection

    monkeypatch.setattr(aiosqlite, "connect", observe)
    try:
        with pytest.raises(ConnectionError) as failed:
            await backend.connect()
        assert any(isinstance(error, sqlite3.OperationalError) for error in execution_evidence_carriers(failed.value))
        assert len(wrappers) == 1 and not aiosqlite_worker(wrappers[0]).is_alive()
        assert backend._connection is None
    finally:
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("private", [False, True])
@pytest.mark.parametrize("close_fault", ["normal", "refused", "blocking-refused", "timeout", "slow-close", "close-owner-withdrawal"])
async def test_actual_constructor_result_survives_original_owner_cancellation(monkeypatch, private, close_fault):
    from kestrel_sovereign._async_ownership import await_owned_task, cancel_owned_task, owned_task_error, raise_owned_outcome

    backend = SQLiteBackend(":memory:")
    loop, previous = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    entered, release = asyncio.Event(), threading.Event()
    native_close_entered, native_close_release = threading.Event(), threading.Event()
    loop_thread = threading.current_thread()
    actual, workers, owners, physical, closers, acquisitions = [], [], [], [], [], []
    caller = None
    original = None
    terminal = None

    async def ownership_close(task):
        outcome = await await_owned_task(task)
        return raise_owned_outcome(outcome, operation="actual retained constructor close", retain_originals=True, retain_observer_cancellations=True)

    def original_fault_replayed(error):
        known = tuple(execution_evidence_carriers(terminal)) if isinstance(terminal, BaseException) else ()
        return close_fault != "normal" and any(node is leaf for node in execution_evidence_carriers(error) for leaf in known)

    class SlowNativeClose(sqlite3.Connection):
        def close(self):
            assert threading.current_thread() is not loop_thread, "Original native close blocked the event loop"
            native_close_entered.set()
            if not native_close_release.wait(HANG_GUARD_SECONDS):
                raise TimeoutError("Actual native physical closer was not released")
            return super().close()

    def constructor(*args, **kwargs):
        native_type = SlowNativeClose if close_fault in {"slow-close", "close-owner-withdrawal"} else sqlite3.Connection
        connection = native_type(*args, **kwargs)
        actual.append(connection)
        workers.append(threading.current_thread())
        loop.call_soon_threadsafe(entered.set)
        if not release.wait(HANG_GUARD_SECONDS):
            connection.close()
            raise TimeoutError("Actual constructor was not released")
        return connection

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        frame = getattr(operation, "cr_frame", None)
        function = frame.f_locals.get("func") if frame is not None else None
        refuse = (
            close_fault == "refused" and code is not None and code.co_qualname == "_close_acquired_sqlite_connection"
            or close_fault == "blocking-refused" and code is not None and code.co_qualname == "to_thread"
            and actual and getattr(function, "__self__", None) is actual[0]
        )
        if refuse:
            descriptor = os.open(os.devnull, os.O_WRONLY)
            try:
                try:
                    os.read(descriptor, 1)
                except OSError as error:
                    physical.append(error)
                    raise
            finally:
                os.close(descriptor)
        task = previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)
        if code is not None and code.co_qualname == "SQLiteBackend._acquire_connection.<locals>.connect_original":
            owners.append(task)
        if code is not None and code.co_qualname == "_close_acquired_sqlite_connection":
            closers.append(task)
            acquisitions.append(frame.f_locals["acquisition"])
        return task

    async def acquire():
        try:
            await backend._acquire_connection(":memory:", factory=constructor, check_same_thread=False)
        except BaseException as error:
            return error

    try:
        loop.set_task_factory(factory)
        caller = asyncio.create_task(acquire())
        await await_within_hang_guard(entered.wait(), step="the actual C constructor before native publication")
        assert len(owners) == 1
        assert cancel_owned_task(owners[0]) if private else owners[0].cancel("Original constructor owner withdrew")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        if close_fault in {"normal", "slow-close", "blocking-refused", "close-owner-withdrawal"}:
            release.set()
        if close_fault == "close-owner-withdrawal":
            await wait_for_thread_event(native_close_entered, step="the exact first native close running off the event loop")
            assert len(closers) == 1
            closers[0].cancel("Original close owner observation withdrew")
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not closers[0].done()
            native_close_release.set()
        terminal = await await_within_hang_guard(caller, step="original owner withdrawal and exact native connection retirement")
        assert isinstance(terminal, BaseException)
        original = owned_task_error(owners[0])
        assert any(error is original for error in execution_evidence_carriers(terminal))
        if close_fault == "close-owner-withdrawal":
            assert acquisitions[0].retirement_confirmed
            assert not execution_cleanup_debts(terminal)
            closer_error = owned_task_error(closers[0])
            assert any(node is closer_error for node in execution_evidence_carriers(terminal))
        elif close_fault != "normal":
            assert is_execution_control_error(terminal) and execution_cleanup_debts(terminal)
            assert backend.connection_retirement_pending
            retained, = backend._retired_connection_closes.values()
            assert retained.acquisition is not None and not retained.acquisition.retirement_confirmed
            if close_fault in {"timeout", "slow-close"}:
                assert retained.close_task is not None and not retained.close_task.done()
                assert retained.close_task.cancelling() == 0
            if close_fault == "slow-close":
                assert native_close_entered.is_set() and not workers[0].is_alive()
            release.set()
            await await_within_hang_guard(asyncio.to_thread(workers[0].join), step="the exact native worker exiting")
            if close_fault in {"refused", "blocking-refused"}:
                assert retained.acquisition.native.done(), "Original native result was lost before retention"
                assert retained.acquisition.native.result() is actual[0]
                assert not retained.acquisition.retirement_confirmed
                assert actual[0].in_transaction is False
                for _ in range(2):
                    with pytest.raises(BaseException) as fenced:
                        await backend.connect()
                    assert is_execution_control_error(fenced.value)
                    assert any(node is physical[0] for node in execution_evidence_carriers(fenced.value))
                return
            native_close_release.set()
            await await_within_hang_guard(ownership_close(retained.close_task), step="the retained first native close after the finite caller timed out")
            assert retained.acquisition.retirement_confirmed
        assert len(actual) == 1 and not workers[0].is_alive()
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            _ = actual[0].in_transaction
    finally:
        loop.set_task_factory(previous)
        release.set()
        native_close_release.set()
        if caller is not None:
            await asyncio.gather(caller, return_exceptions=True)
        # Exact test recovery AFTER the failure snapshot, never a product ACK.
        # Public cross-thread option is explicit for this adversarial probe.
        try:
            for retained in list(backend._retired_connection_closes.values()):
                if retained.close_task is not None:
                    try:
                        await ownership_close(retained.close_task)
                    except BaseException as error:
                        if not original_fault_replayed(error):
                            raise
        finally:
            for connection in actual:
                sqlite3.Connection.close(connection)
        try:
            await backend.close()
        except BaseException as error:
            if not original_fault_replayed(error):
                raise


@pytest.mark.asyncio
async def test_unretirable_thread_policy_refuses_before_constructor_admission(monkeypatch):
    backend = SQLiteBackend(":memory:")

    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported thread policy reached actual constructor admission")

    monkeypatch.setattr(aiosqlite, "connect", forbidden)
    try:
        with pytest.raises(ValueError, match="retirement after its original worker exits"):
            await backend._acquire_connection(":memory:", check_same_thread=True)
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_refused_original_connection_owner_never_starts_a_sqlite_worker(tmp_path, monkeypatch):
    backend = SQLiteBackend(str(tmp_path / "refused-creation.db"))
    original, wrappers, refused = aiosqlite.connect, [], []
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    refusal = RuntimeError("Original SQLite acquisition Task admission refused")

    def observe(*args, **kwargs):
        connection = original(*args, **kwargs)
        wrappers.append(connection)
        return connection

    def factory(loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        code = getattr(operation, "cr_code", None)
        if code is not None and code.co_qualname == "SQLiteBackend._acquire_connection.<locals>.connect_original":
            refused.append(operation)
            raise refusal
        return previous(loop, coroutine, **kwargs) if previous else asyncio.Task(coroutine, loop=loop, **kwargs)

    monkeypatch.setattr(aiosqlite, "connect", observe)
    try:
        loop.set_task_factory(factory)
        with pytest.raises(ConnectionError) as failed:
            await backend.connect()
        loop.set_task_factory(previous)
        assert refused and is_execution_control_error(failed.value)
        assert any(error is refusal for error in execution_evidence_carriers(failed.value))
        assert len(wrappers) == 1 and aiosqlite_worker(wrappers[0]).ident is None
        assert not (tmp_path / "refused-creation.db").exists()
    finally:
        loop.set_task_factory(previous)
        await backend.close()
