"""What an SQLite close reports when it overruns its budget (#3460).

Closing the last connection to a WAL database checkpoints the whole -wal
into the main file and fsyncs it, inside the same 1.0s budget as worker
termination. These tests pin what an overrun reports, so a CI failure shows
how much WAL the close had to checkpoint without any extra step.
"""

import asyncio
import json
import logging
import re
import sqlite3
import sys
import threading
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import pytest

import kestrel_sovereign.storage.db.sqlite as sqlite_backend_module
from kestrel_sovereign.async_ownership import await_owned_operation, await_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCommitOutcomeError,
    execution_observation_error,
    is_execution_control_error,
)
from kestrel_sovereign.storage.db import ConnectionError, QueryError, SQLiteBackend
from tests.utils.aiosqlite_workers import (
    CloseStage,
    aiosqlite_worker,
    block_aiosqlite_native_close,
    delay_aiosqlite_worker_exit,
    normal_close_schedule,
    observe_bounded_close_stages,
)
from tests.utils.virtual_clock import (
    HANG_GUARD_SECONDS as _HANG_GUARD_SECONDS,
    await_within_hang_guard,
    pinned_loop_clock,
    wait_for_thread_event,
)
from tests.utils.original_failure import OriginalFailure


async def _normal_backend_close(backend, *, original_connection=None):
    """Retire the captured original even if an observer lost backend's pointer.

    The existing normal functional-test schedule owns the SAME close. A
    different current pointer is never adopted or closed on its authority.
    Every independent worker/reaper join is attempted before validation.
    """
    connection = (
        backend._ensure_connected()
        if original_connection is None else original_connection
    )
    worker = aiosqlite_worker(connection)
    failures = []
    try:
        async with normal_close_schedule() as finish:
            operation = (
                backend.close() if backend._connection is connection else
                sqlite_backend_module._close_aiosqlite_connection(
                    connection, retained_closes=backend._retired_connection_closes,
                )
            )
            await finish(connection, operation, retained_closes=backend._retired_connection_closes)
    except BaseException as error:  # noqa: BLE001 - keep validation and original native cleanup outcomes
        failures.append(error)
    finally:
        try:
            outcome = await await_within_hang_guard(
                await_owned_operation(
                    asyncio.to_thread(worker.join, _HANG_GUARD_SECONDS),
                    retirement_resource=connection, retain_originals=True,
                ),
                step="the captured original primary worker joining on every exit",
            )
            failures.extend((outcome.error, outcome.cancellation))
        except BaseException as error:  # noqa: BLE001 - still harvest the original reaper after a join failure
            failures.append(error)
        finally:
            retained = backend._retired_connection_closes.get(connection)
            if retained is not None and retained.retirement_task is not None:
                try:
                    outcome = await await_within_hang_guard(
                        await_owned_task(retained.retirement_task),
                        step="the captured original primary retirement outcome",
                    )
                    failures.extend((outcome.error, outcome.cancellation))
                except BaseException as error:  # noqa: BLE001 - preserve the exact retirement source
                    failures.append(error)
    try:
        assert not worker.is_alive()
        assert not backend.connection_retirement_pending
    except BaseException as error:  # noqa: BLE001 - assertions cannot gate independent retirement
        failures.append(error)
    terminal = execution_observation_error(*failures)
    if terminal is not None:
        raise terminal

# The patched worker-shutdown window. Close budgets add one 10ms poll to it.
_SHORT_WINDOW_S = 0.05
_SHORT_BUDGET = "budget=0.06s"


@pytest.fixture
def short_close_window(monkeypatch):
    monkeypatch.setattr(
        sqlite_backend_module,
        "AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S",
        _SHORT_WINDOW_S,
    )


async def _connected_wal_backend(path: Path) -> SQLiteBackend:
    """A file-backed backend with committed pages still in its WAL."""
    backend = SQLiteBackend(str(path))
    await backend.connect()
    await backend.execute("CREATE TABLE payload (body BLOB)")
    for _ in range(20):
        await backend.execute("INSERT INTO payload VALUES (randomblob(4000))")
    return backend


def _wal_frames(wal_bytes: int, page_size: int) -> int:
    return (wal_bytes - 32) // (24 + page_size)


async def _page_size(backend: SQLiteBackend) -> int:
    row = await backend.fetch_one("PRAGMA page_size")
    return int(row[0])


def _block_close(backend: SQLiteBackend):
    """Make the backend's next close block in its worker; return the handles."""
    connection = backend._ensure_connected()
    entered, release = block_aiosqlite_native_close(connection)
    return connection, entered, release


async def _retire(backend: SQLiteBackend, release: threading.Event, worker, *, close_task=None) -> None:
    """Let a blocked close finish and wait for its worker to retire."""
    # Keep already-published reapers before releasing their physical worker;
    # the enclosing close can prune a record while harvesting its result.
    retirement_tasks = {
        retained.retirement_task
        for retained in backend._retired_connection_closes.values()
        if retained.retirement_task is not None
    }
    release.set()
    if close_task is not None:
        await await_within_hang_guard(
            asyncio.gather(close_task, return_exceptions=True),
            step="settling the original bounded SQLite close",
        )
    retirement_tasks.update(
        retained.retirement_task
        for retained in backend._retired_connection_closes.values()
        if retained.retirement_task is not None
    )
    await await_within_hang_guard(
        asyncio.gather(*retirement_tasks, return_exceptions=True),
        step="joining the original retained SQLite close tasks",
    )
    await await_within_hang_guard(
        asyncio.to_thread(worker.join, _HANG_GUARD_SECONDS),
        step="joining the original retained SQLite worker",
    )
    assert not worker.is_alive()
    assert not backend.connection_retirement_pending


def _overrun_warnings(caplog) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == sqlite_backend_module.__name__
        and record.levelno == logging.WARNING
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["before-native", "after-native", "worker-exit"])
async def test_snapshot_close_records_actual_native_and_ack_stages(
    tmp_path, monkeypatch, record_property, stage
):
    """Closed native handle is not an acknowledged close or worker retirement."""
    backend = await _connected_wal_backend(tmp_path / "snapshot-stages.db")
    primary = backend._ensure_connected()
    primary_worker = aiosqlite_worker(primary)
    opened = asyncio.Event()
    transaction_entered = asyncio.Event()
    release_transaction = asyncio.Event()
    release_worker_exit = threading.Event()
    worker_exit_entered = threading.Event()
    captured = {}
    timeline = {"test_started_at": time.monotonic()}
    transaction = snapshot_read = original_close = None
    retirement_tasks = set()
    expected_read_error = None
    failures = []
    original_open_snapshot = backend._open_snapshot_read_connection

    async def observed_open_snapshot():
        connection = await original_open_snapshot()
        captured["connection"] = connection
        captured["worker"] = aiosqlite_worker(connection)
        entered, release = block_aiosqlite_native_close(
            connection, after_native_close=stage == "after-native", timeline=timeline
        )
        captured["native_entered"], captured["native_release"] = entered, release
        if stage == "worker-exit":
            release.set()
        opened.set()
        return connection

    async def hold_transaction():
        async with backend.transaction():
            await backend.execute("UPDATE snapshot_probe SET value=1")
            captured["transaction_owner"] = backend._txn_owner
            transaction_entered.set()
            await release_transaction.wait()

    exit_delay = (
        delay_aiosqlite_worker_exit(
            release_worker_exit, worker_exit_entered,
            should_delay=lambda worker: worker is captured.get("worker"),
        )
        if stage == "worker-exit" else nullcontext()
    )
    with observe_bounded_close_stages() as stages, exit_delay:
        try:
            await backend.execute("CREATE TABLE snapshot_probe (value INTEGER NOT NULL)")
            await backend.execute("INSERT INTO snapshot_probe VALUES (0)")
            monkeypatch.setattr(backend, "_open_snapshot_read_connection", observed_open_snapshot)
            transaction = asyncio.create_task(hold_transaction())
            await await_within_hang_guard(transaction_entered.wait(), step="the original snapshot blocker")
            assert captured["transaction_owner"] is transaction
            snapshot_read = asyncio.create_task(backend.fetch_one("SELECT value FROM snapshot_probe"))
            await await_within_hang_guard(opened.wait(), step="the actual independent snapshot opening")
            connection, worker = captured["connection"], captured["worker"]
            assert connection is not primary and worker is not primary_worker
            await wait_for_thread_event(captured["native_entered"], step="the actual native close boundary")
            await stages.reached(connection, CloseStage.CONNECTION_CLOSE)
            original_close = stages.close_tasks[connection]
            timeline["deadline"] = stages.deadlines[connection]
            timeline["close_wait_entered_at"] = stages.entered_at[(CloseStage.CONNECTION_CLOSE, connection)]
            timeline["loop_observed_at"] = asyncio.get_running_loop().time()
            if stage == "worker-exit":
                await wait_for_thread_event(worker_exit_entered, step="the original snapshot worker exit hold")
                await stages.reached(connection, CloseStage.WORKER_EXIT)
                timeline["worker_wait_entered_at"] = stages.entered_at[(CloseStage.WORKER_EXIT, connection)]
                assert original_close.done() and not original_close.cancelled()
                assert original_close.exception() is None
                assert worker.is_alive()
            with pytest.raises(QueryError) as raised:
                await await_within_hang_guard(snapshot_read, step="the unchanged actual snapshot close deadline")
            expected_read_error = raised.value
            assert isinstance(expected_read_error.__cause__, ConnectionError)
            if stage == "after-native":
                assert "native_close_returned_at" in timeline
                assert isinstance(expected_read_error.__cause__.__cause__, sqlite3.ProgrammingError)
            assert worker.is_alive()
            assert backend.connection_retirement_pending
            retained = backend._retired_connection_closes[connection]
            assert retained.close_task is original_close
            assert retained.retirement_task is not None
            retirement_tasks.add(retained.retirement_task)
            timeline["read_observed_at"] = time.monotonic()
            assert backend._connection is primary and primary_worker.is_alive()
        except BaseException as error:
            failures.append(error)
        finally:
            release_worker_exit.set()
            if "native_release" in captured:
                captured["native_release"].set()
            release_transaction.set()
            if "connection" in captured:
                try:
                    # Reuse the existing exact-owned retirement helper. Harvest
                    # every captured original outcome below; never clear a map.
                    await _retire(backend, captured["native_release"], captured["worker"], close_task=snapshot_read)
                    timeline["snapshot_worker_join_observed_at"] = time.monotonic()
                except BaseException as error:
                    failures.append(error)
            originals = {task for task in (snapshot_read, original_close, transaction) if task is not None}
            originals.update(retirement_tasks)
            for task in originals:
                try:
                    outcome = await await_within_hang_guard(await_owned_task(task), step="harvesting an original snapshot owner")
                    if outcome.error is not None and outcome.error is not expected_read_error:
                        deadline_cancel = (
                            stage in {"before-native", "after-native"}
                            and (task is original_close or task in retirement_tasks)
                            and type(outcome.error) is asyncio.CancelledError
                            and not is_execution_control_error(outcome.error)
                            and outcome.error.__cause__ is None
                        )
                        if not deadline_cancel:
                            failures.append(outcome.error)
                    if outcome.cancellation is not None:
                        failures.append(outcome.cancellation)
                    assert task.done()
                except BaseException as error:
                    failures.append(error)
            try:
                assert backend._connection is primary
                if transaction is not None and not transaction.cancelled() and transaction.exception() is None:
                    assert await backend.fetch_one("SELECT value FROM snapshot_probe") == (1,)
            except BaseException as error:
                failures.append(error)
            finally:
                try:
                    await _normal_backend_close(backend, original_connection=primary)
                    assert not primary_worker.is_alive()
                    assert not backend.connection_retirement_pending
                except BaseException as error:
                    failures.append(error)
            record_property("sqlite_snapshot_close_timeline", json.dumps({
                "stage": stage, "times": timeline,
                "primary_id": id(primary), "primary_worker_id": id(primary_worker),
                "snapshot_id": id(captured.get("connection")),
                "snapshot_worker_id": id(captured.get("worker")),
                "transaction_task_id": id(transaction), "read_task_id": id(snapshot_read),
                "original_close_task_id": id(original_close),
                "retirement_task_ids": sorted(id(task) for task in retirement_tasks),
                "product_budget_seconds": sqlite_backend_module._minimum_close_timeout_s(),
                "unchanged_real_clock": True, "no_provider_or_pg": True,
            }, sort_keys=True))
    terminal = execution_observation_error(*failures)
    if terminal is not None:
        raise terminal


@pytest.mark.asyncio
@pytest.mark.parametrize("fault,stage", [
    pytest.param("row", "before-native", id="row-before-native"),
    pytest.param("row", "after-native", id="row-after-native"),
    pytest.param("row", "worker-exit", id="row-worker-exit"),
    pytest.param("query", "after-native", id="query-error"),
    pytest.param("pointer", "after-native", id="original-pointer-lost"),
    pytest.param("retire", "after-native", id="retirement-observer-error"),
    pytest.param("late-ordinary", "after-native", id="validation-and-late-close-error"),
    pytest.param("late-unknown", "after-native", id="validation-and-late-close-unknown"),
])
async def test_snapshot_validation_failure_still_retires_original_primary(
    tmp_path, monkeypatch, record_property, fault, stage,
):
    """The counterproof owns the real primary even when the inner test fails.

    The outer reconciliation is explicitly NOT the inner original close ACK.
    In particular, losing backend._connection must not lose the native owner.
    """
    original_connect = _connected_wal_backend
    original_fetch = SQLiteBackend.fetch_one
    original_await = await_owned_task
    original_retire = _retire
    invoking_task = asyncio.current_task()
    captured = {}
    facts = {"fault": fault, "stage": stage, "validation_entries": 0,
             "inner_primary_close_entries": 0, "outer_reconciliation": False}
    validation_error = (
        AssertionError("original retirement observer failed") if fault == "retire"
        else OSError("original final primary query observation failed")
    )
    validation_source = OriginalFailure.capture(validation_error)
    late_error = (
        ExecutionCommitOutcomeError("unknown") if fault == "late-unknown"
        else OSError("original primary close epilogue failed")
    )
    late_source = OriginalFailure.capture(late_error)
    body_failure = None
    cleanup_failures = []
    owner_errors = []
    expected_outcomes_asserted = False

    async def record_backend(path):
        backend = await original_connect(path)
        primary = backend._ensure_connected()
        captured.update(backend=backend, primary=primary,
                        worker=aiosqlite_worker(primary), real_close=primary.close)

        async def observed_primary_close():
            facts["inner_primary_close_entries"] += 1
            captured["primary_close_task"] = asyncio.current_task()
            await captured["real_close"]()
            if fault in {"late-ordinary", "late-unknown"}:
                raise late_error

        monkeypatch.setattr(primary, "close", observed_primary_close)
        return backend

    async def observe_final_query(backend, query, *args, **kwargs):
        result = await original_fetch(backend, query, *args, **kwargs)
        if (backend is captured.get("backend")
                and asyncio.current_task() is invoking_task
                and query == "SELECT value FROM snapshot_probe"):
            assert result == (1,), "fault did not reach the real committed final row"
            facts["validation_entries"] += 1
            facts["actual_final_row"] = result
            if fault == "row":
                return (2,)
            raise validation_error
        return result

    async def observe_original_owner(task, **kwargs):
        outcome = await original_await(task, **kwargs)
        if (fault == "pointer" and task.done() and outcome.error is None
                and getattr(task.get_coro(), "cr_code", None) is not None
                and task.get_coro().cr_code.co_name == "hold_transaction"):
            assert captured["backend"]._connection is captured["primary"]
            assert captured["worker"].is_alive()
            facts["validation_entries"] += 1
            captured["backend"]._connection = None
        return outcome

    async def observe_retirement(*args, **kwargs):
        await original_retire(*args, **kwargs)
        if fault == "retire":
            assert not args[2].is_alive(), "observer fault preceded real snapshot retirement"
            facts["validation_entries"] += 1
            raise validation_error

    monkeypatch.setattr(sys.modules[__name__], "_connected_wal_backend", record_backend)
    monkeypatch.setattr(SQLiteBackend, "fetch_one", observe_final_query)
    monkeypatch.setattr(sys.modules[__name__], "await_owned_task", observe_original_owner)
    monkeypatch.setattr(sys.modules[__name__], "_retire", observe_retirement)
    try:
        try:
            if fault == "retire":
                await test_snapshot_close_reports_primary_journal_mode(tmp_path, monkeypatch)
            else:
                await test_snapshot_close_records_actual_native_and_ack_stages(
                    tmp_path, monkeypatch, record_property, stage,
                )
        except BaseException as error:  # noqa: BLE001 - inspect the actual original inner failure
            captured["inner_error"] = error
        assert facts["validation_entries"] == 1
        inner_error = captured["inner_error"]
        if fault in {"row", "pointer"}:
            assert isinstance(inner_error, AssertionError)
        else:
            validation_source.assert_retained(inner_error)
        facts["primary_alive_before_outer_reconciliation"] = captured["worker"].is_alive()
        facts["pointer_is_original_before_outer_reconciliation"] = (
            captured["backend"]._connection is captured["primary"]
        )
        assert not captured["worker"].is_alive(), "validation failure skipped original primary retirement"
        assert facts["inner_primary_close_entries"] == 1
        if fault in {"late-ordinary", "late-unknown"}:
            late_source.assert_retained(inner_error)
        assert not captured["backend"].connection_retirement_pending
        expected_outcomes_asserted = True
    except BaseException as error:  # noqa: BLE001 - keep the regression failure through real outer cleanup
        body_failure = error
    finally:
        if "primary" in captured:
            backend, primary, worker = captured["backend"], captured["primary"], captured["worker"]
            try:
                if worker.is_alive():
                    # No admitted inner closer may be retried. The canonical
                    # helper reuses its retained Task if present; otherwise
                    # this is the outer test owner's FIRST physical close.
                    async def outer_first_close():
                        captured["outer_close_task"] = asyncio.current_task()
                        await captured["real_close"]()

                    monkeypatch.setattr(primary, "close", outer_first_close)
                    facts["outer_reconciliation"] = True
                    outcome = await await_within_hang_guard(
                        await_owned_operation(
                            sqlite_backend_module._close_aiosqlite_connection(
                                primary, retained_closes=backend._retired_connection_closes,
                                close_task=captured.get("primary_close_task"),
                            ),
                            retirement_resource=primary, retain_originals=True,
                        ),
                        step="counterproof reconciliation of the captured original primary",
                    )
                    cleanup_failures.extend((outcome.error, outcome.cancellation))
            except BaseException as error:  # noqa: BLE001 - join the native worker even after close failure
                cleanup_failures.append(error)
            finally:
                try:
                    await await_within_hang_guard(
                        asyncio.to_thread(worker.join, _HANG_GUARD_SECONDS),
                        step="counterproof original primary worker join",
                    )
                    assert not worker.is_alive()
                except BaseException as error:  # noqa: BLE001 - preserve actual physical cleanup failure
                    cleanup_failures.append(error)
            original_close = captured.get("primary_close_task")
            try:
                if original_close is not None:
                    outcome = await await_within_hang_guard(
                        original_await(original_close), step="counterproof original primary close outcome",
                    )
                    owner_errors.extend((outcome.error, outcome.cancellation))
                assert original_close is None or original_close.done()
                assert not backend.connection_retirement_pending
            except BaseException as error:  # noqa: BLE001 - retain late owner/validation errors with the body
                cleanup_failures.append(error)
            facts["primary_alive_after_outer_reconciliation"] = worker.is_alive()
            facts["original_close_task_terminal"] = None if original_close is None else original_close.done()
            outer_close = captured.get("outer_close_task")
            facts["outer_close_task_terminal"] = None if outer_close is None else outer_close.done()
            facts["retirement_pending_after_outer_reconciliation"] = backend.connection_retirement_pending
        record_property("validation_retirement_counterproof", json.dumps(facts, sort_keys=True))
        terminal = execution_observation_error(
            body_failure, *cleanup_failures,
            *(owner_errors if not expected_outcomes_asserted else ()),
        )
        if terminal is not None:
            raise terminal


@pytest.mark.asyncio
async def test_slow_close_reports_elapsed_budget_and_wal_size(
    tmp_path, short_close_window, caplog
):
    db_path = tmp_path / "slow-close.db"
    backend = await _connected_wal_backend(db_path)
    page_size = await _page_size(backend)
    wal_bytes = Path(f"{db_path}-wal").stat().st_size
    assert wal_bytes > 0
    connection, entered, release = _block_close(backend)
    worker = aiosqlite_worker(connection)
    close_task = None
    try:
        with observe_bounded_close_stages() as stages, pinned_loop_clock() as clock:
            close_task = asyncio.create_task(backend.close())
            await wait_for_thread_event(entered, step="native SQLite close entry")
            await stages.reached(connection, CloseStage.CONNECTION_CLOSE)
            assert worker.is_alive() and not close_task.done()
            deadline = stages.deadlines[connection]
            clock.advance(deadline - clock.time() + _SHORT_WINDOW_S)
            with caplog.at_level(logging.WARNING, logger=sqlite_backend_module.__name__):
                with pytest.raises(
                    ConnectionError, match="connection close did not complete"
                ) as raised:
                    await await_within_hang_guard(close_task, step="close deadline expiry")
            assert not release.is_set() and worker.is_alive()
            assert backend.connection_retirement_pending
            retained = backend._retired_connection_closes[connection]
            assert retained.close_task is stages.close_tasks[connection]
            assert retained.retirement_task is not None
    finally:
        await _retire(backend, release, worker, close_task=close_task)

    message = str(raised.value)
    assert _SHORT_BUDGET in message
    elapsed = float(re.search(r"elapsed=([0-9.]+)s", message).group(1))
    assert elapsed >= _SHORT_WINDOW_S
    assert "database=slow-close.db" in message
    assert str(tmp_path) not in message
    assert "journal_mode=wal" in message
    assert f"wal_bytes={wal_bytes}" in message
    assert f"wal_pages={_wal_frames(wal_bytes, page_size)}" in message
    assert [record.getMessage() for record in _overrun_warnings(caplog)] == [
        message
    ]


@pytest.mark.asyncio
async def test_overrun_interrupts_before_it_logs(
    tmp_path, short_close_window, monkeypatch
):
    """A log handler stalled on the same disk must not delay the interrupt."""
    backend = await _connected_wal_backend(tmp_path / "ordered.db")
    connection, _entered, release = _block_close(backend)
    worker = aiosqlite_worker(connection)
    order = []
    real_interrupt = connection.interrupt

    async def recording_interrupt():
        order.append("interrupt")
        await real_interrupt()

    class RecordingHandler(logging.Handler):
        def emit(self, record):
            if record.levelno == logging.WARNING:
                order.append("warning")

    monkeypatch.setattr(connection, "interrupt", recording_interrupt)
    handler = RecordingHandler()
    module_logger = logging.getLogger(sqlite_backend_module.__name__)
    module_logger.addHandler(handler)
    try:
        with pytest.raises(ConnectionError):
            await backend.close()
    finally:
        module_logger.removeHandler(handler)
        await _retire(backend, release, worker)

    assert order == ["interrupt", "warning"]


@pytest.mark.asyncio
async def test_normal_close_logs_nothing_and_checkpoints(tmp_path, caplog):
    db_path = tmp_path / "normal-close.db"
    backend = await _connected_wal_backend(db_path)
    assert Path(f"{db_path}-wal").stat().st_size > 0
    worker = aiosqlite_worker(backend._ensure_connected())

    with caplog.at_level(logging.DEBUG, logger=sqlite_backend_module.__name__):
        await _normal_backend_close(backend)

    assert _overrun_warnings(caplog) == []
    assert not backend.is_connected
    assert not worker.is_alive()
    assert not backend.connection_retirement_pending
    # An idle close is not interrupted, so its final checkpoint ran and
    # removed the WAL, as before.
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "normal-close.db"
    ]


@pytest.mark.asyncio
async def test_failed_wal_read_reports_unknown_and_keeps_close_error(
    tmp_path, short_close_window, monkeypatch, caplog
):
    db_path = tmp_path / "unreadable-wal.db"
    backend = await _connected_wal_backend(db_path)
    connection, _entered, release = _block_close(backend)
    worker = aiosqlite_worker(connection)
    real_stat = Path.stat

    def failing_wal_stat(self, *args, **kwargs):
        if self.name.endswith("-wal"):
            raise PermissionError("injected -wal stat failure")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", failing_wal_stat)
    try:
        with caplog.at_level(logging.WARNING, logger=sqlite_backend_module.__name__):
            with pytest.raises(ConnectionError) as raised:
                await backend.close()
    finally:
        monkeypatch.setattr(Path, "stat", real_stat)
        await _retire(backend, release, worker)

    message = str(raised.value)
    assert message.startswith("SQLite connection close did not complete")
    assert "wal_bytes=unknown wal_pages=unknown" in message
    assert "journal_mode=wal" in message
    assert len(_overrun_warnings(caplog)) == 1


@pytest.mark.asyncio
async def test_non_wal_database_reports_journal_mode_without_wal(
    short_close_window, caplog
):
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    await backend.execute("CREATE TABLE payload (body BLOB)")
    connection, _entered, release = _block_close(backend)
    worker = aiosqlite_worker(connection)
    try:
        with caplog.at_level(logging.WARNING, logger=sqlite_backend_module.__name__):
            with pytest.raises(
                ConnectionError, match="connection close did not complete"
            ) as raised:
                await backend.close()
    finally:
        await _retire(backend, release, worker)

    message = str(raised.value)
    assert "database=:memory:" in message
    assert "journal_mode=memory" in message
    assert "wal_" not in message
    assert _SHORT_BUDGET in message


@pytest.mark.asyncio
async def test_non_wal_database_closes_normally(caplog):
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    assert backend._journal_mode == "memory"

    with caplog.at_level(logging.DEBUG, logger=sqlite_backend_module.__name__):
        await _normal_backend_close(backend)

    assert _overrun_warnings(caplog) == []
    assert not backend.is_connected


@pytest.mark.asyncio
async def test_known_non_wal_journal_mode_never_reads_a_stray_wal(tmp_path):
    db_path = tmp_path / "rollback.db"
    db_path.write_bytes(b"")
    Path(f"{db_path}-wal").write_bytes(b"x" * 100)

    diagnostics = sqlite_backend_module._capture_close_diagnostics(
        db_path, "delete"
    )

    assert diagnostics.journal_mode == "delete"
    assert diagnostics.wal is None
    assert "wal_" not in diagnostics.describe(diagnostics.started_at)


@pytest.mark.asyncio
async def test_worker_termination_overrun_reports_wal_read_before_close(
    tmp_path, short_close_window
):
    """The close checkpointed the WAL away; the report keeps its size."""
    db_path = tmp_path / "stuck-worker.db"
    release_worker = threading.Event()
    worker_exit_delayed = threading.Event()
    with delay_aiosqlite_worker_exit(release_worker, worker_exit_delayed):
        backend = await _connected_wal_backend(db_path)
        page_size = await _page_size(backend)
        wal_bytes = Path(f"{db_path}-wal").stat().st_size
        connection = backend._ensure_connected()
        worker = aiosqlite_worker(connection)
        close_task = None
        try:
            with observe_bounded_close_stages() as stages, pinned_loop_clock() as clock:
                close_task = asyncio.create_task(backend.close())
                await wait_for_thread_event(worker_exit_delayed, step="native worker exit gate")
                await stages.reached(connection, CloseStage.WORKER_EXIT)
                assert worker.is_alive() and not close_task.done()
                deadline = stages.deadlines[connection]
                clock.advance(deadline - clock.time() + _SHORT_WINDOW_S)
                with pytest.raises(
                    ConnectionError, match="worker did not terminate"
                ) as raised:
                    await await_within_hang_guard(close_task, step="worker-exit deadline expiry")
                assert not release_worker.is_set() and worker.is_alive()
                assert backend.connection_retirement_pending
                assert (
                    backend._retired_connection_closes[connection].close_task
                    is stages.close_tasks[connection]
                )
        finally:
            await _retire(backend, release_worker, worker, close_task=close_task)

    assert not Path(f"{db_path}-wal").exists()
    message = str(raised.value)
    assert "database=stuck-worker.db" in message
    assert "journal_mode=wal" in message
    assert f"wal_bytes={wal_bytes}" in message
    assert f"wal_pages={_wal_frames(wal_bytes, page_size)}" in message
    assert _SHORT_BUDGET in message


def test_wal_size_counts_frames_with_the_known_page_size(tmp_path):
    db_path = tmp_path / "framed.db"
    # Enough frames that their 24-byte headers add up to more than a page,
    # so a count that ignores them comes out different.
    frames = 30
    Path(f"{db_path}-wal").write_bytes(bytes(32 + frames * (24 + 512)))

    wal = sqlite_backend_module._read_wal_size(db_path, 512)

    assert wal == sqlite_backend_module._WalSize(
        size=32 + frames * (24 + 512), pages=frames
    )


def test_wal_size_without_a_page_size_reports_bytes_only(tmp_path):
    db_path = tmp_path / "unsized.db"
    Path(f"{db_path}-wal").write_bytes(bytes(64))

    wal = sqlite_backend_module._read_wal_size(db_path, None)

    assert wal == sqlite_backend_module._WalSize(size=64, pages=None)


def test_wal_size_never_opens_the_wal(tmp_path, monkeypatch):
    """Every close reads this; only a stat, never a read of the file."""
    db_path = tmp_path / "stat-only.db"
    Path(f"{db_path}-wal").write_bytes(bytes(32 + 2 * (24 + 512)))

    def refuse_open(self, *args, **kwargs):
        raise AssertionError(f"opened {self.name}")

    monkeypatch.setattr(Path, "open", refuse_open)

    assert sqlite_backend_module._read_wal_size(db_path, 512).pages == 2


@pytest.mark.asyncio
async def test_missing_wal_is_reported_absent(tmp_path):
    diagnostics = sqlite_backend_module._capture_close_diagnostics(
        tmp_path / "no-wal.db", "wal"
    )

    assert diagnostics.wal == sqlite_backend_module._WalSize(absent=True)
    assert "wal_bytes=absent wal_pages=absent" in diagnostics.describe(
        diagnostics.started_at
    )


@pytest.mark.asyncio
async def test_snapshot_close_reports_primary_journal_mode(
    tmp_path, monkeypatch
):
    backend = await _connected_wal_backend(tmp_path / "snapshot.db")
    primary = backend._ensure_connected()
    snapshot = await backend._open_snapshot_read_connection()
    _entered, release = block_aiosqlite_native_close(snapshot)
    worker = aiosqlite_worker(snapshot)
    failures = []
    try:
        # Only the deliberately stalled snapshot close has a 60ms budget.
        # Its independently owned primary connection is ordinary teardown,
        # not another injected slow-close operation under that short window.
        with monkeypatch.context() as fault:
            fault.setattr(sqlite_backend_module, "AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S", _SHORT_WINDOW_S)
            with pytest.raises(ConnectionError) as raised:
                await sqlite_backend_module._close_aiosqlite_connection(
                    snapshot,
                    retained_closes=backend._retired_connection_closes,
                    diagnostics=backend._close_diagnostics(),
                )
    except BaseException as error:  # noqa: BLE001 - do not let cleanup replace the original diagnostic failure
        failures.append(error)
    finally:
        try:
            await _retire(backend, release, worker)
        except BaseException as error:  # noqa: BLE001 - retirement observation cannot skip the primary
            failures.append(error)
        finally:
            try:
                await _normal_backend_close(backend, original_connection=primary)
            except BaseException as error:  # noqa: BLE001 - compose both independent originals
                failures.append(error)
    terminal = execution_observation_error(*failures)
    if terminal is not None:
        raise terminal

    message = str(raised.value)
    assert "database=snapshot.db" in message
    assert _SHORT_BUDGET in message
    assert "journal_mode=wal" in message
    assert "wal_bytes=" in message


@pytest.mark.asyncio
async def test_every_backend_close_names_the_database_it_closes(
    tmp_path, monkeypatch
):
    """Snapshot reads, backups and the primary close each pass their file."""
    backend = await _connected_wal_backend(tmp_path / "primary.db")
    reported = []
    real_close = sqlite_backend_module._close_aiosqlite_connection

    holding = asyncio.Event()
    release = asyncio.Event()

    async def hold_transaction():
        async with backend.transaction():
            await backend.execute("INSERT INTO payload VALUES (x'00')")
            holding.set()
            await release.wait()

    async with normal_close_schedule() as finish:
        async def record_diagnostics(connection, **kwargs):
            reported.append(kwargs.get("diagnostics"))
            await finish(connection, real_close(connection, **kwargs), retained_closes=kwargs["retained_closes"])

        monkeypatch.setattr(
            sqlite_backend_module, "_close_aiosqlite_connection", record_diagnostics
        )
        holder = asyncio.create_task(hold_transaction())
        operations = []
        primary_close = None

        async def run_operation(operation, *, step):
            task = asyncio.create_task(operation)
            operations.append(task)
            return await await_within_hang_guard(task, step=step)

        try:
            try:
                await await_within_hang_guard(holding.wait(), step="the other transaction starting")
                # Another task's open transaction routes this read to a snapshot
                # connection, which is closed when the read finishes.
                assert await run_operation(
                    backend.fetch_one("SELECT count(*) FROM payload"), step="snapshot read and close"
                ) == (20,)
            finally:
                release.set()
                await await_within_hang_guard(holder, step="the other transaction releasing")
            await run_operation(
                backend.backup_to(str(tmp_path / "copy.db")), step="backup and destination close"
            )
            primary_close = asyncio.create_task(backend.close())
            operations.append(primary_close)
            await await_within_hang_guard(primary_close, step="the primary close")
        finally:
            release.set()
            try:
                await await_within_hang_guard(
                    asyncio.gather(holder, *operations, return_exceptions=True),
                    step="routing operations and their original cleanup settling",
                )
            finally:
                # Give the primary its FIRST close if an earlier operation
                # failed. An admitted primary is always the same original Task.
                if primary_close is None:
                    primary_close = asyncio.create_task(backend.close())
                await await_within_hang_guard(
                    asyncio.gather(primary_close, return_exceptions=True),
                    step="the original primary close settling on every exit",
                )

    assert [
        (diagnostics.database, diagnostics.journal_mode)
        for diagnostics in reported
    ] == [
        ("primary.db", "wal"),
        ("copy.db", "unknown"),
        ("primary.db", "wal"),
    ]


@pytest.mark.asyncio
async def test_session_factory_close_passes_its_database_to_driver_closes(
    tmp_path, monkeypatch
):
    import kestrel_sovereign.storage.sqla.session as session_module
    from sqlalchemy import text

    from kestrel_sovereign.storage.async_database import AsyncDatabase
    from kestrel_sovereign.storage.sqla import make_session_factory

    db = await AsyncDatabase.sqlite(str(tmp_path / "factory.db"))
    captured = []
    handed_off = []
    real_capture = session_module._capture_close_diagnostics

    def record_capture(db_path, journal_mode):
        diagnostics = real_capture(db_path, journal_mode)
        captured.append(diagnostics)
        return diagnostics

    monkeypatch.setattr(
        session_module, "_capture_close_diagnostics", record_capture
    )
    try:
        factory = make_session_factory(db)
        assert factory._sqlite_path == db.backend.db_path
        real_close_live = factory._close_live_sqlite_connections

        async def record_close_live(*, deadline, diagnostics=None):
            handed_off.append(diagnostics)
            await real_close_live(deadline=deadline, diagnostics=diagnostics)

        monkeypatch.setattr(
            factory, "_close_live_sqlite_connections", record_close_live
        )
        async with factory.read_session() as session:
            await session.execute(text("SELECT 1"))
        await factory.close()
        # A retried close reports what the first attempt saw before it began.
        await factory.close()
    finally:
        await db.close()

    assert [diagnostics.database for diagnostics in captured] == ["factory.db"]
    assert len(handed_off) >= 2
    assert all(
        (diagnostics.database, diagnostics.wal)
        == (captured[0].database, captured[0].wal)
        for diagnostics in handed_off
    )
    # ...but each attempt is timed from its own start.
    assert handed_off[1].started_at > handed_off[0].started_at


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["snapshot", "backup guard", "primary guard"])
async def test_routing_failure_still_retires_primary(tmp_path, monkeypatch, failure_stage):
    """Routing errors and guarded exits cannot abandon original owners."""
    original_fetch = SQLiteBackend.fetch_one
    original_close = sqlite_backend_module._close_aiosqlite_connection
    original_connect = _connected_wal_backend
    original_guard = await_within_hang_guard
    captured = []
    guarded_operations = []

    async def record_backend(path):
        backend = await original_connect(path)
        captured.append((backend, aiosqlite_worker(backend._ensure_connected())))
        return backend

    async def fail_after_snapshot(backend, query, *args, **kwargs):
        result = await original_fetch(backend, query, *args, **kwargs)
        if failure_stage == "snapshot" and query == "SELECT count(*) FROM payload":
            raise RuntimeError("injected routing failure")
        return result

    async def fail_guard(awaitable, *, step, **kwargs):
        target = {
            "backup guard": "backup and destination close",
            "primary guard": "the primary close",
        }.get(failure_stage)
        if step == target:
            # The caller, not the guard, must retain this original operation.
            if not isinstance(awaitable, asyncio.Task):
                awaitable.close()
                raise AssertionError("routing operation was not retained as a Task")
            guarded_operations.append(awaitable)
            await asyncio.sleep(0)  # Let its real operation enter before failure.
            raise RuntimeError("injected routing failure")
        return await original_guard(awaitable, step=step, **kwargs)

    monkeypatch.setattr(sys.modules[__name__], "_connected_wal_backend", record_backend)
    monkeypatch.setattr(sys.modules[__name__], "await_within_hang_guard", fail_guard)
    monkeypatch.setattr(SQLiteBackend, "fetch_one", fail_after_snapshot)
    try:
        with pytest.raises(RuntimeError, match="injected routing failure"):
            await test_every_backend_close_names_the_database_it_closes(tmp_path, monkeypatch)
        assert len(captured) == 1
        assert all(task.done() for task in guarded_operations)
        backend, worker = captured[0]
        assert not backend.is_connected, "routing failure skipped primary close"
        assert not worker.is_alive()
        assert not backend.connection_retirement_pending
    finally:
        # The counterproof must also clean the pre-fix open primary. This is
        # its first close, never a replacement for an already-admitted owner.
        monkeypatch.setattr(sqlite_backend_module, "_close_aiosqlite_connection", original_close)
        monkeypatch.setattr(sys.modules[__name__], "await_within_hang_guard", original_guard)
        for backend, _worker in captured:
            if backend.is_connected:
                await _normal_backend_close(backend)


@pytest.mark.asyncio
async def test_worker_exit_assertion_failure_joins_retirement(tmp_path, monkeypatch):
    """An observer assertion failure must still join the original reaper."""
    original_observe = observe_bounded_close_stages
    original_retain = sqlite_backend_module._retain_aiosqlite_close
    retirement_tasks = []

    class WrongObservedOwner(dict):
        def __getitem__(self, connection):
            return None  # Only the test observer is wrong; native ownership is real.

    @contextmanager
    def wrong_owner_observation():
        with original_observe() as stages:
            stages.close_tasks = WrongObservedOwner()
            yield stages

    def record_retirement(connection, close_task, retained_closes, **kwargs):
        original_retain(connection, close_task, retained_closes, **kwargs)
        retained = retained_closes.get(connection)
        if retained is not None and retained.retirement_task is not None:
            retirement_tasks.append(retained.retirement_task)

    monkeypatch.setattr(sys.modules[__name__], "observe_bounded_close_stages", wrong_owner_observation)
    monkeypatch.setattr(sqlite_backend_module, "_retain_aiosqlite_close", record_retirement)
    monkeypatch.setattr(sqlite_backend_module, "AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S", _SHORT_WINDOW_S)
    try:
        with pytest.raises(AssertionError):
            await test_worker_termination_overrun_reports_wal_read_before_close(tmp_path, None)
        assert retirement_tasks
        assert all(task.done() for task in retirement_tasks), "assertion failure skipped retirement Task join"
    finally:
        await await_within_hang_guard(
            asyncio.gather(*retirement_tasks, return_exceptions=True),
            step="counterproof original retirement owners",
        )
