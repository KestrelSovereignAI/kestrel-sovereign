"""Native retirement and admission receipts on the review-50 exit paths."""

import asyncio
import time

import aiosqlite
import pytest
from sqlalchemy import text

from kestrel_sovereign._async_ownership import create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    bind_execution_custody,
    is_execution_control_error,
)
from kestrel_sovereign.signals import FAILED, DurableConsumerRegistration
from kestrel_sovereign.storage.async_database import (
    AsyncDatabase,
    DatabaseInitializationCleanupError,
    DatabaseRetirementError,
    _close_failed_database_initialization,
    _close_owned_database,
)
from kestrel_sovereign.storage.db.sqlite import _aiosqlite_worker_is_alive
from kestrel_sovereign.storage.sqla import make_session_factory
from tests.unit.test_durable_signal_delivery import (
    _close,
    _dispatcher,
    _registration,
    _signal,
)
from tests.unit.test_execution_custody_review48_native import Fence, terminal
from tests.unit.test_execution_custody_review49_native import KINDS, join
from tests.unit.test_inference_lease_coordinator import (
    _coordinator,
    _Provider,
    _quote,
    _request,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_successful_cognition_cas_returns_original_stronger_child(tmp_path, kind):
    from tests.utils.original_failure import OriginalFailure
    from kestrel_sovereign.execution_custody import execution_commit_outcome
    backend, agent, dispatcher = await _dispatcher(tmp_path / "cognition.db", "did:test:review50")
    store = dispatcher._durable_store
    await store.register_consumer(DurableConsumerRegistration(
        consumer_id="review50", source="provider.message", agent_id=agent.did,
    ))
    await store.persist_signal(_signal(agent_id=agent.did), agent_id=agent.did,
                               source_event_id="review50", retention_days=1)
    delivery = await store.claim_delivery(agent_id=agent.did, consumer_id="review50",
                                          executor_id=dispatcher._durable_delivery_owner)
    assert delivery is not None
    await backend.execute("CREATE TABLE effects(id INTEGER PRIMARY KEY)")
    original = terminal(kind)
    source = OriginalFailure.capture(original)
    weaker = OriginalFailure.capture(ExecutionAuthorityError("renewal lost authority"))

    async def route():
        await backend.execute("INSERT INTO effects VALUES (1)")
        raise original

    try:
        child = create_owned_task(route())
        assert (await join(child)).error is original
        outcome = await join(create_owned_task(dispatcher._settle_cognition_control(
            delivery, weaker.error, child,
        )))
        assert outcome.error is None
        source.assert_retained(outcome.result)
        weaker.assert_retained(outcome.result)
        assert execution_commit_outcome(outcome.result) == execution_commit_outcome(original)
        assert delivery.delivery_id not in dispatcher._retained_cognition_control_debt
        row = await store.get_delivery(agent_id=agent.did, consumer_id="review50",
                                       delivery_id=delivery.delivery_id)
        assert row.status == FAILED
        assert await backend.fetch_val("SELECT COUNT(*) FROM effects") == 1
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["owned", "initialization"])
async def test_grouped_failed_close_preserves_original_and_native_owner(tmp_path, monkeypatch, route):
    db = await AsyncDatabase.sqlite(str(tmp_path / "owner.db"))
    connection = db.backend._connection
    await db.execute("CREATE TABLE effects(id INTEGER PRIMARY KEY)")
    await db.execute("INSERT INTO effects VALUES (1)")
    original = terminal("group")

    async def failed_close():
        raise original

    monkeypatch.setattr(db, "close", failed_close)
    try:
        call = (_close_owned_database(db) if route == "owned" else
                _close_failed_database_initialization(db, RuntimeError("initialization failed")))
        outcome = await join(create_owned_task(call))
        expected = DatabaseRetirementError if route == "owned" else DatabaseInitializationCleanupError
        assert isinstance(outcome.error, expected)
        assert outcome.error.retirement_owner is db
        assert outcome.error.cleanup_error is original
        assert is_execution_control_error(outcome.error)
        assert _aiosqlite_worker_is_alive(connection)
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
    finally:
        monkeypatch.undo()
        await db.close()
    assert not _aiosqlite_worker_is_alive(connection)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["cancel_unknown", "group"])
async def test_native_cancelled_write_drain_latches_control_and_still_closes(tmp_path, monkeypatch, kind):
    db = await AsyncDatabase.sqlite(str(tmp_path / "drain.db"))
    backend, original = db.backend, terminal(kind)
    connection = backend._connection
    await connection.execute("CREATE TABLE effects(id INTEGER PRIMARY KEY)")
    await connection.commit()
    await connection.execute("BEGIN")
    await connection.execute("INSERT INTO effects VALUES (1)")

    async def rollback_then_receipt(conn, *, started_at):
        await conn.rollback()
        raise original

    monkeypatch.setattr(backend, "_rollback_cancelled_write", rollback_then_receipt)
    try:
        drain = create_owned_task(backend._drain_cancelled_write(
            connection, started_at=asyncio.get_running_loop().time(),
        ))
        backend._cancelled_write_drain = drain
        await join(drain)
        assert backend._cancelled_write_drain_error is original
        assert backend._cancelled_write_drain is None
        assert (await join(create_owned_task(db.fetchval("SELECT COUNT(*) FROM effects")))).error is original
        cursor = await connection.execute("SELECT COUNT(*) FROM effects")
        assert (await cursor.fetchone())[0] == 0
        await cursor.close()
        closed = await join(create_owned_task(db.close()))
        assert closed.error is original
        assert backend._connection is None
        assert not _aiosqlite_worker_is_alive(connection)
    finally:
        monkeypatch.undo()
        if backend._connection is not None:
            await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_sqlite_native_close_receipt_outweighs_caller_cancellation(tmp_path, monkeypatch, kind):
    db = await AsyncDatabase.sqlite(str(tmp_path / "close.db"))
    connection = db.backend._connection
    await db.execute("CREATE TABLE effects(id INTEGER PRIMARY KEY)")
    await db.execute("INSERT INTO effects VALUES (1)")
    native_close, original = connection.close, terminal(kind)
    from tests.utils.original_failure import OriginalFailure
    from kestrel_sovereign.execution_custody import _explicit_control_chain
    from kestrel_sovereign import _async_ownership as ownership
    snapshot = OriginalFailure.capture(original)
    actual_join, withdrawals = ownership.await_owned_task, []

    async def observed_join(*args, **kwargs):
        result = await actual_join(*args, **kwargs)
        if result.cancellation is not None:
            withdrawals.append(result.cancellation)
        return result

    monkeypatch.setattr(ownership, "await_owned_task", observed_join)
    entered, finish = asyncio.Event(), asyncio.Event()

    async def close_then_receipt():
        await native_close()
        entered.set()
        await finish.wait()
        raise original

    monkeypatch.setattr(connection, "close", close_then_receipt)
    owner = create_owned_task(db.close())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        owner.cancel()
        finish.set()
        result = await join(owner)
        snapshot.assert_retained(result.error)
        assert withdrawals and all(error in tuple(_explicit_control_chain(result.error)) for error in withdrawals)
        assert not _aiosqlite_worker_is_alive(connection)
    finally:
        finish.set()
        await join(owner)
        monkeypatch.undo()
        if db.backend._connection is not None:
            await db.backend.close()


@pytest.mark.asyncio
async def test_grouped_engine_disposal_does_not_skip_checked_out_native_workers(tmp_path, monkeypatch):
    db = await AsyncDatabase.sqlite(str(tmp_path / "engine.db"))
    factory = make_session_factory(db)
    sessions = [factory._async_session(), factory._async_session()]
    for session in sessions:
        await session.execute(text("SELECT 1"))
    connections = tuple(factory._sqlite_connections)
    assert len(connections) == 2
    original = terminal("group")

    async def fail_before_dispose(engine, *args, **kwargs):
        raise original

    monkeypatch.setattr(type(factory.engine), "dispose", fail_before_dispose)
    try:
        closed = await join(create_owned_task(db.close()))
        assert closed.error is original
        assert not any(_aiosqlite_worker_is_alive(conn) for conn in connections)
        assert not factory.retirement_pending
        assert db.backend._connection is None
    finally:
        monkeypatch.undo()
        # Driver workers have physically retired; SQLAlchemy can now discard
        # its pool records without trying to roll back their closed handles.
        for session in sessions:
            await session.invalidate()
        await factory.engine.dispose()
        if db.backend._connection is not None:
            await db.backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_caller", [False, True])
async def test_all_native_driver_close_receipts_are_selected_after_physical_join(tmp_path, monkeypatch, cancel_caller):
    db = await AsyncDatabase.sqlite(str(tmp_path / "drivers.db"))
    factory = make_session_factory(db)
    connections = [await aiosqlite.connect(str(tmp_path / f"driver-{i}.db")) for i in range(2)]
    original = terminal("group")
    entered, finish = asyncio.Event(), asyncio.Event()
    for i, connection in enumerate(connections):
        await connection.execute("CREATE TABLE effects(id INTEGER PRIMARY KEY)")
        await connection.execute("INSERT INTO effects VALUES (1)")
        await connection.commit()
        native_close = connection.close

        async def close_then_receipt(native_close=native_close, index=i):
            await native_close()
            if index == 1:
                entered.set()
                await finish.wait()
                raise original
            raise RuntimeError("earlier ordinary driver error")

        monkeypatch.setattr(connection, "close", close_then_receipt)
    factory._sqlite_connections.extend(connections)
    factory._close_started = True
    factory._engine_dispose_task = create_owned_task(factory.engine.dispose())
    await join(factory._engine_dispose_task)
    owner = create_owned_task(factory._close_live_sqlite_connections(
        deadline=asyncio.get_running_loop().time() + 3,
    ))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        if cancel_caller:
            owner.cancel()
        finish.set()
        assert (await join(owner)).error is original
        assert not any(_aiosqlite_worker_is_alive(conn) for conn in connections)
        assert not factory.retirement_pending
    finally:
        finish.set()
        await join(owner)
        monkeypatch.undo()
        await join(create_owned_task(db.close()))


@pytest.mark.asyncio
async def test_late_session_and_engine_retirement_selects_strongest_before_detach(tmp_path, monkeypatch):
    db = await AsyncDatabase.sqlite(str(tmp_path / "late.db"))
    factory = make_session_factory(db)
    original = terminal("cancel_unknown")
    async with factory.write_session() as session:
        await session.execute(text("CREATE TABLE effects(id INTEGER PRIMARY KEY)"))
        await session.execute(text("INSERT INTO effects VALUES (1)"))
    await factory.close()
    factory._session_close_error = RuntimeError("late ordinary session error")
    factory._session_close_error_reported = False
    factory._engine_dispose_error = original
    factory._engine_dispose_error_reported = False
    # Disposal already physically finished; exercise actual native factory
    # finalization, not a substitute factory whose pending predicate is fake.
    with pytest.raises(type(original)) as raised:
        factory.finalize_retirement()
    assert raised.value is original
    assert factory._session_close_error_reported
    assert factory._engine_dispose_error_reported
    assert not factory.retirement_pending
    assert (await join(create_owned_task(db.close()))).error is original
    assert db.backend._connection is None


@pytest.mark.asyncio
async def test_caused_hold_state_failure_is_control_not_a_normal_signal_result(tmp_path, monkeypatch):
    from kestrel_sovereign.hold import HoldStateError
    backend, agent, dispatcher = await _dispatcher(tmp_path / "hold.db", "did:test:hold-control")
    await backend.execute("CREATE TABLE effects(id INTEGER PRIMARY KEY)")
    original = terminal("group")

    async def hold_read():
        await backend.execute("INSERT INTO effects VALUES (1)")
        raise HoldStateError("native hold read failed") from original

    monkeypatch.setattr(dispatcher, "_agent_is_held", hold_read)
    try:
        outcome = await join(create_owned_task(dispatcher._held_signal_result(
            _signal(agent_id=agent.did), _registration(agent), time.monotonic(),
            durable=True, schedule_outcome=False,
        )))
        assert isinstance(outcome.error, HoldStateError)
        assert outcome.error.__cause__ is original
        assert is_execution_control_error(outcome.error)
        assert await backend.fetch_val("SELECT COUNT(*) FROM effects") == 1
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_queued_quote_cannot_dispatch_after_original_admission_retires(tmp_path):
    db = await AsyncDatabase.sqlite(str(tmp_path / "quote.db"))
    await db.execute("CREATE TABLE effects(id INTEGER PRIMARY KEY)")
    provider = _Provider("runpod", total="0.2", ready_seconds=1)
    coordinator, _ = _coordinator({"runpod": provider})
    called = False

    async def native_quote(request):
        nonlocal called
        called = True
        # Direct native SQL models the irreversible provider boundary without
        # invoking a provider, API, or metered adapter. Core's SQL fence cannot
        # rescue a missing pre-provider admission check here.
        cursor = await db.backend._connection.execute("INSERT INTO effects VALUES (1)")
        await cursor.close()
        await db.backend._connection.commit()
        return _quote(request)

    provider.quote = native_quote
    try:
        with bind_execution_custody(Fence()) as original:
            coordinator._llm_service._execution_custody = original
            owner = create_owned_task(coordinator._quote_provider(provider, _request()))
            original.revoke("caller retired before queued quote dispatch")
            outcome = await join(owner)
        assert is_execution_control_error(outcome.error)
        assert not called
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 0
    finally:
        await db.close()
