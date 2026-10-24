"""Original receipts survive actual native effects and consumer retirement."""

import asyncio
from contextvars import Context
from types import SimpleNamespace

import pytest
from tests.utils.receipt_delivery import ReceiptDelivery
from tests.utils.original_failure import OriginalFailure
from kestrel_sdk.hooks.base import Hook, HookEvent, HookInput
from kestrel_sdk.llm import InferenceLeaseState
from sqlalchemy import text

from kestrel_sovereign._async_ownership import (
    await_owned_task,
    create_owned_task,
    gather_owned,
    owned_task_error,
)
from kestrel_sovereign.a2a.task_manager import TaskManager
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from tests.utils.native_a2a import manager_for
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, _explicit_control_chain, execution_cleanup_debts,
    execution_commit_outcome,
    bind_execution_custody,
    is_execution_control_error,
)
from kestrel_sovereign.features.base import Feature
from kestrel_sovereign.features.delivery.queue import DeliveryQueue
from kestrel_sovereign.heartbeat import HeartbeatConfig, HeartbeatRunner
from kestrel_sovereign.hooks import HooksManager
from kestrel_sovereign.llm.inference_leases import InferenceLeaseRecord
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.sqla import make_session_factory
from tests.unit.test_execution_custody_review29 import native_owner, retire
from tests.unit.test_execution_custody_review48_native import Fence, terminal
from tests.unit.test_inference_lease_coordinator import (
    _coordinator,
    _lease,
    _Provider,
    _quote,
    _request,
)

KINDS = ("unknown", "committed", "cancel_unknown", "group")


async def join(task):
    done, _ = await asyncio.wait({task}, timeout=3)
    assert done, "owned operation did not physically settle within its test guard"
    return await asyncio.wait_for(await_owned_task(task), 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_owned_native_cancel_receipt_is_repeatably_harvestable(tmp_path, kind):
    db = await AsyncDatabase.sqlite(str(tmp_path / "effect.db"))
    original = terminal(kind)
    try:
        async def effect():
            await db.execute("CREATE TABLE effects(id INTEGER PRIMARY KEY)")
            await db.execute("INSERT INTO effects VALUES (1)")
            raise original
        task = create_owned_task(effect())
        result = await join(task)
        assert result.error is original
        assert owned_task_error(task) is original
        assert owned_task_error(task) is original
        assert not task._log_traceback, "physically joined native exception remained unobserved"
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["fanout", "hooks-timed", "hooks-human", "quotes"])
@pytest.mark.parametrize("kind", KINDS)
async def test_native_fanout_physically_joins_stronger_late_sibling(tmp_path, route, kind):
    agent, db = await native_owner(tmp_path)
    original = terminal(kind)
    entered = asyncio.Event()
    children = []
    async def weak():
        children.append(asyncio.current_task())
        await entered.wait()
        raise ExecutionAuthorityError("earlier weak child receipt")
    async def late():
        children.append(asyncio.current_task())
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await db.execute("INSERT INTO effects VALUES (1)")
            raise original
    operation = None
    try:
        if route == "fanout":
            coroutine = gather_owned((weak(), late()))
        elif route.startswith("hooks-"):
            class NativeHook(Hook):
                def __init__(self, name, callback):
                    super().__init__(name=name, events=[HookEvent.POST_TOOL_USE],
                                     awaits_user_input=route == "hooks-human")
                    self.callback = callback
                async def execute(self, _input):
                    return await self.callback()
            manager = HooksManager()
            await manager.register_async(NativeHook("weak", weak))
            await manager.register_async(NativeHook("native", late))
            coroutine = manager.execute_hooks_parallel(HookEvent.POST_TOOL_USE, HookInput(session_id="native", hook_event_name="PostToolUse"))
        else:
            first = _Provider("runpod", total="0.2", ready_seconds=1)
            second = _Provider("vast", total="0.2", ready_seconds=1)
            async def first_quote(*args):
                return await weak()
            async def second_quote(*args):
                return await late()
            first.quote, second.quote = first_quote, second_quote
            coordinator, _ = _coordinator({"runpod": first, "vast": second})
            coroutine = coordinator.acquire(_request())
        operation = create_owned_task(coroutine)
        result = await join(operation)
        assert result.error is original
        assert len(children) == 2 and all(task.done() for task in children)
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
    finally:
        if operation is not None:
            if not operation.done():
                operation.cancel()
            await join(operation)
        for task in children:
            if not task.done():
                task.cancel()
            await join(task)
        await retire(agent)
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_a2a_stores_all_close_after_original_receipt_and_later_cancel(tmp_path, kind):
    from tests.unit.test_a2a_native_shutdown import STORE_NAMES

    db = await AsyncDatabase.sqlite(str(tmp_path / "native-stores.db"))
    manager = TaskManager.from_backend(
        db.backend, scope=NativeRecipientScope("did:test:recipient"), owns_backend=False,
    )
    await manager.initialize()
    original = terminal(kind)
    closed, original_closes = [], []
    for name in STORE_NAMES:
        store = getattr(manager, name)
        close = store.close
        original_closes.append(close)
        async def close_then_receipt(name=name, close=close):
            await close()
            closed.append(name)
            if name == "observability_store":
                raise original
            if name == "session_service":
                raise asyncio.CancelledError("later independent store cleanup")
        store.close = close_then_receipt
    try:
        result = await join(create_owned_task(manager.close()))
        assert result.error is original
        assert set(closed) == STORE_NAMES and len(closed) == len(STORE_NAMES)
        assert db.backend.is_connected, "borrowed primary retired before its owner"
    finally:
        for close in original_closes:
            await close()
        await db.close()



@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["feature", "a2a"])
@pytest.mark.parametrize("kind", KINDS)
async def test_cancelled_drain_joins_native_effect_and_retains_receipt(tmp_path, route, kind, monkeypatch):
    agent, db = await native_owner(tmp_path)
    original = terminal(kind)
    source = OriginalFailure.capture(original)
    withdrawals = []
    entered, unwinding, release = (asyncio.Event() for _ in range(3))
    class NativeFeature(Feature):
        async def initialize(self):
            pass
        @property
        def tool_description(self):
            return "native effect ownership"
    feature = NativeFeature(agent)
    hook = object()
    agent.sleep_hooks = []
    feature._register_sleep_hook(agent, hook)
    manager = manager_for(db.backend, NativeRecipientScope(agent.did), owns_backend=False)
    async def effect():
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            unwinding.set()
            await release.wait()
            await db.execute("INSERT INTO effects VALUES (1)")
            raise original
    child = feature._track_owned_background_task(effect(), name="native") if route == "feature" else manager._track_execution_task(effect(), "original", agent.did)
    # No durable A2A task was submitted: exercise the actual owned drain rather
    # than fabricating a stored task to match this internal native effect.
    manager._execution_authorities.pop(child, None)
    if route == "feature":
        from kestrel_sovereign import _async_ownership as ownership_module
        actual_join = ownership_module.await_owned_task
        async def observed_join(task, *args, **kwargs):
            outcome = await actual_join(task, *args, **kwargs)
            if task is child and outcome.cancellation is not None:
                withdrawals.append(outcome.cancellation)
            return outcome
        monkeypatch.setattr(ownership_module, "await_owned_task", observed_join)
    parent = None
    try:
        await entered.wait()
        parent = create_owned_task(feature.shutdown() if route == "feature" else manager.drain_execution_tasks(cancel=True))
        await unwinding.wait()
        parent.cancel()
        await asyncio.sleep(0)
        assert not parent.done()
        release.set()
        result = await join(parent)
        if route == "feature":
            source.assert_retained(result.error)
            assert withdrawals and all(error in tuple(_explicit_control_chain(result.error))
                                       for error in withdrawals)
        else:
            assert result.error is original
        assert child.done()
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
        if route == "feature":
            assert hook not in agent.sleep_hooks
    finally:
        release.set()
        for task in (child, parent):
            if task is not None:
                if not task.done():
                    task.cancel()
                await join(task)
        await retire(agent)
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["transaction", "savepoint", "execute", "many", "script"])
async def test_sqlite_native_rollback_retains_original_cancellation_group(tmp_path, monkeypatch, route):
    db = await AsyncDatabase.sqlite(str(tmp_path / "rollback.db"))
    await db.execute("CREATE TABLE effects(id INTEGER PRIMARY KEY)")
    original = terminal("group")
    graph, debts = tuple(_explicit_control_chain(original)), tuple(execution_cleanup_debts(original))
    rollback_errors = []
    conn = db.backend._connection
    rollback = conn.rollback
    async def rollback_then_cancel():
        await rollback()
        cancellation = asyncio.CancelledError("later native rollback cancellation")
        rollback_errors.append(cancellation)
        raise cancellation
    monkeypatch.setattr(conn, "rollback", rollback_then_cancel)
    execute = conn.execute
    many = conn.executemany
    script = conn.executescript
    async def execute_then_receipt(sql, *args, **kwargs):
        cursor = await execute(sql, *args, **kwargs)
        if sql.startswith("INSERT"):
            raise original
        if sql.startswith("ROLLBACK TO"):
            cancellation = asyncio.CancelledError("later native savepoint rollback")
            rollback_errors.append(cancellation)
            raise cancellation
        return cursor
    async def many_then_receipt(*args, **kwargs):
        await many(*args, **kwargs)
        raise original
    async def script_then_receipt(*args, **kwargs):
        await script(*args, **kwargs)
        raise original
    if route in ("execute", "savepoint"):
        monkeypatch.setattr(conn, "execute", execute_then_receipt)
    elif route == "many":
        monkeypatch.setattr(conn, "executemany", many_then_receipt)
    elif route == "script":
        monkeypatch.setattr(conn, "executescript", script_then_receipt)
    async def operation():
        if route == "execute":
            await db.execute("INSERT INTO effects VALUES (1)")
        elif route == "many":
            await db.backend.execute_many("INSERT INTO effects VALUES (?)", [(1,)])
        elif route == "script":
            await db.backend.execute_script("INSERT INTO effects VALUES (1);")
        else:
            async with db.transaction():
                if route == "savepoint":
                    async with db.backend.transaction(savepoint=True):
                        await db.execute("INSERT INTO effects VALUES (1)")
                else:
                    await db.execute("INSERT INTO effects VALUES (1)")
                    raise original
    try:
        result = await join(create_owned_task(operation()))
        chain = tuple(_explicit_control_chain(result.error))
        assert original in chain and execution_commit_outcome(result.error) == "unknown"
        assert rollback_errors and rollback_errors[0] in chain, "rollback lost its independent original cancellation"
        assert tuple(_explicit_control_chain(original)) == graph
        assert tuple(execution_cleanup_debts(original)) == debts
    finally:
        monkeypatch.undo()
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_grouped_factory_failure_still_retires_native_primary_database(tmp_path, kind):
    db = await AsyncDatabase.sqlite(str(tmp_path / "factory.db"))
    factory = make_session_factory(db)
    async with factory.write_session() as session:
        await session.execute(text("CREATE TABLE effects(id INTEGER PRIMARY KEY)"))
    original = terminal(kind)
    close = factory.close
    async def close_then_receipt():
        await close()
        raise original
    factory.close = close_then_receipt
    try:
        result = await join(create_owned_task(db.close()))
        assert result.error is original
        assert db.backend._connection is None
        assert not factory.retirement_pending
    finally:
        await db.backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["engine", "session"])
async def test_native_sqlalchemy_cancelled_owner_keeps_original_receipt(tmp_path, monkeypatch, route):
    db = await AsyncDatabase.sqlite(str(tmp_path / "sqla.db"))
    factory = make_session_factory(db)
    original = terminal("cancel_unknown")
    session = factory._async_session()
    await session.execute(text("SELECT 1"))
    if route == "session":
        factory._active_sessions.add(session)
        close = session.close
        async def close_then_receipt():
            await close()
            raise original
        monkeypatch.setattr(session, "close", close_then_receipt)
    else:
        await session.close()
        dispose = type(factory.engine).dispose
        async def dispose_then_receipt(engine, *args, **kwargs):
            await dispose(engine, *args, **kwargs)
            raise original
        monkeypatch.setattr(type(factory.engine), "dispose", dispose_then_receipt)
    try:
        result = await join(create_owned_task(db.close()))
        assert result.error is original
        assert db.backend._connection is None
        assert not factory.retirement_pending
        cached = factory._session_close_error if route == "session" else factory._engine_dispose_error
        assert cached is original
    finally:
        monkeypatch.undo()
        await session.close()
        if db.backend._connection is not None:
            await db.backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["heartbeat", "delivery"])
@pytest.mark.parametrize("kind", KINDS)
async def test_native_worker_poll_control_is_not_an_ordinary_result(tmp_path, monkeypatch, route, kind):
    agent, db = await native_owner(tmp_path)
    original_dispatcher = getattr(agent, "dispatcher", None)
    original = terminal(kind)
    finished = asyncio.Event()
    async def poll(*args, **kwargs):
        await db.execute("INSERT INTO effects VALUES (1)")
        finished.set()
        raise original
    if route == "heartbeat":
        config = HeartbeatConfig(enabled=True, interval_seconds=0, heartbeat_file=str(tmp_path / "HEARTBEAT.md"))
        agent.dispatcher = SimpleNamespace(dispatch_signal=poll)
        worker = HeartbeatRunner(agent, config)
    else:
        worker = DeliveryQueue(db, agent.did)
        monkeypatch.setattr(worker, "process_pending", poll)
    try:
        await worker.start()
        await finished.wait()
        result = await join(worker._task)
        assert result.error is original
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
        for _ in range(2):
            stopped = await join(create_owned_task(worker.stop()))
            assert stopped.error is original
    finally:
        if worker._task is not None and not worker._task.done():
            worker._task.cancel()
            await join(worker._task)
        agent.dispatcher = original_dispatcher
        await retire(agent)
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["ambient", "runtime"])
async def test_inference_status_checks_original_runtime_after_lock(tmp_path, source):
    provider = _Provider("runpod", total="0.2", ready_seconds=1)
    coordinator, _ = _coordinator({"runpod": provider})
    request = _request()
    quote = _quote(request)
    record = InferenceLeaseRecord("runpod", request, quote, lease=_lease(request, quote, state=InferenceLeaseState.READY))
    coordinator._record = record
    await coordinator._lock.acquire()
    entered = asyncio.Event()
    async def status():
        entered.set()
        return await coordinator.status(record.lease.lease_id)
    with bind_execution_custody(Fence()) as original:
        if source == "runtime":
            coordinator._llm_service._execution_custody = original
        task = create_owned_task(status(), context=Context()) if source == "runtime" else create_owned_task(status())
        try:
            await entered.wait()
            with bind_execution_custody(Fence()) as replacement:
                original.revoke("original runtime retired while queued")
                coordinator._llm_service._execution_custody = replacement
                coordinator._lock.release()
                result = await join(task)
            assert is_execution_control_error(result.error)
            assert provider.status_calls == 0
        finally:
            if coordinator._lock.locked():
                coordinator._lock.release()
            await join(task)


@pytest.mark.asyncio
async def test_cancelled_read_closes_exact_late_native_cursor_before_next_begin(tmp_path, monkeypatch):
    db = await AsyncDatabase.sqlite(str(tmp_path / "late-cursor.db"))
    conn = db.backend._connection
    execute, interrupt = conn.execute, conn.interrupt
    returned, allow_return, interrupted = (asyncio.Event() for _ in range(3))
    cursors = []
    async def delayed_cursor(sql, *args, **kwargs):
        cursor = await execute(sql, *args, **kwargs)
        if "late_cursor" in sql:
            # A real active SQLite SELECT returned to its native owner, but its
            # caller has not yet received the cursor-result callback.
            cursors.append(cursor)
            returned.set()
            await allow_return.wait()
        return cursor
    async def observed_interrupt():
        await interrupt()
        interrupted.set()
    monkeypatch.setattr(conn, "execute", delayed_cursor)
    monkeypatch.setattr(conn, "interrupt", observed_interrupt)
    read = create_owned_task(db.fetchval("WITH RECURSIVE late_cursor(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM late_cursor WHERE x<10) SELECT x FROM late_cursor"))
    try:
        await returned.wait()
        read.cancel()
        await interrupted.wait()
        allow_return.set()
        await join(read)
        async with db.transaction():
            await db.execute("CREATE TABLE recovered(id INTEGER PRIMARY KEY)")
        assert await db.table_exists("recovered")
        assert all(cursor._cursor.connection is conn._conn for cursor in cursors)
    finally:
        allow_return.set()
        if not read.done():
            read.cancel()
        await join(read)
        for cursor in cursors:
            await cursor.close()
        monkeypatch.undo()
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_a2a_cancel_group_cannot_skip_other_native_workers(tmp_path, kind):
    agent, db = await native_owner(tmp_path)
    manager = manager_for(db.backend, NativeRecipientScope(agent.did), owns_backend=False)
    original = terminal(kind)
    entered = [asyncio.Event(), asyncio.Event()]
    workers = []
    async def effect(index):
        entered[index].set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await db.execute("INSERT INTO effects VALUES (?)", (index,))
    async def cancel_task(*args, **kwargs):
        # Model the original durable cancellation adapter after a native write,
        # not a fake success or an exception before any I/O took place.
        await db.execute("CREATE TABLE IF NOT EXISTS cancellation_receipts(id INTEGER PRIMARY KEY)")
        raise original
    manager.cancel_task = cancel_task
    workers = [manager._track_execution_task(effect(i), str(i), agent.did) for i in range(2)]
    try:
        await asyncio.gather(*(event.wait() for event in entered))
        outcome = await join(create_owned_task(manager.drain_execution_tasks(cancel=True)))
        assert outcome.error is original
        assert all(worker.done() for worker in workers)
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 2
    finally:
        for worker in workers:
            if not worker.done():
                worker.cancel()
            await join(worker)
        await retire(agent)
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_isolated_supervisor_retains_native_stop_failure(tmp_path, monkeypatch, kind):
    from kestrel_sovereign.features import isolated_runtime
    from tests.unit.test_isolated_feature_runtime import (
        _HostIngressClient,
        _initialized_host_ingress_proxy,
    )
    db = await AsyncDatabase.sqlite(str(tmp_path / "isolated-receipt.db"))
    await db.execute("CREATE TABLE effects(id INTEGER PRIMARY KEY)")
    original = terminal(kind)
    unhealthy = asyncio.Event()
    class NativeClient(_HostIngressClient):
        async def health(self):
            await unhealthy.wait()
            return False
        async def stop(self):
            if not getattr(self, "reported_original", False):
                self.reported_original = True
                await db.execute("INSERT INTO effects VALUES (1)")
                raise original
            self.stopped = True
    client = NativeClient()
    # Yield a genuine event-loop turn without the supervisor's production
    # backoff. No background load generators or unbounded CPU loops.
    sleep = asyncio.sleep
    async def immediate_sleep(_seconds):
        await sleep(0)
    monkeypatch.setattr(isolated_runtime.asyncio, "sleep", immediate_sleep)
    feature, _ = await _initialized_host_ingress_proxy(monkeypatch, tmp_path, lambda **_: client)
    supervisor = feature._supervision_task
    try:
        unhealthy.set()
        outcome = await join(supervisor)
        assert outcome.error is original
        assert feature._client is None
        assert feature._traffic_gate.sealed
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
        stopped = await join(create_owned_task(feature.shutdown()))
        assert stopped.error is original
    finally:
        if not supervisor.done():
            supervisor.cancel()
        await join(supervisor)
        await join(create_owned_task(feature.shutdown()))
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", KINDS)
async def test_completed_cognition_selects_original_native_receipt_before_failed_cas(tmp_path, kind):
    from tests.unit.test_durable_signal_delivery import _close, _dispatcher
    from tests.utils.original_failure import OriginalFailure
    from kestrel_sovereign.execution_custody import execution_commit_outcome
    backend, agent, dispatcher = await _dispatcher(tmp_path / "cognition.db", "did:test:original-route")
    await backend.execute("CREATE TABLE effects(id INTEGER PRIMARY KEY)")
    original = terminal(kind)
    source = OriginalFailure.capture(original)
    weaker = OriginalFailure.capture(ExecutionAuthorityError("renewal lost authority"))
    async def route():
        await backend.execute("INSERT INTO effects VALUES (1)")
        raise original
    child = create_owned_task(route())
    # An already-settled original child races a weak renewal failure. A stale
    # lease intentionally cannot CAS a terminal row; its strongest original
    # receipt must still survive that actual native failure path.
    delivery = ReceiptDelivery(delivery_id="stale-original", consumer_id="original-consumer", lease_token="original-token")
    try:
        await join(child)
        result = await join(create_owned_task(dispatcher._settle_cognition_control(delivery, weaker.error, child)))
        source.assert_retained(result.error)
        weaker.assert_retained(result.error)
        retained = dispatcher._retained_cognition_control_debt[delivery.delivery_id][1]
        source.assert_retained(retained)
        weaker.assert_retained(retained)
        assert execution_commit_outcome(retained) == execution_commit_outcome(original)
        assert await backend.fetch_val("SELECT COUNT(*) FROM effects") == 1
    finally:
        # This test's deliberately stale lease owns no durable row. Retire only
        # its local observation record, never retry or erase a real delivery.
        dispatcher._retained_cognition_control_debt.pop(delivery.delivery_id, None)
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.asyncio
async def test_stopped_settlement_callback_abandons_unknown_native_outcome(tmp_path):
    from kestrel_sovereign.agent.request_lifecycle import RequestCompletionDisposition
    from kestrel_sovereign.signals.dispatcher import SignalDispatcher
    from tests.unit.test_durable_signal_delivery import _LifecycleAgent
    db = await AsyncDatabase.sqlite(str(tmp_path / "settlement.db"))
    await db.execute("CREATE TABLE effects(id INTEGER PRIMARY KEY)")
    agent = _LifecycleAgent("did:test:settlement")
    dispatcher = object.__new__(SignalDispatcher)
    dispatcher._agent = agent
    original = terminal("cancel_unknown")
    guards = []
    async def settle():
        guard = await dispatcher._hold_durable_cognition_stop_completion("original-request")
        guards.append(guard)
        guard.operation.cancel()
        await join(guard.operation)
        await db.execute("INSERT INTO effects VALUES (1)")
        guard.stop_terminalized = True
        raise original
    owner = create_owned_task(settle())
    try:
        assert (await join(owner)).error is original
        await asyncio.sleep(0)
        assert await agent.wait_for_request_completion("original-request") is RequestCompletionDisposition.ABANDONED
        assert await db.fetchval("SELECT COUNT(*) FROM effects") == 1
        assert all(guard.operation.done() for guard in guards)
    finally:
        for guard in guards:
            if not guard.operation.done():
                guard.operation.cancel()
            await join(guard.operation)
        await db.close()
