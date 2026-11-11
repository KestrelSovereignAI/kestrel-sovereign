"""A finite tail observer retains actual HTTP/SQLite work, not a fake failure."""

import asyncio
from contextlib import closing
import sqlite3

import pytest
import pytest_asyncio

from kestrel_sovereign import _async_ownership as ownership
from kestrel_sovereign.hold.enforcement import complete_bound_agent_shutdown
from kestrel_sovereign.host_features.context import (
    SovereignHostContext,
    host_context_cleanup_owners,
)
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from tests.unit.test_cleanup_task_admission import _operation_name
from tests.unit.test_host_context_native_cleanup import _actual_operation
from kestrel_sovereign.hold.enforcement import close_bound_host_context
from kestrel_sovereign._async_ownership import (
    await_owned_task,
    create_owned_task,
    pending_owned_retirement_error,
)
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    execution_cleanup_debts,
    execution_terminal_error,
    is_execution_control_error,
)
from kestrel_sovereign.kestrel_agent import (
    KestrelAgent,
    await_agent_shutdown_completion,
)
from tests.integration.test_lighthouse_snapshot_original_control import (
    native_snapshot as native_snapshot,
)
from tests.utils.virtual_clock import await_within_hang_guard


async def _observe_retained_native_tail(agent, checkpoint, *, step, state=None):
    """Observe actual native progress AFTER the finite observer has returned.

    A bounded observer can expire before HTTP reaches its peer. That does not
    retire, cancel or replace its already-admitted original. Wait separately
    for the real peer/closer checkpoint, retaining those same original Tasks;
    this test observation never renews the production tail budget.
    """
    originals = frozenset(agent._durable_shutdown_tail_tasks)
    assert originals and any(not task.done() for task in originals), (
        "the finite observer must retain its pending native originals"
    )
    if state is not None:
        state["upload_admission_barrier"].set()
    await await_within_hang_guard(checkpoint.wait(), step=step)
    assert originals <= agent._durable_shutdown_tail_tasks, (
        "native progress replaced or forgot an original tail owner"
    )
    assert any(not task.done() for task in originals), (
        "native retirement was reported before the held peer/close ACK"
    )
    return originals


@pytest_asyncio.fixture
async def prepared_tail(native_snapshot, tmp_path, request):
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
    state["hold_path"] = "/not-an-endpoint"
    assert (await service.force_snapshot())[target.name].success
    with closing(sqlite3.connect(str(db))) as connection:
        connection.execute("INSERT INTO payload VALUES ('new data before shutdown')")
        connection.commit()
    state["hold_path"] = "/api/v0/add"
    service.wal_sync_interval = None
    await service.start()
    agent = KestrelAgent(
        did="did:test:owned-offboarding",
        storage_path=str(tmp_path / "unopened-agent.db"),
    )
    assert agent.storage is None
    agent._sync_service = service
    admission = asyncio.Event()
    if not getattr(request, "param", False):
        admission.set()
    state["upload_admission_barrier"] = admission
    try:
        yield agent, native_snapshot
    finally:
        admission.set()
        state["release_server"].set()
        release_close.set()
        tasks.extend(agent._durable_shutdown_tail_tasks)


@pytest.mark.asyncio
@pytest.mark.parametrize("close_fails", [False, True])
@pytest.mark.parametrize("cancel_waiter", [False, True])
@pytest.mark.parametrize("prepared_tail", [False, True], indirect=True, ids=["ready-admission", "held-admission"])
async def test_tail_deadline_joins_original_snapshot_and_preserves_actual_close_failure(
    prepared_tail, close_fails, cancel_waiter
):
    agent, native_snapshot = prepared_tail
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
    original_failure = state["close_error"] if close_fails else None
    state["close_error"] = original_failure
    try:
        cancelled, degraded = await agent._run_durable_shutdown_tail(0.15)
        tasks.extend(agent._durable_shutdown_tail_tasks)
        assert not cancelled and degraded
        originals = await _observe_retained_native_tail(
            agent, entered, step="the original retained snapshot reaching its real HTTP peer", state=state,
        )
        assert entered.is_set()
        assert agent._durable_shutdown_tail_tasks
        assert any(not task.done() for task in agent._durable_shutdown_tail_tasks)
        assert not source_errors, "Observation deadline cancelled the original upload"
        assert not close_entered.is_set(), (
            "Native upload was retired before its peer responded"
        )
        assert vars(agent).get("_shutdown_control") is None
        waiter = create_owned_task(
            agent.wait_for_shutdown_completion(), retain_originals=True
        )
        tasks.append(waiter)
        await asyncio.sleep(0)
        if cancel_waiter:
            waiter.cancel("original tail observer withdrew")
            await asyncio.sleep(0)
            assert any(not task.done() for task in agent._durable_shutdown_tail_tasks)
        state["release_server"].set()
        release_close.set()
        outcome = await asyncio.wait_for(await_owned_task(waiter), 3)
        assert close_entered.is_set() and close_tasks
        assert all(task.done() for task in close_tasks)
        assert not agent._durable_shutdown_tail_tasks
        assert all(task.done() for task in originals)
        assert all(
            client._client is None or client._client.is_closed for client in clients
        )
        if cancel_waiter:
            assert any(
                isinstance(error, asyncio.CancelledError)
                and error.args == ("original tail observer withdrew",)
                for error in _explicit_control_chain(outcome.error)
            )
        if close_fails:
            assert original_failure in tuple(_explicit_control_chain(outcome.error))
            assert execution_cleanup_debts(outcome.error)
            later = create_owned_task(
                agent.wait_for_shutdown_completion(), retain_originals=True
            )
            tasks.append(later)
            repeated = await await_owned_task(later)
            assert original_failure in tuple(_explicit_control_chain(repeated.error))
        else:
            if not cancel_waiter:
                assert outcome.error is None and outcome.cancellation is None
            assert vars(agent).get("_shutdown_control") is None
            await agent.wait_for_shutdown_completion()
    finally:
        state["release_server"].set()
        release_close.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("prepared_tail", [False, True], indirect=True, ids=["ready-admission", "held-admission"])
async def test_tail_callback_publication_failure_keeps_accepted_original_work(
    prepared_tail,
):
    agent, native_snapshot = prepared_tail
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
    failure = OSError("original tail callback publication acknowledgement lost")
    accepted = []
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()

    class PublicationTask(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            super().add_done_callback(callback, context=context)
            if (
                not accepted
                and getattr(callback, "__name__", "") == "retain_native_terminal"
            ):
                accepted.append(self)
                raise failure

    def factory(owner_loop, coroutine, **kwargs):
        return PublicationTask(coroutine, loop=owner_loop, **kwargs)

    loop.set_task_factory(factory)
    try:
        with pytest.raises(OSError, match="callback publication") as caught:
            await agent._run_durable_shutdown_tail(0.15)
        assert caught.value is failure
        assert accepted and accepted[0] in agent._durable_shutdown_tail_tasks
        originals = await _observe_retained_native_tail(
            agent, entered, step="the callback-retained original snapshot reaching its real HTTP peer", state=state,
        )
        assert entered.is_set() and not source_errors and not close_entered.is_set()
        assert vars(agent).get("_shutdown_control") is None
    finally:
        loop.set_task_factory(previous)
        state["release_server"].set()
        release_close.set()
    await agent.wait_for_shutdown_completion()
    assert not agent._durable_shutdown_tail_tasks
    assert accepted[0].done()
    assert all(task.done() for task in originals)
    assert all(client._client is None or client._client.is_closed for client in clients)


@pytest.mark.asyncio
@pytest.mark.parametrize("prepared_tail", [False, True], indirect=True, ids=["ready-admission", "held-admission"])
async def test_native_close_can_observe_completion_without_joining_its_own_parent(
    prepared_tail,
):
    agent, native_snapshot = prepared_tail
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
    observed = []

    async def observe_from_original_closer():
        await agent.wait_for_shutdown_completion()
        observed.append(asyncio.current_task())

    state["close_probe"] = observe_from_original_closer
    cancelled, degraded = await agent._run_durable_shutdown_tail(0.15)
    assert not cancelled and degraded
    originals = await _observe_retained_native_tail(
        agent, entered, step="the self-observing original closer's snapshot reaching its real HTTP peer", state=state,
    )
    assert entered.is_set()
    state["release_server"].set()
    release_close.set()
    await asyncio.wait_for(agent.wait_for_shutdown_completion(), 3)
    assert observed and all(task in close_tasks for task in observed)
    assert not agent._durable_shutdown_tail_tasks
    assert all(task.done() for task in originals)
    assert vars(agent).get("_shutdown_control") is None


@pytest.mark.asyncio
async def test_late_native_callback_cancellation_is_retained_as_original_source(
    prepared_tail,
):
    agent, case = prepared_tail
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
    ) = case
    state["close_error"] = None
    state["hold_path"] = None
    state["upload_failure_as_cancel"] = True
    state["release_server"].set()
    await agent._run_durable_shutdown_tail(0.15)
    originals = await _observe_retained_native_tail(
        agent, close_entered, step="the original failed upload reaching its real native-close boundary",
    )
    assert close_entered.is_set() and source_errors
    original = source_errors[0]
    assert (
        isinstance(original, asyncio.CancelledError) and original.__cause__ is not None
    )
    release_close.set()
    waiter = create_owned_task(
        agent.wait_for_shutdown_completion(), retain_originals=True
    )
    tasks.append(waiter)
    outcome = await await_owned_task(waiter)
    assert all(task.done() for task in originals)
    assert original in tuple(_explicit_control_chain(outcome.error))
    assert original in tuple(
        _explicit_control_chain(vars(agent).get("_shutdown_control"))
    ), "Actual native callback cancellation vanished from the Agent lifetime"


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["hold", "server"])
async def test_original_tail_is_joined_before_borrowed_context_or_new_shutdown(
    prepared_tail, monkeypatch, surface
):
    agent, case = prepared_tail
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
    ) = case
    state["close_error"] = None
    await agent._run_durable_shutdown_tail(0.15)
    originals = set(agent._durable_shutdown_tail_tasks)
    joined, premature, refused = asyncio.Event(), asyncio.Event(), asyncio.Event()
    real_join = ownership.await_owned_task

    async def observed_join(task, *args, **kwargs):
        if task in originals:
            joined.set()
        return await real_join(task, *args, **kwargs)

    monkeypatch.setattr(ownership, "await_owned_task", observed_join)
    loop, previous = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
    )

    def factory(owner_loop, coroutine, **kwargs):
        if (
            surface == "hold"
            and _operation_name(coroutine)
            == "KestrelAgent.wait_for_shutdown_completion"
        ):
            refused.set()
            raise RuntimeError("original completion waiter allocation refused")
        return (
            previous(owner_loop, coroutine, **kwargs)
            if previous
            else asyncio.Task(coroutine, loop=owner_loop, **kwargs)
        )

    backend = None
    if surface == "hold":
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        context = SovereignHostContext(db=backend)
        original_close = backend.close

        async def close():
            premature.set()
            return await original_close()

        monkeypatch.setattr(backend, "close", close)
        operation = complete_bound_agent_shutdown(agent, context)
    else:
        from kestrel_sovereign.server import _shutdown_single_agent

        original_shutdown = agent.shutdown

        async def shutdown():
            premature.set()
            return await original_shutdown()

        monkeypatch.setattr(agent, "shutdown", shutdown)
        operation = _shutdown_single_agent(agent, finite_errors=[])
    loop.set_task_factory(factory)
    caller = create_owned_task(operation, retain_originals=True)
    tasks.append(caller)
    joins = create_owned_task(joined.wait(), retain_originals=True)
    closes = create_owned_task(premature.wait(), retain_originals=True)
    tasks.extend([joins, closes])
    try:
        done, _ = await asyncio.wait(
            {joins, closes}, timeout=3, return_when=asyncio.FIRST_COMPLETED
        )
        assert done
        if surface == "hold":
            assert refused.is_set(), (
                "native completion allocation refusal was not exercised"
            )
        assert joined.is_set() and not premature.is_set(), (
            "Borrowed context closed or new shutdown started before the original native tail joined"
        )
    finally:
        loop.set_task_factory(previous)
        state["release_server"].set()
        release_close.set()
        await await_owned_task(caller)
        assert premature.is_set(), (
            "independent lifecycle owner never ran after the join"
        )
        if backend is not None:
            assert backend._connection is None
        for waiter in [joins, closes]:
            if not waiter.done():
                waiter.cancel()
            await await_owned_task(waiter)
        if backend is not None:
            await backend.close()


@pytest.mark.asyncio
async def test_public_shutdown_observer_stays_bounded_while_native_work_is_pending(
    prepared_tail, monkeypatch
):
    from kestrel_sovereign import kestrel_agent as agent_module

    agent, case = prepared_tail
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
    ) = case
    state["close_error"] = None
    monkeypatch.setattr(agent_module, "KESTREL_AGENT_SHUTDOWN_TIMEOUT_S", 0.2)
    monkeypatch.setattr(agent_module, "KESTREL_SHUTDOWN_DURABLE_RESERVE_S", 0.15)
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    agent._standalone_hold_context = SovereignHostContext(db=backend)
    shutdown = create_owned_task(agent.shutdown(), retain_originals=True)
    observer = asyncio.create_task(asyncio.wait_for(shutdown, timeout=0.6))
    tasks.extend([shutdown, observer])
    try:
        await asyncio.wait_for(entered.wait(), 3)
        done, _ = await asyncio.wait({observer}, timeout=1)
        assert observer in done, (
            "Public shutdown swallowed its caller deadline while native upload was still pending"
        )
        assert backend._connection is not None
        assert any(not owner.done() for owner in agent._durable_shutdown_tail_tasks)
    finally:
        state["release_server"].set()
        release_close.set()
        await await_owned_task(observer)
        await await_owned_task(
            create_owned_task(
                agent.wait_for_shutdown_completion(), retain_originals=True
            )
        )
        try:
            assert backend._connection is None
        finally:
            await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("previously_handled_withdrawal", [False, True])
async def test_ordinary_tail_fault_still_retires_the_owned_context_after_native_join(
    prepared_tail, monkeypatch, previously_handled_withdrawal
):
    from kestrel_sovereign import kestrel_agent as agent_module

    agent, case = prepared_tail
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
    ) = case
    state["close_error"] = None
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    agent._standalone_hold_context = SovereignHostContext(db=backend)
    failure = OSError("original native tail observer publication failed")
    delivered = []
    published = asyncio.Event()
    loop, previous = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
    )

    class PublicationTask(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            super().add_done_callback(callback, context=context)
            if (
                not delivered
                and getattr(callback, "__name__", "") == "retain_native_terminal"
            ):
                delivered.append(self)
                published.set()
                raise failure

    def factory(owner_loop, coroutine, **kwargs):
        return PublicationTask(coroutine, loop=owner_loop, **kwargs)

    monkeypatch.setattr(agent_module, "KESTREL_AGENT_SHUTDOWN_TIMEOUT_S", 0.2)
    monkeypatch.setattr(agent_module, "KESTREL_SHUTDOWN_DURABLE_RESERVE_S", 0.15)
    loop.set_task_factory(factory)

    async def invoke_shutdown():
        if previously_handled_withdrawal:
            asyncio.current_task().cancel("previously handled before this shutdown")
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                pass
            assert asyncio.current_task().cancelling() == 1
        await agent.shutdown()

    caller = create_owned_task(invoke_shutdown(), retain_originals=True)
    tasks.append(caller)
    try:
        await asyncio.wait_for(published.wait(), 3)
        assert backend._connection is not None
        done, _ = await asyncio.wait({caller}, timeout=0.3)
        assert caller not in done, (
            "A previously handled withdrawal or ordinary fault skipped the original cleanup join"
        )
        state["release_server"].set()
        release_close.set()
        result = await await_owned_task(caller)
        assert failure in tuple(_explicit_control_chain(result.error))
        assert backend._connection is None, (
            "shutdown skipped independent context retirement before reporting its ordinary fault"
        )
    finally:
        loop.set_task_factory(previous)
        state["release_server"].set()
        release_close.set()
        await await_owned_task(caller)
        await await_owned_task(
            create_owned_task(
                agent.wait_for_shutdown_completion(),
                retain_originals=True,
            )
        )
        await backend.close()


@pytest.mark.asyncio
async def test_owned_context_native_closer_can_observe_its_waiting_parent(
    prepared_tail,
    monkeypatch,
):
    agent, case = prepared_tail
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    agent._standalone_hold_context = SovereignHostContext(db=backend)
    original_close = backend.close
    entries = []

    async def close():
        entries.append(asyncio.current_task())
        assert asyncio.current_task() in host_context_cleanup_owners(
            agent._standalone_hold_context
        )
        await agent.wait_for_shutdown_completion()
        await original_close()

    monkeypatch.setattr(backend, "close", close)
    waiter = create_owned_task(
        agent.wait_for_shutdown_completion(), retain_originals=True
    )
    case[-1].append(waiter)
    try:
        outcome = await await_owned_task(waiter)
        assert outcome.error is None and outcome.cancellation is None
        assert len(entries) == 1
        assert backend._connection is None
        assert agent._standalone_hold_context is None
        assert not host_context_cleanup_owners(agent._standalone_hold_context)
    finally:
        await original_close()


@pytest.mark.asyncio
@pytest.mark.parametrize("grouped", [False, True])
async def test_original_tail_withdrawal_cannot_be_reobserved_as_a_live_shutdown(
    prepared_tail, monkeypatch, grouped
):
    from kestrel_sovereign import kestrel_agent as agent_module

    agent, case = prepared_tail
    state, release_close, tasks = case[-2], case[5], case[-1]
    state["close_error"] = None
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    agent._standalone_hold_context = SovereignHostContext(db=backend)
    monkeypatch.setattr(agent_module, "KESTREL_AGENT_SHUTDOWN_TIMEOUT_S", 0.2)
    monkeypatch.setattr(agent_module, "KESTREL_SHUTDOWN_DURABLE_RESERVE_S", 0.15)
    original_tail = agent._run_durable_shutdown_tail
    delivered = asyncio.Event()
    withdrawals = []

    async def withdraw_after_original_guard(*args, **kwargs):
        # Forward the actual native tail first. The explicit scheduling point
        # deterministically delivers a real cancellation at its caller boundary;
        # it supplies no completion/retirement result for the original owners.
        await original_tail(*args, **kwargs)
        delivered.set()
        asyncio.current_task().cancel("original tail caller withdrew")
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError as withdrawal:
            withdrawals.append(withdrawal)
            if grouped:
                raise execution_terminal_error(
                    OSError("original finite tail report failed"),
                    withdrawal,
                    retain_originals=True,
                    isolate_control_originals=True,
                )
            raise

    monkeypatch.setattr(
        agent, "_run_durable_shutdown_tail", withdraw_after_original_guard
    )
    shutdown = create_owned_task(agent.shutdown(), retain_originals=True)
    tasks.append(shutdown)
    try:
        await asyncio.wait_for(delivered.wait(), 3)
        done, _ = await asyncio.wait({shutdown}, timeout=1)
        assert shutdown in done, (
            "An already-delivered tail withdrawal was lost before the native completion join"
        )
        assert backend._connection is not None
        result = await await_owned_task(shutdown)
        assert len(withdrawals) == 1 and withdrawals[0] in tuple(
            _explicit_control_chain(result.error)
        )
    finally:
        state["release_server"].set()
        release_close.set()
        await await_owned_task(shutdown)
        await await_owned_task(
            create_owned_task(
                agent.wait_for_shutdown_completion(), retain_originals=True
            )
        )
        await backend.close()


@pytest.mark.asyncio
async def test_borrowed_withdrawal_cannot_drop_a_completed_original_tail_failure(
    prepared_tail, monkeypatch
):
    agent, case = prepared_tail
    service, release_close, state, tasks = case[2], case[5], case[-2], case[-1]
    state["close_error"] = None
    original_snapshot = service.force_snapshot
    source = OSError("original snapshot finalizer failed after native HTTP close")
    originals = []

    async def snapshot():
        originals.append(asyncio.current_task())
        await original_snapshot()
        raise source

    monkeypatch.setattr(service, "force_snapshot", snapshot)
    await agent._run_durable_shutdown_tail(0.15)
    assert originals and originals[0] in agent._durable_shutdown_tail_tasks
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    original_close = backend.close
    release_sql, joined_sql = asyncio.Event(), asyncio.Event()

    async def close():
        await release_sql.wait()
        await original_close()

    monkeypatch.setattr(backend, "close", close)
    continuation = create_owned_task(
        backend.close(), retirement_resource=backend, retain_originals=True
    )
    tasks.append(continuation)
    # This is the exact accepted physical SQLite close, not an observer result.
    agent._durable_shutdown_continuation = continuation
    actual_shield = asyncio.shield

    def shield(owner):
        if owner is continuation:
            joined_sql.set()
        return actual_shield(owner)

    monkeypatch.setattr(asyncio, "shield", shield)
    state["release_server"].set()
    release_close.set()
    first = await await_owned_task(originals[0])
    assert source in tuple(_explicit_control_chain(first.error))
    borrower = create_owned_task(
        agent.wait_for_shutdown_completion(defer_cancellation=False),
        retain_originals=True,
    )
    tasks.append(borrower)
    try:
        await asyncio.wait_for(joined_sql.wait(), 3)
        borrower.cancel("borrowed completion withdrew after the original failure")
        first_observation = await await_owned_task(borrower)
        assert source in tuple(_explicit_control_chain(first_observation.error)), (
            "current withdrawal dropped the earlier original tail failure"
        )
        release_sql.set()
        later = create_owned_task(
            agent.wait_for_shutdown_completion(), retain_originals=True
        )
        tasks.append(later)
        result = await await_owned_task(later)
        assert backend._connection is None
        assert source in tuple(_explicit_control_chain(result.error)), (
            "Later original completion lost the real ordinary tail failure"
        )
    finally:
        state["release_server"].set()
        release_close.set()
        release_sql.set()
        await await_owned_task(continuation)
        await await_owned_task(borrower)
        await original_close()


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_callback_published", [False, True])
async def test_refused_completion_fallback_resnapshots_a_new_original_native_child(
    prepared_tail, monkeypatch, terminal_callback_published
):
    from kestrel_sovereign.server import _shutdown_single_agent

    agent, case = prepared_tail
    service, release_close, state, tasks = case[2], case[5], case[-2], case[-1]
    state["close_error"] = None
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    original_close = backend.close
    release_sql, child_admitted, observed = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    children = []
    timeline = []
    actual_snapshot = service.force_snapshot

    async def close():
        await release_sql.wait()
        await original_close()

    monkeypatch.setattr(backend, "close", close)

    async def snapshot():
        await actual_snapshot()
        child = create_owned_task(
            backend.close(), retirement_resource=backend, retain_originals=True
        )
        children.append(child)
        tasks.append(child)
        child_admitted.set()
        failure = pending_owned_retirement_error(
            backend,
            (child,),
            message="original child close remains pending after its receipt publication failed",
        )
        assert failure is not None
        failure.__cause__ = OSError(
            "original native child close receipt publication failed"
        )
        raise failure

    monkeypatch.setattr(service, "force_snapshot", snapshot)
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    callback_fault = OSError("original native terminal callback publication refused")
    unpublished = []

    class NoTerminalCallback(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            if getattr(callback, "__name__", "") == "retain_native_terminal":
                unpublished.append(self)
                raise callback_fault
            return super().add_done_callback(callback, context=context)

    def unpublished_factory(owner_loop, coroutine, **kwargs):
        return NoTerminalCallback(coroutine, loop=owner_loop, **kwargs)

    if not terminal_callback_published:
        loop.set_task_factory(unpublished_factory)
    try:
        if terminal_callback_published:
            await agent._run_durable_shutdown_tail(0.15)
        else:
            with pytest.raises(BaseException) as publication:
                await agent._run_durable_shutdown_tail(0.15)
            assert (
                callback_fault in tuple(_explicit_control_chain(publication.value))
                and unpublished
            )
    finally:
        loop.set_task_factory(previous_factory)
    actual_join = ownership.await_owned_task

    async def join(owner, *args, **kwargs):
        if owner in children:
            timeline.append("child-join")
            observed.set()
        return await actual_join(owner, *args, **kwargs)

    monkeypatch.setattr(ownership, "await_owned_task", join)
    actual_shutdown = agent.shutdown

    async def shutdown():
        timeline.append("new-shutdown")
        observed.set()
        return await actual_shutdown()

    monkeypatch.setattr(agent, "shutdown", shutdown)
    loop = asyncio.get_running_loop()
    factory = loop.get_task_factory()
    refused = []
    refused_event = asyncio.Event()

    def refuse(owner_loop, coroutine, **kwargs):
        if _operation_name(coroutine) == "KestrelAgent.wait_for_shutdown_completion":
            refused.append(coroutine)
            refused_event.set()
            raise OSError("original completion waiter allocation refused")
        return (
            factory(owner_loop, coroutine, **kwargs)
            if factory
            else asyncio.Task(coroutine, loop=owner_loop, **kwargs)
        )

    loop.set_task_factory(refuse)
    caller = create_owned_task(
        _shutdown_single_agent(agent, finite_errors=[]), retain_originals=True
    )
    tasks.append(caller)
    try:
        # Ensure the fallback snapshots before the child is admitted.
        await asyncio.wait_for(refused_event.wait(), 3)
        assert refused and not children
        state["release_server"].set()
        release_close.set()
        await asyncio.wait_for(child_admitted.wait(), 3)
        await asyncio.wait_for(observed.wait(), 3)
        assert timeline[0] == "child-join", (
            "A fresh shutdown started before the fallback joined a newly published native child"
        )
        assert "new-shutdown" not in timeline
    finally:
        loop.set_task_factory(factory)
        state["release_server"].set()
        release_close.set()
        release_sql.set()
        await await_owned_task(caller)
        for child in children:
            await await_owned_task(child)
        await original_close()


@pytest.mark.asyncio
async def test_prefix_grouped_withdrawal_does_not_reenter_borrowed_completion(
    prepared_tail, monkeypatch
):
    from kestrel_sovereign import kestrel_agent as agent_module

    agent, case = prepared_tail
    state, release_close, tasks = case[-2], case[5], case[-1]
    state["close_error"] = None
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    context = SovereignHostContext(db=backend)
    original_llm_close = agent.llm_service.close
    entered = asyncio.Event()
    sources = []
    refused = []
    original_refusal = OSError("original native prefix SQLite close admission refused")
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()

    def factory(owner_loop, coroutine, **kwargs):
        operation = _actual_operation(coroutine)
        frame = getattr(operation, "cr_frame", None)
        code = getattr(operation, "cr_code", None)
        if (
            not refused
            and code is not None
            and code.co_qualname == "SQLiteBackend.close"
            and frame.f_locals.get("self") is backend
        ):
            refused.append(coroutine)
            raise original_refusal
        return (
            previous(owner_loop, coroutine, **kwargs)
            if previous
            else asyncio.Task(coroutine, loop=owner_loop, **kwargs)
        )

    async def close():
        await original_llm_close()
        try:
            await close_bound_host_context(context)
        except BaseException as source:
            assert is_execution_control_error(source) and refused
            sources.append(source)
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # The original native failed-close receipt is still owed.
                # The prefix owns composition with its outer caller withdrawal.
                raise source

    monkeypatch.setattr(agent.llm_service, "close", close)
    monkeypatch.setattr(agent_module, "KESTREL_AGENT_SHUTDOWN_TIMEOUT_S", 2.0)
    monkeypatch.setattr(agent_module, "KESTREL_SHUTDOWN_DURABLE_RESERVE_S", 0.2)
    loop.set_task_factory(factory)
    caller = create_owned_task(agent.shutdown(), retain_originals=True)
    tasks.append(caller)
    try:
        await asyncio.wait_for(entered.wait(), 3)
        caller.cancel("original prefix caller withdrew")
        done, _ = await asyncio.wait({caller}, timeout=1)
        assert caller in done, (
            "A prefix control group consumed the caller withdrawal before a new native completion wait"
        )
        result = await await_owned_task(caller)
        chain = tuple(_explicit_control_chain(result.error))
        assert sources[0] in chain and original_refusal in chain
        assert any(
            isinstance(error, asyncio.CancelledError)
            and error.args == ("original prefix caller withdrew",)
            for error in chain
        )
        assert backend._connection is not None
    finally:
        loop.set_task_factory(previous)
        state["release_server"].set()
        release_close.set()
        await await_owned_task(caller)
        await await_owned_task(
            create_owned_task(
                agent.wait_for_shutdown_completion(), retain_originals=True
            )
        )
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("refuse_waiter", [False, True])
async def test_context_observer_joins_physical_close_without_adopting_its_callers_later_work(
    prepared_tail, monkeypatch, refuse_waiter
):
    agent, case = prepared_tail
    tasks = case[-1]
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    agent._standalone_hold_context = SovereignHostContext(db=backend)
    original_close = backend.close
    entered, release = asyncio.Event(), asyncio.Event()
    observers = []

    async def close():
        entered.set()
        await release.wait()
        await original_close()

    monkeypatch.setattr(backend, "close", close)

    async def owner():
        await agent.wait_for_shutdown_completion()
        await asyncio.shield(observers[0])

    caller = create_owned_task(owner(), retain_originals=True)
    tasks.append(caller)
    observer = None
    loop, previous = (
        asyncio.get_running_loop(),
        asyncio.get_running_loop().get_task_factory(),
    )
    refused = []
    observation_errors = []
    refusal = OSError("original context completion waiter allocation refused")

    def factory(owner_loop, coroutine, **kwargs):
        if (
            refuse_waiter
            and _operation_name(coroutine)
            == "KestrelAgent.wait_for_shutdown_completion"
        ):
            refused.append(coroutine)
            raise refusal
        return (
            previous(owner_loop, coroutine, **kwargs)
            if previous
            else asyncio.Task(coroutine, loop=owner_loop, **kwargs)
        )

    try:
        await asyncio.wait_for(entered.wait(), 3)
        loop.set_task_factory(factory)
        observer = create_owned_task(
            await_agent_shutdown_completion(agent, finite_errors=observation_errors),
            retain_originals=True,
        )
        observers.append(observer)
        tasks.append(observer)
        await asyncio.sleep(0)
        assert not observer.done(), (
            "completion returned before the original physical close"
        )
        if refuse_waiter:
            assert refused
        release.set()
        done, _ = await asyncio.wait({caller, observer}, timeout=1)
        assert done == {caller, observer}, (
            "Context observer adopted the close callers later work and formed a native join cycle"
        )
        assert backend._connection is None
        assert (await await_owned_task(caller)).error is None
        assert (await await_owned_task(observer)).error is None
        if refuse_waiter:
            assert any(
                refusal in tuple(_explicit_control_chain(error))
                for error in observation_errors
            )
    finally:
        loop.set_task_factory(previous)
        release.set()
        if not caller.done():
            caller.cancel("owned test retires only the post-close caller frame")
        await await_owned_task(caller)
        if observer is not None:
            await await_owned_task(observer)
        await original_close()


@pytest.mark.asyncio
@pytest.mark.parametrize("grouped", [False, True])
async def test_ordinary_prefix_failure_preserves_its_actual_caller_withdrawal(
    prepared_tail, monkeypatch, grouped
):
    from kestrel_sovereign import kestrel_agent as agent_module

    agent, case = prepared_tail
    state, release_close, tasks = case[-2], case[5], case[-1]
    state["close_error"] = None
    original_close = agent.llm_service.close
    entered = asyncio.Event()
    source = OSError("original prefix close report failed after native provider close")
    failure = (
        ExceptionGroup(
            "original ordinary prefix failures",
            [source, ValueError("original prefix cleanup report invalid")],
        )
        if grouped
        else source
    )

    async def close():
        await original_close()
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise failure

    monkeypatch.setattr(agent.llm_service, "close", close)
    monkeypatch.setattr(agent_module, "KESTREL_AGENT_SHUTDOWN_TIMEOUT_S", 2.0)
    monkeypatch.setattr(agent_module, "KESTREL_SHUTDOWN_DURABLE_RESERVE_S", 0.2)
    caller = create_owned_task(agent.shutdown(), retain_originals=True)
    tasks.append(caller)
    try:
        await asyncio.wait_for(entered.wait(), 3)
        caller.cancel("original ordinary-prefix caller withdrew")
        done, _ = await asyncio.wait({caller}, timeout=1)
        assert caller in done, (
            "An ordinary prefix failure consumed the caller withdrawal before the retained native tail join"
        )
        result = await await_owned_task(caller)
        chain = tuple(_explicit_control_chain(result.error))
        assert source in chain
        assert any(
            isinstance(error, asyncio.CancelledError)
            and error.args == ("original ordinary-prefix caller withdrew",)
            for error in chain
        )
    finally:
        state["release_server"].set()
        release_close.set()
        await await_owned_task(caller)
        await await_owned_task(
            create_owned_task(
                agent.wait_for_shutdown_completion(), retain_originals=True
            )
        )


@pytest.mark.asyncio
async def test_two_context_callers_retain_the_same_task_after_successful_physical_close(
    monkeypatch,
):
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    context = SovereignHostContext(db=backend)
    original_close = backend.close
    entered, release = asyncio.Event(), asyncio.Event()

    async def close():
        entered.set()
        await release.wait()
        await original_close()

    monkeypatch.setattr(backend, "close", close)
    first = create_owned_task(close_bound_host_context(context), retain_originals=True)
    second = create_owned_task(close_bound_host_context(context), retain_originals=True)
    try:
        await asyncio.wait_for(entered.wait(), 3)
        release.set()
        outcomes = [await await_owned_task(owner) for owner in (first, second)]
        assert backend._connection is None
        assert all(
            result.error is None and result.cancellation is None for result in outcomes
        ), (
            "Concurrent context observer reread the retired mutable task slot instead of the original close owner"
        )
    finally:
        release.set()
        await await_owned_task(first)
        await await_owned_task(second)
        await original_close()
