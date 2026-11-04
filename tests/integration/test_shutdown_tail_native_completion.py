"""A finite tail observer retains actual HTTP/SQLite work, not a fake failure."""

import asyncio
from contextlib import closing
import sqlite3

import pytest
import pytest_asyncio

from kestrel_sovereign import _async_ownership as ownership
from kestrel_sovereign.hold.enforcement import complete_bound_agent_shutdown
from kestrel_sovereign.host_features.context import SovereignHostContext
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from tests.unit.test_cleanup_task_admission import _operation_name
from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    _explicit_control_chain,
    execution_cleanup_debts,
)
from kestrel_sovereign.kestrel_agent import KestrelAgent
from tests.integration.test_lighthouse_snapshot_original_control import native_snapshot  # noqa: F401


@pytest_asyncio.fixture
async def prepared_tail(native_snapshot, tmp_path):
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
    try:
        yield agent, native_snapshot
    finally:
        state["release_server"].set()
        release_close.set()
        tasks.extend(agent._durable_shutdown_tail_tasks)


@pytest.mark.asyncio
@pytest.mark.parametrize("close_fails", [False, True])
@pytest.mark.parametrize("cancel_waiter", [False, True])
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
        assert not cancelled and degraded and entered.is_set()
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
        assert entered.is_set() and not source_errors and not close_entered.is_set()
        assert vars(agent).get("_shutdown_control") is None
    finally:
        loop.set_task_factory(previous)
        state["release_server"].set()
        release_close.set()
    await agent.wait_for_shutdown_completion()
    assert not agent._durable_shutdown_tail_tasks
    assert accepted[0].done()
    assert all(client._client is None or client._client.is_closed for client in clients)


@pytest.mark.asyncio
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
    assert not cancelled and degraded and entered.is_set()
    state["release_server"].set()
    release_close.set()
    await asyncio.wait_for(agent.wait_for_shutdown_completion(), 3)
    assert observed and all(task in close_tasks for task in observed)
    assert not agent._durable_shutdown_tail_tasks
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
async def test_ordinary_tail_fault_still_retires_the_owned_context_after_native_join(
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
    caller = create_owned_task(agent.shutdown(), retain_originals=True)
    tasks.append(caller)
    try:
        await asyncio.wait_for(published.wait(), 3)
        assert backend._connection is not None
        state["release_server"].set()
        release_close.set()
        result = await await_owned_task(caller)
        assert failure in tuple(_explicit_control_chain(result.error))
    finally:
        loop.set_task_factory(previous)
        state["release_server"].set()
        release_close.set()
    try:
        waiter = create_owned_task(
            agent.wait_for_shutdown_completion(), retain_originals=True
        )
        tasks.append(waiter)
        await await_owned_task(waiter)
        assert backend._connection is None, (
            "Ordinary native tail fault skipped the owned standalone context retirement"
        )
    finally:
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
        assert (
            agent._standalone_hold_context_close_task
            in ownership.owned_task_join_ancestors()
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
        assert agent._standalone_hold_context_close_task is None
    finally:
        await original_close()
