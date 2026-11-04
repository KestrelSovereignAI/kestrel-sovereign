"""A finite tail observer retains actual HTTP/SQLite work, not a fake failure."""
import asyncio
from contextlib import closing
import sqlite3

import pytest
import pytest_asyncio

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_cleanup_debts
from kestrel_sovereign.kestrel_agent import KestrelAgent
from tests.integration.test_lighthouse_snapshot_original_control import native_snapshot  # noqa: F401


@pytest_asyncio.fixture
async def prepared_tail(native_snapshot, tmp_path):
    (db, target, service, entered, close_entered, release_close,
     clients, source_errors, close_tasks, state, tasks) = native_snapshot
    state['hold_path'] = '/not-an-endpoint'
    assert (await service.force_snapshot())[target.name].success
    with closing(sqlite3.connect(str(db))) as connection:
        connection.execute("INSERT INTO payload VALUES ('new data before shutdown')")
        connection.commit()
    state['hold_path'] = '/api/v0/add'
    service.wal_sync_interval = None
    await service.start()
    agent = KestrelAgent(did='did:test:owned-offboarding',
                         storage_path=str(tmp_path / 'unopened-agent.db'))
    assert agent.storage is None
    agent._sync_service = service
    try:
        yield agent, native_snapshot
    finally:
        state['release_server'].set()
        release_close.set()
        tasks.extend(agent._durable_shutdown_tail_tasks)


@pytest.mark.asyncio
@pytest.mark.parametrize('close_fails', [False, True])
@pytest.mark.parametrize('cancel_waiter', [False, True])
async def test_tail_deadline_joins_original_snapshot_and_preserves_actual_close_failure(
        prepared_tail, close_fails, cancel_waiter):
    agent, native_snapshot = prepared_tail
    (db, target, service, entered, close_entered, release_close,
     clients, source_errors, close_tasks, state, tasks) = native_snapshot
    original_failure = state['close_error'] if close_fails else None
    state['close_error'] = original_failure
    try:
        cancelled, degraded = await agent._run_durable_shutdown_tail(0.15)
        tasks.extend(agent._durable_shutdown_tail_tasks)
        assert not cancelled and degraded and entered.is_set()
        assert agent._durable_shutdown_tail_tasks
        assert any(not task.done() for task in agent._durable_shutdown_tail_tasks)
        assert not source_errors, 'Observation deadline cancelled the original upload'
        assert not close_entered.is_set(), 'Native upload was retired before its peer responded'
        assert vars(agent).get('_shutdown_control') is None
        waiter = create_owned_task(agent.wait_for_shutdown_completion(), retain_originals=True)
        tasks.append(waiter)
        await asyncio.sleep(0)
        if cancel_waiter:
            waiter.cancel('original tail observer withdrew')
            await asyncio.sleep(0)
            assert any(not task.done() for task in agent._durable_shutdown_tail_tasks)
        state['release_server'].set()
        release_close.set()
        outcome = await asyncio.wait_for(await_owned_task(waiter), 3)
        assert close_entered.is_set() and close_tasks
        assert all(task.done() for task in close_tasks)
        assert not agent._durable_shutdown_tail_tasks
        assert all(client._client is None or client._client.is_closed for client in clients)
        if cancel_waiter:
            assert any(isinstance(error, asyncio.CancelledError) and error.args == ('original tail observer withdrew',)
                       for error in _explicit_control_chain(outcome.error))
        if close_fails:
            assert original_failure in tuple(_explicit_control_chain(outcome.error))
            assert execution_cleanup_debts(outcome.error)
            later = create_owned_task(agent.wait_for_shutdown_completion(), retain_originals=True)
            tasks.append(later)
            repeated = await await_owned_task(later)
            assert original_failure in tuple(_explicit_control_chain(repeated.error))
        else:
            if not cancel_waiter:
                assert outcome.error is None and outcome.cancellation is None
            assert vars(agent).get('_shutdown_control') is None
            await agent.wait_for_shutdown_completion()
    finally:
        state['release_server'].set()
        release_close.set()


@pytest.mark.asyncio
async def test_tail_callback_publication_failure_keeps_accepted_original_work(prepared_tail):
    agent, native_snapshot = prepared_tail
    (db, target, service, entered, close_entered, release_close,
     clients, source_errors, close_tasks, state, tasks) = native_snapshot
    state['close_error'] = None
    failure = OSError('original tail callback publication acknowledgement lost')
    accepted = []
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()

    class PublicationTask(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            super().add_done_callback(callback, context=context)
            if not accepted and getattr(callback, '__name__', '') == 'retain_native_terminal':
                accepted.append(self)
                raise failure

    def factory(owner_loop, coroutine, **kwargs):
        return PublicationTask(coroutine, loop=owner_loop, **kwargs)

    loop.set_task_factory(factory)
    try:
        with pytest.raises(OSError, match='callback publication') as caught:
            await agent._run_durable_shutdown_tail(0.15)
        assert caught.value is failure
        assert accepted and accepted[0] in agent._durable_shutdown_tail_tasks
        assert entered.is_set() and not source_errors and not close_entered.is_set()
        assert vars(agent).get('_shutdown_control') is None
    finally:
        loop.set_task_factory(previous)
        state['release_server'].set()
        release_close.set()
    await agent.wait_for_shutdown_completion()
    assert not agent._durable_shutdown_tail_tasks
    assert accepted[0].done()
    assert all(client._client is None or client._client.is_closed for client in clients)


@pytest.mark.asyncio
async def test_native_close_can_observe_completion_without_joining_its_own_parent(prepared_tail):
    agent, native_snapshot = prepared_tail
    (db, target, service, entered, close_entered, release_close,
     clients, source_errors, close_tasks, state, tasks) = native_snapshot
    state['close_error'] = None
    observed = []

    async def observe_from_original_closer():
        await agent.wait_for_shutdown_completion()
        observed.append(asyncio.current_task())

    state['close_probe'] = observe_from_original_closer
    cancelled, degraded = await agent._run_durable_shutdown_tail(0.15)
    assert not cancelled and degraded and entered.is_set()
    state['release_server'].set()
    release_close.set()
    await asyncio.wait_for(agent.wait_for_shutdown_completion(), 3)
    assert observed and all(task in close_tasks for task in observed)
    assert not agent._durable_shutdown_tail_tasks
    assert vars(agent).get('_shutdown_control') is None
