"""Actual A2A worker failures remain owned through original Root harvesting."""

import asyncio

import pytest

from kestrel_sdk.native_a2a.types import TaskState
from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.execution_custody import execution_evidence_carriers
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.tool_effect_store import ToolEffectReconciliationRequired
from kestrel_sovereign.tools.effect_journal import ToolEffectTurn, current_tool_effect_turn
from tests.unit.test_tool_effect_execution import owner as owner
from tests.unit.test_tool_effect_invocation import governed
from tests.utils.native_a2a import manager_for

pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(30)]


@pytest.mark.parametrize('ordering', ['completed-before-close', 'pending-at-close', 'failed-admission-before-close'])
async def test_native_async_worker_failure_reaches_original_root(owner, monkeypatch, ordering):
    manager = manager_for(owner.tool_effect_store.backend, NativeRecipientScope(owner.did), owns_backend=False)
    await manager.initialize()
    failure = RuntimeError('original accepted asynchronous worker failed before handler entry')
    release, root_closing = asyncio.Event(), asyncio.Event()
    workers, tasks, turns, handler_calls = [], [], [], []

    class WorkerFeature(Feature):
        tool_description = 'Original asynchronous worker failure probe'

        async def initialize(self):
            pass

        @tool(name='owned_worker_failure', description='Never enter after the original worker failed')
        async def owned_worker_failure(self):
            handler_calls.append(True)
            return {'success': True}

    feature = WorkerFeature(owner)
    owner.features[feature.name] = feature
    manager.register_provider(feature.name, feature.get_agent_skills(), feature)
    track = manager._track_execution_task

    def observed_track(*args, **kwargs):
        worker = track(*args, **kwargs)
        workers.append(worker)
        return worker

    monkeypatch.setattr(manager, '_track_execution_task', observed_track)
    close = ToolEffectTurn.close

    async def observed_close(turn, *args, **kwargs):
        if turn.owner is owner:
            root_closing.set()
        return await close(turn, *args, **kwargs)

    monkeypatch.setattr(ToolEffectTurn, 'close', observed_close)
    if ordering == 'failed-admission-before-close':
        lock = owner._get_privacy_transition_lock()
        admit = lock.admit_owned_task

        def admit_then_refuse_publication(parent, worker, admission):
            result = admit(parent, worker, admission)
            if worker.get_name().startswith('a2a-task-'):
                raise failure
            return result

        # Actual constructor and original scope admission run first. The
        # native producer returns its accepted, failed-gated Task unchanged.
        monkeypatch.setattr(lock, 'admit_owned_task', admit_then_refuse_publication)
    else:
        async def fail_before_handler(*args, **kwargs):
            await release.wait()
            raise failure

        monkeypatch.setattr(manager, '_execute_async', fail_before_handler)

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        turns.append(current_tool_effect_turn(agent))
        task = await manager.execute_skill(feature.name, 'owned_worker_failure', {}, sync=False)
        tasks.append(task)
        if ordering != 'pending-at-close':
            release.set()
            await asyncio.wait_for(await_owned_task(workers[0]), 5)
            await asyncio.sleep(0)  # Let actual TaskManager completion callbacks run.
            assert workers[0].done()
        return task

    invocation = create_owned_task(invoke(owner, 'original asynchronous acceptance', invocation_id='original-worker-' + ordering))
    try:
        if ordering == 'pending-at-close':
            await asyncio.wait_for(root_closing.wait(), 5)
            assert len(workers) == 1 and not workers[0].done()
            release.set()
        result = await asyncio.wait_for(await_owned_task(invocation), 5)
        assert result.error is not None, 'Completed asynchronous worker failure escaped its originating Root'
        assert any(carrier is failure for carrier in execution_evidence_carriers(result.error)), (
            'Originating Root replaced the original asynchronous worker failure'
        )
        assert handler_calls == [] and len(workers) == 1 and workers[0].done()
        assert len(turns) == 1 and turns[0].pending == {} and not turns[0].live
        assert len(tasks) == 1
        persisted = await manager.task_store.get_for_recipient(tasks[0].id, manager.did)
        assert persisted.status.state == TaskState.TASK_STATE_SUBMITTED
        assert await owner.tool_effect_store.database.fetchall('SELECT state FROM tool_effect_claims') == [('RETURNED_OK',)]
        with pytest.raises(ToolEffectReconciliationRequired):
            await invoke(owner, 'original asynchronous acceptance', invocation_id='original-worker-' + ordering)
        assert len(tasks) == 1 and handler_calls == []
    finally:
        release.set()
        await await_owned_task(invocation)
        for worker in workers:
            await await_owned_task(worker)
        try:
            await manager.close()
        except BaseException as error:
            assert any(carrier is failure for carrier in execution_evidence_carriers(error)), (
                'Manager cleanup lost or replaced the original worker error'
            )
