"""Real callback admission and physical closure keep finite observer evidence local."""
import asyncio
from contextlib import AsyncExitStack

import pytest

from kestrel_sdk.resources import ResourceOutcomeUnknown
from kestrel_sovereign._async_ownership import OwnedCallbackScope, await_owned_task, create_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCommitOutcomeError, execution_evidence_carriers, execution_native_source_error,
    is_execution_control_error, execution_observer_cancellations,
)
from kestrel_sovereign.llm.codex_app_server import CodexAppServerClient
from kestrel_sovereign.storage.privacy_wrapper import ReentrantTransitionLock
from tests.unit.test_tool_effect_execution import owner as owner
from tests.unit.test_tool_effect_invocation import governed


def native_failure(kind):
    error = (asyncio.CancelledError('original source cancellation') if kind == 'cancel' else
        ResourceOutcomeUnknown('ordinary SDK outcome uncertainty') if kind == 'ordinary' else
        ExecutionCommitOutcomeError('unknown') if kind == 'unknown' else
        ExecutionCommitOutcomeError('committed') if kind == 'committed' else
        BaseExceptionGroup('original native source group',
            [ExecutionCommitOutcomeError('unknown'), asyncio.CancelledError('source withdrawal')]))
    cause = ValueError('original physical source cause')
    error.__cause__ = cause
    return error, cause


def assert_original_graph(error, cause, observed, marker):
    assert any(item is error for item in execution_evidence_carriers(observed))
    assert any(isinstance(item, asyncio.CancelledError) and item.args == (marker,)
        for item in execution_evidence_carriers(observed)), 'finite observation lost its own withdrawal'
    assert error.__cause__ is cause, 'observer withdrawal rewrote the original shared source graph'
    native = execution_native_source_error(observed)
    if is_execution_control_error(error):
        assert any(item is error for item in execution_evidence_carriers(native))
    else:
        assert native is None
    assert not any(isinstance(item, asyncio.CancelledError) and item.args == (marker,)
        for item in execution_evidence_carriers(native)), 'observer withdrawal escaped into native source projection'
    client = CodexAppServerClient(binary='unused-offline-codex')
    client._retain_request_failure(observed)
    shared = getattr(client, '_server_request_failure', None)
    assert not any(isinstance(item, asyncio.CancelledError) and item.args == (marker,)
        for item in execution_evidence_carriers(shared)), 'shared Codex connection retained a foreign observer withdrawal'
    if native is not None:
        assert any(item is error for item in execution_evidence_carriers(shared))


@pytest.mark.asyncio
@pytest.mark.parametrize('mode,kind', [(mode,kind)
    for mode in ['park', 'yield', 'scope-close', 'join']
    for kind in ['cancel', 'ordinary', 'unknown', 'committed', 'group']
    if mode != 'join' or kind not in {'cancel', 'ordinary'}])
async def test_original_source_graph_survives_cancelled_writer_restoration(mode, kind, tmp_path):
    lock = ReentrantTransitionLock()
    entered, release, body_failed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original, cause = native_failure(kind)
    receipt = tmp_path / 'original-writer-settlement'
    children = []
    marker = 'observer withdrew during original writer restoration'

    async def child():
        entered.set()
        await release.wait()
        receipt.write_text('physical writer settled')

    error_release = asyncio.Event()

    async def source():
        children.append(lock.start_owned_native_work(child))
        await entered.wait()
        await error_release.wait()
        body_failed.set()
        raise original

    async def fail_body():
        await entered.wait()
        body_failed.set()
        raise original

    async def operation():
        async with lock.policy_span() if mode in {'scope-close', 'join'} else lock:
            if mode == 'join':
                children.append(lock.start_owned_native_work(source, acquire_span=False))
                await entered.wait()
                error_release.set()
                await lock.join_owned_native_work()
            else:
                task = lock.start_owned_native_work(child)
                children.append(task)
                if mode == 'scope-close':
                    await fail_body()
                else:
                    span = lock.park_native_wait((task,)) if mode == 'park' else lock.yield_native_writer()
                    async with span:
                        await fail_body()

    calling = create_owned_task(operation())
    try:
        await asyncio.wait_for(body_failed.wait(), .5)
        await asyncio.sleep(0)
        assert lock._writer in children and not calling.done()
        calling.cancel(marker)
        await asyncio.sleep(0)
        assert not calling.done(), 'observer escaped before restoring the original writer'
    finally:
        error_release.set()
        release.set()
        outcome = await await_owned_task(calling)
        for task in children:
            await await_owned_task(task)
    assert_original_graph(original, cause, outcome.error, marker)
    if kind == 'cancel':
        observers = execution_observer_cancellations(outcome.error)
        assert sum(error.args == original.args for error in observers) == 1, (
            'finite cancellation envelope counted as another actual withdrawal')
        assert sum(error.args == (marker,) for error in observers) == 1
    assert receipt.read_text() == 'physical writer settled'
    assert not lock.locked() and not lock._native_scopes


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['unknown', 'committed'])
async def test_original_journal_error_does_not_absorb_close_observer_withdrawal(owner, kind, tmp_path):
    from kestrel_sovereign.tools.effect_journal import execute_tool_effect
    entered, release, body_failed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original, cause = native_failure(kind)
    marker = 'observer withdrew while original journal closed'
    receipt = tmp_path / 'original-journal-effect'
    children = []

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        async def effect():
            entered.set()
            await release.wait()
            receipt.write_text('original journal effect acknowledged')
            return 'acknowledged'
        children.append(create_owned_task(execute_tool_effect(agent,
            registration='original-closure-effect', arguments={}, execute=effect)))
        await entered.wait()
        body_failed.set()
        raise original

    calling = create_owned_task(invoke(owner, 'one physical effect', invocation_id='original-closure-observer'))
    try:
        await asyncio.wait_for(body_failed.wait(), 1)
        await asyncio.sleep(0)
        calling.cancel(marker)
        await asyncio.sleep(0)
        assert not calling.done(), 'journal closure abandoned its original claimant'
    finally:
        release.set()
        outcome = await await_owned_task(calling)
        for task in children:
            await await_owned_task(task)
    assert_original_graph(original, cause, outcome.error, marker)
    assert receipt.read_text() == 'original journal effect acknowledged'
    assert await owner.tool_effect_store.database.fetchall('SELECT state FROM tool_effect_claims') == [('RETURNED_OK',)]
    assert await owner.tool_effect_store.database.fetchall('SELECT state FROM tool_effect_operations') == [('CLOSED',)]


@pytest.mark.asyncio
@pytest.mark.parametrize('privacy', [False, True])
@pytest.mark.parametrize('failure', ['start-gate', 'accepted-factory', 'completion', 'metadata', 'cohort', 'active'])
async def test_failed_native_codex_callback_notifies_only_its_original_turn(privacy, failure, tmp_path):
    loop, prior = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    original = RuntimeError('original native callback opening failure')
    allocated = []
    receipt = tmp_path / 'unadmitted-codex-effect'
    client = CodexAppServerClient(binary='unused-offline-codex')
    own = client.open_turn_sink('original-thread')
    foreign = client.open_turn_sink('foreign-thread')
    lock = ReentrantTransitionLock()

    class FailedStartTask(asyncio.Task):
        def add_done_callback(self, callback, **kwargs):
            if (failure == 'start-gate' and getattr(callback, '__name__', '') == 'close_unstarted'
                    or failure == 'completion' and any(name in (getattr(getattr(callback, '__code__', None), 'co_names', ())
                            + getattr(getattr(callback, '__code__', None), 'co_freevars', ()))
                        for name in ('_server_request_done', 'report_completion'))):
                raise original
            return super().add_done_callback(callback, **kwargs)
        def __setattr__(self, name, value):
            if failure == 'metadata' and name == '_kestrel_codex_caller_scopes':
                raise original
            return super().__setattr__(name, value)

    def refuse_after_acceptance(loop, coroutine, **kwargs):
        task = (FailedStartTask if failure in ('start-gate', 'completion', 'metadata') else asyncio.Task)(coroutine, loop=loop, **kwargs)
        allocated.append(task)
        if failure == 'accepted-factory':
            raise original
        return task

    async def handler(params):
        receipt.write_text('unadmitted callback dispatched')
        return {'success': True}

    async with AsyncExitStack() as resources:
        if privacy:
            await resources.enter_async_context(lock.policy_span())
        scope = OwnedCallbackScope()
        handler._codex_native_callback_scope = scope
        class RefusedInventory(set):
            def add(self, task):
                super().add(task)
                raise original
        handler._codex_active_tasks = RefusedInventory() if failure == 'active' else set()
        if failure == 'cohort':
            accept = scope.accept
            def refuse_cohort(task):
                accept(task)
                raise original
            scope.accept = refuse_cohort
        unregister = client.register_server_request_handler('item/tool/call', handler, thread_id='original-thread')
        try:
            loop.set_task_factory(refuse_after_acceptance)
            escaped = None
            try:
                client._dispatch({'id': 41, 'method': 'item/tool/call', 'params': {'threadId': 'original-thread'}})
            except BaseException as error:
                escaped = error
            finally:
                loop.set_task_factory(prior)
            if escaped is not None:
                # A broken producer can strand this test's exact accepted gate.
                # Dispose only these owned Tasks before asserting the escape;
                # a pytest timeout is not evidence that the guard caught it.
                for task in tuple(scope.tasks):
                    task.cancel('dispose fixture-owned unadmitted callback')
                    await await_owned_task(task)
            assert escaped is None, 'native callback publication escaped resident dispatch'
            for task in tuple(scope.tasks):
                await await_owned_task(task)
            await asyncio.sleep(0)
            assert not own.empty(), 'failed native callback admission left its original Codex turn unanswered'
            message = own.get_nowait()
            assert '__bridge_error__' in message
            assert any(error is original for error in execution_evidence_carriers(message['__bridge_error__']))
            assert foreign.empty() and client._closed_error is None
            client._dispatch({'method': 'turn/completed', 'params': {'threadId': 'foreign-thread'}})
            assert foreign.get_nowait()['method'] == 'turn/completed'
            assert not receipt.exists() and len(allocated) == 1 and all(task.done() for task in allocated)
            assert not client._server_request_tasks and not handler._codex_active_tasks
        finally:
            loop.set_task_factory(prior)
            scope.close()
            unregister()
            for task in tuple(scope.tasks):
                await await_owned_task(task)


@pytest.mark.asyncio
async def test_sdk_unknown_envelope_keeps_first_message_and_immutable_originals():
    from kestrel_sovereign.execution_custody import execution_observation_error
    first = ResourceOutcomeUnknown('failed task could not be recorded')
    second = ResourceOutcomeUnknown('provider reply lost')
    first_cause, second_cause = ValueError('original task write'), ValueError('original provider')
    first.__cause__, second.__cause__ = first_cause, second_cause
    observed = execution_observation_error(first, second)
    assert isinstance(observed, ResourceOutcomeUnknown), 'finite join erased public SDK outcome type'
    assert str(observed) == str(first)
    assert observed is not first and observed is not second
    assert first.__cause__ is first_cause and second.__cause__ is second_cause
    assert all(any(node is original for node in execution_evidence_carriers(observed))
        for original in (first, second, first_cause, second_cause))


@pytest.mark.asyncio
@pytest.mark.parametrize('interrupted', [False, True])
async def test_no_claim_uncertainty_fences_reopened_operation(owner, interrupted):
    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from kestrel_sovereign.tool_effect_store import ToolEffectReconciliationRequired, ToolEffectStore
    from kestrel_sovereign.tools.effect_journal import (
        bind_tool_effect_turn, record_tool_effect_completion_failure,
    )
    from tests.unit.test_tool_effect_execution import admit
    turn = await admit(owner, key='original-preclaim-uncertainty')
    store, db = owner.tool_effect_store, owner.tool_effect_store.database
    original = ResourceOutcomeUnknown('required audit ACK unavailable before business claim')
    with bind_tool_effect_turn((turn, None)):
        record_tool_effect_completion_failure(owner, original)
    if not interrupted:
        with pytest.raises(ResourceOutcomeUnknown):
            await turn.close()
    else:
        # Model an interrupted process: no terminal write. Release only local
        # privacy resources so the fixture can dispose this in-process probe.
        turn.live = False
        await turn.privacy_lock.close_native_scope(turn.privacy_scope, None)
    assert await db.fetchone('SELECT COUNT(*) FROM tool_effect_claims') == (0,)
    await store.close()
    await db.backend.close()
    await db.backend.connect()
    restarted = ToolEffectStore(db, scope=NativeRecipientScope(owner.did))
    await restarted.initialize()
    owner.tool_effect_store = restarted
    with pytest.raises(ToolEffectReconciliationRequired) as refused:
        await admit(owner, key='original-preclaim-uncertainty')
    assert all(count == 0 for count in refused.value.counts.values()), 'audit uncertainty fabricated a business claim'
    assert await db.fetchone('SELECT COUNT(*) FROM tool_effect_claims') == (0,)
    fresh = await admit(owner, key='independent-new-operation')
    await fresh.close()
