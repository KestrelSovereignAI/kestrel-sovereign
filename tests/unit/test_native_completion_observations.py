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


def sqlite_commit_ack_loss(connection, table=None, *, after, ready=None):
    import aiosqlite
    assert (table is None) != (ready is None), 'select one original commit readiness source'
    native_commit = aiosqlite.Connection.commit
    original, lost = OSError('original prebody SQLite COMMIT acknowledgment lost'), []

    async def lose_ack(target):
        if target is connection and not lost:
            if ready is not None:
                present = ready()
            else:
                async with target.execute('SELECT COUNT(*) FROM ' + table) as cursor:
                    present = (await cursor.fetchone())[0]
            if present:
                lost.append(True)
                if after:
                    await native_commit(target)
                raise original
        await native_commit(target)

    return original, lost, lose_ack


@pytest.mark.asyncio
@pytest.mark.parametrize('phase,loss', [
    ('claim', 'before'), ('claim', 'after'), ('audit', 'before'), ('audit', 'after'),
    ('claim', 'known-refusal'),
])
async def test_original_prebody_commit_failure_fences_live_effects(owner, monkeypatch, tmp_path, phase, loss):
    import aiosqlite
    from kestrel_sovereign.agent.orchestrator_engine import _run_tool_execution
    from kestrel_sovereign.agent.tool_registry import _ToolExecution
    from kestrel_sovereign.execution_custody import current_execution_custody, execution_commit_outcome
    from kestrel_sovereign.tools.effect_journal import current_tool_effect_turn, execute_tool_effect

    database = owner.tool_effect_store.database
    connection = database.backend._connection
    seen = {}
    witness = tmp_path / 'live-effect-after-prebody-commit-loss'
    table = 'tool_effect_claims' if phase == 'claim' else 'a2a_tool_dispatches'

    original, lost, lose_ack = sqlite_commit_ack_loss(connection, table, after=loss == 'after')

    async def first_effect():
        witness.write_text('first unadmitted effect ran')

    async def later_effect():
        witness.write_text('later effect admitted after original claim/audit uncertainty')
        return 'confirmed later effect'

    async def propagate(error, *args):
        raise error

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        assert not current_execution_custody(agent), 'probe requires the supported standalone SQLite host'
        if loss != 'known-refusal':
            monkeypatch.setattr(aiosqlite.Connection, 'commit', lose_ack)
        try:
            if phase == 'claim':
                await execute_tool_effect(agent, registration='original-uncertain-claim',
                    arguments=({'invalid_json': object()} if loss == 'known-refusal' else {}),
                    execute=first_effect)
            else:
                await _run_tool_execution(agent, lambda: write_required_denial_audit(agent),
                    execution=_ToolExecution(), dispatch_meta={}, session_id='original-native-audit',
                    on_failure=propagate, on_terminal=propagate)
        except BaseException as error:
            seen['original_error'] = error
        seen['unconfirmed'] = tuple(current_tool_effect_turn(agent).unconfirmed)
        try:
            seen['later_result'] = await execute_tool_effect(agent,
                registration='later-live-effect', arguments={}, execute=later_effect)
        except BaseException as error:
            seen['later_error'] = error

    try:
        await invoke(owner, 'original prebody commit uncertainty', invocation_id='original-prebody-commit')
    except BaseException as error:
        seen['root_error'] = error
    if loss == 'known-refusal':
        assert not seen['unconfirmed'] and 'root_error' not in seen and 'later_error' not in seen
        assert seen['later_result'] == 'confirmed later effect'
        assert await database.fetchone('SELECT state,latched FROM tool_effect_operations') == ('CLOSED', 0)
        return
    assert not witness.exists(), 'prebody native COMMIT uncertainty admitted another live effect'
    assert lost == [True] and execution_commit_outcome(seen['original_error']) == 'unknown'
    assert any(error is original for error in execution_evidence_carriers(seen['original_error']))
    assert seen['unconfirmed'] and 'later_error' in seen and 'root_error' in seen
    assert any(error is original for error in execution_evidence_carriers(seen['root_error']))
    assert original.__cause__ is None
    assert await database.fetchone('SELECT state,latched FROM tool_effect_operations') == ('CLOSED', 1)
    assert await database.fetchone('SELECT COUNT(*) FROM ' + table) == (int(loss == 'after'),)
    assert await database.fetchone('SELECT COUNT(*) FROM tool_effect_claims') == (int(phase == 'claim' and loss == 'after'),)


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
@pytest.mark.parametrize('failure,privacy', [(failure, privacy)
    for failure in ['start-gate', 'accepted-factory', 'completion', 'metadata', 'cohort', 'active', 'both-subscriptions', 'start-gate-before-step', 'both-subscriptions-before-step']
    for privacy in [False, True] if failure != 'completion' or privacy])
async def test_failed_native_codex_callback_notifies_only_its_original_turn(privacy, failure, tmp_path):
    stop_before_step = failure.endswith('-before-step')
    failure = failure.removesuffix('-before-step')
    loop, prior = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    original = RuntimeError('original native callback opening failure')
    allocated = []
    receipt = tmp_path / 'unadmitted-codex-effect'
    client = CodexAppServerClient(binary='unused-offline-codex')
    own = client.open_turn_sink('original-thread')
    foreign = client.open_turn_sink('foreign-thread')
    lock = ReentrantTransitionLock()
    if failure == 'metadata':
        class RefusedRequestMetadata(dict):
            def __setitem__(self, task, completion):
                super().__setitem__(task, completion)
                raise original
        client._server_request_tasks = RefusedRequestMetadata()

    class FailedStartTask(asyncio.Task):
        def add_done_callback(self, callback, **kwargs):
            if (failure in ('start-gate', 'both-subscriptions') and getattr(callback, '__name__', '') == 'close_unstarted'
                    or failure in ('completion', 'both-subscriptions') and (
                        getattr(callback, '__name__', '') == 'completed'
                        or any(name in (getattr(getattr(callback, '__code__', None), 'co_names', ())
                            + getattr(getattr(callback, '__code__', None), 'co_freevars', ()))
                            for name in ('_server_request_done', 'report_completion')))):
                raise original
            return super().add_done_callback(callback, **kwargs)

    def refuse_after_acceptance(loop, coroutine, **kwargs):
        task = (FailedStartTask if failure in ('start-gate', 'completion', 'metadata', 'both-subscriptions') else asyncio.Task)(coroutine, loop=loop, **kwargs)
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
            if stop_before_step:
                await client._join_server_requests()
                assert all(task.cancelled() for task in allocated)
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


@pytest.mark.asyncio
@pytest.mark.parametrize('borrowed', [False, True])
async def test_original_preclaim_audit_settles_before_operation_seal(owner, monkeypatch, borrowed):
    from kestrel_sovereign.tool_effect_store import ToolEffectReconciliationRequired
    from kestrel_sovereign.tools.effect_journal import (
        bind_tool_effect_turn, record_tool_effect_completion_failure,
    )
    from tests.unit.test_tool_effect_execution import admit

    loop, prior = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    entered, release, sealing = asyncio.Event(), asyncio.Event(), asyncio.Event()
    joined = loop.create_future()
    original_worker = []
    original = ResourceOutcomeUnknown('original required audit delivery unavailable')
    store, db = owner.tool_effect_store, owner.tool_effect_store.database
    finish = store.finish

    class OriginalAuditTask(asyncio.Task):
        def add_done_callback(self, callback, **kwargs):
            if (original_worker and self is original_worker[0]
                    and getattr(callback, '__name__', '') == 'mark_complete' and not joined.done()):
                joined.set_result(None)
            return super().add_done_callback(callback, **kwargs)

    def allocate(loop, coroutine, **kwargs):
        return OriginalAuditTask(coroutine, loop=loop, **kwargs)

    async def record_seal(*args, **kwargs):
        sealing.set()
        return await finish(*args, **kwargs)

    monkeypatch.setattr(store, 'finish', record_seal)
    async with AsyncExitStack() as resources:
        if borrowed:
            await resources.enter_async_context(owner._get_privacy_transition_lock().policy_span())
        turn = await admit(owner, key='original-pending-preclaim-audit')
        assert turn.owns_privacy_scope is not borrowed

        async def audit():
            entered.set()
            await release.wait()
            await write_required_denial_audit(owner)
            raise record_tool_effect_completion_failure(owner, original)

        with bind_tool_effect_turn((turn, None)):
            try:
                loop.set_task_factory(allocate)
                worker = turn.privacy_lock.start_owned_native_work(audit, acquire_span=False)
                original_worker.append(worker)
            finally:
                loop.set_task_factory(prior)
            await entered.wait()
            closing = create_owned_task(turn.close(), retain_originals=True)
        try:
            # Observe the real original Task's native join, or an erroneous
            # early Root return. Neither outcome requires a synthetic collector.
            done, _ = await asyncio.wait((closing, joined), timeout=1,
                return_when=asyncio.FIRST_COMPLETED)
            made_progress = bool(done)
            sealed_while_pending = sealing.is_set()
        finally:
            release.set()
            outcome = await await_owned_task(closing)
            worker_outcome = await await_owned_task(worker)
        assert made_progress, 'original audit closure made no progress'
    assert not sealed_while_pending, 'journal seal preceded its accepted preclaim audit'
    assert any(error is original for error in execution_evidence_carriers(worker_outcome.error))
    assert any(error is original for error in execution_evidence_carriers(outcome.error)), 'Root discarded its late original audit uncertainty'
    assert await db.fetchone('SELECT COUNT(*) FROM a2a_tool_dispatches') == (1,)
    assert await db.fetchone('SELECT COUNT(*) FROM tool_effect_claims') == (0,)
    assert await db.fetchone('SELECT state,latched FROM tool_effect_operations') == ('CLOSED', 1)
    with pytest.raises(ToolEffectReconciliationRequired) as refused:
        await admit(owner, key='original-pending-preclaim-audit')
    assert all(count == 0 for count in refused.value.counts.values())
    assert worker.done() and closing.done()


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['unknown', 'committed'])
async def test_stream_journal_close_keeps_original_native_cause_immutable(owner, kind):
    from tests.unit.test_tool_effect_invocation import governed_stream
    original, cause = native_failure(kind)

    @governed_stream
    async def invoke(agent, user_input, *, request_id=None):
        try:
            yield 'original stream item'
        finally:
            raise original

    stream = invoke(owner, 'original streaming input', request_id='original-stream-native-closer')
    assert await anext(stream) == 'original stream item'
    observed = None
    try:
        await stream.aclose()
    except BaseException as error:
        observed = error
    assert observed is not None
    assert original.__cause__ is cause, 'stream journal teardown rewrote its original native source cause'
    assert any(error is original for error in execution_evidence_carriers(observed))
    assert any(isinstance(error, GeneratorExit) for error in execution_evidence_carriers(observed))
    assert await owner.tool_effect_store.database.fetchone(
        'SELECT state FROM tool_effect_operations') == ('CLOSED',)


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['success', 'ordinary', 'unknown', 'committed', 'group'])
async def test_original_task_completion_publisher_failure_preserves_all_sources(tmp_path, kind):
    original, cause = (None, None) if kind == 'success' else native_failure(kind)
    publication, publication_cause = OSError('original completion publisher failed'), ValueError('original publication cause')
    publication.__cause__ = publication_cause
    receipt = tmp_path / 'physically-settled-original'
    observed = []

    async def body():
        receipt.write_text('one original physical operation')
        if original is not None:
            raise original
        return 'acknowledged original result'

    def publish(task, error):
        observed.append((task, error))
        raise publication

    task = create_owned_task(body(), retain_originals=True, on_completion=publish)
    outcome = await await_owned_task(task)
    assert task.done() and len(observed) == 1 and observed[0][0] is task
    assert receipt.read_text() == 'one original physical operation'
    assert any(error is publication for error in execution_evidence_carriers(outcome.error))
    assert publication.__cause__ is publication_cause
    if original is not None:
        assert any(error is original for error in execution_evidence_carriers(outcome.error))
        assert any(error is original for error in execution_evidence_carriers(observed[0][1]))
        assert original.__cause__ is cause
    else:
        assert observed[0][1] is None


@pytest.mark.asyncio
@pytest.mark.parametrize('declared_async', [False, True])
async def test_native_completion_publisher_must_finish_synchronously(tmp_path, declared_async):
    receipt, observer_receipt = tmp_path / 'body', tmp_path / 'async-observer'
    async def body():
        receipt.write_text('body settled')
    async def asynchronous_publication(task, error):
        observer_receipt.write_text('invalid deferred completion')
    def concealed_publication(task, error):
        return asynchronous_publication(task, error)
    task, refused = None, None
    try:
        task = create_owned_task(body(), on_completion=(asynchronous_publication
            if declared_async else concealed_publication))
        outcome = await await_owned_task(task)
        refused = outcome.error
    except TypeError as error:
        refused = error
    assert isinstance(refused, TypeError), 'native completion silently accepted an awaitable publisher'
    assert not observer_receipt.exists()
    assert receipt.exists() is not declared_async
    if task is not None:
        assert task.done()


@pytest.mark.asyncio
@pytest.mark.parametrize('form', ['async-callable', 'async-generator', 'async-generator-callable'])
async def test_declared_deferred_publisher_is_refused_before_native_allocation(tmp_path, form):
    body_receipt = tmp_path / 'body'
    allocated = []
    loop, prior = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()

    async def body():
        body_receipt.write_text('body ran before declared async publisher refusal')

    async def generator(task, error):
        yield None

    class DeferredCallable:
        async def __call__(self, task, error):
            return None

    class GeneratorCallable:
        async def __call__(self, task, error):
            yield None

    publisher = {'async-callable': DeferredCallable(), 'async-generator': generator,
        'async-generator-callable': GeneratorCallable()}[form]

    def allocate(loop, coroutine, **kwargs):
        task = asyncio.Task(coroutine, loop=loop, **kwargs)
        allocated.append(task)
        return task

    operation, task, refused = body(), None, None
    try:
        loop.set_task_factory(allocate)
        try:
            task = create_owned_task(operation, on_completion=publisher)
        except TypeError as error:
            refused = error
    finally:
        loop.set_task_factory(prior)
        for accepted in allocated:
            await await_owned_task(accepted)
    assert not allocated, 'declared deferred publisher was accepted by the native Task constructor'
    assert isinstance(refused, TypeError) and task is None
    assert operation.cr_frame is None and not body_receipt.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize('borrowed', [False, True])
@pytest.mark.parametrize('kind', ['unknown', 'committed', 'group'])
async def test_completed_original_native_failure_is_harvested_before_journal_seal(owner, borrowed, kind):
    from kestrel_sovereign.tool_effect_store import ToolEffectReconciliationRequired
    from kestrel_sovereign.tools.effect_journal import bind_tool_effect_turn
    from tests.unit.test_tool_effect_execution import admit
    original, cause = native_failure(kind)
    async with AsyncExitStack() as resources:
        if borrowed:
            await resources.enter_async_context(owner._get_privacy_transition_lock().policy_span())
        turn = await admit(owner, key='completed-original-native-failure')

        async def audit():
            await write_required_denial_audit(owner)
            raise original

        with bind_tool_effect_turn((turn, None)):
            worker = turn.privacy_lock.start_owned_native_work(audit, acquire_span=False)
            outcome = await await_owned_task(worker)
        await asyncio.sleep(0)
        assert worker.done() and worker not in turn.privacy_lock._native_work
        assert not turn.pending and not turn.unconfirmed
        assert any(error is original for error in execution_evidence_carriers(outcome.error))
        terminal = None
        try:
            await turn.close()
        except BaseException as error:
            terminal = error
        # A borrowed lifetime must itself retain its genuine native failure;
        # dispose it here before making assertions about the journal boundary.
        if borrowed:
            try:
                await resources.aclose()
            except BaseException as error:
                assert any(node is original for node in execution_evidence_carriers(error))
    assert any(error is original for error in execution_evidence_carriers(terminal)), (
        'journal retirement discarded an already completed original native failure')
    assert original.__cause__ is cause
    db = owner.tool_effect_store.database
    assert await db.fetchone('SELECT COUNT(*) FROM a2a_tool_dispatches') == (1,)
    assert await db.fetchone('SELECT COUNT(*) FROM tool_effect_claims') == (0,)
    assert await db.fetchone('SELECT state,latched FROM tool_effect_operations') == ('CLOSED', 1)
    with pytest.raises(ToolEffectReconciliationRequired):
        await admit(owner, key='completed-original-native-failure')


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['ordinary', 'unknown', 'committed'])
async def test_original_uncertain_root_terminal_fences_claim_free_replay(owner, kind):
    from kestrel_sovereign.tool_effect_store import ToolEffectReconciliationRequired
    from tests.unit.test_tool_effect_execution import admit
    turn = await admit(owner, key='original-root-finalization-uncertainty')
    original, cause = native_failure(kind)
    await write_required_denial_audit(owner)
    assert not turn.unconfirmed
    observed = None
    try:
        await turn.close(original)
    except BaseException as error:
        observed = error
    assert any(error is original for error in execution_evidence_carriers(observed))
    assert original.__cause__ is cause
    assert await owner.tool_effect_store.database.fetchone('SELECT COUNT(*) FROM a2a_tool_dispatches') == (1,)
    assert await owner.tool_effect_store.database.fetchone('SELECT COUNT(*) FROM tool_effect_claims') == (0,)
    with pytest.raises(ToolEffectReconciliationRequired):
        await admit(owner, key='original-root-finalization-uncertainty')


async def write_required_denial_audit(owner):
    from kestrel_sovereign.a2a.stores.unified.observability_store import ToolDispatchEntry
    await owner.observability_store.log_tool_dispatch(ToolDispatchEntry(
        agent_did=owner.did, session_id='original-audit-session', turn_id='original-audit-turn',
        tool_name='refused-native-tool', adapter='native-audit', args_redacted={},
        result_status='policy_denied', error_class='PolicyDenied', error_message=None,
        latency_ms=0, result_size_bytes=0), required=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('native', [False, True])
async def test_known_claim_free_refusal_retains_classification_and_requires_fresh_intent(owner, native):
    from kestrel_sdk.resources import ResourceConflict
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError
    from kestrel_sovereign.tool_effect_store import ToolEffectReconciliationRequired
    from tests.unit.test_tool_effect_execution import admit
    turn = await admit(owner, key='known-safe-claim-free-refusal')
    original = (ExecutionAuthorityError if native else ResourceConflict)('known admission refusal')
    await write_required_denial_audit(owner)
    with pytest.raises(type(original)) as caught:
        await turn.close(original)
    assert caught.value is original
    assert await owner.tool_effect_store.database.fetchone('SELECT state,latched FROM tool_effect_operations') == ('CLOSED', 0)
    assert await owner.tool_effect_store.database.fetchone('SELECT COUNT(*) FROM tool_effect_claims') == (0,)
    with pytest.raises(ToolEffectReconciliationRequired):
        await admit(owner, key='known-safe-claim-free-refusal')
    successor = await admit(owner, key='fresh-explicit-intent-after-known-refusal')
    await successor.close()


@pytest.mark.asyncio
async def test_journal_closure_joins_its_original_subtree_without_waiting_for_foreign_work(owner, tmp_path):
    from kestrel_sovereign.tools.effect_journal import bind_tool_effect_turn
    from tests.unit.test_tool_effect_execution import admit
    lock = owner._get_privacy_transition_lock()
    loop, prior = asyncio.get_running_loop(), asyncio.get_running_loop().get_task_factory()
    foreign_join = loop.create_future()
    release_foreign, foreign_entered = asyncio.Event(), asyncio.Event()
    foreign_owner = []
    witness = tmp_path / 'unrelated-native-operation'

    class ForeignNativeTask(asyncio.Task):
        def add_done_callback(self, callback, **kwargs):
            if (foreign_owner and self is foreign_owner[0]
                    and getattr(callback, '__name__', '') == 'mark_complete' and not foreign_join.done()):
                foreign_join.set_result(None)
            return super().add_done_callback(callback, **kwargs)

    def allocate(loop, coroutine, **kwargs):
        return ForeignNativeTask(coroutine, loop=loop, **kwargs)

    async def unrelated():
        foreign_entered.set()
        await release_foreign.wait()
        witness.write_text('original unrelated native operation settled')

    async def journal():
        turn = await admit(owner, key='original-subtree-journal')
        assert not turn.owns_privacy_scope and turn.native_root is asyncio.current_task()
        assert turn.privacy_scope.root is not turn.native_root
        with bind_tool_effect_turn((turn, None)):
            turn.privacy_lock.start_owned_native_work(
                lambda: write_required_denial_audit(owner), acquire_span=False)
        await turn.close()

    async with lock.policy_span():
        try:
            loop.set_task_factory(allocate)
            foreign = create_owned_task(unrelated())
            foreign_owner.append(foreign)
        finally:
            loop.set_task_factory(prior)
        await foreign_entered.wait()
        own = create_owned_task(journal(), retain_originals=True)
        try:
            await asyncio.wait((own, foreign_join), return_when=asyncio.FIRST_COMPLETED)
            own_settled = own.done()
            joined_foreign = foreign_join.done()
            foreign_still_pending = not foreign.done() and not witness.exists()
        finally:
            release_foreign.set()
            own_outcome = await await_owned_task(own)
            foreign_outcome = await await_owned_task(foreign)
    assert own_settled and not joined_foreign, 'journal closure joined an unrelated native cohort or its parent'
    assert foreign_still_pending
    assert own_outcome.error is foreign_outcome.error is None
    assert await owner.tool_effect_store.database.fetchone('SELECT COUNT(*) FROM a2a_tool_dispatches') == (1,)
    assert await owner.tool_effect_store.database.fetchone('SELECT state,latched FROM tool_effect_operations') == ('CLOSED', 0)
    assert witness.read_text() == 'original unrelated native operation settled'


@pytest.mark.asyncio
async def test_copied_loop_callback_cannot_construct_an_admitting_journal_root(owner):
    from kestrel_sovereign.tools.effect_journal import ToolEffectTurn
    from tests.unit.test_tool_effect_execution import admit
    turn = await admit(owner, key='original-root-task-required')
    loop = asyncio.get_running_loop()
    result = loop.create_future()
    def callback():
        try:
            ToolEffectTurn(owner, owner.tool_effect_store, turn.attempt, 'unadmitted-loop-callback')
        except RuntimeError as error:
            result.set_result(error)
        else:
            result.set_result(None)
    loop.call_soon(callback)
    refused = await result
    await turn.close()
    assert isinstance(refused, RuntimeError)
    assert str(refused) == 'tool effect journal requires its original admitting Task'


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['ordinary', 'cancel', 'unknown', 'committed', 'group'])
async def test_original_owned_handoff_preserves_every_source_graph(tmp_path, kind):
    from kestrel_sovereign._async_ownership import OwnedTaskHandoff
    original, cause = native_failure(kind)
    delivery = OSError('original consumer result delivery failed')
    witness = tmp_path / 'original-physical-result'

    async def producer():
        witness.write_text('original result physically settled')
        raise original

    task = create_owned_task(producer(), retain_originals=True)
    outcome = await await_owned_task(task)
    before = execution_evidence_carriers(original)
    handoff = OwnedTaskHandoff(outcome, operation='original owned result delivery')
    observed = None
    try:
        with handoff.handoff():
            raise delivery
    except BaseException as error:
        observed = error
    assert execution_evidence_carriers(original) == before, 'finite handoff rewrote its original owned source graph'
    assert original.__cause__ is cause
    if kind == 'cancel':
        assert isinstance(observed, asyncio.CancelledError), 'owned handoff changed its original cancellation contract'
    assert all(any(node is error for node in execution_evidence_carriers(observed))
        for error in (original, cause, delivery))
    assert witness.read_text() == 'original result physically settled'


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['native', 'ordinary-consumer'])
async def test_completed_original_native_control_fences_live_effect_before_root_close(owner, monkeypatch, tmp_path, kind):
    import aiosqlite
    import contextvars
    from kestrel_sovereign.tools.effect_journal import current_tool_effect_turn, execute_tool_effect
    database = owner.tool_effect_store.database
    original, lost, lose_ack = sqlite_commit_ack_loss(database.backend._connection,
        'a2a_tool_dispatches', after=True)
    ordinary, cause = native_failure('ordinary')
    witness, seen = tmp_path / 'effect-after-completed-native-control', {}

    async def audit():
        await write_required_denial_audit(owner)
        if kind == 'ordinary-consumer':
            raise ordinary

    async def effect():
        witness.write_text('later effect')
        return 'acknowledged'

    @governed
    async def invoke(agent, user_input, *, invocation_id=None):
        turn = current_tool_effect_turn(agent)
        if kind == 'native':
            monkeypatch.setattr(aiosqlite.Connection, 'commit', lose_ack)
        # Business context can belong to another consumer. Physical native
        # membership is retained by the original factory, independently.
        worker = turn.privacy_lock.start_owned_native_work(audit,
            context=contextvars.Context(), acquire_span=False)
        outcome = await await_owned_task(worker)
        await asyncio.sleep(0)
        assert worker.done() and worker not in turn.privacy_lock._native_work
        assert not turn.unconfirmed and outcome.error is not None
        seen['source'] = outcome.error
        try:
            seen['result'] = await execute_tool_effect(agent, registration='after-completed-native',
                arguments={}, execute=effect)
        except BaseException as error:
            seen['refused'] = error

    try:
        await invoke(owner, 'original completed native control', invocation_id='completed-native-live-fence')
    except BaseException as error:
        seen['root_error'] = error
    if kind == 'ordinary-consumer':
        assert witness.read_text() == 'later effect' and seen['result'] == 'acknowledged'
        assert 'refused' not in seen and 'root_error' not in seen
        assert ordinary.__cause__ is cause
        assert await database.fetchone('SELECT state,latched FROM tool_effect_operations') == ('CLOSED', 0)
    else:
        assert not witness.exists(), 'completed original native control admitted another live effect'
        assert lost == [True] and 'refused' in seen and 'root_error' in seen
        assert any(error is original for error in execution_evidence_carriers(seen['refused']))
        assert await database.fetchone('SELECT COUNT(*) FROM tool_effect_claims') == (0,)
        assert await database.fetchone('SELECT state,latched FROM tool_effect_operations') == ('CLOSED', 1)
    assert await database.fetchone('SELECT COUNT(*) FROM a2a_tool_dispatches') == (1,)
