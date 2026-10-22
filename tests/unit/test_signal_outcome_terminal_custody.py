"""Original fixed native audit outlives work admission, never grants work."""
import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCustody, bind_execution_runtime,
)
from kestrel_sovereign.signals.outcome_custody import _OriginalSignalOutcome
from tests.unit.test_signals_dispatcher import db_path, dispatcher_components, _action_reg, _signal  # noqa: F401


@dataclass
class _OriginalWork:
    backend_type: str = 'sqlite'

    def require_work(self):
        return None

    async def lock_and_validate(self, connection):
        self.require_work()


@pytest.mark.parametrize('cutoff', ['handler', 'journal'])
async def test_original_native_outcome_persists_after_work_cutoff_without_fresh_work(
    dispatcher_components, monkeypatch, cutoff,
):
    c = dispatcher_components
    await c.backend.execute('CREATE TABLE ordinary_effects(id INTEGER PRIMARY KEY)')
    entered, release = asyncio.Event(), asyncio.Event()
    scope = ExecutionCustody(_OriginalWork(backend_type=c.backend.backend_type))
    owner = SimpleNamespace(_execution_custody=scope)
    handle = None

    async def handler(payload):
        await c.backend.execute('INSERT INTO ordinary_effects VALUES(1)')
        if cutoff == 'handler':
            entered.set()
            await release.wait()
        return {'original': 'completed'}

    original_writer = c.dispatcher._write_outcome_log

    async def held_original_writer(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original_writer(*args, **kwargs)

    if cutoff == 'journal':
        monkeypatch.setattr(c.dispatcher, '_write_outcome_log', held_original_writer)
    c.registry.register(_action_reg('original.audit', handler=handler))
    signal = _signal('original.audit')
    try:
        with bind_execution_runtime(owner):
            handle = await c.dispatcher.enqueue_signal(signal)
            await asyncio.wait_for(entered.wait(), 2)
            scope.revoke('original work cutoff')
            with pytest.raises(ExecutionAuthorityError):
                await c.backend.execute('INSERT INTO ordinary_effects VALUES(2)')
            release.set()
            outcome = await await_owned_task(handle.task)
            assert outcome.error is None
            assert outcome.result.signal_id == signal.id
        assert await c.backend.fetch_val('SELECT COUNT(*) FROM ordinary_effects') == 1
        row = await c.backend.fetch_one('SELECT id,status,source FROM signal_log WHERE id=?', (signal.id,))
        assert tuple(row) == (signal.id, 'ok', 'original.audit')
    finally:
        release.set()
        if handle is not None:
            await await_owned_task(handle.task)


async def test_unadmitted_or_forged_native_journal_never_grants_storage(dispatcher_components):
    c = dispatcher_components
    registration = _action_reg('unadmitted.audit')
    c.registry.register(registration)
    with pytest.raises(ExecutionAuthorityError, match='accepted dispatch'):
        _OriginalSignalOutcome(c.dispatcher, _signal('unadmitted.audit'), registration)
    for forged in (object(), object.__new__(_OriginalSignalOutcome)):
        with pytest.raises(ExecutionAuthorityError):
            await c.backend._append_original_signal_outcome(forged)
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log') == 0


async def test_handler_cannot_use_journal_permission_or_replay_after_success(dispatcher_components):
    c = dispatcher_components
    observed = []

    async def handler(payload):
        journal = c.dispatcher._original_outcome_admission.get()
        assert type(journal) is _OriginalSignalOutcome
        with pytest.raises(ExecutionAuthorityError):
            await c.backend._append_original_signal_outcome(journal)
        observed.append(journal)
        return {'ok': True}

    c.registry.register(_action_reg('bounded.audit', handler=handler))
    signal = _signal('bounded.audit')
    result = await c.dispatcher.dispatch_signal(signal)
    assert result.signal_id == signal.id
    with pytest.raises(ExecutionAuthorityError):
        await c.backend._append_original_signal_outcome(observed[0])
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log') == 1


async def test_handler_cannot_mint_unrelated_journal_and_self_attach(dispatcher_components):
    from kestrel_sdk.signals import SignalResult, Status
    c = dispatcher_components
    rejected = []

    async def handler(payload):
        other = _signal('mint.audit')
        try:
            journal = _OriginalSignalOutcome(c.dispatcher, other, registration)
        except ExecutionAuthorityError:
            rejected.append(True)
        else:
            result = SignalResult(signal_id=other.id, status=Status.OK, mode=other.mode)
            journal.bind(other, registration, result, c.store._prepare_outcome(other, registration, result))
            journal.attach_outer(asyncio.current_task())
            journal.attach_writer(asyncio.current_task())
            await c.backend._append_original_signal_outcome(journal)
        return {'ok': True}

    registration = _action_reg('mint.audit', handler=handler)
    c.registry.register(registration)
    await c.dispatcher.dispatch_signal(_signal('mint.audit'))
    assert rejected == [True]
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log') == 1


async def test_original_journal_destination_cannot_follow_mutable_table(dispatcher_components, monkeypatch):
    c = dispatcher_components
    await c.backend.execute('CREATE TABLE redirected_log AS SELECT * FROM signal_log WHERE 1=0')
    original = c.dispatcher._write_outcome_log

    async def redirect(*args, **kwargs):
        monkeypatch.setattr(c.store, 'TABLE', 'redirected_log')
        return await original(*args, **kwargs)

    monkeypatch.setattr(c.dispatcher, '_write_outcome_log', redirect)
    c.registry.register(_action_reg('destination.audit'))
    signal = _signal('destination.audit')
    await c.dispatcher.dispatch_signal(signal)
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM redirected_log') == 0
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log WHERE id=?', (signal.id,)) == 1


@pytest.mark.parametrize('callback', ['redaction', 'result'])
async def test_projection_control_failure_is_not_a_placeholder(dispatcher_components, callback):
    from dataclasses import replace
    from kestrel_sdk.signals import RedactionPolicy
    c = dispatcher_components
    original = ExecutionAuthorityError('original callback denial')

    def refuse(value):
        raise original

    registration = _action_reg('projection.control')
    registration = replace(registration, **(
        {'log_redaction': RedactionPolicy(summarize=refuse)} if callback == 'redaction'
        else {'result_summary': refuse}))
    c.registry.register(registration)
    with pytest.raises(ExecutionAuthorityError) as caught:
        await c.dispatcher.dispatch_signal(_signal('projection.control'))
    assert caught.value is original
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log') == 0


async def test_ordinary_projection_loss_does_not_relabel_completed_effect(dispatcher_components):
    from kestrel_sdk.signals import Status
    c = dispatcher_components
    effects = []

    async def handler(payload):
        effects.append('completed')
        circular = {}
        circular['self'] = circular
        return circular

    c.registry.register(_action_reg('projection.loss', handler=handler))
    result = await c.dispatcher.dispatch_signal(_signal('projection.loss'))
    assert effects == ['completed']
    assert result.status == Status.OK
    assert c.dispatcher._log_write_failures == 1
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log') == 0


@pytest.mark.parametrize('registered_before_refusal', [False, True])
async def test_rejected_completion_callback_physically_joins_original_writer(dispatcher_components, registered_before_refusal):
    c = dispatcher_components
    loop = asyncio.get_running_loop()
    before = loop.get_task_factory()
    rejected = RuntimeError('original outcome completion publication refused')
    accepted = []

    class RejectCompletion(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            if getattr(callback, '__name__', '') == '_complete':
                if registered_before_refusal:
                    super().add_done_callback(callback, context=context)
                raise rejected
            return super().add_done_callback(callback, context=context)

    def factory(loop, coro, **kwargs):
        task = RejectCompletion(coro, loop=loop, **kwargs)
        accepted.append(task)
        return task

    c.registry.register(_action_reg('publication.audit'))
    loop.set_task_factory(factory)
    try:
        result = await asyncio.wait_for(c.dispatcher.dispatch_signal(_signal('publication.audit')), 2)
    finally:
        loop.set_task_factory(before)
    assert accepted and all(task.done() for task in accepted)
    assert not c.dispatcher._outcome_log_tasks
    assert not c.dispatcher._outcome_log_tasks_by_signal
    assert c.dispatcher._durable_active_admissions == 0
    assert str(rejected) in result.error


async def test_recovered_claim_has_its_own_terminal_journal(tmp_path, monkeypatch):
    import time
    from kestrel_sovereign.signals import DurableConsumerRegistration
    from tests.unit.test_durable_signal_delivery import _channel_dispatcher, _channel_signal, _close
    from kestrel_sovereign.signals.sources.channels import (
        DURABLE_COGNITION_CONSUMER_ID, DURABLE_COGNITION_MARKER, DURABLE_COGNITION_MARKER_VALUE,
    )
    from kestrel_sovereign.execution_custody import is_execution_control_error
    backend, agent, dispatcher = await _channel_dispatcher(tmp_path / 'recovered-native-audit.db', 'did:agent:journal-recovery')
    consumer = DurableConsumerRegistration(consumer_id=DURABLE_COGNITION_CONSUMER_ID,
        source='channel.message', agent_id=agent.did,
        correlation_selector=f'payload.{DURABLE_COGNITION_MARKER}={DURABLE_COGNITION_MARKER_VALUE}', max_attempts=0)
    entered, release = asyncio.Event(), asyncio.Event()
    original = dispatcher._write_outcome_log
    task = None
    scope = ExecutionCustody(_OriginalWork())

    async def held_writer(*args, **kwargs):
        assert dispatcher._original_outcome_writer.get() is not None, (
            dispatcher._original_outcome_admission.get(), dispatcher._accepted_signal_outcomes)
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(dispatcher, '_write_outcome_log', held_writer)
    try:
        await dispatcher.register_durable_consumer(consumer)
        signal = _channel_signal(agent.did, 'original-recovered-audit')
        persisted = await dispatcher._durable_store.persist_signal(signal, agent_id=agent.did,
            source_event_id='recovered-original', retention_days=7,
            caller_identity=dispatcher._protect_durable_caller_identity(signal))
        delivery = await dispatcher.claim_durable_delivery_for_event(consumer_id=consumer.consumer_id,
            event_id=persisted.event_id, executor_id=dispatcher._durable_delivery_owner)
        assert delivery is not None
        seed = dispatcher._durable_retry_seed(delivery.event)
        with bind_execution_runtime(SimpleNamespace(_execution_custody=scope)):
            task = asyncio.create_task(dispatcher._route_durable_cognition_delivery(
                seed, dispatcher._registry.get(seed.source), time.monotonic(),
                persisted_event_id=delivery.event_id, consumer_id=consumer.consumer_id,
                durable_admission=None, durable_created=False, use_live_signal=False,
                claimed_delivery=delivery))
            await asyncio.wait_for(entered.wait(), 5)
            scope.revoke('recovered work cutoff after original delivery settlement')
            release.set()
            outcome = await await_owned_task(task)
            assert outcome.error is None
            await dispatcher._drain_outcome_logs_for_signal(seed.id)
        assert await backend.fetch_val('SELECT COUNT(*) FROM signal_log WHERE id=?', (seed.id,)) == 1
    finally:
        release.set()
        if task is not None:
            await await_owned_task(task)
        try:
            await dispatcher.shutdown_durable_delivery()
        except BaseException as error:
            assert is_execution_control_error(error)
        await _close(backend, agent)


@pytest.mark.parametrize('change', ['signal', 'agent', 'backend'])
async def test_original_journal_refuses_substituted_identity(dispatcher_components, monkeypatch, tmp_path, change):
    from kestrel_sovereign.signals.store import SignalLogStore
    from kestrel_sovereign.storage.db import SQLiteBackend
    from kestrel_sovereign.execution_custody import is_execution_control_error

    c = dispatcher_components
    other = None
    if change == 'backend':
        other = SQLiteBackend(str(tmp_path / 'replacement.db'))
        await other.connect()
        await SignalLogStore(other).initialize()
    initial_agent_id = c.agent._did
    original_writer = c.dispatcher._write_outcome_log
    original_schedule = c.dispatcher._schedule_outcome_log
    c.registry.register(_action_reg('fixed.audit'))

    async def changed_writer(*args, **kwargs):
        if change == 'agent':
            c.agent._did = 'different-agent'
        elif change == 'backend':
            c.store._backend = other
        return await original_writer(*args, **kwargs)

    def changed_signal(signal, registration, result, **kwargs):
        result.signal_id = 'substituted-result'
        return original_schedule(signal, registration, result, **kwargs)

    monkeypatch.setattr(c.dispatcher, '_write_outcome_log', changed_writer)
    if change == 'signal':
        monkeypatch.setattr(c.dispatcher, '_schedule_outcome_log', changed_signal)
    try:
        with pytest.raises(BaseException) as refused:
            await c.dispatcher.dispatch_signal(_signal('fixed.audit'))
        assert is_execution_control_error(refused.value)
    finally:
        c.agent._did = initial_agent_id
        c.store._backend = c.backend
        # Join exactly the original failed terminal owner; no new retry of its
        # already retained failure is interpreted as a successful write.
        try:
            await c.dispatcher.shutdown_durable_delivery()
        except BaseException as error:
            assert is_execution_control_error(error)
        if other is not None:
            assert await other.fetch_val('SELECT COUNT(*) FROM signal_log') == 0
            await other.close()
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log') == 0
