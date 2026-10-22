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
from tests.unit.test_signals_dispatcher import (
    db_path as _db_path_fixture, dispatcher_components as _dispatcher_fixture, _action_reg, _signal,
)

db_path = _db_path_fixture
dispatcher_components = _dispatcher_fixture


@dataclass
class _OriginalWork:
    backend_type: str = 'sqlite'

    def require_work(self):
        return None

    async def lock_and_validate(self, connection):
        self.require_work()


@pytest.mark.parametrize('shape', ['expression_index', 'predicate_index', 'predicate_comment', 'check', 'check_comment'])
async def test_sqlite_journal_rejects_executable_schema_before_routing(dispatcher_components, shape):
    from kestrel_sovereign.execution_custody import is_execution_control_error
    c = dispatcher_components
    await c.backend.execute('CREATE TABLE forbidden_schema_effect(id TEXT)')
    armed, effects = [], []
    def program(value):
        if armed:
            c.backend._connection._conn.execute('INSERT INTO forbidden_schema_effect VALUES(?)', (value,))
        return value
    await c.backend._connection.create_function('journal_program', 1, program, deterministic=True)
    if shape in ('check', 'check_comment'):
        sql = await c.backend.fetch_val("SELECT sql FROM sqlite_schema WHERE name='signal_log'")
        await c.backend.execute('ALTER TABLE signal_log RENAME TO original_audit')
        gap = ' /* native SQL comment */ ' if shape == 'check_comment' else ''
        await c.backend.execute(sql.rstrip()[:-1] + ', CHECK' + gap + '(journal_program(id)=id))')
    elif shape == 'expression_index':
        await c.backend.execute('CREATE INDEX journal_program_index ON signal_log(journal_program(id))')
    elif shape == 'predicate_comment':
        await c.backend.execute('CREATE INDEX journal_program_index ON signal_log(id) WHERE journal_program(id) OR -- WHERE\n constitution_hash IS NOT NULL')
    else:
        await c.backend.execute('CREATE INDEX journal_program_index ON signal_log(id) WHERE journal_program(id)=id')
    scope = ExecutionCustody(_OriginalWork())
    async def handler(payload):
        effects.append('handler completed')
        armed.append(True)
        scope.revoke('actual cutoff before executable schema')
        return {'genuine': True}
    c.registry.register(_action_reg('schema.program', handler=handler))
    error = None
    with bind_execution_runtime(SimpleNamespace(_execution_custody=scope)):
        try:
            await c.dispatcher.dispatch_signal(_signal('schema.program'))
        except BaseException as caught:
            error = caught
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM forbidden_schema_effect') == 0, 'privileged journal executed application schema SQL'
    assert not effects and is_execution_control_error(error)


async def test_original_sqlite_refuses_noop_native_autocommit(dispatcher_components):
    import sqlite3
    from kestrel_sovereign.execution_custody import is_execution_control_error
    c = dispatcher_components
    native = c.backend._connection._conn
    if not hasattr(native, 'autocommit'):
        pytest.skip('Native autocommit property was added in Python 3.12')
    scope = ExecutionCustody(_OriginalWork())
    async def handler(payload):
        await c.backend._connection._execute(setattr, native, 'autocommit', True)
        scope.revoke('actual cutoff after configuring no-op commit mode')
        return {'genuine': True}
    c.registry.register(_action_reg('native.noop.commit', handler=handler))
    error = None
    try:
        with bind_execution_runtime(SimpleNamespace(_execution_custody=scope)):
            try:
                await c.dispatcher.dispatch_signal(_signal('native.noop.commit'))
            except BaseException as caught:
                error = caught
        assert is_execution_control_error(error), 'native no-op COMMIT was acknowledged as durable'
        assert not native.in_transaction, 'original journal returned with an unresolved native transaction'
    finally:
        def restore_native_mode():
            if native.in_transaction:
                native.execute('ROLLBACK').close()
            native.autocommit = sqlite3.LEGACY_TRANSACTION_CONTROL
        await c.backend._connection._execute(restore_native_mode)


@pytest.mark.parametrize('installed', ['before', 'handler'])
async def test_sqlite_journal_refuses_trace_program_through_commit(dispatcher_components, installed):
    from kestrel_sovereign.execution_custody import is_execution_control_error
    c = dispatcher_components
    armed, deleted, effects = [], [], []
    signal = _signal('native.trace.commit')
    def trace(query):
        if armed and not deleted and query.strip().upper() == 'COMMIT':
            deleted.append(True)
            c.backend._connection._conn.execute('DELETE FROM signal_log WHERE id=?', (signal.id,))
    if installed == 'before':
        await c.backend._connection.set_trace_callback(trace)
    scope = ExecutionCustody(_OriginalWork())
    async def handler(payload):
        effects.append(True)
        if installed == 'handler':
            await c.backend._connection.set_trace_callback(trace)
        armed.append(True)
        scope.revoke('actual cutoff before trace COMMIT program')
        return {'genuine': True}
    c.registry.register(_action_reg('native.trace.commit', handler=handler))
    error = None
    try:
        with bind_execution_runtime(SimpleNamespace(_execution_custody=scope)):
            try:
                await c.dispatcher.dispatch_signal(signal)
            except BaseException as caught:
                error = caught
    finally:
        armed.clear()
        await c.backend._connection.set_trace_callback(None)
    assert not deleted, 'application trace deleted the row after native verification'
    assert is_execution_control_error(error)
    assert len(effects) == (installed == 'handler')
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log') == 0


async def test_original_sqlite_program_fence_is_live_at_actual_commit(dispatcher_components, monkeypatch):
    from kestrel_sovereign._async_ownership import create_owned_task, await_owned_task
    from kestrel_sovereign.storage.db.native_journal_programs import NativeJournalConfigurationError
    c = dispatcher_components
    entered, release = asyncio.Event(), asyncio.Event()
    commit = c.backend._commit_write
    async def held_commit(connection, **kwargs):
        if kwargs.get('original_outcome') is not None:
            entered.set()
            await release.wait()
        return await commit(connection, **kwargs)
    monkeypatch.setattr(c.backend, '_commit_write', held_commit)
    scope = ExecutionCustody(_OriginalWork())
    async def handler(payload):
        scope.revoke('actual cutoff before held native COMMIT')
        return {'genuine': True}
    c.registry.register(_action_reg('actual.commit.program.fence', handler=handler))
    traced = []
    with bind_execution_runtime(SimpleNamespace(_execution_custody=scope)):
        task = create_owned_task(c.dispatcher.dispatch_signal(_signal('actual.commit.program.fence')))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        with pytest.raises(NativeJournalConfigurationError):
            await c.backend._connection.set_trace_callback(traced.append)
    finally:
        release.set()
        outcome = await await_owned_task(task)
    assert outcome.error is None and outcome.result.action_result == {'genuine': True}
    assert not traced and await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log') == 1
    assert c.backend._connection._conn._original_journal_programs_supported()


async def test_sqlite_journal_refuses_noncanonical_column_affinity(dispatcher_components):
    from kestrel_sovereign.execution_custody import is_execution_control_error
    c = dispatcher_components
    sql = await c.backend.fetch_val("SELECT sql FROM sqlite_schema WHERE name='signal_log'")
    assert 'result_summary TEXT' in sql
    await c.backend.execute('ALTER TABLE signal_log RENAME TO original_audit')
    await c.backend.execute(sql.replace('result_summary TEXT', 'result_summary INTEGER'))
    effects = []
    async def handler(payload):
        effects.append(True)
        return {'genuine': True}
    registration = _action_reg('noncanonical.affinity', handler=handler)
    registration.result_summary = lambda value: '001'
    c.registry.register(registration)
    error = None
    try:
        await c.dispatcher.dispatch_signal(_signal('noncanonical.affinity'))
    except BaseException as caught:
        error = caught
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log') == 0, 'affinity coerced the prepared text while SQL equality still certified it'
    assert not effects and is_execution_control_error(error)


async def test_sqlite_journal_never_invokes_application_text_factory(dispatcher_components):
    from kestrel_sovereign.execution_custody import is_execution_control_error
    c = dispatcher_components
    calls = []
    await c.backend.execute('CREATE TABLE forbidden_text_factory_effect(id TEXT)')
    def factory(raw):
        calls.append(True)
        c.backend._connection._conn.execute("INSERT INTO forbidden_text_factory_effect VALUES('forbidden')")
        return raw.decode()
    scope = ExecutionCustody(_OriginalWork())
    async def handler(payload):
        c.backend._connection.text_factory = factory
        scope.revoke('actual cutoff before metadata text conversion')
        return {'genuine': True}
    c.registry.register(_action_reg('native.text.factory', handler=handler))
    error = None
    try:
        with bind_execution_runtime(SimpleNamespace(_execution_custody=scope)):
            try:
                await c.dispatcher.dispatch_signal(_signal('native.text.factory'))
            except BaseException as caught:
                error = caught
    finally:
        c.backend._connection.text_factory = str
    assert not calls, 'privileged journal metadata executed an application text factory'
    assert is_execution_control_error(error)
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM forbidden_text_factory_effect') == 0
    assert c.dispatcher._log_write_failures == 1


@pytest.mark.parametrize('projection', ['summary', 'redaction'])
async def test_sqlite_journal_binds_exact_native_projection_text(dispatcher_components, projection):
    from dataclasses import replace
    c = dispatcher_components
    adapted = []
    class AdaptedText(str):
        def __conform__(self, protocol):
            adapted.append(True)
            return 'substituted-through-application-adapter'
    original = AdaptedText('genuine-projection')
    registration = _action_reg('projection.adaptation')
    if projection == 'summary':
        registration.result_summary = lambda result: original
    else:
        registration.log_redaction = replace(registration.log_redaction, summarize=lambda payload: original)
    c.registry.register(registration)
    signal = _signal('projection.adaptation')
    await c.dispatcher.dispatch_signal(signal)
    column = 'result_summary' if projection == 'summary' else 'payload_redacted'
    stored = await c.backend.fetch_val('SELECT '+column+' FROM signal_log WHERE id=?', (signal.id,))
    assert stored == 'genuine-projection', 'native binding substituted canonical prepared text'
    assert not adapted


async def test_sqlite_journal_refuses_registered_native_scalar_adapter(dispatcher_components):
    import sqlite3
    from kestrel_sovereign.execution_custody import is_execution_control_error
    c = dispatcher_components
    adapted = []
    key = (str, sqlite3.PrepareProtocol)
    missing = object()
    before = sqlite3.adapters.get(key, missing)
    def adapter(value):
        adapted.append(True)
        return 'substituted-global-adapter'
    scope = ExecutionCustody(_OriginalWork())
    async def handler(payload):
        sqlite3.register_adapter(str, adapter)
        scope.revoke('actual cutoff after application adapter registration')
        return {'genuine': True}
    c.registry.register(_action_reg('global.scalar.adapter', handler=handler))
    error = None
    try:
        with bind_execution_runtime(SimpleNamespace(_execution_custody=scope)):
            try:
                await c.dispatcher.dispatch_signal(_signal('global.scalar.adapter'))
            except BaseException as caught:
                error = caught
    finally:
        if before is missing:
            sqlite3.adapters.pop(key, None)
        else:
            sqlite3.adapters[key] = before
    assert not adapted, 'privileged SQL invoked an application scalar adapter'
    assert is_execution_control_error(error)
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log') == 0
    assert c.dispatcher._log_write_failures == 1


async def test_emitter_cannot_change_original_writer_tracking_identity(dispatcher_components, monkeypatch):
    from dataclasses import replace
    from kestrel_sdk.signals import Visibility
    from kestrel_sovereign._async_ownership import create_owned_task, await_owned_task
    c = dispatcher_components
    signal = replace(_signal('mutable.emit.identity'), visibility=Visibility.USER_VISIBLE)
    original_id = signal.id
    emitted = asyncio.Event()
    release_drain = asyncio.Event()
    original_drain = c.dispatcher._drain_outcome_logs_for_signal
    async def held_drain(signal_id):
        await release_drain.wait()
        await original_drain(signal_id)
    monkeypatch.setattr(c.dispatcher, '_drain_outcome_logs_for_signal', held_drain)
    async def emit(*args):
        signal.id = 'mutated-after-actual-commit'
        emitted.set()
    c.agent.emit_event = emit
    c.registry.register(_action_reg('mutable.emit.identity'))
    task = create_owned_task(c.dispatcher.dispatch_signal(signal))
    try:
        await asyncio.wait_for(emitted.wait(), 2)
        writers = tuple(c.dispatcher._outcome_log_tasks_by_signal.get(original_id, ()))
        for writer in writers:
            await await_owned_task(writer)
            assert writer.done(), 'test must inspect a physically joined writer'
        assert not c.dispatcher._outcome_log_tasks_by_signal.get(original_id), 'completed original writer remains indexed under its initial ID'
        release_drain.set()
        result = await asyncio.wait_for(asyncio.shield(task), 2)
        assert result.signal_id == original_id
    finally:
        if not task.done():
            task.cancel()
        release_drain.set()
        await await_owned_task(task)
    assert task.done() and not c.dispatcher._outcome_log_tasks
    assert not c.dispatcher._outcome_log_reconciliations
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log WHERE id=?', (original_id,)) == 1


@pytest.mark.parametrize('temporary', [False, True])
async def test_sqlite_original_journal_never_executes_unrelated_trigger_effect(dispatcher_components, temporary):
    from kestrel_sovereign.execution_custody import is_execution_control_error, execution_commit_outcome, execution_native_source_error
    c = dispatcher_components
    await c.backend.execute('CREATE TABLE journal_forbidden_effect(id TEXT PRIMARY KEY)')
    scope = ExecutionCustody(_OriginalWork())
    async def handler(payload):
        await c.backend.execute('CREATE ' + ('TEMP ' if temporary else '') + 'TRIGGER journal_extra AFTER INSERT ON main.signal_log BEGIN INSERT INTO journal_forbidden_effect VALUES(NEW.id); END')
        scope.revoke('actual cutoff before privileged audit trigger')
        return {'genuine': True}
    c.registry.register(_action_reg('extra.trigger.effect', handler=handler))
    signal = _signal('extra.trigger.effect')
    error = None
    with bind_execution_runtime(SimpleNamespace(_execution_custody=scope)):
        try:
            await c.dispatcher.dispatch_signal(signal)
        except BaseException as caught:
            error = caught
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM journal_forbidden_effect') == 0, 'revoked work committed unrelated trigger SQL'
    assert is_execution_control_error(error)
    assert execution_commit_outcome(error) != 'committed'
    assert c.dispatcher._log_write_failures == 1
    assert execution_native_source_error(error) is None


async def test_sqlite_original_journal_refuses_replaced_physical_destination(dispatcher_components):
    from kestrel_sovereign.execution_custody import is_execution_control_error, execution_commit_outcome
    c = dispatcher_components
    scope = ExecutionCustody(_OriginalWork())
    async def handler(payload):
        await c.backend.execute('ALTER TABLE main.signal_log RENAME TO accepted_signal_log')
        await c.backend.execute('CREATE TABLE main.signal_log AS SELECT * FROM main.accepted_signal_log WHERE 0')
        scope.revoke('actual original table replaced')
        return {'genuine': True}
    c.registry.register(_action_reg('replacement.sqlite.audit', handler=handler))
    signal = _signal('replacement.sqlite.audit')
    error = None
    with bind_execution_runtime(SimpleNamespace(_execution_custody=scope)):
        try:
            await c.dispatcher.dispatch_signal(signal)
        except BaseException as caught:
            error = caught
    assert is_execution_control_error(error), 'replacement table inherited original audit permission'
    assert execution_commit_outcome(error) != 'committed'
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM main.signal_log') == 0
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM main.accepted_signal_log') == 0
    assert c.dispatcher._log_write_failures == 1


@pytest.mark.parametrize('cutoff', [False, True])
async def test_projection_retained_alias_cannot_rewrite_actual_result(dispatcher_components, cutoff):
    from kestrel_sovereign.signals.store import _digest
    c = dispatcher_components
    body = {'nested': {'value': 'actual-return'}}
    scope = ExecutionCustody(_OriginalWork(backend_type=c.backend.backend_type))
    async def handler(payload):
        if cutoff:
            scope.revoke('actual cutoff before retained alias mutation')
        return body
    def summary(_copied_argument):
        body['nested']['value'] = 'rewritten-through-retained-alias'
        return '<projection>'
    registration = _action_reg('retained.result.alias', handler=handler)
    registration.result_summary = summary
    c.registry.register(registration)
    signal = _signal('retained.result.alias')
    with bind_execution_runtime(SimpleNamespace(_execution_custody=scope)):
        result = await c.dispatcher.dispatch_signal(signal)
    assert body['nested']['value'] == 'rewritten-through-retained-alias', 'callback did not exercise its original retained alias'
    assert result.action_result == {'nested': {'value': 'actual-return'}}
    assert await c.backend.fetch_val('SELECT action_result_digest FROM signal_log WHERE id=?', (signal.id,)) == _digest({'nested': {'value': 'actual-return'}})
    assert c.dispatcher._log_write_failures == 0


async def test_result_snapshot_never_delegates_to_mutable_copy_hook(dispatcher_components):
    from kestrel_sovereign.signals.store import _digest
    c = dispatcher_components
    copied = []
    class RetainedDictionary(dict):
        def __deepcopy__(self, memo):
            copied.append(True)
            return self
    body = RetainedDictionary(nested={'value': 'actual-return'})
    async def handler(payload):
        return body
    def summary(_argument):
        body['nested']['value'] = 'mutated-original-copy-hook-alias'
        return '<projection>'
    registration = _action_reg('copy.hook.alias', handler=handler)
    registration.result_summary = summary
    c.registry.register(registration)
    signal = _signal('copy.hook.alias')
    result = await c.dispatcher.dispatch_signal(signal)
    assert not copied, 'canonical snapshot invoked an application copy hook'
    assert result.action_result == {'nested': {'value': 'actual-return'}}
    assert await c.backend.fetch_val('SELECT action_result_digest FROM signal_log WHERE id=?', (signal.id,)) == _digest({'nested': {'value': 'actual-return'}})


async def test_accepted_payload_snapshot_never_delegates_to_copy_hook(dispatcher_components):
    from dataclasses import replace
    from kestrel_sdk.signals import Trust
    c = dispatcher_components
    copied = []
    class RetainedDictionary(dict):
        def __deepcopy__(self, memo):
            copied.append(True)
            return self
    body = RetainedDictionary(nested={'value': 'accepted-payload'})
    async def handler(payload):
        body['nested']['value'] = 'handler-mutated-original'
        return {'genuine': True}
    registration = replace(_action_reg('payload.copy.hook', handler=handler), trust=Trust.TRUSTED)
    registration.log_redaction = replace(registration.log_redaction, store_raw_trusted=True)
    c.registry.register(registration)
    signal = _signal('payload.copy.hook', payload=body)
    result = await c.dispatcher.dispatch_signal(signal)
    assert not copied
    assert result.action_result == {'genuine': True}
    import json
    stored = await c.backend.fetch_val('SELECT payload_raw FROM signal_log WHERE id=?', (signal.id,))
    assert json.loads(stored) == {'nested': {'value': 'accepted-payload'}}


def test_data_snapshot_uses_builtin_storage_and_preserves_cycles():
    from kestrel_sovereign.signals.outcome_custody import _snapshot_outcome_data
    class Container(dict):
        def items(self):
            raise AssertionError('application items override ran')
        def __deepcopy__(self, memo):
            raise AssertionError('application copy hook ran')
    class Items(list):
        def __iter__(self):
            raise AssertionError('application iterator ran')
    original = Container(items=Items([1, {'value': 'accepted'}]))
    original['self'] = original
    copied = _snapshot_outcome_data(original)
    assert type(copied) is dict and type(copied['items']) is list
    assert copied is not original and copied['self'] is copied
    original['items'][1]['value'] = 'changed'
    assert copied['items'][1] == {'value': 'accepted'}
    cycle = []
    original_tuple = (cycle,)
    cycle.append(original_tuple)
    copied_tuple = _snapshot_outcome_data(original_tuple)
    assert copied_tuple is not original_tuple
    assert copied_tuple[0][0] is copied_tuple


def test_native_immutable_record_identity_requires_immutable_fields():
    from kestrel_sovereign.features.scheduler.outcome import ScheduledTaskOutcome
    from kestrel_sovereign.signals.outcome_custody import _snapshot_outcome_data
    original = ScheduledTaskOutcome(status='blocked', result_text='actual immutable return')
    assert _snapshot_outcome_data(original) is original
    malformed = ScheduledTaskOutcome(status='blocked', result_text=['mutable'])
    with pytest.raises(TypeError, match='mutable fields'):
        _snapshot_outcome_data(malformed)


def test_snapshot_preserves_native_post_response_verdict_without_retained_alias():
    from kestrel_sovereign.agent.streaming import _PostResponseText
    from kestrel_sovereign.signals.outcome_custody import _snapshot_outcome_data
    original = _PostResponseText('genuine reviewed text', denied=True, modified=True, enforcing=True)
    copied = _snapshot_outcome_data(original)
    assert type(copied) is _PostResponseText and copied is not original
    original.denied = original.modified = original.enforcing = False
    assert copied == 'genuine reviewed text'
    assert copied.denied and copied.modified and copied.enforcing


async def test_uncopyable_result_preserves_actual_return_without_projection_or_replay(dispatcher_components):
    from kestrel_sovereign.execution_custody import is_execution_control_error, execution_native_source_error, _explicit_control_chain
    c = dispatcher_components
    calls, projected = [], []
    class Uncopyable:
        def __deepcopy__(self, memo):
            raise TypeError('original result cannot be safely copied')
    body = {'native': Uncopyable()}
    async def handler(payload):
        calls.append(True)
        return body
    registration = _action_reg('uncopyable.original', handler=handler)
    registration.result_summary = lambda value: projected.append(value) or 'must not project'
    c.registry.register(registration)
    with pytest.raises(BaseException) as caught:
        await c.dispatcher.dispatch_signal(_signal('uncopyable.original'))
    assert is_execution_control_error(caught.value)
    originals = [getattr(item, 'signal_result', None) for item in _explicit_control_chain(caught.value)]
    assert any(result is not None and result.action_result is body for result in originals)
    assert len(calls) == 1 and not projected
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log') == 0
    assert c.dispatcher._log_write_failures == 1
    assert execution_native_source_error(caught.value) is None
    assert c.dispatcher._outcome_log_control_error is None


async def test_original_journal_refuses_overridden_native_privacy_gate(dispatcher_components):
    from contextlib import asynccontextmanager
    from kestrel_sovereign.execution_custody import execution_commit_outcome, is_execution_control_error, _explicit_control_chain
    from kestrel_sovereign.storage.privacy_wrapper import DurablePersistenceGate
    c = dispatcher_components
    entered = []
    class OverriddenGate(DurablePersistenceGate):
        @asynccontextmanager
        async def _original_journal(self, exclusive_hold=None):
            entered.append(True)
            journal = c.dispatcher._original_outcome_writer.get()
            journal.begin(c.backend)
            await c.backend._commit_write(c.backend._connection, original_outcome=journal)
            raise journal.privacy_refusal('overridden_gate_before_insert')
            yield
    gate = OverriddenGate()
    class PrivacyAgent(type(c.agent)):
        def _get_durable_persistence_gate(self):
            return gate
    c.agent.__class__ = PrivacyAgent
    scope = ExecutionCustody(_OriginalWork())
    async def handler(payload):
        scope.revoke('actual original cutoff')
        return {'genuine': True}
    c.registry.register(_action_reg('overridden.gate', handler=handler))
    signal = _signal('overridden.gate')
    with bind_execution_runtime(SimpleNamespace(_execution_custody=scope)):
        with pytest.raises(BaseException) as caught:
            await c.dispatcher.dispatch_signal(signal)
    assert is_execution_control_error(caught.value)
    assert execution_commit_outcome(caught.value) != 'committed'
    assert any(getattr(item, 'reason', None) == 'unsupported_gate' for item in _explicit_control_chain(caught.value))
    assert not entered
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM main.signal_log WHERE id=?', (signal.id,)) == 0
    assert c.dispatcher._log_write_failures == 1
    assert not c.dispatcher._outcome_log_tasks


@pytest.mark.timeout(15, method='thread')
async def test_privacy_getter_cannot_drive_empty_native_commit_helper(dispatcher_components):
    from kestrel_sovereign.execution_custody import execution_commit_outcome, is_execution_control_error
    c = dispatcher_components
    saved = []
    yielded = []
    completed = []
    refused = []
    entered = []
    observation = RuntimeError('one-shot getter error after attempted empty native COMMIT')
    def epoch():
        journal = c.dispatcher._original_outcome_writer.get()
        if journal is None or journal._phase != 'submitted' or entered:
            return 'unchanged-original-policy'
        entered.append(True)
        try:
            if not saved:
                operation = c.backend._commit_write(c.backend._connection, original_outcome=journal)
                saved.append(operation)
                try:
                    yielded.append(operation.send(None))
                except ExecutionAuthorityError as denial:
                    refused.append(denial)
                    raise observation
                except StopIteration:
                    completed.append(True)
                    raise observation
            elif not completed and yielded and yielded[0].done():
                try:
                    saved[0].send(None)
                except StopIteration:
                    completed.append(True)
                    raise observation
        finally:
            entered.pop()
        return 'unchanged-original-policy'
    c.agent._get_privacy_policy_epoch = epoch
    scope = ExecutionCustody(_OriginalWork())
    async def handler(payload):
        scope.revoke('actual cutoff before getter helper reentry')
        return {'genuine': True}
    c.registry.register(_action_reg('empty.commit.reentry', handler=handler))
    signal = _signal('empty.commit.reentry')
    try:
        with bind_execution_runtime(SimpleNamespace(_execution_custody=scope)):
            with pytest.raises(BaseException) as caught:
                await c.dispatcher.dispatch_signal(signal)
        assert is_execution_control_error(caught.value)
        assert await c.backend.fetch_val('SELECT COUNT(*) FROM main.signal_log WHERE id=?', (signal.id,)) == 0
        assert execution_commit_outcome(caught.value) != 'committed', 'empty helper COMMIT was recorded as the absent original row'
        assert c.dispatcher._log_write_failures == 1
        assert refused and not completed
        assert not c.dispatcher._outcome_log_tasks
    finally:
        # The failing implementation may leave a manually driven coroutine.
        # Join its ACTUAL native future without changing future/context state,
        # then close it; never abandon a queued native COMMIT.
        if yielded and not yielded[0].done():
            done, _ = await asyncio.wait({yielded[0]}, timeout=5)
            assert done
        for operation in saved:
            if operation.cr_frame is not None:
                try:
                    operation.close()
                except BaseException as terminal:
                    assert is_execution_control_error(terminal)
        if c.dispatcher._outcome_log_control_error is not None:
            try:
                await c.dispatcher.shutdown_durable_delivery()
            except BaseException as terminal:
                assert is_execution_control_error(terminal)


@pytest.mark.timeout(15, method='thread')
@pytest.mark.parametrize('trigger', ['before_ignore', 'after_delete', 'after_mutate'])
async def test_original_journal_verifies_exact_row_before_native_commit(dispatcher_components, trigger):
    from kestrel_sovereign.execution_custody import execution_commit_outcome, execution_native_source_error, _explicit_control_chain
    c = dispatcher_components
    await c.backend.execute('CREATE TABLE journal_original_effects(id INTEGER PRIMARY KEY)')
    postgres = c.backend.backend_type == 'postgres'
    schema = await c.backend.fetch_val('SELECT current_schema()') if postgres else 'main'
    table = '"' + schema.replace('"', '""') + '".signal_log'
    effects = []
    scope = ExecutionCustody(_OriginalWork(backend_type=c.backend.backend_type))
    owner = SimpleNamespace(_execution_custody=scope)
    async def handler(payload):
        effects.append(len(effects) + 1)
        await c.backend.execute('INSERT INTO journal_original_effects VALUES(?)', (effects[-1],))
        if len(effects) == 1:
            if postgres:
                function = '"' + schema.replace('"', '""') + '".journal_terminal_filter'
                body = ('RETURN NULL;' if trigger == 'before_ignore' else
                    f'DELETE FROM {table} WHERE id=NEW.id; RETURN NULL;' if trigger in ('after_delete', 'deferred_delete') else
                    f"UPDATE {table} SET status='failed' WHERE id=NEW.id; RETURN NULL;")
                await c.backend.execute(f'CREATE FUNCTION {function}() RETURNS trigger LANGUAGE plpgsql AS $$BEGIN {body} END$$')
                if trigger == 'deferred_delete':
                    await c.backend.execute(f'CREATE CONSTRAINT TRIGGER journal_terminal_filter AFTER INSERT ON {table} DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION {function}()')
                else:
                    timing = 'BEFORE' if trigger == 'before_ignore' else 'AFTER'
                    await c.backend.execute(f'CREATE TRIGGER journal_terminal_filter {timing} INSERT ON {table} FOR EACH ROW EXECUTE FUNCTION {function}()')
            else:
                body = ('SELECT RAISE(IGNORE);' if trigger == 'before_ignore' else
                    'DELETE FROM signal_log WHERE id=NEW.id;' if trigger == 'after_delete' else
                    "UPDATE signal_log SET status='failed' WHERE id=NEW.id;")
                timing = 'BEFORE' if trigger == 'before_ignore' else 'AFTER'
                await c.backend.execute(f'CREATE TEMP TRIGGER journal_terminal_filter {timing} INSERT ON main.signal_log BEGIN {body} END')
            scope.revoke('actual work cutoff after trigger installation and committed effect')
        return {'genuine': True}
    c.registry.register(_action_reg('trigger.filtered.audit', handler=handler))
    signal = _signal('trigger.filtered.audit')
    with bind_execution_runtime(owner):
        with pytest.raises(BaseException) as caught:
            await c.dispatcher.dispatch_signal(signal)
    assert execution_commit_outcome(caught.value) != 'committed'
    assert any(getattr(item, 'signal_result', None) is not None for item in _explicit_control_chain(caught.value))
    assert await c.backend.fetch_val(f'SELECT COUNT(*) FROM {table} WHERE id=?', (signal.id,)) == 0
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM journal_original_effects') == 1
    assert c.dispatcher._log_write_failures == 1
    assert execution_native_source_error(caught.value) is None
    assert c.dispatcher._outcome_log_control_error is None
    if postgres:
        await c.backend.execute(f'DROP TRIGGER journal_terminal_filter ON {table}')
    else:
        await c.backend.execute('DROP TRIGGER temp.journal_terminal_filter')
    # This is a NEW signal, not replay of the already completed original effect.
    second = await c.dispatcher.dispatch_signal(_signal('trigger.filtered.audit'))
    assert second.action_result == {'genuine': True}
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM journal_original_effects') == 2
    assert c.dispatcher._log_write_failures == 1


async def test_postcommit_privacy_observation_does_not_poison_healthy_dispatch(dispatcher_components, monkeypatch):
    from kestrel_sovereign.execution_custody import execution_commit_outcome, execution_native_source_error, _explicit_control_chain
    c = dispatcher_components
    actual_commit = []
    raised = []
    observation = RuntimeError('one-time actual post-COMMIT privacy getter observation')
    def epoch():
        if actual_commit and not raised:
            raised.append(True)
            raise observation
        return 'unchanged-original-policy'
    c.agent._get_privacy_policy_epoch = epoch
    if c.backend.backend_type == 'sqlite':
        native_commit = c.backend._connection.commit
        async def after_actual_commit():
            result = await native_commit()
            if c.dispatcher._original_outcome_writer.get() is not None:
                actual_commit.append(True)
            return result
        monkeypatch.setattr(c.backend._connection, 'commit', after_actual_commit)
    else:
        import asyncpg
        native_release = asyncpg.pool.Pool.release
        pool = c.backend._pool
        async def after_actual_reset(original_pool, connection, *args, **kwargs):
            result = await native_release(original_pool, connection, *args, **kwargs)
            if original_pool is pool and c.dispatcher._original_outcome_writer.get() is not None:
                actual_commit.append(True)
            return result
        monkeypatch.setattr(asyncpg.pool.Pool, 'release', after_actual_reset)
    scope = ExecutionCustody(_OriginalWork(backend_type=c.backend.backend_type))
    owner = SimpleNamespace(_execution_custody=scope)
    c.registry.register(_action_reg('healthy.commit.observer'))
    signal = _signal('healthy.commit.observer')
    try:
        with bind_execution_runtime(owner):
            with pytest.raises(BaseException) as caught:
                await c.dispatcher.dispatch_signal(signal)
        assert actual_commit and raised == [True]
        assert execution_commit_outcome(caught.value) == 'committed'
        assert any(item is observation for item in _explicit_control_chain(caught.value))
        # Operator observation of the owned fixture, not a work grant to a
        # possibly poisoned original scope. Verify the real row first.
        assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log WHERE id=?', (signal.id,)) == 1
        assert c.dispatcher._log_write_failures == 0
        assert execution_native_source_error(caught.value) is None, 'healthy post-COMMIT observer became a native-source failure'
        assert c.dispatcher._outcome_log_control_error is None
        with bind_execution_runtime(owner):
            scope.require_work()
            second = await c.dispatcher.dispatch_signal(_signal('healthy.commit.observer'))
            assert second.action_result is not None
            assert c.dispatcher._log_write_failures == 0
    finally:
        if c.dispatcher._outcome_log_control_error is not None:
            from kestrel_sovereign.execution_custody import is_execution_control_error
            try:
                await c.dispatcher.shutdown_durable_delivery()
            except BaseException as terminal:
                assert is_execution_control_error(terminal)


@pytest.mark.timeout(15, method='thread')
@pytest.mark.parametrize('entry', ['dispatch', 'enqueue', 'durable_enqueue'])
async def test_sqlite_original_dispatch_refuses_enclosing_transaction_at_parent_entry(dispatcher_components, entry):
    c = dispatcher_components
    await c.dispatcher.initialize_durable_delivery()
    effects = []
    async def handler(payload):
        effects.append('unsupported effect')
        return {'genuine': True}
    c.registry.register(_action_reg('sqlite.enclosing.audit', handler=handler))
    error = handle = None
    async with c.backend.transaction():
        try:
            signal = _signal('sqlite.enclosing.audit')
            if entry == 'dispatch':
                await c.dispatcher.dispatch_signal(signal)
            elif entry == 'enqueue':
                handle = await c.dispatcher.enqueue_signal(signal)
            else:
                handle = await c.dispatcher.enqueue_durable_cognition(
                    signal, source_event_id=None, consumer_id='owned-test-consumer')
        except BaseException as caught:
            error = caught
    if handle is not None:
        await await_owned_task(handle.task)
    assert isinstance(error, ExecutionAuthorityError) and 'enclosing transaction' in str(error)
    assert effects == []
    assert not c.dispatcher._outcome_log_tasks
    assert all(task.done() for task in c.agent.background_tasks)


@pytest.mark.parametrize('cutoff', ['handler', 'journal'])
async def test_sqlite_original_journal_cannot_be_shadowed_by_temporary_table(dispatcher_components, monkeypatch, cutoff):
    c = dispatcher_components
    scope = ExecutionCustody(_OriginalWork())
    owner = SimpleNamespace(_execution_custody=scope)
    async def handler(payload):
        await c.backend.execute('CREATE TEMP TABLE signal_log AS SELECT * FROM main.signal_log WHERE 0')
        if cutoff == 'handler':
            scope.revoke('actual cutoff after ordinary temporary-table creation')
        return {'genuine': True}
    if cutoff == 'journal':
        original_writer = c.dispatcher._write_outcome_log
        async def late_cutoff(*args, **kwargs):
            scope.revoke('actual cutoff at original native writer')
            return await original_writer(*args, **kwargs)
        monkeypatch.setattr(c.dispatcher, '_write_outcome_log', late_cutoff)
    c.registry.register(_action_reg('sqlite.shadow.audit', handler=handler))
    signal = _signal('sqlite.shadow.audit')
    with bind_execution_runtime(owner):
        result = await c.dispatcher.dispatch_signal(signal)
    assert result.action_result == {'genuine': True}
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM main.signal_log WHERE id=?', (signal.id,)) == 1, 'temporary table redirected the durable original journal'
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM temp.signal_log') == 0
    assert c.dispatcher._log_write_failures == 0


async def test_privacy_getter_cannot_record_sqlite_commit_before_native_sql(dispatcher_components):
    from kestrel_sovereign.execution_custody import execution_commit_outcome, is_execution_control_error
    c = dispatcher_components
    attempted = []
    getter_error = RuntimeError('actual reentrant privacy getter failure before original INSERT')
    def epoch():
        journal = c.dispatcher._original_outcome_writer.get()
        if journal is not None and journal._phase == 'submitted' and not attempted:
            attempted.append(True)
            record = getattr(journal, 'record_sqlite_commit', None)
            if record is not None:
                record(c.backend, c.backend._connection)
            raise getter_error
        return 'original-test-epoch'
    c.agent._get_privacy_policy_epoch = epoch
    scope = ExecutionCustody(_OriginalWork())
    owner = SimpleNamespace(_execution_custody=scope)
    async def handler(payload):
        scope.revoke('actual original work cutoff before native writer')
        return {'genuine': True}
    c.registry.register(_action_reg('sqlite.getter.ack', handler=handler))
    signal = _signal('sqlite.getter.ack')
    with bind_execution_runtime(owner):
        with pytest.raises(BaseException) as caught:
            await c.dispatcher.dispatch_signal(signal)
    assert attempted == [True]
    assert is_execution_control_error(caught.value)
    assert execution_commit_outcome(caught.value) != 'committed', 'getter fabricated SQLite COMMIT before native SQL'
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM main.signal_log WHERE id=?', (signal.id,)) == 0
    assert c.dispatcher._log_write_failures == 1, 'the absent audit was incorrectly counted as committed'
    assert not c.dispatcher._outcome_log_tasks


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
            result = SignalResult(signal_id=other.id, status=Status.OK, mode=other.mode, duration_ms=0)
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


@pytest.mark.parametrize('callback', ['redaction', 'summary'])
@pytest.mark.parametrize('cutoff', [False, True])
async def test_projection_callbacks_cannot_mutate_canonical_data(dispatcher_components, callback, cutoff):
    import json
    from kestrel_sdk.signals import RedactionPolicy, Trust
    from kestrel_sovereign.signals.store import _digest
    c = dispatcher_components
    scope = ExecutionCustody(_OriginalWork(backend_type=c.backend.backend_type))
    owner = SimpleNamespace(_execution_custody=scope)
    completed = []
    accepted = {'nested': {'value': 'accepted'}}
    returned = {'nested': {'value': 'completed'}}
    async def handler(payload):
        completed.append(True)
        if cutoff:
            scope.revoke('actual cutoff before destructive projection')
        return returned
    def destructive(value):
        if completed:
            value['nested']['value'] = 'projection-mutated-canonical-data'
        return '<projected>'
    registration = _action_reg('destructive.projection.audit', handler=handler)
    registration.trust = Trust.TRUSTED
    registration.log_redaction = RedactionPolicy(
        summarize=destructive if callback == 'redaction' else lambda value: '<payload>',
        store_raw_trusted=True)
    if callback == 'summary':
        registration.result_summary = destructive
    c.registry.register(registration)
    signal = _signal('destructive.projection.audit', payload=accepted)
    with bind_execution_runtime(owner):
        result = await c.dispatcher.dispatch_signal(signal)
    row = await c.backend.fetch_one(
        'SELECT payload_digest,payload_raw,action_result_digest FROM signal_log WHERE id=?', (signal.id,))
    assert row[0] == _digest({'nested': {'value': 'accepted'}}), 'projection rewrote the accepted payload digest'
    assert json.loads(row[1]) == {'nested': {'value': 'accepted'}}, 'projection rewrote the canonical raw payload'
    assert row[2] == _digest({'nested': {'value': 'completed'}}), 'projection rewrote the original result digest'
    assert result.action_result == {'nested': {'value': 'completed'}}, 'projection mutated the genuine result returned to the caller'
    assert returned == {'nested': {'value': 'completed'}}
    assert c.dispatcher._log_write_failures == 0


@pytest.mark.parametrize('callback', ['redaction', 'summary'])
@pytest.mark.parametrize('cutoff', [False, True])
@pytest.mark.parametrize('result_kind', ['fabricated', 'genuine'])
async def test_projection_cannot_reauthorize_fabricated_result_through_finalizer(dispatcher_components, callback, cutoff, result_kind):
    from kestrel_sdk.signals import RedactionPolicy, SignalResult, SignalMode, Status
    c = dispatcher_components
    scope = ExecutionCustody(_OriginalWork(backend_type=c.backend.backend_type))
    owner = SimpleNamespace(_execution_custody=scope)
    attempted, refused = [], []
    async def handler(payload):
        if cutoff:
            scope.revoke('actual original work cutoff before projection')
        return {'genuine': True}
    registration = _action_reg('reentrant.projection.audit', handler=handler)
    signal = _signal('reentrant.projection.audit')
    fabricated = SignalResult(signal_id=signal.id, status=Status.FAILED,
        mode=SignalMode.ACTION, duration_ms=0, action_result={'fabricated': True}, error='forged terminal result')
    def projection(payload):
        if not attempted:
            attempted.append(True)
            try:
                requested = (c.dispatcher._original_outcome_admission.get()._completed_result
                             if result_kind == 'genuine' else fabricated)
                c.dispatcher._finalize_deferred_durable_outcome([], fallback_signal=signal,
                    fallback_registration=registration, result=requested)
            except ExecutionAuthorityError as error:
                refused.append(error)
        return '<original projection>'
    if callback == 'redaction':
        registration.log_redaction = RedactionPolicy(summarize=projection)
    else:
        registration.result_summary = projection
    c.registry.register(registration)
    result = error = None
    with bind_execution_runtime(owner):
        try:
            result = await c.dispatcher.dispatch_signal(signal)
        except BaseException as caught:
            error = caught
        escaped = tuple(c.dispatcher._outcome_log_tasks)
        # A failing counterproof must still physically retire the exact Tasks
        # it exposes; no fixture retry or raw native-resource cleanup shortcuts.
        if escaped:
            try:
                await c.dispatcher._join_outcome_log_tasks(escaped)
            except BaseException:
                pass  # original exploit/retirement evidence is checked below
    assert not escaped, 'projection allocated writers that escaped their dispatch owner'
    assert error is None
    assert refused and attempted == [True]
    assert result.status is Status.OK and result.action_result == {'genuine': True}
    assert c.dispatcher._log_write_failures == 0
    assert c.dispatcher._durable_active_admissions == 0
    row = await c.backend.fetch_one('SELECT status,error FROM signal_log WHERE id=?', (signal.id,))
    assert tuple(row) == ('ok', None)
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log') == 1


@pytest.mark.parametrize('cutoff', [False, True])
async def test_error_after_original_writer_acceptance_still_joins_before_owner_exits(dispatcher_components, monkeypatch, cutoff):
    from kestrel_sovereign.execution_custody import execution_commit_outcome, execution_native_source_error, _explicit_control_chain
    c = dispatcher_components
    scope = ExecutionCustody(_OriginalWork(backend_type=c.backend.backend_type))
    owner = SimpleNamespace(_execution_custody=scope)
    original = c.dispatcher._schedule_outcome_log
    publication_error = RuntimeError('actual owner error after original writer accepted')
    async def handler(payload):
        if cutoff:
            scope.revoke('actual original work cutoff')
        return {'genuine': True}
    def after_acceptance(*args, **kwargs):
        before = set(c.dispatcher._outcome_log_tasks)
        original(*args, **kwargs)
        if set(c.dispatcher._outcome_log_tasks) - before:
            raise publication_error
    monkeypatch.setattr(c.dispatcher, '_schedule_outcome_log', after_acceptance)
    c.registry.register(_action_reg('accepted.owner.error', handler=handler))
    signal = _signal('accepted.owner.error')
    error = None
    with bind_execution_runtime(owner):
        try:
            await c.dispatcher.dispatch_signal(signal)
        except BaseException as caught:
            error = caught
        escaped = tuple(c.dispatcher._outcome_log_tasks)
        if escaped:
            try:
                await c.dispatcher._join_outcome_log_tasks(escaped)
            except BaseException:
                pass  # only failing counterproof retirement; assertions below
    assert not escaped, 'accepted original writer escaped the throwing dispatch owner'
    assert execution_commit_outcome(error) == 'committed'
    assert any(member is publication_error for member in _explicit_control_chain(error))
    assert execution_native_source_error(error) is None
    assert c.dispatcher._outcome_log_control_error is None
    assert c.dispatcher._log_write_failures == 0
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log WHERE id=?', (signal.id,)) == 1
    assert c.dispatcher._durable_active_admissions == 0


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
    signal = _signal('publication.audit')
    try:
        with pytest.raises(BaseException) as caught:
            await asyncio.wait_for(c.dispatcher.dispatch_signal(signal), 2)
    finally:
        loop.set_task_factory(before)
    assert accepted and all(task.done() for task in accepted)
    assert not c.dispatcher._outcome_log_tasks
    assert not c.dispatcher._outcome_log_tasks_by_signal
    assert c.dispatcher._durable_active_admissions == 0
    from kestrel_sovereign.execution_custody import execution_commit_outcome, _explicit_control_chain, _acknowledged_commits
    assert execution_commit_outcome(caught.value) == 'committed'
    assert any(error is rejected for error in _explicit_control_chain(caught.value))
    if c.backend.backend_type == 'postgres':
        assert _acknowledged_commits(caught.value)
        assert all(receipt.retired for receipt in _acknowledged_commits(caught.value))
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log WHERE id=?', (signal.id,)) == 1
    assert c.dispatcher._outcome_log_control_error is None
    following = await c.dispatcher.dispatch_signal(_signal('publication.audit'))
    assert following.action_result is not None


@pytest.mark.parametrize('transformation', ['sanitizer', 'schema'])
async def test_terminal_snapshot_matches_canonical_handler_envelope(dispatcher_components, transformation):
    import json
    from kestrel_sdk.signals import SourceRegistration, SignalMode, RedactionPolicy, Trust
    from kestrel_sovereign.signals.store import _digest, _serialize_chain
    c = dispatcher_components
    canonical = {'allowed': 'normal', 'normalized': True}
    observed = []

    async def handler(signal):
        observed.append(dict(signal.payload))
        assert signal.payload == canonical
        return {'genuine': True}

    def schema(payload):
        return {'allowed': payload['allowed'].strip(), 'normalized': True}

    c.registry.register(SourceRegistration(name='canonical.audit', schema=schema,
        default_mode=SignalMode.ARTIFACT, allowed_modes=frozenset({SignalMode.ARTIFACT}),
        trust=Trust.UNTRUSTED if transformation == 'sanitizer' else Trust.TRUSTED,
        sanitizer=(lambda payload: {'allowed': payload['allowed'].strip()}) if transformation == 'sanitizer' else None,
        artifact_handler=handler,
        log_redaction=RedactionPolicy(summarize=lambda payload: json.dumps(payload, sort_keys=True),
                                    store_raw_trusted=True)))
    signal = _signal('canonical.audit', mode=SignalMode.ARTIFACT,
                     payload={'allowed': ' normal ', 'secret': 'never-persist-this'})
    result = await c.dispatcher.dispatch_signal(signal)
    assert result.artifact == {'genuine': True}
    assert observed == [canonical]
    row = await c.backend.fetch_one('SELECT payload_redacted,payload_raw,causation_chain_digest FROM signal_log WHERE id=?', (signal.id,))
    assert json.loads(row[0]) == canonical
    if transformation == 'schema':
        assert json.loads(row[1]) == canonical
    else:
        assert row[1] is None
    assert signal.causation_chain
    assert row[2] == _digest(_serialize_chain(signal.causation_chain))


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('ingress', ['recovered', 'resident'])
async def test_recovered_claim_has_its_own_terminal_journal(tmp_path, monkeypatch, legacy, ingress):
    import time
    from contextlib import nullcontext
    from kestrel_sovereign.signals import DurableConsumerRegistration
    from tests.unit.test_durable_signal_delivery import _channel_dispatcher, _channel_signal, _close
    from kestrel_sovereign.signals.sources.channels import (
        DURABLE_COGNITION_CONSUMER_ID, DURABLE_COGNITION_MARKER, DURABLE_COGNITION_MARKER_VALUE,
    )
    from kestrel_sovereign.execution_custody import is_execution_control_error
    backend, agent, dispatcher = await _channel_dispatcher(tmp_path / 'recovered-native-audit.db', 'did:agent:journal-recovery')
    if legacy:
        from kestrel_sovereign.signals import SignalLogStore
        class LegacySignalLogStore(SignalLogStore):
            pass
        dispatcher._store = LegacySignalLogStore(backend)
    consumer = DurableConsumerRegistration(consumer_id=DURABLE_COGNITION_CONSUMER_ID,
        source='channel.message', agent_id=agent.did,
        correlation_selector=f'payload.{DURABLE_COGNITION_MARKER}={DURABLE_COGNITION_MARKER_VALUE}', max_attempts=0)
    entered, release = asyncio.Event(), asyncio.Event()
    original = dispatcher._write_outcome_log
    task = None
    scope = ExecutionCustody(_OriginalWork())
    if ingress == 'resident' and not legacy:
        # Finite resident-publisher unit fixture: exercise the production
        # context handoff and Task publisher, not a legacy ambient fallback.
        from kestrel_sovereign.kestrel_agent import KestrelAgent
        class HostedPublisher(type(agent)):
            _runtime_owner_context = KestrelAgent._runtime_owner_context
            _track_runtime_task = KestrelAgent._track_runtime_task
            _remember_shutdown_control = KestrelAgent._remember_shutdown_control
        agent.__class__ = HostedPublisher
        agent._execution_custody = scope
        agent._background_tasks = set()
        agent._runtime_publication_ready = asyncio.Event()
        agent._runtime_publication_ready.set()

    async def held_writer(*args, **kwargs):
        assert (dispatcher._original_outcome_writer.get() is not None) == (not legacy), (
            dispatcher._original_outcome_admission.get(), dispatcher._accepted_signal_outcomes)
        if not legacy and ingress == 'resident':
            assert dispatcher._original_outcome_writer.get()._acceptance.owner.get_name().startswith('durable_cognition_owner:')
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(dispatcher, '_write_outcome_log', held_writer)
    try:
        await dispatcher.register_durable_consumer(consumer)
        signal = _channel_signal(agent.did, 'original-recovered-audit')
        if ingress == 'recovered':
            persisted = await dispatcher._durable_store.persist_signal(signal, agent_id=agent.did,
                source_event_id='recovered-original', retention_days=7,
                caller_identity=dispatcher._protect_durable_caller_identity(signal))
            delivery = await dispatcher.claim_durable_delivery_for_event(consumer_id=consumer.consumer_id,
                event_id=persisted.event_id, executor_id=dispatcher._durable_delivery_owner)
            assert delivery is not None
            seed = dispatcher._durable_retry_seed(delivery.event)
        else:
            seed = signal
        with bind_execution_runtime(SimpleNamespace(_execution_custody=scope)) if not legacy else nullcontext():
            if ingress == 'recovered':
                task = asyncio.create_task(dispatcher._route_durable_cognition_delivery(
                    seed, dispatcher._registry.get(seed.source), time.monotonic(),
                    persisted_event_id=delivery.event_id, consumer_id=consumer.consumer_id,
                    durable_admission=None, durable_created=False, use_live_signal=False,
                    claimed_delivery=delivery))
            else:
                handle = await dispatcher.enqueue_durable_cognition(signal, source_event_id='resident-original',
                    consumer_id=consumer.consumer_id)
                task = handle.task
            try:
                await asyncio.wait_for(entered.wait(), 5)
            except TimeoutError:
                assert not task.done(), await await_owned_task(task)
                raise
            if not legacy:
                scope.revoke('recovered work cutoff after original delivery settlement')
            release.set()
            outcome = await await_owned_task(task)
            assert outcome.error is None
            await dispatcher._drain_outcome_logs_for_signal(seed.id)
        assert await backend.fetch_val('SELECT COUNT(*) FROM signal_log WHERE id=?', (seed.id,)) == 1
        assert not dispatcher._accepted_signal_outcomes
    finally:
        release.set()
        if task is not None:
            await await_owned_task(task)
        try:
            await dispatcher.shutdown_durable_delivery()
        except BaseException as error:
            assert is_execution_control_error(error)
        await _close(backend, agent)


async def test_handler_cannot_terminalize_through_scheduler_after_cutoff(dispatcher_components):
    from kestrel_sdk.signals import SignalResult, Status
    from kestrel_sovereign.execution_custody import is_execution_control_error
    c = dispatcher_components
    entered, release = asyncio.Event(), asyncio.Event()
    scope = ExecutionCustody(_OriginalWork(backend_type=c.backend.backend_type))
    task = None

    async def handler(payload):
        entered.set()
        await release.wait()
        fake = SignalResult(signal_id=signal.id, status=Status.OK, mode=signal.mode,
                            duration_ms=0, action_result={'fabricated': True})
        c.dispatcher._schedule_outcome_log(signal, registration, fake)
        journal = c.dispatcher._original_outcome_admission.get()
        assert journal._phase == 'accepted'
        assert journal._entry is None
        assert not c.dispatcher._outcome_log_tasks
        return {'genuine': True}

    registration = _action_reg('scheduler.audit', handler=handler)
    c.registry.register(registration)
    signal = _signal('scheduler.audit')
    try:
        with bind_execution_runtime(SimpleNamespace(_execution_custody=scope)):
            task = asyncio.create_task(c.dispatcher.dispatch_signal(signal))
            await asyncio.wait_for(entered.wait(), 2)
            scope.revoke('cutoff before handler scheduler attempt')
            release.set()
            outcome = await await_owned_task(task)
            assert outcome.error is None
            assert outcome.result.action_result == {'genuine': True}
        assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log WHERE id=?', (signal.id,)) == 1
    finally:
        release.set()
        if task is not None:
            await await_owned_task(task)
        try:
            await c.dispatcher.shutdown_durable_delivery()
        except BaseException as error:
            assert is_execution_control_error(error)


@pytest.mark.parametrize('boundary', ['before_emit', 'emitter_denial', 'emitter_cancel'])
async def test_visible_journal_does_not_admit_ui_work_after_cutoff(dispatcher_components, monkeypatch, boundary):
    from dataclasses import replace
    from kestrel_sdk.signals import Visibility
    from kestrel_sovereign.execution_custody import execution_commit_outcome, is_execution_control_error, _explicit_control_chain
    from kestrel_sovereign._async_ownership import create_owned_task
    c = dispatcher_components
    entered, release = asyncio.Event(), asyncio.Event()
    emitted = []
    scope = ExecutionCustody(_OriginalWork(backend_type=c.backend.backend_type))
    original = c.dispatcher._write_outcome_log
    task = None
    original_denial = (asyncio.CancelledError('original emitter withdrawal') if boundary == 'emitter_cancel'
                       else ExecutionAuthorityError('original emitter denial'))

    async def emit(*args):
        emitted.append(args)
        if boundary in ('emitter_denial', 'emitter_cancel'):
            raise original_denial

    async def held_writer(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    c.agent.emit_event = emit
    monkeypatch.setattr(c.dispatcher, '_write_outcome_log', held_writer)
    c.registry.register(_action_reg('visible.audit'))
    signal = replace(_signal('visible.audit'), visibility=Visibility.USER_VISIBLE)
    try:
        with bind_execution_runtime(SimpleNamespace(_execution_custody=scope)):
            task = create_owned_task(c.dispatcher.dispatch_signal(signal))
            await asyncio.wait_for(entered.wait(), 2)
            if boundary == 'before_emit':
                scope.revoke('original work cutoff before visible journal')
            release.set()
            outcome = await await_owned_task(task)
            assert outcome.error is not None
            if boundary != 'emitter_cancel':
                assert is_execution_control_error(outcome.error)
            assert execution_commit_outcome(outcome.error) == 'committed'
            if boundary == 'before_emit':
                with pytest.raises(ExecutionAuthorityError):
                    scope.require_work()
            if boundary in ('emitter_denial', 'emitter_cancel'):
                assert any(error is original_denial for error in _explicit_control_chain(outcome.error))
        assert len(emitted) == (0 if boundary == 'before_emit' else 1)
        assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log WHERE id=?', (signal.id,)) == 1
        if boundary in ('before_emit', 'emitter_cancel'):
            assert c.dispatcher._outcome_log_control_error is None
            assert (await c.dispatcher.dispatch_signal(_signal('visible.audit'))).action_result is not None
    finally:
        release.set()
        if task is not None:
            await await_owned_task(task)
        try:
            await c.dispatcher.shutdown_durable_delivery()
        except BaseException as error:
            assert is_execution_control_error(error)


@pytest.mark.parametrize('registered_before_refusal', [False, True])
@pytest.mark.parametrize('writer_method', ['_write_outcome_log', '_log_safe'])
async def test_inner_writer_observation_retains_commit_and_original_error(dispatcher_components, registered_before_refusal, writer_method):
    from kestrel_sovereign.execution_custody import execution_commit_outcome, execution_native_source_error, _explicit_control_chain, _acknowledged_commits
    from tests.unit.test_host_context_native_cleanup import _actual_operation
    c = dispatcher_components
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    original = RuntimeError('original inner writer observation refused')
    accepted = []

    class RejectObservation(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            if getattr(callback, '__name__', '') == 'mark_complete':
                if registered_before_refusal:
                    super().add_done_callback(callback, context=context)
                raise original
            return super().add_done_callback(callback, context=context)

    def factory(loop, coro, **kwargs):
        operation = _actual_operation(coro)
        if getattr(operation, 'cr_code', None) is getattr(c.dispatcher, writer_method).__func__.__code__:
            task = RejectObservation(coro, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coro, **kwargs) if previous else asyncio.Task(coro, loop=loop, **kwargs)

    c.registry.register(_action_reg('inner.observation.audit'))
    signal = _signal('inner.observation.audit')
    loop.set_task_factory(factory)
    try:
        with pytest.raises(BaseException) as caught:
            await c.dispatcher.dispatch_signal(signal)
    finally:
        loop.set_task_factory(previous)
    assert accepted and all(task.done() for task in accepted)
    assert any(error is original for error in _explicit_control_chain(caught.value))
    assert execution_commit_outcome(caught.value) == 'committed'
    assert execution_native_source_error(caught.value) is None
    if c.backend.backend_type == 'postgres':
        assert _acknowledged_commits(caught.value)
        assert all(receipt.retired for receipt in _acknowledged_commits(caught.value))
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log WHERE id=?', (signal.id,)) == 1
    assert c.dispatcher._log_write_failures == 0
    assert c.dispatcher._outcome_log_control_error is None
    assert (await c.dispatcher.dispatch_signal(_signal('inner.observation.audit'))).action_result is not None


@pytest.mark.parametrize('callback', ['redaction', 'summary', 'inner_allocation', 'outer_allocation', 'inner_publication', 'outer_publication'])
@pytest.mark.parametrize('representation', ['typed', 'legacy'])
async def test_actual_committed_journal_observation_is_not_a_projection_placeholder(dispatcher_components, callback, representation):
    from kestrel_sdk.signals import RedactionPolicy
    from kestrel_sovereign.execution_custody import execution_commit_outcome, execution_native_source_error, is_execution_control_error, _explicit_control_chain
    from tests.unit.test_host_context_native_cleanup import _actual_operation
    c = dispatcher_components
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    original = RuntimeError('actual original committed journal observation')
    class RejectObservation(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            if getattr(callback, '__name__', '') == 'mark_complete':
                raise original
            return super().add_done_callback(callback, context=context)
    def factory(loop, coro, **kwargs):
        operation = _actual_operation(coro)
        if getattr(operation, 'cr_code', None) is c.dispatcher._write_outcome_log.__func__.__code__:
            return RejectObservation(coro, loop=loop, **kwargs)
        return previous(loop, coro, **kwargs) if previous else asyncio.Task(coro, loop=loop, **kwargs)
    c.registry.register(_action_reg('actual.committed.audit'))
    first = _signal('actual.committed.audit')
    loop.set_task_factory(factory)
    try:
        with pytest.raises(BaseException) as caught:
            await c.dispatcher.dispatch_signal(first)
    finally:
        loop.set_task_factory(previous)
    carrier = caught.value
    assert execution_commit_outcome(carrier) == 'committed'
    assert is_execution_control_error(carrier)
    assert execution_native_source_error(carrier) is None
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log WHERE id=?', (first.id,)) == 1
    if representation == 'legacy':
        # Compatibility projection of this ACTUAL acknowledged native row's
        # original observation error into the previous metadata-only format.
        # No new ACK/Task/authority or product-completion proof is fabricated.
        original._kestrel_unresolved_commit_outcome = execution_commit_outcome(carrier)
        carrier = original
        assert not is_execution_control_error(carrier)
    def rethrow_actual_carrier(payload):
        raise carrier
    registration = _action_reg('committed.projection.audit')
    if callback == 'redaction':
        registration.log_redaction = RedactionPolicy(summarize=rethrow_actual_carrier)
    elif callback == 'summary':
        registration.result_summary = rethrow_actual_carrier
    c.registry.register(registration)
    second = _signal('committed.projection.audit')
    class RejectCommittedStart(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            if getattr(callback, '__name__', '') == 'close_unstarted':
                raise carrier
            return super().add_done_callback(callback, context=context)
    def second_factory(loop, coro, **kwargs):
        operation = _actual_operation(coro)
        method = c.dispatcher._write_outcome_log if callback.startswith('inner_') else c.dispatcher._log_safe
        if getattr(operation, 'cr_code', None) is method.__func__.__code__:
            if callback.endswith('allocation'):
                raise carrier
            return RejectCommittedStart(coro, loop=loop, **kwargs)
        return previous(loop, coro, **kwargs) if previous else asyncio.Task(coro, loop=loop, **kwargs)
    if callback not in ('redaction', 'summary'):
        loop.set_task_factory(second_factory)
    try:
        with pytest.raises(BaseException) as propagated:
            await c.dispatcher.dispatch_signal(second)
    finally:
        loop.set_task_factory(previous)
    assert execution_commit_outcome(propagated.value) == 'committed'
    assert any(error is carrier for error in _explicit_control_chain(propagated.value))
    assert execution_native_source_error(propagated.value) is None
    assert c.dispatcher._outcome_log_control_error is None
    assert c.dispatcher._log_write_failures == 1
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log WHERE id=?', (second.id,)) == 0
    assert (await c.dispatcher.dispatch_signal(_signal('actual.committed.audit'))).action_result is not None


async def test_failed_sqlite_insert_preserves_cancellation_during_successful_rollback(dispatcher_components, monkeypatch):
    import threading
    from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_commit_outcome, execution_native_source_error
    c = dispatcher_components
    c.registry.register(_action_reg('failed.insert.audit'))
    await c.dispatcher.dispatch_signal(_signal('failed.insert.audit'))
    # The original destination must exist at admission. A real uniqueness
    # violation reaches INSERT/rollback without bypassing that new fence.
    await c.backend.execute('CREATE UNIQUE INDEX journal_duplicate_source ON signal_log(source)')
    conn = c.backend._connection
    original_execute = conn._execute
    entered, worker_release = asyncio.Event(), threading.Event()
    body_errors, used_barrier = [], []
    loop = asyncio.get_running_loop()
    async def actual_queue(function, *args, **kwargs):
        journal = c.dispatcher._original_outcome_writer.get()
        if journal is not None and getattr(function, '__name__', '') == 'rollback' and not used_barrier:
            used_barrier.append(True)
            def native_rollback():
                loop.call_soon_threadsafe(entered.set)
                worker_release.wait(5)
                return function(*args, **kwargs)
            return await original_execute(native_rollback)
        try:
            return await original_execute(function, *args, **kwargs)
        except BaseException as error:
            if journal is not None and getattr(function, '__name__', '') == 'execute':
                body_errors.append(error)
            raise
    monkeypatch.setattr(conn, '_execute', actual_queue)
    task = asyncio.create_task(c.dispatcher.dispatch_signal(_signal('failed.insert.audit')))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        journal = next(entry[2] for entry in c.dispatcher._outcome_log_reconciliations.values() if entry[2] is not None)
        journal._writer_admission.task.cancel()
        await asyncio.sleep(0)
        worker_release.set()
        outcome = await await_owned_task(task)
        assert outcome.error is not None, 'ordinary INSERT failure swallowed actual cleanup cancellation'
        chain = tuple(_explicit_control_chain(outcome.error))
        assert len(body_errors) == 1 and any(error is body_errors[0] for error in chain)
        assert any(isinstance(error, asyncio.CancelledError) for error in chain)
        assert execution_commit_outcome(outcome.error) is None
        assert execution_native_source_error(outcome.error) is None
        assert c.dispatcher._log_write_failures == 1
        assert c.dispatcher._outcome_log_control_error is None
    finally:
        worker_release.set()
        await await_owned_task(task)


async def test_sqlite_original_journal_retains_commit_before_identity_refusal(dispatcher_components, monkeypatch):
    from kestrel_sovereign.execution_custody import execution_commit_outcome, is_execution_control_error
    c = dispatcher_components
    conn = c.backend._connection
    original_commit = conn.commit
    original_id = c.agent._did
    async def committed_then_identity_changed():
        await original_commit()
        journal = c.dispatcher._original_outcome_writer.get()
        if journal is not None and journal._phase == 'submitted':
            c.agent._did = 'replacement-after-actual-sqlite-commit'
    c.registry.register(_action_reg('sqlite.committed.audit'))
    signal = _signal('sqlite.committed.audit')
    monkeypatch.setattr(conn, 'commit', committed_then_identity_changed)
    try:
        with pytest.raises(BaseException) as caught:
            await c.dispatcher.dispatch_signal(signal)
        assert is_execution_control_error(caught.value)
        assert execution_commit_outcome(caught.value) == 'committed'
        assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log WHERE id=?', (signal.id,)) == 1
        assert c.dispatcher._log_write_failures == 0
    finally:
        c.agent._did = original_id
        monkeypatch.setattr(conn, 'commit', original_commit)
        try:
            await c.dispatcher.shutdown_durable_delivery()
        except BaseException as error:
            assert is_execution_control_error(error)


@pytest.mark.parametrize('boundary', ['different_sql', 'enclosing_transaction'])
async def test_sqlite_original_journal_native_defenses_are_typed(dispatcher_components, monkeypatch, boundary):
    c = dispatcher_components
    original = c.backend._append_original_signal_outcome
    async def native_defense(journal):
        if boundary == 'different_sql':
            with pytest.raises(ExecutionAuthorityError, match='SQL differs'):
                await c.backend._execute_write_operation(c.backend._connection,
                    'DELETE FROM signal_log', (), original_outcome=journal)
        else:
            async with c.backend.transaction():
                with pytest.raises(ExecutionAuthorityError, match='enclosing transaction'):
                    await original(journal)
        return await original(journal)
    monkeypatch.setattr(c.backend, '_append_original_signal_outcome', native_defense)
    c.registry.register(_action_reg('sqlite.native.defense'))
    signal = _signal('sqlite.native.defense')
    assert (await c.dispatcher.dispatch_signal(signal)).status.value == 'ok'
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log WHERE id=?', (signal.id,)) == 1
    assert c.dispatcher._log_write_failures == 0


@pytest.mark.parametrize('writer_method', ['_write_outcome_log', '_log_safe'])
@pytest.mark.parametrize('failure', ['allocation', 'publication_before', 'publication_after'])
async def test_unstarted_journal_writer_loss_is_counted_once_without_relabelling_effect(dispatcher_components, writer_method, failure):
    from tests.unit.test_host_context_native_cleanup import _actual_operation
    c = dispatcher_components
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    original = RuntimeError('original journal Task admission refused')
    accepted, effects = [], []
    class RejectStartPublication(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            if getattr(callback, '__name__', '') == 'close_unstarted':
                if failure == 'publication_after':
                    super().add_done_callback(callback, context=context)
                raise original
            return super().add_done_callback(callback, context=context)
    def factory(loop, coro, **kwargs):
        operation = _actual_operation(coro)
        if getattr(operation, 'cr_code', None) is getattr(c.dispatcher, writer_method).__func__.__code__:
            if failure == 'allocation':
                raise original
            task = RejectStartPublication(coro, loop=loop, **kwargs)
            accepted.append(task)
            return task
        return previous(loop, coro, **kwargs) if previous else asyncio.Task(coro, loop=loop, **kwargs)
    async def handler(payload):
        effects.append('genuine-completed-effect')
        return {'completed': True}
    c.registry.register(_action_reg('unstarted.audit', handler=handler))
    signal = _signal('unstarted.audit')
    loop.set_task_factory(factory)
    try:
        result = await c.dispatcher.dispatch_signal(signal)
    finally:
        loop.set_task_factory(previous)
    assert effects == ['genuine-completed-effect']
    assert c.dispatcher._log_write_failures == 1
    assert result.status.value == 'ok'
    assert all(task.done() for task in accepted)
    assert not c.dispatcher._outcome_log_tasks
    assert c.dispatcher._durable_active_admissions == 0
    assert c.dispatcher._outcome_log_control_error is None
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log WHERE id=?', (signal.id,)) == 0
    assert (await c.dispatcher.dispatch_signal(_signal('unstarted.audit'))).action_result is not None


@pytest.mark.parametrize('failure', ['normal', 'allocation', 'publication_before', 'publication_after', 'concurrent_close', 'rollback_failure', 'insert_rollback_failure', 'repeat_cancel'])
async def test_sqlite_original_journal_privacy_hold_outlives_cancelled_commit_cleanup(dispatcher_components, monkeypatch, failure):
    from tests.unit.test_host_context_native_cleanup import _actual_operation
    from kestrel_sovereign.storage.privacy_wrapper import DurablePersistenceGate
    c = dispatcher_components
    gate = DurablePersistenceGate()
    class PrivacyAgent(type(c.agent)):
        def _get_durable_persistence_gate(self):
            return gate
    c.agent.__class__ = PrivacyAgent
    conn = c.backend._connection
    original_execute = conn._execute
    original_drain = c.backend._drain_cancelled_write
    original_cleanup = c.backend._rollback_cancelled_write
    cleanup_errors = []
    previous = asyncio.get_running_loop().get_task_factory()
    accepted_drains = []
    class RejectDrainStart(asyncio.Task):
        def add_done_callback(self, callback, *, context=None):
            if getattr(callback, '__name__', '') == 'close_unstarted':
                if failure == 'publication_after':
                    super().add_done_callback(callback, context=context)
                raise RuntimeError('original SQLite drain publication refused')
            return super().add_done_callback(callback, context=context)
    def factory(loop, coro, **kwargs):
        operation = _actual_operation(coro)
        if getattr(operation, 'cr_code', None) is original_drain.__func__.__code__:
            if failure == 'allocation':
                raise RuntimeError('original SQLite drain allocation refused')
            if failure.startswith('publication'):
                task = RejectDrainStart(coro, loop=loop, **kwargs)
                accepted_drains.append(task)
                return task
        return previous(loop, coro, **kwargs) if previous else asyncio.Task(coro, loop=loop, **kwargs)
    native_entered, native_release, drain_entered, drain_release, transition_started, transition_entered = (asyncio.Event() for _ in range(6))
    import threading
    worker_release = threading.Event()
    async def held_actual_execute(function, *args, **kwargs):
        journal = c.dispatcher._original_outcome_writer.get()
        selected = (getattr(function, '__name__', '') == 'execute'
            and journal is not None and args and args[0] is journal._query
            if failure == 'insert_rollback_failure'
            else getattr(function, '__name__', '') == 'commit')
        if journal is None or not selected:
            return await original_execute(function, *args, **kwargs)
        # The enqueued native operation itself calls the real COMMIT after
        # this worker barrier, even if its Python await has been cancelled.
        def worker_barrier():
            loop.call_soon_threadsafe(native_entered.set)
            worker_release.wait(5)
            return function(*args, **kwargs)
        return await original_execute(worker_barrier)
    async def held_original_cleanup(connection, *, started_at):
        drain_entered.set()
        await drain_release.wait()
        if failure.endswith('rollback_failure'):
            await connection.close()
            try:
                await connection.rollback()
            except BaseException as error:
                cleanup_errors.append(error)
                raise
        return await original_cleanup(connection, started_at=started_at)
    async def transition():
        transition_started.set()
        async with gate.exclusive():
            transition_entered.set()
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(conn, '_execute', held_actual_execute)
    monkeypatch.setattr(c.backend, '_rollback_cancelled_write', held_original_cleanup)
    c.registry.register(_action_reg('sqlite.cancelled.audit'))
    task = next_transition = closing = cleanup_wait = None
    try:
        task = asyncio.create_task(c.dispatcher.dispatch_signal(_signal('sqlite.cancelled.audit')))
        await asyncio.wait_for(native_entered.wait(), 2)
        journals = [entry[2] for entry in c.dispatcher._outcome_log_reconciliations.values()]
        journal = next(journal for journal in journals if journal is not None)
        loop.set_task_factory(factory)
        if failure == 'concurrent_close':
            closing = asyncio.create_task(c.backend.close())
            await asyncio.sleep(0)
        journal._writer_admission.task.cancel()
        cleanup_wait = asyncio.create_task(drain_entered.wait())
        first, _ = await asyncio.wait({cleanup_wait, task}, timeout=2,
                                      return_when=asyncio.FIRST_COMPLETED)
        assert first, 'original writer neither reported refusal nor began cleanup'
        if failure == 'repeat_cancel':
            journal._writer_admission.task.cancel()
            await asyncio.sleep(0)
            assert not journal._writer_admission.task.done()
        next_transition = asyncio.create_task(transition())
        await transition_started.wait()
        await asyncio.sleep(0)
        assert not transition_entered.is_set(), 'privacy transition overtook original SQLite COMMIT/rollback cleanup'
    finally:
        loop.set_task_factory(previous)
        native_release.set()
        worker_release.set()
        drain_release.set()
        if task is not None:
            original_outcome = await await_owned_task(task)
        if c.backend._cancelled_write_drain is not None:
            await await_owned_task(c.backend._cancelled_write_drain)
        if next_transition is not None:
            await next_transition
        if cleanup_wait is not None:
            cleanup_wait.cancel()
            await await_owned_task(cleanup_wait)
        if closing is not None:
            await await_owned_task(closing)
        assert all(task.done() for task in accepted_drains)
        from kestrel_sovereign.execution_custody import is_execution_control_error
        try:
            await c.dispatcher.shutdown_durable_delivery()
        except BaseException as error:
            assert is_execution_control_error(error)
        try:
            await c.backend.close()
        except BaseException as error:
            assert is_execution_control_error(error)
    if failure.endswith('rollback_failure'):
        from kestrel_sovereign.execution_custody import _explicit_control_chain
        assert len(cleanup_errors) == 1
        assert any(error is cleanup_errors[0] for error in _explicit_control_chain(original_outcome.error))
        assert any(isinstance(error, asyncio.CancelledError) for error in _explicit_control_chain(original_outcome.error))
    if failure == 'repeat_cancel':
        from kestrel_sovereign.execution_custody import _explicit_control_chain
        assert len([error for error in _explicit_control_chain(original_outcome.error)
                    if isinstance(error, asyncio.CancelledError)]) >= 2


async def test_original_journal_completes_inside_original_exclusive_privacy_hold(dispatcher_components):
    from kestrel_sovereign.storage.privacy_wrapper import DurablePersistenceGate
    c = dispatcher_components
    gate = DurablePersistenceGate()
    class PrivacyAgent(type(c.agent)):
        def _get_durable_persistence_gate(self):
            return gate
    c.agent.__class__ = PrivacyAgent
    c.registry.register(_action_reg('privacy.exclusive.audit'))
    signal = _signal('privacy.exclusive.audit')

    async def inline_dispatch():
        async with gate.exclusive():
            return await c.dispatcher.dispatch_signal(signal)

    task = asyncio.create_task(inline_dispatch())
    completed, _ = await asyncio.wait({task}, timeout=2)
    if not completed:
        # Cancel the actual child waiting for the gate, not the parent whose
        # physical join deliberately defers withdrawal. This unwinds the
        # original exclusive frame without forging a lock release.
        for _, _, journal in tuple(c.dispatcher._outcome_log_reconciliations.values()):
            if journal is not None and journal._writer_admission is not None:
                journal._writer_admission.task.cancel()
    outcome = await await_owned_task(task)
    assert completed, 'original inline dispatch deadlocked on its own exclusive privacy gate'
    assert outcome.error is None
    assert not gate.locked()
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log WHERE id=?', (signal.id,)) == 1


@pytest.mark.parametrize('boundary', ['writer_queued', 'writer_active', 'compatible_reader', 'inherited_shared'])
async def test_original_journal_gate_admission_never_waits_on_a_transition(dispatcher_components, monkeypatch, boundary):
    from kestrel_sovereign.storage.privacy_wrapper import DurablePersistenceGate
    from kestrel_sovereign.signals.outcome_custody import SignalOutcomeJournalPrivacyRefusal
    from kestrel_sovereign.execution_custody import execution_native_source_error, execution_commit_outcome, _explicit_control_chain
    c = dispatcher_components
    gate = DurablePersistenceGate()
    class PrivacyAgent(type(c.agent)):
        def _get_durable_persistence_gate(self):
            return gate
        def _get_privacy_policy_epoch(self):
            return 'unchanged-original-policy'
    c.agent.__class__ = PrivacyAgent
    entered, release, transition_started = (asyncio.Event() for _ in range(3))
    original = c.store._append_original_outcome
    async def held_journal(journal):
        entered.set()
        await release.wait()
        return await original(journal)
    async def transition():
        transition_started.set()
        async with gate.exclusive():
            pass
    monkeypatch.setattr(c.store, '_append_original_outcome', held_journal)
    c.registry.register(_action_reg('privacy.admission.audit'))
    signal = _signal('privacy.admission.audit')
    task = next_transition = None
    try:
        # Only the inherited case is actually created under the retained
        # shared frame. A later holder cannot retroactively lend its lease.
        if boundary != 'inherited_shared':
            task = asyncio.create_task(c.dispatcher.dispatch_signal(signal))
            await asyncio.wait_for(entered.wait(), 2)
        async with gate.exclusive() if boundary == 'writer_active' else gate.shared():
            if task is None:
                task = asyncio.create_task(c.dispatcher.dispatch_signal(signal))
                await asyncio.wait_for(entered.wait(), 2)
            if boundary in ('writer_queued', 'inherited_shared'):
                next_transition = asyncio.create_task(transition())
                await transition_started.wait()
            release.set()
            completed, _ = await asyncio.wait({task}, timeout=2)
            if not completed:
                for _, _, journal in tuple(c.dispatcher._outcome_log_reconciliations.values()):
                    if journal is not None and journal._writer_admission is not None:
                        journal._writer_admission.task.cancel()
            outcome = await await_owned_task(task)
            assert completed, 'original journal queued behind a transition while its owner waited'
            refused = boundary in ('writer_active', 'writer_queued')
            if refused:
                refusals = [error for error in _explicit_control_chain(outcome.error)
                            if isinstance(error, SignalOutcomeJournalPrivacyRefusal)]
                assert len(refusals) == 1 and refusals[0].reason == 'busy'
                assert refusals[0].signal_result.status.value == 'ok'
                assert execution_native_source_error(outcome.error) is None
                assert execution_commit_outcome(outcome.error) is None
            else:
                assert outcome.error is None
            assert c.dispatcher._outcome_log_control_error is None
            assert c.dispatcher._log_write_failures == int(refused)
            assert not c.dispatcher._outcome_log_tasks
            assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log WHERE id=?', (signal.id,)) == int(not refused)
        if next_transition is not None:
            await next_transition
        assert (await c.dispatcher.dispatch_signal(_signal('privacy.admission.audit'))).action_result is not None
    finally:
        release.set()
        if task is not None:
            await await_owned_task(task)
        if next_transition is not None:
            await next_transition


@pytest.mark.parametrize('stale_hook_identity', [False, True])
async def test_ack_refusal_journals_distinct_redelivery_attempt(tmp_path, stale_hook_identity):
    import time
    from kestrel_sovereign.signals import DurableConsumerRegistration
    from tests.unit.test_durable_signal_delivery import _channel_dispatcher, _channel_signal, _close
    from kestrel_sovereign.signals.sources.channels import (
        DURABLE_COGNITION_CONSUMER_ID, DURABLE_COGNITION_MARKER, DURABLE_COGNITION_MARKER_VALUE,
    )
    backend, agent, dispatcher = await _channel_dispatcher(tmp_path / 'ack-refused-audit.db', 'did:agent:ack-journal')
    consumer = DurableConsumerRegistration(consumer_id=DURABLE_COGNITION_CONSUMER_ID,
        source='channel.message', agent_id=agent.did,
        correlation_selector=f'payload.{DURABLE_COGNITION_MARKER}={DURABLE_COGNITION_MARKER_VALUE}', max_attempts=0)
    effects = []
    try:
        await dispatcher.register_durable_consumer(consumer)
        canonical = _channel_signal(agent.did, 'original-ack-refusal')
        persisted = await dispatcher._durable_store.persist_signal(canonical, agent_id=agent.did,
            source_event_id='original-ack-refusal', retention_days=7,
            caller_identity=None if stale_hook_identity else dispatcher._protect_durable_caller_identity(canonical))
        delivery = await dispatcher.claim_durable_delivery_for_event(consumer_id=consumer.consumer_id,
            event_id=persisted.event_id, executor_id=dispatcher._durable_delivery_owner)
        if stale_hook_identity:
            agent.rehydrate_durable_cognition_signal = lambda event, *, dispatch_signal: canonical
            agent.durable_rehydratable_sources = frozenset({'channel.message'})

        async def completed_cognition(prompt):
            effects.append('genuinely completed')
            # Actual original-token NACK means the later actual ACK CAS
            # cannot succeed; no fabricated clock or boolean ACK stub.
            assert await dispatcher.nack_durable_delivery(consumer_id=consumer.consumer_id,
                delivery_id=delivery.delivery_id, lease_token=delivery.lease_token,
                error='original lease released after effect before ACK')
            return 'completed'

        agent.process_input = completed_cognition
        incoming = dispatcher._durable_retry_seed(delivery.event)
        assert incoming.id != canonical.id
        result = await dispatcher._route_durable_cognition_delivery(incoming,
            dispatcher._registry.get(incoming.source), time.monotonic(), persisted_event_id=delivery.event_id,
            consumer_id=consumer.consumer_id, durable_admission=None, durable_created=False,
            use_live_signal=False, claimed_delivery=delivery)
        await dispatcher._drain_outcome_logs_for_signal(result.signal_id)
        assert effects == ['genuinely completed']
        assert 'ACK' in result.error
        assert await backend.fetch_val('SELECT COUNT(*) FROM signal_log WHERE id=?', (result.signal_id,)) == 1
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


async def test_accepted_payload_is_not_replaced_after_cutoff(dispatcher_components):
    import json
    from dataclasses import replace
    from kestrel_sdk.signals import RedactionPolicy, Trust
    c = dispatcher_components
    entered, release = asyncio.Event(), asyncio.Event()
    scope = ExecutionCustody(_OriginalWork(backend_type=c.backend.backend_type))
    task = None

    async def handler(payload):
        entered.set()
        await release.wait()
        payload['authorized'] = 'replacement-after-cutoff'
        return {'genuine': True}

    registration = replace(_action_reg('snapshot.audit', handler=handler), trust=Trust.TRUSTED,
        log_redaction=RedactionPolicy(summarize=lambda payload: payload['authorized'], store_raw_trusted=True))
    c.registry.register(registration)
    signal = _signal('snapshot.audit', payload={'authorized': 'original'})
    try:
        with bind_execution_runtime(SimpleNamespace(_execution_custody=scope)):
            task = asyncio.create_task(c.dispatcher.dispatch_signal(signal))
            await asyncio.wait_for(entered.wait(), 2)
            scope.revoke('original envelope cutoff')
            release.set()
            outcome = await await_owned_task(task)
            assert outcome.error is None
        raw, redacted = await c.backend.fetch_one('SELECT payload_raw,payload_redacted FROM signal_log WHERE id=?', (signal.id,))
        assert json.loads(raw) == {'authorized': 'original'}
        assert redacted == 'original'
    finally:
        release.set()
        if task is not None:
            await await_owned_task(task)


@pytest.mark.parametrize('withdrawal', ['dispatch', 'outer_writer'])
async def test_owner_withdrawal_retains_actual_original_journal_commit(dispatcher_components, monkeypatch, withdrawal):
    from kestrel_sovereign._async_ownership import create_owned_task
    from kestrel_sovereign.execution_custody import execution_commit_outcome, is_execution_control_error, _acknowledged_commits
    c = dispatcher_components
    entered, release = asyncio.Event(), asyncio.Event()
    original = c.dispatcher._write_outcome_log
    journals = []
    task = None

    async def held_writer(*args, **kwargs):
        journals.append(c.dispatcher._original_outcome_writer.get())
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(c.dispatcher, '_write_outcome_log', held_writer)
    c.registry.register(_action_reg('withdrawal.audit'))
    signal = _signal('withdrawal.audit')
    try:
        task = create_owned_task(c.dispatcher.dispatch_signal(signal))
        await asyncio.wait_for(entered.wait(), 2)
        journal = journals[0]
        (task if withdrawal == 'dispatch' else journal._outer_admission.task).cancel('original owner withdrawal')
        release.set()
        outcome = await await_owned_task(task)
        error = outcome.error or outcome.cancellation
        assert error is not None
        assert journal._phase == 'acknowledged'
        assert execution_commit_outcome(error) == 'committed'
        if c.backend.backend_type == 'postgres':
            retained = _acknowledged_commits(error)
            assert retained and all(any(actual is receipt for actual in retained) for receipt in journal._acknowledgments)
        # Healthy physical COMMIT/retirement plus finite withdrawal must not
        # leave the reusable runtime poisoned by an observer's cancellation.
        assert c.dispatcher._outcome_log_control_error is None
        following = await c.dispatcher.dispatch_signal(_signal('withdrawal.audit'))
        assert following.action_result is not None
    finally:
        release.set()
        if task is not None:
            await await_owned_task(task)
        try:
            await c.dispatcher.shutdown_durable_delivery()
        except BaseException as error:
            assert is_execution_control_error(error)
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log WHERE id=?', (signal.id,)) == 1


@pytest.mark.parametrize('boundary', ['held', 'hold_read', 'execution_admission'])
async def test_early_recovered_claim_journals_its_actual_disposition(tmp_path, boundary):
    import time
    from kestrel_sdk.signals import Status
    from kestrel_sovereign.signals import DurableConsumerRegistration
    from kestrel_sovereign.hold import EffectiveHoldState
    from tests.unit.test_durable_signal_delivery import (
        _channel_dispatcher, _channel_signal, _close, _HoldSnapshots, _FailingHoldRead, _held_state,
    )
    from kestrel_sovereign.signals.sources.channels import (
        DURABLE_COGNITION_CONSUMER_ID, DURABLE_COGNITION_MARKER, DURABLE_COGNITION_MARKER_VALUE,
    )
    backend, agent, dispatcher = await _channel_dispatcher(tmp_path / 'early-recovered-audit.db', 'did:agent:early-journal')
    consumer = DurableConsumerRegistration(consumer_id=DURABLE_COGNITION_CONSUMER_ID,
        source='channel.message', agent_id=agent.did,
        correlation_selector=f'payload.{DURABLE_COGNITION_MARKER}={DURABLE_COGNITION_MARKER_VALUE}', max_attempts=0)
    free = EffectiveHoldState(host=None, agent=None)
    try:
        await dispatcher.register_durable_consumer(consumer)
        signal = _channel_signal(agent.did, 'original-early-recovered-audit')
        persisted = await dispatcher._durable_store.persist_signal(signal, agent_id=agent.did,
            source_event_id='early-recovered-original', retention_days=7,
            caller_identity=dispatcher._protect_durable_caller_identity(signal))
        delivery = await dispatcher.claim_durable_delivery_for_event(consumer_id=consumer.consumer_id,
            event_id=persisted.event_id, executor_id=dispatcher._durable_delivery_owner)
        assert delivery is not None
        if boundary == 'held':
            agent._hold_store = _HoldSnapshots(free, _held_state(agent.did))
        elif boundary == 'hold_read':
            agent._hold_store = _FailingHoldRead(fail_on=2)
        else:
            class WithdrawClaim:
                reads = 0
                async def get_effective(self, _agent_id):
                    self.reads += 1
                    if self.reads == 2:
                        assert await dispatcher.nack_durable_delivery(consumer_id=consumer.consumer_id,
                            delivery_id=delivery.delivery_id, lease_token=delivery.lease_token,
                            error='original accepted lease withdrawn before execution')
                    return free
            agent._hold_store = WithdrawClaim()
        seed = dispatcher._durable_retry_seed(delivery.event)
        result = await dispatcher._route_durable_cognition_delivery(seed,
            dispatcher._registry.get(seed.source), time.monotonic(), persisted_event_id=delivery.event_id,
            consumer_id=consumer.consumer_id, durable_admission=None, durable_created=False,
            use_live_signal=False, claimed_delivery=delivery)
        await dispatcher._drain_outcome_logs_for_signal(seed.id)
        assert result.status != Status.OK
        assert await backend.fetch_val('SELECT COUNT(*) FROM signal_log WHERE id=?', (seed.id,)) == 1
        assert not dispatcher._accepted_signal_outcomes
    finally:
        await dispatcher.shutdown_durable_delivery()
        await _close(backend, agent)


@pytest.mark.parametrize('change', ['signal', 'agent', 'backend', 'runtime_scope'])
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
        elif change == 'runtime_scope':
            monkeypatch.setattr(c.agent, '_execution_custody',
                ExecutionCustody(_OriginalWork(backend_type=c.backend.backend_type)), raising=False)
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
        if change == 'runtime_scope':
            monkeypatch.delattr(c.agent, '_execution_custody')
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


@pytest.mark.parametrize('boundary', ['changed_epoch', 'inflight_transition', 'before_binding'])
async def test_original_journal_respects_original_privacy_epoch_and_gate(dispatcher_components, monkeypatch, boundary):
    from kestrel_sovereign.storage.privacy_wrapper import DurablePersistenceGate
    from kestrel_sovereign.execution_custody import is_execution_control_error
    c = dispatcher_components
    gate = DurablePersistenceGate()
    class PrivacyAgent(type(c.agent)):
        def _get_durable_persistence_gate(self):
            return self._test_privacy_gate
        def _get_privacy_policy_epoch(self):
            return self._test_privacy_epoch
    c.agent.__class__ = PrivacyAgent
    c.agent._test_privacy_gate = gate
    c.agent._test_privacy_epoch = 'original-policy'
    entered, release, transition_started, transition_entered = (asyncio.Event() for _ in range(4))
    original = c.backend._append_original_signal_outcome
    task = transition = None

    async def held_native(journal):
        entered.set()
        await release.wait()
        return await original(journal)

    async def change_policy():
        transition_started.set()
        async with gate.exclusive():
            transition_entered.set()
            c.agent._test_privacy_epoch = 'replacement-policy'

    monkeypatch.setattr(c.backend, '_append_original_signal_outcome', held_native)
    if boundary == 'before_binding':
        from kestrel_sdk.signals import RedactionPolicy
        def supersede_during_projection(payload):
            c.agent._test_privacy_epoch = 'replacement-policy'
            return '<redacted>'
        registration = _action_reg('privacy.epoch.audit')
        registration.log_redaction = RedactionPolicy(summarize=supersede_during_projection)
    else:
        registration = _action_reg('privacy.epoch.audit')
    c.registry.register(registration)
    signal = _signal('privacy.epoch.audit')
    try:
        task = asyncio.create_task(c.dispatcher.dispatch_signal(signal))
        if boundary == 'before_binding':
            from kestrel_sovereign.signals.outcome_custody import SignalOutcomeJournalPrivacyRefusal
            from kestrel_sovereign.execution_custody import _explicit_control_chain, execution_native_source_error
            outcome = await await_owned_task(task)
            refusals = [error for error in _explicit_control_chain(outcome.error)
                        if isinstance(error, SignalOutcomeJournalPrivacyRefusal)]
            assert len(refusals) == 1 and refusals[0].reason == 'superseded'
            assert refusals[0].signal_result.status.value == 'ok'
            assert execution_native_source_error(outcome.error) is None
            assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log') == 0
            assert c.dispatcher._log_write_failures == 1
            assert not c.dispatcher._outcome_log_tasks
            return
        await asyncio.wait_for(entered.wait(), 2)
        if boundary == 'changed_epoch':
            # Direct epoch replacement also covers a policy invalidation
            # independent of a queued transition. Never grants new work.
            c.agent._test_privacy_epoch = 'replacement-policy'
            release.set()
            outcome = await await_owned_task(task)
            assert is_execution_control_error(outcome.error)
            assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log') == 0
        else:
            transition = asyncio.create_task(change_policy())
            await transition_started.wait()
            assert not transition_entered.is_set(), 'privacy transition crossed an original native journal'
            release.set()
            outcome = await await_owned_task(task)
            assert outcome.error is None
            await transition
            assert transition_entered.is_set()
            assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log') == 1
    finally:
        release.set()
        if task is not None:
            await await_owned_task(task)
        if transition is not None:
            await transition
        try:
            await c.dispatcher.shutdown_durable_delivery()
        except BaseException as error:
            assert is_execution_control_error(error)
