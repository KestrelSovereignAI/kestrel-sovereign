"""Same fixed native signal journal contracts on isolated real PostgreSQL."""
import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sovereign.signals import OrderedLockManager, SignalDispatcher, SignalLogStore, SourceRegistry
from tests.integration.test_execution_authority_postgres import native_pg as _native_pg_fixture
from tests.unit.test_signals_dispatcher import _FakeAgent
from tests.unit.test_signal_outcome_terminal_custody import (  # noqa: F401
    test_original_native_outcome_persists_after_work_cutoff_without_fresh_work,
    test_unadmitted_or_forged_native_journal_never_grants_storage,
    test_handler_cannot_use_journal_permission_or_replay_after_success,
    test_original_journal_refuses_substituted_identity,
    test_handler_cannot_mint_unrelated_journal_and_self_attach,
    test_original_journal_destination_cannot_follow_mutable_table,
    test_projection_control_failure_is_not_a_placeholder,
    test_ordinary_projection_loss_does_not_relabel_completed_effect,
    test_rejected_completion_callback_physically_joins_original_writer,
    test_handler_cannot_terminalize_through_scheduler_after_cutoff,
    test_visible_journal_does_not_admit_ui_work_after_cutoff,
    test_accepted_payload_is_not_replaced_after_cutoff,
    test_owner_withdrawal_retains_actual_original_journal_commit,
    test_terminal_snapshot_matches_canonical_handler_envelope,
    test_inner_writer_observation_retains_commit_and_original_error,
    test_original_journal_respects_original_privacy_epoch_and_gate,
    test_original_journal_completes_inside_original_exclusive_privacy_hold,
    test_original_journal_gate_admission_never_waits_on_a_transition,
    test_unstarted_journal_writer_loss_is_counted_once_without_relabelling_effect,
    test_actual_committed_journal_observation_is_not_a_projection_placeholder,
    test_projection_cannot_reauthorize_fabricated_result_through_finalizer,
    test_error_after_original_writer_acceptance_still_joins_before_owner_exits,
    test_projection_callbacks_cannot_mutate_canonical_data,
    test_original_journal_verifies_exact_row_before_native_commit,
    test_postcommit_privacy_observation_does_not_poison_healthy_dispatch,
    test_projection_retained_alias_cannot_rewrite_actual_result,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]
native_pg = _native_pg_fixture


async def _fresh_journal_pool(original, **options):
    import asyncpg
    options.setdefault('connection_class', original._connection_class)
    return await asyncpg.create_pool(*original._connect_args, **original._connect_kwargs, **options)


async def test_postgres_program_configuration_stays_fenced_through_actual_reset(dispatcher_components, native_pg, monkeypatch):
    from kestrel_sovereign._async_ownership import create_owned_task, await_owned_task
    from kestrel_sovereign.storage.db.native_journal_programs import NativeJournalPostgresConnection, NativeJournalConfigurationError
    from tests.unit.test_signals_dispatcher import _action_reg, _signal
    c = dispatcher_components
    entered, release = asyncio.Event(), asyncio.Event()
    original_reset = NativeJournalPostgresConnection.reset
    native = []
    async def held_reset(connection, *args, **kwargs):
        await original_reset(connection, *args, **kwargs)
        if c.dispatcher._original_outcome_writer.get() is not None:
            native.append(connection)
            entered.set()
            await release.wait()
    monkeypatch.setattr(NativeJournalPostgresConnection, 'reset', held_reset)
    c.registry.register(_action_reg('actual.reset.program.fence'))
    task = create_owned_task(c.dispatcher.dispatch_signal(_signal('actual.reset.program.fence')))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        with pytest.raises(NativeJournalConfigurationError):
            await native[0].set_type_codec('text', schema='pg_catalog', encoder=str, decoder=str, format='text')
    finally:
        release.set()
        outcome = await await_owned_task(task)
    assert outcome.error is None and outcome.result.action_result == {'received': {}}
    assert native[0]._journal_program_seal is None
    assert native[0]._original_journal_programs_supported()
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log') == 1


async def test_postgres_getter_cannot_unseal_before_original_reset_owner_completes(dispatcher_components):
    from tests.unit.test_signals_dispatcher import _action_reg, _signal
    c = dispatcher_components
    attempted = []
    class ObservingAgent(type(c.agent)):
        @property
        def did(self):
            journal = c.dispatcher._original_outcome_writer.get()
            native = journal._postgres_session if journal is not None else None
            if native is not None and native._journal_program_seal is not None:
                _, native_state = native._journal_program_seal
                if native_state.retirement.owner is None:
                    attempted.append(native._end_original_journal())
            return self._did
    c.agent.__class__ = ObservingAgent
    c.registry.register(_action_reg('early.unseal.getter'))
    result = await c.dispatcher.dispatch_signal(_signal('early.unseal.getter'))
    assert result.action_result == {'received': {}}
    assert attempted and not any(attempted), 'getter released program fence before actual native reset existed'
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log') == 1


@pytest.mark.parametrize('installed', ['before', 'handler'])
async def test_postgres_journal_refuses_custom_connection_codecs(dispatcher_components, native_pg, installed):
    import asyncpg
    from kestrel_sovereign.execution_custody import ExecutionCustody, bind_execution_runtime, is_execution_control_error
    from tests.unit.test_signal_outcome_terminal_custody import _OriginalWork
    from tests.unit.test_signals_dispatcher import _action_reg, _signal
    c = dispatcher_components
    _, control, schema = native_pg
    original_pool = c.backend._pool
    pool = await _fresh_journal_pool(original_pool, min_size=0, max_size=1)
    c.backend._pool = pool
    effects, substituted = [], []
    def encode(value):
        if value == 'original-summary':
            substituted.append(True)
            return 'substituted-summary'
        return value
    async def configure():
        async with pool.acquire() as connection:
            await connection.set_type_codec('text', schema='pg_catalog', encoder=encode, decoder=str, format='text')
    scope = ExecutionCustody(_OriginalWork(backend_type='postgres'))
    async def handler(payload):
        effects.append(True)
        if installed == 'handler':
            await configure()
        scope.revoke('actual cutoff after native connection codec configuration')
        return {'genuine': True}
    registration = _action_reg('connection.codec.audit', handler=handler)
    registration.result_summary = lambda result: 'original-summary'
    c.registry.register(registration)
    error = None
    try:
        if installed == 'before':
            await configure()
        with bind_execution_runtime(SimpleNamespace(_execution_custody=scope)):
            try:
                await c.dispatcher.dispatch_signal(_signal('connection.codec.audit'))
            except BaseException as caught:
                error = caught
        assert not substituted, 'application codec substituted INSERT and exact proof together'
        assert is_execution_control_error(error)
        assert len(effects) == (installed == 'handler')
        assert await control.fetchval(f'SELECT COUNT(*) FROM "{schema}".signal_log') == 0
    finally:
        c.backend._pool = original_pool
        async with asyncio.timeout(10):
            await pool.close()


async def test_postgres_journal_refuses_noncanonical_column_type_and_typmod(dispatcher_components, native_pg):
    from kestrel_sovereign.execution_custody import ExecutionCustody, bind_execution_runtime, is_execution_control_error
    from tests.unit.test_signal_outcome_terminal_custody import _OriginalWork
    from tests.unit.test_signals_dispatcher import _action_reg, _signal
    c = dispatcher_components
    _, control, schema = native_pg
    scope = ExecutionCustody(_OriginalWork(backend_type='postgres'))
    effects = []
    async def handler(payload):
        await c.backend.execute('ALTER TABLE signal_log ALTER COLUMN result_summary TYPE character(2)')
        effects.append(True)
        scope.revoke('actual cutoff after native text coercion configured')
        return {'genuine': True}
    registration = _action_reg('column.coercion.audit', handler=handler)
    registration.result_summary = lambda result: 'ok '
    c.registry.register(registration)
    error = None
    with bind_execution_runtime(SimpleNamespace(_execution_custody=scope)):
        try:
            await c.dispatcher.dispatch_signal(_signal('column.coercion.audit'))
        except BaseException as caught:
            error = caught
    assert await control.fetchval(f'SELECT COUNT(*) FROM "{schema}".signal_log') == 0, 'column coercion committed a different value while equality certified it'
    assert effects == [True] and is_execution_control_error(error)
    assert c.dispatcher._log_write_failures == 1


@pytest.mark.parametrize('callback_kind', ['setup', 'reset', 'init', 'connect', 'adapter', 'connection_class', 'record_class'])
async def test_postgres_journal_refuses_application_pool_programs(dispatcher_components, native_pg, callback_kind):
    import asyncpg
    from contextlib import asynccontextmanager
    from kestrel_sovereign.storage.db.postgres import PostgresPoolAdapter
    from kestrel_sovereign.execution_custody import ExecutionCustody, bind_execution_runtime, is_execution_control_error
    from tests.unit.test_signal_outcome_terminal_custody import _OriginalWork
    from tests.unit.test_signals_dispatcher import _action_reg, _signal
    c = dispatcher_components
    _, control, schema = native_pg
    await c.backend.execute('CREATE TABLE forbidden_pool_effect(id TEXT)')
    signal = _signal('pool.application.program')
    armed, effects = [], []
    async def program(connection):
        if armed:
            if callback_kind != 'reset':
                await connection.execute(f'INSERT INTO "{schema}".forbidden_pool_effect VALUES($1)', signal.id)
            else:
                await connection.execute(f'DELETE FROM "{schema}".signal_log WHERE id=$1', signal.id)
    original_pool = c.backend._pool
    async def connect(*args, **kwargs):
        connection = await asyncpg.connect(*args, **kwargs)
        await program(connection)
        return connection
    class CustomConnection(asyncpg.Connection):
        pass
    class CustomRecord(asyncpg.Record):
        pass
    @asynccontextmanager
    async def wrapper(checkout):
        async with checkout as connection:
            await program(connection)
            yield connection
    custom = ({callback_kind: program} if callback_kind in ('setup', 'reset', 'init') else
              {'connect': connect} if callback_kind == 'connect' else
              {'connection_class': CustomConnection} if callback_kind == 'connection_class' else
              {'record_class': CustomRecord} if callback_kind == 'record_class' else {})
    pool = await _fresh_journal_pool(original_pool, min_size=0, max_size=2, **custom)
    c.backend._pool = PostgresPoolAdapter(pool, wrapper) if callback_kind == 'adapter' else pool
    scope = ExecutionCustody(_OriginalWork(backend_type='postgres'))
    async def handler(payload):
        effects.append('genuine handler completion')
        armed.append(True)
        scope.revoke('actual cutoff before pool application callback')
        return {'genuine': True}
    c.registry.register(_action_reg('pool.application.program', handler=handler))
    error = None
    try:
        with bind_execution_runtime(SimpleNamespace(_execution_custody=scope)):
            try:
                await c.dispatcher.dispatch_signal(signal)
            except BaseException as caught:
                error = caught
        assert await control.fetchval(f'SELECT COUNT(*) FROM "{schema}".forbidden_pool_effect') == 0, 'revoked terminal checkout executed unrelated application SQL'
        if effects:
            assert await control.fetchval(f'SELECT COUNT(*) FROM "{schema}".signal_log WHERE id=$1', signal.id) == 1, 'application reset deleted the acknowledged journal row'
        assert not effects and is_execution_control_error(error)
    finally:
        armed.clear()
        c.backend._pool = original_pool
        async with asyncio.timeout(10):
            await pool.close()


async def test_postgres_original_journal_proof_ignores_shadowed_equality(dispatcher_components, native_pg):
    import asyncpg
    from kestrel_sovereign.execution_custody import ExecutionCustody, bind_execution_runtime, is_execution_control_error, execution_commit_outcome
    from tests.unit.test_signal_outcome_terminal_custody import _OriginalWork
    from tests.unit.test_signals_dispatcher import _action_reg, _signal
    c = dispatcher_components
    _, control, schema = native_pg
    original_pool = c.backend._pool
    kwargs = dict(original_pool._connect_kwargs)
    kwargs['server_settings'] = {**kwargs.get('server_settings', {}), 'search_path': f'"{schema}", pg_catalog'}
    pool = await asyncpg.create_pool(*original_pool._connect_args,
        **kwargs, min_size=0, max_size=2, connection_class=original_pool._connection_class)
    c.backend._pool = pool
    scope = ExecutionCustody(_OriginalWork(backend_type='postgres'))
    signal = _signal('operator.shadow.audit')
    async def handler(payload):
        await c.backend.execute(f'''CREATE FUNCTION "{schema}".journal_equal(l TEXT, r TEXT) RETURNS BOOLEAN LANGUAGE plpgsql VOLATILE AS $$BEGIN
            IF l OPERATOR(pg_catalog.=) '{signal.id}' AND l OPERATOR(pg_catalog.=) r THEN
                DELETE FROM "{schema}".signal_log WHERE id OPERATOR(pg_catalog.=) l;
            END IF;
            RETURN l OPERATOR(pg_catalog.=) r; END$$''')
        await c.backend.execute(f'CREATE OPERATOR "{schema}".= (LEFTARG=TEXT, RIGHTARG=TEXT, PROCEDURE="{schema}".journal_equal)')
        scope.revoke('actual cutoff after work installed shadow operator')
        return {'genuine': True}
    c.registry.register(_action_reg('operator.shadow.audit', handler=handler))
    try:
        with bind_execution_runtime(SimpleNamespace(_execution_custody=scope)):
            result = await c.dispatcher.dispatch_signal(signal)
        assert result.action_result == {'genuine': True}
        assert await control.fetchval(f'SELECT COUNT(*) FROM "{schema}".signal_log WHERE id OPERATOR(pg_catalog.=) $1', signal.id) == 1, 'shadow operator deleted the original row during its proof'
        assert c.dispatcher._log_write_failures == 0
        assert c.dispatcher._outcome_log_control_error is None
    finally:
        c.backend._pool = original_pool
        async with asyncio.timeout(10):
            await pool.close()


@pytest.mark.parametrize('shape', ['rule', 'row_security', 'check', 'extra_column', 'domain', 'generated', 'identity', 'expression_index', 'predicate_index'])
async def test_postgres_original_journal_refuses_custom_shape_before_routing(dispatcher_components, native_pg, shape):
    from kestrel_sovereign.execution_custody import is_execution_control_error
    from tests.unit.test_signals_dispatcher import _action_reg, _signal
    c = dispatcher_components
    _, control, schema = native_pg
    table = f'"{schema}".signal_log'
    if shape == 'rule':
        await c.backend.execute(f'CREATE RULE journal_noop AS ON INSERT TO {table} DO INSTEAD NOTHING')
    elif shape == 'row_security':
        await c.backend.execute(f'ALTER TABLE {table} ENABLE ROW LEVEL SECURITY')
    elif shape == 'check':
        await c.backend.execute(f"ALTER TABLE {table} ADD CONSTRAINT journal_check CHECK(id<>'unsupported')")
    elif shape == 'extra_column':
        await c.backend.execute(f"ALTER TABLE {table} ADD COLUMN unexpected TEXT DEFAULT 'hidden'")
    elif shape == 'domain':
        await c.backend.execute(f'CREATE DOMAIN "{schema}".journal_text AS TEXT')
        await c.backend.execute(f'ALTER TABLE {table} ALTER COLUMN payload_redacted TYPE "{schema}".journal_text')
    elif shape == 'generated':
        await c.backend.execute(f'ALTER TABLE {table} DROP COLUMN payload_redacted')
        await c.backend.execute(f'ALTER TABLE {table} ADD COLUMN payload_redacted TEXT GENERATED ALWAYS AS (id) STORED')
    elif shape == 'expression_index':
        await c.backend.execute(f'CREATE INDEX journal_expression ON {table}(lower(id))')
    elif shape == 'predicate_index':
        await c.backend.execute(f"CREATE INDEX journal_predicate ON {table}(id) WHERE id<>'unsupported'")
    else:
        await c.backend.execute(f'ALTER TABLE {table} ALTER COLUMN duration_ms ADD GENERATED BY DEFAULT AS IDENTITY')
    effects = []
    async def handler(payload):
        effects.append('must not route with unsupported native audit shape')
        return {'genuine': True}
    c.registry.register(_action_reg('custom.native.audit', handler=handler))
    error = None
    try:
        await c.dispatcher.dispatch_signal(_signal('custom.native.audit'))
    except BaseException as caught:
        error = caught
    assert is_execution_control_error(error)
    assert not effects
    assert await control.fetchval(f'SELECT COUNT(*) FROM {table}') == 0
    assert not c.dispatcher._outcome_log_tasks


async def test_postgres_original_journal_rejects_unlogged_partition_leaf(dispatcher_components, native_pg):
    from kestrel_sovereign.execution_custody import is_execution_control_error, execution_commit_outcome
    from tests.unit.test_signals_dispatcher import _action_reg, _signal
    c = dispatcher_components
    _, control, schema = native_pg
    effects = []
    table = f'"{schema}".signal_log'
    await c.backend.execute(f'ALTER TABLE {table} RENAME TO original_signal_log')
    await c.backend.execute(f'CREATE TABLE {table} (LIKE "{schema}".original_signal_log INCLUDING ALL) PARTITION BY HASH(id)')
    await c.backend.execute(f'CREATE UNLOGGED TABLE "{schema}".signal_log_leaf PARTITION OF {table} FOR VALUES WITH (MODULUS 1, REMAINDER 0)')
    assert await control.fetchval("SELECT relpersistence FROM pg_class WHERE oid=$1::regclass", table) == b'p'
    assert await control.fetchval("SELECT relpersistence FROM pg_class WHERE oid=$1::regclass", f'"{schema}".signal_log_leaf') == b'u'
    async def handler(payload):
        effects.append('must not execute against routed nondurable journal')
        return {'genuine': True}
    c.registry.register(_action_reg('partitioned.audit', handler=handler))
    error = None
    try:
        await c.dispatcher.dispatch_signal(_signal('partitioned.audit'))
    except BaseException as caught:
        error = caught
    assert is_execution_control_error(error), 'unlogged physical destination was admitted through its logged parent'
    assert execution_commit_outcome(error) != 'committed'
    assert not effects
    assert await control.fetchval(f'SELECT COUNT(*) FROM {table}') == 0
    assert not c.dispatcher._outcome_log_tasks


async def test_postgres_original_journal_refuses_redeferred_trigger_mutation(dispatcher_components, native_pg):
    from kestrel_sovereign.execution_custody import ExecutionCustody, bind_execution_runtime, is_execution_control_error, execution_commit_outcome, execution_native_source_error, _explicit_control_chain
    from tests.unit.test_signal_outcome_terminal_custody import _OriginalWork
    from tests.unit.test_signals_dispatcher import _action_reg, _signal
    c = dispatcher_components
    _, control, schema = native_pg
    table = f'"{schema}".signal_log'
    queue = f'"{schema}".journal_delete_queue'
    effect = f'"{schema}".journal_handler_effect'
    await c.backend.execute(f'CREATE TABLE {effect}(id INTEGER PRIMARY KEY)')
    effects = []
    scope = ExecutionCustody(_OriginalWork(backend_type='postgres'))
    async def handler(payload):
        effects.append(True)
        await c.backend.execute(f'INSERT INTO {effect} VALUES(?)', (len(effects),))
        if len(effects) == 1:
            await c.backend.execute(f'CREATE TABLE {queue}(id TEXT PRIMARY KEY)')
            await c.backend.execute(f'''CREATE FUNCTION "{schema}".journal_delete() RETURNS trigger LANGUAGE plpgsql AS $$BEGIN
                DELETE FROM {table} WHERE id=NEW.id; RETURN NULL; END$$''')
            await c.backend.execute(f'''CREATE CONSTRAINT TRIGGER queued_delete AFTER INSERT ON {queue}
                DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION "{schema}".journal_delete()''')
            await c.backend.execute(f'''CREATE FUNCTION "{schema}".journal_queue_delete() RETURNS trigger LANGUAGE plpgsql AS $$BEGIN
                SET CONSTRAINTS "{schema}".queued_delete DEFERRED;
                INSERT INTO {queue}(id) VALUES(NEW.id); RETURN NULL; END$$''')
            await c.backend.execute(f'''CREATE CONSTRAINT TRIGGER journal_queue AFTER INSERT ON {table}
                DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION "{schema}".journal_queue_delete()''')
            scope.revoke('actual work cutoff after committed handler effect and trigger configuration')
        return {'genuine': True}
    c.registry.register(_action_reg('redeferred.audit', handler=handler))
    signal = _signal('redeferred.audit')
    error = None
    with bind_execution_runtime(SimpleNamespace(_execution_custody=scope)):
        try:
            await c.dispatcher.dispatch_signal(signal)
        except BaseException as caught:
            error = caught
    assert await control.fetchval(f'SELECT COUNT(*) FROM {table} WHERE id=$1', signal.id) == 0
    assert await control.fetchval(f'SELECT COUNT(*) FROM {effect}') == 1
    assert is_execution_control_error(error), 'absent terminal row was reported as successfully journaled'
    assert execution_commit_outcome(error) != 'committed', 're-deferred deletion escaped row proof and was acknowledged'
    assert any(getattr(item, 'signal_result', None) is not None for item in _explicit_control_chain(error))
    assert c.dispatcher._log_write_failures == 1
    assert execution_native_source_error(error) is None
    assert c.dispatcher._outcome_log_control_error is None
    await c.backend.execute(f'DROP TRIGGER journal_queue ON {table}')
    fresh = await c.dispatcher.dispatch_signal(_signal('redeferred.audit'))
    assert fresh.action_result == {'genuine': True}
    assert await control.fetchval(f'SELECT COUNT(*) FROM {effect}') == 2


@pytest.mark.parametrize('persistence', ['temporary', 'unlogged'])
async def test_postgres_original_journal_rejects_temporary_relation_at_admission(dispatcher_components, native_pg, persistence):
    import asyncpg
    from kestrel_sovereign.execution_custody import ExecutionCustody, bind_execution_runtime, is_execution_control_error, execution_commit_outcome
    from tests.unit.test_signal_outcome_terminal_custody import _OriginalWork
    from tests.unit.test_signals_dispatcher import _action_reg, _signal
    c = dispatcher_components
    _, control, schema = native_pg
    original_pool = c.backend._pool
    pool = await _fresh_journal_pool(original_pool, min_size=0, max_size=1)
    c.backend._pool = pool
    effects = []
    scope = ExecutionCustody(_OriginalWork(backend_type='postgres'))
    async def handler(payload):
        effects.append('must not execute against temporary audit destination')
        scope.revoke('actual work cutoff after admitted handler')
        return {'genuine': True}
    c.registry.register(_action_reg('temporary.native.audit', handler=handler))
    signal = _signal('temporary.native.audit')
    error = None
    try:
        with bind_execution_runtime(SimpleNamespace(_execution_custody=scope)):
            if persistence == 'temporary':
                await c.backend.execute(f'CREATE TEMP TABLE signal_log (LIKE "{schema}".signal_log INCLUDING ALL)')
            else:
                await c.backend.execute(f'ALTER TABLE "{schema}".signal_log SET UNLOGGED')
            try:
                await c.dispatcher.dispatch_signal(signal)
            except BaseException as caught:
                error = caught
        assert is_execution_control_error(error), 'temporary table was admitted as a durable audit destination'
        assert execution_commit_outcome(error) != 'committed'
        assert effects == []
        assert await control.fetchval(f'SELECT COUNT(*) FROM "{schema}".signal_log') == 0
        assert not c.dispatcher._outcome_log_tasks
    finally:
        c.backend._pool = original_pool
        async with asyncio.timeout(10):
            await pool.close()


async def test_postgres_original_journal_checks_deferred_trigger_effect_before_commit(dispatcher_components):
    await test_original_journal_verifies_exact_row_before_native_commit(dispatcher_components, 'deferred_delete')


@pytest.mark.timeout(15, method='thread')
@pytest.mark.parametrize('entry', ['dispatch', 'enqueue', 'durable_enqueue'])
@pytest.mark.parametrize('pool_size', [2, 1])
async def test_original_journal_refuses_enclosing_transaction_before_destination_checkout(dispatcher_components, entry, pool_size):
    import asyncpg
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError, _explicit_control_chain
    from tests.unit.test_signals_dispatcher import _action_reg, _signal
    c = dispatcher_components
    original_pool = c.backend._pool
    pool = await _fresh_journal_pool(original_pool, min_size=0, max_size=pool_size)
    c.backend._pool = pool
    effects = []
    async def handler(payload):
        effects.append('handler must not run inside an unsupported enclosing transaction')
        return {'genuine': True}
    c.registry.register(_action_reg('enclosing.admission.audit', handler=handler))
    error = None
    handle = None
    try:
        # Normal startup completes without holding the sole native pool slot.
        await c.dispatcher.initialize_durable_delivery()
        try:
            async with asyncio.timeout(3):
                async with c.backend.transaction():
                    signal = _signal('enclosing.admission.audit')
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
            from kestrel_sovereign._async_ownership import await_owned_task
            await await_owned_task(handle.task)
        assert error is not None
        assert any(isinstance(item, ExecutionAuthorityError) and 'enclosing transaction' in str(item)
                   for item in _explicit_control_chain(error)), 'destination capture waited for its own enclosing transaction pool slot'
        assert effects == []
        assert not c.dispatcher._outcome_log_tasks
    finally:
        c.backend._pool = original_pool
        async with asyncio.timeout(10):
            await pool.close()


@pytest.mark.parametrize('boundary', ['namespace', 'fake_ack', 'relation'])
async def test_original_journal_native_setup_cannot_redirect_or_forge_commit(dispatcher_components, native_pg, boundary):
    import asyncpg
    from kestrel_sovereign.execution_custody import (
        ExecutionCustody, bind_execution_runtime, execution_commit_outcome, is_execution_control_error,
    )
    from kestrel_sovereign._async_ownership import await_owned_task
    from tests.unit.test_signal_outcome_terminal_custody import _OriginalWork
    from tests.unit.test_signals_dispatcher import _action_reg, _signal
    c = dispatcher_components
    _, control, schema = native_pg
    alternate = schema + '_setup_replacement'
    await control.execute(f'CREATE SCHEMA "{alternate}"')
    await control.execute(f'CREATE TABLE "{alternate}".signal_log (LIKE "{schema}".signal_log INCLUDING ALL)')
    route = [schema]
    attempted = []
    async def setup(connection):
        await connection.execute(f'SET search_path TO "{route[0]}"')
        journal = c.dispatcher._original_outcome_writer.get()
        if boundary == 'fake_ack' and journal is not None:
            attempted.append(True)
            journal.acknowledged(c.backend)
    original_pool = c.backend._pool
    pool = await _fresh_journal_pool(original_pool, min_size=0, max_size=2, setup=setup)
    c.backend._pool = pool
    entered, release = asyncio.Event(), asyncio.Event()
    effects = []
    scope = ExecutionCustody(_OriginalWork(backend_type='postgres'))
    owner = SimpleNamespace(_execution_custody=scope)
    async def handler(payload):
        effects.append('genuine-effect')
        entered.set()
        return {'genuine': True}
    c.registry.register(_action_reg('setup.boundary.audit', handler=handler))
    signal = _signal('setup.boundary.audit')
    handle = None
    try:
        with bind_execution_runtime(owner):
            handle = await c.dispatcher.enqueue_signal(signal)
            outcome = await await_owned_task(handle.task)
        # Application setup is now unsupported for a privileged journal. It
        # must refuse BEFORE the handler, not merely inspect its later SQL.
        assert effects == [] and not entered.is_set() and not attempted
        original_rows = await control.fetchval(f'SELECT COUNT(*) FROM "{schema}".signal_log WHERE id=$1', signal.id)
        replacement_rows = await control.fetchval(f'SELECT COUNT(*) FROM "{alternate}".signal_log')
        assert is_execution_control_error(outcome.error)
        assert execution_commit_outcome(outcome.error) != 'committed'
        assert (original_rows, replacement_rows) == (0, 0)
        assert not c.dispatcher._outcome_log_tasks
    finally:
        release.set()
        if handle is not None:
            await await_owned_task(handle.task)
        c.backend._pool = original_pool
        async with asyncio.timeout(10):
            await pool.close()
        await control.execute(f'DROP SCHEMA "{alternate}" CASCADE')


@pytest.mark.parametrize('boundary', ['handler_generation', 'handler_configuration', 'checkout_both'])
async def test_original_journal_refuses_same_pool_reconfiguration(dispatcher_components, native_pg, monkeypatch, boundary):
    from contextlib import asynccontextmanager
    from kestrel_sovereign.execution_custody import ExecutionCustody, bind_execution_runtime, is_execution_control_error
    from tests.unit.test_signal_outcome_terminal_custody import _OriginalWork
    from tests.unit.test_signals_dispatcher import _action_reg, _signal
    c = dispatcher_components
    _, control, schema = native_pg
    native_pool = c.backend._pool
    original_args, original_kwargs = native_pool._connect_args, native_pool._connect_kwargs
    alternate = schema + '_replacement'
    await control.execute(f'CREATE SCHEMA "{alternate}"')
    await control.execute(f'CREATE TABLE "{alternate}".signal_log (LIKE "{schema}".signal_log INCLUDING ALL)')
    entered, release = asyncio.Event(), asyncio.Event()
    effects = []
    scope = ExecutionCustody(_OriginalWork(backend_type='postgres'))
    owner = SimpleNamespace(_execution_custody=scope)
    original_backend_pool = c.backend._pool
    original_checkout = c.backend._tracked_pool_checkout
    @asynccontextmanager
    async def held_checkout(*args, **kwargs):
        if 'checkout' in boundary and c.dispatcher._original_outcome_writer.get() is not None:
            entered.set()
            await release.wait()
        async with original_checkout(*args, **kwargs) as owned:
            yield owned
    async def handler(payload):
        effects.append('genuine-effect')
        if boundary.startswith('handler_'):
            entered.set()
            await release.wait()
        return {'genuine': True}
    monkeypatch.setattr(c.backend, '_tracked_pool_checkout', held_checkout)
    c.registry.register(_action_reg('pool.configuration.audit', handler=handler))
    signal = _signal('pool.configuration.audit')
    handle = None
    try:
        with bind_execution_runtime(owner):
            handle = await c.dispatcher.enqueue_signal(signal)
            await asyncio.wait_for(entered.wait(), 2)
            if boundary != 'handler_generation':
                settings = dict(original_kwargs.get('server_settings', {}))
                settings['search_path'] = alternate
                native_pool.set_connect_args(original_args[0], **{**original_kwargs, 'server_settings': settings})
            if boundary != 'handler_configuration':
                await native_pool.expire_connections()
            else:
                # Public physical retirement without a generation bump forces
                # the next connection to use the new public configuration.
                held = [await native_pool.acquire() for _ in range(native_pool.get_max_size())]
                for connection in held:
                    await connection.close()
                for connection in held:
                    await native_pool.release(connection)
            scope.revoke('original work cutoff after public pool rotation')
            release.set()
            from kestrel_sovereign._async_ownership import await_owned_task
            outcome = await await_owned_task(handle.task)
            assert is_execution_control_error(outcome.error), 'original journal used the same pool with replacement configuration'
        assert effects == ['genuine-effect']
        assert await control.fetchval(f'SELECT COUNT(*) FROM "{alternate}".signal_log') == 0
        assert await control.fetchval(f'SELECT COUNT(*) FROM "{schema}".signal_log WHERE id=$1', signal.id) == 0
        assert not c.dispatcher._outcome_log_tasks
    finally:
        release.set()
        if handle is not None:
            from kestrel_sovereign._async_ownership import await_owned_task
            await await_owned_task(handle.task)
        c.backend._pool = original_backend_pool
        monkeypatch.setattr(c.backend, '_tracked_pool_checkout', original_checkout)
        native_pool.set_connect_args(original_args[0], **original_kwargs)
        await native_pool.expire_connections()
        await control.execute(f'DROP SCHEMA "{alternate}" CASCADE')


@pytest.fixture
async def dispatcher_components(native_pg):
    backend, _, _ = native_pg
    store = SignalLogStore(backend)
    await store.initialize()
    registry = SourceRegistry()
    agent = _FakeAgent()
    dispatcher = SignalDispatcher(agent=agent, registry=registry, lock_manager=OrderedLockManager(), store=store)
    try:
        yield SimpleNamespace(backend=backend, store=store, registry=registry, agent=agent, dispatcher=dispatcher)
    finally:
        try:
            await dispatcher.shutdown_durable_delivery()
        except BaseException as error:
            from kestrel_sovereign.execution_custody import is_execution_control_error
            assert is_execution_control_error(error)


async def test_original_journal_keeps_acknowledged_commit_after_reset_identity_loss(
    dispatcher_components, monkeypatch,
):
    import asyncpg
    from kestrel_sovereign.execution_custody import execution_commit_outcome, is_execution_control_error
    from tests.unit.test_signals_dispatcher import _action_reg, _signal
    c = dispatcher_components
    pool = c.backend._pool
    original_release = asyncpg.pool.Pool.release
    original_id = c.agent._did
    reset_acknowledged = []

    async def changed_after_actual_reset(original_pool, connection, *args, **kwargs):
        result = await original_release(original_pool, connection, *args, **kwargs)
        journal = c.dispatcher._original_outcome_writer.get()
        if original_pool is pool and journal is not None and journal._phase == 'submitted':
            reset_acknowledged.append(True)
            c.agent._did = 'replacement-after-native-reset'
        return result

    c.registry.register(_action_reg('committed.audit'))
    signal = _signal('committed.audit')
    monkeypatch.setattr(asyncpg.pool.Pool, 'release', changed_after_actual_reset)
    try:
        with pytest.raises(BaseException) as caught:
            await c.dispatcher.dispatch_signal(signal)
        assert reset_acknowledged == [True]
        assert is_execution_control_error(caught.value)
        assert execution_commit_outcome(caught.value) == 'committed'
        assert c.dispatcher._log_write_failures == 0
    finally:
        c.agent._did = original_id
        monkeypatch.setattr(asyncpg.pool.Pool, 'release', original_release)
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log WHERE id=?', (signal.id,)) == 1


@pytest.mark.parametrize('scoped', [False, True])
async def test_original_journal_legacy_pool_compatibility(dispatcher_components, scoped):
    from contextlib import asynccontextmanager, nullcontext
    from kestrel_sovereign.execution_custody import (
        ExecutionAuthorityError, ExecutionCustody, bind_execution_runtime,
    )
    from tests.unit.test_signal_outcome_terminal_custody import _OriginalWork
    from tests.unit.test_signals_dispatcher import _action_reg, _signal
    c = dispatcher_components
    original_pool = c.backend._pool
    handlers = []

    class LegacyPool:
        def __getattr__(self, name):
            return getattr(original_pool, name)

        @asynccontextmanager
        async def acquire(self):
            async with original_pool.acquire() as connection:
                yield connection

    async def handler(payload):
        handlers.append(True)
        return {'genuine': True}

    c.registry.register(_action_reg('legacy.pool.audit', handler=handler))
    c.backend._pool = LegacyPool()
    signal = _signal('legacy.pool.audit')
    owner = SimpleNamespace(_execution_custody=ExecutionCustody(_OriginalWork(backend_type='postgres')))
    try:
        with bind_execution_runtime(owner) if scoped else nullcontext():
            if scoped:
                with pytest.raises(ExecutionAuthorityError, match='supported native backend|PostgresPoolAdapter'):
                    await c.dispatcher.dispatch_signal(signal)
            else:
                result = await c.dispatcher.dispatch_signal(signal)
                assert result.action_result == {'genuine': True}
    finally:
        c.backend._pool = original_pool
    assert handlers == ([] if scoped else [True])
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log WHERE id=?', (signal.id,)) == (0 if scoped else 1)


async def test_original_journal_reset_failure_retains_exact_commit_and_retirement(
    dispatcher_components, monkeypatch,
):
    import asyncpg
    from contextlib import asynccontextmanager
    from kestrel_sovereign._async_ownership import owned_task_error
    from kestrel_sovereign.execution_custody import _acknowledged_commits, execution_cleanup_debts
    from tests.unit.test_signals_dispatcher import _action_reg, _signal
    c = dispatcher_components
    pool = c.backend._pool
    original_release = asyncpg.pool.Pool.release
    tracked_checkout = c.backend._tracked_pool_checkout
    states = []
    reset_failure = OSError('original journal reset acknowledgment failed')

    @asynccontextmanager
    async def observe_checkout(*args, **kwargs):
        async with tracked_checkout(*args, **kwargs) as (connection, state):
            if c.dispatcher._original_outcome_writer.get() is not None:
                states.append(state)
            yield connection, state

    async def fail_original_reset(original_pool, connection, *args, **kwargs):
        result = await original_release(original_pool, connection, *args, **kwargs)
        journal = c.dispatcher._original_outcome_writer.get()
        if original_pool is pool and journal is not None and journal._phase == 'submitted':
            raise reset_failure
        return result

    c.registry.register(_action_reg('reset.failed.audit'))
    signal = _signal('reset.failed.audit')
    monkeypatch.setattr(c.backend, '_tracked_pool_checkout', observe_checkout)
    monkeypatch.setattr(asyncpg.pool.Pool, 'release', fail_original_reset)
    try:
        with pytest.raises(BaseException) as caught:
            await c.dispatcher.dispatch_signal(signal)
        assert len(states) == 1
        state = states[0]
        retained = _acknowledged_commits(caught.value)
        assert retained and state.acknowledgments
        assert all(any(actual is receipt for actual in retained) for receipt in state.acknowledgments)
        for receipt in retained:
            assert receipt.acknowledged
            assert receipt.retirement_owner is state.retirement.owner
            assert receipt.retirement_owner.done()
            assert owned_task_error(receipt.retirement_owner) is reset_failure
        assert any(owner is state.retirement.owner for _, owner in execution_cleanup_debts(caught.value))
    finally:
        monkeypatch.setattr(asyncpg.pool.Pool, 'release', original_release)
        monkeypatch.setattr(c.backend, '_tracked_pool_checkout', tracked_checkout)
    assert await c.backend.fetch_val('SELECT COUNT(*) FROM signal_log WHERE id=?', (signal.id,)) == 1
