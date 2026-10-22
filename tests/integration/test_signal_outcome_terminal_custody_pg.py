"""Same fixed native signal journal contracts on isolated real PostgreSQL."""
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
)

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]
native_pg = _native_pg_fixture


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
