"""Fixed native journal close cannot confer ordinary work or close a successor."""

from dataclasses import replace
import asyncio

import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCustody, ProcessRuntimeExecutionFence,
    bind_execution_cleanup, execution_evidence_carriers, is_execution_control_error,
)
from kestrel_sdk.resources import ResourceConflict
from kestrel_sovereign.tool_effect_store import ToolEffectAttemptSuperseded
from tests.unit.test_tool_effect_execution import admit, owner as owner


pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(30)]


async def test_cleanup_closes_only_original_attempt_and_cannot_admit_fresh_work(owner):
    turn = await admit(owner)
    scope = ExecutionCustody(ProcessRuntimeExecutionFence('owned-terminal-boundary'))
    with bind_execution_cleanup(owner, (scope,)):
        with pytest.raises(ExecutionAuthorityError, match='cleanup-only'):
            await owner.tool_effect_store.database.execute('CREATE TABLE must_not_exist(value INTEGER)')
        await turn.close()
        with pytest.raises(ExecutionAuthorityError, match='cleanup-only'):
            await owner.tool_effect_store.claim(turn.attempt, registration='forbidden', arguments={})
    assert await owner.tool_effect_store.database.fetchone('SELECT state FROM tool_effect_operations') == ('CLOSED',)
    assert await owner.tool_effect_store.database.fetchone('SELECT COUNT(*) FROM tool_effect_claims') == (0,)
    assert await owner.tool_effect_store.database.fetchone("SELECT COUNT(*) FROM sqlite_master WHERE name='must_not_exist'") == (0,)


async def test_original_terminal_cannot_close_a_superseding_attempt(owner):
    original = await admit(owner)
    await original.close()
    successor = await admit(owner)
    await owner.tool_effect_store.finish(original.attempt)
    assert await owner.tool_effect_store.database.fetchone('SELECT attempt_id,state FROM tool_effect_operations') == (successor.attempt.attempt_id, 'OPEN')
    await successor.close()


async def test_terminal_cannot_borrow_a_reconnected_backing_for_an_old_attempt(owner):
    turn = await admit(owner)
    backend = owner.tool_effect_store.database.backend
    original = backend._connection
    await backend.close()
    await backend.connect()
    assert backend._connection is not original
    with pytest.raises(ExecutionAuthorityError, match='original|retired|backing'):
        await turn.close()
    assert await owner.tool_effect_store.database.fetchone(
        'SELECT state FROM tool_effect_operations') == ('OPEN',)


async def test_reconnected_backing_cannot_admit_a_claim_for_the_original_attempt(owner):
    turn = await admit(owner)
    backend = owner.tool_effect_store.backend
    await backend.close()
    await backend.connect()
    with pytest.raises(ExecutionAuthorityError):
        await owner.tool_effect_store.claim(turn.attempt, registration='retired-attempt', arguments={})
    assert await owner.tool_effect_store.database.fetchone('SELECT COUNT(*) FROM tool_effect_claims') == (0,)


async def test_backend_close_fences_new_terminal_writer_admission_before_inventory_snapshot(owner, monkeypatch):
    first, late = await admit(owner, key='first-original'), await admit(owner, key='late-original')
    backend = owner.tool_effect_store.backend
    native_proof = backend._verify_original_signal_outcome
    entered, resume = asyncio.Event(), asyncio.Event()

    async def hold_first_original(connection, journal):
        await native_proof(connection, journal)
        if journal is first.attempt._terminal:
            entered.set()
            await resume.wait()

    monkeypatch.setattr(backend, '_verify_original_signal_outcome', hold_first_original)
    finishing = asyncio.create_task(owner.tool_effect_store.finish(first.attempt))
    closing = late_finish = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        closing = asyncio.create_task(backend.close())
        await asyncio.sleep(0.02)
        assert backend._original_results_closing and not closing.done()
        late_finish = asyncio.create_task(owner.tool_effect_store.finish(late.attempt))
        await asyncio.sleep(0.02)
        assert late.attempt._terminal._writer is None, 'A terminal writer entered after its inventory closed'
        assert late.attempt._terminal not in backend._original_result_holds
        with pytest.raises(ExecutionAuthorityError):
            await late_finish
        resume.set()
        await finishing
        await closing
        assert not backend._original_result_holds
    finally:
        resume.set()
        await asyncio.gather(finishing, *([late_finish] if late_finish else []),
            *([closing] if closing else []), return_exceptions=True)


@pytest.mark.parametrize('field,value', [
    ('scope_digest', 'not-a-digest'), ('operation_digest', '0' * 63),
    ('attempt_id', 'not-an-attempt'), ('attempt_id', None),
])
async def test_fixed_terminal_rejects_malformed_identities_before_sql(owner, field, value):
    turn = await admit(owner)
    with pytest.raises((ResourceConflict, ExecutionAuthorityError)):
        await owner.tool_effect_store.finish(replace(turn.attempt, **{field: value}))
    assert await owner.tool_effect_store.database.fetchone('SELECT state FROM tool_effect_operations') == ('OPEN',)
    await turn.close()


async def test_terminal_refuses_trigger_program_before_any_native_effect(owner):
    turn = await admit(owner)
    db = owner.tool_effect_store.database
    await db.execute('CREATE TABLE forbidden_terminal_effect(value INTEGER)')
    await db.execute('CREATE TRIGGER forbidden_terminal_trigger BEFORE UPDATE ON tool_effect_operations BEGIN INSERT INTO forbidden_terminal_effect VALUES(1); END')
    with pytest.raises(BaseExceptionGroup) as refused:
        await turn.close()
    assert is_execution_control_error(refused.value)
    assert any(isinstance(error, ExecutionAuthorityError) and 'destination_changed' in str(error)
        for error in execution_evidence_carriers(refused.value))
    assert await db.fetchone('SELECT COUNT(*) FROM forbidden_terminal_effect') == (0,)
    assert await db.fetchone('SELECT state FROM tool_effect_operations') == ('OPEN',)


async def test_terminal_refuses_an_enclosing_native_transaction(owner):
    turn = await admit(owner)
    async with owner.tool_effect_store.database.backend.transaction():
        with pytest.raises(ResourceConflict, match='own durable transaction'):
            await owner.tool_effect_store.finish(turn.attempt)
    await turn.close()


async def test_copied_valid_attempt_has_no_original_terminal_custody(owner):
    turn = await admit(owner)
    with pytest.raises(ExecutionAuthorityError):
        await owner.tool_effect_store.finish(replace(turn.attempt))
    assert await owner.tool_effect_store.database.fetchone('SELECT state FROM tool_effect_operations') == ('OPEN',)
    await turn.close()


@pytest.mark.parametrize('timeout', [0, -1, True, float('inf'), float('nan'), '5'])
async def test_terminal_timeout_configuration_requires_a_finite_positive_number(owner, timeout):
    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from kestrel_sovereign.tool_effect_store import ToolEffectStore

    with pytest.raises(ValueError, match='finite and positive'):
        ToolEffectStore(owner.tool_effect_store.database,
            scope=NativeRecipientScope('did:test:owned-timeout', 'tenant'),
            terminal_timeout_seconds=timeout)


async def test_terminal_timeout_captures_the_original_configured_budget(owner, monkeypatch):
    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from kestrel_sovereign.tool_effect_store import ToolEffectStore

    monkeypatch.setenv('KESTREL_TOOL_EFFECT_TERMINAL_TIMEOUT_S', '0.25')
    configured = ToolEffectStore(owner.tool_effect_store.database,
        scope=NativeRecipientScope('did:test:owned-timeout', 'tenant'))
    assert configured.terminal_timeout_seconds == 0.25
    owner.tool_effect_store.terminal_timeout_seconds = configured.terminal_timeout_seconds
    turn = await admit(owner)
    owner.tool_effect_store.terminal_timeout_seconds = 60
    assert turn.attempt._terminal._registration.timeout == 0.25
    await turn.close()


async def test_terminal_refuses_replaced_key_before_any_native_statement(owner):
    turn = await admit(owner)
    store = owner.tool_effect_store
    original = store._root
    store._root = b'changed-original-root' + bytes(11)
    try:
        with pytest.raises(ExecutionAuthorityError):
            await store.finish(turn.attempt)
        assert await store.database.fetchone('SELECT state FROM tool_effect_operations') == ('OPEN',)
    finally:
        store._root = original
    await turn.close()


async def test_terminal_factory_postaccept_failure_joins_unstarted_native_owner(owner):
    turn = await admit(owner)
    loop = asyncio.get_running_loop()
    original_factory = loop.get_task_factory()
    failure = RuntimeError('owned native finish publication refused')
    accepted = []

    def refuse_after_acceptance(loop, coroutine, **kwargs):
        accepted.append(asyncio.Task(coroutine, loop=loop, **kwargs))
        raise failure

    loop.set_task_factory(refuse_after_acceptance)
    try:
        with pytest.raises(RuntimeError) as denied:
            await owner.tool_effect_store.finish(turn.attempt)
        assert any(error is failure for error in execution_evidence_carriers(denied.value))
    finally:
        loop.set_task_factory(original_factory)
    assert accepted and all(task.done() for task in accepted)
    assert not owner.tool_effect_store.backend._original_result_holds
    assert await owner.tool_effect_store.database.fetchone('SELECT state FROM tool_effect_operations') == ('OPEN',)


async def test_terminal_observer_cancel_and_backend_close_join_actual_native_commit(owner, monkeypatch):
    from kestrel_sovereign.execution_custody import execution_commit_outcome

    turn = await admit(owner)
    backend = owner.tool_effect_store.backend
    original_connection = backend._connection
    native_proof = backend._verify_original_signal_outcome
    entered, resume = asyncio.Event(), asyncio.Event()

    async def hold_before_commit(connection, journal):
        await native_proof(connection, journal)
        assert connection is original_connection
        entered.set()
        await resume.wait()

    monkeypatch.setattr(backend, '_verify_original_signal_outcome', hold_before_commit)
    finishing = asyncio.create_task(owner.tool_effect_store.finish(turn.attempt))
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        finishing.cancel('owned terminal observer withdrawal')
        closing = asyncio.create_task(backend.close())
        await asyncio.sleep(0.02)
        assert not closing.done()
        assert backend._connection is original_connection
        assert backend._original_result_holds
        assert not turn.attempt._terminal._writer.done()
        resume.set()
        with pytest.raises(BaseException) as observed:
            await finishing
        assert execution_commit_outcome(observed.value) == 'committed'
        assert any(isinstance(error, asyncio.CancelledError)
            and str(error) == 'owned terminal observer withdrawal'
            for error in execution_evidence_carriers(observed.value))
        await closing
        assert not backend._original_result_holds
        await backend.connect()
        assert await owner.tool_effect_store.database.fetchone('SELECT state FROM tool_effect_operations') == ('CLOSED',)
    finally:
        resume.set()
        await asyncio.gather(finishing, *([closing] if closing is not None else []), return_exceptions=True)
