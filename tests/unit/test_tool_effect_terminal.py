"""Fixed native journal close cannot confer ordinary work or close a successor."""

from dataclasses import replace

import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCustody, ProcessRuntimeExecutionFence,
    bind_execution_cleanup,
)
from kestrel_sovereign.storage.db.tool_effect_terminal import finish_tool_effect_attempt
from kestrel_sdk.resources import ResourceConflict
from kestrel_sovereign.tool_effect_store import ToolEffectAttemptSuperseded
from tests.unit.test_tool_effect_execution import admit, owner


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
    successor = await admit(owner)
    with pytest.raises(ToolEffectAttemptSuperseded):
        await owner.tool_effect_store.finish(original.attempt)
    assert await owner.tool_effect_store.database.fetchone('SELECT attempt_id,state FROM tool_effect_operations') == (successor.attempt.attempt_id, 'OPEN')
    await successor.close()


@pytest.mark.parametrize('field,value', [
    ('scope_digest', 'not-a-digest'), ('operation_digest', '0' * 63),
    ('attempt_id', 'not-an-attempt'), ('attempt_id', None),
])
async def test_fixed_terminal_rejects_malformed_identities_before_sql(owner, field, value):
    turn = await admit(owner)
    with pytest.raises(ValueError):
        await finish_tool_effect_attempt(owner.tool_effect_store.database.backend,
            replace(turn.attempt, **{field: value}))
    assert await owner.tool_effect_store.database.fetchone('SELECT state FROM tool_effect_operations') == ('OPEN',)
    await turn.close()


async def test_terminal_refuses_trigger_program_before_any_native_effect(owner):
    turn = await admit(owner)
    db = owner.tool_effect_store.database
    await db.execute('CREATE TABLE forbidden_terminal_effect(value INTEGER)')
    await db.execute('CREATE TRIGGER forbidden_terminal_trigger BEFORE UPDATE ON tool_effect_operations BEGIN INSERT INTO forbidden_terminal_effect VALUES(1); END')
    with pytest.raises(ExecutionAuthorityError, match='trigger programs'):
        await turn.close()
    assert await db.fetchone('SELECT COUNT(*) FROM forbidden_terminal_effect') == (0,)
    assert await db.fetchone('SELECT state FROM tool_effect_operations') == ('OPEN',)


async def test_terminal_refuses_an_enclosing_native_transaction(owner):
    turn = await admit(owner)
    async with owner.tool_effect_store.database.backend.transaction():
        with pytest.raises(ResourceConflict, match='own durable transaction'):
            await owner.tool_effect_store.finish(turn.attempt)
    await turn.close()
