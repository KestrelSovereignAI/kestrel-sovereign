"""Exact original physical result holds across native backend refinement."""
import asyncio
from dataclasses import replace

import pytest

from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ExecutionCustody, bind_execution_custody_snapshot,
    is_execution_control_error,
)
from kestrel_sovereign.visual_computer.dispatch import original_dispatch
from tests.integration.test_visual_custody_backends import custody as custody, SCOPE
from tests.integration.test_visual_original_result_backends import OriginalTurn, prepared


@pytest.mark.asyncio
async def test_copied_backing_descriptor_cannot_replace_original_result_hold(custody):
    store, _ = custody
    current, session, request = await prepared(store)
    turn = ExecutionCustody(OriginalTurn(store.db.backend_type))
    with bind_execution_custody_snapshot((turn,)):
        with original_dispatch(session, request) as origin:
            async with store.original_action_result(SCOPE, session, request, guard=lambda: None) as result:
                await store.action(SCOPE, session.session_id, request.action_id,
                    dispatch_revision=current.revision, _result_owner=result)
                descriptor = result._native_backing
                turn.revoke('original turn stopped before terminal fact')
                fact = origin.refuse_unentered(asyncio.CancelledError('original stop'))
                result._native_backing = replace(descriptor)
                with pytest.raises(ExecutionAuthorityError):
                    await result.not_dispatched(origin, fact)
                result._native_backing = descriptor
                with pytest.raises(Exception) as denied:
                    await store.db.fetchone('SELECT 1')
                assert is_execution_control_error(denied.value)
                await result.not_dispatched(origin, fact)
    stored = await store.action(SCOPE, session.session_id, request.action_id)
    assert stored.receipt.outcome == 'denied'
    assert stored.receipt.reason_code == 'host_not_dispatched'
    assert not await store.has_pending_actions(SCOPE, session.session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize('custody', ['postgres'], indirect=True)
async def test_native_postgres_capture_refines_same_original_pool_recipe(custody):
    store, _ = custody
    current, session, request = await prepared(store)
    with original_dispatch(session, request) as origin:
        async with store.original_action_result(SCOPE, session, request, guard=lambda: None) as result:
            initial, actual = result._native_backing.postgres_pool, result._postgres_pool
            assert actual is not initial
            assert initial.resolved_configuration is None and actual.resolved_configuration is not None
            assert initial.principal is None and actual.principal is not None
            assert actual.pool is initial.pool
            assert actual.connect_args is initial.connect_args
            assert actual.connect_kwargs is initial.connect_kwargs
            assert actual.configuration is initial.configuration
            await store.action(SCOPE, session.session_id, request.action_id,
                dispatch_revision=current.revision, _result_owner=result)
            await result.not_dispatched(origin, origin.refuse_unentered(RuntimeError('no driver entered')))
    stored = await store.action(SCOPE, session.session_id, request.action_id)
    assert stored.receipt.outcome == 'denied'


@pytest.mark.asyncio
@pytest.mark.parametrize('custody', ['postgres'], indirect=True)
async def test_equal_postgres_recipe_copy_cannot_replace_original_native_recipe(custody):
    store, _ = custody
    current, session, request = await prepared(store)
    with original_dispatch(session, request) as origin:
        async with store.original_action_result(SCOPE, session, request, guard=lambda: None) as result:
            await store.action(SCOPE, session.session_id, request.action_id,
                dispatch_revision=current.revision, _result_owner=result)
            original = result._postgres_pool
            copied = tuple(list(original.configuration))
            assert copied == original.configuration and copied is not original.configuration
            result._postgres_pool = replace(original, configuration=copied)
            with pytest.raises(ExecutionAuthorityError):
                await result.not_dispatched(origin, origin.refuse_unentered(RuntimeError('no driver entered')))
    assert await store.has_pending_actions(SCOPE, session.session_id)
