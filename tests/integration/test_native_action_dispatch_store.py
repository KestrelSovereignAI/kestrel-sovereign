"""Actual SQLite/PostgreSQL migration and privacy of native action audit."""

from dataclasses import replace
import asyncio
import json

import pytest

from kestrel_sovereign.a2a.stores.unified.observability_store import ObservabilityStore, ToolDispatchEntry
from kestrel_sovereign.privacy import get_privacy_preset
from tests.integration.test_observability_privacy_gating_backends import observability_backend as observability_backend
from tests.unit.test_native_action_contract import context as context
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from tests.utils.native_a2a import manager_for


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize('private', [False, True])
async def test_native_action_column_migrates_legacy_rows_and_obeys_privacy(observability_backend, context, private):
    backend = observability_backend
    store = ObservabilityStore(backend, owns_backend=False)
    await store.initialize()
    entry = ToolDispatchEntry('did:test:native-audit', 'owned-session', 'owned-turn', 'owned_tool',
        'mcp:owned.owned_tool', {}, 'policy_denied', None, None, 1, 0)
    await store.log_tool_dispatch(entry, required=True)
    # Recreate a pre-action-column schema with a real legacy row, then run
    # the actual public initializer again against that existing database.
    await backend.execute('ALTER TABLE a2a_tool_dispatches DROP COLUMN native_action')
    restarted = ObservabilityStore(backend, owns_backend=False)
    await restarted.initialize()
    if private:
        restarted.set_privacy_config_provider(lambda: get_privacy_preset('ephemeral'))
    await restarted.log_tool_dispatch(replace(entry, native_action=context.to_dict()), required=True)
    rows = await restarted.recent_failures(entry.agent_did)
    assert len(rows) == 2 and rows[1]['native_action'] is None
    if private:
        assert context.task_handle not in json.dumps(rows[0]['native_action'])
        assert rows[0]['native_action'] != context.to_dict()
    else:
        assert rows[0]['native_action'] == context.to_dict()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_existing_core_6_receives_native_column_through_actual_task_manager_startup(observability_backend, context):
    backend = observability_backend
    scope = NativeRecipientScope('did:test:native-upgrade')
    original = manager_for(backend, scope, owns_backend=False)
    await original.initialize()
    await backend.execute('ALTER TABLE a2a_tool_dispatches DROP COLUMN native_action')
    await backend.execute("UPDATE a2a_native_store_schema SET revision=6 WHERE component='core'")
    await original.close()
    restarted = manager_for(backend, scope, owns_backend=False)
    try:
        initialized, = await asyncio.gather(restarted.initialize(), return_exceptions=True)
        assert initialized is None, 'The additive core-6 upgrade must run at actual startup'
        if backend.backend_type == 'sqlite':
            columns = await backend.fetch_all("SELECT name FROM pragma_table_info('a2a_tool_dispatches')")
        else:
            columns = await backend.fetch_all("SELECT column_name FROM information_schema.columns WHERE table_schema=current_schema() AND table_name='a2a_tool_dispatches'")
        assert 'native_action' in {row[0] for row in columns}
        await restarted.observability_store.log_tool_dispatch(ToolDispatchEntry(
            scope.recipient_id, 'owned-session', 'owned-turn', 'owned_tool', 'direct', {},
            'policy_denied', None, None, 1, 0, native_action=context.to_dict()), required=True)
    finally:
        await restarted.close()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_full_valid_native_identity_is_not_replaced_by_the_argument_preview_cap(observability_backend, context):
    store = ObservabilityStore(observability_backend, owns_backend=False)
    await store.initialize()
    long_context = replace(context, view_label='v'*1024, conversation_id='c'*1024)
    entry = ToolDispatchEntry('did:test:native-audit', 'owned-session', 'owned-turn', 'owned_tool',
        'direct', {}, 'policy_denied', None, None, 1, 0, native_action=long_context.to_dict())
    await store.log_tool_dispatch(entry, required=True)
    rows = await store.recent_failures(entry.agent_did)
    assert rows[0]['native_action'] == long_context.to_dict()


@pytest.mark.asyncio
@pytest.mark.dual_backend
@pytest.mark.parametrize('invalid', [
    {'inputResponses': {'private': 'answer'}},
    {'view_label': 'v' * 1025},
], ids=['private-field', 'oversize-identity'])
async def test_native_audit_accepts_only_the_bounded_identity_dto(observability_backend, context, invalid):
    store = ObservabilityStore(observability_backend, owns_backend=False)
    await store.initialize()
    entry = ToolDispatchEntry('did:test:native-audit', 'owned-session', 'owned-turn', 'owned_tool',
        'direct', {}, 'policy_denied', None, None, 1, 0,
        native_action={**context.to_dict(), **invalid})
    with pytest.raises((TypeError, ValueError)):
        await store.log_tool_dispatch(entry, required=True)
    assert await store.recent_failures(entry.agent_did) == []
