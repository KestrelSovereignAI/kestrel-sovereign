"""The published wait provider reads only the original Task's local custody."""

from uuid import uuid4

import pytest
from fastapi import Request
from kestrel_sdk.tools import Outcome

from kestrel_sovereign.auth import caller_context_scope
from kestrel_sovereign.mcp_task_submission import MCPTaskDelivery, MCPTaskServiceOrigin, MCPTaskSubmission
from kestrel_sovereign.waits.engine import TERMINAL_EVENT_FINAL_KEY, TERMINAL_EVENT_KEY
from kestrel_sovereign.waits.reconciler import _get_reconciler
from tests.integration.test_mcp_apps_host import apps_boundary as apps_boundary
from tests.integration.test_mcp_task_host import task_boundary as task_boundary
from tests.shared.mcp_task_custody import attach, binding, observation, submission


pytestmark = [pytest.mark.mcp_integration, pytest.mark.timeout(30)]


async def test_task_wait_provider_is_installed_before_private_task_host(apps_boundary):
    b = apps_boundary
    provider = b.agent.wait_registry.get('mcp')
    assert provider is not None
    handle = str(uuid4())
    assert not await provider.owns_handle(handle)
    assert (await provider.poll(handle)).outcome is Outcome.PENDING


@pytest.mark.parametrize('apps_boundary', [{'operator_input': 'form'}], indirect=True)
async def test_maintenance_registers_only_original_chat_delivery(task_boundary):
    b, host = task_boundary
    chat_binding = binding(agent_did=b.agent.did)
    original = submission()
    chat = await host.store.reserve(chat_binding, original)
    await attach(host.store, chat.issuance)
    app = MCPTaskSubmission.seal(arguments=original.arguments, contract=original.contract,
        authorization=original.authorization,
        delivery=MCPTaskDelivery('app', chat_binding.origin.session_id, 'physical-view'))
    view = await host.store.reserve(
        binding(agent_did=b.agent.did, operation_id='second-leaf-claim'), app)
    await attach(host.store, view.issuance, task_id='remote-app-task')

    await host.maintain(observe=True)

    store = _get_reconciler(b.agent)._store
    [watched] = await store.list_watched()
    assert (watched.kind, watched.handle) == ('mcp', chat.record.handle)
    assert watched.watch_origin_session_id == original.delivery.session_id
    assert await store.get('mcp', view.record.handle) is None


@pytest.mark.parametrize('apps_boundary', [{'operator_input': 'form'}], indirect=True)
@pytest.mark.parametrize('task_boundary',
    [{'maintenance_page_records': 8, 'observations_per_tick': 2}], indirect=True)
async def test_maintenance_does_not_skip_tasks_beyond_the_observation_budget(task_boundary):
    b, host = task_boundary
    origin = MCPTaskServiceOrigin('internal', 'd' * 64)
    handles = []
    for index in range(5):
        reserved = await host.store.reserve(
            binding(agent_did=b.agent.did, origin=origin, operation_id=f'leaf-{index}'),
            submission(origin=origin))
        attached = await attach(host.store, reserved.issuance, task_id=f'remote-{index}')
        handles.append(attached.handle)

    expected = tuple(sorted(handles))
    assert await host.maintain(observe=True) == expected[:2]
    assert await host.maintain(observe=True) == expected[2:4]
    assert await host.maintain(observe=True) == expected[4:]
    assert await host.maintain(observe=True) == ()
    assert await host.maintain(observe=True) == expected[:2]


@pytest.mark.parametrize('apps_boundary', [{'operator_input': 'form'}], indirect=True)
@pytest.mark.parametrize('state,outcome', [
    ('completed', Outcome.DONE), ('tool-error', Outcome.FAILED),
    ('failed', Outcome.FAILED), ('cancelled', Outcome.PARTIAL),
    ('withheld', Outcome.PARTIAL), ('remote_expired', Outcome.PARTIAL),
    ('resource_rejected', Outcome.PENDING),
])
async def test_local_task_wait_status_uses_the_exact_custody_without_naming_peer_events(
        task_boundary, state, outcome):
    b, host = task_boundary
    provider = b.agent.wait_registry.get('mcp')
    assert provider is not None
    reserved = await host.store.reserve(binding(agent_did=b.agent.did), submission())
    assert not await provider.owns_handle(reserved.record.handle)
    attached = await attach(host.store, reserved.issuance)
    assert await provider.owns_handle(attached.handle)
    assert (await provider.poll(attached.handle)).outcome is Outcome.PENDING
    if state in {'remote_expired', 'resource_rejected'}:
        await host.store.block_observation(attached, reason=state, blocked_at=1234.0,
            authority_epoch='owned-host', admission_serial=1)
    else:
        native = observation('completed' if state in {'withheld', 'tool-error'} else state)
        if state == 'tool-error':
            native['result']['isError'] = True
        elif state in {'completed', 'withheld'}:
            native['result']['isError'] = False
        native[TERMINAL_EVENT_KEY] = 'peer-supplied-event'
        await host.store.observe(attached, native, withheld='contract_drift' if state == 'withheld' else None)
    status = await provider.poll(attached.handle)
    assert status.outcome is outcome
    assert 'mcp_result' not in status.data
    if outcome is Outcome.PENDING:
        assert TERMINAL_EVENT_KEY not in status.data
    else:
        assert status.data[TERMINAL_EVENT_KEY] == f'mcp-task:{attached.handle}'
        assert status.data[TERMINAL_EVENT_FINAL_KEY] is True

    @b.app.get('/api/owned-task-wait/{handle}')
    async def foreign_wait(request: Request, handle: str):
        with caller_context_scope(request.state.caller):
            status = await provider.poll(handle)
            return {'owns': await provider.owns_handle(handle), 'data': status.data}

    denied = await b.client.get(f'/api/owned-task-wait/{attached.handle}', headers=b.headers('bob@apps.test'))
    assert denied.status_code == 200
    assert denied.json() == {'owns': False, 'data': None}
    host.withdraw()
    assert not await provider.owns_handle(attached.handle)
