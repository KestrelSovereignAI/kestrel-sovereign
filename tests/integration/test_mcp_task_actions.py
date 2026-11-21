"""Authenticated Task action dispatch retains the exact approved SQL intent."""

from uuid import uuid4

from fastapi import Request
from fastapi.responses import JSONResponse
from kestrel_sdk.tools import NativeAction
import pytest

from kestrel_sovereign.auth import caller_context_scope, caller_identity_digest
from kestrel_sovereign.features.security.permissions import PermissionLevel
from kestrel_sovereign.mcp_task_submission import MCPTaskInteractiveOrigin, MCPTaskOAuthLocator, task_connection_id
from kestrel_sovereign.security.capability_authority import HostCapabilityAccessDenied
from kestrel_sovereign.security.host_authority import HostAuthorityError
from tests.integration.test_mcp_apps_host import apps_boundary as apps_boundary
from tests.integration.test_mcp_task_host import task_boundary as task_boundary
from tests.shared.mcp_task_custody import attach, binding, observation, submission
from tests.unit.test_governed_tool_bindings import _mount, _Tool


class _TaskTool(_Tool):
    def __init__(self, host):
        super().__init__()
        self.host = host
        self.calls = []
        self.advance_poll_before_reserve = False
        self.native_actions = {
            'update': NativeAction('update', 'tasks/update',
                b'{"type":"object","properties":{"inputResponses":{"type":"object"}},"required":["inputResponses"],'
                b'"additionalProperties":false}', self.mutate),
            'cancel': NativeAction('cancel', 'tasks/cancel',
                b'{"type":"object","additionalProperties":false}', self.mutate),
        }

    async def mutate(self, context, arguments):
        mutation = self.host.actions.round(self.binding, context, arguments)
        self.calls.append(context)
        if self.advance_poll_before_reserve:
            await self.host.store.defer_observation(mutation.record, not_before=10.0)
        await mutation.reserve()
        return {'success': True}


@pytest.mark.mcp_integration
@pytest.mark.timeout(30)
@pytest.mark.parametrize('apps_boundary', [{'operator_input': 'form'}], indirect=True)
@pytest.mark.parametrize('case', ['update', 'foreign-update', 'cancel', 'foreign-cancel', 'changed-pacing'])
async def test_task_mutation_stays_in_its_original_approval_and_operator_scope(task_boundary, case):
    b, host = task_boundary
    tool = _TaskTool(host)
    tool.binding = _mount(b.agent, tool, owner='mcp:owned-task-actions')
    await b.permissions.set_permission(tool.binding.owner, tool.binding.name, PermissionLevel.ALLOW)
    owned_session = 'owned-task-actions-chat'

    @b.app.post('/api/owned-task-mutation-prepare')
    async def prepare(request: Request):
        caller = request.state.caller
        origin = MCPTaskInteractiveOrigin(caller_identity_digest(caller), owned_session)
        locator = MCPTaskOAuthLocator(tool.binding.owner, 'owned', origin.principal_digest, 'e' * 64)
        target = binding(agent_did=b.agent.did, origin=origin, tool_name=tool.binding.name,
            connection_id=task_connection_id(b.agent.did, locator.owner, locator.connection_name))
        reserved = await host.store.reserve(target, submission(origin=origin, authorization=locator))
        attached = await attach(host.store, reserved.issuance)
        return {'handle': (await host.store.observe(attached, observation('input_required'))).handle}

    @b.app.post('/api/owned-task-mutation/{handle}')
    async def mutate(request: Request, handle: str):
        value = await request.json()
        try:
            with caller_context_scope(request.state.caller):
                return await host.actions.invoke(tool.binding, handle=handle, session_id=owned_session,
                    operation_id=uuid4().hex, contract_digest=binding().contract_digest,
                    kind=value['kind'], arguments=value['arguments'])
        except (HostCapabilityAccessDenied, HostAuthorityError):
            return JSONResponse({'denied': True}, status_code=403)

    prepared = await b.client.post('/api/owned-task-mutation-prepare', headers=b.headers('key'))
    assert prepared.status_code == 200, prepared.text
    handle = prepared.json()['handle']
    original = await host.store.read_handle(handle)
    tool.advance_poll_before_reserve = case == 'changed-pacing'
    kind = 'cancel' if case.endswith('cancel') else 'update'
    arguments = {'inputResponses': {'private-key': {'action': 'decline'}}} if kind == 'update' else {}
    foreign = case.startswith('foreign')
    response = await b.client.post('/api/owned-task-mutation/' + handle,
        headers=b.headers('bob@apps.test' if foreign else 'key'), json={'kind': kind, 'arguments': arguments})
    actual = await host.store.read_handle(handle)
    if foreign:
        assert response.status_code == 403 and response.json() == {'denied': True}
        assert actual == original and tool.calls == []
        return
    assert response.status_code == 200 and response.json().get('success') is True, response.text
    context, = tool.calls
    assert context.task_handle == handle and context.conversation_id == owned_session
    receipt, = actual.control['mutations'].values()
    assert receipt['acknowledgement'] is None
    assert receipt['intent'] == {'method': 'tasks/' + kind,
        'params': {'taskId': original.remote_task_id, **arguments}}
    assert actual.phase == 'ATTACHED'
