"""Native approval metadata uses real queue, SQL, agent bus and HTTP routes."""

import asyncio
import json
from uuid import uuid4

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from kestrel_sovereign.endpoints.security import router
from kestrel_sovereign.features.security.approval_queue import ApprovalQueue
from kestrel_sovereign.features.security.feature import SecurityFeature
from kestrel_sovereign.features.security.hooks import SecurityHook
from kestrel_sovereign.features.security.permissions import PermissionStore, PermissionLevel
from kestrel_sovereign.security.auto_approve import AutoApprovePolicy, AutoApproveRule
from tests.unit.test_governed_tool_bindings import host as host, _mount
from tests.unit.test_governed_native_actions import ActionTool, context_for
from kestrel_sdk.tools.result import ToolResultStatus
from kestrel_sovereign.security.input_guardrails import MAX_TOOL_ARG_LENGTH


@pytest.mark.asyncio
@pytest.mark.parametrize('native_consumer', ['deny-tool', 'approve-scope'])
async def test_actual_native_approval_bus_and_http_refuse_remembered_scopes(
        host, tmp_path, isolated_process_rate_limiter, native_consumer):
    agent, _ = host
    tool = ActionTool()
    binding = _mount(agent, tool)
    permissions = PermissionStore(str(tmp_path / 'approval-surfaces.db'))
    await permissions.initialize()
    feature = SecurityFeature(agent)
    events = asyncio.Queue()
    async def listener(event, data):
        if event == 'approval_request':
            await events.put(data)
    agent.add_event_listener(listener)
    queue = ApprovalQueue(on_request_added=feature._emit_approval_request,
                          permission_store=permissions, agent=agent)
    feature.permission_store, feature.approval_queue = permissions, queue
    agent.features['SecurityFeature'] = feature
    agent.hooks_manager.register(SecurityHook(permissions, queue))
    app = FastAPI()
    app.include_router(router)
    app.state.agent = agent
    context = context_for(binding)
    call = asyncio.create_task(binding.invoke_action('cancel', {}, native_action=context,
        session_id='owned-session', operation_id=uuid4().hex))
    try:
        event = await events.get()
        assert event['native_action'] == context.to_dict() and event['args'] == {}
        listing = await feature.pending_approvals()
        assert context.method in listing.confirmation and context.task_handle in listing.confirmation
        assert listing.data['requests'][0].get('native_action') == context.to_dict()
        async with AsyncClient(transport=ASGITransport(app=app), base_url='http://owned.invalid') as client:
            pending = await client.get('/api/security/pending')
            assert pending.status_code == 200
            assert pending.json()['pending'][0]['native_action'] == context.to_dict()
            for scope, remember in [('session', False), ('always', False), ('once', True)]:
                refused = await client.post('/api/security/approve', json={
                    'approval_id': event['id'], 'approved': True, 'scope': scope, 'remember': remember})
                assert refused.status_code == 400 and 'only once' in refused.json()['detail']
                assert not call.done() and queue.pending_count == 1
            accepted = await client.post('/api/security/approve', json={
                'approval_id': event['id'], 'approved': True, 'scope': 'once'})
            assert accepted.status_code == 200 and accepted.json()['success']
        assert (await call)['success'] is True and len(tool.action_calls) == 1
        assert await permissions.get_permission(binding.owner, binding.name) is PermissionLevel.ASK
        call = asyncio.create_task(binding.invoke_action('cancel', {}, native_action=context,
            session_id='owned-session', operation_id=uuid4().hex))
        second = await events.get()
        if native_consumer == 'approve-scope':
            refused = await feature.approve_request(second['id'], scope='session')
            assert refused.status is ToolResultStatus.ERROR
            assert 'only once' in refused.error
            assert queue.pending_count == 1 and not call.done()
        denied = await feature.deny_request(second['id'])
        assert denied.status is ToolResultStatus.OK
        assert (await call)['success'] is False and len(tool.action_calls) == 1
        assert queue.pending_count == 0
        audit = await permissions.get_audit_log()
        assert audit
        for row in audit:
            summary = json.loads(row['args_summary'])
            assert summary['native_action'] == context.to_dict()
            assert summary['parameter_names'] == []
            assert summary['parameters_type'] == 'dict'
    finally:
        queue.cancel_all()
        if not call.done():
            call.cancel()
        await asyncio.gather(call, return_exceptions=True)
        agent.remove_event_listener(listener)
        agent.features.pop('SecurityFeature')


@pytest.mark.asyncio
async def test_actual_ordinary_command_allowlist_cannot_auto_approve_native_params(host, tmp_path):
    agent, _ = host
    binding = _mount(agent, ActionTool())
    permissions = PermissionStore(str(tmp_path / 'native-allowlist.db'))
    await permissions.initialize()
    policy = AutoApprovePolicy([AutoApproveRule(r'^gh issue view ', 'owned/repo')], permissions)
    arguments = {'argv': ['gh', 'issue', 'view', '1', '-R', 'owned/repo']}
    match = await policy.evaluate(agent_name=None, feature_name='computer_use',
                                  tool_name='shell', tool_args=arguments)
    assert match is not None
    queue = ApprovalQueue(permission_store=permissions, auto_approve_policy=policy, agent=agent)
    ordinary = await queue.request_approval('computer_use', 'shell', arguments, allow_blocking=False)
    assert ordinary[0] is True and ordinary[1].startswith('auto_approve:')
    native = await queue.request_approval('computer_use', 'shell', arguments,
        native_action=context_for(binding), allow_blocking=False)
    assert native == (False, 'no_approver') and queue.pending_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('invalid,expected_names', [
    ({'count': 12}, ['count']), ({'inputResponses': {'slot': 'private-form-value'}}, ['inputResponses']),
    ({'extra': 'x' * 10_001}, ['extra']), ({'k' * (MAX_TOOL_ARG_LENGTH + 1): 1}, []),
], ids=['unknown-param', 'private-form', 'generic-guardrail', 'oversize-name'])
async def test_native_validation_refusal_keeps_full_identity_in_actual_security_audit(host, tmp_path, invalid, expected_names):
    agent, _ = host
    binding = _mount(agent, ActionTool())
    permissions = PermissionStore(str(tmp_path / 'native-validation-audit.db'))
    await permissions.initialize()
    feature = SecurityFeature(agent)
    feature.permission_store = permissions
    agent.features['SecurityFeature'] = feature
    context = context_for(binding)
    try:
        result = await binding.invoke_action('cancel', invalid, native_action=context,
            session_id='owned-session', operation_id=uuid4().hex)
        assert result['success'] is False
        rows = await permissions.get_audit_log()
        row, = [row for row in rows if row['action'] == 'tool_validation']
        summary = row['args_summary'].split(' | args=', 1)[1]
        assert not summary.endswith('...'), 'Native validation identity must not use the ordinary preview cap'
        projection = json.loads(summary)
        assert 'native_action' in projection
        assert projection['native_action'] == context.to_dict()
        assert projection['parameter_names'] == expected_names
        assert len(json.dumps(projection['parameter_names'])) <= MAX_TOOL_ARG_LENGTH
        if not expected_names:
            assert projection['parameter_names_omitted'] == 1
        assert 'private-form-value' not in summary
    finally:
        agent.features.pop('SecurityFeature')
