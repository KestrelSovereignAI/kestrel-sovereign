"""Native Task actions retain the original tool's actual dispatch boundary."""

from dataclasses import replace
import asyncio
import hashlib
import json
from uuid import uuid4

import pytest
from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown
from kestrel_sovereign.tool_effect_store import ToolEffectReconciliationRequired
from kestrel_sdk.tools import NativeAction, NativeActionContext
from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput, PermissionDecision
from tests.unit.test_governed_tool_bindings import host as host, _Tool, _mount, _Policy
from kestrel_sovereign.features.security.approval_queue import ApprovalQueue, DecisionResult
from kestrel_sovereign.features.security.hooks import SecurityHook, MCP_TASK_SERVICE_SESSION_ID
from kestrel_sovereign.features.security.permissions import PermissionLevel, PermissionStore
from kestrel_sovereign.features.security.args_summary import summarize_args
from kestrel_sovereign.security.input_guardrails import MAX_TOOL_ARG_LENGTH, validate_tool_arguments


def context_for(binding, **changes):
    digest = hashlib.sha256(b'owned-original').hexdigest()
    context = NativeActionContext('cancel', 'tasks/cancel', 'owned-local', digest,
        binding.owner, binding.name, binding.registration_id, digest, 'service', digest, 0)
    return replace(context, **changes)


class ActionTool(_Tool):
    def __init__(self):
        super().__init__()
        self.action_calls = []
        self.native_actions = {'cancel': NativeAction('cancel', 'tasks/cancel',
            b'{"type":"object","additionalProperties":false}', self.cancel)}

    async def cancel(self, context, params):
        self.action_calls.append((context.to_dict(), params))
        return {'success': True, 'cancel_requested': True}


class UpdateTool(ActionTool):
    def __init__(self):
        super().__init__()
        self.native_actions['update'] = NativeAction('update', 'tasks/update', json.dumps({
            'type': 'object', 'required': ['inputResponses'], 'additionalProperties': False,
            'properties': {'inputResponses': {'type': 'object', 'additionalProperties': {'type': 'string'}}},
        }).encode(), self.update)

    async def update(self, context, params):
        self.action_calls.append((context.to_dict(), params))
        return {'success': True, 'input_updated': True}


@pytest.mark.asyncio
async def test_invalid_action_batch_leaves_original_registry_and_binding(host):
    agent, _ = host
    tool = ActionTool()
    binding = _mount(agent, tool)
    original = (dict(agent._direct_tools), list(agent._direct_tool_defs),
                dict(agent._tool_to_feature), dict(agent._explored_features))
    broken = ActionTool()
    broken.native_actions = {'wrong': broken.native_actions['cancel']}
    with pytest.raises(ValueError, match='mapping key'):
        _mount(agent, broken)
    assert original == (agent._direct_tools, agent._direct_tool_defs,
                        agent._tool_to_feature, agent._explored_features)
    assert binding.is_current


@pytest.mark.asyncio
@pytest.mark.parametrize('changed', ['action', 'method', 'owner', 'tool_name', 'registration_id'])
async def test_foreign_action_identity_is_refused_before_any_dispatch(host, changed):
    agent, backend = host
    tool = ActionTool()
    binding = _mount(agent, tool)
    with pytest.raises(ResourceConflict, match='original tool registration'):
        await binding.invoke_action('cancel', {}, native_action=context_for(binding, **{changed: 'foreign'}),
                                    session_id='owned-session', operation_id=uuid4().hex)
    assert tool.calls == tool.action_calls == []
    assert await backend.fetch_one('SELECT COUNT(*) FROM a2a_tool_dispatches') == (0,)


@pytest.mark.asyncio
async def test_mount_captures_membership_and_schema_without_aliasing_feature(host):
    agent, _ = host
    tool = ActionTool()
    original = tool.native_actions['cancel']
    binding = _mount(agent, tool)
    tool.native_actions.clear()
    object.__setattr__(original, 'schema_json', b'{"type":"null"}')
    assert binding.native_actions[0].schema == {'type': 'object', 'additionalProperties': False}
    assert binding.native_actions[0].handler == tool.cancel


@pytest.mark.asyncio
async def test_cancel_runs_only_captured_action_through_hooks_and_actual_sql(host):
    agent, backend = host
    tool = ActionTool()
    binding = _mount(agent, tool)
    context = context_for(binding)
    policy = _Policy(HookOutput.allow())
    agent.hooks_manager.register(policy)
    result, = await asyncio.gather(binding.invoke_action('cancel', {}, native_action=context,
        session_id='owned-session', operation_id=uuid4().hex), return_exceptions=True)
    assert result == {'success': True, 'cancel_requested': True}
    assert tool.calls == [] and tool.action_calls == [(context.to_dict(), {})]
    assert policy.inputs[0].native_action == context
    row = await backend.fetch_one('SELECT native_action,args_redacted,result_status FROM a2a_tool_dispatches')
    assert row[0] is not None
    assert json.loads(row[0]) == context.to_dict()
    assert json.loads(row[1])['parameter_names'] == [] and row[2] == 'success'


@pytest.mark.asyncio
@pytest.mark.parametrize('output', [HookOutput.deny('owned deny'), HookOutput.ask('owned approval')])
async def test_native_action_deny_and_pending_approval_never_execute(host, output):
    agent, backend = host
    tool = ActionTool()
    binding = _mount(agent, tool)
    context = context_for(binding)
    agent.hooks_manager.register(_Policy(output))
    result = await binding.invoke_action('cancel', {}, native_action=context,
        session_id='owned-session', operation_id=uuid4().hex)
    assert result['success'] is False
    assert tool.calls == tool.action_calls == []
    row = await backend.fetch_one('SELECT native_action,result_status FROM a2a_tool_dispatches')
    expected_status = 'error' if output.permission_decision is PermissionDecision.ASK else 'policy_denied'
    assert json.loads(row[0]) == context.to_dict() and row[1] == expected_status


@pytest.mark.asyncio
async def test_empty_cancel_rewrite_and_invalid_post_hook_params_do_not_fall_through(host):
    agent, backend = host
    tool = ActionTool()
    binding = _mount(agent, tool)
    policy = _Policy(HookOutput.modify({'count': 12}))
    agent.hooks_manager.register(policy)
    result = await binding.invoke_action('cancel', {}, native_action=context_for(binding),
        session_id='owned-session', operation_id=uuid4().hex)
    assert result['success'] is False and tool.calls == tool.action_calls == []
    policy.output = HookOutput.modify({})
    result = await binding.invoke_action('cancel', {}, native_action=context_for(binding),
        session_id='owned-session', operation_id=uuid4().hex)
    assert result['success'] is True and len(tool.action_calls) == 1 and tool.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize('scope', ['always', 'session', 'auto'])
async def test_native_approval_is_once_only_and_cannot_widen_ordinary_tool(host, tmp_path, scope):
    agent, _ = host
    tool = ActionTool()
    binding = _mount(agent, tool)
    permissions = PermissionStore(str(tmp_path / 'native-permissions.db'))
    await permissions.initialize()
    requests = asyncio.Queue()
    async def added(request):
        await requests.put(request)
    queue = ApprovalQueue(on_request_added=added, permission_store=permissions, agent=agent)
    agent.hooks_manager.register(SecurityHook(permissions, queue))
    context = context_for(binding)
    call = asyncio.create_task(binding.invoke_action('cancel', {}, native_action=context,
                              session_id='owned-session', operation_id=uuid4().hex))
    try:
        request = await requests.get()
        assert request.native_action == context and request.to_dict()['native_action'] == context.to_dict()
        refused, = await asyncio.gather(queue.submit_decision(request.id, True, scope), return_exceptions=True)
        assert isinstance(refused, DecisionResult)
        assert not refused.in_memory and not refused.persisted and 'only once' in refused.error
        assert not call.done() and queue.pending_count == 1
        assert await permissions.get_permission(binding.owner, binding.name) is PermissionLevel.ASK
        accepted = await queue.submit_decision(request.id, True, 'once')
        assert accepted.in_memory and accepted.persisted and (await call)['success'] is True
        assert len(tool.action_calls) == 1 and tool.calls == []
        assert await permissions.get_permission(binding.owner, binding.name) is PermissionLevel.ASK
        # The ordinary call still asks and has no action context.
        call = asyncio.create_task(binding.invoke({'count': 3}, session_id='owned-session',
                                                  operation_id=uuid4().hex))
        ordinary = await requests.get()
        assert ordinary.native_action is None and 'native_action' not in ordinary.to_dict()
        await queue.submit_decision(ordinary.id, False, 'once')
        assert (await call)['success'] is False and tool.calls == []
    finally:
        queue.cancel_all()
        if not call.done():
            call.cancel()
        await asyncio.gather(call, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('level', [PermissionLevel.ASK, PermissionLevel.DENY, PermissionLevel.ALLOW])
async def test_service_action_obeys_original_permission_without_waiting_for_approver(host, tmp_path, level):
    agent, _ = host
    tool = ActionTool()
    binding = _mount(agent, tool)
    permissions = PermissionStore(str(tmp_path / 'service-permissions.db'))
    await permissions.initialize()
    await permissions.set_permission(binding.owner, binding.name, level)
    queue = ApprovalQueue(permission_store=permissions, agent=agent)
    agent.hooks_manager.register(SecurityHook(permissions, queue))
    result = await binding.invoke_action('cancel', {}, native_action=context_for(binding),
        session_id=MCP_TASK_SERVICE_SESSION_ID, operation_id=uuid4().hex)
    assert result['success'] is (level is PermissionLevel.ALLOW)
    assert len(tool.action_calls) == (1 if level is PermissionLevel.ALLOW else 0)
    assert tool.calls == [] and queue.pending_count == 0
    rows = await permissions.get_audit_log()
    assert len(rows) == 1
    assert rows[0]['decision'] == {PermissionLevel.ASK: 'no_approver', PermissionLevel.DENY: 'auto_denied',
                                   PermissionLevel.ALLOW: 'auto_allowed'}[level]


def test_native_security_summary_contains_identity_and_never_form_values():
    class Binding:
        owner, name, registration_id = 'mcp:owned', 'owned_tool', 'owned-batch'
    context = context_for(Binding())
    summary = summarize_args({'inputResponses': {'opaque-request': {'answer': 'private answer'}}},
                             native_action=context, max_length=10_000)
    assert 'private answer' not in summary and 'opaque-request' not in summary
    assert json.loads(summary)['native_action']['task_handle'] == context.task_handle


@pytest.mark.asyncio
async def test_native_guardrail_refusal_keeps_private_form_values_and_keys_out_of_audit(host, caplog):
    agent, backend = host
    tool = ActionTool()
    binding = _mount(agent, tool)
    private = {'inputResponses': {'private-request-id': 'private-answer-' * MAX_TOOL_ARG_LENGTH}}
    result = await binding.invoke_action('cancel', private, native_action=context_for(binding),
        session_id='owned-session', operation_id=uuid4().hex)
    assert result['success'] is False and tool.calls == tool.action_calls == []
    row = await backend.fetch_one('SELECT args_redacted,error_message FROM a2a_tool_dispatches')
    for value in [str(tuple(row)), caplog.text, json.dumps(result)]:
        assert 'private-request-id' not in value and 'private-answer-' not in value


@pytest.mark.parametrize('value', [pytest.param(object(), id='invalid-json-type'),
                                  pytest.param('x' * (MAX_TOOL_ARG_LENGTH + 1), id='oversize-string')])
def test_content_free_guardrail_diagnostics_preserve_the_original_refusal(value, caplog):
    valid, error = validate_tool_arguments('owned_tool', {'private-form-key': value}, redact_paths=True)
    assert not valid and 'private-form-key' not in error and 'private-form-key' not in caplog.text
    valid, ordinary = validate_tool_arguments('owned_tool', {'ordinary-key': value})
    assert not valid and 'ordinary-key' in ordinary


@pytest.mark.asyncio
@pytest.mark.parametrize('rewritten', [False, True])
async def test_retained_prior_hook_cannot_change_native_params_while_approval_waits(host, tmp_path, rewritten):
    agent, _ = host
    tool = UpdateTool()
    binding = _mount(agent, tool)
    retained = []
    outputs = []
    class Retain(Hook):
        def __init__(self):
            super().__init__('owned-retainer', [HookEvent.PRE_TOOL_USE], priority=0)
        async def execute(self, value):
            retained.append(value)
            result = HookOutput.modify({'inputResponses': {'key': 'approved'}}) if rewritten else HookOutput.allow()
            outputs.append(result)
            return result
    permissions = PermissionStore(str(tmp_path / 'retained-native-hook.db'))
    await permissions.initialize()
    requests = asyncio.Queue()
    async def added(request):
        await requests.put(request)
    queue = ApprovalQueue(on_request_added=added, permission_store=permissions, agent=agent)
    agent.hooks_manager.register(Retain())
    agent.hooks_manager.register(SecurityHook(permissions, queue))
    context = context_for(binding, action='update', method='tasks/update')
    call = asyncio.create_task(binding.invoke_action('update', {'inputResponses': {'key': 'approved'}},
        native_action=context, session_id='owned-session', operation_id=uuid4().hex))
    try:
        request = await requests.get()
        assert request.tool_args == {'inputResponses': {'key': 'approved'}}
        retained[0].tool_input['inputResponses']['key'] = 'changed-after-approval-shown'
        if rewritten:
            outputs[0].updated_input['inputResponses']['key'] = 'changed-retained-output'
        assert request.tool_args == {'inputResponses': {'key': 'approved'}}
        await queue.submit_decision(request.id, True, 'once')
        assert (await call)['success'] is True
        assert tool.action_calls == [(context.to_dict(), {'inputResponses': {'key': 'approved'}})]
    finally:
        queue.cancel_all()
        if not call.done():
            call.cancel()
        await asyncio.gather(call, return_exceptions=True)


@pytest.mark.asyncio
async def test_two_task_targets_have_distinct_operation_and_effect_digests(host):
    agent, _ = host
    tool = ActionTool()
    binding = _mount(agent, tool)
    for handle in ('owned-first-task', 'owned-second-task'):
        result = await binding.invoke_action('cancel', {}, native_action=context_for(binding, task_handle=handle),
            session_id='owned-session', operation_id=uuid4().hex)
        assert result['success'] is True
    database = agent.tool_effect_store.database
    operations = await database.fetchall('SELECT request_digest FROM tool_effect_operations')
    claims = await database.fetchall('SELECT registration_digest,args_digest FROM tool_effect_claims')
    assert len(operations) == len(claims) == 2
    assert len({row[0] for row in operations}) == len({row[0] for row in claims}) == 2
    assert len({row[1] for row in claims}) == 1
    assert len(tool.action_calls) == 2 and tool.calls == []


@pytest.mark.asyncio
async def test_native_action_and_ordinary_call_have_separate_intent_namespaces(host):
    agent, _ = host
    tool = ActionTool()
    binding = _mount(agent, tool)
    intent = uuid4().hex
    assert (await binding.invoke_action('cancel', {}, native_action=context_for(binding),
        session_id='owned-session', operation_id=intent))['success'] is True
    ordinary, = await asyncio.gather(binding.invoke({'count': 2}, source='mcp-app',
        session_id='owned-session', operation_id=intent), return_exceptions=True)
    assert ordinary == {'success': True, 'count': 2}
    assert len(tool.action_calls) == 1 and tool.calls == [{'count': 2}]
    assert await agent.tool_effect_store.database.fetchone('SELECT COUNT(*) FROM tool_effect_operations') == (2,)


@pytest.mark.asyncio
async def test_original_native_handler_and_context_survive_caller_and_declaration_changes(host):
    agent, backend = host
    tool = ActionTool()
    binding = _mount(agent, tool)
    context = context_for(binding)
    expected = context.to_dict()
    async def changed_handler(context, params):
        pytest.fail('Invocation must use the originally registered handler')
    async def change_after_admission():
        object.__setattr__(context, 'task_handle', 'changed-caller-target')
        object.__setattr__(tool.native_actions['cancel'], 'handler', changed_handler)
        tool.native_actions.clear()
    agent.observability_store.on_admitted = change_after_admission
    assert (await binding.invoke_action('cancel', {}, native_action=context,
        session_id='owned-session', operation_id=uuid4().hex))['success'] is True
    assert tool.action_calls == [(expected, {})] and tool.calls == []
    row = await backend.fetch_one('SELECT native_action FROM a2a_tool_dispatches')
    assert json.loads(row[0]) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['handler', 'audit'])
async def test_native_action_after_effect_failure_is_unknown_and_never_replayed(host, failure):
    agent, _ = host
    class FailingAction(ActionTool):
        async def cancel(self, context, params):
            result = await super().cancel(context, params)
            if failure == 'handler':
                raise ValueError('Native handler failed after the effect')
            return result
    tool = FailingAction()
    binding = _mount(agent, tool)
    if failure == 'audit':
        async def fail_actual_audit_write():
            await agent.observability_store._backend.execute('UPDATE missing_native_action_audit SET value=1')
        agent.observability_store.on_dispatch = fail_actual_audit_write
    intent = uuid4().hex
    with pytest.raises(ResourceOutcomeUnknown):
        await binding.invoke_action('cancel', {}, native_action=context_for(binding),
            session_id='owned-session', operation_id=intent)
    assert len(tool.action_calls) == 1 and tool.calls == []
    with pytest.raises(ToolEffectReconciliationRequired):
        await binding.invoke_action('cancel', {}, native_action=context_for(binding),
            session_id='owned-session', operation_id=intent)
    assert len(tool.action_calls) == 1


@pytest.mark.asyncio
async def test_internal_native_persistence_also_refuses_permission_widening(host, tmp_path):
    agent, _ = host
    binding = _mount(agent, ActionTool())
    permissions = PermissionStore(str(tmp_path / 'native-persistence-defense.db'))
    await permissions.initialize()
    queue = ApprovalQueue(permission_store=permissions, agent=agent)
    with pytest.raises(ValueError, match='only once'):
        await queue._persist_decision(feature_name=binding.owner, tool_name=binding.name,
            tool_args={}, approved=True, scope='always', native_action=context_for(binding))
    assert await permissions.get_permission(binding.owner, binding.name) is PermissionLevel.ASK


@pytest.mark.asyncio
async def test_native_update_decided_in_place_uses_one_parameter_copy_and_content_free_audit(host):
    agent, backend = host
    tool = UpdateTool()
    binding = _mount(agent, tool)
    class Decide(Hook):
        def __init__(self):
            super().__init__('owned-decider', [HookEvent.PRE_TOOL_USE])
        async def execute(self, input):
            input.tool_input['inputResponses']['opaque-request'] = 'private-decided-value'
            return HookOutput.allow()
    agent.hooks_manager.register(Decide())
    result = await binding.invoke_action('update', {'inputResponses': {'opaque-request': 'private-offered-value'}},
        native_action=context_for(binding, action='update', method='tasks/update'),
        session_id='owned-session', operation_id=uuid4().hex)
    assert result['success'] is True and tool.calls == []
    assert tool.action_calls[0][1] == {'inputResponses': {'opaque-request': 'private-decided-value'}}
    rows = await backend.fetch_all('SELECT metadata FROM a2a_observability')
    rows += await backend.fetch_all('SELECT args_redacted FROM a2a_tool_dispatches')
    serialized = repr(rows)
    for private in ('private-decided-value', 'private-offered-value', 'opaque-request'):
        assert private not in serialized
