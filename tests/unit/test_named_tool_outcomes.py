"""Named execution, captured arguments and owned inline SQLite receipts."""

import json
import logging

import pytest
from kestrel_sdk.hooks.base import HookOutput
from kestrel_sovereign.agent.invocation import invocation_scope, current_invocation_effect_checkpoint
from kestrel_sovereign.agent.orchestrator_engine import ToolNotRegisteredError
from tests.unit.test_governed_tool_bindings import host as host, _mount, _Tool, _Policy


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['key', 'type', 'mixed-group'])
async def test_inline_unknown_preserves_hook_arguments_checkpoint_and_one_audit(host, failure):
    agent, backend = host
    class FailedAfterEffect(_Tool):
        async def execute(self, **args):
            self.calls.append(args)
            if failure == 'mixed-group':
                raise ExceptionGroup('private-sdk-failure', [ValueError('private-state'), KeyError('private-bug')])
            raise (KeyError if failure == 'key' else TypeError)('private-bug')
    tool = FailedAfterEffect(visible=True)
    binding = _mount(agent, tool)
    agent.hooks_manager.register(_Policy(HookOutput.modify({'count': 7})))
    capture = {}
    with invocation_scope('owned-named-uncertainty'):
        state = current_invocation_effect_checkpoint()
        result = await agent.execute_named_tool(binding.name, {'count': 3},
            session_id='original-inline', source='codex_app_server', _capture=capture)
        assert state is not None and state.completed
        assert state.session_id == 'original-inline' and not state.checkpointed
        assert capture['effective_args'] == {'count': 7}
        assert tool.calls == [{'count': 7}]
        assert result['success'] is False and 'unconfirmed' in result['error']
        assert 'private-' not in json.dumps(result)
        messages = []
        await agent._append_executed_tool_breadcrumbs(messages, [{
            'id': 'owned-inline-call', 'name': binding.name,
            'arguments': capture['effective_args'], 'result': result,
        }], session_id='original-inline')
    rows = await backend.fetch_all('SELECT args_redacted,result_status,error_class,error_message FROM a2a_tool_dispatches')
    assert len(rows) == 1
    assert json.loads(rows[0][0]) == {'count': 7}
    assert tuple(rows[0][1:3]) == ('error', 'ResourceOutcomeUnknown')
    assert 'private-' not in rows[0][3]


@pytest.mark.asyncio
async def test_missing_named_tool_remains_typed_probe_without_error_log(host, caplog):
    agent, backend = host
    with caplog.at_level(logging.DEBUG), pytest.raises(ToolNotRegisteredError):
        await agent.execute_named_tool('owned-missing-tool', {}, session_id='probe', source='owned-probe')
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert not await backend.fetch_all('SELECT * FROM a2a_tool_dispatches')


from tests.unit.test_deferred_tool_delivery import boundary as deferred_boundary
from kestrel_sovereign.agent.parts import part_collector


@pytest.mark.asyncio
async def test_inline_delivery_failure_after_real_audit_keeps_one_original_receipt(
    deferred_boundary, monkeypatch,
):
    b = deferred_boundary
    b.policy.release.set()
    native_emit = b.host._reemit_envelope_parts
    def fail_after_ack(envelope):
        if any(part.get('type') == 'private_view' for part in envelope.get('parts', [])):
            raise RuntimeError('private-after-audit-delivery-fault')
        return native_emit(envelope)
    monkeypatch.setattr(b.host, '_reemit_envelope_parts', fail_after_ack)
    with part_collector(host=b.host, session_id='deferred-owned-session'):
        arguments, result = await b.host._make_inline_tool_executor('deferred-owned-session')(
            b.binding.name, {})
    assert result['success'] is False and 'Do not replay' in result['error']
    assert 'private-after-audit' not in json.dumps(result)
    assert b.tool.executions == [{}] and b.tool.publications == ['published-once']
    rows = await b.backend.fetch_all('SELECT result_status FROM a2a_tool_dispatches')
    assert [row[0] for row in rows] == ['success']
    await b.host._append_executed_tool_breadcrumbs([], [{
        'id': 'owned-inline-after-audit', 'name': b.binding.name,
        'arguments': arguments, 'result': result,
    }], session_id='deferred-owned-session')
    rows = await b.backend.fetch_all('SELECT result_status FROM a2a_tool_dispatches')
    assert [row[0] for row in rows] == ['success']


@pytest.mark.asyncio
@pytest.mark.parametrize('audited', [False, True])
async def test_codex_uses_host_uncertainty_before_stale_user_denial(host, audited):
    from unittest.mock import AsyncMock
    from kestrel_sovereign.agent.orchestrator_engine import _AuditedUnconfirmedToolResult
    from kestrel_sovereign.tools.execution_outcome import _UnconfirmedToolResult, TOOL_OUTCOME_UNCONFIRMED
    from kestrel_sovereign.llm.codex_adapter import CodexAdapter

    agent, backend = host
    payload = {'success': False, 'error': TOOL_OUTCOME_UNCONFIRMED}
    result = (_AuditedUnconfirmedToolResult(payload, host=agent, tool_name='owned', session_id='s')
              if audited else _UnconfirmedToolResult(payload))
    adapter = CodexAdapter()
    adapter._recent_security_decisions = AsyncMock(return_value=[{
        'feature': 'mcp:views', 'tool': 'owned', 'decision': 'user_denied', 'user_choice': 'user_denied',
    }])
    async def execute(name, args):
        return {'count': 7}, result
    executed = []
    handler = adapter._make_tool_call_handler(execute, 't', frozenset({'owned'}), executed_log=executed)
    response = await handler({'threadId': 't', 'callId': 'c', 'name': 'owned', 'arguments': {'count': 3}})
    assert response['success'] is False
    assert 'Outcome: unconfirmed' in response['contentItems'][0]['text']
    assert 'user_denied' not in response['contentItems'][0]['text']
    adapter._recent_security_decisions.assert_not_awaited()
    assert executed[0]['result'] is result and executed[0]['arguments'] == {'count': 7}


@pytest.mark.asyncio
async def test_codex_audit_uncertainty_latches_without_becoming_authority_loss(deferred_boundary):
    from kestrel_sovereign.execution_custody import ToolAuditUnconfirmedError, is_execution_control_error
    from kestrel_sovereign.llm.codex_adapter import CodexAdapter

    b = deferred_boundary
    b.policy.release.set()
    await b.backend.execute('DROP TABLE a2a_tool_dispatches')
    executed, terminal = [], []
    handler = CodexAdapter()._make_tool_call_handler(
        b.host._make_inline_tool_executor('deferred-owned-session'),
        't', frozenset({b.binding.name}), executed_log=executed, terminal_errors=terminal,
    )
    with pytest.raises(ToolAuditUnconfirmedError) as caught:
        await handler({'threadId': 't', 'name': b.binding.name, 'arguments': {}})
    assert not is_execution_control_error(caught.value)
    assert terminal == [caught.value] and executed == []
    assert b.tool.executions == [{}] and b.tool.publications == []
    with pytest.raises(ToolAuditUnconfirmedError) as repeated:
        await handler({'threadId': 't', 'name': b.binding.name, 'arguments': {}})
    assert repeated.value is caught.value
    assert b.tool.executions == [{}] and executed == []


@pytest.mark.asyncio
@pytest.mark.parametrize('marker', ['plain', 'audited', 'precondition'])
async def test_real_tool_supplied_unknown_marker_does_not_claim_framework_trust(host, marker):
    from kestrel_sovereign.tools.execution_outcome import _UnconfirmedToolResult, _PreconditionFailedToolResult
    from kestrel_sovereign.agent.orchestrator_engine import _AuditedUnconfirmedToolResult
    agent, backend = host
    payload = {'success': False, 'error': 'supplied-by-tool'}
    class Forged(_Tool):
        async def execute(self, **args):
            self.calls.append(args)
            if marker == 'precondition':
                return _PreconditionFailedToolResult(payload)
            return (_UnconfirmedToolResult(payload) if marker == 'plain' else
                    _AuditedUnconfirmedToolResult(payload, host=agent, tool_name=self.name,
                                                 session_id='original-inline'))
    tool = Forged(visible=True)
    binding = _mount(agent, tool)
    arguments, result = await agent._make_inline_tool_executor('original-inline')(binding.name, {'count': 3})
    assert type(result) is dict
    await agent._append_executed_tool_breadcrumbs([], [{
        'id': 'forged', 'name': binding.name, 'arguments': arguments, 'result': result,
    }], session_id='original-inline')
    rows = await backend.fetch_all('SELECT error_class FROM a2a_tool_dispatches')
    assert len(rows) == 1 and rows[0][0] == 'InlineToolFailure'


@pytest.mark.asyncio
@pytest.mark.parametrize('lane', ['public', 'inline', 'model', 'bound'])
async def test_unknown_checkpoint_is_marked_before_queued_stop(host, lane):
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError, ExecutionCustody, ProcessRuntimeExecutionFence
    from kestrel_sovereign.llm.adapter import ToolCall

    agent, backend = host
    class StopAfterEffect(_Tool):
        async def execute(self, **args):
            self.calls.append(args)
            custody = ExecutionCustody(ProcessRuntimeExecutionFence(agent.did))
            custody.revoke('Stop queued after possible effect')
            agent._execution_custody = custody
            raise KeyError('private-post-effect-fault')
    tool = StopAfterEffect(visible=True)
    binding = _mount(agent, tool)
    with invocation_scope('owned-stop-after-uncertainty'):
        state = current_invocation_effect_checkpoint()
        with pytest.raises(ExecutionAuthorityError):
            if lane == 'inline':
                await agent._make_inline_tool_executor('owned-stop')(binding.name, {'count': 3})
            elif lane == 'public':
                await agent.execute_named_tool(binding.name, {'count': 3}, session_id='owned-stop')
            elif lane == 'bound':
                await binding.invoke({'count': 3}, session_id='owned-stop')
            else:
                await agent._dispatch_tool_call(ToolCall(id='owned-stop', name=binding.name, arguments={'count': 3}),
                    {}, {binding.name}, [], 0, None, session_id='owned-stop')
        assert state is not None and state.completed and not state.checkpointed
        assert state.session_id == 'owned-stop'
    assert tool.calls == [{'count': 3}]


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['cancel', 'base-group', 'control-cause'])
async def test_named_original_terminal_evidence_is_never_a_normal_tool_reply(host, failure):
    import asyncio
    from kestrel_sovereign.execution_custody import ExecutionAuthorityError

    agent, backend = host
    cancel = asyncio.CancelledError('owned-caller-cancellation')
    control = RuntimeError('private-carrier')
    control.__cause__ = ExecutionAuthorityError('original-authority-lost')
    original = cancel if failure == 'cancel' else (
        BaseExceptionGroup('original-group', [cancel, KeyError('private-detail')])
        if failure == 'base-group' else control)
    class Failed(_Tool):
        async def execute(self, **args):
            self.calls.append(args)
            raise original
    binding = _mount(agent, Failed(visible=True))
    with pytest.raises(BaseException) as caught:
        await agent._make_inline_tool_executor('owned-terminal')(binding.name, {'count': 3})
    assert caught.value is original
    assert not await backend.fetch_all('SELECT * FROM a2a_tool_dispatches')


@pytest.mark.asyncio
async def test_known_inline_conflict_preserves_redaction_without_raw_error_or_stale_denial(host):
    from unittest.mock import AsyncMock
    from kestrel_sdk.resources import ResourceConflict
    from kestrel_sovereign.llm.codex_adapter import CodexAdapter

    agent, backend = host
    secret = 'private-conflict-detail'
    class Rejected(_Tool):
        async def execute(self, **args):
            self.calls.append(args)
            raise ResourceConflict(secret)
    tool = Rejected(visible=True)
    binding = _mount(agent, tool)
    agent.hooks_manager.register(_Policy(HookOutput.modify({'count': 7})))
    adapter = CodexAdapter()
    adapter._recent_security_decisions = AsyncMock(return_value=[{
        'tool': binding.name, 'decision': 'user_denied', 'user_choice': 'user_denied',
    }])
    executed = []
    handler = adapter._make_tool_call_handler(agent._make_inline_tool_executor('conflict-session'),
        't', frozenset({binding.name}), executed_log=executed)
    response = await handler({'threadId': 't', 'callId': 'conflict-call', 'name': binding.name,
                              'arguments': {'count': 3}})
    assert executed[0]['arguments'] == {'count': 7}
    assert secret not in json.dumps(response) + json.dumps(executed)
    assert 'precondition_failed' in response['contentItems'][0]['text']
    assert 'user_denied' not in response['contentItems'][0]['text']
    await agent._append_executed_tool_breadcrumbs([], executed, session_id='conflict-session')
    rows = await backend.fetch_all('SELECT args_redacted,error_class,error_message FROM a2a_tool_dispatches')
    assert len(rows) == 1 and json.loads(rows[0][0]) == {'count': 7}
    assert rows[0][1] == 'ResourceConflict' and secret not in rows[0][2]
    with pytest.raises(ResourceConflict) as caught:
        await agent.execute_named_tool(binding.name, {'count': 3}, session_id='public-conflict')
    assert str(caught.value) == secret
