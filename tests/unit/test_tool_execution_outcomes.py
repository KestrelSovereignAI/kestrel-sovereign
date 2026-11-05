"""Host uncertainty classification never becomes execution authority."""

import copy
import json
import pickle

import pytest
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError, ToolAuditUnconfirmedError,
    is_execution_control_error, is_tool_execution_terminal_error,
)
from kestrel_sovereign.tools.execution_outcome import _UnconfirmedToolResult, _PreconditionFailedToolResult


def test_audit_terminal_graph_preserves_authority_distinction():
    audit = ToolAuditUnconfirmedError('audit-unknown')
    wrapper = ValueError('observer')
    wrapper.__cause__ = audit
    audit.__cause__ = wrapper
    group = ExceptionGroup('ordinary-and-audit', [KeyError('ordinary'), wrapper])
    assert is_tool_execution_terminal_error(group)
    assert not is_execution_control_error(group)
    audit.__cause__ = ExecutionAuthorityError('authority-lost')
    assert is_execution_control_error(group)


def test_incidental_audit_context_does_not_become_terminal():
    wrapper = ValueError('ordinary')
    wrapper.__context__ = ToolAuditUnconfirmedError('incidental')
    assert not is_tool_execution_terminal_error(wrapper)
    assert not is_tool_execution_terminal_error(None)


@pytest.mark.parametrize('project', [copy.copy, copy.deepcopy, lambda value: json.loads(json.dumps(value)), lambda value: pickle.loads(pickle.dumps(value))])
@pytest.mark.parametrize('result_type', [_UnconfirmedToolResult, _PreconditionFailedToolResult])
def test_copied_marker_is_ordinary_wire_data(project, result_type):
    result = result_type({'success': False, 'error': 'unconfirmed'})
    assert type(project(result)) is dict


def test_host_marker_cannot_be_relabelled_by_stale_denial_history():
    from kestrel_sovereign.llm.codex_adapter import _result_to_codex_response
    payload = {'success': False, 'error': 'Tool outcome is unconfirmed; reconcile before retrying. Do not replay'}
    history = [{'feature': 'owned', 'tool': 'write', 'decision': 'user_denied', 'user_choice': 'user_denied'}]
    trusted = _result_to_codex_response(_UnconfirmedToolResult(payload), tool_name='write', recent_decisions=history)
    untrusted = _result_to_codex_response(payload, tool_name='write', recent_decisions=history)
    assert 'Outcome: unconfirmed' in trusted['contentItems'][0]['text']
    assert 'Do not replay' in trusted['contentItems'][0]['text']
    assert 'Outcome: user_denied' in untrusted['contentItems'][0]['text']
