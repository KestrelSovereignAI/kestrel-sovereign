"""Every transport shares typed, content-free effect failure semantics."""

import pytest
from kestrel_sdk.resources import ResourceOutcomeUnknown

from kestrel_sovereign.api_errors import tool_effect_api_error
from kestrel_sovereign.execution_custody import ExecutionAuthorityError, is_execution_control_error, is_tool_execution_terminal_error
from kestrel_sovereign.tool_effect_store import (
    ToolEffectAttemptClosed, ToolEffectAttemptSuperseded,
    ToolEffectOperationConflict, ToolEffectReconciliationRequired,
)
from kestrel_sovereign.tools.effect_failures import tool_effect_failure
from kestrel_sovereign.tools.effect_journal import (
    ToolEffectJournalUnavailable, ToolEffectPrivacyRefused,
)


@pytest.mark.parametrize('error,code,status,reconcile', [
    (ToolEffectReconciliationRequired({'unsettled': 1}), 'tool_effect_reconciliation_required', 409, True),
    (ToolEffectOperationConflict('private content'), 'tool_effect_operation_conflict', 409, True),
    (ToolEffectAttemptSuperseded('private content'), 'tool_effect_attempt_superseded', 409, True),
    (ToolEffectAttemptClosed('private content'), 'tool_effect_attempt_closed', 409, True),
    (ResourceOutcomeUnknown('private content'), 'tool_effect_outcome_unconfirmed', 502, True),
    (ToolEffectPrivacyRefused('private content'), 'tool_effect_privacy_mode_refused', 403, False),
    (ToolEffectJournalUnavailable('private content'), 'tool_effect_journal_unavailable', 503, False),
])
def test_transport_classification_does_not_reflect_private_failure_content(error, code, status, reconcile):
    failure = tool_effect_failure(error)
    public = tool_effect_api_error(error)
    assert failure.code == public.code == code
    assert failure.status_code == public.status_code == status
    assert failure.reconciliation_required is reconcile
    assert is_tool_execution_terminal_error(error) is reconcile
    assert public.message == failure.message
    assert public.details == failure.details
    assert 'private content' not in public.message
    assert public.headers is None, 'Uncertain effects must not advise automatic HTTP retry'


@pytest.mark.parametrize('kind', ['reconcile', 'conflict', 'superseded', 'closed'])
@pytest.mark.parametrize('shape', ['cause', 'group', 'mixed-group', 'mixed-group-reversed'])
def test_original_reconciliation_classification_survives_finite_delivery_composition(kind, shape):
    original = (ToolEffectReconciliationRequired({'unsettled': 1}) if kind == 'reconcile' else
        {'conflict': ToolEffectOperationConflict, 'superseded': ToolEffectAttemptSuperseded,
         'closed': ToolEffectAttemptClosed}[kind]('private original operation'))
    expected = tool_effect_failure(original)
    if shape == 'cause':
        error = ValueError('private observer diagnostic')
        error.__cause__ = original
    else:
        errors = ([ToolEffectPrivacyRefused('private sibling refusal'), original]
            if shape.startswith('mixed-group') else [original, ValueError('private observer diagnostic')])
        if shape == 'mixed-group-reversed':
            errors.reverse()
        error = ExceptionGroup('actual independent delivery outcomes', errors)
    failure = tool_effect_failure(error)
    assert failure == expected and is_tool_execution_terminal_error(error), (
        'Independent nonterminal failure downgraded original reconciliation disposition')
    assert not is_execution_control_error(error)
    assert 'private' not in tool_effect_api_error(error).message


@pytest.mark.parametrize('error', [ValueError('unrelated failure'), ExecutionAuthorityError('original control')])
def test_unclassified_or_control_errors_are_not_replaced(error):
    assert tool_effect_failure(error) is None
    assert tool_effect_api_error(error) is None


@pytest.mark.parametrize('shape', ['group', 'cause', 'class-hook', 'cause-hook', 'group-hook'])
def test_composite_unknown_has_one_content_free_http_and_sse_failure(shape):
    import json
    from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown
    from kestrel_sovereign.api_errors import tool_effect_api_error
    from kestrel_sovereign.llm.streaming_errors import bridge_sse_error_event
    from kestrel_sovereign.tools.effect_failures import tool_effect_failure

    class OpaqueConflict(ResourceConflict):
        @property
        def __class__(self):
            raise RuntimeError('Application class hook ran during transport classification')
    class OpaqueCause(ResourceConflict):
        @property
        def __cause__(self):
            raise RuntimeError('Application cause hook ran during transport classification')
    class OpaqueGroup(ExceptionGroup):
        @property
        def exceptions(self):
            raise RuntimeError('Application group hook ran during transport classification')

    original = ResourceOutcomeUnknown('PRIVATE_ORIGINAL_EFFECT_MARKER')
    if shape in {'group', 'group-hook'}:
        carrier = (OpaqueGroup if shape == 'group-hook' else ExceptionGroup)(
            'PRIVATE_ENCLOSING_MARKER', [original])
    else:
        carrier = (OpaqueConflict if shape == 'class-hook' else
            OpaqueCause if shape == 'cause-hook' else ResourceConflict)('PRIVATE_ENCLOSING_MARKER')
        BaseException.__cause__.__set__(carrier, original)
    failure = tool_effect_failure(carrier)
    public = tool_effect_api_error(carrier)
    event = bridge_sse_error_event(carrier)
    assert failure is not None and public is not None, 'Explicit unknown outcome lost its typed transport failure'
    assert failure.status_code == public.status_code == 502
    assert failure.code == public.code == 'tool_effect_outcome_unconfirmed'
    assert public.headers is None and failure.reconciliation_required
    assert 'tool_effect_outcome_unconfirmed' in event
    assert 'PRIVATE_' not in json.dumps([failure.message, public.message, event])


def test_transport_unknown_does_not_replace_control_or_borrow_handling_context():
    original = ResourceOutcomeUnknown('Original lost reply')
    control = ExceptionGroup('Original control and unknown', [original, ExecutionAuthorityError('Original authority')])
    assert tool_effect_failure(control) is None
    assert tool_effect_api_error(control) is None
    ordinary = ToolEffectOperationConflict('Original independent conflict')
    ordinary.__context__ = original
    assert tool_effect_failure(ordinary).code == 'tool_effect_operation_conflict'


def test_ordinary_application_class_claim_does_not_run_during_transport_classification():
    class OpaqueFailure(ValueError):
        @property
        def __class__(self):
            raise RuntimeError('Application class hook replaced the ordinary carrier')
    ordinary = OpaqueFailure('Original ordinary failure')
    assert tool_effect_failure(ordinary) is None
    assert tool_effect_api_error(ordinary) is None
