"""Every transport shares typed, content-free effect failure semantics."""

import pytest
from kestrel_sdk.resources import ResourceOutcomeUnknown

from kestrel_sovereign.api_errors import tool_effect_api_error
from kestrel_sovereign.execution_custody import ExecutionAuthorityError
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
    assert public.message == failure.message
    assert public.details == failure.details
    assert 'private content' not in public.message
    assert public.headers is None, 'Uncertain effects must not advise automatic HTTP retry'


@pytest.mark.parametrize('error', [ValueError('unrelated failure'), ExecutionAuthorityError('original control')])
def test_unclassified_or_control_errors_are_not_replaced(error):
    assert tool_effect_failure(error) is None
    assert tool_effect_api_error(error) is None
