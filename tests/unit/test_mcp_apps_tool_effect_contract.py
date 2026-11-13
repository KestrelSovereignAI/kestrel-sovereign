"""Required App intent identity and shared retry-failure HTTP semantics."""

import json
import pytest
from pydantic import ValidationError
from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown

from kestrel_sovereign.mcp_apps import _Call, _error
from kestrel_sovereign.tool_effect_store import (
    ToolEffectOperationConflict, ToolEffectReconciliationRequired,
)


@pytest.mark.parametrize('operation_id', [None, '', 3, 'x' * 257])
def test_app_operation_identity_is_required_bounded_and_strict(operation_id):
    body = dict(name='app_tool', arguments={})
    if operation_id is not None:
        body['operation_id'] = operation_id
    with pytest.raises(ValidationError):
        _Call.model_validate(body)


def test_app_operation_identity_is_independent_of_json_rpc_id():
    assert _Call.model_validate(dict(name='app_tool', arguments={}, operation_id='owned-intent')).operation_id == 'owned-intent'
    with pytest.raises(ValidationError):
        _Call.model_validate(dict(name='app_tool', arguments={}, id='rpc-id'))


@pytest.mark.parametrize('error,status,code', [
    (ToolEffectReconciliationRequired({'unsettled': 1}), 409, 'tool_effect_reconciliation_required'),
    (ToolEffectOperationConflict('private input'), 409, 'tool_effect_operation_conflict'),
    (ResourceOutcomeUnknown('private result'), 502, 'tool_effect_outcome_unconfirmed'),
    (ResourceConflict('expired physical view'), 404, 'mcp_app_unavailable'),
])
def test_app_effect_errors_precede_the_generic_expired_view_branch(error, status, code):
    response = _error(error, False)
    assert response.status_code == status
    assert json.loads(response.body)['error']['code'] == code
    assert b'private input' not in response.body and b'private result' not in response.body
    assert 'Retry-After' not in response.headers


def test_failed_physical_delivery_never_becomes_a_replay_instruction():
    response = _error(ToolEffectReconciliationRequired({'unsettled': 1}), True)
    assert response.status_code == 503
    assert json.loads(response.body)['error']['code'] == 'mcp_app_outcome_unknown_do_not_replay'
