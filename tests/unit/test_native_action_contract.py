"""Native action captures use real schemas, detached identities and no I/O."""

import hashlib
import json
from types import SimpleNamespace

import pytest
from jsonschema.exceptions import SchemaError
from kestrel_sdk.tools import NativeAction, NativeActionContext
from kestrel_sovereign.tools.native_action import (
    NativeActionInvocation, action_audit, capture_native_actions, snapshot_action_context,
)
from kestrel_sovereign.features.security.args_summary import summarize_args
from kestrel_sovereign.security.input_guardrails import MAX_TOOL_ARG_LENGTH


async def handler(context, params):
    return params


@pytest.fixture
def context():
    digest = hashlib.sha256(b'owned-original').hexdigest()
    return NativeActionContext('update', 'tasks/update', 'owned-local', digest,
                              'mcp:owned', 'owned_tool', 'owned-batch', digest,
                              'service', digest, 2)


@pytest.fixture
def invocation(context):
    schema = {'type': 'object', 'required': ['inputResponses'], 'additionalProperties': False,
              'properties': {'inputResponses': {'type': 'object', 'additionalProperties':
                  {'type': 'integer'}}}}
    return NativeActionInvocation(NativeAction('update', 'tasks/update',
                                              json.dumps(schema).encode(), handler), context)


def test_action_snapshot_and_audit_never_alias_private_values(context):
    snapshot = snapshot_action_context(context)
    object.__setattr__(context, 'task_handle', 'caller-mutated')
    assert snapshot.task_handle == 'owned-local'
    audit = action_audit(snapshot, {'inputResponses': {'secret': 'private-value'}})
    assert 'private-value' not in json.dumps(audit)
    assert audit['parameter_names'] == ['inputResponses']
    assert action_audit(snapshot, None)['parameter_names'] == []


def test_static_actions_capture_original_schema_handler_and_membership(context):
    declaration = NativeAction('update', 'tasks/update', b'{"type":"object"}', handler)
    mapping = {'update': declaration}
    captured, = capture_native_actions(SimpleNamespace(native_actions=mapping))
    mapping.clear()
    object.__setattr__(declaration, 'schema_json', b'{"type":"null"}')
    assert captured.schema == {'type': 'object'} and captured.handler is handler
    assert capture_native_actions(object()) == ()


@pytest.mark.parametrize('declarations', [[], {'wrong': NativeAction('cancel', 'tasks/cancel', b'{}', handler)},
                                        {'cancel': object()}])
def test_invalid_mapping_fails_before_mount(declarations):
    with pytest.raises((ValueError, TypeError)):
        capture_native_actions(SimpleNamespace(native_actions=declarations))


@pytest.mark.parametrize('value', [{}, {'inputResponses': {'key': '3'}},
    {'inputResponses': {'key': True}}, {'inputResponses': {'key': float('nan')}},
    {'inputResponses': {'key': float('inf')}}, {'inputResponses': {1: 3}},
    {'inputResponses': {'key': object()}}, {'inputResponses': []}, [],
    {'inputResponses': {}, 'taskId': 'private-id'}])
def test_invalid_action_params_are_refused_without_coercion_or_private_error(invocation, value):
    with pytest.raises(ValueError, match='^Native action parameters do not match the captured schema$'):
        invocation.validate(value)


def test_valid_params_are_detached_and_preserve_json_types(invocation):
    value = {'inputResponses': {'key': 3}}
    snapshot = invocation.validate(value)
    value['inputResponses']['key'] = 4
    assert snapshot == {'inputResponses': {'key': 3}}
    assert type(snapshot['inputResponses']['key']) is int


def test_cyclic_json_is_refused(invocation):
    value = {}
    value['inputResponses'] = value
    with pytest.raises(ValueError, match='captured schema'):
        invocation.validate(value)


def test_external_schema_reference_never_fetches(context, monkeypatch):
    import socket
    def refuse(*args, **kwargs):
        pytest.fail('Native validation must not perform network I/O')
    monkeypatch.setattr(socket, 'socket', refuse)
    monkeypatch.setattr(socket, 'getaddrinfo', refuse)
    declaration = NativeAction('update', 'tasks/update', b'{"$ref":"https://unreachable.invalid/schema"}', handler)
    with pytest.raises(ValueError, match='captured schema'):
        NativeActionInvocation(declaration, context).validate({})


@pytest.mark.parametrize('number', [pytest.param(float('nan'), id='nan'),
    pytest.param(float('inf'), id='infinity'), pytest.param(float('-inf'), id='negative-infinity')])
def test_finite_json_is_required_even_for_an_unconstrained_action_schema(context, number):
    declaration = NativeAction('update', 'tasks/update', b'{"type":"object"}', handler)
    with pytest.raises(ValueError, match='captured schema'):
        NativeActionInvocation(declaration, context).validate({'number': number})


def test_invalid_json_schema_is_rejected_during_static_capture():
    declaration = NativeAction('update', 'tasks/update', b'{"type":"not-a-json-schema-type"}', handler)
    with pytest.raises(SchemaError):
        capture_native_actions(SimpleNamespace(native_actions={'update': declaration}))


def test_default_security_summary_preserves_all_native_identity_and_parameter_names(context):
    summary = summarize_args({'inputResponses': {'opaque-key': 'private-value'}}, native_action=context)
    assert not summary.endswith('...')
    try:
        projection = json.loads(summary)
    except json.JSONDecodeError as error:
        raise AssertionError('Native audit must serialize the complete JSON identity') from error
    assert projection['native_action'] == context.to_dict()
    assert projection['parameter_names'] == ['inputResponses']
    assert projection['parameters_type'] == 'dict'
    assert 'private-value' not in summary and 'opaque-key' not in summary


@pytest.mark.parametrize('arguments', [
    {'n' * (MAX_TOOL_ARG_LENGTH + 1): 1},
    {f'field-{index:05d}': 1 for index in range(MAX_TOOL_ARG_LENGTH)},
], ids=['oversize-name', 'many-names'])
def test_native_parameter_name_audit_is_bounded_without_cutting_identity(context, arguments):
    projection = json.loads(summarize_args(arguments, native_action=context))
    assert projection['native_action'] == context.to_dict()
    assert len(json.dumps(projection['parameter_names'])) <= MAX_TOOL_ARG_LENGTH
    assert projection['parameter_names_omitted'] > 0
