"""Sealed private metadata and actual ambient namespace attribution, no authority."""

from dataclasses import asdict, FrozenInstanceError
import hashlib
import json

import pytest

from kestrel_sovereign.a2a.media import encode_inline_json
from kestrel_sovereign.agent.invocation import invocation_scope, request_provenance
from kestrel_sovereign.auth import CallerContext, caller_context_scope
from kestrel_sovereign.mcp_task_submission import (
    MCPTaskDelivery, MCPTaskInteractiveOrigin, MCPTaskOAuthLocator, MCPTaskServiceOrigin,
    MCPTaskSubmission, decode_origin,
)
from kestrel_sovereign.tools.effect_invocation import current_service_operation_attribution
from kestrel_sovereign.tools.effect_journal import ToolEffectJournalUnavailable


LOCATOR = MCPTaskOAuthLocator('original-owner', 'original-connection', 'a' * 64, 'b' * 64)


def test_original_arguments_contract_and_delivery_are_sealed_without_retained_aliases():
    arguments = {'private-answer': {'city': 'original-private-city'}}
    contract = b'{"tool":{"name":"original-private-tool"}}'
    delivery = MCPTaskDelivery('app', 'original-session', 'original-view')
    submission = MCPTaskSubmission.seal(arguments=arguments, contract=contract, delivery=delivery,
        authorization=LOCATOR)
    arguments['private-answer']['city'] = 'producer-replacement'
    submission.arguments['private-answer']['city'] = 'reader-replacement'
    assert submission.arguments == {'private-answer': {'city': 'original-private-city'}}
    assert submission.contract == contract and submission.delivery == delivery
    assert submission.authorization == LOCATOR
    assert MCPTaskSubmission(submission.canonical_json) == submission
    assert 'private' not in repr(submission) and 'original-view' not in repr(delivery)
    with pytest.raises(FrozenInstanceError):
        submission.canonical_json = b'changed'


@pytest.mark.parametrize('origin', [
    MCPTaskInteractiveOrigin('a' * 64, 'real-session'),
    MCPTaskServiceOrigin('signal', 'b' * 64),
    MCPTaskServiceOrigin('caller', 'c' * 64, 'real-conversation'),
])
def test_tagged_origins_round_trip_without_raw_service_or_caller_identity(origin):
    assert decode_origin(asdict(origin)) == origin
    assert 'real-session' not in repr(origin)
    assert 'real-conversation' not in repr(origin)


@pytest.mark.parametrize('document', [None, {}, [],
    {'kind': 'interactive', 'principal_digest': 'raw-operator', 'session_id': 'real-session'},
    {'kind': 'interactive', 'principal_digest': 'a' * 64, 'session_id': 'mcp-task-service'},
    {'kind': 'service', 'service_kind': 'anonymous', 'namespace_digest': 'a' * 64, 'session_id': None},
    {'kind': 'service', 'service_kind': 'internal', 'namespace_digest': None, 'session_id': None},
    {'kind': 'service', 'service_kind': 'internal', 'namespace_digest': 'a' * 64, 'session_id': 'scheduler'},
    {'kind': 'interactive', 'principal_digest': 'a' * 64, 'session_id': ' scheduler '},
    {'kind': 'service', 'service_kind': 'internal', 'namespace_digest': 'a' * 64, 'session_id': None, 'extra': True},
])
def test_untyped_raw_anonymous_or_fabricated_conversation_origins_refuse(document):
    with pytest.raises((TypeError, ValueError)):
        decode_origin(document)


@pytest.mark.parametrize('kind,session,view', [
    ('internal', 'invented-session', None), ('internal', None, 'invented-view'),
    ('chat', None, None), ('chat', 'real-session', 'unexpected-view'),
    ('app', 'real-session', None), ('app', 'scheduler', 'original-view'),
    ('unknown', None, None),
])
def test_delivery_requires_a_real_conversation_and_only_apps_identify_views(kind, session, view):
    with pytest.raises((TypeError, ValueError)):
        MCPTaskDelivery(kind, session, view)


@pytest.mark.parametrize('arguments,contract', [
    ([], b'original'), ({'private-key': object()}, b'original'),
    ({'private-key': float('nan')}, b'original'), ({}, b'\xff'), ({}, b''),
])
def test_invalid_submission_does_not_echo_private_arguments(arguments, contract):
    with pytest.raises(ValueError) as caught:
        MCPTaskSubmission.seal(arguments=arguments, contract=contract,
            delivery=MCPTaskDelivery('internal'), authorization=LOCATOR)
    assert 'private-key' not in str(caught.value)


@pytest.mark.parametrize('kind', ['mutable', 'text', 'none'])
def test_submission_requires_immutable_bytes(kind):
    raw = MCPTaskSubmission.seal(arguments={}, contract=b'original',
        delivery=MCPTaskDelivery('internal'), authorization=LOCATOR).canonical_json
    value = {'mutable': bytearray(raw), 'text': raw.decode(), 'none': None}[kind]
    with pytest.raises(TypeError, match='immutable'):
        MCPTaskSubmission(value)


def test_direct_submission_bytes_require_the_exact_finite_canonical_shape():
    submission = MCPTaskSubmission.seal(arguments={}, contract=b'original',
        delivery=MCPTaskDelivery('internal'), authorization=LOCATOR)
    value = json.loads(submission.canonical_json)
    malformed = [b'bad-private-json', b'\xff', submission.canonical_json + b' ',
        encode_inline_json({**value, 'extra': 'private-extra'}, sort_keys=True),
        encode_inline_json({**value, 'authorization': {**value['authorization'], 'slot_id': 'raw-id'}}, sort_keys=True),
        encode_inline_json({**value, 'authorization': {**value['authorization'], 'kind': 'other'}}, sort_keys=True),
        json.dumps({**value, 'arguments': {'private': float('nan')}}, sort_keys=True).encode()]
    for raw in malformed:
        with pytest.raises(ValueError, match='Invalid private MCP Task submission') as caught:
            MCPTaskSubmission(raw)
        assert 'private-json' not in str(caught.value) and 'private-extra' not in str(caught.value)


@pytest.mark.parametrize('source,actor,kind,namespace', [
    ('signal', None, 'signal', ['signal', 'owned-original-locator']),
    ('http_request', 'owned-service', 'service', ['service', 'owned-service']),
    ('workflow', None, 'invocation', ['workflow', 'owned-original-locator']),
])
def test_service_attribution_hashes_actual_ambient_namespace_without_exposing_it(source, actor, kind, namespace):
    provenance = request_provenance(source_kind=source, actor=actor, source_locator='owned-original-locator')
    with caller_context_scope(None), invocation_scope('owned-attribution', provenance=provenance):
        actual = current_service_operation_attribution()
    assert actual == (kind, hashlib.sha256(encode_inline_json(
        ['mcp-task-service-origin-v1', namespace], sort_keys=True)).hexdigest())
    assert 'owned-original-locator' not in repr(actual) and 'owned-service' not in repr(actual)


def test_authenticated_caller_attribution_uses_the_canonical_current_namespace():
    with caller_context_scope(CallerContext.authenticated('owned-original-subject')):
        kind, digest = current_service_operation_attribution()
    expected = ['caller', 'authenticated', 'oauth_session', 'owned-original-subject']
    assert kind == 'caller'
    assert digest == hashlib.sha256(encode_inline_json(
        ['mcp-task-service-origin-v1', expected], sort_keys=True)).hexdigest()


def test_anonymous_http_requests_cannot_be_minted_as_service_task_origins():
    provenance = request_provenance(source_kind='http_request', source_locator='/owned-anonymous')
    with caller_context_scope(CallerContext.anonymous()), invocation_scope('anonymous-task', provenance=provenance):
        with pytest.raises(ToolEffectJournalUnavailable, match='Anonymous'):
            current_service_operation_attribution()


def test_bound_anonymous_caller_without_provenance_cannot_be_attributed_as_internal():
    with caller_context_scope(CallerContext.anonymous()), invocation_scope('anonymous-no-provenance'):
        with pytest.raises(ToolEffectJournalUnavailable, match='Anonymous'):
            current_service_operation_attribution()


@pytest.mark.parametrize('source,actor', [('signal', None), ('http_request', 'owned-service')])
def test_bound_anonymous_role_cannot_be_promoted_by_service_or_signal_attribution(source, actor):
    provenance = request_provenance(source_kind=source, actor=actor, source_locator='owned-source')
    with caller_context_scope(CallerContext.anonymous()), invocation_scope('bound-anonymous', provenance=provenance):
        with pytest.raises(ToolEffectJournalUnavailable, match='Anonymous'):
            current_service_operation_attribution()
    with caller_context_scope(None), invocation_scope('actual-service', provenance=provenance):
        assert current_service_operation_attribution()[0] == ('signal' if source == 'signal' else 'service')


def test_schema_three_origin_policy_changes_require_a_stored_format_decision():
    from kestrel_sovereign.features.security.hooks import NON_INTERACTIVE_SESSION_IDS
    from kestrel_sovereign.tools.effect_invocation import SERVICE_OPERATION_KINDS

    message = 'Changing origin policy requires a stored-format decision before retained Task rows are revalidated'
    assert NON_INTERACTIVE_SESSION_IDS == {'scheduler', 'feature-lifecycle', 'mcp-task-service'}, message
    assert SERVICE_OPERATION_KINDS == {'signal', 'caller', 'service', 'internal', 'invocation'}, message
