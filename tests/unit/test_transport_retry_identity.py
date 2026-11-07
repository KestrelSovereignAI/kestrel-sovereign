"""One retry identity across literal body aliases and the header wire form."""

from itertools import combinations
from types import SimpleNamespace

import pytest
from starlette.requests import Request

from kestrel_sovereign.agent.invocation import (
    invocation_id_response_header, resolve_transport_invocation_id,
)
from kestrel_sovereign.api_errors import ApiHTTPException
from kestrel_sovereign.endpoints.agent_helpers import resolve_request_invocation_id


FIELDS = ("request_id", "invocation_id", "id", "header")
SUBSETS = [subset for size in range(1, 5) for subset in combinations(FIELDS, size)]


def inputs(values, *, object_body):
    body = {key: value for key, value in values.items() if key != "header"}
    header = values.get("header")
    request = Request({"type": "http", "headers": [] if header is None else [(b"x-request-id", header.encode("ascii"))]})
    return (SimpleNamespace(**body) if object_body else body), header, request


@pytest.mark.parametrize("object_body", [False, True])
@pytest.mark.parametrize("fields", SUBSETS)
def test_matching_aliases_preserve_literal_unicode_and_one_pass_header_decoding(object_body, fields):
    identity = "owned 私%2F☃"
    values = {field: invocation_id_response_header(identity) if field == "header" else identity for field in fields}
    body, header, request = inputs(values, object_body=object_body)
    assert resolve_transport_invocation_id(body, header) == identity
    assert resolve_request_invocation_id(request, body) == identity


@pytest.mark.parametrize("object_body", [False, True])
@pytest.mark.parametrize("first,second", list(combinations(FIELDS, 2)))
def test_disagreeing_aliases_are_rejected_with_one_content_free_http_code(object_body, first, second):
    body, header, request = inputs({first: "owned-private-first", second: "owned-private-second"}, object_body=object_body)
    with pytest.raises(ValueError):
        resolve_transport_invocation_id(body, header)
    with pytest.raises(ApiHTTPException) as error:
        resolve_request_invocation_id(request, body)
    assert error.value.status_code == 400
    assert error.value.code == "conflicting_request_id"
    assert "owned-private" not in error.value.message


@pytest.mark.parametrize("object_body", [False, True])
@pytest.mark.parametrize("header", ["%", "%E2", "contains/reserved", "%FF"])
def test_present_malformed_header_cannot_be_hidden_by_a_valid_body_alias(object_body, header):
    body, _, request = inputs({"request_id": "owned-body", "header": header}, object_body=object_body)
    with pytest.raises(ValueError):
        resolve_transport_invocation_id(body, header)
    with pytest.raises(ApiHTTPException) as error:
        resolve_request_invocation_id(request, body)
    assert error.value.status_code == 400 and error.value.code == "invalid_request_id"


def test_absent_identity_creates_distinct_server_owned_intents():
    first = resolve_transport_invocation_id(None)
    second = resolve_transport_invocation_id({})
    assert first != second
