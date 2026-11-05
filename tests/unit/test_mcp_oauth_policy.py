"""Host configuration cannot select vendor credentials or change client identity implicitly."""

import json

import pytest
from pydantic import ValidationError

from kestrel_sovereign.mcp_oauth import (
    MCPOAuthAuthorizationRequest, MCPOAuthCallback, MCPOAuthConnectionStatus, MCPOAuthHostPolicy,
)


@pytest.mark.parametrize("origin,canonical", [
    ("https://console.example", "https://console.example"),
    ("HTTPS://Console.Example:443/", "https://console.example"),
    ("https://console.example:8443/", "https://console.example:8443"),
])
def test_public_web_identity_uses_one_canonical_origin(origin, canonical):
    policy = MCPOAuthHostPolicy(console_origin=origin)
    assert policy.console_origin == canonical
    assert policy.redirect_uri == canonical + "/oauth/mcp/callback"
    assert policy.client_id == canonical + "/oauth/mcp/client-metadata.json"


@pytest.mark.parametrize("origin", [
    "http://console.example", "https://localhost", "https://tenant.localhost",
    "https://127.0.0.1", "https://[::1]", "https://192.168.1.2",
    "https://user:password@console.example", "https://@console.example",
    "https://console.example/path", "https://console.example?x=y",
    "https://console.example?", "https://console.example#", "https://console.example#fragment",
    " https://console.example", "https://console.example\n",
])
def test_web_client_identity_refuses_an_ambiguous_or_nonpublic_origin(origin):
    with pytest.raises(ValidationError):
        MCPOAuthHostPolicy(console_origin=origin)


def test_cimd_document_is_a_fresh_public_web_identity():
    policy = MCPOAuthHostPolicy(console_origin="https://console.example", client_name="Owned host")
    expected = {
        "client_id": "https://console.example/oauth/mcp/client-metadata.json",
        "client_name": "Owned host",
        "redirect_uris": ["https://console.example/oauth/mcp/callback"],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"], "token_endpoint_auth_method": "none", "application_type": "web",
    }
    assert policy.client_document() == expected
    copied = policy.client_document()
    copied["redirect_uris"].append("https://other.example/callback")
    assert policy.client_document() == expected
    with pytest.raises(ValidationError):
        policy.console_origin = "https://other.example"


@pytest.mark.parametrize("settings", [
    {"max_operations": 0}, {"max_operations": True}, {"max_operations": 257},
    {"max_operations": 1, "max_operations_per_principal": 2},
    {"max_operations_per_principal": 17}, {"authorization_seconds": 1801},
    {"network_seconds": float("nan")}, {"operation_seconds": float("inf")},
    {"lease_seconds": 0}, {"response_max_bytes": 1_048_577}, {"max_token_bytes": 0},
    {"max_request_bytes": -1}, {"client_secret": "synthetic-forbidden-secret"},
    {"client_id": "https://different.example/client.json"},
    {"client_name": ""}, {"client_name": "a\nb"},
])
def test_host_budgets_and_identity_are_validated_before_activation(settings):
    with pytest.raises(ValidationError):
        MCPOAuthHostPolicy(console_origin="https://console.example", **settings)


def test_unconfigured_host_has_no_oauth_identity():
    assert MCPOAuthHostPolicy.from_environment({}) is None


def test_host_policy_reads_configured_budgets_without_mutating_environment():
    environ = {
        "KESTREL_MCP_OAUTH_CONSOLE_ORIGIN": "https://console.example",
        "KESTREL_MCP_OAUTH_POLICY": json.dumps({"network_seconds": 7.5, "max_operations": 4}),
    }
    original = dict(environ)
    policy = MCPOAuthHostPolicy.from_environment(environ)
    assert policy.network_seconds == 7.5 and policy.max_operations == 4
    assert environ == original


@pytest.mark.parametrize("environ", [
    {"KESTREL_MCP_OAUTH_POLICY": "{}"},
    {"KESTREL_MCP_OAUTH_CONSOLE_ORIGIN": "https://console.example", "KESTREL_MCP_OAUTH_POLICY": "invalid"},
    {"KESTREL_MCP_OAUTH_CONSOLE_ORIGIN": "https://console.example", "KESTREL_MCP_OAUTH_POLICY": "[]"},
    {"KESTREL_MCP_OAUTH_CONSOLE_ORIGIN": "https://console.example",
     "KESTREL_MCP_OAUTH_POLICY": '{"console_origin":"https://other.example"}'},
])
def test_malformed_or_conflicting_configuration_fails_before_serving_identity(environ):
    with pytest.raises(ValueError):
        MCPOAuthHostPolicy.from_environment(environ)


def test_rejected_secret_settings_are_not_rendered_in_configuration_errors():
    with pytest.raises(ValidationError) as failure:
        MCPOAuthHostPolicy(console_origin="https://console.example",
                           client_secret="synthetic-private-value-not-for-logs")
    assert "synthetic-private-value-not-for-logs" not in str(failure.value)


@pytest.mark.parametrize("scopes", [[], [""], ["read", "read"], ["read write"],
    ['read"write'], [r"read\write"], ["read\twrite"], ["é"], ["x" * 257]])
def test_authorization_requires_bounded_distinct_rfc_scope_tokens(scopes):
    with pytest.raises(ValidationError):
        MCPOAuthAuthorizationRequest(server_url="https://remote.example/mcp", scopes=scopes)


@pytest.mark.parametrize("endpoint", ["http://remote.example/mcp", "https://@remote.example/mcp",
    "https://user:secret@remote.example/mcp", "https://remote.example/mcp#", "https://remote.example/mcp#part"])
def test_authorization_refuses_credentials_and_fragments_before_runtime_io(endpoint):
    with pytest.raises(ValidationError):
        MCPOAuthAuthorizationRequest(server_url=endpoint, scopes=["read"])


def test_valid_scope_and_resource_query_are_preserved_without_a_private_protocol_mapping():
    value = MCPOAuthAuthorizationRequest(server_url="https://remote.example/mcp?workspace=one",
                                        scopes=["read:documents", "write:notes"])
    assert value.server_url == "https://remote.example/mcp?workspace=one"
    assert value.scopes == ["read:documents", "write:notes"]


@pytest.mark.parametrize("extra", [{}, {"code": "owned-code", "error": "access_denied"},
    {"code": ""}, {"code": "owned-code", "unexpected": "value"}])
def test_callback_has_exactly_one_bound_outcome(extra):
    with pytest.raises(ValidationError):
        MCPOAuthCallback(state="owned-state-with-entropy", iss="https://issuer.example", **extra)


def test_callback_repr_does_not_render_code_or_state():
    value = MCPOAuthCallback(state="owned-state-with-entropy", code="owned-one-use-code",
                            iss="https://issuer.example")
    assert value.state == "owned-state-with-entropy" and value.code == "owned-one-use-code"
    assert value.code not in repr(value) and value.state not in repr(value)


def connection_status():
    return {"connection_id": "owned-connection", "server_url": "https://remote.example/mcp",
        "issuer": "https://issuer.example", "resource": "https://remote.example/mcp",
        "scopes": ["read"], "state": "authorized", "connected": False}


@pytest.mark.parametrize("secret_field", ["access_token", "refresh_token", "client_secret"])
def test_connection_status_cannot_accidentally_publish_credential_fields(secret_field):
    with pytest.raises(ValidationError) as failure:
        MCPOAuthConnectionStatus.model_validate(connection_status() | {secret_field: "owned-private-value"})
    assert "owned-private-value" not in str(failure.value)


@pytest.mark.parametrize("changes", [{"state": "connected"}, {"connected": True}, {"state": "invented"}])
def test_connection_status_requires_an_exact_known_disposition(changes):
    with pytest.raises(ValidationError):
        MCPOAuthConnectionStatus.model_validate(connection_status() | changes)


@pytest.mark.parametrize('origin', ['https://localhost.', 'https://x.localhost.'])
def test_trailing_dot_loopback_origin_is_not_a_public_web_identity(origin):
    with pytest.raises(ValidationError):
        MCPOAuthHostPolicy(console_origin=origin)


def test_nested_url_validation_never_echoes_malformed_origin_input():
    with pytest.raises(ValidationError) as caught:
        MCPOAuthHostPolicy(console_origin='https://[owned-private-origin-marker')
    assert 'owned-private-origin-marker' not in str(caught.value)


def test_direct_origin_validator_has_content_free_parse_errors():
    with pytest.raises(ValueError) as caught:
        MCPOAuthHostPolicy.public_console_origin('https://[owned-private-validator-marker')
    assert 'owned-private-validator-marker' not in str(caught.value)
