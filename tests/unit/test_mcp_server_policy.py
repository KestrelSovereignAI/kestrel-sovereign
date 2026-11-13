"""Server configuration is explicit and never implies an agent grant."""

import json

import pytest
from pydantic import ValidationError

from kestrel_sovereign.mcp_server.policy import MCPServerPolicy


def configured(**changes):
    value = {"console_origin": "https://kestrel.example", "clients": [{
        "client_id": "owned-provider", "client_name": "Owned provider",
        "redirect_uris": ["https://provider.example/oauth/callback?flow=owned"],
        "client_secret": "owned-fixture-client-secret"}]}
    value.update(changes)
    return value


def test_server_is_disabled_only_when_configuration_is_absent():
    assert MCPServerPolicy.from_environment({}) is None
    for raw in ["", "null", "{}", "[]", "invalid-json"]:
        with pytest.raises(ValidationError):
            MCPServerPolicy.from_environment({"KESTREL_MCP_SERVER": raw})


def test_registered_client_and_origin_keep_their_explicit_identity():
    policy = MCPServerPolicy.from_environment({"KESTREL_MCP_SERVER": json.dumps(configured())})
    assert policy.console_origin == "https://kestrel.example"
    client, = policy.clients
    assert client.scopes == ("memory:read",)
    assert client.redirect_uris == ("https://provider.example/oauth/callback?flow=owned",)
    assert client.client_secret.get_secret_value() == "owned-fixture-client-secret"
    assert "owned-fixture-client-secret" not in repr(policy)
    assert "owned-fixture-client-secret" not in policy.model_dump_json()


@pytest.mark.parametrize("change", [
    {"console_origin": "http://kestrel.example"},
    {"console_origin": "https://kestrel.example/private"},
    {"console_origin": "https://localhost"},
    {"clients": []},
    {"max_pending_per_agent": 0},
    {"access_token_seconds": True},
    {"authorization_seconds": float("inf")},
    {"access_token_seconds": 3600, "refresh_token_seconds": 60},
    {"implicit_dcr": True},
])
def test_invalid_or_ambiguous_server_configuration_fails(change):
    with pytest.raises(ValidationError):
        MCPServerPolicy.model_validate_json(json.dumps(configured(**change)))


@pytest.mark.parametrize("change", [
    {"client_id": " owned-provider"},
    {"client_name": "bad\nname"},
    {"client_secret": None},
    {"token_endpoint_auth_method": "none"},
    {"redirect_uris": ["http://provider.example/callback"]},
    {"redirect_uris": ["https://user:password@provider.example/callback"]},
    {"redirect_uris": ["https://provider.example/callback#fragment"]},
    {"redirect_uris": ["https://provider.example/callback#"]},
    {"redirect_uris": ["https://provider.example:invalid/callback"]},
    {"redirect_uris": ["https://provider.example/callback", "https://provider.example/callback"]},
    {"scopes": ["memory:read", "memory:read"]},
    {"scopes": ["sovereign"]},
])
def test_client_cannot_gain_implicit_authority_or_redirect_credentials(change):
    value = configured()
    value["clients"][0].update(change)
    with pytest.raises(ValidationError):
        MCPServerPolicy.model_validate_json(json.dumps(value))


def test_public_client_is_explicit_and_duplicate_client_ids_are_refused():
    value = configured()
    value["clients"][0].update(token_endpoint_auth_method="none", client_secret=None)
    policy = MCPServerPolicy.model_validate_json(json.dumps(value))
    assert policy.clients[0].token_endpoint_auth_method == "none"
    value["clients"].append(dict(value["clients"][0]))
    with pytest.raises(ValidationError, match="client IDs must be unique"):
        MCPServerPolicy.model_validate_json(json.dumps(value))


@pytest.mark.parametrize("public", [False, True])
def test_actual_official_sdk_receives_only_the_registered_client_information(public):
    from mcp.shared.auth import OAuthClientInformationFull

    value = configured()
    value["clients"][0]["scopes"] = ["memory:read", "conversation:write"]
    if public:
        value["clients"][0].update(token_endpoint_auth_method="none", client_secret=None)
    client = MCPServerPolicy.model_validate_json(json.dumps(value)).clients[0]
    information = client.oauth_information()
    assert isinstance(information, OAuthClientInformationFull)
    assert information.client_id == client.client_id
    assert information.scope == "memory:read conversation:write"
    assert information.token_endpoint_auth_method == client.token_endpoint_auth_method
    assert information.client_secret == (None if public else "owned-fixture-client-secret")
    assert [str(uri) for uri in information.redirect_uris] == list(client.redirect_uris)
    assert information.grant_types == ["authorization_code", "refresh_token"]
    assert information.response_types == ["code"]


@pytest.mark.parametrize("uri", ["https://bücher.example/callback",
    "https://PROVIDER.example/callback"])
def test_registration_rejects_redirect_identity_changed_by_native_sdk(uri):
    value = configured()
    value["clients"][0]["redirect_uris"] = [uri]
    with pytest.raises(ValidationError, match="redirect identity"):
        MCPServerPolicy.model_validate_json(json.dumps(value))


def test_native_sdk_preserves_a_registered_empty_path():
    value = configured()
    value["clients"][0]["redirect_uris"] = ["https://provider.example"]
    client = MCPServerPolicy.model_validate_json(json.dumps(value)).clients[0]
    assert [str(uri) for uri in client.oauth_information().redirect_uris] == ["https://provider.example"]
