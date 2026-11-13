"""Original SQL owns code consumption, opaque token rotation and replay revocation."""

import asyncio
import json

import pytest
from mcp.server.auth.provider import (
    AuthorizationCode,
    AuthorizationParams,
    TokenError,
    principal_components,
)

from kestrel_sovereign.mcp_server.consent import MCPConsentAuthority
from kestrel_sovereign.mcp_server.grants import MCPAuthorizationGrants
from kestrel_sovereign.mcp_server.policy import MCPServerPolicy
from kestrel_sovereign.mcp_server.records import MCPRecordKind
from kestrel_sovereign.storage.db.interface import TransactionError
from tests.integration.test_native_mcp_consent import API_KEY, RESOURCE
from tests.integration.test_native_mcp_consent import consent_http as _consent_http
from tests.integration.test_native_mcp_consent import native_pg as _native_pg

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]
consent_http = _consent_http
native_pg = _native_pg


async def approved_code(boundary):
    client = boundary[0]
    response = await client.post("/api/owned-mcp-consent?mode=issue-code", headers={"x-api-key": API_KEY})
    assert response.status_code == 200, response.text
    return AuthorizationCode.model_validate(response.json())


async def pair(boundary):
    code = await approved_code(boundary)
    token = await boundary[6].exchange(MCPRecordKind.CODE, code, resource=RESOURCE)
    return code, token


async def test_original_consent_commits_only_hashes_and_encrypted_native_models(consent_http):
    _client, pending, handle, _retained, _keys, _clock, grants, _app = consent_http
    code, token = await pair(consent_http)
    assert await pending.read(handle) is None
    rows = await pending.backend.fetch_all("SELECT secret_hash, ciphertext FROM mcp_server_oauth_credentials")
    for row in rows:
        assert all(secret not in " ".join(row) for secret in (code.code, token.access_token, token.refresh_token))
        assert "owned-provider" not in row[1] and "memory:read" not in row[1]
    access = await grants.load(MCPRecordKind.ACCESS, token.access_token, resource=RESOURCE)
    assert access.resource == RESOURCE and access.subject == code.subject
    assert access.claims == {"iss": RESOURCE + "/oauth"} and access.scopes == ["memory:read"]
    assert principal_components(access) == ("owned-provider", RESOURCE + "/oauth", code.subject)
    assert await grants.load(MCPRecordKind.ACCESS, "untrusted-malformed-token", resource=RESOURCE) is None


@pytest.mark.parametrize("kind", [MCPRecordKind.CODE, MCPRecordKind.REFRESH])
async def test_exact_resource_refusal_preserves_the_original_credential(consent_http, kind):
    grants = consent_http[6]
    code = await approved_code(consent_http)
    credential = code
    if kind is MCPRecordKind.REFRESH:
        token = await grants.exchange(MCPRecordKind.CODE, code, resource=RESOURCE)
        credential = await grants.load(kind, token.refresh_token, resource=RESOURCE, client_id=code.client_id)
    with pytest.raises(TokenError) as error:
        await grants.exchange(kind, credential, resource=RESOURCE + "/other")
    assert error.value.error == "invalid_target"
    secret = credential.code if kind is MCPRecordKind.CODE else credential.token
    assert await grants.load(kind, secret, resource=RESOURCE, client_id=credential.client_id) is not None


async def test_refresh_rotates_opaque_tokens_and_replay_revocation_survives_commit(consent_http):
    grants = consent_http[6]
    code, first = await pair(consent_http)
    refresh = await grants.load(MCPRecordKind.REFRESH, first.refresh_token,
        resource=RESOURCE, client_id=code.client_id)
    second = await grants.exchange(MCPRecordKind.REFRESH, refresh, resource=RESOURCE)
    assert first.access_token != second.access_token and first.refresh_token != second.refresh_token
    assert await grants.load(MCPRecordKind.ACCESS, second.access_token, resource=RESOURCE) is not None
    assert await grants.load(MCPRecordKind.REFRESH, first.refresh_token,
        resource=RESOURCE, client_id=code.client_id) is None
    assert await grants.load(MCPRecordKind.ACCESS, second.access_token, resource=RESOURCE) is None, (
        "Replayed refresh token did not durably revoke its family")
    assert await grants.load(MCPRecordKind.REFRESH, second.refresh_token,
        resource=RESOURCE, client_id=code.client_id) is None


async def test_two_native_code_exchanges_cannot_publish_two_token_pairs(consent_http):
    grants = consent_http[6]
    code = await approved_code(consent_http)
    results = await asyncio.gather(grants.exchange(MCPRecordKind.CODE, code, resource=RESOURCE),
        grants.exchange(MCPRecordKind.CODE, code, resource=RESOURCE), return_exceptions=True)
    assert sum(isinstance(result, TokenError) for result in results) == 1
    token, = [result for result in results if not isinstance(result, BaseException)]
    assert await grants.load(MCPRecordKind.ACCESS, token.access_token, resource=RESOURCE) is None, (
        "Concurrent code replay did not revoke the committed original token pair")


async def test_consumed_code_replay_revokes_tokens_even_after_code_expiry(consent_http):
    grants, clock = consent_http[6], consent_http[5]
    code, token = await pair(consent_http)
    clock[0] = code.expires_at + 1
    assert await grants.load(MCPRecordKind.CODE, code.code, resource=RESOURCE, client_id=code.client_id) is None
    assert await grants.load(MCPRecordKind.ACCESS, token.access_token, resource=RESOURCE) is None


@pytest.mark.parametrize("change", ["client", "secret", "scope", "issuer"])
async def test_current_sovereign_key_and_client_registration_control_durable_reads(consent_http, change):
    pending, keys, grants = consent_http[1], consent_http[4], consent_http[6]
    _code, token = await pair(consent_http)
    keys[0] = "owned-rotated-sovereign-key"
    assert await grants.load(MCPRecordKind.ACCESS, token.access_token, resource=RESOURCE) is None
    keys[0] = API_KEY
    config = json.loads(pending.policy.model_dump_json())
    config["clients"][0]["client_secret"] = "owned-secret"
    if change == "client":
        config["clients"][0]["client_id"] = "other-provider"
    elif change == "secret":
        config["clients"][0]["client_secret"] = "owned-rotated-client-secret"
    elif change == "scope":
        config["clients"][0]["scopes"] = ["memory:write"]
    policy = MCPServerPolicy.model_validate_json(json.dumps(config))
    other_pending = type(pending)(pending.backend, pending.agent_id, pending.resource_url, policy, clock=pending.clock)
    authority = MCPConsentAuthority(pending.agent_id, RESOURCE, policy,
        require_current=grants.consent_authority.require_current_host,
        sovereign_key=lambda: keys[0], clock=pending.clock)
    other = MCPAuthorizationGrants(other_pending, grants.issuer + ("/other" if change == "issuer" else ""),
        consent_authority=authority)
    assert await other.load(MCPRecordKind.ACCESS, token.access_token, resource=RESOURCE) is None


async def test_mutated_loaded_credential_cannot_expand_approved_scopes(consent_http):
    grants = consent_http[6]
    code = await approved_code(consent_http)
    code.scopes.append("memory:write")
    with pytest.raises(TokenError):
        await grants.exchange(MCPRecordKind.CODE, code, resource=RESOURCE)
    row = await grants.backend.fetch_one("SELECT COUNT(*) FROM mcp_server_oauth_credentials")
    assert row == (1,), "A mutated loaded code minted tokens outside the original consent"


@pytest.mark.parametrize("kind", [MCPRecordKind.CODE, MCPRecordKind.REFRESH])
async def test_enclosing_sql_transaction_cannot_publish_tentative_oauth_tokens(consent_http, kind):
    grants = consent_http[6]
    code = await approved_code(consent_http)
    credential = code
    if kind is MCPRecordKind.REFRESH:
        token = await grants.exchange(MCPRecordKind.CODE, code, resource=RESOURCE)
        credential = await grants.load(kind, token.refresh_token, resource=RESOURCE, client_id=code.client_id)
    published, refused = None, None
    with pytest.raises(TransactionError, match="owned-outer-rollback"):
        async with grants.backend.transaction():
            try:
                published = await grants.exchange(kind, credential, resource=RESOURCE)
            except RuntimeError as error:
                refused = error
            raise ValueError("owned-outer-rollback")
    assert published is None and refused is not None and "top-level" in str(refused), (
        "An enclosing SQL transaction published OAuth tokens before its original COMMIT")


async def test_code_issuance_requires_the_original_top_level_commit(consent_http):
    client, pending, handle, _retained, _keys, _clock, grants, _app = consent_http
    response = await client.post("/api/owned-mcp-consent?mode=nested-issue-code", headers={"x-api-key": API_KEY})
    assert response.status_code == 403 and "top-level" in response.json()["error"]
    assert await pending.read(handle) is not None
    assert await grants.backend.fetch_one("SELECT COUNT(*) FROM mcp_server_oauth_grants") == (0,)


@pytest.mark.parametrize("operation", ["load", "revoke"])
async def test_replay_and_revocation_cannot_join_an_uncommitted_outer_owner(consent_http, operation):
    grants = consent_http[6]
    code, token = await pair(consent_http)
    async with grants.backend.transaction():
        with pytest.raises(RuntimeError, match="top-level"):
            if operation == "load":
                await grants.load(MCPRecordKind.CODE, code.code, resource=RESOURCE, client_id=code.client_id)
            else:
                await grants.revoke(MCPRecordKind.ACCESS, token.access_token,
                    resource=RESOURCE, client_id=code.client_id)
    assert await grants.load(MCPRecordKind.ACCESS, token.access_token, resource=RESOURCE) is not None


@pytest.mark.parametrize("consent_http", [
    {"backend": "sqlite", "max_grants_per_agent": 1},
    {"backend": "postgres", "max_grants_per_agent": 1},
], indirect=True)
async def test_grant_capacity_refusal_does_not_consume_another_pending_request(consent_http):
    client, pending, _handle, _retained, _keys, _clock, grants, _app = consent_http
    code = await approved_code(consent_http)
    handle = await pending.create(code.client_id, AuthorizationParams(state="owned-second-request",
        scopes=code.scopes, code_challenge=code.code_challenge, redirect_uri=code.redirect_uri,
        redirect_uri_provided_explicitly=True, resource=RESOURCE))
    with pytest.raises(ValueError, match="grant quota"):
        await client.post("/api/owned-mcp-consent?mode=issue-code&handle=" + handle,
            headers={"x-api-key": API_KEY})
    assert await pending.read(handle) is not None
    assert await grants.backend.fetch_one("SELECT COUNT(*) FROM mcp_server_oauth_grants") == (1,)


@pytest.mark.parametrize("consent_http", [
    {"backend": "sqlite", "max_credentials_per_grant": 3},
    {"backend": "postgres", "max_credentials_per_grant": 3},
], indirect=True)
async def test_rotation_capacity_is_finite_and_its_refusal_commits_revocation(consent_http):
    grants = consent_http[6]
    code, token = await pair(consent_http)
    refresh = await grants.load(MCPRecordKind.REFRESH, token.refresh_token,
        resource=RESOURCE, client_id=code.client_id)
    with pytest.raises(TokenError):
        await grants.exchange(MCPRecordKind.REFRESH, refresh, resource=RESOURCE)
    assert await grants.load(MCPRecordKind.ACCESS, token.access_token, resource=RESOURCE) is None
    assert await grants.backend.fetch_one("SELECT COUNT(*) FROM mcp_server_oauth_credentials") == (3,)


async def test_loaded_code_cannot_replace_the_original_pkce_binding(consent_http):
    grants = consent_http[6]
    code = await approved_code(consent_http)
    code.code_challenge = "changed-native-code-binding"
    with pytest.raises(TokenError):
        await grants.exchange(MCPRecordKind.CODE, code, resource=RESOURCE)
    assert await grants.backend.fetch_one("SELECT COUNT(*) FROM mcp_server_oauth_credentials") == (1,)


async def test_refresh_cannot_request_scopes_outside_its_original_native_token(consent_http):
    grants = consent_http[6]
    code, token = await pair(consent_http)
    refresh = await grants.load(MCPRecordKind.REFRESH, token.refresh_token,
        resource=RESOURCE, client_id=code.client_id)
    with pytest.raises(TokenError):
        await grants.exchange(MCPRecordKind.REFRESH, refresh, resource=RESOURCE, scopes=["memory:write"])
    assert await grants.backend.fetch_one("SELECT COUNT(*) FROM mcp_server_oauth_credentials") == (3,)


@pytest.mark.parametrize("change", ["scopes", "client", "resource", "subject", "credential", "copy"])
async def test_only_the_original_human_approval_can_consume_pending_consent(consent_http, change):
    client, pending, handle, _retained, _keys, _clock, grants, _app = consent_http
    response = await client.post("/api/owned-mcp-consent?mode=substitute-approval&change=" + change,
        headers={"x-api-key": API_KEY})
    assert response.status_code == 403, "A copied approval substituted the original human decision"
    assert await pending.read(handle) is not None
    assert await grants.backend.fetch_one("SELECT COUNT(*) FROM mcp_server_oauth_grants") == (0,)


@pytest.mark.parametrize("change", ["sovereign", "client-secret"])
async def test_revocation_tombstone_survives_temporary_credential_configuration_changes(consent_http, change):
    pending, keys, grants = consent_http[1], consent_http[4], consent_http[6]
    code, token = await pair(consent_http)
    revoker = grants
    if change == "sovereign":
        keys[0] = "owned-rotated-sovereign-key"
    else:
        config = json.loads(pending.policy.model_dump_json())
        config["clients"][0]["client_secret"] = "owned-rotated-client-secret"
        policy = MCPServerPolicy.model_validate_json(json.dumps(config))
        other_pending = type(pending)(pending.backend, pending.agent_id, pending.resource_url, policy,
            clock=pending.clock)
        authority = MCPConsentAuthority(pending.agent_id, RESOURCE, policy,
            require_current=grants.consent_authority.require_current_host,
            sovereign_key=lambda: keys[0], clock=pending.clock)
        revoker = MCPAuthorizationGrants(other_pending, grants.issuer, consent_authority=authority)
    await revoker.revoke(MCPRecordKind.ACCESS, token.access_token, resource=RESOURCE, client_id=code.client_id)
    keys[0] = API_KEY
    assert await grants.load(MCPRecordKind.ACCESS, token.access_token, resource=RESOURCE) is None, (
        "An acknowledged revocation revived after restoring earlier credential configuration")
    assert await grants.backend.fetch_one("SELECT revoked FROM mcp_server_oauth_grants") == (1,)
