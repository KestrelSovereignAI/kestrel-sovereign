"""Host contract checks over real HTTP/authentication; no vendor protocol claim.

The contract consumer returns bounded messages to exercise Core's authority
and response lifecycle. Native SDK acquisition and vendor behavior belong to
the feature integration suite.
"""

import asyncio
import base64
from datetime import timedelta
import json
from urllib.parse import urlencode

from itsdangerous import TimestampSigner
import pytest
import pytest_asyncio

from kestrel_sovereign.endpoints.auth_oauth import _create_jwt
from kestrel_sovereign.mcp_oauth import MCPOAuthHostPolicy
from kestrel_sovereign.security.capability_authority import HostCapabilityAccessDenied
from tests.unit.test_mcp_oauth_host import ALICE, API_KEY, oauth_boundary  # noqa: F401


class HostContractConsumer:
    def __init__(self, host):
        self.host = host
        self.calls, self.retired, self.leases = [], [], []
        self.states = {}
        self.started = asyncio.Event()
        self.release = None
        self.modify_url = None
        self.fail_completion = False
        self.fail_close = False
        self.withdrawals = 0

    def status(self, identifier="owned-connection", *, connected=True):
        return dict(connection_id=identifier, server_url="https://resource.example/mcp",
            issuer="https://issuer.example", resource="https://resource.example",
            scopes=["read"], state="connected" if connected else "authorized", connected=connected)

    async def begin(self, context, check, request, identity):
        check()
        handle = object()
        state = f"owned-state-{len(self.states):032d}"
        self.states[handle] = state
        self.calls.append(("begin", context.principal_id, handle))
        self.started.set()
        if self.release is not None:
            await self.release.wait()
        parameters = dict(client_id=identity["client_document"]["client_id"],
            redirect_uri=identity["client_document"]["redirect_uris"][0],
            response_type="code", code_challenge_method="S256", code_challenge="a" * 43,
            state=state, resource="https://resource.example", scope="read")
        url = "https://issuer.example/authorize?" + urlencode(parameters)
        if self.modify_url is not None:
            url = self.modify_url(url)
        return dict(handle=handle, authorization_url=url, state=state,
            issuer="https://issuer.example", resource="https://resource.example", scopes=["read"])

    async def complete(self, handle, context, check, value):
        check()
        assert value["state"] == self.states[handle]
        self.calls.append(("complete", context.principal_id, handle))
        if self.fail_completion:
            raise TimeoutError("private-upstream-code-or-token")
        self.leases.append(self.host.issue_lease(context, connection_id="owned-connection", slot_id="owned-slot"))
        return self.status()

    async def cancel(self, handle, check):
        check()
        self.calls.append(("cancel", handle))

    async def connect(self, context, check, identifier, *, refresh):
        check()
        self.calls.append(("connect", identifier, refresh))
        self.leases.append(self.host.issue_lease(context, connection_id=identifier, slot_id="owned-slot"))
        return self.status(identifier)

    async def disconnect(self, context, check, identifier):
        check()
        self.calls.append(("disconnect", identifier))
        return dict(local_tokens_deleted=True, remote_revocation="not_attempted")

    async def list(self, context):
        self.calls.append(("list", context.principal_id))
        return [self.status()]

    async def retire_authorization(self, handle):
        self.retired.append(handle)

    async def close_authorizations(self):
        if self.fail_close:
            raise RuntimeError("original worker has not retired")
        self.retired.extend(handle for handle in self.states if handle not in self.retired)

    def withdraw_authorizations(self):
        self.withdrawals += 1


@pytest_asyncio.fixture
async def controller(oauth_boundary):
    b = oauth_boundary
    b.runtime = HostContractConsumer(b.host)
    b.host.attach(b.runtime)
    b.app.include_router(b.agent.create_mcp_oauth_router(b.feature, resolve_host=lambda: b.host))
    try:
        yield b
    finally:
        b.runtime.fail_close = False
        await b.host.close()


async def begin(b, *, headers=None, body=None):
    return await b.client.post("/api/mcp/oauth/authorizations", headers=headers or {"X-API-Key": API_KEY},
        json=body or {"server_url": "https://resource.example/mcp", "scopes": ["read"]})


def callback_body(b):
    return dict(state=next(iter(b.runtime.states.values())), code="private-owned-code", iss="https://issuer.example")


@pytest.mark.asyncio
async def test_native_http_complete_connect_read_and_local_disconnect_contract(controller):
    b = controller
    initial = await begin(b)
    assert initial.status_code == 200, initial.text
    assert initial.headers["cache-control"].startswith("private, no-store")
    assert len(b.host._pending) == 1 and b.host._reader.get() is None
    identifier = initial.json()["operation_id"]
    callback = await b.client.post(f"/api/mcp/oauth/authorizations/{identifier}/callback",
        json=callback_body(b), headers={"X-API-Key": API_KEY})
    assert callback.status_code == 200 and callback.json()["connected"] is True, callback.text
    assert "private-owned-code" not in callback.text
    b.runtime.leases[-1].require_background()
    replay = await b.client.post(f"/api/mcp/oauth/authorizations/{identifier}/callback",
        json=callback_body(b), headers={"X-API-Key": API_KEY})
    assert replay.status_code == 409
    assert sum(call[0] == "complete" for call in b.runtime.calls) == 1
    connected = await b.client.post("/api/mcp/oauth/connections/owned-connection/connect",
        json={"refresh": True}, headers={"X-API-Key": API_KEY})
    assert connected.status_code == 200
    assert ("connect", "owned-connection", True) in b.runtime.calls
    listed = await b.client.get("/api/mcp/oauth/connections", headers={"X-API-Key": API_KEY})
    assert listed.status_code == 200 and len(listed.json()["connections"]) == 1
    disconnected = await b.client.delete("/api/mcp/oauth/connections/owned-connection", headers={"X-API-Key": API_KEY})
    assert disconnected.json() == {"local_tokens_deleted": True, "remote_revocation": "not_attempted"}


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["callback", "cancel"])
async def test_foreign_principal_cannot_consume_or_cancel_an_original_operation(controller, route):
    b = controller
    initial = await begin(b)
    identifier = initial.json()["operation_id"]
    headers = {"Authorization": "Bearer " + _create_jwt(ALICE)}
    path = f"/api/mcp/oauth/authorizations/{identifier}"
    response = (await b.client.post(path + "/callback", headers=headers, json=callback_body(b))
                if route == "callback" else await b.client.delete(path, headers=headers))
    assert response.status_code == 403, response.text
    assert b.host._pending[identifier].consumed is False and b.runtime.retired == []
    assert [call[0] for call in b.runtime.calls] == ["begin"]
    own = await b.client.post(path + "/callback", json=callback_body(b), headers={"X-API-Key": API_KEY})
    assert own.status_code == 200, own.text


@pytest.mark.asyncio
async def test_bad_state_retires_original_handle_without_exchange(controller):
    b = controller
    initial = await begin(b)
    identifier = initial.json()["operation_id"]
    response = await b.client.post(f"/api/mcp/oauth/authorizations/{identifier}/callback",
        json=callback_body(b) | {"state": "wrong-state-with-at-least-sixteen-characters"}, headers={"X-API-Key": API_KEY})
    assert response.status_code == 400
    assert identifier not in b.host._pending and len(b.runtime.retired) == 1
    assert [call[0] for call in b.runtime.calls] == ["begin"]


@pytest.mark.asyncio
async def test_consumed_unknown_exchange_is_never_replayed(controller):
    b = controller
    initial = await begin(b)
    identifier = initial.json()["operation_id"]
    b.runtime.fail_completion = True
    for expected in (503, 409):
        response = await b.client.post(f"/api/mcp/oauth/authorizations/{identifier}/callback",
            json=callback_body(b), headers={"X-API-Key": API_KEY})
        assert response.status_code == expected, response.text
        assert "private-upstream" not in response.text
    assert b.host._pending[identifier].consumed and not b.host._pending[identifier].finished
    assert sum(call[0] == "complete" for call in b.runtime.calls) == 1


@pytest.mark.asyncio
async def test_quota_reservation_precedes_concurrent_feature_begin(controller):
    b = controller
    b.app.state.mcp_oauth_policy = MCPOAuthHostPolicy(console_origin="https://console.example",
        max_operations=1, max_operations_per_principal=1)
    b.runtime.release = asyncio.Event()
    first = asyncio.create_task(begin(b))
    try:
        await b.runtime.started.wait()
        second = await begin(b)
        assert second.status_code == 429, second.text
        assert len(b.runtime.calls) == len(b.host._pending) == 1
    finally:
        b.runtime.release.set()
        assert (await first).status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", [
    ("console.example", "evil.example"), ("S256", "plain"), ("scope=read", "scope=write"),
    ("issuer.example/authorize", "evil.example/authorize"),
])
async def test_host_binding_validation_retires_original_returned_handle(controller, replacement):
    b = controller
    b.runtime.modify_url = lambda url: url.replace(*replacement)
    response = await begin(b)
    assert response.status_code == 400, response.text
    assert b.host._pending == {} and len(b.runtime.retired) == 1


@pytest.mark.asyncio
async def test_bounded_request_refuses_before_feature_or_pending_operation(controller):
    b = controller
    response = await b.client.post("/api/mcp/oauth/authorizations", content=b"x" * 16385, headers={"X-API-Key": API_KEY})
    assert response.status_code == 413, response.text
    assert b.runtime.calls == [] and b.host._pending == {}


@pytest.mark.asyncio
async def test_actual_signed_cookie_requires_csrf_before_feature_begin(controller):
    b = controller
    credential = TimestampSigner("owned-oauth-session-key").sign(base64.b64encode(json.dumps({"user_email": ALICE}).encode()))
    b.client.cookies.set("kestrel_session", credential.decode())
    request = dict(server_url="https://resource.example/mcp", scopes=["read"])
    denied = await b.client.post("/api/mcp/oauth/authorizations", json=request)
    assert denied.status_code == 403 and b.runtime.calls == [], denied.text
    b.client.cookies.set("kestrel_csrf", "owned-oauth-csrf")
    accepted = await b.client.post("/api/mcp/oauth/authorizations", json=request, headers={"X-CSRF-Token": "owned-oauth-csrf"})
    assert accepted.status_code == 200, accepted.text


@pytest.mark.asyncio
async def test_expiry_and_withdrawal_retire_without_consuming_an_exchange(controller):
    b = controller
    initial = await begin(b)
    identifier = initial.json()["operation_id"]
    b.clock[0] += timedelta(seconds=601)
    expired = await b.client.post(f"/api/mcp/oauth/authorizations/{identifier}/callback",
        json=callback_body(b), headers={"X-API-Key": API_KEY})
    assert expired.status_code == 410, expired.text
    await b.host.maintain()
    assert identifier not in b.host._pending and len(b.runtime.retired) == 1
    initial = await begin(b)
    identifier = initial.json()["operation_id"]
    b.agent._withdraw_mcp_oauth_for_privacy_transition()
    assert b.runtime.withdrawals == 1 and b.host._pending[identifier].withdrawn
    denied = await b.client.post(f"/api/mcp/oauth/authorizations/{identifier}/callback",
        json=callback_body(b), headers={"X-API-Key": API_KEY})
    assert denied.status_code == 403
    await b.host.maintain()
    assert b.host._pending == {} and len(b.runtime.retired) == 2


@pytest.mark.asyncio
async def test_replacement_refuses_until_original_workers_acknowledge_close(controller):
    b = controller
    await begin(b)
    b.runtime.fail_close = True
    with pytest.raises(RuntimeError, match="worker has not retired"):
        await b.host.close()
    assert b.host.retired is False and b.host._pending
    from kestrel_sdk.resources import ResourceConflict
    with pytest.raises(ResourceConflict):
        b.agent.create_mcp_oauth_host(b.feature, is_current=lambda: True)
    b.runtime.fail_close = False
    await b.host.close()
    assert b.host.retired is True and b.host._pending == {}
    replacement = b.agent.create_mcp_oauth_host(b.feature, is_current=lambda: True)
    await replacement.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("credentials", ["anonymous", "query"])
async def test_no_interactive_header_cannot_start_authorization(controller, credentials):
    b = controller
    path = "/api/mcp/oauth/authorizations" + ("?api_key=" + API_KEY if credentials == "query" else "")
    result = await b.client.post(path, json=dict(server_url="https://resource.example/mcp", scopes=["read"]))
    assert result.status_code in {401, 403}, result.text
    assert b.runtime.calls == [] and b.host._pending == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("method,path,body", [
    ("POST", "/authorizations", dict(server_url="https://resource.example/mcp", scopes=["read"])),
    ("POST", "/authorizations/unknown/callback", dict(state="a" * 32, code="owned", iss="https://issuer.example")),
    ("DELETE", "/authorizations/unknown", None),
    ("POST", "/connections/owned/connect", dict(refresh=False)),
    ("DELETE", "/connections/owned", None),
])
async def test_every_cookie_mutation_requires_native_csrf(controller, method, path, body):
    b = controller
    credential = TimestampSigner("owned-oauth-session-key").sign(base64.b64encode(json.dumps({"user_email": ALICE}).encode()))
    b.client.cookies.set("kestrel_session", credential.decode())
    result = await b.client.request(method, "/api/mcp/oauth" + path, json=body)
    assert result.status_code == 403, result.text
    assert b.runtime.calls == [] and b.host._pending == {}


@pytest.mark.asyncio
async def test_unknown_callback_and_cancellation_are_safe_and_owner_cancel_is_one_use(controller):
    b = controller
    unknown = await b.client.post("/api/mcp/oauth/authorizations/unknown/callback",
        json=dict(state="a" * 32, code="owned", iss="https://issuer.example"), headers={"X-API-Key": API_KEY})
    assert unknown.status_code == 404 and b.runtime.calls == []
    initial = await begin(b)
    path = "/api/mcp/oauth/authorizations/" + initial.json()["operation_id"]
    for expected in (200, 409):
        result = await b.client.delete(path, headers={"X-API-Key": API_KEY})
        assert result.status_code == expected, result.text
    assert sum(call[0] == "cancel" for call in b.runtime.calls) == 1


@pytest.mark.asyncio
async def test_changed_privacy_epoch_cannot_complete_an_original_authorization(controller):
    b = controller
    initial = await begin(b)
    b.agent._privacy_policy_epoch += 1
    result = await b.client.post("/api/mcp/oauth/authorizations/" + initial.json()["operation_id"] + "/callback",
        json=callback_body(b), headers={"X-API-Key": API_KEY})
    assert result.status_code == 403
    assert [call[0] for call in b.runtime.calls] == ["begin"]


@pytest.mark.asyncio
async def test_real_privacy_transition_withdraws_leases_before_installing_new_mode(controller, tmp_path, monkeypatch):
    from kestrel_sovereign.features.privacy import PrivacyAgent
    from kestrel_sovereign.privacy import PrivacyMode
    from kestrel_sovereign.storage import AsyncStorage
    from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage
    b = controller
    connected = await b.client.post("/api/mcp/oauth/connections/owned-connection/connect",
        json={}, headers={"X-API-Key": API_KEY})
    assert connected.status_code == 200
    lease = b.runtime.leases[-1]
    lease.require_background()
    raw = AsyncStorage(str(tmp_path / "owned-privacy-transition.db"), agent_id=b.agent.did)
    try:
        await raw.initialize()
        b.agent._raw_storage = raw
        b.agent.storage = PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL)
        b.agent.privacy_agent = PrivacyAgent(raw, PrivacyMode.NORMAL)
        installed = []
        original = PrivacyEnforcingStorage.set_privacy_mode
        def audit_install(storage, mode):
            assert b.runtime.withdrawals == 1
            assert b.agent.privacy_mode == PrivacyMode.NORMAL
            with pytest.raises(HostCapabilityAccessDenied):
                lease.require_background()
            installed.append(mode)
            return original(storage, mode)
        monkeypatch.setattr(PrivacyEnforcingStorage, "set_privacy_mode", audit_install)
        result = await b.agent.set_privacy_mode(PrivacyMode.PUBLIC)
        assert isinstance(result, str)
        assert installed == [PrivacyMode.PUBLIC]
        assert b.agent.privacy_mode == PrivacyMode.PUBLIC
    finally:
        await raw.close()
