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
from tests.unit.test_mcp_oauth_host import ALICE, API_KEY, oauth_boundary as oauth_boundary


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

    async def begin(self, context, check, request, identity, *, authorization_handle):
        check()
        handle = authorization_handle
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
    assert response.status_code == 500, response.text
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


@pytest.mark.asyncio
@pytest.mark.parametrize("malformation", ["extra", "missing-handle", "null-handle", "other-handle"])
async def test_malformed_begin_result_never_loses_original_authorization_owner(controller, monkeypatch, malformation):
    b = controller
    original = b.runtime.begin
    async def malformed(*args, **kwargs):
        result = await original(*args, **kwargs)
        if malformation == 'extra':
            result['unexpected'] = True
        elif malformation == 'missing-handle':
            result.pop('handle')
        elif malformation == 'null-handle':
            result['handle'] = None
        else:
            result['handle'] = object()
        return result
    monkeypatch.setattr(b.runtime, 'begin', malformed)
    response = await begin(b)
    assert response.status_code == 500, response.text
    assert b.runtime.retired == list(b.runtime.states), 'Malformed metadata lost original authorization worker custody'
    assert b.host._pending == {}


@pytest.mark.asyncio
async def test_cancel_consumes_before_awaiting_feature_join(controller, monkeypatch):
    b = controller
    initial = await begin(b)
    path = '/api/mcp/oauth/authorizations/' + initial.json()['operation_id']
    entered, release = asyncio.Event(), asyncio.Event()
    original = b.runtime.cancel
    async def paused(handle, check):
        entered.set()
        await release.wait()
        await original(handle, check)
    monkeypatch.setattr(b.runtime, 'cancel', paused)
    first = asyncio.create_task(b.client.delete(path, headers={'X-API-Key': API_KEY}))
    try:
        await entered.wait()
        callback = await b.client.post(path + '/callback', json=callback_body(b), headers={'X-API-Key': API_KEY})
        assert callback.status_code == 409, callback.text
        repeated = await b.client.delete(path, headers={'X-API-Key': API_KEY})
        assert repeated.status_code == 409
    finally:
        release.set()
        assert (await first).status_code == 200
    assert [call[0] for call in b.runtime.calls] == ['begin', 'cancel']


@pytest.mark.asyncio
async def test_maintenance_cannot_retire_an_admitted_exchange_at_browser_deadline(controller, monkeypatch):
    b = controller
    b.app.state.mcp_oauth_policy = MCPOAuthHostPolicy(console_origin='https://console.example', authorization_seconds=1.0)
    initial = await begin(b)
    entered, release = asyncio.Event(), asyncio.Event()
    original = b.runtime.complete
    async def paused(*args):
        entered.set()
        await release.wait()
        return await original(*args)
    monkeypatch.setattr(b.runtime, 'complete', paused)
    callback = asyncio.create_task(b.client.post('/api/mcp/oauth/authorizations/' + initial.json()['operation_id'] + '/callback',
        json=callback_body(b), headers={'X-API-Key': API_KEY}))
    try:
        await entered.wait()
        b.clock[0] += timedelta(seconds=2)
        await b.host.maintain()
        assert b.runtime.retired == [] and b.host._pending
    finally:
        release.set()
        assert (await callback).status_code == 200
    await b.host.maintain()
    assert b.host._pending == {} and len(b.runtime.retired) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('condition,expected', [('expired', 410), ('withdrawn', 403)])
async def test_maintenance_preserves_inflight_begin_then_retires_original_before_delivery(controller, condition, expected):
    b = controller
    b.app.state.mcp_oauth_policy = MCPOAuthHostPolicy(console_origin='https://console.example', authorization_seconds=1.0)
    b.runtime.release = asyncio.Event()
    first = asyncio.create_task(begin(b))
    try:
        await b.runtime.started.wait()
        if condition == 'expired':
            b.clock[0] += timedelta(seconds=2)
        else:
            b.host.withdraw()
        await b.host.maintain()
        assert b.runtime.retired == [] and b.host._pending
    finally:
        b.runtime.release.set()
        response = await first
    assert response.status_code == expected, response.text
    assert b.host._pending == {} and len(b.runtime.retired) == 1


@pytest.mark.asyncio
async def test_concurrent_maintenance_retires_one_original_handle_once(controller, monkeypatch):
    b = controller
    await begin(b)
    b.clock[0] += timedelta(seconds=601)
    entered, release, second_started = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = []
    original = b.runtime.retire_authorization
    async def paused(handle):
        calls.append(handle)
        entered.set()
        await release.wait()
        await original(handle)
    monkeypatch.setattr(b.runtime, 'retire_authorization', paused)
    first = asyncio.create_task(b.host.maintain())
    second = None
    async def another():
        second_started.set()
        await b.host.maintain()
    try:
        await entered.wait()
        second = asyncio.create_task(another())
        await second_started.wait()
        await asyncio.sleep(0)
        assert len(calls) == 1
    finally:
        release.set()
        await first
        if second is not None:
            await second
    assert len(b.runtime.retired) == 1 and b.host._pending == {}


@pytest.mark.asyncio
async def test_async_withdraw_callback_is_refused_before_attaching(oauth_boundary):
    b = oauth_boundary
    runtime = HostContractConsumer(b.host)
    async def asynchronous_withdraw():
        raise AssertionError('Async withdrawal must never be admitted')
    runtime.withdraw_authorizations = asynchronous_withdraw
    with pytest.raises(TypeError, match='synchronous'):
        b.host.attach(runtime)
    assert b.host._runtime is None


@pytest.mark.asyncio
async def test_nonempty_withdraw_result_is_an_explicit_contract_failure(controller, monkeypatch):
    b = controller
    original = b.runtime.withdraw_authorizations
    monkeypatch.setattr(b.runtime, 'withdraw_authorizations', lambda: 'not-a-withdrawal-ack')
    try:
        with pytest.raises(TypeError, match='return None'):
            b.host.withdraw()
    finally:
        monkeypatch.setattr(b.runtime, 'withdraw_authorizations', original)


@pytest.mark.asyncio
@pytest.mark.parametrize('policy', ['missing', 'changed'])
async def test_original_route_refuses_unconfigured_or_retargeted_policy_with_safe_status(controller, policy):
    b = controller
    if policy == 'changed':
        assert (await begin(b)).status_code == 200
    b.app.state.mcp_oauth_policy = (None if policy == 'missing' else
        MCPOAuthHostPolicy(console_origin='https://another-console.example'))
    result = await b.client.get('/api/mcp/oauth/connections', headers={'X-API-Key': API_KEY})
    assert result.status_code == 403, result.text
    assert all(call[0] != 'list' for call in b.runtime.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['complete', 'connect', 'disconnect'])
async def test_feature_acted_but_invalid_receipt_reports_unknown_without_suggesting_replay(controller, monkeypatch, operation):
    b = controller
    original = getattr(b.runtime, operation)
    async def invalid(*args, **kwargs):
        result = await original(*args, **kwargs)
        return result | {'unexpected_secret': 'private-feature-receipt'}
    monkeypatch.setattr(b.runtime, operation, invalid)
    if operation == 'complete':
        initial = await begin(b)
        path = '/api/mcp/oauth/authorizations/' + initial.json()['operation_id'] + '/callback'
        result = await b.client.post(path, json=callback_body(b), headers={'X-API-Key': API_KEY})
    elif operation == 'connect':
        result = await b.client.post('/api/mcp/oauth/connections/owned/connect', json={}, headers={'X-API-Key': API_KEY})
    else:
        result = await b.client.delete('/api/mcp/oauth/connections/owned', headers={'X-API-Key': API_KEY})
    assert result.status_code == 503, result.text
    assert result.json()['error']['code'] == 'mcp_oauth_outcome_unknown_do_not_replay'
    assert 'private-feature-receipt' not in result.text
    assert sum(call[0] == operation for call in b.runtime.calls) == 1


@pytest.mark.asyncio
async def test_invalid_read_receipt_is_a_service_failure_without_private_diagnostics(controller, monkeypatch):
    b = controller
    original = b.runtime.list
    async def invalid(context):
        return [row | {'unexpected_secret': 'private-read-value'} for row in await original(context)]
    monkeypatch.setattr(b.runtime, 'list', invalid)
    result = await b.client.get('/api/mcp/oauth/connections', headers={'X-API-Key': API_KEY})
    assert result.status_code == 500, result.text
    assert result.json()['error']['code'] == 'mcp_oauth_service_error'
    assert 'private-read-value' not in result.text


@pytest.mark.asyncio
async def test_retirement_fences_callback_before_awaiting_original_worker(controller, monkeypatch):
    b = controller
    initial = await begin(b)
    path = '/api/mcp/oauth/authorizations/' + initial.json()['operation_id'] + '/callback'
    entered, release = asyncio.Event(), asyncio.Event()
    original = b.runtime.retire_authorization
    async def paused(handle):
        entered.set()
        await release.wait()
        await original(handle)
    monkeypatch.setattr(b.runtime, 'retire_authorization', paused)
    first = asyncio.create_task(b.client.post(path, json=callback_body(b) | {'state': 'incorrect-state-with-sufficient-length'},
        headers={'X-API-Key': API_KEY}))
    try:
        await entered.wait()
        competing = await b.client.post(path, json=callback_body(b), headers={'X-API-Key': API_KEY})
        assert competing.status_code == 403, competing.text
    finally:
        release.set()
        assert (await first).status_code == 400
    assert [call[0] for call in b.runtime.calls] == ['begin']
    assert b.host._pending == {} and len(b.runtime.retired) == 1


@pytest.mark.asyncio
async def test_failed_retirement_is_retained_for_immediate_original_handle_reconciliation(controller, monkeypatch):
    b = controller
    original = b.runtime.retire_authorization
    async def failed_observer(handle):
        raise TimeoutError('owned physical retirement has not acknowledged')
    monkeypatch.setattr(b.runtime, 'retire_authorization', failed_observer)
    b.runtime.modify_url = lambda url: url.replace('S256', 'plain')
    response = await begin(b)
    assert response.status_code == 503
    assert len(b.host._pending) == 1
    assert next(iter(b.host._pending.values())).withdrawn
    monkeypatch.setattr(b.runtime, 'retire_authorization', original)
    await b.host.maintain()
    assert b.host._pending == {} and len(b.runtime.retired) == 1


@pytest.mark.asyncio
async def test_failed_cancel_is_immediately_eligible_for_original_handle_retirement(controller, monkeypatch):
    b = controller
    initial = await begin(b)
    original = b.runtime.cancel
    async def unacknowledged(handle, check):
        raise TimeoutError('original cancellation not physically acknowledged')
    monkeypatch.setattr(b.runtime, 'cancel', unacknowledged)
    response = await b.client.delete('/api/mcp/oauth/authorizations/' + initial.json()['operation_id'], headers={'X-API-Key': API_KEY})
    assert response.status_code == 503
    assert next(iter(b.host._pending.values())).withdrawn
    monkeypatch.setattr(b.runtime, 'cancel', original)
    await b.host.maintain()
    assert b.host._pending == {} and len(b.runtime.retired) == 1


@pytest.mark.asyncio
async def test_begin_refused_before_handoff_never_retires_a_handle_feature_did_not_receive(controller, monkeypatch):
    b = controller
    original = b.host._begin
    async def refused(context, check, value):
        def withdrawn():
            raise HostCapabilityAccessDenied('authority withdrawn before feature admission')
        return await original(context, withdrawn, value)
    monkeypatch.setattr(b.host, '_begin', refused)
    response = await begin(b)
    assert response.status_code == 403
    assert b.runtime.calls == [] and b.runtime.states == {} and b.runtime.retired == []
    assert b.host._pending == {}


@pytest.mark.asyncio
async def test_foreground_begin_does_not_join_unrelated_failed_maintenance(controller, monkeypatch):
    b = controller
    await begin(b)
    b.host.withdraw()
    original = b.runtime.retire_authorization
    async def failed(handle):
        raise TimeoutError('original worker still retained')
    monkeypatch.setattr(b.runtime, 'retire_authorization', failed)
    try:
        with pytest.raises(TimeoutError):
            await b.host.maintain()
        result = await begin(b, headers={'Authorization': 'Bearer ' + _create_jwt(ALICE)})
        assert result.status_code == 200, result.text
        assert len(b.host._pending) == 2
    finally:
        monkeypatch.setattr(b.runtime, 'retire_authorization', original)


@pytest.mark.asyncio
async def test_identity_snapshot_refusal_precedes_original_handle_handoff(controller, monkeypatch):
    b = controller
    def refused(context):
        raise HostCapabilityAccessDenied('identity snapshot authority ended before feature admission')
    monkeypatch.setattr(b.host, 'client_identity', refused)
    result = await begin(b)
    assert result.status_code == 403
    assert b.runtime.calls == [] and b.runtime.states == {} and b.runtime.retired == []
    assert b.host._pending == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("retirement_error", [TimeoutError, asyncio.CancelledError])
async def test_failed_original_retirement_cannot_skip_an_independent_eligible_handle(controller, monkeypatch, retirement_error):
    b = controller
    await begin(b)
    await begin(b, headers={'Authorization': 'Bearer ' + _create_jwt(ALICE)})
    pending = list(b.host._pending.values())
    original = b.runtime.retire_authorization
    attempted = []
    async def failing_first(handle):
        attempted.append(handle)
        if handle is pending[0].handle:
            raise retirement_error('first original handle remains retained')
        await original(handle)
    monkeypatch.setattr(b.runtime, 'retire_authorization', failing_first)
    b.host.withdraw()
    with pytest.raises(retirement_error):
        await b.host.maintain()
    assert attempted == [item.handle for item in pending]
    assert list(b.host._pending.values()) == [pending[0]]
    assert b.runtime.retired == [pending[1].handle]
    monkeypatch.setattr(b.runtime, 'retire_authorization', original)
    await b.host.maintain()
    assert b.host._pending == {}


@pytest.mark.asyncio
async def test_real_maintenance_cancellation_survives_later_independent_join_failure(controller, monkeypatch):
    b = controller
    await begin(b)
    await begin(b, headers={'Authorization': 'Bearer ' + _create_jwt(ALICE)})
    pending = list(b.host._pending.values())
    entered = asyncio.Event()
    attempted = []
    async def unacknowledged(handle):
        attempted.append(handle)
        if handle is pending[0].handle:
            entered.set()
            await asyncio.Event().wait()
        raise TimeoutError('independent original join has not acknowledged closure')
    monkeypatch.setattr(b.runtime, 'retire_authorization', unacknowledged)
    b.host.withdraw()
    sweep = asyncio.create_task(b.host.maintain())
    await entered.wait()
    sweep.cancel()
    with pytest.raises(asyncio.CancelledError):
        await sweep
    assert sweep.cancelled() and attempted == [item.handle for item in pending]
    assert list(b.host._pending.values()) == pending


@pytest.mark.asyncio
async def test_clock_rollback_cannot_retire_a_newly_admitted_original_exchange(controller, monkeypatch):
    b = controller
    await begin(b)
    second = await begin(b, headers={'Authorization': 'Bearer ' + _create_jwt(ALICE)})
    pending = list(b.host._pending.values())
    retire_entered, retire_release = asyncio.Event(), asyncio.Event()
    exchange_entered, exchange_release = asyncio.Event(), asyncio.Event()
    original_retire, original_complete = b.runtime.retire_authorization, b.runtime.complete
    async def retire(handle):
        if handle is pending[0].handle:
            retire_entered.set()
            await retire_release.wait()
        await original_retire(handle)
    async def complete(handle, context, check, value):
        exchange_entered.set()
        await exchange_release.wait()
        return await original_complete(handle, context, check, value)
    monkeypatch.setattr(b.runtime, 'retire_authorization', retire)
    monkeypatch.setattr(b.runtime, 'complete', complete)
    b.clock[0] += timedelta(seconds=601)
    sweep = asyncio.create_task(b.host.maintain())
    callback = None
    try:
        await retire_entered.wait()
        b.clock[0] -= timedelta(seconds=601)
        callback = asyncio.create_task(b.client.post('/api/mcp/oauth/authorizations/'
            + second.json()['operation_id'] + '/callback',
            headers={'Authorization': 'Bearer ' + _create_jwt(ALICE)},
            json={'state': b.runtime.states[pending[1].handle], 'code': 'owned-clock-rollback-code',
                  'iss': 'https://issuer.example'}))
        await exchange_entered.wait()
        retire_release.set()
        await sweep
        assert b.runtime.retired == [pending[0].handle]
        assert pending[1].in_flight and pending[1] in b.host._pending.values()
    finally:
        retire_release.set()
        exchange_release.set()
        if not sweep.done():
            await sweep
        if callback is not None:
            response = await callback
            assert response.status_code == 200, response.text


@pytest.mark.asyncio
@pytest.mark.parametrize('last_retirement', ['await', 'ordinary-error'])
async def test_anyio_timeout_preserves_its_original_cancellation_during_sweep(controller, monkeypatch, last_retirement):
    import anyio
    b = controller
    await begin(b)
    await begin(b, headers={'Authorization': 'Bearer ' + _create_jwt(ALICE)})
    pending = list(b.host._pending.values())
    attempted = []
    async def unacknowledged(handle):
        attempted.append(handle)
        if handle is pending[0].handle or last_retirement == 'await':
            await anyio.sleep_forever()
        raise RuntimeError('independent original closer remains unacknowledged')
    monkeypatch.setattr(b.runtime, 'retire_authorization', unacknowledged)
    b.host.withdraw()
    with pytest.raises(TimeoutError):
        with anyio.fail_after(0.025) as scope:
            await b.host.maintain()
    assert scope.cancelled_caught
    assert attempted == [item.handle for item in pending]
    assert list(b.host._pending.values()) == pending


@pytest.mark.asyncio
async def test_maintenance_preserves_every_independent_native_error(controller, monkeypatch):
    b = controller
    await begin(b)
    await begin(b, headers={'Authorization': 'Bearer ' + _create_jwt(ALICE)})
    pending = list(b.host._pending.values())
    failures = [TimeoutError('first original closer'), ValueError('second original closer')]
    async def unacknowledged(handle):
        raise failures[0 if handle is pending[0].handle else 1]
    monkeypatch.setattr(b.runtime, 'retire_authorization', unacknowledged)
    b.host.withdraw()
    with pytest.raises(ExceptionGroup) as caught:
        await b.host.maintain()
    assert caught.value.exceptions == tuple(failures)
    assert list(b.host._pending.values()) == pending


@pytest.mark.asyncio
async def test_anyio_control_survives_a_consumer_that_suppresses_its_first_cancellation(controller, monkeypatch):
    import anyio
    b = controller
    await begin(b)
    await begin(b, headers={'Authorization': 'Bearer ' + _create_jwt(ALICE)})
    pending = list(b.host._pending.values())
    attempted = []
    async def unacknowledged(handle):
        attempted.append(handle)
        if handle is pending[0].handle:
            try:
                await anyio.sleep_forever()
            except asyncio.CancelledError:
                return
        raise RuntimeError('independent original closer remains unacknowledged')
    monkeypatch.setattr(b.runtime, 'retire_authorization', unacknowledged)
    b.host.withdraw()
    with pytest.raises(TimeoutError):
        with anyio.fail_after(0.025) as scope:
            await b.host.maintain()
    assert scope.cancelled_caught
    assert attempted == [item.handle for item in pending]
    assert list(b.host._pending.values()) == [pending[1]]


@pytest.mark.asyncio
async def test_plain_asyncio_control_survives_a_consumer_that_suppresses_its_first_cancellation(controller, monkeypatch):
    b = controller
    await begin(b)
    await begin(b, headers={'Authorization': 'Bearer ' + _create_jwt(ALICE)})
    pending = list(b.host._pending.values())
    entered = asyncio.Event()
    failures = [RuntimeError('independent original closer remains unacknowledged')]
    attempted = []
    async def unacknowledged(handle):
        attempted.append(handle)
        if handle is pending[0].handle:
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return
        raise failures[0]
    monkeypatch.setattr(b.runtime, 'retire_authorization', unacknowledged)
    b.host.withdraw()
    sweep = asyncio.create_task(b.host.maintain())
    await entered.wait()
    sweep.cancel('owned original sweep cancellation')
    with pytest.raises(asyncio.CancelledError) as caught:
        await sweep
    assert sweep.cancelled()
    assert caught.value.__cause__.exceptions == tuple(failures)
    assert attempted == [item.handle for item in pending]
    assert list(b.host._pending.values()) == [pending[1]]
