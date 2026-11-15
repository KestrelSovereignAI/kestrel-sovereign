"""HTTP/model/feature-subagent admission precedes an actual owned process.

The native HTTPS model is finite. Core, the installed provider, governance,
identity, SQL stores and local script executor are real. No vendor is called.
These cases do not establish a full host lifespan or browser acceptance.
"""

import asyncio
from dataclasses import dataclass
import json
from uuid import uuid4

import httpx2
import pytest
import pytest_asyncio

from kestrel_sovereign.endpoints.agent import router as invoke_router
from kestrel_sovereign.endpoints.security import router as security_router
from kestrel_sovereign.features.security.permissions import PermissionLevel
from tests.integration.mcp_permission_peers import MCPTestPeer, https_peer, owned_provider_network
from tests.integration.test_native_permission_entrypoints import (
    NativePermissionCase,
    native_permission_case as native_permission_case,
    native_permission_inherited_environment as native_permission_inherited_environment,
)
from tests.shared.agent_http import authenticated_agent_app


pytestmark = [pytest.mark.asyncio, pytest.mark.mcp_integration, pytest.mark.timeout(120)]


@dataclass
class NativeSubagentHTTP:
    case: NativePermissionCase
    peer: MCPTestPeer
    client: httpx2.AsyncClient
    dials: list

    def start(self, script):
        return self.case.own_call(self.client.post('/api/agent/invoke', json={
            'input': f'Run owned signed script {script.id} through compute_feature.',
            'provider': 'deepseek:api', 'model': 'owned-permission-model',
            'session_id': 'owned-subagent-permission',
        }))

    async def finish(self, call):
        response = await asyncio.wait_for(call, 30)
        assert response.status_code == 200, response.text

    async def assert_dispatches(self, expected):
        database = self.case.agent.tool_effect_store.database
        leaves = await database.fetchall(
            'SELECT parent_dispatch_id,result_status,args_redacted,adapter FROM a2a_tool_dispatches '
            "WHERE tool_name='run_script' ORDER BY id")
        assert len(leaves) == len(expected), 'Every attempted leaf needs exactly one dispatch row'
        for leaf, (script, status) in zip(leaves, expected, strict=True):
            assert leaf[0], 'The leaf must retain its actual parent subagent dispatch'
            assert leaf[1] == status
            assert json.loads(leaf[2])['script_id'] == script.id
            assert leaf[3] == 'subagent:ComputeFeature.run_script'
            assert await database.fetchone(
                'SELECT tool_name,session_id FROM a2a_tool_dispatches WHERE dispatch_id=?',
                (leaf[0],)) == ('compute_feature', 'owned-subagent-permission')

    def assert_native_subagent(self, *, leaf_advertised):
        requests = [row for row in self.peer.provider_requests()
            if row['path'].endswith('/chat/completions')]
        assert any('compute_feature' in row['tools'] and not row['compute_subagent']
            for row in requests), requests
        children = [row for row in requests if row['compute_subagent']]
        assert children, 'The real feature subagent must call the native provider'
        assert all(('run_script' in row['tools']) == leaf_advertised for row in children), children
        assert self.dials, 'The installed provider must reach the owned native TLS listener'


@pytest_asyncio.fixture
async def native_subagent_http(tmp_path, monkeypatch, native_permission_case):
    peer_root = tmp_path / 'native-model'
    peer_root.mkdir()
    async with https_peer(peer_root, monkeypatch, provider_model='owned-permission-model') as peer:
        (tmp_path / 'kestrel.toml').write_text(
            '[llm]\nroute_priority=["deepseek:api"]\n'
            '[llm.vendors.deepseek.routes.api]\napi_key="owned-synthetic-provider-key"\n'
            'base_url=' + json.dumps(peer.provider_url) + '\nmodel="owned-permission-model"\n'
        )
        with owned_provider_network(monkeypatch, peer) as dials:
            async with native_permission_case(is_test_instance=False) as case:
                assert case.compute.tool_name == 'compute_feature'
                await case.security.permission_store.set_permission(
                    'ComputeFeature', case.compute.tool_name, PermissionLevel.ALLOW)
                app = authenticated_agent_app(case.agent, session_secret=uuid4().hex)
                app.include_router(invoke_router)
                app.include_router(security_router)
                async with httpx2.AsyncClient(
                    transport=httpx2.ASGITransport(app=app), base_url=case.http.base_url,
                    headers=case.http.headers,
                ) as client:
                    try:
                        yield NativeSubagentHTTP(case, peer, client, dials)
                    finally:
                        await case.retire_calls()


@pytest.mark.parametrize('permission', [PermissionLevel.ALLOW, PermissionLevel.DENY])
async def test_native_http_feature_subagent_obeys_leaf_permission(native_subagent_http, permission):
    native = native_subagent_http
    case = native.case
    script = await case.stage_script()
    await case.security.permission_store.set_permission('ComputeFeature', 'run_script', permission)
    await native.finish(native.start(script))
    allowed = permission == PermissionLevel.ALLOW
    case.assert_effect_count(int(allowed))
    audit = await case.audit()
    assert [row['decision'] for row in audit] == (['auto_allowed'] if allowed else []), audit
    executions = await case.compute.script_store.get_executions_for_script(script.id)
    assert len(executions) == int(allowed)
    assert all(row.succeeded and not row.dry_run for row in executions)
    await native.assert_dispatches([(script, 'success' if allowed else 'error')])
    native.assert_native_subagent(leaf_advertised=allowed)
    assert not case.security.approval_queue.pending_requests


@pytest.mark.parametrize('approved', [True, False], ids=['approve-once', 'deny'])
async def test_native_http_subagent_one_shot_and_invalid_decisions(native_subagent_http, approved):
    native = native_subagent_http
    case = native.case
    await case.security.permission_store.set_permission('ComputeFeature', 'run_script', PermissionLevel.ASK)
    first = await case.stage_script()
    call = native.start(first)
    original = await case.next_approval(call)
    assert original.feature_name == 'ComputeFeature' and original.tool_name == 'run_script'
    assert original.tool_args['script_id'] == first.id
    case.assert_effect_count(0)
    payload = {'approval_id': original.id, 'approved': True, 'scope': 'once'}
    refused = await case.http.post('/api/security/approve', json=payload,
        headers={'X-API-Key': 'unrelated-owned-test-key'})
    assert refused.status_code == 401
    for scope, status in [('', 400), ('forever', 400), (None, 422), ({}, 422)]:
        malformed = await case.http.post('/api/security/approve', json={**payload, 'scope': scope})
        assert malformed.status_code == status, malformed.text
        assert case.security.approval_queue.pending_requests == [original] and not call.done()
        assert await case.audit() == []
        case.assert_effect_count(0)
    submitted = await case.http.post('/api/security/approve', json={**payload, 'approved': approved})
    assert submitted.status_code == 200 and submitted.json()['persisted'], submitted.text
    await native.finish(call)
    case.assert_effect_count(int(approved))
    first_status = 'success' if approved else 'policy_denied'
    await native.assert_dispatches([(first, first_status)])
    second = await case.stage_script()
    following = native.start(second)
    fresh = await case.next_approval(following)
    assert fresh.id != original.id and fresh.tool_args['script_id'] == second.id
    replay = await case.http.post('/api/security/approve', json=payload)
    assert replay.status_code == 404 and not following.done()
    case.assert_effect_count(int(approved))
    denied = await case.http.post('/api/security/approve', json={'approval_id': fresh.id, 'approved': False})
    assert denied.status_code == 200 and denied.json()['persisted'], denied.text
    await native.finish(following)
    case.assert_effect_count(int(approved))
    assert await case.compute.script_store.get_executions_for_script(second.id) == []
    assert sorted(row['decision'] for row in await case.audit()) == sorted(
        ['user_approved' if approved else 'user_denied', 'user_denied'])
    assert await case.security.permission_store.get_permission('ComputeFeature', 'run_script') == PermissionLevel.ASK
    await native.assert_dispatches([(first, first_status), (second, 'policy_denied')])
    native.assert_native_subagent(leaf_advertised=True)
    assert not case.security.approval_queue.pending_requests
