"""HTTP/model/feature-subagent admission precedes an actual owned process.

The native HTTPS model is finite. Core, the installed provider, governance,
identity, SQL stores and local script executor are real. No vendor is called.
These cases do not establish a full host lifespan or browser acceptance.
"""

import asyncio
import json
from dataclasses import dataclass
from uuid import uuid4

import httpx2
import pytest
import pytest_asyncio
from kestrel_sdk.hooks.base import Hook, HookEvent, HookOutput

from kestrel_sovereign.endpoints.agent import router as invoke_router
from kestrel_sovereign.endpoints.security import router as security_router
from kestrel_sovereign.features.security.permissions import PermissionLevel
from tests.integration import test_native_permission_entrypoints as entrypoints
from tests.integration.mcp_permission_peers import (
    MCPTestPeer,
    https_peer,
    owned_provider_network,
)
from tests.shared.agent_http import authenticated_agent_app

pytestmark = [pytest.mark.asyncio, pytest.mark.mcp_integration, pytest.mark.timeout(120)]
native_permission_case = entrypoints.native_permission_case
native_permission_inherited_environment = entrypoints.native_permission_inherited_environment


def owned_script_task(script):
    return f'Run owned signed script {script.id} through compute_feature.'


class OriginalSubagentTurnObserver(Hook):
    """Read the original invocation before the ledger writes being tested."""

    def __init__(self, agent):
        super().__init__('owned-native-subagent-turn', [HookEvent.PRE_SUBAGENT_CALL], priority=1000)
        self.agent = agent
        self.observations = []

    async def execute(self, hook_input):
        if hook_input.feature_name == 'ComputeFeature' and hook_input.tool_name == 'compute_feature':
            self.observations.append((hook_input.session_id, self.agent.get_current_turn_id(),
                hook_input.tool_input['task']))
        return HookOutput()


@dataclass
class NativeSubagentHTTP:
    case: entrypoints.NativePermissionCase
    peer: MCPTestPeer
    client: httpx2.AsyncClient
    dials: list
    observer: OriginalSubagentTurnObserver

    def start(self, script):
        return self.case.own_call(self.client.post('/api/agent/invoke', json={
            'input': owned_script_task(script),
            'provider': 'deepseek:api', 'model': 'owned-permission-model',
            'session_id': 'owned-subagent-permission',
        }))

    async def finish(self, call):
        response = await asyncio.wait_for(call, 30)
        assert response.status_code == 200, response.text

    async def assert_dispatches(self, expected):
        database = self.case.agent.tool_effect_store.database
        leaves = await database.fetchall(
            'SELECT parent_dispatch_id,session_id,turn_id,result_status,error_class,args_redacted,adapter '
            'FROM a2a_tool_dispatches '
            "WHERE tool_name='run_script' ORDER BY id")
        assert len(leaves) == len(expected), 'Every attempted leaf needs exactly one dispatch row'
        parents, turns = set(), set()
        for leaf, (script, expected_status, expected_error) in zip(leaves, expected, strict=True):
            parent_id, leaf_session, leaf_turn, status, error_class, arguments, adapter = leaf
            observed, = [(session, turn) for session, turn, task in self.observer.observations
                if task == owned_script_task(script)]
            assert observed[0] == 'owned-subagent-permission'
            assert observed[1] and observed[1] != 'turn_unknown'
            assert leaf_session == observed[0], 'The leaf must retain its original session'
            assert leaf_turn == observed[1], 'The leaf must belong to its original HTTP turn'
            assert parent_id, 'The leaf must retain its actual parent subagent dispatch'
            assert parent_id not in parents, 'Every invocation needs its own parent dispatch'
            parent = await database.fetchone(
                'SELECT tool_name,session_id,turn_id,args_redacted FROM a2a_tool_dispatches '
                'WHERE dispatch_id=?', (parent_id,))
            assert parent is not None
            assert parent[0] == 'compute_feature' and parent[1] == observed[0]
            assert parent[2] == observed[1], 'The parent must retain its original HTTP turn'
            assert json.loads(parent[3])['task'] == owned_script_task(script)
            assert parent[2] not in turns, 'Distinct HTTP invocations need distinct original turns'
            assert status == expected_status
            assert error_class == expected_error, 'The leaf must retain its exact diagnostic class'
            assert json.loads(arguments)['script_id'] == script.id
            assert adapter == 'subagent:ComputeFeature.run_script'
            parents.add(parent_id)
            turns.add(parent[2])

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
                    observer = OriginalSubagentTurnObserver(case.agent)
                    case.agent.hooks_manager.register(observer)
                    try:
                        yield NativeSubagentHTTP(case, peer, client, dials, observer)
                    finally:
                        try:
                            await case.retire_calls()
                        finally:
                            case.agent.hooks_manager.unregister(observer)


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
    await native.assert_dispatches([(script, 'success' if allowed else 'error',
        None if allowed else 'UnknownToolError')])
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
    first_error = None if approved else 'PermissionDenied'
    await native.assert_dispatches([(first, first_status, first_error)])
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
    await native.assert_dispatches([(first, first_status, first_error),
        (second, 'policy_denied', 'PermissionDenied')])
    native.assert_native_subagent(leaf_advertised=True)
    assert not case.security.approval_queue.pending_requests


async def test_native_http_subagent_cleanup_joins_the_original_pending_approval(native_subagent_http):
    native = native_subagent_http
    case = native.case
    await case.security.permission_store.set_permission('ComputeFeature', 'run_script', PermissionLevel.ASK)
    script = await case.stage_script()
    call = native.start(script)
    original = await case.next_approval(call)
    assert original.tool_args['script_id'] == script.id
    case.assert_effect_count(0)
    await case.retire_calls()
    assert call.done() and not case.security.approval_queue.pending_requests
    case.assert_effect_count(0)
    assert await case.compute.script_store.get_executions_for_script(script.id) == []
