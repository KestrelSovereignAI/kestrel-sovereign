"""Actual host lifespan/process restart preserves remote MCP permission effects.

Only the external genesis auditor and finite HTTPS provider/MCP peer are
fixtures. Core host, identity, bootstrap, auth, restore and security are real.
This does not prove vendor OAuth refresh or remote unknown-outcome recovery.
"""

import asyncio
from dataclasses import dataclass
from contextlib import AsyncExitStack, asynccontextmanager
import json
import hashlib
import os
from pathlib import Path
import signal
import shutil
import ssl
import sys
import uuid

import httpx2
import pytest
import pytest_asyncio
from types import SimpleNamespace
from cryptography.fernet import Fernet

import kestrel_sovereign
from kestrel_sovereign.inception_service import create_kestrel_identity_async
from tests.integration.mcp_permission_peers import (
    MCPTestPeer, https_peer, gateway_peer, owned_gateway_network, owned_provider_network,
)
from tests.shared.genesis_audit import deterministic_safe_genesis_auditor


pytestmark = [pytest.mark.mcp_integration, pytest.mark.timeout(120),
    pytest.mark.skipif(sys.platform == 'win32', reason='This test requires native graceful SIGTERM')]


@dataclass
class RunningHost:
    state: dict
    process: asyncio.subprocess.Process
    log_path: Path

    async def stop(self):
        if self.process.returncode is None:
            self.process.terminate()
        try:
            await asyncio.wait_for(self.process.wait(), 35)
        except TimeoutError:
            self.process.kill()
            await self.process.wait()
            raise
        if self.state:
            diagnostic = self.log_path.read_text()
            assert self.process.returncode in {0, -signal.SIGTERM}, diagnostic
            assert 'Application shutdown complete' in diagnostic, diagnostic


@asynccontextmanager
async def native_host(home, descriptor, environment, peer, generation):
    ready = home / f'host-{generation}-ready.json'
    log_path = home / f'host-{generation}.log'
    admissions = ready.with_suffix('.gateway-admissions')
    if peer.kind == 'gateway':
        admissions.mkdir(mode=0o700)
    with log_path.open('wb') as log:
        process = await asyncio.create_subprocess_exec(sys.executable, '-B',
            str(Path(__file__).with_name('mcp_permission_host.py')), str(descriptor), str(ready),
            cwd=home, env=environment, stdout=log, stderr=asyncio.subprocess.STDOUT)
        running = RunningHost({}, process, log_path)
        from tests.integration.mcp_gateway_custody import OriginalHostAdmission
        original_host = OriginalHostAdmission(process) if peer.kind == 'gateway' else None
        try:
            if original_host is not None:
                import psutil
                peer.retain_host_admission(original_host, admissions)
                try:
                    original_host.bind_birth(psutil.Process(process.pid).create_time())
                except psutil.NoSuchProcess:
                    await process.wait()
                    raise AssertionError('Owned host failed before metadata admission: ' + log_path.read_text())
                peer.read_host_admissions(original_host, admissions)
            async with asyncio.timeout(35):
                while not ready.exists():
                    if process.returncode is not None:
                        raise AssertionError('Owned host failed: ' + log_path.read_text())
                    await asyncio.sleep(0.01)
            state = json.loads(ready.read_text())
            assert state['pid'] == process.pid
            running.state = state
            expected = Path(kestrel_sovereign.__file__).resolve().with_name('server.py')
            assert Path(state['server_source']).resolve() == expected, (
                'Owned child Core runtime differs from parent')
            configured = environment.get('KESTREL_AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S')
            if configured is not None:
                assert state['sqlite_close_timeout_s'] >= float(configured)
                assert state['shutdown_prefix_s'] >= state['shutdown_prefix_floor_s'] > 0
                assert state['shutdown_tail_s'] >= state['sqlite_close_timeout_s']
                assert state['agent_shutdown_timeout_s'] >= (
                    state['shutdown_prefix_s'] + state['shutdown_tail_s'])
            peer.local_ports.add(state['port'])
            if peer.kind == 'gateway' and state.get('gateway') is not None:
                peer.observe_host_gateway(state['pid'], state['gateway'])
            yield running
        finally:
            async with AsyncExitStack() as retirement:
                if original_host is not None:
                    retirement.push_async_callback(peer.reconcile_host_gateways, original_host, admissions)
                if running.state:
                    retirement.callback(peer.local_ports.discard, running.state['port'])
                    if peer.kind == 'gateway':
                        retirement.push_async_callback(peer.assert_host_retired, running.state['pid'])
                retirement.push_async_callback(running.stop)


@pytest_asyncio.fixture
async def host_campaign(tmp_path, monkeypatch, request):
    home = tmp_path / 'host'
    home.mkdir(mode=0o700)
    data = home / 'agent'
    data.mkdir(mode=0o700)
    key = Fernet.generate_key().decode()
    api_key = uuid.uuid4().hex
    monkeypatch.chdir(home)
    monkeypatch.setenv('KESTREL_DATA_KEY', key)
    monkeypatch.setenv('KESTREL_HOME', str(home))
    identity = await create_kestrel_identity_async(output_dir=str(data), is_test_instance=False,
        agent_name='OwnedHostRestart', did_web_domain='owned-host-restart.invalid',
        genesis_auditor=deterministic_safe_genesis_auditor,
        genesis_audit_provenance='test:3679-native-host-process-restart')
    (home / '.kestrel-host-features.toml').write_text('[host_features]\ndefault_enabled=false\n')
    peer_root = home / 'peer'
    peer_root.mkdir()
    settings = getattr(request, 'param', {})
    transport = settings.get('transport', 'remote')
    assert transport in {'remote', 'gateway'}
    native_a2a = settings.get('native_a2a', False)
    peer_agents = []
    sender_identity = recipient_identity = None
    if native_a2a:
        from kestrel_sovereign.identity.runtime_identity import load_agent_identity

        sender_data = home / 'sender'
        sender_data.mkdir(mode=0o700)
        await create_kestrel_identity_async(output_dir=str(sender_data), is_test_instance=True,
            agent_name='OwnedA2ASender', did_web_domain='owned-a2a-sender.invalid',
            genesis_auditor=deterministic_safe_genesis_auditor,
            genesis_audit_provenance='test:596-native-a2a-cognition')
        sender_identity = load_agent_identity(storage_dir=sender_data, master_key=key)
        recipient_identity = load_agent_identity(storage_dir=data, master_key=key)
        peer_agents.append({'agent_name': 'a2a-sender', 'agent_data': str(sender_data),
            'features': ['BootstrapFeature']})
    async with AsyncExitStack() as resources:
        provider = await resources.enter_async_context(https_peer(
            peer_root, monkeypatch, provider_model='owned-permission-model',
            drop_first_tool_reply=settings.get('drop_first_tool_reply', False),
            provider_only=transport == 'gateway'))
        peer = provider
        gateway_settings = None
        if transport == 'gateway':
            gateway_root = home / 'gateway-peer'
            gateway_root.mkdir()
            peer = await resources.enter_async_context(gateway_peer(gateway_root, monkeypatch))
            peer.provider_url = provider.provider_url
            peer.provider_witness = provider.provider_witness
            peer.tls_certificate = provider.tls_certificate
            plugin = Path(os.environ['MCP_TEST_DOCKER_PLUGIN']).resolve(strict=True)
            gateway_settings = dict(home=os.environ['HOME'], plugin=str(plugin),
                plugin_sha256=hashlib.sha256(plugin.read_bytes()).hexdigest(),
                docker_socket=os.environ['MCP_TEST_DOCKER_HOST'].removeprefix('unix://'))
        (home / 'kestrel.toml').write_text(
            '[llm]\nroute_priority=["deepseek:api"]\n'
            '[llm.vendors.deepseek.routes.api]\napi_key="owned-synthetic-provider-key"\n'
            'base_url=' + json.dumps(provider.provider_url) + '\nmodel="owned-permission-model"\n')
        descriptor = home / 'host-test.json'
        descriptor.write_text(json.dumps(dict(agent_name='permission-host', agent_data=str(data),
            transport=transport, selector=peer.selector, provider_url=provider.provider_url,
            certificate=str(provider.tls_certificate), gateway=gateway_settings,
            native_a2a=native_a2a, peer_agents=peer_agents,
            features=['MCPAgent', 'BootstrapFeature', 'TaskFeature'] if native_a2a else
                ['MCPAgent', 'BootstrapFeature'])))
        if 'sqlite_worker_budget' in settings:
            monkeypatch.setenv('KESTREL_AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S',
                settings['sqlite_worker_budget'])
        environment = owned_host_environment(home, api_key=api_key, data_key=key)
        if transport == 'gateway':
            # Native CLI stores its catalog/profile DB under HOME; Kestrel's
            # independent explicit home continues to own identity and history.
            for name in ['HOME', 'DOCKER_CONFIG', 'DOCKER_HOST', 'DOCKER_MCP_IN_CONTAINER',
                         'XDG_CACHE_HOME', 'KESTREL_MCP_DOCKER_CATALOG']:
                environment[name] = os.environ[name]
        headers = {'X-API-Key': api_key, 'X-Kestrel-Allow-Destructive': 'true'}
        async with owned_gateway_network(monkeypatch, peer, gateway_settings) as gateway:
            with owned_provider_network(monkeypatch, peer, gateway=gateway):
                yield SimpleNamespace(home=home, descriptor=descriptor, environment=environment,
                    peer=peer, did=identity.agent_did, headers=headers,
                    sender_identity=sender_identity, recipient_identity=recipient_identity)


def owned_host_environment(home, *, api_key, data_key):
    """Own all state while carrying the explicit native SQLite fixture budget."""
    repository = Path(__file__).resolve().parents[2]
    environment = dict(PATH=os.environ['PATH'], HOME=str(home), KESTREL_HOME=str(home),
        KESTREL_DB_BACKEND='sqlite', KESTREL_HOST_DB_PATH=str(home / 'host-control/host.db'),
        KESTREL_MULTI_AGENT_CONFIG=str(home / 'multi_agent.toml'),
        KESTREL_DATA_KEY=data_key, KESTREL_API_KEY=api_key, KESTREL_SKIP_DOTENV='1',
        PYTHON_DOTENV_DISABLED='1', PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=os.pathsep.join([
            str(Path(kestrel_sovereign.__file__).resolve().parent.parent), str(repository)]),
        KESTREL_PHOENIX_ENABLED='0', KESTREL_TRACING_ENABLED='0',
        AWS_EC2_METADATA_DISABLED='true', EMAIL_DRY_RUN='true')
    setting = 'KESTREL_AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S'
    if setting in os.environ:
        environment[setting] = os.environ[setting]
    return environment


def host_client(campaign, state):
    return httpx2.AsyncClient(base_url=f'http://127.0.0.1:{state["port"]}',
        headers=campaign.headers, trust_env=False, timeout=30)


async def observe_registry(campaign, state, nonce):
    receipt = campaign.home / 'observations' / (nonce + '.json')
    async with asyncio.timeout(10):
        while not receipt.exists():
            await asyncio.sleep(0.001)
    observed = json.loads(receipt.read_text())
    assert observed['did'] == campaign.did and observed['pid'] == state['pid']
    if campaign.peer.kind == 'gateway' and observed.get('gateway') is not None:
        state['gateway'] = observed['gateway']
        campaign.peer.observe_host_gateway(state['pid'], observed['gateway'])
    return observed


async def configure_mcp(client, campaign, state):
    prefix = '/api/agents/permission-host'
    denied = await client.get(prefix + '/api/security/permissions/tree',
        headers={'X-API-Key': 'wrong-owned-key'})
    assert denied.status_code == 401
    for text in ['Hello', '!skip-discovery']:
        response = await client.post(prefix + '/api/agent/invoke', json={
            'input': text, 'session_id': 'owned-host-restart'})
        assert response.status_code == 200, response.text
    response = await client.post(prefix + '/api/security/permissions', json={
        'feature': 'MCPAgent', 'tool': 'mcp_load_server', 'level': 'allow'})
    assert response.status_code == 200, response.text
    nonce = uuid.uuid4().hex
    response = await client.post(prefix + '/api/agent/invoke',
        headers={'X-Owned-Registry-Observation': nonce}, json={
        'input': '!mcp-load ' + campaign.peer.selector, 'session_id': 'owned-host-restart'})
    assert response.status_code == 200, response.text
    observed = await observe_registry(campaign, state, nonce)
    assert len(observed['mcp_tools']) == 1, response.text
    tool, owner = next(iter(observed['mcp_tools'].items()))
    return owner, tool


async def set_permission(client, owner, tool, level):
    response = await client.post('/api/agents/permission-host/api/security/permissions',
        json={'feature': owner, 'tool': tool, 'level': level})
    assert response.status_code == 200, response.text


def invoke_marker(client):
    return client.post('/api/agents/permission-host/api/agent/invoke', json={
        'input': 'Record the owned marker using the connected MCP tool.',
        'provider': 'deepseek:api', 'model': 'owned-permission-model',
        'session_id': 'owned-host-restart'})


async def exercise_permission_restart(c):
    """One HTTP/storage/effect oracle shared by native MCP transports."""
    async with native_host(c.home, c.descriptor, c.environment, c.peer, 1) as first:
        assert first.state['did'] == c.did
        async with host_client(c, first.state) as client:
            owner, tool = await configure_mcp(client, c, first.state)
            await set_permission(client, owner, tool, 'deny')
            response = await invoke_marker(client)
            assert response.status_code == 200, response.text
            assert await c.peer.witness(None, None) == []
    async with native_host(c.home, c.descriptor, c.environment, c.peer, 2) as second:
        assert second.state['pid'] != first.state['pid'] and second.state['did'] == c.did
        async with host_client(c, second.state) as client:
            nonce = uuid.uuid4().hex
            tree = (await client.get('/api/agents/permission-host/api/security/permissions/tree',
                headers={'X-Owned-Registry-Observation': nonce})).json()['tree']
            remote, = [entry for entry in tree if entry['name'] == owner]
            restored, = [entry for entry in remote['tools'] if entry['name'] == tool]
            assert restored['level'] == 'deny'
            observed = await observe_registry(c, second.state, nonce)
            assert observed['mcp_tools'] == {tool: owner}
            for level, expected in [('deny', []), ('allow', ['owned-provider-effect'])]:
                if level == 'allow':
                    await set_permission(client, owner, tool, level)
                response = await invoke_marker(client)
                assert response.status_code == 200, response.text
                assert await c.peer.witness(None, None) == expected
            audit = (await client.get('/api/agents/permission-host/api/security/audit')).json()['logs']
            decisions = [entry['decision'] for entry in reversed(audit)
                if entry['feature'] == owner and entry['tool'] == tool]
            assert decisions == ['auto_denied', 'auto_denied', 'auto_allowed']


async def test_actual_host_process_restores_remote_permission_without_new_effect(host_campaign):
    await exercise_permission_restart(host_campaign)

async def pending_approval(client, call):
    async with asyncio.timeout(10):
        while True:
            response = await client.get('/api/agents/permission-host/api/security/pending')
            assert response.status_code == 200, response.text
            pending = response.json()['pending']
            if pending:
                entry, = pending
                return entry
            if call.done():
                response = await call
                pytest.fail('Invocation completed before approval admission: ' + response.text)
            await asyncio.sleep(0.01)


async def exercise_approval_restart(c):
    """Reject stale/replayed host approvals with a real external effect witness."""
    async with native_host(c.home, c.descriptor, c.environment, c.peer, 1) as first:
        async with host_client(c, first.state) as client:
            owner, tool = await configure_mcp(client, c, first.state)
            await set_permission(client, owner, tool, 'always_ask')
            original = asyncio.create_task(invoke_marker(client))
            try:
                old = await pending_approval(client, original)
                assert old['feature'] == owner and old['tool'] == tool
                assert await c.peer.witness(None, None) == []
                await first.stop()
                outcome, = await asyncio.gather(original, return_exceptions=True)
                if isinstance(outcome, httpx2.Response):
                    assert outcome.status_code in {499, 500, 503}, outcome.text
                else:
                    assert isinstance(outcome, (httpx2.ReadError, httpx2.RemoteProtocolError)), type(outcome).__name__
                assert await c.peer.witness(None, None) == []
            finally:
                if not original.done():
                    original.cancel()
                await asyncio.gather(original, return_exceptions=True)
    async with native_host(c.home, c.descriptor, c.environment, c.peer, 2) as second:
        assert second.state['pid'] != first.state['pid'] and second.state['did'] == c.did
        async with host_client(c, second.state) as client:
            current = asyncio.create_task(invoke_marker(client))
            try:
                fresh = await pending_approval(client, current)
                assert fresh['id'] != old['id'] and fresh['feature'] == owner and fresh['tool'] == tool
                refused = await client.post('/api/agents/permission-host/api/security/approve',
                    json={'approval_id': old['id'], 'approved': True, 'scope': 'once'})
                assert refused.status_code == 404, refused.text
                assert await c.peer.witness(None, None) == []
                accepted = await client.post('/api/agents/permission-host/api/security/approve',
                    json={'approval_id': fresh['id'], 'approved': True, 'scope': 'once'})
                assert accepted.status_code == 200, accepted.text
                assert accepted.json()['awaiter_gone'] is False
                response = await current
                assert response.status_code == 200, response.text
                assert await c.peer.witness(None, None) == ['owned-provider-effect']
                current = asyncio.create_task(invoke_marker(client))
                following = await pending_approval(client, current)
                assert following['id'] not in {old['id'], fresh['id']}
                replay = await client.post('/api/agents/permission-host/api/security/approve',
                    json={'approval_id': fresh['id'], 'approved': True, 'scope': 'once'})
                assert replay.status_code == 404, replay.text
                assert await c.peer.witness(None, None) == ['owned-provider-effect']
                rejected = await client.post('/api/agents/permission-host/api/security/approve',
                    json={'approval_id': following['id'], 'approved': False, 'scope': 'once'})
                assert rejected.status_code == 200, rejected.text
                response = await current
                assert response.status_code == 200, response.text
                assert await c.peer.witness(None, None) == ['owned-provider-effect']
            finally:
                if not current.done():
                    current.cancel()
                await asyncio.gather(current, return_exceptions=True)


async def test_old_host_approval_cannot_authorize_a_restarted_host_call(host_campaign):
    await exercise_approval_restart(host_campaign)


async def exercise_approval_decision_race(c, first_approved):
    """Hold real SQLite persistence while competing HTTP decisions enter CAS."""
    from contextlib import closing
    import sqlite3

    async with native_host(c.home, c.descriptor, c.environment, c.peer, 1) as host:
        async with host_client(c, host.state) as client:
            owner, tool = await configure_mcp(client, c, host.state)
            await set_permission(client, owner, tool, 'always_ask')
            call = asyncio.create_task(invoke_marker(client))
            decisions = []
            try:
                pending = await pending_approval(client, call)
                assert (pending['feature'], pending['tool']) == (owner, tool)
                database = Path(host.state['agent_database']).resolve(strict=True)
                assert database.is_relative_to(c.home / 'agent')

                async def snapshot():
                    nonce = uuid.uuid4().hex
                    response = await client.get('/api/agents/permission-host/api/security/pending',
                        headers={'X-Owned-Registry-Observation': nonce})
                    assert response.status_code == 200, response.text
                    observed = await observe_registry(c, host.state, nonce)
                    return observed['approvals'].get(pending['id'])

                expected_claim = dict(status='approved' if first_approved else 'denied', scope='once')
                with closing(sqlite3.connect(database, timeout=0)) as writer:
                    writer.execute('BEGIN IMMEDIATE')
                    try:
                        first = asyncio.create_task(client.post(
                            '/api/agents/permission-host/api/security/approve', json={
                                'approval_id': pending['id'], 'approved': first_approved, 'scope': 'once'}))
                        decisions.append(first)
                        async with asyncio.timeout(3):
                            while await snapshot() != expected_claim:
                                assert not first.done(), 'First decision escaped its original blocked persistence'
                                await asyncio.sleep(0)
                        second = asyncio.create_task(client.post(
                            '/api/agents/permission-host/api/security/approve', json={
                                'approval_id': pending['id'], 'approved': not first_approved, 'scope': 'always'}))
                        decisions.append(second)
                        # A broken CAS changes the real claim before it blocks
                        # on SQLite too. Observe that state, then release both
                        # actual requests; a test timeout is not the oracle.
                        async with asyncio.timeout(3):
                            while not second.done() and await snapshot() == expected_claim:
                                await asyncio.sleep(0)
                        assert not first.done() and not call.done()
                        assert await c.peer.witness(None, None) == []
                    finally:
                        writer.rollback()
                first_response, second_response = await asyncio.gather(*decisions)
                assert first_response.status_code == 200, first_response.text
                assert first_response.json()['persisted'] is True
                assert first_response.json()['awaiter_gone'] is False
                assert second_response.status_code == 404, (
                    'Concurrent decision replaced the original claimed approval', second_response.text)
                response = await call
                assert response.status_code == 200, response.text
                expected_effects = ['owned-provider-effect'] if first_approved else []
                assert await c.peer.witness(None, None) == expected_effects
                audit = (await client.get('/api/agents/permission-host/api/security/audit')).json()['logs']
                assert [row['decision'] for row in audit
                    if row['feature'] == owner and row['tool'] == tool] == [
                        'user_approved' if first_approved else 'user_denied']
                call = asyncio.create_task(invoke_marker(client))
                following = await pending_approval(client, call)
                assert following['id'] != pending['id'], 'Competing decision escaped one-shot scope'
                assert await c.peer.witness(None, None) == expected_effects
                denied = await client.post('/api/agents/permission-host/api/security/approve', json={
                    'approval_id': following['id'], 'approved': False, 'scope': 'once'})
                assert denied.status_code == 200, denied.text
                response = await call
                assert response.status_code == 200, response.text
                assert await c.peer.witness(None, None) == expected_effects
            finally:
                if not call.done():
                    call.cancel()
                await asyncio.gather(call, *decisions, return_exceptions=True)


@pytest.mark.parametrize('first_approved', [False, True], ids=['deny-first', 'approve-first'])
async def test_concurrent_host_approval_cannot_replace_original_decision(host_campaign, first_approved):
    await exercise_approval_decision_race(host_campaign, first_approved)


async def test_owned_host_setup_failure_preserves_original_diagnostic(tmp_path):
    home = tmp_path / 'failed-host'
    home.mkdir()
    repository = Path(__file__).resolve().parents[2]
    environment = dict(PATH=os.environ['PATH'], HOME=str(home), PYTHONPATH=str(repository),
        KESTREL_SKIP_DOTENV='1', PYTHON_DOTENV_DISABLED='1', PYTHONDONTWRITEBYTECODE='1')
    peer = MCPTestPeer('remote', 'unused-before-descriptor-load')
    with pytest.raises(AssertionError, match='Owned host failed') as raised:
        async with native_host(home, home / 'missing-test-descriptor.json', environment, peer, 'failure'):
            pytest.fail('A missing native host descriptor must never reach readiness')
    assert 'FileNotFoundError' in str(raised.value)
    assert not (home / 'host-failure-ready.json').exists()


@pytest.mark.parametrize('host_campaign', [{'sqlite_worker_budget': '5.0'}], indirect=True)
async def test_owned_host_refuses_a_different_core_runtime(host_campaign):
    c = host_campaign
    alternate = c.home / 'alternate-runtime'
    alternate.mkdir()
    shutil.copytree(Path(kestrel_sovereign.__file__).resolve().parent,
        alternate / 'kestrel_sovereign', ignore=shutil.ignore_patterns('__pycache__'))
    repository = Path(__file__).resolve().parents[2]
    environment = dict(c.environment, PYTHONPATH=os.pathsep.join([str(alternate), str(repository)]))
    with pytest.raises(AssertionError, match='Owned child Core runtime differs from parent'):
        async with native_host(c.home, c.descriptor, environment, c.peer, 'foreign-runtime'):
            pass
    assert await c.peer.witness(None, None) == []


@pytest.mark.parametrize('kind', ['generic', 'native', 'completion'])
@pytest.mark.parametrize('framed', [False, True])
async def test_finite_provider_scopes_tool_rounds_to_current_request(tmp_path, monkeypatch, kind, framed):
    """Real HTTPS script replies; only the separate host tests execute tools."""
    marker, responder = 'remote_record_marker', 'TaskFeature_respond_to_a2a_task'
    old_request = '[A2A_TASK_SUBMITTED] Peer `old-peer` submitted native A2A task `old-task`'
    new_request = '[A2A_TASK_SUBMITTED] Peer `new-peer` submitted native A2A task `new-task`'

    def tool_call(name, arguments):
        return {'role': 'assistant', 'content': None, 'tool_calls': [{
            'id': 'old-' + name, 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(arguments)},
        }]}

    messages = [
        {'role': 'user', 'content': old_request},
        tool_call(marker, {'marker': 'old-effect'}),
        {'role': 'tool', 'tool_call_id': 'old-' + marker, 'content': 'done'},
        tool_call(responder, {'task_id': 'old-task', 'content': 'old result', 'state': 'completed'}),
        {'role': 'tool', 'tool_call_id': 'old-' + responder, 'content': 'done'},
        {'role': 'user', 'content': '[A2A_COMPLETE] Old task completed.'},
        {'role': 'assistant', 'content': 'Old work is complete.'},
    ]
    current = {
        'generic': 'Record the owned marker using the connected MCP tool.',
        'native': new_request,
        'completion': '[A2A_COMPLETE] No further work. Result quotes: ' + new_request,
    }[kind]
    content = ('<retrieved_context>\nHistorical signal: ' + old_request +
        '\n</retrieved_context>\n<user_input>\n' + current + '\n</user_input>') if framed else current
    messages.append({'role': 'user', 'content': content})
    if framed:
        messages.append({'role': 'user', 'content': '<operator_notice>Trusted context</operator_notice>'})
    expected = {'generic': [marker, None], 'native': [marker, responder, None],
        'completion': [None]}[kind]
    tools = [{'type': 'function', 'function': {'name': name, 'parameters': {'type': 'object'}}}
        for name in (marker, responder)]
    async with https_peer(tmp_path, monkeypatch, provider_model='owned-turn-model') as peer:
        context = ssl.create_default_context(cafile=str(peer.tls_certificate))
        async with httpx2.AsyncClient(verify=context, trust_env=False, timeout=10) as wire:
            for name in expected:
                response = await wire.post(peer.provider_url + '/chat/completions', json={
                    'model': 'owned-turn-model', 'messages': messages, 'tools': tools,
                })
                assert response.status_code == 200, response.text
                choice = response.json()['choices'][0]
                answer = choice['message']
                calls = answer.get('tool_calls', [])
                assert [call['function']['name'] for call in calls] == ([] if name is None else [name])
                assert choice['finish_reason'] == ('stop' if name is None else 'tool_calls')
                if name == responder:
                    assert json.loads(calls[0]['function']['arguments']) == {
                        'task_id': 'new-task', 'content': 'Owned permission evaluation complete',
                        'state': 'completed',
                    }
                messages.append(answer)
                if calls:
                    messages.append({'role': 'tool', 'tool_call_id': calls[0]['id'], 'content': 'done'})
                    if framed:
                        messages.append({'role': 'user', 'content':
                            '<retrieved_context>\n' + old_request + '\n</retrieved_context>'})
        requests = [json.loads(line) for line in peer.provider_witness.read_text().splitlines()]
        assert len(requests) == len(expected)
        assert all(request['active_user_prefix'] == current[:200] for request in requests)
        assert await peer.witness(None, None) == [], 'Script replies alone must not execute Core tools'


@pytest.mark.parametrize('messages,diagnostic', [
    ([], 'active user request'),
    ([{'role': 'system', 'content': 'not a request'}], 'active user request'),
    ([{'role': 'user', 'content': []}], 'text user request'),
    ([{'role': 'user', 'content': '<retrieved_context>incomplete'}], 'incomplete context envelope'),
])
def test_finite_provider_refuses_missing_or_invalid_request(messages, diagnostic):
    from tests.integration.mcp_permission_peer import active_user_turn

    with pytest.raises(ValueError, match=diagnostic):
        active_user_turn(messages)
