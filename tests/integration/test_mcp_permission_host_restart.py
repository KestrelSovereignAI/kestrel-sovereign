"""Actual host lifespan/process restart preserves remote MCP permission effects.

Only the external genesis auditor and finite HTTPS provider/MCP peer are
fixtures. Core host, identity, bootstrap, auth, restore and security are real.
This does not prove vendor OAuth refresh or remote unknown-outcome recovery.
"""

import asyncio
from dataclasses import dataclass
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import signal
import shutil
import sys
import uuid

import httpx2
import pytest
import pytest_asyncio
from types import SimpleNamespace
from cryptography.fernet import Fernet

import kestrel_sovereign
from kestrel_sovereign.inception_service import create_kestrel_identity_async
from tests.integration.mcp_permission_peers import https_peer, owned_provider_network
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
    with log_path.open('wb') as log:
        process = await asyncio.create_subprocess_exec(sys.executable, '-B',
            str(Path(__file__).with_name('mcp_permission_host.py')), str(descriptor), str(ready),
            cwd=home, env=environment, stdout=log, stderr=asyncio.subprocess.STDOUT)
        running = RunningHost({}, process, log_path)
        try:
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
            yield running
        finally:
            try:
                await running.stop()
            finally:
                if running.state:
                    peer.local_ports.discard(running.state['port'])


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
    async with https_peer(peer_root, monkeypatch, provider_model='owned-permission-model') as peer:
        (home / 'kestrel.toml').write_text(
            '[llm]\nroute_priority=["deepseek:api"]\n'
            '[llm.vendors.deepseek.routes.api]\napi_key="owned-synthetic-provider-key"\n'
            'base_url=' + json.dumps(peer.provider_url) + '\nmodel="owned-permission-model"\n')
        descriptor = home / 'host-test.json'
        descriptor.write_text(json.dumps(dict(agent_name='permission-host', agent_data=str(data),
            selector=peer.selector, provider_url=peer.provider_url, certificate=str(peer.tls_certificate))))
        settings = getattr(request, 'param', {})
        if 'sqlite_worker_budget' in settings:
            monkeypatch.setenv('KESTREL_AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S',
                settings['sqlite_worker_budget'])
        environment = owned_host_environment(home, api_key=api_key, data_key=key)
        headers = {'X-API-Key': api_key, 'X-Kestrel-Allow-Destructive': 'true'}
        with owned_provider_network(monkeypatch, peer):
            yield SimpleNamespace(home=home, descriptor=descriptor, environment=environment,
                peer=peer, did=identity.agent_did, headers=headers)


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
    return observed


async def configure_remote(client, campaign, state):
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
    assert len(observed['remote_tools']) == 1, response.text
    tool, owner = next(iter(observed['remote_tools'].items()))
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


async def test_actual_host_process_restores_remote_permission_without_new_effect(host_campaign):
    c = host_campaign
    async with native_host(c.home, c.descriptor, c.environment, c.peer, 1) as first:
        assert first.state['did'] == c.did
        async with host_client(c, first.state) as client:
            owner, tool = await configure_remote(client, c, first.state)
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
            assert observed['remote_tools'] == {tool: owner}
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


async def test_old_host_approval_cannot_authorize_a_restarted_host_call(host_campaign):
    c = host_campaign
    async with native_host(c.home, c.descriptor, c.environment, c.peer, 1) as first:
        async with host_client(c, first.state) as client:
            owner, tool = await configure_remote(client, c, first.state)
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


async def test_owned_host_setup_failure_preserves_original_diagnostic(tmp_path):
    home = tmp_path / 'failed-host'
    home.mkdir()
    repository = Path(__file__).resolve().parents[2]
    environment = dict(PATH=os.environ['PATH'], HOME=str(home), PYTHONPATH=str(repository),
        KESTREL_SKIP_DOTENV='1', PYTHON_DOTENV_DISABLED='1', PYTHONDONTWRITEBYTECODE='1')
    peer = SimpleNamespace(local_ports=set())
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
