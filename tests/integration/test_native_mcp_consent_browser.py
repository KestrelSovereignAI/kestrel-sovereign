"""Real Chromium consent over HTTPS in an isolated, networkless namespace."""

import asyncio
import base64
import datetime
import hashlib
import json
import os
import socket
from pathlib import Path
from urllib.parse import urlencode, urlsplit

import pytest
import uvicorn
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.applications import Starlette
from starlette.routing import Host

import kestrel_sovereign
from kestrel_sovereign import server as core_server
from tests.integration.test_native_mcp_memory_tools import observe_native_call
from tests.integration.test_native_mcp_publication import KEY
from tests.integration.test_native_mcp_publication import (
    assistant_agent as _assistant_agent,
)
from tests.integration.test_native_mcp_publication import publication as _publication

pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.mcp_integration]
assistant_agent = _assistant_agent
publication = _publication


@pytest.fixture(autouse=True)
def owned_identity_domain(monkeypatch):
    monkeypatch.setenv('KESTREL_DID_WEB_DOMAIN', 'agents.kestrel-sovereign.test')


@pytest.fixture
async def native_consent_https(publication, tmp_path, monkeypatch):
    assert Path('/.dockerenv').is_file(), 'Native consent browser requires an owned process namespace'
    assert not Path('/var/run/docker.sock').exists() and not os.environ.get('DOCKER_HOST')
    owner, _http = publication
    monkeypatch.setattr(core_server, 'SERVE_UI', True)
    static = Path(kestrel_sovereign.__file__).parent / 'static'
    owner.app.mount('/js', StaticFiles(directory=static / 'js'))
    owner.app.mount('/shared', StaticFiles(directory=static / 'shared'))
    callbacks = []
    client_app = FastAPI()

    @client_app.get('/callback')
    async def callback(request: Request):
        callbacks.append((dict(request.query_params), dict(request.headers)))
        return HTMLResponse('<!doctype html><title>Owned connector callback</title><p>Authorization returned.</p>')

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'Owned native MCP browser')])
    now = datetime.datetime.now(datetime.UTC)
    certificate = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
        .public_key(key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(x509.SubjectAlternativeName([
            x509.DNSName('kestrel.example'), x509.DNSName('provider.example')]), critical=False)
        .sign(key, hashes.SHA256()))
    cert_path, key_path = tmp_path / 'owned-browser.pem', tmp_path / 'owned-browser-key.pem'
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    listeners = []
    host = Starlette(routes=[Host('kestrel.example', app=owner.app), Host('provider.example', app=client_app)])
    server = uvicorn.Server(uvicorn.Config(host, log_level='error',
        ssl_certfile=str(cert_path), ssl_keyfile=str(key_path)))
    original = None
    try:
        for address in ('127.0.0.1', '127.0.0.2'):
            listener = socket.create_server((address, urlsplit(owner.policy.console_origin).port or 443))
            listeners.append(listener)
        original = asyncio.create_task(server.serve(sockets=listeners))
        async with asyncio.timeout(5):
            while not server.started:
                if original.done():
                    await original
                    raise RuntimeError('Original native consent HTTPS listener retired during startup')
                await asyncio.sleep(0.01)
        yield callbacks
    finally:
        server.should_exit = True
        try:
            if original is not None:
                async with asyncio.timeout(5):
                    await original
        finally:
            for listener in listeners:
                listener.close()


@pytest.mark.timeout(60)
@pytest.mark.parametrize('assistant_agent', [{'features': ['MemoryFeature']}], indirect=True)
@pytest.mark.parametrize('decision', ['approve', 'deny'])
async def test_original_native_consent_browser_uses_owner_entry_and_exact_callback(
    publication, native_consent_https, tmp_path, decision,
):
    owner, http = publication
    private = 'Owned browser-authorized private conversation'
    await owner.agent.privacy_agent.add_conversation('user', private, session_id='owned-native-browser')
    owner.publish()
    verifier = 'owned-native-browser-verifier-' + 'v' * 43
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip('=')
    authorize = owner.resource + '/oauth/authorize?' + urlencode({
        'response_type': 'code', 'client_id': 'owned-provider',
        'redirect_uri': 'https://provider.example/callback', 'scope': 'memory:read',
        'resource': owner.resource, 'state': 'owned-native-browser-state',
        'code_challenge': challenge, 'code_challenge_method': 'S256'})
    script = tmp_path / 'owned-consent-browser.mjs'
    script.write_text('''import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
const require = createRequire(process.env.MCP_TEST_BROWSER_SUPPORT + '/web/package.json');
const { chromium } = require('playwright');
let raw = ''; for await (const chunk of process.stdin) raw += chunk;
const config = JSON.parse(raw);
const browser = await chromium.launch({ headless: true, args: ['--no-sandbox', '--no-proxy-server',
  '--host-resolver-rules=MAP kestrel.example 127.0.0.1,MAP provider.example 127.0.0.2'] });
try {
  const context = await browser.newContext({ ignoreHTTPSErrors: true });
  const page = await context.newPage();
  const failures = [], requests = [];
  page.on('pageerror', error => failures.push(error.message));
  page.on('request', request => requests.push(request.url()));
  await page.goto(config.authorize);
  await page.locator('#kestrel-api-key-input').waitFor();
  const consent = page.url();
  assert.equal(new URL(consent).search, '', 'Consent hint remained in browser history during owner entry');
  await page.locator('#kestrel-api-key-input').fill(config.key);
  await page.getByRole('button', { name: 'Unlock', exact: true }).click();
  await page.waitForFunction(() => !document.getElementById('consent-allow').disabled);
  assert.equal(await page.locator('#consent-client').textContent(), 'Owned provider');
  assert.equal(await page.locator('#consent-agent').textContent(), config.agent);
  assert.equal(await page.locator('#consent-resource').textContent(), config.resource);
  assert.deepEqual(await page.locator('#consent-scopes input:checked').evaluateAll(
    elements => elements.map(element => element.value)), ['memory:read']);
  await page.screenshot({ path: '/owned/consent-' + config.decision + '.png', fullPage: true });
  const selector = config.decision === 'approve' ? '#consent-allow' : '#consent-deny';
  await Promise.all([page.waitForURL('https://provider.example/callback?**'), page.locator(selector).click()]);
  assert.equal(new URL(page.url()).searchParams.get('state'), 'owned-native-browser-state');
  await page.goto(consent);
  await page.waitForFunction(() => document.getElementById('consent-status').textContent.includes('unavailable'));
  assert.equal(await page.locator('#consent-allow').isDisabled(), true);
  assert.equal(await page.locator('#consent-deny').isDisabled(), true);
  assert.equal(await page.evaluate(() => Object.keys(sessionStorage).some(
    key => key.startsWith('kestrel.mcp-server-consent.'))), false, 'Consumed consent hint survived');
  assert(requests.every(url => !url.includes(config.key)), 'Owner key entered a request URL');
  assert.deepEqual(failures, []);
  await context.close();
  process.stdout.write(JSON.stringify({ decision: config.decision, returned: true, consumedHint: true }));
} finally { await browser.close(); }
''')
    process = await asyncio.create_subprocess_exec('node', str(script), stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        output, errors = await asyncio.wait_for(process.communicate(json.dumps({
            'authorize': authorize, 'key': KEY, 'agent': owner.agent.agent_name,
            'resource': owner.resource, 'decision': decision}).encode()), 30)
    finally:
        if process.returncode is None:
            process.kill()
            await process.communicate()
    assert process.returncode == 0, errors.decode()
    assert json.loads(output) == {'decision': decision, 'returned': True, 'consumedHint': True}
    assert len(native_consent_https) == 1
    parameters, headers = native_consent_https[0]
    assert 'x-api-key' not in headers and 'authorization' not in headers and 'referer' not in headers
    assert parameters['state'] == 'owned-native-browser-state'
    assert parameters['iss'] == owner.resource + '/oauth'
    if decision == 'deny':
        assert parameters == {'state': 'owned-native-browser-state', 'error': 'access_denied',
            'iss': owner.resource + '/oauth'}
        assert await owner.backend.fetch_one('SELECT COUNT(*) FROM mcp_server_oauth_grants') == (0,)
        return
    assert set(parameters) == {'code', 'iss', 'state'}
    response = await http.post(owner.resource + '/oauth/token', data={
        'grant_type': 'authorization_code', 'code': parameters['code'], 'code_verifier': verifier,
        'redirect_uri': 'https://provider.example/callback', 'client_id': 'owned-provider',
        'client_secret': 'owned-secret', 'resource': owner.resource})
    assert response.status_code == 200, response.text
    result, failure, responses = await observe_native_call(owner, response.json()['access_token'],
        'recall_recent', {'limit': 1, 'operation_id': 'owned-browser-recall',
            'conversation_id': 'owned-native-browser'}, discover_first=True)
    assert result is not None and not result.is_error and failure is None
    assert any(private.encode() in body for _status, body in responses)
