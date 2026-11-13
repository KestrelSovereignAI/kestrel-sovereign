"""Native Console forms against actual Core/MCP, inside an owned PID namespace."""

import asyncio
import json
import os
from pathlib import Path
import socket

from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
import kestrel_sovereign
import pytest
import pytest_asyncio
import uvicorn
from kestrel_sovereign import server as core_server
from kestrel_sovereign.features.security.permissions import PermissionLevel

from tests.integration.test_mcp_apps_host import (
    apps_boundary as apps_boundary, mcp_browser_support as mcp_browser_support,
)
from tests.integration.test_mcp_input_host import wire
from tests.utils.native_streaming import prepare_streaming_agent, install_input_checkpoint_provider

RICH_SCHEMA = {'type': 'object', 'properties': {
    'city': {'type': 'string', 'minLength': 2, 'maxLength': 64},
    'count': {'type': 'integer', 'minimum': 1, 'maximum': 4, 'default': 2},
    'ratio': {'type': 'number', 'minimum': 0, 'maximum': 1, 'default': .5},
    'enabled': {'type': 'boolean', 'default': False},
    'color': {'type': 'string', 'oneOf': [{'const': '#a', 'title': 'Amber'},
        {'const': '#b', 'title': 'Blue'}], 'default': '#a'},
    'labels': {'type': 'array', 'items': {'anyOf': [{'const': 'a', 'title': 'Alpha'},
        {'const': 'b', 'title': 'Beta'}, {'const': 'c', 'title': 'Gamma'}]},
        'minItems': 1, 'maxItems': 2, 'default': ['a']},
    'email': {'type': 'string', 'format': 'email', 'default': 'user@example.test'},
    'link': {'type': 'string', 'format': 'uri', 'default': 'https://example.test/path'},
    'day': {'type': 'string', 'format': 'date', 'default': '2026-11-11'},
    'time': {'type': 'string', 'format': 'date-time', 'default': '2026-11-11T12:00:00Z'},
    '__proto__': {'type': 'string', 'title': 'Safe key', 'default': 'ordinary-value'},
}, 'required': ['city', 'count', 'ratio', 'enabled', 'color', 'labels', 'email', 'link', 'day', 'time', '__proto__']}
RICH_CONTENT = {'city': 'owned-private-answer', 'count': 3, 'ratio': .25, 'enabled': False,
    'color': '#b', 'labels': ['a', 'c'], 'email': 'user@example.test', 'link': 'https://example.test/path',
    'day': '2026-11-11', 'time': '2026-11-11T12:00:00Z', '__proto__': 'ordinary-value'}


@pytest.fixture
def input_listener():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        yield listener


@pytest.fixture
def input_host_origin(input_listener):
    return f'http://127.0.0.1:{input_listener.getsockname()[1]}'


@pytest.fixture
def native_console_scripts():
    return {}


@pytest_asyncio.fixture
async def native_console_origin(apps_boundary, input_listener, monkeypatch, native_console_scripts):
    import kestrel_feature_mcp

    assert Path('/.dockerenv').is_file(), 'Native browser tests require an owned process namespace'
    assert not Path('/var/run/docker.sock').exists() and not os.environ.get('DOCKER_HOST')
    monkeypatch.setattr(core_server, 'SERVE_UI', True)
    app = apps_boundary.app

    @app.get('/js/owned-input-test.js')
    async def test_javascript():
        return Response(native_console_scripts['entry'], media_type='text/javascript')

    static = Path(kestrel_sovereign.__file__).parent / 'static'
    app.mount('/js', StaticFiles(directory=static / 'js'))
    app.mount('/shared', StaticFiles(directory=static / 'shared'))
    app.mount('/features/mcpagent/static', StaticFiles(
        directory=Path(kestrel_feature_mcp.__file__).resolve().parent / 'apps_sandbox'))
    server = uvicorn.Server(uvicorn.Config(app, log_level='error'))
    serving = asyncio.create_task(server.serve(sockets=[input_listener]))
    try:
        async with asyncio.timeout(5):
            while not server.started:
                if serving.done():
                    await serving
                    raise RuntimeError('Owned input listener exited before startup')
                await asyncio.sleep(.01)
        yield f'http://127.0.0.1:{input_listener.getsockname()[1]}'
    finally:
        server.should_exit = True
        await asyncio.wait_for(serving, 5)


@pytest.mark.mcp_integration
@pytest.mark.timeout(60)
@pytest.mark.parametrize('apps_boundary', [
    {'operator_input': 'form', 'input_message': '<img src=x onerror="window.untrustedInputRan=true">Private city?'},
    {'operator_input': 'form', 'input_tool': 'mutate_marker', 'browser_app': True},
    {'operator_input': 'form', 'input_schema': RICH_SCHEMA, 'input_content': RICH_CONTENT},
    {'operator_input': 'form', 'action': 'decline'},
    {'operator_input': 'form', 'action': 'cancel'},
    {'operator_input': 'form', 'input_schema': {'type': 'object', 'properties': {
        'city': {'type': 'string', 'maxLength': 0},
        'labels': {'type': 'array', 'items': {'type': 'string', 'enum': ['a', 'b']}, 'minItems': 0},
        'optional_labels': {'type': 'array', 'items': {'type': 'string', 'enum': ['a', 'b']}, 'minItems': 1},
    }, 'required': ['city', 'labels']}, 'input_content': {'city': '', 'labels': []}},
    {'operator_input': 'form', 'input_tool': 'mutate_marker', 'browser_app': True,
        'input_schema': {'type': 'object', 'properties': {'city': {'type': 'string', 'format': 'uri'}},
            'required': ['city']}, 'input_content': {'city': 'https://example.test/private-path'}},
    *[{'operator_input': 'form', **mode, 'input_schema': {'type': 'object', 'properties': {
        'city': {'type': 'string'}, 'count': {'type': 'integer', 'minimum': .5, 'maximum': 2}},
        'required': ['city', 'count']}, 'input_content': {'city': 'owned-private-answer', 'count': 1}}
        for mode in [{}, {'input_tool': 'mutate_marker', 'browser_app': True}]],
    {'operator_input': 'form', 'input_schema': {'type': 'object', 'properties': {
        'city': {'type': 'string', 'maxLength': 1}}, 'required': ['city']},
        'input_content': {'city': '\U0001f422'}},
    {'operator_input': 'form', 'input_schema': {'type': 'object', 'properties': {
        'city': {'type': 'string'}, 'contact': {'type': 'string', 'format': 'email'}},
        'required': ['city', 'contact']},
        'input_content': {'city': 'owned-private-answer', 'contact': '"a b"@example.test'}},
], indirect=True)
async def test_native_console_input_card_and_private_dom_retirement(
    apps_boundary, tmp_path, monkeypatch, request, mcp_browser_support, native_console_origin, native_console_scripts,
):
    assert Path('/.dockerenv').is_file(), 'Native browser tests require an owned process namespace'
    assert not Path('/var/run/docker.sock').exists() and not os.environ.get('DOCKER_HOST')
    b = apps_boundary
    settings = request.node.callspec.params['apps_boundary']
    app_mode, action = settings.get('browser_app', False), settings.get('action', 'accept')
    if app_mode:
        tool = next(item for item in b.tools if item.tool.native_tool.name == 'mutate_marker')
        await b.permissions.set_permission('mcp:owned-apps', tool.binding.name, PermissionLevel.ALLOW)
    monkeypatch.setattr(core_server, 'SERVE_UI', True)
    repository = mcp_browser_support
    build = await asyncio.create_subprocess_exec('node', 'web/test-support/parent-browser-build.mjs', 'input',
        cwd=repository, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    script, errors = await asyncio.wait_for(build.communicate(), 10)
    assert build.returncode == 0, errors.decode()

    @b.app.get('/')
    async def page():
        return HTMLResponse('<!doctype html><head><link rel="stylesheet" href="/features/mcpagent/static/input-entry.css">'
            '</head><body><script type="module" src="/js/owned-input-test.js"></script>')

    native_console_scripts['entry'] = script

    child = None
    try:
        driver = 'input-app-browser-driver.mjs' if app_mode else 'input-browser-driver.mjs'
        child = await asyncio.create_subprocess_exec('node', 'web/test-support/' + driver,
            native_console_origin, '/input-browser.png',
            json.dumps(settings.get('input_content')), action, cwd=repository,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        output, errors = await asyncio.wait_for(child.communicate(), 45)
        assert child.returncode == 0, errors.decode()
        receipt = json.loads(output)
        assert receipt and all(receipt.values())
        assert b.markers == ([] if action != 'accept' else
            ['original-marker', 'app-effect'] if app_mode else ['original-marker'])
        assert len(wire(b)) == 2
        row, = await b.feature._input_host.store.database.fetchall('SELECT state FROM mcp_inputs')
        assert row == ('RESOLVED',)
    finally:
        if child is not None and child.returncode is None:
            child.terminate()
            await asyncio.wait_for(child.wait(), 5)


@pytest.mark.mcp_integration
@pytest.mark.timeout(60)
@pytest.mark.parametrize('apps_boundary', [{'operator_input': 'form', 'input_tool': 'open_marker'}], indirect=True)
@pytest.mark.parametrize('dispatch', ['inline', 'batch'])
@pytest.mark.parametrize('followup', [False, True])
async def test_native_live_console_retracts_only_original_turn_prose(
        apps_boundary, monkeypatch, mcp_browser_support, native_console_origin, native_console_scripts, dispatch, followup):
    from fastapi import Request
    from kestrel_sovereign.endpoints import agent as agent_endpoints
    from kestrel_sovereign.endpoints import conversations as conversation_endpoints

    b = apps_boundary
    registered = {item.tool.native_tool.name: item for item in b.tools}
    await b.permissions.set_permission('mcp:owned-apps', registered['record_marker'].binding.name, PermissionLevel.ALLOW)
    prepare_streaming_agent(b.agent, tools=b.agent._build_all_tools())
    gate = asyncio.Event()
    provider_calls, _ = install_input_checkpoint_provider(b.agent, registered, monkeypatch,
        dispatch=dispatch, followup=followup, before_tools=gate.wait, tool_markers=True, interleaved_parts=True)
    b.app.state.agent = b.agent
    b.app.state.limiter = agent_endpoints.limiter
    b.app.include_router(agent_endpoints.router)
    b.app.include_router(conversation_endpoints.router)

    @b.app.post('/api/owned-console-bootstrap')
    async def bootstrap():
        return {'parts': [{'type': 'notice', 'data': 'PRIOR-CONVERSATION-MUST-REMAIN'}]}

    @b.app.post('/api/owned-release-stream')
    async def release(request: Request):
        assert request.state.caller is not None
        gate.set()
        return {'released': True}

    entry = """import { mountNativeConsole } from './native-console.js';
const native = await mountNativeConsole({getToken:()=> 'owned-apps-host-api-key',
 renderer:['/features/mcpagent/static/parent-entry.js','/features/mcpagent/static/input-entry.js'],
 endpoint:'/api/owned-console-bootstrap', body:{}, partType:'notice'});
const ui=await import('/js/ui.js');
native.chat.initChat(); ui.state.useStreaming=true;
native.chat.mountChatPane(null); native.pane.sessionId='original-chat';
const other=ui.getOrCreateChatPane('another-owned-pane');
await native.chat.addMessage('agent','OTHER-PANE-MUST-REMAIN',other.element);
document.body.append(other.element);
window.startOwnedStream=()=>{document.getElementById('message-input').value='Run the owned tools once';
 window.ownedStreamDone=native.chat.sendMessage();};
window.releaseOwnedStream=async()=>{await native.api.requestForAgent('/api/owned-release-stream',
 {method:'POST',body:'{}'},null,true); await window.ownedStreamDone;};
window.reloadOwnedConversation=async()=>{await import('/js/identity.js');
 await window.loadConversation('original-chat',{force:true,expectedAgent:null});};
window.ownedConsoleReady=true;
"""
    builder = """import {build} from 'esbuild';
const result=await build({stdin:{contents:process.argv[1],resolveDir:process.cwd()+'/test-support'},
 bundle:true,format:'esm',platform:'browser',target:'es2022',write:false,
 external:['/js/*','/shared/*','/features/*']});
process.stdout.write(result.outputFiles[0].text);
"""
    child = await asyncio.create_subprocess_exec('node', '--input-type=module', '-e', builder, entry,
        cwd=mcp_browser_support / 'web', stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    script, errors = await asyncio.wait_for(child.communicate(), 10)
    assert child.returncode == 0, errors.decode()

    native_console_scripts['entry'] = script

    @b.app.get('/')
    async def page():
        return HTMLResponse('<!doctype html><head><link rel="stylesheet" href="/features/mcpagent/static/input-entry.css">'
            '</head><body><div id="chat-container"></div><textarea id="message-input"></textarea>'
            '<button id="send-button">Send</button><div id="thinking-indicator"></div>'
            '<script type="module" src="/js/owned-input-test.js"></script></body>')

    driver = """const {chromium}=require('@playwright/test');
(async()=>{const browser=await chromium.launch(); const page=await browser.newPage(); const errors=[];
const transport=[]; page.on('response',r=>transport.push([r.status(),r.url()]));
 page.on('requestfailed',r=>transport.push(['failed',r.url(),r.failure()]));
 page.on('console',m=>{if(m.type()==='error')transport.push(['console-error',m.text()]);});
 page.on('pageerror',error=>{errors.push(error.message);page.evaluate(message=>window.ownedConsoleFailure=message,error.message).catch(()=>{});});
try {await page.goto(process.argv[1]);
 await page.waitForFunction(()=>window.ownedConsoleReady || window.ownedConsoleFailure);
 if (await page.evaluate(()=>window.ownedConsoleFailure)) throw new Error(await page.evaluate(()=>window.ownedConsoleFailure));
 await page.getByText('PRIOR-CONVERSATION-MUST-REMAIN',{exact:true}).waitFor({state:'visible'});
 await page.getByText('OTHER-PANE-MUST-REMAIN',{exact:true}).waitFor({state:'visible'});
 await page.evaluate(()=>window.startOwnedStream());
 await page.waitForFunction(()=>document.body.textContent.includes('UNFINISHED-PRE-INPUT-PROSE'));
 await page.evaluate(()=>window.releaseOwnedStream());
 await page.getByRole('button',{name:'Review MCP input',exact:true}).click();
 const text=await page.locator('body').innerText();
 const receipt={unfinishedRemoved:!text.includes('UNFINISHED'),
  waiting:text.includes('waiting for operator input'),
  prior:text.includes('PRIOR-CONVERSATION-MUST-REMAIN'), other:text.includes('OTHER-PANE-MUST-REMAIN'),
  forms:await page.locator('.mcp-input-card').count(),
  toolCards:await page.locator('.tool-activity-call').count(), errors};
 await page.evaluate(()=>window.reloadOwnedConversation());
 const reloaded=await page.locator('body').innerText();
 receipt.reloaded={unfinishedRemoved:!reloaded.includes('UNFINISHED'),
  waiting:reloaded.includes('waiting for operator input'),
  forms:await page.locator('.mcp-input-card').count(),
  toolCards:await page.locator('.tool-activity-call').count()};
 process.stdout.write(JSON.stringify(receipt));
} catch(error) {throw new Error(JSON.stringify({message:error.message,errors,transport,body:await page.locator('body').innerText()}));}
 finally {await browser.close();}})().catch(error=>{console.error(error);process.exitCode=1});
"""
    child = None
    try:
        child = await asyncio.create_subprocess_exec('node', '-e', driver, native_console_origin,
            cwd=mcp_browser_support / 'web', stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        output, errors = await asyncio.wait_for(child.communicate(), 45)
        assert child.returncode == 0, errors.decode()
        receipt = json.loads(output)
        assert receipt['unfinishedRemoved'], 'Live Console retained unfinished provider prose after the input checkpoint'
        assert receipt['waiting'] and receipt['prior'] and receipt['other'], receipt
        assert receipt['forms'] == 1, 'Live Console duplicated its original pending form card'
        assert receipt['toolCards'] > 0 and receipt['errors'] == []
        assert receipt['reloaded'] == {'unfinishedRemoved': True, 'waiting': True, 'forms': 1, 'toolCards': 2}, receipt
        assert b.markers == ['completed-before-input'] and len(wire(b)) == 1
        assert len(provider_calls) == (2 if followup else 1)
    finally:
        gate.set()
        if child is not None and child.returncode is None:
            child.terminate()
            await asyncio.wait_for(child.wait(), 5)
