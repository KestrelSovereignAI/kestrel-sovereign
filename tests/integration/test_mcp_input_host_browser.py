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
import uvicorn
from kestrel_sovereign import server as core_server
from kestrel_sovereign.features.security.permissions import PermissionLevel

from tests.integration.test_mcp_apps_host import (
    apps_boundary as apps_boundary, mcp_browser_support as mcp_browser_support,
)
from tests.integration.test_mcp_input_host import wire

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
    apps_boundary, tmp_path, monkeypatch, input_listener, request, mcp_browser_support,
):
    import kestrel_feature_mcp
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
            '</head><body><script type="module" src="/js/input-test.js"></script>')

    @b.app.get('/js/input-test.js')
    async def javascript():
        return Response(script, media_type='text/javascript')

    static = Path(kestrel_sovereign.__file__).parent / 'static'
    b.app.mount('/js', StaticFiles(directory=static / 'js'))
    b.app.mount('/shared', StaticFiles(directory=static / 'shared'))
    b.app.mount('/features/mcpagent/static', StaticFiles(
        directory=Path(kestrel_feature_mcp.__file__).resolve().parent / 'apps_sandbox'))
    listener = input_listener
    server = uvicorn.Server(uvicorn.Config(b.app, log_level='error'))
    serving = asyncio.create_task(server.serve(sockets=[listener]))
    child = None
    try:
        async with asyncio.timeout(5):
            while not server.started:
                if serving.done():
                    await serving
                    raise RuntimeError('Owned input listener exited before startup')
                await asyncio.sleep(.01)
        driver = 'input-app-browser-driver.mjs' if app_mode else 'input-browser-driver.mjs'
        child = await asyncio.create_subprocess_exec('node', 'web/test-support/' + driver,
            f'http://127.0.0.1:{listener.getsockname()[1]}', '/input-browser.png',
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
        server.should_exit = True
        await asyncio.wait_for(serving, 5)
