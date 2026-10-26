import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { JSDOM } from 'jsdom';

let source = await readFile(new URL('../../kestrel_sovereign/features/visual_computer/static/visual-computer.js', import.meta.url), 'utf8');
source = source.replace("import API from '/js/api.js';", 'const API = {};')
    .replace("import { registerPanel } from '/js/ui-ext/panels.js';", 'const registerPanel = () => {};')
    .replace("import bus from '/js/ui-ext/bus.js';", 'const bus = { on() {} };');
const { mountVisualComputer } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const flush = () => new Promise(resolve => setTimeout(resolve, 0));

function setup() {
    const dom = new JSDOM('<main></main>');
    globalThis.document = dom.window.document;
    const calls = [];
    let agent = 'original';
    let changed;
    let screen;
    let controller = 'agent';
    let epoch = 0;
    const api = { getHostAgent: () => agent, onHostAgentChange: fn => { changed = fn; return () => {}; },
        async requestForAgent(path, options, owner) {
            calls.push({path, options, owner});
            if (path.endsWith('/sessions')) return [{session_id:'session', phase:'active'}];
            if (path.endsWith('/observation')) return new Promise(resolve => { screen = resolve; });
            if (path.endsWith('/takeover')) { controller = 'human'; epoch++; return {}; }
            if (path.endsWith('/handback')) { controller = 'agent'; epoch++; return {}; }
            if (path.endsWith('/stop')) { controller = 'none'; epoch++; return {}; }
            return {phase:controller==='agent'?'active':'fenced', session:{controller,control_epoch:epoch}, lease:{state:'ready'}};
        } };
    const body = document.querySelector('main');
    const view = mountVisualComputer(body, api);
    const click = text => [...body.querySelectorAll('button')].find(b => b.textContent === text).click();
    return {view, body, calls, click, resolveScreen: () => screen({frame:{media_type:'image/png'},image_base64:'eA=='}),
        switchAgent: () => {agent='other'; changed();}, dom};
}

test('Stop stays independently usable while a screen request is blocked', async () => {
    const h = setup();
    await h.view.refresh();
    h.click('Show current screen');
    await flush();
    h.click('Stop input');
    await flush();
    assert.ok(h.calls.some(call => call.path.endsWith('/stop')));
    h.resolveScreen();
    await flush();
    assert.equal(h.body.querySelector('img').hidden, true);
    assert.ok(h.calls.every(call => call.owner === 'original'));
    h.view.destroy();
});

test('takeover and agent changes discard late pixels and keep exact control request', async () => {
    const h = setup();
    await h.view.refresh();
    h.click('Show current screen');
    await flush();
    h.click('Take over');
    await flush();
    const call = h.calls.find(call => call.path.endsWith('/takeover'));
    const payload = JSON.parse(call.options.body);
    assert.equal(payload.session_id,'session');
    assert.equal(payload.expected_control_epoch,0);
    assert.equal(payload.operation,'takeover');
    assert.ok(payload.request_id);
    h.resolveScreen();
    await flush();
    assert.equal(h.body.querySelector('img').hasAttribute('src'),false);
    h.switchAgent();
    assert.equal(h.body.querySelector('select').children.length,0);
    h.view.destroy();
});
