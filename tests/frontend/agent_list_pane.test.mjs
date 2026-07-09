// #2279: `mountAgentListPane` is the agent/companion analogue of
// `mountConversationsPane` — the shared `mountAgentList` surface PLUS the SAME
// pane chrome (chevron collapse to fully-hidden, drag-resize with min/max +
// localStorage persistence) AND a component-owned "+ New" header action wired
// via `onNew`. The standalone console and any embedder consume this one export.
// These tests exercise the pane contract directly:
//   - mount builds/adopts chrome and mounts the list;
//   - the chevron closes the pane to fully hidden (#2216), persisted + restored;
//   - the resize handle clamps to min/max and persists the width;
//   - the "+ New" header action fires `onNew`;
//   - destroy() leaves ADOPTED chrome in place;
//   - a console-style adopt (with onNew) GAINS a new-agent affordance.

import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.Node = dom.window.Node;
globalThis.HTMLElement = dom.window.HTMLElement;
if (!globalThis.CSS || typeof globalThis.CSS.escape !== 'function') {
    globalThis.CSS = { escape: (s) => String(s).replace(/[^a-zA-Z0-9_-]/g, '\\$&') };
}
globalThis.location = dom.window.location;
globalThis.window.kicon = (name) => `<span class="ki ki-${name}" aria-hidden="true"></span>`;
globalThis.kicon = globalThis.window.kicon;

// In-memory localStorage so persistence is observable across mounts.
function makeStorage() {
    const map = new Map();
    return {
        getItem: (k) => (map.has(k) ? map.get(k) : null),
        setItem: (k, v) => { map.set(k, String(v)); },
        removeItem: (k) => { map.delete(k); },
        _map: map,
    };
}
globalThis.localStorage = makeStorage();

const { mountAgentListPane } = await import('../../kestrel_sovereign/static/js/agent_list.js');

function tick() { return new Promise((r) => setTimeout(r, 0)); }

function fakeAdapter(items = [], mode = 'multi_agent') {
    return { mode, listAgents: async () => items };
}

// Mirror index.html's static #agents-pane chrome (adopt path).
function makeConsolePane() {
    const el = document.createElement('aside');
    el.id = 'agents-pane';
    el.className = 'pane-sidebar';
    el.innerHTML = `
        <div class="pane-header"><h3>Agents</h3>
            <button id="collapse-agents-btn" class="collapse-btn"></button></div>
        <div id="agents-list" class="pane-content"></div>
        <div id="resize-agents" class="resize-handle"></div>`;
    document.body.appendChild(el);
    return el;
}

test('mount builds pane chrome (header, collapse rail, resize handle) into a bare container', async () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter([{ name: 'Emma', status: 'online' }]),
        storageKey: 'a:test-build',
    });
    await tick();

    assert.ok(el.classList.contains('pane-sidebar'), 'container becomes a pane-sidebar');
    assert.ok(el.classList.contains('agent-list-pane'), 'container tagged agent-list-pane');
    assert.ok(el.querySelector('.pane-header'), 'header built');
    assert.ok(el.querySelector('.collapse-btn'), 'collapse rail built');
    assert.ok(el.querySelector('.resize-handle'), 'resize handle built');
    assert.ok(el.querySelector('.agent-card'), 'list rows rendered inside the pane');
    assert.ok(handle.list, 'inner mountAgentList handle exposed');
    handle.destroy();
});

test('mount ADOPTS an existing static pane header + resize handle (console chrome)', async () => {
    const el = makeConsolePane();
    const headerBefore = el.querySelector('.pane-header');
    const handleBefore = el.querySelector('.resize-handle');
    const listBefore = el.querySelector('#agents-list');

    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter([{ name: 'Emma', status: 'online' }]), storageKey: 'a:test-adopt', autoLoad: false,
    });
    assert.equal(el.querySelectorAll('.pane-header').length, 1, 'no duplicate header');
    assert.equal(el.querySelector('.pane-header'), headerBefore, 'existing header adopted');
    assert.equal(el.querySelector('.resize-handle'), handleBefore, 'existing resize handle adopted');
    assert.equal(el.querySelectorAll('.resize-handle').length, 1, 'no duplicate resize handle');
    // The list mounts into the adopted #agents-list.
    assert.ok(listBefore.querySelector('.agent-list-root'), 'list mounted into adopted #agents-list');
    handle.destroy();
});

test('#2216: the chevron closes the agents pane to fully hidden (display:none), persisted + restored', () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const seen = [];
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter(), storageKey: 'a:test-collapse', autoLoad: false,
        onToggle: (c) => seen.push(c),
    });
    // The agents pane defaults OPEN (it is the primary nav surface).
    assert.equal(handle.collapsed, false, 'starts open by default');
    assert.notEqual(el.style.display, 'none', 'open pane is visible');

    // The chevron CLOSES to fully hidden — no leftover rail.
    el.querySelector('.collapse-btn').click();
    assert.equal(handle.collapsed, true, 'chevron closed the pane');
    assert.equal(el.style.display, 'none', 'closed pane takes zero width (display:none)');
    assert.ok(el.classList.contains('collapsed'), 'collapsed marker in lock-step with display');
    assert.equal(localStorage.getItem('a:test-collapse:collapsed'), '1', 'closed state persisted');
    // The chevron only closes — a second click does NOT reopen it.
    el.querySelector('.collapse-btn').click();
    assert.equal(handle.collapsed, true, 'chevron never reopens (open()/toggle() do)');
    handle.open();
    assert.equal(handle.collapsed, false, 'open() reveals the pane again');
    assert.equal(localStorage.getItem('a:test-collapse:collapsed'), '0', 'open state persisted');
    assert.deepEqual(seen, [false, true, false], 'onToggle fired for init + each change');
    handle.destroy();

    // A persisted CLOSED state is restored (hidden) on the next mount.
    const el2 = document.createElement('div');
    document.body.appendChild(el2);
    localStorage.setItem('a:test-collapse-restore:collapsed', '1');
    const handle2 = mountAgentListPane(el2, {
        adapter: fakeAdapter(), storageKey: 'a:test-collapse-restore', autoLoad: false,
    });
    assert.equal(handle2.collapsed, true, 'restored closed from localStorage');
    assert.equal(el2.style.display, 'none', 'restored closed pane is fully hidden');
    handle2.destroy();
});

test('the resize handle clamps to min/max and persists the width', () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter(), storageKey: 'a:test-resize', autoLoad: false,
        minWidth: 200, maxWidth: 500,
    });
    const rh = el.querySelector('.resize-handle');
    Object.defineProperty(el, 'offsetWidth', { value: 280, configurable: true });
    rh.dispatchEvent(new dom.window.MouseEvent('mousedown', { clientX: 300 }));
    document.dispatchEvent(new dom.window.MouseEvent('mousemove', { clientX: 9999 }));
    assert.equal(el.style.width, '500px', 'width clamped to maxWidth');
    document.dispatchEvent(new dom.window.MouseEvent('mousemove', { clientX: -9999 }));
    assert.equal(el.style.width, '200px', 'width clamped to minWidth');
    document.dispatchEvent(new dom.window.MouseEvent('mouseup', {}));
    assert.ok(localStorage.getItem('a:test-resize:width'), 'width persisted on mouseup');
    handle.destroy();

    // A persisted width is restored (clamped) on the next mount.
    localStorage.setItem('a:test-width:width', '9999');
    const el2 = document.createElement('div');
    document.body.appendChild(el2);
    const handle2 = mountAgentListPane(el2, {
        adapter: fakeAdapter(), storageKey: 'a:test-width', autoLoad: false, maxWidth: 500,
    });
    assert.equal(el2.style.width, '500px', 'oversized persisted width clamped to max');
    handle2.destroy();
});

test('the "+ New" header action fires onNew', () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    let fired = 0;
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter(), storageKey: 'a:test-new', autoLoad: false,
        onNew: () => { fired += 1; },
    });
    const newBtn = el.querySelector('.new-agent-btn');
    assert.ok(newBtn, 'component-owned New button built when onNew is provided');
    newBtn.click();
    assert.equal(fired, 1, 'clicking New fires onNew');
    handle.destroy();
});

test('no "+ New" button is built when onNew is omitted', () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter(), storageKey: 'a:test-nonew', autoLoad: false,
    });
    assert.equal(el.querySelector('.new-agent-btn'), null, 'no New button without an onNew hook');
    handle.destroy();
});

test('destroy() leaves ADOPTED chrome in place (only built chrome is removed)', () => {
    const el = makeConsolePane();
    const headerBefore = el.querySelector('.pane-header');
    const resizeBefore = el.querySelector('.resize-handle');
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter(), storageKey: 'a:test-destroy-adopt', autoLoad: false,
        onNew: () => {},
    });
    handle.destroy();
    assert.equal(el.querySelector('.pane-header'), headerBefore, 'adopted header survives destroy');
    assert.equal(el.querySelector('.resize-handle'), resizeBefore, 'adopted resize handle survives destroy');
});

test('destroy() removes BUILT chrome (bare container)', () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter(), storageKey: 'a:test-destroy-build', autoLoad: false,
        onNew: () => {},
    });
    assert.ok(el.querySelector('.pane-header'), 'header built');
    assert.ok(el.querySelector('.new-agent-btn'), 'New button built');
    handle.destroy();
    assert.equal(el.querySelector('.pane-header'), null, 'built header removed on destroy');
    assert.equal(el.querySelector('.new-agent-btn'), null, 'built New button removed on destroy');
});

test('standalone console (adopt) GAINS a new-agent affordance via onNew', async () => {
    // The console mounts into the static #agents-pane and passes onNew — so the
    // affordance it lacked today appears in the adopted header (same-everywhere).
    const el = makeConsolePane();
    let opened = 0;
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter([{ name: 'Emma', status: 'online' }]),
        storageKey: 'a:test-standalone-new',
        onNew: () => { opened += 1; },
    });
    await tick();
    const newBtn = el.querySelector('.pane-header .new-agent-btn');
    assert.ok(newBtn, 'new-agent affordance present in the console pane header');
    newBtn.click();
    assert.equal(opened, 1, 'the affordance drives the new-agent flow');
    handle.destroy();
});

test('the handle exposes list delegators (refresh/select/setActiveName/getActive) + pane controls', () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter(), storageKey: 'a:test-delegate', autoLoad: false,
    });
    assert.equal(typeof handle.refresh, 'function');
    assert.equal(typeof handle.select, 'function');
    assert.equal(typeof handle.setActiveName, 'function');
    assert.equal(typeof handle.getActive, 'function');
    assert.equal(typeof handle.open, 'function');
    assert.equal(typeof handle.close, 'function');
    assert.equal(typeof handle.toggle, 'function');
    assert.ok(handle.list, 'inner mountAgentList handle exposed');
    handle.destroy();
});

test('destroy() removes the BUILT resize handle from a bare container (codex P2)', async () => {
    const el = document.createElement('div');   // bare embed container
    document.body.appendChild(el);
    const handle = mountAgentListPane(el, {
        adapter: { mode: 'multi_agent', listAgents: async () => [] },
        storageKey: 'test:dz-pane',
    });
    await new Promise((r) => setTimeout(r, 0));
    assert.ok(el.querySelector('.resize-handle'), 'bare mount builds a resize handle');
    handle.destroy();
    assert.equal(el.querySelector('.resize-handle'), null, 'built handle removed on destroy');

    // Adopted chrome stays: pre-existing handle survives destroy.
    const el2 = document.createElement('div');
    const preHandle = document.createElement('div');
    preHandle.className = 'resize-handle';
    el2.appendChild(preHandle);
    document.body.appendChild(el2);
    const handle2 = mountAgentListPane(el2, {
        adapter: { mode: 'multi_agent', listAgents: async () => [] },
        storageKey: 'test:dz-pane2',
    });
    await new Promise((r) => setTimeout(r, 0));
    handle2.destroy();
    assert.ok(el2.querySelector('.resize-handle'), 'adopted handle survives destroy');
});
