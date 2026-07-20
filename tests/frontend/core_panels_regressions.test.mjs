// #2145 review regressions (PR #2150):
//   P2-1  A core panel gated off at boot must have its in-place `#panel-<id>`
//         body removed from the DOM, not just its tab — restoring the pre-#2145
//         `panelIsEnabled` prune contract. Before the fix `_syncNav` dropped only
//         `registryOwned` bodies, stranding a gated-off core panel's in-place
//         index.html body.
//   P2-2  An embedder that calls `destroy()` then `mountPanels()` again gets
//         fresh DOM. The runtime wiring (`_wired`) and its per-mount init/load
//         guards are module singletons, so without a reset the remount's panels
//         never re-init and their loaders are skipped — bodies stay on their
//         "Loading…" placeholder. `destroy()` must reset the runtime.

import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.Node = dom.window.Node;
globalThis.MutationObserver = dom.window.MutationObserver;
if (!globalThis.CSS || typeof globalThis.CSS.escape !== 'function') {
    globalThis.CSS = { escape: (s) => String(s).replace(/[^a-zA-Z0-9_-]/g, '\\$&') };
}
globalThis.location = dom.window.location;
globalThis.sessionStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
// The embed runtime imports the real panel loaders (identity/tasks/…). Those
// modules touch a few browser globals at import time; stub the minimum so the
// heavy chain loads under jsdom. The loaders themselves fail-soft on the 500
// fetch below, replacing their "Loading…" placeholder with an error state —
// which is exactly the "loader ran" signal P2-2 asserts on.
globalThis.fetch = async () => ({ ok: false, status: 500, json: async () => ({}), text: async () => '' });
globalThis.kicon = () => '';
dom.window.kicon = () => '';
globalThis.EventSource = class { constructor() {} close() {} addEventListener() {} };
globalThis.WebSocket = class { constructor() {} close() {} addEventListener() {} send() {} };

const { CORE_PANEL_DEFS } = await import(
    '../../kestrel_sovereign/static/js/ui-ext/core-panels.js'
);
const { mountPanels, registerCorePanels } = await import(
    '../../kestrel_sovereign/static/js/ui-ext/mount-panels.js'
);
const Panels = (await import('../../kestrel_sovereign/static/js/ui-ext/panels.js')).default;

function stubApi(caps = {}) {
    return { hasCapability: (k) => caps[k] !== false };
}

// Let queued async loaders (fetch → catch → DOM replace) drain.
async function flush() {
    for (let i = 0; i < 4; i++) {
        await Promise.resolve();
        await new Promise((r) => setTimeout(r, 0));
    }
}

// --- P2-1: gated-off core panel's in-place body is removed at boot -----------

test('P2-1 boot: a core panel gated off has its in-place body removed, not just its tab', () => {
    Panels._reset();
    document.body.innerHTML = '';
    const nav = document.createElement('div');
    nav.className = 'nav-tabs';
    const host = document.createElement('div');
    host.className = 'main-content';
    document.body.appendChild(nav);
    document.body.appendChild(host);

    // Reproduce index.html's static core tabs + in-place `#panel-<id>` bodies.
    for (const def of CORE_PANEL_DEFS) {
        const tab = document.createElement('button');
        tab.className = 'nav-tab';
        tab.dataset.panel = def.panelId;
        nav.appendChild(tab);
        const panel = document.createElement('div');
        panel.className = 'panel';
        panel.id = `panel-${def.panelId}`;
        panel.innerHTML = '<div class="panel-content"><div class="loading">Loading…</div></div>';
        host.appendChild(panel);
    }

    // Host opts out of `permissions`, which gates the `approvals` panel off.
    const caps = { permissions: false };
    const api = { hasCapability: (k) => caps[k] !== false };
    registerCorePanels({ api });
    Panels.renderNav({ navEl: nav, hostEl: host, ctx: { api } });

    assert.equal(document.getElementById('panel-approvals'), null,
        'gated-off core panel in-place body removed from the DOM (P2-1)');
    assert.equal(nav.querySelector('.nav-tab[data-panel="approvals"]'), null,
        'gated-off core panel tab removed too');
    // A surviving core panel keeps its in-place body untouched.
    assert.ok(document.getElementById('panel-identity'), 'ungated panel body preserved');

    // And a runtime re-enable brings a fresh (registry-owned) body back.
    caps.permissions = true;
    Panels.syncNav();
    assert.ok(document.getElementById('panel-approvals'),
        'panel body recreated when the capability is re-enabled');
});

// --- P2-2: destroy() resets the embed runtime so a remount re-inits/reloads --

test('P2-2 remount: destroy()+mountPanels() again re-runs loaders (body not stuck on "Loading…")', async () => {
    Panels._reset();
    document.body.innerHTML = '';

    const container = document.createElement('div');
    document.body.appendChild(container);
    const h1 = await mountPanels(container, {
        api: stubApi(),
        loadFeatures: false,
        wireRuntime: true,
        activateFirst: false,
    });
    h1.activate('identity');
    await flush();
    // First mount actually ran the identity loader (placeholder replaced).
    const card1 = container.querySelector('#panel-identity #identity-card');
    assert.ok(card1 && !/Loading identity/.test(card1.textContent),
        'first mount: identity loader ran');
    h1.destroy();

    // Fresh remount into a new container (same shared module singletons).
    const container2 = document.createElement('div');
    document.body.appendChild(container2);
    const h2 = await mountPanels(container2, {
        api: stubApi(),
        loadFeatures: false,
        wireRuntime: true,
        activateFirst: false,
    });
    h2.activate('identity');
    await flush();

    const card2 = container2.querySelector('#panel-identity #identity-card');
    assert.ok(card2, 'identity body built on remount');
    // The regression: without destroy() resetting the runtime, the per-mount
    // load guard from mount #1 persists and the loader is skipped, leaving this
    // fresh body stuck on its "Loading identity…" placeholder.
    assert.ok(!/Loading identity/.test(card2.textContent),
        'remount: identity loader re-ran — body not stuck on its loading placeholder (P2-2)');

    h2.destroy();
});

test('mountPanels wires database explorer to the embed-scoped API without app.js', async () => {
    Panels._reset();
    document.body.innerHTML = '';

    const databaseCalls = [];
    const api = {
        hasCapability: () => true,
        getHostAgent: () => 'Embedded Agent',
        onHostAgentChange: () => () => {},
        async getDbTables(agent) {
            databaseCalls.push(agent);
            return { tables: [], table_count: 0, db_size: 0 };
        },
        async queryDbTable() {
            throw new Error('not used');
        },
    };
    const container = document.createElement('div');
    document.body.appendChild(container);
    const handle = await mountPanels(container, {
        api,
        loadFeatures: false,
        wireRuntime: true,
        activateFirst: false,
    });

    handle.activate('sovereignty');
    const toggle = container.querySelector('#toggle-db-explorer');
    assert.ok(toggle, 'embed runtime rendered the explorer control');
    assert.equal(toggle.getAttribute('onclick'), null, 'control does not depend on inline wiring');
    assert.equal(typeof window.toggleDbExplorer, 'function',
        'database runtime is imported without the standalone app entry point');

    toggle.click();
    await flush();

    assert.deepEqual(databaseCalls, ['Embedded Agent'],
        'database request uses config.api and its selected agent');
    assert.equal(container.querySelector('#db-explorer-section').style.display, 'block');

    handle.destroy();
});
