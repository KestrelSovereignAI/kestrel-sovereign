// #2145: core panels migrated onto the ui-ext panel registry + an embeddable
// `mountPanels` host. These tests cover:
//   (a) every core panel is a registry contribution that gates correctly against
//       a capabilities map (mirrors the retired PANEL_CAPABILITIES behavior);
//   (b) mountPanels into a DETACHED container renders the gated nav-tab strip,
//       lazily builds a panel body on first activation, and destroy() cleans up;
//   (c) a gated-off capability removes the tab (standalone parity).

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
// api.js constructs its client from these globals at import time.
globalThis.location = dom.window.location;
globalThis.sessionStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.fetch = async () => ({ ok: false, status: 500, json: async () => ({}) });

const { CORE_PANEL_DEFS, buildCorePanelBody } = await import(
    '../../kestrel_sovereign/static/js/ui-ext/core-panels.js'
);
const { mountPanels, registerCorePanels } = await import(
    '../../kestrel_sovereign/static/js/ui-ext/mount-panels.js'
);
const Panels = (await import('../../kestrel_sovereign/static/js/ui-ext/panels.js')).default;

// A capabilities-driven stub API: every key defaults ON, listed `false` keys off.
function stubApi(caps = {}) {
    return { hasCapability: (k) => caps[k] !== false };
}

// Reconstruct the retired PANEL_CAPABILITIES semantics for parity assertions.
const EXPECTED = {
    identity: ['identity'],
    constitution: ['constitution'],
    memories: ['memory'],
    tasks: ['tasks'],
    sovereignty: ['sovereignty'],
    resources: ['keys', 'wallet'],
    metrics: ['metrics'],
    features: ['featureStore'],
    security: ['audit', 'permissions'],
    approvals: ['permissions'],
};

// --- (a) every core panel registers + gates correctly ----------------------

test('CORE_PANEL_DEFS covers exactly the migrated core panels in nav order', () => {
    assert.deepEqual(
        CORE_PANEL_DEFS.map((d) => d.panelId),
        ['identity', 'constitution', 'memories', 'tasks', 'sovereignty',
            'resources', 'metrics', 'features', 'security', 'approvals'],
    );
    // Every def is a proper contribution: id + label + gate + body markup.
    for (const def of CORE_PANEL_DEFS) {
        assert.equal(typeof def.panelId, 'string');
        assert.ok(def.label, `${def.panelId} needs a label`);
        assert.equal(typeof def.gate, 'function', `${def.panelId} needs a gate`);
        assert.ok(def.bodyHtml && def.bodyHtml.trim().length > 0, `${def.panelId} needs body markup`);
    }
});

test('each core panel gate matches "any listed capability on"', () => {
    const combos = [
        {},
        { keys: false, wallet: false },
        { audit: false },
        { permissions: false },
        { audit: false, permissions: false },
        { keys: false },
        { wallet: false },
        { memory: false },
    ];
    for (const caps of combos) {
        const api = stubApi(caps);
        for (const def of CORE_PANEL_DEFS) {
            const want = EXPECTED[def.panelId].some((c) => api.hasCapability(c));
            assert.equal(!!def.gate(api), want,
                `gate(${def.panelId}) wrong for caps=${JSON.stringify(caps)}`);
        }
    }
});

test('buildCorePanelBody fills an empty container but never clobbers an in-place body', () => {
    const def = CORE_PANEL_DEFS.find((d) => d.panelId === 'memories');

    const empty = document.createElement('div');
    assert.equal(buildCorePanelBody(empty, def), true, 'built into an empty container (embed case)');
    assert.ok(empty.querySelector('#memory-list'), 'memories body markup injected');

    const inPlace = document.createElement('div');
    inPlace.innerHTML = '<span id="sentinel">already here</span>';
    assert.equal(buildCorePanelBody(inPlace, def), false, 'no-op when body pre-exists (standalone case)');
    assert.ok(inPlace.querySelector('#sentinel'), 'in-place body preserved');
});

// --- (b) mountPanels into a detached container -----------------------------

test('mountPanels renders gated tabs, lazily builds a body on first activation, and destroy() cleans up', async () => {
    Panels._reset();
    // The embedder's own container, attached same-document (as a real host
    // does) — NOT index.html markup. The registry resolves `#panel-<id>` via
    // document.getElementById, so the host must be in-document to activate.
    const container = document.createElement('div');
    document.body.appendChild(container);

    const handle = await mountPanels(container, {
        api: stubApi(),
        loadFeatures: false,
        wireRuntime: false,
        activateFirst: false,
    });

    const nav = container.querySelector('.nav-tabs');
    const host = container.querySelector('.main-content');
    assert.ok(nav, 'nav-tabs element created inside the container');
    assert.ok(host, 'panel host element created inside the container');

    // Full gated nav rendered same-document, without index.html markup existing.
    const tabIds = [...nav.querySelectorAll('.nav-tab')].map((t) => t.dataset.panel);
    assert.deepEqual(tabIds, CORE_PANEL_DEFS.map((d) => d.panelId),
        'every core panel tab is rendered in order');

    // Body is lazily rendered — the registry-created container starts empty.
    const memoriesPanel = host.querySelector('#panel-memories .panel-content');
    assert.ok(memoriesPanel, 'registry created the #panel-memories container');
    assert.equal(memoriesPanel.children.length, 0, 'body not built until first activation');

    // First activation builds the body and marks the tab/panel active.
    handle.activate('memories');
    assert.ok(memoriesPanel.querySelector('#memory-list'), 'body built on first activation');
    assert.ok(nav.querySelector('.nav-tab[data-panel="memories"]').classList.contains('active'));
    assert.ok(host.querySelector('#panel-memories').classList.contains('active'));

    // Activating another panel swaps active state (single active panel).
    handle.activate('security');
    assert.ok(host.querySelector('#panel-security .panel-content').querySelector('#permission-tree'),
        'security body built on activation');
    assert.equal(host.querySelector('#panel-memories').classList.contains('active'), false);
    assert.ok(host.querySelector('#panel-security').classList.contains('active'));

    // destroy() removes the mounted nav + host from the container.
    handle.destroy();
    assert.equal(container.querySelector('.nav-tabs'), null, 'nav removed on destroy');
    assert.equal(container.querySelector('.main-content'), null, 'host removed on destroy');
    container.remove();
});

// --- (c) a gated-off capability removes the tab (standalone parity) ---------

test('mountPanels omits a tab whose capability is opted out', async () => {
    Panels._reset();
    const container = document.createElement('div');

    // Host opts out of memory + the whole resources gate (keys AND wallet) +
    // both security sub-caps.
    const handle = await mountPanels(container, {
        api: stubApi({ memory: false, keys: false, wallet: false, audit: false, permissions: false }),
        loadFeatures: false,
        wireRuntime: false,
        activateFirst: false,
    });

    const nav = container.querySelector('.nav-tabs');
    const tabIds = [...nav.querySelectorAll('.nav-tab')].map((t) => t.dataset.panel);
    assert.ok(!tabIds.includes('memories'), 'memory-gated tab removed');
    assert.ok(!tabIds.includes('resources'), 'resources tab removed when keys AND wallet are off');
    assert.ok(!tabIds.includes('security'), 'security tab removed when audit AND permissions are off');
    // approvals is gated on permissions:false too.
    assert.ok(!tabIds.includes('approvals'), 'approvals tab removed when permissions is off');
    // Ungated-on panels survive.
    assert.ok(tabIds.includes('identity'));
    assert.ok(tabIds.includes('metrics'));

    handle.destroy();
});

test('resources tab survives when only ONE of keys/wallet is on (composite gate)', async () => {
    Panels._reset();
    const container = document.createElement('div');
    const handle = await mountPanels(container, {
        api: stubApi({ keys: false }), // wallet still on
        loadFeatures: false,
        wireRuntime: false,
        activateFirst: false,
    });
    const nav = container.querySelector('.nav-tabs');
    const tabIds = [...nav.querySelectorAll('.nav-tab')].map((t) => t.dataset.panel);
    assert.ok(tabIds.includes('resources'), 'resources stays visible while wallet is on');
    handle.destroy();
});

test('a core tab toggled off→on at runtime re-inserts at its original position (#2041 order preserved)', async () => {
    Panels._reset();
    const container = document.createElement('div');
    document.body.appendChild(container);

    // Mutable capabilities so a gate can flip at runtime without a reload.
    const caps = {};
    const api = { hasCapability: (k) => caps[k] !== false };

    const handle = await mountPanels(container, {
        api,
        loadFeatures: false,
        wireRuntime: false,
        activateFirst: false,
    });
    const nav = container.querySelector('.nav-tabs');
    const ids = () => [...nav.querySelectorAll('.nav-tab')].map((t) => t.dataset.panel);
    const full = CORE_PANEL_DEFS.map((d) => d.panelId);
    assert.deepEqual(ids(), full, 'all core tabs render in order at mount');

    // Toggle a MID-STRIP core panel off, then back on (the #2041 use case).
    caps.metrics = false;
    Panels.syncNav();
    assert.ok(!ids().includes('metrics'), 'metrics tab removed when gated off');

    caps.metrics = true;
    Panels.syncNav();
    // Regression: without a `before` anchor the rebuilt tab appended to the end.
    assert.deepEqual(ids(), full, 'metrics re-inserts in its original position, not at the end');

    handle.destroy();
    container.remove();
});

// --- (d) host-provided tabs (embed parity — Chat-first) ---------------------

test('hostTabs renders a host-element tab first and adopts the live element on activation', async () => {
    Panels._reset();
    const container = document.createElement('div');
    document.body.appendChild(container);

    // The host owns a live chat element parked elsewhere in its own DOM.
    const hostHome = document.createElement('div');
    document.body.appendChild(hostHome);
    const chatEl = document.createElement('div');
    chatEl.id = 'host-chat';
    chatEl.__liveMarker = Symbol('sse-listeners'); // survives ⇒ same node identity
    hostHome.appendChild(chatEl);

    const handle = await mountPanels(container, {
        api: stubApi(),
        loadFeatures: false,
        wireRuntime: false,
        activateFirst: false,
        hostTabs: [{ panelId: 'chat', label: 'Chat', element: chatEl }],
    });

    const nav = container.querySelector('.nav-tabs');
    const host = container.querySelector('.main-content');
    const tabIds = [...nav.querySelectorAll('.nav-tab')].map((t) => t.dataset.panel);
    assert.equal(tabIds[0], 'chat', 'host tab (no `before`, registered first) lands first');
    assert.deepEqual(tabIds, ['chat', ...CORE_PANEL_DEFS.map((d) => d.panelId)]);

    // Not adopted until activated (lazy render, same as core panels).
    assert.equal(chatEl.parentNode, hostHome, 'host element stays home until its tab is shown');

    handle.activate('chat');
    const chatPanel = host.querySelector('#panel-chat');
    assert.ok(chatPanel, 'registry created the #panel-chat container');
    assert.ok(chatPanel.classList.contains('active'), 'chat panel active');
    const adopted = host.querySelector('#panel-chat #host-chat');
    assert.equal(adopted, chatEl, 'SAME node moved into the panel body (no clone)');
    assert.equal(adopted.__liveMarker, chatEl.__liveMarker, 'live listeners/state intact');

    handle.destroy();
    container.remove();
    hostHome.remove();
});

test('switching to/from a host tab preserves node identity (no detach/reattach, no clone)', async () => {
    Panels._reset();
    const container = document.createElement('div');
    document.body.appendChild(container);
    const chatEl = document.createElement('div');
    chatEl.id = 'host-chat';
    const marker = Symbol('live');
    chatEl.__liveMarker = marker;

    const handle = await mountPanels(container, {
        api: stubApi(),
        loadFeatures: false,
        wireRuntime: false,
        activateFirst: false,
        hostTabs: [{ panelId: 'chat', label: 'Chat', element: chatEl }],
    });
    const host = container.querySelector('.main-content');

    handle.activate('chat');
    const afterFirst = host.querySelector('#panel-chat #host-chat');
    handle.activate('metrics');
    // Switching away hides the panel but does NOT detach the element.
    assert.equal(host.querySelector('#panel-chat').classList.contains('active'), false,
        'chat panel hidden (display toggle via active class)');
    assert.equal(host.querySelector('#panel-chat #host-chat'), afterFirst,
        'element remains inside its panel while hidden (not detached)');
    handle.activate('chat');
    const afterSecond = host.querySelector('#panel-chat #host-chat');
    assert.equal(afterSecond, afterFirst, 'same node across switches (never re-created)');
    assert.equal(afterSecond.__liveMarker, marker, 'listeners/state survive tab switches');

    handle.destroy();
    container.remove();
});

test('destroy() returns each host element to its original parent/position', async () => {
    Panels._reset();
    const container = document.createElement('div');
    document.body.appendChild(container);

    const hostHome = document.createElement('div');
    document.body.appendChild(hostHome);
    const before = document.createElement('span');
    const chatEl = document.createElement('div');
    chatEl.id = 'host-chat';
    const after = document.createElement('span');
    hostHome.append(before, chatEl, after); // chatEl sits between two siblings

    const handle = await mountPanels(container, {
        api: stubApi(),
        loadFeatures: false,
        wireRuntime: false,
        activateFirst: false,
        hostTabs: [{ panelId: 'chat', label: 'Chat', element: chatEl }],
    });
    handle.activate('chat'); // adopt it into the mount
    assert.notEqual(chatEl.parentNode, hostHome, 'element was adopted into the mount');

    handle.destroy();
    assert.equal(chatEl.parentNode, hostHome, 'element returned to its original parent');
    assert.equal(chatEl.previousSibling, before, 'restored before its original nextSibling');
    assert.equal(chatEl.nextSibling, after, 'restored in its original position');

    container.remove();
    hostHome.remove();
});

test('a no-`before` host tab still leads when the first core panel is gated off', async () => {
    Panels._reset();
    const container = document.createElement('div');
    document.body.appendChild(container);
    const chatEl = document.createElement('div');
    chatEl.id = 'host-chat';

    // The embedder (a companion platform) opts OUT of `identity` —
    // CORE_PANEL_DEFS[0], the anchor a naive "first" default would rely on.
    // Chat must STILL land first, not fall to the end.
    const handle = await mountPanels(container, {
        api: stubApi({ identity: false }),
        loadFeatures: false,
        wireRuntime: false,
        activateFirst: false,
        hostTabs: [{ panelId: 'chat', label: 'Chat', element: chatEl }],
    });

    const nav = container.querySelector('.nav-tabs');
    const tabIds = [...nav.querySelectorAll('.nav-tab')].map((t) => t.dataset.panel);
    assert.ok(!tabIds.includes('identity'), 'identity gated off by the embedder');
    assert.equal(tabIds[0], 'chat', 'host tab still leads the strip (Chat-first preserved)');

    handle.destroy();
    container.remove();
});

test('multiple no-`before` host tabs lead the strip in registration order', async () => {
    Panels._reset();
    const container = document.createElement('div');
    document.body.appendChild(container);
    const chatEl = document.createElement('div');
    const notesEl = document.createElement('div');

    const handle = await mountPanels(container, {
        api: stubApi(),
        loadFeatures: false,
        wireRuntime: false,
        activateFirst: false,
        hostTabs: [
            { panelId: 'chat', label: 'Chat', element: chatEl },
            { panelId: 'notes', label: 'Notes', element: notesEl },
        ],
    });

    const nav = container.querySelector('.nav-tabs');
    const tabIds = [...nav.querySelectorAll('.nav-tab')].map((t) => t.dataset.panel);
    assert.deepEqual(tabIds.slice(0, 2), ['chat', 'notes'],
        'no-`before` host tabs lead in registration order');
    assert.deepEqual(tabIds, ['chat', 'notes', ...CORE_PANEL_DEFS.map((d) => d.panelId)]);

    handle.destroy();
    container.remove();
});

test('a host tab honors `before` ordering like registerPanel', async () => {
    Panels._reset();
    const container = document.createElement('div');
    document.body.appendChild(container);
    const chatEl = document.createElement('div');

    const handle = await mountPanels(container, {
        api: stubApi(),
        loadFeatures: false,
        wireRuntime: false,
        activateFirst: false,
        hostTabs: [{ panelId: 'chat', label: 'Chat', element: chatEl, before: 'metrics' }],
    });
    const nav = container.querySelector('.nav-tabs');
    const tabIds = [...nav.querySelectorAll('.nav-tab')].map((t) => t.dataset.panel);
    const metricsIdx = tabIds.indexOf('metrics');
    assert.equal(tabIds[metricsIdx - 1], 'chat', 'host tab inserted immediately before `metrics`');

    handle.destroy();
    container.remove();
});

test('standalone console mount is unaffected when no hostTabs are passed', async () => {
    Panels._reset();
    const container = document.createElement('div');
    document.body.appendChild(container);
    const handle = await mountPanels(container, {
        api: stubApi(),
        loadFeatures: false,
        wireRuntime: false,
        activateFirst: false,
    });
    const nav = container.querySelector('.nav-tabs');
    const tabIds = [...nav.querySelectorAll('.nav-tab')].map((t) => t.dataset.panel);
    assert.deepEqual(tabIds, CORE_PANEL_DEFS.map((d) => d.panelId),
        'exactly the core panel tabs, no host tabs injected');
    handle.destroy();
    container.remove();
});

test('registerCorePanels is idempotent (re-registration replaces in place)', () => {
    Panels._reset();
    registerCorePanels({ api: stubApi() });
    const first = Panels.panels().map((p) => p.panelId);
    registerCorePanels({ api: stubApi() });
    const second = Panels.panels().map((p) => p.panelId);
    assert.deepEqual(second, first, 'no duplicate registrations after a second call');
    assert.equal(second.filter((id) => id === 'identity').length, 1);
});
