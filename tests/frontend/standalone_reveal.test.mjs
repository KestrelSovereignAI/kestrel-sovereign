// #2229: the standalone console is chat-first — the static `.nav-tabs` strip
// starts hidden and a chat-header "Advanced" toggle reveals the capability-gated
// strip through the SAME reveal implementation the embeddable `mountPanels` host
// uses (`Panels.initReveal`). These tests drive that shared helper against the
// REAL index.html markup (hidden nav strip + `#advanced-toggle-btn`), exactly as
// identity.js wires it at boot, and cover:
//   - boot starts COLLAPSED with the toggle present + aria-pressed=false;
//   - reveal/collapse roundtrip incl. aria + return-to-Chat;
//   - persistence across a "reload" (destroy + re-init reads localStorage);
//   - single-tab console → no visible toggle;
//   - mountPanels reveal persistence (embeds inherit the same behavior).

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { JSDOM } from 'jsdom';

const here = dirname(fileURLToPath(import.meta.url));
const indexHtml = readFileSync(
    resolve(here, '../../kestrel_sovereign/static/index.html'),
    'utf8',
);

// A Map-backed localStorage so a "reload" (a second initReveal against a fresh
// DOM) can read what the first run persisted.
function makeStore(seed = {}) {
    const m = new Map(Object.entries(seed));
    return {
        getItem: (k) => (m.has(k) ? m.get(k) : null),
        setItem: (k, v) => { m.set(k, String(v)); },
        removeItem: (k) => { m.delete(k); },
        _map: m,
    };
}

// Boot the real console markup into a fresh JSDOM and expose the standalone
// reveal wiring inputs (the hidden nav strip + the chat-header Advanced button).
function bootConsole(store) {
    const dom = new JSDOM(indexHtml, { url: 'http://localhost/' });
    globalThis.window = dom.window;
    globalThis.document = dom.window.document;
    globalThis.Node = dom.window.Node;
    globalThis.MutationObserver = dom.window.MutationObserver;
    globalThis.location = dom.window.location;
    globalThis.localStorage = store;
    if (!globalThis.CSS || typeof globalThis.CSS.escape !== 'function') {
        globalThis.CSS = { escape: (s) => String(s).replace(/[^a-zA-Z0-9_-]/g, '\\$&') };
    }
    const navEl = dom.window.document.querySelector('.nav-tabs');
    const toggle = dom.window.document.getElementById('advanced-toggle-btn');
    return { dom, navEl, toggle };
}

// api.js (imported transitively by mount-panels.js) constructs its client from
// these globals AT IMPORT TIME, so seed them from a bootstrap DOM first.
const _bootDom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' });
globalThis.window = _bootDom.window;
globalThis.document = _bootDom.window.document;
globalThis.Node = _bootDom.window.Node;
globalThis.MutationObserver = _bootDom.window.MutationObserver;
globalThis.location = _bootDom.window.location;
if (!globalThis.CSS || typeof globalThis.CSS.escape !== 'function') {
    globalThis.CSS = { escape: (s) => String(s).replace(/[^a-zA-Z0-9_-]/g, '\\$&') };
}
globalThis.sessionStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.localStorage = makeStore();
globalThis.fetch = async () => ({ ok: false, status: 500, json: async () => ({}) });

const Panels = (await import('../../kestrel_sovereign/static/js/ui-ext/panels.js')).default;
const { mountPanels } = await import('../../kestrel_sovereign/static/js/ui-ext/mount-panels.js');

function stubApi(caps = {}) {
    return { hasCapability: (k) => caps[k] !== false };
}

// --- index.html static contract --------------------------------------------

test('index.html ships the nav strip hidden and an Advanced toggle button', () => {
    const { navEl, toggle } = bootConsole(makeStore());
    // The tab strip starts hidden so the console is chat-first before JS runs.
    assert.match(navEl.getAttribute('style') || '', /display:\s*none/,
        'nav-tabs strip starts hidden in markup');
    // The Advanced toggle exists in the chat header, starts hidden + collapsed,
    // and carries the i18n label key (no existing key renamed — #2180 lesson).
    assert.ok(toggle, '#advanced-toggle-btn present in the chat header');
    assert.match(toggle.getAttribute('style') || '', /display:\s*none/,
        'toggle starts hidden until the shared code decides');
    assert.equal(toggle.getAttribute('aria-pressed'), 'false', 'starts un-pressed');
    assert.ok(toggle.querySelector('[data-label-key="btn_advanced"]'),
        'toggle label carries the btn_advanced i18n key');
});

// --- standalone reveal wiring (Panels.initReveal, as identity.js calls it) ---

test('standalone boot starts collapsed: nav hidden, toggle shown + aria-pressed=false, chat active', () => {
    const { navEl, toggle } = bootConsole(makeStore());
    const activated = [];
    const handle = Panels.initReveal({
        navEl,
        activate: (id) => activated.push(id),
        anchor: toggle,
        storageKey: 'kestrel:console-advanced',
    });

    assert.ok(handle, 'reveal handle created');
    assert.equal(handle.revealed, false, 'starts collapsed');
    assert.equal(navEl.style.display, 'none', 'tab strip hidden while collapsed');
    // The pre-existing header button is adopted as the toggle and made visible
    // (the console has many tabs, so there IS something to reveal).
    assert.notEqual(toggle.style.display, 'none', 'toggle shown when >1 tab exists');
    assert.equal(toggle.getAttribute('aria-pressed'), 'false', 'aria-pressed=false collapsed');
    // Collapsed activates the leading (Chat) tab.
    assert.equal(activated[0], 'chat', 'chat is the leading tab activated on boot');

    handle.destroy();
});

test('reveal/collapse roundtrip: click shows the strip, aria tracks, collapse returns to Chat', () => {
    const { dom, navEl, toggle } = bootConsole(makeStore());
    const activated = [];
    const handle = Panels.initReveal({
        navEl,
        activate: (id) => activated.push(id),
        anchor: toggle,
        storageKey: 'kestrel:console-advanced',
    });

    toggle.dispatchEvent(new dom.window.Event('click'));
    assert.equal(handle.revealed, true, 'revealed after click');
    assert.notEqual(navEl.style.display, 'none', 'strip shown when revealed');
    assert.equal(toggle.getAttribute('aria-pressed'), 'true', 'aria-pressed=true revealed');
    // The Chat tab still leads the strip.
    const tabIds = [...navEl.querySelectorAll('.nav-tab')].map((t) => t.dataset.panel);
    assert.equal(tabIds[0], 'chat', 'Chat leads the revealed strip');

    activated.length = 0;
    toggle.dispatchEvent(new dom.window.Event('click'));
    assert.equal(handle.revealed, false, 'collapsed after second click');
    assert.equal(navEl.style.display, 'none', 'strip hidden again');
    assert.equal(toggle.getAttribute('aria-pressed'), 'false', 'aria-pressed=false collapsed');
    assert.equal(activated[activated.length - 1], 'chat', 'collapsing returns to the Chat tab');

    handle.destroy();
});

test('revealed state persists across a reload (localStorage)', () => {
    const store = makeStore();

    // First "load": operator reveals the strip.
    let boot = bootConsole(store);
    let handle = Panels.initReveal({
        navEl: boot.navEl,
        activate: () => {},
        anchor: boot.toggle,
        storageKey: 'kestrel:console-advanced',
    });
    handle.toggleReveal(true);
    assert.equal(store.getItem('kestrel:console-advanced'), '1', 'reveal persisted');
    handle.destroy();

    // Second "load" (fresh DOM, same store): the console comes up REVEALED.
    boot = bootConsole(store);
    handle = Panels.initReveal({
        navEl: boot.navEl,
        activate: () => {},
        anchor: boot.toggle,
        storageKey: 'kestrel:console-advanced',
    });
    assert.equal(handle.revealed, true, 'persisted reveal restored on reload');
    assert.notEqual(boot.navEl.style.display, 'none', 'strip shown after restore');
    assert.equal(boot.toggle.getAttribute('aria-pressed'), 'true', 'toggle reflects restored state');
    handle.destroy();
});

test('collapsed is the persisted default when nothing was stored', () => {
    const boot = bootConsole(makeStore());
    const handle = Panels.initReveal({
        navEl: boot.navEl,
        activate: () => {},
        anchor: boot.toggle,
        storageKey: 'kestrel:console-advanced',
    });
    assert.equal(handle.revealed, false, 'no stored preference → collapsed');
    handle.destroy();
});

test('single-tab console: the Advanced toggle is hidden (nothing to reveal)', () => {
    const boot = bootConsole(makeStore());
    // Reduce the strip to the single Chat tab (every capability gated off).
    boot.navEl.querySelectorAll('.nav-tab').forEach((t) => {
        if (t.dataset.panel !== 'chat') t.remove();
    });
    const handle = Panels.initReveal({
        navEl: boot.navEl,
        activate: () => {},
        anchor: boot.toggle,
        storageKey: 'kestrel:console-advanced',
    });
    assert.equal(boot.toggle.style.display, 'none', 'toggle hidden with a single tab');
    assert.equal(handle.revealed, false, 'nothing revealed');
    handle.destroy();
});

// --- mountPanels persistence (embeds inherit the same reveal) ---------------

test('mountPanels reveal persists to localStorage and restores on remount (#2229)', async () => {
    Panels._reset();
    const store = makeStore();
    const boot = bootConsole(store); // sets globalThis.localStorage = store
    const container = boot.dom.window.document.createElement('div');
    boot.dom.window.document.body.appendChild(container);

    const chatEl = boot.dom.window.document.createElement('div');
    chatEl.id = 'host-chat';

    let handle = await mountPanels(container, {
        api: stubApi(),
        loadFeatures: false,
        wireRuntime: false,
        hostTabs: [{ panelId: 'chat', label: 'Chat', element: chatEl }],
        reveal: { toggleLabel: 'Advanced', storageKey: 'kestrel:embed-advanced' },
    });
    handle.toggleReveal(true);
    assert.equal(store.getItem('kestrel:embed-advanced'), '1', 'embed reveal persisted');
    handle.destroy();

    // Remount with the SAME storage key: reveal state restored from localStorage
    // (no explicit toggleReveal call needed).
    Panels._reset();
    const chatEl2 = boot.dom.window.document.createElement('div');
    chatEl2.id = 'host-chat';
    handle = await mountPanels(container, {
        api: stubApi(),
        loadFeatures: false,
        wireRuntime: false,
        hostTabs: [{ panelId: 'chat', label: 'Chat', element: chatEl2 }],
        reveal: { toggleLabel: 'Advanced', storageKey: 'kestrel:embed-advanced' },
    });
    assert.equal(handle.revealed, true, 'persisted embed reveal restored on remount');
    const nav = container.querySelector('.nav-tabs');
    assert.notEqual(nav.style.display, 'none', 'strip shown after restore');
    handle.destroy();
    container.remove();
});

test('mountPanels reveal persistence can be disabled with storageKey:false', async () => {
    Panels._reset();
    const store = makeStore();
    const boot = bootConsole(store);
    const container = boot.dom.window.document.createElement('div');
    boot.dom.window.document.body.appendChild(container);
    const chatEl = boot.dom.window.document.createElement('div');
    chatEl.id = 'host-chat';

    const handle = await mountPanels(container, {
        api: stubApi(),
        loadFeatures: false,
        wireRuntime: false,
        hostTabs: [{ panelId: 'chat', label: 'Chat', element: chatEl }],
        reveal: { storageKey: false },
    });
    handle.toggleReveal(true);
    assert.equal(store._map.size, 0, 'nothing written when persistence is disabled');
    handle.destroy();
    container.remove();
});
