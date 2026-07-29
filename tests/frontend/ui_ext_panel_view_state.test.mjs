// Panel view-state provider hook (#2802).
//
// A panel contribution may declare `viewState: {key?, getState, setState}` and
// the registry persists its view (sub-tab, zoom/pan, scroll, selection) through
// `ui_state.mjs` — never a bespoke raw-localStorage path — restoring it around a
// body remount and across a full page reload.
//
// These tests assert the contract the issue's acceptance criteria name:
//   - register provider -> after a simulated remount, `setState` receives the
//     value the prior `getState` returned;
//   - missing/corrupt stored state degrades to the provider's OWN default and
//     never throws;
//   - two panels' keys cannot collide;
// plus the surrounding guarantees: the framework-composed key shape, the
// `kestrel:ui:` namespace, the unload flush, and that a panel registering no
// provider behaves exactly as it did before the hook existed.

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

// In-memory localStorage so persistence is directly observable (and so we can
// plant corrupt values). `ui_state.mjs` resolves the global, not jsdom's own.
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

const Panels = (await import('../../kestrel_sovereign/static/js/ui-ext/panels.js')).default;
const { UI_STATE_PREFIX } = await import('../../kestrel_sovereign/static/js/ui_state.mjs');
const { viewStateKey, DEFAULT_VIEW_STATE_KEY } = await import(
    '../../kestrel_sovereign/static/js/ui-ext/view-state.js'
);

function freshNav() {
    globalThis.localStorage._map.clear();
    Panels._reset();
    document.body.innerHTML = '<div id="nav"></div><div id="host"></div>';
    return {
        navEl: document.getElementById('nav'),
        hostEl: document.getElementById('host'),
    };
}

/**
 * A panel whose "view" is a single mutable value, exposing the calls the
 * registry made so a test can assert on them. `defaultState` is what the panel
 * falls back to when nothing valid is stored.
 */
function makePanel(panelId, { key, defaultState = { tab: 'overview', zoom: 1 } } = {}) {
    const rec = {
        panelId,
        live: { ...defaultState },
        defaultState: { ...defaultState },
        renders: 0,
        setStateCalls: [],
        getStateCalls: 0,
    };
    rec.def = {
        panelId,
        label: panelId,
        render: () => { rec.renders += 1; },
        viewState: {
            ...(key ? { key } : {}),
            getState: () => { rec.getStateCalls += 1; return rec.live; },
            setState: (s) => { rec.setStateCalls.push(s); rec.live = s; },
        },
    };
    return rec;
}

// ---------------------------------------------------------------------------
// Acceptance 1 — remount round-trip
// ---------------------------------------------------------------------------

test('after a remount, setState receives the value the prior getState returned', () => {
    const { navEl, hostEl } = freshNav();
    const obs = makePanel('observability');
    const other = makePanel('identity');

    Panels.registerPanel(obs.def);
    Panels.registerPanel(other.def);
    Panels.renderNav({ navEl, hostEl, ctx: {} });

    Panels.activate('observability');
    assert.equal(obs.renders, 1, 'panel body rendered on first activation');
    assert.deepEqual(obs.setStateCalls, [], 'nothing stored yet, so the panel keeps its own default');

    // The operator drives the view: sub-tab + zoom/pan change.
    obs.live = { tab: 'timeline', zoom: 4.5, pan: 120 };

    // Switching tabs is the deactivation boundary — the registry snapshots here.
    Panels.activate('identity');
    assert.equal(obs.getStateCalls, 1, 'deactivation snapshotted the outgoing panel');

    // Simulate the remount: a FRESH panel instance (as a destroy/remount or a
    // re-gate produces) re-registers under the same id, which clears the
    // rendered flag so the next activation re-renders the body from scratch.
    const remounted = makePanel('observability');
    Panels.registerPanel(remounted.def);
    Panels.activate('observability');

    assert.equal(remounted.renders, 1, 'the fresh instance rendered its body');
    assert.equal(remounted.setStateCalls.length, 1, 'setState called exactly once on remount');
    assert.deepEqual(
        remounted.setStateCalls[0],
        { tab: 'timeline', zoom: 4.5, pan: 120 },
        'restored value is exactly what the prior getState returned',
    );
    assert.deepEqual(remounted.live, { tab: 'timeline', zoom: 4.5, pan: 120 });
});

test('state survives a full page reload (fresh registry against the same storage)', () => {
    const { navEl, hostEl } = freshNav();
    const obs = makePanel('observability');
    Panels.registerPanel(obs.def);
    Panels.renderNav({ navEl, hostEl, ctx: {} });
    Panels.activate('observability');
    obs.live = { tab: 'timeline', zoom: 2, follow: true };

    // A reload never calls activate() — `pagehide` is the flush point.
    window.dispatchEvent(new dom.window.Event('pagehide'));
    assert.equal(obs.getStateCalls, 1, 'pagehide snapshotted the active panel');

    // Reload: registry state is gone, localStorage is not.
    Panels._reset();
    document.body.innerHTML = '<div id="nav"></div><div id="host"></div>';
    const reloaded = makePanel('observability');
    Panels.registerPanel(reloaded.def);
    Panels.renderNav({
        navEl: document.getElementById('nav'),
        hostEl: document.getElementById('host'),
        ctx: {},
    });
    Panels.activate('observability');

    assert.deepEqual(
        reloaded.live,
        { tab: 'timeline', zoom: 2, follow: true },
        'the reloaded panel came up on its persisted view',
    );
});

test('a re-gate (feature disabled then re-enabled) round-trips the view state', () => {
    const { navEl, hostEl } = freshNav();
    let gateOpen = true;
    const obs = makePanel('observability');
    obs.def.gate = () => gateOpen;

    Panels.registerPanel(obs.def);
    Panels.renderNav({ navEl, hostEl, ctx: {} });
    Panels.activate('observability');
    obs.live = { tab: 'traces', zoom: 8 };

    // Gate off: the registry drops the body, so it must snapshot first.
    gateOpen = false;
    Panels.syncNav();
    assert.equal(document.getElementById('panel-observability'), null, 'body dropped by the re-gate');
    assert.equal(obs.getStateCalls, 1, 're-gate snapshotted before the body was dropped');

    // Gate back on: fresh body, restored view.
    obs.live = { ...obs.defaultState };
    gateOpen = true;
    Panels.syncNav();
    Panels.activate('observability');
    assert.deepEqual(obs.live, { tab: 'traces', zoom: 8 });
});

// ---------------------------------------------------------------------------
// Acceptance 2 — missing / corrupt storage degrades to the provider's default
// ---------------------------------------------------------------------------

test('missing stored state leaves the panel on its own default and never throws', () => {
    const { navEl, hostEl } = freshNav();
    const obs = makePanel('observability');
    Panels.registerPanel(obs.def);
    Panels.renderNav({ navEl, hostEl, ctx: {} });

    assert.doesNotThrow(() => Panels.activate('observability'));
    assert.deepEqual(obs.setStateCalls, [], 'setState never called when nothing is stored');
    assert.deepEqual(obs.live, obs.defaultState, 'panel kept its own default');
});

test('corrupt stored state degrades to the provider default and never throws', () => {
    const { navEl, hostEl } = freshNav();
    const obs = makePanel('observability');
    // Plant a non-JSON value under the exact key the framework composes.
    globalThis.localStorage.setItem(viewStateKey('observability'), '{not json');

    Panels.registerPanel(obs.def);
    Panels.renderNav({ navEl, hostEl, ctx: {} });

    assert.doesNotThrow(() => Panels.activate('observability'));
    assert.deepEqual(obs.setStateCalls, [], 'a corrupt value is never handed to setState');
    assert.deepEqual(obs.live, obs.defaultState, 'panel kept its own default');
});

test('an unavailable localStorage degrades to the provider default and never throws', () => {
    const { navEl, hostEl } = freshNav();
    const obs = makePanel('observability');
    const original = globalThis.localStorage;
    globalThis.localStorage = {
        getItem() { throw new Error('SecurityError'); },
        setItem() { throw new Error('SecurityError'); },
        removeItem() { throw new Error('SecurityError'); },
    };
    try {
        Panels.registerPanel(obs.def);
        Panels.renderNav({ navEl, hostEl, ctx: {} });
        assert.doesNotThrow(() => Panels.activate('observability'));
        obs.live = { tab: 'timeline', zoom: 3 };
        assert.doesNotThrow(() => Panels.flushViewState('observability'));
        assert.deepEqual(obs.setStateCalls, [], 'no state to restore from a dead store');
    } finally {
        globalThis.localStorage = original;
    }
});

test('a throwing getState/setState is isolated — activation still completes', () => {
    const { navEl, hostEl } = freshNav();
    let rendered = 0;
    Panels.registerPanel({
        panelId: 'bad',
        label: 'Bad',
        render: () => { rendered += 1; },
        viewState: {
            getState: () => { throw new Error('boom'); },
            setState: () => { throw new Error('bang'); },
        },
    });
    Panels.registerPanel({ panelId: 'other', label: 'Other' });
    Panels.renderNav({ navEl, hostEl, ctx: {} });

    assert.doesNotThrow(() => Panels.activate('bad'));
    // A throwing getState writes nothing rather than persisting garbage.
    assert.doesNotThrow(() => Panels.activate('other'));
    assert.equal(globalThis.localStorage.getItem(viewStateKey('bad')), null, 'nothing persisted');

    // Now plant a valid value so the throwing setState is exercised on remount.
    globalThis.localStorage.setItem(viewStateKey('bad'), JSON.stringify({ ok: true }));
    Panels.registerPanel({
        panelId: 'bad',
        label: 'Bad',
        render: () => { rendered += 1; },
        viewState: {
            getState: () => ({ ok: true }),
            setState: () => { throw new Error('bang'); },
        },
    });
    assert.doesNotThrow(() => Panels.activate('bad'));
    assert.equal(rendered, 2, 'the body still rendered despite the throwing provider');
});

test('a malformed provider is ignored, not fatal', () => {
    const { navEl, hostEl } = freshNav();
    Panels.registerPanel({
        panelId: 'sloppy',
        label: 'Sloppy',
        viewState: { key: 'x', getState: () => 1 }, // no setState
    });
    Panels.registerPanel({ panelId: 'other', label: 'Other' });
    Panels.renderNav({ navEl, hostEl, ctx: {} });

    assert.doesNotThrow(() => Panels.activate('sloppy'));
    assert.doesNotThrow(() => Panels.activate('other'));
    assert.equal(globalThis.localStorage._map.size, 0, 'a half-declared provider persists nothing');
});

// ---------------------------------------------------------------------------
// Acceptance 3 — two panels' keys do not collide
// ---------------------------------------------------------------------------

test('two panels declaring the SAME provider key do not collide', () => {
    const { navEl, hostEl } = freshNav();
    // Both panels pick the identical provider key — the framework's panelId
    // segment is what keeps them apart.
    const a = makePanel('observability', { key: 'viewstate', defaultState: { v: 'a-default' } });
    const b = makePanel('database', { key: 'viewstate', defaultState: { v: 'b-default' } });

    Panels.registerPanel(a.def);
    Panels.registerPanel(b.def);
    Panels.renderNav({ navEl, hostEl, ctx: {} });

    Panels.activate('observability');
    a.live = { v: 'A' };
    Panels.activate('database');
    b.live = { v: 'B' };
    Panels.activate('observability'); // snapshots database

    const keyA = viewStateKey('observability', 'viewstate');
    const keyB = viewStateKey('database', 'viewstate');
    assert.notEqual(keyA, keyB, 'composed keys are distinct');
    assert.deepEqual(JSON.parse(globalThis.localStorage.getItem(keyA)), { v: 'A' });
    assert.deepEqual(JSON.parse(globalThis.localStorage.getItem(keyB)), { v: 'B' });

    // And each remounts onto its OWN value, not the other's.
    const a2 = makePanel('observability', { key: 'viewstate', defaultState: { v: 'a-default' } });
    const b2 = makePanel('database', { key: 'viewstate', defaultState: { v: 'b-default' } });
    Panels.registerPanel(a2.def);
    Panels.registerPanel(b2.def);
    Panels.activate('observability');
    Panels.activate('database');
    assert.deepEqual(a2.live, { v: 'A' });
    assert.deepEqual(b2.live, { v: 'B' });
});

test('one panel can hold several independent slices via distinct provider keys', () => {
    assert.notEqual(viewStateKey('obs', 'timeline'), viewStateKey('obs', 'traces'));
});

// ---------------------------------------------------------------------------
// Key composition + namespace
// ---------------------------------------------------------------------------

test('the framework composes kestrel:ui:panel:<panelId>:<key>', () => {
    assert.equal(viewStateKey('observability', 'timeline'), 'kestrel:ui:panel:observability:timeline');
    assert.equal(
        viewStateKey('observability'),
        `${UI_STATE_PREFIX}panel:observability:${DEFAULT_VIEW_STATE_KEY}`,
        'a provider with no key lands in the panel default sub-namespace',
    );
    assert.ok(viewStateKey('x', 'y').startsWith(UI_STATE_PREFIX), 'always under the ui_state namespace');
});

test('persistence goes through the kestrel:ui: namespace, not a bespoke key', () => {
    const { navEl, hostEl } = freshNav();
    const obs = makePanel('observability', { key: 'timeline' });
    Panels.registerPanel(obs.def);
    Panels.registerPanel({ panelId: 'other', label: 'Other' });
    Panels.renderNav({ navEl, hostEl, ctx: {} });

    Panels.activate('observability');
    obs.live = { tab: 'timeline' };
    Panels.activate('other');

    const keys = [...globalThis.localStorage._map.keys()];
    assert.deepEqual(keys, ['kestrel:ui:panel:observability:timeline']);
    // Stored JSON-serialized by ui_state.mjs (not a hand-rolled format).
    assert.deepEqual(JSON.parse(globalThis.localStorage._map.get(keys[0])), { tab: 'timeline' });
});

// ---------------------------------------------------------------------------
// Purely additive
// ---------------------------------------------------------------------------

test('a panel that registers no provider behaves exactly as before', () => {
    const { navEl, hostEl } = freshNav();
    let renders = 0;
    const shown = [];
    Panels.registerPanel({ panelId: 'plain', label: 'Plain', render: () => { renders += 1; } });
    Panels.registerPanel({ panelId: 'other', label: 'Other' });
    Panels.renderNav({ navEl, hostEl, ctx: {} });

    Panels.activate('plain');
    Panels.activate('other');
    Panels.activate('plain');
    shown.push(renders);

    assert.equal(renders, 1, 'still rendered exactly once (lazy-render semantics unchanged)');
    assert.equal(globalThis.localStorage._map.size, 0, 'no storage written for a provider-less panel');
    assert.doesNotThrow(() => window.dispatchEvent(new dom.window.Event('pagehide')));
});

test('an ordinary tab switch does NOT emit panel:hidden (existing subscribers unchanged)', async () => {
    const { navEl, hostEl } = freshNav();
    const bus = (await import('../../kestrel_sovereign/static/js/ui-ext/bus.js')).default;
    const obs = makePanel('observability');
    Panels.registerPanel(obs.def);
    Panels.registerPanel({ panelId: 'other', label: 'Other' });
    Panels.renderNav({ navEl, hostEl, ctx: {} });

    const hidden = [];
    const off = bus.on('panel:hidden', (p) => hidden.push(p && p.panelId));
    Panels.activate('observability');
    Panels.activate('other');
    off();

    assert.equal(obs.getStateCalls, 1, 'the snapshot still happened on the switch');
    assert.deepEqual(hidden, [], 'panel:hidden stayed a re-gate-only teardown signal');
});

// ---------------------------------------------------------------------------
// Snapshot guards
// ---------------------------------------------------------------------------

test('a never-rendered panel cannot clobber a good stored value', () => {
    const { navEl, hostEl } = freshNav();
    const good = JSON.stringify({ tab: 'timeline', zoom: 9 });
    globalThis.localStorage.setItem(viewStateKey('observability'), good);

    const obs = makePanel('observability');
    Panels.registerPanel(obs.def);
    Panels.registerPanel({ panelId: 'other', label: 'Other' });
    Panels.renderNav({ navEl, hostEl, ctx: {} });

    // The operator never opens the panel; a re-gate/unload still must not write
    // the panel's uninitialized default over the stored view.
    Panels.activate('other');
    window.dispatchEvent(new dom.window.Event('pagehide'));
    assert.equal(obs.getStateCalls, 0, 'an unrendered panel is never snapshotted');
    assert.equal(globalThis.localStorage.getItem(viewStateKey('observability')), good);
});

test('a provider returning undefined writes nothing (nothing to persist yet)', () => {
    const { navEl, hostEl } = freshNav();
    const good = JSON.stringify({ tab: 'timeline' });
    globalThis.localStorage.setItem(viewStateKey('observability'), good);

    let ready = false;
    Panels.registerPanel({
        panelId: 'observability',
        label: 'Obs',
        viewState: {
            getState: () => (ready ? { tab: 'traces' } : undefined),
            setState: () => {},
        },
    });
    Panels.registerPanel({ panelId: 'other', label: 'Other' });
    Panels.renderNav({ navEl, hostEl, ctx: {} });

    Panels.activate('observability');
    Panels.activate('other');
    assert.equal(globalThis.localStorage.getItem(viewStateKey('observability')), good, 'stored view untouched');

    ready = true;
    Panels.activate('observability');
    Panels.activate('other');
    assert.deepEqual(
        JSON.parse(globalThis.localStorage.getItem(viewStateKey('observability'))),
        { tab: 'traces' },
        'once the provider has state, it persists',
    );
});

test('flushViewState is the explicit escape hatch for a host-driven teardown', () => {
    const { navEl, hostEl } = freshNav();
    const obs = makePanel('observability');
    Panels.registerPanel(obs.def);
    Panels.renderNav({ navEl, hostEl, ctx: {} });
    Panels.activate('observability');
    obs.live = { tab: 'timeline', zoom: 6 };

    assert.equal(Panels.flushViewState(), true, 'defaults to the active panel');
    assert.deepEqual(
        JSON.parse(globalThis.localStorage.getItem(viewStateKey('observability'))),
        { tab: 'timeline', zoom: 6 },
    );
    assert.equal(Panels.flushViewState('nope'), false, 'unknown panel is a no-op, not a throw');
});

test('unregisterPanel snapshots before dropping the contribution', () => {
    const { navEl, hostEl } = freshNav();
    const obs = makePanel('observability');
    Panels.registerPanel(obs.def);
    Panels.renderNav({ navEl, hostEl, ctx: {} });
    Panels.activate('observability');
    obs.live = { tab: 'traces' };

    Panels.unregisterPanel('observability');
    assert.deepEqual(
        JSON.parse(globalThis.localStorage.getItem(viewStateKey('observability'))),
        { tab: 'traces' },
    );
});
