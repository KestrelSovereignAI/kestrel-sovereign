// #2298: `ui_state.mjs` is the ONE shared UI view-state persistence surface.
// It replaces four ad-hoc localStorage implementations (theme.js,
// conversations.js, agent_list.js, ui-ext/panels.js) plus identity.js one-off
// stashes. These tests exercise the module contract directly:
//   - raw string helpers (storeGet/storeSet/storeRemove) round-trip verbatim;
//   - the JSON API (uiStateGet/uiStateSet/uiStateRemove) round-trips values,
//     applies the fallback on absent/invalid JSON, and never throws;
//   - every operation degrades to a safe no-op when localStorage is disabled
//     or throws (embeds, sandboxed iframes, jsdom).

import test from 'node:test';
import assert from 'node:assert/strict';

// In-memory localStorage so persistence is observable.
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

const mod = await import('../../kestrel_sovereign/static/js/ui_state.mjs');
const {
    UI_STATE_PREFIX,
    storeGet, storeSet, storeRemove,
    uiStateGet, uiStateSet, uiStateRemove,
} = mod;

test('UI_STATE_PREFIX namespaces new keys', () => {
    assert.equal(UI_STATE_PREFIX, 'kestrel:ui:');
});

test('raw storeGet/storeSet/storeRemove round-trip verbatim (no JSON transform)', () => {
    storeSet('raw:k', '1');
    assert.equal(storeGet('raw:k'), '1', 'stored value read back byte-for-byte');
    assert.equal(localStorage.getItem('raw:k'), '1', 'on-disk format is the raw string');
    storeRemove('raw:k');
    assert.equal(storeGet('raw:k'), null, 'removed key reads back null');
});

test('raw storeGet returns null for an absent key', () => {
    assert.equal(storeGet('raw:absent'), null);
});

test('uiStateSet/uiStateGet round-trip JSON values (string, number, bool, object)', () => {
    uiStateSet('js:str', 'Emma');
    assert.equal(uiStateGet('js:str'), 'Emma');
    // JSON-serialized on disk (a quoted string), not the bare value.
    assert.equal(localStorage.getItem('js:str'), '"Emma"');

    uiStateSet('js:num', 42);
    assert.equal(uiStateGet('js:num'), 42);

    uiStateSet('js:bool', true);
    assert.equal(uiStateGet('js:bool'), true);

    uiStateSet('js:obj', { a: 1, b: ['x'] });
    assert.deepEqual(uiStateGet('js:obj'), { a: 1, b: ['x'] });
});

test('uiStateGet returns the fallback for an absent key', () => {
    assert.equal(uiStateGet('js:missing', 'fb'), 'fb');
    assert.equal(uiStateGet('js:missing'), null, 'default fallback is null');
});

test('uiStateGet returns the fallback (never throws) for non-JSON stored values', () => {
    // A stale raw value from before this module (e.g. a legacy '1').
    localStorage.setItem('js:legacy', 'not json {');
    assert.equal(uiStateGet('js:legacy', 'fb'), 'fb');
});

test('uiStateRemove deletes the key', () => {
    uiStateSet('js:del', 'v');
    assert.equal(uiStateGet('js:del'), 'v');
    uiStateRemove('js:del');
    assert.equal(uiStateGet('js:del', 'gone'), 'gone');
});

test('all operations degrade to safe no-ops when localStorage throws', () => {
    const throwing = {
        getItem() { throw new Error('SecurityError'); },
        setItem() { throw new Error('SecurityError'); },
        removeItem() { throw new Error('SecurityError'); },
    };
    const original = globalThis.localStorage;
    globalThis.localStorage = throwing;
    try {
        // None of these throw; reads return the fallback, writes drop silently.
        assert.doesNotThrow(() => storeSet('x', '1'));
        assert.equal(storeGet('x'), null);
        assert.doesNotThrow(() => storeRemove('x'));
        assert.doesNotThrow(() => uiStateSet('x', { a: 1 }));
        assert.equal(uiStateGet('x', 'fb'), 'fb');
        assert.doesNotThrow(() => uiStateRemove('x'));
        assert.equal(uiStateSet('x', 1), false, 'uiStateSet reports failure on a throwing store');
    } finally {
        globalThis.localStorage = original;
    }
});

test('uiStateSet returns false when storage is entirely unavailable', () => {
    const original = globalThis.localStorage;
    // Deleting the global makes typeof-checks resolve to unavailable.
    delete globalThis.localStorage;
    try {
        assert.equal(uiStateSet('x', 1), false);
        assert.equal(uiStateGet('x', 'fb'), 'fb');
        assert.doesNotThrow(() => storeSet('x', '1'));
        assert.equal(storeGet('x'), null);
    } finally {
        globalThis.localStorage = original;
    }
});

test('exposes the same one implementation on globalThis.KestrelUIState (for plain scripts)', () => {
    assert.ok(globalThis.KestrelUIState, 'global shim present');
    assert.equal(globalThis.KestrelUIState.uiStateGet, uiStateGet);
    assert.equal(globalThis.KestrelUIState.storeSet, storeSet);
    assert.equal(globalThis.KestrelUIState.UI_STATE_PREFIX, UI_STATE_PREFIX);
});
