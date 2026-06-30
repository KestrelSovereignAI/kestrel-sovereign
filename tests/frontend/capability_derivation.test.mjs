// #2041: capability set derived from enabled features. Covers the asymmetric
// merge precedence across the three capability classes, the runtime re-derive
// + capabilities:changed emission, and the boot-injection data source.

import test from 'node:test';
import assert from 'node:assert/strict';

import {
    createApiClient,
    mergeCapabilities,
} from '../../kestrel_sovereign/static/js/api_client.mjs';

function createStorage(initial = {}) {
    const store = new Map(Object.entries(initial));
    return {
        getItem(key) { return store.has(key) ? store.get(key) : null; },
        setItem(key, value) { store.set(key, String(value)); },
        removeItem(key) { store.delete(key); },
    };
}

function createLogger() {
    const warnings = [];
    return { log() {}, warn(...a) { warnings.push(a.join(' ')); }, error() {}, warnings };
}

function makeClient({ capabilities = null, featureCapabilities = null, logger = null } = {}) {
    return createApiClient({
        fetchFn: async () => ({ ok: false, status: 404, async json() { return {}; } }),
        sessionStorage: createStorage({ kestrel_api_key: 'k' }),
        location: { href: '/console', search: '' },
        logger: logger || createLogger(),
        capabilities,
        featureCapabilities,
    });
}

// --- mergeCapabilities: feature-backed class ---

test('feature-backed: enabled feature → true, disabled feature → false', () => {
    const merged = mergeCapabilities({ voice: true, spawn: false }, {}, createLogger());
    assert.equal(merged.voice, true);
    assert.equal(merged.spawn, false);
});

test('feature-backed: host override can force an enabled feature OFF', () => {
    const merged = mergeCapabilities({ voice: true }, { voice: false }, createLogger());
    assert.equal(merged.voice, false);
});

test('feature-backed: force-TRUE on a disabled feature is ignored + warns', () => {
    const logger = createLogger();
    const merged = mergeCapabilities({ voice: false }, { voice: true }, logger);
    assert.equal(merged.voice, false, 'disabled feature stays false');
    assert.ok(
        logger.warnings.some((w) => w.includes('voice')),
        'expected a console warning about the ignored force-true override',
    );
});

test('feature-backed: effective value = serverEnabled && hostOverride !== false', () => {
    assert.equal(mergeCapabilities({ k: true }, {}, createLogger()).k, true);
    assert.equal(mergeCapabilities({ k: true }, { k: false }, createLogger()).k, false);
    assert.equal(mergeCapabilities({ k: false }, {}, createLogger()).k, false);
    assert.equal(mergeCapabilities({ k: false }, { k: true }, createLogger()).k, false);
});

// --- mergeCapabilities: core/static class ---

test('core/static: host override (on or off) flows through; absent stays default-on', () => {
    const merged = mergeCapabilities({}, { chat: false, conversations: true }, createLogger());
    assert.equal(merged.chat, false);
    assert.equal(merged.conversations, true);
    // A key in neither map is absent → resolveCapability returns true (default-on).
    assert.equal('metrics' in merged, false);
});

test('core/static: nested object overrides pass through untouched', () => {
    const caps = { agent: false, user: true };
    const merged = mergeCapabilities({}, { keys: caps }, createLogger());
    assert.deepEqual(merged.keys, caps);
});

// --- client wiring: hasCapability reads the merged set ---

test('client derives feature capability from server map (no host edit needed)', () => {
    // A brand-new feature "weather" gains a working gate with zero api_client edits.
    const client = makeClient({ featureCapabilities: { weather: true, voice: false } });
    assert.equal(client.hasCapability('weather'), true);
    assert.equal(client.hasCapability('voice'), false);
});

test('client: server-disabled feature cannot be forced true by host', () => {
    const client = makeClient({
        featureCapabilities: { voice: false },
        capabilities: { voice: true },
    });
    assert.equal(client.hasCapability('voice'), false);
});

test('client: host can force an enabled feature off', () => {
    const client = makeClient({
        featureCapabilities: { voice: true },
        capabilities: { voice: false },
    });
    assert.equal(client.hasCapability('voice'), false);
});

// --- runtime re-derive + capabilities:changed emission ---

test('applyServerCapabilities re-derives the set and emits capabilities:changed', () => {
    // Browsers expose dispatchEvent/addEventListener on window (=globalThis).
    // Node does not, so install a minimal EventTarget-backed bus for the test.
    const bus = new EventTarget();
    const prev = {
        dispatchEvent: globalThis.dispatchEvent,
        addEventListener: globalThis.addEventListener,
        removeEventListener: globalThis.removeEventListener,
    };
    globalThis.dispatchEvent = (evt) => bus.dispatchEvent(evt);
    globalThis.addEventListener = (...a) => bus.addEventListener(...a);
    globalThis.removeEventListener = (...a) => bus.removeEventListener(...a);

    const client = makeClient({ featureCapabilities: { voice: false } });
    assert.equal(client.hasCapability('voice'), false);

    let fired = null;
    const handler = (e) => { fired = e.detail; };
    globalThis.addEventListener('capabilities:changed', handler);
    try {
        client.applyServerCapabilities({ voice: true });
    } finally {
        globalThis.removeEventListener('capabilities:changed', handler);
        Object.assign(globalThis, prev);
    }

    assert.equal(client.hasCapability('voice'), true, 'capability flipped on without reload');
    assert.ok(fired, 'capabilities:changed event fired');
    assert.equal(fired.capabilities.voice, true);
});

test('applyServerCapabilities re-applies host overrides with the same precedence', () => {
    const client = makeClient({
        featureCapabilities: { voice: true },
        capabilities: { voice: false }, // host force-off persists across re-derive
    });
    assert.equal(client.hasCapability('voice'), false);
    client.applyServerCapabilities({ voice: true });
    assert.equal(client.hasCapability('voice'), false, 'host force-off survives re-derive');
});

// --- regression: no server map means no feature downgrade (legacy behavior) ---

test('with no server feature map, feature keys stay default-on (unchanged)', () => {
    const client = makeClient();
    assert.equal(client.hasCapability('voice'), true);
    assert.equal(client.hasCapability('spawn'), true);
});
