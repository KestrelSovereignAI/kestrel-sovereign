// Voice UI migrated onto the slot registry (#2038, ticket 04). It no longer
// exposes `initVoiceUI()` / `mountAgentVoiceControls()` — it self-registers slot
// contributions on import, each gated on the `voice` capability via
// `ctx.api.hasCapability('voice')`. These tests pin that the contributions are
// registered into the expected zones and that the gate blocks mounting when
// voice is off (the behavior the old manual `if (!API.hasCapability('voice'))`
// guard used to enforce).
import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.Node = dom.window.Node;
globalThis.location = dom.window.location;
globalThis.sessionStorage = dom.window.sessionStorage;
globalThis.localStorage = dom.window.localStorage;
// Node 25 exposes `globalThis.navigator` as a read-only getter, so a bare
// assignment throws. Redefine the property to point at jsdom's navigator.
Object.defineProperty(globalThis, 'navigator', {
    value: dom.window.navigator,
    configurable: true,
    writable: true,
});
globalThis.CSS = dom.window.CSS || { escape: (s) => String(s) };
globalThis.fetch = async () => ({ ok: false, status: 404, json: async () => ({}) });
globalThis.kicon = () => '';
globalThis.KESTREL_UI_CONFIG = { capabilities: { voice: false } };

const { UI } = await import('../../kestrel_sovereign/static/js/ui-ext/registry.js');
// Importing the module runs its top-level UI.register(...) calls.
await import('../../kestrel_sovereign/static/js/voice/ui.js');

function anchor() {
    const el = document.createElement('div');
    document.body.appendChild(el);
    return el;
}

function capApi(voiceOn) {
    return { hasCapability: (k) => (k === 'voice' ? voiceOn : false) };
}

test('voice self-registers contributions into its four zones', () => {
    const has = (slot, id) => UI.contributions(slot).some((c) => c.id === id);
    assert.ok(has('chat-input-actions', 'voice-mic'), 'mic button registered');
    assert.ok(has('input-footer-status', 'voice-status'), 'footer status registered');
    assert.ok(has('agent-card-actions', 'voice-controls'), 'agent-card controls registered');
    assert.ok(has('modal-root', 'voice-picker'), 'picker modal registered');
});

test('voice capability false prevents the mic button from mounting', () => {
    const el = anchor();
    UI.renderSlot('chat-input-actions', { element: el, api: capApi(false) });
    assert.equal(el.children.length, 0, 'gate blocked → no container mounted');
});

test('voice capability true mounts the mic button into the zone', () => {
    const el = anchor();
    UI.renderSlot('chat-input-actions', { element: el, api: capApi(true) });
    assert.equal(el.children.length, 1, 'one per-contribution container mounted');
    assert.ok(el.querySelector('#voice-toggle-btn'), 'mic button rendered into the zone');
});

test('runtime capabilities:changed re-gates the mic without a reload', () => {
    // A mutable api whose `voice` answer flips at runtime — mirrors the live
    // `API` object after `applyServerCapabilities()` rewrites its caps map.
    const live = { voice: false };
    const api = { hasCapability: (k) => (k === 'voice' ? live.voice : false) };

    const el = anchor();
    // Boot with voice OFF: nothing mounts, but the instance ctx is retained.
    UI.renderSlot('chat-input-actions', { element: el, api });
    assert.equal(el.children.length, 0, 'voice off at boot → no mic');

    // Feature enabled at runtime: api flips, the bridged bus event fires.
    // (Assert via the scoped class, not `#voice-toggle-btn` — by this point in
    // the file several instances carry a button with that duplicate id, and
    // jsdom's id-selector fast-path resolves it globally to the wrong subtree.)
    live.voice = true;
    UI.emit('capabilities:changed', { capabilities: { voice: true } });
    assert.ok(el.querySelector('.kestrel-voice-btn'), 'mic mounts on enable, no reload');

    // Feature disabled at runtime: the contribution gates out and tears down.
    live.voice = false;
    UI.emit('capabilities:changed', { capabilities: { voice: false } });
    assert.equal(el.children.length, 0, 'mic torn down on disable');
});

test('agent-card controls gate on voice capability', () => {
    const off = anchor();
    UI.renderSlot('agent-card-actions', {
        element: off, api: capApi(false), agentName: 'Alpha', standalone: false,
    });
    assert.equal(off.children.length, 0, 'gated out → no controls');

    const on = anchor();
    UI.renderSlot('agent-card-actions', {
        element: on, api: capApi(true), agentName: 'Alpha', standalone: false,
    });
    assert.ok(on.querySelector('.agent-voice-solo'), 'listen control mounted');
    assert.ok(on.querySelector('.agent-voice-arm'), 'talk control mounted');
});
