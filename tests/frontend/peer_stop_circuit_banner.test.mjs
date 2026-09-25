// #3170: the agents banner shows open peer Stop circuits and wires the
// sovereign Reset. The circuits ride the host Stop status read; the component
// owns the rendering so every embedding of mountAgentListPane receives it.

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
globalThis.sessionStorage = dom.window.sessionStorage;
globalThis.window.kicon = (name) => `<span class="ki ki-${name}" aria-hidden="true"></span>`;
globalThis.kicon = globalThis.window.kicon;
const store = new Map();
globalThis.localStorage = {
    getItem: (k) => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => { store.set(k, String(v)); },
    removeItem: (k) => { store.delete(k); },
};

const { mountAgentListPane } = await import('../../kestrel_sovereign/static/js/agent_list.js');

function tick() { return new Promise((r) => setTimeout(r, 0)); }

const AGENTS = [
    { name: 'Emma', id: 'did:agent:emma', status: 'online' },
    { name: 'Kite', id: 'did:agent:kite', status: 'online' },
];

function openCircuit(target, count = 8) {
    return {
        target_agent_id: target,
        opened_at: '2026-09-25T12:00:00.000+00:00',
        opened_event_id: `event-${target}`,
        admitted_count: count,
        threshold: 8,
        window_seconds: 900,
    };
}

function mount({ statuses, reset = async () => ({ event: { kind: 'reset' } }), ask }) {
    const el = document.createElement('div');
    document.body.appendChild(el);
    let reads = 0;
    const resets = [];
    const handle = mountAgentListPane(el, {
        adapter: { mode: 'multi_agent', listAgents: async () => AGENTS },
        isThinking: () => false,
        api: {
            getHostStopStatus: async () => {
                const status = statuses[Math.min(reads, statuses.length - 1)];
                reads += 1;
                if (status instanceof Error) throw status;
                return status;
            },
            stopHost: async () => ({}),
            resetPeerStopCircuit: async (payload) => {
                resets.push(payload);
                return reset(payload);
            },
        },
        onPrepareStopAll: () => () => {},
        askCircuitResetReason: ask || (() => 'reviewed the peers'),
        hold: false,
        stopAllStatusIntervalMs: 60_000,
        storageKey: `a:circuit-${Math.random()}`,
    });
    return { el, handle, resets, reads: () => reads };
}

test('an open circuit renders by display name with a Reset action', async () => {
    const { el, handle } = mount({
        statuses: [{
            can_stop: true,
            in_flight_count: 0,
            peer_stop_circuit: { available: true, open: [openCircuit('did:agent:kite')] },
        }],
    });
    await tick();
    await tick();

    const banner = el.querySelector('.agent-peer-stop-circuits');
    assert.ok(banner, 'the pane owns the circuit banner');
    assert.equal(banner.hidden, false);
    const rows = banner.querySelectorAll('.agent-peer-stop-circuit');
    assert.equal(rows.length, 1);
    assert.equal(rows[0].dataset.target, 'did:agent:kite');
    assert.match(rows[0].textContent, /peer Stop circuit open: Kite/);
    assert.ok(rows[0].querySelector('.agent-peer-stop-circuit-reset'));
    handle.destroy();
    assert.equal(el.querySelector('.agent-peer-stop-circuits'), null, 'destroy removes it');
});

test('no circuit, or a non-sovereign status without circuits, draws nothing', async () => {
    for (const status of [
        { can_stop: true, in_flight_count: 0, peer_stop_circuit: { available: true, open: [] } },
        { can_stop: false, in_flight_count: 0 },
    ]) {
        const { el, handle } = mount({ statuses: [status] });
        await tick();
        await tick();
        const banner = el.querySelector('.agent-peer-stop-circuits');
        assert.equal(banner.hidden, true);
        assert.equal(banner.querySelectorAll('.agent-peer-stop-circuit').length, 0);
        handle.destroy();
    }
});

test('an unreadable breaker is shown as unavailable, never as clear', async () => {
    const { el, handle } = mount({
        statuses: [{
            can_stop: true,
            in_flight_count: 0,
            peer_stop_circuit: { available: false, open: [] },
        }],
    });
    await tick();
    await tick();
    const banner = el.querySelector('.agent-peer-stop-circuits');
    assert.equal(banner.hidden, false);
    assert.match(banner.textContent, /unavailable/);
    handle.destroy();
});

test('Reset calls the sovereign door with target and reason, then re-reads status', async () => {
    const { el, handle, resets, reads } = mount({
        statuses: [
            {
                can_stop: true,
                in_flight_count: 0,
                peer_stop_circuit: { available: true, open: [openCircuit('did:agent:emma')] },
            },
            {
                can_stop: true,
                in_flight_count: 0,
                peer_stop_circuit: { available: true, open: [] },
            },
        ],
    });
    await tick();
    await tick();
    const before = reads();

    el.querySelector('.agent-peer-stop-circuit-reset').click();
    await tick();
    await tick();

    assert.deepEqual(resets, [{ target: 'did:agent:emma', reason: 'reviewed the peers' }]);
    assert.ok(reads() > before, 'status re-read after the reset');
    assert.equal(el.querySelector('.agent-peer-stop-circuits').hidden, true);
    handle.destroy();
});

test('a cancelled reason never resets, and a refused reset is shown on its row', async () => {
    const cancelled = mount({
        statuses: [{
            can_stop: true,
            in_flight_count: 0,
            peer_stop_circuit: { available: true, open: [openCircuit('did:agent:kite')] },
        }],
        ask: () => null,
    });
    await tick();
    await tick();
    cancelled.el.querySelector('.agent-peer-stop-circuit-reset').click();
    await tick();
    assert.equal(cancelled.resets.length, 0, 'no reason, no reset');
    cancelled.handle.destroy();

    const refused = mount({
        statuses: [{
            can_stop: true,
            in_flight_count: 0,
            peer_stop_circuit: { available: true, open: [openCircuit('did:agent:kite')] },
        }],
        reset: async () => { throw new Error('Host control-plane authority is required.'); },
    });
    await tick();
    await tick();
    refused.el.querySelector('.agent-peer-stop-circuit-reset').click();
    await tick();
    await tick();
    const row = refused.el.querySelector('.agent-peer-stop-circuit');
    assert.ok(row, 'the circuit stays visible after a refused reset');
    assert.match(row.textContent, /Reset failed: Host control-plane authority is required/);
    refused.handle.destroy();
});
