// #3170: the agents banner shows open peer Stop circuits and wires the
// sovereign Reset. The circuits are their own sovereign-only read, polled
// independently of the Stop All status, so neither read's failure blanks the
// other; the component owns the rendering so every embedding of
// mountAgentListPane receives it.

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

// A failing assertion skips its test's destroy(); without this the pane's
// status interval keeps the runner alive until the file times out.
const mounted = [];
test.afterEach(() => {
    for (const handle of mounted.splice(0)) handle.destroy();
});

function httpError(status, message = `HTTP ${status}`) {
    return Object.assign(new Error(message), { status });
}

const CLEAR = { threshold: 8, window_seconds: 900, open: [] };
const STATUS = { can_stop: true, in_flight_count: 1 };

function circuits(...targets) {
    return { threshold: 8, window_seconds: 900, open: targets.map((t) => openCircuit(t)) };
}

function sequence(answers) {
    let reads = 0;
    const read = async () => {
        const answer = answers[Math.min(reads, answers.length - 1)];
        reads += 1;
        if (answer instanceof Error) throw answer;
        return answer;
    };
    return { read, count: () => reads };
}

function mount({
    statuses = [STATUS],
    circuitReads,
    reset = async () => ({ event: { kind: 'reset' } }),
    ask,
}) {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const status = sequence(statuses);
    const circuit = sequence(circuitReads);
    const resets = [];
    const handle = mountAgentListPane(el, {
        adapter: { mode: 'multi_agent', listAgents: async () => AGENTS },
        isThinking: () => false,
        api: {
            getHostStopStatus: status.read,
            getPeerStopCircuits: circuit.read,
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
    mounted.push(handle);
    return {
        el,
        handle,
        resets,
        statusReads: status.count,
        circuitReads: circuit.count,
    };
}

async function settle() {
    for (let i = 0; i < 4; i += 1) await tick();
}

function banner(el) {
    return el.querySelector('.agent-peer-stop-circuits');
}

test('an open circuit renders by display name with a Reset action', async () => {
    const { el, handle } = mount({ circuitReads: [circuits('did:agent:kite')] });
    await settle();

    assert.ok(banner(el), 'the pane owns the circuit banner');
    assert.equal(banner(el).hidden, false);
    const rows = banner(el).querySelectorAll('.agent-peer-stop-circuit');
    assert.equal(rows.length, 1);
    assert.equal(rows[0].dataset.target, 'did:agent:kite');
    assert.match(rows[0].textContent, /peer Stop circuit open: Kite/);
    assert.ok(rows[0].querySelector('.agent-peer-stop-circuit-reset'));
    handle.destroy();
    assert.equal(el.querySelector('.agent-peer-stop-circuits'), null, 'destroy removes it');
});

test('no open circuit, or a caller refused the circuit read, draws nothing', async () => {
    for (const answer of [CLEAR, httpError(403), httpError(401)]) {
        const { el, handle } = mount({ circuitReads: [answer] });
        await settle();
        assert.equal(banner(el).hidden, true);
        assert.equal(banner(el).querySelectorAll('.agent-peer-stop-circuit').length, 0);
        assert.doesNotMatch(banner(el).textContent, /unavailable/);
        handle.destroy();
    }
});

test('an inventory failure never hides an open circuit', async () => {
    const { el, handle, statusReads } = mount({
        statuses: [httpError(503, 'Host Stop target inventory is unavailable.')],
        circuitReads: [circuits('did:agent:kite')],
    });
    await settle();

    assert.ok(statusReads() >= 1, 'the Stop All status was read and failed');
    assert.equal(el.querySelector('.agent-stop-all-btn').disabled, true);
    assert.equal(banner(el).hidden, false);
    const rows = banner(el).querySelectorAll('.agent-peer-stop-circuit');
    assert.equal(rows.length, 1);
    assert.match(rows[0].textContent, /peer Stop circuit open: Kite/);
    handle.destroy();
});

test('an unreadable breaker is shown as unavailable and leaves Stop All intact', async () => {
    const { el, handle } = mount({
        statuses: [STATUS],
        circuitReads: [httpError(503, 'Peer Stop circuit breaker is unavailable.')],
    });
    await settle();

    assert.equal(banner(el).hidden, false);
    assert.equal(banner(el).querySelectorAll('.agent-peer-stop-circuit').length, 0);
    assert.match(banner(el).textContent, /circuit status unavailable/);
    const stopAll = el.querySelector('.agent-stop-all-btn');
    assert.equal(stopAll.disabled, false, 'Stop All still reads its own status');
    assert.equal(stopAll.dataset.inFlightCount, '1');
    handle.destroy();
});

test('a malformed circuit answer is unavailable, never clear', async () => {
    const { el, handle } = mount({ circuitReads: [{ threshold: 8 }] });
    await settle();
    assert.equal(banner(el).hidden, false);
    assert.match(banner(el).textContent, /unavailable/);
    handle.destroy();
});

test('Reset calls the sovereign door with target and reason, then re-reads circuits', async () => {
    const { el, handle, resets, circuitReads } = mount({
        circuitReads: [circuits('did:agent:emma'), CLEAR],
    });
    await settle();
    const before = circuitReads();

    el.querySelector('.agent-peer-stop-circuit-reset').click();
    await settle();

    assert.deepEqual(resets, [{ target: 'did:agent:emma', reason: 'reviewed the peers' }]);
    assert.ok(circuitReads() > before, 'circuits re-read after the reset');
    assert.equal(banner(el).hidden, true);
    handle.destroy();
});

test('a cancelled reason never resets, and a refused reset is shown on its row', async () => {
    const cancelled = mount({
        circuitReads: [circuits('did:agent:kite')],
        ask: () => null,
    });
    await settle();
    cancelled.el.querySelector('.agent-peer-stop-circuit-reset').click();
    await tick();
    assert.equal(cancelled.resets.length, 0, 'no reason, no reset');
    cancelled.handle.destroy();

    const refused = mount({
        circuitReads: [circuits('did:agent:kite')],
        reset: async () => { throw new Error('Host control-plane authority is required.'); },
    });
    await settle();
    refused.el.querySelector('.agent-peer-stop-circuit-reset').click();
    await settle();
    const row = refused.el.querySelector('.agent-peer-stop-circuit');
    assert.ok(row, 'the circuit stays visible after a refused reset');
    assert.match(row.textContent, /Reset failed: Host control-plane authority is required/);
    refused.handle.destroy();
});

test('a failed re-read of a circuit shown open reports it unknown, never cleared', async () => {
    const { el, handle } = mount({
        circuitReads: [circuits('did:agent:kite'), httpError(503)],
    });
    await settle();
    el.querySelector('.agent-peer-stop-circuit-reset').click();
    await settle();
    assert.equal(banner(el).hidden, false);
    assert.equal(banner(el).querySelectorAll('.agent-peer-stop-circuit').length, 0);
    assert.match(banner(el).textContent, /unavailable/);
    handle.destroy();
});

test('the Stop All status read never carries or clears circuit state', async () => {
    const { el, handle } = mount({
        // A stale host still riding circuits on status must not be believed.
        statuses: [{ ...STATUS, peer_stop_circuit: { available: true, open: [] } }],
        circuitReads: [circuits('did:agent:kite')],
    });
    await settle();
    assert.equal(banner(el).querySelectorAll('.agent-peer-stop-circuit').length, 1);
    handle.destroy();
});

test('a poll in flight across a Reset never repaints the pre-reset answer', async () => {
    let releaseStale;
    const stale = new Promise((resolve) => { releaseStale = resolve; });
    let reads = 0;
    const el = document.createElement('div');
    document.body.appendChild(el);
    let resetDone;
    const resetGate = new Promise((resolve) => { resetDone = resolve; });
    const handle = mountAgentListPane(el, {
        adapter: { mode: 'multi_agent', listAgents: async () => AGENTS },
        isThinking: () => false,
        api: {
            getHostStopStatus: async () => STATUS,
            getPeerStopCircuits: async () => {
                reads += 1;
                if (reads === 1) return circuits('did:agent:kite');
                if (reads === 2) return stale;
                return CLEAR;
            },
            stopHost: async () => ({}),
            resetPeerStopCircuit: async () => { await resetGate; return {}; },
        },
        onPrepareStopAll: () => () => {},
        askCircuitResetReason: () => 'reviewed the peers',
        hold: false,
        stopAllStatusIntervalMs: 60_000,
        storageKey: `a:circuit-${Math.random()}`,
    });
    mounted.push(handle);
    await settle();

    el.querySelector('.agent-peer-stop-circuit-reset').click();
    await tick();
    // A poll issued while the reset is still in flight.
    void handle.refreshPeerStopCircuits();
    await tick();
    resetDone();
    await settle();
    assert.equal(banner(el).hidden, true, 'the post-reset read is shown');

    releaseStale(circuits('did:agent:kite'));
    await settle();
    assert.equal(banner(el).hidden, true, 'the stale pre-reset answer is discarded');
    assert.equal(reads, 3);
    handle.destroy();
});
