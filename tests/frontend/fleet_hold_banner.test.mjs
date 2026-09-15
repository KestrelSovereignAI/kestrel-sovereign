// #3165: the agents BANNER carries the fleet Hold actions and a persistent
// held count. These tests pin the acceptance conditions:
//   - the fan-out renders per-agent hold outcomes beside the per-agent stop
//     outcomes, rather than collapsing either to a boolean;
//   - a host resume releases the host latch and leaves an agent someone held
//     individually still held;
//   - the held count is the host's own tally of durable state and converges
//     after a reload rather than living in this document's memory;
//   - empty, partial and refused stay three distinct readings;
//   - the shared pane embedding — which is what Frinz mounts — gets all of it;
//   - and the authority wiring (sovereign-only, host scope, observed-receipt
//     release) is asserted on the wire, not inferred from the rendering.

import test, { afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
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
globalThis.localStorage = {
    getItem: () => null,
    setItem: () => {},
    removeItem: () => {},
};

const { mountAgentListPane } = await import('../../kestrel_sovereign/static/js/agent_list.js');
const { closeKebabMenu } = await import('../../kestrel_sovereign/static/js/kebab_menu.js');

function tick() { return new Promise((r) => setTimeout(r, 0)); }

const EMMA = 'did:agent:emma';
const NELLIE = 'did:agent:nellie';
const KITE = 'did:agent:kite';
const WREN = 'did:agent:wren';

function latch({
    scope = 'agent',
    target = EMMA,
    receipt = 'receipt-agent-1',
    reason = 'runaway loop',
    actor = 'sovereign-key',
    at = '2026-09-15T10:00:00+00:00',
} = {}) {
    return {
        scope,
        target_id: target,
        reason,
        actor_id: actor,
        set_at: at,
        hold_receipt_id: receipt,
        revision: 1,
    };
}

// A stand-in for the host's two doors. The GET composes `held`/`sources` the
// way `EffectiveHoldState` does server-side — the console is never allowed to
// compose that verdict itself, so the double must be the one that does.
function makeHost({
    canHold = true,
    canStop = true,
    inFlight = 0,
    hostHold = null,
    agents = [EMMA, NELLIE, KITE],
    agentHolds = {},
} = {}) {
    const calls = { read: 0, set: [], release: [], stop: [], order: [] };
    // A canned mutation reply still has to leave the host's durable state
    // agreeing with the `current` it just claimed — otherwise the confirming
    // GET contradicts the mutation, and the double, not the component, is what
    // the test measures. (`refused_stale` in particular means the store's live
    // latch IS the superseding one it returned.)
    function canned(reply) {
        if (reply && typeof reply === 'object' && 'current' in reply) {
            host.hostHold = reply.current || null;
        }
        return reply;
    }
    const host = {
        calls,
        canHold,
        canStop,
        inFlight,
        hostHold,
        agents: [...agents],
        agentHolds: { ...agentHolds },
        failRead: false,
        setHostHoldResult: null,
        releaseHostHoldResult: null,
        setHostAgent() {},
        async getHostHoldState() {
            calls.read += 1;
            if (host.failRead) throw new Error('host unreachable');
            return {
                can_hold: host.canHold,
                host_hold: host.hostHold,
                agents: host.agents.map((agentId) => {
                    const agentHold = host.agentHolds[agentId] || null;
                    const sources = [];
                    if (host.hostHold) sources.push('host');
                    if (agentHold) sources.push('agent');
                    return {
                        agent_id: agentId,
                        held: sources.length > 0,
                        sources,
                        agent_hold: agentHold,
                    };
                }),
            };
        },
        async setHostHold(payload) {
            calls.set.push(payload);
            calls.order.push('hold');
            if (host.setHostHoldResult instanceof Error) throw host.setHostHoldResult;
            if (host.setHostHoldResult) return canned(host.setHostHoldResult);
            host.hostHold = latch({
                scope: 'host',
                target: 'host',
                receipt: 'receipt-host-1',
                reason: payload.reason,
            });
            return {
                receipt: {
                    receipt_id: 'receipt-host-1',
                    operation_id: payload.operation_id,
                    action: 'hold',
                    disposition: 'applied',
                },
                current: host.hostHold,
            };
        },
        async releaseHostHold(payload) {
            calls.release.push(payload);
            calls.order.push('release');
            if (host.releaseHostHoldResult instanceof Error) throw host.releaseHostHoldResult;
            if (host.releaseHostHoldResult) return canned(host.releaseHostHoldResult);
            host.hostHold = null;
            return {
                receipt: {
                    receipt_id: 'receipt-host-release',
                    operation_id: payload.operation_id,
                    action: 'release',
                    disposition: 'applied',
                },
                current: null,
            };
        },
        async getHostStopStatus() {
            return { can_stop: host.canStop, in_flight_count: host.inFlight };
        },
        async stopHost(payload) {
            calls.stop.push(payload);
            calls.order.push('stop');
            return hostStopEnvelope(payload.correlation_id, host.agents);
        },
    };
    return host;
}

function hostStopEnvelope(correlationId, agents) {
    const outcomes = agents.map((agent, index) => ({
        scope: 'host',
        requested_target: null,
        resolved_target: agent,
        agent_id: agent,
        disposition: 'stopped',
        correlation_id: correlationId,
        receipt_id: 'receipt-host-stop',
        ordinal: index,
    }));
    return {
        success: true,
        state: 'confirmed',
        target_count: outcomes.length,
        confirmed_count: outcomes.length,
        unconfirmed_count: 0,
        correlation_id: correlationId,
        stop_outcomes: outcomes,
    };
}

const mounted = [];
afterEach(() => {
    closeKebabMenu();
    while (mounted.length) {
        const handle = mounted.pop();
        try { handle.destroy(); } catch (_) { /* best-effort */ }
    }
});

// index.html's static `#agents-pane`: the chrome the console ADOPTS. It
// matters for teardown, because a built header is removed whole on destroy and
// would mask a control this mount leaked inside it — the adopted header
// survives, so only per-control removal can keep it clean.
function makeConsolePane(ownerDocument = document) {
    const el = ownerDocument.createElement('aside');
    el.id = 'agents-pane';
    el.className = 'pane-sidebar';
    el.innerHTML = `
        <div class="pane-header"><h3>Agents</h3>
            <button id="collapse-agents-btn" class="collapse-btn"></button></div>
        <div id="agents-list" class="pane-content"></div>
        <div id="resize-agents" class="resize-handle"></div>`;
    ownerDocument.body.appendChild(el);
    return el;
}

function mountPane({
    host,
    ownerDocument = document,
    stopAll = true,
    container = null,
    extra = {},
} = {}) {
    const el = container || ownerDocument.createElement('div');
    if (!container) ownerDocument.body.appendChild(el);
    const handle = mountAgentListPane(el, {
        api: host,
        adapter: {
            mode: 'multi_agent',
            listAgents: async () => host.agents.map((agentId) => ({
                name: agentId.split(':').pop(),
                displayName: agentId.split(':').pop(),
                id: agentId,
                status: 'online',
            })),
        },
        askHoldReason: () => 'operator reason',
        confirmStopAll: () => true,
        ...(stopAll ? { onPrepareStopAll: () => () => {} } : {}),
        // Long enough that no test observes a poll it did not ask for.
        holdStatusIntervalMs: 1e7,
        stopAllStatusIntervalMs: 1e7,
        ...extra,
    });
    mounted.push(handle);
    return { el, handle };
}

// Settle the mount: the agent list load, the stop-status read and the latch
// read are three independent promises.
async function settle() {
    for (let i = 0; i < 6; i++) await tick();
}

function countEl(el) { return el.querySelector('.agent-hold-count'); }
function kebab(el) { return el.querySelector('.agent-fleet-kebab'); }

function openFleetMenu(el, ownerDocument = document) {
    const btn = kebab(el);
    assert.ok(btn, 'the banner carries a fleet menu');
    btn.click();
    return Array.from(ownerDocument.querySelectorAll('.kebab-menu .kebab-menu-item'));
}

function menuActions(el, ownerDocument = document) {
    return openFleetMenu(el, ownerDocument).map((item) => item.dataset.action);
}

async function chooseFleetAction(el, action, ownerDocument = document) {
    const items = openFleetMenu(el, ownerDocument);
    const item = items.find((candidate) => candidate.dataset.action === action);
    assert.ok(item, `the fleet menu offers ${action} (saw ${items.map((i) => i.dataset.action).join(', ')})`);
    item.click();
    await settle();
}

// ---------------------------------------------------------------------------
// The held count
// ---------------------------------------------------------------------------

test('the banner held count is the host\'s tally, and a partially held fleet says so', async () => {
    const host = makeHost({ agentHolds: { [NELLIE]: latch({ target: NELLIE }) } });
    const { el } = mountPane({ host });
    await settle();

    const badge = countEl(el);
    assert.equal(badge.hidden, false, 'a partially held fleet is legible at rest');
    assert.equal(badge.dataset.holdState, 'partial');
    assert.equal(badge.textContent, '1 of 3 held');
    assert.equal(badge.dataset.heldCount, '1');
    assert.equal(badge.dataset.targetCount, '3');
});

test('a fully held fleet reads as all; an unheld one is quiet but still distinct', async () => {
    const unheld = makeHost();
    const { el: quiet } = mountPane({ host: unheld });
    await settle();
    assert.equal(countEl(quiet).hidden, true, 'nothing held means nothing to say');
    assert.equal(countEl(quiet).dataset.holdState, 'none',
        'but "none held" is a reading, and must not read the same as "unknown"');
    assert.equal(countEl(quiet).dataset.heldCount, '0');
    assert.equal(countEl(quiet).dataset.targetCount, '3');

    const allHeld = makeHost({ hostHold: latch({ scope: 'host', target: 'host', receipt: 'receipt-host-1' }) });
    const { el } = mountPane({ host: allHeld });
    await settle();
    assert.equal(countEl(el).hidden, false);
    assert.equal(countEl(el).dataset.holdState, 'all');
    assert.equal(countEl(el).textContent, 'All 3 held');
});

test('an empty inventory and an unanswered host are each distinct from "none held"', async () => {
    const empty = makeHost({ agents: [] });
    const { el: emptyEl } = mountPane({ host: empty });
    await settle();
    assert.equal(countEl(emptyEl).dataset.holdState, 'empty',
        'a host that named no agents has not told us that nothing is held');
    assert.equal(countEl(emptyEl).dataset.targetCount, '0');

    // `hold: false` is the embedder's opt-out: no door, so no reading at all.
    const noDoor = makeHost();
    const { el: noDoorEl } = mountPane({ host: noDoor, extra: { hold: false } });
    await settle();
    assert.equal(countEl(noDoorEl).dataset.holdState, 'unknown');
    assert.equal(countEl(noDoorEl).hidden, true);
    assert.equal(kebab(noDoorEl).hidden, true, 'and no fleet menu onto a door that is not there');
    assert.equal(noDoor.calls.read, 0, 'nor a probe of it');
});

test('the held count converges after a reload, because it is read not remembered', async () => {
    const host = makeHost({ agentHolds: { [KITE]: latch({ target: KITE }) } });
    const first = mountPane({ host });
    await settle();
    assert.equal(countEl(first.el).textContent, '1 of 3 held');
    first.handle.destroy();
    mounted.pop();

    // A brand new document tree with no memory of the first mount at all.
    const second = mountPane({ host });
    assert.equal(countEl(second.el).dataset.holdState, 'unknown',
        'before the host answers, the banner knows nothing');
    await settle();
    assert.equal(countEl(second.el).textContent, '1 of 3 held',
        'and converges onto the durable reading, unprompted');
    assert.equal(countEl(second.el).dataset.holdState, 'partial');
});

// ---------------------------------------------------------------------------
// Authority
// ---------------------------------------------------------------------------

test('the fleet menu is offered to a sovereign caller only', async () => {
    const host = makeHost({ canHold: false, agentHolds: { [EMMA]: latch() } });
    const { el } = mountPane({ host });
    await settle();

    const btn = kebab(el);
    assert.equal(btn.hidden, false, 'the door exists, so the control is visible');
    assert.equal(btn.disabled, true, 'but this caller may not use it');
    assert.match(btn.title, /Sovereign host authority/);
    btn.click();
    assert.equal(document.querySelector('.kebab-menu'), null,
        'a disabled fleet kebab opens nothing');
    // The reading itself is not an authority, so it is still rendered.
    assert.equal(countEl(el).textContent, '1 of 3 held');
});

test('Hold all latches the HOST scope and names no caller-chosen target', async () => {
    const host = makeHost();
    const { el } = mountPane({ host });
    await settle();

    await chooseFleetAction(el, 'hold-all');

    assert.equal(host.calls.set.length, 1);
    const [payload] = host.calls.set;
    assert.equal(payload.scope, 'host');
    assert.ok(!('target_id' in payload),
        'the host scope has exactly one latch, so there is nothing to name');
    assert.equal(payload.reason, 'operator reason');
    assert.match(payload.operation_id, /^ui-host-hold:/);
    assert.equal(host.calls.stop.length, 0, 'Hold is not Stop');
});

test('the host resume releases exactly the receipt the operator saw', async () => {
    const host = makeHost({
        hostHold: latch({ scope: 'host', target: 'host', receipt: 'receipt-host-observed' }),
    });
    const { el } = mountPane({ host });
    await settle();

    await chooseFleetAction(el, 'resume-host-hold');

    assert.equal(host.calls.release.length, 1);
    const [payload] = host.calls.release;
    assert.equal(payload.scope, 'host');
    assert.equal(payload.expected_hold_receipt_id, 'receipt-host-observed');
    assert.match(payload.operation_id, /^ui-host-resume:/);
});

// ---------------------------------------------------------------------------
// The fan-out
// ---------------------------------------------------------------------------

test('the Hold fan-out renders a per-agent outcome for every agent the host named', async () => {
    const host = makeHost({ agentHolds: { [NELLIE]: latch({ target: NELLIE }) } });
    const { el } = mountPane({ host });
    await settle();

    await chooseFleetAction(el, 'hold-all');

    const results = el.querySelector('.agent-fleet-hold-results');
    assert.equal(results.hidden, false);
    assert.equal(results.dataset.action, 'hold');
    assert.equal(results.dataset.disposition, 'applied');
    assert.equal(results.dataset.holdState, 'all');
    assert.match(results.querySelector('p').textContent, /^Held\. 3 of 3 agents held\./);

    const rows = Array.from(results.querySelectorAll('li'));
    assert.deepEqual(rows.map((row) => row.dataset.agentId), [EMMA, NELLIE, KITE]);
    assert.deepEqual(rows.map((row) => row.dataset.held), ['true', 'true', 'true']);
    assert.deepEqual(rows.map((row) => row.dataset.sources),
        ['host', 'host agent', 'host'],
        'an agent held by BOTH independent latches says so');
    assert.equal(rows[1].textContent, 'nellie: held (host, agent)');
});

// The fan-out rows are a RECEIPT, and the host's mutation reply carries only
// its own latch — so the rows have to come from a read taken after that latch
// committed. Composing them from the last poll instead is confidently wrong
// about everything that moved in between, and a fleet is exactly the thing
// that moves: agents are spawned and retired, and the other latch axis is
// writable from another tab, the CLI, or a mandate holder.
test('the Hold fan-out is the host\'s reading after the mutation, not a memory of one before it', async () => {
    const host = makeHost();
    const { el } = mountPane({ host });
    await settle();
    // Everything the banner has read so far: Emma, Nellie, Kite; none held.
    assert.equal(countEl(el).dataset.targetCount, '3');

    // The fleet moves while the operator is reading the prompt: Kite is
    // retired, Wren is spawned, and somebody holds Nellie individually.
    const originalSet = host.setHostHold.bind(host);
    host.setHostHold = async (payload) => {
        host.agents = [EMMA, NELLIE, WREN];
        host.agentHolds = { [NELLIE]: latch({ target: NELLIE, receipt: 'receipt-nellie' }) };
        return originalSet(payload);
    };

    await chooseFleetAction(el, 'hold-all');

    const results = el.querySelector('.agent-fleet-hold-results');
    assert.equal(results.dataset.fanout, 'confirmed');
    const rows = Array.from(results.querySelectorAll('li'));
    assert.deepEqual(rows.map((row) => row.dataset.agentId), [EMMA, NELLIE, WREN],
        'a receipt naming a retired agent, and silent about a live one, is a fabrication');
    assert.deepEqual(rows.map((row) => row.dataset.sources),
        ['host', 'host agent', 'host'],
        'and a host mutation never touched Nellie\'s own latch, so it may not erase it');
    // Wren is not in the loaded card list yet, so the row falls back to the
    // identity the HOST named rather than inventing a display name for it.
    assert.match(rows[2].textContent, /^did:agent:wren: held/);
});

test('a fan-out the host could not confirm is reported unconfirmed, and names nobody', async () => {
    const host = makeHost({ agentHolds: { [NELLIE]: latch({ target: NELLIE }) } });
    const { el } = mountPane({ host });
    await settle();

    // The mutation lands; the read that would confirm the fan-out does not.
    const originalSet = host.setHostHold.bind(host);
    host.setHostHold = async (payload) => {
        const response = await originalSet(payload);
        host.failRead = true;
        return response;
    };

    await chooseFleetAction(el, 'hold-all');

    const results = el.querySelector('.agent-fleet-hold-results');
    assert.equal(results.dataset.disposition, 'applied',
        'the latch itself is not in doubt — its own receipt came back');
    assert.equal(results.dataset.fanout, 'unconfirmed');
    assert.equal(results.dataset.holdState, 'unknown',
        'and an unread fleet is not "all held" merely because the host latch is set');
    assert.equal(results.querySelectorAll('li').length, 0,
        'a per-agent row nobody read is a per-agent row nobody may print');
    const summary = results.querySelector('p').textContent;
    assert.match(summary, /could not be confirmed/);
    assert.doesNotMatch(summary, /\d+ of \d+/,
        'a tally composed from the pre-mutation reading is the same fabrication');

    // The badge is a different job — best current knowledge, marked
    // unconfirmed — and it must not lose the latch that demonstrably committed.
    assert.equal(countEl(el).textContent, 'All 3 held');
    assert.equal(countEl(el).dataset.holdStale, 'true');
});

test('the confirming read is never a read that was already in flight', async () => {
    const host = makeHost();
    const { el, handle } = mountPane({ host });
    await settle();

    // A read issued BEFORE the gesture and still open across it. Its payload
    // cannot confirm a latch that did not exist when it was sent, and the
    // mutation orphans it by sequence, so a confirming read that merely joined
    // it would resolve to nothing at all — which is how the post-mutation
    // refresh became a no-op until the next poll tick.
    let openTheRead;
    const gate = new Promise((resolve) => { openTheRead = resolve; });
    const live = host.getHostHoldState.bind(host);
    let reads = 0;
    host.getHostHoldState = async () => {
        reads += 1;
        if (reads === 1) await gate;
        return live();
    };
    const stranded = handle.refreshHoldState();
    await tick();
    assert.equal(reads, 1, 'the stranded read is open');

    await chooseFleetAction(el, 'hold-all');

    const results = el.querySelector('.agent-fleet-hold-results');
    assert.equal(results.dataset.fanout, 'confirmed',
        'the fan-out waited for a read that started after the latch committed');
    assert.ok(reads >= 2, 'which means it issued one, rather than joining the open one');

    openTheRead();
    await stranded;
});

test('a mutation this document cannot paint still gets a read of its own', async () => {
    const host = makeHost();
    const { el, handle } = mountPane({ host });
    await settle();

    // A reply whose `current` is unusable: there is nothing for the browser to
    // paint, so nothing moves the read fence either. The confirming read has to
    // be fresh on its own account — an in-flight read issued BEFORE the request
    // does not become evidence about it just because the request happened.
    host.setHostHold = async (payload) => {
        host.calls.set.push(payload);
        return {
            receipt: {
                receipt_id: 'receipt-host-1',
                operation_id: payload.operation_id,
                action: 'hold',
                disposition: 'applied',
            },
            current: { ...latch({ scope: 'host', target: 'host' }), hold_receipt_id: '' },
        };
    };

    let openTheRead;
    const gate = new Promise((resolve) => { openTheRead = resolve; });
    const live = host.getHostHoldState.bind(host);
    let reads = 0;
    host.getHostHoldState = async () => {
        reads += 1;
        if (reads === 1) await gate;
        return live();
    };
    const stranded = handle.refreshHoldState();
    await tick();
    assert.equal(reads, 1, 'the stranded read is open');

    await chooseFleetAction(el, 'hold-all');

    const results = el.querySelector('.agent-fleet-hold-results');
    assert.equal(results.hidden, false,
        'the fan-out must not sit behind a read that predates the request it describes');
    assert.equal(results.dataset.disposition, 'indeterminate',
        'a receipt claiming success while carrying no latch contradicts itself');
    assert.ok(reads >= 2, 'and it issued a read of its own to say so');

    openTheRead();
    await stranded;
});

test('a confirming read that a later mutation orphaned does not become the fan-out', async () => {
    const host = makeHost();
    const { el, handle } = mountPane({ host });
    await settle();

    // Another surface — a card Hold, another tab's poll landing a latch —
    // commits while the confirming read is in flight, which orphans that read
    // by sequence. What is left on screen is a composition again, and it does
    // not become a reading just because the last real one was recent: `stale`
    // is still false, so only `composed` can tell the panel to hold its tongue.
    let orphanNextRead = null;
    const live = host.getHostHoldState.bind(host);
    host.getHostHoldState = async () => {
        const payload = await live();
        if (orphanNextRead) {
            const orphan = orphanNextRead;
            orphanNextRead = null;
            orphan();
        }
        return payload;
    };
    orphanNextRead = () => handle.list.applyHostLatch(
        latch({ scope: 'host', target: 'host', receipt: 'receipt-somebody-else' }),
    );

    await chooseFleetAction(el, 'hold-all');

    const results = el.querySelector('.agent-fleet-hold-results');
    assert.equal(results.dataset.fanout, 'unconfirmed');
    assert.equal(results.querySelectorAll('li').length, 0,
        'the rows would have been composed, not read — and a receipt is not composed');
});

test('Stop all and hold holds first, then stops, and keeps the two receipts apart', async () => {
    const host = makeHost({ inFlight: 2 });
    const { el } = mountPane({ host });
    await settle();

    await chooseFleetAction(el, 'stop-all-and-hold');

    assert.deepEqual(host.calls.order.filter((k) => k !== 'release'), ['hold', 'stop'],
        'latching willingness first closes the window a heartbeat could start a turn in');
    assert.equal(host.calls.set.length, 1);
    assert.equal(host.calls.stop.length, 1);

    const holdResults = el.querySelector('.agent-fleet-hold-results');
    const stopResults = el.querySelector('.agent-stop-all-results');
    assert.equal(holdResults.hidden, false);
    assert.equal(stopResults.hidden, false);
    assert.match(holdResults.textContent, /Held\./);
    assert.match(stopResults.textContent, /Stop All results: 3 stopped\./);
    assert.deepEqual(
        Array.from(stopResults.querySelectorAll('li')).map((row) => row.dataset.disposition),
        ['stopped', 'stopped', 'stopped'],
        'the Stop fan-out keeps reporting per-target dispositions of its own');
    // Two typed requests, two receipts — the gesture is compound, the
    // operations are not.
    assert.ok(holdResults.querySelector('p').title.includes('receipt-host-1'));
});

test('Stop all and hold is not offered when there is nothing in flight', async () => {
    const host = makeHost({ inFlight: 0 });
    const { el } = mountPane({ host });
    await settle();

    assert.deepEqual(menuActions(el), ['hold-all'],
        'with nothing to stop, "stop all and hold" IS the plain Hold above it');
});

test('a failed Hold does not silently degrade "stop all and hold" into a bare Stop all', async () => {
    const host = makeHost({ inFlight: 2 });
    host.setHostHoldResult = new Error('hold door unreachable');
    const { el } = mountPane({ host });
    await settle();

    await chooseFleetAction(el, 'stop-all-and-hold');

    assert.equal(host.calls.stop.length, 0,
        'stopping without latching leaves the fleet free to restart on the next heartbeat');
    const results = el.querySelector('.agent-fleet-hold-results');
    assert.equal(results.dataset.disposition, 'unreachable');
    assert.equal(results.dataset.holdState, 'refused',
        'a refusal is its own state, never "none held"');
    assert.match(results.textContent, /Hold unreachable/);
});

test('a Hold that answered without latching is indeterminate, and stops nothing', async () => {
    const host = makeHost({ inFlight: 2 });
    // A receipt that claims success while carrying no latch contradicts itself.
    host.setHostHoldResult = {
        receipt: { receipt_id: 'receipt-x', operation_id: 'op', action: 'hold', disposition: 'applied' },
        current: null,
    };
    const { el } = mountPane({ host });
    await settle();

    await chooseFleetAction(el, 'stop-all-and-hold');

    assert.equal(host.calls.stop.length, 0);
    const results = el.querySelector('.agent-fleet-hold-results');
    assert.equal(results.dataset.disposition, 'indeterminate');
    assert.equal(results.dataset.holdState, 'refused');
});

// ---------------------------------------------------------------------------
// Independence of the two latches
// ---------------------------------------------------------------------------

test('a host resume leaves an agent held individually still held', async () => {
    const host = makeHost({
        hostHold: latch({ scope: 'host', target: 'host', receipt: 'receipt-host-observed' }),
        agentHolds: { [NELLIE]: latch({ target: NELLIE, receipt: 'receipt-nellie' }) },
    });
    const { el } = mountPane({ host });
    await settle();
    assert.equal(countEl(el).textContent, 'All 3 held');

    await chooseFleetAction(el, 'resume-host-hold');

    const results = el.querySelector('.agent-fleet-hold-results');
    assert.equal(results.dataset.action, 'release');
    assert.equal(results.dataset.disposition, 'applied');
    assert.equal(results.dataset.holdState, 'partial');
    const rows = Array.from(results.querySelectorAll('li'));
    assert.deepEqual(rows.map((row) => row.dataset.held), ['false', 'true', 'false'],
        'releasing the host latch releases the host latch, and nothing else');
    assert.equal(rows[1].dataset.sources, 'agent');
    assert.equal(countEl(el).textContent, '1 of 3 held');
    assert.equal(countEl(el).dataset.holdState, 'partial');
    assert.equal(host.calls.release.length, 1,
        'and it did not fan out a release to each agent on the way');
});

test('a stale release is refused, and the superseding host latch stands', async () => {
    const superseding = latch({
        scope: 'host', target: 'host', receipt: 'receipt-host-2', reason: 'somebody else',
    });
    const host = makeHost({
        hostHold: latch({ scope: 'host', target: 'host', receipt: 'receipt-host-observed' }),
    });
    host.releaseHostHoldResult = {
        receipt: {
            receipt_id: 'receipt-refusal',
            operation_id: 'op',
            action: 'release',
            disposition: 'refused_stale',
        },
        current: superseding,
    };
    const { el } = mountPane({ host });
    await settle();

    await chooseFleetAction(el, 'resume-host-hold');

    const results = el.querySelector('.agent-fleet-hold-results');
    assert.equal(results.dataset.disposition, 'refused_stale');
    assert.equal(results.dataset.holdState, 'refused');
    assert.match(results.textContent, /release refused/);
    // The latch that came back is the one that actually holds the fleet now, so
    // the next Resume's compare-and-set names it.
    await chooseFleetAction(el, 'resume-host-hold');
    assert.equal(host.calls.release[1].expected_hold_receipt_id, 'receipt-host-2');
});

test('a host Hold that committed keeps the banner held when the confirming read fails', async () => {
    const host = makeHost();
    const { el } = mountPane({ host });
    await settle();

    // The mutation lands; the read that would confirm it does not.
    const originalSet = host.setHostHold.bind(host);
    host.setHostHold = async (payload) => {
        const response = await originalSet(payload);
        host.failRead = true;
        return response;
    };

    await chooseFleetAction(el, 'hold-all');

    const badge = countEl(el);
    assert.equal(badge.hidden, false);
    assert.equal(badge.textContent, 'All 3 held',
        'a committed latch is not rolled back by a read that never arrived');
    assert.equal(badge.dataset.holdStale, 'true', 'and it says the reading is unconfirmed');
    assert.match(badge.title, /currently unreadable/);
    assert.equal(kebab(el).disabled, true,
        'a further mutation needs a receipt this read did not get');
});

// ---------------------------------------------------------------------------
// The shared pane embedding
// ---------------------------------------------------------------------------

test('an embedding host mounting into ITS document gets the same banner controls', async () => {
    const other = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' });
    const otherDoc = other.window.document;
    assert.notEqual(otherDoc, document);

    const host = makeHost({ agentHolds: { [EMMA]: latch() } });
    const { el } = mountPane({ host, ownerDocument: otherDoc });
    await settle();

    const btn = kebab(el);
    assert.ok(btn, 'the fleet menu is component-owned, so the embed inherits it');
    assert.equal(btn.ownerDocument, otherDoc,
        'a control built in the console\'s document belongs to a tree this host never shows');
    assert.equal(countEl(el).ownerDocument, otherDoc);
    assert.equal(countEl(el).textContent, '1 of 3 held');

    // And its menu opens in that document too.
    const actions = menuActions(el, otherDoc);
    assert.deepEqual(actions, ['hold-all']);
    assert.equal(document.querySelector('.kebab-menu'), null,
        'nothing leaked into the console\'s document');
    assert.ok(otherDoc.querySelector('.kebab-menu'));
});

test('a pane with no Stop All still gets the held count and Hold all', async () => {
    // Hold and Stop are different types with different doors, so an embedder
    // that never opted into the fleet Stop fence must still be able to latch —
    // and the Hold results must find somewhere to live without Stop's block.
    const host = makeHost({ inFlight: 4, agentHolds: { [EMMA]: latch() } });
    const { el } = mountPane({ host, stopAll: false });
    await settle();

    assert.equal(el.querySelector('.agent-stop-all-btn'), null, 'no Stop All was opted into');
    assert.equal(countEl(el).textContent, '1 of 3 held');
    assert.deepEqual(menuActions(el), ['hold-all'],
        'and "stop all and hold" needs the Stop door this pane does not have');

    await chooseFleetAction(el, 'hold-all');
    const results = el.querySelector('.agent-fleet-hold-results');
    assert.equal(results.hidden, false);
    const listRoot = el.querySelector('.agent-list-root');
    assert.ok(listRoot, 'the list is mounted');
    assert.equal(
        results.compareDocumentPosition(listRoot) & Node.DOCUMENT_POSITION_FOLLOWING,
        Node.DOCUMENT_POSITION_FOLLOWING,
        'the fan-out receipt reads above the list, not appended below it',
    );
    assert.equal(host.calls.stop.length, 0);
});

// Both teardown tests use the ADOPTED console chrome deliberately: a BUILT
// header is removed whole by destroy(), which would mask a leaked control
// inside it and make these two vacuous.
test('re-mounting adopted chrome leaves exactly one fleet control of each', async () => {
    const host = makeHost();
    const pane = makeConsolePane();
    mountPane({ host, container: pane });
    await settle();

    // A host may remount into adopted chrome without retiring the old handle
    // first; the container's owner handle is what retires it (#3155).
    const second = mountAgentListPane(pane, {
        api: host,
        adapter: { mode: 'multi_agent', listAgents: async () => [] },
        askHoldReason: () => 'operator reason',
        onPrepareStopAll: () => () => {},
        holdStatusIntervalMs: 1e7,
        stopAllStatusIntervalMs: 1e7,
    });
    mounted.push(second);
    await settle();

    assert.ok(pane.querySelector('.pane-header'), 'the adopted header is still there');
    assert.equal(pane.querySelectorAll('.agent-fleet-kebab').length, 1,
        'two kebabs would race one gesture into two menus');
    assert.equal(pane.querySelectorAll('.agent-hold-count').length, 1);
    assert.equal(pane.querySelectorAll('.agent-fleet-hold-results').length, 1);
});

test('destroy() takes the fleet Hold chrome out of adopted chrome that survives it', async () => {
    const host = makeHost();
    const pane = makeConsolePane();
    const { handle } = mountPane({ host, container: pane });
    await settle();
    assert.ok(kebab(pane) && countEl(pane) && pane.querySelector('.agent-fleet-hold-results'));

    handle.destroy();
    mounted.pop();

    assert.ok(pane.querySelector('.pane-header'),
        'adopted chrome is left in place, so a leak here would be permanent');
    assert.equal(kebab(pane), null, 'a leaked kebab keeps a dead menu callback alive');
    assert.equal(countEl(pane), null);
    assert.equal(pane.querySelector('.agent-fleet-hold-results'), null);
});

test('the banner reads left to right: held count, Stop all, fleet menu, collapse', async () => {
    const host = makeHost({ inFlight: 1 });
    const { el } = mountPane({ host });
    await settle();

    const header = el.querySelector('.pane-header');
    const order = Array.from(header.children)
        .map((child) => child.className)
        .filter((name) => /agent-hold-count|agent-stop-all-btn|agent-fleet-kebab|collapse-btn/.test(name));
    assert.deepEqual(order.map((name) => name.split(' ').find((cls) => (
        ['agent-hold-count', 'agent-stop-all-btn', 'agent-fleet-kebab', 'collapse-btn'].includes(cls)
    ))), ['agent-hold-count', 'agent-stop-all-btn', 'agent-fleet-kebab', 'collapse-btn']);
});

// ---------------------------------------------------------------------------
// CSS: jsdom has no layout, so the rules that make these controls visible are
// asserted as the stylesheet facts they are.
// ---------------------------------------------------------------------------

test('the banner CSS gives the fleet surface its resting behaviour', () => {
    const here = dirname(fileURLToPath(import.meta.url));
    const css = readFileSync(
        join(here, '..', '..', 'kestrel_sovereign', 'static', 'index.css'),
        'utf8',
    );
    const emitted = ['agent-hold-count', 'agent-fleet-kebab', 'agent-fleet-hold-results'];
    const missing = emitted.filter((cls) => !new RegExp(`\\.${cls}(?![\\w-])`).test(css));
    assert.deepEqual(missing, [], `index.css is missing styles for: ${missing.join(', ')}`);

    assert.match(css, /\.agent-hold-count\[hidden\]\s*\{[^}]*display:\s*none/,
        'the count is hidden when there is nothing to count');
    assert.match(css, /\.agent-fleet-kebab\[hidden\]\s*\{[^}]*display:\s*none/,
        '`.kebab-btn` sets display:inline-flex, which beats the UA [hidden] rule');

    // The shared kebab primitive is a hover-reveal row action (`opacity: 0`,
    // revealed by a `.conversation-item:hover` selector these buttons never
    // match), so both Hold menus need an override to be visible at rest —
    // which is the one thing a latch surface may not fail at.
    //
    // Asserted through the real CASCADE rather than by matching the rule text.
    // The first attempt at this fix wrote `.agent-card-kebab { opacity: 1 }`,
    // which ties `.kebab-btn` on specificity and loses to it on order — a rule
    // that is present in the file and does nothing. A text match would have
    // called that clean.
    const probe = new JSDOM(
        `<!doctype html><html><head><style>${css}</style></head><body>`
        + '<button class="kebab-btn agent-card-kebab"></button>'
        + '<button class="kebab-btn agent-fleet-kebab"></button>'
        + '<button class="kebab-btn"></button></body></html>',
    );
    const [card, fleet, plain] = probe.window.document.querySelectorAll('button');
    assert.equal(probe.window.getComputedStyle(card).opacity, '1',
        'the card kebab (#3164) is visible at rest');
    assert.equal(probe.window.getComputedStyle(fleet).opacity, '1',
        'and so is the fleet kebab');
    assert.equal(probe.window.getComputedStyle(plain).opacity, '0',
        'while the primitive itself still hides — otherwise the override above '
        + 'is testing nothing and the conversation rows lost their hover-reveal');
});
