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
// Explicit, not ambient: Node 25+ exposes a global sessionStorage, Node 22 (CI) does not.
globalThis.sessionStorage = dom.window.sessionStorage;
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

// For the one test that drives the real latch POLL: wait for the condition, not
// for a duration, so a slow machine does not decide the result. A wait that
// runs out is the failure under test, and names the condition that never came.
async function waitFor(predicate, label, timeoutMs = 3000) {
    const deadline = Date.now() + timeoutMs;
    while (!predicate()) {
        if (Date.now() > deadline) assert.fail(`timed out waiting for ${label}`);
        await new Promise((resolve) => setTimeout(resolve, 5));
    }
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

// Two doors fence these controls: the button's `disabled` attribute, and
// `fleetMenuItems()` behind it. jsdom suppresses `click()` on a disabled
// button, so asserting an empty menu without forcing the button open restates
// the attribute and proves nothing about the gate behind it. Force it, read the
// menu, and put the attribute back — so the repaint that ends the fence is
// still the thing under test afterwards.
function menuActionsBehindTheFence(el, ownerDocument = document) {
    const btn = kebab(el);
    const wasDisabled = btn.disabled;
    btn.disabled = false;
    try {
        return menuActions(el, ownerDocument);
    } finally {
        closeKebabMenu();
        btn.disabled = wasDisabled;
    }
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

// A read that never settles is a different failure from a read that fails, and
// it lands somewhere else: the list COALESCES on the read it has in flight and
// retires that join target only when it settles. So a confirming read the host
// accepts and never answers stands in front of every later poll — each tick
// joins a promise that cannot land, no request goes out, and the banner's count
// and Hold authority never recover even once the host does. The leash bounds
// the gesture's WAIT; it has to retire the read as well, or one hung request
// takes the fleet surface down for the life of the mount.
test('a confirming read that never settles is retired by the leash, so the poll recovers', async () => {
    const host = makeHost();
    // Real timers. 250ms is the list's own floor — a smaller interval is
    // silently clamped up to it — and the leash deliberately outlives a tick,
    // so the hung read is demonstrably JOINED by a poll before the leash runs
    // out. With a leash inside one tick nothing polls during the gesture and
    // the coalescing assertion below would hold for want of a poll.
    const { el } = mountPane({
        host,
        extra: { holdStatusIntervalMs: 250, fleetHoldConfirmTimeoutMs: 400 },
    });
    await settle();
    assert.equal(countEl(el).dataset.holdState, 'none');
    assert.equal(kebab(el).disabled, false, 'the fleet menu starts live');

    // Accepted and never answered — a hung connection, a proxy holding the
    // socket. Only reads that actually START are affected, so a poll that joins
    // the hung one is invisible here, which is the point.
    const live = host.getHostHoldState.bind(host);
    let started = 0;
    let hang = true;
    host.getHostHoldState = async (...args) => {
        started += 1;
        if (hang) return new Promise(() => {});
        return live(...args);
    };

    await chooseFleetAction(el, 'hold-all');
    await waitFor(() => started >= 1, 'the confirming read to leave');
    const startedWhileHung = started;
    // The host is reachable again from here; nothing may reach it while the
    // wedge stands, so flipping this now cannot rescue the assertion below.
    hang = false;
    host.failRead = true;
    // A membership only the host can state, so the count below cannot have been
    // composed by this document from anything it already had.
    host.agents = [EMMA, NELLIE, KITE, WREN];

    const results = el.querySelector('.agent-fleet-hold-results');
    await waitFor(() => results.dataset.fanout === 'unconfirmed',
        'the gesture to give up waiting for its confirming read');
    assert.equal(countEl(el).textContent, 'All 3 held',
        'the committed latch still paints, over the pre-mutation membership');
    assert.equal(countEl(el).dataset.holdStale, undefined,
        'and that composition presents itself as a current reading — which is '
        + 'exactly why it must not be the last word the surface ever has');
    assert.equal(started, startedWhileHung,
        'every poll across the gesture joined the hung read rather than issuing');

    // The recovery. A read the poll issues of its own accord is the only thing
    // that can produce either of the two states below.
    await waitFor(() => countEl(el).dataset.holdStale === 'true',
        'the latch poll to issue a NEW read once the leash retired the hung one');
    assert.ok(started > startedWhileHung, 'which means a request actually left');
    assert.equal(kebab(el).disabled, true,
        'a read that failed withdraws Hold authority, as it always did');

    host.failRead = false;
    await waitFor(() => countEl(el).textContent === 'All 4 held',
        'the poll to converge onto the host\'s own tally');
    assert.equal(countEl(el).dataset.holdStale, undefined, 'confirmed again');
    assert.equal(countEl(el).dataset.targetCount, '4',
        'read from the host, not recomposed from the membership it had before');
    assert.equal(kebab(el).disabled, false, 'and the authority came back with it');
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

// ---------------------------------------------------------------------------
// Lifecycle: a fleet gesture outlives the mount that started it
//
// "Stop all and hold" is three awaits deep — mutate, confirm, stop — and a host
// may remount the pane across any of them. The gesture is therefore fenced on
// the CONTAINER, not in the mount's closure, so that a replacement mount
// inherits the fence and the retired continuation can discover it no longer
// owns the gesture. What must hold across every remount point:
//
//   - a retired mount renders nothing and publishes nothing, because its chrome
//     is detached and its embedding host cannot tell which mount is speaking;
//   - the replacement's controls stay fenced while the gesture is in flight,
//     and are released when it ends — a fence nobody clears is a dead control;
//   - and the Stop half never fires from a retired continuation, so exactly one
//     Stop can reach the host for one operator gesture.
// ---------------------------------------------------------------------------

// Remount into the same container, the way a host does: `mountAgentListPane`
// retires the prior owner itself (#3155), so this is one call, not two.
function remount(pane, host) {
    const handle = mountAgentListPane(pane, {
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
        onPrepareStopAll: () => () => {},
        holdStatusIntervalMs: 1e7,
        stopAllStatusIntervalMs: 1e7,
    });
    mounted.push(handle);
    return handle;
}

// Hold the named host call open, and hand back the key. Every lifecycle test
// below needs the remount to land while exactly one await is outstanding.
function gate(host, method) {
    let release;
    const opened = new Promise((resolve) => { release = resolve; });
    const live = host[method].bind(host);
    let entered = null;
    const arrived = new Promise((resolve) => { entered = resolve; });
    // `calls.*` on the double is pushed by `live`, which a gated call has not
    // reached yet — so a test asking "how many requests were STARTED" has to
    // count arrivals here, not completions there.
    const state = { entries: 0 };
    host[method] = async (...args) => {
        state.entries += 1;
        entered();
        await opened;
        return live(...args);
    };
    return {
        release,
        arrived,
        state,
        restore: () => { host[method] = live; },
    };
}

test('a pane remounted mid-Hold neither paints nor publishes from the retired mount', async () => {
    const host = makeHost({ inFlight: 2 });
    const pane = makeConsolePane();
    const published = [];
    const { handle: first } = mountPane({
        host,
        container: pane,
        extra: { onHoldState: (snapshot) => published.push(snapshot) },
    });
    await settle();
    const retiredResults = pane.querySelector('.agent-fleet-hold-results');

    const held = gate(host, 'setHostHold');
    const gesture = chooseFleetAction(pane, 'stop-all-and-hold');
    await held.arrived;

    const publishedBefore = published.length;
    remount(pane, host);
    await settle();

    held.release();
    await gesture;
    await settle();

    assert.equal(published.length, publishedBefore,
        'a destroyed mount must not keep telling its embedding host about fleet state');
    assert.equal(retiredResults.parentNode, null,
        'the retired results node left with its mount');
    assert.equal(retiredResults.hidden, true,
        'and the retired continuation never rendered into it');
    assert.equal(host.calls.stop.length, 0,
        'the Stop half belongs to the gesture the operator was watching, and that '
        + 'pane is gone; a replacement pane never asked for it');
    // The Hold itself DID commit, and that is the point of a latch: it is
    // durable, so the replacement reads it back rather than inheriting a claim.
    assert.equal(host.calls.set.length, 1);
    assert.equal(countEl(pane).dataset.holdState, 'all',
        'the replacement mount shows the fleet as held, read from the host');
});

test('a replacement mount inherits the fence, and is released when the gesture ends', async () => {
    const host = makeHost({ inFlight: 2 });
    const pane = makeConsolePane();
    mountPane({ host, container: pane });
    await settle();
    assert.equal(kebab(pane).disabled, false, 'the fleet menu starts live');

    const held = gate(host, 'setHostHold');
    const gesture = chooseFleetAction(pane, 'stop-all-and-hold');
    await held.arrived;

    remount(pane, host);
    await settle();
    assert.equal(kebab(pane).disabled, true,
        'the replacement must not offer a second fleet gesture over the first');
    assert.deepEqual(menuActionsBehindTheFence(pane), [],
        'and its menu composes nothing while the inherited operation is in flight');

    held.release();
    await gesture;
    await settle();

    assert.equal(kebab(pane).disabled, false,
        'a fence the retired mount never cleared would disable this control for '
        + 'the life of the page, with no gesture left to justify it');
    assert.ok(menuActions(pane).length > 0, 'and the menu works again');
    closeKebabMenu();
});

test('a pane remounted during the CONFIRMING read publishes no fan-out, and its already-started Stop lands once', async () => {
    const host = makeHost({ inFlight: 2 });
    const pane = makeConsolePane();
    mountPane({ host, container: pane });
    await settle();
    const retiredResults = pane.querySelector('.agent-fleet-hold-results');

    // Past the mutation, inside the read that would have become the fan-out.
    // The latch is committed by now, so this is the window where a retired
    // continuation is most tempted to "just finish the job".
    //
    // The Stop half is DELIBERATELY already on the wire here: it is authorised
    // by the committed Hold and started in the same turn, precisely so a host
    // that answers the POST and then never answers this GET cannot leave the
    // fleet held but never stopped. Retiring the pane after that cannot un-send
    // a request, and must not send a second one — so what a remount here owes
    // is silence about the fan-out, not the un-stopping of a stopped fleet.
    const read = gate(host, 'getHostHoldState');
    const gesture = chooseFleetAction(pane, 'stop-all-and-hold');
    await read.arrived;
    assert.equal(host.calls.set.length, 1, 'the mutation already committed');
    assert.equal(host.calls.stop.length, 1,
        'and the Stop it authorised went with it, rather than queueing behind '
        + 'a read that exists only to draw a receipt');

    remount(pane, host);
    await settle();

    read.restore();
    read.release();
    await gesture;
    await settle();

    assert.equal(retiredResults.hidden, true,
        'a fan-out nobody can see is not a receipt');
    assert.equal(retiredResults.querySelectorAll('li').length, 0);
    assert.equal(host.calls.stop.length, 1,
        'and the retired continuation adds no second Stop on its way out');
    assert.deepEqual(host.calls.order.filter((c) => c !== 'release'), ['hold', 'stop'],
        'one gesture, in the order it was asked for');
    assert.equal(kebab(pane).disabled, false, 'the replacement is unfenced when it ends');
    assert.equal(pane.querySelector('.agent-stop-all-btn').disabled, false,
        'and so is the Stop lane the gesture reserved');
});

test('a pane remounted during a host release publishes nothing from the retired mount', async () => {
    const host = makeHost({ hostHold: latch({ scope: 'host', target: 'host', receipt: 'receipt-host-1' }) });
    const pane = makeConsolePane();
    const published = [];
    mountPane({
        host,
        container: pane,
        extra: { onHoldState: (snapshot) => published.push(snapshot) },
    });
    await settle();
    const retiredResults = pane.querySelector('.agent-fleet-hold-results');

    const release = gate(host, 'releaseHostHold');
    const gesture = chooseFleetAction(pane, 'resume-host-hold');
    await release.arrived;

    const publishedBefore = published.length;
    remount(pane, host);
    await settle();
    const readsBefore = host.calls.read;

    release.release();
    await gesture;
    await settle();

    assert.equal(published.length, publishedBefore,
        'the retired mount stays silent about a release it can no longer show');
    assert.equal(retiredResults.hidden, true);
    assert.equal(host.calls.release.length, 1, 'the release itself committed exactly once');
    assert.equal(kebab(pane).disabled, false, 'and the replacement is unfenced');
    assert.equal(countEl(pane).hidden, true,
        'the replacement reads the released state from the host');
    // Exactly one: the convergence the retired mount owes the CURRENT owner.
    // A second would be the retired mount confirming a fan-out for itself —
    // going back to the host for evidence it has nowhere to render.
    assert.equal(host.calls.read - readsBefore, 1,
        'a retired mount converges the new owner and asks nothing for itself');
});

test('a pane remounted during a RELEASE\'s confirming read renders no fan-out', async () => {
    const host = makeHost({ hostHold: latch({ scope: 'host', target: 'host', receipt: 'receipt-host-1' }) });
    const pane = makeConsolePane();
    mountPane({ host, container: pane });
    await settle();
    const retiredResults = pane.querySelector('.agent-fleet-hold-results');

    // Past the release mutation, inside the read that says WHICH agents the
    // resume left held — the one thing this panel exists to show, and the
    // window where ownership is lost after the mount was entitled to ask.
    const read = gate(host, 'getHostHoldState');
    const gesture = chooseFleetAction(pane, 'resume-host-hold');
    await read.arrived;
    assert.equal(host.calls.release.length, 1, 'the release already committed');

    remount(pane, host);
    await settle();

    read.restore();
    read.release();
    await gesture;
    await settle();

    assert.equal(retiredResults.hidden, true,
        'the projection arrived for a pane that can no longer show it');
    assert.equal(retiredResults.querySelectorAll('li').length, 0);
    assert.equal(kebab(pane).disabled, false, 'and the replacement is unfenced when it ends');
});

test('the fence is released by the gesture ending, not by a host read landing', async () => {
    const host = makeHost({ inFlight: 2 });
    const pane = makeConsolePane();
    mountPane({ host, container: pane });
    await settle();

    const held = gate(host, 'setHostHold');
    const gesture = chooseFleetAction(pane, 'stop-all-and-hold');
    await held.arrived;

    remount(pane, host);
    await settle();
    assert.equal(kebab(pane).disabled, true, 'the replacement inherited the fence');

    // From here the host stops answering reads. The convergence refresh the
    // retired mount owes the new owner can no longer land, so the ONLY thing
    // that can give the replacement its menu back is the direct handoff when
    // the operation clears. Leaning on the refresh instead would leave a
    // console whose fleet menu is dead until the network recovers — and the
    // fence is local state, so there is nothing to ask the host about.
    host.getHostHoldState = () => new Promise(() => {});

    held.release();
    await gesture;
    await settle();

    assert.equal(kebab(pane).disabled, false,
        'the operation is over, so the control it fenced is live again');
    assert.ok(menuActions(pane).length > 0);
    closeKebabMenu();
});

test('a destroyed pane publishes nothing, even when its handle is asked to refresh', async () => {
    const host = makeHost();
    const published = [];
    const { handle } = mountPane({
        host,
        extra: { onHoldState: (snapshot) => published.push(snapshot) },
    });
    await settle();
    assert.ok(published.length > 0, 'a live pane does publish');

    handle.destroy();
    mounted.pop();
    const publishedBefore = published.length;

    // The handle survives its pane, and a host holding one has no way to know
    // the mount behind it is gone. A read it starts here resolves into a mount
    // whose chrome is detached — it must not be reported to the embedding host
    // as this pane's fleet state.
    await handle.refreshHoldState({ fresh: true });
    await settle();

    assert.equal(published.length, publishedBefore,
        'a destroyed pane is inert, not merely invisible');
});

test('a second click on a fleet action starts nothing, and asks nothing', async () => {
    const host = makeHost({ inFlight: 2 });
    const pane = makeConsolePane();
    const asked = [];
    mountPane({
        host,
        container: pane,
        extra: { askHoldReason: (message) => { asked.push(message); return 'operator reason'; } },
    });
    await settle();

    // `kebab_menu` closes the menu and THEN calls onSelect, leaving its click
    // listener bound to the now-detached node — so a rapid second click (or a
    // double-click) re-enters the same action while the first is still in
    // flight. The fence has to refuse that before the reason prompt, not after:
    // prompting an operator and then discarding the answer is its own defect.
    const held = gate(host, 'setHostHold');
    const items = openFleetMenu(pane);
    const compound = items.find((item) => item.dataset.action === 'stop-all-and-hold');
    assert.ok(compound, 'the compound gesture is the one with a second half to duplicate');
    compound.click();
    await held.arrived;
    assert.equal(asked.length, 1, 'the first click asked for a reason');

    compound.click();
    await settle();

    assert.equal(asked.length, 1,
        'the second click is refused by the fence, not by discarding an answer');
    assert.equal(held.state.entries, 1, 'and only one Hold request was ever started');

    held.release();
    await settle();
    await settle();
    assert.equal(host.calls.set.length, 1, 'exactly one Hold reached the host');
    assert.equal(host.calls.stop.length, 1, 'one gesture, one Stop');
    closeKebabMenu();
});

test('exactly one Stop reaches the host when the pane survives the whole gesture', async () => {
    const host = makeHost({ inFlight: 2 });
    const pane = makeConsolePane();
    mountPane({ host, container: pane });
    await settle();

    // The control case for the three remount tests above: with nothing retired,
    // the compound gesture must still perform its Stop — otherwise those three
    // would pass against a component that had simply stopped stopping.
    await chooseFleetAction(pane, 'stop-all-and-hold');
    await settle();

    assert.equal(host.calls.set.length, 1);
    assert.equal(host.calls.stop.length, 1, 'one gesture, one Stop');
    assert.deepEqual(host.calls.order.filter((c) => c !== 'release'), ['hold', 'stop']);
});

test('the fence outlives the Hold half, because the Stop half is the same gesture', async () => {
    const host = makeHost({ inFlight: 2 });
    const pane = makeConsolePane();
    mountPane({ host, container: pane });
    await settle();

    // Past the Hold and its confirming read, INSIDE the Stop request. The latch
    // is committed by now, so the menu's live offer here is Resume — and taking
    // it mid-gesture lands the fleet stopped and unheld, free to start again on
    // the next heartbeat. That is the one outcome "Stop all and hold" exists to
    // prevent, so the fence may not end when the Hold half does.
    const stopped = gate(host, 'stopHost');
    const gesture = chooseFleetAction(pane, 'stop-all-and-hold');
    await stopped.arrived;
    assert.equal(host.calls.set.length, 1, 'the Hold half already committed');

    assert.equal(kebab(pane).disabled, true,
        'the gesture is not over until its Stop is');
    assert.deepEqual(menuActionsBehindTheFence(pane), [],
        'so no Resume is on offer while the Stop request is still open');

    stopped.release();
    await gesture;
    await settle();

    assert.equal(host.calls.release.length, 0, 'and none was made');
    assert.equal(host.calls.stop.length, 1, 'one gesture, one Stop');
    assert.equal(kebab(pane).disabled, false, 'the fence ends with the gesture, not before it');
    assert.deepEqual(menuActions(pane), ['resume-host-hold'],
        'and the held fleet is resumable once the gesture has actually finished');
    closeKebabMenu();
});

test('the fence ends with the Stop, not with the bookkeeping read that follows it', async () => {
    const host = makeHost({ inFlight: 2 });
    const pane = makeConsolePane();
    mountPane({ host, container: pane });
    await settle();

    // The Stop lands; the status read AFTER it never does. That read is
    // convergence bookkeeping — the durable Stop is already over — so a fence
    // that waits for it hands a hung host the power to disable this console's
    // fleet menu for the life of the page, with the gesture finished and
    // nothing in flight to justify it. The fence is local state; ending it is
    // not the host's to answer.
    let releaseStatus;
    const hungStatus = new Promise((resolve) => { releaseStatus = resolve; });
    const liveStatus = host.getHostStopStatus.bind(host);
    const liveStop = host.stopHost.bind(host);
    host.stopHost = async (payload) => {
        const envelope = await liveStop(payload);
        host.getHostStopStatus = () => hungStatus.then(() => liveStatus());
        return envelope;
    };

    await chooseFleetAction(pane, 'stop-all-and-hold');
    await settle();

    assert.equal(host.calls.set.length, 1, 'the Hold half committed');
    assert.equal(host.calls.stop.length, 1, 'and so did the Stop half');
    assert.equal(kebab(pane).disabled, false,
        'the gesture is over the moment its Stop is');
    assert.deepEqual(menuActions(pane), ['resume-host-hold'],
        'and the held fleet is resumable again without waiting on a read the '
        + 'host may never answer');
    closeKebabMenu();

    // Let the trailing read land so this test leaves nothing pending behind it.
    releaseStatus();
    await settle();
});

test('a pane remounted during the STOP half inherits the fence, and cannot resume under it', async () => {
    const host = makeHost({ inFlight: 2 });
    const pane = makeConsolePane();
    mountPane({ host, container: pane });
    await settle();

    const stopped = gate(host, 'stopHost');
    const gesture = chooseFleetAction(pane, 'stop-all-and-hold');
    await stopped.arrived;

    remount(pane, host);
    await settle();

    // The replacement read the committed host latch back, so Resume is what its
    // menu would offer — this is the mount a fence scoped to the Hold half
    // hands the release to, with the Stop request still open.
    assert.equal(kebab(pane).disabled, true,
        'the replacement inherited the whole gesture, Stop half included');
    assert.deepEqual(menuActionsBehindTheFence(pane), [],
        'and the gate behind its button refuses the Resume too');

    stopped.release();
    await gesture;
    await settle();

    assert.equal(host.calls.release.length, 0, 'no Resume ran underneath the gesture');
    assert.equal(host.calls.stop.length, 1, 'and the Stop the operator asked for did');
    assert.equal(kebab(pane).disabled, false);
    assert.deepEqual(menuActions(pane), ['resume-host-hold'],
        'the replacement gets its fleet menu back when the gesture ends');
    closeKebabMenu();
});

// ---------------------------------------------------------------------------
// One gesture, two lanes, one reservation.
//
// "Stop all and hold" is ONE operator action whose halves live in two different
// lanes — the fleet Hold lane and the Stop All lane. Guarding each lane with its
// own token leaves a gap between the two claims: a Stop All started while the
// Hold half is awaiting takes the second lane, and the compound gesture's Stop
// then either declines silently (a gesture that reports a Hold and never stops)
// or runs a second Stop behind the competitor off a stale in-flight count.
//
// So the gesture reserves BOTH lanes before its first request leaves, and
// releases both exactly once when it settles. These pin the consequences.
// ---------------------------------------------------------------------------

test('the compound gesture reserves the Stop lane before its Hold leaves, so no Stop can race it', async () => {
    const host = makeHost({ inFlight: 2 });
    const pane = makeConsolePane();
    mountPane({ host, container: pane });
    await settle();
    const stopAllBtn = pane.querySelector('.agent-stop-all-btn');
    assert.equal(stopAllBtn.disabled, false, 'there is work to stop');

    // Inside the Hold half — the window where the Stop lane used to be free,
    // and where a click on the still-live Stop All button took it.
    const held = gate(host, 'setHostHold');
    const gesture = chooseFleetAction(pane, 'stop-all-and-hold');
    await held.arrived;

    assert.equal(stopAllBtn.disabled, true,
        'the lane this gesture reserved reads as busy — a fence the operator '
        + 'cannot see is one they walk straight into');
    assert.match(stopAllBtn.title, /Stop all and hold/,
        'and says WHICH gesture owns it, not merely that it is unavailable');

    // Force the click past the attribute: `disabled` is one door, and the
    // reservation behind it is the one that has to hold.
    stopAllBtn.disabled = false;
    stopAllBtn.click();
    await settle();
    assert.equal(host.calls.stop.length, 0,
        'a competing Stop All cannot start under the reservation');

    held.release();
    await gesture;
    await settle();

    assert.equal(host.calls.set.length, 1);
    assert.equal(host.calls.stop.length, 1, 'one gesture, one Stop');
    assert.equal(host.calls.stop[0].reason, 'Stopped from the agents banner');
    assert.deepEqual(host.calls.order.filter((c) => c !== 'release'), ['hold', 'stop'],
        'Hold first, then Stop, exactly once each');
    assert.equal(stopAllBtn.disabled, false,
        'and the reservation released BOTH lanes when the gesture settled');
});

test('a Stop All started while the menu sat open refuses the compound gesture rather than half-doing it', async () => {
    const host = makeHost({ inFlight: 2 });
    const pane = makeConsolePane();
    const asked = [];
    mountPane({
        host,
        container: pane,
        extra: { askHoldReason: (message) => { asked.push(message); return 'operator reason'; } },
    });
    await settle();

    // A menu is composed when it OPENS. Taking the Stop lane afterwards leaves
    // a compound entry on screen that can no longer be honoured — the reachable
    // half of the race the reservation closes.
    const items = openFleetMenu(pane);
    const compound = items.find((item) => item.dataset.action === 'stop-all-and-hold');
    assert.ok(compound, 'the entry was offered while the lane was free');

    const stopped = gate(host, 'stopHost');
    pane.querySelector('.agent-stop-all-btn').click();
    await stopped.arrived;

    compound.click();
    await settle();

    assert.equal(asked.length, 0,
        'refused before the prompt: asking an operator for a reason and then '
        + 'discarding the answer is its own defect');
    assert.equal(host.calls.set.length, 0,
        'and no Hold went out, because a bare Hold is not what was asked for');

    stopped.release();
    await settle();
    assert.equal(host.calls.stop.length, 1, 'the Stop All the operator did start ran once');
});

test('a replacement mount inherits BOTH reserved lanes, not just the fleet menu', async () => {
    const host = makeHost({ inFlight: 2 });
    const pane = makeConsolePane();
    mountPane({ host, container: pane });
    await settle();

    const held = gate(host, 'setHostHold');
    const gesture = chooseFleetAction(pane, 'stop-all-and-hold');
    await held.arrived;

    remount(pane, host);
    await settle();

    assert.equal(kebab(pane).disabled, true, 'the replacement inherited the fleet lane');
    const replacementStopAll = pane.querySelector('.agent-stop-all-btn');
    assert.equal(replacementStopAll.disabled, true,
        'and the Stop lane with it — a replacement that came up unfenced on one '
        + 'lane is exactly how a second Stop got started');
    replacementStopAll.disabled = false;
    replacementStopAll.click();
    await settle();
    assert.equal(host.calls.stop.length, 0,
        'and the gate behind the button refuses it too');

    held.release();
    await gesture;
    await settle();

    assert.equal(kebab(pane).disabled, false, 'both lanes are released together');
    assert.equal(pane.querySelector('.agent-stop-all-btn').disabled, false,
        'and the replacement gets a FRESH reading of the lane it inherited fenced '
        + '— status reads are suppressed while an operation holds it, so a mount '
        + 'that arrived during the reservation has never read it, and a repaint '
        + 'alone would leave its Stop All dead until the next poll');
});

test('a release reserves only its own lane, so a Stop All still runs beside it', async () => {
    const host = makeHost({
        inFlight: 2,
        hostHold: latch({ scope: 'host', target: 'host', receipt: 'receipt-host-1' }),
    });
    const pane = makeConsolePane();
    mountPane({ host, container: pane });
    await settle();

    // Resume stops nothing, so it has no business reserving the Stop lane: a
    // reservation wider than the gesture is a fence with nothing behind it.
    const release = gate(host, 'releaseHostHold');
    const gesture = chooseFleetAction(pane, 'resume-host-hold');
    await release.arrived;

    const stopAllBtn = pane.querySelector('.agent-stop-all-btn');
    assert.equal(stopAllBtn.disabled, false, 'the Stop lane is untouched by a release');
    stopAllBtn.click();
    await settle();
    assert.equal(host.calls.stop.length, 1, 'and a Stop All runs beside it');

    release.release();
    await gesture;
    await settle();
    assert.equal(host.calls.release.length, 1);
});

test('"Stop all and hold" is not offered while a Stop All is already running', async () => {
    const host = makeHost({ inFlight: 2 });
    const pane = makeConsolePane();
    mountPane({ host, container: pane });
    await settle();

    const stopAllBtn = pane.querySelector('.agent-stop-all-btn');
    assert.equal(stopAllBtn.disabled, false, 'there is work to stop');
    const stopped = gate(host, 'stopHost');
    stopAllBtn.click();
    await stopped.arrived;

    // `runStopAll` admits one operation per container, so the compound
    // gesture's second half would early-return against this one and the panel
    // would report a Stop this gesture never made. The half that remains IS the
    // plain Hold above it, so offering only that takes nothing away.
    assert.deepEqual(menuActions(pane), ['hold-all']);
    closeKebabMenu();

    stopped.release();
    await settle();

    assert.equal(host.calls.stop.length, 1);
    assert.deepEqual(menuActions(pane), ['hold-all', 'stop-all-and-hold'],
        'and the compound gesture comes back when the Stop All is over');
    closeKebabMenu();
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
