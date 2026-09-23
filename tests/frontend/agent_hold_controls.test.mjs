// #3164: Hold and Resume live on the agent CARD, not on the busy-only Stop
// button. These tests pin the six acceptance conditions: an IDLE agent can be
// held; a held card stays legible with nothing in flight and across a remount;
// "Stop and hold" performs two typed operations and keeps both receipts apart;
// Resume releases only the latch the card owns; the context menu is an
// accelerator beside a visible focusable control; and the component — not the
// console — owns all of it, so an embedding host inherits it.

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
// Explicit, not ambient: Node 25+ exposes global Web Storage, Node 22 (CI) does not.
globalThis.sessionStorage = dom.window.sessionStorage;
globalThis.localStorage = dom.window.localStorage;
globalThis.window.kicon = (name) => `<span class="ki ki-${name}" aria-hidden="true"></span>`;
globalThis.kicon = globalThis.window.kicon;

const { mountAgentList, mountAgentListPane } = await import(
    '../../kestrel_sovereign/static/js/agent_list.js'
);
const { closeKebabMenu, createKebabButton } = await import(
    '../../kestrel_sovereign/static/js/kebab_menu.js'
);
const { createApiClient } = await import('../../kestrel_sovereign/static/js/api_client.mjs');

function tick() { return new Promise((r) => setTimeout(r, 0)); }

// The REAL console API client over a scripted host, so the component's
// adoption check is exercised against the client every embedding host
// actually passes it rather than a hand-written stand-in.
function makeRealClient(route) {
    return createApiClient({
        fetchFn: async (url, opts) => {
            const reply = route(url, opts) || { ok: true, status: 200 };
            const payload = reply.body === undefined ? {} : reply.body;
            return {
                ok: reply.ok !== false,
                status: reply.status || 200,
                json: async () => payload,
                text: async () => JSON.stringify(payload),
                headers: { get: () => null },
            };
        },
        sessionStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
        location: { href: '/', search: '' },
        AbortControllerCtor: globalThis.AbortController,
        TextDecoderCtor: globalThis.TextDecoder,
        authProvider: {
            async ensureAuthenticated() {},
            applyAuth: async (h) => ({ ...h, 'X-API-Key': 'k' }),
            async onUnauthorized() { return 'failed'; },
        },
    });
}

const EMMA = 'did:agent:emma';

function agentItem(overrides = {}) {
    return {
        name: 'EmmaRoute',
        displayName: 'Emma',
        id: EMMA,
        status: 'online',
        ...overrides,
    };
}

function latch({ scope = 'agent', target = EMMA, receipt = 'receipt-1', reason = 'runaway loop', actor = 'sovereign-key', at = '2026-09-15T10:00:00+00:00' } = {}) {
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

// A stand-in for the host Hold door. `state` is the authoritative reply; every
// mutation is recorded verbatim so a test can assert the exact wire request.
function holdApi({ canHold = true, hostHold = null, agentHold = null, failRead = false } = {}) {
    const calls = { read: 0, set: [], release: [] };
    const api = {
        setHostAgent() {},
        calls,
        hostHold,
        agentHold,
        failRead,
        canHold,
        setHostHoldResult: null,
        releaseHostHoldResult: null,
        async getHostHoldState() {
            calls.read += 1;
            if (api.failRead) throw new Error('host unreachable');
            const sources = [];
            if (api.hostHold) sources.push('host');
            if (api.agentHold) sources.push('agent');
            return {
                can_hold: api.canHold,
                host_hold: api.hostHold,
                agents: [{
                    agent_id: EMMA,
                    held: sources.length > 0,
                    sources,
                    agent_hold: api.agentHold,
                }],
            };
        },
        async setHostHold(payload) {
            calls.set.push(payload);
            if (api.setHostHoldResult instanceof Error) throw api.setHostHoldResult;
            if (api.setHostHoldResult) return api.setHostHoldResult;
            api.agentHold = latch({ receipt: 'receipt-new', reason: payload.reason });
            return {
                receipt: {
                    receipt_id: 'receipt-new',
                    operation_id: payload.operation_id,
                    action: 'hold',
                    disposition: 'applied',
                },
                current: api.agentHold,
            };
        },
        async releaseHostHold(payload) {
            calls.release.push(payload);
            if (api.releaseHostHoldResult) return api.releaseHostHoldResult;
            api.agentHold = null;
            return {
                receipt: {
                    receipt_id: 'receipt-release',
                    operation_id: payload.operation_id,
                    action: 'release',
                    disposition: 'applied',
                },
                current: null,
            };
        },
    };
    return api;
}

// Count the intervals the component opens against the window it mounted into,
// so "polling stopped" can be asserted as a released timer rather than as a
// request that happened not to be sent.
function watchTimers() {
    const realSet = dom.window.setInterval;
    const realClear = dom.window.clearInterval;
    const live = new Set();
    dom.window.setInterval = function patchedSetInterval(...args) {
        const id = realSet.apply(this, args);
        live.add(id);
        return id;
    };
    dom.window.clearInterval = function patchedClearInterval(id) {
        live.delete(id);
        return realClear.call(this, id);
    };
    return {
        outstanding: () => Array.from(live),
        restore() {
            for (const id of live) realClear.call(dom.window, id);
            live.clear();
            dom.window.setInterval = realSet;
            dom.window.clearInterval = realClear;
        },
    };
}

// A mount holds a latch-poll interval, so a test that throws before its own
// destroy() would leave the event loop open and turn a FAILING suite into a
// HANGING one — which reports nothing. Every mount is torn down here.
const mounted = [];

afterEach(() => {
    closeKebabMenu();
    for (const handle of mounted.splice(0)) {
        try { handle.destroy(); } catch (_) { /* best-effort teardown */ }
    }
});

function mountInto(config, items = [agentItem()]) {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const handle = mountAgentList(el, {
        adapter: { mode: 'multi_agent', listAgents: async () => items },
        // No interval churn in tests: every refresh below is explicit.
        holdStatusIntervalMs: 1e7,
        askHoldReason: () => 'operator reason',
        ...config,
    });
    mounted.push(handle);
    return { el, handle };
}

// `contextmenu` is cancelable in a browser, so dispatch it that way: whether
// the handler took the gesture from the browser is only observable on an event
// that could actually be cancelled.
function rightClick(row) {
    const event = new dom.window.MouseEvent('contextmenu', {
        bubbles: true, cancelable: true, clientX: 5, clientY: 5,
    });
    row.dispatchEvent(event);
    return event;
}

function openContextMenu(row) {
    rightClick(row);
    return document.querySelector('.kebab-menu');
}

function menuLabels() {
    return Array.from(document.querySelectorAll('.kebab-menu-item')).map((b) => b.textContent);
}

function clickMenuAction(action) {
    const btn = document.querySelector(`.kebab-menu-item[data-action="${action}"]`);
    assert.ok(btn, `menu offers "${action}"`);
    btn.click();
}

test('an IDLE agent can be held from the card, and the latch is keyed by its DID', async () => {
    const api = holdApi();
    const { el, handle } = mountInto({
        api,
        isThinking: () => false, // the case Stop's button cannot serve at all
        askHoldReason: () => 'runaway loop',
    });
    await tick();

    const row = el.querySelector('.agent-card');
    // The row is NOT thinking, which is the whole point: `.agent-stop-btn` is
    // CSS-hidden until `.agent-thinking`, so a gesture on Stop would be
    // unreachable exactly in Hold's primary case.
    assert.ok(!row.classList.contains('agent-thinking'));

    const kebab = el.querySelector('.agent-card-kebab');
    assert.ok(kebab, 'the card carries a visible kebab control');
    assert.equal(kebab.disabled, false, 'a loaded inventory enables it');
    kebab.click();
    assert.deepEqual(menuLabels(), ['Hold Emma…'], 'an idle card offers Hold and nothing else');
    clickMenuAction('hold');
    await tick();
    await tick();

    assert.equal(api.calls.set.length, 1);
    const [request] = api.calls.set;
    assert.equal(request.scope, 'agent');
    assert.equal(request.target_id, EMMA, 'the latch targets the DID, never the display name');
    assert.equal(request.reason, 'runaway loop');
    assert.ok(request.operation_id, 'the mutation carries an idempotency key');

    assert.ok(row.classList.contains('agent-held'));
    assert.equal(el.querySelector('.agent-hold-outcome').dataset.disposition, 'applied');
    closeKebabMenu();
    handle.destroy();
});

test('a held card stays legible with nothing in flight, and again after a remount', async () => {
    const api = holdApi({ agentHold: latch({ reason: 'paused for audit', actor: 'ops@example' }) });
    const first = mountInto({ api, isThinking: () => false });
    await tick();
    await tick();

    const badge = first.el.querySelector('.agent-hold-badge');
    assert.equal(badge.hidden, false, 'held is a resting state, not an in-flight one');
    assert.equal(first.el.querySelector('.agent-hold-badge-reason').textContent, 'paused for audit');
    assert.equal(first.el.querySelector('.agent-hold-badge-actor').textContent, 'ops@example');
    assert.ok(first.el.querySelector('.agent-hold-badge-time').textContent,
        'the badge carries when it was set');
    assert.match(badge.title, /ops@example/);
    assert.match(badge.title, /paused for audit/);
    assert.equal(first.el.querySelector('.agent-resume-btn').hidden, false,
        'Resume occupies the Stop slot on a held card');
    assert.ok(!first.el.querySelector('.agent-card').classList.contains('agent-thinking'));
    first.handle.destroy();

    // "Across reload": a fresh mount reads the durable latch and paints it,
    // rather than depending on anything the previous page remembered.
    const second = mountInto({ api, isThinking: () => false });
    await tick();
    await tick();
    assert.equal(second.el.querySelector('.agent-hold-badge').hidden, false);
    assert.ok(second.el.querySelector('.agent-card').classList.contains('agent-held'));
    second.handle.destroy();
});

test('a list refresh repaints the hold badge instead of blanking it for a poll interval', async () => {
    const api = holdApi({ agentHold: latch() });
    const { el, handle } = mountInto({ api });
    await tick();
    await tick();
    assert.equal(el.querySelector('.agent-hold-badge').hidden, false);

    await handle.refresh();
    assert.equal(el.querySelector('.agent-hold-badge').hidden, false,
        'the rebuilt card is repainted from the latch already read');
    handle.destroy();
});

test('Stop and hold performs two typed operations and keeps the two receipts apart', async () => {
    const order = [];
    const api = holdApi();
    const originalSet = api.setHostHold;
    api.setHostHold = async (payload) => { order.push('hold'); return originalSet(payload); };
    const { el, handle } = mountInto({
        api,
        isThinking: () => true,
        onStop: async () => {
            order.push('stop');
            return { outcomes: [{ resolved_target: EMMA, disposition: 'stopped' }] };
        },
    });
    await tick();
    await tick();

    el.querySelector('.agent-card-kebab').click();
    assert.deepEqual(menuLabels(), ['Hold Emma…', 'Stop and hold…', 'Stop Emma'],
        'a busy card offers the compound action explicitly, never as a gesture on Stop');
    clickMenuAction('stop-and-hold');
    await tick();
    await tick();
    await tick();

    assert.deepEqual(order, ['hold', 'stop'],
        'the latch lands first, so no heartbeat can start a turn between the two');
    assert.equal(api.calls.set.length, 1, 'one Hold request');

    const holdOutcome = el.querySelector('.agent-hold-outcome');
    const stopOutcome = el.querySelector('.agent-stop-outcome');
    assert.equal(holdOutcome.dataset.disposition, 'applied');
    assert.equal(holdOutcome.textContent, 'Held');
    assert.equal(stopOutcome.dataset.disposition, 'stopped');
    assert.equal(stopOutcome.textContent, 'Stopped');
    assert.notEqual(holdOutcome, stopOutcome, 'two receipts, two typed surfaces');
    closeKebabMenu();
    handle.destroy();
});

test('a failed Hold does not silently degrade "stop and hold" into a bare Stop', async () => {
    const api = holdApi();
    api.setHostHoldResult = new Error('hold store unavailable');
    let stops = 0;
    const { el, handle } = mountInto({
        api,
        isThinking: () => true,
        onStop: async () => { stops += 1; return { outcomes: [] }; },
    });
    await tick();
    await tick();

    el.querySelector('.agent-card-kebab').click();
    clickMenuAction('stop-and-hold');
    await tick();
    await tick();

    assert.equal(stops, 0, 'the operator asked for a latch; a bare Stop is a different act');
    const outcome = el.querySelector('.agent-hold-outcome');
    assert.equal(outcome.dataset.disposition, 'unreachable');
    assert.match(outcome.title, /hold store unavailable/);
    closeKebabMenu();
    handle.destroy();
});

test('a Hold that answered without latching is indeterminate, and stops nothing', async () => {
    const api = holdApi();
    // A receipt that says "applied" while no latch came back contradicts
    // itself; the compound gesture must not treat that as a hold.
    api.setHostHoldResult = {
        receipt: { receipt_id: 'r', operation_id: 'op', action: 'hold', disposition: 'applied' },
        current: null,
    };
    let stops = 0;
    const { el, handle } = mountInto({
        api,
        isThinking: () => true,
        onStop: async () => { stops += 1; return { outcomes: [] }; },
    });
    await tick();
    await tick();

    el.querySelector('.agent-card-kebab').click();
    clickMenuAction('stop-and-hold');
    await tick();
    await tick();

    assert.equal(stops, 0, 'the Stop half waits on a latch that demonstrably landed');
    assert.equal(el.querySelector('.agent-hold-outcome').dataset.disposition, 'indeterminate');
    closeKebabMenu();
    handle.destroy();
});

test('Resume releases only the intended latch — this agent, this receipt', async () => {
    const api = holdApi({
        agentHold: latch({ receipt: 'agent-receipt-7' }),
        hostHold: latch({ scope: 'host', target: 'host', receipt: 'host-receipt-1', reason: 'fleet freeze' }),
    });
    const { el, handle } = mountInto({ api, askHoldReason: () => 'audit finished' });
    await tick();
    await tick();

    el.querySelector('.agent-resume-btn').click();
    await tick();
    await tick();

    assert.equal(api.calls.release.length, 1);
    const [request] = api.calls.release;
    assert.equal(request.scope, 'agent', 'never the host latch');
    assert.equal(request.target_id, EMMA);
    assert.equal(request.expected_hold_receipt_id, 'agent-receipt-7',
        'the compare-and-set names the receipt the operator saw, not the host latch');
    assert.equal(el.querySelector('.agent-hold-outcome').textContent, 'Resumed');
    // The host latch is untouched, so the card is still held — and says so.
    assert.equal(el.querySelector('.agent-hold-badge').hidden, false);
    assert.equal(el.querySelector('.agent-hold-badge-label').textContent, 'Held by host');
    handle.destroy();
});

test('a card held only by the host latch offers no Resume at all', async () => {
    const api = holdApi({
        hostHold: latch({ scope: 'host', target: 'host', receipt: 'host-receipt-1', reason: 'fleet freeze' }),
    });
    const { el, handle } = mountInto({ api });
    await tick();
    await tick();

    const row = el.querySelector('.agent-card');
    assert.ok(row.classList.contains('agent-held'));
    assert.equal(el.querySelector('.agent-hold-badge-reason').textContent, 'fleet freeze');
    assert.equal(el.querySelector('.agent-resume-btn').hidden, true,
        'releasing the fleet latch from an agent card would release a latch nobody aimed at');
    el.querySelector('.agent-card-kebab').click();
    assert.deepEqual(menuLabels(), ['Hold Emma…'],
        'an independent agent latch is still offered; a host release is not');
    el.querySelector('.agent-resume-btn').click();
    await tick();
    assert.equal(api.calls.release.length, 0);
    closeKebabMenu();
    handle.destroy();
});

// A Hold is committed by the POST, not by the GET that follows it. These two
// pin the split-request failure: the mutation's authoritative `current` is what
// the card renders, so a confirming read that never arrives cannot roll the
// card back to a pre-mutation reading while the outcome reads "Held".
test('a Hold that committed keeps the card held when the confirming read fails', async () => {
    const api = holdApi();
    const originalSet = api.setHostHold;
    api.setHostHold = async (payload) => {
        const response = await originalSet(payload);
        api.failRead = true; // the POST landed; the follow-up GET does not
        return response;
    };
    const { el, handle } = mountInto({ api, askHoldReason: () => 'runaway loop' });
    await tick();
    await tick();
    assert.equal(el.querySelector('.agent-hold-badge').hidden, true);

    el.querySelector('.agent-card-kebab').click();
    clickMenuAction('hold');
    await tick();
    await tick();
    await tick();

    assert.equal(api.calls.set.length, 1, 'the mutation was made');
    assert.equal(el.querySelector('.agent-hold-outcome').textContent, 'Held');
    const badge = el.querySelector('.agent-hold-badge');
    assert.equal(badge.hidden, false,
        'saying "Held" while showing no badge is the state Hold exists to prevent');
    assert.ok(el.querySelector('.agent-card').classList.contains('agent-held'));
    assert.equal(el.querySelector('.agent-hold-badge-reason').textContent, 'runaway loop',
        'and the badge carries the latch the mutation actually committed');
    assert.equal(badge.dataset.holdStale, 'true',
        'the reading behind it is unconfirmed, and says so');
    assert.equal(el.querySelector('.agent-card-kebab').disabled, true,
        'no further mutation until a read succeeds');
    closeKebabMenu();
    handle.destroy();
});

// The 5s latch poll and an operator's Hold are independent, so the read that
// was already in flight when the mutation committed carries a PRE-mutation
// reading of the very latch that just landed. Letting it land last would undo
// a committed fact with a stale one — the same "held card silently unheld"
// failure as the test above, reached by the other door.
test('a poll that started before a Hold cannot overwrite the latch the Hold committed', async () => {
    const api = holdApi();
    const { el, handle } = mountInto({ api, askHoldReason: () => 'runaway loop' });
    await tick();
    await tick();
    assert.equal(el.querySelector('.agent-hold-badge').hidden, true);

    // A poll reads "not held" — and then stalls before it can be applied.
    let releaseRead;
    const stalled = new Promise((resolve) => { releaseRead = resolve; });
    const liveRead = api.getHostHoldState;
    api.getHostHoldState = async () => {
        const payload = await liveRead.call(api); // captured BEFORE the mutation
        await stalled;
        return payload;
    };
    const inFlight = handle.refreshHoldState();
    await tick();
    assert.equal(api.calls.read >= 2, true, 'a read is outstanding');

    // A failed assertion below must not leave the stalled read — and the
    // mounted poll — holding the event loop open.
    try {
        // The operator holds while that read is still outstanding.
        el.querySelector('.agent-card-kebab').click();
        clickMenuAction('hold');
        await tick();
        await tick();
        assert.equal(api.calls.set.length, 1, 'the mutation was made');
        assert.equal(el.querySelector('.agent-hold-badge').hidden, false);

        // Now the stale reading lands, last.
        releaseRead();
        await inFlight;
        await tick();
        await tick();

        assert.equal(el.querySelector('.agent-hold-badge').hidden, false,
            'a read that predates the mutation must not undo the latch it committed');
        assert.ok(el.querySelector('.agent-card').classList.contains('agent-held'));
        assert.equal(el.querySelector('.agent-hold-badge-reason').textContent, 'runaway loop');
    } finally {
        releaseRead();
        closeKebabMenu();
        handle.destroy();
    }
});

test('a Resume that committed stops showing held when the confirming read fails', async () => {
    const api = holdApi({ agentHold: latch({ receipt: 'agent-receipt-7' }) });
    const originalRelease = api.releaseHostHold;
    api.releaseHostHold = async (payload) => {
        const response = await originalRelease(payload);
        api.failRead = true;
        return response;
    };
    const { el, handle } = mountInto({ api, askHoldReason: () => 'audit finished' });
    await tick();
    await tick();
    assert.equal(el.querySelector('.agent-hold-badge').hidden, false);

    el.querySelector('.agent-resume-btn').click();
    await tick();
    await tick();
    await tick();

    assert.equal(api.calls.release.length, 1);
    assert.equal(el.querySelector('.agent-hold-outcome').textContent, 'Resumed');
    assert.equal(el.querySelector('.agent-hold-badge').hidden, true,
        'the release committed, so the card must not keep drawing a latch that is gone');
    assert.ok(!el.querySelector('.agent-card').classList.contains('agent-held'));
    handle.destroy();
});

test('a stale Hold read keeps the badge but withdraws authority', async () => {
    const api = holdApi({ agentHold: latch() });
    const { el, handle } = mountInto({ api });
    await tick();
    await tick();
    assert.equal(el.querySelector('.agent-hold-badge').hidden, false);

    api.failRead = true;
    await handle.refreshHoldState();

    const badge = el.querySelector('.agent-hold-badge');
    assert.equal(badge.hidden, false, 'a blip must not erase the record of a hold');
    assert.equal(badge.dataset.holdStale, 'true');
    assert.match(badge.title, /last confirmed/);
    assert.equal(el.querySelector('.agent-card-kebab').disabled, true,
        'no mutation on an unconfirmed reading');
    el.querySelector('.agent-resume-btn').click();
    await tick();
    assert.equal(api.calls.release.length, 0);
    handle.destroy();
});

test('right-click is an accelerator onto the same menu, beside a focusable button', async () => {
    const api = holdApi();
    const { el, handle } = mountInto({ api });
    await tick();
    await tick();

    const kebab = el.querySelector('.agent-card-kebab');
    assert.equal(kebab.tagName, 'BUTTON');
    assert.equal(kebab.getAttribute('aria-haspopup'), 'menu');
    assert.equal(kebab.getAttribute('aria-label'), 'Actions for Emma');

    kebab.click();
    const fromButton = menuLabels();
    closeKebabMenu();

    const menu = openContextMenu(el.querySelector('.agent-card'));
    assert.ok(menu, 'the row contextmenu opens the menu too');
    assert.deepEqual(menuLabels(), fromButton, 'one menu, two ways in');
    closeKebabMenu();
    handle.destroy();
});

// An accelerator may only take the browser's own context menu when it has one
// of ours to put there. Suppressing the native menu on a card whose kebab is
// disabled leaves the right-click doing nothing at all — the opposite of an
// accelerator beside a visible path.
test('right-click keeps the browser menu when there is no menu of ours to open', async () => {
    const api = holdApi();
    const { el, handle } = mountInto({ api });
    await tick();
    await tick();

    const row = el.querySelector('.agent-card');
    const kebab = el.querySelector('.agent-card-kebab');
    assert.equal(kebab.disabled, false);
    const accelerated = rightClick(row);
    assert.ok(document.querySelector('.kebab-menu'), 'our menu opened');
    assert.equal(accelerated.defaultPrevented, true,
        'so the gesture is ours and the browser must not also answer it');
    closeKebabMenu();

    // The caller turns out to lack sovereign authority, so the kebab disables
    // and no menu of ours can open for this gesture.
    api.canHold = false;
    await handle.refreshHoldState();
    assert.equal(el.querySelector('.agent-card-kebab').disabled, true);

    const suppressed = rightClick(row);
    assert.equal(document.querySelector('.kebab-menu'), null,
        'the accelerator cannot bypass the disabled control');
    assert.equal(suppressed.defaultPrevented, false,
        'and with nothing of ours to show, the browser keeps its own menu');
    handle.destroy();
});

test('the latch is addressed by the host record, not by whatever matched the card', async () => {
    const api = holdApi({ agentHold: latch({ receipt: 'agent-3' }) });
    // The default adapter falls back to the ROUTING KEY for `id` when a payload
    // carries no `did`, so a card can legitimately be matched by a name. What
    // the mutation targets must still be the DID the host named.
    const { el, handle } = mountInto(
        { api, askHoldReason: () => 'why' },
        [{ name: 'EmmaRoute', displayName: 'Emma', id: 'EmmaRoute', status: 'online', raw: { did: EMMA } }],
    );
    await tick();
    await tick();

    el.querySelector('.agent-resume-btn').click();
    await tick();
    await tick();
    assert.equal(api.calls.release[0].target_id, EMMA, 'the DID, not the routing key');

    api.agentHold = null;
    await handle.refreshHoldState();
    el.querySelector('.agent-card-kebab').click();
    clickMenuAction('hold');
    await tick();
    await tick();
    assert.equal(api.calls.set[0].target_id, EMMA, 'the DID, not the routing key');
    closeKebabMenu();
    handle.destroy();
});

test('an unresolved or ambiguous card identity never invents a latch target', async () => {
    const api = holdApi();
    // The host's inventory does not know this card, so there is no DID to latch.
    const { el, handle } = mountInto({ api }, [agentItem({ id: 'did:agent:unknown' })]);
    await tick();
    await tick();

    const kebab = el.querySelector('.agent-card-kebab');
    assert.equal(kebab.disabled, true);
    assert.equal(kebab.title, 'Hold controls are unavailable');
    const event = rightClick(el.querySelector('.agent-card'));
    assert.equal(document.querySelector('.kebab-menu'), null,
        'the accelerator cannot bypass the disabled control');
    assert.equal(event.defaultPrevented, false,
        'and a card with no latch target keeps the browser context menu');
    handle.destroy();
});

test('a non-sovereign caller gets no Hold controls, and the pane inherits the surface', async () => {
    const api = holdApi({ canHold: false });
    const el = document.createElement('div');
    document.body.appendChild(el);
    const pane = mountAgentListPane(el, {
        api,
        adapter: { mode: 'multi_agent', listAgents: async () => [agentItem()] },
        holdStatusIntervalMs: 1e7,
    });
    mounted.push(pane);
    await tick();
    await tick();

    assert.ok(el.querySelector('.agent-card-kebab'), 'the pane mounts the component-owned surface');
    assert.equal(el.querySelector('.agent-card-kebab').disabled, true);
    assert.equal(typeof pane.refreshHoldState, 'function');
    pane.destroy();
});

// jsdom has no layout, so "Resume sits in the Stop slot" is a CSS fact, not a
// DOM one. Assert the two rules that carry it (the #2159 css-coverage gate's
// shape) — without them the component emits controls nothing ever shows.
test('the card CSS gives the hold surface its resting and Stop-slot behaviour', () => {
    const here = dirname(fileURLToPath(import.meta.url));
    const css = readFileSync(
        join(here, '..', '..', 'kestrel_sovereign', 'static', 'index.css'),
        'utf8',
    );
    const emitted = [
        'agent-card-kebab',
        'agent-held',
        'agent-hold-badge',
        'agent-hold-badge-label',
        'agent-hold-badge-reason',
        'agent-hold-badge-actor',
        'agent-hold-badge-time',
        'agent-hold-outcome',
        'agent-resume-btn',
    ];
    const missing = emitted.filter((cls) => !new RegExp(`\\.${cls}(?![\\w-])`).test(css));
    assert.deepEqual(missing, [], `index.css is missing styles for: ${missing.join(', ')}`);

    assert.match(css, /\.agent-card\.agent-held \.agent-resume-btn:not\(\[hidden\]\)\s*\{[^}]*display:\s*inline-flex/,
        'Resume becomes visible on a held card regardless of whether it is thinking');
    assert.match(css, /\.agent-card\.agent-held \.agent-stop-btn\s*\{[^}]*display:\s*none/,
        'a held card gives the Stop slot to Resume');
    assert.match(css, /\.agent-resume-btn\s*\{[^}]*display:\s*none/,
        'and hides it again when the card is not held');
});

test('a host whose API client lacks the Hold door renders exactly the old card', async () => {
    const { el, handle } = mountInto({ api: { setHostAgent() {} } });
    await tick();

    assert.equal(el.querySelector('.agent-card-kebab'), null);
    assert.equal(el.querySelector('.agent-hold-badge'), null);
    assert.equal(el.querySelector('.agent-resume-btn'), null);
    assert.ok(el.querySelector('.agent-stop-btn'), 'Stop is untouched');
    handle.destroy();
});

// The test above hand-writes an API object WITHOUT the three Hold methods, so
// on its own it would pass against a component that read method presence as
// proof of support. The real standard client always exposes all three whatever
// the backend mounts, so the adoption question has to be asked of THAT client
// against a host with no Hold route.
test('the standard API client exposes the Hold methods even where no Hold route is mounted', () => {
    const client = makeRealClient(() => ({ ok: true, status: 200 }));
    for (const method of ['getHostHoldState', 'setHostHold', 'releaseHostHold']) {
        assert.equal(typeof client[method], 'function',
            `${method} is unconditional on the client — presence cannot mean the door exists`);
    }
});

test('the real client against a host with no Hold route leaves the old card alone and stops polling', async () => {
    const requested = [];
    const client = makeRealClient((url) => {
        requested.push(String(url));
        return String(url).includes('/api/host/hold')
            ? { ok: false, status: 404, body: { detail: 'Not Found' } }
            : { ok: true, status: 200 };
    });

    // A short real interval, so "stopped polling" is observed rather than
    // inferred from a cadence too slow to have fired. The timers are watched
    // too: a retired surface must release its interval, not merely decline to
    // use it — an ignored timer left running forever is still a leak.
    const timers = watchTimers();
    let el;
    let handle;
    try {
        ({ el, handle } = mountInto({ api: client, holdStatusIntervalMs: 250 }));
        await tick();
        await tick();
        await tick();

        const holdReads = () => requested.filter((u) => u.includes('/api/host/hold')).length;
        assert.equal(holdReads(), 1, 'the component asked the host exactly once');
        assert.equal(el.querySelector('.agent-card-kebab'), null,
            'a 404 must leave no permanently disabled control behind');
        assert.equal(el.querySelector('.agent-hold-badge'), null);
        assert.equal(el.querySelector('.agent-resume-btn'), null);
        assert.ok(el.querySelector('.agent-stop-btn'), 'Stop is untouched');

        // Past two poll intervals: the retired surface must not keep asking a
        // route that answered "no such door".
        await new Promise((resolve) => setTimeout(resolve, 600));
        assert.equal(holdReads(), 1, 'polling stopped with the surface');
        assert.deepEqual(timers.outstanding(), [],
            'the latch poll interval is cleared, not just ignored');
        assert.equal(await handle.refreshHoldState(), false);
        assert.equal(holdReads(), 1, 'and an explicit refresh does not revive it');
    } finally {
        timers.restore();
        if (handle) handle.destroy();
    }
});

// Retirement only fires on a structural "no such route". A host that answers
// the probe with a 500 (or never answers at all) has told us nothing, so the
// component must not have drawn a Hold control on the strength of the client
// merely having the methods — that is the accidental opt-in, and it leaves a
// permanently disabled kebab on a card that never had a Hold door.
test('a Hold probe that never succeeds never grows a Hold control', async () => {
    const client = makeRealClient((url) => (String(url).includes('/api/host/hold')
        ? { ok: false, status: 500, body: { detail: 'boom' } }
        : { ok: true, status: 200 }));

    const { el, handle } = mountInto({ api: client });
    await tick();
    await tick();
    await tick();

    assert.equal(el.querySelector('.agent-card-kebab'), null,
        'an unanswered probe is not permission to draw the surface');
    assert.equal(el.querySelector('.agent-hold-badge'), null);
    assert.equal(el.querySelector('.agent-resume-btn'), null);
    assert.ok(el.querySelector('.agent-stop-btn'), 'Stop is untouched');
    handle.destroy();
});

// `hold` is the escape hatch from the probe, and the PANE is what Frinz and the
// console actually mount. A pane that accepts the option and drops it on the
// floor is worse than one that never had it: the embedder's configuration is
// silently ignored, so the two paths below assert the option ARRIVES, by the
// two behaviours only the inner component can produce.

test('the pane forwards hold:false — the standard client cannot opt an embedding host in', async () => {
    const requested = [];
    // A host that WOULD answer the probe. If `hold: false` is dropped, the pane
    // probes, the door answers, and the surface appears — which is exactly the
    // accidental opt-in an embedder passed `hold: false` to prevent.
    const client = makeRealClient((url) => {
        requested.push(String(url));
        return String(url).includes('/api/host/hold')
            ? { ok: true, status: 200, body: { can_hold: true, host_hold: null, agents: [] } }
            : { ok: true, status: 200 };
    });

    const timers = watchTimers();
    const el = document.createElement('div');
    document.body.appendChild(el);
    try {
        const pane = mountAgentListPane(el, {
            api: client,
            adapter: { mode: 'multi_agent', listAgents: async () => [agentItem()] },
            hold: false,
            holdStatusIntervalMs: 250,
        });
        mounted.push(pane);
        await tick();
        await tick();
        await tick();

        assert.ok(el.querySelector('.agent-card'), 'the pane still renders its cards');
        assert.ok(el.querySelector('.agent-stop-btn'), 'Stop is untouched');
        assert.equal(el.querySelector('.agent-card-kebab'), null,
            'hold:false must reach the inner component and suppress the surface');
        assert.equal(el.querySelector('.agent-hold-badge'), null);
        assert.equal(el.querySelector('.agent-resume-btn'), null);
        assert.equal(requested.filter((u) => u.includes('/api/host/hold')).length, 0,
            'and it must not probe a door the embedder disabled');

        await new Promise((resolve) => setTimeout(resolve, 600));
        assert.equal(requested.filter((u) => u.includes('/api/host/hold')).length, 0,
            'nor keep probing it on the poll');
        assert.deepEqual(timers.outstanding(), [],
            'a disabled Hold opens no latch-poll interval at all');
        assert.equal(await pane.refreshHoldState(), false);
    } finally {
        timers.restore();
    }
});

test('the pane forwards hold:true — an asserted door draws the surface without waiting for a probe', async () => {
    // The probe never answers, so ONLY a forwarded `hold: true` can attach the
    // surface here. With the option dropped, holdSupported stays null and the
    // controls are built-but-detached exactly as in the unanswered-probe test.
    let reads = 0;
    const api = {
        setHostAgent() {},
        getHostHoldState: () => { reads++; return new Promise(() => {}); },
        setHostHold: async () => ({}),
        releaseHostHold: async () => ({}),
    };
    const el = document.createElement('div');
    document.body.appendChild(el);
    const pane = mountAgentListPane(el, {
        api,
        adapter: { mode: 'multi_agent', listAgents: async () => [agentItem()] },
        hold: true,
        holdStatusIntervalMs: 1e7,
    });
    mounted.push(pane);
    await tick();
    await tick();

    assert.equal(reads, 1, 'an asserted door is still read for its latches');
    const kebab = el.querySelector('.agent-card-kebab');
    assert.ok(kebab, 'hold:true must reach the inner component and attach the surface');
    assert.equal(kebab.disabled, true,
        'asserting the door exists is not asserting this caller may use it');
    assert.ok(el.querySelector('.agent-hold-badge'), 'the resting badge is present, hidden');
    assert.equal(el.querySelector('.agent-hold-badge').hidden, true);
});

// The component derives every node from `containerEl.ownerDocument`, which is
// how an embedding host (#3165's shared pane) mounts this list into ITS
// document rather than the console's. The Hold surface has to honour the same
// abstraction: a kebab built in the module-global `document` belongs to a tree
// the host never renders, so the control is simply absent where it matters.
test('the Hold surface is built in the container\'s document, not the console\'s', async () => {
    const other = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' });
    const otherDoc = other.window.document;
    assert.notEqual(otherDoc, document, 'the embedding host owns a different document');

    const el = otherDoc.createElement('div');
    otherDoc.body.appendChild(el);
    const api = holdApi({ agentHold: latch() });
    const handle = mountAgentList(el, {
        api,
        adapter: { mode: 'multi_agent', listAgents: async () => [agentItem()] },
        holdStatusIntervalMs: 1e7,
        askHoldReason: () => 'operator reason',
    });
    mounted.push(handle);
    try {
        await tick();
        await tick();

        const kebab = el.querySelector('.agent-card-kebab');
        assert.ok(kebab, 'the embedded card carries the Hold kebab');
        assert.equal(kebab.ownerDocument, otherDoc,
            'the kebab belongs to the document the list was mounted into');
        // The siblings it sits beside already do — the kebab was the outlier.
        for (const selector of ['.agent-hold-badge', '.agent-resume-btn', '.agent-hold-outcome']) {
            assert.equal(el.querySelector(selector).ownerDocument, otherDoc, selector);
        }
        assert.equal(document.querySelector('.agent-card-kebab'), null,
            'and nothing was left behind in the console document');

        // `appendChild` ADOPTS across documents, so the assertion above is
        // satisfied even by a button created in the console document and then
        // attached here — it cannot, on its own, tell the defect from the fix.
        // The menu is what adoption does not launder: it is mounted on the
        // creating document's own `body`, so a kebab built in the wrong
        // document opens its menu in a tree the embedding host never renders.
        kebab.click();
        const fromButton = otherDoc.querySelector('.kebab-menu');
        assert.ok(fromButton, 'the menu opens in the embedding document');
        assert.equal(document.querySelector('.kebab-menu'), null);
        closeKebabMenu();

        el.querySelector('.agent-card').dispatchEvent(
            new other.window.MouseEvent('contextmenu', { bubbles: true, clientX: 5, clientY: 5 }),
        );
        assert.ok(otherDoc.querySelector('.kebab-menu'),
            'the right-click accelerator opens the same menu in the same document');
        assert.equal(document.querySelector('.kebab-menu'), null);
    } finally {
        closeKebabMenu();
        handle.destroy();
        other.window.close();
    }
});

// The primitive's own boundary, asserted where adoption cannot launder it: a
// button that has never been appended anywhere. Both ends — the option is
// honoured, and the default that every existing console caller relies on is
// unchanged.
test('the kebab primitive builds in the document it is given, and still defaults to the console\'s', () => {
    const other = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' });
    try {
        const given = createKebabButton(() => [], { ownerDocument: other.window.document });
        assert.equal(given.ownerDocument, other.window.document);
        assert.equal(createKebabButton(() => []).ownerDocument, document,
            'a caller that names no document keeps the module-global one');
    } finally {
        other.window.close();
    }
});
