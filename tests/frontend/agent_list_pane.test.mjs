// #2279: `mountAgentListPane` is the agent/companion analogue of
// `mountConversationsPane` — the shared `mountAgentList` surface PLUS the SAME
// pane chrome (chevron collapse to fully-hidden, drag-resize with min/max +
// localStorage persistence) AND a component-owned "+ New" header action wired
// via `onNew`. The standalone console and any embedder consume this one export.
// These tests exercise the pane contract directly:
//   - mount builds/adopts chrome and mounts the list;
//   - the chevron closes the pane to fully hidden (#2216), persisted + restored;
//   - the resize handle clamps to min/max and persists the width;
//   - the "+ New" header action fires `onNew`;
//   - destroy() leaves ADOPTED chrome in place;
//   - a console-style adopt (with onNew) GAINS a new-agent affordance.

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
globalThis.window.kicon = (name) => `<span class="ki ki-${name}" aria-hidden="true"></span>`;
globalThis.kicon = globalThis.window.kicon;

// In-memory localStorage so persistence is observable across mounts.
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

const { mountAgentListPane } = await import('../../kestrel_sovereign/static/js/agent_list.js');
const { validateHostStopEnvelope } = await import(
    '../../kestrel_sovereign/static/js/stop_evidence.js'
);

function tick() { return new Promise((r) => setTimeout(r, 0)); }

function fakeAdapter(items = [], mode = 'multi_agent') {
    return { mode, listAgents: async () => items };
}

function browserStopFence() { return () => {}; }

function hostStopEnvelope(correlationId, specs) {
    const outcomes = specs.map(({ agent, disposition, detail }, index) => ({
        scope: 'host',
        requested_target: null,
        resolved_target: agent,
        agent_id: agent,
        disposition,
        correlation_id: correlationId,
        receipt_id: 'receipt-host-stop',
        ...(detail ? { detail } : {}),
        ordinal: index,
    }));
    const confirmed = outcomes.filter((outcome) => (
        ['stopped', 'already_complete'].includes(outcome.disposition)
    )).length;
    const unconfirmed = outcomes.length - confirmed;
    return {
        success: unconfirmed === 0,
        state: confirmed && unconfirmed ? 'partial' : (unconfirmed ? 'unconfirmed' : 'confirmed'),
        target_count: outcomes.length,
        confirmed_count: confirmed,
        unconfirmed_count: unconfirmed,
        correlation_id: correlationId,
        stop_outcomes: outcomes,
    };
}

// Mirror index.html's static #agents-pane chrome (adopt path).
function makeConsolePane() {
    const el = document.createElement('aside');
    el.id = 'agents-pane';
    el.className = 'pane-sidebar';
    el.innerHTML = `
        <div class="pane-header"><h3>Agents</h3>
            <button id="collapse-agents-btn" class="collapse-btn"></button></div>
        <div id="agents-list" class="pane-content"></div>
        <div id="resize-agents" class="resize-handle"></div>`;
    document.body.appendChild(el);
    return el;
}

test('mount builds pane chrome (header, collapse rail, resize handle) into a bare container', async () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter([{ name: 'Emma', status: 'online' }]),
        storageKey: 'a:test-build',
    });
    await tick();

    assert.ok(el.classList.contains('pane-sidebar'), 'container becomes a pane-sidebar');
    assert.ok(el.classList.contains('agent-list-pane'), 'container tagged agent-list-pane');
    assert.ok(el.querySelector('.pane-header'), 'header built');
    assert.ok(el.querySelector('.collapse-btn'), 'collapse rail built');
    assert.ok(el.querySelector('.resize-handle'), 'resize handle built');
    assert.ok(el.querySelector('.agent-card'), 'list rows rendered inside the pane');
    assert.ok(handle.list, 'inner mountAgentList handle exposed');
    handle.destroy();
});

test('mount ADOPTS an existing static pane header + resize handle (console chrome)', async () => {
    const el = makeConsolePane();
    const headerBefore = el.querySelector('.pane-header');
    const handleBefore = el.querySelector('.resize-handle');
    const listBefore = el.querySelector('#agents-list');

    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter([{ name: 'Emma', status: 'online' }]), storageKey: 'a:test-adopt', autoLoad: false,
    });
    assert.equal(el.querySelectorAll('.pane-header').length, 1, 'no duplicate header');
    assert.equal(el.querySelector('.pane-header'), headerBefore, 'existing header adopted');
    assert.equal(el.querySelector('.resize-handle'), handleBefore, 'existing resize handle adopted');
    assert.equal(el.querySelectorAll('.resize-handle').length, 1, 'no duplicate resize handle');
    // The list mounts into the adopted #agents-list.
    assert.ok(listBefore.querySelector('.agent-list-root'), 'list mounted into adopted #agents-list');
    handle.destroy();
});

test('#2216: the chevron closes the agents pane to fully hidden (display:none), persisted + restored', () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const seen = [];
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter(), storageKey: 'a:test-collapse', autoLoad: false,
        onToggle: (c) => seen.push(c),
    });
    // The agents pane defaults OPEN (it is the primary nav surface).
    assert.equal(handle.collapsed, false, 'starts open by default');
    assert.notEqual(el.style.display, 'none', 'open pane is visible');

    // The chevron CLOSES to fully hidden — no leftover rail.
    el.querySelector('.collapse-btn').click();
    assert.equal(handle.collapsed, true, 'chevron closed the pane');
    assert.equal(el.style.display, 'none', 'closed pane takes zero width (display:none)');
    assert.ok(el.classList.contains('collapsed'), 'collapsed marker in lock-step with display');
    assert.equal(localStorage.getItem('a:test-collapse:collapsed'), '1', 'closed state persisted');
    // The chevron only closes — a second click does NOT reopen it.
    el.querySelector('.collapse-btn').click();
    assert.equal(handle.collapsed, true, 'chevron never reopens (open()/toggle() do)');
    handle.open();
    assert.equal(handle.collapsed, false, 'open() reveals the pane again');
    assert.equal(localStorage.getItem('a:test-collapse:collapsed'), '0', 'open state persisted');
    assert.deepEqual(seen, [false, true, false], 'onToggle fired for init + each change');
    handle.destroy();

    // A persisted CLOSED state is restored (hidden) on the next mount.
    const el2 = document.createElement('div');
    document.body.appendChild(el2);
    localStorage.setItem('a:test-collapse-restore:collapsed', '1');
    const handle2 = mountAgentListPane(el2, {
        adapter: fakeAdapter(), storageKey: 'a:test-collapse-restore', autoLoad: false,
    });
    assert.equal(handle2.collapsed, true, 'restored closed from localStorage');
    assert.equal(el2.style.display, 'none', 'restored closed pane is fully hidden');
    handle2.destroy();
});

test('the resize handle clamps to min/max and persists the width', () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter(), storageKey: 'a:test-resize', autoLoad: false,
        minWidth: 200, maxWidth: 500,
    });
    const rh = el.querySelector('.resize-handle');
    Object.defineProperty(el, 'offsetWidth', { value: 280, configurable: true });
    rh.dispatchEvent(new dom.window.MouseEvent('mousedown', { clientX: 300 }));
    document.dispatchEvent(new dom.window.MouseEvent('mousemove', { clientX: 9999 }));
    assert.equal(el.style.width, '500px', 'width clamped to maxWidth');
    document.dispatchEvent(new dom.window.MouseEvent('mousemove', { clientX: -9999 }));
    assert.equal(el.style.width, '200px', 'width clamped to minWidth');
    document.dispatchEvent(new dom.window.MouseEvent('mouseup', {}));
    assert.ok(localStorage.getItem('a:test-resize:width'), 'width persisted on mouseup');
    handle.destroy();

    // A persisted width is restored (clamped) on the next mount.
    localStorage.setItem('a:test-width:width', '9999');
    const el2 = document.createElement('div');
    document.body.appendChild(el2);
    const handle2 = mountAgentListPane(el2, {
        adapter: fakeAdapter(), storageKey: 'a:test-width', autoLoad: false, maxWidth: 500,
    });
    assert.equal(el2.style.width, '500px', 'oversized persisted width clamped to max');
    handle2.destroy();
});

test('the "+ New" header action fires onNew', () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    let fired = 0;
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter(), storageKey: 'a:test-new', autoLoad: false,
        onNew: () => { fired += 1; },
    });
    const newBtn = el.querySelector('.new-agent-btn');
    assert.ok(newBtn, 'component-owned New button built when onNew is provided');
    newBtn.click();
    assert.equal(fired, 1, 'clicking New fires onNew');
    handle.destroy();
});

test('no "+ New" button is built when onNew is omitted', () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter(), storageKey: 'a:test-nonew', autoLoad: false,
    });
    assert.equal(el.querySelector('.new-agent-btn'), null, 'no New button without an onNew hook');
    handle.destroy();
});

test('destroy() leaves ADOPTED chrome in place (only built chrome is removed)', () => {
    const el = makeConsolePane();
    const headerBefore = el.querySelector('.pane-header');
    const resizeBefore = el.querySelector('.resize-handle');
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter(), storageKey: 'a:test-destroy-adopt', autoLoad: false,
        onNew: () => {},
    });
    handle.destroy();
    assert.equal(el.querySelector('.pane-header'), headerBefore, 'adopted header survives destroy');
    assert.equal(el.querySelector('.resize-handle'), resizeBefore, 'adopted resize handle survives destroy');
});

test('destroy() removes BUILT chrome (bare container)', () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter(), storageKey: 'a:test-destroy-build', autoLoad: false,
        onNew: () => {},
    });
    assert.ok(el.querySelector('.pane-header'), 'header built');
    assert.ok(el.querySelector('.new-agent-btn'), 'New button built');
    handle.destroy();
    assert.equal(el.querySelector('.pane-header'), null, 'built header removed on destroy');
    assert.equal(el.querySelector('.new-agent-btn'), null, 'built New button removed on destroy');
});

test('standalone console (adopt) GAINS a new-agent affordance via onNew', async () => {
    // The console mounts into the static #agents-pane and passes onNew — so the
    // affordance it lacked today appears in the adopted header (same-everywhere).
    const el = makeConsolePane();
    let opened = 0;
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter([{ name: 'Emma', status: 'online' }]),
        storageKey: 'a:test-standalone-new',
        onNew: () => { opened += 1; },
    });
    await tick();
    const newBtn = el.querySelector('.pane-header .new-agent-btn');
    assert.ok(newBtn, 'new-agent affordance present in the console pane header');
    newBtn.click();
    assert.equal(opened, 1, 'the affordance drives the new-agent flow');
    handle.destroy();
});

test('the handle exposes list delegators (refresh/select/setActiveName/getActive) + pane controls', () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter(), storageKey: 'a:test-delegate', autoLoad: false,
    });
    assert.equal(typeof handle.refresh, 'function');
    assert.equal(typeof handle.select, 'function');
    assert.equal(typeof handle.setActiveName, 'function');
    assert.equal(typeof handle.getActive, 'function');
    assert.equal(typeof handle.open, 'function');
    assert.equal(typeof handle.close, 'function');
    assert.equal(typeof handle.toggle, 'function');
    assert.ok(handle.list, 'inner mountAgentList handle exposed');
    handle.destroy();
});

test('destroy() removes the BUILT resize handle from a bare container (codex P2)', async () => {
    const el = document.createElement('div');   // bare embed container
    document.body.appendChild(el);
    const handle = mountAgentListPane(el, {
        adapter: { mode: 'multi_agent', listAgents: async () => [] },
        storageKey: 'test:dz-pane',
    });
    await new Promise((r) => setTimeout(r, 0));
    assert.ok(el.querySelector('.resize-handle'), 'bare mount builds a resize handle');
    handle.destroy();
    assert.equal(el.querySelector('.resize-handle'), null, 'built handle removed on destroy');

    // Adopted chrome stays: pre-existing handle survives destroy.
    const el2 = document.createElement('div');
    const preHandle = document.createElement('div');
    preHandle.className = 'resize-handle';
    el2.appendChild(preHandle);
    document.body.appendChild(el2);
    const handle2 = mountAgentListPane(el2, {
        adapter: { mode: 'multi_agent', listAgents: async () => [] },
        storageKey: 'test:dz-pane2',
    });
    await new Promise((r) => setTimeout(r, 0));
    handle2.destroy();
    assert.ok(el2.querySelector('.resize-handle'), 'adopted handle survives destroy');
});

test('built header sits directly above the body when the pane has foreign leading chrome (contiguous two-part unit)', async () => {
    // An embedder (Frinz) nests its own user panel INSIDE the pane so the whole
    // column — chrome + list — collapses together. The component must not pin its
    // header to paneEl.firstChild and sandwich that chrome between header and list.
    const el = document.createElement('div');
    const userChrome = document.createElement('div');
    userChrome.className = 'user-info';
    userChrome.textContent = 'user@example.com';
    el.appendChild(userChrome);            // pre-existing leading child
    document.body.appendChild(el);

    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter([{ name: 'Sally', status: 'online' }]),
        onNew: () => {},
        newLabel: 'Add a companion',
        title: 'Companions',
        storageKey: 'a:test-foreign-child',
        autoLoad: false,
    });
    await tick();

    const kids = Array.from(el.children);
    const iChrome = kids.indexOf(userChrome);
    const iHeader = kids.indexOf(el.querySelector('.pane-header'));
    const iBody = kids.indexOf(el.querySelector('.agent-list-pane-body'));
    const iHandle = kids.indexOf(el.querySelector('.resize-handle'));

    // Foreign chrome stays first; the component's header + body are contiguous
    // below it (header immediately followed by body), handle last.
    assert.equal(iChrome, 0, 'foreign user chrome stays at the top');
    assert.equal(iHeader, 1, 'built header sits right below the foreign chrome');
    assert.equal(iBody, 2, 'list body is immediately after the header (contiguous)');
    assert.equal(iBody - iHeader, 1, 'header and body are adjacent — chrome is not sandwiched between them');
    assert.ok(iHandle > iBody, 'resize handle stays last');

    // Collapse still hides the whole pane (chrome included), since chrome is
    // inside paneEl.
    handle.close();
    assert.equal(el.style.display, 'none', 'collapse hides the whole pane, user chrome included');
    handle.destroy();
});

test('adopted body nested inside foreign chrome does not throw (built-header reposition guard)', async () => {
    // Unusual shape: an existing list body is nested INSIDE a foreign wrapper, so
    // it is not a direct child of the pane. The built-header reposition must skip
    // (guarded on body.parentNode === paneEl) rather than throw NotFoundError.
    // Uses the class-based body selector (.agent-list-pane-body) rather than the
    // #agents-list id to avoid colliding with the console-adopt test's leftover
    // pane in the shared JSDOM document.
    const el = document.createElement('div');
    const wrapper = document.createElement('div');
    wrapper.className = 'host-wrapper';
    const nestedBody = document.createElement('div');
    nestedBody.className = 'pane-content agent-list-pane-body';
    wrapper.appendChild(nestedBody);
    el.appendChild(wrapper);
    document.body.appendChild(el);

    let handle;
    assert.doesNotThrow(() => {
        handle = mountAgentListPane(el, {
            adapter: fakeAdapter([{ name: 'Emma', status: 'online' }]),
            storageKey: 'a:test-nested-body',
            autoLoad: false,
        });
    }, 'mount with a nested adopted body must not throw');
    await tick();
    // The guard's contract: the nested body is adopted in place (not moved), a
    // header was still built, and the list mounted into the adopted body.
    assert.equal(nestedBody.parentNode, wrapper, 'nested body stays where the host put it');
    assert.ok(el.querySelector('.pane-header'), 'header still built');
    assert.ok(nestedBody.querySelector('.agent-list-root'), 'list mounts into the adopted nested body');
    handle.destroy();
});

test('Stop All is disabled without live work and confirms the exact in-flight count', async () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const busy = new Set(['Emma', 'Kite']);
    const confirmations = [];
    const stopCalls = [];
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter([
            { name: 'Emma', id: 'did:agent:emma', status: 'online' },
            { name: 'Kite', id: 'did:agent:kite', status: 'online' },
            { name: 'Talon', id: 'did:agent:talon', status: 'online' },
        ]),
        isThinking: (name) => busy.has(name),
        api: {
            getHostStopStatus: async () => ({ can_stop: true, in_flight_count: 2 }),
            stopHost: async (payload) => {
                stopCalls.push(payload);
                return hostStopEnvelope(payload.correlation_id, [
                    { agent: 'did:agent:emma', disposition: 'stopped' },
                    {
                        agent: 'did:agent:kite',
                        disposition: 'refused',
                        detail: 'target declined cooperative Stop',
                    },
                ]);
            },
        },
        onPrepareStopAll: browserStopFence,
        confirmStopAll: (message) => {
            confirmations.push(message);
            return true;
        },
        storageKey: 'a:test-stop-all-count',
    });
    await tick();

    const button = el.querySelector('.agent-stop-all-btn');
    assert.ok(button, 'pane owns a visible Stop All control');
    assert.equal(button.disabled, false, 'control is enabled while work is live');
    button.click();
    await tick();

    assert.equal(stopCalls.length, 1, 'one host Stop request fan-outs server-side');
    assert.match(stopCalls[0].correlation_id, /^ui-host-stop:/,
        'browser owns the retryable durable operation identity');
    assert.match(confirmations[0], /2 in-flight agents/, 'confirmation names the live count');
    const report = el.querySelector('.agent-stop-all-results');
    assert.match(report.textContent, /Emma: stopped/, 'successful target remains visible');
    assert.match(report.textContent, /Kite: refused/, 'partial refusal remains visible');
    assert.match(report.textContent, /target declined cooperative Stop/, 'typed detail is not collapsed');
    handle.destroy();
});

test('Stop All never calls the host seam when no agent is in flight', async () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    let calls = 0;
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter([{ name: 'Emma', status: 'online' }]),
        isThinking: () => false,
        api: {
            getHostStopStatus: async () => ({ can_stop: true, in_flight_count: 0 }),
            stopHost: async () => { calls += 1; return { stop_outcomes: [] }; },
        },
        onPrepareStopAll: browserStopFence,
        confirmStopAll: () => true,
        storageKey: 'a:test-stop-all-idle',
    });
    await tick();

    const button = el.querySelector('.agent-stop-all-btn');
    assert.equal(button.disabled, true, 'idle fleet cannot issue Stop All');
    button.click();
    await tick();
    assert.equal(calls, 0, 'disabled action never calls lifecycle or Stop APIs');
    handle.destroy();
});

test('a click does not abandon accepted Stop intent to a second status read', async () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    let statusCalls = 0;
    let fenceCalls = 0;
    let stopCalls = 0;
    let statusCallsAtStop = null;
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter([{ name: 'Emma', id: 'did:agent:emma' }]),
        api: {
            getHostStopStatus: async () => ({
                can_stop: true,
                in_flight_count: ++statusCalls === 1 ? 1 : 0,
            }),
            stopHost: async (payload) => {
                statusCallsAtStop = statusCalls;
                stopCalls += 1;
                return hostStopEnvelope(payload.correlation_id, [
                    { agent: 'did:agent:emma', disposition: 'already_complete' },
                ]);
            },
        },
        onPrepareStopAll: () => { fenceCalls += 1; return () => {}; },
        confirmStopAll: () => true,
        stopAllStatusIntervalMs: 999999,
    });
    await tick();
    await tick();

    el.querySelector('.agent-stop-all-btn').click();
    await tick();
    await tick();

    handle.destroy();
    assert.equal(fenceCalls, 1, 'the accepted user intent fences local queued work');
    assert.equal(statusCallsAtStop, 1,
        'the accepted action does not depend on a racy second read');
    assert.equal(stopCalls, 1, 'the host authority resolves the accepted Stop intent');
});

test('Stop All waits for adapter identity inventory before status or enablement', async () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    let resolveItems;
    const items = new Promise((resolve) => { resolveItems = resolve; });
    let statusCalls = 0;
    const handle = mountAgentListPane(el, {
        adapter: { mode: 'multi_agent', listAgents: () => items },
        api: {
            getHostStopStatus: async () => {
                statusCalls += 1;
                return { can_stop: true, in_flight_count: 1 };
            },
            stopHost: async () => ({}),
        },
        onPrepareStopAll: browserStopFence,
        stopAllStatusIntervalMs: 250,
    });
    const button = el.querySelector('.agent-stop-all-btn');
    await tick();
    const statusCallsBeforeInventory = statusCalls;
    const disabledBeforeInventory = button.disabled;

    resolveItems([{ name: 'Emma', id: 'did:agent:emma' }]);
    await tick();
    await tick();
    handle.destroy();
    assert.equal(statusCallsBeforeInventory, 0,
        'identity inventory is load-bearing for settlement');
    assert.equal(disabledBeforeInventory, true);
    assert.equal(statusCalls, 1);
    assert.equal(button.disabled, false);
});

test('autoLoad false defers Stop All authority polling until explicit refresh', async () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    let statusCalls = 0;
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter([{ name: 'Emma', id: 'did:agent:emma' }]),
        autoLoad: false,
        api: {
            getHostStopStatus: async () => {
                statusCalls += 1;
                return { can_stop: true, in_flight_count: 1 };
            },
            stopHost: async () => ({}),
        },
        onPrepareStopAll: browserStopFence,
        stopAllStatusIntervalMs: 250,
    });
    await tick();
    assert.equal(statusCalls, 0);
    assert.equal(el.querySelector('.agent-stop-all-btn').disabled, true);

    await handle.refresh();
    await tick();
    assert.equal(statusCalls, 1);
    assert.equal(el.querySelector('.agent-stop-all-btn').disabled, false);
    handle.destroy();
});

test('slow Stop All status polling serializes one authoritative request', async () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    let resolveStatus;
    const status = new Promise((resolve) => { resolveStatus = resolve; });
    let statusCalls = 0;
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter([{ name: 'Emma', id: 'did:agent:emma' }]),
        api: {
            getHostStopStatus: () => {
                statusCalls += 1;
                return status;
            },
            stopHost: async () => ({}),
        },
        onPrepareStopAll: browserStopFence,
        stopAllStatusIntervalMs: 250,
    });
    await tick();
    await new Promise((resolve) => setTimeout(resolve, 300));

    const callsBeforeSettlement = statusCalls;
    resolveStatus({ can_stop: true, in_flight_count: 1 });
    await tick();
    assert.equal(el.querySelector('.agent-stop-all-btn').disabled, false);
    handle.destroy();
    assert.equal(callsBeforeSettlement, 1,
        'an interval tick joins the in-flight status request instead of superseding it');
});

test('Stop All fails closed when an embed omits the browser-work fence contract', async () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    let calls = 0;
    let statusCalls = 0;
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter([{ name: 'Emma', status: 'online' }]),
        api: {
            getHostStopStatus: async () => {
                statusCalls += 1;
                return { can_stop: true, in_flight_count: 1 };
            },
            stopHost: async () => { calls += 1; },
        },
        confirmStopAll: () => true,
        storageKey: 'a:test-stop-all-fence-required',
    });
    await tick();

    try {
        assert.equal(el.querySelector('.agent-stop-all-btn'), null,
            'Stop All is an explicit embed opt-in, not dead default chrome');
        assert.equal(statusCalls, 0, 'unopted embeds never poll a host authority door');
        assert.equal(calls, 0, 'an unfenced embed cannot issue Host Stop');
    } finally {
        handle.destroy();
    }
});

test('Stop All reports an empty or malformed fan-out as indeterminate, never success', async () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter([{ name: 'Emma', status: 'online' }]),
        isThinking: () => true,
        api: {
            getHostStopStatus: async () => ({ can_stop: true, in_flight_count: 1 }),
            stopHost: async (payload) => {
                const response = hostStopEnvelope(payload.correlation_id, [
                    { agent: 'did:agent:emma', disposition: 'stopped' },
                ]);
                delete response.stop_outcomes[0].receipt_id;
                return response;
            },
        },
        onPrepareStopAll: browserStopFence,
        confirmStopAll: () => true,
        storageKey: 'a:test-stop-all-empty',
    });
    await tick();

    el.querySelector('.agent-stop-all-btn').click();
    await tick();
    const report = el.querySelector('.agent-stop-all-results');
    assert.match(report.textContent, /malformed or incomplete/);
    assert.doesNotMatch(report.textContent, /all stopped/i);
    handle.destroy();
});

test('an ambiguous Host Stop retry reuses the browser-owned correlation id', async () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const operationIds = [];
    let stopCalls = 0;
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter([
            { name: 'Emma', id: 'did:agent:emma', status: 'online' },
        ]),
        api: {
            getHostStopStatus: async () => ({
                can_stop: true,
                in_flight_count: stopCalls === 0 ? 1 : 0,
            }),
            stopHost: async (payload) => {
                operationIds.push(payload.correlation_id);
                stopCalls += 1;
                if (stopCalls === 1) throw new Error('response lost');
                return hostStopEnvelope(payload.correlation_id, [
                    { agent: 'did:agent:emma', disposition: 'stopped' },
                ]);
            },
        },
        onPrepareStopAll: browserStopFence,
        confirmStopAll: () => true,
        storageKey: 'a:test-stop-all-retry-identity',
    });
    await tick();

    const button = el.querySelector('.agent-stop-all-btn');
    button.click();
    await tick();
    await tick();
    assert.equal(button.disabled, false,
        'lost response remains retryable even after live count reaches zero');

    button.click();
    await tick();
    await tick();
    assert.equal(stopCalls, 2);
    assert.equal(operationIds[1], operationIds[0],
        'retry replays the exact durable Stop identity');
    assert.equal(button.disabled, true, 'recovered evidence clears the retry handle');
    handle.destroy();
});

test('new work after an ambiguous Host Stop gets a fresh operation after recovery', async () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const operationIds = [];
    let stopCalls = 0;
    let fenceCalls = 0;
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter([
            { name: 'Emma', id: 'did:agent:emma', status: 'online' },
        ]),
        api: {
            getHostStopStatus: async () => ({ can_stop: true, in_flight_count: 1 }),
            stopHost: async (payload) => {
                operationIds.push(payload.correlation_id);
                stopCalls += 1;
                if (stopCalls === 1) throw new Error('response lost');
                return hostStopEnvelope(payload.correlation_id, [
                    { agent: 'did:agent:emma', disposition: 'stopped' },
                ]);
            },
        },
        onPrepareStopAll: () => {
            fenceCalls += 1;
            return () => {};
        },
        confirmStopAll: () => true,
        stopAllStatusIntervalMs: 999999,
    });
    await tick();
    await tick();

    const button = el.querySelector('.agent-stop-all-btn');
    button.click();
    await tick();
    await tick();
    button.click();
    await tick();
    await tick();
    await tick();

    handle.destroy();
    assert.equal(operationIds.length, 3);
    assert.equal(operationIds[1], operationIds[0], 'the ambiguous operation is recovered');
    assert.notEqual(operationIds[2], operationIds[0], 'current work receives a fresh operation');
    assert.equal(fenceCalls, 2, 'recovery does not create a second browser fence');
});

test('a hung recovery cannot delay the fresh Stop for current work', async () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const operationIds = [];
    let stopCalls = 0;
    const never = new Promise(() => {});
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter([{ name: 'Emma', id: 'did:agent:emma' }]),
        api: {
            getHostStopStatus: async () => ({ can_stop: true, in_flight_count: 1 }),
            stopHost: async (payload) => {
                operationIds.push(payload.correlation_id);
                stopCalls += 1;
                if (stopCalls === 1) throw new Error('response lost');
                if (stopCalls === 2) return never;
                return hostStopEnvelope(payload.correlation_id, [
                    { agent: 'did:agent:emma', disposition: 'stopped' },
                ]);
            },
        },
        onPrepareStopAll: browserStopFence,
        confirmStopAll: () => true,
        stopAllStatusIntervalMs: 999999,
    });
    await tick();
    await tick();

    const button = el.querySelector('.agent-stop-all-btn');
    button.click();
    await tick();
    await tick();
    button.click();
    await tick();
    await tick();

    const callsBeforeDestroy = stopCalls;
    handle.destroy();
    assert.equal(callsBeforeDestroy, 3, 'fresh Stop is issued without awaiting old recovery');
    assert.equal(operationIds[1], operationIds[0]);
    assert.notEqual(operationIds[2], operationIds[0]);
});

test('an ambiguous gateway response is recovered before a fresh current-work Stop', async () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const operationIds = [];
    let stopCalls = 0;
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter([{ name: 'Emma', id: 'did:agent:emma' }]),
        api: {
            getHostStopStatus: async () => ({ can_stop: true, in_flight_count: 1 }),
            stopHost: async (payload) => {
                operationIds.push(payload.correlation_id);
                stopCalls += 1;
                if (stopCalls === 1) {
                    const error = new Error('gateway lost the upstream response');
                    error.status = 504;
                    throw error;
                }
                return hostStopEnvelope(payload.correlation_id, [
                    { agent: 'did:agent:emma', disposition: 'stopped' },
                ]);
            },
        },
        onPrepareStopAll: browserStopFence,
        confirmStopAll: () => true,
        stopAllStatusIntervalMs: 999999,
    });
    await tick();
    await tick();

    const button = el.querySelector('.agent-stop-all-btn');
    button.click();
    await tick();
    await tick();
    button.click();
    await tick();
    await tick();

    handle.destroy();
    assert.equal(operationIds.length, 3);
    assert.equal(operationIds[1], operationIds[0],
        'an upstream gateway failure is transport-ambiguous and recovered');
    assert.notEqual(operationIds[2], operationIds[0],
        'the live fleet is stopped under a fresh identity');
});

test('a terminal unreceipted response gets a fresh operation identity', async () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const operationIds = [];
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter([{ name: 'Emma', id: 'did:agent:emma' }]),
        api: {
            getHostStopStatus: async () => ({ can_stop: true, in_flight_count: 1 }),
            stopHost: async (payload) => {
                operationIds.push(payload.correlation_id);
                const response = hostStopEnvelope(payload.correlation_id, [
                    { agent: 'did:agent:emma', disposition: 'refused' },
                ]);
                response.stop_outcomes[0].receipt_id = null;
                return response;
            },
        },
        onPrepareStopAll: browserStopFence,
        confirmStopAll: () => true,
        stopAllStatusIntervalMs: 999999,
    });
    await tick();
    await tick();
    const button = el.querySelector('.agent-stop-all-btn');
    button.click();
    await tick();
    await tick();
    button.click();
    await tick();
    await tick();

    handle.destroy();
    assert.equal(operationIds.length, 2);
    assert.notEqual(operationIds[1], operationIds[0],
        'a received terminal refusal is not an ambiguous replay');
});

test('canonical host evidence rejects duplicate and cross-wired target identities', () => {
    const duplicate = hostStopEnvelope('stop:duplicate', [
        { agent: 'did:agent:emma', disposition: 'stopped' },
        { agent: 'did:agent:emma', disposition: 'stopped' },
    ]);
    assert.equal(validateHostStopEnvelope(duplicate, 'stop:duplicate'), null);

    const crossWired = hostStopEnvelope('stop:cross-wired', [
        { agent: 'did:agent:emma', disposition: 'stopped' },
    ]);
    crossWired.stop_outcomes[0].resolved_target = 'did:agent:kite';
    assert.equal(validateHostStopEnvelope(crossWired, 'stop:cross-wired'), null);
});

test('re-mounting an adopted pane replaces Stop All ownership without duplicate controls or handlers', async () => {
    const el = makeConsolePane();
    let firstCalls = 0;
    let secondCalls = 0;
    mountAgentListPane(el, {
        adapter: fakeAdapter([{ name: 'Emma', status: 'online' }]),
        isThinking: () => true,
        api: {
            getHostStopStatus: async () => ({ can_stop: true, in_flight_count: 1 }),
            stopHost: async () => { firstCalls += 1; return { stop_outcomes: [] }; },
        },
        onPrepareStopAll: browserStopFence,
        confirmStopAll: () => true,
        storageKey: 'a:test-stop-all-remount-one',
    });
    await tick();

    const second = mountAgentListPane(el, {
        adapter: fakeAdapter([{ name: 'Emma', status: 'online' }]),
        isThinking: () => true,
        api: {
            getHostStopStatus: async () => ({ can_stop: true, in_flight_count: 1 }),
            stopHost: async () => { secondCalls += 1; return { stop_outcomes: [] }; },
        },
        onPrepareStopAll: browserStopFence,
        confirmStopAll: () => true,
        storageKey: 'a:test-stop-all-remount-two',
    });
    await tick();

    assert.equal(el.querySelectorAll('.agent-stop-all-btn').length, 1, 'one adopted control');
    el.querySelector('.agent-stop-all-btn').click();
    await tick();
    assert.equal(firstCalls, 0, 'prior mount no longer owns a click handler');
    assert.equal(secondCalls, 1, 'current mount owns exactly one handler');
    second.destroy();
});

test('re-mounting during Stop All preserves the operation fence and retires stale UI continuations', async () => {
    const el = makeConsolePane();
    let finishStop;
    const stopPromise = new Promise((resolve) => { finishStop = resolve; });
    let firstOutcomeRenders = 0;
    const first = mountAgentListPane(el, {
        adapter: fakeAdapter([{ name: 'Emma', id: 'did:agent:emma', status: 'online' }]),
        api: {
            getHostStopStatus: async () => ({ can_stop: true, in_flight_count: 1 }),
            stopHost: async () => stopPromise,
        },
        onPrepareStopAll: browserStopFence,
        onStopAllOutcomes: () => { firstOutcomeRenders += 1; },
        confirmStopAll: () => true,
        storageKey: 'a:test-stop-all-active-remount-one',
    });
    await tick();
    el.querySelector('.agent-stop-all-btn').click();
    await tick();

    const second = mountAgentListPane(el, {
        adapter: fakeAdapter([{ name: 'Emma', id: 'did:agent:emma', status: 'online' }]),
        api: {
            getHostStopStatus: async () => ({ can_stop: true, in_flight_count: 1 }),
            stopHost: async () => { throw new Error('overlapping Stop must stay fenced'); },
        },
        onPrepareStopAll: browserStopFence,
        confirmStopAll: () => true,
        storageKey: 'a:test-stop-all-active-remount-two',
    });
    await tick();
    const adoptedButton = el.querySelector('.agent-stop-all-btn');
    assert.equal(adoptedButton.disabled, true,
        'the new owner inherits the still-running host operation fence');

    finishStop({ stop_outcomes: [] });
    await tick();
    await tick();

    assert.equal(firstOutcomeRenders, 0, 'retired owner cannot render after its awaited POST');
    assert.equal(adoptedButton.disabled, false,
        'the current owner refreshes after the inherited operation settles');
    assert.equal(el.querySelectorAll('.agent-stop-all-results').length, 1,
        'retired result surfaces are removed during adoption');
    first.destroy();
    second.destroy();
});

test('Stop All uses sovereign host status rather than this tab\'s busy cards', async () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const confirmations = [];
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter([{ name: 'Emma', id: 'did:agent:emma', status: 'online' }]),
        isThinking: () => false,
        api: {
            getHostStopStatus: async () => ({ can_stop: true, in_flight_count: 3 }),
            stopHost: async (payload) => hostStopEnvelope(payload.correlation_id, [
                { agent: 'did:agent:emma', disposition: 'stopped' },
            ]),
        },
        onPrepareStopAll: browserStopFence,
        confirmStopAll: (message) => { confirmations.push(message); return true; },
        storageKey: 'a:test-stop-all-host-inventory',
    });
    await tick();

    const button = el.querySelector('.agent-stop-all-btn');
    const disabled = button.disabled;
    button.click();
    await tick();
    const confirmation = confirmations[0];
    handle.destroy();
    assert.equal(disabled, false, 'remote/API/signal work keeps the host action enabled');
    assert.match(confirmation, /3 in-flight agents/, 'confirmation uses the fresh host count');
});

test('Stop All fails closed for callers without advertised sovereign authority', async () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    let stopCalls = 0;
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter([{ name: 'Emma', status: 'online' }]),
        isThinking: () => true,
        api: {
            getHostStopStatus: async () => ({ can_stop: false, in_flight_count: 1 }),
            stopHost: async () => { stopCalls += 1; return { stop_outcomes: [] }; },
        },
        onPrepareStopAll: browserStopFence,
        confirmStopAll: () => true,
        storageKey: 'a:test-stop-all-authority',
    });
    await tick();

    const button = el.querySelector('.agent-stop-all-btn');
    assert.equal(button.disabled, true);
    button.click();
    await tick();
    assert.equal(stopCalls, 0, 'unauthorized UI never attempts the sovereign operation');
    handle.destroy();
});

test('Stop All fences local queues synchronously before awaiting the host request', async () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const order = [];
    let resolveStop;
    const stopPromise = new Promise((resolve) => { resolveStop = resolve; });
    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter([{ name: 'Emma', id: 'did:agent:emma', status: 'online' }]),
        api: {
            getHostStopStatus: async () => ({ can_stop: true, in_flight_count: 1 }),
            stopHost: async () => { order.push('post'); return stopPromise; },
        },
        onPrepareStopAll: () => {
            order.push('fence');
            return () => order.push('settle');
        },
        confirmStopAll: () => true,
        storageKey: 'a:test-stop-all-local-fence',
    });
    await tick();

    el.querySelector('.agent-stop-all-btn').click();
    await tick();
    const beforeSettlement = [...order];
    resolveStop({
        stop_outcomes: [{
            agent_id: 'did:agent:emma',
            resolved_target: 'did:agent:emma',
            disposition: 'stopped',
        }],
    });
    await tick();
    const afterSettlement = [...order];
    handle.destroy();
    assert.deepEqual(beforeSettlement, ['fence', 'post']);
    assert.deepEqual(afterSettlement, ['fence', 'post', 'settle']);
});

test('a failed agent-list fetch does not permanently retire fleet Stop', async () => {
    // The control's authority is the HOST status endpoint, not the agent list.
    // A transient /api/agents failure used to latch listLoaded=false, which
    // gated the render, the poller and the click handler alike -- so the button
    // stayed disabled for the rest of the page session. Deterministic: drive
    // the recovery with an explicit refresh rather than waiting on a timer.
    const el = document.createElement('div');
    document.body.appendChild(el);
    let listCalls = 0;
    const adapter = {
        mode: 'multi_agent',
        listAgents: async () => {
            listCalls += 1;
            if (listCalls === 2) throw new Error('transient network failure');
            return [{ name: 'Emma', id: 'did:agent:emma' }];
        },
    };
    const handle = mountAgentListPane(el, {
        adapter,
        api: {
            getHostStopStatus: async () => ({ can_stop: true, in_flight_count: 1 }),
            stopHost: async () => ({}),
        },
        onPrepareStopAll: browserStopFence,
        stopAllStatusIntervalMs: 999999,
    });
    try {
        await tick();
        await tick();
        const button = el.querySelector('.agent-stop-all-btn');
        assert.equal(button.disabled, false, 'precondition: enabled after first load');

        await handle.refresh();          // call 2: the failing fetch
        await tick();

        await handle.refresh();          // call 3: the network recovers
        await tick();
        await tick();

        assert.equal(button.disabled, false,
            'a transient list failure must not disable fleet Stop for the session');
    } finally {
        handle.destroy();
    }
});

test('an adopted header whose collapse button is nested still mounts', async () => {
    // collapseBtn is found with a DESCENDANT query, so it need not be a direct
    // child. insertBefore on the header then threw NotFoundError and aborted
    // the mount before the owner handle was ever attached.
    const el = document.createElement('div');
    el.innerHTML = `
        <div class="agent-pane">
          <div class="pane-header">
            <div class="header-tools"><button class="collapse-btn">v</button></div>
          </div>
          <div class="pane-body"></div>
        </div>`;
    document.body.appendChild(el);

    const handle = mountAgentListPane(el, {
        adapter: fakeAdapter([{ name: 'Emma', id: 'did:agent:emma' }]),
        api: {
            getHostStopStatus: async () => ({ can_stop: true, in_flight_count: 1 }),
            stopHost: async () => ({}),
        },
        onPrepareStopAll: browserStopFence,
    });
    await tick();

    assert.ok(handle, 'mount must not abort on a nested collapse button');
    assert.ok(el.querySelector('.agent-stop-all-btn'),
        'the Stop All control is still placed');
    assert.equal(typeof handle.destroy, 'function');
    handle.destroy();
});
