// Unit tests for the UI extension slot registry + event bus (#2038, ticket 02).
//
// Covers the registry invariants the slot model promises (and that
// registerHeaderAction lacked): order sort with stable ties, gate
// re-evaluation on a bus event, per-contribution teardown-on-rerender, render
// error isolation, dedupe-by-id, empty-zone no-op, detached-instance pruning,
// and — the headline guarantee — updating ONE contribution does not tear down
// a sibling.
import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.Node = dom.window.Node;

const { UI } = await import('../../kestrel_sovereign/static/js/ui-ext/registry.js');
const bus = (await import('../../kestrel_sovereign/static/js/ui-ext/bus.js')).default;

function anchor() {
    const el = document.createElement('div');
    document.body.appendChild(el);
    return el;
}

function reset() {
    UI._reset();
    document.body.innerHTML = '';
}

function mountedIds(el) {
    return [...el.children].map((c) => c.dataset.uiExtId);
}

test('renderSlot mounts contributions sorted by order (stable on ties)', () => {
    reset();
    const el = anchor();
    UI.register({ slot: 'z', id: 'c', order: 30, render: () => {} });
    UI.register({ slot: 'z', id: 'a', order: 10, render: () => {} });
    UI.register({ slot: 'z', id: 'b', order: 20, render: () => {} });
    // Two with equal order keep registration order (b1 before b2).
    UI.register({ slot: 'z', id: 'b1', order: 20, render: () => {} });
    UI.register({ slot: 'z', id: 'b2', order: 20, render: () => {} });
    UI.renderSlot('z', { element: el });
    assert.deepEqual(mountedIds(el), ['a', 'b', 'b1', 'b2', 'c']);
});

test('default order is 100', () => {
    reset();
    const el = anchor();
    UI.register({ slot: 'z', id: 'late', order: 200, render: () => {} });
    UI.register({ slot: 'z', id: 'default', render: () => {} }); // 100
    UI.register({ slot: 'z', id: 'early', order: 1, render: () => {} });
    UI.renderSlot('z', { element: el });
    assert.deepEqual(mountedIds(el), ['early', 'default', 'late']);
});

test('empty zone renders no children and does not throw', () => {
    reset();
    const el = anchor();
    UI.renderSlot('nobody', { element: el });
    assert.equal(el.children.length, 0);
});

test('dedupe by id: re-registering an id replaces the prior contribution', () => {
    reset();
    const el = anchor();
    UI.register({ slot: 'z', id: 'dup', render: (c) => { c.textContent = 'first'; } });
    UI.register({ slot: 'z', id: 'dup', render: (c) => { c.textContent = 'second'; } });
    assert.equal(UI.contributions('z').length, 1);
    UI.renderSlot('z', { element: el });
    assert.equal(el.children.length, 1);
    assert.equal(el.children[0].textContent, 'second');
});

test('render error in one contribution is isolated; siblings still mount', () => {
    reset();
    const el = anchor();
    UI.register({ slot: 'z', id: 'boom', render: () => { throw new Error('nope'); } });
    UI.register({ slot: 'z', id: 'ok', render: (c) => { c.textContent = 'fine'; } });
    UI.renderSlot('z', { element: el });
    // The throwing contribution still gets a (empty) container; the healthy one renders.
    const ok = [...el.children].find((c) => c.dataset.uiExtId === 'ok');
    assert.ok(ok, 'healthy sibling mounted despite the thrower');
    assert.equal(ok.textContent, 'fine');
});

test('teardown fires before re-render of the same contribution', () => {
    reset();
    const el = anchor();
    let teardowns = 0;
    let renders = 0;
    UI.register({
        slot: 'z',
        id: 'live',
        render: () => { renders++; return () => { teardowns++; }; },
    });
    UI.renderSlot('z', { element: el });
    assert.equal(renders, 1);
    assert.equal(teardowns, 0);
    UI.refreshContribution('z', 'live');
    assert.equal(renders, 2);
    assert.equal(teardowns, 1, 'prior teardown ran exactly once before re-render');
});

test('updating ONE contribution does not tear down a sibling', () => {
    reset();
    const el = anchor();
    let aTear = 0;
    let bTear = 0;
    let bRenders = 0;
    UI.register({ slot: 'z', id: 'A', render: () => () => { aTear++; } });
    UI.register({ slot: 'z', id: 'B', render: () => { bRenders++; return () => { bTear++; }; } });
    UI.renderSlot('z', { element: el });
    const bContainerBefore = [...el.children].find((c) => c.dataset.uiExtId === 'B');

    UI.refreshContribution('z', 'A');

    assert.equal(aTear, 1, 'A torn down for its re-render');
    assert.equal(bTear, 0, "sibling B's teardown must NOT fire");
    assert.equal(bRenders, 1, 'sibling B not re-rendered');
    const bContainerAfter = [...el.children].find((c) => c.dataset.uiExtId === 'B');
    assert.equal(bContainerBefore, bContainerAfter, "sibling B's DOM element identity preserved");
});

test('gate is re-evaluated on a declared bus event (appear and disappear)', () => {
    reset();
    const el = anchor();
    let visible = false;
    UI.register({
        slot: 'z',
        id: 'gated',
        events: ['agent:switch'],
        gate: () => visible,
        render: (c) => { c.textContent = 'shown'; },
    });
    UI.renderSlot('z', { element: el });
    assert.equal(el.children.length, 0, 'gated out initially -> no container');

    visible = true;
    bus.emit('agent:switch');
    assert.equal(el.children.length, 1, 'event re-gates it in');
    assert.equal(el.children[0].textContent, 'shown');

    visible = false;
    bus.emit('agent:switch');
    assert.equal(el.children.length, 0, 'event re-gates it out and removes the container');
});

test('a contribution only reacts to events it declared', () => {
    reset();
    const el = anchor();
    let renders = 0;
    UI.register({
        slot: 'z',
        id: 'narrow',
        events: ['session:change'],
        render: () => { renders++; },
    });
    UI.renderSlot('z', { element: el });
    assert.equal(renders, 1);
    bus.emit('agent:switch'); // not declared -> ignored
    assert.equal(renders, 1);
    bus.emit('session:change'); // declared -> re-render
    assert.equal(renders, 2);
});

test('unregister tears down and removes a single contribution', () => {
    reset();
    const el = anchor();
    let tear = 0;
    UI.register({ slot: 'z', id: 'A', render: () => () => { tear++; } });
    UI.register({ slot: 'z', id: 'B', render: () => {} });
    UI.renderSlot('z', { element: el });
    assert.equal(el.children.length, 2);

    UI.unregister('z', 'A');
    assert.equal(tear, 1);
    assert.deepEqual(mountedIds(el), ['B']);
    assert.equal(UI.contributions('z').length, 1);
});

test('detached instances are pruned (teardown fires) on next renderSlot', () => {
    reset();
    const el1 = anchor();
    let tear = 0;
    UI.register({ slot: 'z', id: 'A', render: () => () => { tear++; } });
    UI.renderSlot('z', { element: el1 });
    assert.equal(tear, 0);

    el1.remove(); // simulate the agents-list innerHTML rebuild detaching the card
    const el2 = anchor();
    UI.renderSlot('z', { element: el2 });
    assert.equal(tear, 1, 'stale detached instance was torn down');
});

test('renderSlot requires ctx.element (no silent crash)', () => {
    reset();
    UI.register({ slot: 'z', id: 'A', render: () => {} });
    assert.doesNotThrow(() => UI.renderSlot('z', {}));
});

test('contribution registered AFTER renderSlot mounts into the live instance', () => {
    reset();
    const el = anchor();
    UI.renderSlot('z', { element: el }); // empty zone rendered first (boot order)
    assert.equal(el.children.length, 0);

    UI.register({ slot: 'z', id: 'late', render: (c) => { c.textContent = 'here'; } });
    // No second core renderSlot — the late registration must appear on its own.
    assert.equal(el.children.length, 1, 'late registration mounted without a re-render');
    assert.equal(el.children[0].textContent, 'here');
});

test('modal-root style zone: late registration mounts with no retrigger event', () => {
    reset();
    const root = anchor();
    UI.renderSlot('modal-root', { element: root });
    UI.register({ slot: 'modal-root', id: 'm', render: (c) => { c.textContent = 'modal'; } });
    assert.deepEqual(mountedIds(root), ['m']);
});

test('late registration of a lower-order contribution preserves DOM order', () => {
    reset();
    const el = anchor();
    UI.register({ slot: 'z', id: 'B', order: 20, render: () => {} });
    UI.renderSlot('z', { element: el });
    assert.deepEqual(mountedIds(el), ['B']);

    // A registers later but sorts earlier — it must land before B, not after.
    UI.register({ slot: 'z', id: 'A', order: 10, render: () => {} });
    assert.deepEqual(mountedIds(el), ['A', 'B']);
});

test('renderSlot realigns DOM order when an earlier contribution is added between renders', () => {
    reset();
    const el = anchor();
    UI.register({ slot: 'z', id: 'B', order: 20, render: () => {} });
    UI.renderSlot('z', { element: el });
    const bBefore = [...el.children].find((c) => c.dataset.uiExtId === 'B');

    UI.register({ slot: 'z', id: 'A', order: 10, render: () => {} });
    UI.renderSlot('z', { element: el }); // re-render the whole zone
    assert.deepEqual(mountedIds(el), ['A', 'B']);
    // B's container identity is preserved (moved, not recreated).
    const bAfter = [...el.children].find((c) => c.dataset.uiExtId === 'B');
    assert.equal(bBefore, bAfter, "sibling B's DOM element identity preserved across reorder");
});

test('multiple instances of one slot render independently with their own ctx', () => {
    reset();
    const card1 = anchor();
    const card2 = anchor();
    UI.register({
        slot: 'agent-card',
        id: 'name',
        render: (c, ctx) => { c.textContent = ctx.agentName; },
    });
    UI.renderSlot('agent-card', { element: card1, agentName: 'Alpha' });
    UI.renderSlot('agent-card', { element: card2, agentName: 'Beta' });
    assert.equal(card1.children[0].textContent, 'Alpha');
    assert.equal(card2.children[0].textContent, 'Beta');
});
