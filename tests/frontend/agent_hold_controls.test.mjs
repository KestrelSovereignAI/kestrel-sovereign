// #3164: the shared agent-list component owns Hold/Resume presentation and
// actions so both the Sovereign console and custom-rendered adopters use the
// same visible kebab/context-menu path.

import test, { afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.Node = dom.window.Node;
globalThis.HTMLElement = dom.window.HTMLElement;
globalThis.location = dom.window.location;
globalThis.window.kicon = (name) => `<span class="ki ki-${name}" aria-hidden="true"></span>`;
globalThis.kicon = globalThis.window.kicon;

const { mountAgentList, createDefaultAgentAdapter } = await import(
    '../../kestrel_sovereign/static/js/agent_list.js'
);
const { closeKebabMenu } = await import(
    '../../kestrel_sovereign/static/js/kebab_menu.js'
);

function tick() { return new Promise((resolve) => setTimeout(resolve, 0)); }

function holdState(overrides = {}) {
    return {
        available: true,
        held: true,
        sources: ['agent'],
        host: null,
        agent: {
            scope: 'agent',
            target_id: 'did:test:emma',
            reason: 'maintenance window',
            actor_id: 'sovereign-key',
            set_at: '2026-08-30T12:00:00+00:00',
            hold_receipt_id: 'hold-receipt-1',
            revision: 1,
            ...overrides,
        },
    };
}

function fakeAdapter(items) {
    return { mode: 'multi_agent', listAgents: async () => items };
}

function mount(config = {}, items = [{
    name: 'Emma',
    id: 'did:test:emma',
    displayName: 'Emma',
    status: 'online',
    hold: { available: true, held: false, sources: [], host: null, agent: null },
}]) {
    const container = document.createElement('div');
    document.body.appendChild(container);
    const handle = mountAgentList(container, {
        adapter: fakeAdapter(items),
        ...config,
    });
    return { container, handle };
}

function menuItem(action) {
    return document.querySelector(`.kebab-menu-item[data-action="${action}"]`);
}

afterEach(() => {
    closeKebabMenu();
    document.body.innerHTML = '';
});

test('idle cards expose Hold through a visible accessible kebab', async () => {
    const calls = [];
    const { container, handle } = mount({
        requestHoldReason: async (item) => {
            calls.push(['reason', item.name]);
            return 'operator pause';
        },
        onHold: async (item, reason) => calls.push(['hold', item.id, reason]),
    });
    await tick();

    const kebab = container.querySelector('.agent-card-kebab');
    assert.ok(kebab, 'component renders a visible kebab');
    assert.equal(kebab.getAttribute('aria-label'), 'Actions for Emma');
    kebab.click();
    assert.equal(menuItem('hold')?.textContent, 'Hold');
    menuItem('hold').click();
    await tick();
    await tick();

    assert.deepEqual(calls, [
        ['reason', 'Emma'],
        ['hold', 'did:test:emma', 'operator pause'],
    ]);
    handle.destroy();
});

test('busy cards offer one Stop and hold action that performs both typed operations', async () => {
    const calls = [];
    const results = [];
    const { container, handle } = mount({
        isThinking: () => true,
        requestHoldReason: async () => 'handoff',
        onStop: async (name) => {
            calls.push(['stop', name]);
            return true;
        },
        onHold: async (item, reason) => {
            calls.push(['hold', item.id, reason]);
            return { receipt: { receipt_id: 'hold-receipt-1' } };
        },
        onLifecycleResult: (result) => results.push(result),
    });
    await tick();

    assert.ok(container.querySelector('.agent-stop-btn'), 'busy card retains direct Stop');
    container.querySelector('.agent-card-kebab').click();
    assert.equal(menuItem('stop-and-hold')?.textContent, 'Stop and hold');
    menuItem('stop-and-hold').click();
    await tick();
    await tick();

    assert.deepEqual(calls, [
        ['stop', 'Emma'],
        ['hold', 'did:test:emma', 'handoff'],
    ], 'Stop and Hold remain distinct calls in causal order');
    assert.deepEqual(results, [{
        stop: true,
        hold: { receipt: { receipt_id: 'hold-receipt-1' } },
    }], 'the two typed operation results remain separate');
    handle.destroy();
});

test('held card shows durable evidence and Resume in the Stop slot', async () => {
    const calls = [];
    const item = {
        name: 'Emma',
        id: 'did:test:emma',
        displayName: 'Emma',
        status: 'online',
        hold: holdState(),
    };
    const { container, handle } = mount({
        onResume: async (target, receiptId) => calls.push([target.id, receiptId]),
    }, [item]);
    await tick();

    const card = container.querySelector('.agent-card');
    assert.ok(card.classList.contains('agent-held'));
    const badge = card.querySelector('.agent-hold-badge');
    assert.ok(badge, 'held evidence is persistent on the card');
    assert.match(badge.textContent, /maintenance window/);
    assert.match(badge.textContent, /sovereign-key/);
    assert.match(badge.textContent, /2026-08-30/);

    const resume = card.querySelector('.agent-resume-btn');
    assert.ok(resume, 'Resume replaces the primary Stop slot');
    assert.equal(card.querySelector('.agent-stop-btn:not(.agent-resume-btn)'), null);
    resume.click();
    await tick();
    assert.deepEqual(calls, [['did:test:emma', 'hold-receipt-1']]);
    handle.destroy();
});

test('right-click opens the exact same Hold menu as the visible kebab', async () => {
    const { container, handle } = mount({
        requestHoldReason: async () => 'pause',
        onHold: async () => {},
    });
    await tick();

    const card = container.querySelector('.agent-card');
    card.dispatchEvent(new dom.window.MouseEvent('contextmenu', {
        bubbles: true,
        cancelable: true,
        clientX: 30,
        clientY: 40,
    }));
    assert.equal(menuItem('hold')?.textContent, 'Hold');
    handle.destroy();
});

test('component-owned Hold controls survive a custom card renderer', async () => {
    const { container, handle } = mount({
        renderCard: (item) => {
            const portrait = document.createElement('div');
            portrait.className = 'portrait-card';
            portrait.textContent = item.displayName;
            return portrait;
        },
        requestHoldReason: async () => 'pause',
        onHold: async () => {},
    });
    await tick();

    assert.ok(container.querySelector('.portrait-card'));
    assert.ok(container.querySelector('.agent-card-kebab'),
        'the lifecycle controls belong to the component, not the console renderer');
    handle.destroy();
});

test('default adapter preserves effective durable Hold state from /api/agents', async () => {
    const durableHold = holdState();
    const adapter = createDefaultAgentAdapter({
        getAgents: async () => ({
            mode: 'multi_agent',
            agents: [{
                id: 'did:test:emma',
                name: 'Emma',
                routing_name: 'Emma',
                status: 'online',
                hold: durableHold,
            }],
        }),
    });

    const [item] = await adapter.listAgents();

    assert.deepEqual(item.hold, durableHold);
    assert.equal(item.hold.agent.hold_receipt_id, 'hold-receipt-1');
});

test('unknown Hold state is visible and cannot be mutated from the card', async () => {
    const item = {
        name: 'Emma',
        id: 'did:test:emma',
        displayName: 'Emma',
        status: 'online',
        hold: { available: false, held: null, sources: [], host: null, agent: null },
    };
    const { container, handle } = mount({
        requestHoldReason: async () => 'pause',
        onHold: async () => assert.fail('unavailable state must not mutate'),
    }, [item]);
    await tick();

    assert.match(container.querySelector('.agent-hold-unavailable').textContent, /unavailable/i);
    assert.equal(container.querySelector('.agent-card-kebab'), null);
    handle.destroy();
});

test('failed post-mutation refresh stays visible instead of repainting stale controls', async () => {
    const item = {
        name: 'Emma',
        id: 'did:test:emma',
        displayName: 'Emma',
        status: 'online',
        hold: { available: true, held: false, sources: [], host: null, agent: null },
    };
    let reads = 0;
    const adapter = {
        mode: 'multi_agent',
        async listAgents() {
            reads += 1;
            if (reads === 1) return [item];
            throw new Error('hold state read failed');
        },
    };
    const { container, handle } = mount({
        adapter,
        requestHoldReason: async () => 'pause',
        onHold: async () => ({ receipt: { receipt_id: 'hold-receipt-1' } }),
    });
    await tick();

    container.querySelector('.agent-card-kebab').click();
    menuItem('hold').click();
    await tick();
    await tick();

    assert.equal(reads, 2);
    assert.match(container.querySelector('.agent-list-error').textContent, /failed to load/i);
    assert.equal(container.querySelector('.agent-card-kebab'), null,
        'stale pre-Hold controls must not replace the refresh error');
    handle.destroy();
});
