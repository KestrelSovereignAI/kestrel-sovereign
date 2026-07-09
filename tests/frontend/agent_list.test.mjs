// #2278: `mountAgentList` is the shared agent-list surface — adapter-fed fetch,
// an owned card shell, a pluggable `renderCard(item, ctx)` hook (default =
// console row), the shared host-agent selection path (setHostAgent in
// multi-agent mode ONLY), `select()` / `setActiveName()` on the handle, a
// per-card `agent-card-actions` slot via the ui-ext registry, refresh, and the
// active highlight. These tests exercise the contract directly, plus a
// source-contract check that identity.js no longer hand-rolls the agent loop.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { JSDOM } from 'jsdom';

const here = dirname(fileURLToPath(import.meta.url));

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

const { mountAgentList, createDefaultAgentAdapter } = await import(
    '../../kestrel_sovereign/static/js/agent_list.js'
);
const { UI } = await import('../../kestrel_sovereign/static/js/ui-ext/registry.js');

function tick() { return new Promise((r) => setTimeout(r, 0)); }

function fakeAdapter(items, mode = 'multi_agent') {
    return { mode, listAgents: async () => items };
}

function mountInto(config) {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const handle = mountAgentList(el, config);
    return { el, handle };
}

test('adapter feed → rows render with the default console-row renderer', async () => {
    const { el, handle } = mountInto({
        adapter: fakeAdapter([
            { name: 'Emma', description: 'orchestrator', status: 'online' },
            { name: 'Nellie', description: 'verifier', status: 'offline' },
        ]),
    });
    await tick();

    const rows = el.querySelectorAll('.agent-card');
    assert.equal(rows.length, 2, 'one card per adapter item');
    assert.ok(rows[0].classList.contains('agent-item'), 'default renderer tags the shell .agent-item');
    assert.equal(rows[0].dataset.agentName, 'Emma');
    assert.equal(el.querySelector('.agent-name').textContent, 'Emma');
    assert.equal(el.querySelector('.agent-description').textContent, 'orchestrator');
    assert.ok(el.querySelector('.agent-status-dot.online'), 'status dot rendered by default');
    assert.ok(rows[1].classList.contains('offline'), 'offline agent gets the offline class');
    assert.ok(el.querySelector('.agent-stop-btn'), 'console row has the stop control');
    handle.destroy();
});

test('renderCard override replaces the body and receives ctx (actionsAnchor + escapeHtml)', async () => {
    let seenCtx = null;
    const { el, handle } = mountInto({
        adapter: fakeAdapter([{ name: 'Frin', status: 'online', raw: { mood: 'happy' } }]),
        renderCard: (item, ctx) => {
            seenCtx = ctx;
            const body = document.createElement('div');
            body.className = 'portrait-card';
            body.textContent = item.raw.mood;
            body.appendChild(ctx.actionsAnchor); // host positions the anchor
            return body;
        },
    });
    await tick();

    assert.ok(el.querySelector('.portrait-card'), 'host body used');
    assert.equal(el.querySelector('.portrait-card').firstChild.textContent, 'happy');
    assert.equal(el.querySelectorAll('.agent-name').length, 0, 'console markup not rendered');
    // A custom renderer gets a clean shell (no `.agent-item` console layout).
    assert.ok(!el.querySelector('.agent-card').classList.contains('agent-item'));
    assert.ok(seenCtx && seenCtx.actionsAnchor, 'ctx carries the actionsAnchor');
    assert.equal(typeof seenCtx.escapeHtml, 'function', 'ctx carries escapeHtml');
    assert.equal(seenCtx.standalone, false);
    // The host placed the anchor inside its body — the component must not
    // re-append it to the shell.
    assert.equal(seenCtx.actionsAnchor.parentNode.className, 'portrait-card');
    handle.destroy();
});

test('selection fires setHostAgent ONLY in multi-agent mode', async () => {
    const calls = [];
    const api = { setHostAgent: (n) => calls.push(n) };
    const selected = [];

    // multi-agent: click pins routing then fires onSelect.
    const multi = mountInto({
        api,
        adapter: fakeAdapter([{ name: 'Emma', status: 'online' }], 'multi_agent'),
        onSelect: (item, meta) => selected.push([item.name, meta.standalone]),
    });
    await tick();
    multi.el.querySelector('.agent-card').dispatchEvent(new dom.window.Event('click'));
    assert.deepEqual(calls, ['Emma'], 'setHostAgent called on multi-agent select');
    assert.deepEqual(selected, [['Emma', false]], 'onSelect fired with standalone=false');
    multi.handle.destroy();

    // standalone: rows are NOT clickable, and programmatic select() must NOT
    // install a host-agent prefix (it 404s the un-prefixed routes) — but still
    // fires onSelect.
    calls.length = 0; selected.length = 0;
    const solo = mountInto({
        api,
        adapter: fakeAdapter([{ name: 'Solo', status: 'online' }], 'standalone'),
        onSelect: (item, meta) => selected.push([item.name, meta.standalone]),
    });
    await tick();
    solo.el.querySelector('.agent-card').dispatchEvent(new dom.window.Event('click'));
    assert.deepEqual(calls, [], 'standalone rows are not click-selectable');
    solo.handle.select('Solo');
    assert.deepEqual(calls, [], 'programmatic select does not setHostAgent in standalone');
    assert.deepEqual(selected, [['Solo', true]], 'onSelect still fires with standalone=true');
    solo.handle.destroy();
});

test('setActiveName repaints the highlight only — no setHostAgent, no onSelect', async () => {
    const calls = [];
    const selected = [];
    const { el, handle } = mountInto({
        api: { setHostAgent: (n) => calls.push(n) },
        adapter: fakeAdapter([
            { name: 'Emma', status: 'online' },
            { name: 'Nellie', status: 'online' },
        ]),
        onSelect: (item) => selected.push(item.name),
    });
    await tick();

    handle.setActiveName('Nellie');
    assert.deepEqual(calls, [], 'setActiveName does not pin routing');
    assert.deepEqual(selected, [], 'setActiveName does not fire onSelect');
    const nellie = el.querySelector('[data-agent-name="Nellie"]');
    const emma = el.querySelector('[data-agent-name="Emma"]');
    assert.ok(nellie.classList.contains('selected'), 'active row highlighted');
    assert.ok(!emma.classList.contains('selected'), 'other row not highlighted');
    handle.destroy();
});

test('the agent-card-actions slot renders into a per-card anchor', async () => {
    const seenAgents = [];
    UI.register({
        slot: 'agent-card-actions',
        id: 'test-actions',
        render: (container, ctx) => {
            seenAgents.push(ctx.agentName);
            const btn = document.createElement('button');
            btn.className = 'voice-marker';
            container.appendChild(btn);
        },
    });

    const { el, handle } = mountInto({
        adapter: fakeAdapter([
            { name: 'Emma', status: 'online' },
            { name: 'Nellie', status: 'online' },
        ], 'multi_agent'),
    });
    await tick();

    const anchors = el.querySelectorAll('.agent-card-actions[data-slot="agent-card-actions"]');
    assert.equal(anchors.length, 2, 'one actions anchor per card');
    assert.equal(el.querySelectorAll('.voice-marker').length, 2, 'slot rendered into each anchor');
    assert.deepEqual(seenAgents.sort(), ['Emma', 'Nellie'], 'agentName passed per card');

    UI.unregister('agent-card-actions', 'test-actions');
    handle.destroy();
});

test('default adapter maps /api/agents fields and resolves avatar_hash', async () => {
    const api = {
        getAgents: async () => ({
            mode: 'multi_agent',
            server_demo_mode: false,
            agents: [{ name: 'Emma', description: 'd', status: 'online', avatar_hash: 'abc', is_demo: false }],
        }),
    };
    const adapter = createDefaultAgentAdapter(api);
    const items = await adapter.listAgents();
    assert.equal(items.length, 1);
    assert.equal(items[0].name, 'Emma');
    assert.equal(items[0].avatarUrl, '/api/files/abc');
    assert.equal(items[0].isDemo, false);
    assert.equal(items[0].raw.description, 'd', 'raw source record rides along');
    assert.equal(adapter.mode, 'multi_agent', 'adapter.mode mirrors the response');
});

test('source-contract: identity.js drives mountAgentList, no hand-rolled agent loop', () => {
    const src = readFileSync(
        resolve(here, '../../kestrel_sovereign/static/js/identity.js'),
        'utf8',
    );
    assert.match(src, /import \{ mountAgentList, createDefaultAgentAdapter \} from '\.\/agent_list\.js'/);
    assert.match(src, /mountAgentList\(container, \{/, 'loadAgents mounts the component');
    // The bespoke per-agent innerHTML loop is gone (the tell-tale inline markup).
    assert.ok(!src.includes('class="agent-status-dot'), 'no hand-rolled status-dot markup');
    assert.ok(!/for \(const agent of agents\)/.test(src), 'no hand-rolled agent loop');
});
