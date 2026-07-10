// #2351: the standalone console's "+ New" agent affordance opens a Create Agent
// dialog that POSTs to `/api/agents` (a fresh top-level, parentless agent —
// distinct from Spawn's child-of-a-parent). These tests exercise the dialog
// contract directly against the SHARED Modal (so overlay-root rendering is real):
//   - a valid name POSTs to /api/agents, then refreshes + selects on success;
//   - a 409 duplicate-name failure renders inline in the dialog (not a toast);
//   - a client-invalid name is rejected before any POST;
//   - the secondary "Spawn a child agent…" link routes to the spawn flow.

import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.Node = dom.window.Node;
globalThis.HTMLElement = dom.window.HTMLElement;
globalThis.window.kicon = (name) => `<span class="ki ki-${name}" aria-hidden="true"></span>`;
globalThis.kicon = globalThis.window.kicon;

const { Modal } = await import('../../kestrel_sovereign/static/js/ui.js');
const { openCreateAgentDialog } = await import('../../kestrel_sovereign/static/js/new_agent_dialog.js');

function tick() { return new Promise((r) => setTimeout(r, 0)); }

function typeName(value) {
    const input = document.getElementById('create-agent-name-input');
    assert.ok(input, 'name input rendered');
    input.value = value;
    return input;
}

function clickCreate() {
    const btns = [...document.querySelectorAll('.modal-btn')];
    const create = btns.find((b) => b.textContent === 'Create');
    assert.ok(create, 'Create button rendered');
    create.click();
}

function errorText() {
    const el = document.getElementById('create-agent-error');
    return el && el.style.display !== 'none' ? el.textContent : '';
}

test('valid name POSTs to /api/agents, then refreshes + selects on success', async () => {
    Modal.hide();
    const posted = [];
    const refreshed = [];
    const selected = [];
    const api = {
        createAgent: async (name) => { posted.push(name); return { success: true, agent: { name } }; },
    };
    openCreateAgentDialog({
        modal: Modal,
        api,
        onCreated: async (name) => { refreshed.push('refresh'); selected.push(name); },
    });
    await tick();
    typeName('Kestrel');
    clickCreate();
    await tick();
    await tick();

    assert.deepEqual(posted, ['Kestrel'], 'POST /api/agents called with the typed name');
    assert.deepEqual(refreshed, ['refresh'], 'list refreshed on success');
    assert.deepEqual(selected, ['Kestrel'], 'new agent selected on success');
    assert.equal(document.getElementById('modal-overlay'), null, 'dialog closed on success');
});

test('a 409 duplicate-name failure renders inline (not a toast)', async () => {
    Modal.hide();
    const err = new Error("Agent 'Kestrel' already exists.");
    err.status = 409;
    err.body = { detail: "Agent 'Kestrel' already exists." };
    const api = { createAgent: async () => { throw err; } };
    let createdCalled = false;
    openCreateAgentDialog({
        modal: Modal,
        api,
        onCreated: async () => { createdCalled = true; },
    });
    await tick();
    typeName('Kestrel');
    clickCreate();
    await tick();
    await tick();

    assert.match(errorText(), /already exists/, 'the 409 detail is shown inline in the dialog');
    assert.equal(createdCalled, false, 'no refresh/select on failure');
    assert.ok(document.getElementById('modal-overlay'), 'dialog stays open so the name can be corrected');
});

test('a client-invalid name is rejected before any POST', async () => {
    Modal.hide();
    let posted = false;
    const api = { createAgent: async () => { posted = true; return {}; } };
    openCreateAgentDialog({ modal: Modal, api, onCreated: async () => {} });
    await tick();
    typeName('1bad name');
    clickCreate();
    await tick();

    assert.equal(posted, false, 'no POST for a client-invalid name');
    assert.match(errorText(), /must start with a letter/, 'inline validation message shown');
});

test('the secondary spawn link routes to the spawn flow', async () => {
    Modal.hide();
    let spawned = false;
    openCreateAgentDialog({
        modal: Modal,
        api: { createAgent: async () => ({}) },
        onCreated: async () => {},
        spawnAvailable: true,
        onSpawn: () => { spawned = true; },
    });
    await tick();
    const link = document.getElementById('create-agent-spawn-link');
    assert.ok(link, 'spawn link rendered when spawnAvailable');
    link.click();
    await tick();
    assert.equal(spawned, true, 'clicking the spawn link runs the spawn flow');
    assert.equal(document.getElementById('modal-overlay'), null, 'dialog closes when routing to spawn');
});

test('the spawn link is absent when the spawn capability is not present', async () => {
    Modal.hide();
    openCreateAgentDialog({
        modal: Modal,
        api: { createAgent: async () => ({}) },
        onCreated: async () => {},
        spawnAvailable: false,
    });
    await tick();
    assert.equal(document.getElementById('create-agent-spawn-link'), null, 'no spawn link without the capability');
    Modal.hide();
});

test('a pending create resolved AFTER dismissal never hides an unrelated modal or paints stale errors (codex P2)', async () => {
    Modal.hide();
    let rejectCreate;
    const api = {
        createAgent: () => new Promise((_, rej) => { rejectCreate = rej; }),
    };
    openCreateAgentDialog({ modal: Modal, api, onCreated: () => {} });
    await tick();
    typeName('Kestrel');
    clickCreate();          // submit hangs on the pending promise
    Modal.hide();           // user dismisses mid-flight...
    Modal.show({ title: 'Unrelated', content: '<p id="unrelated-marker">other</p>', buttons: [] });
    assert.ok(document.getElementById('unrelated-marker'), 'unrelated modal open');

    // The stale request fails now — it must NOT touch the unrelated modal.
    rejectCreate(new Error('boom'));
    await tick();
    await tick();
    assert.ok(document.getElementById('unrelated-marker'),
        'unrelated modal untouched by the stale failure');
    const err = document.getElementById('create-agent-error');
    assert.ok(!err || err.textContent === '',
        'no stale error painted into a foreign modal');
    Modal.hide();
});
