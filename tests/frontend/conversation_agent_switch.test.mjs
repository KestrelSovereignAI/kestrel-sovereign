// #2199: the standalone conversations pane is a `mountConversations` consumer.
// identity.js no longer keeps its own `loadConversations` / request-sequence
// guard — the single list orchestrator (fetch / refresh / seq-guard / views)
// lives in the shared component. These tests exercise the sidebar's remaining
// job: retargeting that mount on an agent switch (#1358 pinning) and routing a
// row click through `window.loadConversation` pinned to the agent the list was
// loaded for.
//
// The seq-guard itself is verified generically in conversations.test.mjs
// ("stale refresh: a slow active-list response never clobbers a newer view
// switch"); here we assert the sidebar drives it correctly and that the
// standalone pane owns NO duplicate guard (grep -c conversationListRequestSeq
// == 0 is a hard acceptance gate).

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
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
globalThis.sessionStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.fetch = async () => ({ ok: false, status: 500, json: async () => ({}) });
globalThis.kicon = () => '';
globalThis.window.kicon = globalThis.kicon;
globalThis.confirm = () => true;
globalThis.window.confirm = globalThis.confirm;
globalThis.window.SharedMarkdown = {
    renderMarkdown: () => '',
    renderStreamingMarkdown: () => '',
    highlightCodeBlocks: () => {},
    renderMermaidDiagrams: () => {},
    finalizeMarkdown: async () => {},
};

const apiModule = await import('../../kestrel_sovereign/static/js/api.js');
const API = apiModule.default;
// The pane is gated on the 'conversations' capability; force it on for the test
// host so refreshConversationsPane mounts instead of hiding the pane.
API.hasCapability = () => true;

const { refreshConversationsPane } = await import('../../kestrel_sovereign/static/js/identity.js');

// The sidebar mounts into `#conversations-list`; the pane wrapper carries the
// #879 hide toggle. Provide both plus a chat container the loader can paint to.
// identity.js keeps ONE module-level mount handle bound to `#conversations-list`,
// so the container must stay stable across tests — set it up once and reuse it.
for (const id of ['conversations-pane', 'conversations-list', 'chat-container']) {
    const el = document.createElement('div');
    el.id = id;
    document.body.appendChild(el);
}

function conversation(sessionId, preview) {
    return {
        session_id: sessionId,
        preview,
        started_at: '2026-06-09T15:00:00Z',
        last_message_at: '2026-06-09T15:00:00Z',
    };
}

function renderedSessionIds() {
    return Array.from(document.querySelectorAll('.conversation-item'))
        .map((row) => row.dataset.sessionId);
}

test('standalone pane keeps NO bespoke request-sequence guard (#2199)', () => {
    const src = readFileSync(
        new URL('../../kestrel_sovereign/static/js/identity.js', import.meta.url),
        'utf8',
    );
    const count = (src.match(/conversationListRequestSeq/g) || []).length;
    assert.equal(count, 0, 'identity.js must not reimplement the list seq-guard');
    assert.equal(
        (src.match(/\bfunction loadConversations\b/g) || []).length, 0,
        'identity.js must not reimplement loadConversations',
    );
});

test('agent switch retargets the mount, dropping stale LIST rows before they render under the new agent', async () => {
    const pending = new Map();
    API.getConversations = () => {
        const agent = API.getHostAgent();
        return new Promise((resolve) => pending.set(agent, resolve));
    };

    // First agent: mount + retarget kicks off a held load.
    API.setHostAgent('Meridian');
    refreshConversationsPane();
    // Switch agent before Meridian's list resolves — retarget bumps the
    // component's refreshSeq so the older response loses.
    API.setHostAgent('Emma');
    refreshConversationsPane();

    // Meridian's list lands late; the seq-guard in the component must drop it.
    pending.get('Meridian')({
        conversations: [conversation('1325', 'Meridian old row'), conversation('1278', 'Meridian older row')],
    });
    await new Promise((r) => setTimeout(r, 0));
    assert.deepEqual(renderedSessionIds(), [], 'stale Meridian rows must not render under Emma');

    // Emma's list resolves and renders.
    pending.get('Emma')({ conversations: [conversation('991', 'Emma row')] });
    await new Promise((r) => setTimeout(r, 0));
    assert.deepEqual(renderedSessionIds(), ['991'], 'Emma pane shows only Emma rows after her list resolves');
});

test('a row click routes through window.loadConversation pinned to the agent the list was loaded for (#1358)', async () => {
    API.getConversations = async () => ({ conversations: [conversation('991', 'Emma row')] });
    API.setHostAgent('Emma');
    refreshConversationsPane();
    await new Promise((r) => setTimeout(r, 0));

    const calls = [];
    const realLoad = window.loadConversation;
    window.loadConversation = (sessionId, options = {}) => calls.push({ sessionId, options });
    try {
        const row = document.querySelector('.conversation-item');
        assert.ok(row, 'Emma row rendered');
        row.click();
        assert.deepEqual(
            calls,
            [{ sessionId: '991', options: { expectedAgent: 'Emma' } }],
            'the row load is pinned to Emma (the list-fetch agent)',
        );
    } finally {
        window.loadConversation = realLoad;
    }
});

test('the pinned expectedAgent makes window.loadConversation drop a stale row under a switched host (#1358/#1604)', async () => {
    API.getConversations = async () => ({ conversations: [conversation('991', 'Emma row')] });
    API.setHostAgent('Emma');
    refreshConversationsPane();
    await new Promise((r) => setTimeout(r, 0));

    const fetched = [];
    API.getConversation = async (sessionId) => { fetched.push(sessionId); return { messages: [] }; };

    const row = document.querySelector('.conversation-item');
    row.click();
    await new Promise((r) => setTimeout(r, 0));
    assert.deepEqual(fetched, ['991'], 'under Emma the pinned load fetches the conversation');

    // Operator switches to a different host; the still-mounted Emma row must
    // not dispatch a GET against Meridian with Emma's session_id.
    API.setHostAgent('Meridian');
    row.click();
    await new Promise((r) => setTimeout(r, 0));
    assert.deepEqual(fetched, ['991'], 'a stale Emma row must not fetch under Meridian routing');
});

test('a stale row clicked DURING the retarget fetch window stays pinned to the old agent (#2199 render-time snapshot)', async () => {
    // Emma's list resolves synchronously so her rows render pinned to Emma.
    API.getConversations = async () => {
        if (API.getHostAgent() === 'Emma') {
            return { conversations: [conversation('991', 'Emma row')] };
        }
        // Meridian's list is HELD — this is the in-flight retarget window.
        return new Promise(() => {});
    };
    API.setHostAgent('Emma');
    refreshConversationsPane();
    await new Promise((r) => setTimeout(r, 0));

    const fetched = [];
    API.getConversation = async (sessionId) => { fetched.push(sessionId); return { messages: [] }; };

    // Switch to Meridian: retarget() sets the component's agentName to Meridian
    // synchronously, then fires an async refresh whose fetch never resolves.
    // Emma's rows remain in the DOM and clickable during that window.
    API.setHostAgent('Meridian');
    refreshConversationsPane();

    const row = document.querySelector('.conversation-item');
    assert.ok(row, 'Emma row still rendered during Meridian retarget fetch window');
    row.click();
    await new Promise((r) => setTimeout(r, 0));
    // The row was rendered for Emma; the render-time snapshot pins its load to
    // Emma even though the list agentName has already flipped to Meridian. With
    // the host now Meridian, loadConversation's agent gate drops it — no
    // cross-agent load of Emma's conversation under Meridian routing.
    assert.deepEqual(fetched, [], 'a stale Emma row clicked mid-retarget must not fetch under Meridian');
});
