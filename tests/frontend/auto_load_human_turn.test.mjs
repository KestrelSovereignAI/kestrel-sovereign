// #3653: the chat console auto-opened the conversation with the newest
// `last_message_at`. For an agent that chains its own work that is its
// autonomous chain session (#3429: one session per chain), so the operator saw
// the agent's wake log every time instead of their own conversation.
//
// These drive the real sidebar (identity.js -> conversations.js -> the #714
// auto-load) with a stubbed list, so the picker is exercised where it runs: on
// the list a page load fetched and on the list an agent switch fetched, which
// both arrive through refreshConversationsPane and the component's `onLoaded`.
//
// Its own file because identity.js keeps module-level state (the one mount
// handle) that the other sidebar test files also drive.

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
API.hasCapability = () => true;

const { getOrCreateChatPane } = await import('../../kestrel_sovereign/static/js/ui.js');
const { wipeAgentChatPane } = await import('../../kestrel_sovereign/static/js/chat.js');
const { refreshConversationsPane, initTrashToggle } = await import('../../kestrel_sovereign/static/js/identity.js');

for (const id of ['conversations-pane', 'conversations-list', 'chat-container']) {
    const el = document.createElement('div');
    el.id = id;
    document.body.appendChild(el);
}
const trashButton = document.createElement('button');
trashButton.id = 'trash-toggle-btn';
document.body.appendChild(trashButton);
initTrashToggle();

function chainSession(sessionId, lastMessageAt) {
    // As the list endpoint serializes a wake-only autonomous session.
    return {
        session_id: sessionId,
        preview: 'Autonomous wake — talon.job_complete',
        has_human_turn: false,
        started_at: lastMessageAt,
        last_message_at: lastMessageAt,
    };
}

function operatorSession(sessionId, lastMessageAt) {
    return {
        session_id: sessionId,
        preview: 'can you check the deploy',
        has_human_turn: true,
        started_at: lastMessageAt,
        last_message_at: lastMessageAt,
    };
}

// Newest first, as the server pages them. In every list the newest session is
// the agent's autonomous chain.
const LISTS = {
    Emma: [
        chainSession('emma-chain', '2026-10-15T15:06:00Z'),
        operatorSession('emma-operator', '2026-10-14T13:40:00Z'),
        chainSession('emma-older-chain', '2026-10-09T08:00:00Z'),
    ],
    Meridian: [
        chainSession('meridian-chain', '2026-10-15T12:00:00Z'),
        operatorSession('meridian-operator', '2026-10-13T09:00:00Z'),
    ],
    Kite: [
        chainSession('kite-chain', '2026-10-15T14:00:00Z'),
        chainSession('kite-older-chain', '2026-10-12T14:00:00Z'),
    ],
};

// Agents whose list runs past page one, as the server pages it: each page is
// keyed by the cursor that asks for it ('' is page one) and carries the token
// for the one after. The default page size is 50, so an agent with 50 newer
// chain sessions has its operator conversation on page two (#3653).
const PAGED = {
    Wren: {
        '': {
            conversations: [
                chainSession('wren-chain-1', '2026-10-15T15:00:00Z'),
                chainSession('wren-chain-2', '2026-10-15T14:00:00Z'),
            ],
            next_cursor: 'wren-p2',
        },
        'wren-p2': {
            conversations: [
                chainSession('wren-chain-3', '2026-10-15T13:00:00Z'),
                operatorSession('wren-operator', '2026-10-14T13:40:00Z'),
            ],
            next_cursor: 'wren-p3',
        },
        'wren-p3': {
            conversations: [operatorSession('wren-older-operator', '2026-10-01T09:00:00Z')],
            next_cursor: null,
        },
    },
    Lark: {
        '': { conversations: [chainSession('lark-chain-1', '2026-10-15T15:00:00Z')], next_cursor: 'lark-p2' },
        'lark-p2': { conversations: [chainSession('lark-chain-2', '2026-10-14T15:00:00Z')], next_cursor: 'lark-p3' },
        'lark-p3': { conversations: [chainSession('lark-chain-3', '2026-10-13T15:00:00Z')], next_cursor: null },
    },
    Finch: {
        '': { conversations: [chainSession('finch-chain', '2026-10-15T15:00:00Z')], next_cursor: 'finch-p2' },
        'finch-p2': { conversations: [operatorSession('finch-operator', '2026-10-14T09:00:00Z')], next_cursor: null },
    },
    Robin: {
        '': { conversations: [chainSession('robin-chain', '2026-10-15T15:00:00Z')], next_cursor: 'robin-p2' },
        'robin-p2': { conversations: [operatorSession('robin-operator', '2026-10-14T09:00:00Z')], next_cursor: null },
    },
};

// Every list request, with the agent it was routed to.
const requests = [];
// Cursor -> a gate the request for that page waits on, to hold a page in flight.
const held = new Map();

function holdPage(cursor) {
    let release;
    const gate = new Promise((resolve) => { release = resolve; });
    held.set(cursor, { gate, release });
    return () => { held.delete(cursor); release(); };
}

API.getConversations = async (decrypt, view, cursor = null) => {
    const agent = API.getHostAgent();
    requests.push({ agent, view, cursor });
    const hold = cursor && held.get(cursor);
    if (hold) await hold.gate;
    if (PAGED[agent]) return PAGED[agent][cursor || ''];
    return { conversations: LISTS[agent] || [], next_cursor: null };
};

async function settle() {
    for (let i = 0; i < 10; i += 1) await new Promise((r) => setTimeout(r, 0));
}

const loads = [];
window.loadConversation = (sessionId, options = {}) => {
    loads.push({ sessionId, options });
};

async function showAgent(agentName) {
    // A cold pane: nothing streamed, nothing typed, no session claimed.
    getOrCreateChatPane(agentName);
    API.setHostAgent(agentName);
    refreshConversationsPane();
    await settle();
}

function renderedSessionIds() {
    return Array.from(document.querySelectorAll('.conversation-item'))
        .map((row) => row.dataset.sessionId);
}

test('page load opens the newest conversation a person took part in, not the newer chain', async () => {
    loads.length = 0;
    await showAgent('Emma');
    assert.deepEqual(
        loads.map((load) => load.sessionId),
        ['emma-operator'],
        'the auto-load opened the operator conversation',
    );
    assert.equal(loads[0].options.auto, true, 'it is the cold-pane auto-load');
    assert.equal(loads[0].options.expectedAgent, 'Emma');
});

test('an agent switch uses the same picker as the page load', async () => {
    loads.length = 0;
    await showAgent('Meridian');
    assert.deepEqual(
        loads.map((load) => load.sessionId),
        ['meridian-operator'],
        'switching agents skips the newer chain exactly as a page load does',
    );
    assert.equal(loads[0].options.auto, true);
    assert.equal(loads[0].options.expectedAgent, 'Meridian');
});

test('an agent whose sessions are all autonomous opens nothing', async () => {
    loads.length = 0;
    await showAgent('Kite');
    assert.deepEqual(loads, [], 'no fallback to an autonomous session');
    assert.equal(
        getOrCreateChatPane('Kite').element.children.length, 0,
        'the pane stays empty, as for a new agent',
    );
});

test('autonomous sessions stay listed and open on click', async () => {
    loads.length = 0;
    await showAgent('Kite');
    assert.deepEqual(renderedSessionIds(), ['kite-chain', 'kite-older-chain']);

    const row = document.querySelector('.conversation-item[data-session-id="kite-chain"]');
    assert.ok(row, 'the chain session is listed');
    row.click();
    assert.deepEqual(
        loads,
        [{ sessionId: 'kite-chain', options: { expectedAgent: 'Kite' } }],
        'a click opens the autonomous session like any other',
    );
});

// ── Past page one ─────────────────────────────────────────────────────────

test('a conversation a person took part in on page two is opened', async () => {
    loads.length = 0;
    requests.length = 0;
    await showAgent('Wren');
    assert.deepEqual(
        loads.map((load) => load.sessionId),
        ['wren-operator'],
        'the newest eligible session, found by following the cursor',
    );
    assert.equal(loads[0].options.auto, true);
    assert.equal(loads[0].options.expectedAgent, 'Wren');
    assert.equal(
        loads[0].options.expectedGeneration, getOrCreateChatPane('Wren').generation,
        'pinned to the pane context the walk started in',
    );
    assert.deepEqual(
        requests.map((r) => [r.agent, r.view, r.cursor]),
        [['Wren', 'active', null], ['Wren', 'active', 'wren-p2']],
        'stops at the first page holding an eligible session',
    );
    assert.deepEqual(
        renderedSessionIds(), ['wren-chain-1', 'wren-chain-2'],
        'the pages read to decide are not painted into the list',
    );
});

test('an agent with no human conversation on any page opens nothing', async () => {
    loads.length = 0;
    requests.length = 0;
    await showAgent('Lark');
    assert.deepEqual(loads, [], 'no fallback to an autonomous session');
    assert.deepEqual(
        requests.map((r) => r.cursor),
        [null, 'lark-p2', 'lark-p3'],
        'the whole list was searched, and the walk ended with it',
    );
});

test('an agent switch while an older page is in flight abandons the walk', async () => {
    loads.length = 0;
    requests.length = 0;
    const release = holdPage('finch-p2');
    await showAgent('Finch');
    assert.ok(
        requests.some((r) => r.agent === 'Finch' && r.cursor === 'finch-p2'),
        'the walk is fetching page two',
    );
    await showAgent('Kite');
    release();
    await settle();
    assert.deepEqual(
        loads, [],
        'neither the previous agent\'s conversation nor an autonomous one is opened',
    );
});

test('a wipe of the pane while an older page is in flight abandons the walk', async () => {
    loads.length = 0;
    requests.length = 0;
    const release = holdPage('robin-p2');
    await showAgent('Robin');
    assert.ok(requests.some((r) => r.cursor === 'robin-p2'), 'the walk is fetching page two');
    // New Chat (or any deliberate wipe) bumps the pane's generation and leaves
    // it empty, exactly the cold state the auto-load fires on.
    wipeAgentChatPane('Robin');
    release();
    await settle();
    assert.deepEqual(loads, [], 'the conversation the user cleared away is not reopened');
});

for (const replacement of ['trash', 'refresh', 'failed-refresh']) {
    test(`a pending ${replacement} invalidates the old walk before its replacement loads`, async () => {
        loads.length = 0;
        requests.length = 0;
        const releaseOld = holdPage('finch-p2');
        const originalGet = API.getConversations;
        const originalTrash = API.listTrash;
        let releaseReplacement;
        let rejectReplacement;
        const pending = new Promise((resolve, reject) => {
            releaseReplacement = resolve;
            rejectReplacement = reject;
        });
        try {
            await showAgent('Finch');
            assert.ok(requests.some(r => r.cursor === 'finch-p2'));
            if (replacement === 'trash') {
                API.listTrash = () => pending;
                trashButton.click();
                assert.equal(trashButton.dataset.mode, 'trash');
            } else {
                API.getConversations = (decrypt, view, cursor = null) => cursor
                    ? originalGet(decrypt, view, cursor) : pending;
                refreshConversationsPane();
            }
            releaseOld();
            await settle();
            assert.deepEqual(loads, [], 'the superseded active-list walk opened during replacement load');
            if (replacement === 'failed-refresh') rejectReplacement(new Error('offline replacement refused'));
            else releaseReplacement(replacement === 'trash'
                ? { messages: [] } : { conversations: [], next_cursor: null });
            await settle();
            assert.deepEqual(loads, [], 'a failed or empty replacement resurrected the retired walk');
        } finally {
            releaseOld();
            releaseReplacement({ messages: [], conversations: [], next_cursor: null });
            await settle();
            API.getConversations = originalGet;
            API.listTrash = originalTrash;
            if (trashButton.dataset.mode === 'trash') {
                trashButton.click();
                await settle();
            }
        }
    });
}
