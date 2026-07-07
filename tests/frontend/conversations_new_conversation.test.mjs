// #2222: New Conversation is component-owned. Creating a new conversation must
// (a) immediately add a tile to the pane and (b) make it the CURRENT
// conversation (active highlight; subsequent messages land in it). The New
// button lives INSIDE the component (built for bare embed containers, adopted
// from the standalone console's static markup) so late-import embeds get it
// too. These tests exercise the component surface directly:
//   - the pane's New button prepends exactly one active tile BEFORE the
//     reconciling list refetch resolves;
//   - setActiveSessionId moves the highlight; getActiveSessionId seeds it;
//   - built-header embeds get a New button; an adopted static button is wired
//     exactly once (no double-binding);
//   - a later refresh() keeps the marker-backed session present + active.

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
globalThis.confirm = () => true;
globalThis.window.confirm = globalThis.confirm;

function makeStorage() {
    const map = new Map();
    return {
        getItem: (k) => (map.has(k) ? map.get(k) : null),
        setItem: (k, v) => { map.set(k, String(v)); },
        removeItem: (k) => { map.delete(k); },
    };
}
globalThis.localStorage = makeStorage();

const { mountConversations, mountConversationsPane } = await import(
    '../../kestrel_sovereign/static/js/conversations.js'
);

const tick = () => new Promise((r) => setTimeout(r, 0));

function el() {
    const node = document.createElement('div');
    document.body.appendChild(node);
    return node;
}

function conv(id, preview, startedAt = '2026-06-09T15:00:00Z') {
    return {
        session_id: id, preview,
        started_at: startedAt, last_message_at: startedAt, message_count: 2,
    };
}

function rowsIn(container) {
    return Array.from(container.querySelectorAll('.conversation-item'));
}

function newBtnIn(container) {
    return container.querySelector('#new-conversation-sidebar-btn')
        || container.querySelector('.new-conversation-btn');
}

test('the pane New button prepends exactly one active tile BEFORE the reconcile refetch resolves', async () => {
    let getCount = 0;
    let releaseReconcile;
    const gate = new Promise((r) => { releaseReconcile = r; });
    const api = {
        getConversations: async () => {
            getCount += 1;
            // The reconcile refetch (2nd call) is held so we can assert the
            // optimistic tile lands before it resolves.
            if (getCount >= 2) await gate;
            return { conversations: [conv('old-1', 'old row')] };
        },
        listTrash: async () => ({ messages: [] }),
        newConversation: async () => ({
            success: true, session_id: 'new-xyz', started_at: '2026-07-07T10:00:00Z',
        }),
    };
    const container = el();
    const handle = mountConversationsPane(container, {
        api, storageKey: 'k:new-optimistic', collapsed: false,
    });
    await tick(); // initial load (getCount === 1)

    newBtnIn(container).click();
    await tick(); // newConversation resolves + optimistic prepend; reconcile held on gate

    const rows = rowsIn(container);
    const news = rows.filter((r) => r.dataset.sessionId === 'new-xyz');
    assert.equal(news.length, 1, 'exactly one tile for the minted session_id');
    assert.equal(rows[0].dataset.sessionId, 'new-xyz', 'the new tile is prepended (first)');
    assert.ok(rows[0].classList.contains('active'), 'the new tile is the active/current conversation');
    assert.ok(
        !rows.some((r) => r.dataset.sessionId === 'old-1' && r.classList.contains('active')),
        'the previously-listed row is not active',
    );

    releaseReconcile();
    handle.destroy();
});

test('setActiveSessionId moves the highlight; getActiveSessionId seeds it; roundtrip', async () => {
    const api = {
        getConversations: async () => ({ conversations: [conv('a', 'A'), conv('b', 'B')] }),
        listTrash: async () => ({ messages: [] }),
    };
    let active = 'a';
    const container = el();
    const handle = mountConversations(container, {
        api, autoLoad: false, getActiveSessionId: () => active,
    });
    await handle.refresh();

    const byId = (id) => rowsIn(container).find((r) => r.dataset.sessionId === id);
    assert.ok(byId('a').classList.contains('active'), 'getActiveSessionId seeds the highlight on a');
    assert.ok(!byId('b').classList.contains('active'), 'b not active initially');

    // The handle override wins and moves the highlight synchronously.
    handle.setActiveSessionId('b');
    assert.ok(!byId('a').classList.contains('active'), 'previous active tile loses highlight');
    assert.ok(byId('b').classList.contains('active'), 'setActiveSessionId marks b active');

    // Roundtrip back.
    handle.setActiveSessionId('a');
    assert.ok(byId('a').classList.contains('active'), 'roundtrip: a active again');
    assert.ok(!byId('b').classList.contains('active'), 'roundtrip: b no longer active');
    handle.destroy();
});

test('built-header embeds get a New button; an adopted static button is wired exactly once', async () => {
    // --- Built header (bare embed container) ---
    let builtCalls = 0;
    const builtApi = {
        getConversations: async () => ({ conversations: [] }),
        listTrash: async () => ({ messages: [] }),
        newConversation: async () => { builtCalls += 1; return { session_id: `b${builtCalls}` }; },
    };
    const bare = el();
    const h1 = mountConversationsPane(bare, {
        api: builtApi, storageKey: 'k:new-built', autoLoad: false, collapsed: false,
    });
    const built = bare.querySelector('.new-conversation-btn');
    assert.ok(built, 'a bare embed container gets a component-built New button');
    built.click();
    await tick();
    assert.equal(builtCalls, 1, 'built New button drives the new-conversation action');
    h1.destroy();

    // --- Adopted static header (standalone console markup) ---
    let adoptCalls = 0;
    const adoptApi = {
        getConversations: async () => ({ conversations: [] }),
        listTrash: async () => ({ messages: [] }),
        newConversation: async () => { adoptCalls += 1; return { session_id: `x${adoptCalls}` }; },
    };
    const adopt = el();
    adopt.className = 'pane-sidebar';
    adopt.innerHTML = `
        <div class="pane-header">
            <h3 id="conversations-pane-title" class="conversations-pane-title">History</h3>
            <button id="new-conversation-sidebar-btn" class="btn-icon"><span class="ki ki-plus"></span></button>
            <button class="collapse-btn"></button>
        </div>
        <div id="conversations-list" class="pane-content"></div>
        <div class="resize-handle"></div>`;
    const staticBtn = adopt.querySelector('#new-conversation-sidebar-btn');
    const h2 = mountConversationsPane(adopt, {
        api: adoptApi, storageKey: 'k:new-adopt', autoLoad: false, collapsed: false,
    });
    assert.equal(
        adopt.querySelectorAll('#new-conversation-sidebar-btn, .new-conversation-btn').length, 1,
        'no duplicate New button — the static one is adopted, not rebuilt',
    );
    assert.equal(adopt.querySelector('#new-conversation-sidebar-btn'), staticBtn, 'the existing button node is reused');
    staticBtn.click();
    await tick();
    assert.equal(adoptCalls, 1, 'adopted button wired exactly once (no double-binding)');
    h2.destroy();
});

test('a later refresh() keeps the marker-backed session present and active (no vanish/flicker)', async () => {
    // The stubbed server behaves like the real one: creating a conversation
    // makes it list-visible (the session-marker row), so the reconcile refetch
    // and any later refresh() both return it.
    const listRef = { current: [conv('old-1', 'old row')] };
    const api = {
        getConversations: async () => ({ conversations: listRef.current }),
        listTrash: async () => ({ messages: [] }),
        newConversation: async () => {
            const sid = 'mk-1';
            listRef.current = [
                { session_id: sid, preview: '', started_at: '2026-07-07T10:00:00Z', message_count: 0 },
                ...listRef.current,
            ];
            return { success: true, session_id: sid, started_at: '2026-07-07T10:00:00Z' };
        },
    };
    const container = el();
    const handle = mountConversationsPane(container, {
        api, storageKey: 'k:new-marker', collapsed: false,
    });
    await tick(); // initial load

    newBtnIn(container).click();
    await tick(); // optimistic prepend + reconcile refetch (now includes mk-1)

    // A later, independent refresh — the marker-backed session must survive.
    await handle.refresh();
    await tick();

    const rows = rowsIn(container);
    const mk = rows.filter((r) => r.dataset.sessionId === 'mk-1');
    assert.equal(mk.length, 1, 'new session still present after refresh (no vanish, no duplicate)');
    assert.ok(mk[0].classList.contains('active'), 'new session stays the active/current conversation');
    handle.destroy();
});

test('a NULLISH setActiveSessionId clears the override — it never masks a session the host getter learns later (codex P2)', async () => {
    // identity.js's retarget seeds the highlight with its per-agent memory,
    // which is null for an agent with no current session. That seed must NOT
    // pin "nothing active": when chat.js learns the real session id on the
    // first message (surfaced via the config getter), the list must highlight
    // it on the next repaint.
    const api = {
        getConversations: async () => ({ conversations: [conv('sess-1', 'first msg')] }),
        listTrash: async () => ({ messages: [] }),
    };
    let hostActive = null; // agent has no current session yet
    const container = el();
    const handle = mountConversationsPane(container, {
        api, storageKey: 'k:null-seed', collapsed: false,
        getActiveSessionId: () => hostActive,
    });
    await tick();

    handle.setActiveSessionId(null); // the identity.js retarget seed
    hostActive = 'sess-1';           // chat.js learns the session on first message
    await handle.refresh();
    await tick();

    const row = rowsIn(container).find((r) => r.dataset.sessionId === 'sess-1');
    assert.ok(row, 'row rendered');
    assert.ok(
        row.classList.contains('active'),
        'the host-getter session is highlighted — a null seed must not stick as a winning override',
    );
    handle.destroy();
});

test('SOURCE CONTRACT: the chat-toolbar New Chat door converges on the pane path (two doors, one outcome)', async () => {
    // User requirement: BOTH entry points — the pane + button AND the chat
    // toolbar's New Chat (clearChat -> window.startNewConversation) — must
    // produce the same tile+active outcome. The toolbar door converges by
    // routing through window.newConversationViaPane (defined by identity.js,
    // backed by the same component newConversation() the pane button uses).
    // Pin that wiring so a refactor can't silently fork the flows again.
    const { readFileSync } = await import('node:fs');
    const history = readFileSync(
        new URL('../../kestrel_sovereign/static/js/history.js', import.meta.url), 'utf8');
    const identity = readFileSync(
        new URL('../../kestrel_sovereign/static/js/identity.js', import.meta.url), 'utf8');
    const chat = readFileSync(
        new URL('../../kestrel_sovereign/static/js/chat.js', import.meta.url), 'utf8');

    assert.match(chat, /startNewConversation/,
        'clearChat (New Chat button) still routes into startNewConversation');
    assert.match(history, /window\.newConversationViaPane/,
        'startNewConversation prefers the shared-pane path when mounted');
    assert.match(identity, /window\.newConversationViaPane\s*=/,
        'identity.js provides the pane bridge');
    assert.match(identity, /\.newConversation\(\)/,
        'the bridge calls the component-owned newConversation action');
});
