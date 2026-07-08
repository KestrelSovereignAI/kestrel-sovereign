// #2248: the conversations pane's new-conversation tile stayed "0 msgs" while
// the user chatted in it. Two bugs, fixed together:
//   (a) No post-turn refresh — chat.js never told the pane a turn completed, so
//       the optimistic tile (hardcoded message_count:0) never learned the count.
//   (b) Fragile session anchoring — startNewConversationForPane set the current
//       session id everywhere EXCEPT pane.sessionId (which the send path reads),
//       so the first turn relied on an implicit last-message derive race.
//
// These tests pin: (1) the pane reloads on the `kestrel:conversations-stale`
// window event chat.js now fires on turn end (so the tile's count + preview
// learn the exchange), and the SOURCE CONTRACTs that (2) chat.js dispatches
// that event when a turn completes and (3) identity.js sets the explicit
// pane.sessionId when minting a new conversation.

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
globalThis.window.kicon = (name) => `<span class="ki ki-${name}" aria-hidden="true"></span>`;
globalThis.kicon = globalThis.window.kicon;

function makeStorage() {
    const map = new Map();
    return {
        getItem: (k) => (map.has(k) ? map.get(k) : null),
        setItem: (k, v) => { map.set(k, String(v)); },
        removeItem: (k) => { map.delete(k); },
    };
}
globalThis.localStorage = makeStorage();

const { mountConversationsPane } = await import(
    '../../kestrel_sovereign/static/js/conversations.js'
);

const tick = () => new Promise((r) => setTimeout(r, 0));

function el() {
    const node = document.createElement('div');
    document.body.appendChild(node);
    return node;
}

function msgCountText(container, sid) {
    const row = Array.from(container.querySelectorAll('.conversation-item'))
        .find((r) => r.dataset.sessionId === sid);
    return row ? row.querySelector('.conversation-msg-count')?.textContent : null;
}

test('the pane reloads on kestrel:conversations-stale — the active tile learns the new message_count + preview', async () => {
    // The server-side count/preview grow as the user chats; the optimistic tile
    // starts at 0 msgs. A completed turn fires the stale event, the (identity.js)
    // listener calls handle.refresh(), and the tile must pick up the new values.
    let served = { session_id: 'sess-1', preview: '', message_count: 0, started_at: '2026-07-08T10:00:00Z' };
    const api = {
        getConversations: async () => ({ conversations: [{ ...served, last_message_at: served.started_at }] }),
        listTrash: async () => ({ messages: [] }),
    };
    const container = el();
    const handle = mountConversationsPane(container, {
        api, storageKey: 'k:turn-refresh', collapsed: false,
    });
    await tick();
    assert.equal(msgCountText(container, 'sess-1'), '0 msgs', 'tile starts at 0 msgs');

    // Mirror identity.js's listener wiring so this proves the end-to-end path.
    window.addEventListener('kestrel:conversations-stale', () => { handle.refresh(); });

    // A turn completes server-side: two more messages, a preview.
    served = { session_id: 'sess-1', preview: 'hello there', message_count: 2, started_at: '2026-07-08T10:00:00Z' };
    window.dispatchEvent(new window.Event('kestrel:conversations-stale'));
    await tick();
    await tick();

    assert.equal(msgCountText(container, 'sess-1'), '2 msgs',
        'the tile refreshed to the server-side count (no longer stuck at 0)');
    const row = Array.from(container.querySelectorAll('.conversation-item'))
        .find((r) => r.dataset.sessionId === 'sess-1');
    assert.match(row.textContent, /hello there/, 'refresh() also corrected the preview text');
    handle.destroy();
});

test('SOURCE CONTRACT: chat.js signals the conversations pane on turn completion', async () => {
    const chat = readFileSync(
        new URL('../../kestrel_sovereign/static/js/chat.js', import.meta.url), 'utf8');

    assert.match(chat, /kestrel:conversations-stale/,
        'chat.js fires the conversations-stale signal so the pane refreshes after a turn');
    // The signal must live in the turn-teardown path (the finally block) gated
    // on the owning, non-aborted turn — not in some unrelated helper.
    assert.match(chat, /ownsStream\(\)\s*&&\s*!wasAborted\)\s*\{[\s\S]*?notifyConversationsStale\(/s,
        'the stale signal fires from the owning, non-aborted turn on completion');
});

test('SOURCE CONTRACT: identity.js explicitly anchors the new conversation to pane.sessionId', async () => {
    const identity = readFileSync(
        new URL('../../kestrel_sovereign/static/js/identity.js', import.meta.url), 'utf8');

    // startNewConversationForPane must set pane.sessionId (not only the current-
    // session getters) so the first turn is anchored without the implicit-derive
    // race the send path (pane.sessionId) would otherwise lose.
    const start = identity.indexOf('function startNewConversationForPane');
    const fn = identity.slice(start, identity.indexOf('\n}', start));
    assert.match(fn, /pane\.sessionId\s*=\s*sid/,
        'startNewConversationForPane sets the host pane.sessionId to the minted session');
});
