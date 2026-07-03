// #2149: the history slideout no longer owns any conversation-list or rename
// logic. `loadConversationHistory` now mounts the shared conversations.js
// component into `#history-container`, and the old prompt()-based
// `window.renameConversation` path is retired (the component's inline rename is
// the single rename implementation). These tests pin that consolidation:
//   - loadConversationHistory renders the shared component's rows (with kebab)
//     into the history container, sourced from API.getConversations
//   - no second rename implementation survives on history.js
//     (window.renameConversation is gone; prompt() is never called)

import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const dom = new JSDOM('<!doctype html><html><body><div id="history-container"></div></body></html>', {
    url: 'http://localhost/',
});
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
globalThis.kicon = (name) => `<span class="ki ki-${name}"></span>`;
globalThis.window.kicon = globalThis.kicon;
let promptCalls = 0;
globalThis.prompt = () => { promptCalls += 1; return null; };
globalThis.window.prompt = globalThis.prompt;

const apiModule = await import('../../kestrel_sovereign/static/js/api.js');
const { loadConversationHistory } = await import('../../kestrel_sovereign/static/js/history.js');

test('history slideout mounts the shared conversation-list component', async () => {
    apiModule.default.getConversations = async () => ({
        encrypted_at_rest: false,
        conversations: [
            {
                session_id: 'sess-1', name: 'Debugging Thread', preview: 'first user message',
                started_at: '2026-06-23T12:00:00Z', message_count: 3,
            },
            {
                session_id: 'sess-2', preview: 'unnamed preview',
                started_at: '2026-06-23T13:00:00Z', message_count: 1,
            },
        ],
    });

    await loadConversationHistory();
    // The mount's refresh resolves on the getConversations microtask.
    await new Promise((r) => setTimeout(r, 0));

    const container = document.getElementById('history-container');
    assert.match(container.innerHTML, /Debugging Thread/);
    assert.match(container.innerHTML, /unnamed preview/);
    // Shared component markers: kebab overflow button + the .conversation-item
    // rows produced by conversations.js (NOT history.js's old bespoke list).
    assert.ok(container.querySelector('.conversation-item'), 'rows come from the shared component');
    assert.ok(container.querySelector('.kebab-btn'), 'rows carry the shared kebab menu');
});

test('the retired prompt()-based rename path no longer exists on history.js', () => {
    // history.js used to define a SECOND rename implementation using prompt().
    // The consolidation deletes it; the component owns inline rename now.
    assert.equal(typeof globalThis.window.renameConversation, 'undefined',
        'window.renameConversation must be gone');
    assert.equal(promptCalls, 0, 'no prompt() rename path runs during history render');
});
