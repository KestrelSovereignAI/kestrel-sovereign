// EMBED-CONTEXT regression for the #2250/#2254 turn-end sync (user report:
// selected conversation card STILL not highlighted in the frinz embed after
// #2257 shipped). Root cause: the `kestrel:conversations-stale` listener was
// registered inside identity.js's DOMContentLoaded handler — embedding hosts
// dynamically import the module long AFTER DOMContentLoaded fired, so the
// wiring never ran there. The sibling test file masked it by manually
// re-dispatching DOMContentLoaded after import; this file deliberately does
// NOT — it exercises exactly what an embed gets: module import, no lifecycle
// event, then the turn-end event must still sync the highlight + refresh.

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

const { refreshConversationsPane } = await import('../../kestrel_sovereign/static/js/identity.js');

for (const id of ['conversations-pane', 'conversations-list', 'chat-container']) {
    const node = document.createElement('div');
    node.id = id;
    document.body.appendChild(node);
}

// NO DOMContentLoaded dispatch — this is the embed context.

const tick = () => new Promise((r) => setTimeout(r, 0));

test('embed context: the turn-end event syncs the organic highlight WITHOUT DOMContentLoaded ever running', async () => {
    API.getConversations = async () => ({
        conversations: [{
            session_id: 'sess-embed-1',
            preview: 'typed organically',
            started_at: '2026-07-10T15:00:00Z',
            last_message_at: '2026-07-10T15:00:00Z',
            message_count: 2,
        }],
    });
    API.setHostAgent('Emma');
    refreshConversationsPane();
    await tick();

    const row = Array.from(document.querySelectorAll('.conversation-item'))
        .find((r) => r.dataset.sessionId === 'sess-embed-1');
    assert.ok(row, 'organic session row rendered');
    assert.ok(!row.classList.contains('active'), 'not highlighted before the event');

    window.dispatchEvent(new dom.window.CustomEvent('kestrel:conversations-stale', {
        detail: { sessionId: 'sess-embed-1', agent: 'Emma' },
    }));
    await tick();
    await tick();

    const after = Array.from(document.querySelectorAll('.conversation-item'))
        .find((r) => r.dataset.sessionId === 'sess-embed-1');
    assert.ok(after, 'row still present after refresh');
    assert.ok(
        after.classList.contains('active'),
        'the turn-end event synced the highlight with NO DOMContentLoaded — the embed path',
    );
});
