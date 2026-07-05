// codex round-2 P2 on #2199: hosts WITHOUT the multi_agent agent-select flow
// (embeds like Frinz) never call refreshConversationsPane() from loadAgents /
// selectAgent — the chat-header history trigger is the ONLY path that reveals
// the conversations pane. Because the pane mounts with autoLoad:false, the
// trigger must fire the FIRST list load itself, otherwise the pane opens
// permanently empty.
//
// This lives in its own file (not conversation_agent_switch.test.mjs) on
// purpose: identity.js keeps module-level state (`conversationsPaneTargeted`,
// the mount handle), and this scenario requires a fresh module instance where
// refreshConversationsPane has NEVER run before the trigger is used.

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
// Embed shape: conversations on, multi_agent off — like Frinz's capabilities map.
API.hasCapability = (cap) => cap !== 'multi_agent';

await import('../../kestrel_sovereign/static/js/identity.js');

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

test('history trigger on an embed host fires the FIRST list load — the revealed pane is never permanently empty (#2199 codex round-2)', async () => {
    let fetchCount = 0;
    API.getConversations = async () => {
        fetchCount += 1;
        return { conversations: [conversation('501', 'Clara row')] };
    };
    API.setHostAgent('Clara');

    const pane = document.getElementById('conversations-pane');
    pane.style.display = 'none';

    // No refreshConversationsPane() has ever run in this module instance —
    // the trigger is the first and only entry point, exactly the embed flow.
    window.toggleConversationsPane();
    await new Promise((r) => setTimeout(r, 0));

    assert.equal(pane.style.display, 'flex', 'trigger reveals the hidden pane');
    assert.equal(pane.classList.contains('collapsed'), false, 'revealed pane is open');
    assert.equal(fetchCount, 1, 'revealing an untargeted pane fetches the list exactly once');
    const rows = Array.from(document.querySelectorAll('.conversation-item'))
        .map((row) => row.dataset.sessionId);
    assert.deepEqual(rows, ['501'], 'conversations render in the revealed pane');

    // Subsequent toggles are pure collapse/expand — no re-fetch storm.
    window.toggleConversationsPane();
    window.toggleConversationsPane();
    await new Promise((r) => setTimeout(r, 0));
    assert.equal(fetchCount, 1, 'later toggles do not refetch');
});
