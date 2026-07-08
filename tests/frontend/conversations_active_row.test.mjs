// #2254: two conversations-pane polish bugs.
//
// (1) Active-row highlight regression for ORGANIC sessions. `.conversation-item
//     .active` exists and buildConversationRow applies it, but only the EXPLICIT
//     paths (loadConversation, new-conversation, trash-mutation) synced the
//     active id. A session the user reaches by just typing learns its effective
//     id from the X-Session-Id header onto `pane.sessionId` — identity.js's
//     `activeConversationId` was never updated, so no row highlighted, even
//     after the #2250 turn-end refresh repainted. Fix: the turn-end
//     `kestrel:conversations-stale` event now carries `{ sessionId, agent }`;
//     identity.js's listener adopts it into the per-agent active-id map (and the
//     live highlight for the current host) BEFORE the refresh repaints.
//
// (2) The kebab (⋯) is pinned to the RIGHT of the tile, VERTICALLY CENTERED in
//     the row (was awkwardly at the right end of the time+count meta line).
//
// These tests pin: the organic-session end-to-end flow through the REAL
// identity.js listener (jsdom), the SOURCE CONTRACTs on both files' halves of
// the event wiring, and the kebab's right-centered CSS contract.

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
// #879 hide toggle. identity.js keeps ONE module-level mount handle bound to
// `#conversations-list`, so the container must stay stable across tests.
for (const id of ['conversations-pane', 'conversations-list', 'chat-container']) {
    const node = document.createElement('div');
    node.id = id;
    document.body.appendChild(node);
}

// The `kestrel:conversations-stale` listener is registered inside identity.js's
// DOMContentLoaded handler; jsdom's document is already 'complete' at import, so
// fire the event manually to run that wiring.
document.dispatchEvent(new dom.window.Event('DOMContentLoaded'));

const tick = () => new Promise((r) => setTimeout(r, 0));

function conversation(sessionId, preview) {
    return {
        session_id: sessionId,
        preview,
        started_at: '2026-06-09T15:00:00Z',
        last_message_at: '2026-06-09T15:00:00Z',
        message_count: 2,
    };
}

function rowById(sessionId) {
    return Array.from(document.querySelectorAll('.conversation-item'))
        .find((row) => row.dataset.sessionId === sessionId) || null;
}

test('organic session: after the turn-end event carrying { sessionId, agent }, the matching row is highlighted', async () => {
    API.getConversations = async () => ({ conversations: [conversation('sess-1', 'hello there')] });
    API.setHostAgent('Emma');
    // Mount + retarget. NO explicit loadConversation runs — this is the organic
    // path where the id would only be learned from the X-Session-Id header.
    refreshConversationsPane();
    await tick();

    const before = rowById('sess-1');
    assert.ok(before, 'the organic session row rendered');
    assert.ok(
        !before.classList.contains('active'),
        'no row is highlighted yet — the organic id was never synced (the bug)',
    );

    // chat.js fires this on turn teardown, now carrying the learned session id +
    // the dispatch agent. The real identity.js listener must adopt it and paint.
    window.dispatchEvent(new dom.window.CustomEvent('kestrel:conversations-stale', {
        detail: { sessionId: 'sess-1', agent: 'Emma' },
    }));
    await tick();
    await tick();

    const after = rowById('sess-1');
    assert.ok(after, 'the row still rendered after the refresh');
    assert.ok(
        after.classList.contains('active'),
        'the organic session row is now highlighted (active id synced from the event detail)',
    );
});

test('the turn-end event highlight is scoped to the agent it fired for (companion/agent switch)', async () => {
    // A turn on Emma must not highlight a row while the operator is viewing
    // another host — the id lands in the per-agent map, not the live highlight.
    API.getConversations = async () => ({ conversations: [conversation('m-1', 'Meridian row')] });
    API.setHostAgent('Meridian');
    refreshConversationsPane();
    await tick();

    // The event fires for Emma while Meridian is the current host.
    window.dispatchEvent(new dom.window.CustomEvent('kestrel:conversations-stale', {
        detail: { sessionId: 'emma-99', agent: 'Emma' },
    }));
    await tick();

    const meridianRow = rowById('m-1');
    assert.ok(meridianRow, 'Meridian row rendered');
    assert.ok(
        !meridianRow.classList.contains('active'),
        'an Emma turn does not highlight a row under Meridian routing',
    );

    // Switching to Emma surfaces the id the event stored in the per-agent map.
    API.getConversations = async () => ({ conversations: [conversation('emma-99', 'Emma row')] });
    API.setHostAgent('Emma');
    refreshConversationsPane();
    await tick();
    const emmaRow = rowById('emma-99');
    assert.ok(emmaRow, 'Emma row rendered after switch');
    assert.ok(
        emmaRow.classList.contains('active'),
        'the id the Emma turn stored is highlighted once Emma is current',
    );
});

test('SOURCE CONTRACT: chat.js carries { sessionId, agent } in the turn-end stale event', () => {
    const chat = readFileSync(
        new URL('../../kestrel_sovereign/static/js/chat.js', import.meta.url), 'utf8');

    // The debounced dispatch builds a detail-bearing event...
    assert.match(chat, /new\s+\w*Event\w*\(\s*['"]kestrel:conversations-stale['"]\s*,\s*\{\s*detail\s*\}/s,
        'chat.js dispatches the stale event with a detail payload');
    // ...and the turn-teardown call feeds it this turn's learned session + agent.
    assert.match(chat, /notifyConversationsStale\(\s*pane\.sessionId\s*,\s*dispatchAgent\s*\)/,
        'the turn-end signal passes pane.sessionId + dispatchAgent so the pane can highlight the active row');
});

test('SOURCE CONTRACT: identity.js listener syncs the active id from the event detail before refresh', () => {
    const identity = readFileSync(
        new URL('../../kestrel_sovereign/static/js/identity.js', import.meta.url), 'utf8');

    const start = identity.indexOf("addEventListener('kestrel:conversations-stale'");
    assert.ok(start > -1, 'the stale listener exists');
    const block = identity.slice(start, start + 1400);
    assert.match(block, /event\s*&&\s*event\.detail/, 'the listener reads event.detail');
    assert.match(block, /activeConversationIdsByAgent\.set\(\s*detail\.agent\s*,\s*detail\.sessionId\s*\)/,
        'the listener records the id in the per-agent active map');
    assert.match(block, /setActiveSessionId\(activeConversationId\)/,
        'the listener drives the component highlight before the refresh repaints');
});

test('CSS CONTRACT: the kebab is absolutely pinned right and vertically centered on the tile', () => {
    const css = readFileSync(
        new URL('../../kestrel_sovereign/static/index.css', import.meta.url), 'utf8');

    const idx = css.indexOf('.conversation-item .kebab-btn {');
    assert.ok(idx > -1, 'a tile-scoped kebab rule exists');
    const block = css.slice(idx, css.indexOf('}', idx) + 1);
    assert.match(block, /position:\s*absolute/, 'kebab is absolutely positioned on the tile');
    assert.match(block, /right:\s*0\.5rem/, 'kebab pinned to the right edge');
    assert.match(block, /top:\s*50%/, 'kebab anchored to the vertical midpoint');
    assert.match(block, /transform:\s*translateY\(-50%\)/, 'kebab is vertically centered');

    // The tile reserves right padding so text never underlaps the kebab.
    const item = css.slice(css.indexOf('.conversation-item {'));
    assert.match(item.slice(0, item.indexOf('}')), /padding:\s*0\.75rem\s+2\.25rem/,
        'the tile reserves right padding for the absolutely-positioned kebab');
});
