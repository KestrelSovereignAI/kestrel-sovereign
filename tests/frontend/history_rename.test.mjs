// #2149 / #2171: the standalone console collapsed to a SINGLE conversation
// surface — the `#conversations-pane` sidebar. The history slideout (and the
// `#history-container` mount history.js used to own) is gone. These tests pin
// that consolidation:
//   - loadConversationHistory no longer mounts a second list surface; it only
//     signals the sidebar owner to refresh via the shared stale event
//   - no second rename implementation survives on history.js
//     (window.renameConversation is gone; prompt() is never called)
//   - the standalone console markup carries no history-slideout / toggle button

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { JSDOM } from 'jsdom';

const dom = new JSDOM('<!doctype html><html><body></body></html>', {
    url: 'http://localhost/',
});
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.Node = dom.window.Node;
globalThis.HTMLElement = dom.window.HTMLElement;
globalThis.Event = dom.window.Event;
globalThis.CustomEvent = dom.window.CustomEvent;
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

await import('../../kestrel_sovereign/static/js/api.js');
const { loadConversationHistory } = await import('../../kestrel_sovereign/static/js/history.js');

test('loadConversationHistory signals the sidebar to refresh (no second surface)', async () => {
    let staleEvents = 0;
    window.addEventListener('kestrel:conversations-stale', () => { staleEvents += 1; });

    await loadConversationHistory();

    assert.equal(staleEvents, 1,
        'loadConversationHistory fires the shared conversations-stale event');
    // #2171: the slideout mount is gone — nothing should mount a conversation
    // list into the document from this call.
    assert.equal(document.querySelector('.conversation-item'), null,
        'no second conversation-list surface is rendered by history.js');
    assert.equal(document.getElementById('history-container'), null,
        'the history slideout container no longer exists');
});

test('the retired prompt()-based rename path no longer exists on history.js', () => {
    // history.js used to define a SECOND rename implementation using prompt().
    // The consolidation deletes it; the shared component owns inline rename now.
    assert.equal(typeof globalThis.window.renameConversation, 'undefined',
        'window.renameConversation must be gone');
    assert.equal(promptCalls, 0, 'no prompt() rename path runs during history render');
});

test('the standalone console has one conversation surface — the sidebar, not the slideout', () => {
    const indexHtml = readFileSync(
        fileURLToPath(new URL('../../kestrel_sovereign/static/index.html', import.meta.url)),
        'utf8',
    );
    // The deleted slideout + its toggle button are gone.
    assert.ok(!indexHtml.includes('id="history-sidebar"'),
        'the #history-sidebar slideout is removed');
    assert.ok(!indexHtml.includes('id="toggle-history-btn"'),
        'the #toggle-history-btn is removed');
    assert.ok(!indexHtml.includes('toggleHistorySidebar'),
        'no toggleHistorySidebar wiring remains in the markup');
    // The single surviving surface is the conversations pane.
    assert.ok(indexHtml.includes('id="conversations-pane"'),
        'the #conversations-pane sidebar remains the one conversation surface');
    assert.ok(indexHtml.includes('id="conversations-list"'),
        'the conversations pane keeps its list container');
});
