// Full-text conversation search in the pane component.
//
// The search box used to be a pure client-side name/preview filter. It now
// ALSO fires a debounced server search (GET /api/conversations?q=) whose
// content-level hits replace the rows and carry match snippets:
//   - typing paints the instant client filter first, then the server results
//   - a stale server response (older keystroke) never paints
//   - clearing the term restores the plain list
//   - snippets render with the term <mark>-highlighted via DOM nodes (no
//     innerHTML — message content must not be able to inject markup)
//   - an api without searchConversations degrades to the client filter

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

const { mountConversations, appendHighlighted } = await import(
    '../../kestrel_sovereign/static/js/conversations.js'
);

const LIST = [
    {
        session_id: 's1', name: 'Debug thread', preview: 'first message',
        started_at: '2026-06-23T12:00:00Z', message_count: 3,
    },
    {
        session_id: 's2', preview: 'unnamed preview',
        started_at: '2026-06-23T13:00:00Z', message_count: 1,
    },
];

function stubApi(overrides = {}) {
    const calls = [];
    const api = {
        calls,
        getConversations: async () => ({ conversations: LIST }),
        searchConversations: async (q, view, decrypt) => {
            calls.push({ name: 'searchConversations', args: [q, view, decrypt] });
            return {
                conversations: [{
                    session_id: 's9', preview: 'server hit',
                    started_at: '2026-06-23T14:00:00Z', message_count: 4,
                    match_count: 2, match_role: 'assistant',
                    match_snippet: 'we discussed penguin husbandry at length',
                }],
                query: q,
            };
        },
    };
    return Object.assign(api, overrides);
}

function makeContainer() {
    const el = document.createElement('div');
    document.body.appendChild(el);
    return el;
}

function rowsIn(container) {
    return Array.from(container.querySelectorAll('.conversation-item'));
}

function type(container, term) {
    const input = container.querySelector('.conversations-search');
    input.value = term;
    input.dispatchEvent(new dom.window.Event('input', { bubbles: true }));
}

const settle = (ms = 0) => new Promise((r) => setTimeout(r, ms));
const DEBOUNCE = 300; // component debounce is 250ms

test('typing fires a debounced server search whose hits replace the rows', async () => {
    const api = stubApi();
    const container = makeContainer();
    const handle = mountConversations(container, { api });
    await settle();
    assert.equal(rowsIn(container).length, 2);

    type(container, 'penguin');
    // Instant client filter paints first: no name/preview matches "penguin".
    assert.equal(rowsIn(container).length, 0);
    assert.match(container.textContent, /No matching conversations/);

    await settle(DEBOUNCE);
    const rows = rowsIn(container);
    assert.equal(rows.length, 1);
    assert.equal(rows[0].dataset.sessionId, 's9');
    assert.equal(api.calls.filter((c) => c.name === 'searchConversations').length, 1);
    assert.deepEqual(api.calls[0].args[0], 'penguin');

    const snip = rows[0].querySelector('.conversation-match-snippet');
    assert.ok(snip, 'search hit renders its match snippet');
    const mark = snip.querySelector('mark');
    assert.ok(mark, 'matched term is highlighted');
    assert.equal(mark.textContent, 'penguin');

    handle.destroy();
});

test('a stale server response never paints over a newer term', async () => {
    let release;
    const gate = new Promise((r) => { release = r; });
    const api = stubApi({
        searchConversations: async (q) => {
            if (q === 'slow') {
                await gate;
                return { conversations: [{ session_id: 'stale', preview: 'stale', match_snippet: 'stale' }] };
            }
            return { conversations: [{ session_id: 'fresh', preview: 'fresh', match_snippet: `hit ${q}` }] };
        },
    });
    const container = makeContainer();
    const handle = mountConversations(container, { api });
    await settle();

    type(container, 'slow');
    await settle(DEBOUNCE); // "slow" request in flight, blocked on the gate
    type(container, 'fresh');
    await settle(DEBOUNCE); // "fresh" resolves and paints
    release();              // stale "slow" resolves late
    await settle();

    const rows = rowsIn(container);
    assert.equal(rows.length, 1);
    assert.equal(rows[0].dataset.sessionId, 'fresh');
    handle.destroy();
});

test('clearing the search restores the plain list without a server call', async () => {
    const api = stubApi();
    const container = makeContainer();
    const handle = mountConversations(container, { api });
    await settle();

    type(container, 'penguin');
    await settle(DEBOUNCE);
    assert.equal(rowsIn(container).length, 1);

    type(container, '');
    await settle(DEBOUNCE);
    assert.equal(rowsIn(container).length, 2, 'plain list is back');
    assert.equal(
        api.calls.filter((c) => c.name === 'searchConversations').length, 1,
        'no server search for an empty term',
    );
    handle.destroy();
});

test('an api without searchConversations degrades to the client filter', async () => {
    const api = stubApi();
    delete api.searchConversations;
    const container = makeContainer();
    const handle = mountConversations(container, { api });
    await settle();

    type(container, 'debug');
    await settle(DEBOUNCE);
    const rows = rowsIn(container);
    assert.equal(rows.length, 1, 'client name filter still works');
    assert.equal(rows[0].dataset.sessionId, 's1');
    handle.destroy();
});

test('a mutation drops the search hit immediately and revalidates the search', async () => {
    // codex P2: with an active search, archiving/trashing a hit used to leave
    // it in `searchResults`, so the mutation's refresh() repainted the stale
    // row. The hit must vanish on mutation and the search re-run server-side.
    let searchCalls = 0;
    const api = stubApi({
        archiveConversation: async () => ({ success: true }),
        searchConversations: async (q, view, decrypt) => {
            searchCalls += 1;
            // After the archive, the server no longer returns the hit.
            const conversations = searchCalls === 1
                ? [{
                    session_id: 's9', preview: 'server hit', message_count: 4,
                    started_at: '2026-06-23T14:00:00Z',
                    match_count: 1, match_role: 'user', match_snippet: 'penguin plans',
                }]
                : [];
            return { conversations, query: q };
        },
    });
    const container = makeContainer();
    const handle = mountConversations(container, { api });
    await settle();

    type(container, 'penguin');
    await settle(DEBOUNCE);
    assert.equal(rowsIn(container).length, 1);

    // Archive the hit via its kebab menu.
    const kebab = container.querySelector('.conv-kebab-btn');
    kebab.click();
    const archiveItem = Array.from(document.querySelectorAll('.kebab-menu .kebab-menu-item'))
        .find((i) => /Archive/.test(i.textContent));
    archiveItem.click();
    await settle();

    assert.equal(rowsIn(container).length, 0, 'stale hit no longer painted');
    assert.ok(searchCalls >= 2, 'server search revalidated after the mutation');
    handle.destroy();
});

test('appendHighlighted builds text nodes + <mark>, never markup from content', () => {
    const el = document.createElement('div');
    appendHighlighted(el, '<img src=x onerror=alert(1)> penguin <b>Penguin</b>', 'penguin');
    assert.equal(el.querySelectorAll('img, b').length, 0, 'content cannot inject elements');
    const marks = Array.from(el.querySelectorAll('mark')).map((m) => m.textContent);
    assert.deepEqual(marks, ['penguin', 'Penguin'], 'case-insensitive, case-preserving');
    assert.ok(el.textContent.includes('<img src=x onerror=alert(1)>'), 'literal text preserved');
});
