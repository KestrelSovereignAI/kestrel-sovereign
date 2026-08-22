// #2960: the conversation list pages. Before this ticket the server answered
// with one fixed window of history — measured on a live agent, 34% of its
// conversations fell outside it and no `limit` could reach them — so the pane
// rendered whatever one response contained and there was nothing to continue.
//
// These cover the component's half of that: it must ASK for the next page with
// the token it was given, APPEND rather than replace, stop when the server says
// there is no next page, and never paint one agent's page under another's.

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

const { mountConversations } = await import('../../kestrel_sovereign/static/js/conversations.js');

// Several ticks: a page is a fetch, a state update and a repaint.
const settle = async () => { for (let i = 0; i < 5; i++) await new Promise((r) => setTimeout(r, 0)); };

function conversation(id) {
    return {
        session_id: id,
        preview: `preview ${id}`,
        started_at: '2026-06-23T12:00:00Z',
        last_message_at: '2026-06-23T12:00:00Z',
        message_count: 2,
    };
}

// A server holding `pages` of session ids. Page N's `next_cursor` names page
// N+1, and handing that token back is the only way to reach it — which is the
// contract the component has to honour.
function pagingApi(pages) {
    const calls = [];
    const token = (index) => `cursor-for-page-${index}`;
    return {
        calls,
        getConversations: async (decrypt, view, cursor) => {
            calls.push({ decrypt, view, cursor: cursor ?? null });
            const index = cursor ? pages.findIndex((_, i) => token(i) === cursor) : 0;
            if (index < 0) throw new Error(`unknown cursor ${cursor}`);
            return {
                conversations: pages[index].map(conversation),
                next_cursor: index + 1 < pages.length ? token(index + 1) : null,
            };
        },
        listTrash: async () => ({ messages: [] }),
    };
}

function makeContainer() {
    const el = document.createElement('div');
    document.body.appendChild(el);
    return el;
}

const rows = (el) => Array.from(el.querySelectorAll('.conversation-item'));
const moreBtn = (el) => el.querySelector('.conversations-load-more');

test('a next_cursor paints a Load more button; its absence does not', async () => {
    const el = makeContainer();
    const api = pagingApi([['a', 'b'], ['c']]);
    const handle = mountConversations(el, { api, agentName: 'Emma' });
    await settle();

    assert.equal(rows(el).length, 2);
    assert.ok(moreBtn(el), 'a page with a next_cursor offers to continue');
    assert.equal(handle.hasMore, true);

    moreBtn(el).click();
    await settle();

    // Appended, not replaced — the pages already read stay on screen.
    assert.deepEqual(rows(el).map((r) => r.dataset.sessionId), ['a', 'b', 'c']);
    assert.equal(api.calls[1].cursor, 'cursor-for-page-1',
        'the server is asked with the token IT minted, not one the client built');
    assert.equal(moreBtn(el), null, 'no next_cursor is the end of the list');
    assert.equal(handle.hasMore, false);
    handle.destroy();
});

test('a session served on two pages is rendered once', async () => {
    // Keyset paging over a table the agent is still writing to can hand back a
    // session whose activity moved across the cursor. A repeat is the benign
    // direction; a repeated ROW in the list is not.
    const el = makeContainer();
    const api = pagingApi([['a', 'b'], ['b', 'c']]);
    const handle = mountConversations(el, { api, agentName: 'Emma' });
    await settle();
    moreBtn(el).click();
    await settle();

    assert.deepEqual(rows(el).map((r) => r.dataset.sessionId), ['a', 'b', 'c']);
    handle.destroy();
});

test('a refresh starts over at the top rather than continuing', async () => {
    const el = makeContainer();
    const api = pagingApi([['a', 'b'], ['c']]);
    const handle = mountConversations(el, { api, agentName: 'Emma' });
    await settle();
    moreBtn(el).click();
    await settle();
    assert.equal(rows(el).length, 3);

    await handle.refresh();
    await settle();

    assert.deepEqual(rows(el).map((r) => r.dataset.sessionId), ['a', 'b']);
    assert.equal(api.calls[api.calls.length - 1].cursor, null,
        'a reload asks for page one, not for the page after the last one');
    handle.destroy();
});

test('a page that lands after a view switch is dropped, not appended', async () => {
    // The stale-response guard the list already has for refresh(), applied to
    // the continuation: appending a page belonging to the previous view paints
    // active conversations under the Archived actions.
    const el = makeContainer();
    let release;
    const gate = new Promise((resolve) => { release = resolve; });
    const calls = [];
    const api = {
        getConversations: async (decrypt, view, cursor) => {
            calls.push({ view, cursor });
            if (cursor === 'tok-1') {
                await gate;
                return { conversations: [conversation('late')], next_cursor: null };
            }
            return {
                conversations: [conversation(`${view}-1`)],
                next_cursor: cursor ? null : 'tok-1',
            };
        },
        listTrash: async () => ({ messages: [] }),
    };
    const handle = mountConversations(el, { api, agentName: 'Emma' });
    await settle();
    moreBtn(el).click();          // in flight, waiting on the gate
    handle.setView('archived');   // ...and the view changes underneath it
    await settle();
    release();
    await settle();

    const ids = rows(el).map((r) => r.dataset.sessionId);
    assert.ok(!ids.includes('late'), 'the previous view\'s page must not be painted');
    assert.deepEqual(ids, ['archived-1']);
    // ...and the pane can still page. A losing continuation that left the
    // in-flight flag set would disable Load more for the rest of its life.
    assert.equal(calls.some((c) => c.view === 'archived' && c.cursor === null), true);
    const more = moreBtn(el);
    assert.ok(more, 'the new view offers its own next page');
    assert.equal(more.disabled, false, 'a lost continuation left the pane wedged');
    more.click();
    await settle();
    assert.deepEqual(rows(el).map((r) => r.dataset.sessionId), ['archived-1', 'late']);
    handle.destroy();
});
