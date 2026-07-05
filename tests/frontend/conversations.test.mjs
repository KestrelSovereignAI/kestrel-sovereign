// #2149: the single conversation-list component. Before this module the
// console shipped two divergent list UIs (the identity.js sidebar + the
// history.js slideout) with two renders and two rename implementations. These
// tests exercise the consolidated component's public surface:
//   - renders timeline-grouped rows, each with a kebab (⋯) overflow button
//   - the kebab menu carries Rename / Archive / Move to Trash / Delete
//     Permanently, and each item calls the right API method
//   - contextmenu (right-click) opens the SAME menu as an accelerator
//   - views/filters: Active (default), Archived (Unarchive item), Trash
//     (Restore + purge, loaded from listTrash)
//   - mount / retarget / destroy lifecycle

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
// The component's trash/purge handlers gate on confirm(); auto-approve so the
// action fires. Individual tests flip this to test the cancel path.
let confirmReturn = true;
globalThis.confirm = () => confirmReturn;
globalThis.window.confirm = globalThis.confirm;

const { mountConversations, buildConversationRow, beginInlineRename, formatConversationTime } = await import(
    '../../kestrel_sovereign/static/js/conversations.js'
);

// A call-recording stub API. Every list method resolves to canned data; every
// mutation records its arguments so tests can assert the row action wired to
// the right endpoint.
function stubApi(overrides = {}) {
    const calls = [];
    const rec = (name) => (...args) => { calls.push({ name, args }); return Promise.resolve({ success: true }); };
    const api = {
        calls,
        getConversations: async (decrypt, view) => {
            calls.push({ name: 'getConversations', args: [decrypt, view] });
            return {
                conversations: [
                    {
                        session_id: 's1', name: 'Debug thread', preview: 'first message',
                        started_at: '2026-06-23T12:00:00Z', message_count: 3,
                    },
                    {
                        session_id: 's2', preview: 'unnamed preview',
                        started_at: '2026-06-23T13:00:00Z', message_count: 1,
                    },
                ],
            };
        },
        listTrash: async (limit) => {
            calls.push({ name: 'listTrash', args: [limit] });
            return {
                messages: [
                    {
                        content: 'trashed msg', role: 'user',
                        deleted_at: '2026-06-22T09:00:00Z',
                        metadata: { session_id: 't1' },
                    },
                ],
            };
        },
        renameConversation: (id, name) => { calls.push({ name: 'renameConversation', args: [id, name] }); return Promise.resolve({ success: true, session_id: id, name }); },
        archiveConversation: rec('archiveConversation'),
        unarchiveConversation: rec('unarchiveConversation'),
        deleteConversation: rec('deleteConversation'),
        purgeConversation: rec('purgeConversation'),
        restoreConversation: rec('restoreConversation'),
        restoreMessage: rec('restoreMessage'),
        purgeMessage: rec('purgeMessage'),
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

function openKebab(row) {
    const btn = row.querySelector('.kebab-btn');
    assert.ok(btn, 'row must carry a kebab button');
    btn.click();
    return Array.from(document.querySelectorAll('.kebab-menu .kebab-menu-item'));
}

function menuLabels(items) {
    return items.map((i) => i.textContent.trim());
}

// #2165: compact meta-row time. Same-day → time only; this year → "Mon D";
// older → "M/D/YY". The full timestamp always rides the title attribute.
test('formatConversationTime: same-day renders time only', () => {
    const now = new Date();
    const iso = now.toISOString();
    const { text, title } = formatConversationTime(iso);
    // Time-only: has a colon (H:MM) and no month/day separators.
    assert.match(text, /\d{1,2}:\d{2}/, 'same-day shows H:MM time');
    assert.doesNotMatch(text, /\//, 'same-day carries no date slashes');
    // Full timestamp preserved for hover.
    assert.equal(title, new Date(iso).toLocaleString());
});

test('formatConversationTime: this-year renders month + day (no year)', () => {
    const now = new Date();
    // Pick a day in this year that is not today (Jan 1 unless today is Jan 1).
    const isToday = now.getMonth() === 0 && now.getDate() === 1;
    const month = isToday ? 5 : 0; // June vs January
    const d = new Date(Date.UTC(now.getFullYear(), month, isToday ? 15 : 1, 12, 0, 0));
    const { text, title } = formatConversationTime(d.toISOString());
    assert.match(text, /[A-Za-z]{3}\s+\d{1,2}/, 'this-year shows "Mon D"');
    assert.doesNotMatch(text, /\d{4}/, 'this-year omits the year');
    assert.equal(title, new Date(d.toISOString()).toLocaleString());
});

test('formatConversationTime: older renders a compact 2-digit-year date', () => {
    const now = new Date();
    const d = new Date(Date.UTC(now.getFullYear() - 2, 6, 4, 12, 0, 0));
    const { text, title } = formatConversationTime(d.toISOString());
    assert.match(text, /\d{1,2}\/\d{1,2}\/\d{2}/, 'older shows M/D/YY');
    assert.doesNotMatch(text, /\d{4}/, 'older uses a 2-digit year');
    assert.equal(title, new Date(d.toISOString()).toLocaleString());
});

test('formatConversationTime: empty input yields empty text and title', () => {
    assert.deepEqual(formatConversationTime(''), { text: '', title: '' });
    assert.deepEqual(formatConversationTime(null), { text: '', title: '' });
});

test('buildConversationRow: meta-row time carries the full timestamp as title', () => {
    const row = buildConversationRow(
        { session_id: 'z', preview: 'p', started_at: '2024-07-04T11:05:05Z' },
        {},
    );
    const time = row.querySelector('.conversation-time');
    assert.ok(time, 'row carries a time span');
    assert.equal(time.title, new Date(require_ts('2024-07-04T11:05:05Z')).toLocaleString());
    // Older date → compact M/D/YY, no seconds, no 4-digit year.
    assert.doesNotMatch(time.textContent, /\d{4}/);
    assert.doesNotMatch(time.textContent, /:\d{2}:\d{2}/, 'no seconds in the compact label');
});

// Helper: parse a tz-marked ISO string to ms (mirrors timelineTs for a
// tz-suffixed string — Date.parse handles the 'Z' directly).
function require_ts(iso) { return Date.parse(iso); }

test('mount renders timeline-grouped rows each with a kebab button', async () => {
    const container = makeContainer();
    const api = stubApi();
    const handle = mountConversations(container, { api, autoLoad: false });
    await handle.refresh();

    const rows = rowsIn(container);
    assert.equal(rows.length, 2, 'both conversations render');
    // Named conversation shows its name; unnamed falls back to preview.
    assert.match(container.innerHTML, /Debug thread/);
    assert.match(container.innerHTML, /unnamed preview/);
    // Grouped under a date-group header (history.js grouping preserved).
    assert.ok(container.querySelector('.date-group'), 'rows are timeline-grouped');
    for (const row of rows) {
        assert.ok(row.querySelector('.kebab-btn'), 'every row has a kebab button');
    }
    handle.destroy();
});

test('kebab menu on an active row carries Rename / Archive / Trash / Delete Permanently', async () => {
    const container = makeContainer();
    const handle = mountConversations(container, { api: stubApi(), autoLoad: false });
    await handle.refresh();

    const items = openKebab(rowsIn(container)[0]);
    const labels = menuLabels(items);
    assert.deepEqual(labels, ['Rename', 'Archive', 'Move to Trash', 'Delete Permanently']);
    // Delete Permanently is danger-styled and separated (the #765 slow-down).
    const purge = items.find((i) => i.dataset.action === 'purge');
    assert.ok(purge.classList.contains('kebab-menu-item-danger'), 'purge is danger-styled');
    handle.destroy();
});

test('Archive action calls archiveConversation', async () => {
    const container = makeContainer();
    const api = stubApi();
    const handle = mountConversations(container, { api, autoLoad: false });
    await handle.refresh();

    const items = openKebab(rowsIn(container)[0]);
    items.find((i) => i.dataset.action === 'archive').click();
    await new Promise((r) => setTimeout(r, 0));
    assert.ok(api.calls.some((c) => c.name === 'archiveConversation' && c.args[0] === 's1'));
    handle.destroy();
});

test('Move to Trash confirms then calls deleteConversation; cancel skips it', async () => {
    const container = makeContainer();
    const api = stubApi();
    const handle = mountConversations(container, { api, autoLoad: false });
    await handle.refresh();

    confirmReturn = false;
    openKebab(rowsIn(container)[0]).find((i) => i.dataset.action === 'trash').click();
    await new Promise((r) => setTimeout(r, 0));
    assert.ok(!api.calls.some((c) => c.name === 'deleteConversation'), 'cancel skips the delete');

    confirmReturn = true;
    openKebab(rowsIn(container)[0]).find((i) => i.dataset.action === 'trash').click();
    await new Promise((r) => setTimeout(r, 0));
    assert.ok(api.calls.some((c) => c.name === 'deleteConversation' && c.args[0] === 's1'));
    handle.destroy();
});

test('Delete Permanently confirms then calls purgeConversation', async () => {
    const container = makeContainer();
    const api = stubApi();
    const handle = mountConversations(container, { api, autoLoad: false });
    await handle.refresh();

    confirmReturn = true;
    openKebab(rowsIn(container)[0]).find((i) => i.dataset.action === 'purge').click();
    await new Promise((r) => setTimeout(r, 0));
    assert.ok(api.calls.some((c) => c.name === 'purgeConversation' && c.args[0] === 's1'));
    handle.destroy();
});

test('Rename item begins inline edit and commits through renameConversation', async () => {
    const container = makeContainer();
    const api = stubApi();
    const handle = mountConversations(container, { api, autoLoad: false });
    await handle.refresh();

    const row = rowsIn(container)[0];
    openKebab(row).find((i) => i.dataset.action === 'rename').click();
    const input = row.querySelector('.conversation-rename-input');
    assert.ok(input, 'inline rename input appears (no prompt())');
    input.value = 'Renamed thread';
    input.dispatchEvent(new dom.window.Event('blur'));
    await new Promise((r) => setTimeout(r, 0));
    assert.ok(api.calls.some((c) => c.name === 'renameConversation'
        && c.args[0] === 's1' && c.args[1] === 'Renamed thread'));
    handle.destroy();
});

test('Archived view loads view=archived and menu offers Unarchive', async () => {
    const container = makeContainer();
    const api = stubApi();
    const handle = mountConversations(container, { api, autoLoad: false });
    await handle.refresh();

    handle.setView('archived');
    await new Promise((r) => setTimeout(r, 0));
    assert.ok(api.calls.some((c) => c.name === 'getConversations' && c.args[1] === 'archived'));

    const items = openKebab(rowsIn(container)[0]);
    const labels = menuLabels(items);
    assert.ok(labels.includes('Unarchive'), 'archived rows offer Unarchive');
    assert.ok(!labels.includes('Archive'), 'archived rows do not offer Archive');
    items.find((i) => i.dataset.action === 'unarchive').click();
    await new Promise((r) => setTimeout(r, 0));
    assert.ok(api.calls.some((c) => c.name === 'unarchiveConversation'));
    handle.destroy();
});

test('Trash view loads listTrash and menu offers Restore + Delete Permanently', async () => {
    const container = makeContainer();
    const api = stubApi();
    const handle = mountConversations(container, { api, autoLoad: false });
    await handle.refresh();

    handle.setView('trash');
    await new Promise((r) => setTimeout(r, 0));
    assert.ok(api.calls.some((c) => c.name === 'listTrash'), 'trash view uses listTrash');

    const items = openKebab(rowsIn(container)[0]);
    const labels = menuLabels(items);
    assert.deepEqual(labels, ['Restore', 'Delete Permanently']);
    items.find((i) => i.dataset.action === 'restore').click();
    await new Promise((r) => setTimeout(r, 0));
    assert.ok(api.calls.some((c) => c.name === 'restoreConversation' && c.args[0] === 't1'));
    handle.destroy();
});

// #2199 P2-1: individually-deleted messages (no session_id) are "orphans" in
// groupTrashBySession. They must still surface as restore/purge-actionable rows
// — dropping them makes trashed messages invisible and unmanageable. Their
// actions act at the MESSAGE level (restoreMessage / purgeMessage).
test('Trash view surfaces orphan messages with message-level restore/purge (#2199 P2-1)', async () => {
    const container = makeContainer();
    const api = stubApi({
        listTrash: async () => ({
            messages: [
                {
                    content: 'session msg', role: 'user',
                    deleted_at: '2026-06-22T09:00:00Z',
                    metadata: { session_id: 't1' },
                },
                {
                    id: 'orphan-1', content: 'a lone deleted message', role: 'assistant',
                    deleted_at: '2026-06-22T10:00:00Z',
                    metadata: {}, // no session_id → orphan
                },
            ],
        }),
    });
    const handle = mountConversations(container, { api, autoLoad: false });
    await handle.refresh();
    handle.setView('trash');
    await new Promise((r) => setTimeout(r, 0));

    const rows = rowsIn(container);
    assert.equal(rows.length, 2, 'both the session row and the orphan row render');

    // The orphan row (sorted newest-first by deleted_at) purges at the message
    // level, not the conversation level.
    const orphanRow = rows.find((r) => /lone deleted message/.test(r.textContent));
    assert.ok(orphanRow, 'orphan message row is rendered');
    const items = openKebab(orphanRow);
    items.find((i) => i.dataset.action === 'purge').click();
    await new Promise((r) => setTimeout(r, 0));
    assert.ok(
        api.calls.some((c) => c.name === 'purgeMessage' && c.args[0] === 'orphan-1'),
        'orphan purge routes through purgeMessage',
    );
    assert.ok(
        !api.calls.some((c) => c.name === 'purgeConversation'),
        'orphan purge never hits purgeConversation',
    );

    const items2 = openKebab(rowsIn(container).find((r) => /lone deleted message/.test(r.textContent)));
    items2.find((i) => i.dataset.action === 'restore').click();
    await new Promise((r) => setTimeout(r, 0));
    assert.ok(
        api.calls.some((c) => c.name === 'restoreMessage' && c.args[0] === 'orphan-1'),
        'orphan restore routes through restoreMessage',
    );
    handle.destroy();
});

// #2199 P2-3: during retarget()'s async reload the OLD agent's rows stay in the
// DOM. A kebab Archive/Trash/Purge fired in that window would send the previous
// agent's session_id against the API layer, which the host has already routed
// to the NEW agent. Such stale actions must be dropped until the reload paints.
test('a mutation fired during a retarget-in-flight is dropped (#2199 P2-3)', async () => {
    const container = makeContainer();
    // A listTrash/getConversations that never resolves within the test tick, so
    // the retarget reload stays in flight while we click a stale row.
    let releaseReload;
    const gate = new Promise((r) => { releaseReload = r; });
    const api = stubApi({
        getConversations: async (decrypt, view) => {
            api.calls.push({ name: 'getConversations', args: [decrypt, view] });
            // First call (initial render) resolves immediately; the retarget
            // reload blocks on the gate.
            if (api.calls.filter((c) => c.name === 'getConversations').length > 1) {
                await gate;
            }
            return {
                conversations: [
                    { session_id: 's1', preview: 'first', started_at: '2026-06-23T12:00:00Z', message_count: 1 },
                ],
            };
        },
    });
    const handle = mountConversations(container, { api, agentName: 'AgentA', autoLoad: false });
    await handle.refresh();

    // Kick off a retarget; its reload is stuck on the gate. The old row is still
    // interactive in the DOM.
    handle.retarget('AgentB');
    const staleRow = rowsIn(container)[0];
    const items = openKebab(staleRow);
    items.find((i) => i.dataset.action === 'archive').click();
    await new Promise((r) => setTimeout(r, 0));
    assert.ok(
        !api.calls.some((c) => c.name === 'archiveConversation'),
        'a stale-agent row action is dropped, not sent to the new agent route',
    );

    // Once the reload resolves and repaints, actions work again.
    releaseReload();
    await new Promise((r) => setTimeout(r, 0));
    const freshRow = rowsIn(container)[0];
    const items2 = openKebab(freshRow);
    items2.find((i) => i.dataset.action === 'archive').click();
    await new Promise((r) => setTimeout(r, 0));
    assert.ok(
        api.calls.some((c) => c.name === 'archiveConversation'),
        'after the retarget repaints, row actions dispatch normally',
    );
    handle.destroy();
});

// #2199 P2-3 regression: renderCurrent() also runs on every search keystroke to
// re-filter the cached list WITHOUT a reload. If the stale-row anchor were set in
// renderCurrent() (rather than in refresh()'s seq-guarded success block), typing
// in search during a retarget-in-flight would advance the anchor to the new agent
// while still painting the OLD agent's rows — defeating the guard. This types in
// search mid-retarget and asserts the mutation is still dropped.
test('a search keystroke during retarget-in-flight does not defeat the stale guard (#2199 P2-3)', async () => {
    const container = makeContainer();
    let releaseReload;
    const gate = new Promise((r) => { releaseReload = r; });
    const api = stubApi({
        getConversations: async (decrypt, view) => {
            api.calls.push({ name: 'getConversations', args: [decrypt, view] });
            if (api.calls.filter((c) => c.name === 'getConversations').length > 1) {
                await gate;
            }
            return {
                conversations: [
                    { session_id: 's1', preview: 'first', started_at: '2026-06-23T12:00:00Z', message_count: 1 },
                ],
            };
        },
    });
    const handle = mountConversations(container, { api, agentName: 'AgentA', autoLoad: false });
    await handle.refresh();

    // Retarget; its reload blocks on the gate. Now type in the search box — this
    // synchronously re-filters AgentA's cached rows via renderCurrent().
    handle.retarget('AgentB');
    const searchInput = container.querySelector('.conversations-search');
    searchInput.value = 'f';
    searchInput.dispatchEvent(new dom.window.Event('input', { bubbles: true }));

    // The still-AgentA row must remain stale: a mutation is dropped, not sent.
    const staleRow = rowsIn(container)[0];
    const items = openKebab(staleRow);
    items.find((i) => i.dataset.action === 'archive').click();
    await new Promise((r) => setTimeout(r, 0));
    assert.ok(
        !api.calls.some((c) => c.name === 'archiveConversation'),
        'a stale-agent row action stays dropped even after a search keystroke repaints it',
    );

    releaseReload();
    handle.destroy();
});

test('contextmenu (right-click) opens the same menu as the kebab', async () => {
    const container = makeContainer();
    const handle = mountConversations(container, { api: stubApi(), autoLoad: false });
    await handle.refresh();

    const row = rowsIn(container)[0];
    const evt = new dom.window.MouseEvent('contextmenu', { bubbles: true, clientX: 10, clientY: 10 });
    row.dispatchEvent(evt);
    const items = Array.from(document.querySelectorAll('.kebab-menu .kebab-menu-item'));
    assert.deepEqual(menuLabels(items), ['Rename', 'Archive', 'Move to Trash', 'Delete Permanently']);
    handle.destroy();
});

test('onSelect fires with the conversation when a row is clicked', async () => {
    const container = makeContainer();
    const selected = [];
    const handle = mountConversations(container, {
        api: stubApi(), autoLoad: false, onSelect: (c) => selected.push(c.session_id),
    });
    await handle.refresh();

    rowsIn(container)[0].click();
    assert.deepEqual(selected, ['s1']);
    handle.destroy();
});

test('retarget reloads the list; destroy clears the container', async () => {
    const container = makeContainer();
    const api = stubApi();
    const handle = mountConversations(container, { api, autoLoad: false });
    await handle.refresh();
    const before = api.calls.filter((c) => c.name === 'getConversations').length;

    handle.retarget('OtherAgent');
    await new Promise((r) => setTimeout(r, 0));
    const after = api.calls.filter((c) => c.name === 'getConversations').length;
    assert.ok(after > before, 'retarget triggers a reload');

    handle.destroy();
    assert.equal(rowsIn(container).length, 0, 'destroy empties the container');
});

test('buildConversationRow escapes user-supplied names (XSS posture)', () => {
    const row = buildConversationRow(
        { session_id: 'x', name: '<img src=x onerror=alert(1)>', preview: 'p' },
        {},
    );
    // The name is rendered as a text node, never parsed as markup, so no real
    // <img> element exists and the raw string survives verbatim as text.
    assert.equal(row.querySelector('img'), null, 'a hostile name must not spawn an element');
    const preview = row.querySelector('.conversation-preview');
    assert.equal(preview.textContent, '<img src=x onerror=alert(1)>', 'name preserved as text');
    assert.ok(row.dataset.sessionId === 'x');
});

// Codex P2-1 (PR #2151): a refresh that resolves AFTER a newer view switch
// must be discarded, or active rows render under the Archived view with the
// wrong actions.
test('stale refresh: a slow active-list response never clobbers a newer view switch', async () => {
    const container = makeContainer();
    const api = stubApi();
    let releaseActive;
    const activeGate = new Promise((r) => { releaseActive = r; });
    const baseGet = api.getConversations;
    api.getConversations = async (decrypt, view) => {
        if (view === 'active') {
            await activeGate; // hold the active response until after the switch
            api.calls.push({ name: 'getConversations', args: [decrypt, view] });
            return {
                conversations: [{
                    session_id: 'stale-active', preview: 'stale row',
                    started_at: '2026-06-23T12:00:00Z', message_count: 1,
                }],
            };
        }
        return baseGet(decrypt, view);
    };

    const handle = mountConversations(container, { api, autoLoad: false });
    const slow = handle.refresh();      // active view — response held
    handle.setView('archived');         // newer refresh wins
    await new Promise((r) => setTimeout(r, 0));
    releaseActive();                    // stale active response lands late
    await slow;
    await new Promise((r) => setTimeout(r, 0));

    const ids = rowsIn(container).map((r) => r.dataset.sessionId);
    assert.ok(!ids.includes('stale-active'), 'stale active rows must not render');
    const labels = menuLabels(openKebab(rowsIn(container)[0]));
    assert.ok(labels.includes('Unarchive'), 'view is still Archived after the stale response');
    handle.destroy();
});

// Codex P2-2 (PR #2151): committing an inline rename with Enter must not
// bubble into the row's Enter/select handler and load the conversation.
test('rename Enter commits without triggering row onSelect', async () => {
    const container = makeContainer();
    const api = stubApi();
    const selected = [];
    const handle = mountConversations(container, {
        api, autoLoad: false, onSelect: (c) => selected.push(c.session_id),
    });
    await handle.refresh();

    const row = rowsIn(container)[0];
    openKebab(row).find((i) => i.dataset.action === 'rename').click();
    const input = row.querySelector('.conversation-rename-input');
    input.value = 'Renamed via Enter';
    input.dispatchEvent(new dom.window.KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    input.dispatchEvent(new dom.window.Event('blur'));
    await new Promise((r) => setTimeout(r, 0));

    assert.ok(api.calls.some((c) => c.name === 'renameConversation'
        && c.args[1] === 'Renamed via Enter'), 'rename committed');
    assert.deepEqual(selected, [], 'Enter in the rename input must not select/load the row');
    handle.destroy();
});
