// #2199: `mountConversationsPane` is the ONE collapsible pane unit — the
// embeddable list surface PLUS the pane chrome (collapse rail, drag-resize with
// min/max + localStorage persistence, and a search/view-bar/stats disclosure).
// The standalone console AND any embedder consume this single export. These
// tests exercise the pane contract directly:
//   - mount builds/adopts chrome and mounts the list;
//   - toggle()/open()/close() flip collapse state + fire onToggle + persist;
//   - the resize handle clamps to min/max and persists the width;
//   - the filters disclosure hides/shows the controls + stats, persisted;
//   - destroy() tears the mount down.
// A separate test confirms the standalone console (identity.js) drives THIS
// export (its handle exposes open/close/toggle), not `mountConversations`.

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

// In-memory localStorage so persistence is observable across mounts.
function makeStorage() {
    const map = new Map();
    return {
        getItem: (k) => (map.has(k) ? map.get(k) : null),
        setItem: (k, v) => { map.set(k, String(v)); },
        removeItem: (k) => { map.delete(k); },
        _map: map,
    };
}
globalThis.localStorage = makeStorage();

const { mountConversationsPane } = await import('../../kestrel_sovereign/static/js/conversations.js');

function fakeApi(conversations = []) {
    return {
        getConversations: async () => ({ conversations }),
        listTrash: async () => ({ messages: [] }),
    };
}

function conv(id, preview) {
    return {
        session_id: id, preview,
        started_at: '2026-06-09T15:00:00Z',
        last_message_at: '2026-06-09T15:00:00Z',
        message_count: 2,
    };
}

test('mount builds pane chrome (header, collapse rail, resize handle) into a bare container', async () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const handle = mountConversationsPane(el, { api: fakeApi([conv('a', 'hi')]), storageKey: 'k:test-build' });
    await new Promise((r) => setTimeout(r, 0));

    assert.ok(el.classList.contains('pane-sidebar'), 'container becomes a pane-sidebar');
    assert.ok(el.classList.contains('conversations-pane'), 'container tagged conversations-pane');
    assert.ok(el.querySelector('.pane-header'), 'header built');
    assert.ok(el.querySelector('.collapse-btn'), 'collapse rail built');
    assert.ok(el.querySelector('.resize-handle'), 'resize handle built');
    assert.ok(el.querySelector('.conversations-filters-toggle'), 'filters disclosure built');
    assert.ok(el.querySelector('.conversation-item'), 'list rows rendered inside the pane');
    handle.destroy();
});

test('mount ADOPTS an existing static pane header + resize handle (console chrome)', () => {
    // Mirror index.html's static #conversations-pane chrome.
    const el = document.createElement('div');
    el.className = 'pane-sidebar';
    el.innerHTML = `
        <div class="pane-header"><h3 id="conversations-pane-title">Conversations</h3>
            <button class="collapse-btn"></button></div>
        <div id="conversations-list" class="pane-content"></div>
        <div class="resize-handle" id="resize-conversations"></div>`;
    document.body.appendChild(el);

    const headerBefore = el.querySelector('.pane-header');
    const handleBefore = el.querySelector('.resize-handle');
    const listBefore = el.querySelector('#conversations-list');

    const handle = mountConversationsPane(el, { api: fakeApi(), storageKey: 'k:test-adopt', autoLoad: false });
    assert.equal(el.querySelectorAll('.pane-header').length, 1, 'no duplicate header');
    assert.equal(el.querySelector('.pane-header'), headerBefore, 'existing header adopted');
    assert.equal(el.querySelector('.resize-handle'), handleBefore, 'existing resize handle adopted');
    // The list mounts into the adopted #conversations-list.
    assert.ok(listBefore.querySelector('.conversations-root'), 'list mounted into adopted #conversations-list');
    handle.destroy();
});

test('toggle()/open()/close() flip collapse state, fire onToggle, and persist', () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const seen = [];
    const handle = mountConversationsPane(el, {
        api: fakeApi(), storageKey: 'k:test-toggle', autoLoad: false,
        onToggle: (c) => seen.push(c),
    });
    assert.equal(handle.collapsed, false, 'starts expanded');
    handle.toggle();
    assert.equal(handle.collapsed, true, 'toggled to collapsed');
    assert.ok(el.classList.contains('collapsed'), 'collapsed class applied');
    assert.equal(localStorage.getItem('k:test-toggle:collapsed'), '1', 'collapse persisted');
    handle.open();
    assert.equal(handle.collapsed, false, 'open() expands');
    handle.close();
    assert.equal(handle.collapsed, true, 'close() collapses');
    assert.deepEqual(seen, [true, false, true], 'onToggle fired for each change');
    handle.destroy();
});

test('a persisted collapsed state is restored on the next mount', () => {
    localStorage.setItem('k:test-restore:collapsed', '1');
    const el = document.createElement('div');
    document.body.appendChild(el);
    const handle = mountConversationsPane(el, { api: fakeApi(), storageKey: 'k:test-restore', autoLoad: false });
    assert.equal(handle.collapsed, true, 'restored collapsed from localStorage');
    handle.destroy();
});

test('the resize handle clamps to min/max and persists the width', () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const handle = mountConversationsPane(el, {
        api: fakeApi(), storageKey: 'k:test-resize', autoLoad: false,
        minWidth: 200, maxWidth: 500,
    });
    const rh = el.querySelector('.resize-handle');
    // Fake a drag from x=300 that would push far past the max.
    Object.defineProperty(el, 'offsetWidth', { value: 280, configurable: true });
    rh.dispatchEvent(new dom.window.MouseEvent('mousedown', { clientX: 300 }));
    document.dispatchEvent(new dom.window.MouseEvent('mousemove', { clientX: 9999 }));
    assert.equal(el.style.width, '500px', 'width clamped to maxWidth');
    document.dispatchEvent(new dom.window.MouseEvent('mousemove', { clientX: -9999 }));
    assert.equal(el.style.width, '200px', 'width clamped to minWidth');
    document.dispatchEvent(new dom.window.MouseEvent('mouseup', {}));
    assert.ok(localStorage.getItem('k:test-resize:width'), 'width persisted on mouseup');
    handle.destroy();
});

test('a persisted width is restored (clamped) on mount', () => {
    localStorage.setItem('k:test-width:width', '9999');
    const el = document.createElement('div');
    document.body.appendChild(el);
    const handle = mountConversationsPane(el, {
        api: fakeApi(), storageKey: 'k:test-width', autoLoad: false, maxWidth: 500,
    });
    assert.equal(el.style.width, '500px', 'oversized persisted width clamped to max');
    handle.destroy();
});

test('the filters disclosure hides/shows the controls + stats and persists', () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const handle = mountConversationsPane(el, {
        api: fakeApi(), storageKey: 'k:test-filters', autoLoad: false,
        showViewBar: true, showSearch: true, showStats: true,
    });
    const controls = el.querySelector('.conversations-controls');
    const stats = el.querySelector('.conversations-stats');
    assert.ok(controls, 'controls block present');
    assert.equal(handle.filtersOpen, true, 'default open');
    assert.notEqual(controls.style.display, 'none', 'controls visible by default');

    handle.setFiltersOpen(false);
    assert.equal(controls.style.display, 'none', 'controls hidden when disclosure closed');
    if (stats) assert.equal(stats.style.display, 'none', 'stats hidden when disclosure closed');
    assert.equal(localStorage.getItem('k:test-filters:filters'), '0', 'disclosure state persisted');

    // The header toggle button flips it too.
    el.querySelector('.conversations-filters-toggle').click();
    assert.equal(handle.filtersOpen, true, 'header toggle re-opens filters');
    handle.destroy();
});

test('destroy() tears down the inner mount', () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const handle = mountConversationsPane(el, { api: fakeApi([conv('a', 'hi')]), storageKey: 'k:test-destroy' });
    handle.destroy();
    assert.equal(el.querySelector('.conversation-item'), null, 'rows removed on destroy');
});

test('the handle exposes list delegators (refresh/retarget/setView)', () => {
    const el = document.createElement('div');
    document.body.appendChild(el);
    const handle = mountConversationsPane(el, { api: fakeApi(), storageKey: 'k:test-delegate', autoLoad: false });
    assert.equal(typeof handle.refresh, 'function');
    assert.equal(typeof handle.retarget, 'function');
    assert.equal(typeof handle.setView, 'function');
    assert.equal(typeof handle.open, 'function');
    assert.equal(typeof handle.close, 'function');
    assert.equal(typeof handle.toggle, 'function');
    assert.ok(handle.conversations, 'inner mountConversations handle exposed');
    handle.destroy();
});
