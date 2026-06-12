// #1662 PR C — attachment composer: tray render, paste→upload→stage, remove,
// and the read-only message-attachment renderer (XSS escaping).
import test from 'node:test';
import assert from 'node:assert/strict';

globalThis.window = globalThis.window || {};
globalThis.window.SharedMarkdown = {
    renderMarkdown: () => '',
    renderStreamingMarkdown: () => '',
    highlightCodeBlocks: () => {},
    renderMermaidDiagrams: () => {},
    finalizeMarkdown: async () => {},
};

// DOM mock that RECORDS event listeners so tests can dispatch synthetic events
// (the shared harness's addEventListener is a no-op).
function makeNode(tag = 'div') {
    const node = {
        tagName: tag.toUpperCase(),
        nodeType: 1,
        id: '',
        children: [],
        childNodes: [],
        parentNode: null,
        _listeners: {},
        classList: {
            _set: new Set(),
            add(c) { this._set.add(c); },
            remove(c) { this._set.delete(c); },
            contains(c) { return this._set.has(c); },
            toggle(c, on) { on ? this._set.add(c) : this._set.delete(c); return on; },
        },
        dataset: {},
        style: {},
        innerHTML: '',
        textContent: '',
        value: '',
        files: null,
        scrollTop: 0,
        scrollHeight: 0,
        addEventListener(type, fn) {
            (this._listeners[type] = this._listeners[type] || []).push(fn);
        },
        dispatch(type, event = {}) {
            for (const fn of this._listeners[type] || []) fn({ target: this, ...event });
        },
        getAttribute(name) { return this._attrs?.[name] ?? null; },
        setAttribute(name, v) { (this._attrs = this._attrs || {})[name] = v; },
        closest() { return null; },
        click() { this.dispatch('click', {}); },
        focus() {},
        querySelector(sel) {
            if (!sel.startsWith('#')) return null;
            const id = sel.slice(1);
            const stack = [...this.children];
            while (stack.length) {
                const c = stack.shift();
                if (c.id === id) return c;
                stack.push(...c.children);
            }
            return null;
        },
        querySelectorAll() { return []; },
        appendChild(child) { child.parentNode = this; this.children.push(child); this.childNodes.push(child); return child; },
        insertAdjacentHTML() {},
        remove() {},
        get firstChild() { return this.children[0] || null; },
    };
    return node;
}

const documentRoot = makeNode();
globalThis.document = {
    getElementById(id) { return documentRoot.querySelector('#' + id); },
    createElement(tag) { return makeNode(tag); },
    head: makeNode('head'),
    body: makeNode('body'),
    addEventListener() {},
    querySelector(sel) { return documentRoot.querySelector(sel); },
    querySelectorAll() { return []; },
};
globalThis.sessionStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.location = { href: '/', search: '' };
globalThis.fetch = async () => ({ ok: false, status: 500 });
globalThis.kicon = () => '';
globalThis.CSS = { escape: (s) => String(s) };

const chat = await import('../../kestrel_sovereign/static/js/chat.js');

// The shared escapeHtml uses document.createElement (textContent→innerHTML),
// which the DOM mock doesn't emulate; inject a real string escaper so the
// renderer's output is meaningful to assert on.
chat.setChatDeps({
    escapeHtml: (s) => String(s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;'),
});

// --- pure renderer: escaping + image vs doc ---------------------------------

test('messageAttachmentsHtml renders an image thumbnail by url', () => {
    const html = chat.messageAttachmentsHtml([
        { hash: 'a'.repeat(64), kind: 'image', name: 'shot.png', url: '/api/files/' + 'a'.repeat(64) },
    ]);
    assert.match(html, /<img src="\/api\/files\/a{64}"/);
    assert.match(html, /class="message-attachments"/);
});

test('messageAttachmentsHtml falls back to /api/files/<hash> when no url', () => {
    const html = chat.messageAttachmentsHtml([
        { hash: 'b'.repeat(64), kind: 'image', name: 'x.png' },
    ]);
    assert.match(html, /\/api\/files\/b{64}/);
});

test('messageAttachmentsHtml renders a document as a named link, not an image', () => {
    const html = chat.messageAttachmentsHtml([
        { hash: 'c'.repeat(64), kind: 'document', name: 'report.pdf', url: '/api/files/' + 'c'.repeat(64) },
    ]);
    assert.match(html, /msg-attachment-doc/);
    assert.match(html, /report\.pdf/);
    assert.doesNotMatch(html, /<img/);
});

test('messageAttachmentsHtml escapes attachment names (no HTML injection)', () => {
    const html = chat.messageAttachmentsHtml([
        { hash: 'd'.repeat(64), kind: 'document', name: '<img src=x onerror=alert(1)>', url: '/x' },
    ]);
    assert.doesNotMatch(html, /<img src=x/);
    assert.match(html, /&lt;img src=x/);
});

test('messageAttachmentsHtml is empty for no attachments', () => {
    assert.equal(chat.messageAttachmentsHtml([]), '');
    assert.equal(chat.messageAttachmentsHtml(null), '');
});

// --- composer wiring: paste → upload → stage → tray → remove ----------------

function wireComposer() {
    // Build the DOM the composer binds to.
    const container = makeNode('section');
    for (const id of ['chat-container', 'message-input', 'send-button',
        'model-selector', 'thinking-indicator', 'composer-mode-toggle',
        'attach-button', 'attach-input', 'attachment-tray', 'input-area']) {
        const n = makeNode(id === 'message-input' ? 'textarea' : 'div');
        n.id = id;
        container.appendChild(n);
    }
    documentRoot.appendChild(container);

    const uploaded = [];
    let pendingPane = { pendingAttachments: [], element: makeNode(), scrollPos: 0 };
    chat.mount(container, {
        deps: {
            api: {
                hasCapability: () => true,
                getHostAgent: () => 'agent-1',
                uploadAttachment: async (file) => {
                    uploaded.push(file);
                    return { hash: 'e'.repeat(64), kind: 'image', mime: 'image/png',
                        name: file.name, url: '/api/files/' + 'e'.repeat(64) };
                },
            },
            getOrCreateChatPane: () => pendingPane,
        },
    });
    return { container, uploaded, pane: () => pendingPane };
}

test('pasting an image uploads it and stages it inline on the pane', async () => {
    const { container, uploaded, pane } = wireComposer();
    const input = container.querySelector('#message-input');
    const file = { name: 'pasted.png', type: 'image/png' };
    input.dispatch('paste', {
        clipboardData: { items: [{ kind: 'file', type: 'image/png', getAsFile: () => file }] },
        preventDefault: () => {},
    });
    // Let the async upload settle.
    await new Promise((r) => setTimeout(r, 0));
    assert.equal(uploaded.length, 1);
    const staged = pane().pendingAttachments;
    assert.equal(staged.length, 1);
    assert.equal(staged[0].kind, 'image');
    assert.equal(staged[0].inline, true, 'pasted image must be marked inline (eager vision)');
});

test('the attach button stages a NON-inline (lazy) ref', async () => {
    const { container, pane } = wireComposer();
    const attachInput = container.querySelector('#attach-input');
    attachInput.files = [{ name: 'doc.pdf', type: 'application/pdf' }];
    attachInput.dispatch('change', {});
    await new Promise((r) => setTimeout(r, 0));
    const staged = pane().pendingAttachments;
    assert.equal(staged.length, 1);
    assert.equal(staged[0].inline, false, 'attach-button file must be a lazy ref, not inline');
});
