// Typed component parts in the console (#1914).
//
// Covers the reload segmentation (splitContentByParts), the console-wired core
// renderer (notice) proving the system is no longer host-only, and the safety
// contract (renderer escapes host-influenceable text; unregistered type
// degrades to escaped text).
import test from 'node:test';
import assert from 'node:assert/strict';

globalThis.window = globalThis.window || {};
globalThis.window.SharedMarkdown = {
    renderMarkdown: (s) => s,
    renderStreamingMarkdown: (s) => s,
    highlightCodeBlocks: () => {},
    renderMermaidDiagrams: () => {},
    finalizeMarkdown: async () => {},
};

function makeNode(tag = 'div') {
    const node = {
        tagName: tag.toUpperCase(),
        nodeType: 1,
        id: '',
        children: [],
        childNodes: [],
        parentNode: null,
        classList: {
            _set: new Set(),
            add(c) { this._set.add(c); },
            remove(c) { this._set.delete(c); },
            contains(c) { return this._set.has(c); },
        },
        dataset: {},
        style: {},
        innerHTML: '',
        textContent: '',
        value: '',
        scrollTop: 0,
        scrollHeight: 0,
        addEventListener() {},
        focus() {},
        querySelector(selector) {
            if (!selector.startsWith('#')) return null;
            const id = selector.slice(1);
            const stack = [...this.children];
            while (stack.length) {
                const child = stack.shift();
                if (child.id === id) return child;
                stack.push(...child.children);
            }
            return null;
        },
        querySelectorAll() { return []; },
        closest(selector) {
            // Minimal ancestor match for `.class` / tag selectors. Real DOM
            // syncs className↔classList; the mock doesn't, so check both the
            // classList set and the raw className string.
            let n = this;
            while (n) {
                if (selector.startsWith('.')) {
                    const cls = selector.slice(1);
                    const inList = n.classList && n.classList.contains(cls);
                    const inStr = typeof n.className === 'string'
                        && n.className.split(/\s+/).includes(cls);
                    if (inList || inStr) return n;
                } else if (n.tagName === selector.toUpperCase()) {
                    return n;
                }
                n = n.parentNode;
            }
            return null;
        },
        appendChild(child) {
            child.parentNode = this;
            this.children.push(child);
            this.childNodes.push(child);
            return child;
        },
        remove() {
            if (!this.parentNode) return;
            const i = this.parentNode.children.indexOf(this);
            if (i >= 0) this.parentNode.children.splice(i, 1);
            this.parentNode = null;
        },
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
    querySelector(selector) { return documentRoot.querySelector(selector); },
    querySelectorAll() { return []; },
};
globalThis.sessionStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.location = { href: '/', search: '' };
globalThis.fetch = async () => ({ ok: false, status: 500 });
globalThis.kicon = () => '';
globalThis.CSS = { escape: (s) => String(s) };

const chatModule = await import('../../kestrel_sovereign/static/js/chat.js');

function makeChatContainer(hostAgent) {
    const container = makeNode('section');
    for (const id of [
        'chat-container', 'message-input', 'send-button',
        'model-selector', 'thinking-indicator', 'composer-mode-toggle',
    ]) {
        const node = makeNode(id === 'message-input' ? 'textarea' : 'div');
        node.id = id;
        container.appendChild(node);
    }
    return chatModule.mount(container, {
        deps: {
            api: { hasCapability: () => true, getHostAgent: () => hostAgent },
            // Real string-based escape so the renderer's sanitization is
            // exercised (the mock DOM can't do the textContent→innerHTML trick).
            escapeHtml: (s) => String(s)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#39;'),
        },
    });
}

// --------------------------------------------------------------------------
// splitContentByParts — reload interleave segmentation
// --------------------------------------------------------------------------

test('splitContentByParts returns one whole-content prose segment with no parts', () => {
    const segs = chatModule.splitContentByParts('hello world', []);
    assert.deepEqual(segs, [
        { kind: 'prose', text: 'hello world', start: 0, end: 11 },
    ]);
});

test('splitContentByParts interleaves a part at its position', () => {
    const part = { type: 'todo', data: { t: 1 }, pos: 3 };
    const segs = chatModule.splitContentByParts('abcdef', [part]);
    assert.equal(segs.length, 3);
    assert.deepEqual(segs[0], { kind: 'prose', text: 'abc', start: 0, end: 3 });
    assert.equal(segs[1].kind, 'part');
    assert.equal(segs[1].part, part);
    assert.deepEqual(segs[2], { kind: 'prose', text: 'def', start: 3, end: 6 });
});

test('splitContentByParts sorts parts and groups same-position parts adjacently', () => {
    const a = { type: 'todo', data: {}, pos: 4 };
    const b = { type: 'notice', data: {}, pos: 2 };
    const c = { type: 'todo', data: {}, pos: 2 };
    const segs = chatModule.splitContentByParts('123456', [a, b, c]);
    // prose[0,2), part(b), part(c), prose[2,4), part(a), prose[4,6)
    assert.deepEqual(segs.map((s) => s.kind), [
        'prose', 'part', 'part', 'prose', 'part', 'prose',
    ]);
    assert.equal(segs[1].part, b);
    assert.equal(segs[2].part, c);
    assert.equal(segs[4].part, a);
});

test('splitContentByParts clamps out-of-range positions to content end', () => {
    const part = { type: 'todo', data: {}, pos: 999 };
    const segs = chatModule.splitContentByParts('abc', [part]);
    assert.deepEqual(segs.map((s) => s.kind), ['prose', 'part']);
    assert.deepEqual(segs[0], { kind: 'prose', text: 'abc', start: 0, end: 3 });
});

test('splitContentByParts on empty content yields only the part (no blank prose)', () => {
    const part = { type: 'notice', data: {}, pos: 0 };
    const segs = chatModule.splitContentByParts('', [part]);
    assert.deepEqual(segs.map((s) => s.kind), ['part']);
});

test('splitContentByParts drops malformed parts lacking a string type', () => {
    const ok = { type: 'todo', data: {}, pos: 1 };
    const bad = { data: {}, pos: 2 };
    const segs = chatModule.splitContentByParts('abcd', [ok, bad]);
    assert.deepEqual(segs.map((s) => s.kind), ['prose', 'part', 'prose']);
    assert.equal(segs[1].part, ok);
});

// --------------------------------------------------------------------------
// Console-wired core renderer (notice) + safety
// --------------------------------------------------------------------------

test('mount registers the core notice renderer (no longer host-only)', () => {
    const api = makeChatContainer('notice-agent');
    // initChat (run by mount) registered core parts; appendMessagePart('notice')
    // must render the card, NOT the no-renderer escaped-text fallback.
    const div = api.appendMessagePart('notice', { title: 'Heads up', body: 'all good', level: 'success' });
    const content = div.children[0];
    assert.match(content.innerHTML, /part-notice part-notice-success/);
    assert.match(content.innerHTML, /Heads up/);
    assert.match(content.innerHTML, /all good/);
});

test('notice renderer escapes host-influenceable text (XSS-safe)', () => {
    const api = makeChatContainer('xss-agent');
    api.registerCoreParts();
    const div = api.appendMessagePart('notice', {
        title: '<script>alert(1)</script>',
        body: '<img src=x onerror=y>',
    });
    const html = div.children[0].innerHTML;
    assert.ok(!html.includes('<script>'), 'raw <script> must not survive');
    assert.ok(!html.includes('<img src=x'), 'raw <img> must not survive');
    assert.match(html, /&lt;script&gt;/);
});

test('notice renderer defaults an unknown level to info', () => {
    const api = makeChatContainer('lvl-agent');
    const div = api.appendMessagePart('notice', { body: 'x', level: 'bogus' });
    assert.match(div.children[0].innerHTML, /part-notice-info/);
});

test('appendMessagePart for an unregistered part type degrades to escaped text', () => {
    const api = makeChatContainer('unreg-agent');
    const div = api.appendMessagePart('totally-unknown-type', 'plain payload');
    // No renderer → safe escaped text fallback.
    assert.equal(div.children[0].textContent, 'plain payload');
});

// --------------------------------------------------------------------------
// #2081: persisted channel_link pairing card (renderer resolves live QR state)
// --------------------------------------------------------------------------

function makeChannelChatContainer() {
    const container = makeNode('section');
    for (const id of [
        'chat-container', 'message-input', 'send-button',
        'model-selector', 'thinking-indicator', 'composer-mode-toggle',
    ]) {
        const node = makeNode(id === 'message-input' ? 'textarea' : 'div');
        node.id = id;
        container.appendChild(node);
    }
    return chatModule.mount(container, {
        deps: {
            api: {
                hasCapability: () => true,
                getHostAgent: () => 'wa-agent',
                buildAgentUrl: (p) => `/api/agents/Emma${p}`,
                getApiKey: () => 'secret-key',
            },
            kicon: () => '',
            escapeHtml: (s) => String(s).replace(/</g, '&lt;').replace(/>/g, '&gt;'),
        },
    });
}

test('channel_link part renders a card that fetches the link-qr.png endpoint', () => {
    const api = makeChannelChatContainer();
    const div = api.appendMessagePart('channel_link', { channel_type: 'whatsapp' });
    const card = div.children[0].children[0];
    // The card carries the QR <img>; its src resolves the current QR from the
    // host endpoint (multi-agent routed + api_key + cache-bust ts).
    const img = card.children.find((c) => c.tagName === 'IMG');
    assert.ok(img, 'card must contain the QR image');
    assert.match(img.src, /\/api\/agents\/Emma\/api\/agent\/channels\/whatsapp\/link-qr\.png/);
    assert.match(img.src, /api_key=secret-key/);
    assert.match(img.src, /ts=/);
});

test('channel_link part retries a first-load 404 instead of giving up (fresh-link race)', () => {
    const api = makeChannelChatContainer();
    const div = api.appendMessagePart('channel_link', { channel_type: 'whatsapp' });
    const card = div.children[0].children[0];
    const img = card.children.find((c) => c.tagName === 'IMG');
    // A FRESH link's QR PNG can land a beat after the card renders (#2081): a
    // first-load 404 must NOT permanently kill the card, or the QR that arrives
    // milliseconds later is lost. One error → retry, no note yet, img retained.
    img.onerror();
    const note = card.children.find((c) => c.className === 'channel-link-note');
    assert.ok(!note, 'a single first-load 404 must not show the expired note');
    assert.ok(card.children.includes(img), 'the QR image must be retained for retry');
});

test('channel_link part shows expired/linked note after a loaded QR 404s', () => {
    const api = makeChannelChatContainer();
    const div = api.appendMessagePart('channel_link', { channel_type: 'whatsapp' });
    const card = div.children[0].children[0];
    const img = card.children.find((c) => c.tagName === 'IMG');
    // Once a real QR has painted (onload), a later 404 is a genuine expiry/link
    // — the honest note replaces it immediately, no grace window.
    img.onload();
    img.onerror();
    const note = card.children.find((c) => c.className === 'channel-link-note');
    assert.ok(note, 'a note must replace the dead image');
    assert.match(note.textContent, /expired or Whatsapp already linked/i);
});

test('channel_link part gives up after the bounded fresh-link retry window', () => {
    const api = makeChannelChatContainer();
    const div = api.appendMessagePart('channel_link', { channel_type: 'whatsapp' });
    const card = div.children[0].children[0];
    const img = card.children.find((c) => c.tagName === 'IMG');
    // Exhaust the bounded retry budget (never loaded) — the card finally
    // resolves to the expired/linked note rather than retrying forever.
    for (let i = 0; i < 20; i += 1) img.onerror();
    const note = card.children.find((c) => c.className === 'channel-link-note');
    assert.ok(note, 'the bounded retry window must terminate in the expired note');
});

test('channel_link card pins the QR URL to the pane-owning agent, not the selection', () => {
    // The card must resolve the QR from the agent that OWNS its pane (#2081),
    // even when a different host agent is currently selected — a live stream can
    // keep painting into a detached pane after the user switches, and a history
    // replay can render while the selection has changed. The pane element
    // carries ``dataset.agent``; ``buildAgentUrlFor`` pins to it.
    const container = makeNode('section');
    for (const id of [
        'chat-container', 'message-input', 'send-button',
        'model-selector', 'thinking-indicator', 'composer-mode-toggle',
    ]) {
        const node = makeNode(id === 'message-input' ? 'textarea' : 'div');
        node.id = id;
        container.appendChild(node);
    }
    // Owner (mounted) agent is 'Emma'; the "selected" agent buildAgentUrl would
    // use is 'Nellie' — the card must ignore the latter.
    const api = chatModule.mount(container, {
        deps: {
            api: {
                hasCapability: () => true,
                getHostAgent: () => 'Emma',
                buildAgentUrl: (p) => `/api/agents/Nellie${p}`,
                buildAgentUrlFor: (p, agent) => `/api/agents/${agent}${p}`,
                getApiKey: () => 'k',
            },
            kicon: () => '',
            escapeHtml: (s) => String(s).replace(/</g, '&lt;').replace(/>/g, '&gt;'),
        },
    });
    const div = api.appendMessagePart('channel_link', { channel_type: 'whatsapp' });
    const card = div.children[0].children[0];
    const img = card.children.find((c) => c.tagName === 'IMG');
    assert.ok(img, 'card must contain the QR image');
    assert.match(img.src, /\/api\/agents\/Emma\/api\/agent\/channels\/whatsapp\/link-qr\.png/);
    assert.ok(!img.src.includes('/Nellie/'), 'must NOT use the selected agent');
});

test('channel_link part rejects an invalid channel_type', () => {
    const api = makeChannelChatContainer();
    const div = api.appendMessagePart('channel_link', { channel_type: '../etc' });
    const card = div.children[0].children[0];
    assert.equal(card.children.length, 0);
    assert.match(card.textContent, /unavailable/i);
});
