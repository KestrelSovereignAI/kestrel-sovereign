// Regression tests for the registerHeaderAction XSS fix (#1650).
//
// registerHeaderAction is the exported embedder API (#1623/#1627), so a
// third party controls `label`/`icon`. `label` must always be escaped as
// text; an `icon` Node must be appended (not stringified). These tests use a
// purpose-built document mock whose `textContent` setter escapes & < > like a
// browser, so the real ui.js `escapeHtml` round-trips faithfully.
import test from 'node:test';
import assert from 'node:assert/strict';

function escapeRef(s) {
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

function makeNode(tag = 'div') {
    return {
        tagName: String(tag).toUpperCase(),
        nodeType: 1,
        id: '',
        className: '',
        title: '',
        value: '',
        children: [],
        childNodes: [],
        parentNode: null,
        style: {},
        dataset: {},
        _ih: '',
        classList: { add() {}, remove() {}, contains() { return false; }, toggle() {} },
        get innerHTML() { return this._ih; },
        set innerHTML(v) {
            this._ih = v;
            if (v === '') { this.children = []; this.childNodes = []; }
        },
        // Browser textContent -> innerHTML escaping (what escapeHtml relies on).
        get textContent() { return this._ih; },
        set textContent(v) { this._ih = escapeRef(v); },
        addEventListener() {},
        appendChild(child) {
            child.parentNode = this;
            this.children.push(child);
            this.childNodes.push(child);
            return child;
        },
        querySelector(sel) {
            const match = (n) => {
                if (sel.startsWith('#')) return n.id === sel.slice(1);
                if (sel.startsWith('.')) {
                    return String(n.className || '').split(/\s+/).includes(sel.slice(1));
                }
                return n.tagName === sel.toUpperCase();
            };
            const stack = [...this.children];
            while (stack.length) {
                const c = stack.shift();
                if (match(c)) return c;
                stack.push(...c.children);
            }
            return null;
        },
        querySelectorAll() { return []; },
    };
}

const documentRoot = makeNode();
globalThis.window = globalThis.window || {};
globalThis.window.SharedMarkdown = {
    renderMarkdown: () => '',
    renderStreamingMarkdown: () => '',
    highlightCodeBlocks: () => {},
    renderMermaidDiagrams: () => {},
    finalizeMarkdown: async () => {},
};
globalThis.document = {
    getElementById(id) { return documentRoot.querySelector('#' + id); },
    createElement(tag) { return makeNode(tag); },
    createTextNode(text) {
        return { nodeType: 3, textContent: String(text), parentNode: null };
    },
    head: makeNode('head'),
    body: makeNode('body'),
    addEventListener() {},
    querySelector(sel) { return documentRoot.querySelector(sel); },
    querySelectorAll() { return []; },
};
globalThis.sessionStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.location = { href: '/', search: '' };
globalThis.fetch = async () => ({ ok: false, status: 500 });
globalThis.kicon = () => '';
globalThis.CSS = { escape: (s) => String(s) };
globalThis.EventSource = class { close() {} addEventListener() {} };

const chat = await import('../../kestrel_sovereign/static/js/chat.js');
const { registerHeaderAction, setChatRoot } = chat;

function freshRootWithHeader() {
    const root = makeNode('section');
    const header = makeNode('div');
    header.className = 'chat-header';
    root.appendChild(header);
    setChatRoot(root);
    return root;
}

function buttonByTitle(root, title) {
    const slot = root.querySelector('#chat-header-actions');
    assert.ok(slot, 'header-actions slot was created');
    return slot.children.find((b) => b.title === title);
}

test('registerHeaderAction escapes a third-party label string (no raw HTML)', () => {
    const root = freshRootWithHeader();
    registerHeaderAction({
        id: 'xss-label',
        title: 'evil-label',
        label: '<img src=x onerror=alert(1)>',
        onClick() {},
    });

    const btn = buttonByTitle(root, 'evil-label');
    assert.ok(btn, 'button rendered');
    assert.ok(!btn.innerHTML.includes('<img'), 'raw <img must not reach innerHTML');
    assert.ok(btn.innerHTML.includes('&lt;img'), 'label is HTML-escaped');
});

test('registerHeaderAction appends a DOM Node icon instead of stringifying it', () => {
    const root = freshRootWithHeader();
    const iconNode = document.createElement('span');
    iconNode.id = 'safe-icon';
    registerHeaderAction({
        id: 'node-icon',
        title: 'node-icon-btn',
        icon: iconNode,
        label: 'Selfie',
        onClick() {},
    });

    const btn = buttonByTitle(root, 'node-icon-btn');
    assert.ok(btn, 'button rendered');
    assert.equal(btn.children[0], iconNode, 'icon Node appended as a child');
    // Label rides along as a real text node (nodeType 3), never as innerHTML
    // markup — so a malicious label can't inject even when an icon Node is used.
    const textNode = btn.childNodes[1];
    assert.equal(textNode.nodeType, 3, 'label is appended as a text node');
    assert.equal(textNode.textContent, ' Selfie', 'label text node carries the raw label as text');
    assert.equal(btn.innerHTML, '', 'no markup written to button innerHTML in the Node-icon path');
});
