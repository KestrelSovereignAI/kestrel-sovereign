import test from 'node:test';
import assert from 'node:assert/strict';

function makeNode(tag = 'div') {
    return {
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
            toggle(c, on) {
                if (on === undefined) {
                    this._set.has(c) ? this._set.delete(c) : this._set.add(c);
                } else if (on) {
                    this._set.add(c);
                } else {
                    this._set.delete(c);
                }
                return this._set.has(c);
            },
            contains(c) { return this._set.has(c); },
        },
        dataset: {},
        style: {},
        innerHTML: '',
        textContent: '',
        addEventListener() {},
        appendChild(child) {
            child.parentNode = this;
            this.children.push(child);
            this.childNodes.push(child);
            return child;
        },
        querySelector(selector) {
            if (!selector.startsWith('.')) return null;
            const klass = selector.slice(1);
            const stack = [...this.children];
            while (stack.length) {
                const child = stack.shift();
                if (child.classList?.contains(klass)) return child;
                stack.push(...(child.children || []));
            }
            return null;
        },
        querySelectorAll() { return []; },
    };
}

const header = makeNode('header');
header.classList.add('chat-header');

globalThis.window = globalThis.window || {};
globalThis.document = {
    querySelector(selector) {
        return selector === '.chat-header' ? header : null;
    },
    querySelectorAll() { return []; },
    createElement: (tag) => makeNode(tag),
    head: makeNode('head'),
    body: makeNode('body'),
    addEventListener() {},
};
globalThis.sessionStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.location = { href: '/', search: '' };
globalThis.fetch = async () => ({ ok: false, status: 404 });
globalThis.kicon = () => '';
globalThis.CSS = { escape: (s) => String(s) };
globalThis.KESTREL_UI_CONFIG = { capabilities: { voice: false } };

const { initVoiceUI, mountAgentVoiceControls } = await import(
    '../../kestrel_sovereign/static/js/voice/ui.js'
);

test('voice capability false prevents mic UI from mounting', () => {
    initVoiceUI();
    assert.equal(header.querySelector('.voice-toggle-btn'), null);
    assert.equal(header.children.length, 0);

    const row = makeNode('div');
    mountAgentVoiceControls(row, 'agent-a');
    assert.equal(row.children.length, 0);
});
