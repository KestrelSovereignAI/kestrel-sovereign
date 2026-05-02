import test from 'node:test';
import assert from 'node:assert/strict';

// voice/ui.js calls addMessageStreaming('agent') with no paneElement
// arg. The pane-aware refactor of chat.js's helpers must keep that
// no-arg path working — the helpers default to the currently-mounted
// agent's pane. This test pins that contract so a future signature
// tweak can't silently break voice mode.

globalThis.window = globalThis.window || {};
globalThis.window.SharedMarkdown = {
    renderMarkdown: () => '',
    renderStreamingMarkdown: () => '',
    highlightCodeBlocks: () => {},
    renderMermaidDiagrams: () => {},
    finalizeMarkdown: async () => {},
};
function makeNode(tag = 'div') {
    return {
        tagName: tag.toUpperCase(), nodeType: 1, children: [], childNodes: [], parentNode: null,
        classList: { _set: new Set(), add(c){this._set.add(c);}, remove(c){this._set.delete(c);}, toggle(){}, contains(c){return this._set.has(c);} },
        dataset: {}, style: {}, innerHTML: '', textContent: '',
        scrollTop: 0, scrollHeight: 0,
        addEventListener() {},
        querySelector(sel) {
            if (sel === '.message-content') {
                for (const c of this.children) {
                    if (c.classList && c.classList.contains('message-content')) return c;
                }
            }
            return null;
        },
        querySelectorAll() { return []; },
        appendChild(c) { c.parentNode = this; this.children.push(c); this.childNodes.push(c); return c; },
        remove() { if (this.parentNode) { const i=this.parentNode.children.indexOf(this); if (i>=0) this.parentNode.children.splice(i,1); this.parentNode=null; } },
        get firstChild(){return this.children[0]||null;},
    };
}
const chatContainer = makeNode(); chatContainer.id = 'chat-container';
globalThis.document = {
    getElementById(id) { return id === 'chat-container' ? chatContainer : null; },
    createElement: (t) => makeNode(t),
    head: makeNode(), body: makeNode(),
    addEventListener() {}, querySelector() { return null; }, querySelectorAll() { return []; },
};
globalThis.sessionStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.location = { href: '/', search: '' };
globalThis.fetch = async () => ({ ok: false, status: 500 });
globalThis.kicon = () => '';
globalThis.CSS = { escape: (s) => String(s) };

const { state, getOrCreateChatPane } = await import('../../kestrel_sovereign/static/js/ui.js');
const { mountChatPane, addMessageStreaming, addMessage } = await import(
    '../../kestrel_sovereign/static/js/chat.js'
);

test('addMessageStreaming() with no paneElement appends into the currently-mounted pane', () => {
    const paneA = getOrCreateChatPane('voice-test-a');
    mountChatPane('voice-test-a');

    const div = addMessageStreaming('agent');

    assert.ok(div, 'must return the message div');
    assert.equal(div.parentNode, paneA.element,
        'no-arg call must land in the mounted pane element, not the bare viewport');
});

test('switching mounted agent retargets the no-arg helper to the new pane', () => {
    getOrCreateChatPane('voice-test-x');
    const paneY = getOrCreateChatPane('voice-test-y');
    mountChatPane('voice-test-x');
    mountChatPane('voice-test-y');

    const div = addMessageStreaming('agent');
    assert.equal(div.parentNode, paneY.element);
});

test('addMessage() with no paneElement also defaults to the mounted pane', async () => {
    const paneZ = getOrCreateChatPane('voice-test-z');
    mountChatPane('voice-test-z');

    await addMessage('user', 'hi');

    assert.equal(paneZ.element.children.length, 1);
});
