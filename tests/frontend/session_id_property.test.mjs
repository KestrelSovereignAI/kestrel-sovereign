import test from 'node:test';
import assert from 'node:assert/strict';

// state.currentSessionId is now per-agent. The property is wired so
// reads/writes route into whichever agent's pane is currently
// mounted. This preserves all the existing call sites in
// chat.js / history.js / identity.js that read or assign it directly,
// while making each agent retain its own session across switches.
//
// Goal of this test: lock the routing semantics so a future refactor
// can't silently break the "Agent B remembers its session when I
// switch back" property.

globalThis.window = globalThis.window || {};
globalThis.window.SharedMarkdown = {
    renderMarkdown: () => '',
    renderStreamingMarkdown: () => '',
    highlightCodeBlocks: () => {},
    renderMermaidDiagrams: () => {},
    finalizeMarkdown: async () => {},
};
function makeNode() {
    return {
        nodeType: 1, children: [], childNodes: [], parentNode: null,
        classList: { _set: new Set(), add(c){this._set.add(c);}, remove(c){this._set.delete(c);}, toggle(){}, contains(c){return this._set.has(c);} },
        dataset: {}, style: {}, innerHTML: '', textContent: '',
        scrollTop: 0, scrollHeight: 0,
        addEventListener() {}, querySelector() { return null; }, querySelectorAll() { return []; },
        appendChild(c) { c.parentNode = this; this.children.push(c); this.childNodes.push(c); return c; },
        remove() { if (this.parentNode) { const i=this.parentNode.children.indexOf(this); if (i>=0) this.parentNode.children.splice(i,1); this.parentNode=null; } },
        get firstChild(){return this.children[0]||null;},
    };
}
const chatContainer = makeNode(); chatContainer.id = 'chat-container';
globalThis.document = {
    getElementById(id) { return id === 'chat-container' ? chatContainer : null; },
    createElement: () => makeNode(),
    head: makeNode(), body: makeNode(),
    addEventListener() {}, querySelector() { return null; }, querySelectorAll() { return []; },
};
globalThis.sessionStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.location = { href: '/', search: '' };
globalThis.fetch = async () => ({ ok: false, status: 500 });
globalThis.kicon = () => '';
globalThis.CSS = { escape: (s) => String(s) };

const { state, getOrCreateChatPane } = await import('../../kestrel_sovereign/static/js/ui.js');
const { mountChatPane } = await import('../../kestrel_sovereign/static/js/chat.js');

test('reading currentSessionId before any mount returns null (no phantom pane)', () => {
    state.mountedChatAgent = undefined;  // simulate pre-init
    assert.equal(state.currentSessionId, null);
});

test('writing currentSessionId before any mount is a no-op (no phantom pane)', () => {
    state.mountedChatAgent = undefined;
    const before = state.chatPanes.size;
    state.currentSessionId = 'phantom-sess';
    assert.equal(state.chatPanes.size, before, 'must not create a pane keyed on undefined');
    assert.equal(state.currentSessionId, null, 'still null because nothing was actually written');
});

test('writes route into the currently-mounted agent\'s pane', () => {
    getOrCreateChatPane('agent-x');
    getOrCreateChatPane('agent-y');
    mountChatPane('agent-x');

    state.currentSessionId = 'sess-x-1';
    assert.equal(state.chatPanes.get('agent-x').sessionId, 'sess-x-1');
    assert.equal(state.chatPanes.get('agent-y').sessionId, null);

    mountChatPane('agent-y');
    state.currentSessionId = 'sess-y-7';
    assert.equal(state.chatPanes.get('agent-y').sessionId, 'sess-y-7');
    assert.equal(state.chatPanes.get('agent-x').sessionId, 'sess-x-1', 'X must keep its session');
});

test('reading currentSessionId after switching back to A returns A\'s session, not B\'s', () => {
    getOrCreateChatPane('agent-A');
    getOrCreateChatPane('agent-B');

    mountChatPane('agent-A');
    state.currentSessionId = 'A-sess';
    mountChatPane('agent-B');
    state.currentSessionId = 'B-sess';

    mountChatPane('agent-A');
    assert.equal(state.currentSessionId, 'A-sess',
        'currentSessionId must follow the mounted agent — that\'s how each agent keeps its own conversation');
});

test('null agent (standalone mode) is a valid pane key', () => {
    getOrCreateChatPane(null);
    mountChatPane(null);
    state.currentSessionId = 'standalone-sess';
    assert.equal(state.chatPanes.get(null).sessionId, 'standalone-sess');
    assert.equal(state.currentSessionId, 'standalone-sess');
});
