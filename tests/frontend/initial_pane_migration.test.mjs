import test from 'node:test';
import assert from 'node:assert/strict';

// initChat() must move the welcome content baked into #chat-container
// (see index.html line 1282+) into the first pane's element. Bare
// children left in the viewport would survive selectAgent's
// mount/detach swap and end up rendered alongside the new agent's
// pane — a visible regression where the welcome card never goes
// away when the user picks an agent.

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
        querySelector() { return null; },
        querySelectorAll() { return []; },
        appendChild(c) {
            // Real DOM semantics: if c already has a parent, detach it
            // first. Otherwise the migration loop in initChat (which
            // walks chatContainer.firstChild) would never terminate.
            if (c.parentNode && c.parentNode !== this) {
                const i = c.parentNode.children.indexOf(c);
                if (i >= 0) c.parentNode.children.splice(i, 1);
                const j = c.parentNode.childNodes.indexOf(c);
                if (j >= 0) c.parentNode.childNodes.splice(j, 1);
            }
            c.parentNode = this; this.children.push(c); this.childNodes.push(c);
            return c;
        },
        remove() { if (this.parentNode) { const i=this.parentNode.children.indexOf(this); if (i>=0) this.parentNode.children.splice(i,1); this.parentNode=null; } },
        get firstChild(){return this.children[0]||null;},
    };
}
const chatContainer = makeNode(); chatContainer.id = 'chat-container';
const messageInput = makeNode('input'); messageInput.id = 'message-input';
const sendButton = makeNode('button'); sendButton.id = 'send-button';

// Pre-seed the viewport with welcome content the way index.html does.
const welcomeMsg = makeNode('div');
welcomeMsg.classList._set.add('message');
welcomeMsg.classList._set.add('agent-message');
chatContainer.appendChild(welcomeMsg);

globalThis.document = {
    getElementById(id) {
        if (id === 'chat-container') return chatContainer;
        if (id === 'message-input') return messageInput;
        if (id === 'send-button') return sendButton;
        return null;
    },
    createElement: (t) => makeNode(t),
    head: makeNode(), body: makeNode(),
    addEventListener() {}, querySelector() { return null; }, querySelectorAll() { return []; },
};
globalThis.sessionStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.location = { href: '/', search: '' };
globalThis.fetch = async () => ({ ok: false, status: 500 });
globalThis.kicon = () => '';
globalThis.CSS = { escape: (s) => String(s) };

// Stub API.hasCapability('chat') = true via the api module's defaults
// — we don't need the real auth provider, just enough surface for
// initChat() to proceed past the capability gate.
const { state, getOrCreateChatPane } = await import('../../kestrel_sovereign/static/js/ui.js');
const chatModule = await import('../../kestrel_sovereign/static/js/chat.js');
const apiModule = await import('../../kestrel_sovereign/static/js/api.js');

test('initChat moves pre-existing #chat-container children into the first pane element', () => {
    // Sanity check: the welcome content is in the viewport before init.
    assert.equal(chatContainer.children.length, 1);
    assert.equal(chatContainer.children[0], welcomeMsg);

    // Force the capability + host-agent state initChat() reads.
    apiModule.default.setHostAgent(null);  // standalone-mode key

    chatModule.initChat();

    // After init, the viewport's only child should be the pane element,
    // and the welcome message should have moved INTO that pane.
    assert.equal(chatContainer.children.length, 1, 'viewport should hold exactly the pane element');
    const mountedPane = state.chatPanes.get(null);
    assert.ok(mountedPane, 'first-mount pane (null key) must exist');
    assert.equal(chatContainer.children[0], mountedPane.element,
        'pane element must be the viewport\'s only direct child');
    assert.equal(mountedPane.element.children[0], welcomeMsg,
        'welcome content must have moved into the pane');
});
