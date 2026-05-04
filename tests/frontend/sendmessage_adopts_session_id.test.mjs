import test from 'node:test';
import assert from 'node:assert/strict';

// On the very first turn against an agent's pane, pane.sessionId is
// null. sendMessage must adopt the server-resolved session_id so the
// next turn sends it back explicitly — anchoring the pane to a
// durable conversation id and letting auto-load + context-status
// behave correctly. Reviewer flagged the prior null-stays-null hole.

globalThis.window = globalThis.window || {};
globalThis.window.SharedMarkdown = {
    renderMarkdown: () => '',
    renderStreamingMarkdown: () => '',
    highlightCodeBlocks: () => {},
    renderMermaidDiagrams: () => {},
    finalizeMarkdown: async () => {},
};
function makeNode(tag = 'div') {
    const node = {
        tagName: tag.toUpperCase(), nodeType: 1, children: [], childNodes: [], parentNode: null,
        classList: { _set: new Set(), add(c){this._set.add(c);}, remove(c){this._set.delete(c);}, toggle(c, on){if(on===undefined){this._set.has(c)?this._set.delete(c):this._set.add(c);}else if(on){this._set.add(c);}else{this._set.delete(c);}return this._set.has(c);}, contains(c){return this._set.has(c);} },
        dataset: {}, style: {}, _innerHTML: '', textContent: '',
        scrollTop: 0, scrollHeight: 0, value: '', disabled: false,
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
        appendChild(c) {
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
        get firstChild() { return this.children[0] || null; },
    };
    Object.defineProperty(node, 'innerHTML', {
        get() { return node._innerHTML; },
        set(v) {
            node._innerHTML = String(v);
            for (const c of node.children) c.parentNode = null;
            node.children = []; node.childNodes = [];
        },
    });
    return node;
}

const chatContainer = makeNode(); chatContainer.id = 'chat-container';
const messageInput = makeNode('input'); messageInput.id = 'message-input';
const sendButton = makeNode('button'); sendButton.id = 'send-button';
const thinkingIndicator = makeNode(); thinkingIndicator.id = 'thinking-indicator';
globalThis.document = {
    getElementById(id) {
        if (id === 'chat-container') return chatContainer;
        if (id === 'message-input') return messageInput;
        if (id === 'send-button') return sendButton;
        if (id === 'thinking-indicator') return thinkingIndicator;
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

const { state, getOrCreateChatPane } = await import('../../kestrel_sovereign/static/js/ui.js');
const chatModule = await import('../../kestrel_sovereign/static/js/chat.js');
const { mountChatPane, sendMessage } = chatModule;
const apiModule = await import('../../kestrel_sovereign/static/js/api.js');

chatModule.initChat();

function controlledStream() {
    let resolveNext = null;
    let buffer = [];
    let done = false;
    const iter = (async function* () {
        while (true) {
            if (buffer.length) { yield buffer.shift(); continue; }
            if (done) return;
            await new Promise((r) => { resolveNext = r; });
            resolveNext = null;
        }
    })();
    return {
        iter,
        push(chunk) { buffer.push(chunk); if (resolveNext) resolveNext(); },
        end() { done = true; if (resolveNext) resolveNext(); },
    };
}

test('first turn: sendMessage adopts server-resolved session_id onto pane.sessionId', async () => {
    const pane = getOrCreateChatPane('first-A');
    apiModule.default.setHostAgent('first-A');
    mountChatPane('first-A');
    pane.sessionId = null;  // virgin pane

    const ctrl = controlledStream();
    const origStream = apiModule.default.streamInvoke;
    const origGetSid = apiModule.default.getEffectiveSessionId;
    apiModule.default.streamInvoke = () => ctrl.iter;
    apiModule.default.getEffectiveSessionId = (a) =>
        a === 'first-A' ? 'sess-from-server' : null;

    messageInput.value = 'hi';
    const sendPromise = sendMessage();
    await Promise.resolve(); await Promise.resolve();

    ctrl.push('hello');
    await new Promise((r) => setTimeout(r, 5));
    ctrl.end();
    await sendPromise;

    apiModule.default.streamInvoke = origStream;
    apiModule.default.getEffectiveSessionId = origGetSid;

    assert.equal(pane.sessionId, 'sess-from-server',
        'pane.sessionId must be adopted from API.getEffectiveSessionId on the first chunk');
});

test('subsequent turn: sendMessage does NOT overwrite an existing pane.sessionId', async () => {
    // Once a pane is anchored, the explicit session_id rides on every
    // subsequent turn. The server echoes it back unchanged. The
    // adoption logic must not re-write it (avoids a UI flicker if the
    // server happened to return a different id for any reason).
    const pane = getOrCreateChatPane('anchored-A');
    apiModule.default.setHostAgent('anchored-A');
    mountChatPane('anchored-A');
    pane.sessionId = 'pre-existing-sess';

    const ctrl = controlledStream();
    const origStream = apiModule.default.streamInvoke;
    const origGetSid = apiModule.default.getEffectiveSessionId;
    apiModule.default.streamInvoke = () => ctrl.iter;
    apiModule.default.getEffectiveSessionId = () => 'different-sess-the-server-derived';

    messageInput.value = 'hi again';
    const sendPromise = sendMessage();
    await Promise.resolve(); await Promise.resolve();
    ctrl.push('reply');
    ctrl.end();
    await sendPromise;

    apiModule.default.streamInvoke = origStream;
    apiModule.default.getEffectiveSessionId = origGetSid;

    assert.equal(pane.sessionId, 'pre-existing-sess',
        'an already-anchored pane.sessionId must NOT be overwritten');
});

test('non-streaming fallback also adopts session_id from response.session_id', async () => {
    const pane = getOrCreateChatPane('nonstream-A');
    apiModule.default.setHostAgent('nonstream-A');
    mountChatPane('nonstream-A');
    pane.sessionId = null;

    const origInvokeForAgent = apiModule.default.invokeForAgent;
    const origUseStreaming = state.useStreaming;
    state.useStreaming = false;  // force non-streaming path
    apiModule.default.invokeForAgent = async () => ({
        response: 'plain reply',
        session_id: 'json-sess-7',
    });

    messageInput.value = 'hi';
    await sendMessage();

    state.useStreaming = origUseStreaming;
    apiModule.default.invokeForAgent = origInvokeForAgent;

    assert.equal(pane.sessionId, 'json-sess-7',
        'non-streaming path must adopt pane.sessionId from response.session_id');
});
