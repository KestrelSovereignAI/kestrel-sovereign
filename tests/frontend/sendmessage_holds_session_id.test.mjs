import test from 'node:test';
import assert from 'node:assert/strict';

// #3654: a chat pane never sends a turn without a session.
//
// A pane that held none (a cold load before auto-load committed, a purge of the
// open conversation) used to send session_id=null and adopt whatever the
// server's 30-minute rule picked, from the X-Session-Id header or the
// non-streaming body. That rule picks the newest conversation, which for an
// agent that chains its own work is its autonomous session: the operator's
// turn, and every later turn from the pane, was filed into the chain. The pane
// now mints a fresh session itself before it sends, and only ever sends what
// it holds.

const UUID_V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const AUTONOMOUS = 'autonomous-chain-session';

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
const { mintChatSessionId, mountChatPane, sendMessage, wipeAgentChatPane } = chatModule;
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

function captureStream() {
    const ctrl = controlledStream();
    const sent = [];
    const origStream = apiModule.default.streamInvoke;
    const origGetSid = apiModule.default.getEffectiveSessionId;
    apiModule.default.streamInvoke = (...args) => {
        sent.push(args[2]);
        return ctrl.iter;
    };
    // What the server would have answered for a session-less turn: the agent's
    // own chain, newest conversation by the 30-minute rule.
    apiModule.default.getEffectiveSessionId = () => AUTONOMOUS;
    return {
        ctrl,
        sent,
        restore() {
            apiModule.default.streamInvoke = origStream;
            apiModule.default.getEffectiveSessionId = origGetSid;
        },
    };
}

async function streamOneTurn(text, capture, { midTurn } = {}) {
    messageInput.value = text;
    const sendPromise = sendMessage();
    await Promise.resolve(); await Promise.resolve();
    if (midTurn) midTurn();
    capture.ctrl.push('reply');
    await new Promise((r) => setTimeout(r, 5));
    capture.ctrl.end();
    await sendPromise;
}

test('a pane with no session mints one and sends it explicitly', async () => {
    const pane = getOrCreateChatPane('cold-A');
    apiModule.default.setHostAgent('cold-A');
    mountChatPane('cold-A');
    pane.sessionId = null;  // a cold pane

    const capture = captureStream();
    try {
        await streamOneTurn('hi', capture);
    } finally {
        capture.restore();
    }

    assert.equal(capture.sent.length, 1);
    assert.match(capture.sent[0], UUID_V4, 'the turn carries a session of its own');
    assert.equal(pane.sessionId, capture.sent[0], 'the pane holds the session it sent');
});

test('a UI send from a cleared pane does not adopt the autonomous session id', async () => {
    const pane = getOrCreateChatPane('cleared-A');
    apiModule.default.setHostAgent('cleared-A');
    mountChatPane('cleared-A');
    pane.sessionId = 'the-conversation-just-purged';
    wipeAgentChatPane('cleared-A');
    assert.equal(pane.sessionId, null, 'the wipe leaves the pane without a session');

    const capture = captureStream();
    try {
        await streamOneTurn('first after the purge', capture);
        await streamOneTurn('and the next one', capture);
    } finally {
        capture.restore();
    }

    const [first, second] = capture.sent;
    assert.match(first, UUID_V4);
    assert.notEqual(first, AUTONOMOUS);
    assert.notEqual(first, 'the-conversation-just-purged');
    assert.equal(second, first, 'later turns from the pane stay in the session it started');
    assert.equal(pane.sessionId, first, 'the server\'s choice is never adopted');
});

test('subsequent turn: sendMessage sends and keeps an existing pane.sessionId', async () => {
    const pane = getOrCreateChatPane('anchored-A');
    apiModule.default.setHostAgent('anchored-A');
    mountChatPane('anchored-A');
    pane.sessionId = 'pre-existing-sess';

    const capture = captureStream();
    try {
        await streamOneTurn('hi again', capture);
    } finally {
        capture.restore();
    }

    assert.deepEqual(capture.sent, ['pre-existing-sess']);
    assert.equal(pane.sessionId, 'pre-existing-sess',
        'an already-anchored pane.sessionId must NOT be overwritten');
});

test('a pane wiped mid-turn is not re-anchored to the turn it left behind', async () => {
    const pane = getOrCreateChatPane('wiped-A');
    apiModule.default.setHostAgent('wiped-A');
    mountChatPane('wiped-A');
    pane.sessionId = 'conversation-being-left';

    const capture = captureStream();
    try {
        await streamOneTurn('long question', capture, {
            midTurn: () => wipeAgentChatPane('wiped-A'),
        });
    } finally {
        capture.restore();
    }

    assert.deepEqual(capture.sent, ['conversation-being-left']);
    assert.equal(pane.sessionId, null,
        'a cleared pane stays cleared until its own next send starts a session');
});

test('non-streaming path sends the pane\'s minted session and keeps it', async () => {
    const pane = getOrCreateChatPane('nonstream-A');
    apiModule.default.setHostAgent('nonstream-A');
    mountChatPane('nonstream-A');
    pane.sessionId = null;

    const origInvokeForAgent = apiModule.default.invokeForAgent;
    const origCompleteRequest = apiModule.default.completeCurrentStreamRequestId;
    const origUseStreaming = state.useStreaming;
    state.useStreaming = false;  // force non-streaming path
    let invokeArgs;
    const completedRequests = [];
    apiModule.default.invokeForAgent = async (...args) => {
        invokeArgs = args;
        return {
            response: 'plain reply',
            session_id: AUTONOMOUS,
        };
    };
    apiModule.default.completeCurrentStreamRequestId = (...args) => {
        completedRequests.push(args);
        return true;
    };

    messageInput.value = 'hi';
    try {
        await sendMessage();
    } finally {
        state.useStreaming = origUseStreaming;
        apiModule.default.invokeForAgent = origInvokeForAgent;
        apiModule.default.completeCurrentStreamRequestId = origCompleteRequest;
    }

    assert.match(invokeArgs[2], UUID_V4, 'the non-streaming turn carries a session too');
    assert.equal(pane.sessionId, invokeArgs[2],
        'the body\'s session_id does not replace the one the pane sent');
    assert.equal(typeof invokeArgs[5], 'string');
    assert.ok(invokeArgs[5].length > 0,
        'non-streaming chat must carry an exact Stop request id');
    assert.deepEqual(completedRequests, [['nonstream-A', invokeArgs[5]]],
        'the owning UI turn must release the exact Stop address with busy state');
});

test('mintChatSessionId builds a v4 UUID where randomUUID is unavailable', () => {
    // A console served over plain http on a LAN address is not a secure
    // context: it has getRandomValues but no randomUUID.
    const original = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
    const realCrypto = globalThis.crypto;
    Object.defineProperty(globalThis, 'crypto', {
        configurable: true,
        value: { getRandomValues: (array) => realCrypto.getRandomValues(array) },
    });
    try {
        assert.equal(typeof globalThis.crypto.randomUUID, 'undefined');
        const minted = new Set(Array.from({ length: 64 }, () => mintChatSessionId()));
        assert.equal(minted.size, 64);
        for (const id of minted) assert.match(id, UUID_V4);
    } finally {
        Object.defineProperty(globalThis, 'crypto', original);
    }
    assert.match(mintChatSessionId(), UUID_V4);
});

test('a superseded non-streaming response cannot paint into the replacement turn', async () => {
    const pane = getOrCreateChatPane('nonstream-stale');
    apiModule.default.setHostAgent('nonstream-stale');
    mountChatPane('nonstream-stale');

    const origInvokeForAgent = apiModule.default.invokeForAgent;
    const origUseStreaming = state.useStreaming;
    state.useStreaming = false;
    let release;
    apiModule.default.invokeForAgent = () => new Promise((resolve) => {
        release = resolve;
    });

    messageInput.value = 'old turn';
    const pending = sendMessage();
    await Promise.resolve();
    await Promise.resolve();
    pane.activeTurnId += 1;
    release({ response: 'must not paint' });
    await pending;

    state.useStreaming = origUseStreaming;
    apiModule.default.invokeForAgent = origInvokeForAgent;
    state.waitingAgents.delete('nonstream-stale');

    assert.equal(
        pane.element.children.filter(
            (child) => child.className?.includes('agent-message'),
        ).length,
        0,
        'only the current turn owner may render a non-streaming response',
    );
});
