import test from 'node:test';
import assert from 'node:assert/strict';

// Issue #1255: when the user hits Enter while the dispatch agent is
// already streaming, sendMessage must interrupt the in-flight turn
// (stopAgent → abort + POST /api/agent/stop) BEFORE dispatching the
// new turn. The prior behavior bailed early at the busy check, so
// Enter mid-stream was a no-op and the user had to click Stop, wait,
// then start typing again.
//
// What this test pins:
//   1. Composer-disable removal — messageInput.disabled / sendButton
//      .disabled stay false even when ``state.waitingAgents.has
//      (current)`` is true. Typing while the agent thinks is the
//      whole point of the issue.
//   2. Interrupt ordering — when busy at sendMessage time, the
//      client-side AbortController.abort() and POST /api/agent/stop
//      both fire before the new streamInvoke() begins consuming
//      chunks. Otherwise two streams race on the same pane.

globalThis.window = globalThis.window || {};
globalThis.window.SharedMarkdown = {
    renderMarkdown: () => '',
    renderStreamingMarkdown: (s) => s,
    highlightCodeBlocks: () => {},
    renderMermaidDiagrams: () => {},
    finalizeMarkdown: async (el, content) => { el.textContent = content; },
};

function makeNode(tag = 'div') {
    const node = {
        tagName: tag.toUpperCase(), nodeType: 1, children: [], childNodes: [], parentNode: null,
        classList: {
            _set: new Set(),
            add(c) { this._set.add(c); },
            remove(c) { this._set.delete(c); },
            toggle(c, on) {
                if (on === undefined) {
                    this._set.has(c) ? this._set.delete(c) : this._set.add(c);
                } else if (on) { this._set.add(c); } else { this._set.delete(c); }
                return this._set.has(c);
            },
            contains(c) { return this._set.has(c); },
        },
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
        remove() {
            if (this.parentNode) {
                const i = this.parentNode.children.indexOf(this);
                if (i >= 0) this.parentNode.children.splice(i, 1);
                this.parentNode = null;
            }
        },
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
    let _className = '';
    Object.defineProperty(node, 'className', {
        get() { return _className; },
        set(v) {
            _className = String(v);
            node.classList._set = new Set(_className.split(/\s+/).filter(Boolean));
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
const { mountChatPane, sendMessage, updateThinkingIndicator } = chatModule;
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
        push(chunk) {
            buffer.push(chunk);
            if (resolveNext) resolveNext();
        },
        end() {
            done = true;
            if (resolveNext) resolveNext();
        },
    };
}


test('composer is NOT disabled while the agent is streaming (#1255)', () => {
    // The old code disabled messageInput + sendButton whenever the
    // selected agent was in state.waitingAgents. Now they stay
    // editable so the user can type their redirect while the agent
    // is still thinking — Enter routes through the interrupt branch
    // in sendMessage.
    getOrCreateChatPane('agent-busy');
    apiModule.default.setHostAgent('agent-busy');
    mountChatPane('agent-busy');

    state.waitingAgents.add('agent-busy');
    messageInput.disabled = false;
    sendButton.disabled = false;

    updateThinkingIndicator();

    assert.equal(messageInput.disabled, false,
        'messageInput must stay editable while the agent is streaming');
    assert.equal(sendButton.disabled, false,
        'sendButton must stay clickable while the agent is streaming');
    // Cleanup
    state.waitingAgents.delete('agent-busy');
});


test('sendMessage interrupts the in-flight turn before dispatching the new one (#1255)', async () => {
    getOrCreateChatPane('agent-interrupt');
    apiModule.default.setHostAgent('agent-interrupt');
    mountChatPane('agent-interrupt');

    // Track ordering: which calls fire, in what order, relative to
    // the new streamInvoke.
    const eventLog = [];

    // Pre-seed the per-agent stream state so stopAgent has something
    // to abort + ack.
    const fakeAbortController = {
        abort() { eventLog.push('abort'); },
        signal: {},
    };
    // streamAbortControllers / currentStreamRequestIds are private to
    // the api_client module; reach in via the documented API surface.
    apiModule.default.getStreamAbortController = () => fakeAbortController;
    apiModule.default.getCurrentStreamRequestId = () => 'prior-req-id';

    // Stop ack: pretend the server returns 200.
    apiModule.default.stop = async (requestId, agent) => {
        eventLog.push(`stop:${requestId}:${agent}`);
        return { ok: true };
    };

    // New stream: capture that we got there, then end cleanly so
    // sendMessage's await resolves.
    const ctrl = controlledStream();
    apiModule.default.streamInvoke = (...args) => {
        eventLog.push('streamInvoke');
        return ctrl.iter;
    };
    apiModule.default.invoke = async () => ({ response: '' });

    // Simulate the busy state — the prior turn is still streaming.
    state.waitingAgents.add('agent-interrupt');

    messageInput.value = 'wait, redirect';
    const sendPromise = sendMessage();

    // Let the interrupt-await tick through. stopAgent does an
    // `abort()` synchronously and `await API.stop(...)` once.
    await new Promise((r) => setTimeout(r, 5));

    // Close the new stream so the promise can resolve.
    ctrl.end();
    await sendPromise;

    // Assert the ordering: abort → stop POST → new streamInvoke.
    const abortIdx = eventLog.indexOf('abort');
    const stopIdx = eventLog.findIndex((e) => e.startsWith('stop:'));
    const streamIdx = eventLog.indexOf('streamInvoke');

    assert.ok(abortIdx >= 0, 'AbortController.abort() must fire on interrupt');
    assert.ok(stopIdx >= 0, 'API.stop(prior-req-id, agent) must fire on interrupt');
    assert.ok(streamIdx >= 0, 'new streamInvoke must fire after the interrupt');
    assert.ok(stopIdx < streamIdx,
        'POST /api/agent/stop must resolve BEFORE the new streamInvoke opens');
    assert.ok(abortIdx <= stopIdx,
        'client-side abort must precede or coincide with the stop POST');
});


test('sendMessage does NOT call stopAgent when the agent is idle (#1255)', async () => {
    getOrCreateChatPane('agent-idle');
    apiModule.default.setHostAgent('agent-idle');
    mountChatPane('agent-idle');

    const eventLog = [];
    apiModule.default.getStreamAbortController = () => ({
        abort() { eventLog.push('abort-should-not-fire'); },
        signal: {},
    });
    apiModule.default.getCurrentStreamRequestId = () => null;
    apiModule.default.stop = async () => {
        eventLog.push('stop-should-not-fire');
        return { ok: true };
    };

    const ctrl = controlledStream();
    apiModule.default.streamInvoke = () => {
        eventLog.push('streamInvoke');
        return ctrl.iter;
    };
    apiModule.default.invoke = async () => ({ response: '' });

    // No prior turn — agent is idle.
    state.waitingAgents.delete('agent-idle');

    messageInput.value = 'first message';
    const sendPromise = sendMessage();
    await new Promise((r) => setTimeout(r, 5));
    ctrl.end();
    await sendPromise;

    assert.equal(
        eventLog.filter((e) => e.startsWith('abort-')).length, 0,
        'abort must not fire when the agent was idle',
    );
    assert.equal(
        eventLog.filter((e) => e.startsWith('stop-')).length, 0,
        '/api/agent/stop must not fire when the agent was idle',
    );
    assert.ok(eventLog.includes('streamInvoke'),
        'streamInvoke must fire on a normal send');
});


test('sendMessage with empty text returns early regardless of busy state (#1255)', async () => {
    getOrCreateChatPane('agent-empty');
    apiModule.default.setHostAgent('agent-empty');
    mountChatPane('agent-empty');

    const eventLog = [];
    apiModule.default.getStreamAbortController = () => ({
        abort() { eventLog.push('abort'); },
        signal: {},
    });
    apiModule.default.getCurrentStreamRequestId = () => 'rid';
    apiModule.default.stop = async () => {
        eventLog.push('stop');
        return { ok: true };
    };
    apiModule.default.streamInvoke = () => {
        eventLog.push('streamInvoke');
        return (async function*() {})();
    };

    state.waitingAgents.add('agent-empty');
    messageInput.value = '   ';   // whitespace-only

    await sendMessage();

    assert.deepEqual(eventLog, [],
        'whitespace-only input must NOT trigger stop OR a new send, ' +
        'even when the agent is busy. The user is presumed to be still ' +
        'typing — interrupting on a stray Enter would be hostile.');

    state.waitingAgents.delete('agent-empty');
});
