import test from 'node:test';
import assert from 'node:assert/strict';

// Wave 5C of #1048: when the server fires a `revising` SSE event for
// the dispatch's request_id, the chat UI must drop the pre-tool prose
// it already painted into the in-flight bubble and let the next
// streamed chunk start fresh.
//
// The protocol pieces:
// * Wave 5B emits `revising` from agent/streaming.py the moment a
//   ToolCallStarted marker arrives in the LLM stream.
// * /api/agent/notifications/sse forwards the event with payload
//   { request_id, index, tool_name, ... } so multi-pane / multi-tab
//   clients can route to the right bubble.
// * Wave 5C (this) registers a listener that flips
//   `pane.pendingRevise = true` and the streaming loop honors that
//   on the next chunk by resetting the accumulated text.
//
// These tests exercise the streaming-loop reset directly. The SSE
// listener registration is covered by inspecting the connectNotifications
// machinery indirectly — flipping `pane.pendingRevise` is the
// observable contract.

// Spy hooks: every call to renderStreamingMarkdown / finalizeMarkdown
// captures its content arg so tests can assert on the FULL accumulated
// text fed through the rendering pipeline at each chunk boundary +
// finalization. Tests reset the buffer between cases.
const renderCalls = [];
const finalizeCalls = [];
globalThis.window = globalThis.window || {};
globalThis.window.SharedMarkdown = {
    renderMarkdown: () => '',
    renderStreamingMarkdown: (s) => { renderCalls.push(s); return ''; },
    highlightCodeBlocks: () => {},
    renderMermaidDiagrams: () => {},
    finalizeMarkdown: async (_, s) => { finalizeCalls.push(s); },
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
    // Setting className must populate classList so querySelector
    // by class can find the node — the chat code does
    // contentDiv.className = 'message-content streaming' before
    // updateStreamingMessage walks up via querySelector('.message-content').
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

test('pendingRevise flag pinned on pane shape (initial value + reset on wipe)', async () => {
    const pane = getOrCreateChatPane('shape-test');
    assert.equal(pane.pendingRevise, false,
        'getOrCreateChatPane must initialize pendingRevise=false');

    // Simulate a server-fired revise mid-stream
    pane.pendingRevise = true;
    chatModule.wipeAgentChatPane('shape-test');
    assert.equal(pane.pendingRevise, false,
        'wipeAgentChatPane must reset pendingRevise so a stale flag from '
        + 'the wiped conversation can\'t bleed into the next dispatch');
});

test('streaming loop drops pre-tool prose when pane.pendingRevise is set', async () => {
    // Canonical Wave 5D shape: agent emits "Saving that now..." pre-tool,
    // server fires revising on ToolCallStarted, agent emits post-tool
    // synthesis chunks. The bubble must end up rendering ONLY the
    // post-tool synthesis — the pre-tool optimistic text is retracted.
    renderCalls.length = 0; finalizeCalls.length = 0;

    const pane = getOrCreateChatPane('revise-A');
    apiModule.default.setHostAgent('revise-A');
    mountChatPane('revise-A');
    pane.sessionId = 'sess-1';

    const ctrl = controlledStream();
    const origStream = apiModule.default.streamInvoke;
    apiModule.default.streamInvoke = () => ctrl.iter;

    messageInput.value = 'save my color';
    const sendPromise = sendMessage();
    await Promise.resolve(); await Promise.resolve();

    // Pre-tool chunks paint into the streaming bubble.
    ctrl.push('Saving that now');
    ctrl.push('...');
    await new Promise((r) => setTimeout(r, 5));

    // Pre-revise renders include the pre-tool prose — invariant of
    // the optimistic-streaming UX.
    const preRevisePreview = renderCalls[renderCalls.length - 1];
    assert.ok(preRevisePreview.includes('Saving that now'),
        `pre-revise renders must show pre-tool prose, got: ${JSON.stringify(preRevisePreview)}`);

    // Server fires `revising` SSE → listener flips this flag.
    pane.pendingRevise = true;

    // Post-tool synthesis chunks arrive AFTER tool execution.
    ctrl.push('Looking at the result, ');
    ctrl.push('the save did not persist.');
    await new Promise((r) => setTimeout(r, 5));
    ctrl.end();
    await sendPromise;

    apiModule.default.streamInvoke = origStream;

    // The finalized text fed to finalizeMarkdown must contain ONLY
    // the post-tool synthesis. The pre-tool optimistic claim was
    // dropped at the revise boundary.
    assert.equal(finalizeCalls.length, 1,
        'finalizeMarkdown should run exactly once at end of stream');
    const finalText = finalizeCalls[0];
    assert.ok(
        finalText.includes('did not persist'),
        `post-tool synthesis must reach finalizeMarkdown, got: ${JSON.stringify(finalText)}`,
    );
    assert.ok(
        !finalText.includes('Saving that now'),
        `pre-tool optimistic prose must NOT reach finalizeMarkdown, got: ${JSON.stringify(finalText)}`,
    );

    // Flag is cleared once the stream completes so the next turn
    // starts clean.
    assert.equal(pane.pendingRevise, false);
});

test('streaming loop without revise leaves pre-tool prose intact (regression guard)', async () => {
    // Stream that never has a revise event must concatenate every
    // chunk verbatim — protects existing single-pass behavior.
    renderCalls.length = 0; finalizeCalls.length = 0;

    const pane = getOrCreateChatPane('no-revise-A');
    apiModule.default.setHostAgent('no-revise-A');
    mountChatPane('no-revise-A');
    pane.sessionId = 'sess-2';

    const ctrl = controlledStream();
    const origStream = apiModule.default.streamInvoke;
    apiModule.default.streamInvoke = () => ctrl.iter;

    messageInput.value = 'just chat';
    const sendPromise = sendMessage();
    await Promise.resolve(); await Promise.resolve();

    ctrl.push('Hello! ');
    ctrl.push('No tools needed today.');
    await new Promise((r) => setTimeout(r, 5));
    ctrl.end();
    await sendPromise;

    apiModule.default.streamInvoke = origStream;

    const finalText = finalizeCalls[finalizeCalls.length - 1] || '';
    assert.ok(finalText.includes('Hello!'),
        `regression: no-revise final text dropped 'Hello!', got: ${JSON.stringify(finalText)}`);
    assert.ok(finalText.includes('No tools needed today.'));
});

test('SSE listener routes revising event to the matching pane only', async () => {
    // Codex P3: prior tests flipped pane.pendingRevise directly, so
    // a broken listener registration / request-id matcher would
    // still pass. Drive the path end-to-end through a fake
    // EventSource: register the listener, dispatch an event, assert
    // ONLY the matching pane gets the placeholder + flag.
    renderCalls.length = 0; finalizeCalls.length = 0;

    // Fake EventSource that captures handlers and exposes a fire helper.
    const handlers = new Map();
    class FakeES {
        constructor(_url) { FakeES.last = this; }
        addEventListener(name, h) {
            const list = handlers.get(name) || [];
            list.push(h);
            handlers.set(name, list);
        }
        close() {}
    }
    const origES = globalThis.EventSource;
    globalThis.EventSource = FakeES;

    // Two panes with distinct active request_ids.
    const paneA = getOrCreateChatPane('sse-A');
    const paneB = getOrCreateChatPane('sse-B');
    paneA.pendingRevise = false; paneB.pendingRevise = false;

    // Mount A so we have a streaming bubble there too — addMessageStreaming
    // requires a mounted target.
    apiModule.default.setHostAgent('sse-A');
    mountChatPane('sse-A');

    // Stub request-id lookup so the SSE listener can match the event.
    const origGetReqId = apiModule.default.getCurrentStreamRequestId;
    apiModule.default.getCurrentStreamRequestId = (agent) => {
        if (agent === 'sse-A') return 'rid-A';
        if (agent === 'sse-B') return 'rid-B';
        return null;
    };

    // Give each pane an in-flight bubble to retract.
    const msgA = chatModule.addMessageStreaming('agent', paneA.element);
    const msgB = chatModule.addMessageStreaming('agent', paneB.element);
    paneA.streamingMsgDiv = msgA;
    paneB.streamingMsgDiv = msgB;

    // Register the listener by reconnecting notifications.
    chatModule.connectNotifications();
    const reviseHandlers = handlers.get('revising') || [];
    assert.ok(reviseHandlers.length >= 1,
        'connectNotifications must register a revising listener');

    // Fire a revising event for rid-A.
    for (const h of reviseHandlers) {
        h({ data: JSON.stringify({
            type: 'revising', request_id: 'rid-A',
            index: 0, tool_name: 'save_fact',
        }) });
    }

    assert.equal(paneA.pendingRevise, true,
        'pane A (matching request_id) must have pendingRevise flipped');
    assert.equal(paneB.pendingRevise, false,
        'pane B (non-matching) must NOT be touched');

    // The matching pane's bubble shows the placeholder; the
    // non-matching pane's bubble is untouched.
    const slotA = msgA.querySelector('.message-content') || msgA;
    const slotB = msgB.querySelector('.message-content') || msgB;
    assert.ok(
        (slotA._innerHTML || '').includes('Revising'),
        `pane A bubble must show the revising placeholder, got: ${JSON.stringify(slotA._innerHTML)}`,
    );
    assert.ok(
        !(slotB._innerHTML || '').includes('Revising'),
        'pane B bubble must NOT show the placeholder',
    );

    // A revising event with no matching active request_id is a no-op
    // (silent drop) — covers the dispatch-already-finished case.
    paneA.pendingRevise = false; // re-arm
    for (const h of reviseHandlers) {
        h({ data: JSON.stringify({ request_id: 'rid-NOBODY' }) });
    }
    assert.equal(paneA.pendingRevise, false,
        'unmatched request_id must not flip any pane');

    globalThis.EventSource = origES;
    apiModule.default.getCurrentStreamRequestId = origGetReqId;
});

test('multiple concurrent panes: revise on pane A leaves pane B untouched', async () => {
    // Each pane has independent pendingRevise state. A revising event
    // for dispatch on agent A must not affect the in-flight stream on
    // agent B.
    const paneA = getOrCreateChatPane('multi-A');
    const paneB = getOrCreateChatPane('multi-B');
    paneB.pendingRevise = false;

    apiModule.default.setHostAgent('multi-A');
    mountChatPane('multi-A');
    paneA.sessionId = 'sess-A';

    const ctrlA = controlledStream();
    const origStream = apiModule.default.streamInvoke;
    apiModule.default.streamInvoke = () => ctrlA.iter;

    messageInput.value = 'do thing A';
    const sendPromiseA = sendMessage();
    await Promise.resolve(); await Promise.resolve();

    ctrlA.push('Saving A');
    await new Promise((r) => setTimeout(r, 5));
    paneA.pendingRevise = true;
    ctrlA.push('post-tool A');
    await new Promise((r) => setTimeout(r, 5));
    ctrlA.end();
    await sendPromiseA;

    apiModule.default.streamInvoke = origStream;

    // pane B must not have inherited the revise flag from pane A.
    assert.equal(paneB.pendingRevise, false,
        'a revise on agent A must not flip agent B\'s pendingRevise');
});
