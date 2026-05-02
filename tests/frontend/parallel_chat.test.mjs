import test from 'node:test';
import assert from 'node:assert/strict';

// End-to-end-ish test of the parallel-agent chat invariant:
//   * A stream dispatched against Agent A keeps painting into A's
//     pane element while the user views Agent B.
//   * When the user comes back to A, the streaming text is already
//     there — no DB re-fetch, no lost mid-stream content.
//   * When A finishes while B is visible, the user gets a Toast.
//   * Agent switch does NOT invalidate A's stream gate; only a
//     within-agent context change does.
//
// The test wires a controllable async-iterator fake for streamInvoke
// so we can interleave chunk arrivals with mountChatPane calls.

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
        get firstChild(){return this.children[0]||null;},
    };
    // innerHTML setter mirrors real DOM: writing a string replaces
    // children. wipeAgentChatPane relies on this — without it, post-
    // wipe DOM still contains pre-wipe message bubbles.
    Object.defineProperty(node, 'innerHTML', {
        get() { return node._innerHTML; },
        set(v) {
            node._innerHTML = String(v);
            // Detach all current children. The stub doesn't try to
            // parse HTML into nodes — non-empty strings are stored
            // verbatim and not exposed via .children.
            for (const c of node.children) c.parentNode = null;
            node.children = [];
            node.childNodes = [];
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

const { state, getOrCreateChatPane, Toast } = await import('../../kestrel_sovereign/static/js/ui.js');
const chatModule = await import('../../kestrel_sovereign/static/js/chat.js');
const { mountChatPane, wipeAgentChatPane, sendMessage } = chatModule;
const apiModule = await import('../../kestrel_sovereign/static/js/api.js');

// Initialize the chat surface (registers DOM refs, performs initial-
// pane migration). API host-agent starts at null — set it explicitly
// per test.
chatModule.initChat();

// Helper: replace API.streamInvoke with a controllable async iterator
// that lets a test push chunks at specific moments.
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

test('mid-stream return: chunks dispatched to A keep landing in A\'s pane while B is visible', async () => {
    const paneA = getOrCreateChatPane('agent-A');
    const paneB = getOrCreateChatPane('agent-B');
    apiModule.default.setHostAgent('agent-A');
    mountChatPane('agent-A');

    const ctrl = controlledStream();
    const origStream = apiModule.default.streamInvoke;
    apiModule.default.streamInvoke = () => ctrl.iter;
    apiModule.default.invoke = async () => ({ response: 'fallback' });

    messageInput.value = 'hi A';
    const sendPromise = sendMessage();

    // Let sendMessage capture the dispatch agent + pane and create
    // the streaming msgDiv on A.
    await Promise.resolve();
    await Promise.resolve();

    // First chunk lands while A is visible.
    ctrl.push('chunk-1 ');
    await new Promise((r) => setTimeout(r, 5));

    // User switches to B mid-stream.
    apiModule.default.setHostAgent('agent-B');
    mountChatPane('agent-B');

    // More chunks arrive while B is mounted — they MUST still land in
    // A's pane element (not B's, not the bare viewport).
    ctrl.push('chunk-2 ');
    ctrl.push('chunk-3');
    await new Promise((r) => setTimeout(r, 5));
    ctrl.end();

    await sendPromise;

    apiModule.default.streamInvoke = origStream;

    // A's pane should have: user message + agent message bubble
    // containing the streamed content. B's pane stays clean.
    const aMessages = paneA.element.children;
    assert.ok(aMessages.length >= 2,
        "A's pane must contain user message + agent stream bubble");
    assert.equal(paneB.element.children.length, 0,
        "B's pane must remain untouched by A's stream");
});

test('background completion fires a Toast when the dispatch agent is no longer visible', async () => {
    getOrCreateChatPane('toast-A');
    getOrCreateChatPane('toast-B');
    apiModule.default.setHostAgent('toast-A');
    mountChatPane('toast-A');

    const toasts = [];
    const origInfo = Toast.info;
    Toast.info = (msg) => toasts.push(msg);

    const ctrl = controlledStream();
    const origStream = apiModule.default.streamInvoke;
    apiModule.default.streamInvoke = () => ctrl.iter;

    messageInput.value = 'hi';
    const sendPromise = sendMessage();
    await Promise.resolve();
    await Promise.resolve();

    // Switch to B before the stream completes.
    apiModule.default.setHostAgent('toast-B');
    mountChatPane('toast-B');

    // Finish A's stream while B is visible.
    ctrl.push('done');
    ctrl.end();
    await sendPromise;

    apiModule.default.streamInvoke = origStream;
    Toast.info = origInfo;

    assert.ok(toasts.some((m) => m.includes('toast-A')),
        'completing-agent name must appear in the background-finish toast');
});

test('within-agent conversation switch (wipeAgentChatPane) gates out chunks dispatched before the switch', async () => {
    const paneA = getOrCreateChatPane('switch-A');
    apiModule.default.setHostAgent('switch-A');
    mountChatPane('switch-A');

    const ctrl = controlledStream();
    const origStream = apiModule.default.streamInvoke;
    apiModule.default.streamInvoke = () => ctrl.iter;

    messageInput.value = 'q1';
    const sendPromise = sendMessage();
    await Promise.resolve();
    await Promise.resolve();

    ctrl.push('preWipe ');
    await new Promise((r) => setTimeout(r, 5));

    const sizeBefore = paneA.element.children.length;

    // User picks a different conversation on the same agent — wipe
    // the pane and bump its generation. Chunks dispatched against
    // the old generation must stop painting from this point.
    wipeAgentChatPane('switch-A');

    ctrl.push('postWipe-should-not-render');
    ctrl.end();
    await sendPromise;

    apiModule.default.streamInvoke = origStream;

    // After the wipe, the pane was reset (innerHTML=''). Any chunk
    // arriving after the wipe is dispatchGeneration-stale and must
    // NOT have re-populated the pane.
    assert.equal(paneA.element.children.length, 0,
        'post-wipe chunks must not paint into the pane');
});

test('agent switch does NOT abort or detach an in-flight stream\'s pane writes', async () => {
    // Direct invariant: switching to B does not bump A's generation,
    // so isPaneFresh() in sendMessage stays true and chunks keep
    // landing in A's pane element.
    const paneA = getOrCreateChatPane('keep-A');
    const genA0 = paneA.generation;

    apiModule.default.setHostAgent('keep-A');
    mountChatPane('keep-A');

    const ctrl = controlledStream();
    const origStream = apiModule.default.streamInvoke;
    apiModule.default.streamInvoke = () => ctrl.iter;

    messageInput.value = 'hi';
    const sendPromise = sendMessage();
    await Promise.resolve();
    await Promise.resolve();

    // Multiple agent switches while streaming — none bumps gen.
    apiModule.default.setHostAgent('keep-B');
    mountChatPane(getOrCreateChatPane('keep-B') && 'keep-B');
    apiModule.default.setHostAgent('keep-A');
    mountChatPane('keep-A');
    apiModule.default.setHostAgent('keep-B');
    mountChatPane('keep-B');

    assert.equal(paneA.generation, genA0,
        "A's pane generation must not have moved across agent switches");

    ctrl.push('hello world');
    ctrl.end();
    await sendPromise;

    apiModule.default.streamInvoke = origStream;
});
