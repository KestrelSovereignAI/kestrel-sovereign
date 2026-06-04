import test from 'node:test';
import assert from 'node:assert/strict';

// #1522: an autonomous cognition wake (an A2A reply resuming an asked
// question, a scheduled wake, etc.) runs in the dispatcher background —
// it never streams through the chat composer. The dispatcher emits a
// `signal_completed` SSE event after the turn logs; the chat UI must
// render the response into the active pane in real time, otherwise the
// open chat stays blank until a manual refresh (the reported bug:
// "seems like it did not wake you. at least not in this CHAT").
//
// These tests drive the path end-to-end through a fake EventSource:
// connectNotifications registers the listener, an event is dispatched,
// and we assert the wake is (or is not) painted into the visible pane.

globalThis.window = globalThis.window || {};
globalThis.window.SharedMarkdown = {
    renderMarkdown: (s) => String(s),
    renderStreamingMarkdown: (s) => String(s),
    highlightCodeBlocks: () => {},
    renderMermaidDiagrams: () => {},
    finalizeMarkdown: async (node, s) => { node.innerHTML = String(s); },
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

const { getOrCreateChatPane } = await import('../../kestrel_sovereign/static/js/ui.js');
const chatModule = await import('../../kestrel_sovereign/static/js/chat.js');
const { mountChatPane, handleSignalCompleted, connectNotifications } = chatModule;
const apiModule = await import('../../kestrel_sovereign/static/js/api.js');

chatModule.initChat();

function mountFreshPane(name) {
    const pane = getOrCreateChatPane(name);
    pane.element.children.length = 0;
    apiModule.default.setHostAgent(name);
    mountChatPane(name);
    return pane;
}

function wakePayload(overrides = {}) {
    return {
        signal_id: 'sig-' + Math.random().toString(36).slice(2),
        source: 'a2a.question_answered',
        kind: 'answered',
        mode: 'cognition',
        target_agent: 'did:test:sender',
        session_id: null,
        caller: 'Meridian',
        visibility: 'user_visible',
        status: 'ok',
        result_summary: 'Meridian says the deploy is green.',
        ...overrides,
    };
}

test('user_visible cognition wake renders into the active pane', async () => {
    const pane = mountFreshPane('wake-render');
    await handleSignalCompleted(wakePayload());

    const wakeMsg = pane.element.children.find(
        (c) => c.classList && c.classList.contains('signal-wake-message'));
    assert.ok(wakeMsg, 'a signal-wake-message must be appended to the pane');

    const content = wakeMsg.querySelector('.message-content');
    assert.ok(content, 'wake message must carry a .message-content slot');
    assert.match(content.innerHTML, /deploy is green/,
        'the agent response (result_summary) must be rendered into the pane');
});

test('metadata-only wake (no result_summary) paints nothing', async () => {
    const pane = mountFreshPane('wake-empty');
    await handleSignalCompleted(wakePayload({ result_summary: null }));
    assert.equal(pane.element.children.length, 0,
        'a wake with no body has nothing to render — the pane stays empty');
});

test('INTERNAL / non-cognition signals never reach the chat stream', async () => {
    const pane = mountFreshPane('wake-internal');
    await handleSignalCompleted(wakePayload({ visibility: 'internal' }));
    await handleSignalCompleted(wakePayload({ visibility: 'admin_visible' }));
    await handleSignalCompleted(wakePayload({ mode: 'action' }));
    assert.equal(pane.element.children.length, 0,
        'only USER_VISIBLE cognition wakes render as chat messages');
});

test('duplicate signal_id is rendered at most once (SSE replay/reconnect)', async () => {
    const pane = mountFreshPane('wake-dedupe');
    const payload = wakePayload({ signal_id: 'dupe-1' });
    await handleSignalCompleted(payload);
    await handleSignalCompleted(payload);
    const wakeMsgs = pane.element.children.filter(
        (c) => c.classList && c.classList.contains('signal-wake-message'));
    assert.equal(wakeMsgs.length, 1,
        'the dispatcher startup-replay sweep + live subscription must not '
        + 'double-paint one wake');
});

test('SSE listener routes signal_completed end-to-end into the pane', async () => {
    // Drive through a fake EventSource so a broken listener registration
    // (the actual bug surface) fails the test, not just the handler.
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

    const pane = mountFreshPane('wake-sse');
    connectNotifications();

    const sigHandlers = handlers.get('signal_completed') || [];
    assert.ok(sigHandlers.length >= 1,
        'connectNotifications must register a signal_completed listener');

    for (const h of sigHandlers) {
        h({ data: JSON.stringify(wakePayload({
            signal_id: 'sse-wake-1',
            result_summary: 'Live wake via the SSE stream.',
        })) });
    }
    // Handler is async; let its microtasks flush.
    await new Promise((r) => setTimeout(r, 5));

    const wakeMsg = pane.element.children.find(
        (c) => c.classList && c.classList.contains('signal-wake-message'));
    assert.ok(wakeMsg, 'the registered listener must paint the wake into the pane');
    assert.match(wakeMsg.querySelector('.message-content').innerHTML, /Live wake/);

    globalThis.EventSource = origES;
});
