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

// The notification stream is pinned to the agent captured at connect time
// (`notificationAgent`), NOT whatever pane is mounted now — so a test that
// only mounts a pane would assert against a DIFFERENT pane than the handler
// writes to, and its "nothing was painted" assertions would pass vacuously.
// Rebind the stream to the freshly-mounted pane so the destination the
// handler resolves is the one under test.
function mountNotificationPane(name) {
    const pane = mountFreshPane(name);
    class SilentES {
        constructor(_url) {}
        addEventListener() {}
        close() {}
    }
    const origES = globalThis.EventSource;
    globalThis.EventSource = SilentES;
    try {
        connectNotifications();
    } finally {
        globalThis.EventSource = origES;
    }
    return pane;
}

test('mountNotificationPane binds the handler destination to the pane', async () => {
    // Guards the guard: if this stops holding, every assertion below that
    // expects NOTHING to be painted would pass for the wrong reason.
    const pane = mountNotificationPane('wake-binding-selfcheck');
    pane.sessionId = null;
    await handleSignalCompleted(wakePayload({
        signal_id: 'selfcheck-1',
        result_summary: 'Self-check body.',
    }));
    assert.ok(pane.element.children.find(
        (c) => c.classList && c.classList.contains('signal-wake-message')),
        'the pane under test must be the handler destination');
});

// #2877: a wake can now be BOUND to the chat session that registered the
// work (a Talon job completion resumes the session that dispatched it, as a
// restart wake does since #1809). The notifications stream is pinned to the
// agent, not to a session, so agent-pinning alone no longer picks the right
// destination: if the pane has been switched to another conversation while
// the job ran, painting the wake there shows a turn that is not in the
// transcript the reader is looking at — and that reappears in the OTHER
// conversation on reload.

test('a session-bound wake paints into its own conversation', async () => {
    const pane = mountNotificationPane('wake-session-match');
    pane.sessionId = 'sess-A';
    await handleSignalCompleted(wakePayload({
        signal_id: 'bound-match-1',
        session_id: 'sess-A',
        result_summary: 'Talon job finished; dispatched attempt 4.',
    }));

    const wakeMsg = pane.element.children.find(
        (c) => c.classList && c.classList.contains('signal-wake-message'));
    assert.ok(wakeMsg, 'a wake bound to the displayed session must render');
    assert.match(wakeMsg.querySelector('.message-content').innerHTML,
        /dispatched attempt 4/);
});

test('a wake bound to another conversation is not painted here', async () => {
    const pane = mountNotificationPane('wake-session-mismatch');
    pane.sessionId = 'sess-B';
    await handleSignalCompleted(wakePayload({
        signal_id: 'bound-mismatch-1',
        session_id: 'sess-A',
        result_summary: 'Belongs to the other thread.',
    }));

    assert.equal(pane.element.children.length, 0,
        'the wake persisted into sess-A; painting it into sess-B would show '
        + 'a turn that is not in this transcript');
});

test('a session-less wake still renders (unattended cron/CLI work)', async () => {
    const pane = mountNotificationPane('wake-session-none');
    pane.sessionId = 'sess-B';
    await handleSignalCompleted(wakePayload({
        signal_id: 'unbound-1',
        session_id: null,
        result_summary: 'Unattended job finished.',
    }));

    const wakeMsg = pane.element.children.find(
        (c) => c.classList && c.classList.contains('signal-wake-message'));
    assert.ok(wakeMsg,
        'an unattended wake has no originating conversation — the pre-#2877 '
        + 'agent-pinned destination stays correct');
});

test('a bound wake renders when the pane has no conversation yet', async () => {
    const pane = mountNotificationPane('wake-session-unbound-pane');
    pane.sessionId = null;
    await handleSignalCompleted(wakePayload({
        signal_id: 'bound-nopane-1',
        session_id: 'sess-A',
        result_summary: 'Arrived before the pane bound a conversation.',
    }));

    assert.ok(pane.element.children.find(
        (c) => c.classList && c.classList.contains('signal-wake-message')),
        'an unbound pane has nothing to mismatch against — do not drop the wake');
});

test('a filtered wake is not consumed by the dedupe set', async () => {
    // The mismatch check must run BEFORE the signal_id is recorded as
    // rendered; otherwise switching back to the originating conversation
    // (or a reconnect replay) would find the wake already "seen" and the
    // turn would never paint anywhere.
    const away = mountNotificationPane('wake-session-refilter');
    away.sessionId = 'sess-B';
    await handleSignalCompleted(wakePayload({
        signal_id: 'refilter-1', session_id: 'sess-A',
        result_summary: 'Deferred body.',
    }));
    assert.equal(away.element.children.length, 0);

    const home = mountNotificationPane('wake-session-refilter-home');
    home.sessionId = 'sess-A';
    await handleSignalCompleted(wakePayload({
        signal_id: 'refilter-1', session_id: 'sess-A',
        result_summary: 'Deferred body.',
    }));
    assert.ok(home.element.children.find(
        (c) => c.classList && c.classList.contains('signal-wake-message')),
        'a wake dropped for a session mismatch must stay renderable');
});
