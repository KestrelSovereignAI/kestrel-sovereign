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
        _listeners: {},
        addEventListener(type, handler) {
            (this._listeners[type] = this._listeners[type] || []).push(handler);
        },
        click() {
            for (const handler of this._listeners.click || []) handler({});
        },
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
        result_summary_complete: true,
        reply_message_id: null,
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

// #3655: `result_summary` is capped (2 KB on the event, less for sources that
// shorten it themselves), so a long reply arrived cut off — and the bubble
// presented the cut-off text as the reply. The dispatcher now says whether
// the summary is the whole reply (`result_summary_complete`) and names the
// stored row holding it (`reply_message_id`). The bubble loads the full reply
// from that row, retrying briefly when the row is not readable yet, and
// otherwise labels the summary a truncated preview with a load action.

const TRUNCATED = 'After a time limit...(truncated)';
const FULL_REPLY = 'After a time limit expires, the restart proceeds. ' + 'x'.repeat(3000);

function truncatedWake(overrides = {}) {
    return wakePayload({
        source: 'restart.completed',
        result_summary: TRUNCATED,
        result_summary_complete: false,
        reply_message_id: 4242,
        ...overrides,
    });
}

function wakeBubbles(pane) {
    return pane.element.children.filter(
        (c) => c.classList && c.classList.contains('signal-wake-message'));
}

function noticeOf(bubble) {
    return bubble.children.find(
        (c) => c.classList && c.classList.contains('signal-wake-truncation')) || null;
}

function noticeText(notice) {
    return notice.children.map((c) => c.textContent).join(' ');
}

function loadButtonOf(notice) {
    return notice.children.find(
        (c) => c.classList && c.classList.contains('signal-wake-load-full')) || null;
}

// A fake `getMessage` that answers from a script: each entry is either the
// stored row or an Error (the row not readable yet). Records every call.
function scriptedGetMessage(script) {
    const calls = [];
    const fn = async (messageId, agent) => {
        calls.push({ messageId, agent });
        const next = script.length > 1 ? script.shift() : script[0];
        if (next instanceof Error) throw next;
        return next;
    };
    fn.calls = calls;
    return fn;
}

function storedReply(content = FULL_REPLY, id = 4242) {
    return { id, role: 'assistant', content, session_id: null, metadata: {} };
}

function notFound() {
    const err = new Error('Message not found.');
    err.status = 404;
    return err;
}

async function flushMicrotasks() {
    for (let i = 0; i < 50; i++) await Promise.resolve();
}

// Drive a handler that awaits mocked timers: flush its microtasks, then
// advance the mocked clock, until it settles. No real time passes.
async function settleWithMockedTimers(t, promise) {
    let done = false;
    promise.then(() => { done = true; }, () => { done = true; });
    for (let i = 0; i < 20 && !done; i++) {
        await flushMicrotasks();
        if (!done) t.mock.timers.tick(5000);
    }
    await promise;
}

function withGetMessage(t, fn) {
    const api = apiModule.default;
    const original = api.getMessage;
    api.getMessage = fn;
    t.after(() => { api.getMessage = original; });
}

test('truncated summary: the bubble shows the full stored reply', async (t) => {
    const getMessage = scriptedGetMessage([storedReply()]);
    withGetMessage(t, getMessage);
    const pane = mountNotificationPane('wake-full-reply');
    pane.sessionId = null;

    await handleSignalCompleted(truncatedWake({ signal_id: 'full-1' }));

    const [bubble] = wakeBubbles(pane);
    assert.ok(bubble, 'the wake must render');
    const content = bubble.querySelector('.message-content');
    assert.equal(content.innerHTML, FULL_REPLY,
        'the stored reply replaces the capped summary');
    assert.ok(!content.classList.contains('signal-wake-preview'));
    assert.equal(noticeOf(bubble), null,
        'a resolved reply carries no truncation notice');
    assert.equal(bubble.dataset.messageId, '4242');
    assert.deepEqual(getMessage.calls, [{ messageId: 4242, agent: 'wake-full-reply' }],
        'the reply is read by the id on the event, from the stream\'s agent');
});

test('truncated summary: the row read is retried until readable, without real sleeps', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const getMessage = scriptedGetMessage([notFound(), notFound(), storedReply()]);
    withGetMessage(t, getMessage);
    const pane = mountNotificationPane('wake-retry');
    pane.sessionId = null;

    const handled = handleSignalCompleted(truncatedWake({ signal_id: 'retry-1' }));
    await flushMicrotasks();

    // While the row is unreadable the summary is shown, labelled a preview.
    const [bubble] = wakeBubbles(pane);
    const content = bubble.querySelector('.message-content');
    assert.equal(content.innerHTML, TRUNCATED);
    assert.ok(content.classList.contains('signal-wake-preview'));
    assert.match(noticeText(noticeOf(bubble)), /Truncated preview/,
        'the capped summary is never presented as the reply');

    await settleWithMockedTimers(t, handled);

    assert.equal(getMessage.calls.length, 3);
    assert.equal(content.innerHTML, FULL_REPLY);
    assert.equal(noticeOf(bubble), null);
});

test('truncated summary: an unreadable row leaves a labelled preview with a load action', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const getMessage = scriptedGetMessage([notFound()]);
    withGetMessage(t, getMessage);
    const pane = mountNotificationPane('wake-unreadable');
    pane.sessionId = null;

    await settleWithMockedTimers(
        t, handleSignalCompleted(truncatedWake({ signal_id: 'unreadable-1' })));

    assert.equal(getMessage.calls.length, 4,
        'one read plus three bounded retries, then it stops');
    const [bubble] = wakeBubbles(pane);
    const content = bubble.querySelector('.message-content');
    assert.equal(content.innerHTML, TRUNCATED);
    assert.ok(content.classList.contains('signal-wake-preview'));
    const notice = noticeOf(bubble);
    assert.match(noticeText(notice), /Truncated preview/);
    const button = loadButtonOf(notice);
    assert.ok(button, 'the bubble offers to load the full message');
    assert.equal(button.textContent, 'Load full message');

    // The row becomes readable; the load action fetches it.
    getMessage.calls.length = 0;
    const api = apiModule.default;
    api.getMessage = scriptedGetMessage([storedReply()]);
    button.click();
    await flushMicrotasks();
    assert.equal(content.innerHTML, FULL_REPLY);
    assert.equal(noticeOf(bubble), null);
});

test('truncated summary with no stored row: preview label, load reopens the conversation', async (t) => {
    const getMessage = scriptedGetMessage([storedReply()]);
    withGetMessage(t, getMessage);
    const loads = [];
    const originalLoad = globalThis.window.loadConversation;
    globalThis.window.loadConversation = (sessionId, options) => {
        loads.push({ sessionId, options });
    };
    t.after(() => { globalThis.window.loadConversation = originalLoad; });

    const pane = mountNotificationPane('wake-unstored');
    pane.sessionId = 'sess-W';
    await handleSignalCompleted(truncatedWake({
        signal_id: 'unstored-1', session_id: 'sess-W', reply_message_id: null,
    }));

    assert.equal(getMessage.calls.length, 0, 'there is no row id to read');
    const [bubble] = wakeBubbles(pane);
    assert.equal(bubble.querySelector('.message-content').innerHTML, TRUNCATED);
    const notice = noticeOf(bubble);
    assert.match(noticeText(notice), /Truncated preview/);
    const button = loadButtonOf(notice);
    assert.ok(button);
    button.click();
    assert.deepEqual(loads, [{
        sessionId: 'sess-W',
        options: { force: true, expectedAgent: 'wake-unstored' },
    }]);
});

test('an untruncated summary renders as the reply with no extra request', async (t) => {
    const getMessage = scriptedGetMessage([storedReply()]);
    withGetMessage(t, getMessage);
    const pane = mountNotificationPane('wake-complete');
    pane.sessionId = null;

    await handleSignalCompleted(wakePayload({
        signal_id: 'complete-1',
        result_summary: 'Short and whole.',
        result_summary_complete: true,
        reply_message_id: 77,
    }));

    assert.equal(getMessage.calls.length, 0,
        'a summary that is the whole reply has nothing to fetch');
    const [bubble] = wakeBubbles(pane);
    assert.equal(bubble.querySelector('.message-content').innerHTML, 'Short and whole.');
    assert.equal(noticeOf(bubble), null);
});

test('a summary not marked complete is never rendered as the reply', async (t) => {
    // An event without the flag (or with any value but true) is a preview.
    const getMessage = scriptedGetMessage([storedReply('The whole reply.', 9)]);
    withGetMessage(t, getMessage);
    const pane = mountNotificationPane('wake-unflagged');
    pane.sessionId = null;

    await handleSignalCompleted(wakePayload({
        signal_id: 'unflagged-1',
        result_summary: 'The whole',
        result_summary_complete: undefined,
        reply_message_id: 9,
    }));

    const [bubble] = wakeBubbles(pane);
    assert.equal(bubble.querySelector('.message-content').innerHTML, 'The whole reply.');
});

test('a redelivered truncated wake renders once, while its reply is still loading', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const getMessage = scriptedGetMessage([notFound(), storedReply()]);
    withGetMessage(t, getMessage);
    const pane = mountNotificationPane('wake-dedupe-loading');
    pane.sessionId = null;

    const first = handleSignalCompleted(truncatedWake({ signal_id: 'dupe-loading-1' }));
    await flushMicrotasks();
    await handleSignalCompleted(truncatedWake({ signal_id: 'dupe-loading-1' }));
    await settleWithMockedTimers(t, first);

    assert.equal(wakeBubbles(pane).length, 1);
    assert.equal(getMessage.calls.length, 2, 'the redelivery issues no read of its own');
});

test('a pane that already shows the stored reply gets no second copy', async (t) => {
    // History loaded after the reply was stored and before the event arrived.
    const getMessage = scriptedGetMessage([storedReply()]);
    withGetMessage(t, getMessage);
    const pane = mountNotificationPane('wake-history-first');
    pane.sessionId = 'sess-H';
    const historical = makeNode();
    historical.className = 'message agent-message';
    historical.dataset.messageId = '4242';
    pane.element.appendChild(historical);

    await handleSignalCompleted(truncatedWake({
        signal_id: 'history-first-1', session_id: 'sess-H',
    }));

    assert.equal(wakeBubbles(pane).length, 0);
    assert.equal(getMessage.calls.length, 0);
});

test('a row that is not an assistant reply is never shown as the reply', async (t) => {
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const getMessage = scriptedGetMessage([{
        id: 4242, role: 'user', content: 'a wake prompt', metadata: {},
    }]);
    withGetMessage(t, getMessage);
    const pane = mountNotificationPane('wake-wrong-role');
    pane.sessionId = null;

    await settleWithMockedTimers(
        t, handleSignalCompleted(truncatedWake({ signal_id: 'wrong-role-1' })));

    const [bubble] = wakeBubbles(pane);
    assert.equal(bubble.querySelector('.message-content').innerHTML, TRUNCATED);
    assert.match(noticeText(noticeOf(bubble)), /could not be loaded/);
});

test('an agent switch while the bubble renders still reads the reply from the waking agent', async (t) => {
    // The stream is rebound to another agent when the user switches agents,
    // and the handler awaits rendering before it reads the stored row. Each
    // agent has its own database, so the same id can name an unrelated
    // message in the other one: the read must target the agent whose stream
    // announced the wake, captured before the first await.
    const replies = {
        'wake-switch-A': storedReply('Agent A full reply.', 4242),
        'wake-switch-B': storedReply('Agent B unrelated message.', 4242),
    };
    const calls = [];
    withGetMessage(t, async (messageId, agent) => {
        calls.push({ messageId, agent });
        return replies[agent];
    });

    const paneA = mountNotificationPane('wake-switch-A');
    paneA.sessionId = null;
    const handled = handleSignalCompleted(truncatedWake({ signal_id: 'switch-1' }));

    // Switch to B before the handler has read the row.
    assert.equal(calls.length, 0, 'the switch must land before the read');
    const paneB = mountNotificationPane('wake-switch-B');
    paneB.sessionId = null;
    await handled;

    assert.deepEqual(calls, [{ messageId: 4242, agent: 'wake-switch-A' }],
        'the row is read from the agent that wrote it');
    const [bubble] = wakeBubbles(paneA);
    assert.ok(bubble, 'the wake renders into the waking agent\'s pane');
    assert.equal(bubble.querySelector('.message-content').innerHTML, 'Agent A full reply.');
    assert.equal(wakeBubbles(paneB).length, 0, 'nothing lands in the other agent\'s pane');
});
