import test from 'node:test';
import assert from 'node:assert/strict';

// Issue #1257 (Phase 2 of epic #1254): per-pane "Queue" mode. In
// queue mode, hitting Enter while the agent is streaming stores the
// message instead of interrupting (#1255's default), and the message
// dispatches automatically when the in-flight turn finishes.
//
// Pinned behavior:
//   1. Queue mode: Enter-while-busy stores to pane.queuedMessage,
//      renders a chip, does NOT call stopAgent (no interrupt).
//   2. The queued message dispatches when the in-flight turn's
//      finally runs — via queueMicrotask, after the prior turn's
//      async context unwinds (the #1255 ordering lesson).
//   3. The queued dispatch targets the ORIGINAL agent even if the
//      user switched agents while it was queued.
//   4. Stop = stop everything: clears the queue, nothing dispatches.
//   5. A conversation switch (wipeAgentChatPane) discards the queue.
//   6. Re-Enter while queued REPLACES the queued message (1 slot).
//   7. Interrupt mode (default) still interrupts — regression guard.
//   8. A prior-turn ERROR still dispatches the queued message.
//   9. The toggle is per-pane and reflects the mounted agent's mode.

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
        scrollTop: 0, scrollHeight: 0, value: '', disabled: false, title: '', type: '',
        _listeners: {},
        addEventListener(ev, fn) { (this._listeners[ev] ||= []).push(fn); },
        click() { (this._listeners.click || []).forEach((fn) => fn()); },
        querySelector(sel) {
            // Depth-first search by single class selector ".cls".
            const cls = sel.startsWith('.') ? sel.slice(1) : null;
            const walk = (n) => {
                for (const c of n.children) {
                    if (cls && c.classList && c.classList.contains(cls)) return c;
                    const found = walk(c);
                    if (found) return found;
                }
                return null;
            };
            return walk(this);
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
const messageInput = makeNode('textarea'); messageInput.id = 'message-input';
const sendButton = makeNode('button'); sendButton.id = 'send-button';
const thinkingIndicator = makeNode(); thinkingIndicator.id = 'thinking-indicator';
const composerModeToggle = makeNode('button'); composerModeToggle.id = 'composer-mode-toggle';

globalThis.document = {
    getElementById(id) {
        if (id === 'chat-container') return chatContainer;
        if (id === 'message-input') return messageInput;
        if (id === 'send-button') return sendButton;
        if (id === 'thinking-indicator') return thinkingIndicator;
        if (id === 'composer-mode-toggle') return composerModeToggle;
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
const { mountChatPane, wipeAgentChatPane, sendMessage, stopAgent } = chatModule;
const apiModule = await import('../../kestrel_sovereign/static/js/api.js');

chatModule.initChat();

function controlledStream() {
    let resolveNext = null;
    let buffer = [];
    let done = false;
    let pendingError = null;
    const iter = (async function* () {
        while (true) {
            if (pendingError) { const e = pendingError; pendingError = null; throw e; }
            if (buffer.length) { yield buffer.shift(); continue; }
            if (done) return;
            await new Promise((r) => { resolveNext = r; });
            resolveNext = null;
        }
    })();
    return {
        iter,
        push(c) { buffer.push(c); if (resolveNext) resolveNext(); },
        end() { done = true; if (resolveNext) resolveNext(); },
        error(err) { pendingError = err; if (resolveNext) resolveNext(); },
    };
}

function setQueueMode(agent) {
    const pane = getOrCreateChatPane(agent);
    pane.composerMode = 'queue';
    return pane;
}


test('queue mode: Enter-while-busy stores the message and renders a chip, no interrupt', async () => {
    const agent = 'q-store';
    apiModule.default.setHostAgent(agent);
    mountChatPane(agent);
    const pane = setQueueMode(agent);

    let stopCalled = false;
    apiModule.default.getStreamAbortController = () => ({
        abort() { stopCalled = true; }, signal: {},
    });
    apiModule.default.getCurrentStreamRequestId = () => 'rid';
    apiModule.default.stop = async () => { stopCalled = true; return { ok: true }; };

    // Simulate the agent already streaming.
    state.waitingAgents.add(agent);

    messageInput.value = 'follow-up question';
    await sendMessage();

    assert.equal(pane.queuedMessage, 'follow-up question',
        'queue mode must store the message instead of interrupting');
    assert.equal(stopCalled, false, 'queue mode must NOT call stopAgent');
    assert.equal(messageInput.value, '', 'composer cleared after queueing');
    const chip = pane.element.querySelector('.queued-message-chip');
    assert.ok(chip, 'a pending-queued chip must be rendered');
    assert.ok(state.waitingAgents.has(agent),
        'the in-flight turn keeps running (still busy)');

    state.waitingAgents.delete(agent);
    pane.queuedMessage = null;
});


test('queued message dispatches when the in-flight turn finishes', async () => {
    const agent = 'q-dispatch';
    apiModule.default.setHostAgent(agent);
    mountChatPane(agent);
    setQueueMode(agent);

    const ctrl = controlledStream();
    const dispatched = [];
    apiModule.default.streamInvoke = (input, ...rest) => {
        dispatched.push(input);
        return ctrl.iter;
    };
    apiModule.default.invoke = async () => ({ response: '' });

    // First (in-flight) turn.
    messageInput.value = 'first turn';
    const firstPromise = sendMessage();
    await new Promise((r) => setTimeout(r, 5));
    assert.deepEqual(dispatched, ['first turn']);

    // Queue a follow-up while it streams.
    messageInput.value = 'queued follow-up';
    await sendMessage();
    const pane = getOrCreateChatPane(agent);
    assert.equal(pane.queuedMessage, 'queued follow-up');

    // Set up the SECOND streamInvoke (the queued dispatch).
    const ctrl2 = controlledStream();
    apiModule.default.streamInvoke = (input, ...rest) => {
        dispatched.push(input);
        return ctrl2.iter;
    };

    // Finish the first turn → finally fires → queued dispatch.
    ctrl.end();
    await firstPromise;
    // Let the queueMicrotask + the queued sendMessage's first awaits run.
    await new Promise((r) => setTimeout(r, 10));

    assert.deepEqual(dispatched, ['first turn', 'queued follow-up'],
        'the queued message must dispatch as the next turn');
    assert.equal(pane.queuedMessage, null, 'queued slot cleared after dispatch');
    assert.equal(pane.element.querySelector('.queued-message-chip'), null,
        'chip removed once the queued message is dispatched');

    ctrl2.end();
    await new Promise((r) => setTimeout(r, 5));
});


test('Stop with a SLOW /stop POST and a normally-completing turn still cancels the queue (codex P2)', async () => {
    // Regression for the codex review finding on PR #1276: stopAgent
    // used to clear the queue AFTER `await API.stop()`. If the POST
    // is slow and the in-flight turn completes NORMALLY (e.g. no
    // abort controller was registered, so wasAborted stays false),
    // the turn's finally runs first with pane.queuedMessage still set
    // and dispatches the message the user just cancelled. The fix
    // clears the queue synchronously at the top of stopAgent, before
    // any await — this test fails if that clear regresses behind an
    // await.
    const agent = 'q-slow-stop';
    apiModule.default.setHostAgent(agent);
    mountChatPane(agent);
    const pane = setQueueMode(agent);

    const ctrl = controlledStream();
    const dispatched = [];
    apiModule.default.streamInvoke = (input) => { dispatched.push(input); return ctrl.iter; };
    apiModule.default.invoke = async () => ({ response: '' });
    // No abort controller registered → stopAgent's abort() is skipped
    // and the in-flight stream completes NORMALLY (wasAborted=false).
    apiModule.default.getStreamAbortController = () => null;
    apiModule.default.getCurrentStreamRequestId = () => 'rid';
    // SLOW /stop POST — resolves well after the in-flight turn's
    // finally would have run.
    apiModule.default.stop = async () => {
        await new Promise((r) => setTimeout(r, 30));
        return { ok: true };
    };

    messageInput.value = 'in-flight turn';
    const firstPromise = sendMessage();
    await new Promise((r) => setTimeout(r, 5));

    messageInput.value = 'queued, user will press Stop';
    await sendMessage();
    assert.equal(pane.queuedMessage, 'queued, user will press Stop');

    // User presses Stop. Do NOT await it yet — simulate the POST
    // being in flight while the turn finishes.
    const stopPromise = stopAgent(agent);

    // Synchronous-clear means the queue is already gone right here,
    // before the slow POST resolves.
    assert.equal(pane.queuedMessage, null,
        'Stop must clear the queue synchronously, before awaiting /stop');

    // The in-flight turn completes normally NOW, while /stop is still
    // in flight. Its finally must not find a queued message.
    ctrl.end();
    await firstPromise;
    await new Promise((r) => setTimeout(r, 10));

    assert.deepEqual(dispatched, ['in-flight turn'],
        'the queued message must NOT dispatch — the user pressed Stop, ' +
        'even though /stop was slow and the turn completed normally');

    await stopPromise;
});


test('queued dispatch targets the original agent even after an agent switch', async () => {
    const agentA = 'q-A';
    const agentB = 'q-B';
    apiModule.default.setHostAgent(agentA);
    mountChatPane(agentA);
    setQueueMode(agentA);
    getOrCreateChatPane(agentB);

    const ctrl = controlledStream();
    const dispatchedAgents = [];
    apiModule.default.streamInvoke = (input, model, sid, provider, retried, agent) => {
        dispatchedAgents.push(agent);
        return ctrl.iter;
    };
    apiModule.default.invoke = async () => ({ response: '' });

    messageInput.value = 'A first';
    const firstPromise = sendMessage();
    await new Promise((r) => setTimeout(r, 5));

    messageInput.value = 'A queued';
    await sendMessage();   // queued against agentA

    // User switches to agent B before A's turn finishes.
    apiModule.default.setHostAgent(agentB);
    mountChatPane(agentB);

    const ctrl2 = controlledStream();
    apiModule.default.streamInvoke = (input, model, sid, provider, retried, agent) => {
        dispatchedAgents.push(agent);
        return ctrl2.iter;
    };

    ctrl.end();
    await firstPromise;
    await new Promise((r) => setTimeout(r, 10));

    assert.deepEqual(dispatchedAgents, [agentA, agentA],
        'the queued message must dispatch against the ORIGINAL agent ' +
        '(agentA), not the currently-mounted agentB');

    ctrl2.end();
    await new Promise((r) => setTimeout(r, 5));
    apiModule.default.setHostAgent(agentA);
});


test('Stop while queued = stop everything: queue cleared, nothing dispatches', async () => {
    const agent = 'q-stop';
    apiModule.default.setHostAgent(agent);
    mountChatPane(agent);
    const pane = setQueueMode(agent);

    const ctrl = controlledStream();
    const dispatched = [];
    apiModule.default.streamInvoke = (input) => { dispatched.push(input); return ctrl.iter; };
    apiModule.default.invoke = async () => ({ response: '' });
    apiModule.default.getStreamAbortController = () => ({
        abort() { ctrl.error(Object.assign(new Error('aborted'), { name: 'AbortError' })); },
        signal: {},
    });
    apiModule.default.getCurrentStreamRequestId = () => 'rid';
    apiModule.default.stop = async () => ({ ok: true });

    messageInput.value = 'turn one';
    const firstPromise = sendMessage();
    await new Promise((r) => setTimeout(r, 5));

    messageInput.value = 'queued, should be cancelled by Stop';
    await sendMessage();
    assert.equal(pane.queuedMessage, 'queued, should be cancelled by Stop');
    assert.ok(pane.element.querySelector('.queued-message-chip'));

    // Hit Stop.
    await stopAgent(agent);

    assert.equal(pane.queuedMessage, null,
        'Stop must clear the queued message (Stop = stop everything)');
    assert.equal(pane.element.querySelector('.queued-message-chip'), null,
        'Stop must remove the queued chip immediately');

    await firstPromise;
    await new Promise((r) => setTimeout(r, 10));

    assert.deepEqual(dispatched, ['turn one'],
        'no second dispatch — the queued message was cancelled by Stop');
});


test('conversation switch (wipeAgentChatPane) discards the queued message', async () => {
    const agent = 'q-wipe';
    apiModule.default.setHostAgent(agent);
    mountChatPane(agent);
    const pane = setQueueMode(agent);

    const ctrl = controlledStream();
    apiModule.default.streamInvoke = () => ctrl.iter;
    apiModule.default.invoke = async () => ({ response: '' });

    messageInput.value = 'turn';
    const firstPromise = sendMessage();
    await new Promise((r) => setTimeout(r, 5));

    messageInput.value = 'queued, belongs to old conversation';
    await sendMessage();
    assert.equal(pane.queuedMessage, 'queued, belongs to old conversation');

    // User switches conversation on this agent.
    wipeAgentChatPane(agent);
    assert.equal(pane.queuedMessage, null,
        'a within-agent context change must discard the queued message');

    ctrl.end();
    await firstPromise;
    await new Promise((r) => setTimeout(r, 10));
    // composerMode preference must SURVIVE the wipe (UI pref, not
    // conversation state).
    assert.equal(pane.composerMode, 'queue',
        'the mode toggle preference must survive a conversation switch');
});


test('re-Enter while queued replaces the queued message (single slot)', async () => {
    const agent = 'q-replace';
    apiModule.default.setHostAgent(agent);
    mountChatPane(agent);
    const pane = setQueueMode(agent);
    state.waitingAgents.add(agent);

    messageInput.value = 'first queued';
    await sendMessage();
    assert.equal(pane.queuedMessage, 'first queued');

    messageInput.value = 'second queued (replaces first)';
    await sendMessage();
    assert.equal(pane.queuedMessage, 'second queued (replaces first)',
        're-Enter must REPLACE the queued message, not append');

    // Exactly one chip (not two).
    let chipCount = 0;
    const walk = (n) => {
        for (const c of n.children) {
            if (c.classList && c.classList.contains('queued-message-chip')) chipCount++;
            walk(c);
        }
    };
    walk(pane.element);
    assert.equal(chipCount, 1, 'exactly one queued chip after replace');

    state.waitingAgents.delete(agent);
    pane.queuedMessage = null;
});


test('interrupt mode (default) still interrupts — regression guard', async () => {
    const agent = 'q-interrupt-still';
    apiModule.default.setHostAgent(agent);
    mountChatPane(agent);
    const pane = getOrCreateChatPane(agent);
    pane.composerMode = 'interrupt';   // explicit default

    let stopCalled = false;
    apiModule.default.getStreamAbortController = () => ({
        abort() { stopCalled = true; }, signal: {},
    });
    apiModule.default.getCurrentStreamRequestId = () => 'rid';
    apiModule.default.stop = async () => { stopCalled = true; return { ok: true }; };
    const ctrl = controlledStream();
    apiModule.default.streamInvoke = () => ctrl.iter;
    apiModule.default.invoke = async () => ({ response: '' });

    state.waitingAgents.add(agent);
    messageInput.value = 'interrupt please';
    const p = sendMessage();
    await new Promise((r) => setTimeout(r, 5));

    assert.equal(stopCalled, true,
        'interrupt mode must still call stopAgent (Phase 1 behavior)');
    assert.equal(pane.queuedMessage, null,
        'interrupt mode must not populate the queue');

    ctrl.end();
    await p;
    await new Promise((r) => setTimeout(r, 5));
});


test('prior-turn error still dispatches the queued message', async () => {
    const agent = 'q-error';
    apiModule.default.setHostAgent(agent);
    mountChatPane(agent);
    const pane = setQueueMode(agent);

    const ctrl = controlledStream();
    const dispatched = [];
    apiModule.default.streamInvoke = (input) => { dispatched.push(input); return ctrl.iter; };
    apiModule.default.invoke = async () => ({ response: '' });

    messageInput.value = 'turn that will error';
    const firstPromise = sendMessage();
    await new Promise((r) => setTimeout(r, 5));

    messageInput.value = 'queued after error';
    await sendMessage();
    assert.equal(pane.queuedMessage, 'queued after error');

    const ctrl2 = controlledStream();
    apiModule.default.streamInvoke = (input) => { dispatched.push(input); return ctrl2.iter; };

    // Non-abort error on the first turn.
    ctrl.error(new Error('model route exploded'));
    await firstPromise;
    await new Promise((r) => setTimeout(r, 10));

    assert.deepEqual(dispatched, ['turn that will error', 'queued after error'],
        'a prior-turn error (not user-abort) must still dispatch the queue');

    ctrl2.end();
    await new Promise((r) => setTimeout(r, 5));
});


test('mode toggle is per-pane and reflects the mounted agent', () => {
    const agentX = 'tog-X';
    const agentY = 'tog-Y';
    const px = getOrCreateChatPane(agentX);
    const py = getOrCreateChatPane(agentY);
    px.composerMode = 'queue';
    py.composerMode = 'interrupt';

    apiModule.default.setHostAgent(agentX);
    mountChatPane(agentX);
    assert.equal(composerModeToggle.dataset.mode, 'queue',
        'toggle reflects agentX (queue)');
    assert.equal(composerModeToggle.textContent, 'Queue');

    apiModule.default.setHostAgent(agentY);
    mountChatPane(agentY);
    assert.equal(composerModeToggle.dataset.mode, 'interrupt',
        'toggle reflects agentY (interrupt) after switch');
    assert.equal(composerModeToggle.textContent, 'Interrupt');

    // Clicking toggles the MOUNTED agent's mode only.
    composerModeToggle.click();
    assert.equal(py.composerMode, 'queue', 'click flipped agentY to queue');
    assert.equal(px.composerMode, 'queue', 'agentX unaffected by agentY toggle');
    assert.equal(composerModeToggle.textContent, 'Queue');
});


test('concurrent: queued dispatch fires AFTER the prior turn finally unwinds', async () => {
    // The queued sendMessage must start from a clean async context —
    // dispatched via queueMicrotask from the finally, NOT re-entered
    // synchronously mid-cleanup. We assert the prior turn's finally
    // bookkeeping (waitingAgents.delete) has completed before the
    // queued turn re-adds the agent.
    const agent = 'q-concurrent';
    apiModule.default.setHostAgent(agent);
    mountChatPane(agent);
    const pane = setQueueMode(agent);

    const ctrl = controlledStream();
    const ctrl2 = controlledStream();
    const observed = [];
    apiModule.default.streamInvoke = (input) => {
        observed.push(input);
        return input === 'c1' ? ctrl.iter : ctrl2.iter;
    };
    apiModule.default.invoke = async () => ({ response: '' });

    messageInput.value = 'c1';
    const firstPromise = sendMessage();
    await new Promise((r) => setTimeout(r, 5));

    messageInput.value = 'c2-queued';
    await sendMessage();

    ctrl.end();
    await firstPromise;
    // Immediately after firstPromise resolves the finally has run
    // (waitingAgents.delete fired). The queued dispatch is scheduled
    // via queueMicrotask — it has NOT run synchronously inside the
    // finally.
    assert.deepEqual(observed, ['c1'],
        'queued dispatch must not run synchronously inside the finally');

    await new Promise((r) => setTimeout(r, 10));
    assert.deepEqual(observed, ['c1', 'c2-queued'],
        'queued dispatch runs as its own turn after the microtask drains');

    ctrl2.end();
    await new Promise((r) => setTimeout(r, 5));
});
