import test from 'node:test';
import assert from 'node:assert/strict';

// Issue #1573: a mid-stream interrupt must be a hard TURN BOUNDARY in the
// live stream. The Sovereign typed a steer into the chat while the agent
// was streaming; the agent's response to it welded into the SAME bubble as
// the pre-interrupt prose ("…org chart.You're right.") — no new bubble, no
// newline.
//
// Root cause (client): the pane's streaming paint target
// (`pane.streamingMsgDiv`) and `pane.streamBaseline` were shared across
// turns, gated only by `pane.generation` — which bumps on a *conversation*
// change, never on a new turn. When the user interrupts, the prior turn's
// loop can still be alive (the #1255 comment notes the case where no abort
// controller was registered, so the prior stream runs to completion). After
// the new turn opens bubble B and points `pane.streamingMsgDiv` at it, a
// trailing paint from the PRIOR turn lands in B — welding the two turns.
//
// The fix gives each dispatched turn a monotonic `pane.activeTurnId`; the
// streaming loop only paints / recreates / tears down the pane bubble while
// it still owns the turn. This test pins that a prior turn's late paint can
// never reach the new turn's bubble.

globalThis.window = globalThis.window || {};
globalThis.window.SharedMarkdown = {
    renderMarkdown: (s) => String(s),
    renderStreamingMarkdown: (s) => String(s),
    highlightCodeBlocks: () => {},
    renderMermaidDiagrams: () => {},
    finalizeMarkdown: async (el, content) => { el.innerHTML = String(content); },
};

function makeNode(tag = 'div') {
    const node = {
        tagName: tag.toUpperCase(), nodeType: 1, children: [], childNodes: [], parentNode: null,
        classList: {
            _set: new Set(),
            add(c) { this._set.add(c); },
            remove(c) { this._set.delete(c); },
            toggle(c, on) {
                if (on === undefined) { this._set.has(c) ? this._set.delete(c) : this._set.add(c); }
                else if (on) { this._set.add(c); } else { this._set.delete(c); }
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

const tick = () => new Promise((r) => setTimeout(r, 5));

// Collect the agent-message bubbles in a pane, with their rendered content.
function agentBubbles(paneEl) {
    return paneEl.children
        .filter((c) => c.classList && c.classList.contains('agent-message'))
        .map((bubble) => {
            const content = bubble.querySelector('.message-content');
            return {
                bubble,
                text: content ? String(content._innerHTML || content.textContent || '') : '',
                streaming: !!(content && content.classList.contains('streaming')),
            };
        });
}

test('a prior turn\'s late paint cannot weld into the interrupting turn\'s bubble (#1573)', async () => {
    getOrCreateChatPane('agent-weld');
    apiModule.default.setHostAgent('agent-weld');
    mountChatPane('agent-weld');
    const pane = getOrCreateChatPane('agent-weld');

    // --- Prior turn: an agent answer that is mid-stream. Crucially, NO
    // abort controller is registered (the #1255 race note: the prior
    // stream then runs on instead of being aborted), so its loop stays
    // alive and will try one more paint AFTER the interrupt. ---
    const ctrlPrior = controlledStream();
    apiModule.default.streamInvoke = () => ctrlPrior.iter;
    apiModule.default.invoke = async () => ({ response: '' });
    apiModule.default.getStreamAbortController = () => null;  // no controller → no abort
    apiModule.default.getCurrentStreamRequestId = () => 'prior-req';
    apiModule.default.stop = async () => ({ ok: true });

    messageInput.value = 'do the soul work';
    const priorPromise = sendMessage();
    await tick();

    // Prior turn paints its pre-interrupt prose into bubble A.
    ctrlPrior.push('soul added to existing, not rewrite to match the org chart.');
    await tick();

    let bubbles = agentBubbles(pane.element);
    assert.equal(bubbles.length, 1, 'prior turn should have one agent bubble');
    const priorBubble = bubbles[0].bubble;
    assert.match(bubbles[0].text, /org chart/, 'prior bubble shows the pre-interrupt prose');

    // --- Interrupt: the user types a steer mid-stream. composerMode
    // defaults to 'interrupt' → sendMessage routes through stopAgent (a
    // no-op abort here, since no controller) then dispatches the new turn
    // into a FRESH bubble B. ---
    const ctrlNew = controlledStream();
    apiModule.default.streamInvoke = () => ctrlNew.iter;

    messageInput.value = "NO. Don't pretend, surface problems.";
    const newPromise = sendMessage();
    await tick();

    // New turn paints its response into bubble B.
    ctrlNew.push("You're right. I blurred the line.");
    await tick();

    // --- THE WELD ATTEMPT: the prior turn (still alive) emits one more
    // chunk. Pre-fix it painted into `pane.streamingMsgDiv` — now bubble
    // B — welding the prior turn's text into the new turn's bubble. ---
    ctrlPrior.push(' …completed=33 tasks.');
    await tick();

    bubbles = agentBubbles(pane.element);

    // The interrupting turn's bubble (B) must contain ONLY its own
    // response — never the prior turn's late chunk.
    const newBubble = bubbles.find((b) => /You're right/.test(b.text));
    assert.ok(newBubble, 'the interrupting turn must have its own bubble');
    assert.doesNotMatch(
        newBubble.text, /completed=33/,
        'WELD: the prior turn\'s late chunk must NOT land in the new turn\'s bubble',
    );
    assert.doesNotMatch(
        newBubble.text, /org chart/,
        'the new turn\'s bubble must not contain the prior turn\'s pre-interrupt prose',
    );

    // And there must be (at least) two distinct agent bubbles — a hard
    // turn boundary, not one merged bubble.
    assert.ok(bubbles.length >= 2,
        'interrupt must produce a distinct new bubble, not a merged one');

    // The prior bubble was sealed (its live `.streaming` affordance
    // stripped) when the new turn took over.
    const priorAfter = bubbles.find((b) => b.bubble === priorBubble);
    assert.ok(priorAfter, 'prior bubble should still exist as its own turn');
    assert.equal(priorAfter.streaming, false,
        'prior bubble should be sealed (no live streaming affordance) after interrupt');

    // The interrupting turn is still streaming, so the agent must remain
    // marked busy — a prior turn ending must NOT un-mark the active turn
    // (codex P1: shared waitingAgents teardown is now turn-scoped).
    assert.equal(state.waitingAgents.has('agent-weld'), true,
        'the active interrupting turn must stay busy while it streams');

    // Now let the PRIOR (orphaned) stream end naturally while the new
    // turn is still streaming. Its `finally` runs but must not clear the
    // agent's busy flag — that belongs to the active turn.
    ctrlPrior.end();
    await priorPromise.catch(() => {});
    assert.equal(state.waitingAgents.has('agent-weld'), true,
        'a prior orphaned turn ending must NOT un-mark the active turn as busy');

    // Finish the active turn — now busy clears.
    ctrlNew.end();
    await newPromise.catch(() => {});
    assert.equal(state.waitingAgents.has('agent-weld'), false,
        'once the owning turn settles, the agent is no longer busy');
    state.waitingAgents.delete('agent-weld');
});
