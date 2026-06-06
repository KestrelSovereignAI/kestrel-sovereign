import test from 'node:test';
import assert from 'node:assert/strict';

// #1560: ``handleRestartStatus`` must dedupe by the stable
// ``dedupe_signature = "{request_id}:{state}"`` field from the
// payload (introduced by #1562), updating the existing bubble in
// place when the same (request, state) is re-emitted with only a
// volatile ``deferral_reason`` age substring change.
//
// And: a status that lands while an assistant stream is in flight
// must NOT sit underneath a still-growing assistant bubble. The
// active streaming bubble must be finalized at its current content
// and a fresh bubble must open below the status for any later
// stream chunks — chronological DOM order.

globalThis.window = globalThis.window || {};
globalThis.window.SharedMarkdown = {
    renderMarkdown: () => '',
    renderStreamingMarkdown: (s) => '',
    highlightCodeBlocks: () => {},
    renderMermaidDiagrams: () => {},
    finalizeMarkdown: async () => {},
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
            const m = sel.match(/^\.message-content$/);
            if (m) {
                for (const c of this.children) {
                    if (c.classList && c.classList.contains('message-content')) return c;
                }
                return null;
            }
            const attrMatch = sel.match(/^\.restart-status-message\[data-dedupe-signature="(.+)"\]$/);
            if (attrMatch) {
                const wanted = attrMatch[1].replace(/\\"/g, '"').replace(/\\\\/g, '\\');
                for (const c of this.children) {
                    if (c.classList && c.classList.contains('restart-status-message')
                        && c.dataset && c.dataset.dedupeSignature === wanted) {
                        return c;
                    }
                }
                return null;
            }
            return null;
        },
        querySelectorAll(sel) {
            const m = sel.match(/^\.restart-status-message$/);
            if (m) {
                return this.children.filter(
                    (c) => c.classList && c.classList.contains('restart-status-message'),
                );
            }
            return [];
        },
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
    // Real DOM: setting textContent also serializes to innerHTML.
    // ``escapeHtml`` in ui.js relies on this round-trip to encode
    // strings safely, so the mock needs to mirror it or every
    // ``escapeHtml(text)`` call returns ''.
    let _textContent = '';
    Object.defineProperty(node, 'textContent', {
        get() { return _textContent; },
        set(v) {
            _textContent = String(v);
            node._innerHTML = _textContent
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;');
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

const pane = getOrCreateChatPane('test-agent');
state.mountedChatAgent = 'test-agent';
chatContainer.appendChild(pane.element);


function basePayload(overrides = {}) {
    return {
        request_id: '7f9ee2dab18b4f079ce2e03ba7122b9d',
        requested_by_agent: 'did:test:emma',
        operation: 'restart_only',
        urgency: 'normal',
        policy: 'idle_agents_only',
        status: 'pending',
        reason: 'config landed',
        target_ref: '',
        update_profile: '',
        deferral_reason: '',
        status_reason: '',
        completed_at: null,
        dedupe_signature: '7f9ee2dab18b4f079ce2e03ba7122b9d:pending',
        ...overrides,
    };
}


function resetPane() {
    pane.element.children = [];
    pane.element.childNodes = [];
    pane.streamingMsgDiv = null;
    pane.streamBaseline = 0;
}


test('same dedupe_signature updates one bubble in place (no duplicates from volatile age text)', () => {
    resetPane();
    chatModule.handleRestartStatus(basePayload({
        deferral_reason: 'agent busy (1 active request id(s); oldest 43s of 900s stale window)',
    }));
    assert.equal(
        pane.element.querySelectorAll('.restart-status-message').length, 1,
        'first emit should append one bubble',
    );
    const first = pane.element.querySelectorAll('.restart-status-message')[0];
    assert.equal(first.dataset.dedupeSignature, '7f9ee2dab18b4f079ce2e03ba7122b9d:pending');
    assert.match(first._innerHTML, /oldest 43s of 900s/);

    chatModule.handleRestartStatus(basePayload({
        deferral_reason: 'agent busy (1 active request id(s); oldest 87s of 900s stale window)',
    }));
    assert.equal(
        pane.element.querySelectorAll('.restart-status-message').length, 1,
        'second same-signature poll must update in place, not duplicate',
    );
    assert.match(
        pane.element.querySelectorAll('.restart-status-message')[0]._innerHTML,
        /oldest 87s of 900s/,
        'updated bubble must show the latest age string',
    );

    chatModule.handleRestartStatus(basePayload({
        status: 'deferred',
        dedupe_signature: '7f9ee2dab18b4f079ce2e03ba7122b9d:deferred',
        deferral_reason: 'agent busy (1 active request id(s); oldest 120s of 900s stale window)',
    }));
    assert.equal(
        pane.element.querySelectorAll('.restart-status-message').length, 2,
        'distinct state must append a new bubble (lifecycle remains visible)',
    );
});


test('stream-boundary: status mid-stream finalizes current bubble and arms a fresh one below', () => {
    resetPane();
    const streamingBubble = makeNode();
    streamingBubble.className = 'message agent-message';
    const contentDiv = makeNode();
    contentDiv.className = 'message-content streaming';
    contentDiv.textContent = 'Hello world';
    streamingBubble.appendChild(contentDiv);
    pane.element.appendChild(streamingBubble);
    pane.streamingMsgDiv = streamingBubble;
    pane.streamBaseline = 0;
    // The streaming loop publishes the raw cumulative fullContent
    // length here. Codex round 1 P1: using rendered textContent
    // length would diverge for markdown/code/thinking bubbles —
    // for this minimal test the raw and rendered lengths match.
    pane.streamRawContentLength = 'Hello world'.length;

    chatModule.handleRestartStatus(basePayload());

    assert.ok(
        !contentDiv.classList.contains('streaming'),
        'pre-existing streaming bubble must be finalized (streaming class cleared)',
    );
    assert.equal(
        pane.streamBaseline, 'Hello world'.length,
        'streamBaseline must advance to the raw cumulative content length',
    );
    assert.equal(
        pane.streamingMsgDiv, null,
        'streamingMsgDiv must be cleared so the next chunk starts fresh',
    );
    const statusBubbles = pane.element.querySelectorAll('.restart-status-message');
    assert.equal(statusBubbles.length, 1);
    assert.equal(pane.element.children[0], streamingBubble);
    assert.equal(pane.element.children[1], statusBubbles[0]);
});


test('stream-boundary at the no-content preamble removes the empty bubble (codex P2 r1)', () => {
    resetPane();
    // A bubble was preallocated by addMessageStreaming but no chunks
    // have arrived yet: streamRawContentLength is 0.
    const emptyBubble = makeNode();
    emptyBubble.className = 'message agent-message';
    const contentDiv = makeNode();
    contentDiv.className = 'message-content streaming';
    contentDiv.textContent = '';
    emptyBubble.appendChild(contentDiv);
    pane.element.appendChild(emptyBubble);
    pane.streamingMsgDiv = emptyBubble;
    pane.streamBaseline = 0;
    pane.streamRawContentLength = 0;

    chatModule.handleRestartStatus(basePayload());

    // Empty bubble removed entirely (no blank assistant above status).
    assert.equal(emptyBubble.parentNode, null);
    assert.equal(
        pane.streamBaseline, 0,
        'baseline stays 0 — the next chunk renders from the start',
    );
    // Status bubble is now the only child.
    const statusBubbles = pane.element.querySelectorAll('.restart-status-message');
    assert.equal(statusBubbles.length, 1);
    assert.equal(pane.element.children.length, 1);
    assert.equal(pane.element.children[0], statusBubbles[0]);
});


test('legacy payload without dedupe_signature falls back to {request_id}:{state}', () => {
    resetPane();
    const legacy = basePayload();
    delete legacy.dedupe_signature;
    chatModule.handleRestartStatus(legacy);
    const bubble = pane.element.querySelectorAll('.restart-status-message')[0];
    assert.equal(
        bubble.dataset.dedupeSignature,
        '7f9ee2dab18b4f079ce2e03ba7122b9d:pending',
    );
});


test('no pane mounted: handleRestartStatus is a no-op (does not throw)', () => {
    const restore = state.mountedChatAgent;
    state.mountedChatAgent = undefined;
    try {
        chatModule.handleRestartStatus(basePayload());
    } finally {
        state.mountedChatAgent = restore;
    }
});
