import test from 'node:test';
import assert from 'node:assert/strict';

// Race: an agent-select fires loadConversations() in parallel with the
// rest of the UI init. If the user starts typing and submits a message
// before the most-recent-conversation auto-load resolves, the auto-load
// would call wipeAgentChatPane() on the agent's pane — bumping its
// generation — and the in-flight stream would gate out mid-answer.
//
// User-facing symptom: "the agent keeps stopping."
//
// The fix is two-layered:
//   1. Synchronous cold-pane check at the auto-load fire site
//      (loadConversations in identity.js)
//   2. Post-await re-check inside window.loadConversation when called
//      with {auto: true} — catches races that happen between the sync
//      check and the actual wipe.
//
// This file exercises (2) — the defense-in-depth — by calling
// loadConversation directly and arranging the race state before the
// fetch resolves.

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
        querySelector() { return null; },
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
globalThis.document = {
    getElementById(id) { return id === 'chat-container' ? chatContainer : null; },
    createElement: (t) => makeNode(t),
    head: makeNode(), body: makeNode(),
    addEventListener() {}, querySelector() { return null; }, querySelectorAll() { return []; },
};
globalThis.sessionStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.location = { href: '/', search: '' };
globalThis.kicon = () => '';
globalThis.CSS = { escape: (s) => String(s) };

const { state, getOrCreateChatPane } = await import('../../kestrel_sovereign/static/js/ui.js');
const { mountChatPane } = await import('../../kestrel_sovereign/static/js/chat.js');
const apiModule = await import('../../kestrel_sovereign/static/js/api.js');

// Stub API.getConversation so we can hold the auto-load in-flight while
// we mutate the pane state to simulate the user starting a turn.
let releaseGetConv = null;
const heldConv = { messages: [{ role: 'user', content: 'historical' }] };
apiModule.default.getConversation = (_id) => new Promise((r) => { releaseGetConv = () => r(heldConv); });

// Import identity.js LAST — its top-level execution wires window.loadConversation.
await import('../../kestrel_sovereign/static/js/identity.js');

test('auto-load drops silently when the user has begun a turn during the fetch', async () => {
    // Build a pane and mount it like selectAgent would.
    const pane = getOrCreateChatPane('race-A');
    apiModule.default.setHostAgent('race-A');
    mountChatPane('race-A');

    const genBefore = pane.generation;

    // Fire the auto-load. It awaits getConversation — held by us.
    const autoLoadPromise = window.loadConversation('historical-sess', { auto: true });

    // Simulate the user starting a turn during the fetch:
    //   - sendMessage adds a user bubble to the pane
    //   - sendMessage adds the agent to waitingAgents
    //   - sendMessage installs a streamingMsgDiv on the pane
    pane.element.appendChild(makeNode());     // user message bubble
    state.waitingAgents.add('race-A');
    pane.streamingMsgDiv = makeNode();

    // Now release the auto-load's awaited fetch. The defense-in-depth
    // re-check must see the pane is no longer cold and bail out without
    // calling wipeAgentChatPane().
    releaseGetConv();
    await autoLoadPromise;

    assert.equal(pane.generation, genBefore,
        "auto-load must NOT bump the pane generation when the user has begun a turn");
    assert.ok(pane.streamingMsgDiv,
        'streaming bubble must survive the auto-load');
    assert.equal(state.waitingAgents.has('race-A'), true,
        "waitingAgents must still mark the agent busy");

    // Cleanup for next test.
    state.waitingAgents.delete('race-A');
    pane.streamingMsgDiv = null;
});

test('auto-load proceeds when the pane is genuinely cold', async () => {
    const pane = getOrCreateChatPane('cold-A');
    apiModule.default.setHostAgent('cold-A');
    mountChatPane('cold-A');

    const genBefore = pane.generation;

    const autoLoadPromise = window.loadConversation('cold-sess', { auto: true });
    // Don't simulate any user activity. Pane stays cold. Release the fetch.
    releaseGetConv();
    await autoLoadPromise;

    // Generation must have bumped (wipe ran) and the historical
    // message from the stub must have rendered into the pane.
    assert.equal(pane.generation, genBefore + 1,
        'cold pane should accept the auto-load (generation bumps)');
});

test('user-explicit click (no auto flag) skips the defense-in-depth check', async () => {
    // Even if the pane has activity, an explicit user click on a
    // sidebar conversation must still load — the user's intent
    // overrides the in-flight stream. This guards against the fix
    // accidentally breaking the user-explicit path.
    const pane = getOrCreateChatPane('explicit-A');
    apiModule.default.setHostAgent('explicit-A');
    mountChatPane('explicit-A');
    pane.element.appendChild(makeNode());     // simulate prior message
    pane.streamingMsgDiv = makeNode();

    const genBefore = pane.generation;
    const explicitPromise = window.loadConversation('explicit-sess');  // no {auto: true}
    releaseGetConv();
    await explicitPromise;

    assert.equal(pane.generation, genBefore + 1,
        'explicit click must wipe (bump generation) regardless of pane busyness');
});
