import test from 'node:test';
import assert from 'node:assert/strict';

// Per-pane generation gating replaced the old global uiGeneration
// regime. The invariant flipped:
//
//   * Agent switch (A→B→A) does NOT bump any generation. Streams
//     dispatched against A's pane keep painting into A's pane element
//     even while B is mounted. Agent equality is no longer the gate;
//     pane-attachment is.
//
//   * Within-agent context changes (clear chat, new chat, conversation
//     switch on the same agent, soft/hard delete of the active
//     conversation) bump THAT agent's pane.generation. The dispatch
//     captured a generation token at start; the gate fails the moment
//     the pane's generation moves past it.
//
// These tests exercise the per-pane gate directly via wipeAgentChatPane
// and the chatPanes Map, without spinning up the full sendMessage path.

globalThis.window = globalThis.window || {};
globalThis.window.SharedMarkdown = {
    renderMarkdown: () => '',
    renderStreamingMarkdown: () => '',
    highlightCodeBlocks: () => {},
    renderMermaidDiagrams: () => {},
    finalizeMarkdown: async () => {},
};

// Minimal jsdom-ish stubs. createElement returns a node with the
// children / appendChild semantics we need for pane attachment tests.
function makeNode(tag = 'div') {
    const node = {
        tagName: tag.toUpperCase(),
        nodeType: 1,
        children: [],
        childNodes: [],
        parentNode: null,
        classList: { _set: new Set(), add(c){this._set.add(c);}, remove(c){this._set.delete(c);}, toggle(c, on){if(on===undefined){this._set.has(c)?this._set.delete(c):this._set.add(c);}else if(on){this._set.add(c);}else{this._set.delete(c);}return this._set.has(c);}, contains(c){return this._set.has(c);} },
        dataset: {},
        style: {},
        innerHTML: '',
        textContent: '',
        scrollTop: 0,
        scrollHeight: 0,
        addEventListener() {},
        querySelector() { return null; },
        querySelectorAll() { return []; },
        appendChild(child) {
            child.parentNode = this;
            this.children.push(child);
            this.childNodes.push(child);
            return child;
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
    return node;
}

let chatContainer = makeNode('div');
chatContainer.id = 'chat-container';

globalThis.document = globalThis.document || {
    getElementById(id) { if (id === 'chat-container') return chatContainer; return null; },
    createElement(tag) { return makeNode(tag); },
    head: makeNode(), body: makeNode(),
    addEventListener() {},
    querySelector() { return null; },
    querySelectorAll() { return []; },
};
globalThis.sessionStorage = globalThis.sessionStorage || {
    getItem: () => null, setItem: () => {}, removeItem: () => {},
};
globalThis.location = globalThis.location || { href: '/', search: '' };
globalThis.fetch = globalThis.fetch || (async () => ({ ok: false, status: 500 }));
globalThis.kicon = globalThis.kicon || (() => '');
globalThis.CSS = globalThis.CSS || { escape: (s) => String(s).replace(/"/g, '\\"') };

const { mountChatPane, wipeAgentChatPane } = await import(
    '../../kestrel_sovereign/static/js/chat.js'
);
const { state, getOrCreateChatPane } = await import(
    '../../kestrel_sovereign/static/js/ui.js'
);

test('agent switch (A→B→A) does NOT bump any pane generation', () => {
    const paneA = getOrCreateChatPane('agent-a');
    const paneB = getOrCreateChatPane('agent-b');
    const genA0 = paneA.generation;
    const genB0 = paneB.generation;

    mountChatPane('agent-a');
    mountChatPane('agent-b');
    mountChatPane('agent-a');

    assert.equal(paneA.generation, genA0, "A's generation must not move on agent switches");
    assert.equal(paneB.generation, genB0, "B's generation must not move on agent switches");
});

test('within-agent context change bumps ONLY that agent\'s generation', () => {
    const paneA = getOrCreateChatPane('agent-a');
    const paneB = getOrCreateChatPane('agent-b');
    const genA0 = paneA.generation;
    const genB0 = paneB.generation;

    wipeAgentChatPane('agent-a');

    assert.equal(paneA.generation, genA0 + 1, "A's generation must bump");
    assert.equal(paneB.generation, genB0, "B's generation must NOT move when A is wiped");
});

test('per-pane gate: dispatch captured at gen N becomes stale only after a within-agent wipe', () => {
    const pane = getOrCreateChatPane('agent-a');
    const dispatchGeneration = pane.generation;

    // No within-agent change: gate holds.
    assert.equal(pane.generation === dispatchGeneration, true);

    // Agent switches do NOT invalidate the gate (the very point of
    // the per-pane regime — streams keep painting through switches).
    mountChatPane('agent-b');
    mountChatPane('agent-a');
    assert.equal(pane.generation === dispatchGeneration, true,
        'agent switches must not invalidate the dispatch generation');

    // A within-agent context change DOES invalidate the gate.
    wipeAgentChatPane('agent-a');
    assert.equal(pane.generation === dispatchGeneration, false,
        'within-agent context change must move the pane generation');
});

test('mountChatPane attaches the target pane and detaches the previous one', () => {
    const paneA = getOrCreateChatPane('mount-a');
    const paneB = getOrCreateChatPane('mount-b');

    mountChatPane('mount-a');
    assert.equal(paneA.element.parentNode, chatContainer, 'A must be mounted');

    mountChatPane('mount-b');
    assert.equal(paneA.element.parentNode, null, 'A must be detached');
    assert.equal(paneB.element.parentNode, chatContainer, 'B must be mounted');

    assert.equal(state.mountedChatAgent, 'mount-b');
});
