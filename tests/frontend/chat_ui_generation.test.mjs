import test from 'node:test';
import assert from 'node:assert/strict';

// chat.js touches `window.SharedMarkdown` at module top, and pulls in
// api.js (which expects browser globals). Stub just enough of that
// surface so we can import the module and exercise the UI-generation
// seam in isolation.
globalThis.window = globalThis.window || {};
globalThis.window.SharedMarkdown = {
    renderMarkdown: () => '',
    renderStreamingMarkdown: () => '',
    highlightCodeBlocks: () => {},
    renderMermaidDiagrams: () => {},
    finalizeMarkdown: async () => {},
};
function stubElement() {
    return {
        classList: { add() {}, remove() {}, toggle() {} },
        style: {},
        addEventListener() {},
        appendChild() {},
        querySelector: () => null,
        querySelectorAll: () => [],
        innerHTML: '',
        textContent: '',
    };
}
globalThis.document = globalThis.document || {
    getElementById: () => null,
    createElement: () => stubElement(),
    head: stubElement(),
    body: stubElement(),
    addEventListener: () => {},
    querySelector: () => null,
    querySelectorAll: () => [],
};
globalThis.sessionStorage = globalThis.sessionStorage || {
    getItem: () => null, setItem: () => {}, removeItem: () => {},
};
globalThis.location = globalThis.location || { href: '/', search: '' };
globalThis.fetch = globalThis.fetch || (async () => ({ ok: false, status: 500 }));
// `kicon` is a global helper loaded via <script> in production; stub it.
globalThis.kicon = globalThis.kicon || (() => '');

const { bumpUiGeneration, _getUiGeneration, wipeChatPane } = await import(
    '../../kestrel_sovereign/static/js/chat.js'
);

test('UI generation counter starts at 0 and increments on each bump (PR #874)', () => {
    // The counter is module-scoped so we can't reset it between tests —
    // capture the baseline and assert deltas.
    const baseline = _getUiGeneration();

    bumpUiGeneration();
    assert.equal(_getUiGeneration(), baseline + 1);

    bumpUiGeneration();
    bumpUiGeneration();
    assert.equal(_getUiGeneration(), baseline + 3);
});

test('wipeChatPane() bumps the UI generation BEFORE the DOM mutation (PR #874 review-2)', () => {
    // Every chat-pane wipe path (selectAgent, loadConversation,
    // startNewConversation, clearChat, delete/purgeConversation) must go
    // through wipeChatPane() so the generation counter moves with the
    // wipe. Bare innerHTML='' wipes leak: a stream dispatched against the
    // pre-wipe pane keeps thinking it's current and writes into the
    // freshly-rebuilt one.
    const fakeContainer = { innerHTML: '<old>previous</old>' };
    let lookups = 0;
    globalThis.document.getElementById = (id) => {
        if (id === 'chat-container') {
            lookups += 1;
            return fakeContainer;
        }
        return null;
    };

    const before = _getUiGeneration();
    wipeChatPane('<div>fresh</div>');

    assert.equal(_getUiGeneration(), before + 1, 'generation must bump');
    assert.equal(fakeContainer.innerHTML, '<div>fresh</div>', 'pane must be replaced');
    assert.ok(lookups >= 1, 'must look up #chat-container fresh, not rely on a stale ref');
});

test('wipeChatPane() called with no argument clears to empty string (PR #874 review-2)', () => {
    const fakeContainer = { innerHTML: '<old>previous</old>' };
    globalThis.document.getElementById = (id) =>
        id === 'chat-container' ? fakeContainer : null;

    wipeChatPane();
    assert.equal(fakeContainer.innerHTML, '');
});

test('UI generation gate: dispatch captured at gen N becomes stale after a bump (PR #874)', () => {
    // Models the A→B→A timing case the reviewer flagged. agent-equality
    // alone says "current!" after switching back to A — but the chat
    // pane was wiped twice in between, orphaning the in-flight stream's
    // msgDiv. The generation token catches that: the captured
    // dispatchGeneration freezes; uiGeneration moves on with each
    // selectAgent() call.
    const dispatchGeneration = _getUiGeneration();

    // Equivalent of "current pane unchanged" — generation matches.
    assert.equal(_getUiGeneration() === dispatchGeneration, true);

    // selectAgent('B') would call bumpUiGeneration() before wiping the pane.
    bumpUiGeneration();
    assert.equal(_getUiGeneration() === dispatchGeneration, false);

    // selectAgent('A') again — agent equality is restored from the
    // dispatcher's perspective, but the generation has moved on twice.
    bumpUiGeneration();
    assert.equal(_getUiGeneration() === dispatchGeneration, false);
    assert.equal(_getUiGeneration(), dispatchGeneration + 2);
});
