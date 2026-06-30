import test from 'node:test';
import assert from 'node:assert/strict';

// #2068: when the LLM calls the `set_model` tool, the MODEL_CHANGED marker
// lands in the tool-result card, not the streamed assistant text, so
// `checkForModelChange` never fires and the selector goes stale. chat.js must
// detect a `set_model` tool event in the completed turn and re-sync the
// selector from /api/model/current. These tests pin that DETECTION wiring —
// the actual bug was the missing detection, not the sync mechanism (which is
// covered separately in model_selector.test.mjs::syncWithServer).

// chat.js touches a handful of globals at import time; stub the minimum.
globalThis.window = globalThis.window || {};
globalThis.window.SharedMarkdown = {
    renderMarkdown: () => '',
    renderStreamingMarkdown: (s) => s,
    highlightCodeBlocks: () => {},
    renderMermaidDiagrams: () => {},
    finalizeMarkdown: async () => {},
};
const stubNode = () => ({
    classList: { add() {}, remove() {} },
    style: {},
    dataset: {},
    appendChild() {},
    setAttribute() {},
    addEventListener() {},
});
globalThis.document = {
    getElementById: () => null,
    createElement: () => stubNode(),
    head: stubNode(),
    body: stubNode(),
    addEventListener() {},
    querySelector() { return null; },
    querySelectorAll() { return []; },
};
globalThis.sessionStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.location = { href: '/', search: '' };
globalThis.fetch = async () => ({ ok: false, status: 500 });
globalThis.CSS = { escape: (s) => String(s) };
globalThis.kicon = () => '';

const { syncSelectorIfModelToolUsed, setChatDeps } = await import(
    '../../kestrel_sovereign/static/js/chat.js'
);

function fakeSelector() {
    return {
        synced: 0,
        selectedModel: 'gpt-5.4-mini',
        selectedProvider: 'openai',
        selectedRoute: 'api',
        async syncWithServer() { this.synced += 1; },
    };
}

test('agent set_model tool event re-syncs the selector (#2068)', async () => {
    const state = {};
    setChatDeps({ state });
    const selector = fakeSelector();
    const pane = { toolEvents: [
        { name: 'recall', phase: 'done' },
        { name: 'set_model', phase: 'done' },
    ] };

    await syncSelectorIfModelToolUsed(pane, selector);

    assert.equal(selector.synced, 1, 'syncWithServer must be called when set_model ran');
    assert.equal(state.selectedModel, 'gpt-5.4-mini');
    assert.equal(state.selectedProvider, 'openai');
    assert.equal(state.selectedVendor, 'openai');
    assert.equal(state.selectedRoute, 'api');
});

test('a turn WITHOUT set_model does not re-sync (#2068)', async () => {
    const state = { selectedModel: 'untouched' };
    setChatDeps({ state });
    const selector = fakeSelector();
    const pane = { toolEvents: [
        { name: 'recall', phase: 'done' },
        { name: 'web_search', phase: 'done' },
    ] };

    await syncSelectorIfModelToolUsed(pane, selector);

    assert.equal(selector.synced, 0, 'no set_model event → no re-sync');
    assert.equal(state.selectedModel, 'untouched', 'state must be left alone');
});

test('tool events may carry the name under `tool` instead of `name` (#2068)', async () => {
    const state = {};
    setChatDeps({ state });
    const selector = fakeSelector();
    // normalizeToolEvents reads `e.name || e.tool`; the detector accepts both.
    const pane = { toolEvents: [{ tool: 'set_model', phase: 'done' }] };

    await syncSelectorIfModelToolUsed(pane, selector);

    assert.equal(selector.synced, 1);
});
