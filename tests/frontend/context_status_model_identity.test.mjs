import test from 'node:test';
import assert from 'node:assert/strict';

globalThis.window = globalThis.window || {};
globalThis.window.SharedMarkdown = {
    renderMarkdown: () => '',
    renderStreamingMarkdown: () => '',
    highlightCodeBlocks: () => {},
    renderMermaidDiagrams: () => {},
    finalizeMarkdown: async () => {},
};

function makeNode(tag = 'div') {
    return {
        tagName: tag.toUpperCase(),
        nodeType: 1,
        children: [],
        childNodes: [],
        parentNode: null,
        classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
        dataset: {},
        style: {},
        innerHTML: '',
        textContent: '',
        value: '',
        addEventListener() {},
        querySelector() { return null; },
        querySelectorAll() { return []; },
        appendChild(c) {
            c.parentNode = this;
            this.children.push(c);
            this.childNodes.push(c);
            return c;
        },
        remove() {},
    };
}

const nodes = new Map([
    ['context-status', makeNode()],
    ['model-selector', makeNode('select')],
]);
nodes.get('model-selector').value = 'gpt-5-text-dropdown';

globalThis.document = {
    getElementById(id) { return nodes.get(id) || null; },
    createElement: (t) => makeNode(t),
    head: makeNode(),
    body: makeNode(),
    addEventListener() {},
    querySelector() { return null; },
    querySelectorAll() { return []; },
};
globalThis.sessionStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.location = { href: '/', search: '' };
globalThis.fetch = async () => ({ ok: false, status: 500 });
globalThis.kicon = () => '!';
globalThis.CSS = { escape: (s) => String(s) };

const { updateContextStatus, setChatDeps } = await import('../../kestrel_sovereign/static/js/chat.js');
const apiModule = await import('../../kestrel_sovereign/static/js/api.js');

function baseStatus(overrides = {}) {
    return {
        model: 'gpt-4o-realtime-preview',
        provider: 'openai:realtime',
        context_model: 'openai:realtime/gpt-4o-realtime-preview',
        model_source: 'assistant_turn',
        message_count: 3,
        utilization_percent: 12.4,
        status: 'healthy',
        warnings: [],
        route_cap: null,
        codex_thread: null,
        ...overrides,
    };
}

test('context status pill renders latest assistant model/provider, not dropdown value', async () => {
    const origGetContextStatus = apiModule.default.getContextStatus;
    const origHasCapability = apiModule.default.hasCapability;
    apiModule.default.hasCapability = () => true;
    apiModule.default.getContextStatus = async () => baseStatus();
    setChatDeps({ state: { currentSessionId: 'sess-voice' } });

    try {
        await updateContextStatus();

        const html = nodes.get('context-status').innerHTML;
        assert.match(html, /openai:realtime\/gpt-4o-realtime-preview/);
        assert.doesNotMatch(html, /gpt-5-text-dropdown/);
    } finally {
        apiModule.default.getContextStatus = origGetContextStatus;
        apiModule.default.hasCapability = origHasCapability;
        setChatDeps({ state: null });
    }
});

test('context status pill renders legacy assistant rows with a safe placeholder', async () => {
    const origGetContextStatus = apiModule.default.getContextStatus;
    const origHasCapability = apiModule.default.hasCapability;
    apiModule.default.hasCapability = () => true;
    apiModule.default.getContextStatus = async () => baseStatus({
        model: null,
        provider: null,
        context_model: 'legacy/unknown',
        model_source: 'legacy_assistant_turn',
    });
    setChatDeps({ state: { currentSessionId: 'sess-legacy' } });

    try {
        await updateContextStatus();

        assert.match(nodes.get('context-status').innerHTML, /Legacy turn .* model unknown/);
    } finally {
        apiModule.default.getContextStatus = origGetContextStatus;
        apiModule.default.hasCapability = origHasCapability;
        setChatDeps({ state: null });
    }
});
