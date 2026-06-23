import test from 'node:test';
import assert from 'node:assert/strict';

function makeNode(tag = 'div') {
    const node = {
        tagName: tag.toUpperCase(),
        children: [],
        parentNode: null,
        style: {},
        dataset: {},
        value: '',
        _innerHTML: '',
        appendChild(child) {
            child.parentNode = this;
            this.children.push(child);
            return child;
        },
        addEventListener() {},
        remove() {},
    };
    Object.defineProperty(node, 'innerHTML', {
        get() { return node._innerHTML; },
        set(value) { node._innerHTML = String(value); },
    });
    Object.defineProperty(node, 'textContent', {
        get() { return node._textContent || ''; },
        set(value) {
            node._textContent = String(value);
            node._innerHTML = String(value)
                .replaceAll('&', '&amp;')
                .replaceAll('<', '&lt;')
                .replaceAll('>', '&gt;')
                .replaceAll('"', '&quot;')
                .replaceAll("'", '&#39;');
        },
    });
    return node;
}

const nodes = new Map([
    ['history-container', makeNode()],
]);

globalThis.window = globalThis.window || {};
globalThis.window.innerWidth = 1024;
globalThis.window.SharedMarkdown = {
    renderMarkdown: (text) => String(text || ''),
    renderMath: () => {},
};
globalThis.document = {
    getElementById(id) { return nodes.get(id) || null; },
    createElement: (tag) => makeNode(tag),
    head: makeNode('head'),
    body: makeNode('body'),
    addEventListener() {},
    querySelector() { return null; },
    querySelectorAll() { return []; },
};
globalThis.sessionStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.location = { href: '/', search: '' };
globalThis.fetch = async () => ({ ok: false, status: 500, json: async () => ({}) });
globalThis.kicon = (name) => `<span class="ki ki-${name}"></span>`;
globalThis.window.kicon = globalThis.kicon;
globalThis.prompt = () => null;

const apiModule = await import('../../kestrel_sovereign/static/js/api.js');
const { state } = await import('../../kestrel_sovereign/static/js/ui.js');
const { loadConversationHistory } = await import('../../kestrel_sovereign/static/js/history.js');

function resetHistoryContainer() {
    const container = nodes.get('history-container');
    container.innerHTML = '';
    return container;
}

test('history sidebar shows conversation name before preview and renders rename affordance', async () => {
    const container = resetHistoryContainer();
    apiModule.default.getConversations = async () => ({
        encrypted_at_rest: false,
        conversations: [
            {
                session_id: 'sess-1',
                name: 'Debugging Thread',
                preview: 'first user message',
                started_at: '2026-06-23T12:00:00Z',
                message_count: 3,
            },
            {
                session_id: 'sess-2',
                preview: 'unnamed preview',
                started_at: '2026-06-23T13:00:00Z',
                message_count: 1,
            },
        ],
    });

    await loadConversationHistory();

    assert.match(container.innerHTML, /Debugging Thread/);
    assert.match(container.innerHTML, /unnamed preview/);
    assert.match(container.innerHTML, /class="conv-rename-btn"/);
    assert.match(container.innerHTML, /Rename conversation/);
});

test('renameConversation patches backend and updates rendered item without reloading list', async () => {
    const container = resetHistoryContainer();
    const calls = [];
    state.conversations = [
        {
            session_id: 'sess-1',
            preview: 'fallback preview',
            started_at: '2026-06-23T12:00:00Z',
            message_count: 3,
        },
    ];
    state.encryptedAtRest = false;
    apiModule.default.renameConversation = async (sessionId, name) => {
        calls.push({ sessionId, name });
        return { success: true, session_id: sessionId, name: 'Renamed Thread' };
    };
    globalThis.prompt = () => 'Renamed Thread';

    await globalThis.window.renameConversation('sess-1', {
        preventDefault() {},
        stopPropagation() {},
    });

    assert.deepEqual(calls, [{ sessionId: 'sess-1', name: 'Renamed Thread' }]);
    assert.equal(state.conversations[0].name, 'Renamed Thread');
    assert.match(container.innerHTML, /Renamed Thread/);
});
