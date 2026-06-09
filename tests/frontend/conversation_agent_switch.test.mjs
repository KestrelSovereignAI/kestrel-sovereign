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

function makeClassList() {
    const set = new Set();
    return {
        _set: set,
        add(c) { set.add(c); },
        remove(c) { set.delete(c); },
        contains(c) { return set.has(c); },
        toggle(c, on) {
            const shouldAdd = on === undefined ? !set.has(c) : !!on;
            if (shouldAdd) set.add(c);
            else set.delete(c);
            return shouldAdd;
        },
    };
}

function makeNode(tag = 'div') {
    const listeners = new Map();
    const node = {
        tagName: tag.toUpperCase(),
        nodeType: 1,
        children: [],
        childNodes: [],
        parentNode: null,
        classList: makeClassList(),
        dataset: {},
        style: {},
        _innerHTML: '',
        textContent: '',
        title: '',
        disabled: false,
        scrollTop: 0,
        scrollHeight: 0,
        value: '',
        addEventListener(type, fn) {
            if (!listeners.has(type)) listeners.set(type, []);
            listeners.get(type).push(fn);
        },
        dispatchEvent(event) {
            for (const fn of listeners.get(event.type) || []) fn(event);
        },
        click() {
            this.dispatchEvent({ type: 'click', stopPropagation() {}, preventDefault() {} });
        },
        querySelector() { return null; },
        querySelectorAll() { return []; },
        appendChild(child) {
            if (child.parentNode && child.parentNode !== this) {
                const i = child.parentNode.children.indexOf(child);
                if (i >= 0) child.parentNode.children.splice(i, 1);
                const j = child.parentNode.childNodes.indexOf(child);
                if (j >= 0) child.parentNode.childNodes.splice(j, 1);
            }
            child.parentNode = this;
            this.children.push(child);
            this.childNodes.push(child);
            return child;
        },
        remove() {
            if (!this.parentNode) return;
            const i = this.parentNode.children.indexOf(this);
            if (i >= 0) this.parentNode.children.splice(i, 1);
            const j = this.parentNode.childNodes.indexOf(this);
            if (j >= 0) this.parentNode.childNodes.splice(j, 1);
            this.parentNode = null;
        },
        get firstChild() { return this.children[0] || null; },
    };
    Object.defineProperty(node, 'innerHTML', {
        get() { return node._innerHTML; },
        set(value) {
            node._innerHTML = String(value);
            for (const child of node.children) child.parentNode = null;
            node.children = [];
            node.childNodes = [];
        },
    });
    return node;
}

const nodes = new Map([
    ['conversations-list', makeNode()],
    ['conversations-pane', makeNode()],
    ['chat-container', makeNode()],
]);

globalThis.document = {
    getElementById(id) { return nodes.get(id) || null; },
    createElement: (tag) => makeNode(tag),
    head: makeNode('head'),
    body: makeNode('body'),
    addEventListener() {},
    querySelector() { return null; },
    querySelectorAll(selector) {
        if (selector !== '.conversation-item') return [];
        return nodes.get('conversations-list').children
            .filter((child) => child.classList.contains('conversation-item'));
    },
};
globalThis.sessionStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.location = { href: '/', search: '' };
globalThis.fetch = async () => ({ ok: false, status: 500, json: async () => ({}) });
globalThis.kicon = () => '';
globalThis.CSS = { escape: (s) => String(s) };

const apiModule = await import('../../kestrel_sovereign/static/js/api.js');
const { loadConversations } = await import('../../kestrel_sovereign/static/js/identity.js');

function conversation(sessionId, preview) {
    return {
        session_id: sessionId,
        preview,
        started_at: '2026-06-09T15:00:00Z',
        last_message_at: '2026-06-09T15:00:00Z',
    };
}

test('agent switch drops stale conversation LIST rows before they can render under the new agent', async () => {
    const pending = new Map();
    apiModule.default.getConversations = () => {
        const agent = apiModule.default.getHostAgent();
        return new Promise((resolve) => pending.set(agent, resolve));
    };

    const list = nodes.get('conversations-list');

    apiModule.default.setHostAgent('Meridian');
    const meridianLoad = loadConversations('Meridian');

    apiModule.default.setHostAgent('Emma');
    const emmaLoad = loadConversations('Emma');

    pending.get('Meridian')({
        conversations: [
            conversation('1325', 'Meridian old row'),
            conversation('1278', 'Meridian older row'),
        ],
    });
    await meridianLoad;

    assert.equal(list.dataset.agentKey, 'Emma');
    assert.deepEqual(
        list.children.map((row) => row.dataset.sessionId),
        [],
        'Meridian session_ids must not render while Emma is selected',
    );

    pending.get('Emma')({
        conversations: [
            conversation('991', 'Emma row'),
        ],
    });
    await emmaLoad;

    assert.deepEqual(
        list.children.map((row) => row.dataset.sessionId),
        ['991'],
        'Emma sidebar must contain only Emma session_ids after her LIST resolves',
    );
});

test('conversation rows are pinned to the agent that produced them', async () => {
    apiModule.default.getConversations = async () => ({
        conversations: [conversation('991', 'Emma row')],
    });

    apiModule.default.setHostAgent('Emma');
    await loadConversations('Emma');

    const row = nodes.get('conversations-list').children[0];
    const calls = [];
    window.loadConversation = (sessionId, options = {}) => calls.push({ sessionId, options });

    row.click();
    assert.deepEqual(calls, [{ sessionId: '991', options: { expectedAgent: 'Emma' } }]);

    apiModule.default.setHostAgent('Meridian');
    row.click();
    assert.deepEqual(
        calls,
        [{ sessionId: '991', options: { expectedAgent: 'Emma' } }],
        'a stale Emma row must not dispatch under Meridian routing',
    );
});
