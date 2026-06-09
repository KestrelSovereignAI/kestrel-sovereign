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
    const node = {
        tagName: tag.toUpperCase(),
        nodeType: 1,
        id: '',
        children: [],
        childNodes: [],
        parentNode: null,
        classList: {
            _set: new Set(),
            add(c) { this._set.add(c); },
            remove(c) { this._set.delete(c); },
            toggle(c, on) {
                if (on === undefined) {
                    this._set.has(c) ? this._set.delete(c) : this._set.add(c);
                } else if (on) {
                    this._set.add(c);
                } else {
                    this._set.delete(c);
                }
                return this._set.has(c);
            },
            contains(c) { return this._set.has(c); },
        },
        dataset: {},
        style: {},
        innerHTML: '',
        textContent: '',
        value: '',
        scrollTop: 0,
        scrollHeight: 0,
        addEventListener() {},
        focus() {},
        querySelector(selector) {
            if (!selector.startsWith('#')) return null;
            const id = selector.slice(1);
            const stack = [...this.children];
            while (stack.length) {
                const child = stack.shift();
                if (child.id === id) return child;
                stack.push(...child.children);
            }
            return null;
        },
        querySelectorAll() { return []; },
        appendChild(child) {
            if (child.parentNode) {
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
    return node;
}

const documentRoot = makeNode();
globalThis.document = {
    getElementById(id) { return documentRoot.querySelector('#' + id); },
    createElement(tag) { return makeNode(tag); },
    head: makeNode('head'),
    body: makeNode('body'),
    addEventListener() {},
    querySelector(selector) { return documentRoot.querySelector(selector); },
    querySelectorAll() { return []; },
};
globalThis.sessionStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.location = { href: '/', search: '' };
globalThis.fetch = async () => ({ ok: false, status: 500 });
globalThis.kicon = () => '';
globalThis.CSS = { escape: (s) => String(s) };

const chatModule = await import('../../kestrel_sovereign/static/js/chat.js');
const { state } = await import('../../kestrel_sovereign/static/js/ui.js');

test('mount initializes chat in provided container and returns public API', () => {
    const container = makeNode('section');
    const chatContainer = makeNode();
    chatContainer.id = 'chat-container';
    const messageInput = makeNode('textarea');
    messageInput.id = 'message-input';
    const sendButton = makeNode('button');
    sendButton.id = 'send-button';
    const modelSelector = makeNode('select');
    modelSelector.id = 'model-selector';
    const thinkingIndicator = makeNode();
    thinkingIndicator.id = 'thinking-indicator';
    const composerModeToggle = makeNode('button');
    composerModeToggle.id = 'composer-mode-toggle';

    container.appendChild(chatContainer);
    container.appendChild(messageInput);
    container.appendChild(sendButton);
    container.appendChild(modelSelector);
    container.appendChild(thinkingIndicator);
    container.appendChild(composerModeToggle);

    let capabilityChecked = false;
    const api = {
        hasCapability(capability) {
            capabilityChecked = capability === 'chat';
            return true;
        },
        getHostAgent() { return 'factory-agent'; },
    };

    const component = chatModule.mount(container, { deps: { api } });

    assert.equal(typeof component.initChat, 'function');
    assert.equal(typeof component.sendMessage, 'function');
    assert.equal(capabilityChecked, true);
    assert.equal(state.mountedChatAgent, 'factory-agent');
    assert.equal(chatContainer.children.length, 1);
    assert.equal(state.chatPanes.get('factory-agent').element.parentNode, chatContainer);
});

