import test from 'node:test';
import assert from 'node:assert/strict';

globalThis.window = globalThis.window || {};
globalThis.window.SharedMarkdown = {
    renderMarkdown: (text) => String(text || ''),
    renderStreamingMarkdown: (text) => String(text || ''),
    highlightCodeBlocks: () => {},
    renderMermaidDiagrams: async () => {},
    finalizeMarkdown: async (el, text) => {
        el.textContent = String(text || '');
    },
};

function makeNode(tag = 'div') {
    return {
        tagName: tag.toUpperCase(),
        nodeType: 1,
        children: [],
        childNodes: [],
        parentNode: null,
        className: '',
        dataset: {},
        style: {},
        innerHTML: '',
        textContent: '',
        scrollTop: 0,
        scrollHeight: 0,
        classList: {
            _set: new Set(),
            add(c) { this._set.add(c); },
            remove(c) { this._set.delete(c); },
            contains(c) { return this._set.has(c); },
        },
        addEventListener() {},
        appendChild(child) {
            child.parentNode = this;
            this.children.push(child);
            this.childNodes.push(child);
            return child;
        },
        insertAdjacentHTML(_pos, html) {
            this.innerHTML += html;
            if (html.includes('message-model-footer')) {
                const footer = makeNode('div');
                footer.className = 'message-model-footer';
                footer.classList.add('message-model-footer');
                footer.textContent = html.replace(/<[^>]+>/g, '');
                this.appendChild(footer);
            }
        },
        querySelector(sel) {
            if (sel === '.message-model-footer') {
                return this.children.find(
                    (c) => c.classList?.contains('message-model-footer'),
                ) || null;
            }
            if (sel === '.message-content') {
                return this.children.find(
                    (c) => c.classList?.contains('message-content'),
                ) || null;
            }
            return null;
        },
        querySelectorAll() { return []; },
        remove() {},
        get firstChild() { return this.children[0] || null; },
    };
}

const chatContainer = makeNode('div');
chatContainer.id = 'chat-container';

globalThis.document = {
    getElementById(id) {
        return id === 'chat-container' ? chatContainer : null;
    },
    createElement(tag) {
        return makeNode(tag);
    },
    head: makeNode(),
    body: makeNode(),
    addEventListener() {},
    querySelector() { return null; },
    querySelectorAll() { return []; },
};
globalThis.sessionStorage = {
    getItem: () => null,
    setItem: () => {},
    removeItem: () => {},
};
globalThis.location = { href: '/', search: '' };
globalThis.fetch = async () => ({ ok: false, status: 500 });
globalThis.kicon = () => '';
globalThis.CSS = { escape: (s) => String(s) };

const chat = await import('../../kestrel_sovereign/static/js/chat.js');

const selectedState = {
    selectedModel: 'gpt-5-mini',
    selectedProvider: 'openai',
};

chat.setChatDeps({
    state: selectedState,
    markdown: window.SharedMarkdown,
    escapeHtml: (text) => String(text || '').replace(/[&<>"']/g, (ch) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
    }[ch])),
});

test('assistant model footer renders when row model differs from current selector', async () => {
    const pane = makeNode('div');
    await chat.addMessage(
        'agent',
        'voice reply',
        pane,
        null,
        { model: 'gpt-realtime-2', provider: 'openai' },
    );

    const footer = pane.children[0].querySelector('.message-model-footer');
    assert.ok(footer, 'assistant bubble should include model footer');
    assert.equal(footer.textContent, 'via gpt-realtime-2 · openai');
});

test('assistant model footer is silent for matching current selector', async () => {
    const pane = makeNode('div');
    await chat.addMessage(
        'agent',
        'normal reply',
        pane,
        null,
        { model: 'gpt-5-mini', provider: 'openai' },
    );

    assert.equal(pane.children[0].querySelector('.message-model-footer'), null);
});

test('assistant model footer is silent for legacy null rows', () => {
    assert.equal(chat.renderModelFooterHtml({ model: null, provider: null }), '');
});

test('assistant model footer escapes model and provider labels', () => {
    const html = chat.renderModelFooterHtml({
        model: '<img src=x onerror=alert(1)>',
        provider: 'openai',
    });

    assert.match(html, /&lt;img/);
    assert.doesNotMatch(html, /<img/);
});
