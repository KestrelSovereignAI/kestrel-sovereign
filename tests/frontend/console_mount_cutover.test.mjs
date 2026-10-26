// Real-DOM (jsdom) verification of the #1644 console cutover: app.js now mounts
// the chat component into #panel-chat via mount() instead of initChat(). This
// exercises REAL querySelector scoping against the actual index.html chat-panel
// structure, so a container-scoping regression (e.g. an el() lookup for an id
// outside #panel-chat) fails here rather than silently in the browser.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { JSDOM } from 'jsdom';
import createDOMPurify from 'dompurify';

// Load the REAL console markup so the cutover is verified against the actual
// #panel-chat structure (ids, nesting) the browser uses — not a simplified
// fixture. jsdom does not run the page's scripts by default, so this only
// parses the DOM; we drive the component ourselves below.
const here = dirname(fileURLToPath(import.meta.url));
const indexHtml = readFileSync(
    resolve(here, '../../kestrel_sovereign/static/index.html'),
    'utf8',
);
const dom = new JSDOM(indexHtml, { url: 'http://localhost/' });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.Node = dom.window.Node;
globalThis.getComputedStyle = dom.window.getComputedStyle.bind(dom.window);
globalThis.CSS = dom.window.CSS || { escape: (s) => String(s) };
globalThis.location = dom.window.location;
globalThis.sessionStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.fetch = async () => ({ ok: false, status: 500, json: async () => ({}) });
globalThis.kicon = () => '';
globalThis.EventSource = class { close() {} addEventListener() {} };
window.SharedMarkdown = {
    renderMarkdown: () => '',
    renderStreamingMarkdown: () => '',
    highlightCodeBlocks: () => {},
    renderMermaidDiagrams: () => {},
    finalizeMarkdown: async () => {},
    sanitizeHtml: createDOMPurify(dom.window).sanitize,
};

const chat = await import('../../kestrel_sovereign/static/js/chat.js');
const example = await import(
    '../../kestrel_sovereign/static/examples/embed-chat-example.mjs'
);

test('console mounts into #panel-chat and the embedding hooks resolve in real DOM', () => {
    const panel = document.getElementById('panel-chat');
    const api = chat.mount(panel, {
        deps: { api: { hasCapability: () => true, getHostAgent: () => 'jsdom-agent' } },
    });
    assert.equal(typeof api.appendMessagePart, 'function');

    // The chat-container the component scrolls/mounts panes into resolves
    // within #panel-chat (at whatever depth index.html nests it) — proven by
    // real scoped querySelector against the production markup.
    assert.ok(panel.querySelector('#chat-container'));
    assert.equal(document.querySelectorAll('#chat-container').length >= 1, true);

    // A header action renders a real <button> into .chat-header inside the panel.
    chat.registerHeaderAction({
        id: 'jsdom-img', title: 'Insert image', icon: '🖼️', label: 'Image', onClick() {},
    });
    const slot = panel.querySelector('#chat-header-actions');
    assert.ok(slot, 'header-actions slot created inside the mounted panel');
    const btn = slot.querySelector('button.chat-header-action');
    assert.ok(btn, 'header action button rendered');
    assert.ok(btn.textContent.includes('Image'));

    // The example image part renderer yields a real <img> via appendMessagePart.
    chat.registerPartRenderer('image', example.imagePartRenderer);
    const div = api.appendMessagePart('image', { src: '/p.png', alt: 'p' });
    const img = div.querySelector('img');
    assert.ok(img, 'image part rendered as an <img> node');
    assert.equal(img.getAttribute('src'), '/p.png');
    assert.equal(img.getAttribute('alt'), 'p');
});


const { registerRenderer, unregisterRenderer } = await import(
    '../../kestrel_sovereign/static/js/ui-ext/renderers.js'
);
const { state } = await import('../../kestrel_sovereign/static/js/ui.js');

for (const placement of ['mounted', 'background', 'nested-background']) {
    test(`feature part callbacks keep the owning agent in a ${placement} pane`, () => {
        const ownerAgent = `part-owner-${placement}`;
        const type = `test/owner-${placement}`;
        chat.mountChatPane(ownerAgent);
        const ownerPane = state.chatPanes.get(ownerAgent).element;
        if (placement !== 'mounted') chat.mountChatPane('other-agent');
        assert.equal(ownerPane.isConnected, placement === 'mounted');
        let target = ownerPane;
        if (placement === 'nested-background') {
            target = document.createElement('section');
            ownerPane.appendChild(target);
        }
        const callbacks = [];
        let clickedOwner;
        const verifyContext = (stage, ctx) => {
            callbacks.push(stage);
            assert.equal(ctx.type, type);
            assert.equal(ctx.ownerAgent, ownerAgent);
            assert.equal(ctx.paneElement, target);
        };
        registerRenderer({
            id: type,
            match: { contentType: type },
            gate(ctx) {
                verifyContext('gate', ctx);
                return true;
            },
            render(payload, ctx) {
                verifyContext('render', ctx);
                return `<button>${payload.label}</button><img src=x onerror="alert(1)">`;
            },
            mount(root, ctx) {
                verifyContext('mount', ctx);
                // The root is not attached to its pane yet: discovering the
                // owner with closest() here cannot replace the supplied context.
                assert.equal(root.closest('.chat-container-pane'), null);
                root.querySelector('button').addEventListener('click', () => {
                    clickedOwner = ctx.ownerAgent;
                });
            },
        });
        try {
            const part = chat.appendMessagePart(type, { label: 'Open' }, target);
            assert.equal(part.parentElement, target);
            assert.deepEqual(callbacks, ['gate', 'render', 'mount']);
            assert.equal(part.querySelector('img').getAttribute('onerror'), null,
                'owner context does not bypass core sanitization');
            const button = part.querySelector('button');
            assert.equal(button.textContent, 'Open');
            chat.mountChatPane('third-agent');
            button.click();
            assert.equal(clickedOwner, ownerAgent,
                'later interaction retains the part owner after selection changes');
        } finally {
            unregisterRenderer(type);
        }
    });
}

test('an unowned feature part does not borrow the selected agent', () => {
    const type = 'test/unowned';
    const target = document.createElement('section');
    let seen;
    chat.mountChatPane('selected-agent');
    registerRenderer({
        id: type,
        match: { contentType: type },
        gate(ctx) {
            seen = ctx;
            return ctx.ownerAgent !== undefined;
        },
        render() { assert.fail('an owner-dependent renderer must decline an unowned part'); },
    });
    try {
        const part = chat.appendMessagePart(type, 'literal payload', target);
        assert.equal(seen.ownerAgent, undefined);
        assert.equal(seen.paneElement, target);
        assert.equal(part.querySelector('.message-content').textContent, 'literal payload');
    } finally {
        unregisterRenderer(type);
    }
});
