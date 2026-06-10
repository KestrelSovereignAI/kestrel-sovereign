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
