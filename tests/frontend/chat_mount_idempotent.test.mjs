// #2197: mount() / initChat() is idempotent per chat surface. An embedder mount
// race (two async callers both reaching mount(container)) used to run the pane
// bootstrap twice: the second pass took the already-mounted PANE as
// #chat-container's first child and appended it into itself, throwing
// HierarchyRequestError and aborting init mid-way. jsdom gives the real DOM
// semantics that throw, against the production index.html markup.
//
// The cases run in declaration order and share chat.js's module state: each
// builds on the surface the previous one left mounted.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { JSDOM } from 'jsdom';

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
window.SharedMarkdown = {
    renderMarkdown: () => '',
    renderStreamingMarkdown: () => '',
    highlightCodeBlocks: () => {},
    renderMermaidDiagrams: () => {},
    finalizeMarkdown: async () => {},
};

// Every SSE stream the chat component opens goes through this constructor.
let eventSourcesOpened = 0;
globalThis.EventSource = class {
    constructor() { eventSourcesOpened += 1; }
    close() {}
    addEventListener() {}
};

// Record every listener registration on any DOM node or the window, so a
// second mount can be shown to add none.
const registrations = [];
const realAddEventListener = dom.window.EventTarget.prototype.addEventListener;
dom.window.EventTarget.prototype.addEventListener = function (type, ...rest) {
    registrations.push({ target: this, type });
    return realAddEventListener.call(this, type, ...rest);
};

const chat = await import('../../kestrel_sovereign/static/js/chat.js');
const { state } = await import('../../kestrel_sovereign/static/js/ui.js');

function chatDeps(hostAgent) {
    return { api: { hasCapability: () => true, getHostAgent: () => hostAgent } };
}

function mountRecording(container, hostAgent) {
    const start = registrations.length;
    chat.mount(container, { deps: chatDeps(hostAgent) });
    return registrations.slice(start);
}

function panesIn(chatContainer) {
    return chatContainer.querySelectorAll('.chat-container-pane');
}

test('mounting the same container twice neither throws nor duplicates panes, listeners or SSE', () => {
    const panel = document.getElementById('panel-chat');
    const chatContainer = panel.querySelector('#chat-container');

    const first = mountRecording(panel, 'agent-a');
    assert.ok(first.length > 0, 'the first mount wires the composer and viewport');
    assert.ok(first.some((r) => r.target === dom.window && r.type === 'resize'),
        'the window resize listener is part of what a mount wires');
    assert.equal(chatContainer.getAttribute('data-kestrel-chat-mounted'), 'true',
        'a completed mount marks its chat surface');

    let second;
    assert.doesNotThrow(() => { second = mountRecording(panel, 'agent-a'); });

    assert.deepEqual(second.map((r) => r.type), [],
        'the second mount registers no listener on any element or the window');
    assert.equal(chatContainer.children.length, 1, 'the viewport holds one child');
    assert.equal(panesIn(chatContainer).length, 1, 'exactly one pane, not nested');
    const pane = state.chatPanes.get('agent-a');
    assert.equal(chatContainer.children[0], pane.element);
    assert.equal(state.mountedChatAgent, 'agent-a');

    // The same chat DOM reached through the document root is the same surface.
    let viaDocument;
    assert.doesNotThrow(() => { viaDocument = mountRecording(null, 'agent-a'); });
    assert.deepEqual(viaDocument.map((r) => r.type), []);
    assert.equal(panesIn(chatContainer).length, 1);
    assert.equal(eventSourcesOpened, 0, 'mounting opens no SSE stream, once or twice');
});

test('a second mount leaves the mounted pane alone after an agent switch', () => {
    // The Frinz race: the first caller mounts and selects a companion (its pane
    // replaces the initial one); the second caller then reaches mount() with
    // that companion as the host agent. Its pane is the viewport's child, so a
    // re-run bootstrap would append it into itself.
    const panel = document.getElementById('panel-chat');
    const chatContainer = panel.querySelector('#chat-container');
    chat.mountChatPane('companion');
    const companionPane = state.chatPanes.get('companion');
    assert.equal(chatContainer.children[0], companionPane.element);

    let added;
    assert.doesNotThrow(() => { added = mountRecording(panel, 'companion'); });

    assert.deepEqual(added.map((r) => r.type), []);
    assert.equal(chatContainer.children.length, 1);
    assert.equal(chatContainer.children[0], companionPane.element);
    assert.equal(panesIn(chatContainer).length, 1);
    assert.equal(state.mountedChatAgent, 'companion',
        'the second mount does not reset the mounted agent');
});

test('the pane bootstrap never moves the pane into itself', () => {
    // Defence in depth below the marker: a bootstrap that runs over a viewport
    // already holding the initial pane moves only the other children.
    const panel = document.getElementById('panel-chat');
    const chatContainer = panel.querySelector('#chat-container');
    const companionPane = state.chatPanes.get('companion');
    delete chatContainer.dataset.kestrelChatMounted;
    const stray = document.createElement('div');
    stray.className = 'stray-welcome';
    chatContainer.appendChild(stray);

    assert.doesNotThrow(() => chat.mount(panel, { deps: chatDeps('companion') }));

    assert.equal(chatContainer.children.length, 1);
    assert.equal(chatContainer.children[0], companionPane.element);
    assert.equal(panesIn(chatContainer).length, 1);
    assert.equal(stray.parentNode, companionPane.element,
        'the bare child still moves into the pane');
    assert.equal(chatContainer.getAttribute('data-kestrel-chat-mounted'), 'true');
});

function freshChatHost(id, viewportHtml) {
    const host = document.createElement('section');
    host.id = id;
    host.innerHTML = `<div id="chat-container">${viewportHtml}</div>
        <textarea id="message-input"></textarea>
        <button id="send-button"></button>`;
    document.body.appendChild(host);
    return host;
}

test('a fresh container still mounts as before, and only once', () => {
    // The embedder tears the old chat down first: jsdom resolves a scoped
    // `#id` query through the document, so duplicate ids would hide the new one.
    document.getElementById('panel-chat').remove();
    const host = freshChatHost('fresh-host',
        'banner text <div class="demo-banner">demo</div>'
        + '<div class="message agent-message">welcome</div>\n');
    const chatContainer = host.querySelector('#chat-container');
    const original = Array.from(chatContainer.childNodes);
    assert.equal(original.length, 4, 'two elements and two text nodes to move');

    const first = mountRecording(host, 'fresh-agent');

    assert.ok(first.some((r) => r.target === chatContainer && r.type === 'click'),
        'the fresh viewport gets its delegated link listener');
    assert.ok(first.some((r) => r.target === host.querySelector('#send-button')
        && r.type === 'click'), 'the fresh send button is wired');
    const pane = state.chatPanes.get('fresh-agent');
    assert.deepEqual(Array.from(chatContainer.childNodes), [pane.element],
        'the pane is the viewport\'s only node');
    assert.deepEqual(Array.from(pane.element.childNodes), original,
        'every node, text included, moved into the pane in order');
    assert.equal(state.mountedChatAgent, 'fresh-agent');
    assert.equal(chatContainer.getAttribute('data-kestrel-chat-mounted'), 'true');

    let second;
    assert.doesNotThrow(() => { second = mountRecording(host, 'fresh-agent'); });
    assert.deepEqual(second.map((r) => r.type), []);
    assert.equal(panesIn(chatContainer).length, 1);
    assert.equal(eventSourcesOpened, 0);
});

test('a mount that fails partway leaves its surface unmarked, so a retry completes it', () => {
    document.getElementById('fresh-host').remove();
    const host = freshChatHost('retry-host', '');
    const chatContainer = host.querySelector('#chat-container');
    const failingDeps = {
        ...chatDeps('retry-agent'),
        getOrCreateChatPane: () => { throw new Error('pane store unavailable'); },
    };

    assert.throws(() => chat.mount(host, { deps: failingDeps }), /pane store unavailable/);
    assert.equal(chatContainer.getAttribute('data-kestrel-chat-mounted'), null,
        'an init that did not complete does not mark the surface');

    chat.mount(host, { deps: { ...chatDeps('retry-agent'), getOrCreateChatPane: null } });

    const pane = state.chatPanes.get('retry-agent');
    assert.deepEqual(Array.from(chatContainer.childNodes), [pane.element]);
    assert.equal(state.mountedChatAgent, 'retry-agent');
    assert.equal(chatContainer.getAttribute('data-kestrel-chat-mounted'), 'true');
});

test('remounting an existing surface after another one rebinds cached DOM without rewiring', async () => {
    document.getElementById('retry-host').remove();
    const first = freshChatHost('remount-first', 'first surface');
    first.insertAdjacentHTML('beforeend', '<div id="thinking-indicator"></div>'
        + '<button id="composer-mode-toggle"></button>');
    chat.mount(first, { deps: chatDeps('remount-a') });
    const firstViewport = first.querySelector('#chat-container');
    first.remove();
    const second = freshChatHost('remount-second', 'second surface');
    second.insertAdjacentHTML('beforeend', '<div id="thinking-indicator"></div>'
        + '<button id="composer-mode-toggle"></button>');
    chat.mount(second, { deps: chatDeps('remount-b') });
    const secondViewport = second.querySelector('#chat-container');
    const secondPane = state.chatPanes.get('remount-b').element;
    second.remove();
    document.body.appendChild(first);
    const added = mountRecording(first, 'remount-a');
    assert.deepEqual(added, [], 'cached reference refresh must not duplicate listeners');
    const firstPane = state.chatPanes.get('remount-a');
    firstPane.composerMode = 'queue';
    chat.updateComposerModeToggle();
    assert.equal(first.querySelector('#composer-mode-toggle').dataset.mode, 'queue');
    assert.equal(second.querySelector('#composer-mode-toggle').dataset.mode, 'interrupt');
    state.waitingAgents.add('remount-a');
    try {
        chat.updateThinkingIndicator();
        assert.equal(first.querySelector('#thinking-indicator').style.display, 'flex');
        assert.notEqual(second.querySelector('#thinking-indicator').style.display, 'flex');
        first.querySelector('#message-input').value = 'queue from reactivated surface';
        second.querySelector('#message-input').value = 'must remain untouched';
        await chat.sendMessage();
        assert.equal(firstPane.queuedMessage, 'queue from reactivated surface');
        assert.equal(first.querySelector('#message-input').value, '');
        assert.equal(second.querySelector('#message-input').value, 'must remain untouched');
        assert.equal(eventSourcesOpened, 0, 'queueing uses no live provider or stream');
    } finally {
        state.waitingAgents.delete('remount-a');
    }
    chat.mountChatPane('remount-target');
    assert.equal(firstViewport.children[0], state.chatPanes.get('remount-target').element,
        'the reactivated surface, not the detached previous one, owns the new pane');
    assert.equal(secondViewport.children[0], secondPane, 'inactive surface stays untouched');
});
