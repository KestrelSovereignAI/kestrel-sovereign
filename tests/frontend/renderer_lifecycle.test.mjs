import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import createDOMPurify from 'dompurify';
import { readFileSync } from 'node:fs';

const dom = new JSDOM('<!doctype html><body></body>', {
    url: 'https://console.example/', runScripts: 'outside-only',
});
Object.assign(globalThis, {
    window: dom.window,
    document: dom.window.document,
    location: dom.window.location,
    sessionStorage: dom.window.sessionStorage,
    localStorage: dom.window.localStorage,
});
const purify = createDOMPurify(dom.window);
dom.window.DOMPurify = purify;
for (const file of ['parse.js', 'highlight.js', 'mermaid.js', 'katex.js', 'index.js']) {
    dom.window.eval(readFileSync(new URL(
        '../../kestrel_sovereign/static/shared/markdown/' + file, import.meta.url,
    ), 'utf8'));
}
const registry = await import('../../kestrel_sovereign/static/js/ui-ext/renderers.js');
await import('../../kestrel_sovereign/static/js/icons.js');
globalThis.kicon = window.kicon;
const chat = await import('../../kestrel_sovereign/static/js/chat.js');
const { getOrCreateChatPane } = await import('../../kestrel_sovereign/static/js/ui.js');
const voice = await import('../../kestrel_sovereign/static/js/voice/ui.js');
const history = await import('../../kestrel_sovereign/static/js/history.js');
chat.setChatDeps({ markdown: dom.window.SharedMarkdown });

function interactiveRenderer(id, witness) {
    return {
        id, match: { contentType: id },
        render: () => '<button type="button">View action</button>',
        mount(root) {
            witness.mounts += 1;
            const button = root.querySelector('button');
            const click = () => { witness.clicks += 1; };
            button.addEventListener('click', click);
            return () => {
                witness.disposals += 1;
                button.removeEventListener('click', click);
            };
        },
    };
}

function mountedFixture(id) {
    const witness = { mounts: 0, clicks: 0, disposals: 0 };
    const renderer = interactiveRenderer(id, witness);
    registry.registerRenderer(renderer);
    const container = document.createElement('div');
    container.innerHTML = registry.renderToSafeHtml(
        renderer, {}, {}, purify.sanitize, (s) => String(s),
    );
    document.body.append(container);
    const dispose = registry.mountRenderers(container);
    return { witness, container, dispose, button: container.querySelector('button') };
}

test('disposing a mount permits a fresh mount of the same actual DOM', () => {
    const { witness, container, dispose, button } = mountedFixture('remount-view');
    try {
        button.click();
        dispose();
        dispose();
        button.click();
        assert.equal(witness.disposals, 1);
        const remounted = registry.mountRenderers(container);
        dispose();
        registry.mountRenderers(container);
        button.click();
        assert.deepEqual(witness, { mounts: 2, clicks: 2, disposals: 1 });
        remounted();
        dispose();
        assert.equal(witness.disposals, 2);
    } finally {
        registry._reset();
        container.remove();
    }
});

test('unregistering a renderer disposes its existing native event listeners', () => {
    const { witness, container, button } = mountedFixture('retired-view');
    try {
        registry.unregisterRenderer('retired-view');
        button.click();
        assert.deepEqual(witness, { mounts: 1, clicks: 0, disposals: 1 });
    } finally {
        registry._reset();
        container.remove();
    }
});

test('synchronous retirement inside mount releases the eventual teardown once', () => {
    const witness = { mounts: 0, clicks: 0, disposals: 0 };
    const renderer = interactiveRenderer('self-retiring-view', witness);
    const mount = renderer.mount;
    renderer.mount = (root) => {
        const dispose = mount(root);
        registry.unregisterRenderer(renderer.id);
        return dispose;
    };
    registry.registerRenderer(renderer);
    const container = document.createElement('div');
    container.innerHTML = registry.renderToSafeHtml(renderer, {}, {}, purify.sanitize, String);
    try {
        const dispose = registry.mountRenderers(container);
        container.querySelector('button').click();
        dispose();
        assert.deepEqual(witness, { mounts: 1, clicks: 0, disposals: 1 });
    } finally {
        registry._reset();
        container.remove();
    }
});

test('replacing a renderer releases the previous implementation before remount', () => {
    const { witness, container, button } = mountedFixture('replaced-view');
    const replacement = { mounts: 0, clicks: 0, disposals: 0 };
    try {
        registry.registerRenderer(interactiveRenderer('replaced-view', replacement));
        button.click();
        assert.deepEqual(witness, { mounts: 1, clicks: 0, disposals: 1 });
        registry.mountRenderers(container);
        button.click();
        assert.deepEqual(replacement, { mounts: 1, clicks: 1, disposals: 0 });
    } finally {
        registry._reset();
        container.remove();
    }
});

test('explicit container teardown is idempotent and scoped to its own DOM', () => {
    const first = mountedFixture('first-view');
    const second = mountedFixture('second-view');
    try {
        assert.equal(typeof registry.unmountRenderers, 'function');
        registry.unmountRenderers(first.container.firstElementChild);
        first.button.click();
        assert.deepEqual(first.witness, { mounts: 1, clicks: 0, disposals: 1 });
        registry.unmountRenderers(first.container);
        second.button.click();
        assert.deepEqual(first.witness, { mounts: 1, clicks: 0, disposals: 1 });
        assert.deepEqual(second.witness, { mounts: 1, clicks: 1, disposals: 0 });
    } finally {
        registry._reset();
        first.container.remove();
        second.container.remove();
    }
});

test('a failed mount releases its claim so the actual root can recover', () => {
    const witness = { mounts: 0, clicks: 0, disposals: 0 };
    const renderer = interactiveRenderer('recovering-view', witness);
    const mount = renderer.mount;
    let attempts = 0;
    renderer.mount = (root) => {
        attempts += 1;
        if (attempts === 1) throw new Error('Controlled renderer startup failure');
        return mount(root);
    };
    registry.registerRenderer(renderer);
    const container = document.createElement('div');
    container.innerHTML = registry.renderToSafeHtml(renderer, {}, {}, purify.sanitize, String);
    try {
        registry.mountRenderers(container);
        const dispose = registry.mountRenderers(container);
        container.querySelector('button').click();
        assert.equal(attempts, 2);
        assert.deepEqual(witness, { mounts: 1, clicks: 1, disposals: 0 });
        dispose();
        assert.equal(witness.disposals, 1);
    } finally {
        registry._reset();
        container.remove();
    }
});

test('wiping one real chat pane releases its parts and preserves a background pane', () => {
    const first = { mounts: 0, clicks: 0, disposals: 0 };
    const second = { mounts: 0, clicks: 0, disposals: 0 };
    registry.registerRenderer(interactiveRenderer('part-first', first));
    registry.registerRenderer(interactiveRenderer('part-second', second));
    const a = getOrCreateChatPane('lifecycle-agent-a');
    const b = getOrCreateChatPane('lifecycle-agent-b');
    document.body.append(a.element);
    try {
        chat.appendMessagePart('part-first', {}, a.element);
        chat.appendMessagePart('part-second', {}, b.element);
        const aButton = a.element.querySelector('button');
        const bButton = b.element.querySelector('button');
        assert.ok(aButton && bButton);
        assert.equal(b.element.isConnected, false);
        chat.wipeAgentChatPane('lifecycle-agent-a');
        aButton.click();
        bButton.click();
        assert.deepEqual(first, { mounts: 1, clicks: 0, disposals: 1 });
        assert.deepEqual(second, { mounts: 1, clicks: 1, disposals: 0 });
        chat.wipeAgentChatPane('lifecycle-agent-b');
        assert.equal(second.disposals, 1);
    } finally {
        registry._reset();
        a.element.remove();
        b.element.remove();
    }
});

for (const phase of ['streaming', 'finalization', 'thinking-finalization']) {
    test(`${phase} replaces tool-card mounts without retaining old listeners`, async () => {
        const id = 'lifecycle-tool-' + phase;
        const witness = { mounts: 0, clicks: 0, disposals: 0 };
        const renderer = interactiveRenderer(id, witness);
        renderer.match = { tool: id };
        registry.registerRenderer(renderer);
        const pane = getOrCreateChatPane(id);
        const message = document.createElement('div');
        message.innerHTML = '<div class="message-content streaming"></div>';
        pane.element.append(message);
        const events = [{ phase: 'start', name: id, pos: 0 }];
        try {
            chat.updateStreamingMessage(message, '', pane.element, [], events);
            const oldButton = message.querySelector('button');
            assert.ok(oldButton);
            if (phase === 'streaming') {
                chat.updateStreamingMessage(message, '', pane.element, [], events);
            } else {
                pane.toolEvents = events;
                if (phase === 'thinking-finalization') pane.thinkingItems = [''];
                await chat.finalizeStreamingMessage(message, '', pane);
            }
            oldButton.click();
            const currentButton = message.querySelector('button');
            assert.notEqual(currentButton, oldButton);
            currentButton.click();
            const mounts = phase === 'thinking-finalization' ? 3 : 2;
            assert.deepEqual(witness, { mounts, clicks: 1, disposals: mounts - 1 });
            chat.wipeAgentChatPane(id);
            assert.equal(witness.disposals, mounts);
        } finally {
            registry._reset();
            pane.element.remove();
        }
    });
}

test('voice tool-card replacement retires its old renderer', () => {
    const id = 'voice-card-lifecycle';
    const witness = { mounts: 0, clicks: 0, disposals: 0 };
    const renderer = interactiveRenderer(id, witness);
    renderer.match = { tool: id };
    registry.registerRenderer(renderer);
    const message = document.createElement('div');
    message.innerHTML = '<div class="message-content"></div>';
    const card = { name: id, status: 'running', events: [{ phase: 'start', name: id }] };
    try {
        voice.updateVoiceToolCard(message, card);
        const previous = message.querySelector('button');
        assert.ok(previous);
        voice.updateVoiceToolCard(message, { ...card, status: 'done' });
        previous.click();
        message.querySelector('button').click();
        assert.deepEqual(witness, { mounts: 2, clicks: 1, disposals: 1 });
    } finally {
        registry._reset();
        message.remove();
    }
});

test('deleting all bubbles of one message retires their renderer mounts', async () => {
    const a = mountedFixture('deleted-message-first');
    const b = mountedFixture('deleted-message-second');
    for (const fixture of [a, b]) {
        fixture.container.classList.add('message');
        fixture.container.dataset.messageId = 'shared-message';
    }
    try {
        const removed = new Promise(resolve => {
            const observer = new dom.window.MutationObserver(() => {
                if (!a.container.parentNode && !b.container.parentNode) {
                    observer.disconnect();
                    resolve();
                }
            });
            observer.observe(document.body, { childList: true });
        });
        history.fadeRemoveMessageNodes('shared-message', a.container);
        await removed;
        a.button.click(); b.button.click();
        assert.deepEqual(a.witness, { mounts: 1, clicks: 0, disposals: 1 });
        assert.deepEqual(b.witness, { mounts: 1, clicks: 0, disposals: 1 });
    } finally {
        registry._reset();
        a.container.remove(); b.container.remove();
    }
});

test('mounting a pane retires leftover non-pane content', () => {
    const leftover = mountedFixture('leftover-content');
    const container = document.createElement('div');
    container.id = 'chat-container';
    container.append(leftover.container);
    document.body.append(container);
    try {
        chat.mountChatPane('new-leftover-owner');
        leftover.button.click();
        assert.deepEqual(leftover.witness, { mounts: 1, clicks: 0, disposals: 1 });
    } finally {
        chat.wipeAgentChatPane('new-leftover-owner');
        registry._reset();
        container.remove();
    }
});

test('a cleared pane cannot acquire new mounts after asynchronous finalization', async () => {
    const id = 'late-finalization';
    const witness = { mounts: 0, clicks: 0, disposals: 0 };
    const renderer = interactiveRenderer(id, witness);
    renderer.match = { tool: id };
    registry.registerRenderer(renderer);
    const pane = getOrCreateChatPane(id);
    const message = document.createElement('div');
    message.innerHTML = '<div class="message-content streaming"></div>';
    pane.element.append(message);
    pane.toolEvents = [{ phase: 'start', name: id, pos: 0 }];
    let received, release;
    const started = new Promise(resolve => { received = resolve; });
    const resumed = new Promise(resolve => { release = resolve; });
    chat.setChatDeps({ markdown: { ...dom.window.SharedMarkdown,
        renderMermaidDiagrams: async element => {
            await dom.window.SharedMarkdown.renderMermaidDiagrams(element);
            received();
            await resumed;
        },
    } });
    try {
        const finalizing = chat.finalizeStreamingMessage(message, '', pane);
        await started;
        chat.wipeAgentChatPane(id);
        pane.thinkingItems = ['']; // a new turn started while the old finalizer was held
        release();
        await finalizing;
        assert.equal(message.parentNode, null);
        assert.deepEqual(witness, { mounts: 1, clicks: 0, disposals: 1 });
    } finally {
        release();
        chat.setChatDeps({ markdown: dom.window.SharedMarkdown });
        registry._reset();
        pane.element.remove();
    }
});

test.after(() => { dom.window.close(); });
