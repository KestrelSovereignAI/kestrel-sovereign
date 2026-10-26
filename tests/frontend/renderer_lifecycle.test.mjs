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
for (const file of ['highlight.js', 'mermaid.js']) {
    dom.window.eval(readFileSync(new URL(
        '../../kestrel_sovereign/static/shared/markdown/' + file, import.meta.url,
    ), 'utf8'));
}
const registry = await import('../../kestrel_sovereign/static/js/ui-ext/renderers.js');
await import('../../kestrel_sovereign/static/js/icons.js');
globalThis.kicon = window.kicon;
const chat = await import('../../kestrel_sovereign/static/js/chat.js');
const { getOrCreateChatPane } = await import('../../kestrel_sovereign/static/js/ui.js');
chat.setChatDeps({ markdown: {
    sanitizeHtml: purify.sanitize,
    highlightCodeBlocks: dom.window.highlightCodeBlocks,
    renderMermaidDiagrams: dom.window.renderMermaidDiagrams,
} });

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

for (const phase of ['streaming', 'finalization']) {
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
                await chat.finalizeStreamingMessage(message, '', pane);
            }
            oldButton.click();
            const currentButton = message.querySelector('button');
            assert.notEqual(currentButton, oldButton);
            currentButton.click();
            assert.deepEqual(witness, { mounts: 2, clicks: 1, disposals: 1 });
            chat.wipeAgentChatPane(id);
            assert.equal(witness.disposals, 2);
        } finally {
            registry._reset();
            pane.element.remove();
        }
    });
}

test.after(() => { dom.window.close(); });
