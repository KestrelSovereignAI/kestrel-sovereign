import test from 'node:test';
import assert from 'node:assert/strict';

globalThis.window = globalThis.window || {};
globalThis.window.SharedMarkdown = {
    renderMarkdown: (s) => `<p>${s}</p>`,
    renderStreamingMarkdown: (s) => `<p>${s}</p>`,
    highlightCodeBlocks: () => {},
    renderMermaidDiagrams: () => {},
    finalizeMarkdown: async (el, s) => { el.innerHTML = `<p>${s}</p>`; },
};

function makeNode(tag = 'div') {
    const node = {
        tagName: tag.toUpperCase(),
        nodeType: 1,
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
        _innerHTML: '',
        _textContent: '',
        scrollTop: 0,
        scrollHeight: 0,
        addEventListener() {},
        querySelector(sel) {
            if (sel === '.message-content') {
                return this.children.find((c) => c.classList?.contains('message-content')) || null;
            }
            return null;
        },
        querySelectorAll() { return []; },
        appendChild(c) {
            c.parentNode = this;
            this.children.push(c);
            this.childNodes.push(c);
            return c;
        },
        remove() {},
        get firstChild() { return this.children[0] || null; },
    };
    Object.defineProperty(node, 'className', {
        get() { return [...node.classList._set].join(' '); },
        set(v) {
            node.classList._set = new Set(String(v).split(/\s+/).filter(Boolean));
        },
    });
    Object.defineProperty(node, 'innerHTML', {
        get() { return node._innerHTML; },
        set(v) {
            node._innerHTML = String(v);
            node.children = [];
            node.childNodes = [];
        },
    });
    Object.defineProperty(node, 'textContent', {
        get() { return node._textContent; },
        set(v) {
            node._textContent = String(v);
            node._innerHTML = String(v)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');
        },
    });
    return node;
}

const chatContainer = makeNode();
chatContainer.id = 'chat-container';

globalThis.document = {
    getElementById(id) {
        if (id === 'chat-container') return chatContainer;
        return null;
    },
    createElement: (tag) => makeNode(tag),
    head: makeNode('head'),
    body: makeNode('body'),
    addEventListener() {},
    querySelector() { return null; },
    querySelectorAll() { return []; },
};
globalThis.sessionStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.location = { href: '/', search: '' };
globalThis.fetch = async () => ({ ok: false, status: 500 });
globalThis.kicon = () => '';
globalThis.CSS = { escape: (s) => String(s) };

const {
    renderAgentContentHtml,
    renderToolCardsHtml,
    normalizeToolEvents,
    updateStreamingMessage,
    finalizeStreamingMessage,
} = await import(
    '../../kestrel_sovereign/static/js/chat.js'
);

// #1659: tool activity is structured data. renderAgentContentHtml normalizes
// both the persisted metadata shape ({type,tool,ms,error,pos}) and the live
// sentinel shape ({phase,name,ms,detail,pos}) and places cards by position.

test('renderAgentContentHtml renders a collapsed details card per tool call', () => {
    const html = renderAgentContentHtml('', {
        toolEvents: [
            { type: 'start', tool: 'save_fact', pos: 0 },
            { type: 'complete', tool: 'save_fact', ms: 12, pos: 0 },
        ],
    });
    assert.match(html, /<div class="tool-activity-container">/);
    assert.match(html, /<details class="tool-activity-expandable tool-activity-call">/);
    assert.match(html, /Tool call: save_fact/);
    assert.match(html, /complete · 12ms · 2 events/);
    assert.doesNotMatch(html, /<details[^>]*\sopen[\s>]/);
});

test('renders multiple tool calls as separate details blocks', () => {
    const html = renderAgentContentHtml('', {
        toolEvents: [
            { type: 'start', tool: 'first_tool', pos: 0 },
            { type: 'start', tool: 'second_tool', pos: 0 },
            { type: 'complete', tool: 'first_tool', ms: 3, pos: 0 },
            { type: 'error', tool: 'second_tool', error: 'denied', pos: 0 },
        ],
    });
    assert.equal((html.match(/tool-activity-call/g) || []).length, 2);
    assert.match(html, /Tool call: first_tool/);
    assert.match(html, /complete · 3ms · 2 events/);
    assert.match(html, /Tool call: second_tool/);
    assert.match(html, /error · denied · 2 events/);
});

test('escapes tool error detail before inserting HTML', () => {
    const html = renderAgentContentHtml('', {
        toolEvents: [
            { type: 'start', tool: 'dangerous_tool', pos: 0 },
            { type: 'error', tool: 'dangerous_tool', error: '<img src=x onerror=alert(1)>', pos: 0 },
        ],
    });
    assert.match(html, /&lt;img src=x onerror=alert\(1\)&gt;/);
    assert.doesNotMatch(html, /<img src=x/);
});

test('places tool cards between prose by position (pre, card, post)', () => {
    const pre = 'I will check that now.';
    const post = 'The lookup finished.';
    const html = renderAgentContentHtml(pre + post, {
        toolEvents: [
            { phase: 'start', name: 'lookup', pos: pre.length },
            { phase: 'done', name: 'lookup', ms: 7, pos: pre.length },
        ],
    });
    assert.match(html, /response-prelude/);
    assert.match(html, /I will check that now\./);
    assert.match(html, /Tool call: lookup/);
    assert.match(html, /The lookup finished\./);
    assert.ok(
        html.indexOf('I will check that now') < html.indexOf('tool-activity-container'),
        'pre-tool prose precedes the card',
    );
    assert.ok(
        html.indexOf('tool-activity-container') < html.indexOf('The lookup finished'),
        'post-tool prose follows the card',
    );
});

test('no toolEvents → plain prose, no card container', () => {
    const html = renderAgentContentHtml('Just an answer.', { toolEvents: [] });
    assert.doesNotMatch(html, /tool-activity-container/);
    assert.match(html, /Just an answer\./);
});

test('normalizeToolEvents maps metadata and sentinel shapes, drops junk', () => {
    const norm = normalizeToolEvents([
        { type: 'complete', tool: 'a', ms: 5, pos: 3 },
        { phase: 'start', name: 'b', detail: 'x', pos: 1 },
        { type: 'bogus' },
        null,
    ]);
    assert.deepEqual(norm.map((e) => e.phase), ['done', 'start']);
    assert.equal(norm[0].name, 'a');
    assert.equal(norm[1].detail, 'x');
});

test('updateStreamingMessage renders tool card + prose from toolEvents', () => {
    const msgDiv = makeNode();
    const contentDiv = makeNode();
    contentDiv.className = 'message-content streaming';
    msgDiv.appendChild(contentDiv);
    updateStreamingMessage(
        msgDiv, 'The lookup finished.', null, [],
        [
            { phase: 'start', name: 'lookup', pos: 0 },
            { phase: 'done', name: 'lookup', ms: 7, pos: 0 },
        ],
    );
    assert.match(contentDiv.innerHTML, /tool-activity-expandable/);
    assert.match(contentDiv.innerHTML, /Tool call: lookup/);
    assert.match(contentDiv.innerHTML, /The lookup finished\./);
});

test('finalizeStreamingMessage renders tool cards from pane.toolEvents', async () => {
    const msgDiv = makeNode();
    const contentDiv = makeNode();
    contentDiv.className = 'message-content streaming';
    msgDiv.appendChild(contentDiv);
    const pane = {
        element: chatContainer,
        toolEvents: [
            { phase: 'start', name: 'memory_agency_feature', pos: 0 },
            { phase: 'start', name: 'response_audit_feature', pos: 0 },
        ],
        streamBaseline: 0,
        thinkingItems: [],
    };
    await finalizeStreamingMessage(msgDiv, 'I tried to save it.', pane);
    assert.match(contentDiv.innerHTML, /tool-activity-expandable/);
    assert.equal((contentDiv.innerHTML.match(/tool-activity-call/g) || []).length, 2);
    assert.match(contentDiv.innerHTML, /memory_agency_feature/);
    assert.match(contentDiv.innerHTML, /I tried to save it\./);
});

test('finalizeStreamingMessage keeps pre-tool prose before the card', async () => {
    const msgDiv = makeNode();
    const contentDiv = makeNode();
    contentDiv.className = 'message-content streaming';
    msgDiv.appendChild(contentDiv);
    const pre = 'I will check that now.';
    const post = 'The lookup finished.';
    const pane = {
        element: chatContainer,
        toolEvents: [
            { phase: 'start', name: 'lookup', pos: pre.length },
            { phase: 'done', name: 'lookup', ms: 7, pos: pre.length },
        ],
        streamBaseline: 0,
        thinkingItems: [],
    };
    await finalizeStreamingMessage(msgDiv, pre + post, pane);
    assert.match(contentDiv.innerHTML, /response-prelude/);
    assert.ok(
        contentDiv.innerHTML.indexOf('I will check that now')
            < contentDiv.innerHTML.indexOf('tool-activity-container'),
        'pre-tool prose must render before the tool card',
    );
});

test('sliceToolEvents drop: a baseline excludes pre-baseline cards (via finalize)', async () => {
    // streamBaseline simulates a mid-stream restart_status (#1560): only the
    // post-baseline tool card should render against the sliced content.
    const msgDiv = makeNode();
    const contentDiv = makeNode();
    contentDiv.className = 'message-content streaming';
    msgDiv.appendChild(contentDiv);
    const pane = {
        element: chatContainer,
        toolEvents: [
            { phase: 'start', name: 'old_tool', pos: 0 },
            { phase: 'start', name: 'new_tool', pos: 30 },
        ],
        streamBaseline: 10,
        thinkingItems: [],
    };
    await finalizeStreamingMessage(msgDiv, 'post-baseline content here', pane);
    assert.match(contentDiv.innerHTML, /new_tool/);
    assert.doesNotMatch(contentDiv.innerHTML, /old_tool/);
});
