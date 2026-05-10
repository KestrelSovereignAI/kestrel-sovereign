import test from 'node:test';
import assert from 'node:assert/strict';

globalThis.window = globalThis.window || {};
globalThis.window.SharedMarkdown = {
    renderMarkdown: () => '',
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
    renderToolActivityHtml,
    splitToolActivity,
    updateStreamingMessage,
    finalizeStreamingMessage,
} = await import(
    '../../kestrel_sovereign/static/js/chat.js'
);

test('renderToolActivityHtml returns a collapsed details block for tool calls', () => {
    const html = renderToolActivityHtml([
        '\u{1F527} Calling save_fact...',
        '\u2713 save_fact complete (12ms)',
    ].join('\n'));

    assert.match(html, /<details class="tool-activity-container tool-activity-expandable">/);
    assert.match(html, /<summary class="tool-activity-summary">/);
    assert.match(html, /Tool call: save_fact/);
    assert.match(html, /2 events/);
    assert.doesNotMatch(html, /<details[^>]*\sopen[\s>]/);
});

test('renderToolActivityHtml escapes tool names and errors before inserting HTML', () => {
    const html = renderToolActivityHtml(
        '\u274C dangerous_tool failed: <img src=x onerror=alert(1)>',
    );

    assert.match(html, /dangerous_tool failed:/);
    assert.match(html, /&lt;img src=x onerror=alert\(1\)&gt;/);
    assert.doesNotMatch(html, /<img src=x/);
});

test('updateStreamingMessage renders expandable tool activity above response text', () => {
    const msgDiv = makeNode();
    const contentDiv = makeNode();
    contentDiv.className = 'message-content streaming';
    msgDiv.appendChild(contentDiv);

    updateStreamingMessage(
        msgDiv,
        '\u{1F527} Calling lookup...\n\u2713 lookup complete (7ms)\n---\nThe lookup finished.',
    );

    assert.match(contentDiv.innerHTML, /tool-activity-expandable/);
    assert.match(contentDiv.innerHTML, /Tool call: lookup/);
    assert.match(contentDiv.innerHTML, /response-content/);
    assert.match(contentDiv.innerHTML, /The lookup finished\./);
});

test('finalizeStreamingMessage preserves expandable tool activity after stream completion', async () => {
    const msgDiv = makeNode();
    const contentDiv = makeNode();
    contentDiv.className = 'message-content streaming';
    msgDiv.appendChild(contentDiv);

    await finalizeStreamingMessage(
        msgDiv,
        '\u{1F527} Calling memory_agency_feature...\n\u{1F527} Calling response_audit_feature...\n---\nI tried to save it.',
    );

    assert.match(contentDiv.innerHTML, /tool-activity-expandable/);
    assert.match(contentDiv.innerHTML, /2 tool call events/);
    assert.match(contentDiv.innerHTML, /Calling memory_agency_feature/);
    const responseDiv = contentDiv.children.find((child) => child.classList?.contains('response-content'));
    assert.ok(responseDiv, 'final renderer must append a response-content child');
    assert.match(responseDiv.innerHTML, /I tried to save it\./);
});

test('splitToolActivity leaves normal markdown horizontal rules alone', () => {
    const content = 'Intro paragraph\n---\nDetails after a normal markdown rule';
    const split = splitToolActivity(content);

    assert.equal(split.hasToolActivity, false);
    assert.equal(split.toolActivity, '');
    assert.equal(split.response, content);
});

test('splitToolActivity preserves final text when no separator is emitted', () => {
    const split = splitToolActivity([
        '\u{1F527} Calling search_memory...',
        '\u2713 search_memory complete (11ms)',
        'Error: Maximum tool call iterations exceeded',
    ].join('\n'));

    assert.equal(split.hasToolActivity, true);
    assert.equal(
        split.toolActivity,
        '\u{1F527} Calling search_memory...\n\u2713 search_memory complete (11ms)',
    );
    assert.equal(split.response, 'Error: Maximum tool call iterations exceeded');
});

test('splitToolActivity does not collapse ordinary checkmark replies', () => {
    const content = '\u2713 Done - here is the summary.';
    const split = splitToolActivity(content);

    assert.equal(split.hasToolActivity, false);
    assert.equal(split.toolActivity, '');
    assert.equal(split.response, content);
});

test('splitToolActivity requires a tool start line before completion or error statuses', () => {
    for (const content of [
        '\u2713 Migration complete',
        '\u274C Build failed: missing dependency',
    ]) {
        const split = splitToolActivity(content);

        assert.equal(split.hasToolActivity, false);
        assert.equal(split.toolActivity, '');
        assert.equal(split.response, content);
    }
});
