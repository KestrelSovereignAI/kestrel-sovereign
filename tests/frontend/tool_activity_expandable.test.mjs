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
    renderToolActivityHtml,
    segmentToolActivity,
    updateStreamingMessage,
    finalizeStreamingMessage,
} = await import(
    '../../kestrel_sovereign/static/js/chat.js'
);

const toolSegments = (segs) => segs.filter((s) => s.kind === 'tools');
const proseText = (segs) => segs.filter((s) => s.kind === 'prose').map((s) => s.text).join('\n');

test('renderToolActivityHtml returns a collapsed details block per tool call', () => {
    const html = renderToolActivityHtml([
        '\u{1F527} Calling save_fact...',
        '\u2713 save_fact complete (12ms)',
    ].join('\n'));

    assert.match(html, /<div class="tool-activity-container">/);
    assert.match(html, /<details class="tool-activity-expandable tool-activity-call">/);
    assert.match(html, /<summary class="tool-activity-summary">/);
    assert.match(html, /Tool call: save_fact/);
    assert.match(html, /complete · 12ms · 2 events/);
    assert.match(html, /2 events/);
    assert.doesNotMatch(html, /<details[^>]*\sopen[\s>]/);
});

test('renderToolActivityHtml renders multiple tool calls as separate details blocks', () => {
    const html = renderToolActivityHtml([
        '\u{1F527} Calling first_tool...',
        '\u{1F527} Calling second_tool...',
        '\u2713 first_tool complete (3ms)',
        '\u274C second_tool failed: denied',
    ].join('\n'));

    assert.equal((html.match(/<details class="tool-activity-expandable tool-activity-call">/g) || []).length, 2);
    assert.match(html, /Tool call: first_tool/);
    assert.match(html, /complete · 3ms · 2 events/);
    assert.match(html, /Tool call: second_tool/);
    assert.match(html, /error · denied · 2 events/);
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
    assert.equal((contentDiv.innerHTML.match(/tool-activity-call/g) || []).length, 2);
    assert.match(contentDiv.innerHTML, /Calling memory_agency_feature/);
    assert.match(contentDiv.innerHTML, /class="response-content"/);
    assert.match(contentDiv.innerHTML, /I tried to save it\./);
});

test('finalizeStreamingMessage keeps pre-tool prose outside tool calls', async () => {
    const msgDiv = makeNode();
    const contentDiv = makeNode();
    contentDiv.className = 'message-content streaming';
    msgDiv.appendChild(contentDiv);

    await finalizeStreamingMessage(
        msgDiv,
        'I will check that now.\n\u{1F527} Calling lookup...\n\u2713 lookup complete (7ms)\n---\nThe lookup finished.',
    );

    assert.match(contentDiv.innerHTML, /response-prelude/);
    assert.match(contentDiv.innerHTML, /I will check that now\./);
    assert.match(contentDiv.innerHTML, /tool-activity-container/);
    assert.match(contentDiv.innerHTML, /Tool call: lookup/);
    assert.match(contentDiv.innerHTML, /class="response-content"/);
    assert.match(contentDiv.innerHTML, /The lookup finished\./);
    // Prelude prose precedes the tool card, response follows it.
    assert.ok(
        contentDiv.innerHTML.indexOf('I will check that now') <
            contentDiv.innerHTML.indexOf('tool-activity-container'),
        'pre-tool prose must render before the tool card',
    );
});


test('segmentToolActivity leaves a tool-free message (with markdown rule) whole', () => {
    const content = 'Intro paragraph\n---\nDetails after a normal markdown rule';
    const segs = segmentToolActivity(content);

    assert.equal(toolSegments(segs).length, 0, 'no tool activity without a \u{1F527} start');
    assert.equal(segs.length, 1);
    assert.equal(segs[0].kind, 'prose');
    // The markdown horizontal rule is preserved verbatim — only the wire
    // delimiter adjacent to a tools block is stripped.
    assert.equal(segs[0].text, content);
});

test('segmentToolActivity keeps trailing non-marker text as a prose block', () => {
    const segs = segmentToolActivity([
        '\u{1F527} Calling search_memory...',
        '✓ search_memory complete (11ms)',
        'Error: Maximum tool call iterations exceeded',
    ].join('\n'));

    const tools = toolSegments(segs);
    assert.equal(tools.length, 1);
    assert.equal(
        tools[0].text,
        '\u{1F527} Calling search_memory...\n✓ search_memory complete (11ms)',
    );
    assert.equal(proseText(segs), 'Error: Maximum tool call iterations exceeded');
});

test('segmentToolActivity keeps prose before the first tool call ahead of the card', () => {
    const segs = segmentToolActivity([
        'I will check that now.',
        '\u{1F527} Calling lookup...',
        '✓ lookup complete (7ms)',
        '---',
        'The lookup finished.',
    ].join('\n'));

    assert.deepEqual(segs.map((s) => s.kind), ['prose', 'tools', 'prose']);
    assert.equal(segs[0].text, 'I will check that now.');
    assert.equal(segs[1].text, '\u{1F527} Calling lookup...\n✓ lookup complete (7ms)');
    assert.equal(segs[2].text, 'The lookup finished.');
});

test('segmentToolActivity does not card ordinary checkmark replies', () => {
    const content = '✓ Done - here is the summary.';
    const segs = segmentToolActivity(content);

    assert.equal(toolSegments(segs).length, 0);
    assert.equal(proseText(segs), content);
});

test('segmentToolActivity requires a start marker before done/error markers count', () => {
    for (const content of [
        '✓ Migration complete',
        '❌ Build failed: missing dependency',
    ]) {
        const segs = segmentToolActivity(content);
        assert.equal(toolSegments(segs).length, 0, content);
        assert.equal(proseText(segs), content);
    }
});

test('segmentToolActivity recognizes a tool start glued onto the end of prose', () => {
    // Server emitters yield the start marker with only a trailing newline;
    // the LLM's last text chunk often lacks one, so the buffer glues prose
    // and marker onto a single source line.
    const segs = segmentToolActivity(
        'I will check that now.\u{1F527} Calling lookup...\n'
        + '✓ lookup complete (7ms)\n'
        + '---\n'
        + 'The lookup finished.',
    );

    assert.deepEqual(segs.map((s) => s.kind), ['prose', 'tools', 'prose']);
    assert.equal(segs[0].text, 'I will check that now.');
    assert.equal(segs[1].text, '\u{1F527} Calling lookup...\n✓ lookup complete (7ms)');
    assert.equal(segs[2].text, 'The lookup finished.');
});

test('segmentToolActivity leaves marker-shaped prose untouched when no tool start exists', () => {
    for (const content of [
        'Done: ✓ migration complete and ready to ship',
        'Heads up❌ build failed: missing dependency',
        'Status report: ✓ phase 1 complete, ✓ phase 2 complete',
    ]) {
        const segs = segmentToolActivity(content);
        assert.equal(toolSegments(segs).length, 0, content);
        assert.equal(proseText(segs), content);
    }
});

test('segmentToolActivity recovers a completion marker glued onto a start marker', () => {
    // A fast item collapses start+complete in one chunk with no newline.
    const segs = segmentToolActivity('\u{1F527} Calling lookup...✓ lookup complete');

    const tools = toolSegments(segs);
    assert.equal(tools.length, 1);
    assert.equal(tools[0].text, '\u{1F527} Calling lookup...\n✓ lookup complete');
});

test('segmentToolActivity cards EVERY iteration of a multi-batch turn (#1547 follow-up)', () => {
    // The screenshot bug: later tool runs glued onto prose
    // ("✓ github complete The merged fix is installed.") rendered as
    // inline text instead of cards. Each run must become its own card
    // group with surrounding prose split out — no newlines required.
    const content =
        'Picking up the loop.'
        + '\u{1F527} Calling memory_feature... ✓ memory_feature complete '
        + '\u{1F527} Calling github... ✓ github complete '
        + 'The merged fix is installed, but the gate still fails.'
        + '\u{1F527} Calling talon... ✓ talon complete '
        + 'Dispatched Talon.';
    const segs = segmentToolActivity(content);

    assert.deepEqual(segs.map((s) => s.kind), ['prose', 'tools', 'prose', 'tools', 'prose']);
    assert.equal(segs[0].text, 'Picking up the loop.');
    assert.equal(segs[2].text, 'The merged fix is installed, but the gate still fails.');
    assert.equal(segs[4].text, 'Dispatched Talon.');
    assert.match(segs[1].text, /memory_feature complete/);
    assert.match(segs[1].text, /github complete/);
});

test('segmentToolActivity bounds an error detail glued onto the next marker', () => {
    // Codex review: `❌ X failed: <detail>` must not swallow a following
    // glued marker into its detail. Each run stays a distinct token and
    // trailing prose stays prose.
    const segs = segmentToolActivity(
        '\u{1F527} Calling search...❌ search failed: timeout'
        + '\u{1F527} Calling fallback...✓ fallback complete Final answer.',
    );

    assert.deepEqual(segs.map((s) => s.kind), ['tools', 'prose']);
    // Both runs coalesce into one card group (no prose between them)...
    assert.match(segs[0].text, /search failed: timeout/);
    assert.match(segs[0].text, /Calling fallback\.\.\./);
    assert.match(segs[0].text, /fallback complete/);
    // ...and the error detail did NOT eat the fallback start marker.
    assert.ok(
        !/timeout\u{1F527}/u.test(segs[0].text),
        'error detail must terminate at the next marker, got: ' + JSON.stringify(segs[0].text),
    );
    assert.equal(segs[1].text, 'Final answer.');
});

test('segmentToolActivity preserves a post-tool indented code block (codex review)', () => {
    // Blanket-trimming prose stripped the leading 4 spaces of a markdown
    // code block that followed a tool call, so it stopped rendering as
    // code. Indentation that begins on a NEW line must survive; only the
    // inline separator space and the wire delimiter are removed.
    const segs = segmentToolActivity(
        '\u{1F527} Calling lookup...✓ lookup complete\n---\n    indented code\nnext',
    );

    assert.deepEqual(segs.map((s) => s.kind), ['tools', 'prose']);
    assert.equal(segs[1].text, '    indented code\nnext');
});

test('segmentToolActivity strips the inline separator but keeps later indentation', () => {
    // "✓ x complete The answer" → inline space dropped; but a fenced/
    // indented block after a newline keeps its leading spaces.
    const segs = segmentToolActivity(
        '\u{1F527} Calling lookup...✓ lookup complete The answer is:\n    code line',
    );
    assert.deepEqual(segs.map((s) => s.kind), ['tools', 'prose']);
    assert.equal(segs[1].text, 'The answer is:\n    code line');
});
