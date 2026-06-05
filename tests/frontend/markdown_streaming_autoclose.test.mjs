import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

// #1547: renderStreamingMarkdown used to count `*`/`**`/`` ` `` with naive
// regexes and append synthetic closers mid-stream. That mis-fired on list
// bullets ("* item"), multiplication, and stray asterisks — wrapping a
// synthetic delimiter around a large span so the whole bubble flipped
// bold/italic for a frame and then reverted. These tests pin the new
// contract: ONLY an unterminated fenced code block (```) is auto-closed;
// inline emphasis/code is passed through verbatim and resolves naturally
// once the real closer streams in.

const here = dirname(fileURLToPath(import.meta.url));
const parseSrc = readFileSync(
    resolve(here, '../../kestrel_sovereign/static/shared/markdown/parse.js'),
    'utf8',
);

// A stub that records exactly what string parse.js hands to marked.parse,
// so we can assert on the pre-processing (synthetic closers) directly.
function loadParseModule() {
    const seen = { lastParsed: null };
    const marked = {
        use() {},
        parse(md) { seen.lastParsed = md; return md; },
    };
    const factory = new Function(
        'marked',
        `${parseSrc}\nreturn { renderStreamingMarkdown };`,
    );
    return { ...factory(marked), seen };
}

test('unbalanced ** is NOT auto-closed (no synthetic bold)', () => {
    const { renderStreamingMarkdown, seen } = loadParseModule();
    renderStreamingMarkdown('Here is **important');
    assert.equal(seen.lastParsed, 'Here is **important',
        'streaming render must not append a synthetic ** closer');
});

test('a growing bullet list (odd * count) is NOT italicized', () => {
    const { renderStreamingMarkdown, seen } = loadParseModule();
    // Three bullets → odd number of leading `*` — the old code appended a
    // synthetic `*` and italicized everything after the first bullet.
    const list = '* one\n* two\n* three';
    renderStreamingMarkdown(list);
    assert.equal(seen.lastParsed, list,
        'list bullets must not be treated as an unclosed italic span');
});

test('a stray single * is NOT auto-closed', () => {
    const { renderStreamingMarkdown, seen } = loadParseModule();
    renderStreamingMarkdown('2 * 3 = 6 and counting');
    assert.equal(seen.lastParsed, '2 * 3 = 6 and counting',
        'a multiplication asterisk must not synthesize an italic closer');
});

test('unterminated inline code is NOT auto-closed', () => {
    const { renderStreamingMarkdown, seen } = loadParseModule();
    renderStreamingMarkdown('run `git status');
    assert.equal(seen.lastParsed, 'run `git status',
        'inline code is rendered literally until its real closer arrives');
});

test('an unterminated fenced code block IS auto-closed', () => {
    const { renderStreamingMarkdown, seen } = loadParseModule();
    renderStreamingMarkdown('```python\nprint(1)');
    assert.equal(seen.lastParsed, '```python\nprint(1)\n```',
        'a fence is a block construct and must be bounded so the rest of '
        + 'the bubble does not become one giant code block');
});

test('a balanced fenced code block is left untouched', () => {
    const { renderStreamingMarkdown, seen } = loadParseModule();
    const md = '```js\nconst x = 1;\n```';
    renderStreamingMarkdown(md);
    assert.equal(seen.lastParsed, md, 'balanced fence needs no synthetic closer');
});
