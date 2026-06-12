// #1660 block-based streaming markdown: fence-aware stable/tail split,
// tail-scoped complete-all, and stable-prefix memoization.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const parseSrc = readFileSync(
    resolve(here, '../../kestrel_sovereign/static/shared/markdown/parse.js'),
    'utf8',
);

// Load parse.js with a marked stub (wraps input in <p>, records calls) and a
// passthrough DOMPurify stub. The split/complete helpers are pure strings and
// need neither.
function load() {
    const calls = [];
    const marked = {
        use() {},
        parse(md) { calls.push(md); return `<p>${md}</p>`; },
    };
    const DOMPurify = {
        sanitize: (h) => h, addHook() {}, removeHook() {}, removeAllHooks() {},
    };
    const factory = new Function(
        'marked', 'DOMPurify',
        `${parseSrc}\nreturn { renderStreamingMarkdown, _splitStreamingTail, _completeStreamingInline };`,
    );
    const mod = factory(marked, DOMPurify);
    mod._markedCalls = calls;
    return mod;
}

// --- fence-aware split ------------------------------------------------------

test('split puts the last block in the tail, earlier blocks in stable', () => {
    const { _splitStreamingTail } = load();
    const { stable, tail } = _splitStreamingTail('para one\n\npara two');
    assert.equal(tail, 'para two');
    assert.match(stable, /para one/);
    assert.doesNotMatch(stable, /para two/);
});

test('an OPEN fence keeps its whole region in the tail', () => {
    const { _splitStreamingTail } = load();
    const { stable, tail } = _splitStreamingTail('intro\n\n```js\nconst x = 1;');
    assert.equal(stable.includes('```'), false);
    assert.match(tail, /```js\nconst x = 1;/);
});

test('a blank line INSIDE a closed fence does not split the block', () => {
    const { _splitStreamingTail } = load();
    const { stable, tail } = _splitStreamingTail('```\na\n\nb\n```\n\nafter');
    assert.match(stable, /```\na\n\nb\n```/);  // whole fence stays together
    assert.equal(tail, 'after');
});

test('a loose list (blank line between items) is NOT split into two lists', () => {
    const { _splitStreamingTail } = load();
    const { stable, tail } = _splitStreamingTail('- first\n\n- second');
    // The whole loose list stays together (in the tail) — not finalized as a
    // standalone <ul> before the second item arrives.
    assert.equal(stable, '');
    assert.match(tail, /- first\n\n- second/);
});

test('a non-list block after a loose list still finalizes', () => {
    const { _splitStreamingTail } = load();
    const { stable, tail } = _splitStreamingTail('- a\n\n- b\n\nparagraph');
    assert.match(stable, /- a\n\n- b/);   // the loose list memoizes together
    assert.equal(tail, 'paragraph');
});

test('a ``` fence containing a ~~~ line stays open (marker-aware)', () => {
    const { _splitStreamingTail, _completeStreamingInline } = load();
    const { tail } = _splitStreamingTail('```\n~~~\n\nstill code');
    assert.match(tail, /```\n~~~\n\nstill code/);  // blank inside the fence didn't split
    assert.equal(_completeStreamingInline('```\n~~~\ncode'), '```\n~~~\ncode\n```');
});

// --- tail-scoped complete-all -----------------------------------------------

test('completes unclosed bold, inline code, link, and math', () => {
    const { _completeStreamingInline } = load();
    assert.equal(_completeStreamingInline('**bold'), '**bold**');
    assert.equal(_completeStreamingInline('`code'), '`code`');
    assert.equal(_completeStreamingInline('[text](http://x'), '[text](http://x)');
    assert.equal(_completeStreamingInline('$$x = y'), '$$x = y$$');
    assert.equal(_completeStreamingInline('\\(a + b'), '\\(a + b\\)');
});

test('closes an open fenced code block (and leaves inline alone inside it)', () => {
    const { _completeStreamingInline } = load();
    assert.equal(_completeStreamingInline('```js\nconst x = 1;'), '```js\nconst x = 1;\n```');
});

test('closes a TILDE fence with a tilde marker, not backticks', () => {
    const { _completeStreamingInline } = load();
    assert.equal(_completeStreamingInline('~~~\ncode'), '~~~\ncode\n~~~');
});

test('does NOT complete ambiguous single-* or bare $ (the #1547 trap)', () => {
    const { _completeStreamingInline } = load();
    assert.equal(_completeStreamingInline('* a list item'), '* a list item');
    assert.equal(_completeStreamingInline('5 * 3 = 15'), '5 * 3 = 15');
    assert.equal(_completeStreamingInline('it costs $5 today'), 'it costs $5 today');
});

test('leaves already-balanced / plain text untouched', () => {
    const { _completeStreamingInline } = load();
    assert.equal(_completeStreamingInline('hello world'), 'hello world');
    assert.equal(_completeStreamingInline('**done** and `ok`'), '**done** and `ok`');
});

// --- stable-prefix memoization ----------------------------------------------

test('the stable prefix is parsed once across growing chunks', () => {
    const mod = load();
    mod.renderStreamingMarkdown('para one\n\npar');
    mod.renderStreamingMarkdown('para one\n\npara t');
    mod.renderStreamingMarkdown('para one\n\npara two');
    // The stable prefix ("para one\n") must be handed to marked.parse exactly
    // once — the three calls only re-parse the growing tail.
    const stableParses = mod._markedCalls.filter((s) => s.includes('para one') && !s.includes('para t'));
    assert.equal(stableParses.length, 1);
});

test('renderStreamingMarkdown concatenates stable + completed tail', () => {
    const mod = load();
    const html = mod.renderStreamingMarkdown('intro para\n\nstart **bo');
    assert.match(html, /intro para/);
    assert.match(html, /start \*\*bo\*\*/);  // tail bold completed
});
