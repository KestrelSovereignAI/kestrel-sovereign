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

test('a paragraph before a list IS a boundary (no bolding across blocks)', () => {
    const { _splitStreamingTail } = load();
    // The blank after a paragraph is a real boundary even though a list
    // follows, so tail completion can only touch the list, not the paragraph.
    const { stable, tail } = _splitStreamingTail('Intro **bold\n\n- item');
    assert.match(stable, /Intro \*\*bold/);
    assert.equal(tail, '- item');
});

test('an indented loose-list continuation stays with the list', () => {
    const { _splitStreamingTail } = load();
    // "  more" continues the list item across the blank line — must not split.
    const { stable, tail } = _splitStreamingTail('- item\n\n  more text');
    assert.equal(stable, '');
    assert.match(tail, /- item\n\n  more text/);
});

test('a ```js info-string line inside a ``` block does not close it', () => {
    const { _completeStreamingInline, _splitStreamingTail } = load();
    assert.equal(_completeStreamingInline('```\n```js\nx'), '```\n```js\nx\n```');
    const { tail } = _splitStreamingTail('```\n```js\n\nx');
    assert.match(tail, /```\n```js\n\nx/);  // info-string line + blank don't split
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

test('respects fence delimiter length (```` is not closed by ```)', () => {
    const { _completeStreamingInline, _splitStreamingTail } = load();
    assert.equal(_completeStreamingInline('````\n```\ncode'), '````\n```\ncode\n````');
    const { tail } = _splitStreamingTail('````\n```\n\ncode');
    assert.match(tail, /````\n```\n\ncode/);  // inner ``` + blank line don't split
});

test('does NOT count ** delimiters that live inside an inline code span', () => {
    const { _completeStreamingInline } = load();
    assert.equal(_completeStreamingInline('Use `**` for bold'), 'Use `**` for bold');
    // an open inline-code span swallows the ** — close the span, not add bold
    assert.equal(_completeStreamingInline('text `code with **'), 'text `code with **`');
});

test('does NOT count delimiters inside a CLOSED fenced code block', () => {
    const { _completeStreamingInline } = load();
    // `**/*.py` and a lone backtick inside the fence must not draw a synthetic
    // closer after the fence.
    assert.equal(
        _completeStreamingInline('```txt\n**/*.py\n```'),
        '```txt\n**/*.py\n```');
    // a real unclosed ** AFTER a closed fence still completes
    assert.equal(
        _completeStreamingInline('```\nx **y** z\n```\nthen **open'),
        '```\nx **y** z\n```\nthen **open**');
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

test('does not bold a literal ** glob/operator (only genuine emphasis)', () => {
    const { _completeStreamingInline } = load();
    assert.equal(_completeStreamingInline('Use **/*.py to match'), 'Use **/*.py to match');
    assert.equal(_completeStreamingInline('a trailing **'), 'a trailing **');
    assert.equal(_completeStreamingInline('really **important'), 'really **important**');
});

test('balanced multi-backtick code span is not treated as unclosed', () => {
    const { _completeStreamingInline } = load();
    // `` ` `` is a valid 2-backtick span containing a literal backtick.
    assert.equal(_completeStreamingInline('Use `` ` `` here'), 'Use `` ` `` here');
    // a genuinely open span still closes
    assert.equal(_completeStreamingInline('run `git'), 'run `git`');
    // a balanced code span crossing a newline is not treated as unclosed
    assert.equal(_completeStreamingInline('Use `foo\nbar` now'), 'Use `foo\nbar` now');
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

test('reference-style link content is parsed whole (not split), so refs resolve', () => {
    const mod = load();
    mod.renderStreamingMarkdown('[docs]: https://example.com\n\nsee [docs]');
    // One whole-content parse (the definition + reference together), not two
    // independent block parses that would lose the document-global definition.
    const refParses = mod._markedCalls.filter((s) => s.includes('[docs]:'));
    assert.equal(refParses.length, 1);
    assert.match(refParses[0], /see \[docs\]/);  // ref + def in the SAME parse
});

test('renderStreamingMarkdown concatenates stable + completed tail', () => {
    const mod = load();
    const html = mod.renderStreamingMarkdown('intro para\n\nstart **bo');
    assert.match(html, /intro para/);
    assert.match(html, /start \*\*bo\*\*/);  // tail bold completed
});

test('streaming math delimiters survive marked (protected like finalize)', () => {
    // A marked stub that mimics marked eating \( \) [ ] backslashes — without
    // _protectMath the delimiters would be destroyed mid-stream and the render
    // would disagree with finalize.
    const marked = {
        use() {},
        parse(md) { return `<p>${md.replace(/\\([()\[\]])/g, '$1')}</p>`; },
    };
    const DOMPurify = {
        sanitize: (h) => h, addHook() {}, removeHook() {}, removeAllHooks() {},
    };
    const factory = new Function(
        'marked', 'DOMPurify',
        `${parseSrc}\nreturn { renderStreamingMarkdown };`,
    );
    const { renderStreamingMarkdown } = factory(marked, DOMPurify);
    const html = renderStreamingMarkdown('see \\(x + y');
    assert.match(html, /\\\(x \+ y\\\)/);  // completed AND protected
});
