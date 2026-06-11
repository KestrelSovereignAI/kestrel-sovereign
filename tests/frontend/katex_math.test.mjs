import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { JSDOM } from 'jsdom';

// katex.js is a classic script declaring its functions at top level. Eval the
// source in a sandbox backed by a real (jsdom) DOM, appending a return for the
// internals we assert on — the same pattern mermaid_cache.test.mjs uses.
// window.renderMathInElement is the stubbed KaTeX auto-render fn (or the
// _katexLoader.import seam supplies it) so no CDN fetch happens.

const here = dirname(fileURLToPath(import.meta.url));
const src = readFileSync(
    resolve(here, '../../kestrel_sovereign/static/shared/markdown/katex.js'),
    'utf8',
);

function loadKatex(win) {
    const factory = new Function(
        'window', 'document', 'console',
        `${src}\nreturn { renderMath, ensureKatex, _katexLoader, _hasMath };`,
    );
    return factory(win, win.document, { warn() {}, log() {}, error() {} });
}

function freshDom() {
    return new JSDOM('<!doctype html><head></head><body></body>').window;
}

function el(win, html) {
    const node = win.document.createElement('div');
    node.innerHTML = html;
    return node;
}

test('renderMath skips (no lazy-load) when the element has no math', async () => {
    const win = freshDom();
    const mod = loadKatex(win);
    let imported = false;
    mod._katexLoader.import = async () => { imported = true; return { default() {} }; };
    await mod.renderMath(el(win, 'just plain text, no dollars'));
    assert.equal(imported, false, 'must not lazy-load KaTeX when there is no math');
});

test('renderMath calls KaTeX auto-render with safe delimiters + ignoredTags', async () => {
    const win = freshDom();
    let opts = null;
    win.renderMathInElement = (_node, o) => { opts = o; };
    const mod = loadKatex(win);
    await mod.renderMath(el(win, 'mass-energy $$E = mc^2$$ here'));
    assert.ok(opts, 'renderMathInElement was called');
    assert.equal(opts.throwOnError, false);
    // code/pre keep shell `$` out of math; svg keeps mermaid diagrams intact.
    assert.ok(['code', 'pre', 'svg'].every((t) => opts.ignoredTags.includes(t)));
    const lefts = opts.delimiters.map((d) => d.left);
    assert.ok(lefts.includes('$$') && lefts.includes('\\(') && lefts.includes('\\['));
    // No bare single-$ delimiter (currency/shell-var safety).
    assert.ok(!lefts.includes('$'));
});

test('_hasMath recognizes $$, \\(, \\[ but NOT a bare single $', () => {
    const win = freshDom();
    const { _hasMath } = loadKatex(win);
    assert.equal(_hasMath('display $$x$$'), true);
    assert.equal(_hasMath('display \\[x\\]'), true);
    assert.equal(_hasMath('paren \\(x\\)'), true);
    // Currency / shell prose must not look like math.
    assert.equal(_hasMath('it costs $5 today and $10 tomorrow'), false);
    assert.equal(_hasMath('export $PATH'), false);
    assert.equal(_hasMath('no math here at all'), false);
});

test('ensureKatex reuses an already-present window.renderMathInElement', async () => {
    const win = freshDom();
    const stub = () => {};
    win.renderMathInElement = stub;
    const { ensureKatex } = loadKatex(win);
    assert.equal(await ensureKatex(), stub);
});

test('a slow first load still renders via deferred retry + injects CSS once', async () => {
    const win = freshDom();
    const mod = loadKatex(win);
    mod._katexLoader.maxWait = 5;
    let resolveImport;
    let calls = 0;
    mod._katexLoader.import = () => new Promise((res) => { resolveImport = res; });

    const node = el(win, 'the identity $$e^{i\\pi}+1=0$$');
    await mod.renderMath(node);
    assert.equal(calls, 0, 'not rendered before the import resolves');

    resolveImport({ default: () => { calls += 1; } });
    for (let i = 0; i < 6; i++) await new Promise((r) => setTimeout(r, 0));
    assert.equal(calls, 1, 'deferred retry rendered after the slow load');
    // CSS injected exactly once.
    assert.equal(win.document.querySelectorAll('link[data-katex-css]').length, 1);
});
