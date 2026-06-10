import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { JSDOM } from 'jsdom';
import createDOMPurify from 'dompurify';

// parse.js routes every rendered-HTML string through DOMPurify.sanitize before
// it reaches innerHTML. This test exercises that path against a REAL DOMPurify
// (backed by jsdom), proving actual XSS removal rather than mocking it away.
//
// The `marked` stub is a passthrough: marked with default options does NOT
// escape raw HTML, so whatever HTML a malicious markdown document produces
// reaches our sanitizer verbatim — a passthrough stub is a faithful proxy for
// "marked emitted this HTML".

const here = dirname(fileURLToPath(import.meta.url));
const parseSrc = readFileSync(
    resolve(here, '../../kestrel_sovereign/static/shared/markdown/parse.js'),
    'utf8',
);

function makeMarked() {
    return { use() {}, parse(md) { return String(md); } };
}

// Evaluate parse.js in a sandbox. `DOMPurify` and `console` are injected as
// params so we can supply a real sanitizer and capture warnings. Each call
// builds a fresh instance, so module-level latches (_sanitizerWarned) reset.
function loadParse({ withPurify = true } = {}) {
    let DOMPurify;
    if (withPurify) {
        const { window } = new JSDOM('');
        DOMPurify = createDOMPurify(window);
    }
    const warnings = [];
    const fakeConsole = { warn: (...a) => warnings.push(a.map(String).join(' ')) };
    const factory = new Function(
        'marked', 'DOMPurify', 'console',
        `${parseSrc}\nreturn { renderMarkdown, renderStreamingMarkdown, sanitizeHtml };`,
    );
    const api = factory(makeMarked(), DOMPurify, fakeConsole);
    api.__warnings = warnings;
    return api;
}

test('strips <script> tags', () => {
    const { renderMarkdown } = loadParse();
    const html = renderMarkdown('<script>alert(1)</script>hello');
    assert.doesNotMatch(html, /<script/i);
    assert.match(html, /hello/);
});

test('strips inline event handlers (onerror)', () => {
    const { renderMarkdown } = loadParse();
    const html = renderMarkdown('<img src=x onerror=alert(1)>');
    assert.doesNotMatch(html, /onerror/i);
});

test('strips javascript: hrefs', () => {
    const { renderMarkdown } = loadParse();
    const html = renderMarkdown('<a href="javascript:alert(1)">x</a>');
    assert.doesNotMatch(html, /javascript:/i);
});

test('blocks data: image sources (allowDataImages:false)', () => {
    const { renderMarkdown } = loadParse();
    const html = renderMarkdown('<img src="data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=">');
    assert.doesNotMatch(html, /src="data:/i);
});

test('drops inline SVG — no <svg><image href="data:"> bypass', () => {
    const { renderMarkdown } = loadParse();
    const html = renderMarkdown('<svg width="1"><image href="data:image/png;base64,AAAA"></image></svg>');
    assert.doesNotMatch(html, /<svg/i);
    assert.doesNotMatch(html, /data:/i);
});

test('forces rel="noopener noreferrer" on target=_blank anchors', () => {
    const { renderMarkdown } = loadParse();
    const html = renderMarkdown('<a href="https://example.com" target="_blank">x</a>');
    assert.match(html, /href="https:\/\/example\.com"/);
    assert.match(html, /target="_blank"/);
    assert.match(html, /rel="noopener noreferrer"/);
});

test('enforces rel on case-variant target=_BLANK (keyword is case-insensitive)', () => {
    const { renderMarkdown } = loadParse();
    const html = renderMarkdown('<a href="https://example.com" target="_BLANK">x</a>');
    assert.match(html, /rel="noopener noreferrer"/);
});

test('preserves benign markdown HTML and code language classes', () => {
    const { renderMarkdown } = loadParse();
    const html = renderMarkdown('<p>hi <strong>there</strong></p><pre><code class="language-js">x</code></pre>');
    assert.match(html, /<p>hi <strong>there<\/strong><\/p>/);
    assert.match(html, /class="language-js"/);
});

test('streaming render path is also sanitized', () => {
    const { renderStreamingMarkdown } = loadParse();
    const html = renderStreamingMarkdown('<img src=x onerror=alert(1)>ok');
    assert.doesNotMatch(html, /onerror/i);
    assert.match(html, /ok/);
});

test('fails closed (escapes to inert text) + warns once when DOMPurify is absent', () => {
    const api = loadParse({ withPurify: false });
    const dirty = '<script>alert(1)</script>';
    const out = api.sanitizeHtml(dirty);
    // Fail closed: no live tag survives; it renders as visible escaped text.
    assert.doesNotMatch(out, /<script/);
    assert.match(out, /&lt;script&gt;/);
    api.sanitizeHtml(dirty);
    // Warns exactly once so a missing sanitizer is visible in prod.
    assert.equal(api.__warnings.length, 1);
    assert.match(api.__warnings[0], /DOMPurify not loaded/);
});
