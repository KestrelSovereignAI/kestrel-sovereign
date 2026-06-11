import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { JSDOM } from 'jsdom';
import createDOMPurify from 'dompurify';

// #1661 KaTeX: marked escapes `\(`/`\[` (→ `(`/`[`) and HTML-escapes `<`/`&`,
// which would destroy the math delimiters before the KaTeX post-pass sees
// them. renderMarkdown protects math spans from marked and restores them
// HTML-escaped. These tests run the REAL sanitizer (jsdom DOMPurify) with a
// `marked` stub that mimics the escaping that breaks math.

const here = dirname(fileURLToPath(import.meta.url));
const parseSrc = readFileSync(
    resolve(here, '../../kestrel_sovereign/static/shared/markdown/parse.js'),
    'utf8',
);

function load() {
    const { window } = new JSDOM('');
    const DOMPurify = createDOMPurify(window);
    const marked = {
        use() {},
        parse(md) {
            // Mimic marked: strip a backslash before ASCII punctuation (its
            // escape handling) then HTML-escape, then wrap in a paragraph.
            let s = md.replace(/\\([\\(){}\[\]!#.\-_>*+`~])/g, '$1');
            s = s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            return `<p>${s}</p>`;
        },
    };
    const factory = new Function(
        'marked', 'DOMPurify',
        `${parseSrc}\nreturn { renderMarkdown };`,
    );
    return factory(marked, DOMPurify);
}

test('inline \\(…\\) math delimiters survive marked escaping', () => {
    const { renderMarkdown } = load();
    const html = renderMarkdown('the value \\(x^2 + 1\\) here');
    // Backslash-paren delimiters preserved so KaTeX's post-pass can find them.
    assert.match(html, /\\\(x\^2 \+ 1\\\)/);
});

test('display \\[…\\] math survives too', () => {
    const { renderMarkdown } = load();
    const html = renderMarkdown('block \\[a+b\\] end');
    assert.match(html, /\\\[a\+b\\\]/);
});

test('$$…$$ math with < is HTML-escaped (innerHTML-safe), not dropped', () => {
    const { renderMarkdown } = load();
    const html = renderMarkdown('eq $$a < b$$ done');
    // The < inside math is escaped to &lt; (so innerHTML is safe); KaTeX reads
    // the real `<` back from textContent.
    assert.match(html, /\$\$a &lt; b\$\$/);
    assert.doesNotMatch(html, /\$\$a < b\$\$/);
});

test('unbalanced \\( (no closing) is NOT protected — marked escapes it', () => {
    const { renderMarkdown } = load();
    const html = renderMarkdown('just a paren \\(b with no close');
    assert.doesNotMatch(html, /\\\(/);
});

test('ordinary prose with no math is unaffected', () => {
    const { renderMarkdown } = load();
    const html = renderMarkdown('plain answer, costs $5 and $10');
    assert.match(html, /costs \$5 and \$10/);
    assert.doesNotMatch(html, /\uE000/);
});
