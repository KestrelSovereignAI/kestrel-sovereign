import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

// parse.js installs a custom marked renderer so every external link rendered
// in chat carries target="_blank" rel="noopener noreferrer". The two
// guarantees this test pins down:
//   1. external https://… links open in a new tab (target=_blank)
//   2. in-page #anchor links do NOT — keep their default same-view behavior

const here = dirname(fileURLToPath(import.meta.url));
const parseSrc = readFileSync(
    resolve(here, '../../kestrel_sovereign/static/shared/markdown/parse.js'),
    'utf8',
);

// Build a single sandbox per test so the install-once latch resets cleanly.
function loadParseModule() {
    // Minimal stub mimicking the bits of marked v11 that parse.js relies on:
    //   - marked.use({renderer: {link}}) registers a renderer
    //   - marked.parse(md) renders [text](href "title") via that renderer
    let installed = null;
    const marked = {
        use(config) {
            if (config?.renderer?.link) installed = config.renderer.link;
        },
        parse(md /* , opts */) {
            // Tiny regex parser — only intended to exercise our link renderer.
            return md.replace(
                /\[([^\]]+)\]\(([^)\s]+)(?:\s+"([^"]*)")?\)/g,
                (_match, text, href, title) => {
                    const token = { href, title, text };
                    return installed
                        ? installed.call({ parser: null }, token)
                        : `<a href="${href}">${text}</a>`;
                },
            );
        },
    };

    const factory = new Function(
        'marked',
        `${parseSrc}\nreturn { renderMarkdown, renderStreamingMarkdown };`,
    );
    return factory(marked);
}

test('external markdown links get target="_blank" rel="noopener noreferrer"', () => {
    const { renderMarkdown } = loadParseModule();
    const html = renderMarkdown('See [docs](https://example.com)');
    assert.match(html, /<a\s[^>]*href="https:\/\/example\.com"/);
    assert.match(html, /target="_blank"/);
    assert.match(html, /rel="noopener noreferrer"/);
});

test('external links preserve a title attribute', () => {
    const { renderMarkdown } = loadParseModule();
    const html = renderMarkdown('[click](https://example.com "Tooltip")');
    assert.match(html, /title="Tooltip"/);
    assert.match(html, /target="_blank"/);
});

test('in-page #anchor links keep default behavior (no target=_blank)', () => {
    const { renderMarkdown } = loadParseModule();
    const html = renderMarkdown('[jump](#section-2)');
    assert.match(html, /<a\s[^>]*href="#section-2"/);
    assert.doesNotMatch(html, /target="_blank"/);
    assert.doesNotMatch(html, /rel="noopener noreferrer"/);
});

test('streaming render path also applies the link renderer', () => {
    const { renderStreamingMarkdown } = loadParseModule();
    const html = renderStreamingMarkdown('see [x](https://example.com)');
    assert.match(html, /target="_blank"/);
    assert.match(html, /rel="noopener noreferrer"/);
});

// Capture the options each render path passes to marked.parse so the
// streaming and finalize paths can be compared. The streaming path used
// to omit options entirely (CommonMark default), collapsing single `\n`
// into a space — chat bubbles "scrunched" mid-stream and re-flowed only
// at finalize. The two paths must agree on `breaks: true` (and the
// other layout-affecting options) or the bubble's wire form will look
// different at stream-time vs. finalize.
function loadParseModuleCapturingOptions() {
    const calls = [];
    const marked = {
        use() {},
        parse(md, opts) {
            calls.push({ md, opts });
            return '';
        },
    };
    const factory = new Function(
        'marked',
        `${parseSrc}\nreturn { renderMarkdown, renderStreamingMarkdown };`,
    );
    return { mod: factory(marked), calls };
}

test('renderStreamingMarkdown passes the same options as renderMarkdown', () => {
    const { mod, calls } = loadParseModuleCapturingOptions();
    mod.renderMarkdown('line one\nline two');
    mod.renderStreamingMarkdown('line one\nline two');
    assert.equal(calls.length, 2);
    assert.deepEqual(calls[0].opts, calls[1].opts);
    assert.equal(calls[0].opts.breaks, true);
    assert.equal(calls[1].opts.breaks, true);
});
