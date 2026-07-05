// #2199: the conversations-pane chat-header trigger requests kicon('history')
// (a rollback-clock glyph — NOT ki-scroll). Guard the icon's presence so the
// button renders a real glyph instead of the unknown-icon "?" fallback.

import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const dom = new JSDOM('<!doctype html><html><head></head><body></body></html>', {
    url: 'http://localhost/',
});
globalThis.window = dom.window;
globalThis.document = dom.window.document;

// icons.js is a plain <script> IIFE (no exports); importing it executes the
// IIFE, which installs window.kicon and injects the mask-image CSS.
await import('../../kestrel_sovereign/static/js/icons.js');

test('history is a known icon', () => {
    assert.ok(window.KI_PATHS['history'], 'PATHS must contain history');
    assert.ok(window.KI_NAMES.includes('history'), 'KI_NAMES must list history');
});

test('history renders a real, non-empty SVG glyph (not the "?" fallback)', () => {
    const inner = window.KI_PATHS['history'];
    // A rollback clock: at least a couple of <path> segments (the arc arrow +
    // the clock hands), not an empty/placeholder string.
    assert.ok(/<path/.test(inner), 'history glyph must contain path geometry');
    assert.ok(inner.length > 20, 'history glyph must be non-trivial');
});

test('kicon("history") returns a real .ki.ki-history span, not the fallback', () => {
    const html = window.kicon('history');
    assert.match(html, /class="ki ki-history"/);
    assert.doesNotMatch(html, />\?</, 'must not be the unknown-icon "?" fallback');
});
