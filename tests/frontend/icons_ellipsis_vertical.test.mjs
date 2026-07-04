// #2159: kebab_menu.js requests kicon('ellipsis-vertical') for the ⋯ button,
// but the name was missing from icons.js PATHS, so the button rendered the
// unknown-icon fallback (a solid box). Guard the icon's presence.

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

test('ellipsis-vertical is a known icon', () => {
    assert.ok(window.KI_PATHS['ellipsis-vertical'], 'PATHS must contain ellipsis-vertical');
    assert.ok(
        window.KI_NAMES.includes('ellipsis-vertical'),
        'KI_NAMES must list ellipsis-vertical',
    );
});

test('kicon("ellipsis-vertical") returns a real .ki.ki-ellipsis-vertical span, not the fallback', () => {
    const html = window.kicon('ellipsis-vertical');
    assert.match(html, /class="ki ki-ellipsis-vertical"/);
    // The unknown-icon fallback renders `<span class="ki" title="...">?</span>`
    // with no ki-<name> modifier and a literal "?" — make sure we're not that.
    assert.doesNotMatch(html, />\?</, 'must not be the unknown-icon "?" fallback');
});
