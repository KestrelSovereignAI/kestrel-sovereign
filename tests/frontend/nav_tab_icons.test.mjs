// #2821: nav tab icon coverage, and the guard that keeps it honest.
//
// The bug this file exists to prevent: the Approvals tab shipped declaring
// `ki ki-check`, which is not a glyph in icons.js (the set has `checkmark`,
// `check-box`, `check-square`, `check-circle` — no `check`). icons.js emits one
// `.ki-<name>{mask-image:…}` rule per KNOWN name, so an unknown name got no
// rule and kept only the base `.ki` declaration — `background: currentColor`
// with no mask, i.e. a solid 1em block. It rendered as a filled square next to
// "Approvals" and read as an intentional design element rather than a typo.
//
// Two layers are asserted here:
//   1. Every `ki-<name>` class token written anywhere in the shipped HTML/JS
//      resolves to a real glyph. This is the check that would have caught it.
//   2. The base `.ki` rule carries a fallback mask, so if a future unknown name
//      slips past layer 1 it renders a visible "?" rather than a solid block —
//      wrong in a way that looks wrong.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { JSDOM } from 'jsdom';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PKG = path.resolve(HERE, '../../kestrel_sovereign');

const dom = new JSDOM('<!doctype html><html><head></head><body></body></html>', {
    url: 'http://localhost/',
});
globalThis.window = dom.window;
globalThis.document = dom.window.document;

// icons.js is a plain <script> IIFE (no exports); importing it executes the
// IIFE, which installs window.kicon/KI_PATHS and injects the mask-image CSS.
await import('../../kestrel_sovereign/static/js/icons.js');

const { CORE_PANEL_DEFS } = await import(
    '../../kestrel_sovereign/static/js/ui-ext/core-panels.js'
);

// ---- source scan ------------------------------------------------------------

function* walk(dir) {
    for (const entry of readdirSync(dir)) {
        if (entry === 'node_modules' || entry.startsWith('.')) continue;
        const full = path.join(dir, entry);
        if (statSync(full).isDirectory()) {
            yield* walk(full);
        } else if (/\.(html|js|mjs)$/.test(entry)) {
            yield full;
        }
    }
}

// Only class ATTRIBUTE values, so prose in a comment that happens to name an
// icon is not mistaken for a usage. A `className = ` template literal with an
// interpolated icon (the registry's tab builder) cannot be resolved statically
// — those are covered by the descriptor assertions further down, which check
// the values that actually feed it.
const CLASS_ATTR = /class\s*=\s*["']([^"']*)["']/g;
const KICON_CALL = /\bkicon\(\s*['"]([a-zA-Z0-9_-]+)['"]/g;

function iconUsages() {
    const found = [];
    for (const file of walk(PKG)) {
        const text = readFileSync(file, 'utf8');
        for (const m of text.matchAll(CLASS_ATTR)) {
            for (const token of m[1].split(/\s+/)) {
                if (token.startsWith('ki-')) {
                    found.push({ file, name: token.slice(3), how: `class="${m[1]}"` });
                }
            }
        }
        for (const m of text.matchAll(KICON_CALL)) {
            found.push({ file, name: m[1], how: `kicon('${m[1]}')` });
        }
    }
    return found;
}

test('every ki-<name> class in the shipped console resolves to a real glyph', () => {
    const usages = iconUsages();
    // Sanity: the scan must actually be finding things, or it guards nothing.
    assert.ok(usages.length > 30, `expected many icon usages, found ${usages.length}`);

    const unknown = usages.filter((u) => !window.KI_PATHS[u.name]);
    const detail = unknown
        .map((u) => `  ${path.relative(PKG, u.file)}: ${u.how} → no such glyph "${u.name}"`)
        .join('\n');
    assert.equal(
        unknown.length, 0,
        `unknown icon name(s) — these render as the "?" fallback, not an icon:\n${detail}`,
    );
});

test('the base .ki rule carries a fallback mask so an unknown name is not a solid block', () => {
    const style = document.getElementById('kestrel-icons');
    assert.ok(style, 'icons.js must inject its <style>');
    const base = style.textContent.match(/(?:^|\})\.ki\{([^{}]*)\}/);
    // `[^{}]*` plus the explicit checks below keep this honest: a malformed
    // (unclosed) base rule must fail here rather than quietly matching across
    // into the first `.ki-<name>` rule and borrowing ITS mask-image.
    assert.ok(base, 'a well-formed base .ki rule must exist');
    assert.doesNotMatch(base[1], /\.ki-/, 'the base rule must not run into the per-glyph rules');
    assert.match(
        base[1], /(^|;)mask-image:url\("data:image\/svg\+xml,/,
        'the base .ki rule must declare its own mask-image; without one, '
        + 'background:currentColor paints the full 1em box as a solid square',
    );
});

test('an unknown ki-<name> gets no rule of its own, so it inherits the fallback', () => {
    const style = document.getElementById('kestrel-icons');
    // `check` is the exact name that caused the Approvals square. It must still
    // be absent — the fix was to correct the call site, not to add the glyph.
    assert.equal(window.KI_PATHS['check'], undefined, 'no `check` glyph should exist');
    assert.doesNotMatch(style.textContent, /\.ki-check\{/, 'no .ki-check rule should be emitted');
});

// ---- nav tab coverage -------------------------------------------------------

test('every core panel descriptor declares an icon, and it is a real glyph', () => {
    for (const def of CORE_PANEL_DEFS) {
        assert.ok(def.icon, `core panel "${def.panelId}" must declare an icon`);
        assert.match(
            def.icon, /^ki ki-[a-z0-9-]+$/,
            `core panel "${def.panelId}" icon must be "ki ki-<name>", got "${def.icon}"`,
        );
        const name = def.icon.replace(/^ki ki-/, '');
        assert.ok(
            window.KI_PATHS[name],
            `core panel "${def.panelId}" declares unknown glyph "${name}"`,
        );
    }
});

test('every nav tab in the standalone console carries an icon span', () => {
    const html = readFileSync(path.join(PKG, 'static/index.html'), 'utf8');
    const page = new JSDOM(html).window.document;
    const tabs = page.querySelectorAll('.nav-tabs .nav-tab');
    assert.ok(tabs.length >= 10, `expected the full tab strip, found ${tabs.length}`);
    for (const tab of tabs) {
        const icon = tab.querySelector('.nav-tab-icon');
        const panelId = tab.dataset.panel;
        assert.ok(icon, `tab "${panelId}" must have a .nav-tab-icon span`);
        assert.ok(
            /\bki-[a-z0-9-]+\b/.test(icon.className),
            `tab "${panelId}" icon span must carry a ki-<name> class, got "${icon.className}"`,
        );
        assert.equal(
            icon.getAttribute('aria-hidden'), 'true',
            `tab "${panelId}" icon is decorative and must be aria-hidden`,
        );
        assert.ok(
            tab.querySelector('.nav-tab-label'),
            `tab "${panelId}" must have a .nav-tab-label span`,
        );
    }
});

test('the standalone Approvals tab no longer declares the phantom ki-check', () => {
    const html = readFileSync(path.join(PKG, 'static/index.html'), 'utf8');
    assert.doesNotMatch(html, /ki ki-check(?![a-z-])/, 'ki-check is not a glyph');
});
