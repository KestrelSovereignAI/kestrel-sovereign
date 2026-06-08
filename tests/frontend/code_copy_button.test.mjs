import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

// highlight.js (#1574) injects a hover-reveal "Copy" button onto every fenced
// `pre > code` block, in one place so all three render paths (streaming,
// finalize, history reload) get it for free. These tests pin down:
//   1. a button is added to each fenced block, scoped to pre > code
//   2. re-running the pass does NOT stack duplicate buttons (streaming safety)
//   3. inline `code` and tool-activity cards are left untouched
//   4. clicking copies the raw textContent of the <code>, not the label/markup

const here = dirname(fileURLToPath(import.meta.url));
const highlightSrc = readFileSync(
    resolve(here, '../../kestrel_sovereign/static/shared/markdown/highlight.js'),
    'utf8',
);

// --- Minimal DOM mock --------------------------------------------------------
// Just enough of the element surface that addCodeCopyButtons touches:
// dataset, classList, querySelectorAll('pre > code'), parentElement,
// closest(), appendChild, createElement, addEventListener/click, textContent.

function makeEl(tag) {
    const listeners = {};
    const el = {
        tagName: tag.toUpperCase(),
        children: [],
        parentElement: null,
        dataset: {},
        _classes: new Set(),
        _attrs: {},
        _text: '',
        style: {},
        classList: {
            add: (c) => el._classes.add(c),
            remove: (c) => el._classes.delete(c),
            toggle: (c, on) => {
                const want = on === undefined ? !el._classes.has(c) : !!on;
                if (want) el._classes.add(c); else el._classes.delete(c);
                return want;
            },
            contains: (c) => el._classes.has(c),
        },
        get className() { return Array.from(el._classes).join(' '); },
        set className(v) {
            el._classes = new Set(String(v).split(/\s+/).filter(Boolean));
        },
        setAttribute: (k, v) => { el._attrs[k] = v; },
        getAttribute: (k) => el._attrs[k],
        get textContent() { return el._text; },
        set textContent(v) { el._text = v; },
        select: () => {},
        focus: () => {},
        appendChild: (child) => { child.parentElement = el; el.children.push(child); return child; },
        addEventListener: (evt, fn) => { (listeners[evt] ||= []).push(fn); },
        dispatch: (evt, payload) => Promise.all((listeners[evt] || []).map((fn) => fn(payload))),
        closest: (sel) => {
            // sel is always '.tool-activity-container' in this code.
            const cls = sel.replace(/^\./, '');
            let node = el;
            while (node) {
                if (node._classes && node._classes.has(cls)) return node;
                node = node.parentElement;
            }
            return null;
        },
        querySelectorAll: (sel) => {
            // Only 'pre > code' is queried by addCodeCopyButtons.
            if (sel !== 'pre > code') return [];
            const out = [];
            const walk = (node) => {
                for (const child of node.children) {
                    if (child.tagName === 'CODE' && child.parentElement && child.parentElement.tagName === 'PRE') {
                        out.push(child);
                    }
                    walk(child);
                }
            };
            walk(el);
            return out;
        },
    };
    return el;
}

function loadHighlight({ clipboard } = {}) {
    const copied = { value: null };
    const documentMock = {
        createElement: (tag) => makeEl(tag),
        body: { appendChild() {}, removeChild() {} },
        execCommand: () => { copied.value = '<execCommand>'; return true; },
    };
    const windowMock = { isSecureContext: true };
    const navigatorMock = {
        clipboard: clipboard === false ? undefined : {
            writeText: async (t) => { copied.value = t; },
        },
    };
    const factory = new Function(
        'document', 'window', 'navigator', 'hljs', 'setTimeout', 'clearTimeout',
        `${highlightSrc}\nreturn { highlightCodeBlocks, addCodeCopyButtons, copyTextToClipboard };`,
    );
    const mod = factory(
        documentMock, windowMock, navigatorMock, undefined,
        (fn) => 0, // setTimeout: don't auto-revert during tests
        () => {},
    );
    return { mod, documentMock, copied };
}

function preWithCode(text) {
    const container = makeEl('div');
    const pre = makeEl('pre');
    const code = makeEl('code');
    code.textContent = text;
    pre.appendChild(code);
    container.appendChild(pre);
    return { container, pre, code };
}

test('adds a copy button to a fenced pre > code block', () => {
    const { mod } = loadHighlight();
    const { container, pre } = preWithCode('echo hi');
    mod.addCodeCopyButtons(container);
    const btn = pre.children.find((c) => c._classes.has('code-copy-btn'));
    assert.ok(btn, 'expected a .code-copy-btn appended to the pre');
    assert.equal(btn.tagName, 'BUTTON');
    assert.equal(btn.type, 'button');
    assert.equal(btn.getAttribute('aria-label'), 'Copy code');
    assert.equal(pre.dataset.copyReady, '1');
});

test('re-running the pass does not stack duplicate buttons (streaming safety)', () => {
    const { mod } = loadHighlight();
    const { container, pre } = preWithCode('echo hi');
    mod.addCodeCopyButtons(container);
    mod.addCodeCopyButtons(container);
    mod.addCodeCopyButtons(container);
    const buttons = pre.children.filter((c) => c._classes.has('code-copy-btn'));
    assert.equal(buttons.length, 1, 'exactly one button after repeated renders');
});

test('does not decorate code inside a tool-activity card', () => {
    const { mod } = loadHighlight();
    const container = makeEl('div');
    const card = makeEl('div');
    card.classList.add('tool-activity-container');
    const pre = makeEl('pre');
    const code = makeEl('code');
    code.textContent = 'internal';
    pre.appendChild(code);
    card.appendChild(pre);
    container.appendChild(card);
    mod.addCodeCopyButtons(container);
    const buttons = pre.children.filter((c) => c._classes.has('code-copy-btn'));
    assert.equal(buttons.length, 0, 'tool-activity pre must not get a copy button');
});

test('clicking copies the raw code textContent, not the label', async () => {
    const { mod, copied } = loadHighlight();
    const { container, pre } = preWithCode('const x = 1;\nconsole.log(x);');
    mod.addCodeCopyButtons(container);
    const btn = pre.children.find((c) => c._classes.has('code-copy-btn'));
    await btn.dispatch('click', { preventDefault() {}, stopPropagation() {} });
    assert.equal(copied.value, 'const x = 1;\nconsole.log(x);');
    assert.equal(btn.textContent, 'Copied');
    // Accessible name tracks the visible state so SR users hear the result.
    assert.equal(btn.getAttribute('aria-label'), 'Copied code');
    assert.ok(btn._classes.has('code-copy-btn--ok'));
});

test('falls back to execCommand when clipboard API is unavailable', async () => {
    const { mod, copied } = loadHighlight({ clipboard: false });
    const { container, pre } = preWithCode('payload');
    mod.addCodeCopyButtons(container);
    const btn = pre.children.find((c) => c._classes.has('code-copy-btn'));
    await btn.dispatch('click', { preventDefault() {}, stopPropagation() {} });
    assert.equal(copied.value, '<execCommand>');
    assert.equal(btn.textContent, 'Copied');
});

test('highlightCodeBlocks runs the copy-button pass even when hljs is absent', () => {
    const { mod } = loadHighlight();
    const { container, pre } = preWithCode('echo hi');
    // hljs is undefined in the sandbox. Highlighting is skipped, but the
    // copy-button pass is decoupled from the hljs guard and must still run, so
    // a code block stays copyable even if hljs failed to load.
    mod.highlightCodeBlocks(container);
    const buttons = pre.children.filter((c) => c._classes.has('code-copy-btn'));
    assert.equal(buttons.length, 1, 'copy button present despite missing hljs');
});
