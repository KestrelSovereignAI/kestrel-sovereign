import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { JSDOM } from 'jsdom';

// mermaid.js is a classic script that declares its functions at top level. We
// eval the source in a sandbox backed by a real (jsdom) DOM, appending a return
// that hands back the internals we want to assert on — the same pattern the
// other shared/markdown tests use. window.mermaid is pre-set to a stub, so the
// lazy `import()` of the CDN bundle is never reached.

const here = dirname(fileURLToPath(import.meta.url));
const src = readFileSync(
    resolve(here, '../../kestrel_sovereign/static/shared/markdown/mermaid.js'),
    'utf8',
);

function loadMermaid(win) {
    const factory = new Function(
        'window', 'document', 'console',
        `${src}\nreturn { renderMermaidDiagrams, ensureMermaid, _uniquifySvgIds, _svgCache, _loader };`,
    );
    return factory(win, win.document, { warn() {}, log() {}, error() {} });
}

function freshDom() {
    return new JSDOM('<!doctype html><body></body>').window;
}

function blockEl(win, source) {
    const el = win.document.createElement('div');
    const pre = win.document.createElement('pre');
    const code = win.document.createElement('code');
    code.className = 'language-mermaid';
    code.textContent = source;
    pre.appendChild(code);
    el.appendChild(pre);
    return el;
}

test('identical diagram source renders once (content-hash cache)', async () => {
    const win = freshDom();
    let renderCount = 0;
    win.mermaid = {
        initialize() {},
        async render(id) {
            renderCount += 1;
            return { svg: `<svg id="root"><marker id="m1"></marker><path marker-end="url(#m1)"></path></svg>` };
        },
    };
    const mod = loadMermaid(win);
    const SRC = 'graph TD; A-->B';

    const a = blockEl(win, SRC);
    const b = blockEl(win, SRC);
    await mod.renderMermaidDiagrams(a);
    await mod.renderMermaidDiagrams(b);

    assert.equal(renderCount, 1, 'second identical diagram must come from cache');
    assert.equal(mod._svgCache.get(SRC) !== undefined, true);
    assert.match(a.querySelector('.mermaid-wrapper').innerHTML, /<svg/);
    assert.match(b.querySelector('.mermaid-wrapper').innerHTML, /<svg/);
});

test('each injected SVG gets a unique id namespace (collision-safe reuse)', async () => {
    const win = freshDom();
    win.mermaid = {
        initialize() {},
        async render() {
            return { svg: `<svg id="root"><marker id="m1"></marker><path marker-end="url(#m1)"></path></svg>` };
        },
    };
    const mod = loadMermaid(win);
    const SRC = 'graph TD; A-->B';

    const a = blockEl(win, SRC);
    const b = blockEl(win, SRC);
    await mod.renderMermaidDiagrams(a);
    await mod.renderMermaidDiagrams(b);

    const idA = a.querySelector('svg').getAttribute('id');
    const idB = b.querySelector('svg').getAttribute('id');
    assert.notEqual(idA, idB, 'reused SVG instances must not share the root id');
    // The url(#…) reference must track the rewritten marker id, not dangle.
    const markerIdA = a.querySelector('marker').getAttribute('id');
    assert.match(a.querySelector('path').getAttribute('marker-end'), new RegExp(`#${markerIdA}\\)`));
});

test('_uniquifySvgIds rewrites ids and their url(#…) references together', () => {
    const win = freshDom();
    win.mermaid = { initialize() {}, async render() { return { svg: '' }; } };
    const { _uniquifySvgIds } = loadMermaid(win);
    const out = _uniquifySvgIds('<svg id="a"><marker id="b"></marker><path marker-end="url(#b)"></path></svg>');
    const markerId = out.match(/<marker id="([^"]+)"/)[1];
    assert.match(out, new RegExp(`url\\(#${markerId}\\)`));
    assert.doesNotMatch(out, /url\(#b\)/, 'old reference must be rewritten');
});

test('_uniquifySvgIds rewrites scoped CSS selectors and aria references', () => {
    const win = freshDom();
    win.mermaid = { initialize() {}, async render() { return { svg: '' }; } };
    const { _uniquifySvgIds } = loadMermaid(win);
    const out = _uniquifySvgIds(
        '<svg id="root" aria-labelledby="title0">'
        + '<style>#root .node{fill:red} #root .edge{stroke:#000}</style>'
        + '<title id="title0">t</title></svg>',
    );
    const rootId = out.match(/<svg id="([^"]+)"/)[1]; // root-iN
    // Scoped CSS selectors must track the new root id, not dangle on #root.
    assert.match(out, new RegExp(`#${rootId} \\.node`));
    assert.doesNotMatch(out, /#root \.node/);
    // The literal color #000 (not an id) must be left untouched.
    assert.match(out, /stroke:#000/);
    // aria-labelledby must follow the renamed title id.
    const titleId = out.match(/<title id="([^"]+)"/)[1];
    assert.match(out, new RegExp(`aria-labelledby="${titleId}"`));
});

test('ensureMermaid returns an already-present window.mermaid without importing', async () => {
    const win = freshDom();
    const stub = { initialize() {}, async render() { return { svg: '' }; } };
    win.mermaid = stub;
    const { ensureMermaid } = loadMermaid(win);
    assert.equal(await ensureMermaid(), stub);
});

test('a diagram still renders when the first load resolves after the wait cap', async () => {
    const win = freshDom();
    const mod = loadMermaid(win);
    mod._loader.maxWait = 5; // force the race to give up before the import resolves
    let resolveImport;
    mod._loader.import = () => new Promise((res) => { resolveImport = res; });

    const el = blockEl(win, 'graph TD; A-->B');
    await mod.renderMermaidDiagrams(el);
    assert.equal(el.querySelector('.mermaid-wrapper'), null, 'not rendered before load completes');

    // The lazy import finally resolves; the deferred retry must render it.
    resolveImport({ default: { initialize() {}, async render() { return { svg: '<svg id="r"></svg>' }; } } });
    for (let i = 0; i < 6; i++) await new Promise((r) => setTimeout(r, 0));
    assert.ok(el.querySelector('.mermaid-wrapper'), 'deferred retry rendered the diagram after the slow load');
});

test('render failure escapes the error message and the source', async () => {
    const win = freshDom();
    win.mermaid = {
        initialize() {},
        async render() { throw new Error('boom <script>'); },
    };
    const mod = loadMermaid(win);
    const el = blockEl(win, 'graph <bad>');
    await mod.renderMermaidDiagrams(el);
    const html = el.querySelector('.mermaid-wrapper').innerHTML;
    assert.match(html, /mermaid-error/);
    assert.doesNotMatch(html, /<script>/);
    assert.match(html, /&lt;script&gt;/);
    assert.match(html, /graph &lt;bad&gt;/);
});
