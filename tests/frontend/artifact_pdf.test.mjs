import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { createPdfPreview } from '../../kestrel_sovereign/static/js/artifact_pdf.mjs';

function fixture({ render = () => Promise.resolve(), getPage, timeoutMs = 1000 } = {}) {
    const dom = new JSDOM('<!doctype html><body></body>');
    dom.window.HTMLCanvasElement.prototype.getContext = () => ({});
    const observed = { pages: [], cancelled: 0, destroyed: 0, workerDestroyed: 0 };
    const library = { GlobalWorkerOptions: {}, PDFWorker: class { destroy() { observed.workerDestroyed++; } },
        getDocument(options) {
            observed.options = options;
            return { destroy: async () => { observed.destroyed++; }, promise: Promise.resolve({ numPages: 2,
                getPage: async number => {
                    observed.pages.push(number);
                    if (getPage) return getPage(number);
                    return { getViewport: ({scale}) => ({width: 1e6 * scale, height: 1e6 * scale}),
                        render: options => { observed.render = options; return { promise: render(number), cancel() { observed.cancelled++; } }; },
                        cleanup() {} };
                } }) };
        } };
    const bytes = new Uint8Array([1,2,3]);
    const preview = createPdfPreview(bytes, dom.window.document, { loadLibrary: async () => library, timeoutMs });
    dom.window.document.body.append(preview.element);
    return { dom, preview, observed, bytes, close() { preview.dispose(); dom.window.close(); } };
}
const tick = () => new Promise(resolve => setTimeout(resolve, 0));

test('PDF renderer copies source bytes, uses local assets and caps canvas dimensions', async () => {
    const item = fixture(); await item.preview.ready;
    const { options, render } = item.observed;
    assert.notEqual(options.data, item.bytes); assert.deepEqual(options.data, item.bytes);
    assert.equal(options.enableXfa, false); assert.equal(options.disableFontFace, true);
    for (const field of ['cMapUrl', 'standardFontDataUrl', 'wasmUrl']) assert.match(options[field], /static\/vendor\/pdfjs\//);
    assert.ok(render.canvas.width * render.canvas.height <= 4_000_000);
    assert.ok(render.canvas.width <= 2048 && render.canvas.height <= 2048);
    assert.equal(item.preview.element.querySelector('p').textContent, 'Page 1 of 2');
    assert.equal(item.preview.element.querySelectorAll('iframe,object,embed,a').length, 0);
    item.preview.element.querySelectorAll('button')[1].click(); await tick();
    assert.deepEqual(item.observed.pages, [1,2]);
    assert.equal(item.preview.element.querySelector('p').textContent, 'Page 2 of 2'); item.close();
    assert.equal(item.observed.destroyed, 1); assert.equal(item.observed.workerDestroyed, 1);
});

test('closing a rendering preview cancels the task and ignores late completion', async () => {
    let finish; const item = fixture({ render: () => new Promise(resolve => { finish = resolve; }) });
    await tick(); item.preview.dispose(); finish(); await item.preview.ready;
    assert.equal(item.observed.cancelled, 1); assert.equal(item.observed.workerDestroyed, 1);
    assert.equal(item.preview.element.childElementCount, 0); item.close();
});

test('PDF deadline terminates the owned worker and leaves an explicit error', async () => {
    let finish; const item = fixture({ render: () => new Promise(resolve => { finish = resolve; }), timeoutMs: 10 });
    await new Promise(resolve => setTimeout(resolve, 30));
    assert.equal(item.observed.workerDestroyed, 1);
    assert.match(item.preview.element.textContent, /preview is unavailable/);
    assert.equal(item.preview.element.querySelector('canvas').width, 0);
    finish(); await item.preview.ready;
    assert.match(item.preview.element.textContent, /preview is unavailable/); item.close();
});

test('closing before library load never starts a PDF worker', async () => {
    const dom = new JSDOM('<!doctype html><body></body>');
    let finish; const preview = createPdfPreview(new Uint8Array(), dom.window.document,
        { loadLibrary: () => new Promise(resolve => { finish = resolve; }) });
    preview.dispose(); finish({ getDocument() { assert.fail('closed preview started'); } });
    await preview.ready; assert.equal(preview.element.childElementCount, 0); dom.window.close();
});

test('invalid page sizes fail without allocating an unbounded canvas', async () => {
    const item = fixture({ getPage: () => ({ getViewport: () => ({ width: Infinity, height: 12 }) }) });
    await item.preview.ready;
    assert.match(item.preview.element.textContent, /preview is unavailable/);
    assert.equal(item.observed.workerDestroyed, 1); item.close();
});
