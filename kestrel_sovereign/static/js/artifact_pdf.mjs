/** Local, canvas-only PDF rendering: no document actions, links or form scripts. */
const ASSETS = new URL('../vendor/pdfjs/', import.meta.url);
const MAX_PIXELS = 4_000_000;
const MAX_SIDE = 2048;
const PREVIEW_TIMEOUT_MS = 15_000;

export function createPdfPreview(bytes, doc = globalThis.document, {
    loadLibrary = () => import('../vendor/pdfjs/pdf.min.mjs'),
    timeoutMs = PREVIEW_TIMEOUT_MS,
} = {}) {
    const root = doc.createElement('section'); root.setAttribute('aria-label', 'PDF preview');
    const status = doc.createElement('p'); status.setAttribute('role', 'status');
    const canvas = doc.createElement('canvas'); canvas.setAttribute('aria-label', 'PDF page');
    canvas.style.cssText = 'max-width:100%;height:auto;background:white;';
    const previous = doc.createElement('button'), next = doc.createElement('button');
    previous.textContent = 'Previous page'; next.textContent = 'Next page';
    for (const button of [previous, next]) { button.type = 'button'; button.disabled = true; button.className = 'btn btn-secondary'; }
    root.append(status, previous, next, canvas);
    let disposed = false, failed = false, busy = true, pageNumber = 1;
    let loadingTask, pdf, worker, rendering, timer;
    const stop = () => {
        clearTimeout(timer); rendering?.cancel();
        // PDF.js destruction may race worker initialization or a rejected load.
        if (loadingTask) void loadingTask.destroy().catch(() => {});
        worker?.destroy(); canvas.width = canvas.height = 0;
    };
    const fail = () => {
        if (disposed || failed) return;
        failed = true; previous.disabled = next.disabled = true;
        status.textContent = 'PDF preview is unavailable. Download the saved revision to open it.';
        stop();
    };
    const deadline = () => { clearTimeout(timer); timer = setTimeout(fail, timeoutMs); };
    async function renderPage(number) {
        busy = true; previous.disabled = next.disabled = true; deadline();
        status.textContent = `Opening page ${number}…`;
        try {
            const page = await pdf.getPage(number);
            if (disposed || failed) return;
            const original = page.getViewport({ scale: 1 });
            if (!Number.isFinite(original.width) || !Number.isFinite(original.height)
                || original.width <= 0 || original.height <= 0) throw new Error('Invalid PDF page size');
            const scale = Math.min(1.5, MAX_SIDE / original.width, MAX_SIDE / original.height,
                Math.sqrt(MAX_PIXELS / (original.width * original.height)));
            const viewport = page.getViewport({ scale });
            canvas.width = Math.max(1, Math.floor(viewport.width));
            canvas.height = Math.max(1, Math.floor(viewport.height));
            rendering = page.render({ canvas, canvasContext: canvas.getContext('2d'), viewport });
            await rendering.promise;
            if (disposed || failed) return;
            rendering = null; page.cleanup(); clearTimeout(timer); busy = false; pageNumber = number;
            status.textContent = `Page ${number} of ${pdf.numPages}`;
            previous.disabled = number === 1; next.disabled = number === pdf.numPages;
        } catch (_) { fail(); }
    }
    previous.addEventListener('click', () => { if (!busy && !disposed && !failed && pageNumber > 1) void renderPage(pageNumber - 1); });
    next.addEventListener('click', () => { if (!busy && !disposed && !failed && pageNumber < pdf.numPages) void renderPage(pageNumber + 1); });
    status.textContent = 'Opening PDF…'; deadline();
    const ready = (async () => {
        try {
            const library = await loadLibrary();
            if (disposed || failed) return;
            library.GlobalWorkerOptions.workerSrc = new URL('pdf.worker.min.mjs', ASSETS).href;
            worker = new library.PDFWorker();
            loadingTask = library.getDocument({ data: bytes.slice(), worker, enableXfa: false,
                stopAtErrors: true, useSystemFonts: false, disableFontFace: true,
                maxImageSize: MAX_PIXELS, canvasMaxAreaInBytes: MAX_PIXELS * 4,
                cMapUrl: new URL('cmaps/', ASSETS).href,
                standardFontDataUrl: new URL('standard_fonts/', ASSETS).href,
                wasmUrl: new URL('wasm/', ASSETS).href });
            pdf = await loadingTask.promise;
            if (disposed || failed) return;
            await renderPage(1);
        } catch (_) { fail(); }
    })();
    return { element: root, ready, dispose() {
        if (disposed) return;
        disposed = true; stop(); root.replaceChildren();
    } };
}
