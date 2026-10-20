/** Real Chromium regression for the blank native-PDF-frame failure. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { deflateSync } from 'node:zlib';
import { readFile } from 'node:fs/promises';
import { chromium } from '@playwright/test';

function pdfFixture() {
    const stream = text => { const content = `BT /F1 20 Tf 20 80 Td (${text}) Tj ET`;
        return `<< /Length ${content.length} >>\nstream\n${content}\nendstream`; };
    const objects = [
        '<< /Type /Catalog /Pages 2 0 R /OpenAction << /S /JavaScript /JS (globalThis.__pdfActionExecuted=true) >> >>',
        '<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>',
        '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 320 160] /Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R >>',
        '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 320 160] /Resources << /Font << /F1 5 0 R >> >> /Contents 7 0 R >>',
        '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
        stream('Saved artifact PDF'), stream('Second saved page'),
    ];
    return assemblePdf(objects.map(object => Buffer.from(object)));
}

function assemblePdf(objects) {
    const parts = [Buffer.from('%PDF-1.7\n')], offsets = [0];
    let size = parts[0].length;
    objects.forEach((object, index) => {
        offsets.push(size);
        const part = Buffer.concat([Buffer.from(`${index + 1} 0 obj\n`), object, Buffer.from('\nendobj\n')]);
        parts.push(part); size += part.length;
    });
    const xref = `xref\n0 ${offsets.length}\n0000000000 65535 f \n`
        + offsets.slice(1).map(offset => `${String(offset).padStart(10, '0')} 00000 n \n`).join('');
    parts.push(Buffer.from(xref + `trailer\n<< /Size ${offsets.length} /Root 1 0 R >>\nstartxref\n${size}\n%%EOF\n`));
    return Buffer.concat(parts);
}

function scanFixture(width, height) {
    const image = deflateSync(Buffer.alloc(width * height, 24));
    const content = Buffer.from('q 612 0 0 792 0 0 cm /Scan Do Q');
    return assemblePdf([
        Buffer.from('<< /Type /Catalog /Pages 2 0 R >>'),
        Buffer.from('<< /Type /Pages /Kids [3 0 R] /Count 1 >>'),
        Buffer.from('<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /XObject << /Scan 4 0 R >> >> /Contents 5 0 R >>'),
        Buffer.concat([Buffer.from(`<< /Type /XObject /Subtype /Image /Width ${width} /Height ${height} /ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode /Length ${image.length} >>\nstream\n`), image, Buffer.from('\nendstream')]),
        Buffer.concat([Buffer.from(`<< /Length ${content.length} >>\nstream\n`), content, Buffer.from('\nendstream')]),
    ]);
}

test('real PDF bytes paint two navigable canvas pages without running document JavaScript', { timeout: 30000 }, async () => {
    const source = pdfFixture().toString('base64');
    const scan = scanFixture(2550, 3300).toString('base64');
    const oversized = scanFixture(10_000, 10_000).toString('base64');
    const assetRoot = new URL('../../kestrel_sovereign/static/', import.meta.url);
    const server = createServer(async (request, response) => {
        try {
            const path = new URL(request.url, 'http://localhost').pathname;
            if (path === '/') {
                response.setHeader('Content-Type', 'text/html');
                response.end(`<!doctype html><body><script type="module">
                    import { createPdfPreview } from '/js/artifact_pdf.mjs';
                    window.preview = createPdfPreview(Uint8Array.from(atob('${source}'), c => c.charCodeAt(0)));
                    document.body.append(preview.element); await preview.ready; window.ready = true;
                    window.openPdf = async source => { preview.dispose(); window.preview = createPdfPreview(Uint8Array.from(atob(source), c => c.charCodeAt(0))); document.body.append(preview.element); await preview.ready; };
                </script></body>`); return;
            }
            // Match the product's /js mount; /vendor is deliberately absent.
            if (path !== '/js/artifact_pdf.mjs' && !path.startsWith('/js/vendor/pdfjs/')) {
                response.writeHead(404); response.end(); return;
            }
            const file = new URL(`js/${path.slice('/js/'.length)}`, assetRoot);
            if (!file.href.startsWith(assetRoot.href)) throw new Error('Outside test assets');
            response.setHeader('Content-Type', path.endsWith('.mjs') || path.endsWith('.js') ? 'text/javascript'
                : path.endsWith('.wasm') ? 'application/wasm' : 'application/octet-stream');
            response.end(await readFile(file));
        } catch (_) { response.writeHead(404); response.end(); }
    });
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    let browser;
    try {
        browser = await chromium.launch({ headless: true });
        const page = await browser.newPage();
        const origin = `http://127.0.0.1:${server.address().port}`;
        const external = [];
        await page.route('**/*', route => {
            if (new URL(route.request().url()).origin !== origin) { external.push(route.request().url()); return route.abort(); }
            return route.continue();
        });
        await page.goto(origin); await page.waitForFunction(() => window.ready === true);
        assert.equal(await page.locator('p').textContent(), 'Page 1 of 2');
        const painted = async () => page.locator('canvas').evaluate(canvas => {
            const bytes = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
            let dark = 0;
            for (let i = 0; i < bytes.length; i += 4) if (bytes[i] < 100 && bytes[i + 1] < 100 && bytes[i + 2] < 100 && bytes[i + 3] > 0) dark++;
            return dark;
        });
        assert.ok(await painted() > 100, 'first page is visibly painted');
        await page.getByRole('button', { name: 'Next page' }).click();
        await page.getByText('Page 2 of 2', { exact: true }).waitFor();
        assert.ok(await painted() > 100, 'second page is visibly painted');
        assert.equal(await page.evaluate(() => window.__pdfActionExecuted), undefined);
        assert.equal(await page.locator('iframe,embed,object,a').count(), 0);
        assert.deepEqual(external, []);
        await page.evaluate(source => window.openPdf(source), scan);
        assert.equal(await page.locator('p').textContent(), 'Page 1 of 1');
        assert.ok(await painted() > 100_000, '300-dpi scan is visibly painted');
        const size = await page.locator('canvas').evaluate(canvas => canvas.width * canvas.height);
        assert.ok(size <= 4_000_000, 'scan output remains bounded');
        await page.evaluate(source => window.openPdf(source), oversized);
        assert.match(await page.locator('p').textContent(), /PDF preview is unavailable/);
        assert.equal(await page.locator('canvas').evaluate(canvas => canvas.width), 0);
        assert.equal(await page.getByRole('button', { name: 'Next page' }).isDisabled(), true);
        const error = await page.evaluate(async source => {
            const { getDocument } = await import('/js/vendor/pdfjs/pdf.min.mjs');
            const task = getDocument({ data: Uint8Array.from(atob(source), c => c.charCodeAt(0)),
                stopAtErrors: true, maxImageSize: 80_000_000 });
            try { const pdf = await task.promise; const page = await pdf.getPage(1); await page.getOperatorList(); return null; }
            catch (error) { return error.message; }
            finally { await task.destroy(); }
        }, oversized);
        assert.match(error, /Image exceeded maximum allowed size/);
        await page.evaluate(() => window.preview.dispose());
        assert.equal(await page.locator('canvas').count(), 0);
    } finally {
        await browser?.close();
        await new Promise(resolve => server.close(resolve));
    }
});
