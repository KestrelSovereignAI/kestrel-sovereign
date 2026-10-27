/** Driven only by test_artifact_product_browser.py against its own real host. */
const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const { expect } = require('@playwright/test');
(async () => {
    const base = process.env.ARTIFACT_TEST_URL;
    assert.match(base, /^http:\/\/127\.0\.0\.1:\d+$/);
    const browser = await chromium.launch({ headless: true, channel: 'chromium' });
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.addInitScript(key => sessionStorage.setItem('kestrel_api_key', key), process.env.ARTIFACT_TEST_KEY);
    try {
        await page.goto(base);
        const launch = page.getByRole('button', { name: 'Artifacts', exact: true });
        await expect(launch).toBeVisible({ timeout: 30000 }); await launch.click();
        const library = page.getByRole('region', { name: 'Artifact library' });
        await expect(library.getByRole('button', { name: 'New text artifact' })).toBeEnabled();
        const workspace = page.getByRole('complementary', { name: 'Artifacts', exact: true });
        const narrow = await workspace.boundingBox(); assert.equal(narrow.x + narrow.width, 1440);
        await workspace.getByRole('button', { name: 'Expand artifacts', exact: true }).click();
        assert.ok((await workspace.boundingBox()).width > narrow.width);
        await workspace.getByRole('button', { name: 'Restore panel width' }).click();
        if (process.env.ARTIFACT_BROWSER_SCREENSHOT) {
            await page.screenshot({ path: process.env.ARTIFACT_BROWSER_SCREENSHOT.replace(/\.png$/, '-library.png') });
            await page.setViewportSize({ width: 390, height: 844 });
            const mobile = await workspace.boundingBox();
            assert.equal(Math.round(mobile.x + mobile.width), 390);
            assert.equal(await workspace.evaluate(node => node.scrollWidth <= node.clientWidth), true);
            await expect(library.getByLabel('New artifact title')).toBeVisible();
            await page.screenshot({ path: process.env.ARTIFACT_BROWSER_SCREENSHOT.replace(/\.png$/, '-mobile-library.png') });
            await page.setViewportSize({ width: 1440, height: 1000 });
        }
        await library.getByLabel('Upload artifact').setInputFiles({ name: 'evidence.md', mimeType: 'text/markdown', buffer: Buffer.from('# Uploaded source\n') });
        let panel = page.getByRole('region', { name: 'Saved artifact' }).last();
        await expect(panel.getByRole('status')).toHaveText('Unsaved changes');
        await expect(panel.getByLabel('Artifact source')).toHaveValue('# Uploaded source\n');
        await panel.getByRole('button', { name: 'Save revision', exact: true }).click();
        await expect(panel.getByRole('status')).toHaveText('Saved revision');
        const revision1 = await panel.locator('.artifact-revision').textContent();
        await panel.getByLabel('Artifact source').fill('# Edited after upload\n');
        let dropped = false;
        await page.route('**/api/artifacts/publications', async route => {
            if (dropped) return route.continue();
            const response = await route.fetch(); assert.equal(response.status(), 200); dropped = true;
            await route.abort('failed');
        });
        await panel.getByRole('button', { name: 'Save revision', exact: true }).click();
        await expect(panel.getByRole('status')).toContainText('Save status is unknown');
        assert.equal(dropped, true);
        await panel.getByRole('button', { name: 'Check save', exact: true }).click();
        await expect(panel.getByRole('status')).toHaveText('Saved revision');
        const revision2 = await panel.locator('.artifact-revision').textContent(); assert.notEqual(revision2, revision1);
        await page.unroute('**/api/artifacts/publications');
        await page.reload(); await expect(launch).toBeVisible(); await launch.click();
        await expect(library.getByRole('button', { name: 'New text artifact' })).toBeEnabled();
        await library.getByLabel('Search artifacts').fill('EVIDENCE.MD');
        await library.getByRole('button', { name: 'Search', exact: true }).click();
        await expect(library.locator('.artifact-card')).toHaveCount(1);
        await library.getByRole('button', { name: 'Open saved artifact' }).click();
        panel = page.getByRole('region', { name: 'Saved artifact' }).last();
        await expect(panel.getByLabel('Artifact source')).toHaveValue('# Edited after upload\n');
        await expect(panel.locator('.artifact-revision')).toHaveText(revision2);
        await panel.getByRole('button', { name: 'Revision history' }).click();
        await expect(panel.locator('h4')).toHaveText('Saved revisions');
        // Open a second working copy of the same current head for real CAS.
        await library.getByRole('button', { name: 'Open saved artifact' }).click();
        const first = page.getByRole('region', { name: 'Saved artifact' }).first();
        const stale = page.getByRole('region', { name: 'Saved artifact' }).last();
        await expect(stale.getByLabel('Artifact source')).toHaveValue('# Edited after upload\n');
        await first.getByLabel('Artifact source').fill('Winner');
        await first.getByRole('button', { name: 'Save revision', exact: true }).click();
        await expect(first.getByRole('status')).toHaveText('Saved revision');
        await stale.getByLabel('Artifact source').fill('Retained conflicting draft');
        await stale.getByRole('button', { name: 'Save revision', exact: true }).click();
        await expect(stale.getByRole('status')).toContainText('Another revision conflicts');
        await stale.getByRole('button', { name: 'Compare latest', exact: true }).click();
        await expect(stale.locator('pre')).toHaveText('Winner');
        await stale.getByRole('button', { name: 'Use latest as save base' }).click();
        await stale.getByRole('button', { name: 'Save revision', exact: true }).click();
        await expect(stale.getByRole('status')).toHaveText('Saved revision');
        await expect(stale.getByLabel('Artifact source')).toHaveValue('Retained conflicting draft');
        if (process.env.ARTIFACT_BROWSER_SCREENSHOT) {
            await stale.getByLabel('Artifact source').scrollIntoViewIfNeeded();
            await page.screenshot({ path: process.env.ARTIFACT_BROWSER_SCREENSHOT.replace(/\.png$/, '-editor.png') });
            await page.setViewportSize({ width: 390, height: 844 });
            await stale.getByLabel('Artifact source').scrollIntoViewIfNeeded();
            assert.equal(await workspace.evaluate(node => node.scrollWidth <= node.clientWidth), true);
            await page.screenshot({ path: process.env.ARTIFACT_BROWSER_SCREENSHOT.replace(/\.png$/, '-mobile-editor.png') });
            await page.setViewportSize({ width: 1440, height: 1000 });
        }
        const previewFixtures = JSON.parse(process.env.ARTIFACT_TEST_PREVIEW_FIXTURES || '[]');
        for (const fixture of previewFixtures) {
            await library.getByLabel('Upload artifact').setInputFiles({ name: fixture.name, mimeType: fixture.mimeType, buffer: Buffer.from(fixture.base64, 'base64') });
            const previewPanel = page.getByRole('region', { name: 'Saved artifact' }).last();
            await expect(previewPanel.getByRole('status')).toHaveText('Unsaved changes');
            await previewPanel.getByRole('button', { name: 'Save revision', exact: true }).click();
            await expect(previewPanel.getByRole('status')).toHaveText('Saved revision');
            const image = previewPanel.locator('img.artifact-preview');
            await expect(image).toBeVisible({ timeout: 20000 });
            const dimensions = await image.evaluate(node => [node.naturalWidth, node.naturalHeight]);
            assert.ok(dimensions.every(value => value > 0 && value <= 1024));
            assert.match(await image.getAttribute('src'), /^blob:/);
            await expect(previewPanel.getByRole('button', { name: 'Download source file' })).toBeEnabled();
            if (process.env.ARTIFACT_BROWSER_SCREENSHOT) {
                await image.scrollIntoViewIfNeeded();
                await page.screenshot({ path: process.env.ARTIFACT_BROWSER_SCREENSHOT.replace(/\.png$/, `-${fixture.name.replace('.', '-')}.png`) });
            }
        }
        assert.deepEqual(errors, []);
        if (process.env.ARTIFACT_BROWSER_SCREENSHOT) await page.screenshot({ path: process.env.ARTIFACT_BROWSER_SCREENSHOT });
        console.log(JSON.stringify({ verdict: 'PASS', path: 'production console header → library → upload → durable receipt → lost ACK reconciliation → reload → encrypted head discovery → CAS conflict → explicit rebase', revision1, revision2, previewKinds: previewFixtures.map(item => item.mimeType), cleanup: 'browser closed by finally' }));
    } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
