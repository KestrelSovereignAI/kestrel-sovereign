/** Shipped panel editing a durable source after its real Docker source died. */
const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const { expect } = require('@playwright/test');
(async () => {
    assert.match(process.env.ARTIFACT_TEST_URL, /^http:\/\/127\.0\.0\.1:\d+$/);
    const browser = await chromium.launch({ headless: true });
    try {
        const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
        const errors = [];
        page.on('pageerror', error => errors.push(error.message));
        await page.addInitScript(key => sessionStorage.setItem('kestrel_api_key', key), process.env.ARTIFACT_TEST_KEY);
        await page.goto(process.env.ARTIFACT_TEST_URL);
        const launch = page.getByRole('button', { name: 'Artifacts', exact: true });
        await expect(launch).toBeVisible({ timeout: 30000 });
        await launch.click();
        const library = page.getByRole('region', { name: 'Artifact library' });
        await library.getByLabel('Search artifacts').fill('Broker durable output');
        await library.getByRole('button', { name: 'Search', exact: true }).click();
        await expect(library.locator('.artifact-card')).toHaveCount(1);
        await library.getByRole('button', { name: 'Open saved artifact' }).click();
        const panel = page.getByRole('region', { name: 'Saved artifact' }).last();
        const editor = panel.getByLabel('Artifact source');
        await expect(editor).toContainText('');
        await expect(editor).toHaveValue(/Broker durable source/);
        const original = await editor.inputValue();
        const edited = original.replace('Broker durable source', 'Edited after Docker destruction') + '\n<script>globalThis.__artifactExecuted = true</script>';
        await editor.fill(edited);
        await panel.getByRole('button', { name: 'Save revision', exact: true }).click();
        await expect(panel.getByRole('status')).toHaveText('Saved revision');
        await expect(editor).toHaveValue(edited);
        assert.equal(await page.evaluate(() => globalThis.__artifactExecuted), undefined);
        assert.equal(await panel.locator('script,iframe,embed,object,img,canvas').count(), 0);
        const revision_id = (await panel.locator('.artifact-revision').textContent()).replace(/^Revision /, '');
        if (process.env.ARTIFACT_BROWSER_SCREENSHOT) await page.screenshot({ path: process.env.ARTIFACT_BROWSER_SCREENSHOT });
        assert.deepEqual(errors, []);
        console.log(JSON.stringify({ verdict:'PASS', revision_id, source_text:edited, path:'destroyed Docker source → authenticated library → inert HTML source editor → durable revision2' }));
    } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
