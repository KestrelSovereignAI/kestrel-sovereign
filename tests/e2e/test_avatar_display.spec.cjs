// @ts-check
const { test, expect } = require('@playwright/test');

const KESTREL_URL = process.env.KESTREL_URL || 'http://localhost:8888';

test.describe('Avatar Display in Kestrel UI', () => {

    test.beforeEach(async ({ page }) => {
        await page.goto(KESTREL_URL);
        // Wait for DOM to be ready (not networkidle - that can hang on polling pages)
        await page.waitForLoadState('domcontentloaded');
        // Chat is the default tab — navigate to Identity to render its panel.
        await page.locator('.nav-tab').filter({ hasText: /identity/i }).click();
        await page.waitForSelector('.identity-header', { timeout: 15000 });
    });

    test('identity panel renders loaded identity content', async ({ page }) => {
        const identityCard = page.locator('#identity-card');
        await expect(identityCard).toBeVisible();
        await expect(identityCard.locator('.identity-header')).toBeVisible();
        await expect(identityCard.locator('#profile-name')).toBeVisible();
        await expect(identityCard.locator('.identity-did-text')).not.toHaveText('');
    });

    test('avatar container has correct dimensions', async ({ page }) => {
        const avatar = page.locator('.identity-avatar');
        await expect(avatar).toBeVisible();
        const box = await avatar.boundingBox();

        expect(box).not.toBeNull();
        expect(box.width).toBeGreaterThan(50);
        expect(box.height).toBeGreaterThan(50);
        const ratio = box.width / box.height;
        expect(ratio).toBeGreaterThan(0.8);
        expect(ratio).toBeLessThan(1.2);
    });

    test('avatar image has a concrete custom or fallback source', async ({ page }) => {
        const avatarImg = page.locator('.identity-avatar-img');
        await expect(avatarImg).toBeVisible();

        const src = await avatarImg.getAttribute('src');
        expect(src).toBeTruthy();
        expect(src).toMatch(/\/api\/files\/.+|\/api\/kestrel\/companions\/.+\/avatar|data:image\/svg\+xml|\/static\/favicon\.svg/);

        const loaded = await avatarImg.evaluate((img) => img.complete && img.naturalWidth > 0);
        expect(loaded).toBe(true);
    });

    test('identity API returns avatar fields', async ({ page, request }) => {
        // Direct API test
        const response = await request.get(`${KESTREL_URL}/api/identity`);

        // Skip if agent not initialized
        if (response.status() === 503) {
            test.skip();
            return;
        }

        expect(response.status()).toBe(200);
        const data = await response.json();

        // Verify avatar fields exist in response
        expect(data).toHaveProperty('avatar_hash');
        expect(data).toHaveProperty('avatar_url');
    });

    test('files endpoint serves avatar images', async ({ page, request }) => {
        // First get identity to find avatar URL
        const identityResponse = await request.get(`${KESTREL_URL}/api/identity`);

        if (identityResponse.status() === 503) {
            test.skip();
            return;
        }

        const identity = await identityResponse.json();

        if (identity.avatar_url) {
            // Fetch the avatar
            const avatarResponse = await request.get(`${KESTREL_URL}${identity.avatar_url}`);

            expect(avatarResponse.status()).toBe(200);

            // Should have image content type
            const contentType = avatarResponse.headers()['content-type'];
            expect(contentType).toMatch(/image\/.*/);
        }
    });

    test('files endpoint is publicly accessible', async ({ request }) => {
        // Files should be accessible without authentication
        const response = await request.get(`${KESTREL_URL}/api/files/nonexistent_test_hash`);

        // Should get 404 (not found), not 401 (unauthorized)
        expect(response.status()).toBe(404);
    });

    test('avatar styling applied correctly', async ({ page }) => {
        const avatar = page.locator('.identity-avatar');
        await expect(avatar).toBeVisible();
        const borderRadius = await avatar.evaluate(el =>
            window.getComputedStyle(el).borderRadius
        );

        expect(borderRadius).not.toBe('0px');
    });

    test('avatar container has overflow hidden', async ({ page }) => {
        const avatar = page.locator('.identity-avatar');
        await expect(avatar).toBeVisible();
        const overflow = await avatar.evaluate(el =>
            window.getComputedStyle(el).overflow
        );

        expect(overflow).toBe('hidden');
    });

    test('avatar image uses object-fit cover', async ({ page }) => {
        const avatarImg = page.locator('.identity-avatar-img');
        await expect(avatarImg).toBeVisible();
        const objectFit = await avatarImg.evaluate(el =>
            window.getComputedStyle(el).objectFit
        );

        expect(objectFit).toBe('cover');
    });
});
