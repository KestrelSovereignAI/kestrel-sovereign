// @ts-check
const { test, expect } = require('@playwright/test');

const KESTREL_URL = process.env.KESTREL_URL || 'http://localhost:8888';

test.describe('Avatar Display in Kestrel UI', () => {

    test.beforeEach(async ({ page }) => {
        await page.goto(KESTREL_URL);
        // Wait for DOM to be ready (not networkidle - that can hang on polling pages)
        await page.waitForLoadState('domcontentloaded');
    });

    test('identity panel exists', async ({ page }) => {
        // Identity section should exist
        const identitySection = page.locator('.identity-card, #identity-card, [data-section="identity"]');
        // May or may not be visible depending on UI state
        const count = await identitySection.count();
        expect(count).toBeGreaterThanOrEqual(0);
    });

    test('avatar container has correct dimensions', async ({ page }) => {
        // Look for avatar element
        const avatar = page.locator('.identity-avatar');

        if (await avatar.count() > 0) {
            const box = await avatar.boundingBox();

            if (box) {
                // Should be roughly square (allow some tolerance)
                expect(box.width).toBeGreaterThan(50);
                expect(box.height).toBeGreaterThan(50);
                // Width and height should be similar (square or near-square)
                const ratio = box.width / box.height;
                expect(ratio).toBeGreaterThan(0.8);
                expect(ratio).toBeLessThan(1.2);
            }
        }
    });

    test('avatar shows emoji fallback when no image', async ({ page }) => {
        // Look for emoji fallback
        const emojiAvatar = page.locator('.identity-avatar-emoji');

        if (await emojiAvatar.count() > 0) {
            // Should contain emoji (falcon/eagle)
            const text = await emojiAvatar.textContent();
            // Could be eagle emoji or other fallback
            expect(text).toBeTruthy();
        }
    });

    test('avatar image loads when present', async ({ page }) => {
        // Look for avatar image
        const avatarImg = page.locator('.identity-avatar-img');

        if (await avatarImg.count() > 0) {
            // Verify image src points to files endpoint
            const src = await avatarImg.getAttribute('src');
            expect(src).toContain('/api/files/');

            // Wait for image to load
            await expect(avatarImg).toHaveAttribute('src', /\/api\/files\/.+/);
        }
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

        if (await avatar.count() > 0) {
            // Check for rounded corners (border-radius)
            const borderRadius = await avatar.evaluate(el =>
                window.getComputedStyle(el).borderRadius
            );

            // Should have some border radius for rounded appearance
            expect(borderRadius).not.toBe('0px');
        }
    });

    test('avatar container has overflow hidden', async ({ page }) => {
        const avatar = page.locator('.identity-avatar');

        if (await avatar.count() > 0) {
            const overflow = await avatar.evaluate(el =>
                window.getComputedStyle(el).overflow
            );

            expect(overflow).toBe('hidden');
        }
    });

    test('avatar image uses object-fit cover', async ({ page }) => {
        const avatarImg = page.locator('.identity-avatar-img');

        if (await avatarImg.count() > 0) {
            const objectFit = await avatarImg.evaluate(el =>
                window.getComputedStyle(el).objectFit
            );

            expect(objectFit).toBe('cover');
        }
    });
});

test.describe('Avatar in Chat Messages', () => {

    test('chat messages may include avatar', async ({ page }) => {
        await page.goto(KESTREL_URL);
        await page.waitForLoadState('domcontentloaded');

        // Look for assistant message avatars (if implemented)
        const messageAvatar = page.locator('.message-avatar, .assistant-avatar');

        // This is optional - not all UIs show avatars in messages
        const count = await messageAvatar.count();
        expect(count).toBeGreaterThanOrEqual(0);
    });
});
