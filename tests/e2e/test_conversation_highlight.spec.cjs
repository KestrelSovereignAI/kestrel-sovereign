/**
 * Conversation-highlight E2E (#2380)
 *
 * Regression guard for the duplicate ``window.loadConversation`` bug: the
 * legacy history.js loader (no #2222 highlight wiring) clobbered the canonical
 * identity.js loader, so clicking a conversation row in the standalone console
 * loaded its messages but never moved the ``.active`` highlight off the
 * auto-loaded most-recent row.
 *
 * This spec seeds two conversations, lets the sidebar auto-load + highlight the
 * most recent, then CLICKS the older row and asserts the ``.active`` highlight
 * MOVES to the clicked row (exactly one active item, matching the clicked
 * session id). It FAILS on pre-fix main (legacy loader wins → highlight never
 * moves) and PASSES once the single canonical loader carries the highlight.
 *
 * NO MOCKS — real Kestrel agent, real database. Requires the server running on
 * localhost:8888.
 */
const { test, expect } = require('@playwright/test');

const BASE_URL = process.env.KESTREL_URL || 'http://localhost:8888';

async function getApiKey(request) {
    if (process.env.KESTREL_API_KEY) return process.env.KESTREL_API_KEY;
    try {
        const response = await request.get(`${BASE_URL}/api/auth/key`);
        if (response.ok()) return (await response.json()).key;
    } catch (e) {
        console.warn('Could not fetch API key:', e.message);
    }
    return null;
}

function authHeaders(apiKey) {
    return apiKey ? { 'X-API-Key': apiKey } : {};
}

test.describe('Conversation highlight follows the clicked row (#2380)', () => {
    let apiKey = null;

    test.beforeAll(async ({ request }) => {
        apiKey = await getApiKey(request);
    });

    test('clicking an older conversation moves .active to it', async ({ page, request }) => {
        const headers = authHeaders(apiKey);

        // --- Seed two distinct conversations (real invoke, no mocks) ---------
        const markerOld = 'HILITE_OLD_' + Date.now();
        const seedOld = await request.post(`${BASE_URL}/agent/invoke`, {
            data: { input: `Remember marker ${markerOld}` },
            headers,
            timeout: 90000,
        });
        expect(seedOld.status()).toBe(200);

        // Force a NEW session so there is an older + a newer conversation.
        const newConv = await request.post(`${BASE_URL}/api/conversations/new`, { headers });
        expect(newConv.status()).toBe(200);
        const session2Id = (await newConv.json()).session_id;

        const markerNew = 'HILITE_NEW_' + Date.now();
        const seedNew = await request.post(`${BASE_URL}/agent/invoke`, {
            data: { input: `Remember marker ${markerNew}`, session_id: session2Id },
            headers,
            timeout: 90000,
        });
        expect(seedNew.status()).toBe(200);

        // Resolve the older session id from the list (most-recent first).
        const convList = await request.get(`${BASE_URL}/api/conversations?limit=20`, { headers });
        expect(convList.status()).toBe(200);
        const conversations = (await convList.json()).conversations || [];
        // The two seeds above guarantee ≥2 conversations — a shortfall means
        // seeding misfiled (e.g. both turns landed in one session) and must
        // FAIL, not skip (no skip-on-failure).
        expect(conversations.length).toBeGreaterThanOrEqual(2);
        // The just-seeded newer conversation is most-recent; pick an OLDER row.
        const olderSessionId = conversations[1].session_id;

        // --- Drive the standalone console ------------------------------------
        // Chat is the console's default surface (post-#2149 layout — no nav
        // tab); the conversation list lives in the hidden `#conversations-pane`
        // sidebar, opened via the chat-header trigger (#2171/#2216).
        await page.goto(BASE_URL);
        await expect(page.locator('#conversations-toggle-btn')).toBeVisible({ timeout: 15000 });
        const pane = page.locator('#conversations-pane');
        if (!(await pane.isVisible())) {
            await page.click('#conversations-toggle-btn');
            await pane.waitFor({ state: 'visible', timeout: 10000 });
        }

        // Wait for the list to render and auto-load to highlight the most recent.
        const olderRow = page.locator(
            `#conversations-list .conversation-item[data-session-id="${olderSessionId}"]`,
        );
        await expect(olderRow).toHaveCount(1, { timeout: 10000 });

        // Pre-click, the older row must not be active. (Whether the auto-loaded
        // most-recent row is already highlighted is NOT asserted — pre-fix, the
        // clobbered legacy loader meant even auto-load painted no highlight, so
        // pinning "exactly one active" here would fail on setup rather than on
        // the behavior under test. The click below is the load-bearing check.)
        await expect(olderRow).not.toHaveClass(/\bactive\b/);

        // --- Click the older row: the highlight must MOVE to it --------------
        await olderRow.click();

        const activeItems = page.locator('.conversation-item.active');
        await expect(activeItems).toHaveCount(1, { timeout: 10000 });
        await expect(activeItems).toHaveAttribute('data-session-id', olderSessionId);
        await expect(olderRow).toHaveClass(/\bactive\b/);
    });
});
