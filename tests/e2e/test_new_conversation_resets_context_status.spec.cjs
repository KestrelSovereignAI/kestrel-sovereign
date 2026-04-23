/**
 * E2E: Starting a new conversation resets the context-status footer.
 *
 * Regression for: "when i start a new chat the context usage stays the same.
 * It still says 22 messages 22%". The footer was reading the previous
 * session's message count / utilization because ``startNewConversation``
 * set ``state.currentSessionId`` but never called ``updateContextStatus``.
 *
 * Runs against the live rookery on ``KESTREL_URL`` (default localhost:8888).
 */
const { test, expect } = require('@playwright/test');

const BASE_URL = process.env.KESTREL_URL || 'http://localhost:8888';
const AGENT = process.env.KESTREL_AGENT || 'Meridian';

async function getApiKey(request) {
    if (process.env.KESTREL_API_KEY) return process.env.KESTREL_API_KEY;
    try {
        const response = await request.get(`${BASE_URL}/api/auth/key`);
        if (response.ok()) return (await response.json()).key;
    } catch (e) {}
    return null;
}


test('context-status footer resets to 0 messages after startNewConversation', async ({ page, request }) => {
    const apiKey = await getApiKey(request);

    await page.goto(`${BASE_URL}/`);
    await page.evaluate((k) => {
        try { window.localStorage.setItem('kestrel_api_key', k); } catch (_) {}
    }, apiKey);
    await page.reload();

    await page.waitForFunction(
        () => typeof window.selectAgent === 'function',
        { timeout: 15000 },
    );
    await page.evaluate((a) => window.selectAgent(a), AGENT);

    // The footer lives inside panel-chat; the element is in the DOM but
    // not visible until the chat panel is active. Find it via JS (not
    // visibility-aware locator) and wait for the async status round-trip
    // to populate its text.
    await page.waitForSelector('#context-status', { state: 'attached', timeout: 15000 });
    await page.waitForFunction(
        () => {
            const el = document.getElementById('context-status');
            return el && /\d+\s*msgs/.test(el.textContent || '');
        },
        { timeout: 20000 },
    );

    // Trigger the canonical new-conversation flow. If the agent has prior
    // history, this is where the regression used to surface — the footer
    // kept showing the previous session's count.
    await page.evaluate(() => window.startNewConversation());

    // After the fix, the footer must re-poll /agent/context-status with the
    // newly-created session_id and settle to "0 msgs · 0%". Allow a short
    // window for the async round-trip to land.
    await page.waitForFunction(
        () => {
            const el = document.getElementById('context-status');
            if (!el) return false;
            const txt = (el.textContent || '').trim();
            // Accept "0 msgs · 0%" with optional whitespace and an icon prefix.
            return /0\s*msgs/.test(txt) && /\b0\s*%/.test(txt);
        },
        { timeout: 10000 },
    );
});
