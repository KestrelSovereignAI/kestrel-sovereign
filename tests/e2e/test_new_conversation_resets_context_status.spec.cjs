/**
 * E2E: Starting a new conversation resets the context-status footer.
 *
 * Regression for: "when i start a new chat the context usage stays the same.
 * It still says 22 messages 22%". The footer was reading the previous
 * session's message count / utilization because ``startNewConversation``
 * set ``state.currentSessionId`` but never called ``updateContextStatus``.
 *
 * Runs against the live multi_agent on ``KESTREL_URL`` (default localhost:8888).
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
    // not visible until the chat panel is active.  After issue #713 we
    // no longer populate the footer before a conversation is active —
    // selectAgent leaves ``state.currentSessionId = null`` and
    // ``updateContextStatus`` hides the indicator in that case, rather
    // than showing the agent's cross-session aggregate ("472 msgs ·
    // 100% Compact" on an empty pane).  So we wait for the element to
    // be ATTACHED but NOT for it to contain a count.
    await page.waitForSelector('#context-status', { state: 'attached', timeout: 15000 });

    // Trigger the canonical new-conversation flow.  Two regressions this
    // pins: (a) the original — after ``startNewConversation`` the footer
    // used to keep showing the PREVIOUS session's count; (b) #713 —
    // before the new conversation, the footer used to show the agent's
    // cross-session aggregate.  After both fixes: the footer starts
    // empty, then settles to "0 msgs · 0%" once the fresh session is
    // created.
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
