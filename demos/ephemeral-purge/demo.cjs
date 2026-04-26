/**
 * EPHEMERAL Purge Vignette — defense-in-depth for privacy mode (#767)
 *
 * Why this exists:
 *   EPHEMERAL is a contract: nothing is stored, local LLM only.  The
 *   privacy wrapper is supposed to gate every persistent write.  But a
 *   gate has bugs, integrations leak, and a feature added next week may
 *   not know about the wrapper.  When the user toggles OUT of EPHEMERAL,
 *   the agent runs a hard-purge of anything that did slip through —
 *   conversation rows, graph nodes — owned by this agent.  That's
 *   defense-in-depth: even if the gate failed, the leak is scrubbed
 *   before the user re-enters a persistent mode.
 *
 * Beats:
 *   1. NORMAL state — badge shows the current mode, conversation persists.
 *   2. Switch to EPHEMERAL via the privacy badge.
 *   3. Send a message in EPHEMERAL — it renders but doesn't persist.
 *   4. Switch back to NORMAL — the purge fires.
 *   5. Audit log shows the purge entry.
 *
 * Run: demos/run.sh ephemeral-purge
 */
const { test } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

const {
  NarrationEngine,
  demoScreenshot,
  demoPause,
  highlightElement,
  clearHighlights,
  getApiKey,
  authHeaders,
  demoGoto,
  demoSendMessage,
  navigateToPanel,
  dismissContextWarning,
  scrollChatToBottom,
} = require('../shared/demo_helpers.cjs');

const BASE_URL = process.env.KESTREL_URL || 'http://localhost:8900';
const OUTPUT_DIR = path.join(__dirname, 'demo-output');

const narrator = new NarrationEngine({
  title: 'Kestrel Sovereign — EPHEMERAL Purge Vignette Transcript',
});
let apiKey = null;

async function setPrivacyModeViaApi(request, mode) {
  return await request.post(`${BASE_URL}/agent/privacy-mode`, {
    headers: { ...authHeaders(apiKey), 'Content-Type': 'application/json' },
    data: { mode },
  });
}

test.describe.serial('EPHEMERAL Purge Vignette', () => {
  test.beforeAll(async ({ request }) => {
    if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    apiKey = await getApiKey(request, BASE_URL);
    narrator.act(0, 'Setup');
    narrator.narrate(apiKey ? 'API key acquired' : 'No API key (public mode)');
    // Ensure we start in NORMAL — prior runs might have left a different mode.
    try { await setPrivacyModeViaApi(request, 'normal'); } catch { /* best-effort */ }
  });

  test.afterAll(async ({ request }) => {
    // Restore NORMAL on the way out so the demo agent isn't left in EPHEMERAL.
    try { await setPrivacyModeViaApi(request, 'normal'); } catch { /* best-effort */ }
    const narrationPath = path.join(OUTPUT_DIR, 'narration.md');
    fs.writeFileSync(narrationPath, narrator.toMarkdown(), 'utf-8');
    console.log(`[DEMO] Narration written to ${narrationPath}`);
  });

  // -------------------------------------------------------------------------
  // Beat 1 — In NORMAL.  Show the privacy badge.
  // -------------------------------------------------------------------------
  test('Beat 1: NORMAL — the persistent baseline', async ({ page }) => {
    narrator.act(1, 'NORMAL baseline');
    narrator.narrate(
      'The privacy badge in the chat header shows the current mode.  In ' +
      'NORMAL the wrapper persists everything to disk: conversation history, ' +
      'memory pins, graph nodes.'
    );

    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await navigateToPanel(page, 'chat');
    await dismissContextWarning(page);
    await demoPause(page, 1500);
    try {
      await highlightElement(page, '#chat-privacy-indicator', 'Current privacy mode');
      await demoPause(page, 1500);
    } catch { /* best-effort */ }
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'normal-baseline');
    await clearHighlights(page);
  });

  // -------------------------------------------------------------------------
  // Beat 2 — Open the privacy dropdown, switch to EPHEMERAL.
  // -------------------------------------------------------------------------
  test('Beat 2: Switch to EPHEMERAL', async ({ page }) => {
    narrator.act(2, 'Flip to EPHEMERAL');
    narrator.narrate(
      'Clicking the badge opens the privacy selector.  EPHEMERAL is the ' +
      'tightest mode: nothing stored, local LLM only.  The wrapper is ' +
      'supposed to silently drop every write — but bugs happen.'
    );

    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await navigateToPanel(page, 'chat');
    await dismissContextWarning(page);
    await demoPause(page, 1000);

    // Open the dropdown via the function exposed on window.
    const opened = await page.evaluate(() => {
      if (typeof window.showPrivacySelector === 'function') {
        window.showPrivacySelector();
        return true;
      }
      return false;
    });
    if (!opened) {
      narrator.narrate('showPrivacySelector not exposed — UI may have changed.');
    }
    await demoPause(page, 1500);
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'privacy-dropdown-open');

    // Pick EPHEMERAL — the dropdown rows have data attributes / inline handlers
    // that ultimately call API.setPrivacyMode.  Calling it directly is the
    // most reliable path for a vignette.
    await page.evaluate(async () => {
      const select = document.querySelector('select[id*="privacy"], #privacy-mode-select');
      // Modern UI uses a custom dropdown rendered by showPrivacySelector — find
      // the EPHEMERAL row and click it.
      const rows = document.querySelectorAll('#privacy-dropdown [data-mode]');
      for (const row of rows) {
        if (row.dataset.mode === 'ephemeral') { row.click(); return; }
      }
      // Fallback — call the API directly via api_client (loaded as a module).
      const mod = await import('/static/js/api_client.mjs').catch(() => null);
      if (mod?.default?.setPrivacyMode) await mod.default.setPrivacyMode('ephemeral');
    });
    await demoPause(page, 2500);
    try {
      await highlightElement(page, '#chat-privacy-indicator', 'EPHEMERAL is active');
      await demoPause(page, 1500);
    } catch { /* best-effort */ }
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'now-in-ephemeral');
    await clearHighlights(page);
    narrator.narrate('Badge updates to EPHEMERAL — wrapper now drops persistent writes.', { callout: true });
  });

  // -------------------------------------------------------------------------
  // Beat 3 — Send a message while in EPHEMERAL.  No persistent row created.
  // -------------------------------------------------------------------------
  test('Beat 3: A conversation that leaves no trace', async ({ page }) => {
    narrator.act(3, 'Ephemeral conversation');
    narrator.narrate(
      'A message sent in EPHEMERAL renders in the chat — but the wrapper ' +
      'rejects the persistent write.  Nothing lands in conversation_history, ' +
      'no graph node is added, no memory pin is created.'
    );

    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await navigateToPanel(page, 'chat');
    await dismissContextWarning(page);
    await demoPause(page, 1000);

    await demoSendMessage(
      page,
      'A short reply, please — this conversation should leave no trace.',
      90000,
    );
    await scrollChatToBottom(page);
    await demoPause(page, 1500);
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'ephemeral-conversation');
  });

  // -------------------------------------------------------------------------
  // Beat 4 — Switch back to NORMAL.  This triggers _purge_ephemeral_leaks().
  // -------------------------------------------------------------------------
  test('Beat 4: Exit EPHEMERAL — the purge fires', async ({ page, request }) => {
    narrator.act(4, 'Defense-in-depth fires');
    narrator.narrate(
      'Switching out of EPHEMERAL is the trigger.  The agent runs ' +
      '_purge_ephemeral_leaks() which hard-deletes any conversation rows ' +
      'or graph nodes owned by this agent — the leaks the wrapper missed.  ' +
      'A purge audit row lands in security_audit_log with the row counts.'
    );

    // The mode change goes through the wrapper transition handler that
    // invokes _purge_ephemeral_leaks().  Use the API directly so we don't
    // depend on the dropdown UI being open.
    const resp = await setPrivacyModeViaApi(request, 'normal');
    let body = null;
    try { body = await resp.json(); } catch { /* not JSON */ }
    if (resp.status() === 200) {
      narrator.narrate(`Mode change accepted: ${JSON.stringify(body)}`);
    } else {
      narrator.narrate(`Mode change returned ${resp.status()}: ${JSON.stringify(body)}`);
    }

    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await navigateToPanel(page, 'chat');
    await demoPause(page, 1500);
    try {
      await highlightElement(page, '#chat-privacy-indicator', 'Back in NORMAL');
      await demoPause(page, 1500);
    } catch { /* best-effort */ }
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'back-to-normal');
    await clearHighlights(page);
  });

  // -------------------------------------------------------------------------
  // Beat 5 — Audit log shows the purge entry.
  // -------------------------------------------------------------------------
  test('Beat 5: Bookend — audit trail of the purge', async ({ page }) => {
    narrator.act(5, 'Bookend');
    narrator.narrate(
      'Every defense-in-depth purge writes an audit row that names the row ' +
      'counts scrubbed.  An operator can answer "did the leak rail fire?" ' +
      'and "what did it remove?" without reading code.'
    );

    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await navigateToPanel(page, 'security');
    await demoPause(page, 1500);
    // Audit log is the last section in the Security panel — load it via the
    // exposed Security.loadAuditLog() and scroll into view so the screenshot
    // captures the audit surface, not the unrelated Permission Tree above.
    await page.evaluate(() => {
      if (window.Security?.loadAuditLog) window.Security.loadAuditLog();
    });
    await demoPause(page, 1500);
    try {
      await page.locator('#security-audit-log').scrollIntoViewIfNeeded();
      await demoPause(page, 800);
      await highlightElement(page, '#security-audit-log', 'Security audit log');
      await demoPause(page, 1500);
    } catch { /* best-effort */ }
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'audit-log-purge');
    await clearHighlights(page);
    narrator.narrate('Demo complete — privacy is a layered defense, not a single gate.', { callout: true });
  });
});
