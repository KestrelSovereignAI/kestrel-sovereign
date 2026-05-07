/**
 * Privacy Modes Vignette — the 5-level spectrum
 *
 * Why this exists:
 *   "Privacy" is not a checkbox.  Kestrel ships five privacy modes that
 *   trade storage and LLM-routing differently.  This vignette walks the
 *   user across the whole spectrum so they can see what changes when they
 *   move from EPHEMERAL to PUBLIC.
 *
 *   EPHEMERAL  — nothing stored, local LLM only.
 *   ISOLATED   — temporary session storage, local LLM only.
 *   ANONYMOUS  — scrubbed storage (PII removed), cloud LLM allowed.
 *   NORMAL     — standard persistent storage.
 *   PUBLIC     — shareable and exportable.
 *
 *   Source of truth: kestrel_sovereign/privacy.py — PrivacyMode enum.
 *
 * Beats:
 *   1. The selector — all five modes side by side.
 *   2-6. Walk each mode in turn (EPHEMERAL → ISOLATED → ANONYMOUS → NORMAL → PUBLIC).
 *   7. Bookend in NORMAL.
 *
 * Run: kestrel demo run privacy-modes
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
  navigateToPanel,
  dismissContextWarning,
} = require('../shared/demo_helpers.cjs');

const BASE_URL = process.env.KESTREL_URL || 'http://localhost:8900';
const OUTPUT_DIR = path.join(__dirname, 'demo-output');

// The full spectrum, in increasing-openness order.
const MODES = [
  { key: 'ephemeral', label: 'EPHEMERAL', tagline: 'Nothing stored, local LLM only' },
  { key: 'isolated',  label: 'ISOLATED',  tagline: 'Temporary session storage, local LLM only' },
  { key: 'anonymous', label: 'ANONYMOUS', tagline: 'Scrubbed storage (PII removed), cloud LLM allowed' },
  { key: 'normal',    label: 'NORMAL',    tagline: 'Standard persistent storage' },
  { key: 'public',    label: 'PUBLIC',    tagline: 'Shareable and exportable' },
];

const narrator = new NarrationEngine({
  title: 'Kestrel Sovereign — Privacy Modes Vignette Transcript',
});
let apiKey = null;

async function setMode(request, mode) {
  return await request.post(`${BASE_URL}/agent/privacy-mode`, {
    headers: { ...authHeaders(apiKey), 'Content-Type': 'application/json' },
    data: { mode },
  });
}

test.describe.serial('Privacy Modes Vignette', () => {
  test.beforeAll(async ({ request }) => {
    if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    apiKey = await getApiKey(request, BASE_URL);
    narrator.act(0, 'Setup');
    narrator.narrate(apiKey ? 'API key acquired' : 'No API key (public mode)');
    try { await setMode(request, 'normal'); } catch { /* best-effort */ }
  });

  test.afterAll(async ({ request }) => {
    try { await setMode(request, 'normal'); } catch { /* best-effort */ }
    const narrationPath = path.join(OUTPUT_DIR, 'narration.md');
    fs.writeFileSync(narrationPath, narrator.toMarkdown(), 'utf-8');
    console.log(`[DEMO] Narration written to ${narrationPath}`);
  });

  // -------------------------------------------------------------------------
  // Beat 1 — Open the selector.  All 5 modes visible at once.
  // -------------------------------------------------------------------------
  test('Beat 1: The five-mode selector', async ({ page }) => {
    narrator.act(1, 'The five-mode selector');
    narrator.narrate(
      'Privacy is a spectrum, not a checkbox.  The selector shows the same ' +
      'agent\'s five settings, ordered from "leaves no trace" on the left ' +
      'to "shareable on the public web" on the right.'
    );

    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await navigateToPanel(page, 'chat');
    await dismissContextWarning(page);
    await demoPause(page, 1500);

    await page.evaluate(() => {
      if (typeof window.showPrivacySelector === 'function') {
        window.showPrivacySelector();
      }
    });
    await demoPause(page, 1500);
    try {
      await highlightElement(page, '#privacy-dropdown', 'Five privacy modes');
      await demoPause(page, 1500);
    } catch { /* element may not exist */ }
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'selector-all-modes');
    await clearHighlights(page);
    // Close the dropdown for subsequent beats.
    await page.evaluate(() => document.getElementById('privacy-dropdown')?.remove());
  });

  // -------------------------------------------------------------------------
  // Beats 2-6 — Walk each mode.
  // -------------------------------------------------------------------------
  for (let i = 0; i < MODES.length; i++) {
    const mode = MODES[i];
    const beatNum = i + 2;
    test(`Beat ${beatNum}: ${mode.label} — ${mode.tagline}`, async ({ page, request }) => {
      narrator.act(beatNum, `${mode.label}: ${mode.tagline}`);
      narrator.narrate(
        `${mode.label} — ${mode.tagline}.  Source of truth: PrivacyMode enum ` +
        `in kestrel_sovereign/privacy.py.  The badge in the chat header is ` +
        `the user\'s constant signal of which contract is in force.`
      );

      const resp = await setMode(request, mode.key);
      if (resp.status() !== 200) {
        let body = null; try { body = await resp.json(); } catch {}
        narrator.narrate(`Mode change returned ${resp.status()}: ${JSON.stringify(body)}`);
      }

      await demoGoto(page, BASE_URL, apiKey);
      await dismissContextWarning(page);
      await navigateToPanel(page, 'chat');
      await dismissContextWarning(page);
      await demoPause(page, 1500);

      try {
        await highlightElement(page, '#chat-privacy-indicator', `${mode.label} mode`);
        await demoPause(page, 1500);
      } catch { /* best-effort */ }
      const slug = String(beatNum).padStart(2, '0') + '-' + mode.key;
      await demoScreenshot(narrator, page, OUTPUT_DIR, slug);
      await clearHighlights(page);
    });
  }

  // -------------------------------------------------------------------------
  // Beat 7 — Bookend in NORMAL.
  // -------------------------------------------------------------------------
  test('Beat 7: Bookend — back in NORMAL', async ({ page, request }) => {
    narrator.act(7, 'Bookend');
    narrator.narrate(
      'A demo agent should never be left in EPHEMERAL or PUBLIC after a ' +
      'tour — both have user-visible side effects.  The vignette restores ' +
      'NORMAL on exit and shows the badge so the operator can confirm.'
    );

    try { await setMode(request, 'normal'); } catch { /* best-effort */ }
    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await navigateToPanel(page, 'chat');
    await demoPause(page, 1500);
    try {
      await highlightElement(page, '#chat-privacy-indicator', 'Restored to NORMAL');
      await demoPause(page, 1500);
    } catch { /* best-effort */ }
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'bookend-normal');
    await clearHighlights(page);
    narrator.narrate('Demo complete — five contracts, one toggle, every state visible.', { callout: true });
  });
});
