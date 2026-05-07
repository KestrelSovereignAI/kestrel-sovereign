/**
 * Vignette Template
 *
 * Copy this whole directory to demos/<feature-name>/ and rewrite the beats.
 * A vignette covers ONE feature with 5–8 screenshots.
 *
 * Run: kestrel demo run <feature-name>
 *
 * Output (in demo-output/):
 *   - narration.md    — generated transcript with screenshot refs
 *   - NN-name.png     — screenshots prefixed by NarrationEngine
 *   - video (.webm)   — Playwright recording
 *
 * This is a DEMO, not a test — never aborts. Failures get narrated and the
 * next beat runs. CI gates live in tests/e2e/test_*.spec.cjs.
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
  demoGoto,
  navigateToPanel,
  dismissContextWarning,
} = require('../shared/demo_helpers.cjs');

const BASE_URL = process.env.KESTREL_URL || 'http://localhost:8900';
const OUTPUT_DIR = path.join(__dirname, 'demo-output');

const narrator = new NarrationEngine({
  title: 'Kestrel Sovereign — TEMPLATE Vignette Transcript',
});
let apiKey = null;

test.describe.serial('TEMPLATE Vignette', () => {
  test.beforeAll(async ({ request }) => {
    if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    apiKey = await getApiKey(request, BASE_URL);
    narrator.act(0, 'Setup');
    narrator.narrate(apiKey ? 'API key acquired' : 'No API key (public mode)');
  });

  test.afterAll(() => {
    const narrationPath = path.join(OUTPUT_DIR, 'narration.md');
    fs.writeFileSync(narrationPath, narrator.toMarkdown(), 'utf-8');
    console.log(`[DEMO] Narration written to ${narrationPath}`);
    console.log(`[DEMO] Screenshots in ${OUTPUT_DIR}`);
  });

  // -------------------------------------------------------------------------
  // Beat 1: Land somewhere meaningful for this feature
  // -------------------------------------------------------------------------
  test('Beat 1: Open the panel where the feature lives', async ({ page }) => {
    narrator.act(1, 'The starting view');
    narrator.narrate(
      'Replace this with one sentence about why the feature exists — not what ' +
      'the screen shows. The screenshot can speak for what is visible.'
    );
    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await navigateToPanel(page, 'chat');  // change to your panel
    await demoPause(page, 1500);
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'starting-view');
  });

  // -------------------------------------------------------------------------
  // Beat 2: Take the headline action
  // -------------------------------------------------------------------------
  test('Beat 2: Trigger the feature', async ({ page }) => {
    narrator.act(2, 'Trigger');
    narrator.narrate('Describe the user intent driving this action.');
    // Drive a click, fill a form, call an API — whatever your feature does.
    await demoPause(page, 1000);
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'after-trigger');
  });

  // -------------------------------------------------------------------------
  // Beat 3: Show the consequence
  // -------------------------------------------------------------------------
  test('Beat 3: See the result', async ({ page }) => {
    narrator.act(3, 'Result');
    narrator.narrate('What the user now sees that they did not before.');
    try {
      await highlightElement(page, '#some-result-region', 'The new state');
      await demoPause(page, 1500);
    } catch { /* element may not exist — keep going */ }
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'result');
    await clearHighlights(page);
  });

  // -------------------------------------------------------------------------
  // Beat 4: Bookend (optional — shows the user can recover or move on)
  // -------------------------------------------------------------------------
  test('Beat 4: Bookend', async ({ page }) => {
    narrator.act(4, 'Bookend');
    narrator.narrate(
      'Optional closing shot — the surface returned to a clean state, ' +
      'or the user moved on. Helps the demo feel like a complete loop.'
    );
    await demoPause(page, 1000);
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'bookend');
  });
});
