/**
 * Kestrel Feature Store Demo
 *
 * Playwright-scripted demo of the Feature Store panel:
 *   Act 1: Feature Store grid
 *   Act 2: Search + filter
 *   Act 3: Drill into a feature detail
 *   Act 4: Skills list highlight
 *   Act 5: Back to grid bookend
 *
 * Run: kestrel demo run feature-store
 *
 * Output (in demo-output/):
 *   - narration.md   — timestamped transcript
 *   - NN-name.png    — screenshots (prefix from NarrationEngine)
 *   - video (.webm)  — browser recording
 *
 * DEMO, not test — never aborts.
 */
const { test, expect } = require('@playwright/test');
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

const narrator = new NarrationEngine({ title: 'Kestrel Feature Store — Transcript' });
let apiKey = null;

test.describe.serial('Kestrel Feature Store Demo', () => {
  test.beforeAll(async ({ request }) => {
    if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    apiKey = await getApiKey(request, BASE_URL);
  });

  test.afterAll(() => {
    const narrationPath = path.join(OUTPUT_DIR, 'narration.md');
    fs.writeFileSync(narrationPath, narrator.toMarkdown(), 'utf-8');
    console.log(`[DEMO] Narration written to ${narrationPath}`);
    console.log(`[DEMO] Screenshots in ${OUTPUT_DIR}`);
  });

  test('Act 1: Feature Store panel', async ({ page }) => {
    narrator.act(1, 'The Feature Store');
    narrator.narrate(
      'The Feature Store lists every feature package available to this agent. ' +
      'Each card shows name, description, install status, and the number of skills ' +
      'the feature provides.'
    );
    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await navigateToPanel(page, 'features');
    await demoPause(page, 2500);
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'feature-grid');
  });

  test('Act 2: Search and filter', async ({ page }) => {
    narrator.act(2, 'Search and filter');
    narrator.narrate(
      'The search box spans name, description, tags, and skills. Filter chips narrow ' +
      'to installed or available. An agent can carry dozens of features — search keeps ' +
      'the store navigable.'
    );
    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await navigateToPanel(page, 'features');
    await demoPause(page, 1500);

    try {
      const search = page.locator('#feature-search');
      await search.click();
      await search.fill('memory');
      await demoPause(page, 1500);
      narrator.narrate('Search: "memory" — grid narrows');
      await demoScreenshot(narrator, page, OUTPUT_DIR, 'search-memory');

      await search.fill('');
      await demoPause(page, 500);

      const installedBtn = page.locator('.feature-filter-btn[data-filter="installed"]');
      await installedBtn.click();
      await demoPause(page, 1500);
      narrator.narrate('Filter: Installed — only active features');
      await demoScreenshot(narrator, page, OUTPUT_DIR, 'filter-installed');

      await page.locator('.feature-filter-btn[data-filter="all"]').click();
      await demoPause(page, 500);
    } catch (e) {
      narrator.narrate(`[narrator] Search/filter interaction issue: ${e.message}`);
    }
  });

  test('Act 3: Drill into a feature', async ({ page }) => {
    narrator.act(3, 'Feature detail');
    narrator.narrate(
      'Click any feature to see its full description, author, version, and — most ' +
      'importantly — the skills it provides. Skills are the atomic units the agent ' +
      'orchestrator actually invokes.'
    );
    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await navigateToPanel(page, 'features');
    await demoPause(page, 2000);

    try {
      const firstCard = page.locator('#feature-grid > *').first();
      await firstCard.click();
      await demoPause(page, 2000);
      await demoScreenshot(narrator, page, OUTPUT_DIR, 'feature-detail');
    } catch (e) {
      narrator.narrate(`[narrator] Could not open feature detail: ${e.message}`);
      await demoScreenshot(narrator, page, OUTPUT_DIR, 'feature-detail-fallback');
    }
  });

  test('Act 4: Skills highlighted', async ({ page }) => {
    narrator.act(4, 'Skills');
    narrator.narrate(
      'Skills are the procedural surface of a feature. Each has a JSON schema the LLM ' +
      'sees — that\'s how the orchestrator picks the right skill for a given user intent.'
    );
    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await navigateToPanel(page, 'features');
    await demoPause(page, 1500);

    try {
      const firstCard = page.locator('#feature-grid > *').first();
      await firstCard.click();
      await demoPause(page, 2000);

      const skillsHeading = page.getByText(/^Skills\s*\(\d+\)/i).first();
      if (await skillsHeading.count() > 0) {
        await skillsHeading.scrollIntoViewIfNeeded();
        await demoPause(page, 1500);
        narrator.narrate('Skills list — each skill has a schema the LLM reasons over');
        await demoScreenshot(narrator, page, OUTPUT_DIR, 'skills-highlighted');
      } else {
        narrator.narrate('This feature has no skills — not every feature needs them.');
        await demoScreenshot(narrator, page, OUTPUT_DIR, 'no-skills');
      }
    } catch (e) {
      narrator.narrate(`[narrator] Skills highlight issue: ${e.message}`);
    }
  });

  test('Act 5: Back to grid bookend', async ({ page }) => {
    narrator.act(5, 'Bookend');
    narrator.narrate(
      'Features ship independently, skills extend behavior, the agent\'s core stays ' +
      'stable. The Feature Store is the composition layer.'
    );
    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await navigateToPanel(page, 'features');
    await demoPause(page, 2000);
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'grid-bookend');
  });
});
