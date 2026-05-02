/**
 * Kestrel Tasks & Activity Demo
 *
 * Playwright-scripted demo of the Tasks panel:
 *   Act 1: Empty tasks view (fresh agent)
 *   Act 2: Trigger a background task (chat message with tool use)
 *   Act 3: Task list populated
 *   Act 4: Filter by state
 *   Act 5: Activity Log view
 *
 * Run: demos/run.sh tasks
 *
 * Output (in demo-output/):
 *   - narration.md
 *   - NN-name.png screenshots (prefix from NarrationEngine)
 *   - video (.webm)
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
  demoSendMessage,
  navigateToPanel,
  dismissContextWarning,
} = require('../shared/demo_helpers.cjs');

const BASE_URL = process.env.KESTREL_URL || 'http://localhost:8900';
const OUTPUT_DIR = path.join(__dirname, 'demo-output');

const narrator = new NarrationEngine({ title: 'Kestrel Tasks & Activity — Transcript' });
let apiKey = null;

test.describe.serial('Kestrel Tasks Demo', () => {
  test.beforeAll(async ({ request }) => {
    if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    apiKey = await getApiKey(request, BASE_URL);
    // Approval modal is suppressed server-side via KESTREL_DEMO_SERVER=1
    // (set by demos/run.sh). See SecurityFeature._register_all_tools.
  });

  test.afterAll(() => {
    const narrationPath = path.join(OUTPUT_DIR, 'narration.md');
    fs.writeFileSync(narrationPath, narrator.toMarkdown(), 'utf-8');
    console.log(`[DEMO] Narration written to ${narrationPath}`);
    console.log(`[DEMO] Screenshots in ${OUTPUT_DIR}`);
  });

  test('Act 1: Empty Tasks view', async ({ page }) => {
    narrator.act(1, 'Empty Tasks');
    narrator.narrate(
      'The Tasks panel has two views — Background Tasks and Activity Log. A fresh ' +
      'agent has nothing in flight, so both are quiet.'
    );
    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await navigateToPanel(page, 'tasks');
    await demoPause(page, 2000);
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'tasks-empty');
  });

  test('Act 2: Trigger activity', async ({ page }) => {
    narrator.act(2, 'Trigger activity');
    narrator.narrate(
      'Send the agent a question that requires tool use. The chat message itself is ' +
      'quick, but the tool calls and LLM hops underneath populate the activity log.'
    );
    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await navigateToPanel(page, 'chat');
    await demoPause(page, 1000);

    await demoSendMessage(page,
      'What\'s today\'s date and what time is it? Use tools to find out.',
      60000,
    );
    await demoPause(page, 1500);
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'chat-triggered');
  });

  test('Act 3: Task list populated', async ({ page }) => {
    narrator.act(3, 'Task list populated');
    narrator.narrate(
      'Return to Tasks. Depending on what the agent did, the Background Tasks queue ' +
      'may show a completed entry. Either way, the Activity Log has fresh events.'
    );
    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await navigateToPanel(page, 'tasks');
    await demoPause(page, 2000);

    try {
      const refresh = page.locator('#btn-refresh-tasks');
      await refresh.click();
      await demoPause(page, 1500);
    } catch { /* refresh optional */ }

    try { await highlightElement(page, '#task-list'); } catch { /* best effort */ }
    await demoPause(page, 1000);
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'task-list');
    try { await clearHighlights(page); } catch { /* best effort */ }
  });

  test('Act 4: Filter by state', async ({ page }) => {
    narrator.act(4, 'Filter by state');
    narrator.narrate(
      'The state filter limits the view to working / completed / failed. Operators ' +
      'use this to zero in on active work or to audit recent failures.'
    );
    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await navigateToPanel(page, 'tasks');
    await demoPause(page, 1500);

    try {
      const filter = page.locator('#task-filter');
      await filter.selectOption('completed');
      await demoPause(page, 1500);
      narrator.narrate('Filter = Completed');
      await demoScreenshot(narrator, page, OUTPUT_DIR, 'filter-completed');

      await filter.selectOption('failed');
      await demoPause(page, 1500);
      narrator.narrate('Filter = Failed');
      await demoScreenshot(narrator, page, OUTPUT_DIR, 'filter-failed');

      await filter.selectOption('');
      await demoPause(page, 500);
    } catch (e) {
      narrator.narrate(`[narrator] Filter interaction issue: ${e.message}`);
    }
  });

  test('Act 5: Activity Log', async ({ page }) => {
    narrator.act(5, 'Activity Log');
    narrator.narrate(
      'Switch to Activity Log. Tool calls, LLM invocations, feature executions — ' +
      'everything the agent did, timestamped. The raw truth layer that Metrics ' +
      'aggregates and Spawn summarizes.'
    );
    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await navigateToPanel(page, 'tasks');
    await demoPause(page, 1000);

    try {
      const activityBtn = page.locator('#btn-view-activity');
      await activityBtn.click();
      await demoPause(page, 2000);
      await demoScreenshot(narrator, page, OUTPUT_DIR, 'activity-log');
    } catch (e) {
      narrator.narrate(`[narrator] Activity Log switch issue: ${e.message}`);
    }

    try {
      await page.locator('#btn-view-tasks').click();
      await demoPause(page, 1500);
    } catch { /* ignore */ }
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'bookend');
  });
});
