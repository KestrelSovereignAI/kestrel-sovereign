/**
 * Kestrel Metrics Dashboard Demo
 *
 * Playwright-scripted demo of the observability panel:
 *   Act 1: Empty dashboard (fresh agent, no activity)
 *   Act 2: Generate activity (chat messages, tool calls)
 *   Act 3: KPI cards light up
 *   Act 4: Timeline + duration + distribution charts
 *   Act 5: Errors table (if any) + bookend
 *
 * Run: demos/run.sh metrics
 *
 * Output (in demo-output/):
 *   - narration.md   — timestamped transcript
 *   - NN-name.png    — screenshots at each beat (numeric prefix added by NarrationEngine)
 *   - video (.webm)  — full browser recording
 *
 * DEMO, not test — never aborts. Failures are narrated gracefully.
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

const narrator = new NarrationEngine({ title: 'Kestrel Metrics Dashboard — Transcript' });
let apiKey = null;

test.describe.serial('Kestrel Metrics Dashboard Demo', () => {
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

  test('Act 1: Empty metrics dashboard', async ({ page }) => {
    narrator.act(1, 'Empty metrics dashboard');
    narrator.narrate(
      'A freshly spawned agent has done nothing yet. The Metrics panel reflects that — ' +
      'zero events, flat charts. Every number here comes from the agent\'s audit stream.'
    );
    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await navigateToPanel(page, 'metrics');
    await demoPause(page, 1500);
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'metrics-empty');
  });

  test('Act 2: Generate activity', async ({ page }) => {
    narrator.act(2, 'Generating activity');
    narrator.narrate(
      'Let\'s give the agent something to do. Each chat message triggers LLM calls, ' +
      'tool selection, and downstream events — all of which flow into the metrics stream.'
    );
    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await navigateToPanel(page, 'chat');
    await demoPause(page, 1000);

    await demoSendMessage(page, 'Briefly introduce yourself and describe what you can do.', 60000);
    await demoPause(page, 1000);
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'chat-activity-1');

    await demoSendMessage(page, 'What\'s today\'s date? Use whatever tools you need.', 60000);
    await demoPause(page, 1000);
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'chat-activity-2');
  });

  test('Act 3: KPI cards populated', async ({ page }) => {
    narrator.act(3, 'KPI cards populated');
    narrator.narrate(
      'Back in Metrics, the KPI cards now have real numbers. Event count, error rate, ' +
      'average tool duration — all live, all derived from the audit log.'
    );
    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await navigateToPanel(page, 'metrics');
    await demoPause(page, 2500);

    try { await highlightElement(page, '#metrics-kpi-cards', 'KPI cards'); } catch { /* best effort */ }
    await demoPause(page, 1500);
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'kpi-cards');
    try { await clearHighlights(page); } catch { /* best effort */ }
  });

  test('Act 4: Charts — timeline, duration, distribution', async ({ page }) => {
    narrator.act(4, 'Charts — timeline, duration, distribution');
    narrator.narrate(
      'Three charts complete the picture. Timeline shows events over time, colored by type. ' +
      'Duration shows p50/p95 per tool. Distribution breaks down event types.'
    );
    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await navigateToPanel(page, 'metrics');
    await demoPause(page, 2500);

    const charts = [
      { sel: '#metrics-timeline-chart',     name: 'timeline-chart',     caption: 'Timeline — events over time, colored by type' },
      { sel: '#metrics-duration-chart',     name: 'duration-chart',     caption: 'Duration distribution — p50 / p95 per tool' },
      { sel: '#metrics-distribution-chart', name: 'distribution-chart', caption: 'Event type distribution' },
    ];

    for (const c of charts) {
      try {
        await page.locator(c.sel).scrollIntoViewIfNeeded();
        narrator.narrate(c.caption);
        try { await highlightElement(page, c.sel); } catch { /* best effort */ }
        await demoPause(page, 800);
        await demoScreenshot(narrator, page, OUTPUT_DIR, c.name);
        try { await clearHighlights(page); } catch { /* best effort */ }
      } catch (e) {
        narrator.narrate(`[narrator] ${c.name} not captured: ${e.message}`);
      }
    }
  });

  test('Act 5: Errors table + final bookend', async ({ page }) => {
    narrator.act(5, 'Errors and bookend');
    narrator.narrate(
      'The errors table surfaces failures transparently — nothing is silently dropped. ' +
      'Close on the full dashboard: this is what operational accountability looks like.'
    );
    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await navigateToPanel(page, 'metrics');
    await demoPause(page, 2000);

    try { await page.locator('#metrics-errors-list').scrollIntoViewIfNeeded(); } catch { /* best effort */ }
    try { await highlightElement(page, '#metrics-errors-list'); } catch { /* best effort */ }
    await demoPause(page, 1200);
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'errors-table');
    try { await clearHighlights(page); } catch { /* best effort */ }

    await page.evaluate(() => window.scrollTo(0, 0));
    await demoPause(page, 1000);
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'dashboard-final');
  });
});
