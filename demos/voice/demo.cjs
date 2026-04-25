/**
 * Kestrel Sovereign — Voice Demo (Squack II, epic #721)
 *
 * Narrated demo of the voice UI that lights up after Phases 1-4 of the
 * Squack II epic merged. Beats:
 *
 *   Beat 1:  Console loads; mic button is in the chat header.
 *   Beat 2:  Voice picker opens; discovered voices are listed.
 *   Beat 3:  User picks a voice + writes a session directive.
 *   Beat 4:  Click the mic — session engages.
 *   Beat 5:  Path badge shows the active route (Realtime or Pipeline).
 *   Beat 6:  Transcript drawer renders the user + agent turns live.
 *   Beat 7:  Esc-to-stop returns to idle.
 *   Beat 8:  Voice picker survives a refresh (localStorage persistence).
 *
 * Run: cd demos/voice && npx playwright test --config=config.cjs
 *
 * Output (in demo-output/):
 *   - narration.md   — timestamped transcript with screenshot references
 *   - NN-name.png    — screenshots at each narrative beat
 *   - video (.webm)  — full browser recording
 *
 * This is a DEMO, not a test. Failures are narrated gracefully — the
 * intent is to capture the visible behavior for review and video editing,
 * not to gate CI. CI gates live in tests/e2e/test_voice_ui.spec.cjs which
 * mocks the network boundary so the spec runs without OpenAI / ElevenLabs
 * keys.
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
} = require('../shared/demo_helpers.cjs');

const BASE_URL = process.env.KESTREL_URL || 'http://localhost:8888';
const OUTPUT_DIR = path.join(__dirname, 'demo-output');

const narrator = new NarrationEngine({ title: 'Kestrel Sovereign — Voice Demo Transcript' });
let apiKey = null;

test.describe.serial('Kestrel Voice Demo', () => {
  test.beforeAll(async ({ request }) => {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    narrator.act(0, 'Setup');
    try {
      const resp = await request.get(`${BASE_URL}/health`);
      const data = await resp.json();
      narrator.narrate(`Server health: ${data.status}, agent_initialized: ${data.agent_initialized}`);
    } catch (e) {
      narrator.narrate(`Server health check failed: ${e.message} — demo may not work`);
    }
    apiKey = await getApiKey(request, BASE_URL);
    narrator.narrate(apiKey ? 'API key acquired' : 'No API key (public mode)');
  });

  test.afterAll(async () => {
    const transcript = narrator.render();
    fs.writeFileSync(path.join(OUTPUT_DIR, 'narration.md'), transcript);
  });

  test('Voice end-to-end', async ({ page }) => {
    test.setTimeout(600000);

    // Beat 1: Console loads, mic button mounts.
    narrator.act(1, 'Mic button arrives in the chat header');
    await demoGoto(page, BASE_URL, narrator);
    await page.waitForLoadState('domcontentloaded');
    await page.waitForSelector('#voice-toggle-btn', { timeout: 15000 });
    await highlightElement(page, '#voice-toggle-btn');
    await demoScreenshot(page, OUTPUT_DIR, '01-mic-button', narrator);
    await clearHighlights(page);

    // Beat 2: Open voice picker (right-click).
    narrator.act(2, 'Voice picker opens, discovered voices listed');
    await page.locator('#voice-toggle-btn').click({ button: 'right' });
    await page.waitForSelector('#voice-picker-modal:not([hidden])', { timeout: 5000 });
    await demoPause(800, narrator);
    await highlightElement(page, '#voice-picker-select');
    await demoScreenshot(page, OUTPUT_DIR, '02-voice-picker', narrator);
    await clearHighlights(page);

    // Beat 3: Pick a voice + directive.
    narrator.act(3, 'User picks a voice and writes a session directive');
    const select = page.locator('#voice-picker-select');
    const optionCount = await select.locator('option').count();
    if (optionCount > 0) {
      await select.selectOption({ index: 0 });
    }
    await page.locator('#voice-picker-instructions').fill(
      "Speak warmly, like reading a children's story by lamplight.",
    );
    await demoScreenshot(page, OUTPUT_DIR, '03-picker-filled', narrator);
    await page.locator('#voice-picker-save').click();
    await page.waitForSelector('#voice-picker-modal[hidden]', { timeout: 3000 });

    // Beat 4: Click mic — session engages.
    narrator.act(4, 'Mic clicked, session engages');
    await page.locator('#voice-toggle-btn').click();
    await page.waitForSelector('#voice-drawer:not([hidden])', { timeout: 10000 });
    await demoPause(800, narrator);
    await demoScreenshot(page, OUTPUT_DIR, '04-session-engaged', narrator);

    // Beat 5: Path badge shows the active route.
    narrator.act(5, 'Path badge displays the active route (Realtime or Pipeline)');
    const badgeText = await page.locator('.kestrel-voice-path-badge').textContent();
    narrator.narrate(`Path badge: ${badgeText || '(empty)'} — depends on the resolver's decision given the active LLM + privacy mode.`);
    await highlightElement(page, '.kestrel-voice-path-badge');
    await demoScreenshot(page, OUTPUT_DIR, '05-path-badge', narrator);
    await clearHighlights(page);

    // Beat 6: Transcript drawer (rendering depends on real STT/agent activity).
    narrator.act(6, 'Transcript drawer hosts live user + agent turns');
    await highlightElement(page, '.kestrel-voice-transcript');
    await demoPause(1500, narrator);
    await demoScreenshot(page, OUTPUT_DIR, '06-transcript', narrator);
    await clearHighlights(page);

    // Beat 7: Esc returns to idle.
    narrator.act(7, 'Esc returns to idle');
    await page.locator('body').click({ position: { x: 0, y: 0 } });
    await page.keyboard.press('Escape');
    await page.waitForFunction(() => {
      const btn = document.getElementById('voice-toggle-btn');
      return btn && btn.dataset.state === 'idle';
    }, { timeout: 5000 });
    await demoScreenshot(page, OUTPUT_DIR, '07-idle-after-esc', narrator);

    // Beat 8: Settings persist across reload (localStorage).
    narrator.act(8, 'Settings persist across reload');
    await page.reload();
    await page.waitForSelector('#voice-toggle-btn', { timeout: 15000 });
    await page.locator('#voice-toggle-btn').click({ button: 'right' });
    await page.waitForSelector('#voice-picker-modal:not([hidden])', { timeout: 5000 });
    const persistedDirective = await page.locator('#voice-picker-instructions').inputValue();
    narrator.narrate(`Persisted directive after reload: "${persistedDirective}"`);
    await demoScreenshot(page, OUTPUT_DIR, '08-persisted-settings', narrator);
    await page.locator('#voice-picker-cancel').click();

    narrator.narrate('End of voice demo.');
  });
});
