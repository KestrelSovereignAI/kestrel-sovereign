// @ts-check
const { defineConfig, devices } = require('@playwright/test');
const path = require('path');

/**
 * Kestrel Falconer Product Demo Configuration
 *
 * Separate config for the Falconer product demo.
 * Always records video, uses slowMo for viewer clarity, larger viewport.
 *
 * Run: npx playwright test --config=demo_falconer_config.cjs
 * Env: DEMO_SLOWMO=200, KESTREL_URL, KESTREL_API_KEY
 */
module.exports = defineConfig({
  testDir: './',
  testMatch: 'demo_falconer.demo.cjs',
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [['html', { outputFolder: 'demo-falconer-report' }], ['list']],
  timeout: 300000, // 5 minutes — morning signal + dispatch involve live GitHub/LLM calls

  use: {
    baseURL: process.env.KESTREL_URL || 'http://localhost:8888',

    // Always capture video + trace for the demo
    video: 'on',
    trace: 'on',
    screenshot: 'off', // Manual screenshots at specific moments

    // Slow motion so actions are visible in recording
    launchOptions: {
      slowMo: parseInt(process.env.DEMO_SLOWMO || '150', 10),
    },

    actionTimeout: 30000,

    // Larger viewport for readability
    viewport: { width: 1440, height: 900 },
  },

  projects: [
    {
      name: 'falconer-demo',
      use: { ...devices['Desktop Chrome'] },
    },
  ],

  outputDir: path.join(__dirname, 'demo-output-falconer', 'playwright'),
});
