// @ts-check
const { defineConfig, devices } = require('@playwright/test');
const path = require('path');

/**
 * Kestrel Spawn Demo — Playwright Configuration
 *
 * Narrated demo of the full spawn agent lifecycle (Issue #354).
 * Always records video, uses slowMo for viewer clarity, larger viewport.
 *
 * Run: cd demos/spawn && npx playwright test --config=demo_config.cjs
 * Env: DEMO_SLOWMO=200, KESTREL_URL, KESTREL_API_KEY
 */
module.exports = defineConfig({
  testDir: './',
  testMatch: 'demo_spawn.demo.cjs',
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [['html', { outputFolder: 'demo-report' }], ['list']],
  timeout: 600000, // 10 minutes — spawn lifecycle involves multiple LLM calls

  use: {
    baseURL: process.env.KESTREL_URL || 'http://localhost:8888',

    // Always capture video + trace for the demo
    video: 'on',
    trace: 'on',
    screenshot: 'off', // We take manual screenshots at specific moments

    // Slow motion so actions are visible in recording
    launchOptions: {
      slowMo: parseInt(process.env.DEMO_SLOWMO || '150', 10),
    },

    actionTimeout: 60000,

    // Larger viewport for readability
    viewport: { width: 1440, height: 900 },
  },

  projects: [
    {
      name: 'spawn-demo',
      use: { ...devices['Desktop Chrome'] },
    },
  ],

  outputDir: path.join(__dirname, 'demo-output', 'playwright'),
});
