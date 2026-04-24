// @ts-check
const path = require('path');
const { buildDemoConfig } = require('@kestrel/flight');

/**
 * Kestrel Spawn Demo — Playwright Configuration
 *
 * Narrated demo of the full spawn agent lifecycle (Issue #354).
 * Always records video, uses slowMo for viewer clarity, larger viewport.
 *
 * Run: cd demos/spawn && npx playwright test --config=config.cjs
 * Env: DEMO_SLOWMO=200, KESTREL_URL, KESTREL_API_KEY
 */
module.exports = buildDemoConfig(
  [{ name: 'spawn-demo', testMatch: 'demo.cjs', timeout: 600000 }],
  {
    baseURL: process.env.KESTREL_URL || 'http://localhost:8888',
    outputDir: path.join(__dirname, 'demo-output', 'playwright'),
  },
);
