// @ts-check
const path = require('path');
const { buildDemoConfig } = require('@kestrel/flight');

/**
 * Kestrel Sovereign Technical Demo Configuration
 *
 * Separate config for the scripted demo (issue #133, Track A).
 * Always records video, uses slowMo for viewer clarity, larger viewport.
 *
 * Run: npx playwright test --config=demo_config.cjs
 * Env: DEMO_SLOWMO=200, KESTREL_URL, KESTREL_API_KEY
 */
module.exports = buildDemoConfig(
  [{ name: 'demo', testMatch: 'demo_technical.demo.cjs', timeout: 600000 }],
  {
    baseURL: process.env.KESTREL_URL || 'http://localhost:8888',
    outputDir: path.join(__dirname, 'demo-output', 'playwright'),
  },
);
