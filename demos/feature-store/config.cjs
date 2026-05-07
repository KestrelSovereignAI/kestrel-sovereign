// @ts-check
const path = require('path');
const { buildDemoConfig } = require('@kestrel/flight');

/**
 * Kestrel Feature Store Demo — Playwright Configuration
 *
 * Narrated demo of the Feature Store panel: browse, search, drill into a
 * feature, inspect its skills.
 *
 * Run: kestrel demo run feature-store
 * Env: DEMO_SLOWMO=200, KESTREL_URL (set by run.sh)
 */
module.exports = buildDemoConfig(
  [{ name: 'feature-store-demo', testMatch: 'demo.cjs', timeout: 300000 }],
  {
    baseURL: process.env.KESTREL_URL || 'http://localhost:8900',
    outputDir: path.join(__dirname, 'demo-output', 'playwright'),
  },
);
