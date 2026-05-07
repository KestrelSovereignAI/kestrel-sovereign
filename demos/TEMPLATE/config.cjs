// @ts-check
const path = require('path');
const { buildDemoConfig } = require('@kestrel/flight');

/**
 * Demo Template — Playwright Configuration
 *
 * Run via the canonical isolating runner:
 *   kestrel demo run <feature-name>
 *
 * Direct invocation (skips isolation; you've been warned):
 *   cd demos/<feature-name> && npx playwright test --config=config.cjs
 */
module.exports = buildDemoConfig(
  [{ name: 'TEMPLATE-demo', testMatch: 'demo.cjs', timeout: 300000 }],
  {
    baseURL: process.env.KESTREL_URL || 'http://localhost:8900',
    outputDir: path.join(__dirname, 'demo-output', 'playwright'),
  },
);
