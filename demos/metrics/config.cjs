// @ts-check
const path = require('path');
const { buildDemoConfig } = require('@kestrel/flight');

/**
 * Kestrel Metrics Dashboard Demo — Playwright Configuration
 *
 * Narrated demo of the observability / KPI dashboard panel.
 *
 * Run: demos/run.sh metrics
 * Env: DEMO_SLOWMO=200, KESTREL_URL (set by run.sh), KESTREL_API_KEY (unset by run.sh)
 */
module.exports = buildDemoConfig(
  [{ name: 'metrics-demo', testMatch: 'demo.cjs', timeout: 300000 }],
  {
    baseURL: process.env.KESTREL_URL || 'http://localhost:8900',
    outputDir: path.join(__dirname, 'demo-output', 'playwright'),
  },
);
