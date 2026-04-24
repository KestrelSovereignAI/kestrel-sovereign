// @ts-check
const path = require('path');
const { buildDemoConfig } = require('@kestrel/flight');

/**
 * Kestrel Tasks & Activity Demo — Playwright Configuration
 *
 * Narrated demo of the Tasks panel: background task queue + real-time
 * activity log. Complements the spawn demo by showing the observability side
 * of asynchronous work.
 *
 * Run: demos/run.sh tasks
 */
module.exports = buildDemoConfig(
  [{ name: 'tasks-demo', testMatch: 'demo.cjs', timeout: 300000 }],
  {
    baseURL: process.env.KESTREL_URL || 'http://localhost:8900',
    outputDir: path.join(__dirname, 'demo-output', 'playwright'),
  },
);
