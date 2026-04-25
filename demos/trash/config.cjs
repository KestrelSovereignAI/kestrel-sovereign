// @ts-check
const path = require('path');
const { buildDemoConfig } = require('@kestrel/flight');

/**
 * Trash Vignette — Playwright Configuration
 *
 * Soft-delete (#763) + Trash sub-view (#765) end-to-end.
 *
 * Run via the canonical isolating runner:
 *   demos/run.sh trash
 */
module.exports = buildDemoConfig(
  [{ name: 'trash-demo', testMatch: 'demo.cjs', timeout: 300000 }],
  {
    baseURL: process.env.KESTREL_URL || 'http://localhost:8900',
    outputDir: path.join(__dirname, 'demo-output', 'playwright'),
  },
);
