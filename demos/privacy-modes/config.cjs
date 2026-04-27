// @ts-check
const path = require('path');
const { buildDemoConfig } = require('@kestrel/flight');

/**
 * Privacy Modes Vignette — Playwright Configuration
 *
 * Tour the 5 privacy modes (EPHEMERAL → PUBLIC) — same agent, same chat,
 * different contracts.
 *
 * Run: demos/run.sh privacy-modes
 */
module.exports = buildDemoConfig(
  [{ name: 'privacy-modes-demo', testMatch: 'demo.cjs', timeout: 240000 }],
  {
    baseURL: process.env.KESTREL_URL || 'http://localhost:8900',
    outputDir: path.join(__dirname, 'demo-output', 'playwright'),
  },
);
