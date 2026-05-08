// @ts-check
const path = require('path');
const { buildDemoConfig } = require('@kestrel/flight');

/**
 * Demo Isolation Vignette — Playwright Configuration
 *
 * Server-side rail (#766) that refuses destructive ops on live agents
 * unless the X-Kestrel-Allow-Destructive header is present.
 *
 * Run: kestrel demo run demo-isolation
 */
module.exports = buildDemoConfig(
  [{ name: 'demo-isolation-demo', testMatch: 'demo.cjs', timeout: 180000 }],
  {
    baseURL: process.env.KESTREL_URL || 'http://localhost:8900',
    outputDir: path.join(__dirname, 'demo-output', 'playwright'),
  },
);
