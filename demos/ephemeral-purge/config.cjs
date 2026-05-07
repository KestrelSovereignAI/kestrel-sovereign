// @ts-check
const path = require('path');
const { buildDemoConfig } = require('@kestrel/flight');

/**
 * EPHEMERAL Purge Vignette — Playwright Configuration
 *
 * Defense-in-depth (#767): when the user toggles out of EPHEMERAL, the
 * agent hard-purges any rows the privacy wrapper accidentally let through.
 *
 * Run: kestrel demo run ephemeral-purge
 */
module.exports = buildDemoConfig(
  [{ name: 'ephemeral-purge-demo', testMatch: 'demo.cjs', timeout: 240000 }],
  {
    baseURL: process.env.KESTREL_URL || 'http://localhost:8900',
    outputDir: path.join(__dirname, 'demo-output', 'playwright'),
  },
);
