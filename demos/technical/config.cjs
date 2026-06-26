// @ts-check
const path = require('path');
const { buildDemoConfig } = require('@kestrel/flight');

/**
 * Kestrel Sovereign Technical Demo Configuration
 *
 * Separate config for the scripted demo (issue #133, Track A).
 * Always records video, uses slowMo for viewer clarity, larger viewport.
 *
 * Run: `kestrel demo run technical` — the ONLY sanctioned entry point. It
 * starts an isolated demo server (its own port + KESTREL_DB_PATH sandbox),
 * verifies only demo agents are loaded, and tears down on exit. The demo
 * mutates databases and permissions; a raw `npx playwright test` against a
 * live instance would corrupt real data (issue #1973), and demo.cjs now
 * refuses to run unless it detects an isolated demo server.
 *
 * Env (set by the runner): KESTREL_URL, KESTREL_DEMO_SERVER=1, KESTREL_DB_PATH.
 * The default baseURL is the demo port (8900), never the live server (8888).
 */
module.exports = buildDemoConfig(
  [{ name: 'demo', testMatch: 'demo.cjs', timeout: 900000 }],
  {
    baseURL: process.env.KESTREL_URL || 'http://localhost:8900',
    outputDir: path.join(__dirname, 'demo-output', 'playwright'),
  },
);
