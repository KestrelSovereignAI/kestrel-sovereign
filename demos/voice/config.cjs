// @ts-check
const path = require('path');
const { buildDemoConfig } = require('@kestrel/flight');

/**
 * Kestrel Voice Demo — Playwright configuration.
 *
 * Narrated end-to-end voice flow for Gabi (and the GTM team) showing the
 * Squack II epic shipped (#721): mic button mounts, voice picker lists
 * discovered voices, click engages a session, path badge shows whether
 * Realtime or Pipeline is active, mid-conversation model switch surfaces
 * cleanly. Records video + a narration.md transcript per beat.
 *
 *   cd demos/voice && npx playwright test --config=config.cjs
 *
 * Requires a running Kestrel server with at least one TTS provider
 * available (OpenAI for Realtime, ElevenLabs / Piper for Pipeline).
 * Env: DEMO_SLOWMO=200, KESTREL_URL, KESTREL_API_KEY.
 */
module.exports = buildDemoConfig(
  [{ name: 'voice-demo', testMatch: 'demo.cjs', timeout: 600000 }],
  {
    baseURL: process.env.KESTREL_URL || 'http://localhost:8888',
    outputDir: path.join(__dirname, 'demo-output', 'playwright'),
  },
);
