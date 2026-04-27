/**
 * voice_helpers.cjs — shared helpers for voice E2E tests.
 *
 * Pulled out of test_voice_real.spec.cjs so future voice specs can reuse the
 * auth, route-introspection, and Chromium-flag bits without copy-paste.
 */

const fs = require('fs');
const path = require('path');

const BASE_URL = process.env.KESTREL_URL || 'http://localhost:8888';

/**
 * Path to a deterministic 24kHz mono PCM16 WAV the AudioWorklet capture
 * worklet expects (the existing 16kHz tone_440hz.wav is the wrong rate;
 * regenerate via voice_helpers.regenerateMicFixtureIfNeeded() below).
 */
const MIC_FIXTURE = path.resolve(
  __dirname,
  '..',
  'fixtures',
  'voice_mic_24khz.wav',
);

/**
 * Single-source agent name. The repo runs in rookery mode by default; the
 * voice WS lives at /api/agents/{name}/voice/*. KESTREL_E2E_AGENT lets a CI
 * job swap to a fixture agent without touching specs.
 */
const AGENT = process.env.KESTREL_E2E_AGENT || 'Nellie';

/**
 * Auth helper — single-agent mode hits /api/auth/key, rookery mode honors the
 * KESTREL_API_KEY env var directly. Mirrors the pattern in
 * test_heartbeat_and_bootstrap.spec.cjs.
 */
async function getApiKey(request) {
  if (process.env.KESTREL_API_KEY) return process.env.KESTREL_API_KEY;
  try {
    const res = await request.get(`${BASE_URL}/api/auth/key`);
    if (res.ok()) return (await res.json()).key;
  } catch (_) { /* fall through */ }
  return null;
}

function authHeaders(apiKey) {
  return apiKey ? { 'X-API-Key': apiKey } : {};
}

function agentUrl(suffix) {
  return `${BASE_URL}/api/agents/${AGENT}${suffix}`;
}

/**
 * Verify the mic fixture is the format the AudioWorklet expects (24kHz mono
 * PCM16). A pre-existing fixture at 16kHz would be silently downsampled by
 * the browser AudioContext but produces unstable input timing across runs;
 * we want exact-match. Regenerates with `ffmpeg` when missing/wrong.
 */
function ensureMicFixture() {
  if (fs.existsSync(MIC_FIXTURE)) return MIC_FIXTURE;
  const src = path.resolve(__dirname, '..', 'fixtures', 'tone_440hz.wav');
  if (!fs.existsSync(src)) {
    throw new Error(`Source fixture missing: ${src}. Cannot generate ${MIC_FIXTURE}.`);
  }
  // Resample 16kHz → 24kHz mono PCM16 via ffmpeg. ffmpeg is required by
  // demos/voice/demo.cjs as well so it's a reasonable test-time dep.
  const { execSync } = require('child_process');
  execSync(`ffmpeg -y -i "${src}" -ac 1 -ar 24000 -sample_fmt s16 "${MIC_FIXTURE}"`,
    { stdio: 'pipe' });
  return MIC_FIXTURE;
}

/**
 * Chromium launch args that:
 *   - auto-grant mic permission (no system prompt blocks the test)
 *   - replace the system mic with a fake source
 *   - feed our deterministic WAV as that source
 */
function fakeMicLaunchArgs(wavPath) {
  return [
    '--use-fake-ui-for-media-stream',
    '--use-fake-device-for-media-stream',
    `--use-file-for-fake-audio-capture=${wavPath}`,
    // Disable autoplay block — the playback worklet needs the AudioContext
    // resumed; without this the synthesized audio path can stall.
    '--autoplay-policy=no-user-gesture-required',
  ];
}

module.exports = {
  BASE_URL,
  AGENT,
  MIC_FIXTURE,
  getApiKey,
  authHeaders,
  agentUrl,
  ensureMicFixture,
  fakeMicLaunchArgs,
};
