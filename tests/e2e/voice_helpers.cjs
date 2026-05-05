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
 * Auth helper — single-agent mode hits /api/auth/key, multi_agent mode honors the
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

/**
 * Build the multi_agent-scoped URL for an agent. Replaces the older
 * `agentUrl(suffix)` helper that hardcoded one agent name — every spec now
 * resolves an agent first via `discoverVoiceCapableAgent` and passes the
 * resolved name in.
 */
function agentUrlFor(agentName, suffix) {
  return `${BASE_URL}/api/agents/${encodeURIComponent(agentName)}${suffix}`;
}

/**
 * Pick the first agent on the multi_agent that has voice configured. Used by E2E
 * specs so they don't hardcode an agent name a CI server may not have.
 *
 * Selection order:
 *   1. KESTREL_E2E_AGENT env var, if the named agent exists.
 *   2. The first agent whose /voice/realtime/route returns a non-error
 *      response (i.e. has VoiceFeature enabled and reachable).
 *   3. Throws with a clear message if no agent qualifies.
 */
async function discoverVoiceCapableAgent(request, apiKey) {
  const explicit = process.env.KESTREL_E2E_AGENT;
  const list = await request.get(`${BASE_URL}/api/agents`, {
    headers: authHeaders(apiKey),
  });
  if (!list.ok()) {
    throw new Error(`Cannot list agents at ${BASE_URL}/api/agents: ${list.status()}`);
  }
  const body = await list.json();
  const names = (body.agents || []).map((a) => a.name).filter(Boolean);
  if (names.length === 0) {
    throw new Error('No agents on this server — cannot run voice E2E.');
  }

  const candidates = explicit && names.includes(explicit) ? [explicit] : names;

  for (const name of candidates) {
    const route = await request.get(
      agentUrlFor(name, '/voice/realtime/route'),
      { headers: authHeaders(apiKey) },
    );
    if (route.ok()) return name;
  }
  throw new Error(
    `No voice-capable agent found. Tried: ${candidates.join(', ')}. ` +
    'Set KESTREL_E2E_AGENT to override or add VoiceFeature to an agent.',
  );
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

/**
 * Fetch the voice ID set for a provider from the live API. Used by E2E
 * assertions that need to test "picker shows voices from provider X"
 * without coupling to specific voice names that may rename or churn.
 *
 * Returns a `Set` of voice_id strings; empty Set if the provider isn't
 * configured (caller decides whether that's a skip or a hard fail).
 */
async function fetchProviderVoiceIds(request, apiKey, agentName, providerName) {
  const res = await request.get(
    agentUrlFor(agentName, `/voice/voices?provider=${encodeURIComponent(providerName)}`),
    { headers: authHeaders(apiKey) },
  );
  if (!res.ok()) {
    throw new Error(`/voice/voices?provider=${providerName} → ${res.status()}`);
  }
  const body = await res.json();
  return new Set((body.voices || []).map((v) => v.voice_id).filter(Boolean));
}

module.exports = {
  BASE_URL,
  MIC_FIXTURE,
  getApiKey,
  authHeaders,
  agentUrlFor,
  discoverVoiceCapableAgent,
  ensureMicFixture,
  fakeMicLaunchArgs,
  fetchProviderVoiceIds,
};
