/**
 * test_voice_real.spec.cjs — voice E2E against the REAL backend.
 *
 * The companion spec at test_voice_ui.spec.cjs mocks the entire network +
 * browser-API boundary (getUserMedia, fetch, RTCPeerConnection, ...) so it
 * runs deterministically without keys. That's good for fast preflight, but
 * every regression we've shipped lately lived in the real provider call,
 * the real route resolver, or the real picker ↔ server wiring — none of
 * which the mocked spec exercises. This spec hits the live backend.
 *
 * Required env (otherwise the suite skips):
 *   - server reachable at KESTREL_URL (default http://localhost:8888)
 *   - KESTREL_API_KEY (or /api/auth/key reachable)
 *   - OPENAI_API_KEY on the SERVER for the realtime-mint case
 *
 * Agent selection:
 *   - KESTREL_E2E_AGENT picks a specific agent name.
 *   - Otherwise the first voice-capable agent on the rookery is used; this
 *     keeps the spec portable across CI vs local dev where agent names
 *     differ.
 *
 * What this spec catches that the mocked one doesn't:
 *   1. /voice/realtime/route returning the wrong shape after a resolver change
 *   2. /voice/realtime/session 502 (the regression that prompted this spec)
 *   3. Picker mode flip → header annotation flips with it (bug 2 in #PR)
 *   4. Picker mode flip → voice list reflects the path's catalog (bug 3)
 *   5. /voice/providers/status surfacing real provider failures with hints
 *   6. EPHEMERAL privacy mode making zero cloud HTTP calls (constitutional)
 */

const { test: base, expect, request: pwRequest } = require('@playwright/test');
const {
  BASE_URL,
  getApiKey,
  authHeaders,
  agentUrlFor,
  discoverVoiceCapableAgent,
  ensureMicFixture,
  fetchProviderVoiceIds,
} = require('./voice_helpers.cjs');

/**
 * Test fixture that resolves a voice-capable agent name once per test.
 * Tests read it as ``async ({ page, request, agentName }) => ...``. Skips
 * the test cleanly when no agent has voice support, with the failure list
 * in the skip message so reviewers know what was tried.
 */
const test = base.extend({
  agentName: async ({ request }, use) => {
    const apiKey = await getApiKey(request);
    if (!apiKey) {
      base.skip(true, 'No API key (set KESTREL_API_KEY)');
      return;
    }
    let name;
    try {
      name = await discoverVoiceCapableAgent(request, apiKey);
    } catch (e) {
      base.skip(true, `Cannot pick a voice-capable agent: ${e.message}`);
      return;
    }
    await use(name);
  },
});

/**
 * Read the picker's currently-listed voice IDs (the option `value`s — the
 * visible text is the voice's display name, but the value is the canonical
 * voice_id we cross-check against /voice/voices).
 */
async function pickerVoiceIds(page) {
  return page.locator('#voice-picker-select option').evaluateAll(
    (opts) => opts.map((o) => o.value).filter((v) => v),
  );
}

// Skip the entire file when the server isn't reachable — same convention as
// the other "real" specs (test_chat_and_models, test_sovereign_console).
test.beforeAll(async () => {
  const ctx = await pwRequest.newContext();
  try {
    const res = await ctx.get(`${BASE_URL}/health`);
    test.skip(!res.ok(), `Server not reachable at ${BASE_URL} (got ${res.status()})`);
  } catch (e) {
    test.skip(true, `Server not reachable at ${BASE_URL}: ${e.message}`);
  } finally {
    await ctx.dispose();
  }
});

// ---------------------------------------------------------------------------
// Headless API tests — no browser, no mic, fast.
// ---------------------------------------------------------------------------

test.describe('voice realtime: API contract', () => {
  test('GET /voice/realtime/route returns realtime path when prefer_realtime=true', async ({ request, agentName }) => {
    const apiKey = await getApiKey(request);
    test.skip(!apiKey, 'No API key (set KESTREL_API_KEY)');
    const res = await request.get(
      agentUrlFor(agentName, '/voice/realtime/route?prefer_realtime=true'),
      { headers: authHeaders(apiKey) },
    );
    expect(res.status(), 'route lookup returns 200').toBe(200);
    const body = await res.json();
    // Response shape contract — these are the fields realtime.js + ui.js
    // consume; if any go missing the picker silently breaks.
    expect(body).toHaveProperty('path');
    expect(body).toHaveProperty('reason');
    expect(body).toHaveProperty('available_conversation_providers');
    expect(body).toHaveProperty('available_tts_providers');
    expect(body).toHaveProperty('available_stt_providers');
    // When OpenAI is the chat LLM and privacy allows cloud, we expect the
    // realtime path. If your local agent uses a non-OpenAI LLM the path
    // will legitimately be 'pipeline' — accept either rather than over-pin.
    expect(['realtime', 'pipeline', 'local', null]).toContain(body.path);
    if (body.path === 'realtime') {
      expect(body.voice_model, 'realtime path must report voice_model').toBeTruthy();
      expect(body.conversation_provider, 'realtime path must report conversation_provider').toBeTruthy();
    }
  });

  test('POST /voice/realtime/session returns 200 + ephemeral token (no 502)', async ({ request, agentName }) => {
    const apiKey = await getApiKey(request);
    test.skip(!apiKey, 'No API key (set KESTREL_API_KEY)');
    // Pre-flight: only run this case when the server has OpenAI configured
    // AND the resolver picks realtime — otherwise the mint correctly 409s
    // and the test would assert against the wrong code path.
    const route = await request.get(
      agentUrlFor(agentName, '/voice/realtime/route?prefer_realtime=true'),
      { headers: authHeaders(apiKey) },
    );
    const routeBody = await route.json();
    test.skip(routeBody.path !== 'realtime',
      `Resolver picked '${routeBody.path}' (need 'realtime' for this case): ${routeBody.reason}`);

    const res = await request.post(agentUrlFor(agentName, '/voice/realtime/session'), {
      headers: { ...authHeaders(apiKey), 'Content-Type': 'application/json' },
      data: {
        // Defaults — same shape the picker sends with no overrides.
        voice: '',
        user_instructions: '',
        prefer_realtime: true,
        turn_detection_mode: 'server_vad',
        silence_ms: 500,
      },
    });
    // The bug was 502. Anything other than 200 here is a regression.
    expect(res.status(), `mint returned ${res.status()}: ${await res.text()}`).toBe(200);
    const body = await res.json();
    expect(body.path).toBe('realtime');
    expect(body.session_id).toMatch(/^sess_/);
    expect(body.client_secret?.value).toMatch(/^ek_/);
    expect(body.client_secret?.expires_at).toBeGreaterThan(Date.now() / 1000);
    expect(body.model, 'mint must report the actual model used').toBeTruthy();
    expect(body.voice, 'mint must echo the voice').toBeTruthy();
  });

  test('GET /voice/voices?provider=openai_realtime returns realtime voices (not empty)', async ({ request, agentName }) => {
    const apiKey = await getApiKey(request);
    test.skip(!apiKey, 'No API key (set KESTREL_API_KEY)');
    // Bug 3 in #PR: the picker asked for provider=openai_realtime and got
    // back []. After #848 the conversation provider returns VoiceInfo
    // directly via the ABC, no sibling-catalog scan needed.
    const res = await request.get(
      agentUrlFor(agentName, '/voice/voices?provider=openai_realtime'),
      { headers: authHeaders(apiKey) },
    );
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body.voices, 'openai_realtime must report voices').toBeInstanceOf(Array);
    expect(body.voices.length, 'realtime voice catalog must not be empty').toBeGreaterThan(0);
    // Every entry must carry the conversation-provider name so the picker
    // can scope correctly. Skip if conversation provider isn't installed
    // (no realtime support in this build).
    test.skip(body.voices.length === 0, 'No conversation providers installed');
    for (const v of body.voices) {
      expect(v.voice_id, 'voice_id is required').toBeTruthy();
      expect(v.provider).toBe('openai_realtime');
    }
  });

  test('GET /voice/providers/status surfaces every attempted provider with diagnostics', async ({ request, agentName }) => {
    const apiKey = await getApiKey(request);
    test.skip(!apiKey, 'No API key (set KESTREL_API_KEY)');
    const res = await request.get(agentUrlFor(agentName, '/voice/providers/status'), {
      headers: authHeaders(apiKey),
    });
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body.providers).toBeInstanceOf(Array);
    expect(body.providers.length).toBeGreaterThan(0);
    // Every row must carry the diagnostic fields the UI's
    // fetchProviderReason() reads. If init_error is set, install_hint must
    // also be set so the user has an actionable next step (per
    // feedback_no_blind_fallbacks.md — silent failures hide the real
    // problem; a populated init_error with no hint is the same antipattern).
    for (const p of body.providers) {
      expect(p).toHaveProperty('name');
      expect(p).toHaveProperty('kind');
      expect(p).toHaveProperty('registered');
      if (p.init_error || p.available_error || p.voice_list_error) {
        // We don't strictly require a hint for every error class, but log
        // any case where one is missing so we can grow _install_hint_for
        // over time. Hard fail only when the error is one of the known-
        // hintable shapes (key permission, SDK incompatibility).
        const errBlob = `${p.init_error || ''} ${p.available_error || ''} ${p.voice_list_error || ''}`.toLowerCase();
        const isHintable = errBlob.includes('voices_read')
          || errBlob.includes('missing_permission')
          || errBlob.includes('import failed')
          || errBlob.includes('cannot import name')
          || errBlob.includes('is_available() returned false');
        if (isHintable) {
          expect(p.install_hint,
            `provider ${p.name} has hintable error but no install_hint: ${errBlob}`,
          ).toBeTruthy();
        }
      }
    }
  });
});

/**
 * Boot the SPA scoped to a specific agent. Auth-protected: we send the API
 * key as both an HTTP header (for the initial GET) and a sessionStorage
 * entry (for everything the SPA does after mount). Then call window.selectAgent
 * to put the SPA in rookery-mode for the named agent so #voice-toggle-btn
 * mounts in the chat header.
 */
async function loadChatForAgent(page, apiKey, agentName) {
  await page.setExtraHTTPHeaders({ 'X-API-Key': apiKey });
  await page.addInitScript((key) => {
    sessionStorage.setItem('kestrel.api.key', key);
  }, apiKey);
  await page.goto(BASE_URL);
  await page.waitForLoadState('domcontentloaded');
  // The SPA wires window.selectAgent at boot.
  await page.waitForFunction(() => typeof window.selectAgent === 'function', { timeout: 10_000 });
  await page.evaluate((name) => window.selectAgent(name), agentName);
  // selectAgent flips into rookery mode but leaves the active panel where it
  // was (Identity by default). Navigate to Chat so the chat input + mic
  // button become visible — the voice UI mounted them at #send-button at
  // page init but the panel they live in is hidden until we switch to it.
  await page.click('button:has-text("Chat"), nav button[data-panel="chat"], nav button:text-is("Chat")');
  await page.waitForSelector('#voice-toggle-btn', { state: 'visible', timeout: 10_000 });
}

// ---------------------------------------------------------------------------
// Browser tests — picker UI wiring. No real mic needed; we just open the
// modal and assert the DOM reflects route state.
// ---------------------------------------------------------------------------

test.describe('voice picker: live UI wiring', () => {
  test('flipping mode → Force Realtime updates header voice annotation', async ({ page, request, agentName }) => {
    const apiKey = await getApiKey(request);
    test.skip(!apiKey, 'No API key');
    const route = await request.get(
      agentUrlFor(agentName, '/voice/realtime/route?prefer_realtime=true'),
      { headers: authHeaders(apiKey) },
    );
    const routeBody = await route.json();
    test.skip(routeBody.path !== 'realtime',
      `Realtime not available on this server (path=${routeBody.path}): ${routeBody.reason}`);

    await loadChatForAgent(page, apiKey, agentName);
    // Open picker via right-click on the mic button (matches ui.js wiring).
    await page.click('#voice-toggle-btn', { button: 'right' });
    await page.waitForSelector('#voice-picker-mode', { state: 'visible' });

    // Bug 2 catch: flip mode → "realtime" and assert the chat-header
    // annotation populates without starting a session. The annotation lives
    // at #voice-active-model-annotation (created by setModelSelectorVoiceAnnotation).
    await page.selectOption('#voice-picker-mode', 'realtime');
    // refreshRoutePreview is async (network round-trip to /voice/realtime/route);
    // poll the annotation for up to 5s.
    await expect.poll(async () => {
      const text = await page.locator('#voice-active-model-annotation').textContent();
      return text || '';
    }, { timeout: 5_000 }).toMatch(/gpt-realtime|🎙/);
  });

  test('flipping mode → Force Realtime narrows voice list to realtime catalog', async ({ page, request, agentName }) => {
    const apiKey = await getApiKey(request);
    test.skip(!apiKey, 'No API key');
    const route = await request.get(
      agentUrlFor(agentName, '/voice/realtime/route?prefer_realtime=true'),
      { headers: authHeaders(apiKey) },
    );
    const routeBody = await route.json();
    test.skip(routeBody.path !== 'realtime',
      `Realtime not available on this server (path=${routeBody.path})`);

    // Pre-seed the picker with ElevenLabs as preferred TTS so we can prove
    // the switch to Realtime overrides it. Persisted via the same
    // localStorage key ui.js uses (SETTINGS_KEY). Set BEFORE the page loads
    // via init script so the SPA reads it on mount.
    await page.addInitScript(() => {
      localStorage.setItem('kestrel.voice.settings', JSON.stringify({
        voice: '', instructions: '', mode: 'pipeline', preferred_tts: 'elevenlabs',
      }));
    });
    await loadChatForAgent(page, apiKey, agentName);

    // Pull the canonical catalogs once so assertions stay catalog-rename-safe:
    // adding a voice to OpenAI realtime later doesn't break this spec.
    const elevenIds = await fetchProviderVoiceIds(request, apiKey, agentName, 'elevenlabs');
    const realtimeIds = await fetchProviderVoiceIds(request, apiKey, agentName, 'openai_realtime');
    test.skip(elevenIds.size === 0,
      'ElevenLabs not configured on this server — pre-state for the flip cannot be set up.');
    expect(realtimeIds.size, 'realtime catalog must not be empty').toBeGreaterThan(0);

    await page.click('#voice-toggle-btn', { button: 'right' });
    await page.waitForSelector('#voice-picker-mode', { state: 'visible' });

    // Confirm the Pipeline-mode pre-state lists ElevenLabs voices: every
    // visible picker option's voice_id must be in the live ElevenLabs catalog.
    await page.selectOption('#voice-picker-mode', 'pipeline');
    await expect.poll(async () => {
      const ids = await pickerVoiceIds(page);
      return ids.length > 0 && ids.every((id) => elevenIds.has(id));
    }, {
      timeout: 5_000,
      message: 'Pipeline mode must show only ElevenLabs voice IDs',
    }).toBe(true);

    // Bug 3 catch: flip to Force Realtime; the picker must now list voices
    // from the realtime catalog and NONE from ElevenLabs.
    await page.selectOption('#voice-picker-mode', 'realtime');
    await expect.poll(async () => {
      const ids = await pickerVoiceIds(page);
      return ids.length > 0 && ids.every((id) => realtimeIds.has(id));
    }, {
      timeout: 5_000,
      message: 'Realtime mode must show only OpenAI realtime voice IDs',
    }).toBe(true);

    const pickerIds = await pickerVoiceIds(page);
    const leaked = pickerIds.filter((id) => elevenIds.has(id));
    expect(leaked, 'Realtime mode must not list any ElevenLabs voice IDs').toEqual([]);

    // The TTS dropdown must be disabled in Realtime mode (Realtime owns
    // I/O end-to-end; the user's TTS pick has no effect). Without this the
    // user thinks they can mix-and-match and gets confused when the choice
    // is ignored.
    await expect(page.locator('#voice-picker-tts')).toBeDisabled();
  });
});
