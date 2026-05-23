/**
 * Voice UI E2E tests.
 *
 * Drives the voice shell (issue #730) through the chat UI and asserts the
 * routing matrix from the resolver (#723) flows correctly:
 *
 * - Mic button mounts in the chat header.
 * - Toggle drives the session lifecycle (idle → connecting → listening).
 * - Path badge displays "Realtime" / "Pipeline" based on the resolver's
 *   decision, conveyed via the ephemeral-token endpoint (#726).
 * - 409 fallback from Realtime → Pipeline triggers the Pipeline WebSocket
 *   client transparently.
 * - Voice picker modal lists voices fetched from /voice/voices, persists
 *   the user's selection to localStorage, and pushes mid-session
 *   instructions when a session is open.
 * - Push-to-talk (Space) + Esc-to-stop keybindings work and don't steal
 *   keystrokes from the message input.
 *
 * Mocks at the network boundary via `page.route()` so the spec runs
 * without OpenAI / ElevenLabs / Piper credentials. WebRTC is stubbed via
 * `page.addInitScript()` so the test never actually negotiates SDP.
 *
 * The companion narrated demo (demos/voice/demo.cjs) exercises the same
 * flows against real services for video output — that's where Gabi sees
 * voice working end-to-end.
 */

// @ts-check
const { test, expect } = require('@playwright/test');

const BASE_URL = process.env.KESTREL_URL || 'http://localhost:8888';

// ---------------------------------------------------------------------------
// Browser-side stubs — installed before page scripts run.
// ---------------------------------------------------------------------------

/**
 * Stub WebRTC + getUserMedia + WebSocket so the voice clients can run
 * without real browser audio devices or network peers. The stubs expose
 * a `__voiceStub` global the spec uses to drive events into the shell.
 */
const VOICE_STUB_INIT_SCRIPT = `
(() => {
  const listeners = { event: [] };
  const recorded = { rtcOffers: 0, rtcCloses: 0, wsOpens: 0, wsSends: 0, wsCloses: 0, micRequests: 0 };
  let lastDC = null;

  // Fake mic stream — empty MediaStream, satisfies addTrack.
  navigator.mediaDevices.getUserMedia = async () => {
    recorded.micRequests++;
    return new MediaStream();
  };

  // Stub RTCPeerConnection — captures createOffer / setLocalDescription /
  // close and resolves SDP exchange immediately.
  window.RTCPeerConnection = class {
    constructor() {
      this._tracks = [];
      this._dataChannels = [];
      this.connectionState = 'new';
      this.ontrack = null;
      this.onconnectionstatechange = null;
    }
    addTrack(track, stream) { this._tracks.push({ track, stream }); }
    createDataChannel(label) {
      const dc = {
        label,
        readyState: 'open',
        send: (_msg) => { recorded.wsSends++; },
        close: () => { dc.readyState = 'closed'; if (dc.onclose) dc.onclose(); },
        onopen: null, onmessage: null, onclose: null, onerror: null,
      };
      this._dataChannels.push(dc);
      lastDC = dc;
      // Emit a synthetic SESSION_READY-equivalent OpenAI event after the
      // page processes the SDP answer. The shell handler translates it.
      setTimeout(() => {
        if (dc.onmessage) {
          dc.onmessage({ data: JSON.stringify({ type: 'session.created', session: { id: 'sess_stub', model: 'gpt-realtime-stub' }}) });
        }
      }, 50);
      return dc;
    }
    async createOffer() { recorded.rtcOffers++; return { type: 'offer', sdp: 'v=0\\r\\n' }; }
    async setLocalDescription(_d) {}
    async setRemoteDescription(_d) { this.connectionState = 'connected'; if (this.onconnectionstatechange) this.onconnectionstatechange(); }
    close() { recorded.rtcCloses++; this.connectionState = 'closed'; if (this.onconnectionstatechange) this.onconnectionstatechange(); }
  };

  // Stub fetch for the SDP exchange to avoid hitting OpenAI directly.
  // Pass-through everything else.
  const realFetch = window.fetch.bind(window);
  window.fetch = async (input, init) => {
    const url = typeof input === 'string' ? input : input.url;
    if (url.startsWith('https://api.openai.com/v1/realtime')) {
      return new Response('v=0\\r\\n', { status: 200, headers: { 'Content-Type': 'application/sdp' } });
    }
    return realFetch(input, init);
  };

  // Stub WebSocket so the Pipeline client can "open" without a server.
  // The spec drives messages into it via window.__voiceStub.pushWs.
  const realWS = window.WebSocket;
  let lastWS = null;
  window.WebSocket = class extends EventTarget {
    constructor(url) {
      super();
      this.url = url;
      this.readyState = 0;  // CONNECTING
      this.binaryType = 'arraybuffer';
      recorded.wsOpens++;
      lastWS = this;
      // Fire 'open' on next tick.
      setTimeout(() => {
        this.readyState = 1;  // OPEN
        if (this.onopen) this.onopen({});
      }, 10);
    }
    send(data) { recorded.wsSends++; }
    close() {
      recorded.wsCloses++;
      this.readyState = 3;  // CLOSED
      if (this.onclose) this.onclose({ wasClean: true });
    }
  };
  // Preserve the real constants (WebSocket.OPEN etc.).
  window.WebSocket.CONNECTING = 0;
  window.WebSocket.OPEN = 1;
  window.WebSocket.CLOSING = 2;
  window.WebSocket.CLOSED = 3;

  // AudioContext shim that returns a no-op AudioWorklet so capture/playback
  // worklets can "load" without actually decoding audio.
  const realAudioContext = window.AudioContext || window.webkitAudioContext;
  if (realAudioContext) {
    const FakeAudioWorklet = { addModule: async () => {} };
    const wrap = (Cls) => class extends Cls {
      constructor(...args) {
        super(...args);
      }
      get audioWorklet() { return FakeAudioWorklet; }
      createMediaStreamSource(_stream) {
        return {
          connect: () => {},
          disconnect: () => {},
        };
      }
    };
    window.AudioContext = wrap(realAudioContext);
  }
  // Stub AudioWorkletNode so capture.js / playback.js don't throw.
  window.AudioWorkletNode = class extends EventTarget {
    constructor() {
      super();
      this.port = {
        postMessage: () => {},
        onmessage: null,
      };
    }
    connect() {}
    disconnect() {}
  };

  window.__voiceStub = {
    recorded,
    pushRtc(event) { if (lastDC && lastDC.onmessage) lastDC.onmessage({ data: JSON.stringify(event) }); },
    pushWs(frame) { if (lastWS && lastWS.onmessage) lastWS.onmessage({ data: frame }); },
    closeWs() { if (lastWS) lastWS.close(); },
  };
})();
`;

// ---------------------------------------------------------------------------
// Mock backend responses
// ---------------------------------------------------------------------------

const MOCK_VOICES = {
  voices: [
    { voice_id: 'cedar', name: 'Cedar', provider: 'openai', gender: 'masculine', accent: 'american', age: 'middle', energy: 'warm', language: 'en', preview_url: '' },
    { voice_id: 'marin', name: 'Marin', provider: 'openai', gender: 'feminine', accent: 'american', age: 'middle', energy: 'warm', language: 'en', preview_url: '' },
  ],
  count: 2,
};

const MOCK_REALTIME_SESSION = {
  path: 'realtime',
  session_id: 'sess_stub',
  client_secret: { value: 'ek_stub', expires_at: 9999999999 },
  model: 'gpt-realtime-stub',
  voice: 'cedar',
};

const MOCK_REALTIME_409 = {
  path: 'pipeline',
  reason: 'Pipeline path: Anthropic LLM, Realtime requires OpenAI.',
  fallback_tts: 'elevenlabs',
  fallback_stt: 'openai',
};

const MOCK_REALTIME_UNAVAILABLE_409 = {
  path: null,
  reason: 'Voice unavailable: no TTS and STT provider installed.',
  fallback_tts: null,
  fallback_stt: null,
};

/**
 * Wire up route mocks. Pass `realtimeStatus` to control the
 * /voice/realtime/session response: 200 → Realtime path, 409 → Pipeline
 * fallback.
 */
async function installRoutes(page, { realtimeStatus = 200, realtime409Body = MOCK_REALTIME_409 } = {}) {
  // Voices endpoint — same in both paths.
  await page.route('**/voice/voices', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(MOCK_VOICES) }),
  );
  await page.route('**/voice/realtime/session', (route) =>
    route.fulfill({
      status: realtimeStatus,
      contentType: 'application/json',
      body: JSON.stringify(realtimeStatus === 409 ? realtime409Body : MOCK_REALTIME_SESSION),
    }),
  );
  // /voice/config + /voice/chat are the existing endpoints; let the WS stub
  // handle /voice/chat upgrade. /voice/config not exercised here.
}

async function openChatPanel(page) {
  await page.getByRole('button', { name: 'Chat' }).click();
}

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

test.describe.configure({ mode: 'serial' });

test.describe('Voice UI shell', () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(VOICE_STUB_INIT_SCRIPT);
  });

  test('mic button mounts in the chat header', async ({ page }) => {
    await installRoutes(page);
    await page.goto(BASE_URL);
    await page.waitForLoadState('domcontentloaded');
    await openChatPanel(page);

    const btn = page.locator('#voice-toggle-btn');
    await expect(btn).toBeVisible({ timeout: 10000 });
    await expect(btn).toHaveAttribute('data-state', 'idle');
    await expect(btn).toHaveAttribute('aria-label', /Start voice session/i);
  });

  test('Realtime path: click engages session, badge shows "Realtime"', async ({ page }) => {
    await installRoutes(page, { realtimeStatus: 200 });
    await page.goto(BASE_URL);
    await page.waitForLoadState('domcontentloaded');
    await openChatPanel(page);

    await page.locator('#voice-toggle-btn').click();
    const badge = page.locator('.kestrel-voice-path-badge');
    await expect(badge).toBeVisible({ timeout: 5000 });
    await expect(badge).toContainText('Realtime', { timeout: 5000 });
    await expect(badge).toHaveAttribute('data-path', 'realtime');

    // Mic button transitions to listening once the stub fires session.created.
    await expect(page.locator('#voice-toggle-btn')).toHaveAttribute('data-state', 'listening', { timeout: 5000 });
  });

  test('Realtime path keeps user transcript before agent response when events race', async ({ page }) => {
    await installRoutes(page, { realtimeStatus: 200 });
    await page.goto(BASE_URL);
    await page.waitForLoadState('domcontentloaded');
    await openChatPanel(page);

    await page.locator('#voice-toggle-btn').click();
    await expect(page.locator('#voice-toggle-btn')).toHaveAttribute('data-state', 'listening', { timeout: 5000 });

    await page.evaluate(() => {
      window.__voiceStub.pushRtc({ type: 'input_audio_buffer.speech_started' });
      window.__voiceStub.pushRtc({ type: 'input_audio_buffer.speech_stopped' });
      window.__voiceStub.pushRtc({ type: 'response.created' });
      window.__voiceStub.pushRtc({ type: 'response.output_audio_transcript.delta', delta: 'Agent answer first.' });
      window.__voiceStub.pushRtc({
        type: 'conversation.item.input_audio_transcription.completed',
        transcript: 'User question arrived late.',
      });
      window.__voiceStub.pushRtc({ type: 'response.output_audio_transcript.done', transcript: 'Agent answer first.' });
      window.__voiceStub.pushRtc({ type: 'response.done' });
    });

    await expect(page.locator('.message')).toHaveCount(2, { timeout: 5000 });
    const messages = await page.locator('.message').evaluateAll((nodes) =>
      nodes.map((node) => ({
        cls: node.className,
        text: node.textContent.trim(),
      })),
    );
    expect(messages[0]).toMatchObject({ text: 'User question arrived late.' });
    expect(messages[0].cls).toContain('user-message');
    expect(messages[1]).toMatchObject({ text: 'Agent answer first.' });
    expect(messages[1].cls).toContain('agent-message');
  });

  test('Realtime path removes stale transcribing placeholder when speech restarts', async ({ page }) => {
    await installRoutes(page, { realtimeStatus: 200 });
    await page.goto(BASE_URL);
    await page.waitForLoadState('domcontentloaded');
    await openChatPanel(page);

    await page.locator('#voice-toggle-btn').click();
    await expect(page.locator('#voice-toggle-btn')).toHaveAttribute('data-state', 'listening', { timeout: 5000 });

    await page.evaluate(() => {
      window.__voiceStub.pushRtc({ type: 'input_audio_buffer.speech_started' });
      window.__voiceStub.pushRtc({ type: 'input_audio_buffer.speech_stopped' });
      window.__voiceStub.pushRtc({ type: 'input_audio_buffer.speech_started' });
      window.__voiceStub.pushRtc({
        type: 'conversation.item.input_audio_transcription.completed',
        transcript: 'Actual transcript.',
      });
    });

    await expect(page.locator('.message')).toHaveCount(1, { timeout: 5000 });
    await expect(page.locator('.message')).toHaveText('Actual transcript.');
    await expect(page.locator('#chat-container')).not.toContainText('Transcribing...');
  });

  test('microphone permission denial shows actionable voice error', async ({ page }) => {
    await page.addInitScript(() => {
      navigator.mediaDevices.getUserMedia = async () => {
        const err = new Error('Permission denied');
        err.name = 'NotAllowedError';
        throw err;
      };
    });
    await installRoutes(page, { realtimeStatus: 200 });
    await page.goto(BASE_URL);
    await page.waitForLoadState('domcontentloaded');
    await openChatPanel(page);

    await page.locator('#voice-toggle-btn').click();

    await expect(page.locator('#voice-toggle-btn')).toHaveAttribute('data-state', 'error', { timeout: 5000 });
    await expect(page.locator('#chat-container')).toContainText(
      'Microphone permission denied. Allow microphone access for this site, then click the mic again. If this browser has no microphone permission control, open Kestrel in Chrome or Safari.',
    );
  });

  test('Pipeline fallback: 409 from realtime endpoint engages WebSocket path', async ({ page }) => {
    await installRoutes(page, { realtimeStatus: 409 });
    await page.goto(BASE_URL);
    await page.waitForLoadState('domcontentloaded');
    await openChatPanel(page);

    await page.locator('#voice-toggle-btn').click();

    const badge = page.locator('.kestrel-voice-path-badge');
    await expect(badge).toBeVisible({ timeout: 5000 });
    await expect(badge).toHaveText('Pipeline', { timeout: 5000 });
    await expect(badge).toHaveAttribute('data-path', 'pipeline');

    // The Pipeline client opened a WebSocket — confirm via the stub recorder.
    const wsOpens = await page.evaluate(() => window.__voiceStub.recorded.wsOpens);
    expect(wsOpens).toBe(1);
  });

  test('Unavailable voice: 409 without Pipeline providers does not open WebSocket', async ({ page }) => {
    await installRoutes(page, {
      realtimeStatus: 409,
      realtime409Body: MOCK_REALTIME_UNAVAILABLE_409,
    });
    await page.goto(BASE_URL);
    await page.waitForLoadState('domcontentloaded');
    await openChatPanel(page);

    await page.locator('#voice-toggle-btn').click();

    await expect(page.locator('#voice-toggle-btn')).toHaveAttribute('data-state', 'error', { timeout: 5000 });
    await expect(page.locator('#chat-container')).toContainText('Voice unavailable: no TTS and STT provider installed.');

    const wsOpens = await page.evaluate(() => window.__voiceStub.recorded.wsOpens);
    expect(wsOpens).toBe(0);
  });

  test('Esc closes an active session and returns to idle', async ({ page }) => {
    await installRoutes(page, { realtimeStatus: 200 });
    await page.goto(BASE_URL);
    await page.waitForLoadState('domcontentloaded');
    await openChatPanel(page);

    await page.locator('#voice-toggle-btn').click();
    await expect(page.locator('#voice-toggle-btn')).toHaveAttribute('data-state', 'listening', { timeout: 5000 });

    // Move focus off any text input so the Esc keybinding fires the shell handler.
    await page.locator('body').click({ position: { x: 0, y: 0 } });
    await page.keyboard.press('Escape');

    await expect(page.locator('#voice-toggle-btn')).toHaveAttribute('data-state', 'idle', { timeout: 5000 });
    await expect(page.locator('.kestrel-voice-path-badge')).toBeHidden();
  });

  test('voice picker lists discovered voices and persists selection', async ({ page }) => {
    await installRoutes(page);
    await page.goto(BASE_URL);
    await page.waitForLoadState('domcontentloaded');
    await openChatPanel(page);

    // Right-click the mic to open the picker without starting a session.
    await page.locator('#voice-toggle-btn').click({ button: 'right' });

    const modal = page.locator('#voice-picker-modal');
    await expect(modal).toBeVisible({ timeout: 5000 });

    const select = modal.locator('#voice-picker-select');
    // Wait for the options to render from the mocked /voice/voices.
    await expect(select.locator('option')).toHaveCount(2, { timeout: 5000 });
    await expect(select.locator('option')).toContainText(['Cedar', 'Marin']);

    // Pick Marin + a custom directive, save, modal closes, settings persisted.
    await select.selectOption('marin');
    await modal.locator('#voice-picker-instructions').fill('Speak like a sympathetic pirate.');
    await modal.locator('#voice-picker-save').click();
    await expect(modal).toBeHidden();

    const saved = await page.evaluate(() => localStorage.getItem('kestrel.voice.settings'));
    expect(saved).toBeTruthy();
    const parsed = JSON.parse(saved);
    expect(parsed.voice).toBe('marin');
    expect(parsed.instructions).toContain('sympathetic pirate');
  });


  test('spacebar push-to-talk does not steal keystrokes from the message input', async ({ page }) => {
    await installRoutes(page);
    await page.goto(BASE_URL);
    await page.waitForLoadState('domcontentloaded');
    await openChatPanel(page);

    // Focus the textarea and type a space — voice should NOT engage because
    // focus is on a text input.
    const input = page.locator('#message-input');
    await input.focus();
    await page.keyboard.press('Space');
    await expect(page.locator('#voice-toggle-btn')).toHaveAttribute('data-state', 'idle');

    // Now move focus off the input and press Space — voice engages.
    await page.locator('body').click({ position: { x: 0, y: 0 } });
    await page.keyboard.down('Space');
    await expect(page.locator('#voice-toggle-btn')).toHaveAttribute('data-state', /connecting|listening/, { timeout: 5000 });
    await page.keyboard.up('Space');
    await expect(page.locator('#voice-toggle-btn')).toHaveAttribute('data-state', 'idle', { timeout: 5000 });
  });

  test('mic button cycles back to idle after stopping the session', async ({ page }) => {
    await installRoutes(page, { realtimeStatus: 200 });
    await page.goto(BASE_URL);
    await page.waitForLoadState('domcontentloaded');
    await openChatPanel(page);

    const btn = page.locator('#voice-toggle-btn');
    await btn.click();
    await expect(btn).toHaveAttribute('data-state', 'listening', { timeout: 5000 });

    // Click again → stops.
    await btn.click();
    await expect(btn).toHaveAttribute('data-state', 'idle', { timeout: 5000 });
    await expect(page.locator('.kestrel-voice-path-badge')).toBeHidden();
  });
});
