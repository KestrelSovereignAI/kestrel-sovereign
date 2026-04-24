/**
 * pipeline.js — Browser WebSocket client for the Pipeline voice path.
 *
 *   const client = await createPipelineClient({ onEvent, apiKey });
 *   await client.start();
 *   client.cancelResponse();   // flushes local playback (best-effort barge-in)
 *   await client.close();
 *
 * Architecture:
 *
 * 1. Open a WebSocket to `/voice/chat` (the server-side pipeline defined in
 *    endpoints/voice.py). Binary frame protocol: byte 0 is a type tag:
 *
 *      0x00 — JSON control (UTF-8 encoded after the header)
 *      0x01 — PCM16 audio (24 kHz mono, little-endian)
 *
 * 2. Feed the capture AudioWorklet (#727) with mic audio; send each chunk
 *    as a 0x01 frame.
 *
 * 3. Decode incoming frames. Audio frames → playback worklet. JSON frames →
 *    translated to the shared voice event vocabulary (#728's events.js) and
 *    forwarded to the caller. The UI shell (#730) uses the same event
 *    handler for both voice paths.
 *
 * 4. Barge-in: the server's existing VAD drives turn boundaries, so client
 *    can't directly tell the LLM to stop. Best effort is to `flush()` the
 *    local playback buffer so audio in-flight doesn't keep playing after
 *    the user starts speaking.
 *
 * Unlike the Realtime client, the Pipeline path preserves the user's chosen
 * LLM (Claude, Gemini, local Ollama, whatever). Slower round-trip, weaker
 * interruption, but model-sovereign.
 */

import { Events, makeEvent } from './events.js';
import { createVoiceCapture } from './capture.js';
import { createVoicePlayback } from './playback.js';

const FRAME_JSON = 0x00;
const FRAME_AUDIO = 0x01;

/**
 * @param {Object} opts
 * @param {(event: {kind: string}) => void} opts.onEvent
 *   Called for every Kestrel voice event. MUST NOT throw.
 * @param {string} [opts.apiKey]  API key for WebSocket auth via query param.
 * @param {string} [opts.wsPath='/voice/chat']
 * @param {number} [opts.sampleRate=24000]  PCM16 sample rate both directions.
 */
export async function createPipelineClient({
  onEvent,
  apiKey = '',
  wsPath = '/voice/chat',
  sampleRate = 24000,
} = {}) {
  if (typeof onEvent !== 'function') {
    throw new Error('createPipelineClient requires an onEvent callback');
  }

  let ws = null;
  let capture = null;
  let playback = null;
  let closed = false;
  let sessionReady = false;

  /** Translate a server control message to a Kestrel voice event. */
  function handleControl(msg) {
    switch (msg?.type) {
      case 'status':
        switch (msg.state) {
          case 'listening':
            onEvent(makeEvent(Events.LISTENING_STARTED, {}));
            break;
          case 'thinking':
            onEvent(makeEvent(Events.LISTENING_STOPPED, {}));
            onEvent(makeEvent(Events.THINKING_STARTED, {}));
            break;
          case 'speaking':
            onEvent(makeEvent(Events.SPEAKING_STARTED, {}));
            break;
          default:
            // Unknown state — skip silently.
            break;
        }
        break;

      case 'transcript':
        if (msg.final) {
          onEvent(makeEvent(Events.USER_TRANSCRIPT_FINAL, { text: msg.text ?? '' }));
        } else {
          onEvent(makeEvent(Events.USER_TRANSCRIPT_DELTA, {
            text: msg.text ?? '',
            is_final: false,
          }));
        }
        break;

      case 'response':
        onEvent(makeEvent(Events.AGENT_TEXT_DELTA, { text: msg.text ?? '' }));
        break;

      case 'response_done':
        onEvent(makeEvent(Events.SPEAKING_STOPPED, {}));
        onEvent(makeEvent(Events.RESPONSE_DONE, {}));
        break;

      case 'error':
        onEvent(makeEvent(Events.ERROR, {
          message: msg.message ?? 'Unknown pipeline error',
          fatal: !!msg.fatal,
        }));
        break;

      default:
        // Unknown control type — ignore to stay forward-compatible with
        // server-side additions.
        break;
    }
  }

  function encodeAudioFrame(pcm) {
    const frame = new Uint8Array(1 + pcm.byteLength);
    frame[0] = FRAME_AUDIO;
    frame.set(pcm, 1);
    return frame.buffer;
  }

  function decodeFrame(buffer) {
    const bytes = buffer instanceof ArrayBuffer ? new Uint8Array(buffer) : buffer;
    if (bytes.byteLength === 0) return null;
    const tag = bytes[0];
    const payload = bytes.subarray(1);
    return { tag, payload };
  }

  async function start() {
    if (ws) throw new Error('Pipeline client already started');

    // 1. Set up audio pipes first so the first mic frame after open() has
    // somewhere to go. Playback starts muted until audio arrives.
    capture = await createVoiceCapture({ targetSampleRate: sampleRate });
    playback = await createVoicePlayback({ sampleRate });

    capture.onchunk((pcm) => {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      try {
        ws.send(encodeAudioFrame(pcm));
      } catch (err) {
        if (!closed) {
          onEvent(makeEvent(Events.ERROR, {
            message: `WebSocket send failed: ${err.message}`,
            fatal: false,
          }));
        }
      }
    });

    // 2. Open the WebSocket.
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = new URL(`${proto}//${window.location.host}${wsPath}`);
    if (apiKey) url.searchParams.set('api_key', apiKey);
    ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      sessionReady = true;
      onEvent(makeEvent(Events.SESSION_READY, {
        session_id: '',  // Pipeline has no discrete session ID today.
        path: 'pipeline',
      }));
    };
    ws.onmessage = (ev) => {
      // Server can send either binary (audio/JSON-framed) or text (rare).
      if (typeof ev.data === 'string') {
        try {
          handleControl(JSON.parse(ev.data));
        } catch (err) {
          onEvent(makeEvent(Events.ERROR, {
            message: `Malformed control message: ${err.message}`,
            fatal: false,
          }));
        }
        return;
      }
      const frame = decodeFrame(ev.data);
      if (!frame) return;
      if (frame.tag === FRAME_JSON) {
        try {
          const text = new TextDecoder().decode(frame.payload);
          handleControl(JSON.parse(text));
        } catch (err) {
          onEvent(makeEvent(Events.ERROR, {
            message: `Malformed control frame: ${err.message}`,
            fatal: false,
          }));
        }
      } else if (frame.tag === FRAME_AUDIO) {
        if (playback) playback.enqueue(frame.payload);
      }
      // Unknown tags are silently dropped — the server may add frame types
      // later without breaking old clients.
    };
    ws.onerror = () => {
      if (closed) return;
      onEvent(makeEvent(Events.ERROR, {
        message: 'WebSocket error',
        code: 'ws_error',
        fatal: !sessionReady,  // before open() it's fatal; after, could be transient
      }));
    };
    ws.onclose = () => {
      if (closed) return;
      onEvent(makeEvent(Events.SESSION_CLOSED, { reason: 'ws_closed' }));
    };
  }

  async function close() {
    if (closed) return;
    closed = true;
    try { ws?.close(); } catch (_) {}
    try { await capture?.destroy(); } catch (_) {}
    try { await playback?.destroy(); } catch (_) {}
    onEvent(makeEvent(Events.SESSION_CLOSED, { reason: 'client_close' }));
  }

  /**
   * Best-effort barge-in: flush local playback buffer so stale TTS doesn't
   * keep playing while the user is talking. Doesn't instruct the server to
   * interrupt LLM generation — the pipeline path doesn't support that today.
   */
  function cancelResponse() {
    try { playback?.flush(); } catch (_) {}
  }

  /**
   * No-op on the pipeline path — provided only for API parity with the
   * Realtime client so the UI shell can call the same method on either.
   */
  function updateInstructions(_instructions) {
    // Intentionally empty: Pipeline has no concept of per-session
    // instructions; the sovereign-side normalizer + TTS adapter handle
    // per-chunk steering via `instructions` on the cloud provider call.
  }

  /**
   * No-op on the pipeline path — tool dispatch is server-side in the
   * cascaded pipeline.
   */
  function commitToolResult(_callId, _result) {}

  function getInputLevel() {
    return capture ? capture.getLevel() : 0;
  }

  return {
    path: 'pipeline',
    start,
    close,
    cancelResponse,
    updateInstructions,
    commitToolResult,
    getInputLevel,
    /** Live mic stream — lets the UI attach an AnalyserNode for the meter. */
    get micStream() {
      return capture?.micStream ?? null;
    },
    /** Pipeline has no remote audio stream; playback happens via the worklet. */
    get remoteStream() {
      return null;
    },
    /** Pipeline has no discrete session object today. */
    get session() {
      return sessionReady ? { session_id: '', path: 'pipeline' } : null;
    },
  };
}
