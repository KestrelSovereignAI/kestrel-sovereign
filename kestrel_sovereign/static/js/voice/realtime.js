/**
 * realtime.js — Browser WebRTC client for the OpenAI Realtime path.
 *
 *   const client = await createRealtimeClient({ onEvent });
 *   await client.start();
 *   // UI receives events; later:
 *   await client.sendText('hi');
 *   client.cancelResponse();   // barge-in
 *   client.updateInstructions('speak excitedly');
 *   await client.close();
 *
 * Architecture:
 *
 * 1. Fetch a short-lived session from the backend ephemeral-token endpoint
 *    (POST /voice/realtime/session, ticket #726). Response carries
 *    { client_secret.value, expires_at, session_id, model, voice }.
 *
 * 2. Open an RTCPeerConnection. Audio is handled natively: mic track added
 *    via `getUserMedia`, remote audio played via an `<audio>` sink. The
 *    capture/playback AudioWorklets (#727) aren't needed here — OpenAI
 *    Realtime wants a live MediaStreamTrack, not raw PCM16 buffers.
 *
 * 3. Open a data channel ("oai-events") for JSON control: session updates,
 *    response cancels, tool-result commits, and incoming events from OpenAI.
 *
 * 4. SDP exchange: send the local offer to OpenAI directly
 *    (`POST https://api.openai.com/v1/realtime?model=...` with the ephemeral
 *    Bearer token). No backend proxy — the long-lived API key never appears
 *    in the browser.
 *
 * 5. Translate OpenAI server events → Kestrel voice events (see events.js)
 *    and forward to the caller via `onEvent`. The UI shell doesn't know
 *    anything about OpenAI's event taxonomy.
 *
 * Barge-in: OpenAI server-side VAD handles this natively — when the user
 * starts speaking, OpenAI stops sending response audio automatically.
 * We also forward `speech_started` as LISTENING_STARTED so the UI can drop
 * its speaking indicator immediately.
 *
 * Tool calls: emitted as TOOL_CALL_REQUESTED events. The caller is
 * responsible for running the tool and calling `client.commitToolResult`
 * which forwards a `conversation.item.create` (type=function_call_output)
 * back over the data channel. Dispatch is not wired to sovereign tools here
 * — that's a follow-up that bridges the frontend into the agent tool
 * registry.
 */

import { Events, makeEvent } from './events.js';

// GA WebRTC endpoint.  The Beta path was ``/v1/realtime`` (with the
// model as a query string), but OpenAI's GA Realtime moved WebRTC SDP
// exchange under ``/v1/realtime/calls`` (the SDK exposes it as
// ``client.realtime.calls.create``).  Posting to the old path now
// 400s with the SDP body — the browser sees "SDP exchange failed:
// HTTP 400".  See kestrel-voice-openai#16 (Beta -> GA migration).
const REALTIME_SDP_URL = 'https://api.openai.com/v1/realtime/calls';

/**
 * @param {Object} opts
 * @param {(event: {kind: string}) => void} opts.onEvent
 *   Called for every Kestrel voice event. MUST NOT throw — the client
 *   doesn't wrap callback errors.
 * @param {string} [opts.endpoint='/voice/realtime/session']  Backend mint endpoint.
 * @param {RequestInit} [opts.sessionRequestInit]  Extra fetch options (headers, credentials).
 * @param {Object} [opts.sessionRequestBody]  Extra fields sent to the mint endpoint
 *   (voice, user_instructions, turn_detection_mode, silence_ms).
 */
export async function createRealtimeClient({
  onEvent,
  endpoint = '/voice/realtime/session',
  getAuthHeaders = () => ({}),
  sessionRequestInit = {},
  sessionRequestBody = {},
} = {}) {
  if (typeof onEvent !== 'function') {
    throw new Error('createRealtimeClient requires an onEvent callback');
  }

  let session = null;    // { session_id, model, voice, client_secret }
  let pc = null;         // RTCPeerConnection
  let dc = null;         // RTCDataChannel
  let micStream = null;  // MediaStream from getUserMedia
  let audioSink = null;  // <audio> element for remote playback
  let closed = false;

  /**
   * Send a JSON control message through the data channel.
   * Swallows send errors when the channel is closing — barge-in races are
   * common and we don't want to promote them to UI errors.
   */
  function sendJSON(msg) {
    if (!dc || dc.readyState !== 'open') return;
    try {
      dc.send(JSON.stringify(msg));
    } catch (err) {
      if (!closed) {
        onEvent(makeEvent(Events.ERROR, {
          message: `data channel send failed: ${err.message}`,
          fatal: false,
        }));
      }
    }
  }

  /**
   * Translate OpenAI server events to Kestrel voice events.
   *
   * Unknown types are logged at debug level and skipped — OpenAI can add new
   * events without breaking this client. Missing fields fall back to empty
   * values so the UI shell never sees undefined text.
   */
  function handleOpenAIEvent(raw) {
    switch (raw.type) {
      case 'session.created':
        onEvent(makeEvent(Events.SESSION_READY, {
          session_id: raw.session?.id ?? session?.session_id ?? '',
          model: session?.model ?? raw.session?.model ?? '',
          path: 'realtime',
        }));
        break;

      case 'input_audio_buffer.speech_started':
        onEvent(makeEvent(Events.LISTENING_STARTED, {}));
        break;
      case 'input_audio_buffer.speech_stopped':
        onEvent(makeEvent(Events.LISTENING_STOPPED, {}));
        break;

      case 'conversation.item.input_audio_transcription.delta':
        onEvent(makeEvent(Events.USER_TRANSCRIPT_DELTA, {
          text: raw.delta ?? '',
          is_final: false,
        }));
        break;
      case 'conversation.item.input_audio_transcription.completed':
        onEvent(makeEvent(Events.USER_TRANSCRIPT_FINAL, {
          text: raw.transcript ?? '',
        }));
        break;

      case 'response.created':
        onEvent(makeEvent(Events.SPEAKING_STARTED, {}));
        break;
      case 'response.audio_transcript.delta':
        onEvent(makeEvent(Events.AGENT_TEXT_DELTA, { text: raw.delta ?? '' }));
        break;
      case 'response.audio_transcript.done':
        onEvent(makeEvent(Events.AGENT_TEXT_FINAL, { text: raw.transcript ?? '' }));
        break;
      case 'response.done':
        onEvent(makeEvent(Events.SPEAKING_STOPPED, {}));
        onEvent(makeEvent(Events.RESPONSE_DONE, {}));
        break;

      case 'response.function_call_arguments.done': {
        let args = {};
        try {
          args = raw.arguments ? JSON.parse(raw.arguments) : {};
        } catch (_err) {
          args = { _raw: raw.arguments ?? '' };
        }
        onEvent(makeEvent(Events.TOOL_CALL_REQUESTED, {
          call_id: raw.call_id ?? '',
          name: raw.name ?? '',
          arguments: args,
        }));
        break;
      }

      case 'error':
        onEvent(makeEvent(Events.ERROR, {
          message: raw.error?.message ?? 'Unknown Realtime error',
          code: raw.error?.code ?? null,
          fatal: false,
        }));
        break;

      default:
        // Unmapped event — fine to ignore. Uncomment for debugging.
        // console.debug('unmapped realtime event', raw);
        break;
    }
  }

  async function start() {
    if (session) throw new Error('Realtime client already started');

    // 1. Mint session from backend. Auth headers come from `getAuthHeaders`
    // so the voice UI shell can wire in `API.applyAuth({})` (which honors
    // whatever provider is active — API key, JWT, OAuth) the same way every
    // other Kestrel endpoint authenticates. Without this the request gets a
    // 401 against any server with auth enabled.
    const authHeaders = (await getAuthHeaders()) || {};
    const resp = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...authHeaders,
      },
      body: JSON.stringify(sessionRequestBody),
      ...sessionRequestInit,
    });
    if (resp.status === 409) {
      // Resolver rejected realtime — caller should fall back to pipeline.
      const body = await resp.json();
      const err = new Error(body.reason || 'Realtime unavailable');
      err.code = 'REALTIME_UNAVAILABLE';
      err.fallback = body;  // { path, fallback_tts, fallback_stt, reason }
      throw err;
    }
    if (!resp.ok) {
      throw new Error(`Session mint failed: HTTP ${resp.status}`);
    }
    session = await resp.json();

    // 2. RTCPeerConnection + mic track.
    pc = new RTCPeerConnection();
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    for (const track of micStream.getTracks()) {
      pc.addTrack(track, micStream);
    }

    // 3. Remote audio → <audio> sink.
    audioSink = document.createElement('audio');
    audioSink.autoplay = true;
    // Don't attach to the DOM — it's a headless sink, avoids CSS interactions.
    pc.ontrack = (ev) => {
      if (ev.streams && ev.streams[0]) {
        audioSink.srcObject = ev.streams[0];
      }
    };

    // 4. Data channel — OpenAI accepts either a client-created or
    // server-created channel; creating it client-side gives us more control
    // over the `ordered` + `negotiated` knobs later.
    dc = pc.createDataChannel('oai-events');
    dc.onmessage = (ev) => {
      try {
        const parsed = JSON.parse(ev.data);
        handleOpenAIEvent(parsed);
      } catch (err) {
        onEvent(makeEvent(Events.ERROR, {
          message: `Malformed event from OpenAI: ${err.message}`,
          fatal: false,
        }));
      }
    };
    dc.onclose = () => {
      if (!closed) {
        onEvent(makeEvent(Events.SESSION_CLOSED, { reason: 'data_channel_closed' }));
      }
    };

    // 5. Track connection-level failures fatally so the UI can reset.
    pc.onconnectionstatechange = () => {
      if (closed || !pc) return;
      if (pc.connectionState === 'failed') {
        onEvent(makeEvent(Events.ERROR, {
          message: 'RTC connection failed',
          code: 'rtc_failed',
          fatal: true,
        }));
      } else if (pc.connectionState === 'disconnected') {
        onEvent(makeEvent(Events.ERROR, {
          message: 'RTC connection lost',
          code: 'rtc_disconnected',
          fatal: true,
        }));
      }
    };

    // 6. SDP exchange — direct to OpenAI with the ephemeral token.
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    const sdpResp = await fetch(
      `${REALTIME_SDP_URL}?model=${encodeURIComponent(session.model)}`,
      {
        method: 'POST',
        body: offer.sdp,
        headers: {
          'Authorization': `Bearer ${session.client_secret.value}`,
          'Content-Type': 'application/sdp',
        },
      },
    );
    if (!sdpResp.ok) {
      throw new Error(`SDP exchange failed: HTTP ${sdpResp.status}`);
    }
    const answer = { type: 'answer', sdp: await sdpResp.text() };
    await pc.setRemoteDescription(answer);
  }

  async function close() {
    if (closed) return;
    closed = true;
    try { dc?.close(); } catch (_) {}
    try {
      if (micStream) {
        for (const t of micStream.getTracks()) t.stop();
      }
    } catch (_) {}
    try { pc?.close(); } catch (_) {}
    try {
      if (audioSink) {
        audioSink.pause();
        audioSink.srcObject = null;
      }
    } catch (_) {}
    onEvent(makeEvent(Events.SESSION_CLOSED, { reason: 'client_close' }));
  }

  /** Queue a text message into the conversation (without speaking it aloud). */
  function sendText(text) {
    sendJSON({
      type: 'conversation.item.create',
      item: {
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text }],
      },
    });
    sendJSON({ type: 'response.create' });
  }

  /** Barge-in: abort the in-flight agent response. */
  function cancelResponse() {
    sendJSON({ type: 'response.cancel' });
  }

  /**
   * Replace the session's ``instructions`` field mid-session. Used by the
   * sovereign Realtime tag adapter (#724) to surface per-turn expressive
   * directives composed from canonical tags.
   */
  function updateInstructions(instructions) {
    sendJSON({
      type: 'session.update',
      session: { instructions },
    });
  }

  /**
   * Return a tool's result to a pending call_id so the model continues.
   * `result` will be JSON-stringified; pass a plain serializable value.
   */
  function commitToolResult(callId, result) {
    sendJSON({
      type: 'conversation.item.create',
      item: {
        type: 'function_call_output',
        call_id: callId,
        output: JSON.stringify(result),
      },
    });
    sendJSON({ type: 'response.create' });
  }

  /** Input-level sampling for the UI meter. Returns 0..1. */
  function getInputLevel() {
    if (!micStream || !pc) return 0;
    // Minimal implementation — the UI shell can instantiate its own
    // AnalyserNode over `micStream` if it wants a real meter. Returning 0
    // here avoids adding a persistent AudioContext just for measurement.
    return 0;
  }

  return {
    path: 'realtime',
    start,
    close,
    sendText,
    cancelResponse,
    updateInstructions,
    commitToolResult,
    getInputLevel,
    /** MediaStream of remote agent audio — for UI meters / visualizers. */
    get remoteStream() {
      return audioSink?.srcObject ?? null;
    },
    /** Live mic stream — lets the UI attach an AnalyserNode for a real level meter. */
    get micStream() {
      return micStream;
    },
    /** Currently-active session bundle (null before start()). */
    get session() {
      return session;
    },
  };
}
