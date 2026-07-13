/**
 * realtime.js — Provider-neutral browser client for Realtime voice paths.
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
 * 2. Open the provider-declared browser transport. WebRTC providers use a
 *    native microphone track and remote audio stream; WebSocket providers use
 *    Kestrel's PCM16 capture/playback AudioWorklets.
 *
 * 3. Translate the compatible provider event stream into Kestrel voice events
 *    (see events.js). The UI shell does not branch on provider event taxonomies.
 *
 * Barge-in: provider-side VAD handles turn detection; when the user starts
 * speaking, Kestrel also flushes queued WebSocket audio immediately.
 * We also forward `speech_started` as LISTENING_STARTED so the UI can drop
 * its speaking indicator immediately.
 *
 * Tool calls are collected for the complete response, dispatched as one batch
 * through Kestrel governance, and committed before a single continuation.
 */

import { Events, makeEvent } from './events.js';
import { createVoiceCapture } from './capture.js';
import { createVoicePlayback } from './playback.js';

// GA WebRTC endpoint.  The Beta path was ``/v1/realtime`` (with the
// model as a query string), but OpenAI's GA Realtime moved WebRTC SDP
// exchange under ``/v1/realtime/calls`` (the SDK exposes it as
// ``client.realtime.calls.create``).  Posting to the old path now
// 400s with the SDP body — the browser sees "SDP exchange failed:
// HTTP 400".  See kestrel-voice-openai#16 (Beta -> GA migration).
const REALTIME_SDP_URL = 'https://api.openai.com/v1/realtime/calls';

export function buildRealtimeToolsSessionUpdate(tools = []) {
  return {
    type: 'session.update',
    session: {
      tools: Array.isArray(tools) ? tools : [],
    },
  };
}

export function bytesToBase64(bytes) {
  let binary = '';
  for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

export function base64ToBytes(value) {
  const binary = atob(value || '');
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

export function applyTranscriptUpdate(current, text, cumulative = false) {
  return cumulative ? (text || '') : `${current || ''}${text || ''}`;
}

export function normalizeToolBatchResults(calls = [], results = []) {
  const byId = new Map(
    (Array.isArray(results) ? results : [])
      .filter((item) => item && typeof item === 'object')
      .map((item) => [item.call_id, item]),
  );
  return (Array.isArray(calls) ? calls : []).map((call) => {
    const item = byId.get(call.call_id);
    const hasResult = item && Object.prototype.hasOwnProperty.call(item, 'result');
    return {
      call_id: call.call_id,
      result: hasResult
        ? item.result
        : { error: 'tool dispatch returned no result' },
    };
  });
}

export function resolveRealtimeSDPEndpoint(session = {}) {
  if (session.endpoint) return session.endpoint;
  const model = String(session.model || '').trim();
  return model
    ? `${REALTIME_SDP_URL}?model=${encodeURIComponent(model)}`
    : REALTIME_SDP_URL;
}

export async function waitForPlaybackIdle(playbackController, timeoutMs = 2000) {
  if (typeof playbackController?.whenIdle !== 'function') return;
  let timer = null;
  try {
    await Promise.race([
      playbackController.whenIdle(),
      new Promise((resolve) => { timer = setTimeout(resolve, timeoutMs); }),
    ]);
  } finally {
    if (timer !== null) clearTimeout(timer);
  }
}

export function responseAllowsToolDispatch(raw = {}) {
  const status = String(raw.response?.status ?? raw.status ?? '').toLowerCase();
  return !['cancelled', 'canceled', 'failed', 'incomplete'].includes(status);
}


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
  let ws = null;         // WebSocket realtime transport (xAI and future providers)
  let micStream = null;  // MediaStream from getUserMedia
  let audioSink = null;  // <audio> element for remote playback
  let capture = null;    // PCM capture for WebSocket transports
  let playback = null;   // PCM playback for WebSocket transports
  let inputMuted = false;
  let outputMuted = false;
  let closed = false;
  const persistedTurnIds = new Set();  // item_ids already sent to /transcript
  const pendingPersists = new Set();   // in-flight /transcript POST promises
  let pendingToolEvents = [];          // tool cards to attach to next assistant turn
  let pendingToolCalls = [];           // collected until response.done
  let cumulativeUserTranscript = '';
  let earlyAudioChunks = [];
  let earlyAudioBytes = 0;
  const maxEarlyAudioBytes = 24000 * 2 * 2; // two seconds of mono PCM16

  /**
   * Send a JSON control message through the data channel.
   * Swallows send errors when the channel is closing — barge-in races are
   * common and we don't want to promote them to UI errors.
   */
  function sendJSON(msg) {
    const channelOpen = dc && dc.readyState === 'open';
    const socketOpen = ws && ws.readyState === WebSocket.OPEN;
    if (!channelOpen && !socketOpen) return;
    try {
      const payload = JSON.stringify(msg);
      if (channelOpen) dc.send(payload);
      else ws.send(payload);
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
   * Report a finalized turn to the backend so it lands in the agent's
   * conversation history (#1808). The realtime path is end-to-end between this
   * browser and its provider, so the server never sees the transcript unless we send
   * it. Fire-and-forget: a persistence failure must never interrupt the live
   * call, so errors are surfaced as non-fatal events only.
   *
   * The transcript endpoint is derived from the mint `endpoint` by swapping the
   * trailing `/session` for `/transcript/<session_id>`, reusing the same auth.
   *
   * `dedupeId` guards against double-persisting one turn: GA emits BOTH
   * `response.output_audio_transcript.done` and `response.output_text.done`
   * for the same assistant item (audio + text content parts share an item_id),
   * which would otherwise write the reply twice.
   */
  function persistTurn(role, text, dedupeId) {
    const content = (text ?? '').trim();
    const sessionId = session?.session_id;
    if (!content || !sessionId) return;
    if (dedupeId) {
      if (persistedTurnIds.has(dedupeId)) return;
      persistedTurnIds.add(dedupeId);
    }
    // Drain buffered tool cards onto this assistant turn so they persist
    // alongside the spoken reply (realtime runs tools before speaking, so they
    // belong to the reply that follows). ONLY consume the buffer for assistant
    // turns: a user transcript can finalize late (after a tool already ran),
    // and draining it there would lose the cards before the assistant turn.
    let toolEvents = [];
    if (role === 'assistant' && pendingToolEvents.length) {
      toolEvents = pendingToolEvents;
      pendingToolEvents = [];
    }
    const transcriptUrl =
      endpoint.replace(/\/session$/, '/transcript/') +
      encodeURIComponent(sessionId);
    const p = Promise.resolve(getAuthHeaders())
      .then((authHeaders) =>
        fetch(transcriptUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...(authHeaders || {}) },
          body: JSON.stringify({
            role,
            content,
            item_id: dedupeId || '',
            tool_events: toolEvents,  // only populated for assistant turns
          }),
        }),
      )
      .then((resp) => {
        // fetch() only rejects on network failure — an HTTP 401/404/500
        // still resolves. Without this check a failed persist would look
        // like success and the turn would be silently dropped from history.
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      })
      .catch((err) => {
        if (!closed) {
          onEvent(makeEvent(Events.ERROR, {
            message: `transcript persist failed: ${err.message}`,
            fatal: false,
          }));
        }
      });
    // Track the in-flight POST so callers (the UI's sidebar refresh on session
    // close) can wait for persistence to settle before refetching history —
    // otherwise the refresh races ahead of the write and shows a stale list.
    pendingPersists.add(p);
    p.finally(() => pendingPersists.delete(p));
  }

  /**
   * Resolve once all transcript POSTs issued so far have settled (succeeded
   * or failed). Lets the UI refresh the conversation list only after the new
   * turns are actually persisted. Never rejects.
   */
  function whenPersisted() {
    return Promise.allSettled([...pendingPersists]);
  }

  /**
   * Buffer a tool-activity event ({type, tool, ms?, error?, pos}) to attach to
   * the next assistant turn that persists. The UI's tool-dispatch handler calls
   * this as a realtime tool call starts and completes, so the cards land on the
   * spoken reply that follows and survive a reload. `pos` is 0 — realtime runs
   * tools before the reply, so the cards render above its prose.
   */
  function recordToolEvent(ev) {
    if (ev && typeof ev === 'object') pendingToolEvents.push(ev);
  }

  /**
   * Translate the OpenAI-compatible provider event stream to Kestrel events.
   *
   * Unknown types are logged at debug level and skipped — OpenAI can add new
   * events without breaking this client. Missing fields fall back to empty
   * values so the UI shell never sees undefined text.
   */
  function handleProviderEvent(raw) {
    switch (raw.type) {
      case 'session.created':
        onEvent(makeEvent(Events.SESSION_READY, {
          session_id: raw.session?.id ?? session?.session_id ?? '',
          model: session?.model ?? raw.session?.model ?? '',
          path: 'realtime',
        }));
        break;

      case 'input_audio_buffer.speech_started':
        // A barge-in abandons the prior response. Completed arguments from
        // that interrupted response must not execute or bleed into the next.
        pendingToolCalls = [];
        playback?.flush?.();
        onEvent(makeEvent(Events.LISTENING_STARTED, {}));
        break;
      case 'input_audio_buffer.speech_stopped':
        onEvent(makeEvent(Events.LISTENING_STOPPED, {}));
        break;

      case 'conversation.item.input_audio_transcription.delta':
        cumulativeUserTranscript = applyTranscriptUpdate(
          cumulativeUserTranscript, raw.delta ?? '', false,
        );
        onEvent(makeEvent(Events.USER_TRANSCRIPT_DELTA, {
          text: cumulativeUserTranscript,
          is_final: false,
        }));
        break;
      case 'conversation.item.input_audio_transcription.updated': {
        cumulativeUserTranscript = applyTranscriptUpdate(
          cumulativeUserTranscript, raw.transcript ?? '', true,
        );
        onEvent(makeEvent(Events.USER_TRANSCRIPT_DELTA, {
          text: cumulativeUserTranscript,
          is_final: false,
        }));
        break;
      }
      case 'conversation.item.input_audio_transcription.completed':
        onEvent(makeEvent(Events.USER_TRANSCRIPT_FINAL, {
          text: raw.transcript ?? '',
        }));
        persistTurn('user', raw.transcript ?? '', raw.item_id);
        cumulativeUserTranscript = '';
        break;

      case 'response.created':
        onEvent(makeEvent(Events.SPEAKING_STARTED, {}));
        break;
      // GA renamed audio-transcript events to the `output_audio_transcript`
      // namespace; the text-modality stream is `output_text`. Without these
      // the agent's spoken reply never reaches the chat window because the
      // delta events fall through to the default branch.  See
      // kestrel-voice-openai#16 (Beta -> GA migration).
      case 'response.output_audio_transcript.delta':
      case 'response.output_text.delta':
        onEvent(makeEvent(Events.AGENT_TEXT_DELTA, { text: raw.delta ?? '' }));
        break;
      case 'response.output_audio_transcript.done':
      case 'response.output_text.done':
        onEvent(makeEvent(Events.AGENT_TEXT_FINAL, {
          text: raw.transcript ?? raw.text ?? '',
        }));
        persistTurn('assistant', raw.transcript ?? raw.text ?? '', raw.item_id);
        break;
      case 'response.output_audio.delta':
        if (playback && raw.delta) playback.enqueue(base64ToBytes(raw.delta));
        break;
      case 'response.done':
        playback?.endOfStream?.();
        if (pendingToolCalls.length && responseAllowsToolDispatch(raw)) {
          const calls = pendingToolCalls;
          pendingToolCalls = [];
          onEvent(makeEvent(Events.TOOL_CALL_BATCH_REQUESTED, {
            batch_id: raw.response?.id ?? raw.response_id ?? '',
            calls,
          }));
        } else {
          pendingToolCalls = [];
        }
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
        pendingToolCalls.push({
          call_id: raw.call_id ?? '',
          name: raw.name ?? '',
          arguments: args,
        });
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

  async function startWebRTC() {
    pc = new RTCPeerConnection();
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    for (const track of micStream.getTracks()) pc.addTrack(track, micStream);

    audioSink = document.createElement('audio');
    audioSink.autoplay = true;
    audioSink.muted = outputMuted;
    pc.ontrack = (ev) => {
      if (ev.streams && ev.streams[0]) audioSink.srcObject = ev.streams[0];
    };

    dc = pc.createDataChannel('oai-events');
    dc.onmessage = (ev) => {
      try { handleProviderEvent(JSON.parse(ev.data)); }
      catch (err) {
        onEvent(makeEvent(Events.ERROR, {
          message: `Malformed realtime event: ${err.message}`, fatal: false,
        }));
      }
    };
    dc.onclose = () => {
      if (!closed) onEvent(makeEvent(Events.SESSION_CLOSED, { reason: 'data_channel_closed' }));
    };
    pc.onconnectionstatechange = () => {
      if (closed || !pc) return;
      if (pc.connectionState === 'failed' || pc.connectionState === 'disconnected') {
        onEvent(makeEvent(Events.ERROR, {
          message: `RTC connection ${pc.connectionState}`,
          code: `rtc_${pc.connectionState}`,
          fatal: true,
        }));
      }
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    const sdpResp = await fetch(resolveRealtimeSDPEndpoint(session), {
      method: 'POST',
      body: offer.sdp,
      headers: {
        'Authorization': `Bearer ${session.client_secret.value}`,
        'Content-Type': 'application/sdp',
      },
    });
    if (!sdpResp.ok) throw new Error(`SDP exchange failed: HTTP ${sdpResp.status}`);
    await pc.setRemoteDescription({ type: 'answer', sdp: await sdpResp.text() });
  }

  async function startWebSocket() {
    capture = await createVoiceCapture({ targetSampleRate: 24000 });
    playback = await createVoicePlayback({ sampleRate: 24000 });
    playback.setMuted?.(outputMuted);
    capture.onchunk((pcm) => {
      if (inputMuted || closed) return;
      if (ws?.readyState === WebSocket.OPEN) {
        sendJSON({ type: 'input_audio_buffer.append', audio: bytesToBase64(pcm) });
        return;
      }
      const copy = new Uint8Array(pcm);
      earlyAudioChunks.push(copy);
      earlyAudioBytes += copy.byteLength;
      while (earlyAudioBytes > maxEarlyAudioBytes && earlyAudioChunks.length) {
        earlyAudioBytes -= earlyAudioChunks.shift().byteLength;
      }
    });

    const subprotocols = session.vendor === 'xai'
      ? [`xai-client-secret.${session.client_secret.value}`]
      : [];
    ws = new WebSocket(session.endpoint, subprotocols);
    ws.onmessage = (ev) => {
      try { handleProviderEvent(JSON.parse(ev.data)); }
      catch (err) {
        onEvent(makeEvent(Events.ERROR, {
          message: `Malformed realtime event: ${err.message}`, fatal: false,
        }));
      }
    };
    await new Promise((resolve, reject) => {
      ws.onopen = () => {
        if (session.session_config && Object.keys(session.session_config).length) {
          sendJSON({ type: 'session.update', session: session.session_config });
        }
        for (const pcm of earlyAudioChunks) {
          sendJSON({ type: 'input_audio_buffer.append', audio: bytesToBase64(pcm) });
        }
        earlyAudioChunks = [];
        earlyAudioBytes = 0;
        resolve();
      };
      ws.onerror = () => reject(new Error(`${session.provider || 'Realtime'} WebSocket failed`));
    });
    ws.onclose = () => {
      if (!closed) onEvent(makeEvent(Events.SESSION_CLOSED, { reason: 'websocket_closed' }));
    };
    ws.onerror = () => {
      if (!closed) {
        onEvent(makeEvent(Events.ERROR, {
          message: `${session.provider || 'Realtime'} WebSocket error`,
          code: 'websocket_error',
          fatal: true,
        }));
      }
    };
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

    if (session.transport === 'websocket') await startWebSocket();
    else if (session.transport === 'webrtc' || !session.transport) await startWebRTC();
    else throw new Error(`Unsupported realtime transport: ${session.transport}`);
  }

  async function close() {
    if (closed) return;
    closed = true;
    try { dc?.close(); } catch (_) {}
    try { ws?.close(); } catch (_) {}
    try {
      if (micStream) {
        for (const t of micStream.getTracks()) t.stop();
      }
    } catch (_) {}
    try { pc?.close(); } catch (_) {}
    try { await capture?.destroy?.(); } catch (_) {}
    try { await playback?.destroy?.(); } catch (_) {}
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
    // Typed turns don't produce an input_audio_transcription.completed event,
    // so persist the user side here or backend history would be one-sided
    // (assistant reply only). The assistant turn still persists via
    // response.output_*.done.
    persistTurn('user', text);
  }

  /** Barge-in: abort the in-flight agent response. */
  function cancelResponse() {
    pendingToolCalls = [];
    sendJSON({ type: 'response.cancel' });
    playback?.flush?.();
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
   * Replace the session's advertised realtime tools mid-call. The server owns
   * progressive disclosure and sends the current bounded registry view; the
   * browser only forwards it over the existing data channel.
   */
  function updateTools(tools) {
    sendJSON(buildRealtimeToolsSessionUpdate(tools));
  }

  /**
   * Return a tool's result to a pending call_id so the model continues.
   * `result` will be JSON-stringified; pass a plain serializable value.
   */
  async function commitToolResults(results = []) {
    if (!results.length) return;
    for (const { call_id: callId, result } of results) {
      sendJSON({
        type: 'conversation.item.create',
        item: {
          type: 'function_call_output',
          call_id: callId,
          output: JSON.stringify(result),
        },
      });
    }
    // xAI can finish delivering response audio before the browser has played
    // it. Waiting here prevents the post-tool response from overlapping the
    // prior response; WebRTC providers resolve immediately.
    await waitForPlaybackIdle(playback);
    sendJSON({ type: 'response.create' });
  }

  async function commitToolResult(callId, result) {
    await commitToolResults([{ call_id: callId, result }]);
  }

  /** Input-level sampling for the UI meter. Returns 0..1. */
  function getInputLevel() {
    if (capture) return capture.getLevel();
    if (!micStream || !pc) return 0;
    // Minimal implementation — the UI shell can instantiate its own
    // AnalyserNode over `micStream` if it wants a real meter. Returning 0
    // here avoids adding a persistent AudioContext just for measurement.
    return 0;
  }

  function setMuted(muted) {
    outputMuted = !!muted;
    if (audioSink) audioSink.muted = !!muted;
    playback?.setMuted?.(!!muted);
  }

  // Gate the OUTGOING mic path. Disabling the local audio track makes WebRTC
  // transmit silence, so a backgrounded agent stops hearing the user (no
  // hidden turns under its pane/privacy mode) without tearing down the peer
  // connection — re-enabling resumes instantly on return.
  function setInputMuted(muted) {
    inputMuted = !!muted;
    if (capture) {
      if (inputMuted) capture.pause();
      else capture.resume();
    }
    if (!micStream) return;
    for (const t of micStream.getAudioTracks()) t.enabled = !muted;
  }

  return {
    path: 'realtime',
    start,
    close,
    sendText,
    cancelResponse,
    updateInstructions,
    updateTools,
    commitToolResult,
    commitToolResults,
    whenPersisted,
    recordToolEvent,
    getInputLevel,
    setMuted,
    setInputMuted,
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
