/**
 * events.js — Shared event shape for both voice clients.
 *
 * The Realtime WebRTC client (#728) and the Pipeline WebSocket client (#729)
 * wrap very different transports but both drive the same UI shell (#730).
 * This module defines the single event vocabulary both of them emit, so the
 * UI shell branches on semantic state — never on transport.
 *
 * Every event is an object with a `kind` string discriminator. Payload fields
 * are documented per kind below. Clients that fire an event type MUST include
 * all documented fields (use empty string / null for missing). Unknown kinds
 * MAY be emitted for forward compatibility but the UI shell will ignore them.
 */

export const Events = Object.freeze({
  // Session lifecycle
  SESSION_READY: 'session_ready',        // { session_id, path, model? }
  SESSION_CLOSED: 'session_closed',      // { reason }

  // Turn state (for UI state indicator + level meter gating)
  LISTENING_STARTED: 'listening_started',   // user mic active, agent waiting
  LISTENING_STOPPED: 'listening_stopped',   // user stopped speaking; response coming
  THINKING_STARTED:  'thinking_started',    // agent is processing (pipeline path)
  SPEAKING_STARTED:  'speaking_started',    // agent response audio began
  SPEAKING_STOPPED:  'speaking_stopped',    // response completed or canceled

  // Streamed transcripts (for live captions)
  USER_TRANSCRIPT_DELTA: 'user_transcript_delta',   // { text, is_final:false }
  USER_TRANSCRIPT_FINAL: 'user_transcript_final',   // { text }
  AGENT_TEXT_DELTA:      'agent_text_delta',        // { text } (partial)
  AGENT_TEXT_FINAL:      'agent_text_final',        // { text } (complete)

  // Response completion
  RESPONSE_DONE: 'response_done',        // {}

  // Tool calls — realtime only; pipeline doesn't surface these
  TOOL_CALL_REQUESTED: 'tool_call_requested', // { call_id, name, arguments }

  // Errors. `fatal` signals the session is finished; UI should reset.
  ERROR: 'error',                        // { message, code?, fatal }
});

/**
 * Create an event object safely — ensures `kind` is one of the known values
 * and payload fields aren't forgotten.
 */
export function makeEvent(kind, payload = {}) {
  if (!Object.values(Events).includes(kind)) {
    throw new Error(`Unknown voice event kind: ${kind}`);
  }
  return { kind, ...payload };
}
