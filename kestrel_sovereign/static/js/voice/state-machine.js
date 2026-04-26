/**
 * state-machine.js — Pure mapping from voice events to UI states.
 *
 * Extracted from ui.js so the state transitions can be unit-tested in Node
 * without a DOM. The UI shell calls `nextStateForEvent(currentState, eventKind)`
 * on every event from the active client and only mutates DOM when the
 * function returns a state that differs from the current one.
 *
 * Returns the new state, or `null` to mean "no transition for this event".
 *
 * The mapping is tolerant: unknown event kinds → null (preserves state),
 * unknown current states → still applies the event-driven mapping if the
 * target makes sense, otherwise null.
 */

import { Events } from './events.js';

export const State = Object.freeze({
  IDLE: 'idle',
  CONNECTING: 'connecting',
  LISTENING: 'listening',
  THINKING: 'thinking',
  SPEAKING: 'speaking',
  ERROR: 'error',
});

/**
 * @param {string} currentState  — current State.* value.
 * @param {string} eventKind     — Events.* value.
 * @param {Object} [eventPayload]  — optional event payload (used only for
 *   ERROR.fatal — fatal escalates to ERROR state, non-fatal stays put).
 * @returns {string|null}  next State.* value, or null for no transition.
 */
export function nextStateForEvent(currentState, eventKind, eventPayload = {}) {
  switch (eventKind) {
    case Events.SESSION_READY:
      // First server ack — session is alive, mic is hot.
      return State.LISTENING;

    case Events.SESSION_CLOSED:
      // Server-side close — drop to idle unless we already escalated to
      // ERROR (in which case the user needs to dismiss the error visually).
      return currentState === State.ERROR ? null : State.IDLE;

    case Events.LISTENING_STARTED:
      return State.LISTENING;

    case Events.LISTENING_STOPPED:
      // Both Realtime and Pipeline lead to "thinking" once the user stops.
      return State.THINKING;

    case Events.THINKING_STARTED:
      return State.THINKING;

    case Events.SPEAKING_STARTED:
      return State.SPEAKING;

    case Events.SPEAKING_STOPPED:
    case Events.RESPONSE_DONE:
      // Back to listening — mic is still open for the next user turn.
      return State.LISTENING;

    case Events.ERROR:
      // Only fatal errors escalate to the ERROR state; transient errors
      // are surfaced as transcript lines but the session keeps running.
      return eventPayload.fatal ? State.ERROR : null;

    case Events.USER_TRANSCRIPT_DELTA:
    case Events.USER_TRANSCRIPT_FINAL:
    case Events.AGENT_TEXT_DELTA:
    case Events.AGENT_TEXT_FINAL:
    case Events.TOOL_CALL_REQUESTED:
      // Pure-content events don't change the visible state — they're
      // rendered into the transcript by ui.js, but the mic-button state
      // stays where it is.
      return null;

    default:
      return null;
  }
}
