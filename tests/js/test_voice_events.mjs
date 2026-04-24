/**
 * Node-run unit tests for the shared voice event module.
 *
 *   node --test tests/js/test_voice_events.mjs
 *
 * The voice event module is pure JS — no DOM, no browser APIs — so it runs
 * in Node's built-in test runner. The clients that consume it
 * (realtime.js, pipeline.js) require browser APIs (RTCPeerConnection,
 * WebSocket, AudioContext) and are exercised via Playwright E2E in #731.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { Events, makeEvent } from '../../kestrel_sovereign/static/js/voice/events.js';

test('Events constants include all documented kinds', () => {
  const required = [
    'SESSION_READY', 'SESSION_CLOSED',
    'LISTENING_STARTED', 'LISTENING_STOPPED',
    'THINKING_STARTED', 'SPEAKING_STARTED', 'SPEAKING_STOPPED',
    'USER_TRANSCRIPT_DELTA', 'USER_TRANSCRIPT_FINAL',
    'AGENT_TEXT_DELTA', 'AGENT_TEXT_FINAL',
    'RESPONSE_DONE', 'TOOL_CALL_REQUESTED',
    'ERROR',
  ];
  for (const name of required) {
    assert.ok(Events[name], `Events.${name} missing`);
    assert.equal(typeof Events[name], 'string');
  }
});

test('Events object is frozen', () => {
  // Freezing prevents accidental mutation when clients share the module.
  assert.throws(() => {
    Events.NEW_EVENT = 'new_event';
  });
});

test('makeEvent attaches the kind', () => {
  const ev = makeEvent(Events.LISTENING_STARTED);
  assert.deepEqual(ev, { kind: 'listening_started' });
});

test('makeEvent merges payload without overwriting kind', () => {
  const ev = makeEvent(Events.USER_TRANSCRIPT_DELTA, { text: 'hello', is_final: false });
  assert.equal(ev.kind, 'user_transcript_delta');
  assert.equal(ev.text, 'hello');
  assert.equal(ev.is_final, false);
});

test('makeEvent rejects unknown kinds', () => {
  assert.throws(() => makeEvent('not_a_real_kind'), /Unknown voice event kind/);
});

test('makeEvent survives missing payload', () => {
  const ev = makeEvent(Events.RESPONSE_DONE);
  assert.equal(ev.kind, 'response_done');
  assert.equal(Object.keys(ev).length, 1);
});
