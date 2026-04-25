/**
 * Node-run unit tests for the voice UI state machine.
 *
 *   node --test tests/js/test_voice_state_machine.mjs
 *
 * Pure-function transitions (no DOM, no clients), so these run in Node's
 * built-in test runner. The DOM glue in ui.js is exercised via Playwright
 * E2E in #731.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { Events } from '../../kestrel_sovereign/static/js/voice/events.js';
import {
  State,
  nextStateForEvent,
} from '../../kestrel_sovereign/static/js/voice/state-machine.js';

test('SESSION_READY → LISTENING from any state', () => {
  for (const start of Object.values(State)) {
    assert.equal(nextStateForEvent(start, Events.SESSION_READY), State.LISTENING);
  }
});

test('SESSION_CLOSED returns IDLE except from ERROR', () => {
  for (const start of [State.IDLE, State.LISTENING, State.SPEAKING, State.THINKING, State.CONNECTING]) {
    assert.equal(nextStateForEvent(start, Events.SESSION_CLOSED), State.IDLE);
  }
  // From ERROR, returns null so the user has to dismiss the error visually.
  assert.equal(nextStateForEvent(State.ERROR, Events.SESSION_CLOSED), null);
});

test('LISTENING_STARTED → LISTENING', () => {
  assert.equal(nextStateForEvent(State.IDLE, Events.LISTENING_STARTED), State.LISTENING);
  assert.equal(nextStateForEvent(State.SPEAKING, Events.LISTENING_STARTED), State.LISTENING);
});

test('LISTENING_STOPPED → THINKING', () => {
  assert.equal(nextStateForEvent(State.LISTENING, Events.LISTENING_STOPPED), State.THINKING);
});

test('SPEAKING_STARTED → SPEAKING', () => {
  assert.equal(nextStateForEvent(State.THINKING, Events.SPEAKING_STARTED), State.SPEAKING);
});

test('SPEAKING_STOPPED → LISTENING (mic stays open for next turn)', () => {
  assert.equal(nextStateForEvent(State.SPEAKING, Events.SPEAKING_STOPPED), State.LISTENING);
});

test('RESPONSE_DONE → LISTENING', () => {
  assert.equal(nextStateForEvent(State.SPEAKING, Events.RESPONSE_DONE), State.LISTENING);
});

test('ERROR with fatal=true → ERROR state', () => {
  assert.equal(
    nextStateForEvent(State.LISTENING, Events.ERROR, { fatal: true }),
    State.ERROR,
  );
});

test('ERROR with fatal=false → null (no transition)', () => {
  assert.equal(
    nextStateForEvent(State.LISTENING, Events.ERROR, { fatal: false }),
    null,
  );
  assert.equal(nextStateForEvent(State.LISTENING, Events.ERROR), null);
});

test('content-only events return null (no state transition)', () => {
  for (const ek of [
    Events.USER_TRANSCRIPT_DELTA,
    Events.USER_TRANSCRIPT_FINAL,
    Events.AGENT_TEXT_DELTA,
    Events.AGENT_TEXT_FINAL,
    Events.TOOL_CALL_REQUESTED,
  ]) {
    assert.equal(nextStateForEvent(State.LISTENING, ek), null);
  }
});

test('unknown event kind returns null (no transition)', () => {
  assert.equal(nextStateForEvent(State.LISTENING, 'made_up_event'), null);
});

test('full session: ready → listen → think → speak → listen → close', () => {
  let s = State.CONNECTING;
  s = nextStateForEvent(s, Events.SESSION_READY);
  assert.equal(s, State.LISTENING);
  s = nextStateForEvent(s, Events.LISTENING_STARTED);
  assert.equal(s, State.LISTENING);
  s = nextStateForEvent(s, Events.LISTENING_STOPPED);
  assert.equal(s, State.THINKING);
  s = nextStateForEvent(s, Events.SPEAKING_STARTED);
  assert.equal(s, State.SPEAKING);
  s = nextStateForEvent(s, Events.SPEAKING_STOPPED);
  assert.equal(s, State.LISTENING);
  s = nextStateForEvent(s, Events.SESSION_CLOSED);
  assert.equal(s, State.IDLE);
});

test('State enum has all required values', () => {
  for (const name of ['IDLE', 'CONNECTING', 'LISTENING', 'THINKING', 'SPEAKING', 'ERROR']) {
    assert.ok(State[name], `State.${name} missing`);
  }
});

test('State enum is frozen', () => {
  assert.throws(() => {
    State.NEW = 'new';
  });
});
