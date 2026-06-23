/**
 * Node-run unit tests for Realtime long-running tool progress hints.
 *
 *   node --test tests/js/test_voice_realtime_tool_progress.mjs
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  DEFAULT_TOOL_PROGRESS_HINT_DELAY_MS,
  buildToolProgressHintMessages,
  createToolProgressHintScheduler,
  resolveToolProgressHintDelay,
} from '../../kestrel_sovereign/static/js/voice/realtime.js';

function makeTimerHarness() {
  const timers = [];
  const cleared = new Set();
  return {
    timers,
    cleared,
    setTimeoutFn(fn, ms) {
      const id = timers.length;
      timers.push({ fn, ms });
      return id;
    },
    clearTimeoutFn(id) {
      cleared.add(id);
    },
    fire(id = 0) {
      if (!cleared.has(id)) timers[id].fn();
    },
  };
}

test('tool progress delay defaults to three seconds and accepts config override', () => {
  assert.equal(DEFAULT_TOOL_PROGRESS_HINT_DELAY_MS, 3000);
  assert.equal(resolveToolProgressHintDelay({ tool_progress_hint_delay_ms: 1250 }), 1250);
  assert.equal(resolveToolProgressHintDelay({ toolBridgeDelayMs: 10 }), 10);
  assert.equal(resolveToolProgressHintDelay({ tool_progress_hint_delay_ms: 'bad' }), 3000);
});

test('fast tool call cancels progress hint before it fires', () => {
  const harness = makeTimerHarness();
  const sent = [];
  const scheduler = createToolProgressHintScheduler({
    delayMs: 3000,
    sendHint: (payload) => sent.push(payload),
    setTimeoutFn: harness.setTimeoutFn.bind(harness),
    clearTimeoutFn: harness.clearTimeoutFn.bind(harness),
  });

  const pending = scheduler.start({ callId: 'call_fast', toolName: 'quick_lookup' });
  assert.equal(harness.timers[0].ms, 3000);
  pending.finish();
  harness.fire(0);

  assert.deepEqual(sent, []);
  assert.ok(harness.cleared.has(0));
});

test('slow tool call emits one working hint after threshold', () => {
  const harness = makeTimerHarness();
  const sent = [];
  const scheduler = createToolProgressHintScheduler({
    delayMs: 3000,
    sendHint: (payload) => sent.push(payload),
    setTimeoutFn: harness.setTimeoutFn.bind(harness),
    clearTimeoutFn: harness.clearTimeoutFn.bind(harness),
  });

  const pending = scheduler.start({ callId: 'call_slow', toolName: 'web_research' });
  harness.fire(0);
  harness.fire(0);
  pending.finish();

  assert.deepEqual(sent, [{
    phase: 'working',
    callId: 'call_slow',
    toolName: 'web_research',
  }]);
  assert.equal(harness.cleared.size, 0);
});

test('tool progress hint uses conversation item plus response create', () => {
  const messages = buildToolProgressHintMessages({
    callId: 'call_123',
    toolName: 'subagent_dispatch',
  });

  assert.equal(messages.length, 2);
  assert.equal(messages[0].type, 'conversation.item.create');
  assert.equal(messages[0].item.type, 'message');
  assert.equal(messages[0].item.role, 'user');
  assert.match(messages[0].item.content[0].text, /tool status: working/);
  assert.match(messages[0].item.content[0].text, /subagent_dispatch/);
  assert.equal(messages[1].type, 'response.create');
  assert.match(messages[1].response.instructions, /one short bridge phrase/);
});
