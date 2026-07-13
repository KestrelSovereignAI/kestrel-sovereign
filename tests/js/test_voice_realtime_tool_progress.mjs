/** Provider-neutral Realtime utility tests. */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  applyTranscriptUpdate,
  base64ToBytes,
  buildRealtimeToolsSessionUpdate,
  bytesToBase64,
  normalizeToolBatchResults,
  responseAllowsToolDispatch,
  resolveRealtimeSDPEndpoint,
  waitForPlaybackIdle,
} from '../../kestrel_sovereign/static/js/voice/realtime.js';

test('realtime tool disclosure update uses session.update tools payload', () => {
  const tools = [{ type: 'function', name: 'memory_feature', parameters: { type: 'object' } }];
  assert.deepEqual(buildRealtimeToolsSessionUpdate(tools), {
    type: 'session.update',
    session: { tools },
  });
  assert.deepEqual(buildRealtimeToolsSessionUpdate(null), {
    type: 'session.update',
    session: { tools: [] },
  });
});

test('PCM chunks round trip through provider WebSocket base64 encoding', () => {
  const input = new Uint8Array([0, 1, 127, 128, 254, 255]);
  assert.deepEqual(base64ToBytes(bytesToBase64(input)), input);
});

test('incremental and cumulative provider transcripts share one UI contract', () => {
  assert.equal(applyTranscriptUpdate('hello', ' world', false), 'hello world');
  assert.equal(applyTranscriptUpdate('hello wurld', 'hello world', true), 'hello world');
});

test('partial batch responses still produce one result per requested tool', () => {
  const calls = [
    { call_id: 'one', name: 'first' },
    { call_id: 'two', name: 'second' },
  ];
  assert.deepEqual(normalizeToolBatchResults(calls, [
    { call_id: 'one', result: { ok: true } },
  ]), [
    { call_id: 'one', result: { ok: true } },
    { call_id: 'two', result: { error: 'tool dispatch returned no result' } },
  ]);
  assert.equal(normalizeToolBatchResults(calls, null).length, 2);
});

test('WebRTC fallback keeps the discovered model query', () => {
  assert.equal(
    resolveRealtimeSDPEndpoint({ model: 'runtime model' }),
    'https://api.openai.com/v1/realtime/calls?model=runtime%20model',
  );
  assert.equal(
    resolveRealtimeSDPEndpoint({ endpoint: 'https://provider.example/calls', model: 'ignored' }),
    'https://provider.example/calls',
  );
});

test('playback idle wait has a bounded continuation timeout', async () => {
  const started = Date.now();
  await waitForPlaybackIdle({ whenIdle: () => new Promise(() => {}) }, 5);
  assert.ok(Date.now() - started < 250);
});

test('cancelled and failed responses never dispatch collected tools', () => {
  assert.equal(responseAllowsToolDispatch({ response: { status: 'completed' } }), true);
  assert.equal(responseAllowsToolDispatch({}), true); // xAI-compatible absent status
  assert.equal(responseAllowsToolDispatch({ response: { status: 'cancelled' } }), false);
  assert.equal(responseAllowsToolDispatch({ response: { status: 'failed' } }), false);
  assert.equal(responseAllowsToolDispatch({ response: { status: 'incomplete' } }), false);
});
