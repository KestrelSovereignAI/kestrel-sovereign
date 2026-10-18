/**
 * Node-run contract tests for the browser-side PCM playback controller.
 *
 *   node --test tests/js/test_voice_playback.mjs
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

let lastNode;
let lastGain;
let lastContext;

class FakeAudioContext {
  constructor({ sampleRate }) {
    this.sampleRate = sampleRate;
    this.state = 'running';
    this.destination = {};
    this.audioWorklet = { addModule: async () => {} };
    lastContext = this;
  }

  createGain() {
    lastGain = {
      gain: { value: 1 },
      connect() {},
      disconnect() {},
    };
    return lastGain;
  }

  async close() { this.state = 'closed'; }
}

globalThis.window = { AudioContext: FakeAudioContext };
globalThis.AudioWorkletNode = class {
  constructor(context, name, options) {
    this.context = context;
    this.name = name;
    this.options = options;
    this.messages = [];
    this.port = {
      onmessage: null,
      postMessage: (message) => this.messages.push(message),
    };
    lastNode = this;
  }

  connect() {}
  disconnect() {}
};

// Load the browser ES module through a data URL so this test remains portable
// to Node versions that treat repository `.js` files as CommonJS in the
// absence of a package-wide `type` declaration.
const playbackSource = await readFile(
  new URL('../../kestrel_sovereign/static/js/voice/playback.js', import.meta.url),
  'utf8',
);
const { createVoicePlayback } = await import(
  `data:text/javascript;base64,${Buffer.from(playbackSource).toString('base64')}`
);

test('configures worklet pre-roll and sends PCM immediately', async () => {
  const playback = await createVoicePlayback({ sampleRate: 24_000, preRollMs: 400 });
  const pcm = new Uint8Array([1, 2, 3, 4]);

  assert.deepEqual(lastNode.options.processorOptions, {
    preRollSamples: 9600,
    underflowSamples: 4800,
  });
  playback.enqueue(pcm);

  assert.equal(lastNode.messages.length, 1);
  assert.equal(lastNode.messages[0].type, 'push');
  assert.deepEqual([...new Uint8Array(lastNode.messages[0].pcm)], [...pcm]);
  await playback.destroy();
});

test('waits for drained after end while underflow remains non-terminal', async () => {
  const playback = await createVoicePlayback();
  playback.enqueue(new Uint8Array([1, 2]));
  playback.endOfStream();

  assert.deepEqual(lastNode.messages.map((message) => message.type), ['push', 'end']);
  let idle = false;
  const idlePromise = playback.whenIdle().then(() => { idle = true; });

  lastNode.port.onmessage({ data: { type: 'underflow', generation: 0 } });
  await Promise.resolve();
  assert.equal(idle, false);
  assert.equal(playback.isPlaying(), true);
  assert.equal(playback.underflowCount(), 1);

  lastNode.port.onmessage({ data: { type: 'drained', generation: 0 } });
  await idlePromise;
  assert.equal(playback.isPlaying(), false);
  await playback.destroy();
});

test('flush terminates playback and releases idle waiters', async () => {
  const playback = await createVoicePlayback();
  playback.enqueue(new Uint8Array([1, 2]));
  const idlePromise = playback.whenIdle();

  const clear = playback.flush();
  await idlePromise;

  assert.equal(playback.isPlaying(), false);
  assert.equal(lastNode.messages.at(-1).type, 'flush');
  playback.setMuted(false);
  assert.equal(lastGain.gain.value, 0);
  lastNode.port.onmessage({ data: { type: 'cleared', generation: 1 } });
  assert.deepEqual(await clear, { generation: 1, cleared: true });
  await playback.destroy();
});

test('repeated flush shares one owner and stale ACK/drain cannot open fresh playback', async () => {
  const playback = await createVoicePlayback();
  const first = playback.flush();
  assert.equal(playback.flush(), first);
  playback.enqueue(new Uint8Array([1, 2]));
  playback.endOfStream();
  assert.deepEqual(lastNode.messages.map(message => message.type), ['flush', 'flush']);
  let settled = false;
  first.then(() => { settled = true; });
  lastNode.port.onmessage({ data: { type: 'cleared', generation: 1 } });
  await Promise.resolve();
  assert.equal(settled, false);
  assert.equal(lastGain.gain.value, 0);
  playback.setMuted(true);
  lastNode.port.onmessage({ data: { type: 'cleared', generation: 2 } });
  assert.deepEqual(await first, { generation: 2, cleared: true });
  assert.equal(lastGain.gain.value, 0, 'clear ACK bypassed independent user mute');
  assert.deepEqual(lastNode.messages.slice(2).map(message => [message.type, message.generation]), [['push', 2], ['end', 2]]);
  playback.setMuted(false);
  assert.equal(lastGain.gain.value, 1);
  lastNode.port.onmessage({ data: { type: 'drained', generation: 1 } });
  assert.equal(playback.isPlaying(), true);
  lastNode.port.onmessage({ data: { type: 'drained', generation: 2 } });
  assert.equal(playback.isPlaying(), false);
  await playback.destroy();
});

test('a missing clear ACK retires playback and exposes the original failure', async () => {
  const errors = [];
  const playback = await createVoicePlayback({ clearTimeoutMs: 10, onError: error => errors.push(error) });
  await assert.rejects(playback.flush(), /did not acknowledge/);
  assert.equal(errors.length, 1);
  assert.throws(() => playback.enqueue(new Uint8Array([1, 2])), error => error === errors[0]);
  playback.setMuted(false);
  assert.equal(lastGain.gain.value, 0);
  await playback.destroy();
  assert.equal(lastContext.state, 'closed');
});

test('a later pending PCM push supersedes an earlier end marker before clear ACK', async () => {
  const playback = await createVoicePlayback();
  const clear = playback.flush();
  try {
    playback.enqueue(new Uint8Array([1, 2]));
    playback.endOfStream();
    playback.enqueue(new Uint8Array([3, 4]));
    lastNode.port.onmessage({ data: { type: 'cleared', generation: 1 } });
    await clear;
    assert.deepEqual(lastNode.messages.map(message => message.type), ['flush', 'push', 'end', 'push']);
    assert.equal(playback.isPlaying(), true);
  } finally {
    await playback.destroy();
  }
});

test('pending-clear audio overflow fails explicitly instead of discarding frames', async () => {
  const playback = await createVoicePlayback({ maxPendingAudioMs: 1 });
  const clear = playback.flush();
  assert.throws(() => playback.enqueue(new Uint8Array(50)), /budget exceeded/);
  await assert.rejects(clear, /budget exceeded/);
  assert.equal(lastGain.gain.value, 0);
  await playback.destroy();
});
