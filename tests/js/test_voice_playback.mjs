/**
 * Node-run contract tests for the browser-side PCM playback controller.
 *
 *   node --test tests/js/test_voice_playback.mjs
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

let lastNode;

class FakeAudioContext {
  constructor({ sampleRate }) {
    this.sampleRate = sampleRate;
    this.state = 'running';
    this.destination = {};
    this.audioWorklet = { addModule: async () => {} };
  }

  createGain() {
    return {
      gain: { value: 1 },
      connect() {},
      disconnect() {},
    };
  }

  async close() {}
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

  lastNode.port.onmessage({ data: { type: 'underflow' } });
  await Promise.resolve();
  assert.equal(idle, false);
  assert.equal(playback.isPlaying(), true);
  assert.equal(playback.underflowCount(), 1);

  lastNode.port.onmessage({ data: { type: 'drained' } });
  await idlePromise;
  assert.equal(playback.isPlaying(), false);
  await playback.destroy();
});

test('flush terminates playback and releases idle waiters', async () => {
  const playback = await createVoicePlayback();
  playback.enqueue(new Uint8Array([1, 2]));
  const idlePromise = playback.whenIdle();

  playback.flush();
  await idlePromise;

  assert.equal(playback.isPlaying(), false);
  assert.equal(lastNode.messages.at(-1).type, 'flush');
  await playback.destroy();
});
