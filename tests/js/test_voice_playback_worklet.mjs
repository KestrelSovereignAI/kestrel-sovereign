/**
 * Node-run regression tests for the PCM playback AudioWorklet.
 *
 *   node --test tests/js/test_voice_playback_worklet.mjs
 *
 * AudioWorklet globals are stubbed just enough to instantiate the real
 * processor. The tests then drive its message port and render quantum-sized
 * output buffers exactly as a browser AudioContext would.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

let PlaybackProcessor;

globalThis.AudioWorkletProcessor = class {
  constructor() {
    this.port = {
      messages: [],
      onmessage: null,
      postMessage: (message) => this.port.messages.push(message),
    };
  }
};
globalThis.registerProcessor = (name, processor) => {
  assert.equal(name, 'kestrel-playback');
  PlaybackProcessor = processor;
};

await import('../../kestrel_sovereign/static/js/voice/playback-worklet.js');

function createProcessor({ preRollSamples = 0, underflowSamples = 2400 } = {}) {
  return new PlaybackProcessor({
    processorOptions: { preRollSamples, underflowSamples },
  });
}

function send(processor, data) {
  processor.port.onmessage({ data });
}

function render(processor, sampleCount) {
  const rendered = new Float32Array(sampleCount);
  let offset = 0;
  while (offset < sampleCount) {
    const quantumLength = Math.min(128, sampleCount - offset);
    const quantum = new Float32Array(quantumLength);
    processor.process([], [[quantum]]);
    rendered.set(quantum, offset);
    offset += quantumLength;
  }
  return rendered;
}

function toFloat32(sample) {
  return sample < 0 ? sample / 0x8000 : sample / 0x7fff;
}

test('preserves a provider burst larger than the old four-second ring', () => {
  const processor = createProcessor();
  const pcm = new Int16Array(120_000);
  for (let index = 0; index < pcm.length; index++) {
    pcm[index] = (index % 30_000) + 1;
  }

  send(processor, { type: 'push', pcm: pcm.buffer });
  const rendered = render(processor, pcm.length);

  assert.equal(rendered.length, pcm.length);
  for (let index = 0; index < pcm.length; index++) {
    assert.equal(rendered[index], Math.fround(toFloat32(pcm[index])));
  }
});

test('preserves sample order while reclaiming many consumed chunks', () => {
  const processor = createProcessor();
  const expected = [];

  for (let chunkIndex = 0; chunkIndex < 130; chunkIndex++) {
    const pcm = new Int16Array(3);
    for (let sampleIndex = 0; sampleIndex < pcm.length; sampleIndex++) {
      const sample = (chunkIndex * pcm.length) + sampleIndex + 1;
      pcm[sampleIndex] = sample;
      expected.push(Math.fround(toFloat32(sample)));
    }
    send(processor, { type: 'push', pcm: pcm.buffer });
  }
  send(processor, { type: 'end' });

  assert.deepEqual([...render(processor, expected.length)], expected);
  assert.deepEqual(processor.port.messages, [{ type: 'drained' }]);
});

test('holds initial audio until pre-roll fills', () => {
  const processor = createProcessor({ preRollSamples: 4 });
  const first = new Int16Array([1000, 2000]);
  const second = new Int16Array([3000, 4000]);

  send(processor, { type: 'push', pcm: first.buffer });
  assert.deepEqual([...render(processor, 2)], [0, 0]);

  send(processor, { type: 'push', pcm: second.buffer });
  assert.deepEqual(
    [...render(processor, 4)],
    [1000, 2000, 3000, 4000].map((sample) => Math.fround(toFloat32(sample))),
  );
});

test('end releases a short response and reports drained after playback', () => {
  const processor = createProcessor({ preRollSamples: 8 });
  const pcm = new Int16Array([1000, 2000, 3000]);

  send(processor, { type: 'push', pcm: pcm.buffer });
  send(processor, { type: 'end' });

  assert.deepEqual(
    [...render(processor, pcm.length)],
    [...pcm].map((sample) => Math.fround(toFloat32(sample))),
  );
  assert.deepEqual(processor.port.messages, [{ type: 'drained' }]);
});

test('temporary underflow re-establishes pre-roll without losing later audio', () => {
  const processor = createProcessor({ preRollSamples: 4, underflowSamples: 2 });
  const first = new Int16Array([1000, 2000, 3000, 4000]);
  const second = new Int16Array([5000, 6000]);
  const third = new Int16Array([7000, 8000]);

  send(processor, { type: 'push', pcm: first.buffer });
  render(processor, first.length);
  assert.deepEqual([...render(processor, 2)], [0, 0]);
  assert.deepEqual(processor.port.messages, [{ type: 'underflow' }]);

  send(processor, { type: 'push', pcm: second.buffer });
  assert.deepEqual([...render(processor, 2)], [0, 0]);
  send(processor, { type: 'push', pcm: third.buffer });
  assert.deepEqual(
    [...render(processor, 4)],
    [5000, 6000, 7000, 8000].map((sample) => Math.fround(toFloat32(sample))),
  );
});

test('flush discards queued audio for barge-in', () => {
  const processor = createProcessor();
  const pcm = new Int16Array([1000, 2000, 3000]);

  send(processor, { type: 'push', pcm: pcm.buffer });
  send(processor, { type: 'flush' });

  assert.deepEqual([...render(processor, pcm.length)], [0, 0, 0]);
  assert.deepEqual(processor.port.messages, []);
});
