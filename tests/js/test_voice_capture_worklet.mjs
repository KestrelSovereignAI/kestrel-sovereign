/** PCM transform units only; real microphone/delayed-frame proof is native. */
import { test } from 'node:test';
import assert from 'node:assert/strict';

let CaptureProcessor;
globalThis.sampleRate = 48000;
globalThis.AudioWorkletProcessor = class {
  constructor() {
    this.port = { messages: [], onmessage: null,
      postMessage: message => this.port.messages.push(message) };
  }
};
globalThis.registerProcessor = (name, processor) => {
  assert.equal(name, 'kestrel-capture');
  CaptureProcessor = processor;
};
await import('../../kestrel_sovereign/static/js/voice/capture-worklet.js');

function render(processor, value, quanta) {
  for (let count = 0; count < quanta; count++) {
    processor.process([[new Float32Array(128).fill(value)]]);
  }
}

test('reset discards partial old PCM before the first fresh generation frame', () => {
  const processor = new CaptureProcessor();
  render(processor, .5, 1); // 64 old resampled samples, short of a frame.
  processor.port.onmessage({ data: { type: 'reset', generation: 1 } });
  render(processor, -.5, 8);
  const chunks = processor.port.messages.filter(message => message.type === 'chunk');
  assert.equal(chunks.length, 1);
  assert.ok([...new Int16Array(chunks[0].pcm)].every(sample => sample < 0), 'old partial microphone prefix survived reset');
  assert.equal(chunks[0].generation, 1);
});

test('stale reset cannot reopen an older capture generation', () => {
  const processor = new CaptureProcessor();
  processor.port.onmessage({ data: { type: 'reset', generation: 2 } });
  render(processor, .5, 1);
  processor.port.onmessage({ data: { type: 'reset', generation: 1 } });
  render(processor, .5, 7);
  const chunks = processor.port.messages.filter(message => message.type === 'chunk');
  assert.equal(chunks.length, 1, 'stale reset discarded current partial PCM');
  assert.equal(chunks[0].generation, 2);
});

test('pause/resume preserves the current utterance and does not invent a reset', () => {
  const processor = new CaptureProcessor();
  render(processor, .5, 1);
  processor.port.onmessage({ data: { type: 'pause' } });
  render(processor, -.5, 8);
  processor.port.onmessage({ data: { type: 'resume' } });
  render(processor, .5, 7);
  const chunks = processor.port.messages.filter(message => message.type === 'chunk');
  assert.equal(chunks.length, 1);
  assert.ok([...new Int16Array(chunks[0].pcm)].every(sample => sample > 0));
  assert.equal(chunks[0].generation, 0);
});
