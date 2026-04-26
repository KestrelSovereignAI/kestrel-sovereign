/**
 * playback.js — Jitter-buffered PCM16 playback.
 *
 * Public API:
 *   const pb = createVoicePlayback({ sampleRate: 24000 });
 *   pb.enqueue(uint8Array);   // PCM16 samples at configured rate
 *   pb.flush();               // drop all buffered audio (barge-in)
 *   pb.isPlaying();
 *   pb.destroy();
 *
 * The worklet manages a ring buffer and accepts 'push' / 'flush' control
 * messages from main. Jitter-buffer pre-roll (30ms by default) smooths
 * network irregularity without adding perceptible latency.
 */

const WORKLET_URL = '/static/js/voice/playback-worklet.js';

// Default preroll = 400ms. The server's TTS coalescer paces audio at ~120ms
// cadence, but the FIRST chunk waits on upstream OpenAI/ElevenLabs round-trip
// (~360ms observed). 400ms preroll absorbs that initial latency once and the
// jitter buffer stays full thereafter; smaller preroll causes audible stutter
// at the start of every reply.
export async function createVoicePlayback({ sampleRate = 24000, preRollMs = 400 } = {}) {
  if (typeof AudioWorkletNode === 'undefined') {
    throw new Error('AudioWorklet is not supported in this browser.');
  }
  const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate });
  if (ctx.state === 'suspended') {
    try { await ctx.resume(); } catch (_) { /* non-fatal; user gesture may be required */ }
  }
  await ctx.audioWorklet.addModule(WORKLET_URL);

  const node = new AudioWorkletNode(ctx, 'kestrel-playback', {
    numberOfInputs: 0,
    numberOfOutputs: 1,
    outputChannelCount: [1],
  });
  node.connect(ctx.destination);

  // Pre-roll: hold the first N ms of audio before connecting so short initial
  // chunks don't stutter. We implement this by buffering on main until
  // enough samples have arrived, then flushing them to the worklet at once.
  const preRollSamples = Math.floor((preRollMs / 1000) * sampleRate);
  let preRollBuffer = [];
  let preRollFilled = false;
  let preRollCount = 0;

  let playing = false;
  let underflowCount = 0;

  node.port.onmessage = (ev) => {
    const msg = ev.data;
    if (!msg) return;
    if (msg.type === 'underflow') {
      underflowCount++;
      playing = false;
    }
  };

  function pushToWorklet(uint8) {
    // uint8 is a PCM16 view (even byte count).
    const copy = new Uint8Array(uint8);
    node.port.postMessage({ type: 'push', pcm: copy.buffer }, [copy.buffer]);
  }

  return {
    enqueue(uint8) {
      if (!uint8 || uint8.byteLength === 0) return;
      playing = true;
      if (!preRollFilled) {
        preRollBuffer.push(uint8);
        preRollCount += uint8.byteLength / 2; // 2 bytes per Int16 sample
        if (preRollCount >= preRollSamples) {
          for (const c of preRollBuffer) pushToWorklet(c);
          preRollBuffer = [];
          preRollFilled = true;
        }
      } else {
        pushToWorklet(uint8);
      }
    },
    flush() {
      preRollBuffer = [];
      preRollFilled = false;
      preRollCount = 0;
      playing = false;
      node.port.postMessage({ type: 'flush' });
    },
    isPlaying() {
      return playing;
    },
    underflowCount() {
      return underflowCount;
    },
    async destroy() {
      try { node.disconnect(); } catch (_) {}
      try { await ctx.close(); } catch (_) {}
    },
  };
}
