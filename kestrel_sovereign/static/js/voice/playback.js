/**
 * playback.js — Lossless jitter-buffered PCM16 playback.
 *
 * Public API:
 *   const pb = createVoicePlayback({ sampleRate: 24000 });
 *   pb.enqueue(uint8Array);   // PCM16 samples at configured rate
 *   pb.endOfStream();         // release a short response still in pre-roll
 *   await pb.whenIdle();      // wait until queued samples finish playing
 *   pb.flush();               // drop all buffered audio (barge-in)
 *   pb.isPlaying();
 *   pb.destroy();
 *
 * The worklet manages a lossless PCM chunk queue and accepts 'push' / 'end' /
 * 'flush' control messages from main. Jitter-buffer pre-roll smooths network
 * irregularity without imposing a fixed cap on provider delivery bursts.
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
    processorOptions: {
      preRollSamples: Math.floor((preRollMs / 1000) * sampleRate),
      underflowSamples: Math.floor(sampleRate / 5),
    },
  });
  const gain = ctx.createGain();
  node.connect(gain);
  gain.connect(ctx.destination);

  let playing = false;
  let underflowCount = 0;
  const idleWaiters = new Set();

  function resolveIdleWaiters() {
    for (const resolve of idleWaiters) resolve();
    idleWaiters.clear();
  }

  node.port.onmessage = (ev) => {
    const msg = ev.data;
    if (!msg) return;
    if (msg.type === 'underflow') {
      underflowCount++;
    } else if (msg.type === 'drained') {
      playing = false;
      resolveIdleWaiters();
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
      pushToWorklet(uint8);
    },
    endOfStream() {
      node.port.postMessage({ type: 'end' });
      if (!playing) resolveIdleWaiters();
    },
    whenIdle() {
      if (!playing) return Promise.resolve();
      return new Promise((resolve) => idleWaiters.add(resolve));
    },
    flush() {
      playing = false;
      node.port.postMessage({ type: 'flush' });
      resolveIdleWaiters();
    },
    isPlaying() {
      return playing;
    },
    underflowCount() {
      return underflowCount;
    },
    setMuted(muted) {
      gain.gain.value = muted ? 0 : 1;
    },
    async destroy() {
      playing = false;
      resolveIdleWaiters();
      try { node.disconnect(); } catch (_) {}
      try { gain.disconnect(); } catch (_) {}
      try { await ctx.close(); } catch (_) {}
    },
  };
}
