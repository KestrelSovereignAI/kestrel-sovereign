/**
 * capture.js — Microphone capture → 24kHz PCM16 chunks.
 *
 * Public API:
 *   const cap = await createVoiceCapture({ targetSampleRate: 24000 });
 *   cap.onchunk((pcm) => sendToServer(pcm));    // Uint8Array, ~20ms frames
 *   cap.getLevel();    // peak 0..1 for UI level meter
 *   cap.pause(); cap.resume(); cap.destroy();
 *
 * Uses AudioWorklet (required; ScriptProcessorNode is deprecated). The
 * worklet runs on the audio thread and posts PCM16 frames back here.
 */

const WORKLET_URL = '/static/js/voice/capture-worklet.js';

export async function createVoiceCapture({ targetSampleRate = 24000 } = {}) {
  if (typeof AudioWorkletNode === 'undefined') {
    throw new Error('AudioWorklet is not supported in this browser.');
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error('getUserMedia is not available (requires HTTPS or localhost).');
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });

  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  // Safari may start the context in "suspended" state on a gesture-less page.
  if (ctx.state === 'suspended') {
    try { await ctx.resume(); } catch (_) { /* non-fatal */ }
  }

  await ctx.audioWorklet.addModule(WORKLET_URL);

  const source = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, 'kestrel-capture', {
    numberOfInputs: 1,
    numberOfOutputs: 0,  // processor consumes; nothing routes to speakers
    channelCount: 1,
  });

  node.port.postMessage({ type: 'configure', targetSampleRate });

  let chunkCallback = null;
  let currentLevel = 0;

  node.port.onmessage = (ev) => {
    const msg = ev.data;
    if (!msg || msg.type !== 'chunk') return;
    currentLevel = msg.level || 0;
    if (chunkCallback) {
      chunkCallback(new Uint8Array(msg.pcm));
    }
  };

  source.connect(node);
  // Don't connect `node` to destination — we don't want to echo the mic.

  return {
    onchunk(fn) {
      chunkCallback = fn;
    },
    getLevel() {
      return currentLevel;
    },
    pause() {
      node.port.postMessage({ type: 'pause' });
    },
    resume() {
      node.port.postMessage({ type: 'resume' });
    },
    async destroy() {
      try { source.disconnect(); } catch (_) {}
      try { node.disconnect(); } catch (_) {}
      for (const t of stream.getTracks()) t.stop();
      try { await ctx.close(); } catch (_) {}
    },
  };
}
