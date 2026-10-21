/**
 * capture.js — Microphone capture → 24kHz PCM16 chunks.
 *
 * Public API:
 *   const cap = await createVoiceCapture({ targetSampleRate: 24000 });
 *   cap.onchunk((pcm) => sendToServer(pcm));    // Uint8Array, ~20ms frames
 *   cap.getLevel();    // peak 0..1 for UI level meter
 *   cap.pause(); cap.resume(); cap.destroy();
 *   const requestedGeneration = cap.reset(); // NOT a completion receipt
 *
 * Uses AudioWorklet (required; ScriptProcessorNode is deprecated). The
 * worklet runs on the audio thread and posts PCM16 frames back here.
 */

const WORKLET_URL = '/static/js/voice/capture-worklet.js';

export async function createVoiceCapture({ targetSampleRate = 24000, onAcquire } = {}) {
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

  let ctx;
  let source;
  let node;
  let chunkCallback = null;
  let currentLevel = 0;
  let retired = false;
  let closeOwner;
  let generation = 0;
  function postReset() {
    if (node) node.port.postMessage({ type: 'reset', generation });
  }
  const owner = {
    // Preserve the ORIGINAL stream even after retirement so consumers can
    // inspect its ended tracks instead of borrowing a replacement capture.
    get micStream() { return stream; },
    onchunk(fn) { chunkCallback = fn; },
    getLevel() { return currentLevel; },
    reset() {
      requireActive();
      generation += 1;
      currentLevel = 0;
      postReset();
      // This is a request token only. A matching ACTUAL native PCM frame
      // proves that reset was processed; this number grants no admission.
      return generation;
    },
    pause() { node?.port.postMessage({ type: 'pause' }); },
    resume() { if (!retired) node?.port.postMessage({ type: 'resume' }); },
    destroy() {
      retired = true;
      try { source?.disconnect(); } catch (_) {}
      try { node?.disconnect(); } catch (_) {}
      for (const track of stream.getTracks()) track.stop();
      if (!closeOwner) closeOwner = (async () => {
        try { await ctx?.close(); } catch (_) {}
      })();
      return closeOwner;
    },
  };
  function requireActive() {
    if (retired) throw new Error('Microphone capture retired during initialization');
  }
  try {
  // Publish custody before any subsequent suspension. Stop can retire a real
  // microphone even when resume/worklet loading never finishes.
  onAcquire?.(owner);
  requireActive();
  ctx = new (window.AudioContext || window.webkitAudioContext)();
  // Safari may start the context in "suspended" state on a gesture-less page.
  if (ctx.state === 'suspended') {
    try { await ctx.resume(); } catch (_) { /* non-fatal */ }
  }
  requireActive();

  await ctx.audioWorklet.addModule(WORKLET_URL);
  requireActive();

  source = ctx.createMediaStreamSource(stream);
  node = new AudioWorkletNode(ctx, 'kestrel-capture', {
    numberOfInputs: 1,
    numberOfOutputs: 0,  // processor consumes; nothing routes to speakers
    channelCount: 1,
  });

  node.port.postMessage({ type: 'configure', targetSampleRate });

  node.port.onmessage = (ev) => {
    if (retired) return;
    const msg = ev.data;
    if (!msg || msg.type !== 'chunk') return;
    if (msg.generation !== generation) return;
    currentLevel = msg.level || 0;
    if (chunkCallback) {
      chunkCallback(new Uint8Array(msg.pcm), { generation: msg.generation });
    }
  };

  // Stop can request an input boundary through the onAcquire owner while its
  // original worklet is loading. Deliver the latest generation, not a new
  // capture/stream or a fabricated ready signal.
  if (generation > 0) postReset();

  source.connect(node);
  // Don't connect `node` to destination — we don't want to echo the mic.

  return owner;
  } catch (error) {
    await owner.destroy();
    throw error;
  }
}
