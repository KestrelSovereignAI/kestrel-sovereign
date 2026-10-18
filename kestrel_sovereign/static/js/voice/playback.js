/**
 * playback.js — Lossless jitter-buffered PCM16 playback.
 *
 * Public API:
 *   const pb = createVoicePlayback({ sampleRate: 24000 });
 *   pb.enqueue(uint8Array);   // PCM16 samples at configured rate
 *   pb.endOfStream();         // release a short response still in pre-roll
 *   await pb.whenIdle();      // wait until queued samples finish playing
 *   await pb.flush();         // gate immediately; join real worklet clear ACK
 *   pb.isPlaying();
 *   pb.destroy();
 *
 * The worklet manages a lossless PCM chunk queue and accepts 'push' / 'end' /
 * 'flush' control messages tagged with their original playback generation.
 * User mute is independent of the clear gate; queued fresh PCM waits for the
 * exact clear ACK. Repeated clears share one bounded owner/deadline. Missing
 * ACK or pending-audio overflow retires playback and exposes an error.
 * Jitter-buffer pre-roll smooths network
 * irregularity without imposing a fixed cap on provider delivery bursts.
 */

const WORKLET_URL = '/static/js/voice/playback-worklet.js';

// Default preroll = 400ms. The server's TTS coalescer paces audio at ~120ms
// cadence, but the FIRST chunk waits on upstream OpenAI/ElevenLabs round-trip
// (~360ms observed). 400ms preroll absorbs that initial latency once and the
// jitter buffer stays full thereafter; smaller preroll causes audible stutter
// at the start of every reply.
export async function createVoicePlayback({ sampleRate = 24000, preRollMs = 400, onAcquire,
  onError, clearTimeoutMs = 2000, maxPendingAudioMs = 2000 } = {}) {
  if (typeof AudioWorkletNode === 'undefined') {
    throw new Error('AudioWorklet is not supported in this browser.');
  }
  if (![clearTimeoutMs, maxPendingAudioMs].every(value => Number.isFinite(value) && value > 0)) {
    throw new Error('Playback clear timeout and pending-audio budget must be finite and positive.');
  }
  const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate });
  let node;
  let gain;
  let retired = false;
  let closeOwner;
  let playing = false;
  let underflowCount = 0;
  let generation = 0;
  let muted = false;
  let failure = null;
  let pendingClear = null;
  let pendingOperations = [];
  let pendingBytes = 0;
  const maxPendingBytes = Math.ceil(sampleRate * 2 * maxPendingAudioMs / 1000);
  const idleWaiters = new Set();
  function resolveIdleWaiters() {
    for (const resolve of idleWaiters) resolve();
    idleWaiters.clear();
  }
  function requireActive() {
    if (failure) throw failure;
    if (retired) throw new Error('Audio playback retired during initialization');
  }
  function applyGain() {
    if (gain) gain.gain.value = muted || pendingClear || retired || failure ? 0 : 1;
  }
  function settleClear(error) {
    const original = pendingClear;
    if (!original) return;
    pendingClear = null;
    clearTimeout(original.timer);
    if (error) original.reject(error);
    else original.resolve({ generation, cleared: true });
  }
  function fail(error) {
    if (failure || retired) return;
    failure = error;
    applyGain();
    settleClear(error);
    void owner.destroy();
    onError?.(error);
  }
  function pushToWorklet(uint8) {
    // uint8 is a PCM16 view (even byte count).
    const copy = new Uint8Array(uint8);
    try {
      node.port.postMessage({ type: 'push', generation, pcm: copy.buffer }, [copy.buffer]);
    } catch (error) { fail(error); throw error; }
  }
  function postClear() {
    // onAcquire publishes the owner BEFORE worklet loading can suspend.
    // Remember an early Stop; setup forwards only its latest generation once
    // the original node/handler exist, without restarting its deadline.
    if (!node) return;
    try { node.port.postMessage({ type: 'flush', generation }); }
    catch (error) { fail(error); throw error; }
  }
  const owner = {
    enqueue(uint8) {
      if (failure) throw failure;
      if (retired || !uint8 || uint8.byteLength === 0) return;
      playing = true;
      if (pendingClear) {
        if (pendingBytes + uint8.byteLength > maxPendingBytes) {
          const error = new Error('Playback clear acknowledgement pending-audio budget exceeded.');
          fail(error);
          throw error;
        }
        pendingOperations.push({ type: 'push', pcm: new Uint8Array(uint8) });
        pendingBytes += uint8.byteLength;
        return;
      }
      pushToWorklet(uint8);
    },
    endOfStream() {
      if (failure) throw failure;
      if (pendingClear) {
        // Preserve the actual producer order (push/end/push is NOT
        // push/push/end). Adjacent duplicate ends are idempotent and coalesce,
        // so metadata cannot grow without the bounded PCM-byte inventory.
        if (pendingOperations.at(-1)?.type !== 'end') pendingOperations.push({ type: 'end' });
      }
      else if (!retired) {
        try { node?.port.postMessage({ type: 'end', generation }); }
        catch (error) { fail(error); throw error; }
      }
      if (!playing) resolveIdleWaiters();
    },
    whenIdle() {
      if (failure) return Promise.reject(failure);
      if (!playing) return Promise.resolve();
      return new Promise((resolve) => idleWaiters.add(resolve));
    },
    flush() {
      if (failure) throw failure;
      if (retired) return Promise.resolve({ generation, cleared: false, retired: true });
      generation += 1;
      playing = false;
      pendingOperations = [];
      pendingBytes = 0;
      // Repeated Stop shares ONE bounded owner, not an ever-growing waiter
      // inventory or a deadline that every new Stop can extend indefinitely.
      if (!pendingClear) {
        let resolve, reject;
        const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
        // Legacy synchronous Stop callers need no rejection handler. They do
        // not turn failure into success: the owner retires, emits onError and
        // all later operations expose its original failure.
        promise.catch(() => {});
        pendingClear = { promise, resolve, reject, timer: setTimeout(() => {
          fail(new Error('Playback worklet did not acknowledge the original clear before its deadline.'));
        }, clearTimeoutMs) };
      }
      applyGain(); // Physical gate FIRST; user unmute cannot reopen it.
      const promise = pendingClear.promise;
      postClear();
      resolveIdleWaiters();
      return promise;
    },
    isPlaying() { return playing; },
    underflowCount() { return underflowCount; },
    setMuted(value) { muted = !!value; applyGain(); },
    destroy() {
      retired = true;
      applyGain();
      settleClear(failure || new Error('Playback retired before worklet clear acknowledgement.'));
      pendingOperations = [];
      pendingBytes = 0;
      playing = false;
      resolveIdleWaiters();
      try { node?.disconnect(); } catch (_) {}
      try { gain?.disconnect(); } catch (_) {}
      if (!closeOwner) closeOwner = (async () => {
        try { await ctx.close(); } catch (_) {}
      })();
      return closeOwner;
    },
  };
  try {
  onAcquire?.(owner);
  requireActive();
  if (ctx.state === 'suspended') {
    try { await ctx.resume(); } catch (_) { /* non-fatal; user gesture may be required */ }
  }
  requireActive();
  await ctx.audioWorklet.addModule(WORKLET_URL);
  requireActive();

  node = new AudioWorkletNode(ctx, 'kestrel-playback', {
    numberOfInputs: 0,
    numberOfOutputs: 1,
    outputChannelCount: [1],
    processorOptions: {
      preRollSamples: Math.floor((preRollMs / 1000) * sampleRate),
      underflowSamples: Math.floor(sampleRate / 5),
    },
  });
  gain = ctx.createGain();
  applyGain();
  node.connect(gain);
  gain.connect(ctx.destination);

  node.port.onmessage = (ev) => {
    if (retired) return;
    const msg = ev.data;
    if (!msg) return;
    if (msg.generation !== generation) return;
    if (msg.type === 'cleared') {
      if (!pendingClear) return;
      try {
        for (const operation of pendingOperations) {
          if (operation.type === 'push') pushToWorklet(operation.pcm);
          else node.port.postMessage({ type: 'end', generation });
        }
      } catch (error) {
        fail(error);
        return;
      }
      pendingOperations = [];
      pendingBytes = 0;
      settleClear(null);
      applyGain();
    } else if (msg.type === 'underflow') {
      underflowCount++;
    } else if (msg.type === 'drained') {
      playing = false;
      resolveIdleWaiters();
    }
  };

  if (pendingClear) postClear();

  return owner;
  } catch (error) {
    await owner.destroy();
    throw error;
  }
}
