/**
 * playback-worklet.js — Lossless jitter-buffered PCM16 playback processor.
 *
 * Receives Int16 PCM chunks from the main thread and plays them through the
 * audio graph in arrival order. Provider delivery is allowed to run ahead of
 * wall-clock playback: queued chunks are retained losslessly until rendered.
 * On a temporary underflow, the processor emits silence and re-establishes
 * pre-roll before resuming. On 'flush' it drops everything queued — used for
 * barge-in when the user starts speaking mid-response.
 *
 * Messaging contract:
 *   main → worklet:  { type: 'push', pcm: ArrayBuffer }  // Int16 samples
 *   main → worklet:  { type: 'end' }                     // response complete
 *   main → worklet:  { type: 'flush' }
 *   worklet → main:  { type: 'underflow' }               // network gap
 *   worklet → main:  { type: 'drained' }                 // ended + fully played
 *
 * PCM remains Int16 while queued and is converted to Float32 only as the
 * browser renders it. This keeps burst buffering compact without imposing a
 * fixed-duration cap that would discard audio from fast realtime providers.
 */

const DEFAULT_UNDERFLOW_SAMPLES = 4800; // ~200ms @ 24kHz
const COMPACT_HEAD_THRESHOLD = 64;

class PlaybackProcessor extends AudioWorkletProcessor {
  constructor(options = {}) {
    super();
    const processorOptions = options.processorOptions ?? {};
    this._preRollSamples = Math.max(0, processorOptions.preRollSamples ?? 0);
    this._underflowSamples = Math.max(
      1,
      processorOptions.underflowSamples ?? DEFAULT_UNDERFLOW_SAMPLES,
    );

    this._chunks = [];
    this._chunkHead = 0;
    this._chunkOffset = 0;
    this._queuedSamples = 0;
    this._started = false;
    this._streamEnded = false;
    this._drainedReported = true;
    this._emptySamples = 0;

    this.port.onmessage = (ev) => {
      const msg = ev.data;
      if (!msg) return;
      if (msg.type === 'push') {
        this._enqueue(msg.pcm);
      } else if (msg.type === 'end') {
        this._endStream();
      } else if (msg.type === 'flush') {
        this._flush();
      }
    };
  }

  _enqueue(arrayBuffer) {
    const int16 = new Int16Array(arrayBuffer);
    if (int16.length === 0) return;

    this._chunks.push(int16);
    this._queuedSamples += int16.length;
    this._streamEnded = false;
    this._drainedReported = false;
    this._emptySamples = 0;
    if (!this._started && this._queuedSamples >= this._preRollSamples) {
      this._started = true;
    }
  }

  _endStream() {
    this._streamEnded = true;
    // Release a short utterance that did not reach the normal pre-roll.
    if (this._queuedSamples > 0) this._started = true;
    this._reportDrainedIfReady();
  }

  _flush() {
    this._chunks = [];
    this._chunkHead = 0;
    this._chunkOffset = 0;
    this._queuedSamples = 0;
    this._started = false;
    this._streamEnded = false;
    this._drainedReported = true;
    this._emptySamples = 0;
  }

  _dequeue() {
    const chunk = this._chunks[this._chunkHead];
    const sample = chunk[this._chunkOffset];
    this._chunkOffset += 1;
    this._queuedSamples -= 1;

    if (this._chunkOffset >= chunk.length) {
      this._chunkHead += 1;
      this._chunkOffset = 0;
      if (
        this._chunkHead >= COMPACT_HEAD_THRESHOLD
        && this._chunkHead * 2 >= this._chunks.length
      ) {
        this._chunks = this._chunks.slice(this._chunkHead);
        this._chunkHead = 0;
      }
    }

    return sample < 0 ? sample / 0x8000 : sample / 0x7fff;
  }

  _reportDrainedIfReady() {
    if (
      this._streamEnded
      && this._queuedSamples === 0
      && !this._drainedReported
    ) {
      this._drainedReported = true;
      this._started = false;
      this._chunks = [];
      this._chunkHead = 0;
      this._chunkOffset = 0;
      this.port.postMessage({ type: 'drained' });
    }
  }

  process(inputs, outputs /* , parameters */) {
    const output = outputs[0];
    if (!output || output.length === 0) return true;
    const channel = output[0];

    for (let index = 0; index < channel.length; index++) {
      if (this._started && this._queuedSamples > 0) {
        channel[index] = this._dequeue();
        this._emptySamples = 0;
        this._reportDrainedIfReady();
      } else {
        channel[index] = 0;
        if (this._started && !this._streamEnded) {
          this._emptySamples += 1;
          if (this._emptySamples >= this._underflowSamples) {
            this._emptySamples = 0;
            this._started = false;
            this.port.postMessage({ type: 'underflow' });
          }
        }
      }
    }

    // Copy to other channels (we produce mono; AudioContext may be stereo).
    for (let channelIndex = 1; channelIndex < output.length; channelIndex++) {
      output[channelIndex].set(channel);
    }

    return true;
  }
}

registerProcessor('kestrel-playback', PlaybackProcessor);
