/**
 * playback-worklet.js — Jitter-buffered PCM16 playback processor.
 *
 * Receives Int16 PCM chunks from the main thread, maintains a ring buffer,
 * and plays them out through the audio graph. On underflow (buffer empty)
 * emits silence rather than glitching. On 'flush' drops everything queued
 * — used for barge-in when the user starts speaking mid-response.
 *
 * Messaging contract:
 *   main → worklet:  { type: 'push', pcm: ArrayBuffer }  // Int16 samples
 *   main → worklet:  { type: 'flush' }
 *   worklet → main:  { type: 'playhead', samples: Number, playing: Boolean }
 *   worklet → main:  { type: 'underflow' }   // buffer went empty
 *
 * Ring-buffer cap: ~2 seconds at 24kHz = 48000 samples. Overflow drops
 * oldest (very unusual; indicates a bug upstream).
 */

const RING_CAPACITY = 48000 * 2; // ~4 seconds headroom, tight enough to trip bugs fast
const UNDERFLOW_REPORT_INTERVAL = 4800; // ~100ms between underflow reports @ 24kHz

class PlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._ring = new Float32Array(RING_CAPACITY);
    this._read = 0;
    this._write = 0;
    this._fill = 0;
    this._playing = false;
    this._samplesElapsed = 0;
    this._sinceUnderflowReport = 0;

    this.port.onmessage = (ev) => {
      const msg = ev.data;
      if (!msg) return;
      if (msg.type === 'push') {
        this._enqueue(msg.pcm);
      } else if (msg.type === 'flush') {
        this._read = 0;
        this._write = 0;
        this._fill = 0;
        this._playing = false;
      }
    };
  }

  _enqueue(arrayBuffer) {
    const int16 = new Int16Array(arrayBuffer);
    // Convert Int16 → Float32 [-1, 1] at the ring-buffer edge.
    for (let i = 0; i < int16.length; i++) {
      const f = int16[i] < 0 ? int16[i] / 0x8000 : int16[i] / 0x7fff;
      if (this._fill >= RING_CAPACITY) {
        // Overflow — drop oldest sample to make room. Unusual; logs nothing
        // from the worklet thread, but main-side getLevel can detect.
        this._read = (this._read + 1) % RING_CAPACITY;
        this._fill--;
      }
      this._ring[this._write] = f;
      this._write = (this._write + 1) % RING_CAPACITY;
      this._fill++;
    }
    this._playing = this._fill > 0;
  }

  process(inputs, outputs /* , parameters */) {
    const output = outputs[0];
    if (!output || output.length === 0) return true;
    const channel = output[0];

    for (let i = 0; i < channel.length; i++) {
      if (this._fill > 0) {
        channel[i] = this._ring[this._read];
        this._read = (this._read + 1) % RING_CAPACITY;
        this._fill--;
        this._samplesElapsed++;
      } else {
        channel[i] = 0;
        if (this._playing) {
          this._sinceUnderflowReport += 1;
          if (this._sinceUnderflowReport >= UNDERFLOW_REPORT_INTERVAL) {
            this.port.postMessage({ type: 'underflow' });
            this._sinceUnderflowReport = 0;
          }
          // Stop "playing" state once drained a whole underflow window —
          // helps the UI stop the speaking indicator cleanly.
          if (this._sinceUnderflowReport === 0) {
            this._playing = false;
          }
        }
      }
    }

    // Copy to other channels (we produce mono; AudioContext may be stereo).
    for (let c = 1; c < output.length; c++) {
      output[c].set(channel);
    }

    return true;
  }
}

registerProcessor('kestrel-playback', PlaybackProcessor);
