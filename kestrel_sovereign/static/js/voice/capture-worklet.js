/**
 * capture-worklet.js — AudioWorkletProcessor for mic capture.
 *
 * Runs on the audio thread. Each process() call receives 128 samples of
 * Float32 input at the AudioContext's native rate (usually 48000 on a Mac,
 * 44100 on many phones). We downsample to a target rate (24000 by default,
 * matching OpenAI Realtime's PCM16 spec), convert to Int16, and post bytes
 * back to the main thread via port.postMessage.
 *
 * Also tracks peak amplitude per frame for the level meter.
 *
 * Messaging contract:
 *   main → worklet:  { type: 'configure', targetSampleRate }
 *   main → worklet:  { type: 'pause' }  |  { type: 'resume' }
 *   worklet → main:  { type: 'chunk', pcm: ArrayBuffer, level: Number }
 *
 * `level` is the peak sample magnitude in this frame, normalized to 0..1.
 */

class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Configured on first 'configure' message.
    this._inputRate = sampleRate; // globally provided by AudioWorkletGlobalScope
    this._targetRate = 24000;
    this._ratio = this._inputRate / this._targetRate;
    this._paused = false;

    // Accumulator for fractional-sample resampling: keeps the fractional
    // offset across frames so we don't drift over time.
    this._resampleFrac = 0;

    // Output buffer size in resampled samples. 480 samples @ 24kHz = 20ms
    // which is a natural frame size for speech codecs and keeps round-trip
    // latency low.
    this._outFrame = 480;
    this._outBuffer = new Int16Array(this._outFrame);
    this._outWrite = 0;

    this.port.onmessage = (ev) => {
      const msg = ev.data;
      if (!msg) return;
      if (msg.type === 'configure') {
        this._targetRate = msg.targetSampleRate || 24000;
        this._ratio = this._inputRate / this._targetRate;
      } else if (msg.type === 'pause') {
        this._paused = true;
      } else if (msg.type === 'resume') {
        this._paused = false;
      }
    };
  }

  process(inputs /*, outputs, parameters */) {
    if (this._paused) return true;
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channel = input[0];
    if (!channel || channel.length === 0) return true;

    let peak = 0;
    let srcIdx = this._resampleFrac;
    // Linear resample from input rate to target rate. For speech, simple
    // interpolation is audibly transparent above ~16kHz target; at 24kHz
    // we're well clear of human speech content.
    while (srcIdx < channel.length) {
      const i = Math.floor(srcIdx);
      const f = srcIdx - i;
      const s0 = channel[i];
      const s1 = i + 1 < channel.length ? channel[i + 1] : s0;
      const sample = s0 + (s1 - s0) * f;

      const mag = Math.abs(sample);
      if (mag > peak) peak = mag;

      // Convert Float32 [-1, 1] → Int16 [-32768, 32767].
      let v = Math.max(-1, Math.min(1, sample));
      v = v < 0 ? Math.round(v * 0x8000) : Math.round(v * 0x7fff);
      this._outBuffer[this._outWrite++] = v;

      if (this._outWrite === this._outFrame) {
        // Ship a complete frame.
        const out = new Int16Array(this._outBuffer); // copy
        this.port.postMessage(
          { type: 'chunk', pcm: out.buffer, level: peak },
          [out.buffer],
        );
        this._outWrite = 0;
        peak = 0;
      }

      srcIdx += this._ratio;
    }

    this._resampleFrac = srcIdx - channel.length;
    return true;
  }
}

registerProcessor('kestrel-capture', CaptureProcessor);
