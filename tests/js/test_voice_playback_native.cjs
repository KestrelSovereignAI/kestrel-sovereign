/** Real browser AudioWorklet queue fencing, loopback only; no provider calls. */
const { test } = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');
const fs = require('node:fs/promises');
const path = require('node:path');
const { chromium } = require('playwright');
const staticRoot = process.env.KESTREL_TEST_STATIC_ROOT
  ? path.resolve(process.env.KESTREL_TEST_STATIC_ROOT)
  : path.resolve(__dirname, '../../kestrel_sovereign/static');
let origin, browser, context, page;
const server = http.createServer(async (req, res) => {
  try {
    if (req.url === '/') {
      res.setHeader('Content-Type', 'text/html');
      return res.end('<!doctype html><title>Native playback epoch proof</title>');
    }
    const pathname = new URL(req.url, origin).pathname.replace(/^\/static\//, '/');
    const target = path.resolve(staticRoot, '.' + pathname);
    if (!target.startsWith(staticRoot + path.sep) || !target.endsWith('.js')) {
      return res.writeHead(404).end();
    }
    res.setHeader('Content-Type', 'text/javascript');
    res.end(await fs.readFile(target));
  } catch (error) { res.writeHead(500).end(String(error)); }
});
test('actual playback gate excludes old PCM and renders fresh PCM after clear ACK', { timeout: 30000 }, async () => {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  origin = 'http://127.0.0.1:' + server.address().port;
  try {
    browser = await chromium.launch({headless:true, args:['--autoplay-policy=no-user-gesture-required']});
    context = await browser.newContext();
    page = await context.newPage();
    await page.route('**/*', route => new URL(route.request().url()).origin === origin
      ? route.continue() : route.abort());
    await page.goto(origin);
    await page.evaluate(async () => {
      const nativeNode = window.AudioWorkletNode;
      window.nativeClearMessages = [];
      window.AudioWorkletNode = class extends nativeNode {
        constructor(...args) {
          super(...args);
          if (args[1] === 'kestrel-playback') {
            window.nativePlaybackNode = this;
            this.port.addEventListener('message', event => {
              if (event.data?.type === 'cleared') window.nativeClearMessages.push(event.data);
            });
            this.port.start();
          }
        }
      };
      const nativeGain = AudioContext.prototype.createGain;
      AudioContext.prototype.createGain = function() {
        const gain = nativeGain.call(this);
        if (!window.nativePlaybackGain) {
          window.nativePlaybackGain = gain;
          window.nativeProbe = this.createAnalyser();
          window.nativeProbe.fftSize = 256;
          const silent = nativeGain.call(this);
          silent.gain.value = 0;
          gain.connect(window.nativeProbe);
          window.nativeProbe.connect(silent);
          silent.connect(this.destination);
        }
        return gain;
      };
      const {createVoicePlayback} = await import('/js/voice/playback.js');
      window.pb = await createVoicePlayback({sampleRate:24000, preRollMs:0});
      const old = new Int16Array(24000).fill(19660);
      window.pb.enqueue(new Uint8Array(old.buffer));
      window.sampleProbe = () => {
        const samples = new Float32Array(window.nativeProbe.fftSize);
        window.nativeProbe.getFloatTimeDomainData(samples);
        return samples.reduce((sum, value) => sum + value, 0) / samples.length;
      };
    });
    await page.waitForFunction(() => window.sampleProbe() > .4, null, {timeout:5000});
    const evidence = await page.evaluate(async () => {
      const clear = window.pb.flush();
      window.pb.setMuted(false);
      const immediatelyGated = window.nativePlaybackGain.gain.value === 0;
      const receipt = await clear;
      const ack = window.nativeClearMessages.find(message => message.generation === receipt?.generation);
      if (!ack) return {immediatelyGated, acked:false};
      const generation = receipt.generation;
      // Real messages enter the real worklet. The stale producer is deliberately
      // retained; queue fencing, not test-side suppression, must exclude it.
      const fresh = new Int16Array(12000).fill(-13107);
      window.pb.enqueue(new Uint8Array(fresh.buffer));
      const old = new Int16Array(12000).fill(19660);
      window.nativePlaybackNode.port.postMessage(
        {type:'push', generation:generation - 1, pcm:old.buffer}, [old.buffer]);
      window.pb.endOfStream();
      let minimum = 1, maximum = -1;
      const start = performance.now();
      while (performance.now() - start < 1200) {
        await new Promise(resolve => requestAnimationFrame(resolve));
        // Analyser history from before the synchronous gate is not new output.
        if (performance.now() - start < 100) continue;
        const value = window.sampleProbe();
        minimum = Math.min(minimum, value);
        maximum = Math.max(maximum, value);
      }
      return {immediatelyGated, acked:true, generation, minimum, maximum,
        nativeContextState:window.nativeProbe.context.state};
    });
    console.log(JSON.stringify(evidence));
    assert.equal(evidence.immediatelyGated, true, 'flush or user unmute left the real output gain open before worklet clear');
    assert.equal(evidence.acked, true, 'flush did not join an actual worklet clear acknowledgement');
    assert.ok(evidence.minimum < -.2, 'fresh PCM did not render through the real post-gain audio probe');
    assert.ok(evidence.maximum < .03, 'old-generation PCM rendered after fresh output resumed');
    assert.equal(evidence.nativeContextState, 'running');
  } finally {
    if (page) await page.evaluate(async () => { await window.pb?.destroy(); }).catch(() => {});
    await context?.close();
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
  }
});
