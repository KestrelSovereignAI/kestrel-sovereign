/** Offline browser proofs using native WebRTC, microphone and HTTP resources. */
const { test } = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');
const fs = require('node:fs/promises');
const path = require('node:path');
const { chromium } = require('playwright');

async function nativeVoice(body, transcript = '{}', { autoStart = true, holdMint = false, failWorklet = '' } = {}) {
  const staticRoot = process.env.KESTREL_TEST_STATIC_ROOT
    ? path.resolve(process.env.KESTREL_TEST_STATIC_ROOT)
    : path.resolve(__dirname, '../../kestrel_sovereign/static');
  let page;
  let batchResponse;
  let batchEntered;
  let mintResponse;
  let mintEntered;
  let sdpRequests = 0;
  const enteredMint = new Promise(resolve => { mintEntered = resolve; });
  const entered = new Promise(resolve => { batchEntered = resolve; });
  const server = http.createServer(async (req, res) => {
    try {
      const chunks = [];
      for await (const chunk of req) chunks.push(chunk);
      if (req.url === '/voice/realtime/session') {
        res.setHeader('Content-Type', 'application/json');
        const payload = JSON.stringify({ session_id: 'native-offline', transport: 'webrtc',
          client_secret: { value: 'offline-peer-only' }, endpoint: `${origin}/sdp` });
        mintResponse = { res, payload };
        mintEntered();
        if (!holdMint) res.end(payload);
      } else if (req.url === '/sdp') {
        sdpRequests++;
        const sdp = await page.evaluate(async offer => {
          window.nativePeer = new RTCPeerConnection();
          window.nativeMessages = [];
          window.nativePeer.ondatachannel = event => {
            window.nativeChannel = event.channel;
            event.channel.onmessage = message => window.nativeMessages.push(JSON.parse(message.data));
          };
          await window.nativePeer.setRemoteDescription({ type: 'offer', sdp: offer });
          await window.nativePeer.setLocalDescription(await window.nativePeer.createAnswer());
          await new Promise(resolve => {
            if (window.nativePeer.iceGatheringState === 'complete') return resolve();
            window.nativePeer.onicegatheringstatechange = () => {
              if (window.nativePeer.iceGatheringState === 'complete') resolve();
            };
          });
          // Negotiate a real bounded SCTP message size; Chromium otherwise
          // advertises an unlimited peer receive size on loopback.
          return window.nativePeer.localDescription.sdp.replace(
            /a=max-message-size:\d+/g, 'a=max-message-size:65536');
        }, Buffer.concat(chunks).toString());
        res.setHeader('Content-Type', 'application/sdp');
        res.end(sdp);
      } else if (req.url.startsWith('/voice/realtime/transcript/')) {
        res.setHeader('Content-Type', 'application/json');
        res.end(transcript);
      } else if (req.url.includes('/voice/realtime/tools/')) {
        batchResponse = res;
        batchEntered();
      } else if (req.url === '/') {
        res.setHeader('Content-Type', 'text/html');
        res.end('<!doctype html><html><head><script src="/js/icons.js"></script>'
          + ['parse', 'highlight', 'mermaid', 'index'].map(name =>
            `<script src="/shared/markdown/${name}.js"></script>`).join('')
          + '</head><body><div id="chat-container"></div></body></html>');
      } else if (failWorklet && req.url.endsWith(`/${failWorklet}-worklet.js`)) {
        res.writeHead(503).end('offline worklet refusal');
      } else {
        const relative = decodeURIComponent(req.url.split('?')[0]);
        const target = path.resolve(staticRoot, `.${relative.replace(/^\/static\//, '/')}`);
        if (!target.startsWith(staticRoot + path.sep) || !/\.m?js$/.test(target)) {
          res.writeHead(404).end();
          return;
        }
        let source = await fs.readFile(target, 'utf8');
        // Expose existing module-private UI functions, without replacing any
        // production branch, rendering, HTTP, close or transport operation.
        if (relative === '/js/voice/ui.js') source += '\nexport { createSession, handleToolBatch, surfaceFatalError };\n';
        res.setHeader('Content-Type', 'text/javascript');
        res.end(source);
      }
    } catch (error) {
      console.error('offline fixture resource failed', req.url, String(error));
      res.writeHead(500).end(String(error));
    }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const origin = `http://127.0.0.1:${server.address().port}`;
  const browser = await chromium.launch({ headless: true, args: [
    '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
    `--use-file-for-fake-audio-capture=${path.resolve(__dirname, '../fixtures/tone_440hz.wav')}`,
  ] });
  const context = await browser.newContext({ permissions: ['microphone'] });
  const pageErrors = [];
  try {
    page = await context.newPage();
    page.on('pageerror', error => pageErrors.push(error.message));
    page.on('response', response => {
      if (response.status() >= 400) console.error('offline HTTP error', response.status(), response.url());
    });
    await page.route('**/*', route => new URL(route.request().url()).origin === origin
      ? route.continue() : route.abort());
    await page.goto(origin);
    await page.evaluate(async autoStart => {
      const { createRealtimeClient } = await import('/js/voice/realtime.js');
      window.nativeEvents = [];
      window.client = await createRealtimeClient({ onEvent: event => window.nativeEvents.push(event) });
      if (autoStart) await window.client.start();
    }, autoStart);
    if (autoStart) await page.waitForFunction(() => window.nativeChannel?.readyState === 'open');
    await body({ page, entered, enteredMint,
      releaseMint: () => mintResponse.res.end(mintResponse.payload),
      sdpRequests: () => sdpRequests,
      releaseBatch: (ack = {}) => batchResponse.end(JSON.stringify(ack)) });
    assert.deepEqual(pageErrors, [], 'production console raised an unhandled browser error');
  } finally {
    if (mintResponse && !mintResponse.res.writableEnded) mintResponse.res.end(mintResponse.payload);
    if (batchResponse && !batchResponse.writableEnded) batchResponse.end('{}');
    if (page) await page.evaluate(async () => {
      await window.client?.close();
      window.nativePeer?.close();
    }).catch(() => {});
    await context.close();
    await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
}

test('native RTC transport retirement closes original resources without model continuation', { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page }) => {
    // Retire the actual peer, without replacing RTCDataChannel.send or the
    // production transport. The required tool result must not be dropped.
    await page.evaluate(() => window.nativePeer.close());
    await page.waitForFunction(() => window.nativeChannel.readyState === 'closed');
    const result = await page.evaluate(async () => {
      let error;
      try {
        await window.client.commitToolResults([{ call_id: 'original', result: { text: 'native result' } }]);
      } catch (caught) { error = { name: caught.name, message: caught.message }; }
      return { error, tracks: window.client.micStream.getTracks().map(track => track.readyState),
        continuation: window.nativeMessages.some(message => message.type === 'response.create') };
    });
    assert.ok(result.error, 'native RTC refusal was swallowed');
    assert.match(result.error.message, /transport|open|send/i);
    assert.equal(result.continuation, false);
    assert.ok(result.tracks.every(state => state === 'ended'));
  });
});

for (const acknowledgement of ['{}', 'not-json']) {
  test(`native transcript ${acknowledgement} acknowledgement closes original session`, { timeout: 30000 }, async () => {
    await nativeVoice(async ({ page }) => {
      const result = await page.evaluate(async () => {
        window.client.sendText('offline native transcript');
        await window.client.whenPersisted();
        return window.client.micStream.getTracks().map(track => track.readyState);
      });
      assert.ok(result.every(state => state === 'ended'));
    }, acknowledgement);
  });
}

test('explicit persisted:false is an acknowledged nonpersistent transcript', { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page }) => {
    const result = await page.evaluate(async () => {
      window.client.sendText('acknowledged ephemeral transcript');
      await window.client.whenPersisted();
      return window.client.micStream.getTracks().map(track => track.readyState);
    });
    assert.ok(result.every(state => state === 'live'));
  }, '{"persisted":false}');
});

for (const acknowledged of [false, true]) {
test(`${acknowledged ? 'valid' : 'uncertain'} tool acknowledgement closes captured native client after UI pointer clears`, { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page, entered, releaseBatch }) => {
    await page.evaluate(async () => {
      window.voiceUI = await import('/js/voice/ui.js');
      window.uiSession = window.voiceUI.createSession(null);
      window.uiSession.client = window.client;
      window.uiTask = window.voiceUI.handleToolBatch(window.uiSession, { calls: [
        { call_id: 'original', name: 'native-no-execution', arguments: {} },
      ] });
    });
    await entered;
    await page.evaluate(() => window.voiceUI.surfaceFatalError(new Error('RTC transient disconnect'), window.uiSession));
    assert.equal(await page.evaluate(() => window.uiSession.client), null);
    releaseBatch(acknowledged ? { results: [{ call_id: 'original', result: { text: 'native receipt' } }] } : {});
    const result = await page.evaluate(async () => {
      await window.uiTask;
      return { tracks: window.client.micStream.getTracks().map(track => track.readyState),
        continuation: window.nativeMessages.some(message => message.type === 'response.create') };
    });
    assert.ok(result.tracks.every(state => state === 'ended'));
    assert.equal(result.continuation, false);
  });
});
}

test('ordinary native RTC channel closure retires microphone without UI or tool calls', { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page }) => {
    await page.evaluate(() => window.nativeChannel.close());
    await page.waitForFunction(() => window.client.micStream.getTracks().every(track => track.readyState === 'ended')
      && window.nativeEvents.some(event => event.kind === 'session_closed'));
    assert.equal(await page.evaluate(() => window.nativeEvents.filter(event => event.kind === 'session_closed').length), 1);
  });
});

test('Stop while actual HTTP mint is pending cannot acquire microphone or SDP', { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page, enteredMint, releaseMint, sdpRequests }) => {
    await page.evaluate(() => {
      window.startup = window.client.start().then(() => null, error => error.message);
    });
    await enteredMint;
    await page.evaluate(() => window.client.close());
    releaseMint();
    assert.ok(await page.evaluate(() => window.startup));
    assert.equal(await page.evaluate(() => window.client.micStream), null);
    assert.equal(sdpRequests(), 0);
  }, '{}', { autoStart: false, holdMint: true });
});

test('Stop retires a late actual microphone acquisition after original close finished', { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page, sdpRequests }) => {
    await page.evaluate(() => {
      const nativeCapture = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
      window.acquiredStreams = [];
      const release = new Promise(resolve => { window.releasePermission = resolve; });
      navigator.mediaDevices.getUserMedia = async (...args) => {
        const stream = await nativeCapture(...args);
        window.acquiredStreams.push(stream);
        await release;
        return stream;
      };
      window.startup = window.client.start().then(() => null, error => error.message);
    });
    await page.waitForFunction(() => window.acquiredStreams.length === 1);
    await page.evaluate(() => window.client.close());
    await page.evaluate(() => window.releasePermission());
    assert.ok(await page.evaluate(() => window.startup));
    assert.ok(await page.evaluate(() => window.acquiredStreams[0].getTracks().every(track => track.readyState === 'ended')));
    assert.equal(sdpRequests(), 0);
  }, '{}', { autoStart: false });
});

for (const factory of ['capture', 'playback']) {
test(`failed actual ${factory} AudioWorklet HTTP initialization retires partial resources`, { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page }) => {
    const state = await page.evaluate(async factory => {
      window.acquiredContexts = [];
      window.acquiredStreams = [];
      const NativeContext = window.AudioContext;
      window.AudioContext = class extends NativeContext {
        constructor(...args) { super(...args); window.acquiredContexts.push(this); }
      };
      const nativeCapture = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
      navigator.mediaDevices.getUserMedia = async (...args) => {
        const stream = await nativeCapture(...args);
        window.acquiredStreams.push(stream);
        return stream;
      };
      const module = await import(`/js/voice/${factory}.js`);
      let failed = false;
      try { await module[factory === 'capture' ? 'createVoiceCapture' : 'createVoicePlayback'](); }
      catch (_) { failed = true; }
      return { failed, contexts: window.acquiredContexts.map(ctx => ctx.state),
        tracks: window.acquiredStreams.flatMap(stream => stream.getTracks().map(track => track.readyState)) };
    }, factory);
    assert.equal(state.failed, true);
    assert.deepEqual(state.contexts, ['closed']);
    assert.ok(state.tracks.every(state => state === 'ended'));
    if (factory === 'capture') assert.ok(state.tracks.length);
  }, '{}', { autoStart: false, failWorklet: factory });
});
}
