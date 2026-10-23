/** Offline browser proofs using native WebRTC, microphone and HTTP resources. */
const { test } = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');
const fs = require('node:fs/promises');
const path = require('node:path');
const { chromium } = require('playwright');
// Use the real WebSocket server already bundled with the browser test runtime.
const { wsServer: WebSocketServer } = require(path.join(
  path.dirname(require.resolve('playwright-core/package.json')), 'lib/utilsBundle.js'));

async function nativeVoice(body, transcript = '{}', { autoStart = true, holdMint = false, failWorklet = '', holdWorklet = '', transport = 'webrtc', declineMint = false, responseStartTimeoutMs = 30000, dropPlaybackClearAck = false, framed = false, holdSessionAck = false, holdInputClear = false, ackIdentity = true, socketIdentity = true } = {}) {
  const staticRoot = process.env.KESTREL_TEST_STATIC_ROOT
    ? path.resolve(process.env.KESTREL_TEST_STATIC_ROOT)
    : path.resolve(__dirname, '../../kestrel_sovereign/static');
  let page;
  let batchResponse;
  let batchEntered;
  let mintResponse;
  let mintEntered;
  let sdpRequests = 0;
  let toolPosts = 0;
  let pipelineFrames = 0;
  let mintRequests = 0;
  const realtimeMessages = [];
  const realtimePeers = [];
  const framedConfig = { type: 'realtime', output_modalities: ['audio'], audio: {
    input: { format: { type: 'audio/pcm', rate: 24000 }, turn_detection: {
      type: 'server_vad', create_response: false, interrupt_response: false,
    } }, output: { format: { type: 'audio/pcm', rate: 24000 }, voice: 'offline' },
  } };
  let workletResponse;
  let workletEntered;
  const enteredWorklet = new Promise(resolve => { workletEntered = resolve; });
  const enteredMint = new Promise(resolve => { mintEntered = resolve; });
  const entered = new Promise(resolve => { batchEntered = resolve; });
  const server = http.createServer(async (req, res) => {
    try {
      const chunks = [];
      for await (const chunk of req) chunks.push(chunk);
      if (req.url === '/voice/realtime/session') {
        mintRequests++;
        res.setHeader('Content-Type', 'application/json');
        if (declineMint) {
          res.writeHead(409).end(JSON.stringify({ path: 'pipeline', fallback_tts: 'offline',
            fallback_stt: 'offline', reason: 'offline pipeline route selected' }));
          return;
        }
        const strict = JSON.parse(Buffer.concat(chunks).toString() || '{}').output_profile;
        const actualTransport = strict ? 'websocket' : transport;
        const payload = JSON.stringify({ session_id: 'native-offline', transport: actualTransport,
          ...(strict ? { output_profile: strict, vendor: 'openai', model: 'offline-model',
            protocol: 'openai-realtime-ga-framed-v1', session_config: framedConfig } : {}),
          client_secret: { value: 'offline-peer-only' }, endpoint: actualTransport === 'websocket'
            ? `${origin.replace('http:', 'ws:')}/realtime` : `${origin}/sdp` });
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
        toolPosts++;
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
        if (dropPlaybackClearAck && relative.endsWith('/playback-worklet.js')) {
          const acknowledgement = "this.port.postMessage({ type: 'cleared', generation: this._generation });";
          assert.ok(source.includes(acknowledgement), 'clear-ACK loss fixture no longer matches actual worker');
          source = source.replace(acknowledgement, 'void 0; // deliberate offline clear-ACK loss');
        }
        if (holdWorklet && relative.endsWith(`/${holdWorklet}-worklet.js`)) {
          workletResponse = { res, source };
          workletEntered();
          return;
        }
        // Expose existing module-private UI functions, without replacing any
        // production branch, rendering, HTTP, close or transport operation.
        if (relative === '/js/voice/ui.js') source += '\nexport { createSession, sessionForAgent, startSession, stopSession, handleToolBatch, surfaceFatalError };\n';
        res.setHeader('Content-Type', 'text/javascript');
        res.end(source);
      }
    } catch (error) {
      console.error('offline fixture resource failed', req.url, String(error));
      res.writeHead(500).end(String(error));
    }
  });
  const wsServer = new WebSocketServer({ noServer: true });
  server.on('upgrade', (request, socket, head) => {
    wsServer.handleUpgrade(request, socket, head, connection => {
      wsServer.emit('connection', connection, request);
    });
  });
  wsServer.on('connection', connection => {
    realtimePeers.push(connection);
    const effectiveConfig = JSON.parse(JSON.stringify({ model: 'offline-model', ...framedConfig }));
    connection.send(JSON.stringify({ type: 'session.created', session: { model: 'offline-model',
      ...(socketIdentity ? { id: 'actual-socket-id' } : {}) } }));
    connection.send(JSON.stringify({ type: 'status', state: 'listening' }));
    connection.on('message', (bytes, binary) => {
      if (binary && bytes[0] === 1) pipelineFrames++;
      if (!binary) {
        const message = JSON.parse(bytes.toString());
        realtimeMessages.push(message);
        if (message.type === 'session.update') {
          // Actual GA SessionUpdateEventParam requires session.type. A socket
          // fixture must not ACK a malformed update that the provider rejects.
          if (connection.protocol === 'realtime' && (message.session?.type !== 'realtime' || Object.hasOwn(message.session, 'model'))) {
            connection.send(JSON.stringify({ type: 'error', error: {
              code: 'invalid_request_error', message: 'GA session update requires type and cannot change model',
            } }));
          } else {
            Object.assign(effectiveConfig, message.session);
            if (!holdSessionAck) connection.send(JSON.stringify({ type: 'session.updated',
              session: { ...(ackIdentity ? { id: 'actual-socket-id' } : {}), ...effectiveConfig } }));
          }
        }
        if (message.type === 'input_audio_buffer.clear' && !holdInputClear) connection.send(JSON.stringify({ type: 'input_audio_buffer.cleared' }));
      }
    });
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
    await page.evaluate(async ({ autoStart, responseStartTimeoutMs, framed }) => {
      const { createRealtimeClient } = await import('/js/voice/realtime.js');
      window.nativeEvents = [];
      if (framed) {
        const NativeContext = window.AudioContext, NativeSocket = window.WebSocket, NativeNode = window.AudioWorkletNode;
        window.nativeContexts = []; window.nativeSockets = []; window.nativeCaptureGenerations = []; window.nativeWorkletNodes = [];
        window.AudioContext = class extends NativeContext { constructor(...args) { super(...args); window.nativeContexts.push(this); } };
        window.WebSocket = class extends NativeSocket { constructor(...args) { super(...args); window.nativeSockets.push(this); } };
        window.AudioWorkletNode = class extends NativeNode { constructor(...args) {
          super(...args);
          window.nativeWorkletNodes.push(this);
          if (args[1] === 'kestrel-capture') {
            this.port.addEventListener('message', event => {
              if (event.data?.type === 'chunk') window.nativeCaptureGenerations.push(event.data.generation);
            });
            this.port.start();
          }
        } };
      }
      // These long-standing RTC/xAI compatibility cases explicitly exercise
      // the legacy API. The default Console/new client profile stays strict.
      window.client = await createRealtimeClient({ onEvent: event => window.nativeEvents.push(event), responseStartTimeoutMs,
        outputProfile: framed ? 'response-framed-pcm-v1' : '' });
      if (autoStart) await window.client.start();
    }, { autoStart, responseStartTimeoutMs, framed });
    if (autoStart && transport === 'webrtc' && !framed) await page.waitForFunction(() => window.nativeChannel?.readyState === 'open');
    await body({ page, entered, enteredMint, enteredWorklet,
      releaseWorklet: () => {
        workletResponse.res.setHeader('Content-Type', 'text/javascript');
        workletResponse.res.end(workletResponse.source);
      },
      toolPosts: () => toolPosts,
      pipelineFrames: () => pipelineFrames,
      mintRequests: () => mintRequests,
      realtimeMessages: () => realtimeMessages,
      sessionAck: config => realtimePeers[0].send(JSON.stringify({ type: 'session.updated',
        session: config || { id: 'actual-socket-id', model: 'offline-model', ...framedConfig } })),
      sendRealtime: payload => realtimePeers[0].send(JSON.stringify(payload)),
      sendRealtimeRaw: bytes => realtimePeers[0].send(bytes),
      releaseMint: () => mintResponse.res.end(mintResponse.payload),
      sdpRequests: () => sdpRequests,
      releaseBatch: (ack = {}) => batchResponse.end(JSON.stringify(ack)) });
    assert.deepEqual(pageErrors, [], 'production console raised an unhandled browser error');
  } finally {
    if (workletResponse && !workletResponse.res.writableEnded) {
      workletResponse.res.setHeader('Content-Type', 'text/javascript');
      workletResponse.res.end(workletResponse.source);
    }
    if (mintResponse && !mintResponse.res.writableEnded) mintResponse.res.end(mintResponse.payload);
    if (batchResponse && !batchResponse.writableEnded) batchResponse.end('{}');
    if (page) await page.evaluate(async () => {
      await window.client?.close();
      window.nativePeer?.close();
    }).catch(() => {});
    await context.close();
    await browser.close();
    for (const connection of wsServer.clients) connection.terminate();
    await new Promise(resolve => wsServer.close(resolve));
    await new Promise(resolve => server.close(resolve));
  }
}

test('framed native startup withholds actual microphone PCM until effective session ACK', { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page, realtimeMessages, sessionAck }) => {
    await page.evaluate(() => { window.startup = window.client.start(); });
    await page.waitForFunction(() => window.client.micStream?.getTracks().every(track => track.readyState === 'live'));
    await page.waitForTimeout(350);
    assert.equal(realtimeMessages().filter(message => message.type === 'input_audio_buffer.append').length, 0);
    assert.equal(await page.evaluate(() => window.nativeEvents.some(event => event.kind === 'session_ready')), false);
    sessionAck();
    await page.evaluate(() => window.startup);
    await page.waitForTimeout(150);
    const frames = realtimeMessages().filter(message => message.type === 'input_audio_buffer.append');
    assert.ok(frames.length > 0);
    assert.ok(frames.some(frame => [...Buffer.from(frame.audio, 'base64')].some(byte => byte !== 0)), 'actual original microphone did not deliver fresh nonzero PCM');
    assert.equal(await page.evaluate(() => window.nativeEvents.find(event => event.kind === 'session_ready').session_id), 'actual-socket-id');
  }, '{"persisted":false}', { autoStart: false, framed: true, holdSessionAck: true });
});

for (const failure of ['absent', 'incompatible']) {
test(`framed native ${failure} config ACK automatically retires original resources`, { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page, sessionAck, realtimeMessages }) => {
    await page.evaluate(() => { window.startup = window.client.start().then(() => null, error => error.message); });
    await page.waitForFunction(() => window.client.micStream?.getTracks().some(track => track.readyState === 'live'));
    for (let step = 0; step < 100 && !realtimeMessages().some(message => message.type === 'session.update'); step++) await page.waitForTimeout(10);
    assert.ok(realtimeMessages().some(message => message.type === 'session.update'));
    if (failure === 'incompatible') sessionAck({ type: 'realtime', model: 'wrong-model' });
    assert.ok(await page.evaluate(() => window.startup));
    assert.ok(await page.evaluate(() => window.client.micStream.getTracks().every(track => track.readyState === 'ended')));
    assert.ok(await page.evaluate(() => window.nativeContexts.length === 2 && window.nativeContexts.every(context => context.state === 'closed')));
    await page.waitForFunction(() => window.nativeSockets.length === 1 && window.nativeSockets[0].readyState === WebSocket.CLOSED);
    assert.equal(realtimeMessages().filter(message => message.type === 'input_audio_buffer.append').length, 0);
  }, '{"persisted":false}', { autoStart: false, framed: true, holdSessionAck: true, responseStartTimeoutMs: 700 });
});
}

test('framed native Stop and typed replacement exclude delayed responses and admit fresh original microphone output', { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page, realtimeMessages, sendRealtime }) => {
    await page.evaluate(async () => {
      const nativeGain = AudioContext.prototype.createGain;
      AudioContext.prototype.createGain = function() {
        const gain = nativeGain.call(this);
        window.playbackGain = gain;
        window.probe = this.createAnalyser();
        window.probe.fftSize = 256;
        const sink = nativeGain.call(this); sink.gain.value = 0;
        gain.connect(window.probe); window.probe.connect(sink); sink.connect(this.destination);
        return gain;
      };
      window.originalMicTracks = [];
      window.startup = window.client.start();
      await window.startup;
      window.originalMicTracks = window.client.micStream.getTracks();
      window.sample = () => { const samples = new Float32Array(window.probe.fftSize);
        window.probe.getFloatTimeDomainData(samples); return samples.reduce((sum, x) => sum + x, 0) / samples.length; };
    });
    const waitRequest = async count => {
      for (let step = 0; step < 100 && realtimeMessages().filter(message => message.type === 'response.create').length < count; step++) await page.waitForTimeout(10);
      const requests = realtimeMessages().filter(message => message.type === 'response.create');
      assert.equal(requests.length, count);
      return requests.at(-1).response.metadata;
    };
    const pcm = value => Buffer.from(new Int16Array(24000).fill(value).buffer).toString('base64');
    const stalePcm = pcm(19660), freshPcm = pcm(-13107);
    const startSpeech = async id => {
      sendRealtime({ type: 'input_audio_buffer.speech_started', item_id: id });
      sendRealtime({ type: 'input_audio_buffer.committed', item_id: id });
    };
    await startSpeech('original-speech');
    const original = await waitRequest(1);
    sendRealtime({ type: 'response.created', response: { id: 'original', metadata: original } });
    sendRealtime({ type: 'response.output_audio.delta', response_id: 'original', delta: stalePcm });
    await page.waitForFunction(() => window.sample() > .4);
    await page.evaluate(() => window.client.cancelResponse());
    await page.waitForFunction(() => window.nativeCaptureGenerations.includes(1), null, { timeout: 3000 });
    // Input-clear ACK intentionally withheld while actual original mic runs.
    const beforeClear = realtimeMessages().filter(message => message.type === 'input_audio_buffer.append').length;
    await page.waitForTimeout(160);
    assert.equal(realtimeMessages().filter(message => message.type === 'input_audio_buffer.append').length, beforeClear);
    sendRealtime({ type: 'response.output_audio.delta', response_id: 'original', delta: stalePcm });
    sendRealtime({ type: 'input_audio_buffer.cleared' });
    await page.waitForTimeout(100);
    assert.ok(realtimeMessages().filter(message => message.type === 'input_audio_buffer.append').length > beforeClear);
    await startSpeech('pre-id-speech');
    const preId = await waitRequest(2);
    assert.equal(await page.evaluate(() => Math.max(...window.nativeCaptureGenerations)), 1, 'ordinary VAD must preserve the current microphone generation');
    await page.evaluate(() => window.client.sendText('new typed turn'));
    await page.waitForFunction(() => window.nativeCaptureGenerations.includes(2), null, { timeout: 3000 });
    sendRealtime({ type: 'response.created', response: { id: 'delayed-before-Stop-id', metadata: preId } });
    sendRealtime({ type: 'response.output_audio.delta', response_id: 'delayed-before-Stop-id', delta: stalePcm });
    sendRealtime({ type: 'input_audio_buffer.cleared' });
    const typed = await waitRequest(3);
    assert.notEqual(typed.kestrel_response_nonce, preId.kestrel_response_nonce);
    sendRealtime({ type: 'response.created', response: { id: 'fresh', metadata: typed } });
    // Missing/foreign IDs and old EOF must not publish or terminate fresh PCM.
    for (const response_id of [undefined, 'foreign', 'original']) {
      sendRealtime({ type: 'response.output_audio.delta', response_id, delta: stalePcm });
      sendRealtime({ type: 'response.output_text.delta', response_id, delta: 'STALE' });
      sendRealtime({ type: 'response.function_call_arguments.done', response_id, call_id: 'stale-call', name: 'never', arguments: '{}' });
      sendRealtime({ type: 'response.done', response_id, response: { id: response_id, status: 'completed' } });
    }
    sendRealtime({ type: 'response.output_audio.delta', response_id: 'fresh', delta: freshPcm });
    await page.waitForFunction(() => window.sample() < -.3);
    const state = await page.evaluate(() => ({ value: window.sample(),
      same: window.client.micStream.getTracks().every((track, index) => track === window.originalMicTracks[index] && track.readyState === 'live'),
      stale: window.nativeEvents.some(event => event.text?.includes('STALE') || event.kind === 'tool_call_batch_requested'),
    }));
    assert.ok(state.same && !state.stale && state.value < -.3, JSON.stringify(state));
    sendRealtime({ type: 'response.function_call_arguments.done', response_id: 'fresh', call_id: 'fresh-call', name: 'allowed', arguments: '{}' });
    sendRealtime({ type: 'response.done', response: { id: 'fresh', status: 'completed' } });
    await page.waitForFunction(() => window.nativeEvents.some(event => event.kind === 'tool_call_batch_requested'));
    await page.evaluate(() => window.client.commitToolResults([{ call_id: 'fresh-call', result: { ok: true } }]));
    const continuation = await waitRequest(4);
    assert.equal(continuation.kestrel_input_generation, typed.kestrel_input_generation);
    assert.notEqual(continuation.kestrel_response_nonce, typed.kestrel_response_nonce);
    sendRealtime({ type: 'response.created', response: { id: 'stale-continuation', metadata: typed } });
    sendRealtime({ type: 'response.output_text.delta', response_id: 'stale-continuation', delta: 'STALE' });
    await page.waitForTimeout(100);
    assert.equal(await page.evaluate(() => window.nativeEvents.some(event => event.text?.includes('STALE'))), false);
    sendRealtime({ type: 'response.created', response: { id: 'continuation', metadata: continuation } });
    sendRealtime({ type: 'response.output_text.delta', response_id: 'continuation', delta: 'FRESH' });
    await page.waitForFunction(() => window.nativeEvents.some(event => event.text === 'FRESH'));
    assert.equal(await page.evaluate(() => window.nativeEvents.some(event => event.text?.includes('STALE'))), false);
  }, '{"persisted":false}', { autoStart: false, framed: true, holdInputClear: true });
});

test('framed native unsupported bootstrap is refused before microphone or socket acquisition', { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page }) => {
    const result = await page.evaluate(async () => {
      const { createRealtimeClient } = await import('/js/voice/realtime.js');
      window.client = await createRealtimeClient({ outputProfile: 'unsupported', onEvent: event => window.nativeEvents.push(event) });
      try { await window.client.start(); return null; } catch (error) { return error.message; }
    });
    assert.match(result, /Required framed realtime profile/);
    assert.equal(await page.evaluate(() => window.client.micStream), null);
    assert.equal(await page.evaluate(() => window.nativeContexts.length + window.nativeSockets.length), 0);
  }, '{"persisted":false}', { autoStart: false, framed: true });
});

for (const failure of ['input-clear', 'response-correlation', 'startup-overflow']) {
test(`framed native ${failure} failure closes original capture, contexts and socket without fixture cleanup`, { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page, realtimeMessages, sendRealtime }) => {
    if (failure === 'startup-overflow') {
      await page.evaluate(() => { window.startup = window.client.start().then(() => null, error => error.message); });
      assert.match(await page.evaluate(() => window.startup), /bounded audio budget/);
      assert.equal(realtimeMessages().filter(message => message.type === 'input_audio_buffer.append').length, 0);
    } else {
      await page.evaluate(() => window.client.start());
      if (failure === 'input-clear') await page.evaluate(() => window.client.cancelResponse());
      else {
        sendRealtime({ type: 'input_audio_buffer.speech_started', item_id: 'fresh' });
        sendRealtime({ type: 'input_audio_buffer.committed', item_id: 'fresh' });
        // Uncorrelated provider output cannot turn request timeout into success.
        sendRealtime({ type: 'response.created', response: { id: 'unproven' } });
      }
      await page.waitForFunction(() => window.nativeEvents.some(event => event.kind === 'error' && event.fatal));
    }
    await page.waitForFunction(() => window.client.micStream?.getTracks().every(track => track.readyState === 'ended')
      && window.nativeContexts.length === 2 && window.nativeContexts.every(context => context.state === 'closed')
      && window.nativeSockets.length === 1 && window.nativeSockets[0].readyState === WebSocket.CLOSED);
    assert.equal(await page.evaluate(() => window.nativeEvents.some(event => event.kind === 'speaking_started')), false);
  }, '{"persisted":false}', { autoStart: false, framed: true,
    holdInputClear: true, holdSessionAck: failure === 'startup-overflow',
    responseStartTimeoutMs: failure === 'startup-overflow' ? 10000 : 700 });
});
}

for (const operation of ['tools', 'instructions']) {
test(`review100 public strict ${operation} update satisfies native GA schema and preserves original call`, { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page, realtimeMessages, sendRealtime }) => {
    await page.evaluate(operation => {
      if (operation === 'tools') window.client.updateTools([{ type: 'function', name: 'offline_no_execution', parameters: { type: 'object', properties: {} } }]);
      else window.client.updateInstructions('new original instructions');
    }, operation);
    for (let step = 0; step < 100 && realtimeMessages().filter(message => message.type === 'session.update').length < 2; step++) await page.waitForTimeout(10);
    const update = realtimeMessages().filter(message => message.type === 'session.update').at(-1);
    assert.equal(update.session.type, 'realtime');
    assert.equal(Object.hasOwn(update.session, 'model'), false);
    await page.waitForTimeout(100);
    assert.ok(await page.evaluate(() => window.client.micStream.getTracks().every(track => track.readyState === 'live')));
    sendRealtime({ type: 'input_audio_buffer.speech_started', item_id: 'after-update' });
    sendRealtime({ type: 'input_audio_buffer.committed', item_id: 'after-update' });
    for (let step = 0; step < 100 && !realtimeMessages().some(message => message.type === 'response.create'); step++) await page.waitForTimeout(10);
    const request = realtimeMessages().find(message => message.type === 'response.create');
    assert.ok(request);
    sendRealtime({ type: 'response.created', response: { id: 'after-update', metadata: request.response.metadata } });
    sendRealtime({ type: 'response.output_text.delta', response_id: 'after-update', delta: 'FRESH_AFTER_UPDATE' });
    await page.waitForFunction(() => window.nativeEvents.some(event => event.text === 'FRESH_AFTER_UPDATE'));
  }, '{"persisted":false}', { framed: true });
});
}

test('review100 Pipeline startup preserves original playback failure through public and late native settlement', { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page, enteredWorklet, releaseWorklet }) => {
    await page.evaluate(async () => {
      const { createPipelineClient } = await import('/js/voice/pipeline.js');
      // Observe the actual native settlement without replacing its promise,
      // resource operation, result or rejection.
      window.originalNativeSettlements = [];
      const originalAllSettled = Promise.allSettled;
      Promise.allSettled = function (operations) {
        const actual = originalAllSettled.call(this, operations);
        actual.then(records => window.originalNativeSettlements.push(records));
        return actual;
      };
      window.client = await createPipelineClient({ onEvent: event => window.nativeEvents.push(event) });
      window.startup = window.client.start().then(() => null, error => { window.publicFailure = error; return error.message; });
    });
    await enteredWorklet;
    await page.evaluate(() => window.client.cancelResponse());
    const error = await page.evaluate(() => window.startup);
    assert.match(error, /worklet did not acknowledge/);
    releaseWorklet();
    const native = await page.evaluate(async () => {
      const settled = await window.client.whenStartupSettled();
      return { same: settled[0].publicReason === window.publicFailure,
        actual: settled[0].reason === window.originalNativeSettlements[0][0].reason,
        closed: window.nativeContexts.every(context => context.state === 'closed'),
        tracks: window.client.micStream.getTracks().every(track => track.readyState === 'ended') };
    });
    assert.ok(native.same && native.actual && native.closed && native.tracks, JSON.stringify(native));
  }, '{"persisted":false}', { autoStart: false, framed: true, holdWorklet: 'playback' });
});

test('review100 malformed native socket JSON before ACK retains original parse failure', { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page, realtimeMessages, sendRealtimeRaw }) => {
    await page.evaluate(() => {
      window.startup = window.client.start().then(() => null, error => {
        window.publicFailure = error; return { name: error.name, message: error.message };
      });
    });
    for (let step = 0; step < 100 && !realtimeMessages().some(message => message.type === 'session.update'); step++) await page.waitForTimeout(10);
    assert.ok(realtimeMessages().some(message => message.type === 'session.update'));
    sendRealtimeRaw('{ malformed native JSON');
    const error = await page.evaluate(() => window.startup);
    assert.equal(error.name, 'SyntaxError');
    assert.match(error.message, /JSON|property|Unexpected/);
    const native = await page.evaluate(async () => {
      const settled = await window.client.whenStartupSettled();
      return { same: settled[0].reason === window.publicFailure,
        closed: window.nativeContexts.every(context => context.state === 'closed'),
        tracks: window.client.micStream.getTracks().every(track => track.readyState === 'ended') };
    });
    assert.ok(native.same && native.closed && native.tracks);
  }, '{"persisted":false}', { autoStart: false, framed: true, holdSessionAck: true });
});

for (const operation of ['Stop', 'typed']) {
test(`review99 ${operation} while original mint is pending retains the latest local boundary`, { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page, enteredMint, releaseMint, realtimeMessages }) => {
    await page.evaluate(() => { window.startup = window.client.start().then(() => null, error => error.message); });
    await enteredMint;
    const errors = await page.evaluate(operation => {
      const errors = [];
      for (let attempt = 0; attempt < 2; attempt++) {
        try { if (operation === 'Stop') window.client.cancelResponse(); else window.client.sendText('mint-replacement-' + attempt); }
        catch (error) { errors.push(error.message); }
      }
      return errors;
    }, operation);
    assert.deepEqual(errors, []);
    releaseMint();
    assert.equal(await page.evaluate(() => window.startup), null);
    await page.waitForFunction(() => window.nativeCaptureGenerations.includes(1));
    if (operation === 'typed') {
      for (let step = 0; step < 100 && !realtimeMessages().some(message => message.type === 'response.create'); step++) await page.waitForTimeout(10);
      assert.equal(realtimeMessages().filter(message => message.type === 'response.create').length, 1);
      const items = realtimeMessages().filter(message => message.type === 'conversation.item.create');
      assert.equal(items.length, 1);
      assert.equal(items[0].item.content[0].text, 'mint-replacement-1');
    }
  }, '{"persisted":false}', { autoStart: false, framed: true, holdMint: true });
});
}

for (const factory of ['capture', 'playback']) {
for (const operation of ['Stop', 'typed']) {
test(`review99 ${operation} during strict native ${factory} acquisition preserves a pending local boundary`, { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page, enteredWorklet, releaseWorklet, realtimeMessages }) => {
    await page.evaluate(() => { window.startup = window.client.start().then(() => null, error => error.message); });
    await enteredWorklet;
    const errors = await page.evaluate(operation => {
      const errors = [];
      for (let attempt = 0; attempt < 2; attempt++) {
        try { if (operation === 'Stop') window.client.cancelResponse(); else window.client.sendText('replacement-' + attempt); }
        catch (error) { errors.push(error.message); }
      }
      return errors;
    }, operation);
    assert.deepEqual(errors, [], 'early local abandonment depended on an unopened transport');
    releaseWorklet();
    assert.equal(await page.evaluate(() => window.startup), null);
    await page.waitForFunction(() => window.nativeCaptureGenerations.includes(2));
    assert.ok(await page.evaluate(() => window.client.micStream.getTracks().every(track => track.readyState === 'live')));
    if (operation === 'typed') {
      for (let step = 0; step < 100 && !realtimeMessages().some(message => message.type === 'response.create'); step++) await page.waitForTimeout(10);
      assert.equal(realtimeMessages().filter(message => message.type === 'response.create').length, 1);
      const texts = realtimeMessages().filter(message => message.type === 'conversation.item.create');
      assert.equal(texts.length, 1);
      assert.equal(texts[0].item.content[0].text, 'replacement-1');
    }
  }, '{"persisted":false}', { autoStart: false, framed: true, holdWorklet: factory });
});
}
for (const clientMode of ['framed', 'pipeline']) {
test(`review99 close settles ${clientMode} public startup before native ${factory} worklet HTTP is released`, { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page, enteredWorklet, releaseWorklet }) => {
    await page.evaluate(async clientMode => {
      if (clientMode === 'pipeline') {
        const { createPipelineClient } = await import('/js/voice/pipeline.js');
        window.client = await createPipelineClient({ onEvent: event => window.nativeEvents.push(event) });
      }
      window.startup = window.client.start().then(() => 'started', () => 'closed');
    }, clientMode);
    await enteredWorklet;
    const state = await page.evaluate(async () => {
      await window.client.close();
      const outcome = await Promise.race([window.startup, new Promise(resolve => setTimeout(() => resolve('pending'), 150))]);
      return { outcome, nodes: window.nativeWorkletNodes.length, contexts: window.nativeContexts.map(context => context.state),
        tracks: window.client.micStream.getTracks().map(track => track.readyState) };
    });
    assert.equal(state.outcome, 'closed', 'public startup remained pending until fixture released HTTP');
    assert.ok(state.contexts.every(value => value === 'closed') && state.tracks.every(value => value === 'ended'));
    releaseWorklet();
    const late = await page.evaluate(async () => {
      const actual = await window.client.whenStartupSettled();
      return { status: actual[0].status, nodes: window.nativeWorkletNodes.length,
        sockets: window.nativeSockets.length, closed: window.nativeContexts.every(context => context.state === 'closed'),
        tracks: window.client.micStream.getTracks().every(track => track.readyState === 'ended') };
    });
    assert.equal(late.status, 'rejected');
    assert.equal(late.nodes, state.nodes, 'late native acquisition created a successor worklet');
    assert.equal(late.sockets, 0, 'retired startup opened a late socket');
    assert.ok(late.closed && late.tracks);
  }, '{"persisted":false}', { autoStart: false, framed: true, holdWorklet: factory });
});
}
}

for (const socketIdentity of [false, true]) {
test(`review99 schema-valid effective config ACK need not include socket identity (created ID ${socketIdentity})`, { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page }) => {
    const ready = await page.evaluate(() => window.nativeEvents.find(event => event.kind === 'session_ready'));
    assert.equal(ready.session_id, socketIdentity ? 'actual-socket-id' : '');
    assert.ok(await page.evaluate(() => window.client.micStream.getTracks().every(track => track.readyState === 'live')));
  }, '{"persisted":false}', { framed: true, ackIdentity: false, socketIdentity });
});
}

test('review99 harmless no-active-response cancel error preserves fresh native input and output', { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page, sendRealtime, realtimeMessages }) => {
    await page.evaluate(() => window.client.cancelResponse());
    sendRealtime({ type: 'error', error: { code: 'response_cancel_not_active', message: 'No response is in progress' } });
    await page.waitForTimeout(150);
    assert.ok(await page.evaluate(() => window.client.micStream.getTracks().every(track => track.readyState === 'live')));
    assert.equal(await page.evaluate(() => window.nativeEvents.some(event => event.kind === 'error' && event.fatal)), false);
    sendRealtime({ type: 'input_audio_buffer.speech_started', item_id: 'after-cancel' });
    sendRealtime({ type: 'input_audio_buffer.committed', item_id: 'after-cancel' });
    for (let step = 0; step < 100 && !realtimeMessages().some(message => message.type === 'response.create'); step++) await page.waitForTimeout(10);
    const request = realtimeMessages().find(message => message.type === 'response.create');
    assert.ok(request);
    sendRealtime({ type: 'response.created', response: { id: 'after-cancel', metadata: request.response.metadata } });
    sendRealtime({ type: 'response.output_text.delta', response_id: 'after-cancel', delta: 'FRESH_AFTER_CANCEL' });
    await page.waitForFunction(() => window.nativeEvents.some(event => event.text === 'FRESH_AFTER_CANCEL'));
  }, '{"persisted":false}', { framed: true });
});

test('review99 legacy API opt-out retains explicit fallback description without opening a microphone', { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page }) => {
    const refusal = await page.evaluate(async () => {
      try { await window.client.start(); return null; }
      catch (error) { return { code: error.code, fallback: error.fallback }; }
    });
    assert.equal(refusal.code, 'REALTIME_UNAVAILABLE');
    assert.equal(refusal.fallback.path, 'pipeline');
    assert.equal(await page.evaluate(() => window.client.micStream), null);
  }, '{}', { autoStart: false, declineMint: true });
});

test('review99 provider input protocol rejection after ready automatically retires originals', { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page, sendRealtime }) => {
    sendRealtime({ type: 'error', error: { code: 'invalid_value', message: 'input audio append rejected' } });
    await page.waitForTimeout(150);
    assert.ok(await page.evaluate(() => window.nativeEvents.some(event => event.kind === 'error' && event.fatal)));
    await page.waitForFunction(() => window.nativeContexts.every(context => context.state === 'closed')
      && window.nativeSockets[0].readyState === WebSocket.CLOSED
      && window.client.micStream.getTracks().every(track => track.readyState === 'ended'));
  }, '{"persisted":false}', { framed: true });
});

test('review99 strict Console refusal cannot silently start a Pipeline transport', { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page, pipelineFrames }) => {
    await page.evaluate(async () => {
      const voiceUI = await import('/js/voice/ui.js');
      window.uiSession = voiceUI.sessionForAgent(null);
      await voiceUI.startSession(window.uiSession);
    });
    await page.waitForTimeout(150);
    assert.notEqual(await page.evaluate(() => window.uiSession.path), 'pipeline');
    assert.equal(pipelineFrames(), 0);
  }, '{"persisted":false}', { autoStart: false, declineMint: true });
});

for (const path of ['realtime', 'pipeline']) {
test(`native ${path} clear-ACK failure retires original microphone, playback and socket`, { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page }) => {
    await page.evaluate(async path => {
      const NativeContext = window.AudioContext;
      const NativeSocket = window.WebSocket;
      window.originalContexts = [];
      window.originalSockets = [];
      window.AudioContext = class extends NativeContext {
        constructor(...args) { super(...args); window.originalContexts.push(this); }
      };
      window.WebSocket = class extends NativeSocket {
        constructor(...args) { super(...args); window.originalSockets.push(this); }
      };
      if (path === 'pipeline') {
        await window.client.close();
        const { createPipelineClient } = await import('/js/voice/pipeline.js');
        window.client = await createPipelineClient({
          wsPath: '/realtime', onEvent: event => window.nativeEvents.push(event),
        });
      }
      await window.client.start();
      window.originalMic = window.client.micStream;
    }, path);
    await page.waitForFunction(() => window.client.session !== null);
    await page.evaluate(() => window.client.cancelResponse());
    await page.waitForFunction(() => window.nativeEvents.some(event =>
      event.code === 'AUDIO_OUTPUT_CONTROL_FAILED' && event.fatal), null, { timeout: 4000 });
    // Observe automatic retirement FIRST. The fixture's finally owns any
    // later cleanup and must not make a broken error callback look correct.
    await page.waitForFunction(() => window.originalMic.getTracks().every(track => track.readyState === 'ended')
      && window.originalContexts.every(context => context.state === 'closed')
      && window.originalSockets.length === 1
      && window.originalSockets.every(socket => socket.readyState === WebSocket.CLOSED), null, { timeout: 2000 });
    const state = await page.evaluate(() => {
      return { tracks: window.originalMic.getTracks().map(track => track.readyState),
        contexts: window.originalContexts.map(context => context.state) };
    });
    assert.ok(state.tracks.length && state.tracks.every(state => state === 'ended'));
    assert.equal(state.contexts.length, 2);
    assert.ok(state.contexts.every(state => state === 'closed'));
    await page.waitForFunction(() => window.originalSockets.length === 1
      && window.originalSockets.every(socket => socket.readyState === WebSocket.CLOSED), null, { timeout: 2000 });
  }, '{}', { autoStart: false, transport: 'websocket', dropPlaybackClearAck: true });
});
}

for (const stops of [1, 2]) {
test(`native pending playback initialization forwards latest clear after ${stops} Stop request(s)`, { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page, enteredWorklet, releaseWorklet }) => {
    await page.evaluate(() => {
      const NativeNode = window.AudioWorkletNode;
      window.initializationClearAcks = [];
      window.AudioWorkletNode = class extends NativeNode {
        constructor(...args) {
          super(...args);
          if (args[1] === 'kestrel-playback') {
            this.port.addEventListener('message', event => {
              if (event.data?.type === 'cleared') window.initializationClearAcks.push(event.data);
            });
            this.port.start();
          }
        }
      };
      window.initializing = window.client.start();
    });
    await enteredWorklet;
    await page.evaluate(stops => {
      for (let count = 0; count < stops; count++) window.client.cancelResponse();
    }, stops);
    releaseWorklet();
    await page.evaluate(() => window.initializing);
    await page.waitForFunction(stops => window.initializationClearAcks.some(ack => ack.generation === stops), stops, { timeout: 1000 });
    assert.ok(await page.evaluate(() => window.client.micStream.getTracks().every(track => track.readyState === 'live')));
  }, '{}', { autoStart: false, transport: 'websocket', holdWorklet: 'playback' });
});
}

test('native WebSocket typed replacement clears actual playback without closing fresh microphone', { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page, sendRealtime }) => {
    await page.evaluate(async () => {
      const NativeNode = window.AudioWorkletNode;
      window.actualClearAcks = [];
      window.AudioWorkletNode = class extends NativeNode {
        constructor(...args) {
          super(...args);
          if (args[1] === 'kestrel-playback') {
            this.port.addEventListener('message', event => {
              if (event.data?.type === 'cleared') window.actualClearAcks.push(event.data);
            });
            this.port.start();
          }
        }
      };
      await window.client.start();
    });
    sendRealtime({ type: 'response.created', response: { id: 'old-playback' } });
    sendRealtime({ type: 'response.output_audio.delta', response_id: 'old-playback',
      delta: Buffer.from(new Int16Array(24000).fill(19660).buffer).toString('base64') });
    await page.waitForFunction(() => window.nativeEvents.some(event => event.kind === 'speaking_started'));
    await page.evaluate(() => window.client.sendText('new local input'));
    await page.waitForFunction(() => window.actualClearAcks.length === 1, null, { timeout: 2000 });
    assert.ok(await page.evaluate(() => window.client.micStream.getTracks().every(track => track.readyState === 'live')));
  }, '{"persisted":true}', { autoStart: false, transport: 'websocket' });
});

for (const early of [false, true]) {
test(`native microphone reset excludes a delayed original frame and preserves fresh PCM (early=${early})`, { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page }) => {
    try {
      await page.evaluate(async early => {
        const NativeNode = window.AudioWorkletNode;
        window.AudioWorkletNode = class extends NativeNode {
          constructor(...args) {
            super(...args);
            if (args[1] === 'kestrel-capture') window.originalCaptureNode = this;
          }
        };
        const { createVoiceCapture } = await import('/js/voice/capture.js');
        window.captureProof = await createVoiceCapture({ onAcquire: owner => {
          if (early) { owner.reset(); owner.reset(); }
        } });
        window.originalProofMic = window.captureProof.micStream;
        window.capturedProofFrames = [];
        window.captureProof.onchunk((pcm, metadata) => window.capturedProofFrames.push({
          generation: metadata?.generation,
          nonzero: [...new Int16Array(pcm.buffer, pcm.byteOffset, pcm.byteLength / 2)].some(sample => sample !== 0),
        }));
        // Delay ONE ACTUAL native MessageEvent/PCM frame, not a fabricated
        // frame or completion. Replay its original delivery after reset.
        window.originalCaptureDelivery = window.originalCaptureNode.port.onmessage;
        window.originalCaptureNode.port.onmessage = event => {
          if (event.data?.type === 'chunk' && !window.delayedOriginalFrame) {
            window.delayedOriginalFrame = event;
            return;
          }
          window.originalCaptureDelivery(event);
        };
      }, early);
      await page.waitForFunction(() => !!window.delayedOriginalFrame);
      assert.equal(await page.evaluate(() => window.delayedOriginalFrame.data.generation), early ? 2 : 0,
        'latest reset requested before worklet creation was not delivered');
      await page.evaluate(() => {
        window.capturedProofFrames = [];
        window.requestedCaptureGeneration = window.captureProof.reset();
      });
      await page.waitForFunction(() => window.capturedProofFrames.some(frame =>
        frame.generation === window.requestedCaptureGeneration && frame.nonzero));
      const state = await page.evaluate(() => {
        const before = window.capturedProofFrames.length;
        window.originalCaptureDelivery(window.delayedOriginalFrame);
        return { before, after: window.capturedProofFrames.length,
          generations: window.capturedProofFrames.map(frame => frame.generation),
          requested: window.requestedCaptureGeneration,
          sameMic: window.captureProof.micStream === window.originalProofMic,
          tracks: window.originalProofMic.getTracks().map(track => track.readyState) };
      });
      assert.equal(state.after, state.before, 'delayed original native microphone frame crossed reset');
      assert.ok(state.generations.length && state.generations.every(generation => generation === state.requested));
      assert.ok(state.sameMic && state.tracks.every(state => state === 'live'));
    } finally {
      await page.evaluate(() => window.captureProof?.destroy());
    }
  }, '{}', { autoStart: false });
});
}

test('native WebRTC buffered old input commit cannot reopen pre-ID cancellation', { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page }) => {
    await page.evaluate(() => {
      window.client.cancelResponse();
      for (const payload of [
        { type: 'input_audio_buffer.committed', item_id: 'old-buffer' },
        { type: 'response.created', response: { id: 'late-old' } },
        { type: 'response.function_call_arguments.done', response_id: 'late-old', call_id: 'old-call', name: 'offline', arguments: '{}' },
        { type: 'response.done', response: { id: 'late-old', status: 'completed' } },
        { type: 'conversation.item.input_audio_transcription.completed', item_id: 'barrier-old', transcript: 'input barrier' },
      ]) window.nativeChannel.send(JSON.stringify(payload));
    });
    await page.waitForFunction(() => window.nativeEvents.some(event => event.item_id === 'barrier-old'));
    assert.equal(await page.evaluate(() => window.nativeEvents.filter(event => event.kind === 'tool_call_batch_requested').length), 0);
    assert.equal(await page.evaluate(() => window.nativeEvents.filter(event => event.kind === 'speaking_started').length), 0);
    await page.evaluate(() => {
      for (const payload of [
        { type: 'input_audio_buffer.cleared' },
        { type: 'input_audio_buffer.committed', item_id: 'fresh-buffer' },
        { type: 'response.created', response: { id: 'fresh-response' } },
        { type: 'response.function_call_arguments.done', response_id: 'fresh-response', call_id: 'fresh-call', name: 'offline', arguments: '{}' },
        { type: 'response.done', response: { id: 'fresh-response', status: 'completed' } },
      ]) window.nativeChannel.send(JSON.stringify(payload));
    });
    await page.waitForFunction(() => window.nativeEvents.some(event => event.kind === 'tool_call_batch_requested'));
    assert.deepEqual(await page.evaluate(() => window.nativeEvents.filter(event => event.kind === 'tool_call_batch_requested').map(event => event.batch_id)), ['fresh-response']);
  }, '{"persisted":true,"status":"ok"}');
});

test('native server VAD does not clear the fresh utterance that triggered speech', { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page }) => {
    await page.evaluate(() => {
      for (const payload of [
        { type: 'input_audio_buffer.speech_started', item_id: 'original-fresh-input' },
        { type: 'input_audio_buffer.speech_stopped', item_id: 'original-fresh-input' },
        { type: 'input_audio_buffer.committed', item_id: 'original-fresh-input' },
        { type: 'response.created', response: { id: 'original-fresh-response' } },
        { type: 'conversation.item.input_audio_transcription.completed', item_id: 'original-fresh-input', transcript: 'fresh input barrier' },
      ]) window.nativeChannel.send(JSON.stringify(payload));
    });
    await page.waitForFunction(() => window.nativeEvents.some(event => event.item_id === 'original-fresh-input'));
    const state = await page.evaluate(() => ({
      clears: window.nativeMessages.filter(message => message.type === 'input_audio_buffer.clear').length,
      speaking: window.nativeEvents.filter(event => event.kind === 'speaking_started').length,
    }));
    assert.equal(state.clears, 0, 'ordinary VAD speech destructively cleared newly detected input');
    assert.equal(state.speaking, 1);
  }, '{"persisted":true,"status":"ok"}');
});

test('native typed input requires matching original response correlation after pre-ID Stop', { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page }) => {
    await page.evaluate(() => {
      window.client.cancelResponse();
      window.client.sendText('new original typed input');
      for (const payload of [
        { type: 'response.created', response: { id: 'old-before-clear' } },
        { type: 'response.function_call_arguments.done', response_id: 'old-before-clear', call_id: 'old-call', name: 'offline', arguments: '{}' },
        { type: 'response.done', response: { id: 'old-before-clear', status: 'completed' } },
        { type: 'input_audio_buffer.cleared' },
        { type: 'response.created', response: { id: 'old-after-clear' } },
        { type: 'response.function_call_arguments.done', response_id: 'old-after-clear', call_id: 'later-old-call', name: 'offline', arguments: '{}' },
        { type: 'response.done', response: { id: 'old-after-clear', status: 'completed' } },
        { type: 'conversation.item.input_audio_transcription.completed', item_id: 'typed-old-barrier', transcript: 'old output barrier' },
      ]) window.nativeChannel.send(JSON.stringify(payload));
    });
    await page.waitForFunction(() => window.nativeEvents.some(event => event.item_id === 'typed-old-barrier'));
    assert.equal(await page.evaluate(() => window.nativeEvents.filter(event => event.kind === 'tool_call_batch_requested').length), 0,
      'new text admitted buffered output from the cancelled original response');
    await page.waitForFunction(() => window.nativeMessages.some(message => message.type === 'response.create'));
    const metadata = await page.evaluate(() => window.nativeMessages.find(message => message.type === 'response.create').response?.metadata);
    assert.ok(metadata?.kestrel_input_generation, 'typed native request lacks original response correlation');
    await page.evaluate(metadata => {
      for (const payload of [
        { type: 'response.created', response: { id: 'fresh-typed-response', metadata } },
        { type: 'response.function_call_arguments.done', response_id: 'fresh-typed-response', call_id: 'fresh-typed-call', name: 'offline', arguments: '{}' },
        { type: 'response.done', response: { id: 'fresh-typed-response', status: 'completed' } },
      ]) window.nativeChannel.send(JSON.stringify(payload));
    }, metadata);
    await page.waitForFunction(() => window.nativeEvents.some(event => event.kind === 'tool_call_batch_requested'));
    assert.deepEqual(await page.evaluate(() => window.nativeEvents.filter(event => event.kind === 'tool_call_batch_requested').map(event => event.batch_id)),
      ['fresh-typed-response']);
  }, '{"persisted":true,"status":"ok"}');
});

test('native typed response without correlation closes the original transport within its bound', { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page }) => {
    await page.evaluate(() => {
      window.client.sendText('bounded original input');
      window.nativeChannel.send(JSON.stringify({ type: 'response.created', response: { id: 'uncorrelated' } }));
    });
    await page.waitForFunction(() => window.nativeEvents.some(event => event.kind === 'session_closed'));
    assert.equal(await page.evaluate(() => window.nativeEvents.some(event =>
      event.code === 'RESPONSE_CORRELATION_UNCONFIRMED' && event.fatal)), true);
    assert.equal(await page.evaluate(() => window.nativeEvents.filter(event => event.kind === 'speaking_started').length), 0);
    assert.equal(await page.evaluate(() => window.nativeChannel.readyState), 'closed');
  }, '{"persisted":true,"status":"ok"}', { responseStartTimeoutMs: 1000 });
});

test('native websocket barge-in fences playback continuation and late publication', { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page, sendRealtime, realtimeMessages }) => {
    await page.evaluate(async () => {
      window.playbackPushes = 0;
      const NativeNode = window.AudioWorkletNode;
      window.AudioWorkletNode = class extends NativeNode {
        constructor(...args) {
          super(...args);
          const original = this.port.postMessage.bind(this.port);
          this.port.postMessage = (...messages) => {
            if (messages[0]?.type === 'push') window.playbackPushes++;
            return original(...messages);
          };
        }
      };
      await window.client.start();
    });
    sendRealtime({ type: 'response.created', response: { id: 'original' } });
    await page.waitForFunction(() => window.nativeEvents.some(event => event.kind === 'speaking_started'));
    const delta = Buffer.alloc(24000 * 2 * 4, 1).toString('base64');
    sendRealtime({ type: 'response.output_audio.delta', response_id: 'original', delta });
    await page.waitForFunction(() => window.playbackPushes === 1);
    await page.evaluate(() => {
      window.continuationSettled = false;
      window.continuation = window.client.commitToolResults([
        { call_id: 'original-call', result: { native: true } },
      ]).then(() => { window.continuationSettled = true; });
    });
    assert.equal(await page.evaluate(() => window.continuationSettled), false);
    sendRealtime({ type: 'input_audio_buffer.speech_started' });
    await page.evaluate(() => window.continuation);
    sendRealtime({ type: 'response.output_audio.delta', response_id: 'original', delta });
    sendRealtime({ type: 'response.output_text.delta', response_id: 'original', delta: 'cancelled-private-text' });
    sendRealtime({ type: 'response.created', response: { id: 'delayed-cancelled' } });
    sendRealtime({ type: 'response.output_audio.delta', response_id: 'delayed-cancelled', delta });
    sendRealtime({ type: 'input_audio_buffer.speech_stopped', item_id: 'input-barrier' });
    await page.waitForFunction(() => window.nativeEvents.some(event => event.kind === 'listening_stopped'));
    assert.equal(await page.evaluate(() => window.playbackPushes), 1);
    assert.equal(await page.evaluate(() => window.nativeEvents.filter(event => event.kind === 'speaking_started').length), 1);
    assert.equal(await page.evaluate(() => window.nativeEvents.some(event => event.text === 'cancelled-private-text')), false);
    assert.equal(realtimeMessages().filter(event => event.type === 'response.create').length, 0);
  }, '{}', { autoStart: false, transport: 'websocket' });
});

for (const mode of ['pipeline', 'realtime']) {
for (const factory of ['capture', 'playback']) {
test(`Stop reaches partially acquired native ${factory} while ${mode} worklet HTTP is pending`, { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page, enteredWorklet, releaseWorklet }) => {
    await page.evaluate(async mode => {
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
      if (mode === 'pipeline') {
        const { createPipelineClient } = await import('/js/voice/pipeline.js');
        window.client = await createPipelineClient({ onEvent: event => window.nativeEvents.push(event) });
      }
      window.startup = window.client.start().then(() => null, error => error.message);
    }, mode);
    await enteredWorklet;
    const state = await page.evaluate(async () => {
      await window.client.close();
      return { contexts: window.acquiredContexts.map(ctx => ctx.state),
        tracks: window.acquiredStreams.flatMap(stream => stream.getTracks().map(track => track.readyState)),
        closed: window.nativeEvents.filter(event => event.kind === 'session_closed').length };
    });
    releaseWorklet();
    assert.ok(await page.evaluate(() => window.startup), 'retired startup must not create later native nodes');
    assert.equal(state.closed, 1);
    assert.ok(state.tracks.length && state.tracks.every(state => state === 'ended'), 'Stop returned with acquired microphone live');
    assert.ok(state.contexts.length && state.contexts.every(state => state === 'closed'), 'Stop returned with acquired AudioContext live');
  }, '{}', { autoStart: false, holdWorklet: factory, transport: 'websocket' });
});
}
}

for (const interruption of ['Stop', 'Barge-in']) {
test(`${interruption} during native tool authentication cannot dispatch the captured session batch`, { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page, toolPosts, releaseBatch }) => {
    await page.evaluate(async () => {
      const { default: API } = await import('/js/api.js');
      const apply = API.applyAuth.bind(API);
      const release = new Promise(resolve => { window.releaseAuth = resolve; });
      API.applyAuth = async (...args) => {
        const headers = await apply(...args);
        window.authEntered = true;
        await release;
        return headers;
      };
      window.voiceUI = await import('/js/voice/ui.js');
      window.uiSession = window.voiceUI.createSession(null);
      window.uiSession.client = window.client;
      window.uiTask = window.voiceUI.handleToolBatch(window.uiSession, { calls: [
        { call_id: 'original', name: 'native-no-execution', arguments: {} },
      ] });
    });
    await page.waitForFunction(() => window.authEntered);
    await page.evaluate(interruption => {
      if (interruption === 'Stop') window.voiceUI.surfaceFatalError(new Error('operator Stop'), window.uiSession);
      else window.client.cancelResponse();
    }, interruption);
    await page.evaluate(() => window.releaseAuth());
    await page.waitForTimeout(150);
    if (toolPosts()) releaseBatch({ results: [{ call_id: 'original', result: { text: 'offline' } }] });
    await page.evaluate(() => window.uiTask);
    assert.equal(toolPosts(), 0, 'authentication completion dispatched a retired session tool batch');
  });
});
}

test('obsolete native microphone startup cannot close the replacement UI session', { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page }) => {
    await page.evaluate(async () => {
      const nativeCapture = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
      const release = new Promise(resolve => { window.releaseFirstPermission = resolve; });
      window.acquiredStreams = [];
      navigator.mediaDevices.getUserMedia = async (...args) => {
        const stream = await nativeCapture(...args);
        window.acquiredStreams.push(stream);
        if (window.acquiredStreams.length === 1) await release;
        return stream;
      };
      window.voiceUI = await import('/js/voice/ui.js');
      window.uiSession = window.voiceUI.sessionForAgent(null);
      window.firstStartup = window.voiceUI.startSession(window.uiSession);
    });
    await page.waitForFunction(() => window.acquiredStreams.length === 1);
    await page.evaluate(() => window.voiceUI.stopSession(window.uiSession));
    await page.evaluate(async () => {
      await window.voiceUI.startSession(window.uiSession);
      window.replacement = window.uiSession.client;
      window.client = window.replacement;
      window.releaseFirstPermission();
      await window.firstStartup;
    });
    const state = await page.evaluate(() => ({
      same: window.uiSession.client === window.replacement,
      first: window.acquiredStreams[0].getTracks().map(track => track.readyState),
      replacement: window.replacement.micStream.getTracks().map(track => track.readyState),
    }));
    assert.ok(state.same, 'obsolete startup error cleared the replacement UI client');
    assert.ok(state.first.every(state => state === 'ended'));
    assert.ok(state.replacement.length && state.replacement.every(state => state === 'live'), 'obsolete startup retired replacement microphone');
  }, '{}', { autoStart: false });
});

test('explicit operator Pipeline selection preserves armed native microphone without a realtime mint', { timeout: 30000 }, async () => {
  await nativeVoice(async ({ page, pipelineFrames, mintRequests }) => {
    await page.evaluate(async () => {
      localStorage.setItem('kestrel.voice.settings', JSON.stringify({ mode: 'pipeline' }));
      const voiceUI = await import('/js/voice/ui.js');
      window.uiSession = voiceUI.sessionForAgent(null);
      await voiceUI.startSession(window.uiSession);
      window.client = window.uiSession.client;
    });
    assert.equal(await page.evaluate(() => window.uiSession.path), 'pipeline');
    await page.waitForTimeout(600);
    assert.ok(pipelineFrames() > 0, 'fallback policy disarmed the actual microphone input');
    assert.equal(mintRequests(), 0);
  }, '{}', { autoStart: false, declineMint: true });
});

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
