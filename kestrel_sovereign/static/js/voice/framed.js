/** Explicit OpenAI GA framed-PCM protocol, not a replacement transport owner.
 * All sends, worklets and closure belong to the original realtime client.
 * Metadata is response routing evidence, never tool execution authority.
 */
export const FRAMED_OUTPUT_PROFILE = 'response-framed-pcm-v1';
export const FRAMED_PROTOCOL = 'openai-realtime-ga-framed-v1';

export function requireFramedBootstrap(session, requested) {
  if (requested !== FRAMED_OUTPUT_PROFILE || session.output_profile !== requested
      || session.vendor !== 'openai' || session.transport !== 'websocket'
      || session.protocol !== FRAMED_PROTOCOL || !session.model
      || !session.client_secret?.value) throw new Error('Required framed realtime profile is unavailable.');
  requireFramedConfiguration({ ...session.session_config, model: session.model }, session.model);
  if (Object.hasOwn(session.session_config, 'model')) throw new Error('Realtime session.update cannot change model.');
}

export function requireFramedConfiguration(config, model) {
  const input = config?.audio?.input;
  const output = config?.audio?.output;
  const detection = input?.turn_detection;
  if (config?.type !== 'realtime' || config.model !== model
      || config.output_modalities?.length !== 1 || config.output_modalities[0] !== 'audio'
      || input?.format?.type !== 'audio/pcm' || input.format.rate !== 24000
      || output?.format?.type !== 'audio/pcm' || output.format.rate !== 24000
      || !['server_vad', 'semantic_vad'].includes(detection?.type)
      || detection.create_response !== false || detection.interrupt_response !== false) {
    throw new Error('Provider did not acknowledge the required framed PCM configuration.');
  }
}

export function createFramedProtocol({ getSession, send: nativeSend, resetCapture, fail, onReady,
  timeoutMs = 30000, maxAudioBytes = 96000 }) {
  let stopped = false;
  let ready = false;
  let startupResolve;
  let startupReject;
  let startupTimer;
  let responseTimer;
  let clearTimer;
  let clears = 0;
  let unsentClear = false;
  let pendingCancel = false;
  let socketSessionId = '';
  let chunks = [];
  let bytes = 0;
  let generation = crypto.randomUUID();
  let speechItem = '';
  let pending = null;
  let active = '';
  let afterClear = null;
  const completedCalls = new Set();
  const initialized = new Promise((resolve, reject) => { startupResolve = resolve; startupReject = reject; });
  // Acquisition may fail before start() joins this original promise.
  initialized.catch(() => {});

  function fatal(error) {
    if (stopped) return;
    close(error);
    fail(error);
  }
  function send(message) {
    try { nativeSend(message); }
    catch (error) { fatal(error); throw error; }
  }
  function flushControls() {
    if (!ready || stopped) return;
    if (pendingCancel) { pendingCancel = false; send({ type: 'response.cancel' }); }
    if (unsentClear) {
      unsentClear = false;
      clears += 1;
      if (clears > 16) return fatal(new Error('Too many unacknowledged input clears.'));
      send({ type: 'input_audio_buffer.clear' });
    }
  }
  function requestCancel() {
    if (stopped) return;
    pendingCancel = true;
    flushControls();
  }
  function flushInput() {
    if (!ready || clears || unsentClear || stopped) return;
    for (const pcm of chunks) send({ type: 'input_audio_buffer.append', audio: pcm });
    chunks = [];
    bytes = 0;
  }
  function audio(base64, byteLength) {
    if (stopped) return;
    if (ready && !clears && !unsentClear) {
      send({ type: 'input_audio_buffer.append', audio: base64 });
    } else {
      bytes += byteLength;
      if (bytes > maxAudioBytes) return fatal(new Error('Framed input buffer exceeded its bounded audio budget.'));
      chunks.push(base64);
    }
  }
  function abandon() {
    generation = crypto.randomUUID();
    speechItem = '';
    pending = null;
    active = '';
    completedCalls.clear();
    afterClear = null;
    clearTimeout(responseTimer);
  }
  function clearInput(action = null) {
    if (stopped) return;
    // Local boundary FIRST, independent of whether the original socket exists.
    try { resetCapture(); }
    catch (error) { fatal(error); throw error; }
    chunks = [];
    bytes = 0;
    afterClear = action;
    // Before configuration readiness, repeated requests coalesce. Once sent,
    // every original clear is counted until its ordered acknowledgement.
    unsentClear = true;
    // Repeated Stop does not extend the original native acknowledgement budget.
    if (!clearTimer) clearTimer = setTimeout(() => fatal(new Error('Provider input clear was not acknowledged.')), timeoutMs);
    flushControls();
  }
  function request() {
    if (stopped || !ready || clears || unsentClear || pending) throw new Error('Original framed response is not ready for admission.');
    const nonce = crypto.randomUUID();
    pending = { generation, nonce };
    active = '';
    send({ type: 'response.create', response: { metadata: {
      kestrel_input_generation: generation, kestrel_response_nonce: nonce,
    } } });
    clearTimeout(responseTimer);
    responseTimer = setTimeout(() => fatal(new Error('Provider response correlation was not acknowledged.')), timeoutMs);
  }
  function accept(raw) {
    if (stopped) return false;
    if (raw.type === 'session.created') {
      const id = raw.session?.id;
      if (typeof id === 'string' && id) {
        if (socketSessionId && socketSessionId !== id) fatal(new Error('Original socket session identity changed.'));
        else socketSessionId = id;
      }
      return false;
    }
    if (raw.type === 'session.updated') {
      try {
        requireFramedConfiguration(raw.session, getSession().model);
        if (socketSessionId && raw.session.id && raw.session.id !== socketSessionId) throw new Error('Original socket session identity changed.');
      }
      catch (error) { fatal(error); return false; }
      if (!ready) {
        ready = true;
        clearTimeout(startupTimer);
        flushControls();
        flushInput();
        startupResolve();
        // SessionUpdatedEvent's configuration schema does not require id.
        // Preserve an actual created identity when supplied, never substitute
        // the separate ephemeral-registration ID or invent an identity.
        onReady({ ...raw.session, id: socketSessionId || raw.session.id || '' });
      }
      return false;
    }
    if (!ready) {
      if (raw.type === 'error') fatal(new Error(raw.error?.message || 'Provider rejected framed session configuration.'));
      return false;
    }
    if (raw.type === 'error' && raw.error?.code !== 'response_cancel_not_active') {
      fatal(new Error(raw.error?.message || 'Provider rejected the original framed protocol operation.'));
      return false;
    }
    if (raw.type === 'input_audio_buffer.cleared') {
      if (clears) {
        clears -= 1;
        if (!clears) {
          clearTimeout(clearTimer);
          clearTimer = null;
          const action = afterClear;
          afterClear = null;
          // The replacement typed item precedes fresh buffered microphone data.
          action?.();
          flushInput();
        }
      }
      return false;
    }
    if (raw.type.startsWith('input_audio_buffer.') && (clears || unsentClear)) return false;
    if (raw.type === 'response.created') {
      const metadata = raw.response?.metadata;
      const id = raw.response?.id;
      if (!id || !pending || metadata?.kestrel_input_generation !== pending.generation
          || metadata.kestrel_response_nonce !== pending.nonce) return false;
      pending = null;
      active = id;
      clearTimeout(responseTimer);
      return true;
    }
    if (raw.type.startsWith('response.')) {
      const id = raw.response?.id ?? raw.response_id;
      return !!active && id === active;
    }
    return true;
  }
  function speechStarted(item) {
    speechItem = item || '';
  }
  function committed(item) {
    if (!speechItem || speechItem !== item || clears || unsentClear || stopped) return;
    speechItem = '';
    request();
  }
  function completed(calls, successful) {
    active = '';
    completedCalls.clear();
    if (successful) for (const call of calls) completedCalls.add(call.call_id);
  }
  function reserveToolResults(results) {
    const ids = new Set(results.map(item => item.call_id));
    if (!ids.size || ids.size !== results.length || ids.size !== completedCalls.size
        || [...ids].some(id => !id || !completedCalls.has(id))) throw new Error('Tool results do not belong to the original completed response.');
    completedCalls.clear(); // uncertain/submitted batches cannot be replayed
  }
  function close(error = new Error('Original framed realtime client closed.')) {
    if (stopped) return;
    stopped = true;
    clearTimeout(startupTimer);
    clearTimeout(responseTimer);
    clearTimeout(clearTimer);
    chunks = [];
    bytes = 0;
    afterClear = null;
    unsentClear = false;
    pendingCancel = false;
    completedCalls.clear();
    startupReject(error);
  }
  return { audio, accept, abandon, clearInput, requestCancel, request, speechStarted, committed,
    completed, reserveToolResults, close,
    async start() {
      startupTimer = setTimeout(() => fatal(new Error('Provider session configuration was not acknowledged.')), timeoutMs);
      send({ type: 'session.update', session: getSession().session_config });
      return initialized;
    },
  };
}
