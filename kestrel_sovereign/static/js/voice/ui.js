/**
 * ui.js — Voice UI shell.
 *
 * The single visible surface that lets a user start, run, and end a voice
 * conversation. Wraps the Realtime WebRTC client (#728) and the Pipeline
 * WebSocket client (#729); the resolver server-side picks which path is
 * legal given the user's privacy mode + LLM vendor (#723), and this shell
 * does only fallback orchestration: try Realtime first, fall back to
 * Pipeline on the documented HTTP 409.
 *
 * Components rendered:
 *
 * - Voice toggle button in the chat header (mic icon).
 * - Per-state mic icon + ring animation: idle / listening / thinking /
 *   speaking / error.
 * - Transcript drawer that appears under the chat header during a session,
 *   showing live user + agent text.
 * - Path badge ("Realtime" / "Pipeline") with a tooltip explaining why.
 * - Voice picker modal (provider voices + session-instructions field).
 * - Privacy banner when the active mode is non-NORMAL.
 *
 * Single export `initVoiceUI()` is called once from app.js after
 * `initChat()`. Everything else is internal.
 */

import API from '../api.js';
import { addMessage, addMessageStreaming, finalizeStreamingMessage } from '../chat.js';
import { getOrCreateChatPane } from '../ui.js';
import { Events } from './events.js';
import { createRealtimeClient } from './realtime.js';
import { createPipelineClient } from './pipeline.js';
import { State, nextStateForEvent } from './state-machine.js';

// State.* + nextStateForEvent are imported from state-machine.js so the
// pure transition logic stays Node-testable.

/**
 * Build the auth header bag for voice fetch calls.
 *
 * Voice endpoints sit behind the same auth middleware as every other
 * Kestrel HTTP route, so the mint/voices fetches need the same `X-API-Key`
 * (or `Authorization: Bearer …`) the app's regular API client sends.
 * Without this any server with auth enabled returns 401, surfacing as a
 * fatal voice error in the UI.
 */
async function voiceAuthHeaders() {
  // Honors the active auth provider (API key, JWT, OAuth) — see #863.
  return typeof API.applyAuth === 'function' ? await API.applyAuth({}) : {};
}

const STATE_LABELS = {
  [State.IDLE]: 'Start voice session',
  [State.CONNECTING]: 'Connecting...',
  [State.LISTENING]: 'Listening — click to stop',
  [State.THINKING]: 'Thinking...',
  [State.SPEAKING]: 'Speaking — click to interrupt',
  [State.ERROR]: 'Voice session error — click to retry',
};

const STATE_GLYPH = {
  [State.IDLE]: '🎙️',
  [State.CONNECTING]: '⏳',
  [State.LISTENING]: '🎙️',
  [State.THINKING]: '💭',
  [State.SPEAKING]: '🔊',
  [State.ERROR]: '⚠️',
};


// ---------------------------------------------------------------------------
// Module-level handles to rendered DOM + per-agent sessions
// ---------------------------------------------------------------------------

let buttonEl = null;
let pathBadgeEl = null;
let pickerModalEl = null;
let privacyBannerEl = null;

const sessionByAgent = new Map();
// Sentinel for "no agent soloed/armed". A plain `null` can't serve here because
// `null` IS the valid host-agent key in standalone mode — conflating the two
// makes the standalone row read as soloed/armed by default and impossible to
// toggle off. A Symbol never equals any real agent key (string or null).
const NO_AGENT = Symbol('no-agent');
let soloAgent = NO_AGENT;
let armedAgent = NO_AGENT;

function createSession(agent) {
  return {
    agent,
    client: null,
    state: State.IDLE,
    agentMsgDiv: null,
    agentTextBuffer: '',
    userMsgDiv: null,
    userTranscriptBuffer: '',
    pathLabel: '',
    pathTooltip: '',
    realtimeModel: '',
    ownsSelector: false,
    startSeq: 0,
    explicitMuted: null,
  };
}

function sessionForAgent(agent) {
  const key = agent === undefined ? currentAgentKey() : agent;
  let session = sessionByAgent.get(key);
  if (!session) {
    session = createSession(key);
    sessionByAgent.set(key, session);
  }
  return session;
}

function activeSession() {
  return sessionForAgent(currentAgentKey());
}

function controlSession() {
  return armedAgent !== NO_AGENT ? sessionForAgent(armedAgent) : activeSession();
}

function paneForSession(session) {
  return getOrCreateChatPane(session.agent);
}

function isActiveSession(session) {
  return session.agent === currentAgentKey();
}

// User-overridable session settings, persisted to localStorage so they survive
// page reloads.  Voice is picked from the provider's list; instructions is the
// free-form steering directive forwarded to gpt-4o-mini-tts / Realtime
// session.instructions.
//
// Voice is identity — each agent has its own persona, voice, and steering
// directive.  Settings are therefore keyed by the currently-selected host
// agent (``API.getHostAgent()``); switching agents in multi-agent mode picks
// up a different localStorage slot rather than leaking the prior agent's
// picker state across the tenant.  See #1347.
//
// First-load order on each agent:
//   1. per-agent localStorage key (operator's most recent override)
//   2. hydrate from ``GET /voice/config`` (agent's persisted voice_config)
//   3. hardcoded defaults
//
// (2) is async and happens on first picker open via
// ``hydrateSettingsFromServer()`` so we don't block module init on a network
// call.  Reads before hydration completes get the localStorage / defaults.
const SETTINGS_KEY_PREFIX = 'kestrel.voice.settings';
const _settingsByAgent = new Map();
const _serverHydrated = new Set();
// Per-agent monotonic counter bumped on every ``savePicker``.  Hydration
// captures the counter when its GET starts; if it differs when the GET
// returns, the operator saved during the fetch and we must NOT overlay
// stale server data on top of their just-saved state.  In particular:
// without this, a "clear the directive" save that races with a slow
// hydration GET would see ``s.instructions`` blank, treat it as
// "no local override," and write the OLD ``cfg.voice_directive`` back
// — resurrecting the cleared persona.  See #1352 codex round-2.
const _saveGenByAgent = new Map();

// In-flight ``POST /voice/config`` promise per agent.  Hydration awaits
// this before issuing its GET so a save→reopen flow doesn't race the
// POST commit and resurrect stale server state.  Without it, the
// sequence (1) clear directive → (2) Save (POST in flight) →
// (3) reopen picker → hydration GET fires → might land before POST
// commits → reads OLD ``voice_directive`` → resurrects the cleared
// persona.  Codex round-6 catch on #1352.
const _pendingSaveByAgent = new Map();
let pickerRequestId = 0;

function settingsKeyForAgent(agentName) {
  // ``null`` host-agent (standalone single-agent mode) gets the unscoped
  // key, matching the pre-#1347 storage shape so existing single-agent
  // installs see their settings carry over.
  return agentName ? `${SETTINGS_KEY_PREFIX}.${agentName}` : SETTINGS_KEY_PREFIX;
}

function currentAgentKey() {
  try { return API.getHostAgent() || null; } catch (_) { return null; }
}

function buildAgentUrlForAgent(path, agent) {
  if (!agent) return path;
  return `/api/agents/${encodeURIComponent(agent)}${path}`;
}

function settings() {
  // Lazy accessor — always resolves against the CURRENT host agent.
  // Caches by agent so repeated reads in a single render don't re-parse
  // localStorage, but a host-agent switch picks up the new agent's slot
  // on the next access without explicit invalidation.
  const agent = currentAgentKey();
  let s = _settingsByAgent.get(agent);
  if (!s) {
    s = loadSettings(agent);
    _settingsByAgent.set(agent, s);
  }
  return s;
}

function settingsForAgent(agent) {
  let s = _settingsByAgent.get(agent);
  if (!s) {
    s = loadSettings(agent);
    _settingsByAgent.set(agent, s);
  }
  return s;
}


// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------


export function initVoiceUI() {
  const header = document.querySelector('.chat-header');
  if (!header) {
    // Voice UI is opt-in — if the chat header isn't present (e.g. mounted
    // somewhere unusual) we don't crash the page.
    console.warn('[voice/ui] chat header not found; voice UI not mounted');
    return;
  }

  injectStyles();
  mountButton(header);
  mountStatusIndicator();
  mountPickerModal();
  applyActiveSessionState();

  // Push-to-talk: hold spacebar (when not in a text input) to start voice.
  bindGlobalShortcuts();

  // Crash-recovery: if the tab is hidden / unloaded while a session is live,
  // the SESSION_CLOSED event may not reach us before teardown.  Release the
  // chat-model selector lock so the next page load isn't stuck on the
  // Realtime model.  See #1371.  ``pagehide`` is the bfcache-safe spelling
  // of unload; ``visibilitychange``→hidden catches mobile background.
  const onLeave = () => releaseSelectorOwnership(activeSession());
  window.addEventListener('pagehide', onLeave);
  window.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') onLeave();
  });
}

export function onAgentSwitch(prevAgent, nextAgent) {
  // No-op until the voice UI is actually mounted. When chat/voice capability
  // is disabled, initVoiceUI() returns before creating `buttonEl`, but
  // selectAgent() still fires this hook — and applyActiveSessionState() →
  // setState() would dereference the null `buttonEl` and crash the switch.
  if (!buttonEl) return;
  const prevSession = prevAgent === undefined ? null : sessionForAgent(prevAgent);
  const nextSession = sessionForAgent(nextAgent);
  if (prevSession?.ownsSelector) releaseSelectorOwnership(prevSession);

  applyActiveSessionPolicy();
  applyActiveSessionState();

  const nextPane = getOrCreateChatPane(nextAgent);
  if (nextPane.micArmed && nextSession.client && nextSession.realtimeModel) {
    acquireSelectorOwnership(nextSession.realtimeModel, nextSession);
  }
}

export function mountAgentVoiceControls(item, agentName) {
  if (!item) return;
  // Voice cards are located by their OWN attribute, NOT `data-agent-name`.
  // The voice session key for standalone mode is `null` (≠ the real agent name
  // the row still carries in `data-agent-name` for thinking-dot / stop-button
  // lookups), so the two identities genuinely differ and must not share a key.
  item.dataset.voiceAgentKey = voiceKeyToAttr(agentName);
  const controls = document.createElement('div');
  controls.className = 'agent-voice-controls';
  controls.innerHTML = `
    <span class="agent-voice-state" title="Voice state"></span>
    <button type="button" class="agent-voice-control agent-voice-mute" title="Mute playback" aria-label="Mute playback">🔇</button>
    <button type="button" class="agent-voice-control agent-voice-solo" title="Solo playback" aria-label="Solo playback">🎧</button>
    <button type="button" class="agent-voice-control agent-voice-arm" title="Arm microphone" aria-label="Arm microphone">●</button>
  `;
  controls.querySelector('.agent-voice-mute')?.addEventListener('click', (ev) => {
    ev.stopPropagation();
    toggleAgentMute(agentName);
  });
  controls.querySelector('.agent-voice-solo')?.addEventListener('click', (ev) => {
    ev.stopPropagation();
    toggleAgentSolo(agentName);
  });
  controls.querySelector('.agent-voice-arm')?.addEventListener('click', (ev) => {
    ev.stopPropagation();
    toggleAgentArm(agentName);
  });
  item.appendChild(controls);
  refreshAgentVoiceCard(agentName);
}

// Map a voice session key (string agent name, or `null` for standalone) to/from
// the stable DOM attribute used to locate that agent's card.
function voiceKeyToAttr(agent) {
  return agent === null || agent === undefined ? '__standalone__' : String(agent);
}
function attrToVoiceKey(attr) {
  return attr === '__standalone__' || attr === undefined ? null : attr;
}
function cssEscape(s) {
  return (typeof CSS !== 'undefined' && typeof CSS.escape === 'function')
    ? CSS.escape(s)
    : String(s).replace(/["\\]/g, '\\$&');
}

export function refreshAgentVoiceCard(agentName) {
  const selector = `.agent-item[data-voice-agent-key="${cssEscape(voiceKeyToAttr(agentName))}"]`;
  const row = document.querySelector(selector);
  if (!row) return;
  const session = sessionForAgent(agentName);
  const pane = getOrCreateChatPane(agentName);
  const outputMuted = isOutputMuted(agentName, session);
  row.dataset.voiceState = session.state;
  row.classList.toggle('agent-voice-live', session.state !== State.IDLE && session.state !== State.ERROR);
  row.classList.toggle('agent-voice-speaking', session.state === State.SPEAKING);
  row.classList.toggle('agent-voice-muted', outputMuted);
  row.classList.toggle('agent-voice-soloed', soloAgent === agentName);
  row.classList.toggle('agent-voice-armed', armedAgent === agentName);

  const stateEl = row.querySelector('.agent-voice-state');
  if (stateEl) {
    stateEl.textContent = stateLabel(session.state);
    stateEl.title = `Voice: ${stateLabel(session.state)}`;
  }
  const muteBtn = row.querySelector('.agent-voice-mute');
  if (muteBtn) {
    muteBtn.classList.toggle('active', session.explicitMuted === true);
    muteBtn.setAttribute('aria-pressed', session.explicitMuted === true ? 'true' : 'false');
    muteBtn.title = session.explicitMuted === true ? 'Unmute playback' : 'Mute playback';
  }
  const soloBtn = row.querySelector('.agent-voice-solo');
  if (soloBtn) {
    soloBtn.classList.toggle('active', soloAgent === agentName);
    soloBtn.setAttribute('aria-pressed', soloAgent === agentName ? 'true' : 'false');
  }
  const armBtn = row.querySelector('.agent-voice-arm');
  if (armBtn) {
    armBtn.classList.toggle('active', pane.micArmed);
    armBtn.setAttribute('aria-pressed', pane.micArmed ? 'true' : 'false');
  }
}

function refreshAllAgentVoiceCards() {
  document.querySelectorAll('.agent-item[data-voice-agent-key]').forEach((row) => {
    refreshAgentVoiceCard(attrToVoiceKey(row.dataset.voiceAgentKey));
  });
}

function stateLabel(state) {
  if (state === State.CONNECTING) return 'connecting';
  return state || State.IDLE;
}

function toggleAgentMute(agentName) {
  const session = sessionForAgent(agentName);
  session.explicitMuted = session.explicitMuted === true ? false : true;
  applyActiveSessionPolicy();
}

function toggleAgentSolo(agentName) {
  soloAgent = soloAgent === agentName ? NO_AGENT : agentName;
  applyActiveSessionPolicy();
}

function toggleAgentArm(agentName) {
  setArmedAgent(armedAgent === agentName ? NO_AGENT : agentName);
}

function setArmedAgent(agentName) {
  armedAgent = agentName;
  for (const [agent] of sessionByAgent.entries()) {
    getOrCreateChatPane(agent).micArmed = agent === armedAgent;
  }
  for (const row of document.querySelectorAll('.agent-item[data-voice-agent-key]')) {
    const agent = attrToVoiceKey(row.dataset.voiceAgentKey);
    getOrCreateChatPane(agent).micArmed = agent === armedAgent;
  }
  applyActiveSessionPolicy();
  if (buttonEl) applyActiveSessionState();
}

// Re-lock the chat-model selector to the active agent's live Realtime session.
// `selectAgent()` calls `loadModels()` AFTER `onAgentSwitch()`, and loadModels
// rebuilds `window._sharedModelSelector` from scratch — discarding any lock
// onAgentSwitch just acquired. Without re-locking, switching back to an agent
// with a running Realtime session leaves the mic live but the selector
// editable, letting the user swap the chat model out from under voice. Call
// this once the model list (and its selector) has been rebuilt.
export function reapplyActiveSelectorLock() {
  if (!buttonEl) return;  // voice UI not mounted — nothing can hold the lock
  const session = activeSession();
  // Pipeline sessions don't own the selector (no realtimeModel), so this is a
  // no-op for them — only a live Realtime session re-locks.
  if (session.client && session.realtimeModel) {
    acquireSelectorOwnership(session.realtimeModel, session);
  }
}


// ---------------------------------------------------------------------------
// Mounting
// ---------------------------------------------------------------------------


function mountButton() {
  // Mic lives in the chat input row, immediately to the LEFT of #send-button
  // — same affordance as ChatGPT's voice button. Voice is treated as an
  // input modality on the existing chat, NOT a parallel session.
  const sendBtn = document.getElementById('send-button');
  if (!sendBtn || !sendBtn.parentElement) {
    console.warn('[voice/ui] #send-button not found; voice button not mounted');
    return;
  }
  buttonEl = document.createElement('button');
  buttonEl.id = 'voice-toggle-btn';
  buttonEl.type = 'button';
  buttonEl.className = 'kestrel-voice-btn';
  buttonEl.title = STATE_LABELS[State.IDLE];
  buttonEl.setAttribute('aria-label', STATE_LABELS[State.IDLE]);
  buttonEl.setAttribute('aria-live', 'polite');
  buttonEl.textContent = STATE_GLYPH[State.IDLE];
  buttonEl.addEventListener('click', toggleSession);
  // Right-click opens the voice picker so power users can reach voice
  // settings without a separate UI affordance.
  buttonEl.addEventListener('contextmenu', (ev) => {
    ev.preventDefault();
    openPicker();
  });
  // Insert before the send button so the row reads: textarea | mic | send.
  sendBtn.parentElement.insertBefore(buttonEl, sendBtn);
}


function mountStatusIndicator() {
  // Tiny path/state indicator floats in the input footer next to the
  // context-status. Replaces the old separate-drawer header. Hidden when
  // idle so the existing chat UI is visually unchanged outside a session.
  const footer = document.querySelector('.input-footer');
  if (!footer) return;

  pathBadgeEl = document.createElement('span');
  pathBadgeEl.className = 'kestrel-voice-path-badge';
  pathBadgeEl.hidden = true;

  privacyBannerEl = document.createElement('span');
  privacyBannerEl.className = 'kestrel-voice-privacy-banner';
  privacyBannerEl.hidden = true;

  // Insert at the left of the footer so it doesn't fight the existing
  // right-aligned context-status text.
  footer.insertBefore(privacyBannerEl, footer.firstChild);
  footer.insertBefore(pathBadgeEl, footer.firstChild);
}


function mountPickerModal() {
  // Lazy-render: build the modal nodes upfront but keep them hidden.
  pickerModalEl = document.createElement('div');
  pickerModalEl.id = 'voice-picker-modal';
  pickerModalEl.className = 'kestrel-voice-modal';
  pickerModalEl.hidden = true;

  const card = document.createElement('div');
  card.className = 'kestrel-voice-modal-card';

  card.innerHTML = `
    <h3 class="kestrel-voice-modal-title">Voice settings</h3>
    <div id="voice-picker-route-preview" class="kestrel-voice-route-preview">
      Loading current route...
    </div>
    <label class="kestrel-voice-field">
      <span>Voice path</span>
      <select id="voice-picker-mode" class="model-selector">
        <option value="auto">Auto (Realtime when on OpenAI; Pipeline otherwise)</option>
        <option value="realtime">Force Realtime (uses gpt-realtime, your chat LLM is bypassed)</option>
        <option value="pipeline">Force Pipeline (STT → your chat LLM → TTS)</option>
      </select>
    </label>
    <label class="kestrel-voice-field">
      <span>TTS provider (Pipeline only)</span>
      <select id="voice-picker-tts" class="model-selector">
        <option value="">Auto (resolver picks)</option>
      </select>
    </label>
    <label class="kestrel-voice-field">
      <span>Voice</span>
      <select id="voice-picker-select" class="model-selector"></select>
    </label>
    <label class="kestrel-voice-field">
      <span>Session directive (optional)</span>
      <textarea id="voice-picker-instructions" rows="2"
        placeholder="Speak like a 1920s newscaster, measured but excited..."></textarea>
    </label>
    <p class="kestrel-voice-hint">
      Instructions are forwarded to the model that supports them
      (gpt-4o-mini-tts / Realtime). Local providers ignore them.
    </p>
    <div class="kestrel-voice-modal-actions">
      <button type="button" id="voice-picker-cancel" class="btn btn-secondary">Cancel</button>
      <button type="button" id="voice-picker-save" class="btn btn-primary">Save</button>
    </div>
  `;

  pickerModalEl.appendChild(card);
  pickerModalEl.addEventListener('click', (ev) => {
    if (ev.target === pickerModalEl) closePicker();  // click outside card
  });
  document.body.appendChild(pickerModalEl);

  card.querySelector('#voice-picker-cancel').addEventListener('click', closePicker);
  card.querySelector('#voice-picker-save').addEventListener('click', savePicker);
}


// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------


function setState(next, session = activeSession()) {
  session.state = next;
  refreshAgentVoiceCard(session.agent);
  if (session === controlSession()) {
    buttonEl.textContent = STATE_GLYPH[next];
    buttonEl.title = STATE_LABELS[next];
    buttonEl.setAttribute('aria-label', STATE_LABELS[next]);
    buttonEl.dataset.state = next;
    // Path/privacy badge visible whenever a session is in progress.
    if (pathBadgeEl) pathBadgeEl.hidden = next === State.IDLE;
  }
}

function applyActiveSessionState() {
  const session = controlSession();
  setState(session.state, session);
  setPathBadge(session.pathLabel, session.pathTooltip);
}

// Mixing-board policy:
// - Output defaults to v0 behavior: with no explicit mute/solo choice, the
//   active agent is audible and background agents are muted.
// - A per-agent mute toggle is explicit state and wins over focus changes:
//   explicit unmute keeps a background agent audible; explicit mute keeps the
//   active agent silent. Solo is exclusive and temporarily mutes every other
//   output without erasing their mute choices.
// - Input is a single armed target. Focus changes never move it.
function applyActiveSessionPolicy() {
  for (const [agent, session] of sessionByAgent.entries()) {
    try { session.client?.setMuted?.(isOutputMuted(agent, session)); } catch (_) {}
    try { session.client?.setInputMuted?.(agent !== armedAgent); } catch (_) {}
  }
  refreshAllAgentVoiceCards();
}

function isOutputMuted(agent, session = sessionForAgent(agent)) {
  if (soloAgent !== NO_AGENT) return agent !== soloAgent;
  if (session.explicitMuted !== null) return !!session.explicitMuted;
  return agent !== currentAgentKey();
}


// ---------------------------------------------------------------------------
// Session control
// ---------------------------------------------------------------------------


async function toggleSession() {
  const session = controlSession();
  if (session.state === State.IDLE || session.state === State.ERROR) {
    await startSession(session);
  } else {
    await stopSession(session);
  }
}


// ---------------------------------------------------------------------------
// Chat-model selector ownership (#1371).
//
// When a Realtime voice session engages, the chat-LLM selector in the chat
// header stops being honest — the user's selected text model is bypassed,
// and the Realtime model (e.g. ``gpt-realtime-2``) is what actually produces
// every turn.  acquire/release flip the selector to display the Realtime
// model and lock it, then restore the user's prior selection on every exit
// path (clean close, fatal error, fallback-to-pipeline, page unload).
//
// Release is idempotent: multiple exit paths may fire for the same session
// (e.g. ``close()`` then SESSION_CLOSED then page-unload), and only the
// first non-noop call actually restores.
// ---------------------------------------------------------------------------


function _chatModelSelector() {
  // Exposed by chat.js loadModels().  Returns null when chat isn't initialized
  // (e.g. headless dev pages), in which case ownership becomes a noop.
  return (typeof window !== 'undefined' && window._sharedModelSelector) || null;
}


function acquireSelectorOwnership(realtimeModelId, session = activeSession()) {
  if (!isActiveSession(session)) return;
  const sel = _chatModelSelector();
  if (!sel || typeof sel.lockToVoiceModel !== 'function') return;
  // OpenAI is the only Realtime vendor today; if that ever changes the
  // session payload should carry the vendor so we can pass it through here.
  sel.lockToVoiceModel(
    { vendor: 'openai', model: realtimeModelId },
    '🎙 voice owns this — stop voice to change',
  );
  session.realtimeModel = realtimeModelId;
  session.ownsSelector = true;
}


function releaseSelectorOwnership(session = activeSession()) {
  if (!session.ownsSelector) return;
  const sel = _chatModelSelector();
  if (sel && typeof sel.unlockToPrior === 'function') {
    sel.unlockToPrior();
  }
  session.ownsSelector = false;
}


async function startSession(session = activeSession()) {
  const agent = session.agent;
  // Pin the initiating agent's settings NOW. settings() resolves against the
  // CURRENT host agent, and the route/mint work below has awaits — if the user
  // switches agents mid-startup, a bare settings() call would resolve against
  // the new agent and we'd mint agent A's session with agent B's voice/config.
  const pinnedSettings = settingsForAgent(agent);
  // Monotonic start token: lets late events from a superseded same-agent client
  // (notably an old client's async-teardown SESSION_CLOSED) be ignored so they
  // can't clobber a newer session. Bumped per start, captured in `onEvent`.
  session.startSeq = (session.startSeq || 0) + 1;
  const startSeq = session.startSeq;
  setState(State.CONNECTING, session);
  resetTurnState(session);
  if (session === controlSession()) setPathBadge('', '');
  session.pathLabel = '';
  session.pathTooltip = '';
  session.realtimeModel = '';
  setArmedAgent(agent);

  const onEvent = (ev) => handleClientEvent(agent, ev, startSeq);

  // Apply user picker overrides (mode + TTS) to drive routing.
  const overrides = pickerOverridesFromUI(pinnedSettings.mode || 'auto', pinnedSettings.preferred_tts || '');

  // If the user explicitly picked Pipeline, skip Realtime entirely. The
  // previous flow round-tripped to /voice/realtime/session, ate a 409,
  // logged a scary console error, then fell back. That round-trip is wasted
  // work and the 409 looks like a bug to anyone reading the network tab.
  // Mode selection is the user's contract — honor it.
  if (overrides.prefer_realtime !== false) {
    try {
      session.client = await createRealtimeClient({
        onEvent,
        // Rewrite to /api/agents/<host>/voice/realtime/session in multi_agent
        // mode; identity in standalone mode.
        endpoint: buildAgentUrlForAgent('/voice/realtime/session', agent),
        getAuthHeaders: voiceAuthHeaders,
        sessionRequestBody: {
          voice: pinnedSettings.voice || '',
          user_instructions: pinnedSettings.instructions || '',
          prefer_realtime: overrides.prefer_realtime,
          preferred_tts: overrides.preferred_tts || '',
        },
      });
      await session.client.start();
      applyActiveSessionPolicy();
      const realtimeModel = session.client.session?.model || 'gpt-realtime';
      session.realtimeModel = realtimeModel;
      // Take ownership of the chat-model selector for the lifetime of this
      // session.  Acquired AFTER ``start()`` so a failed mint doesn't strand
      // the selector locked.  See #1371.
      acquireSelectorOwnership(realtimeModel, session);
      const label = `Realtime · ${realtimeModel}`;
      const tooltip =
        `OpenAI Realtime: voice + reasoning answered by ${realtimeModel}, NOT your selected chat LLM. Switch to Pipeline in voice settings (right-click 🎙) to keep your chat LLM as the brain.`;
      session.pathLabel = label;
      session.pathTooltip = tooltip;
      if (session === controlSession()) setPathBadge(
        label,
        tooltip,
      );
      return;
    } catch (err) {
      if (err && err.code === 'REALTIME_UNAVAILABLE') {
        const fallback = err.fallback || {};
        const canUsePipeline =
          (fallback.path === 'pipeline' || fallback.path === 'local') &&
          fallback.fallback_tts &&
          fallback.fallback_stt;
        if (!canUsePipeline) {
          surfaceFatalError(new Error(fallback.reason || err.message || 'Voice unavailable'), session);
          return;
        }
        console.info('[voice/ui] Realtime declined, falling back to Pipeline:', fallback.reason);
        session.client = null;
      } else {
        surfaceFatalError(err, session);
        return;
      }
    }
  } else {
    console.info('[voice/ui] Pipeline mode forced by user picker; skipping Realtime.');
  }

  // Pipeline path.
  try {
    session.client = await createPipelineClient({
      onEvent,
      apiKey: API.getApiKey() || '',
      wsPath: buildAgentUrlForAgent('/voice/chat', agent),
      // Honor the picker's voice + provider choice. Without these the server
      // falls back to its config-file voice and the picker is decorative.
      voiceId: pinnedSettings.voice || '',
      preferredTts: (overrides && overrides.preferred_tts) || pinnedSettings.preferred_tts || '',
      // Pin STT to the browser's primary language tag (e.g. "en-US" → "en")
      // so Whisper doesn't hallucinate language switches mid-utterance.
      language: (navigator.language || 'en').split('-')[0],
    });
    await session.client.start();
    applyActiveSessionPolicy();
    session.pathLabel = 'Pipeline';
    session.pathTooltip = 'Cascaded STT → your LLM → TTS. Slower than Realtime, preserves your model choice.';
    if (session === controlSession()) setPathBadge(session.pathLabel, session.pathTooltip);
  } catch (err) {
    surfaceFatalError(err, session);
  }
}


async function stopSession(session = activeSession()) {
  const c = session.client;
  session.client = null;
  session.pathLabel = '';
  session.pathTooltip = '';
  const wasControlSession = session === controlSession();
  if (armedAgent === session.agent) setArmedAgent(NO_AGENT);
  setState(State.IDLE, session);
  if (wasControlSession) applyActiveSessionState();
  // Release the chat-model selector lock BEFORE awaiting close so the user
  // sees their text model restored immediately, even if the WebRTC teardown
  // hangs.  Release is idempotent: the SESSION_CLOSED handler below will be
  // a noop if this already restored.
  releaseSelectorOwnership(session);
  if (c) {
    try { await c.close(); } catch (_) {}
  }
}


function surfaceFatalError(err, session = activeSession()) {
  console.error('[voice/ui] fatal voice error:', err);
  setState(State.ERROR, session);
  // Surface as an agent message so the user sees it inline with the chat.
  addMessage('agent', `⚠ Voice error: ${formatVoiceError(err)}`, paneForSession(session).element);
  session.client = null;
  session.pathLabel = '';
  session.pathTooltip = '';
  const wasControlSession = session === controlSession();
  if (armedAgent === session.agent) setArmedAgent(NO_AGENT);
  if (wasControlSession) applyActiveSessionState();
  // Restore the chat-model selector — fatal errors take a different code
  // path than stopSession() but the lock still needs releasing or the user
  // is stranded on the Realtime model for the next text turn.  See #1371.
  releaseSelectorOwnership(session);
}


function formatVoiceError(err) {
  const message = err?.message || 'session failed';
  if (
    err?.name === 'NotAllowedError' ||
    err?.name === 'SecurityError' ||
    /permission denied|permission dismissed|not allowed|denied/i.test(message)
  ) {
    return 'Microphone permission denied. Allow microphone access for this site, then click the mic again. If this browser has no microphone permission control, open Kestrel in Chrome or Safari.';
  }
  if (/getUserMedia is not available/i.test(message)) {
    return 'Microphone capture is unavailable. Voice requires a browser with microphone support on localhost or HTTPS.';
  }
  return message;
}


// ---------------------------------------------------------------------------
// Event handling — voice turns render directly into the existing chat
// container so a voice session is a continuation of the same conversation,
// not a parallel one. The same handler drives both Realtime + Pipeline
// clients (they emit identical events from events.js).
// ---------------------------------------------------------------------------


function handleClientEvent(agent, ev, startSeq) {
  const session = sessionForAgent(agent);
  // Drop events from a superseded client: a newer startSession() on this agent
  // bumped `session.startSeq`, so a stale client's late events — especially an
  // async-teardown SESSION_CLOSED — must not clear the newer session's client,
  // path state, or selector lock. `startSeq` is always set by the start-bound
  // `onEvent`; the undefined guard is belt-and-suspenders for any other caller.
  if (startSeq !== undefined && session.startSeq !== startSeq) return;
  // Apply the pure state transition first so the mic-button visual updates
  // before any DOM mutation below.
  const nextState = nextStateForEvent(session.state, ev.kind, ev);
  if (nextState !== null) setState(nextState, session);

  switch (ev.kind) {
    // User-side transcript: reserve/update one user bubble per voice turn.
    // Realtime can start the agent answer before final transcription arrives;
    // the reserved bubble keeps the visual order as user -> agent.
    case Events.USER_TRANSCRIPT_DELTA:
      updateUserTurn(session, ev.text || '');
      break;
    case Events.USER_TRANSCRIPT_FINAL:
      finalizeUserTurn(session, ev.text || session.userTranscriptBuffer);
      break;

    // SPEAKING_STARTED marks the start of a NEW agent response (mapped
    // from `response.created` in realtime.js). Defensive turn boundary:
    // if a prior agent bubble is still open here, the previous turn's
    // RESPONSE_DONE / AGENT_TEXT_FINAL didn't fire (or fired late), so
    // close it now. Without this, AGENT_TEXT_DELTA for the new turn
    // appends to the old bubble forever.
    case Events.SPEAKING_STARTED:
      if (session.agentMsgDiv) finalizeAgentTurn(session, session.agentTextBuffer);
      ensureUserTurn(session);
      break;

    // LISTENING_STARTED (user begins speaking) is also an unambiguous
    // "current agent turn is over" signal. Same defensive close.
    case Events.LISTENING_STARTED:
      if (session.agentMsgDiv) finalizeAgentTurn(session, session.agentTextBuffer);
      if (session.userMsgDiv) finalizeUserTurn(session, session.userTranscriptBuffer);
      break;

    case Events.LISTENING_STOPPED:
      ensureUserTurn(session);
      break;

    // Agent reply streams into a single message bubble. AGENT_TEXT_DELTA
    // appends; AGENT_TEXT_FINAL / RESPONSE_DONE finalize and reset for
    // the next turn.
    case Events.AGENT_TEXT_DELTA:
      if (ev.text) {
        if (!session.agentMsgDiv) {
          session.agentMsgDiv = addMessageStreaming('agent', paneForSession(session).element);
          session.agentTextBuffer = '';
        }
        session.agentTextBuffer += ev.text;
        const contentDiv = session.agentMsgDiv.querySelector('.message-content');
        if (contentDiv) contentDiv.textContent = session.agentTextBuffer;
      }
      break;
    case Events.AGENT_TEXT_FINAL:
      finalizeAgentTurn(session, ev.text || session.agentTextBuffer);
      break;
    case Events.RESPONSE_DONE:
      if (session.userMsgDiv) finalizeUserTurn(session, session.userTranscriptBuffer);
      finalizeAgentTurn(session, session.agentTextBuffer);
      break;

    case Events.TOOL_CALL_REQUESTED:
      handleToolCall(session, ev).catch((err) => {
        console.error('[voice/ui] tool dispatch failed:', err);
      });
      break;

    case Events.SESSION_CLOSED:
      // Drop any in-flight agent message back into the chat so the user
      // sees what they got even if the session ended mid-response.
      if (session.userMsgDiv) finalizeUserTurn(session, session.userTranscriptBuffer);
      if (session.agentMsgDiv) finalizeAgentTurn(session, session.agentTextBuffer);
      session.client = null;
      session.pathLabel = '';
      session.pathTooltip = '';
      const wasControlSession = session === controlSession();
      if (armedAgent === session.agent) setArmedAgent(NO_AGENT);
      if (wasControlSession) applyActiveSessionState();
      // Belt-and-suspenders selector restore.  ``stopSession`` and
      // ``surfaceFatalError`` already release, but events like
      // ``data_channel_closed`` (network drop, server-side close) reach us
      // through here without going through either site.  Idempotent — see
      // #1371.
      releaseSelectorOwnership(session);
      break;

    case Events.ERROR:
      if (ev.fatal) {
        surfaceFatalError(new Error(ev.message), session);
      } else {
        // Non-fatal: log to console; don't pollute chat history.
        console.warn('[voice/ui]', ev.message);
      }
      break;

    // SESSION_READY / SPEAKING_STOPPED / THINKING_STARTED handled entirely by
    // the state-machine transition above.
    default:
      break;
  }
}


function finalizeAgentTurn(session, text) {
  const div = session.agentMsgDiv;
  const buf = text || '';
  session.agentMsgDiv = null;
  session.agentTextBuffer = '';
  if (!div) {
    if (buf.trim()) addMessage('agent', buf, paneForSession(session).element);
    return;
  }
  // Use the chat module's finalizer so markdown / code blocks / mermaid get
  // the same treatment as text-chat agent messages. Pass the full pane so
  // pane targeting AND deferred-mermaid-on-mount work for a detached pane,
  // but with includePaneArtifacts:false so this pane's text-chat thinking
  // bubbles / tool cards aren't prepended onto the voice bubble.
  finalizeStreamingMessage(div, buf, paneForSession(session), { includePaneArtifacts: false }).catch((err) =>
    console.error('[voice/ui] finalize failed:', err),
  );
}


function ensureUserTurn(session) {
  if (session.userMsgDiv) return;
  session.userMsgDiv = addMessageStreaming('user', paneForSession(session).element);
  session.userTranscriptBuffer = '';
  const contentDiv = session.userMsgDiv.querySelector('.message-content');
  if (contentDiv) contentDiv.textContent = 'Transcribing...';
}


function updateUserTurn(session, text) {
  ensureUserTurn(session);
  session.userTranscriptBuffer = text || session.userTranscriptBuffer;
  const contentDiv = session.userMsgDiv?.querySelector('.message-content');
  if (contentDiv && session.userTranscriptBuffer.trim()) {
    contentDiv.textContent = session.userTranscriptBuffer;
  }
}


function finalizeUserTurn(session, text) {
  const div = session.userMsgDiv;
  const buf = (text || '').trim();
  session.userMsgDiv = null;
  session.userTranscriptBuffer = '';
  if (!div) {
    if (buf) addMessage('user', buf, paneForSession(session).element);
    return;
  }

  const contentDiv = div.querySelector('.message-content');
  if (!buf) {
    div.remove();
    return;
  }
  if (contentDiv) {
    contentDiv.classList.remove('streaming');
    contentDiv.textContent = buf;
  }
}


function resetTurnState(session = activeSession()) {
  session.agentMsgDiv = null;
  session.agentTextBuffer = '';
  session.userMsgDiv = null;
  session.userTranscriptBuffer = '';
}


// ---------------------------------------------------------------------------
// Tool dispatch — when the Realtime model invokes a tool, POST to the
// backend tool-runner endpoint, then commit the result back over the data
// channel so the model can continue.
// ---------------------------------------------------------------------------


async function handleToolCall(session, ev) {
  // Capture the client at function entry. The per-agent session client can be
  // nulled by stopSession()/surfaceFatalError() during the await on the
  // tool-dispatch fetch; without this snapshot, commitToolResult below would
  // crash on a closed session.
  const sessionClient = session.client;
  if (!sessionClient || !sessionClient.session?.session_id) {
    console.warn('[voice/ui] tool call before session ready:', ev);
    return;
  }
  const sessionId = sessionClient.session.session_id;
  const url = buildAgentUrlForAgent(
    `/voice/realtime/tools/${encodeURIComponent(sessionId)}`,
    session.agent,
  );
  let body;
  try {
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...await voiceAuthHeaders() },
      body: JSON.stringify({
        call_id: ev.call_id,
        name: ev.name,
        arguments: ev.arguments,
      }),
    });
    body = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      body = { error: body.detail || `tool dispatch HTTP ${resp.status}` };
    }
  } catch (err) {
    body = { error: `tool dispatch threw: ${err.message}` };
  }

  // If the session ended while we were awaiting the dispatch, the model is
  // gone and there's nowhere to commit. Drop silently — the model already
  // closed.
  if (sessionClient !== session.client) {
    console.info('[voice/ui] session ended before tool result; dropping commit');
    return;
  }

  // Always commit SOMETHING when the session is alive — silence wedges the model.
  try {
    sessionClient.commitToolResult(ev.call_id, body.result ?? body);
  } catch (err) {
    console.error('[voice/ui] commitToolResult failed:', err);
  }
}


// ---------------------------------------------------------------------------
// Path badge
// ---------------------------------------------------------------------------


function setPathBadge(label, tooltip) {
  if (!pathBadgeEl) return;  // status indicator wasn't mountable
  pathBadgeEl.textContent = label || '';
  pathBadgeEl.title = tooltip || '';
  pathBadgeEl.dataset.path = (label || '').split(/\s+/)[0].toLowerCase();
  // Mirror the active voice model into the chat-header model selector area
  // so the user can see at a glance that voice is using a different model
  // than text chat. The chat model selector itself is unchanged (text chat
  // still uses the user's selection); we just ANNOTATE it.
  setModelSelectorVoiceAnnotation(label, tooltip);
}


let _voiceAnnotationEl = null;

function setModelSelectorVoiceAnnotation(label, tooltip) {
  // Find or create the annotation element. Anchored to the chat header, to
  // the right of the model selector group. Hidden when no label is set.
  const modelSelector = document.getElementById('model-selector');
  if (!modelSelector) return;
  if (!_voiceAnnotationEl) {
    _voiceAnnotationEl = document.createElement('span');
    _voiceAnnotationEl.id = 'voice-active-model-annotation';
    _voiceAnnotationEl.className = 'kestrel-voice-active-annotation';
    _voiceAnnotationEl.hidden = true;
    // Insert immediately after the model selector so it sits in the same
    // visual cluster as the chat-LLM controls.
    modelSelector.insertAdjacentElement('afterend', _voiceAnnotationEl);
  }
  if (label) {
    _voiceAnnotationEl.textContent = `🎙 ${label}`;
    _voiceAnnotationEl.title = tooltip || '';
    _voiceAnnotationEl.hidden = false;
  } else {
    _voiceAnnotationEl.textContent = '';
    _voiceAnnotationEl.hidden = true;
  }
}


// ---------------------------------------------------------------------------
// Voice picker modal
// ---------------------------------------------------------------------------


async function openPicker() {
  // Populate from /voice/voices + show a live preview of the resolved
  // route given the current overrides. Both fetches happen on every open
  // so a chat-LLM swap (or restart) is reflected immediately.
  const requestId = ++pickerRequestId;
  const voiceSel = document.getElementById('voice-picker-select');
  const ttsSel = document.getElementById('voice-picker-tts');
  const modeSel = document.getElementById('voice-picker-mode');
  const instructionsEl = document.getElementById('voice-picker-instructions');

  voiceSel.innerHTML = '<option value="">Loading voices...</option>';
  modeSel.value = settings().mode || 'auto';
  instructionsEl.value = settings().instructions || '';
  // Stash the textarea's seeded value (a) so the hydration .then()
  // below can distinguish "still untouched" (safe to repaint) from
  // "operator already edited" (must preserve), AND (b) so savePicker
  // knows whether the operator saw a non-empty value before clicking
  // Save with a blank textarea — that turns the otherwise-ambiguous
  // pre-hydration empty save into an explicit clear of a seen
  // value.  Codex round-4/5 catches on #1352.
  const instructionsSeededValue = instructionsEl.value;
  instructionsEl.dataset.seededValue = instructionsSeededValue;
  // Reset the TTS dropdown from THIS agent's settings.  The select
  // element is reused across picker opens, so without this reset
  // ``refreshRoutePreview()`` reads the prior agent's provider out of
  // the stale DOM value and resolves voice for the wrong agent.
  // Codex round-4 catch on #1347.
  ttsSel.value = settings().preferred_tts || '';

  // Show the modal before provider calls finish. A slow auth bootstrap,
  // rate-limit, or provider probe should render as a loading/error state in
  // the picker, not as an apparently frozen right-click.
  pickerModalEl.hidden = false;
  instructionsEl.focus();

  // Hydrate this agent's settings from the server's persisted
  // ``voice_config`` so first-open shows the agent's identity-set voice
  // instead of empty defaults.  Fire-and-forget so a slow / stalled
  // ``/voice/config`` doesn't keep the picker invisible — modal is
  // already rendered above.  When hydration lands:
  //   * re-apply ``voice`` and ``preferred_tts`` to the rendered fields,
  //   * if hydration CHANGED ``preferred_tts``, re-run the route
  //     preview + voice list refresh so the picker isn't showing the
  //     wrong provider's options/badge (codex round-2 catch on #1347).
  // Pin the agent-id we captured at open time — if the operator
  // switched host agents during the hydration fetch, this picker is
  // stale and we should NOT paint it with another agent's config.
  const pinnedAgent = currentAgentKey();
  hydrateSettingsFromServer()
    .then(({ changedTts } = { changedTts: false }) => {
      // Picker was closed / reopened, OR host agent switched —
      // earlier requestId is stale, don't paint.
      if (requestId !== pickerRequestId || pickerModalEl.hidden) return;
      if (currentAgentKey() !== pinnedAgent) return;
      const s = settings();
      if (s.voice) {
        for (const opt of voiceSel.options) {
          if (opt.value === s.voice) { voiceSel.value = s.voice; break; }
        }
      }
      if (s.preferred_tts) {
        for (const opt of ttsSel.options) {
          if (opt.value === s.preferred_tts) { ttsSel.value = s.preferred_tts; break; }
        }
      }
      // #1352 — repaint the instructions textarea after hydration so
      // the operator sees the agent's persisted ``voice_directive``
      // before they hit Save.  Without this, a fresh-browser open
      // shows a blank persona field (the textarea was seeded BEFORE
      // hydration ran), and Save would POST ``voice_directive: ""``
      // — silently clearing the identity directive.
      //
      // Only paint when the textarea is STILL at its seeded value —
      // checking against ``''`` alone would stomp a mid-edit clear
      // (operator deleted the persona while the GET was in flight).
      // Codex round-4 catch.
      if (instructionsEl.value === instructionsSeededValue && s.instructions) {
        instructionsEl.value = s.instructions;
      }
      if (changedTts) {
        // Provider changed — the route badge + voice catalog scoped to
        // the prior provider need a fresh resolve so the user sees the
        // right voices and the right "Realtime / Pipeline" path.
        refreshRoutePreview(++pickerRequestId);
      }
    })
    .catch(() => {});

  // Route preview + TTS provider list (both come from /voice/realtime/route).
  refreshRoutePreview(requestId);
  // Re-preview when the user toggles mode or TTS — gives instant feedback.
  modeSel.onchange = () => refreshRoutePreview(++pickerRequestId);
  ttsSel.onchange = () => refreshRoutePreview(++pickerRequestId);
}


async function refreshRoutePreview(requestId = pickerRequestId) {
  const previewEl = document.getElementById('voice-picker-route-preview');
  const ttsSel = document.getElementById('voice-picker-tts');
  const modeSel = document.getElementById('voice-picker-mode');
  const voiceSel = document.getElementById('voice-picker-select');
  const previousTts = ttsSel.value || settings().preferred_tts || '';
  previewEl.textContent = 'Resolving...';

  const overrides = pickerOverridesFromUI(modeSel.value, previousTts);
  let route;
  try {
    route = await fetchRoute(overrides);
  } catch (err) {
    if (requestId !== pickerRequestId || pickerModalEl.hidden) return;
    previewEl.textContent = `Route preview failed: ${err.message}`;
    await refreshVoiceList(voiceSel, '', requestId);
    return;
  }
  if (requestId !== pickerRequestId || pickerModalEl.hidden) return;

  // Render the human summary.
  const parts = [];
  if (route.path === 'realtime') {
    parts.push(`🟢 Realtime — model: ${route.voice_model || '(discovering)'}`);
  } else if (route.path === 'pipeline') {
    parts.push(`🟠 Pipeline — your chat LLM (${route.llm_vendor || 'unknown'}) → TTS: ${route.tts_provider || 'auto'} / STT: ${route.stt_provider || 'auto'}`);
  } else if (route.path === 'local') {
    parts.push(`🔒 Local — TTS: ${route.tts_provider} / STT: ${route.stt_provider}`);
  } else {
    parts.push(`⚠ Voice unavailable`);
  }
  if (route.reason) parts.push(`<small>${route.reason}</small>`);
  previewEl.innerHTML = parts.join('<br>');

  // Refresh the TTS dropdown from the discovered providers (preserve selection).
  // Disable the dropdown in Realtime mode — Realtime owns audio I/O end-to-end
  // (no separate TTS provider), so the user's TTS pick has no effect there.
  const installed = route.available_tts_providers || [];
  ttsSel.innerHTML = '<option value="">Auto (resolver picks)</option>';
  for (const name of installed) {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    if (name === previousTts) opt.selected = true;
    ttsSel.appendChild(opt);
  }
  ttsSel.disabled = (route.path === 'realtime');

  // Refresh the voice list scoped to the path's actual voice catalog.
  //   Realtime → conversation provider (e.g. openai_realtime; voices like
  //              Marin/Cedar/Alloy on the Realtime model).
  //   Pipeline → user-picked TTS, falling back to the resolver's choice.
  //   Local    → resolver-picked local TTS (Piper).
  // Without the path branch, `previousTts` ("elevenlabs") sticks across mode
  // flips and the user sees Sarah/Roger in the picker even after switching
  // to Realtime where those voices don't exist.
  let voiceListProvider;
  if (route.path === 'realtime') {
    voiceListProvider = route.conversation_provider || 'openai_realtime';
  } else {
    voiceListProvider = previousTts || route.tts_provider || '';
  }
  await refreshVoiceList(voiceSel, voiceListProvider, requestId);

  // Bug #2: live-update the chat-header annotation when the picker mode
  // changes, not only after a session successfully starts. Otherwise the
  // user toggles to "Force Realtime", sees the route preview update, but
  // the header still reads the prior (Pipeline) annotation.
  if (route.path === 'realtime') {
    setModelSelectorVoiceAnnotation(
      route.voice_model || 'gpt-realtime',
      `Voice will use ${route.voice_model || 'gpt-realtime'} (Realtime). Click 🎙 to start.`,
    );
  } else if (route.path === 'pipeline') {
    setModelSelectorVoiceAnnotation(
      `Pipeline · ${route.tts_provider || 'auto'}`,
      `Voice will use ${route.llm_vendor || 'your chat LLM'} for reasoning, ${route.tts_provider || 'auto-picked TTS'} for speech.`,
    );
  } else if (route.path === 'local') {
    setModelSelectorVoiceAnnotation(
      `Local · ${route.tts_provider || 'piper'}`,
      `Voice will run fully local (no cloud calls).`,
    );
  } else {
    setModelSelectorVoiceAnnotation('', '');
  }
}


async function refreshVoiceList(selectEl, providerName, requestId = pickerRequestId) {
  selectEl.innerHTML = '<option value="">Loading voices...</option>';
  try {
    const voices = await fetchVoices(providerName);
    if (requestId !== pickerRequestId || pickerModalEl.hidden) return;
    selectEl.innerHTML = '';
    if (voices.length === 0) {
      // Empty list: ask /voice/providers/status for the actual reason and
      // surface it inline. Without this, the user sees "No voices reported
      // by elevenlabs" with no way to tell it's an API-key permission
      // issue, package version mismatch, etc.
      const reason = await fetchProviderReason(providerName);
      if (requestId !== pickerRequestId || pickerModalEl.hidden) return;
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = reason || (providerName
        ? `No voices reported by ${providerName}`
        : 'No voices available — install a voice provider');
      opt.disabled = true;
      selectEl.appendChild(opt);
      return;
    }
    for (const v of voices) {
      const opt = document.createElement('option');
      opt.value = v.voice_id;
      // Suffix with provider name when "auto" so the user can see the
      // multi-provider mix; redundant when scoped to one.
      const providerLabel = providerName ? '' : ` · ${v.provider}`;
      opt.textContent = `${v.name} (${v.gender}, ${v.accent})${providerLabel}`;
      if (v.voice_id === settings().voice) opt.selected = true;
      selectEl.appendChild(opt);
    }
  } catch (err) {
    if (requestId !== pickerRequestId || pickerModalEl.hidden) return;
    selectEl.innerHTML = `<option value="">Failed to load voices: ${err.message}</option>`;
  }
}


function pickerOverridesFromUI(mode, preferredTts) {
  // Translate the mode dropdown into the resolver's prefer_realtime knob.
  // "auto" leaves prefer_realtime at the default (true) — the resolver
  // still falls back to Pipeline based on privacy + LLM vendor.
  const overrides = { preferred_tts: preferredTts || '' };
  if (mode === 'pipeline') overrides.prefer_realtime = false;
  else overrides.prefer_realtime = true;
  return overrides;
}


async function fetchRoute(overrides = {}) {
  const params = new URLSearchParams();
  if (overrides.prefer_realtime === false) params.set('prefer_realtime', 'false');
  if (overrides.preferred_tts) params.set('preferred_tts', overrides.preferred_tts);
  if (overrides.preferred_stt) params.set('preferred_stt', overrides.preferred_stt);
  const qs = params.toString();
  const url = API.buildAgentUrl(`/voice/realtime/route${qs ? `?${qs}` : ''}`);
  const resp = await fetch(url, { headers: await voiceAuthHeaders() });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json();
}


function closePicker() {
  pickerRequestId++;
  pickerModalEl.hidden = true;
}


function savePicker() {
  const voice = document.getElementById('voice-picker-select').value;
  const instructions = document.getElementById('voice-picker-instructions').value;
  const mode = document.getElementById('voice-picker-mode').value || 'auto';
  const preferredTts = document.getElementById('voice-picker-tts').value || '';
  // Per-agent storage: the cache for the current host agent gets the
  // new picker state in-place, then we persist to that agent's
  // localStorage slot.  Other agents' caches are untouched — switching
  // to a different agent later shows that agent's own settings.
  const agent = currentAgentKey();
  const next = { voice, instructions, mode, preferred_tts: preferredTts };
  _settingsByAgent.set(agent, next);
  saveSettings(agent, next);
  // Bump the save generation so any hydration that started before this
  // save returns and skips writing stale server data on top of the
  // operator's just-saved intent.  See #1352 codex round-2.
  _saveGenByAgent.set(agent, (_saveGenByAgent.get(agent) || 0) + 1);
  // #1352 — also persist the directive to the agent's IDENTITY so the
  // persona survives browser changes / operator changes / A2A peer
  // interactions.  ``instructions`` in the picker is the operator-facing
  // surface for ``voice_config.voice_directive``.  Persist VOICE and TTS
  // provider through the same write — they're identity-shaped too.
  // Fire-and-forget; failures shouldn't block the modal close (operator
  // explicitly hit Save, the localStorage write above already happened).
  //
  // Include ``voice_directive`` in the POST unless we're in the ONE
  // truly ambiguous case: textarea is empty AND it was seeded blank
  // AND hydration hasn't returned yet — in that combination we
  // genuinely can't tell whether the operator wants to clear the
  // persona or hasn't seen the persisted server value yet.  Omit
  // the field so the server preserves whatever's there.
  //
  // Other combinations are all safe to send — the operator has seen
  // SOMETHING (either the local-seeded value OR the server-hydrated
  // value) and the current textarea content reflects their intent:
  //   * non-empty current → operator typed/kept content; persist it
  //   * empty + seeded non-empty → operator EXPLICITLY cleared the
  //     value they saw; persist the clear.  Without this case,
  //     codex round-5 caught the scenario where a localStorage-
  //     seeded directive gets cleared, save fires pre-hydration,
  //     the field gets omitted, and the next hydration resurrects
  //     the value the operator just deleted.
  //   * empty + hydrated → operator cleared after seeing server
  //     value; persist the clear.
  //
  // ``tts_voice_id`` and ``tts_provider`` send the operator's
  // current selection unconditionally — including empty strings.
  // The current kestrel-feature-voice 0.4.0 endpoint preserves
  // empty (legacy ``or prev`` semantics for those fields) so
  // sending "" is a no-op today; once
  // KestrelSovereignAI/kestrel-feature-voice#6 ships explicit-clear
  // semantics for these fields too, sending "" will properly clear.
  // Forward-compatible.  Codex round-5 catch on #1352.
  const seeded = document.getElementById('voice-picker-instructions').dataset.seededValue || '';
  const operatorHasSeenSomething = (
    seeded !== '' || _serverHydrated.has(agent)
  );
  const payload = {
    tts_voice_id: voice,
    tts_provider: preferredTts,
  };
  if (instructions !== '' || operatorHasSeenSomething) {
    payload.voice_directive = instructions;
  }
  // Track the in-flight POST so hydration on a quick reopen awaits
  // the commit before issuing its GET — see ``_pendingSaveByAgent``.
  const inflight = persistConfigToIdentity(agent, payload)
    .catch(() => {})
    .finally(() => {
      // Only clear if THIS save is still the in-flight one — another
      // save fired during this one would have replaced the entry
      // already and we shouldn't trample its tracking.
      if (_pendingSaveByAgent.get(agent) === inflight) {
        _pendingSaveByAgent.delete(agent);
      }
    });
  _pendingSaveByAgent.set(agent, inflight);
  // If a session is active, push the new instructions immediately —
  // Realtime accepts session.update mid-call. Voice/path change requires a
  // new session (OpenAI Realtime can't hot-swap voice or model).
  const session = sessionForAgent(agent);
  if (session.client && instructions) {
    try { session.client.updateInstructions(instructions); } catch (_) {}
  }
  closePicker();
}


async function persistConfigToIdentity(pinnedAgent, payload) {
  // POST /voice/config with the agent's persistent voice fields so the
  // identity-side directive (and chosen voice/provider) survives across
  // browsers, operators, and A2A peer interactions.  See #1352.
  //
  // Pin the agent to the one we captured at savePicker-time — if the
  // operator switched host agents while the POST was in flight, the
  // ``API.buildAgentUrl`` prefix would route to the WRONG agent and
  // write into the wrong identity.  Skip the write if the pinned agent
  // no longer matches the active host.  (Operator can still re-save
  // on the right agent.)
  if (pinnedAgent !== currentAgentKey()) return;
  const url = API.buildAgentUrl('/voice/config');
  const headers = await voiceAuthHeaders();
  headers['Content-Type'] = 'application/json';
  await fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(payload),
  });
}


async function fetchVoices(providerName = '') {
  // /voice/voices already supports `?provider=<name>` to scope the list.
  // When providerName is empty, returns voices from all installed providers.
  const url = API.buildAgentUrl(
    `/voice/voices${providerName ? `?provider=${encodeURIComponent(providerName)}` : ''}`,
  );
  const resp = await fetch(url, { headers: await voiceAuthHeaders() });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  const body = await resp.json();
  return Array.isArray(body.voices) ? body.voices : [];
}


/**
 * Look up the diagnostic reason a TTS provider has no voices. Calls
 * /voice/providers/status and returns the install hint or raw error message
 * for the named provider, or null when no specific diagnostic is available.
 */
async function fetchProviderReason(providerName) {
  if (!providerName) return null;
  try {
    const url = API.buildAgentUrl('/voice/providers/status');
    const res = await fetch(url, { headers: await voiceAuthHeaders() });
    if (!res.ok) return null;
    const body = await res.json();
    const rows = (body && body.providers) || [];
    const target = providerName.toLowerCase();
    const match = rows.find(
      r => r.kind === 'tts'
        && (r.provider_name === providerName
            || (r.name || '').toLowerCase().includes(target))
    );
    if (!match) return null;
    return match.install_hint
      || match.voice_list_error
      || match.available_error
      || match.init_error
      || null;
  } catch (_) {
    return null;
  }
}


// ---------------------------------------------------------------------------
// Push-to-talk + Esc-to-stop
// ---------------------------------------------------------------------------


function bindGlobalShortcuts() {
  let spaceHeld = false;
  // The session push-to-talk started on. The user can switch agents while
  // holding Space, so keyup must stop the SAME session it started — not
  // whatever happens to be active on release — or A's session leaks.
  let pttSession = null;
  document.addEventListener('keydown', (ev) => {
    const session = controlSession();
    // Esc: stop active session from anywhere.
    if (ev.key === 'Escape' && session.state !== State.IDLE) {
      ev.preventDefault();
      stopSession(session);
      return;
    }
    // Space: only when focus isn't in a text input — avoids stealing
    // typing on the message input.
    if (ev.code === 'Space' && !isTypingTarget(ev.target) && !ev.repeat) {
      if (session.state === State.IDLE && !spaceHeld) {
        spaceHeld = true;
        pttSession = session;
        ev.preventDefault();
        startSession(session);
      }
    }
  });
  document.addEventListener('keyup', (ev) => {
    if (ev.code === 'Space' && spaceHeld) {
      spaceHeld = false;
      // Push-to-talk release stops the session it started, even if the user
      // switched agents while holding Space. Users who prefer a long-running
      // session can click the button instead.
      const target = pttSession;
      pttSession = null;
      if (target && target.state !== State.IDLE) stopSession(target);
    }
  });
}


function isTypingTarget(el) {
  if (!el) return false;
  const tag = el.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || el.isContentEditable;
}


// ---------------------------------------------------------------------------
// Settings persistence
// ---------------------------------------------------------------------------


function loadSettings(agentName) {
  const defaults = { voice: '', instructions: '', mode: 'auto', preferred_tts: '' };
  try {
    const raw = localStorage.getItem(settingsKeyForAgent(agentName));
    if (!raw) return defaults;
    const parsed = JSON.parse(raw);
    return {
      voice: typeof parsed.voice === 'string' ? parsed.voice : '',
      instructions: typeof parsed.instructions === 'string' ? parsed.instructions : '',
      // Legacy stored objects won't have `mode` / `preferred_tts` — fall
      // back to "auto" + "" so existing users don't get a broken picker.
      mode: typeof parsed.mode === 'string' ? parsed.mode : 'auto',
      preferred_tts: typeof parsed.preferred_tts === 'string' ? parsed.preferred_tts : '',
    };
  } catch (_) {
    return defaults;
  }
}


function saveSettings(agentName, value) {
  try {
    localStorage.setItem(settingsKeyForAgent(agentName), JSON.stringify(value));
  } catch (_) {
    // Quota or disabled storage — ignore; settings stay in-memory for the
    // current session.
  }
}


async function hydrateSettingsFromServer() {
  // First time the picker opens for an agent in this browser session,
  // pull the agent's persisted ``voice_config`` from the server and seed
  // the in-memory picker cache.  This makes the picker reflect the
  // agent's identity-set voice rather than showing empty defaults that
  // look like "no voice configured."
  //
  // CRITICAL: hydrated values land in the in-memory ``_settingsByAgent``
  // cache only — NEVER in localStorage.  localStorage holds operator
  // overrides; if we persisted server defaults there, a later server-
  // side change to ``voice_config`` would be silently shadowed by the
  // cached snapshot forever (the next hydration call sees non-empty
  // local fields and refuses to update them).  Only ``savePicker`` —
  // when the operator clicks Save — writes to localStorage.  On a fresh
  // page load, hydration runs again and picks up the current server
  // config.  Codex round-3 catch on #1347.
  //
  // Returns ``{ changedTts }`` so callers can re-run dependent
  // refreshes (route preview, voice list) only when the hydration
  // actually altered the TTS provider.
  //
  // The ``_serverHydrated`` flag is set ONLY after a successful read so
  // a transient first-open failure (401 during auth bootstrap, brief
  // network error) leaves the agent open to retry on the next picker
  // open instead of being stuck on defaults until a full page reload.
  // Codex round-3 catch on #1347.
  const agent = currentAgentKey();
  if (_serverHydrated.has(agent)) return { changedTts: false };
  // Drain ALL in-flight ``POST /voice/config`` saves before issuing
  // our GET.  A single ``await`` of the current entry wouldn't be
  // enough: if a second save fires while we're awaiting the first,
  // ``_pendingSaveByAgent`` gets replaced with the newer POST and the
  // GET would race that one.  Loop until the map entry settles to
  // ``undefined``, then proceed.  Best-effort: the POST's own
  // ``.catch`` swallows errors so awaits can't reject.
  // Codex round-6/7 catches on #1352.
  while (_pendingSaveByAgent.has(agent)) {
    const pendingSave = _pendingSaveByAgent.get(agent);
    try { await pendingSave; } catch (_) { /* swallowed above */ }
  }
  // Capture the save generation BEFORE the fetch.  If ``savePicker``
  // fires for this agent while the GET is in flight, ``_saveGenByAgent``
  // increments — the post-await check below skips applying the stale
  // server payload over the operator's just-saved state.  Particularly
  // protects "clear the directive" saves from being undone by a slow
  // first-open hydration GET.  Codex round-2 catch on #1352.
  const saveGenAtStart = _saveGenByAgent.get(agent) || 0;
  try {
    const url = API.buildAgentUrl('/voice/config');
    const resp = await fetch(url, { headers: await voiceAuthHeaders() });
    if (!resp.ok) return { changedTts: false };
    const cfg = await resp.json();
    // Operator saved during the fetch — their just-saved state is the
    // ground truth.  Skip applying the now-stale GET response, and
    // skip marking hydrated so the NEXT picker open will retry the
    // GET (which will then return the post-save server state cleanly).
    if ((_saveGenByAgent.get(agent) || 0) !== saveGenAtStart) {
      return { changedTts: false };
    }
    // Pin the read+write to the agent we captured BEFORE the await.
    // If the operator switched host agents while ``/voice/config`` was
    // in flight, ``currentAgentKey()`` now points at the new agent —
    // re-resolving ``settings()`` here would mutate the new agent's
    // entry with the OLD agent's config.  Codex round-2 catch on #1347.
    let s = _settingsByAgent.get(agent);
    if (!s) {
      s = loadSettings(agent);
      _settingsByAgent.set(agent, s);
    }
    let changedTts = false;
    if (!s.voice && typeof cfg.tts_voice_id === 'string' && cfg.tts_voice_id) {
      s.voice = cfg.tts_voice_id;
    }
    if (!s.preferred_tts && typeof cfg.tts_provider === 'string' && cfg.tts_provider) {
      s.preferred_tts = cfg.tts_provider;
      changedTts = true;
    }
    // #1352 — agent's persisted ``voice_directive`` IS the persona;
    // the picker's ``instructions`` textarea is the operator-facing
    // surface for it.  Hydrate when the local picker doesn't already
    // have an operator override.  Empty server values don't override
    // a non-empty local one — same rule as the other fields.
    if (!s.instructions
        && typeof cfg.voice_directive === 'string'
        && cfg.voice_directive) {
      s.instructions = cfg.voice_directive;
    }
    _serverHydrated.add(agent);
    return { changedTts };
  } catch (_) {
    // Hydration is best-effort — a network blip leaves the picker on the
    // localStorage / defaults path, which is still per-agent.  Don't
    // mark hydrated so the next open retries.
    return { changedTts: false };
  }
}


// ---------------------------------------------------------------------------
// CSS — kept inline to match the rest of static/* (no build step)
// ---------------------------------------------------------------------------


function injectStyles() {
  if (document.getElementById('kestrel-voice-ui-styles')) return;
  const style = document.createElement('style');
  style.id = 'kestrel-voice-ui-styles';
  style.textContent = `
    /* Mic button in the chat input row — sits between the textarea and Send. */
    .kestrel-voice-btn {
      background: transparent;
      border: 1px solid var(--border-color, #2d3748);
      color: var(--text-primary, #e5e7eb);
      cursor: pointer;
      font-size: 1.05rem;
      line-height: 1;
      padding: 0.45rem 0.6rem;
      border-radius: 6px;
      align-self: stretch;
      display: inline-flex;
      align-items: center;
    }
    .kestrel-voice-btn:hover { background: var(--bg-tertiary, #1f2937); }
    .kestrel-voice-btn[data-state="listening"] {
      background: var(--error-color, #ef4444);
      color: #fff;
      border-color: var(--error-color, #ef4444);
      animation: kestrel-voice-pulse 1.4s infinite;
    }
    .kestrel-voice-btn[data-state="speaking"] {
      background: var(--accent-color, #3b82f6);
      color: #fff;
      border-color: var(--accent-color, #3b82f6);
    }
    .kestrel-voice-btn[data-state="thinking"] { opacity: 0.85; }
    .kestrel-voice-btn[data-state="error"] {
      background: var(--error-color, #ef4444);
      color: #fff;
      border-color: var(--error-color, #ef4444);
    }
    @keyframes kestrel-voice-pulse {
      0%   { box-shadow: 0 0 0 0   rgba(239, 68, 68, 0.55); }
      70%  { box-shadow: 0 0 0 9px rgba(239, 68, 68, 0);    }
      100% { box-shadow: 0 0 0 0   rgba(239, 68, 68, 0);    }
    }

    /* Live route-preview block at the top of the voice picker — tells the
       user which model would actually answer if they started a session right
       now, given their current overrides. */
    .kestrel-voice-route-preview {
      background: var(--bg-tertiary, #1f2937);
      color: var(--text-secondary, #d1d5db);
      padding: 0.5rem 0.7rem;
      border-radius: 6px;
      font-size: 0.8rem;
      line-height: 1.35;
    }
    .kestrel-voice-route-preview small {
      color: var(--text-tertiary, #9ca3af);
      font-size: 0.7rem;
    }

    /* Annotation in the chat header next to the model selector — surfaces
       which model is actually answering when voice is on (which can be
       gpt-realtime instead of the user's selected text-chat model). */
    .kestrel-voice-active-annotation {
      background: #16a34a;
      color: #fff;
      padding: 0.15rem 0.55rem;
      margin-left: 0.5rem;
      border-radius: 999px;
      font-size: 0.72rem;
      font-weight: 600;
      cursor: help;
      white-space: nowrap;
    }
    .kestrel-voice-active-annotation[hidden] { display: none; }

    /* Path/privacy chips live in the input footer next to context-status. */
    .kestrel-voice-path-badge {
      background: var(--bg-tertiary, #1f2937);
      color: var(--text-secondary, #d1d5db);
      padding: 0.1rem 0.5rem;
      margin-right: 0.5rem;
      border-radius: 999px;
      font-size: 0.7rem;
      font-weight: 600;
      cursor: help;
    }
    .kestrel-voice-path-badge[data-path="realtime"] { background: #16a34a; color: #fff; }
    .kestrel-voice-path-badge[data-path="pipeline"] { background: #d97706; color: #fff; }
    .kestrel-voice-privacy-banner {
      background: #6b21a8;
      color: #fff;
      padding: 0.1rem 0.5rem;
      margin-right: 0.5rem;
      border-radius: 999px;
      font-size: 0.7rem;
    }

    .agent-voice-controls {
      display: grid;
      grid-template-columns: minmax(4.6rem, 1fr) repeat(3, 1.55rem);
      gap: 0.25rem;
      align-items: center;
      flex-shrink: 0;
    }
    .agent-voice-state {
      color: var(--text-tertiary, #9ca3af);
      font-size: 0.68rem;
      line-height: 1;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      text-align: right;
    }
    .agent-item.selected .agent-voice-state {
      color: rgba(255, 255, 255, 0.78);
    }
    .agent-item.agent-voice-live .agent-voice-state {
      color: #16a34a;
      font-weight: 700;
    }
    .agent-item.agent-voice-speaking .agent-voice-state {
      color: var(--accent-color, #3b82f6);
    }
    .agent-voice-control {
      width: 1.55rem;
      height: 1.55rem;
      border: 1px solid var(--border-color, #2d3748);
      border-radius: 6px;
      background: transparent;
      color: var(--text-secondary, #d1d5db);
      cursor: pointer;
      font-size: 0.78rem;
      line-height: 1;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 0;
    }
    .agent-voice-control:hover {
      background: var(--bg-tertiary, #1f2937);
      color: var(--text-primary, #e5e7eb);
    }
    .agent-voice-control.active {
      background: var(--accent-color, #3b82f6);
      border-color: var(--accent-color, #3b82f6);
      color: #fff;
    }
    .agent-voice-mute.active {
      background: var(--error-color, #ef4444);
      border-color: var(--error-color, #ef4444);
    }
    .agent-voice-arm {
      color: #ef4444;
    }
    .agent-voice-arm.active {
      background: #ef4444;
      border-color: #ef4444;
      color: #fff;
    }
    .agent-item.offline .agent-voice-controls {
      pointer-events: none;
      opacity: 0.5;
    }

    .kestrel-voice-modal {
      position: fixed; inset: 0;
      background: rgba(0, 0, 0, 0.55);
      display: flex; align-items: center; justify-content: center;
      z-index: 9999;
    }
    .kestrel-voice-modal[hidden] { display: none; }
    .kestrel-voice-modal-card {
      background: var(--bg-primary);
      color: var(--text-primary);
      padding: 1.25rem 1.5rem;
      border-radius: 8px;
      width: min(440px, 90vw);
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
      display: flex; flex-direction: column; gap: 0.75rem;
    }
    .kestrel-voice-modal-title { margin: 0; font-size: 1.1rem; }
    .kestrel-voice-field { display: flex; flex-direction: column; gap: 0.25rem; font-size: 0.85rem; }
    .kestrel-voice-field textarea, .kestrel-voice-field select {
      width: 100%;
      padding: 0.4rem 0.55rem;
      background: var(--bg-secondary);
      color: var(--text-primary);
      border: 1px solid var(--border-color);
      border-radius: 4px;
      font: inherit;
    }
    .kestrel-voice-hint { color: var(--text-tertiary); font-size: 0.75rem; margin: 0; }
    .kestrel-voice-modal-actions {
      display: flex; justify-content: flex-end; gap: 0.5rem; margin-top: 0.5rem;
    }
  `;
  document.head.appendChild(style);
}
