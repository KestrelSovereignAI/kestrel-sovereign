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

import { Events } from './events.js';
import { createRealtimeClient } from './realtime.js';
import { createPipelineClient } from './pipeline.js';
import { State, nextStateForEvent } from './state-machine.js';

// State.* + nextStateForEvent are imported from state-machine.js so the
// pure transition logic stays Node-testable.

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
// Module-level handles to rendered DOM + active client
// ---------------------------------------------------------------------------

let buttonEl = null;
let drawerEl = null;
let transcriptEl = null;
let pathBadgeEl = null;
let pickerModalEl = null;
let privacyBannerEl = null;

let client = null;       // active createRealtimeClient or createPipelineClient
let currentState = State.IDLE;

// User-overridable session settings, persisted to localStorage so they survive
// page reloads. Voice is picked from the provider's list; instructions is the
// free-form steering directive forwarded to gpt-4o-mini-tts / Realtime
// session.instructions.
const SETTINGS_KEY = 'kestrel.voice.settings';
let settings = loadSettings();


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
  mountDrawer();
  mountPickerModal();
  setState(State.IDLE);

  // Push-to-talk: hold spacebar (when not in a text input) to start voice.
  bindGlobalShortcuts();
}


// ---------------------------------------------------------------------------
// Mounting
// ---------------------------------------------------------------------------


function mountButton(header) {
  // Place the button in the header's left-hand button group so it sits next
  // to History + New Chat. If we can't find that group, fall back to header
  // append so the button is at least reachable.
  const leftGroup = header.querySelector('div[style*="display: flex"]') || header;
  buttonEl = document.createElement('button');
  buttonEl.id = 'voice-toggle-btn';
  buttonEl.className = 'btn btn-secondary kestrel-voice-btn';
  buttonEl.title = STATE_LABELS[State.IDLE];
  buttonEl.setAttribute('aria-label', STATE_LABELS[State.IDLE]);
  buttonEl.setAttribute('aria-live', 'polite');
  buttonEl.style.padding = '0.4rem 0.6rem';
  buttonEl.style.fontSize = '0.85rem';
  buttonEl.textContent = STATE_GLYPH[State.IDLE];
  buttonEl.addEventListener('click', toggleSession);
  // Right-click opens the voice picker so power users can reach voice
  // settings without a separate UI affordance.
  buttonEl.addEventListener('contextmenu', (ev) => {
    ev.preventDefault();
    openPicker();
  });
  leftGroup.appendChild(buttonEl);
}


function mountDrawer() {
  // Drawer holds: path badge, privacy banner, transcript. Slides in below
  // the chat header during a session.
  drawerEl = document.createElement('div');
  drawerEl.id = 'voice-drawer';
  drawerEl.className = 'kestrel-voice-drawer';
  drawerEl.hidden = true;

  pathBadgeEl = document.createElement('span');
  pathBadgeEl.className = 'kestrel-voice-path-badge';
  pathBadgeEl.textContent = '';

  privacyBannerEl = document.createElement('span');
  privacyBannerEl.className = 'kestrel-voice-privacy-banner';
  privacyBannerEl.hidden = true;

  const headerRow = document.createElement('div');
  headerRow.className = 'kestrel-voice-drawer-header';
  headerRow.appendChild(pathBadgeEl);
  headerRow.appendChild(privacyBannerEl);

  const settingsBtn = document.createElement('button');
  settingsBtn.type = 'button';
  settingsBtn.className = 'kestrel-voice-icon-btn';
  settingsBtn.title = 'Voice settings';
  settingsBtn.setAttribute('aria-label', 'Open voice settings');
  settingsBtn.textContent = '⚙';
  settingsBtn.addEventListener('click', openPicker);
  headerRow.appendChild(settingsBtn);

  transcriptEl = document.createElement('div');
  transcriptEl.className = 'kestrel-voice-transcript';
  transcriptEl.setAttribute('aria-live', 'polite');

  drawerEl.appendChild(headerRow);
  drawerEl.appendChild(transcriptEl);

  // Insert under the chat header.
  const chatHeader = document.querySelector('.chat-header');
  chatHeader?.insertAdjacentElement('afterend', drawerEl);
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


function setState(next) {
  if (next === currentState) return;
  currentState = next;
  buttonEl.textContent = STATE_GLYPH[next];
  buttonEl.title = STATE_LABELS[next];
  buttonEl.setAttribute('aria-label', STATE_LABELS[next]);
  buttonEl.dataset.state = next;
  // Toggle drawer visibility based on session lifecycle: visible whenever
  // a session is in progress, hidden in idle.
  drawerEl.hidden = next === State.IDLE;
}


// ---------------------------------------------------------------------------
// Session control
// ---------------------------------------------------------------------------


async function toggleSession() {
  if (currentState === State.IDLE || currentState === State.ERROR) {
    await startSession();
  } else {
    await stopSession();
  }
}


async function startSession() {
  setState(State.CONNECTING);
  clearTranscript();
  setPathBadge('', '');

  const onEvent = handleClientEvent;

  // Try Realtime first. The server-side resolver returns 409 with a
  // `fallback` payload when Realtime isn't legal under the current privacy
  // mode + LLM vendor — we use that to pick Pipeline cleanly.
  try {
    client = await createRealtimeClient({
      onEvent,
      sessionRequestBody: {
        voice: settings.voice || '',
        user_instructions: settings.instructions || '',
      },
    });
    await client.start();
    setPathBadge('Realtime', 'OpenAI Realtime: low-latency speech-to-speech.');
    return;
  } catch (err) {
    if (err && err.code === 'REALTIME_UNAVAILABLE') {
      console.info('[voice/ui] Realtime declined, falling back to Pipeline:', err.fallback?.reason);
      client = null;
    } else {
      surfaceFatalError(err);
      return;
    }
  }

  // Fallback path.
  try {
    client = await createPipelineClient({
      onEvent,
      apiKey: getApiKey(),
    });
    await client.start();
    setPathBadge('Pipeline', 'Cascaded STT → your LLM → TTS. Slower than Realtime, preserves your model choice.');
  } catch (err) {
    surfaceFatalError(err);
  }
}


async function stopSession() {
  const c = client;
  client = null;
  setState(State.IDLE);
  if (c) {
    try { await c.close(); } catch (_) {}
  }
}


function surfaceFatalError(err) {
  console.error('[voice/ui] fatal voice error:', err);
  setState(State.ERROR);
  appendTranscriptLine('error', err?.message || 'Voice session failed.');
  client = null;
}


// ---------------------------------------------------------------------------
// Event handling — the same handler drives both Realtime + Pipeline clients
// because they emit identical events (events.js).
// ---------------------------------------------------------------------------


function handleClientEvent(ev) {
  // Apply the pure state transition first so the mic-button visual updates
  // before any DOM mutation below. ERROR.fatal escalates to ERROR state via
  // the surfaceFatalError path so we get the right transcript line + reset.
  const nextState = nextStateForEvent(currentState, ev.kind, ev);
  if (nextState !== null) setState(nextState);

  // Side-effects: transcript rendering + error surfacing.
  switch (ev.kind) {
    case Events.USER_TRANSCRIPT_DELTA:
      appendTranscriptDelta('user', ev.text);
      break;
    case Events.USER_TRANSCRIPT_FINAL:
      finalizeTranscriptLine('user', ev.text);
      break;
    case Events.AGENT_TEXT_DELTA:
      appendTranscriptDelta('agent', ev.text);
      break;
    case Events.AGENT_TEXT_FINAL:
      finalizeTranscriptLine('agent', ev.text);
      break;

    case Events.TOOL_CALL_REQUESTED:
      // Surface for visibility; tool dispatch into the agent registry is a
      // follow-up wiring ticket.
      appendTranscriptLine('tool', `Tool call: ${ev.name}(${JSON.stringify(ev.arguments)})`);
      break;

    case Events.ERROR:
      if (ev.fatal) {
        surfaceFatalError(new Error(ev.message));
      } else {
        appendTranscriptLine('error', ev.message);
      }
      break;

    // SESSION_READY / SESSION_CLOSED / LISTENING_* / SPEAKING_* /
    // THINKING_STARTED / RESPONSE_DONE are handled entirely by the state
    // transition above — no transcript side-effect.
    default:
      break;
  }
}


// ---------------------------------------------------------------------------
// Transcript rendering — line per turn, deltas append to the current line
// ---------------------------------------------------------------------------


function clearTranscript() {
  transcriptEl.innerHTML = '';
}


function ensureCurrentLine(role) {
  let last = transcriptEl.lastElementChild;
  if (!last || last.dataset.role !== role || last.dataset.final === 'true') {
    last = document.createElement('div');
    last.className = `kestrel-voice-line kestrel-voice-line--${role}`;
    last.dataset.role = role;
    last.dataset.final = 'false';
    last.textContent = '';
    transcriptEl.appendChild(last);
  }
  return last;
}


function appendTranscriptDelta(role, text) {
  if (!text) return;
  const line = ensureCurrentLine(role);
  line.textContent += text;
  scrollTranscriptToBottom();
}


function finalizeTranscriptLine(role, text) {
  const line = ensureCurrentLine(role);
  if (text) line.textContent = text;  // overwrite with the canonical final text
  line.dataset.final = 'true';
  scrollTranscriptToBottom();
}


function appendTranscriptLine(role, text) {
  const line = document.createElement('div');
  line.className = `kestrel-voice-line kestrel-voice-line--${role}`;
  line.dataset.role = role;
  line.dataset.final = 'true';
  line.textContent = text;
  transcriptEl.appendChild(line);
  scrollTranscriptToBottom();
}


function scrollTranscriptToBottom() {
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}


// ---------------------------------------------------------------------------
// Path badge
// ---------------------------------------------------------------------------


function setPathBadge(label, tooltip) {
  pathBadgeEl.textContent = label || '';
  pathBadgeEl.title = tooltip || '';
  pathBadgeEl.dataset.path = label.toLowerCase();
}


// ---------------------------------------------------------------------------
// Voice picker modal
// ---------------------------------------------------------------------------


async function openPicker() {
  // Populate the voice select lazily — first time we need it, fetch from
  // the active provider's discovered list. If no session is open yet we
  // use the default backend voice list.
  const sel = document.getElementById('voice-picker-select');
  sel.innerHTML = '<option value="">Loading...</option>';

  try {
    const voices = await fetchVoices();
    sel.innerHTML = '';
    if (voices.length === 0) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'No voices available — install a voice provider';
      sel.appendChild(opt);
    } else {
      for (const v of voices) {
        const opt = document.createElement('option');
        opt.value = v.voice_id;
        opt.textContent = `${v.name} (${v.gender}, ${v.accent})`;
        if (v.voice_id === settings.voice) opt.selected = true;
        sel.appendChild(opt);
      }
    }
  } catch (err) {
    sel.innerHTML = `<option value="">Failed to load voices: ${err.message}</option>`;
  }

  document.getElementById('voice-picker-instructions').value = settings.instructions || '';
  pickerModalEl.hidden = false;
  document.getElementById('voice-picker-instructions').focus();
}


function closePicker() {
  pickerModalEl.hidden = true;
}


function savePicker() {
  const voice = document.getElementById('voice-picker-select').value;
  const instructions = document.getElementById('voice-picker-instructions').value;
  settings = { voice, instructions };
  saveSettings();
  // If a session is active, push the new instructions immediately —
  // Realtime accepts session.update mid-call. Voice change requires a new
  // session (OpenAI Realtime can't hot-swap voices).
  if (client && instructions) {
    try { client.updateInstructions(instructions); } catch (_) {}
  }
  closePicker();
}


async function fetchVoices() {
  // /voice/voices is the existing endpoint that returns the active
  // provider's discovered voice list filtered by privacy mode.
  const resp = await fetch('/voice/voices');
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  const body = await resp.json();
  return Array.isArray(body.voices) ? body.voices : [];
}


// ---------------------------------------------------------------------------
// Push-to-talk + Esc-to-stop
// ---------------------------------------------------------------------------


function bindGlobalShortcuts() {
  let spaceHeld = false;
  document.addEventListener('keydown', (ev) => {
    // Esc: stop active session from anywhere.
    if (ev.key === 'Escape' && currentState !== State.IDLE) {
      ev.preventDefault();
      stopSession();
      return;
    }
    // Space: only when focus isn't in a text input — avoids stealing
    // typing on the message input.
    if (ev.code === 'Space' && !isTypingTarget(ev.target) && !ev.repeat) {
      if (currentState === State.IDLE && !spaceHeld) {
        spaceHeld = true;
        ev.preventDefault();
        startSession();
      }
    }
  });
  document.addEventListener('keyup', (ev) => {
    if (ev.code === 'Space' && spaceHeld) {
      spaceHeld = false;
      // Push-to-talk release stops the session. Users who prefer a
      // long-running session can click the button instead.
      if (currentState !== State.IDLE) stopSession();
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


function loadSettings() {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) return { voice: '', instructions: '' };
    const parsed = JSON.parse(raw);
    return {
      voice: typeof parsed.voice === 'string' ? parsed.voice : '',
      instructions: typeof parsed.instructions === 'string' ? parsed.instructions : '',
    };
  } catch (_) {
    return { voice: '', instructions: '' };
  }
}


function saveSettings() {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch (_) {
    // Quota or disabled storage — ignore; settings stay in-memory for the
    // current session.
  }
}


function getApiKey() {
  // The chat already authenticates via cookie/header — this helper only
  // exists for the Pipeline WebSocket which expects ?api_key=. When no
  // key is configured (open-access servers), pass empty.
  return localStorage.getItem('kestrel.apiKey') || '';
}


// ---------------------------------------------------------------------------
// CSS — kept inline to match the rest of static/* (no build step)
// ---------------------------------------------------------------------------


function injectStyles() {
  if (document.getElementById('kestrel-voice-ui-styles')) return;
  const style = document.createElement('style');
  style.id = 'kestrel-voice-ui-styles';
  style.textContent = `
    .kestrel-voice-btn { font-size: 1rem; line-height: 1; }
    .kestrel-voice-btn[data-state="listening"] {
      box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.6);
      animation: kestrel-voice-pulse 1.4s infinite;
    }
    .kestrel-voice-btn[data-state="speaking"] {
      background: var(--accent-color, #3b82f6);
      color: #fff;
    }
    .kestrel-voice-btn[data-state="thinking"] { opacity: 0.85; }
    .kestrel-voice-btn[data-state="error"] {
      background: var(--error-color, #ef4444);
      color: #fff;
    }
    @keyframes kestrel-voice-pulse {
      0%   { box-shadow: 0 0 0 0   rgba(239, 68, 68, 0.55); }
      70%  { box-shadow: 0 0 0 9px rgba(239, 68, 68, 0);    }
      100% { box-shadow: 0 0 0 0   rgba(239, 68, 68, 0);    }
    }

    .kestrel-voice-drawer {
      border-bottom: 1px solid var(--border-color);
      background: var(--bg-secondary);
      padding: 0.5rem 1rem;
      max-height: 220px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }
    .kestrel-voice-drawer[hidden] { display: none; }
    .kestrel-voice-drawer-header {
      display: flex;
      align-items: center;
      gap: 0.6rem;
      font-size: 0.8rem;
    }
    .kestrel-voice-path-badge {
      background: var(--bg-tertiary, #1f2937);
      color: var(--text-secondary, #d1d5db);
      padding: 0.15rem 0.55rem;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 600;
      cursor: help;
    }
    .kestrel-voice-path-badge[data-path="realtime"] { background: #16a34a; color: #fff; }
    .kestrel-voice-path-badge[data-path="pipeline"] { background: #d97706; color: #fff; }
    .kestrel-voice-privacy-banner {
      background: #6b21a8;
      color: #fff;
      padding: 0.15rem 0.55rem;
      border-radius: 999px;
      font-size: 0.7rem;
    }
    .kestrel-voice-icon-btn {
      margin-left: auto;
      background: none;
      border: none;
      color: var(--text-secondary);
      cursor: pointer;
      font-size: 1rem;
    }
    .kestrel-voice-transcript {
      flex: 1;
      overflow-y: auto;
      font-size: 0.85rem;
      line-height: 1.4;
      padding-right: 0.4rem;
    }
    .kestrel-voice-line {
      padding: 0.15rem 0;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .kestrel-voice-line--user::before { content: 'you · '; opacity: 0.6; font-weight: 600; }
    .kestrel-voice-line--agent::before { content: 'agent · '; opacity: 0.6; font-weight: 600; }
    .kestrel-voice-line--tool { color: var(--text-tertiary); font-style: italic; }
    .kestrel-voice-line--tool::before { content: '⚙ '; }
    .kestrel-voice-line--error { color: var(--error-color, #ef4444); }
    .kestrel-voice-line--error::before { content: '⚠ '; }

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
