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
function voiceAuthHeaders() {
  const headers = {};
  const apiKey = typeof API.getApiKey === 'function' ? API.getApiKey() : '';
  if (apiKey) headers['X-API-Key'] = apiKey;
  return headers;
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
// Module-level handles to rendered DOM + active client
// ---------------------------------------------------------------------------

let buttonEl = null;
// `agentMsgDiv` holds the in-flight agent chat message during a voice turn so
// AGENT_TEXT_DELTA events can stream into it. Reset to null when finalized.
let agentMsgDiv = null;
let agentTextBuffer = '';
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
  mountStatusIndicator();
  mountPickerModal();
  setState(State.IDLE);

  // Push-to-talk: hold spacebar (when not in a text input) to start voice.
  bindGlobalShortcuts();
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
  // Path/privacy badge visible whenever a session is in progress.
  if (pathBadgeEl) pathBadgeEl.hidden = next === State.IDLE;
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
  resetTurnState();
  setPathBadge('', '');

  const onEvent = handleClientEvent;

  // Try Realtime first. The server-side resolver returns 409 with a
  // `fallback` payload when Realtime isn't legal under the current privacy
  // mode + LLM vendor — we use that to pick Pipeline cleanly.
  try {
    client = await createRealtimeClient({
      onEvent,
      // Rewrite to /api/agents/<host>/voice/realtime/session in rookery
      // mode; identity in standalone mode.
      endpoint: API.buildAgentUrl('/voice/realtime/session'),
      getAuthHeaders: voiceAuthHeaders,
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
      apiKey: API.getApiKey() || '',
      // Same rookery URL rewrite for the WebSocket route.
      wsPath: API.buildAgentUrl('/voice/chat'),
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
  // Surface as an agent message so the user sees it inline with the chat.
  addMessage('agent', `⚠ Voice error: ${err?.message || 'session failed'}`);
  client = null;
}


// ---------------------------------------------------------------------------
// Event handling — voice turns render directly into the existing chat
// container so a voice session is a continuation of the same conversation,
// not a parallel one. The same handler drives both Realtime + Pipeline
// clients (they emit identical events from events.js).
// ---------------------------------------------------------------------------


function handleClientEvent(ev) {
  // Apply the pure state transition first so the mic-button visual updates
  // before any DOM mutation below.
  const nextState = nextStateForEvent(currentState, ev.kind, ev);
  if (nextState !== null) setState(nextState);

  switch (ev.kind) {
    // User-side transcript: only the FINAL adds a chat message. Live
    // partials would create N nested user bubbles; chat history wants one.
    case Events.USER_TRANSCRIPT_FINAL:
      if (ev.text && ev.text.trim()) {
        addMessage('user', ev.text);
      }
      break;

    // Agent reply streams into a single message bubble. AGENT_TEXT_DELTA
    // appends; AGENT_TEXT_FINAL / RESPONSE_DONE finalize and reset for
    // the next turn.
    case Events.AGENT_TEXT_DELTA:
      if (ev.text) {
        if (!agentMsgDiv) {
          agentMsgDiv = addMessageStreaming('agent');
          agentTextBuffer = '';
        }
        agentTextBuffer += ev.text;
        const contentDiv = agentMsgDiv.querySelector('.message-content');
        if (contentDiv) contentDiv.textContent = agentTextBuffer;
      }
      break;
    case Events.AGENT_TEXT_FINAL:
      finalizeAgentTurn(ev.text || agentTextBuffer);
      break;
    case Events.RESPONSE_DONE:
      finalizeAgentTurn(agentTextBuffer);
      break;

    case Events.TOOL_CALL_REQUESTED:
      handleToolCall(ev).catch((err) => {
        console.error('[voice/ui] tool dispatch failed:', err);
      });
      break;

    case Events.SESSION_CLOSED:
      // Drop any in-flight agent message back into the chat so the user
      // sees what they got even if the session ended mid-response.
      if (agentMsgDiv) finalizeAgentTurn(agentTextBuffer);
      break;

    case Events.ERROR:
      if (ev.fatal) {
        surfaceFatalError(new Error(ev.message));
      } else {
        // Non-fatal: log to console; don't pollute chat history.
        console.warn('[voice/ui]', ev.message);
      }
      break;

    // SESSION_READY / LISTENING_* / SPEAKING_* / THINKING_STARTED handled
    // entirely by the state-machine transition above — no chat side-effect.
    default:
      break;
  }
}


function finalizeAgentTurn(text) {
  const div = agentMsgDiv;
  const buf = text || '';
  agentMsgDiv = null;
  agentTextBuffer = '';
  if (!div) {
    if (buf.trim()) addMessage('agent', buf);
    return;
  }
  // Use the chat module's finalizer so markdown / code blocks / mermaid
  // get the same treatment as text-chat agent messages.
  finalizeStreamingMessage(div, buf).catch((err) =>
    console.error('[voice/ui] finalize failed:', err),
  );
}


function resetTurnState() {
  agentMsgDiv = null;
  agentTextBuffer = '';
}


// ---------------------------------------------------------------------------
// Tool dispatch — when the Realtime model invokes a tool, POST to the
// backend tool-runner endpoint, then commit the result back over the data
// channel so the model can continue.
// ---------------------------------------------------------------------------


async function handleToolCall(ev) {
  if (!client || !client.session?.session_id) {
    console.warn('[voice/ui] tool call before session ready:', ev);
    return;
  }
  const sessionId = client.session.session_id;
  const url = API.buildAgentUrl(`/voice/realtime/tools/${encodeURIComponent(sessionId)}`);
  let body;
  try {
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...voiceAuthHeaders() },
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
  // Always commit SOMETHING — silence on a tool call wedges the model.
  try {
    client.commitToolResult(ev.call_id, body.result ?? body);
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
  pathBadgeEl.dataset.path = (label || '').toLowerCase();
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
  // provider's discovered voice list filtered by privacy mode. Auth + URL
  // rewriting via the global API client so the request behaves the same
  // as every other Kestrel HTTP call.
  const resp = await fetch(API.buildAgentUrl('/voice/voices'), {
    headers: voiceAuthHeaders(),
  });
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
