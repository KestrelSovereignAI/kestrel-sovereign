/**
 * Kestrel Sovereign Console - Chat Component
 * Chat panel and command autocomplete
 */

import API from './api.js';
import { state, AGENT_COMMANDS, Toast, getOrCreateChatPane, escapeHtml } from './ui.js';

// Shared markdown utilities - loaded via script tag before this module
const {
    renderMarkdown,
    renderStreamingMarkdown,
    highlightCodeBlocks,
    renderMermaidDiagrams,
    finalizeMarkdown
} = window.SharedMarkdown;

// Wave 5E in-band revising sentinel — pairs with kestrel_sovereign/
// agent/streaming.py:_build_revise_sentinel. Format:
//   \x1eKESTREL:REVISE:<json>\x1e
// Strictly ordered with the chunks on /api/agent/stream so it can't
// race post-tool synthesis the way the parallel SSE channel can.
// Both signals stay wired (Wave 5C SSE is the reliability backup);
// pendingRevise is the single shared state — whichever signal hits
// first sets it; the other becomes a no-op.
const REVISE_SENTINEL_PREFIX = '\x1eKESTREL:REVISE:';
const REVISE_SENTINEL_SUFFIX = '\x1e';
const THINKING_SENTINEL_PREFIX = '\x1eKESTREL:THINK:';
const THINKING_SENTINEL_SUFFIX = '\x1e';
const STREAM_SENTINELS = [
    { kind: 'revise', prefix: REVISE_SENTINEL_PREFIX, suffix: REVISE_SENTINEL_SUFFIX },
    { kind: 'thinking', prefix: THINKING_SENTINEL_PREFIX, suffix: THINKING_SENTINEL_SUFFIX },
];

function renderToolActivityLineHtml(line) {
    const escaped = escapeHtml(line);
    if (line.startsWith('\u{1F527}')) return `<div class="tool-activity tool-start">${escaped}</div>`;
    if (line.startsWith('\u2713')) return `<div class="tool-activity tool-done">${escaped}</div>`;
    if (line.startsWith('\u274C')) return `<div class="tool-activity tool-error">${escaped}</div>`;
    return `<div class="tool-activity">${escaped}</div>`;
}

function parseToolActivityLine(line) {
    const text = String(line || '').trim();
    const startMatch = text.match(/^\u{1F527}\s*(?:Calling\s+)?(.+?)(?:\.\.\.)?$/u);
    const doneMatch = text.match(/^\u2713\s*(.+?)\s+(complete|done)(?:\s+\((.+)\))?$/u);
    const errorMatch = text.match(/^\u274C\s*(.+?)\s+failed(?::\s*(.*))?$/u);
    if (startMatch) return { kind: 'start', name: startMatch[1], line: text };
    if (doneMatch) return { kind: 'done', name: doneMatch[1], detail: doneMatch[3] || '', line: text };
    if (errorMatch) return { kind: 'error', name: errorMatch[1], detail: errorMatch[2] || '', line: text };
    return { kind: 'note', name: '', line: text };
}

function groupToolActivity(lines) {
    const groups = [];
    for (const line of lines) {
        const parsed = parseToolActivityLine(line);
        if (parsed.kind === 'start') {
            groups.push({ name: parsed.name, status: 'running', detail: '', lines: [line] });
            continue;
        }

        const target = [...groups].reverse().find((group) => group.name === parsed.name && group.status === 'running');
        if (target && (parsed.kind === 'done' || parsed.kind === 'error')) {
            target.status = parsed.kind === 'done' ? 'complete' : 'error';
            target.detail = parsed.detail;
            target.lines.push(line);
            continue;
        }

        groups.push({
            name: parsed.name || line,
            status: parsed.kind === 'error' ? 'error' : parsed.kind === 'done' ? 'complete' : 'note',
            detail: parsed.detail || '',
            lines: [line],
        });
    }
    return groups;
}

function isToolActivityLine(line) {
    const text = String(line || '').trim();
    return (
        isToolActivityStartLine(text) ||
        /^\u2713\s+.+\s+(?:complete|done)(?:\s+\(.+\))?$/u.test(text) ||
        /^\u274C\s+.+\s+failed(?::.*)?$/u.test(text)
    );
}

function isToolActivityStartLine(line) {
    return /^\u{1F527}\s+Calling\s+.+(?:\.\.\.)?$/u.test(String(line || '').trim());
}

// Server emitters yield tool markers with only a TRAILING newline
// (orchestrator_engine.py and codex_adapter.py). The LLM's preceding
// text chunk is not guaranteed to end with `\n`, so the accumulated
// buffer often looks like `"...so we get a grounded success or
// failure.🔧 Calling list_peers..."` — a single glued line that the
// anchored `^🔧` matchers below can't see. Normalize here so the
// downstream line-prefix logic recognizes the section boundary
// regardless of upstream newline discipline. Gated on the presence
// of a 🔧-Calling marker so ordinary assistant prose containing
// phrases like "Done: ✓ migration complete" is left untouched.
const TOOL_MARKER_PRESENCE_PATTERN = /\u{1F527}\s+Calling\s+/u;
const TOOL_MARKER_PREFIX_PATTERN = /([^\n])(\u{1F527}\s+Calling\s+|✓\s+\S[^\n]*?\s+(?:complete|done)\b|❌\s+\S[^\n]*?\s+failed\b)/gu;
function normalizeToolMarkerLineBreaks(text) {
    if (!TOOL_MARKER_PRESENCE_PATTERN.test(text)) return text;
    return text.replace(TOOL_MARKER_PREFIX_PATTERN, '$1\n$2');
}

export function splitToolActivity(content) {
    const text = normalizeToolMarkerLineBreaks(String(content || ''));
    const [beforeSeparator, ...afterSeparatorParts] = text.split('\n---\n');
    const beforeLines = beforeSeparator.split('\n');
    const toolStartIndex = beforeLines.findIndex(isToolActivityStartLine);
    if (toolStartIndex < 0) {
        return {
            prelude: '',
            toolActivity: '',
            response: text,
            hasToolActivity: false,
        };
    }

    const prelude = beforeLines.slice(0, toolStartIndex).join('\n').trimEnd();
    if (afterSeparatorParts.length > 0) {
        const toolActivity = beforeLines.slice(toolStartIndex).join('\n');
        const response = afterSeparatorParts.join('\n---\n');
        return {
            prelude,
            toolActivity,
            response,
            hasToolActivity: !!toolActivity.trim(),
        };
    }

    const toolAndMaybeResponse = beforeLines.slice(toolStartIndex);
    const responseStart = toolAndMaybeResponse.findIndex((line) => line.trim() && !isToolActivityLine(line));
    if (responseStart >= 0) {
        const toolActivity = toolAndMaybeResponse.slice(0, responseStart).join('\n');
        const response = toolAndMaybeResponse.slice(responseStart).join('\n');
        return {
            prelude,
            toolActivity,
            response,
            hasToolActivity: true,
        };
    }

    const toolActivity = toolAndMaybeResponse.join('\n');
    return {
        prelude,
        toolActivity,
        response: '',
        hasToolActivity: !!toolActivity.trim(),
    };
}

function renderStreamingResponseSections({ prelude, toolActivity, response, hasToolActivity }, originalContent) {
    if (!hasToolActivity) {
        return renderStreamingMarkdown(originalContent);
    }

    const sections = [];
    if (prelude) {
        sections.push(`<div class="response-content response-prelude">${renderStreamingMarkdown(prelude)}</div>`);
    }
    if (toolActivity) {
        sections.push(renderToolActivityHtml(toolActivity));
    }
    if (response) {
        sections.push(`<div class="response-content">${renderStreamingMarkdown(response)}</div>`);
    }
    return sections.join('');
}

async function finalizeAgentContent(contentDiv, content) {
    const split = splitToolActivity(content);
    const { toolActivity, response, hasToolActivity, prelude } = split;
    if (!hasToolActivity) {
        await finalizeMarkdown(contentDiv, content);
        return;
    }

    const preludeHtml = prelude
        ? `<div class="response-content response-prelude">${renderMarkdown(prelude)}</div>`
        : '';
    contentDiv.innerHTML = `${preludeHtml}${renderToolActivityHtml(toolActivity)}`;
    if (response) {
        const responseDiv = document.createElement('div');
        responseDiv.className = 'response-content';
        contentDiv.appendChild(responseDiv);
        await finalizeMarkdown(responseDiv, response);
    }
}

export function renderToolActivityHtml(activityText) {
    const lines = String(activityText || '')
        .split('\n')
        .map((line) => line.trim())
        .filter(Boolean);
    if (lines.length === 0) return '';

    const groups = groupToolActivity(lines);
    const callsHtml = groups.map((group) => {
        const eventCount = group.lines.length === 1 ? '1 event' : `${group.lines.length} events`;
        const detail = group.detail ? ` · ${group.detail}` : '';
        const meta = `${group.status}${detail} · ${eventCount}`;
        const activityHtml = group.lines.map(renderToolActivityLineHtml).join('');

        return `
        <details class="tool-activity-expandable tool-activity-call">
            <summary class="tool-activity-summary">
                <span>${escapeHtml(`Tool call: ${group.name}`)}</span>
                <span class="tool-activity-count">${escapeHtml(meta)}</span>
            </summary>
            <div class="tool-activity-list">${activityHtml}</div>
        </details>
    `;
    }).join('');

    return `<div class="tool-activity-container">${callsHtml}</div>`;
}

/**
 * Detect and strip in-band revise sentinels from a chat-stream chunk.
 *
 * Returns ``{ textBefore, textAfter, sawSentinel }``. ``textBefore``
 * contains the chunk with all complete sentinel markers removed.
 * ``textAfter`` is kept for the legacy call shape but is always empty.
 * Incomplete sentinels are buffered by the streaming loop before this
 * helper runs, so wire metadata should never be rendered as prose.
 */
function stripReviseSentinel(chunk) {
    return stripStreamSentinels(chunk);
}

function findNextStreamSentinel(chunk, fromIndex = 0) {
    let found = null;
    for (const spec of STREAM_SENTINELS) {
        const idx = chunk.indexOf(spec.prefix, fromIndex);
        if (idx >= 0 && (!found || idx < found.idx)) {
            found = { ...spec, idx };
        }
    }
    return found;
}

function stripStreamSentinels(chunk) {
    let textBefore = '';
    let sawSentinel = false;
    const thoughts = [];
    let pos = 0;

    while (pos < chunk.length) {
        const found = findNextStreamSentinel(chunk, pos);
        if (!found) {
            textBefore += chunk.slice(pos);
            break;
        }

        textBefore += chunk.slice(pos, found.idx);

        const payloadStart = found.idx + found.prefix.length;
        const closeIdx = chunk.indexOf(found.suffix, payloadStart);
        if (closeIdx < 0) {
            if (found.kind === 'revise') sawSentinel = true;
            break;
        }

        const payloadText = chunk.slice(payloadStart, closeIdx);
        if (found.kind === 'revise') {
            sawSentinel = true;
        } else {
            try {
                const payload = JSON.parse(payloadText);
                if (payload && payload.content) thoughts.push(payload);
            } catch (_) {
                // Malformed UI metadata should not corrupt visible text.
            }
        }
        pos = closeIdx + found.suffix.length;
    }

    return {
        textBefore,
        textAfter: '',
        sawSentinel,
        thoughts,
    };
}

function appendThinkingItems(target, thoughts) {
    if (!target || !thoughts || thoughts.length === 0) return;
    for (const thought of thoughts) {
        const content = typeof thought === 'string' ? thought : (thought.content || '');
        if (!content) continue;
        const provider = typeof thought === 'object' ? (thought.provider || '') : '';
        const last = target[target.length - 1];
        const lastProvider = typeof last === 'object' ? (last.provider || '') : '';
        if (last && lastProvider === provider) {
            if (typeof last === 'string') {
                target[target.length - 1] = last + content;
            } else {
                last.content = (last.content || '') + content;
            }
        } else {
            target.push(typeof thought === 'string' ? { content, provider } : { ...thought, content, provider });
        }
    }
}

function findLastOpenStreamSentinel(text) {
    let found = null;
    for (const spec of STREAM_SENTINELS) {
        const idx = text.lastIndexOf(spec.prefix);
        if (idx >= 0 && (!found || idx > found.idx)) {
            found = { ...spec, idx };
        }
    }
    return found;
}

function trimPartialStreamSentinel(text) {
    for (const spec of STREAM_SENTINELS) {
        const maxCheck = Math.min(text.length, spec.prefix.length - 1);
        for (let i = maxCheck; i > 0; i--) {
            const tail = text.slice(text.length - i);
            if (spec.prefix.startsWith(tail)) {
                return {
                    text: text.slice(0, text.length - i),
                    buffer: tail,
                };
            }
        }
    }
    return { text, buffer: '' };
}

// ============================================================================
// DOM References
// ============================================================================

let chatContainer = null;
let messageInput = null;
let sendButton = null;
let modelSelector = null;
let thinkingIndicator = null;
let composerModeToggle = null;  // #1257 send-while-busy mode toggle

// Shared model selector instance
let sharedModelSelector = null;

// Autocomplete state
let autocompleteSelectedIndex = -1;

/**
 * Resolve the live #chat-container scroll viewport. Looked up fresh
 * because callers may run before initChat() has populated the cached
 * ref (e.g. early agent-select firing in parallel with init).
 */
function getChatContainer() {
    return chatContainer || document.getElementById('chat-container');
}

/**
 * Resolve the chat pane element a write should target. When called
 * with no arg, defaults to the currently-mounted agent's pane — this
 * is what voice/ui.js and other no-arg consumers rely on so a single
 * helper signature works for both "write to the visible chat" and
 * "write to a specific agent's detached pane".
 */
function resolvePaneElement(paneElement) {
    if (paneElement) return paneElement;
    if (state.mountedChatAgent === undefined) return null;
    const pane = state.chatPanes.get(state.mountedChatAgent);
    return pane ? pane.element : null;
}

/**
 * Mount the named agent's pane into #chat-container, swapping out
 * whichever pane (if any) is currently mounted. Does NOT bump any
 * generation — agent switching never invalidates a stream's pane. If
 * the incoming pane has unrendered mermaid (deferred during streaming
 * on a detached fragment, see updateStreamingMessage / finalize), run
 * the mermaid pass now that it's live and clear the flag.
 */
export function mountChatPane(agentName) {
    const target = getOrCreateChatPane(agentName);
    const container = getChatContainer();
    if (!container) {
        // Pre-init or chat-disabled host — record the intent so the
        // first real mount picks up the right agent. The pane stays
        // detached in the JS heap.
        state.mountedChatAgent = agentName;
        return target;
    }

    // Detach the currently-mounted pane (if any), saving its scroll.
    if (state.mountedChatAgent !== undefined) {
        const current = state.chatPanes.get(state.mountedChatAgent);
        if (current && current.element.parentNode === container) {
            current.scrollPos = container.scrollTop;
            current.element.remove();
        }
    }

    // Empty the container of any leftover non-pane children (defensive
    // against pre-migration HTML that mounted welcome content directly
    // into #chat-container without going through a pane).
    container.innerHTML = '';
    container.appendChild(target.element);
    state.mountedChatAgent = agentName;

    // Restore scroll to where the user left this agent's conversation.
    container.scrollTop = target.scrollPos;

    // Mermaid finalization was deferred while the pane was detached —
    // some renderers refuse to operate on disconnected nodes. Render
    // now that the pane is live.
    if (target.hasUnrenderedMermaid) {
        try {
            renderMermaidDiagrams(target.element);
        } catch (e) {
            console.warn('mermaid render on mount failed:', e);
        }
        target.hasUnrenderedMermaid = false;
    }
    // #1257: mode is per-pane (like session). Reflect the now-visible
    // agent's send-while-busy mode on the toggle.
    updateComposerModeToggle();
    return target;
}

/**
 * Wipe ONE agent's pane and bump that agent's pane-local generation.
 * Used for within-agent context changes — clear chat, new chat,
 * conversation switch, soft/hard delete of the active conversation.
 * Streams dispatched against the old generation drop their DOM writes
 * (their server-side response still persists to the agent's DB).
 *
 * Crucially, this does NOT touch any other agent's pane: a switch of
 * conversations on Agent A while Agent B is mid-stream must leave B's
 * chunks painting into B's pane uninterrupted.
 */
export function wipeAgentChatPane(agentName, html = '') {
    const pane = getOrCreateChatPane(agentName);
    pane.generation += 1;
    pane.streamingMsgDiv = null;
    pane.fullContent = '';
    pane.thinkingItems = [];
    pane.sessionId = null;
    pane.hasUnrenderedMermaid = false;
    pane.pendingRevise = false;
    pane.reviseConsumedRequestId = null;
    // #1257: a within-agent context change (clear chat, new chat,
    // conversation switch, delete) discards any queued follow-up — it
    // belonged to the conversation the user just left. The chip DOM
    // goes with the innerHTML reset below; null the field too.
    pane.queuedMessage = null;
    pane.element.innerHTML = html;
    pane.scrollPos = 0;
}

// ============================================================================
// Initialization
// ============================================================================

export function initChat() {
    // #879: skip wiring when the host has its own chat surface.  The chat
    // panel's DOM was removed by initNavigation(); attaching listeners now
    // would just no-op against missing nodes, but the explicit gate makes
    // the intent legible and keeps SharedModelSelector / autocomplete from
    // initializing in a host that doesn't render any of it.
    if (!API.hasCapability('chat')) return;
    chatContainer = document.getElementById('chat-container');
    messageInput = document.getElementById('message-input');
    sendButton = document.getElementById('send-button');
    modelSelector = document.getElementById('model-selector');
    thinkingIndicator = document.getElementById('thinking-indicator');
    composerModeToggle = document.getElementById('composer-mode-toggle');

    // Initial-pane migration: HTML ships welcome content baked into
    // #chat-container. Move that content into a pane element so the
    // pane-cache invariants hold from the very first frame — bare
    // chatContainer children would survive selectAgent's swap and end
    // up rendered alongside the new agent's pane.
    //
    // The host-agent is null at init (selectAgent hasn't run); the
    // null-keyed pane is what standalone mode uses for its only
    // conversation. In multi_agent mode, the first selectAgent call swaps
    // this pane out via mountChatPane.
    if (chatContainer) {
        const initialAgent = API.getHostAgent();
        const initialPane = getOrCreateChatPane(initialAgent);
        // Move existing children (welcome card, demo banners, etc.)
        // into the pane element and clear the container before mount.
        while (chatContainer.firstChild) {
            initialPane.element.appendChild(chatContainer.firstChild);
        }
        chatContainer.appendChild(initialPane.element);
        state.mountedChatAgent = initialAgent;
    }

    // Event listeners
    sendButton?.addEventListener('click', () => sendMessage());
    // #1257 send-while-busy mode toggle. Reflect the initial mounted
    // agent's mode (default 'interrupt').
    composerModeToggle?.addEventListener('click', toggleComposerMode);
    updateComposerModeToggle();

    // Ensure every external link rendered in chat opens in a new tab.
    // Markdown links go through marked's renderer (already adds target=_blank),
    // but anchors injected via innerHTML (system messages, notifications,
    // tool-result HTML) bypass marked. A delegated listener on the chat
    // container backstops both paths without preventing default — middle-click,
    // Cmd-click, and right-click "open in new tab" keep their native behavior.
    chatContainer?.addEventListener('click', (event) => {
        const anchor = event.target instanceof Element
            ? event.target.closest('a[href]')
            : null;
        if (!anchor) return;
        const href = anchor.getAttribute('href') || '';
        // Leave in-page anchors and javascript: hrefs alone.
        if (!href || href.startsWith('#') || href.toLowerCase().startsWith('javascript:')) return;
        if (anchor.getAttribute('target') !== '_blank') {
            anchor.setAttribute('target', '_blank');
            anchor.setAttribute('rel', 'noopener noreferrer');
        }
    });

    // Stop button for cancelling requests
    const stopButton = document.getElementById('stop-button');
    stopButton?.addEventListener('click', stopRequest);

    // Input events for autocomplete
    messageInput?.addEventListener('input', handleInput);
    messageInput?.addEventListener('keydown', handleKeydown);
    messageInput?.addEventListener('blur', () => {
        setTimeout(hideCommandAutocomplete, 200);
    });

    // Note: Model selector events are now handled by SharedModelSelector in loadModels()

    // SSE notifications and context status are loaded after agent selection:
    // - MultiAgent mode: selectAgent() handles both
    // - Standalone mode: app.js init handles both after loadAgents()
}

// ============================================================================
// SSE Task Notifications
// ============================================================================

let notificationEventSource = null;
let notificationReconnectTimeout = null;

// Subscribers registered via subscribeSSE. Kept in module state (not attached
// directly to notificationEventSource) so subscriptions survive reconnects —
// each new EventSource re-attaches every handler in this list. See #748.
const sseSubscribers = [];

/**
 * Subscribe to a named SSE event on the notification stream.
 *
 * Other modules (e.g. Security) call this to receive server-pushed events
 * without reaching into `notificationEventSource` directly. The handler is
 * stored in a module-scoped list and re-attached on every (re)connect, so
 * subscribers keep working across network drops.
 *
 * @param {string} eventType  Name matching the server-side `event:` field
 * @param {(evt: MessageEvent) => void} handler
 */
export function subscribeSSE(eventType, handler) {
    sseSubscribers.push({ eventType, handler });
    if (notificationEventSource) {
        notificationEventSource.addEventListener(eventType, handler);
    }
}

/**
 * Connect to the SSE notifications endpoint for real-time task updates.
 * Automatically reconnects on disconnect with exponential backoff.
 */
export function connectNotifications() {
    // #879: the SSE notification stream is multiplexed — chat consumes
    // task/streaming events for the thinking indicator, Security.init()
    // consumes approval_request / approval_withdrawn for the modal queue
    // (#748).  A host that disables chat but keeps permissions/approval
    // prompts on still needs the stream open, otherwise the approval
    // modal never fires.  Open the connection if EITHER consumer is
    // active; only skip when none of them are.
    const needed = API.hasCapability('chat')
        || API.hasCapability('permissions')
        || API.hasCapability('audit');
    if (!needed) return;
    if (notificationEventSource) {
        notificationEventSource.close();
    }

    // EventSource can't send custom headers. Standalone server: pass the
    // bootstrap API key as a query param. OAuth: the session cookie rides
    // automatically. Embedded hosts using a non-API-key auth scheme (e.g.
    // BearerToken) must authenticate /api/agent/notifications/sse via cookie
    // or their own middleware — the host controls that path.
    const apiKey = API.getApiKey();

    try {
        const ssePath = API.buildAgentUrl('/api/agent/notifications/sse');
        const sseUrl = apiKey ? `${ssePath}?api_key=${encodeURIComponent(apiKey)}` : ssePath;
        notificationEventSource = new EventSource(sseUrl);

        notificationEventSource.addEventListener('connected', (e) => {
            console.log('SSE notifications connected');
            // Reset backoff counter — connection is healthy
            reconnectAttempts = 0;
            // Clear any pending reconnect
            if (notificationReconnectTimeout) {
                clearTimeout(notificationReconnectTimeout);
                notificationReconnectTimeout = null;
            }
        });

        notificationEventSource.addEventListener('task_notification', (e) => {
            try {
                const data = JSON.parse(e.data);
                showTaskNotification(data.message, data.type);
            } catch (err) {
                console.error('Failed to parse task notification:', err);
            }
        });

        // Constitutional honesty signal — Wave 5C of #1048.
        //
        // The event marks the tool boundary, but it should not clear
        // already-visible prose. The in-band sentinel is stripped from
        // the chat stream; prose before and after it remains visible.
        notificationEventSource.addEventListener('revising', (e) => {
            try {
                const data = JSON.parse(e.data);
                const targetRequestId = data && data.request_id;
                if (!targetRequestId) return;
                for (const [agentName, pane] of state.chatPanes) {
                    const paneRequestId = API.getCurrentStreamRequestId(agentName);
                    if (paneRequestId !== targetRequestId) continue;
                    // Already consumed (likely by the in-band path
                    // landing first) — don't re-arm.
                    if (pane.reviseConsumedRequestId === targetRequestId) return;
                    if (!pane.streamingMsgDiv) return;
                    pane.pendingRevise = true;
                    pane.reviseConsumedRequestId = targetRequestId;
                    return;
                }
            } catch (err) {
                console.error('Failed to handle revising event:', err);
            }
        });

        notificationEventSource.addEventListener('ping', () => {
            // Keepalive - no action needed
        });

        // Re-attach every subscriber to this fresh EventSource so subscriptions
        // survive reconnects (the old EventSource is discarded).
        for (const { eventType, handler } of sseSubscribers) {
            notificationEventSource.addEventListener(eventType, handler);
        }

        notificationEventSource.addEventListener('error', (e) => {
            console.warn('SSE notification error, will reconnect...');
            notificationEventSource?.close();
            notificationEventSource = null;
            scheduleReconnect();
        });

        notificationEventSource.onerror = () => {
            // Fallback error handler
            notificationEventSource?.close();
            notificationEventSource = null;
            scheduleReconnect();
        };
    } catch (e) {
        console.error('Failed to connect to SSE notifications:', e);
        scheduleReconnect();
    }
}

/**
 * Schedule a reconnection attempt with exponential backoff.
 *
 * Stops retrying after MAX_RECONNECT_ATTEMPTS consecutive failures so a
 * persistent auth error (e.g. wrong API key) doesn't flood the server with
 * repeated 401 requests. reconnectAttempts is reset to 0 when a connection
 * is established successfully (see 'connected' handler in connectNotifications).
 */
const MAX_RECONNECT_ATTEMPTS = 10;
let reconnectAttempts = 0;
function scheduleReconnect() {
    if (notificationReconnectTimeout) return;

    if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
        console.warn(
            `SSE notifications: giving up after ${MAX_RECONNECT_ATTEMPTS} failed attempts. ` +
            'Check that KESTREL_API_KEY matches the running server.'
        );
        return;
    }

    // Exponential backoff: 1s, 2s, 4s, 8s, max 30s
    const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
    reconnectAttempts++;

    notificationReconnectTimeout = setTimeout(() => {
        notificationReconnectTimeout = null;
        connectNotifications();
    }, delay);
}

/**
 * Show a task notification in the chat interface.
 *
 * Notifications target the visible (mounted) agent's pane — task
 * notifications come over a single SSE stream pinned to the selected
 * agent, so by definition the visible pane is the right destination.
 * Per-agent task notifications for non-visible agents would require
 * one SSE per loaded agent and is out of scope for the parallel-chat
 * change.
 */
function showTaskNotification(message, type) {
    // Reset reconnect attempts on successful notification
    reconnectAttempts = 0;

    const paneElement = resolvePaneElement();
    if (!paneElement) return;

    const div = document.createElement('div');
    div.className = 'message notification-message';

    // Style based on type
    let bgColor, borderColor;
    switch (type) {
        case 'completed':
            bgColor = 'rgba(34, 197, 94, 0.1)';  // green
            borderColor = 'rgba(34, 197, 94, 0.3)';
            break;
        case 'failed':
            bgColor = 'rgba(239, 68, 68, 0.1)';  // red
            borderColor = 'rgba(239, 68, 68, 0.3)';
            break;
        case 'canceled':
            bgColor = 'rgba(245, 158, 11, 0.1)';  // amber
            borderColor = 'rgba(245, 158, 11, 0.3)';
            break;
        default:
            bgColor = 'rgba(59, 130, 246, 0.1)';  // blue
            borderColor = 'rgba(59, 130, 246, 0.3)';
    }

    div.style.cssText = `
        background: ${bgColor};
        border: 1px solid ${borderColor};
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.875rem;
    `;

    div.innerHTML = `
        <div class="notification-content">
            ${message}
        </div>
    `;

    paneElement.appendChild(div);
    const c = getChatContainer();
    if (c) c.scrollTop = c.scrollHeight;

    // Also show a Toast notification
    Toast.show(message, type === 'failed' ? 'error' : 'info');
}

/**
 * Disconnect SSE notifications (call when leaving page).
 */
export function disconnectNotifications() {
    if (notificationReconnectTimeout) {
        clearTimeout(notificationReconnectTimeout);
        notificationReconnectTimeout = null;
    }
    if (notificationEventSource) {
        notificationEventSource.close();
        notificationEventSource = null;
    }
}

// ============================================================================
// Thinking Indicator
// ============================================================================

/**
 * Refresh the thinking indicator + input/send disable state from
 * `state.waitingAgents`. Only the *currently selected* agent's status
 * controls the visible UI — that way switching to Agent B while Agent A
 * is mid-stream doesn't make B's input look busy. Call this whenever
 * the waiting set changes OR the selected agent changes.
 */
export function updateThinkingIndicator() {
    const current = API.getHostAgent();
    const busy = state.waitingAgents.has(current);
    if (thinkingIndicator) {
        thinkingIndicator.style.display = busy ? 'flex' : 'none';
    }
    // #1255: composer stays editable while the agent is streaming.
    // Hitting Enter mid-stream is the universal interrupt pattern
    // (ChatGPT, Claude.ai, Cursor); sendMessage detects the busy
    // state and routes through stopAgent before dispatching the new
    // turn. The disable lines that used to live here forced the user
    // to click Stop, wait, then start typing — an extra click and a
    // state transition for what is by far the most common path of
    // "I'd like to redirect mid-answer."
}

/**
 * Toggle the per-agent thinking pulse + per-agent stop control on a
 * sidebar agent row. Driven by `state.waitingAgents`. Called from
 * sendMessage start / finally / catch so the dot lights up while the
 * agent has work in flight, even when a different agent is visible.
 *
 * Selector matches the row rendered in identity.js loadAgents().
 * No-op if the row hasn't been rendered yet (early init).
 */
export function refreshAgentThinkingDot(agentName) {
    if (typeof document === 'undefined') return;
    const row = document.querySelector(`.agent-item[data-agent-name="${CSS.escape(String(agentName))}"]`);
    if (!row) return;
    const busy = state.waitingAgents.has(agentName);
    row.classList.toggle('agent-thinking', busy);
}

// ============================================================================
// Stop Request
// ============================================================================

/**
 * Stop the currently-visible agent's streaming request. Wired to the
 * chat-pane Stop button. Use stopAgent(name) for the per-agent stop
 * control rendered in the sidebar agent list.
 */
async function stopRequest() {
    return stopAgent(API.getHostAgent());
}

/**
 * Stop a specific agent's in-flight stream. Aborts client-side via
 * the per-agent AbortController, and tells the server to halt by
 * routing the /stop POST to that agent's endpoint (NOT the currently-
 * selected agent's, which is what the un-overloaded API.stop would do).
 *
 * Exposed so the sidebar agent list can render a per-agent stop
 * affordance — clicking "Stop A" while viewing B must reach A's
 * backend, not B's.
 */
export async function stopAgent(agentName) {
    // #1257: Stop = stop everything. Clear any queued follow-up
    // SYNCHRONOUSLY, before the abort and before the awaited /stop
    // POST. This must precede every await: if `API.stop()` is slow,
    // the in-flight turn's stream can complete (normally — e.g. no
    // abort controller was registered yet, so `wasAborted` stays
    // false) and reach its `finally` with `pane.queuedMessage` still
    // set, dispatching the message the user just cancelled. Clearing
    // first closes that window — by the time any other async context
    // runs, the queue is already gone. The finally's `!wasAborted`
    // guard is a redundant second layer, not the primary one. (The
    // queued chip's own × cancels ONLY the queued message while
    // letting the turn run; that path is in renderQueuedChip.) Use
    // the map directly so we don't conjure a pane for an agent that
    // never had one.
    const pane = state.chatPanes.get(agentName);
    if (pane) {
        pane.queuedMessage = null;
        clearQueuedChip(pane);
    }

    const abortController = API.getStreamAbortController(agentName);
    if (abortController) {
        try { abortController.abort(); } catch (_) { /* noop */ }
    }

    const requestId = API.getCurrentStreamRequestId(agentName);
    try {
        // Pass agentName explicitly so the stop POST hits this agent's
        // endpoint regardless of which agent is currently selected.
        await API.stop(requestId, agentName);
    } catch (e) {
        console.error(`Error stopping request on ${agentName}:`, e);
    }

    state.waitingAgents.delete(agentName);
    refreshAgentThinkingDot(agentName);
    if (agentName === API.getHostAgent()) {
        updateThinkingIndicator();
    }
}

// ============================================================================
// Queue Mode (#1257)
// ============================================================================

const QUEUED_CHIP_CLASS = 'queued-message-chip';

/**
 * Render (or replace) the pending-queued-message chip at the bottom of
 * an agent's pane. Queue mode holds exactly ONE pending message; a
 * re-Enter replaces the chip so the user always sees precisely what
 * will be sent when the in-flight turn finishes. The chip lives in
 * pane.element (like message bubbles) so it travels with the pane
 * across agent switches and shows when the user returns.
 */
function renderQueuedChip(pane, agentName, text) {
    clearQueuedChip(pane);
    if (!pane || !pane.element) return;
    const chip = document.createElement('div');
    chip.className = QUEUED_CHIP_CLASS;
    const label = document.createElement('span');
    label.className = 'queued-message-text';
    label.textContent = text;
    const cancel = document.createElement('button');
    cancel.type = 'button';
    cancel.className = 'queued-message-cancel';
    cancel.title = 'Cancel queued message';
    cancel.textContent = '×';  // ×
    // Cancel ONLY the queued message — the in-flight turn keeps
    // running. (The big Stop button cancels both; see stopAgent.)
    cancel.addEventListener('click', () => {
        pane.queuedMessage = null;
        clearQueuedChip(pane);
    });
    chip.appendChild(label);
    chip.appendChild(cancel);
    pane.element.appendChild(chip);
    // Scroll the real viewport, not pane.element: `.chat-container-pane`
    // is `display: contents` so it has no scroll box. Mirror the
    // addMessage/updateStreamingMessage pattern — only scroll when this
    // pane is the one actually mounted into #chat-container, so a chip
    // queued for a backgrounded agent doesn't yank the visible pane.
    const c = getChatContainer();
    if (c && pane.element.parentNode === c) {
        c.scrollTop = c.scrollHeight;
    }
}

/** Remove the queued-message chip from a pane, if present. */
function clearQueuedChip(pane) {
    if (!pane || !pane.element) return;
    const existing = pane.element.querySelector('.' + QUEUED_CHIP_CLASS);
    if (existing) existing.remove();
}

/**
 * Reflect the mounted agent's composerMode on the toggle button.
 * Mode is per-pane (like session), so the control always shows the
 * VISIBLE agent's mode — called on init and on every pane mount.
 */
export function updateComposerModeToggle() {
    if (!composerModeToggle) return;
    const current = API.getHostAgent();
    const pane = state.chatPanes.get(current);
    const mode = (pane && pane.composerMode) || 'interrupt';
    composerModeToggle.dataset.mode = mode;
    composerModeToggle.textContent = mode === 'queue' ? 'Queue' : 'Interrupt';
    composerModeToggle.title = mode === 'queue'
        ? 'Send-while-busy: queue this message and send it when the agent finishes. Click to switch to Interrupt.'
        : 'Send-while-busy: stop the current turn and send now. Click to switch to Queue.';
}

/** Toggle the mounted agent's send-while-busy mode (per-pane). */
function toggleComposerMode() {
    const pane = getOrCreateChatPane(API.getHostAgent());
    pane.composerMode = pane.composerMode === 'queue' ? 'interrupt' : 'queue';
    updateComposerModeToggle();
}

// ============================================================================
// Send Message
// ============================================================================

/**
 * Dispatch a chat turn.
 *
 * Normal call (Enter / Send click): reads the composer textarea and
 * the currently-mounted host agent.
 *
 * #1257 queued re-dispatch: the completing turn's finally calls
 * ``sendMessage(queuedText, originalAgent)`` so the queued message
 * lands against the agent it was queued for — NOT whatever agent the
 * user happens to be viewing when the prior turn finishes. Without
 * the explicit agent, an agent switch between queueing and dispatch
 * would misroute the message.
 */
export async function sendMessage(overrideText, overrideAgent) {
    const fromComposer = overrideText === undefined;
    const text = (fromComposer ? messageInput.value : overrideText).trim();

    // Capture the dispatch agent and pin the dispatch's pane up front,
    // BEFORE any await — the user could mount a different agent's pane
    // before the first stream chunk arrives, and the user's typed text
    // must always land in the agent it was dispatched against.
    const dispatchAgent = overrideAgent !== undefined
        ? overrideAgent
        : API.getHostAgent();
    if (!text) return;

    const pane = getOrCreateChatPane(dispatchAgent);

    // Send-while-busy. Behavior depends on the pane's composerMode.
    if (state.waitingAgents.has(dispatchAgent)) {
        if (pane.composerMode === 'queue') {
            // #1257 queue mode: stash the message and surface it as a
            // pending chip. The completing turn's finally dispatches
            // it. Re-Enter while something is already queued REPLACES
            // it — single-slot queue (multi-message queue is a
            // deferred follow-up). Do NOT interrupt the in-flight turn.
            pane.queuedMessage = text;
            if (fromComposer) messageInput.value = '';
            renderQueuedChip(pane, dispatchAgent, text);
            return;
        }
        // #1255 interrupt mode (default): stop the in-flight turn
        // before dispatching the new one. ``stopAgent`` aborts the
        // client-side fetch synchronously and awaits the server's
        // ``/api/agent/stop`` ack, so the cancellation is registered
        // server-side BEFORE the next ``streamInvoke`` POST opens; the
        // backend's loop-level checkpoints (#1256) then halt the prior
        // turn cleanly. Without this, two streams for the same agent
        // would race and the prior turn's chunks could keep painting
        // into the pane after the new user message had already been
        // rendered. Note: ``stopAgent`` removes the agent from
        // ``state.waitingAgents`` itself, so the subsequent ``add``
        // below is the correct next state.
        await stopAgent(dispatchAgent);
    }

    // Capture the pane-local generation. This dispatch's DOM writes
    // gate on `pane.generation === dispatchGeneration`. Agent switches
    // do NOT bump generation — only within-agent context changes
    // (clear chat, new conversation, soft/hard delete) do, so streams
    // keep painting through agent switches but stop the moment the
    // user changes the conversation underneath them.
    const dispatchGeneration = pane.generation;

    // Read the dispatch agent's session id from its pane (the
    // currentSessionId getter resolves to the mounted agent — at this
    // point the user could have already switched, so go through the
    // pane directly).
    const sessionId = pane.sessionId || null;

    await addMessage('user', text, pane.element);
    // Only clear the composer when the text CAME from it. A #1257
    // queued re-dispatch passes overrideText and must not wipe
    // whatever the user has since typed for the (possibly different)
    // currently-mounted agent.
    if (fromComposer) messageInput.value = '';
    state.waitingAgents.add(dispatchAgent);
    updateThinkingIndicator();
    refreshAgentThinkingDot(dispatchAgent);

    // DO NOT send model/provider overrides from the chat UI. The server
    // already knows each agent's persisted mandate (set via POST
    // /api/model/set when the user picked from the dropdown). Sending
    // state.selectedModel/selectedProvider here was a bug: those vars
    // don't update when the user switches agents, so a stale override
    // from the previous agent would silently reroute the new agent's
    // chat to a different model. Source of truth: server mandate.

    // Pane-fresh = "no within-agent context change since dispatch."
    // Visible-and-fresh = also the agent the user is actively viewing;
    // used to gate global-singleton updates (model selector, footer).
    const isPaneFresh = () => pane.generation === dispatchGeneration;
    const isCurrentVisible = () =>
        isPaneFresh() && API.getHostAgent() === dispatchAgent;

    let wasAborted = false;

    try {
        if (state.useStreaming) {
            const msgDiv = addMessageStreaming('agent', pane.element);
            pane.streamingMsgDiv = msgDiv;
            pane.thinkingItems = [];
            let fullContent = '';

            try {
                // Pass dispatchAgent EXPLICITLY to streamInvoke so the
                // URL is pinned to this dispatch's agent, even if the
                // user switches mid-flight or a 401 refresh forces a
                // retry. Without this the URL would be recaptured
                // from state.selectedHostAgent at fetch time.
                let learnedSessionId = false;
                // Wave 5E: cross-chunk parser buffer for partial
                // sentinels. ReadableStream chunking isn't guaranteed
                // to preserve server yields, so a sentinel can split
                // anywhere — including INSIDE the prefix string
                // ("\x1eKESTREL:REV" / "ISE:{...}\x1e..."). We hold
                // back two cases:
                //   (A) Full prefix present but no closing \\x1e
                //   (B) Partial prefix at chunk tail (any non-empty
                //       prefix of REVISE_SENTINEL_PREFIX)
                let sentinelBuffer = '';
                for await (const rawChunk of API.streamInvoke(text, null, sessionId, null, false, dispatchAgent)) {
                    const merged = sentinelBuffer + rawChunk;
                    sentinelBuffer = '';
                    let processable = merged;
                    // Case A: a full prefix without close in this
                    // chunk — buffer everything from prefix on.
                    const lastOpen = findLastOpenStreamSentinel(merged);
                    if (lastOpen) {
                        const closeAfter = merged.indexOf(
                            lastOpen.suffix,
                            lastOpen.idx + lastOpen.prefix.length,
                        );
                        if (closeAfter < 0) {
                            processable = merged.slice(0, lastOpen.idx);
                            sentinelBuffer = merged.slice(lastOpen.idx);
                        }
                    }
                    // Strip any complete sentinels in processable.
                    // Revise markers and thinking markers are wire
                    // metadata; visible prose stays in the accumulator.
                    let { textBefore, textAfter, sawSentinel, thoughts } =
                        stripStreamSentinels(processable);
                    if (thoughts.length) {
                        appendThinkingItems(pane.thinkingItems, thoughts);
                    }
                    // Case B: a PARTIAL prefix at the tail of post-
                    // strip output — happens when a chunk splits
                    // INSIDE the prefix string ("\x1eKESTREL:REV" then
                    // "ISE:..."). Run AFTER strip so we don't
                    // misidentify the closing \\x1e of a just-stripped
                    // sentinel as a new prefix start. Codex P2 of #1089.
                    if (!sentinelBuffer) {
                        const target = textBefore;
                        if (target.length > 0) {
                            const trimmed = trimPartialStreamSentinel(target);
                            if (trimmed.buffer) {
                                sentinelBuffer = trimmed.buffer;
                                textBefore = trimmed.text;
                            }
                        }
                    }
                    if (sawSentinel) {
                        // Mark this request's revise as consumed so a
                        // delayed SSE arriving after the in-band path is
                        // treated as a no-op.
                        const rid = API.getCurrentStreamRequestId(dispatchAgent);
                        if (rid) pane.reviseConsumedRequestId = rid;
                    }
                    // A legacy SSE revising event can still arrive before
                    // the in-band sentinel. Clear the flag without clearing
                    // the visible accumulator; the sentinel itself is the
                    // only thing that should disappear.
                    if (pane.pendingRevise) {
                        pane.pendingRevise = false;
                    }
                    const chunk = sawSentinel ? textBefore + textAfter : textBefore;
                    if (!chunk) continue;
                    fullContent += chunk;
                    // The server resolves the effective session_id and
                    // returns it as the X-Session-Id response header,
                    // which streamInvoke captures before yielding the
                    // first body chunk. Adopt it onto the pane so the
                    // next turn sends it back explicitly, anchoring
                    // the pane to a durable conversation id. Only when
                    // pane.sessionId was null — never overwrite an
                    // explicit user-clicked conversation.
                    if (!learnedSessionId && !pane.sessionId) {
                        const effective = API.getEffectiveSessionId(dispatchAgent);
                        if (effective) {
                            pane.sessionId = effective;
                        }
                        learnedSessionId = true;
                    }
                    // Per-pane gate only — chunks DO paint into the
                    // dispatch agent's pane element even when that
                    // pane is detached (the user is viewing a
                    // different agent). When they come back, the
                    // streaming text is already there.
                    if (isPaneFresh()) {
                        updateStreamingMessage(msgDiv, fullContent, pane.element, pane.thinkingItems);
                    }
                }
                if (isPaneFresh()) {
                    await finalizeStreamingMessage(msgDiv, fullContent, pane);
                }
                if (isCurrentVisible()) {
                    await checkForModelChange(fullContent);
                }
            } catch (streamError) {
                if (streamError.name === 'AbortError') {
                    wasAborted = true;
                    throw streamError;
                }
                if (streamError.message.includes('404') || streamError.message.includes('405')) {
                    console.log('Streaming not available, falling back to standard invoke');
                    state.useStreaming = false;
                    msgDiv.remove();
                    // invokeForAgent pins the URL to dispatchAgent —
                    // unprefixed invoke() routes via the currently
                    // selected agent and would land on the wrong
                    // backend if the user has switched.
                    const response = await API.invokeForAgent(text, null, sessionId, null, dispatchAgent);
                    if (response && response.session_id && !pane.sessionId) {
                        pane.sessionId = response.session_id;
                    }
                    if (isPaneFresh()) {
                        await addMessage('agent', response.response, pane.element);
                    }
                    if (isCurrentVisible()) {
                        await checkForModelChange(response.response);
                    }
                } else {
                    throw streamError;
                }
            }
        } else {
            const response = await API.invokeForAgent(text, null, sessionId, null, dispatchAgent);
            if (response && response.session_id && !pane.sessionId) {
                pane.sessionId = response.session_id;
            }
            if (isPaneFresh()) {
                await addMessage('agent', response.response, pane.element);
            }
            if (isCurrentVisible()) {
                await checkForModelChange(response.response);
            }
        }
    } catch (e) {
        if (e && e.name === 'AbortError') {
            wasAborted = true;
        } else if (isPaneFresh()) {
            await addMessage('agent', `Error: ${e.message}`, pane.element);
        } else {
            console.warn(`stream error on ${dispatchAgent} (pane stale):`, e.message);
        }
    } finally {
        pane.streamingMsgDiv = null;
        pane.pendingRevise = false;
        pane.reviseConsumedRequestId = null;
        state.waitingAgents.delete(dispatchAgent);
        refreshAgentThinkingDot(dispatchAgent);
        // Drive the visible thinking indicator from whatever agent the
        // user is currently looking at, not the dispatch agent.
        updateThinkingIndicator();
        // Context status touches a global singleton — gate on visible.
        if (isCurrentVisible()) {
            updateContextStatus();
        }
        // Toast the user when a non-visible agent finishes responding,
        // so a long-running answer on Agent A surfaces while they're
        // chatting with Agent B. Skipped on aborts and on stale panes.
        if (!wasAborted && isPaneFresh() && API.getHostAgent() !== dispatchAgent) {
            const label = dispatchAgent || 'agent';
            Toast.info(`${label} finished responding`);
        }
        // #1257 queue mode: if a follow-up was queued while this turn
        // streamed, dispatch it now the turn has fully settled. ALWAYS
        // clear the stored message + chip here; only RE-dispatch when
        //   * the turn wasn't user-aborted — Stop = stop everything,
        //     including the queue (stopAgent also clears it eagerly so
        //     the chip vanishes on click; this is the belt-and-braces
        //     half), and
        //   * the pane is still fresh — a conversation switch
        //     mid-turn discards the stale queue.
        // A prior-turn ERROR still dispatches (the user queued it
        // deliberately; a route failure shouldn't silently eat their
        // next input). queueMicrotask defers the dispatch past this
        // finally's own unwind so the queued sendMessage starts from a
        // clean async context instead of re-entering mid-cleanup —
        // the ordering lesson from #1255's review.
        if (pane.queuedMessage != null) {
            const queued = pane.queuedMessage;
            pane.queuedMessage = null;
            clearQueuedChip(pane);
            if (!wasAborted && isPaneFresh()) {
                queueMicrotask(() => {
                    // Re-check generation at fire time — a conversation
                    // switch could land between this finally and the
                    // microtask draining.
                    if (pane.generation !== dispatchGeneration) return;
                    sendMessage(queued, dispatchAgent);
                });
            }
        }
    }
}

// ============================================================================
// Context Status Indicator
// ============================================================================

let contextStatusElement = null;

/**
 * Update the context status indicator.
 * Shows messages count and utilization percentage with color coding.
 *
 * When no conversation is active (`state.currentSessionId` is null, e.g. a
 * fresh chat pane before the user starts or selects a conversation) the
 * indicator is hidden rather than showing stale/aggregate data.  See #713 —
 * previously this case leaked the agent's global cross-session message
 * count into the footer and offered a Compress button based on that
 * aggregate, which made no sense.
 */
export async function updateContextStatus() {
    // #879: context-status footer is part of the chat surface.  No-op when
    // the host opted out so /api/agent/context-status isn't called on every
    // conversation change.
    if (!API.hasCapability('chat')) return;
    try {
        if (!contextStatusElement) {
            createContextStatusElement();
        }
        if (!contextStatusElement) return;

        const sessionId = state.currentSessionId || null;

        // No active conversation → hide the footer outright. The server
        // also returns idle/zeroed values for this case, but hiding
        // client-side avoids any flash of "0 msgs · 0%" during the RTT.
        if (!sessionId) {
            contextStatusElement.innerHTML = '';
            return;
        }

        const status = await API.getContextStatus(sessionId);
        const { message_count, utilization_percent, status: contextState, warnings, route_cap } = status;

        // Color based on utilization
        let color, icon;
        if (utilization_percent < 50) {
            color = '#22c55e';  // green
            icon = '●';
        } else if (utilization_percent < 80) {
            color = '#eab308';  // yellow
            icon = '●';
        } else if (utilization_percent < 95) {
            color = '#f97316';  // orange
            icon = kicon('warning');
        } else {
            color = '#ef4444';  // red
            icon = kicon('warning');
        }

        // #1503: route per-turn cap labeling. On capped routes (notably
        // ``openai:plan`` on ChatGPT-Plus), ``TokenCounter`` already
        // returns the cap as ``context_limit`` — so the existing
        // utilization % above is ALREADY measured against the route
        // cap, modulo the response reserve. The route_cap block exists
        // so the UI can NAME the cap (instead of letting the operator
        // think they're full of the model's 256K window) and surface
        // the actionable knob in the tooltip. A separate percentage
        // segment would be redundant — operator just needs to know
        // what they're being limited by (codex round 2 P2 on #1503).
        let routeCapBadge = '';
        let routeCapTooltip = '';
        if (route_cap && route_cap.cap_tokens) {
            const routeLabel = _esc(route_cap.route || 'route');
            const capTok = Number(route_cap.cap_tokens || 0).toLocaleString();
            const projected = Number(route_cap.projected_turn_payload || 0).toLocaleString();
            const headroom = Number(route_cap.headroom_tokens || 0).toLocaleString();
            const knobHint = route_cap.knob
                ? `\nRaise via ${route_cap.knob} or [llm.route_context_caps] in kestrel.toml.`
                : '\nRaise via [llm.route_context_caps] in kestrel.toml.';
            // Cheap-poll path (full=false) skips RAG measurement —
            // mark the pill % as a FLOOR for capped routes with RAG
            // (codex round 1 P2 on #1503).
            const ragFloorHint = route_cap.includes_rag === false
                ? '\nRetrieval/RAG not measured in this poll — open breakdown for full picture.'
                : '';
            routeCapTooltip = `\nRoute cap (${routeLabel}): ${projected} / ${capTok} tokens used; ${headroom} tokens headroom.${_esc(knobHint)}${_esc(ragFloorHint)}`;
            routeCapBadge = ` <span style="color:${color};opacity:0.75;font-size:0.65rem;">@ ${routeLabel}</span>`;
        }

        contextStatusElement.style.color = color;

        // Show compress button when utilization is 70%+
        const showCompress = utilization_percent >= 70;
        const compressButton = showCompress
            // The Compress button lives inside the clickable pill
            // span, so its click would bubble up to the pill's
            // ``openContextBreakdownPopup`` handler. Stop propagation
            // so clicking Compress only sends !compress (codex round
            // 1 P2 caught this regression).
            ? `<button onclick="event.stopPropagation(); window.compressContext()" style="
                    margin-left: 0.5rem;
                    padding: 0.125rem 0.375rem;
                    font-size: 0.625rem;
                    background: ${color};
                    color: white;
                    border: none;
                    border-radius: 3px;
                    cursor: pointer;
                    opacity: 0.9;
                " title="Compress older messages to free up context space">Compress</button>`
            : '';

        // Pill is clickable — opens the breakdown popup (#1310) which
        // renders the layered context taxonomy + warning labels +
        // silently-pruned auto-detect. ``onclick`` lives in a global
        // (``window.openContextBreakdownPopup``) so the popup module
        // can lazy-load without forcing a dependency cycle through
        // chat.js' export surface.
        contextStatusElement.innerHTML = `
            <span class="context-pill"
                  role="button"
                  tabindex="0"
                  onclick="window.openContextBreakdownPopup()"
                  onkeydown="if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); window.openContextBreakdownPopup(); }"
                  style="cursor: pointer; user-select: none;"
                  title="Click for per-section context breakdown · ${message_count} messages · ${utilization_percent.toFixed(1)}% of window used${warnings.length ? '\nWarnings: ' + warnings.join(', ') : ''}${_esc(routeCapTooltip)}">
                ${icon} ${message_count} msgs · ${utilization_percent.toFixed(0)}%${routeCapBadge}${compressButton}
            </span>
        `;

        // Show warning in chat if context is critically full
        if (contextState === 'critical' && warnings.length > 0) {
            showContextWarning(warnings);
        }
    } catch (e) {
        console.debug('Context status unavailable:', e.message);
    }
}

/**
 * Create the context status element in the UI.
 * Uses the existing #context-status element in the input footer.
 */
function createContextStatusElement() {
    // The element already exists in HTML (input footer), just return it
    contextStatusElement = document.getElementById('context-status');
    return contextStatusElement;
}

/**
 * Show a context warning message in the chat. Targets the visible
 * agent's pane — context warnings are emitted from updateContextStatus
 * which itself only runs for the visible agent.
 */
function showContextWarning(warnings, paneElement = null) {
    const target = resolvePaneElement(paneElement);
    if (!target) return;

    // Don't show duplicate warnings
    if (target.querySelector('.context-warning')) return;

    const div = document.createElement('div');
    div.className = 'message system-message context-warning';
    div.style.cssText = `
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.875rem;
    `;
    div.innerHTML = `
        <strong>${kicon('warning')} Context Warning:</strong> ${warnings.join('. ')}
        <br><small>Use <code>!compress</code> to summarize older messages, or start fresh with <code>!new-session</code></small>
    `;
    target.appendChild(div);
    const c = getChatContainer();
    if (c) c.scrollTop = c.scrollHeight;
}

/**
 * Open the context-breakdown popup (#1310 / epic #1307).
 *
 * Reads ``/api/agent/context-status?full=true`` so RAG retrieval runs
 * for this on-demand call (the frequent footer poll passes
 * ``full=false`` and skips RAG). Renders the layered taxonomy Emma
 * signed off on in PR #1306:
 *
 *   System / Governance — mandatory (non-borrowable) vs optional rows
 *   Tools — schemas + scaffolding, "estimated" badge
 *   Conversation — total / kept-after-pruning / raw vs effective
 *   Memories — count + "not counted" badge if no retriever wired
 *   Retrieval / RAG — chunks + "estimated" badge, "skipped" when poll
 *   Reserve / Overhead — dynamic_context_overhead + response_reserve
 *
 * UI honesty invariant (Emma): never imply "compression saved this"
 * when only the silent-prune path executed. While #1311 is unshipped,
 * the popup unconditionally surfaces "silently-pruned path still
 * active" — the auto-detect invariant from the design doc.
 */
window.openContextBreakdownPopup = async function () {
    const { Modal } = await import('./ui.js');
    const sessionId = state.currentSessionId || null;
    if (!sessionId) {
        Modal.show({
            title: 'Context breakdown',
            content: '<p style="margin:0;color:var(--text-secondary)">No active conversation — context breakdown is only meaningful within a session.</p>',
        });
        return;
    }

    Modal.show({
        title: 'Context breakdown',
        content: '<p style="margin:0;color:var(--text-secondary)">Loading…</p>',
    });

    let status;
    try {
        status = await API.getContextStatus(sessionId, { full: true });
    } catch (e) {
        // Backend error.detail from a non-OK response is surfaced as
        // ``e.message`` by the API client; that string is **not**
        // trusted (could contain HTML from a proxy / framework error
        // page). Escape before injecting (codex round 2 residual P1).
        Modal.show({
            title: 'Context breakdown',
            content: `<p style="margin:0;color:var(--error)">Could not load breakdown: ${_esc(e && e.message ? e.message : e)}</p>`,
        });
        return;
    }

    Modal.show({
        title: 'Context breakdown',
        content: renderContextBreakdown(status),
        buttons: [
            ...((status.compression_recommended)
                ? [{
                    label: 'Save older turns into a durable note (!compress)',
                    type: 'primary',
                    onClick: () => {
                        Modal.hide();
                        window.compressContext();
                    },
                }]
                : []),
            { label: 'Close', type: 'secondary', onClick: () => Modal.hide() },
        ],
    });
};

/** Escape a string for safe injection into HTML.
 *
 * Codex round 1 P1 caught a real XSS surface: backend strings
 * (subsection names, exception messages in ``breakdown.notes``, model
 * name) were being interpolated into a template literal handed to
 * ``Modal.show``, which calls ``innerHTML``. Any string that contains
 * ``<script>`` or event-handler-bearing tags would execute. All
 * dynamic text in ``renderContextBreakdown`` now flows through this.
 */
function _esc(value) {
    return String(value == null ? '' : value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

/** Render the layered breakdown HTML for ``openContextBreakdownPopup``. */
function renderContextBreakdown(status) {
    const fmt = (n) => Number(n || 0).toLocaleString();
    const breakdown = status.breakdown;
    if (!breakdown) {
        return '<p style="margin:0;color:var(--text-secondary)">Breakdown unavailable for this session.</p>';
    }
    const sections = breakdown.sections || {};
    const total = breakdown.total_measured || 0;
    const budget = breakdown.total_budget || 1;
    const pct = (n) => ((n / budget) * 100).toFixed(1);

    // ``badge`` text is hard-coded by the renderer (never user-supplied)
    // — colors come from the renderer too. Safe to inline.
    const badge = (text, color) => `<span style="
        font-size: 0.625rem; font-weight: 600; padding: 0.125rem 0.375rem;
        border-radius: 999px; background: ${color}; color: white;
        margin-left: 0.5rem;">${_esc(text)}</span>`;

    // ``sectionRow`` ``name`` is hard-coded by callers; ``extras`` is
    // assembled from safe badge() output + escaped fragments; ``warning``
    // is renderer-supplied static text. Tokens are coerced to Number.
    const sectionRow = (name, tokens, extras = '', warning = '') => `
        <div style="display:flex; justify-content:space-between; align-items:center; padding:0.4rem 0; border-bottom:1px solid var(--border-color);">
            <div>
                <span style="font-weight:500">${name}</span>
                ${extras}
                ${warning ? `<div style="font-size:0.7rem;color:#f97316;margin-top:0.15rem">${_esc(warning)}</div>` : ''}
            </div>
            <div style="font-variant-numeric: tabular-nums; color: var(--text-secondary);">
                <span style="color:var(--text-primary); font-weight:500">${fmt(tokens)}</span>
                <span style="margin-left:0.5rem; font-size:0.75rem">(${pct(tokens)}%)</span>
            </div>
        </div>`;

    // System sub-rows. Mandatory vs optional split per Emma's
    // taxonomy: anything in MANDATORY_SYSTEM_SUBSECTIONS (from B) is
    // marked non-borrowable.
    const MANDATORY = new Set(['constitution', 'soul', 'bootstrap_agents', 'state_of_mind']);
    const sys = sections.system || {};
    const sysSubs = (sys.subsections || []).map(s => `
        <div style="display:flex; justify-content:space-between; padding:0.2rem 0 0.2rem 1.5rem; font-size:0.85rem; color:var(--text-secondary)">
            <span>
                ${_esc(s.name)}
                ${MANDATORY.has(s.name) ? badge('mandatory', '#7c3aed') : badge('optional', '#64748b')}
            </span>
            <span style="font-variant-numeric: tabular-nums">${fmt(s.tokens)}</span>
        </div>
    `).join('');

    const tools = sections.tools || {};
    const toolsBadge = (tools.tokens || 0) > 0 ? badge('estimated', '#0891b2') : badge('not counted', '#64748b');

    const hist = sections.history || {};
    const histExtras = `<div style="font-size:0.7rem;color:var(--text-secondary);margin-top:0.15rem">
        ${fmt(hist.messages_kept_after_pruning || 0)} of ${fmt(hist.messages_total || 0)} messages kept after pruning ·
        raw ${fmt(hist.raw_tokens || 0)} tokens
    </div>`;
    // Per Emma's canonical taxonomy and the design doc's UI honesty
    // invariant, the conversation row can show four state badges
    // depending on what the section reports. ``pending fold`` and
    // ``failed fold`` are reserved for C/#1311 (durable salvage); the
    // slots render unconditionally so the popup is ready when C ships,
    // and a "silently-pruned path still active" warning fires while
    // C is unshipped (auto-detect invariant).
    // C / #1311: salvage-state badges come from
    // ``history.salvages`` which the endpoint attaches once C's
    // feature flag is enabled. Until then ``hist.salvages`` is
    // absent (or all-zero) and the badge row stays empty.
    const histBadges = [];
    const salv = (hist && hist.salvages) || {};
    if (salv.pointer_only_count) {
        histBadges.push(badge(`pointer-only salvage · ${salv.pointer_only_count}`, '#0891b2'));
    }
    if (salv.pointer_only_terminal_count) {
        histBadges.push(badge(`pointer-only — summary delayed · ${salv.pointer_only_terminal_count}`, '#0e7490'));
    }
    if (salv.pending_count) {
        histBadges.push(badge(`pending fold · ${salv.pending_count}`, '#f97316'));
    }
    if (salv.folded_count) {
        histBadges.push(badge(`folded · ${salv.folded_count}`, '#16a34a'));
    }
    if (salv.failed_count) {
        histBadges.push(badge(`failed fold · ${salv.failed_count}`, '#dc2626'));
    }
    // Legacy slot names retained for back-compat with D's existing
    // test fixtures (hist.pending_fold / hist.failed_fold). When the
    // backend rolls over fully to `history.salvages`, the legacy
    // slots will be unused and these two lines become dead code.
    if (hist.pending_fold) histBadges.push(badge('pending fold', '#f97316'));
    if (hist.failed_fold) histBadges.push(badge('failed fold', '#dc2626'));
    // Pre-C boundary annotation (Emma 2026-05-21 refinement on
    // question (a) — Option 1, no backfill, surfaces the boundary
    // honestly so the operator can see where salvage history begins).
    const preCNote = salv.pre_c_boundary_at
        ? `<div style="font-size:0.7rem;color:var(--text-secondary);margin-top:0.15rem">Salvage history begins ${_esc(salv.pre_c_boundary_at)} — older silently-pruned spans reachable via <code>!context restore</code> with <code>include_excluded=True</code> but were not salvaged at prune time.</div>`
        : '';
    const histExtrasFull = histBadges.join('') + histExtras + preCNote;
    // Compose the warning row. Two independent triggers:
    //   1. Legacy silent-prune still active (feature flag off / pre-C).
    //   2. Summariser falling behind: pending_count above the
    //      warn-threshold the backend surfaces (Emma 2026-05-21
    //      refinement on back-pressure).
    const warningParts = [];
    if (status.silently_pruned_path_active) {
        warningParts.push(
            'silently-pruned path still active — older messages may have been dropped without a durable summary (until #1311 ships)'
        );
    }
    const warnThreshold = salv.warn_threshold || 10;
    if (salv.pending_count && salv.pending_count > warnThreshold) {
        warningParts.push(
            `Summary worker is falling behind — ${salv.pending_count} spans waiting; older ones may surface as pointer-only-terminal. Salvage is still durable; only the summary is delayed.`
        );
    }
    const histWarning = warningParts.join(' · ');

    const ep = sections.episodes || {};
    const epExtras = (ep.count || 0) > 0
        ? `<span style="font-size:0.75rem;color:var(--text-secondary)"> · ${ep.count} ${ep.count === 1 ? 'episode' : 'episodes'}</span>`
        : '';

    const mem = sections.memories || {};
    const memBadge = mem.wired ? '' : badge('not counted', '#64748b');
    const memExcluded = mem.excluded ? badge('excluded — over budget', '#dc2626') : '';
    const memExtras = (mem.wired
        ? `<span style="font-size:0.75rem;color:var(--text-secondary)"> · ${mem.count || 0} memories</span>`
        : memBadge) + memExcluded;

    const rag = sections.rag || {};
    const ragBadge = rag.skipped
        ? badge('skipped (cheap poll)', '#64748b')
        : (rag.excluded ? badge('excluded — over budget', '#dc2626') : badge('estimated', '#0891b2'));
    // Codex round 1 P2 caught the empty-query gap: the popup's
    // ``full=true`` call runs RAG against the last user turn (so the
    // figure matches what the next LLM turn would see). If we couldn't
    // reach a last-user-turn at request time, the endpoint passes an
    // empty query and the popup labels the row to avoid overstating
    // accuracy. ``rag.query_used_label`` is the explicit annotation
    // the endpoint surfaces; rendered only when present.
    const ragQueryLabel = rag.query_used_label
        ? `<div style="font-size:0.7rem;color:var(--text-secondary);margin-top:0.15rem">${_esc(rag.query_used_label)}</div>`
        : '';
    const ragExtras = ragBadge + (rag.chunks ? `<span style="font-size:0.75rem;color:var(--text-secondary)"> · ${rag.chunks} chunks</span>` : '') + ragQueryLabel;

    const dyn = sections.dynamic_context_overhead || {};
    const overheadRow = (dyn.applied)
        ? sectionRow('Reserve / Overhead — &lt;retrieved_context&gt; envelope', dyn.tokens)
        : '';

    const dynamicSection = `
        ${sectionRow('System / Governance', sys.tokens || 0)}
        ${sysSubs}
        ${sectionRow('Tools', tools.tokens || 0, toolsBadge + (tools.count ? `<span style="font-size:0.75rem;color:var(--text-secondary)"> · ${tools.count} ${tools.count === 1 ? 'tool' : 'tools'}</span>` : ''))}
        ${sectionRow('Conversation', hist.tokens || 0, histExtrasFull, histWarning)}
        ${sectionRow('Memories — episodes', ep.tokens || 0, epExtras)}
        ${sectionRow('Memories — retrieved', mem.tokens || 0, memExtras)}
        ${sectionRow('Retrieval / RAG', rag.tokens || 0, ragExtras)}
        ${overheadRow}
    `;

    const notes = (breakdown.notes || []).filter(Boolean);
    const notesBlock = notes.length
        ? `<details style="margin-top:1rem; font-size:0.8rem; color:var(--text-secondary)">
              <summary style="cursor:pointer">Measurement notes (${notes.length})</summary>
              <ul style="margin:0.5rem 0 0; padding-left:1.25rem">${notes.map(n => `<li>${_esc(n)}</li>`).join('')}</ul>
           </details>`
        : '';

    // Soft-session note: Emma's review explicitly asked to surface
    // that Kestrel sessions are tag filters on continuous agent
    // memory, not hard threads.
    const sessionNote = `
        <div style="margin-top:1rem; padding:0.75rem; background:var(--bg-tertiary, rgba(0,0,0,0.05)); border-radius:8px; font-size:0.75rem; color:var(--text-secondary)">
            This is one session's slice of context. Kestrel agents carry continuous cross-session episodic + semantic memory — sessions are tag filters on the same conversation store, not hard threads.
        </div>
    `;

    // #1503: Route per-turn cap section. Names the cap that the existing
    // whole-window number is already measuring against on capped routes
    // (TokenCounter returns the cap as context_limit on routes like
    // ``openai:plan``), and surfaces the actionable knob the operator
    // can raise. Hidden when no route cap applies.
    const rc = status.route_cap;
    let routeCapBlock = '';
    if (rc && rc.cap_tokens) {
        const rcUtil = Number(rc.utilization_percent || 0);
        let rcColor;
        if (rcUtil < 50) rcColor = '#22c55e';
        else if (rcUtil < 80) rcColor = '#eab308';
        else if (rcUtil < 95) rcColor = '#f97316';
        else rcColor = '#ef4444';
        const knobLine = rc.knob
            ? `Raise via <code>${_esc(rc.knob)}</code> env var, or <code>[llm.route_context_caps]</code> in <code>kestrel.toml</code>.`
            : `Raise via <code>[llm.route_context_caps]</code> in <code>kestrel.toml</code>.`;
        const headroomNote = rcUtil >= 95
            ? `<div style="font-size:0.75rem;color:${rcColor};margin-top:0.25rem;font-weight:500">Next turn will likely bust this cap and stall upstream.</div>`
            : '';
        // Popup runs full=true → includes_rag=true, so projection is
        // accurate. Surface the affirmation so users know the popup
        // figure isn't the same floor the pill shows.
        const ragNote = rc.includes_rag
            ? `<span style="color:#22c55e">✓ includes retrieval</span>`
            : `<span style="color:var(--text-secondary)">projection excludes retrieval — open the popup once for a RAG-included figure</span>`;
        routeCapBlock = `
            <div style="margin-bottom:0.5rem;padding:0.75rem;background:${rcColor}15;border-left:3px solid ${rcColor};border-radius:4px">
                <div style="display:flex;justify-content:space-between;align-items:baseline">
                    <div>
                        <div style="font-size:0.7rem;color:var(--text-secondary);text-transform:uppercase">Route per-turn cap</div>
                        <div style="font-size:1.1rem;font-weight:600;color:${rcColor}">${_esc(rc.route || 'route')} — ${rcUtil.toFixed(1)}%</div>
                    </div>
                    <div style="text-align:right;color:var(--text-secondary);font-size:0.85rem">
                        <div>${fmt(rc.projected_turn_payload)} / ${fmt(rc.cap_tokens)} tokens</div>
                        <div style="font-size:0.7rem;margin-top:0.15rem">${fmt(rc.headroom_tokens)} tokens of headroom</div>
                    </div>
                </div>
                ${headroomNote}
                <div style="font-size:0.7rem;margin-top:0.4rem">${ragNote}</div>
                <div style="font-size:0.7rem;color:var(--text-secondary);margin-top:0.4rem">
                    This cap is below the model's full context window — it's the binding constraint for the next turn on this route. ${knobLine}
                </div>
            </div>
        `;
    }

    return `
        <div style="font-size:0.875rem">
            <div style="display:flex; justify-content:space-between; align-items:baseline; padding-bottom:0.75rem; border-bottom:2px solid var(--border-color); margin-bottom:0.5rem">
                <div>
                    <div style="font-size:0.7rem; color:var(--text-secondary); text-transform:uppercase">Window utilization</div>
                    <div style="font-size:1.5rem; font-weight:600">${(breakdown.utilization_percent || 0).toFixed(1)}%</div>
                </div>
                <div style="text-align:right; color:var(--text-secondary)">
                    <div>${fmt(total)} / ${fmt(budget)} tokens</div>
                    <div style="font-size:0.7rem; margin-top:0.15rem">${_esc(breakdown.model || '')} · reserve ${fmt(breakdown.response_reserve || 0)} · limit ${fmt(breakdown.context_limit || 0)}</div>
                </div>
            </div>
            ${routeCapBlock}
            ${dynamicSection}
            ${notesBlock}
            ${sessionNote}
        </div>
    `;
}

/**
 * Compress context by sending !compress command.
 * Called from the compress button in the context status indicator.
 */
window.compressContext = async function() {
    if (!messageInput) return;

    // Send the compress command
    const originalValue = messageInput.value;
    messageInput.value = '!compress';
    await sendMessage();

    // Restore original input if user was typing something
    if (originalValue && !originalValue.startsWith('!')) {
        messageInput.value = originalValue;
    }
}

// ============================================================================
// Message Rendering
// ============================================================================

/**
 * Append a streaming message bubble. Optional paneElement targets a
 * specific agent's detached pane; without it, defaults to the visible
 * (mounted) agent's pane so single-agent and voice-pipe call sites
 * continue to work without modification. Scroll-syncs only when the
 * write lands on the currently-mounted pane.
 */
export function addMessageStreaming(role, paneElement = null) {
    const target = resolvePaneElement(paneElement);
    const div = document.createElement('div');
    div.className = `message ${role === 'user' ? 'user-message' : 'agent-message'}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content streaming';
    contentDiv.textContent = '';

    div.appendChild(contentDiv);
    if (target) {
        target.appendChild(div);
        const c = getChatContainer();
        if (c && target.parentNode === c) {
            c.scrollTop = c.scrollHeight;
        }
    }

    return div;
}

function renderThinkingBubbles(thinkingItems = []) {
    if (!thinkingItems || thinkingItems.length === 0) return '';
    return thinkingItems.map((item, idx) => {
        const content = typeof item === 'string' ? item : (item.content || '');
        const providerText = typeof item === 'object' && item.provider ? ` · ${item.provider}` : '';
        const provider = escapeThinkingLabel(providerText);
        const rendered = renderStreamingMarkdown(content);
        return `<details class="thinking-bubble">
            <summary>Thinking ${idx + 1}${provider}</summary>
            <div class="thinking-bubble-content">${rendered}</div>
        </details>`;
    }).join('');
}

function escapeThinkingLabel(text) {
    return String(text || '').replace(/[&<>"']/g, (ch) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
    }[ch]));
}

export function updateStreamingMessage(msgDiv, content, paneElement = null, thinkingItems = []) {
    const contentDiv = msgDiv.querySelector('.message-content');
    if (contentDiv) {
        const thinkingHtml = renderThinkingBubbles(thinkingItems);
        const split = splitToolActivity(content);

        if (split.hasToolActivity) {
            contentDiv.innerHTML = `${thinkingHtml}${renderStreamingResponseSections(split, content)}`;
            highlightCodeBlocks(contentDiv, true);
        } else {
            // No tool indicators - regular markdown
            contentDiv.innerHTML = `${thinkingHtml}${renderStreamingMarkdown(content)}`;
            highlightCodeBlocks(contentDiv, true);
        }

        // Scroll-sync only when this msgDiv is in the live viewport;
        // detached panes update their `scrollPos` lazily on remount.
        const target = paneElement || msgDiv.parentNode;
        const c = getChatContainer();
        if (c && target && target.parentNode === c) {
            c.scrollTop = c.scrollHeight;
        }
    }
}

/**
 * Final-render a streaming bubble. `paneOrElement` accepts either a
 * pane element directly or a full pane object — passing the pane lets
 * us defer mermaid rendering when the pane is currently detached. The
 * caller in sendMessage passes the pane; voice/ui.js passes nothing
 * and gets the visible-pane default (mermaid runs immediately because
 * the pane is already mounted).
 *
 * Note: this no longer calls checkForModelChange(). The shared model
 * selector is a global singleton — letting helpers mutate it from a
 * background-pane finalize would corrupt the visible agent's UI.
 * sendMessage gates that call on isCurrentVisible() instead.
 */
export async function finalizeStreamingMessage(msgDiv, content, paneOrElement = null) {
    const contentDiv = msgDiv.querySelector('.message-content');
    if (!contentDiv) return;

    contentDiv.classList.remove('streaming');

    // Detect which pane this write lands on so we can choose between
    // running the full markdown finalization (incl. mermaid) now or
    // deferring the mermaid pass until the pane is mounted. mermaid
    // 10+ technically renders on detached nodes, but we've seen
    // flakiness — be conservative and defer.
    let pane = null;
    let paneEl = null;
    if (paneOrElement && typeof paneOrElement === 'object') {
        if (paneOrElement.element) {
            pane = paneOrElement;
            paneEl = paneOrElement.element;
        } else if (paneOrElement.appendChild) {
            paneEl = paneOrElement;
        }
    }
    if (!paneEl) paneEl = msgDiv.parentNode;
    const c = getChatContainer();
    const mounted = !!(c && paneEl && paneEl.parentNode === c);

    await finalizeAgentContent(contentDiv, content);
    const thinkingItems = pane && pane.thinkingItems ? pane.thinkingItems : [];
    if (thinkingItems.length) {
        contentDiv.innerHTML = `${renderThinkingBubbles(thinkingItems)}${contentDiv.innerHTML}`;
        highlightCodeBlocks(contentDiv, true);
    }
    if (!mounted && pane && /```mermaid/.test(content)) {
        // Mark the pane so mountChatPane re-runs the mermaid pass.
        pane.hasUnrenderedMermaid = true;
    }

    if (mounted && c) c.scrollTop = c.scrollHeight;
}

/**
 * Append a non-streaming message bubble. See addMessageStreaming for
 * the paneElement contract; the same default applies. checkForModelChange
 * is intentionally NOT invoked here — see finalizeStreamingMessage.
 */
export async function addMessage(role, content, paneElement = null) {
    const target = resolvePaneElement(paneElement);
    const div = document.createElement('div');
    div.className = `message ${role === 'user' ? 'user-message' : 'agent-message'}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    if (role === 'user') {
        // User messages: just escape HTML and convert newlines to <br>
        const escaped = content.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        contentDiv.innerHTML = escaped.replace(/\n/g, '<br>');
    } else {
        // Agent messages: use shared markdown utilities
        await finalizeAgentContent(contentDiv, content);
    }

    div.appendChild(contentDiv);
    if (target) target.appendChild(div);

    const c = getChatContainer();
    if (c && target && target.parentNode === c) {
        c.scrollTop = c.scrollHeight;
    }
}

// ============================================================================
// Model Management (uses SharedModelSelector from /shared/model-selector/)
// ============================================================================

/**
 * Initialize the shared model selector component
 */
export async function loadModels() {
    // #879: model selector lives in the chat header — skip when chat is off.
    if (!API.hasCapability('chat')) return;
    // Check if SharedModelSelector is available (loaded via script tag)
    if (!window.SharedModelSelector) {
        console.error('SharedModelSelector not loaded. Include /shared/model-selector/index.js');
        return;
    }

    // Blank the dropdowns IMMEDIATELY before building the new selector. Without
    // this, the previous agent's options remain in the DOM while the async
    // /api/models + /api/model/current round-trips for the new agent complete,
    // and the user briefly sees the previous agent's vendor/model (the "flash"
    // Jason saw when switching agents).
    for (const id of ['provider-selector', 'route-selector', 'model-selector']) {
        const el = document.getElementById(id);
        if (el) {
            el.innerHTML = '<option value="">Loading…</option>';
            el.value = '';
            if (id === 'route-selector') el.style.display = 'none';
        }
    }

    // Create the shared model selector instance
    // Use API.buildAgentUrl() for proper multi_agent routing and pass auth headers
    sharedModelSelector = new window.SharedModelSelector({
        providerSelectId: 'provider-selector',
        routeSelectId: 'route-selector',
        modelSelectId: 'model-selector',
        apiEndpoint: API.buildAgentUrl('/api/models'),
        currentModelEndpoint: API.buildAgentUrl('/api/model/current'),
        storagePrefix: `kestrel_${API.getHostAgent() || 'default'}`,
        getAuthHeader: async () => await API.applyAuth({}),
        onModelChange: async (vendor, model, isInitialLoad, route) => {
            if (isInitialLoad) return;

            // Direct REST call to /api/model/set — NOT a chat message. The old
            // flow (write "!model-set ..." to messageInput, sendMessage) went
            // through the chat stream, so switching agents mid-stream left the
            // response's MODEL_CHANGED marker to land on whatever selector was
            // currently visible, corrupting state across agents.
            //
            // We capture the host agent at dispatch time and discard the
            // response if the user has switched agents before it lands.
            const dispatchAgent = API.getHostAgent();

            // Persist vendor/route/model in chat state immediately so the UI
            // reflects the user's intent without waiting for the round-trip.
            state.selectedModel = model;
            state.selectedProvider = vendor;    // legacy name retained
            state.selectedVendor = vendor;
            state.selectedRoute = route || null;

            const body = { vendor, model };
            if (route) body.route = route;
            const headers = await API.applyAuth({ 'Content-Type': 'application/json' });

            try {
                const resp = await fetch(API.buildAgentUrl('/api/model/set'), {
                    method: 'POST',
                    headers,
                    body: JSON.stringify(body),
                });
                if (!resp.ok) {
                    console.warn(`set model failed (${dispatchAgent}): HTTP ${resp.status}`);
                    return;
                }
                if (API.getHostAgent() !== dispatchAgent) {
                    // User switched agents before the server acked. Silently
                    // succeed — the change on dispatchAgent is persisted; don't
                    // overwrite the NEW agent's state.
                    return;
                }
                // No UI update needed here: the selector already reflects the
                // user's click; the server is the source of truth from here on.
            } catch (e) {
                console.warn(`set model request error (${dispatchAgent}):`, e);
            }
        }
    });

    // Initialize - loads models, binds events, syncs with server
    await sharedModelSelector.init();

    // Expose globally so other modules (identity.js) can auto-switch on privacy change
    window._sharedModelSelector = sharedModelSelector;

    // Update state with initial selection (both provider and model)
    const selection = sharedModelSelector.getSelection();
    state.selectedModel = selection.model;
    state.selectedProvider = selection.provider;

    // Discover the voice route's Realtime model and mark it unpickable in the
    // dropdown.  The mic button owns this model (see #1371) — the user can see
    // it exists but should not pick it manually for text chat.  Fire-and-forget:
    // a missing voice feature or auth failure just means no unpickable models
    // (the selector stays usable).
    (async () => {
        try {
            const headers = await API.applyAuth({});
            const resp = await fetch(
                API.buildAgentUrl('/voice/realtime/route'),
                { headers },
            );
            if (!resp.ok) return;
            const route = await resp.json();
            if (route?.voice_model) {
                sharedModelSelector.setUnpickableModels([route.voice_model]);
            }
        } catch (_) {
            // Voice not configured / network noise — selector stays usable.
        }
    })();
}

/**
 * Check for model change events from agent responses
 * and update the selector accordingly
 */
function checkForModelChange(content) {
    if (sharedModelSelector) {
        const changed = sharedModelSelector.checkForModelChange(content);
        if (changed) {
            // Update state with both provider and model
            const selection = sharedModelSelector.getSelection();
            state.selectedModel = selection.model;
            state.selectedProvider = selection.provider;

            // Visual feedback on model selector
            const modelSelect = document.getElementById('model-selector');
            if (modelSelect) {
                modelSelect.style.transition = 'background-color 0.3s';
                modelSelect.style.backgroundColor = '#22c55e33';
                setTimeout(() => {
                    modelSelect.style.backgroundColor = '';
                }, 1000);
            }
        }
    }
}

// ============================================================================
// Command Autocomplete
// ============================================================================

function handleInput(e) {
    const value = e.target.value;

    if (value === '!' || value.match(/\s!$/)) {
        showCommandAutocomplete('');
    } else if (value.startsWith('!') && !value.includes(' ')) {
        showCommandAutocomplete(value.substring(1));
    } else {
        hideCommandAutocomplete();
    }
}

function handleKeydown(e) {
    const dropdown = document.getElementById('command-autocomplete');

    if (dropdown) {
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            navigateAutocomplete(1);
            return;
        }
        if (e.key === 'ArrowUp') {
            e.preventDefault();
            navigateAutocomplete(-1);
            return;
        }
        if (e.key === 'Tab' || (e.key === 'Enter' && autocompleteSelectedIndex >= 0)) {
            e.preventDefault();
            if (selectHighlightedCommand()) return;
            if (e.key === 'Tab') {
                highlightCommand(0);
                selectHighlightedCommand();
                return;
            }
        }
        if (e.key === 'Escape') {
            e.preventDefault();
            hideCommandAutocomplete();
            return;
        }
    }

    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

function showCommandAutocomplete(filter = '') {
    hideCommandAutocomplete();

    if (!messageInput) return;

    const filterLower = filter.toLowerCase();
    const filteredCommands = AGENT_COMMANDS.filter(c =>
        c.cmd.toLowerCase().includes(filterLower) ||
        c.description.toLowerCase().includes(filterLower)
    );

    if (filteredCommands.length === 0) return;

    const rect = messageInput.getBoundingClientRect();

    const dropdown = document.createElement('div');
    dropdown.id = 'command-autocomplete';
    dropdown.style.cssText = `
        position: fixed;
        bottom: ${window.innerHeight - rect.top + 8}px;
        left: ${rect.left}px;
        width: ${rect.width}px;
        max-height: 300px;
        overflow-y: auto;
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        box-shadow: 0 -4px 20px rgba(0,0,0,0.15);
        z-index: 1000;
    `;

    dropdown.innerHTML = `
        <div style="padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--border-color); font-size: 0.75rem; color: var(--text-tertiary);">
            Commands ${filter ? `matching "${filter}"` : '(type to filter)'}
        </div>
        ${filteredCommands.map((cmd, idx) => `
            <div class="command-option" data-index="${idx}" data-cmd="${cmd.cmd}" style="
                padding: 0.625rem 0.75rem;
                cursor: pointer;
                display: flex;
                align-items: center;
                gap: 0.75rem;
                transition: background 0.1s;
            ">
                <code style="
                    background: var(--bg-tertiary);
                    padding: 0.25rem 0.5rem;
                    border-radius: 4px;
                    font-size: 0.8rem;
                    font-weight: 500;
                    color: var(--accent-color);
                    white-space: nowrap;
                ">${cmd.cmd}${cmd.args ? ' <span style="color: var(--text-tertiary);">' + cmd.args + '</span>' : ''}</code>
                <span style="font-size: 0.8rem; color: var(--text-secondary); flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${cmd.description}</span>
            </div>
        `).join('')}
    `;

    dropdown.querySelectorAll('.command-option').forEach(opt => {
        opt.addEventListener('click', () => selectCommand(opt.dataset.cmd));
        opt.addEventListener('mouseover', () => {
            highlightCommand(parseInt(opt.dataset.index));
        });
    });

    document.body.appendChild(dropdown);
    autocompleteSelectedIndex = -1;
}

function hideCommandAutocomplete() {
    const existing = document.getElementById('command-autocomplete');
    if (existing) existing.remove();
    autocompleteSelectedIndex = -1;
}

function highlightCommand(index) {
    const dropdown = document.getElementById('command-autocomplete');
    if (!dropdown) return;

    const options = dropdown.querySelectorAll('.command-option');
    options.forEach((opt, i) => {
        opt.style.background = i === index ? 'var(--bg-tertiary)' : 'transparent';
    });
    autocompleteSelectedIndex = index;
}

function selectCommand(cmd) {
    if (!messageInput) return;

    const cmdInfo = AGENT_COMMANDS.find(c => c.cmd === cmd);

    messageInput.value = cmd + ' ';
    messageInput.focus();
    hideCommandAutocomplete();

    if (cmdInfo && cmdInfo.args && cmdInfo.args.startsWith('<')) {
        messageInput.setSelectionRange(messageInput.value.length, messageInput.value.length);
    }
}

function navigateAutocomplete(direction) {
    const dropdown = document.getElementById('command-autocomplete');
    if (!dropdown) return false;

    const options = dropdown.querySelectorAll('.command-option');
    if (options.length === 0) return false;

    let newIndex = autocompleteSelectedIndex + direction;
    if (newIndex < 0) newIndex = options.length - 1;
    if (newIndex >= options.length) newIndex = 0;

    highlightCommand(newIndex);
    options[newIndex].scrollIntoView({ block: 'nearest' });

    return true;
}

function selectHighlightedCommand() {
    const dropdown = document.getElementById('command-autocomplete');
    if (!dropdown || autocompleteSelectedIndex < 0) return false;

    const options = dropdown.querySelectorAll('.command-option');
    if (options[autocompleteSelectedIndex]) {
        selectCommand(options[autocompleteSelectedIndex].dataset.cmd);
        return true;
    }
    return false;
}

// ============================================================================
// New Chat / Clear Chat
// ============================================================================

/**
 * Clear the chat and start fresh (called via onclick from HTML)
 */
window.clearChat = function() {
    if (!chatContainer) return;

    // Clear ONLY the visible agent's pane and bump that agent's
    // pane-local generation. Other agents' panes (and their in-flight
    // streams) are untouched.
    wipeAgentChatPane(API.getHostAgent(), `
        <div class="message agent-message">
            <div class="message-content">
                <p>Hello! I am your Kestrel AI agent, bound by the Kestrel Constitution to be your truthful and honorable assistant. How can I help you today?</p>
            </div>
        </div>
    `);

    // Clear message input
    if (messageInput) {
        messageInput.value = '';
        messageInput.focus();
    }

    // Optionally start a new conversation in the backend
    import('./history.js').then(module => {
        if (window.startNewConversation) {
            window.startNewConversation();
        }
    }).catch(() => {
        // history.js not loaded, that's OK
    });

    Toast.success('Chat cleared');
};
