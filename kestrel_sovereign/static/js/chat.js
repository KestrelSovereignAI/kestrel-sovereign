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

export function splitToolActivity(content) {
    const text = String(content || '');
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
    sendButton?.addEventListener('click', sendMessage);

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
    if (messageInput) messageInput.disabled = busy;
    if (sendButton) sendButton.disabled = busy;
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
// Send Message
// ============================================================================

export async function sendMessage() {
    const text = messageInput.value.trim();

    // Capture the dispatch agent and pin the dispatch's pane up front,
    // BEFORE any await — the user could mount a different agent's pane
    // before the first stream chunk arrives, and the user's typed text
    // must always land in the agent it was dispatched against.
    const dispatchAgent = API.getHostAgent();
    if (!text || state.waitingAgents.has(dispatchAgent)) return;

    const pane = getOrCreateChatPane(dispatchAgent);
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
    messageInput.value = '';
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
        const { message_count, utilization_percent, status: contextState, warnings } = status;

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

        contextStatusElement.style.color = color;

        // Show compress button when utilization is 70%+
        const showCompress = utilization_percent >= 70;
        const compressButton = showCompress
            ? `<button onclick="window.compressContext()" style="
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

        contextStatusElement.innerHTML = `
            <span title="Context window: ${message_count} messages, ${utilization_percent.toFixed(1)}% used${warnings.length ? '\nWarnings: ' + warnings.join(', ') : ''}">
                ${icon} ${message_count} msgs · ${utilization_percent.toFixed(0)}%${compressButton}
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
