/**
 * Kestrel Sovereign Console - Chat Component
 * Chat panel and command autocomplete
 */

import API from './api.js';
import { state, AGENT_COMMANDS, Toast } from './ui.js';

// Shared markdown utilities - loaded via script tag before this module
const {
    renderMarkdown,
    renderStreamingMarkdown,
    highlightCodeBlocks,
    renderMermaidDiagrams,
    finalizeMarkdown
} = window.SharedMarkdown;

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

// UI generation counter. Bumps every time `selectAgent()` calls
// `bumpUiGeneration()` — i.e. every time the chat pane is wiped and
// rebuilt for a (re)selected agent. A stream captures this value at
// dispatch time and treats its UI as "stale" the moment the global
// counter moves past it. Agent-equality alone can't catch this:
// A→B→A re-enters with `selectedHostAgent === dispatchAgent` but
// `chatContainer.innerHTML = ''` already orphaned the old msgDiv.
let uiGeneration = 0;

export function bumpUiGeneration() {
    uiGeneration++;
}

// Test seam — frontend tests need to inspect/reset the counter without
// reaching through `selectAgent`'s DOM dependencies.
export function _getUiGeneration() {
    return uiGeneration;
}

/**
 * Canonical chat-pane wipe-and-rebuild. Bumps the UI generation counter
 * BEFORE the DOM mutation so any in-flight stream's
 * `dispatchGeneration === uiGeneration` check fails immediately — that
 * stops chunks from painting a now-stale msgDiv during/after the wipe.
 *
 * Every code path that clears or rebuilds the chat container (agent
 * switch, conversation load, new chat, clear chat, soft/hard delete of
 * the active conversation) must go through this helper. Bare
 * `chatContainer.innerHTML = ...` calls leak: the generation counter
 * doesn't move, so a stream dispatched against the old pane keeps
 * thinking it's still current and writes into the freshly-rebuilt pane.
 *
 * The element is looked up fresh rather than relying on the cached
 * `chatContainer` ref because callers may run before `initChat()` has
 * resolved the ref or in contexts where the cache hasn't been populated.
 */
export function wipeChatPane(html = '') {
    bumpUiGeneration();
    const el = chatContainer || document.getElementById('chat-container');
    if (el) el.innerHTML = html;
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
    // - Rookery mode: selectAgent() handles both
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
 */
let reconnectAttempts = 0;
function scheduleReconnect() {
    if (notificationReconnectTimeout) return;

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
 */
function showTaskNotification(message, type) {
    // Reset reconnect attempts on successful notification
    reconnectAttempts = 0;

    // Add as a special notification message in the chat
    if (!chatContainer) return;

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

    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;

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

// ============================================================================
// Stop Request
// ============================================================================

/**
 * Stop the current streaming request.
 * Called when user clicks the Stop button.
 */
async function stopRequest() {
    console.log('Stop button clicked');
    
    // Abort client-side fetch using API's AbortController
    const abortController = API.getStreamAbortController();
    if (abortController) {
        abortController.abort();
    }
    
    // Tell server to stop processing
    try {
        const result = await API.stop(API.getCurrentStreamRequestId());
        console.log('Stop result:', result);
    } catch (e) {
        console.error('Error stopping request:', e);
    }
    
    // Reset UI state for the currently-selected agent only. If the user
    // switched away while a different stream was running, that stream's
    // bookkeeping is cleared by its own sendMessage finally block.
    state.waitingAgents.delete(API.getHostAgent());
    updateThinkingIndicator();
}

// ============================================================================
// Send Message
// ============================================================================

export async function sendMessage() {
    const text = messageInput.value.trim();
    // Capture the agent AND the UI generation this dispatch is bound to.
    // The URL inside streamInvoke() is locked to dispatchAgent at fetch
    // time, so server-side routing is safe. We use dispatchAgent +
    // dispatchGeneration together to gate UI updates so chunks from a
    // dispatch issued before a pane wipe (selectAgent → A→B→A) can't
    // paint a now-orphaned msgDiv in the freshly-rebuilt pane.
    const dispatchAgent = API.getHostAgent();
    const dispatchGeneration = uiGeneration;
    if (!text || state.waitingAgents.has(dispatchAgent)) return;

    await addMessage('user', text);
    messageInput.value = '';
    state.waitingAgents.add(dispatchAgent);
    updateThinkingIndicator();

    // Get current session ID for context-aware conversation
    const sessionId = state.currentSessionId || null;

    // DO NOT send model/provider overrides from the chat UI. The server
    // already knows each agent's persisted mandate (set via POST
    // /api/model/set when the user picked from the dropdown). Sending
    // state.selectedModel/selectedProvider here was a bug: those vars
    // don't update when the user switches agents, so a stale override
    // from the previous agent would silently reroute the new agent's
    // chat to a different model. Source of truth: server mandate.

    // A stream's UI is "current" only if BOTH the active agent matches
    // AND the chat pane hasn't been wiped/rebuilt since dispatch. Agent
    // equality alone misses A→B→A: the pane was wiped on each switch,
    // so the captured msgDiv is detached even though the agent matches.
    const isCurrent = () =>
        API.getHostAgent() === dispatchAgent && uiGeneration === dispatchGeneration;

    try {
        if (state.useStreaming) {
            const msgDiv = addMessageStreaming('agent');
            let fullContent = '';

            try {
                for await (const chunk of API.streamInvoke(text, null, sessionId, null)) {
                    fullContent += chunk;
                    // Skip DOM repaint when the user has switched agents —
                    // chatContainer was wiped on switch, msgDiv is now an
                    // orphan, and updating the shared model selector with
                    // this stream's content would corrupt the new agent's UI.
                    if (isCurrent()) {
                        updateStreamingMessage(msgDiv, fullContent);
                    }
                }
                if (isCurrent()) {
                    await finalizeStreamingMessage(msgDiv, fullContent);
                    await checkForModelChange(fullContent);
                }
            } catch (streamError) {
                if (streamError.message.includes('404') || streamError.message.includes('405')) {
                    console.log('Streaming not available, falling back to standard invoke');
                    state.useStreaming = false;
                    msgDiv.remove();
                    const response = await API.invoke(text, null, sessionId, null);
                    if (isCurrent()) {
                        await addMessage('agent', response.response);
                        await checkForModelChange(response.response);
                    }
                } else {
                    throw streamError;
                }
            }
        } else {
            const response = await API.invoke(text, null, sessionId, null);
            if (isCurrent()) {
                await addMessage('agent', response.response);
                await checkForModelChange(response.response);
            }
        }
    } catch (e) {
        if (isCurrent()) {
            await addMessage('agent', `Error: ${e.message}`);
        } else {
            console.warn(`stream error on ${dispatchAgent} (no longer current):`, e.message);
        }
    } finally {
        state.waitingAgents.delete(dispatchAgent);
        updateThinkingIndicator();
        // Update context status only for the currently-visible agent.
        // Updating from a stale dispatch would query the wrong session.
        if (isCurrent()) {
            updateContextStatus();
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
 * Show a context warning message in the chat.
 */
function showContextWarning(warnings) {
    if (!chatContainer) return;

    // Don't show duplicate warnings
    if (chatContainer.querySelector('.context-warning')) return;

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
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
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

export function addMessageStreaming(role) {
    const div = document.createElement('div');
    div.className = `message ${role === 'user' ? 'user-message' : 'agent-message'}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content streaming';
    contentDiv.textContent = '';

    div.appendChild(contentDiv);
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    return div;
}

export function updateStreamingMessage(msgDiv, content) {
    const contentDiv = msgDiv.querySelector('.message-content');
    if (contentDiv) {
        // Split content into tool activity and response at the --- separator
        const parts = content.split('\n---\n');
        const toolActivity = parts.length > 1 ? parts[0] : '';
        const response = parts.length > 1 ? parts.slice(1).join('\n---\n') : content;

        // Check if content has tool indicators (before separator or if no separator yet)
        const hasToolIndicators = /^[🔧✓❌]/.test(content.trim());
        
        if (hasToolIndicators && !content.includes('\n---\n')) {
            // Still in tool execution phase - show as activity
            const activityHtml = content.split('\n').map(line => {
                if (line.startsWith('🔧')) return `<div class="tool-activity tool-start">${line}</div>`;
                if (line.startsWith('✓')) return `<div class="tool-activity tool-done">${line}</div>`;
                if (line.startsWith('❌')) return `<div class="tool-activity tool-error">${line}</div>`;
                return line;
            }).join('');
            contentDiv.innerHTML = `<div class="tool-activity-container">${activityHtml}</div>`;
        } else if (toolActivity) {
            // Have both tool activity and response
            const activityHtml = toolActivity.split('\n').map(line => {
                if (line.startsWith('🔧')) return `<div class="tool-activity tool-start">${line}</div>`;
                if (line.startsWith('✓')) return `<div class="tool-activity tool-done">${line}</div>`;
                if (line.startsWith('❌')) return `<div class="tool-activity tool-error">${line}</div>`;
                return line;
            }).join('');
            contentDiv.innerHTML = `<div class="tool-activity-container">${activityHtml}</div><div class="response-content">${renderStreamingMarkdown(response)}</div>`;
            highlightCodeBlocks(contentDiv, true);
        } else {
            // No tool indicators - regular markdown
            contentDiv.innerHTML = renderStreamingMarkdown(content);
            highlightCodeBlocks(contentDiv, true);
        }

        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
}

export async function finalizeStreamingMessage(msgDiv, content) {
    const contentDiv = msgDiv.querySelector('.message-content');
    if (contentDiv) {
        contentDiv.classList.remove('streaming');
        // Use shared markdown utilities for full render with mermaid support
        await finalizeMarkdown(contentDiv, content);

        await checkForModelChange(content);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
}

export async function addMessage(role, content) {
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
        await finalizeMarkdown(contentDiv, content);
    }

    div.appendChild(contentDiv);
    chatContainer.appendChild(div);

    if (role === 'agent') {
        await checkForModelChange(content);
    }

    chatContainer.scrollTop = chatContainer.scrollHeight;
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
    // Use API.buildAgentUrl() for proper rookery routing and pass auth headers
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

    // Clear all messages except the welcome message. wipeChatPane() bumps
    // the UI generation so any in-flight stream against this pane gates
    // out before its chunks can paint the welcome view.
    wipeChatPane(`
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
