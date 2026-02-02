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

// ============================================================================
// Initialization
// ============================================================================

export function initChat() {
    chatContainer = document.getElementById('chat-container');
    messageInput = document.getElementById('message-input');
    sendButton = document.getElementById('send-button');
    modelSelector = document.getElementById('model-selector');
    thinkingIndicator = document.getElementById('thinking-indicator');

    // Event listeners
    sendButton?.addEventListener('click', sendMessage);

    // Input events for autocomplete
    messageInput?.addEventListener('input', handleInput);
    messageInput?.addEventListener('keydown', handleKeydown);
    messageInput?.addEventListener('blur', () => {
        setTimeout(hideCommandAutocomplete, 200);
    });

    // Note: Model selector events are now handled by SharedModelSelector in loadModels()

    // Connect to SSE notifications
    connectNotifications();

    // Load initial context status
    updateContextStatus();
}

// ============================================================================
// SSE Task Notifications
// ============================================================================

let notificationEventSource = null;
let notificationReconnectTimeout = null;

/**
 * Connect to the SSE notifications endpoint for real-time task updates.
 * Automatically reconnects on disconnect with exponential backoff.
 */
function connectNotifications() {
    if (notificationEventSource) {
        notificationEventSource.close();
    }

    // Get API key for authentication (EventSource can't send headers)
    const apiKey = API.getApiKey();
    if (!apiKey) {
        console.warn('No API key available for SSE notifications, will retry later');
        scheduleReconnect();
        return;
    }

    try {
        // Pass API key as query parameter since EventSource can't send headers
        notificationEventSource = new EventSource(`/agent/notifications/sse?api_key=${encodeURIComponent(apiKey)}`);

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

function showThinking(show) {
    if (thinkingIndicator) {
        thinkingIndicator.style.display = show ? 'flex' : 'none';
    }
    if (messageInput) messageInput.disabled = show;
    if (sendButton) sendButton.disabled = show;
}

// ============================================================================
// Send Message
// ============================================================================

export async function sendMessage() {
    const text = messageInput.value.trim();
    if (!text || state.isWaiting) return;

    await addMessage('user', text);
    messageInput.value = '';
    state.isWaiting = true;
    showThinking(true);

    // Don't pass model here - it's already set via !model-set command which stores
    // the preference in llm_service._mandate_preference. Passing it again via
    // model_override causes parsing issues with OpenRouter models like "google/gemma-2-27b-it"
    // which get incorrectly split into provider="google" and model="gemma-2-27b-it".

    // Get current session ID for context-aware conversation
    const sessionId = state.currentSessionId || null;

    try {
        if (state.useStreaming) {
            const msgDiv = addMessageStreaming('agent');
            let fullContent = '';

            try {
                for await (const chunk of API.streamInvoke(text, state.selectedModel, sessionId)) {
                    fullContent += chunk;
                    updateStreamingMessage(msgDiv, fullContent);
                }
                await finalizeStreamingMessage(msgDiv, fullContent);
                await checkForModelChange(fullContent);
            } catch (streamError) {
                if (streamError.message.includes('404') || streamError.message.includes('405')) {
                    console.log('Streaming not available, falling back to standard invoke');
                    state.useStreaming = false;
                    msgDiv.remove();
                    const response = await API.invoke(text, state.selectedModel, sessionId);
                    await addMessage('agent', response.response);
                    await checkForModelChange(response.response);
                } else {
                    throw streamError;
                }
            }
        } else {
            const response = await API.invoke(text, state.selectedModel, sessionId);
            await addMessage('agent', response.response);
            await checkForModelChange(response.response);
        }
    } catch (e) {
        await addMessage('agent', `Error: ${e.message}`);
    } finally {
        state.isWaiting = false;
        showThinking(false);
        // Update context status after each message
        updateContextStatus();
    }
}

// ============================================================================
// Context Status Indicator
// ============================================================================

let contextStatusElement = null;

/**
 * Update the context status indicator.
 * Shows messages count and utilization percentage with color coding.
 */
export async function updateContextStatus() {
    try {
        const sessionId = state.currentSessionId || null;
        const status = await API.getContextStatus(sessionId);

        if (!contextStatusElement) {
            createContextStatusElement();
        }

        if (!contextStatusElement) return;

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
            icon = '⚠';
        } else {
            color = '#ef4444';  // red
            icon = '⚠';
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
        <strong>⚠ Context Warning:</strong> ${warnings.join('. ')}
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

function addMessageStreaming(role) {
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

function updateStreamingMessage(msgDiv, content) {
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

async function finalizeStreamingMessage(msgDiv, content) {
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
    // Check if SharedModelSelector is available (loaded via script tag)
    if (!window.SharedModelSelector) {
        console.error('SharedModelSelector not loaded. Include /shared/model-selector/index.js');
        return;
    }

    // Create the shared model selector instance
    sharedModelSelector = new window.SharedModelSelector({
        providerSelectId: 'provider-selector',
        modelSelectId: 'model-selector',
        apiEndpoint: '/api/models',
        currentModelEndpoint: '/api/model/current',
        storagePrefix: 'kestrel',
        onModelChange: async (provider, model) => {
            // Update state
            state.selectedModel = model;

            // Send model-set command to agent with explicit provider
            // Format: !model-set <provider> <model>
            if (messageInput) {
                messageInput.value = `!model-set ${provider} ${model}`;
                await sendMessage();
            }
        }
    });

    // Initialize - loads models, binds events, syncs with server
    await sharedModelSelector.init();

    // Update state with initial selection
    const selection = sharedModelSelector.getSelection();
    state.selectedModel = selection.model;
}

/**
 * Check for model change events from agent responses
 * and update the selector accordingly
 */
function checkForModelChange(content) {
    if (sharedModelSelector) {
        const changed = sharedModelSelector.checkForModelChange(content);
        if (changed) {
            // Update state
            const selection = sharedModelSelector.getSelection();
            state.selectedModel = selection.model;

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

    // Clear all messages except the welcome message
    chatContainer.innerHTML = `
        <div class="message agent-message">
            <div class="message-content">
                <p>Hello! I am your Kestrel AI agent, bound by the Kestrel Constitution to be your truthful and honorable assistant. How can I help you today?</p>
            </div>
        </div>
    `;

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
