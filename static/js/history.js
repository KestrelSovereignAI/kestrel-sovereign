/**
 * Kestrel Sovereign Console - History Module
 * Chat History Browser
 */

import API from './api.js';
import { state, Toast, escapeHtml } from './ui.js';
import { updateContextStatus } from './chat.js';

// ============================================================================
// Chat History Browser
// ============================================================================

state.conversations = null;
state.currentSessionId = null;
state.historyVisible = false;

export async function loadConversationHistory() {
    try {
        const data = await API.getConversations(state.showDecrypted);
        state.conversations = data.conversations;
        renderConversationHistory(data);
    } catch (e) {
        const container = document.getElementById('history-container');
        if (container) {
            container.innerHTML = `<p style="color: var(--error); padding: 1rem;">Failed to load history: ${e.message}</p>`;
        }
    }
}

function formatDateLabel(dateStr) {
    const date = new Date(dateStr);
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);

    if (date.toDateString() === today.toDateString()) {
        return 'Today';
    } else if (date.toDateString() === yesterday.toDateString()) {
        return 'Yesterday';
    } else {
        return dateStr;
    }
}

function renderConversationHistory(data) {
    const container = document.getElementById('history-container');
    if (!container) return;

    if (data.encrypted_at_rest !== undefined) {
        state.encryptedAtRest = data.encrypted_at_rest;
    }

    if (!data.conversations || data.conversations.length === 0) {
        container.innerHTML = `
            <p style="color: var(--text-secondary); text-align: center; padding: 2rem;">
                No conversation history found. Start chatting to create some!
            </p>
        `;
        return;
    }

    const groupedByDate = {};
    data.conversations.forEach(conv => {
        const date = new Date(conv.started_at);
        const dateKey = date.toLocaleDateString();
        if (!groupedByDate[dateKey]) {
            groupedByDate[dateKey] = [];
        }
        groupedByDate[dateKey].push(conv);
    });

    container.innerHTML = `
        ${state.encryptedAtRest ? `
        <button id="encryption-toggle" onclick="toggleEncryptionView()" style="
            width: 100%;
            padding: 0.5rem 0.75rem;
            margin-bottom: 1rem;
            background: ${state.showDecrypted ? 'var(--bg-tertiary)' : '#22c55e'};
            color: ${state.showDecrypted ? 'var(--text-secondary)' : 'white'};
            border: 1px solid ${state.showDecrypted ? 'var(--border-color)' : '#22c55e'};
            border-radius: 8px;
            font-size: 0.75rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            transition: all 0.2s;
        ">
            ${state.showDecrypted ? '\u{1F513} Viewing Decrypted' : '\u{1F510} Viewing Encrypted'}
            <span style="
                font-size: 0.65rem;
                opacity: 0.8;
            ">(click to toggle)</span>
        </button>
        ` : ''}

        <div style="
            display: grid;
            grid-template-columns: ${state.encryptedAtRest ? 'repeat(3, 1fr)' : 'repeat(2, 1fr)'};
            gap: 0.75rem;
            margin-bottom: 1rem;
        ">
            <div style="
                background: var(--bg-tertiary);
                padding: 0.75rem;
                border-radius: 8px;
                text-align: center;
            ">
                <div style="font-size: 1.25rem; font-weight: 600;">${data.conversations.length}</div>
                <div style="font-size: 0.7rem; color: var(--text-secondary);">Sessions</div>
            </div>
            <div style="
                background: var(--bg-tertiary);
                padding: 0.75rem;
                border-radius: 8px;
                text-align: center;
            ">
                <div style="font-size: 1.25rem; font-weight: 600;">${data.conversations.reduce((sum, c) => sum + c.message_count, 0)}</div>
                <div style="font-size: 0.7rem; color: var(--text-secondary);">Messages</div>
            </div>
            ${state.encryptedAtRest ? `
            <div style="
                background: var(--bg-tertiary);
                padding: 0.75rem;
                border-radius: 8px;
                text-align: center;
            ">
                <div style="font-size: 1.25rem;">\u{1F512}</div>
                <div style="font-size: 0.7rem; color: var(--text-secondary);">Encrypted</div>
            </div>
            ` : ''}
        </div>

        <div id="history-list" style="display: flex; flex-direction: column; gap: 0.75rem; max-height: 450px; overflow-y: auto;">
            ${Object.entries(groupedByDate).map(([dateKey, convs]) => `
                <div class="date-group">
                    <div style="
                        font-size: 0.7rem;
                        font-weight: 600;
                        color: var(--text-tertiary);
                        text-transform: uppercase;
                        margin-bottom: 0.5rem;
                        padding-left: 0.25rem;
                    ">${formatDateLabel(dateKey)}</div>
                    ${convs.map(conv => renderConversationItem(conv)).join('')}
                </div>
            `).join('')}
        </div>
    `;
}

function renderConversationItem(conv) {
    const isActive = state.currentSessionId === conv.session_id;
    const time = new Date(conv.started_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const preview = conv.preview || 'Empty conversation';
    const isEncryptedPreview = conv.preview_encrypted;

    const previewStyle = isEncryptedPreview
        ? `font-family: 'Monaco', 'Menlo', 'Courier New', monospace; color: #22c55e;`
        : '';
    const borderColor = isEncryptedPreview && !isActive ? '#22c55e' : (isActive ? 'var(--accent-color)' : 'var(--border-color)');

    return `
        <div class="conversation-item ${isActive ? 'active' : ''}"
             data-session-id="${conv.session_id}"
             onclick="loadConversation('${conv.session_id}')"
             style="
                background: ${isActive ? 'var(--accent-color)' : (isEncryptedPreview ? 'linear-gradient(135deg, #1a1a2e, #16213e)' : 'var(--bg-secondary)')};
                color: ${isActive ? 'white' : 'var(--text-primary)'};
                border: 1px solid ${borderColor};
                border-radius: 8px;
                padding: 0.75rem;
                cursor: pointer;
                transition: all 0.2s;
             "
             onmouseover="if(!this.classList.contains('active')) this.style.borderColor='var(--accent-color)'"
             onmouseout="if(!this.classList.contains('active')) this.style.borderColor='${borderColor}'">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.375rem;">
                <span style="font-size: 0.75rem; opacity: ${isActive ? '0.9' : '0.7'};">
                    ${isEncryptedPreview ? '\u{1F510} ' : ''}${time}
                </span>
                <span style="
                    font-size: 0.65rem;
                    background: ${isActive ? 'rgba(255,255,255,0.2)' : 'var(--bg-tertiary)'};
                    padding: 0.125rem 0.5rem;
                    border-radius: 10px;
                ">${conv.message_count} msgs</span>
            </div>
            <div style="
                font-size: 0.8rem;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                opacity: ${isActive ? '1' : '0.9'};
                ${previewStyle}
            " title="${escapeHtml(conv.preview || '')}">${escapeHtml(preview)}</div>
        </div>
    `;
}

window.loadConversation = async function(sessionId) {
    try {
        const data = await API.getConversation(sessionId, state.showDecrypted);
        state.currentSessionId = sessionId;

        if (data.encrypted_at_rest !== undefined) {
            state.encryptedAtRest = data.encrypted_at_rest;
        }

        const chatContainer = document.getElementById('chat-container');
        if (chatContainer) {
            chatContainer.innerHTML = '';
        }

        if (!state.showDecrypted && state.encryptedAtRest) {
            const banner = document.createElement('div');
            banner.style.cssText = `
                background: linear-gradient(135deg, #22c55e, #16a34a);
                color: white;
                padding: 0.75rem 1rem;
                margin: 0.5rem;
                border-radius: 8px;
                font-size: 0.85rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            `;
            banner.innerHTML = `
                \u{1F510} <strong>Viewing raw encrypted data</strong> - This is how your messages are stored at rest.
                <button onclick="toggleEncryptionView()" style="
                    margin-left: auto;
                    background: rgba(255,255,255,0.2);
                    border: none;
                    color: white;
                    padding: 0.25rem 0.75rem;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 0.75rem;
                ">Decrypt</button>
            `;
            chatContainer.appendChild(banner);
        }

        data.messages.forEach(msg => {
            if (msg.role !== 'system') {
                addMessageToChat(msg.role, msg.content, msg.encrypted && !state.showDecrypted, msg.id);
            }
        });

        renderConversationHistory({ conversations: state.conversations, encrypted_at_rest: state.encryptedAtRest });

        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        const statusText = state.showDecrypted ? '' : ' (encrypted view)';
        Toast.success(`Loaded conversation with ${data.message_count} messages${statusText}`);

        if (window.innerWidth < 768) {
            toggleHistorySidebar();
        }

        // Update context status for the new session
        updateContextStatus();
    } catch (e) {
        Toast.error(`Failed to load conversation: ${e.message}`);
    }
};

window.toggleEncryptionView = async function() {
    state.showDecrypted = !state.showDecrypted;

    await loadConversationHistory();

    if (state.currentSessionId) {
        await window.loadConversation(state.currentSessionId);
    }

    Toast.info(state.showDecrypted ? '\u{1F513} Now viewing decrypted content' : '\u{1F510} Now viewing raw encrypted data');
};

window.startNewConversation = async function() {
    try {
        const result = await API.newConversation();
        state.currentSessionId = result.session_id;

        const chatContainer = document.getElementById('chat-container');
        if (chatContainer) {
            chatContainer.innerHTML = `
                <div style="text-align: center; padding: 2rem; color: var(--text-secondary);">
                    <span style="font-size: 2rem;">\u{2728}</span>
                    <p style="margin-top: 0.5rem;">New conversation started. Say hello!</p>
                </div>
            `;
        }

        await loadConversationHistory();

        Toast.success('New conversation started');
    } catch (e) {
        Toast.error(`Failed to start new conversation: ${e.message}`);
    }
};

window.toggleHistorySidebar = function() {
    const sidebar = document.getElementById('history-sidebar');
    const toggleBtn = document.getElementById('toggle-history-btn');

    if (!sidebar || !toggleBtn) return;

    state.historyVisible = !state.historyVisible;

    if (state.historyVisible) {
        sidebar.style.display = 'block';
        toggleBtn.innerHTML = '\u{1F4DC} Hide History';
        if (!state.conversations) {
            loadConversationHistory();
        }
    } else {
        sidebar.style.display = 'none';
        toggleBtn.innerHTML = '\u{1F4DC} Show History';
    }
};

window.deleteMessage = async function(messageId, messageDiv) {
    if (!confirm('Delete this message? This cannot be undone.')) return;

    try {
        await API.deleteMessage(messageId);
        if (messageDiv) {
            messageDiv.style.transition = 'opacity 0.2s, transform 0.2s';
            messageDiv.style.opacity = '0';
            messageDiv.style.transform = 'scale(0.95)';
            setTimeout(() => messageDiv.remove(), 200);
        }
        Toast.info('Message deleted');
    } catch (e) {
        Toast.error(`Failed to delete message: ${e.message}`);
    }
};

function addMessageToChat(role, content, isEncrypted = false, messageId = null) {
    const chatContainer = document.getElementById('chat-container');
    if (!chatContainer) return;

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    if (messageId) messageDiv.dataset.messageId = messageId;

    // Add hover-reveal delete button
    if (messageId) {
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'msg-delete-btn';
        deleteBtn.title = 'Delete message';
        deleteBtn.textContent = '\u2715';
        deleteBtn.onclick = (e) => {
            e.stopPropagation();
            window.deleteMessage(messageId, messageDiv);
        };
        messageDiv.appendChild(deleteBtn);
    }

    if (isEncrypted) {
        messageDiv.style.cssText = `
            padding: 1rem;
            margin-bottom: 0.75rem;
            border-radius: 12px;
            max-width: 85%;
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            border: 1px solid #22c55e;
            margin-${role === 'user' ? 'left' : 'right'}: auto;
            font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
            font-size: 0.75rem;
            word-break: break-all;
            color: #22c55e;
        `;
        const header = document.createElement('div');
        header.style.cssText = `
            font-size: 0.65rem;
            opacity: 0.7;
            margin-bottom: 0.5rem;
            color: #4ade80;
        `;
        header.textContent = `\u{1F510} ${role.toUpperCase()} (encrypted)`;
        messageDiv.appendChild(header);

        const contentSpan = document.createElement('span');
        contentSpan.textContent = content;
        messageDiv.appendChild(contentSpan);
    } else {
        messageDiv.style.cssText = `
            padding: 1rem;
            margin-bottom: 0.75rem;
            border-radius: 12px;
            max-width: 85%;
            ${role === 'user'
                ? 'background: var(--accent-color); color: white; margin-left: auto;'
                : 'background: var(--bg-tertiary); margin-right: auto;'}
        `;

        if (role === 'assistant' && window.marked) {
            messageDiv.innerHTML = marked.parse(content);
        } else {
            messageDiv.textContent = content;
        }
    }

    chatContainer.appendChild(messageDiv);
}
