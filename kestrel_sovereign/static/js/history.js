/**
 * Kestrel Sovereign Console - History Module
 * Chat History Browser
 */

import API from './api.js';
import { state, Toast, escapeHtml } from './ui.js';
import {
    updateContextStatus,
    wipeAgentChatPane,
    renderAgentContentHtml,
} from './chat.js';

// #1659: tool cards on reload come from the structured, position-stamped
// ``tool_events`` metadata; new rows persist clean prose. Rows persisted
// BEFORE the cutover instead carry emoji marker tokens inline (and their
// tool_events, if any, lack ``pos``). For those we parse the markers back
// into structured events WITH positions, so the cards render where the tools
// actually occurred — not all bunched at the top.
//
// Self-terminating marker TOKENS (not line-anchored): the old stream could
// glue a marker onto adjacent prose without a newline ("I'll check🔧 Calling
// lookup..."). Mirrors the deleted chat.js TOOL_MARKER_TOKEN.
const LEGACY_TOOL_MARKER_TOKEN = /\u{1F527}\s+Calling\s+\S[^\n]*?\.\.\.|✓\s+\S[^\n]*?\s+(?:complete|done)\b(?:\s+\([^\n)]*\))?|❌\s+\S[^\n]*?\s+failed\b(?::[^\n\u{1F527}✓❌]*)?/gu;
// Gate legacy handling on a 🔧 Calling START being present — exactly the old
// TOOL_START_PRESENCE rule. Without it, ordinary replies like "✓ Migration
// complete" would be mistaken for tool markers.
const LEGACY_TOOL_START_PRESENCE = /\u{1F527}\s+Calling\s+/u;

// Parse a pre-cutover assistant message's inline emoji markers into
// { clean, events }: the prose with ONLY the markers removed, and the tool
// events with their positions into that prose. Nothing else is mutated, so
// the positions stay exact — the shared renderer (buildToolSegmentsByPos)
// already strips the `---` wire delimiter and leading blanks PER SEGMENT, so
// doing it here too would shift every later card's position (codex review).
function legacyToolEventsFromText(content) {
    const src = String(content || '');
    const re = new RegExp(LEGACY_TOOL_MARKER_TOKEN.source, 'gu');
    const events = [];
    let clean = '';
    let last = 0;
    let m;
    while ((m = re.exec(src)) !== null) {
        clean += src.slice(last, m.index);
        const marker = m[0];
        const pos = clean.length;
        let mm;
        if ((mm = marker.match(/^\u{1F527}\s+Calling\s+(\S[^\n]*?)(?::[^\n]*)?\.\.\.$/u))) {
            events.push({ phase: 'start', name: mm[1].trim(), pos });
        } else if ((mm = marker.match(/^✓\s+(\S[^\n]*?)\s+(?:complete|done)\b/u))) {
            events.push({ phase: 'done', name: mm[1].trim(), pos });
        } else if ((mm = marker.match(/^❌\s+(\S[^\n]*?)\s+failed\b/u))) {
            events.push({ phase: 'error', name: mm[1].trim(), pos });
        }
        last = re.lastIndex;
    }
    clean += src.slice(last);
    return { clean, events };
}

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
    // Don't reload the conversation we're already viewing — preserves in-flight content
    if (sessionId === state.currentSessionId) {
        return;
    }

    try {
        const data = await API.getConversation(sessionId, state.showDecrypted);

        if (data.encrypted_at_rest !== undefined) {
            state.encryptedAtRest = data.encrypted_at_rest;
        }

        // Wipe the visible agent's pane and bump that agent's pane-
        // local generation — a conversation switch is a within-agent
        // context change. Streams in flight on OTHER agents are NOT
        // touched. Setting currentSessionId via the property routes
        // into the visible agent's pane.
        wipeAgentChatPane(API.getHostAgent());
        state.currentSessionId = sessionId;
        // viewport is the scroll host (#chat-container); pane is its
        // child and where messages are appended.
        const viewport = document.getElementById('chat-container');
        const visiblePane = state.chatPanes.get(state.mountedChatAgent);
        const paneEl = visiblePane ? visiblePane.element : viewport;

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
            paneEl.appendChild(banner);
        }

        data.messages.forEach(msg => {
            if (msg.role === 'system') return;
            const isEncrypted = msg.encrypted && !state.showDecrypted;
            let toolHtml = '';
            let bodyHtml = null;
            const content = msg.content;
            if (msg.role === 'assistant' && !isEncrypted) {
                const toolEvents = msg.metadata?.tool_events;
                const hasPos = !!toolEvents
                    && toolEvents.some((e) => typeof e.pos === 'number');
                if (hasPos) {
                    // New (post-cutover) row: position-stamped tool_events are
                    // authoritative and the prose is clean. Cards + prose in
                    // document order, identical to the live bubble.
                    bodyHtml = renderAgentContentHtml(content, { toolEvents });
                } else if (LEGACY_TOOL_START_PRESENCE.test(content)) {
                    // Pre-cutover row: derive cards AND their positions from the
                    // inline emoji markers so they render where the tools ran,
                    // not bunched at the top.
                    const legacy = legacyToolEventsFromText(content);
                    bodyHtml = renderAgentContentHtml(
                        legacy.clean, { toolEvents: legacy.events },
                    );
                } else if (toolEvents && toolEvents.length) {
                    // Old metadata with neither pos nor inline markers \u2014 no
                    // placement survives; render cards at the top rather than
                    // dropping them.
                    bodyHtml = renderAgentContentHtml(content, { toolEvents });
                }
            }
            addMessageToChat(
                msg.role,
                content,
                isEncrypted,
                msg.id,
                toolHtml,
                '',
                bodyHtml,
            );
        });

        renderConversationHistory({ conversations: state.conversations, encrypted_at_rest: state.encryptedAtRest });

        if (viewport) {
            viewport.scrollTop = viewport.scrollHeight;
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

        // Wipe the visible agent's pane and bump that agent's pane-
        // local generation so any stream still running against the
        // previous (now-replaced) session gates out. Other agents are
        // unaffected. Then write currentSessionId via the property,
        // which writes into the visible agent's pane.
        wipeAgentChatPane(API.getHostAgent(), `
            <div style="text-align: center; padding: 2rem; color: var(--text-secondary);">
                <span style="font-size: 2rem;">\u{2728}</span>
                <p style="margin-top: 0.5rem;">New conversation started. Say hello!</p>
            </div>
        `);
        state.currentSessionId = result.session_id;

        await loadConversationHistory();

        // Refresh the context status for the new (empty) session so the
        // footer stops reporting stale message count / utilization from the
        // previous conversation.
        updateContextStatus();

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
    // Soft-delete (#763) — moves the message to Trash, recoverable from
    // the trash sub-view (#765).
    if (!confirm('Move this message to Trash? You can restore it from the trash view.')) return;

    try {
        await API.deleteMessage(messageId);
        if (messageDiv) {
            messageDiv.style.transition = 'opacity 0.2s, transform 0.2s';
            messageDiv.style.opacity = '0';
            messageDiv.style.transform = 'scale(0.95)';
            setTimeout(() => messageDiv.remove(), 200);
        }
        Toast.info('Message moved to trash');
    } catch (e) {
        Toast.error(`Failed to delete message: ${e.message}`);
    }
};

window.purgeMessage = async function(messageId, messageDiv) {
    // Permanent delete (#765) — hard SQL DELETE, no recovery.
    if (!confirm(
        `Delete this message PERMANENTLY?\n\n`
        + `This is a hard delete and CANNOT be restored. Soft-delete first `
        + `(the regular ✕) is the recoverable path.`
    )) return;

    try {
        await API.purgeMessage(messageId, 'user-initiated-ui');
        if (messageDiv) {
            messageDiv.style.transition = 'opacity 0.2s, transform 0.2s';
            messageDiv.style.opacity = '0';
            messageDiv.style.transform = 'scale(0.95)';
            setTimeout(() => messageDiv.remove(), 200);
        }
        Toast.info('Message permanently deleted');
    } catch (e) {
        Toast.error(`Failed to permanently delete: ${e.message}`);
    }
};

function addMessageToChat(
    role,
    content,
    isEncrypted = false,
    messageId = null,
    toolActivityHtml = '',
    preludeContent = '',
    bodyHtml = null,
) {
    // Append into the visible (mounted) agent's pane element — the
    // viewport (#chat-container) is now the scroll host and panes are
    // its children, so writing directly to the viewport would orphan
    // the message outside any agent's pane.
    const visibleAgent = state.mountedChatAgent;
    const visiblePane = visibleAgent === undefined ? null : state.chatPanes.get(visibleAgent);
    const target = visiblePane ? visiblePane.element : document.getElementById('chat-container');
    if (!target) return;

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    if (messageId) messageDiv.dataset.messageId = messageId;

    // Add hover-reveal delete buttons (#763 / #765).
    // Soft delete -> moves to Trash, recoverable.
    // Permanent delete -> hard SQL DELETE.
    if (messageId) {
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'msg-delete-btn';
        deleteBtn.title = 'Move to trash';
        deleteBtn.textContent = '\u2715';
        deleteBtn.onclick = (e) => {
            e.stopPropagation();
            window.deleteMessage(messageId, messageDiv);
        };
        messageDiv.appendChild(deleteBtn);

        const purgeBtn = document.createElement('button');
        purgeBtn.className = 'msg-purge-btn';
        purgeBtn.title = 'Delete permanently (cannot be restored)';
        purgeBtn.textContent = '⊘';
        purgeBtn.onclick = (e) => {
            e.stopPropagation();
            window.purgeMessage(messageId, messageDiv);
        };
        messageDiv.appendChild(purgeBtn);
    }

    if (preludeContent) {
        const preludeDiv = document.createElement('div');
        preludeDiv.className = 'message-content response-prelude';
        if (role === 'assistant' && window.marked) {
            preludeDiv.innerHTML = marked.parse(preludeContent);
        } else {
            preludeDiv.textContent = preludeContent;
        }
        messageDiv.appendChild(preludeDiv);
    }

    // Render tool activity above the final message content
    if (toolActivityHtml) {
        messageDiv.insertAdjacentHTML('beforeend', toolActivityHtml);
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

        // Wrap rendered content in a child div — assigning to messageDiv's
        // innerHTML/textContent directly would wipe the .msg-delete-btn /
        // .msg-purge-btn that were just appended above (they're children of
        // messageDiv).  See #765 — without this wrapper the soft-delete and
        // hard-purge affordances render briefly then get blown away.
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        if (bodyHtml !== null) {
            // Pre-rendered cards + prose (assistant turns with tool activity),
            // already in document order — see renderAgentContentHtml.
            contentDiv.innerHTML = bodyHtml;
        } else if (role === 'assistant' && window.marked) {
            contentDiv.innerHTML = marked.parse(content);
        } else {
            contentDiv.textContent = content;
        }
        messageDiv.appendChild(contentDiv);
    }

    target.appendChild(messageDiv);
}
