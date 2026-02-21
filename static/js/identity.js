/**
 * Kestrel Sovereign Console - Identity Module
 * Agent selection, navigation, identity panel, privacy indicator, sidebar
 */

import API from './api.js';
import { state, PRIVACY_MODES, Toast, loadCommands } from './ui.js';
import { disconnectNotifications, connectNotifications, loadModels, updateContextStatus } from './chat.js';

// ============================================================================
// Agent Selection (Multi-Agent Support)
// ============================================================================

// Local reference to current agent ID (for use in loadIdentity title update)
let currentAgentId = null;

export function initAgentFromUrl() {
    const params = new URLSearchParams(window.location.search);
    const agentParam = params.get('agent');
    if (agentParam) {
        currentAgentId = agentParam;
        // Set the agent ID in the API module for endpoint rewriting
        API.setAgentId(agentParam);
        console.log('Kestrel Sovereign Console: Agent mode enabled for', currentAgentId);
        document.title = `Sovereign Console - Agent ${currentAgentId.slice(0, 8)}...`;
    } else {
        console.log('Kestrel Sovereign Console: Single-agent mode (default)');
    }
}

// ============================================================================
// Navigation
// ============================================================================

// Forward declarations for lazy-loaded panels
let loadConstitution = null;
let loadMemories = null;
let loadExports = null;
let loadTasks = null;
let loadResources = null;

export function setLazyLoaders(loaders) {
    loadConstitution = loaders.loadConstitution;
    loadMemories = loaders.loadMemories;
    loadExports = loaders.loadExports;
    loadTasks = loaders.loadTasks;
    loadResources = loaders.loadResources;
}

export function initNavigation() {
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const panelId = tab.dataset.panel;

            document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            document.getElementById(`panel-${panelId}`).classList.add('active');

            state.currentPanel = panelId;

            if (panelId === 'constitution' && !state.constitution && loadConstitution) loadConstitution();
            if (panelId === 'memories' && !state.memories && loadMemories) loadMemories();
            if (panelId === 'sovereignty' && !state.exports && loadExports) loadExports();
            if (panelId === 'tasks' && loadTasks) loadTasks();
            if (panelId === 'resources' && loadResources) loadResources();
        });
    });
}

// ============================================================================
// Identity Panel
// ============================================================================

export async function loadIdentity() {
    try {
        const identity = await API.getIdentity();
        state.identity = identity;

        // Update page title with companion name if in multi-agent mode
        if (currentAgentId && identity.name) {
            document.title = `${identity.name} - Sovereign Console`;
        }

        // Build avatar URL - prefer serving from identity chain via avatar endpoint
        let avatarUrl = null;
        if (identity.avatar_hash || identity.avatar_url) {
            // In multi-agent mode, serve from Kestrel identity chain via platform endpoint
            if (currentAgentId) {
                avatarUrl = `/api/kestrel/companions/${encodeURIComponent(currentAgentId)}/avatar`;
            } else if (identity.avatar_url) {
                avatarUrl = identity.avatar_url;
            }
        }

        // Update nav header avatar and name
        const navIcon = document.getElementById('nav-agent-icon');
        const navAvatar = document.getElementById('nav-agent-avatar');
        const navName = document.getElementById('nav-agent-name');

        if (navAvatar && navIcon) {
            if (avatarUrl) {
                navAvatar.src = avatarUrl;
                navAvatar.style.display = 'block';
                navAvatar.onerror = () => {
                    navAvatar.style.display = 'none';
                    navIcon.style.display = 'inline';
                };
                navIcon.style.display = 'none';
            } else {
                navAvatar.style.display = 'none';
                navIcon.style.display = 'inline';
            }
        }
        if (navName && identity.name) {
            navName.textContent = identity.name;
        }

        // Avatar: use stored image if available, fallback to kestrel logo
        const defaultAvatarSvg = `<img src="/static/favicon.svg" alt="Kestrel" class="identity-avatar-img" style="padding: 8px; background: #fff; border-radius: 12px;">`;
        const avatarHtml = avatarUrl
            ? `<img src="${avatarUrl}" alt="Avatar" class="identity-avatar-img" onerror="this.parentElement.innerHTML='${defaultAvatarSvg.replace(/'/g, "\\'")}';">`
            : defaultAvatarSvg;

        const card = document.getElementById('identity-card');
        card.innerHTML = `
            <div class="identity-header">
                <div class="identity-avatar">${avatarHtml}</div>
                <div class="identity-info">
                    <h2>${identity.name || 'Kestrel Agent'}</h2>
                    <div class="identity-did" title="${identity.did}">
                        ${identity.did || 'No DID assigned'}
                        <button onclick="copyToClipboard('${identity.did}')" style="background:none;border:none;cursor:pointer;margin-left:0.5rem;" title="Copy DID">\u{1F4CB}</button>
                    </div>
                </div>
            </div>
            <div class="identity-stats">
                <div class="stat-item">
                    <div class="value">${identity.constitution_hash ? '\u{2705}' : '\u{274C}'}</div>
                    <div class="label">Constitution</div>
                </div>
                <div class="stat-item">
                    <div class="value">${identity.avatar_hash ? '\u{1F5BC}' : '\u{2796}'}</div>
                    <div class="label">Avatar</div>
                </div>
                <div class="stat-item">
                    <div class="value">${identity.initial_balance || 0}</div>
                    <div class="label">FIL Balance</div>
                </div>
            </div>
        `;

        if (identity.genesis_audit) {
            const auditEl = document.getElementById('genesis-audit');
            auditEl.innerHTML = `
                <div class="identity-card" style="margin-top: 1rem;">
                    <h3 style="margin: 0 0 1rem 0;">Genesis Audit</h3>
                    <div style="display: grid; gap: 0.5rem; font-size: 0.875rem;">
                        <div><strong>Risk Level:</strong> ${identity.genesis_audit.risk_level}</div>
                        <div><strong>Summary:</strong> ${identity.genesis_audit.summary || 'No issues found'}</div>
                        ${identity.genesis_audit.findings ? `<div><strong>Findings:</strong> ${JSON.stringify(identity.genesis_audit.findings)}</div>` : ''}
                    </div>
                </div>
            `;
        }
    } catch (e) {
        const card = document.getElementById('identity-card');
        if (card) card.innerHTML = `<div style="color: var(--error); padding: 1rem;">Failed to load identity: ${e.message}</div>`;
    }
}

// ============================================================================
// Privacy Indicator
// ============================================================================

export async function loadPrivacyMode() {
    try {
        const data = await API.getPrivacyMode();
        state.privacyMode = data.privacy_mode;
        updatePrivacyIndicator(data.privacy_mode);
    } catch (e) {
        console.error('Failed to load privacy mode:', e);
    }
}

export function updatePrivacyIndicator(mode) {
    const config = PRIVACY_MODES[mode] || PRIVACY_MODES.normal;
    // Use the new chat header element, fallback to old nav element
    const el = document.getElementById('chat-privacy-indicator') || document.getElementById('privacy-indicator');
    if (!el) return;

    el.innerHTML = `
        <span style="
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.3rem 0.6rem;
            border-radius: 16px;
            font-size: 0.75rem;
            font-weight: 600;
            background-color: ${config.color};
            color: white;
            cursor: pointer;
        " onclick="showPrivacySelector()" title="Click to change privacy mode">
            <span>${config.icon}</span>
            <span>${config.label}</span>
        </span>
    `;
}

window.showPrivacySelector = function() {
    const existing = document.getElementById('privacy-dropdown');
    if (existing) {
        existing.remove();
        return;
    }

    // Use the new chat header element, fallback to old nav element
    const indicator = document.getElementById('chat-privacy-indicator') || document.getElementById('privacy-indicator');
    if (!indicator) return;
    const rect = indicator.getBoundingClientRect();

    const dropdown = document.createElement('div');
    dropdown.id = 'privacy-dropdown';
    dropdown.style.cssText = `
        position: fixed;
        top: ${rect.bottom + 8}px;
        right: ${window.innerWidth - rect.right}px;
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        z-index: 1000;
        min-width: 280px;
        overflow: hidden;
    `;

    dropdown.innerHTML = `
        <div style="padding: 0.75rem 1rem; border-bottom: 1px solid var(--border-color); font-weight: 600; font-size: 0.875rem;">
            Privacy Mode
        </div>
        ${Object.entries(PRIVACY_MODES).map(([mode, config]) => `
            <div class="privacy-option" data-mode="${mode}" style="
                padding: 0.75rem 1rem;
                cursor: pointer;
                display: flex;
                align-items: center;
                gap: 0.75rem;
                transition: background 0.15s;
                ${state.privacyMode === mode ? 'background: var(--bg-tertiary);' : ''}
            " onmouseover="this.style.background='var(--bg-tertiary)'" onmouseout="this.style.background='${state.privacyMode === mode ? 'var(--bg-tertiary)' : 'transparent'}'">
                <span style="
                    width: 32px;
                    height: 32px;
                    border-radius: 8px;
                    background: ${config.color};
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 1rem;
                ">${config.icon}</span>
                <div style="flex: 1;">
                    <div style="font-weight: 500; font-size: 0.875rem; display: flex; align-items: center; gap: 0.5rem;">
                        ${config.label}
                        ${state.privacyMode === mode ? '<span style="font-size: 0.75rem; color: var(--accent-color);">✓ Current</span>' : ''}
                    </div>
                    <div style="font-size: 0.75rem; color: var(--text-secondary);">${config.description}</div>
                </div>
            </div>
        `).join('')}
    `;

    dropdown.querySelectorAll('.privacy-option').forEach(opt => {
        opt.addEventListener('click', async () => {
            const mode = opt.dataset.mode;
            if (mode === state.privacyMode) {
                dropdown.remove();
                return;
            }
            try {
                await API.setPrivacyMode(mode);
                state.privacyMode = mode;
                updatePrivacyIndicator(mode);
                dropdown.remove();
                Toast.success(`Privacy mode set to ${PRIVACY_MODES[mode]?.label || mode}`);
            } catch (e) {
                Toast.error(`Failed to set privacy mode: ${e.message}`);
            }
        });
    });

    setTimeout(() => {
        document.addEventListener('click', function closeDropdown(e) {
            if (!dropdown.contains(e.target) && !indicator.contains(e.target)) {
                dropdown.remove();
                document.removeEventListener('click', closeDropdown);
            }
        });
    }, 0);

    document.body.appendChild(dropdown);
};

// ============================================================================
// Utilities
// ============================================================================

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ============================================================================
// Agents Pane (Rookery)
// ============================================================================

let selectedAgentName = null;

export async function loadAgents() {
    try {
        const data = await API.getAgents();
        const agents = data.agents || [];

        const container = document.getElementById('agents-list');
        if (agents.length === 0) {
            container.innerHTML = '<p style="color: var(--text-secondary); padding: 1rem; text-align: center;">No agents available</p>';
            return;
        }

        container.innerHTML = '';
        for (const agent of agents) {
            const isOnline = agent.status !== 'offline';
            const item = document.createElement('div');
            item.className = `agent-item${selectedAgentName === agent.name ? ' selected' : ''}${!isOnline ? ' offline' : ''}`;
            item.dataset.agentName = agent.name;

            if (isOnline) {
                item.addEventListener('click', () => window.selectAgent(agent.name));
            }

            item.innerHTML = `
                <span class="agent-status-dot ${isOnline ? 'online' : 'offline'}"></span>
                <div class="agent-info">
                    <div class="agent-name">${escapeHtml(agent.name || 'Unnamed Agent')}</div>
                    <div class="agent-description">${escapeHtml(agent.description || 'No description')}</div>
                </div>
            `;
            container.appendChild(item);
        }

        // Auto-select first online agent if none selected
        if (!selectedAgentName) {
            const firstOnline = agents.find(a => a.status !== 'offline');
            if (firstOnline) {
                window.selectAgent(firstOnline.name);
            }
        }
    } catch (e) {
        const container = document.getElementById('agents-list');
        container.innerHTML = '<p style="color: var(--error); padding: 1rem;">Failed to load agents</p>';
    }
}

window.selectAgent = async function(agentName) {
    selectedAgentName = agentName;

    // Set host agent routing in API layer
    API.setHostAgent(agentName);

    // Update selection UI
    document.querySelectorAll('.agent-item').forEach(item => {
        item.classList.toggle('selected', item.dataset.agentName === agentName);
    });

    // Update chat header with agent name
    const navName = document.getElementById('nav-agent-name');
    if (navName) navName.textContent = agentName;

    // Show conversations pane
    const conversationsPane = document.getElementById('conversations-pane');
    conversationsPane.style.display = 'flex';

    // Clear chat messages — previous agent's messages shouldn't persist
    const chatContainer = document.getElementById('chat-container');
    if (chatContainer) {
        chatContainer.innerHTML = '';
    }

    // Reset session and cached panel data so they reload from the new agent
    state.currentSessionId = null;
    state.identity = null;
    state.constitution = null;
    state.memories = null;
    state.exports = null;
    state.storage = null;
    state.wallet = null;

    // Reconnect SSE notifications to the new agent
    disconnectNotifications();
    connectNotifications();

    // Reload all agent-specific data in parallel
    await Promise.all([
        loadIdentity(),
        loadPrivacyMode(),
        loadConversations(agentName),
        loadModels(),
        loadCommands(API),
        updateContextStatus(),
    ]);
};

// ============================================================================
// Conversations Pane
// ============================================================================

let activeConversationId = null;

export async function loadConversations(_agentName) {
    // Agent routing is handled by API.setHostAgent() — all calls auto-prefix
    try {
        const data = await API.getConversations();
        const conversations = data.conversations || [];

        const container = document.getElementById('conversations-list');
        if (conversations.length === 0) {
            container.innerHTML = '<p style="color: var(--text-secondary); padding: 1rem; text-align: center;">No conversations yet</p>';
            return;
        }

        container.innerHTML = '';
        for (const conv of conversations) {
            const date = new Date(conv.started_at);
            const timeStr = date.toLocaleString();

            const item = document.createElement('div');
            item.className = `conversation-item ${activeConversationId === conv.session_id ? 'active' : ''}`;
            item.dataset.sessionId = conv.session_id;
            item.addEventListener('click', () => window.loadConversation(conv.session_id));

            const preview = document.createElement('div');
            preview.className = 'conversation-preview';
            preview.textContent = conv.preview || 'New conversation';

            const time = document.createElement('div');
            time.className = 'conversation-time';
            time.textContent = timeStr;

            item.appendChild(preview);
            item.appendChild(time);
            container.appendChild(item);
        }
    } catch (e) {
        const container = document.getElementById('conversations-list');
        container.innerHTML = '<p style="color: var(--error); padding: 1rem;">Failed to load conversations</p>';
    }
}

window.loadConversation = async function(sessionId) {
    activeConversationId = sessionId;

    // Update selection UI
    document.querySelectorAll('.conversation-item').forEach(item => {
        item.classList.toggle('active', item.dataset.sessionId === sessionId);
    });

    // Load conversation messages into chat panel
    try {
        const data = await API.getConversation(sessionId);
        const messages = data.messages || [];

        const chatContainer = document.getElementById('chat-container');
        chatContainer.innerHTML = '';

        const renderMd = window.SharedMarkdown?.renderMarkdown;

        for (const msg of messages) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${msg.role === 'user' ? 'user-message' : 'agent-message'}`;

            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';

            if (msg.role === 'assistant' && renderMd) {
                contentDiv.innerHTML = renderMd(msg.content);
            } else {
                contentDiv.textContent = msg.content;
            }

            messageDiv.appendChild(contentDiv);
            chatContainer.appendChild(messageDiv);
        }

        chatContainer.scrollTop = chatContainer.scrollHeight;

        // Switch to chat panel
        document.querySelector('[data-panel="chat"]')?.click();
    } catch (e) {
        console.error('Failed to load conversation:', e);
    }
};

// ============================================================================
// Pane Collapse/Expand + Resize
// ============================================================================

window.togglePane = function(paneId) {
    const pane = document.getElementById(paneId);
    if (pane) {
        pane.classList.toggle('collapsed');
    }
};

function initPaneResize(handleId, paneId) {
    const handle = document.getElementById(handleId);
    const pane = document.getElementById(paneId);
    if (!handle || !pane) return;

    let startX, startWidth;

    handle.addEventListener('mousedown', (e) => {
        startX = e.clientX;
        startWidth = pane.offsetWidth;
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';

        function onMouseMove(e) {
            const diff = e.clientX - startX;
            const newWidth = Math.max(200, Math.min(500, startWidth + diff));
            pane.style.width = newWidth + 'px';
        }

        function onMouseUp() {
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        }

        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
    });
}

// Setup collapse buttons and resize handles
document.addEventListener('DOMContentLoaded', () => {
    const collapseAgentsBtn = document.getElementById('collapse-agents-btn');
    if (collapseAgentsBtn) {
        collapseAgentsBtn.addEventListener('click', () => togglePane('agents-pane'));
    }

    const collapseConversationsBtn = document.getElementById('collapse-conversations-btn');
    if (collapseConversationsBtn) {
        collapseConversationsBtn.addEventListener('click', () => togglePane('conversations-pane'));
    }

    const newConversationSidebarBtn = document.getElementById('new-conversation-sidebar-btn');
    if (newConversationSidebarBtn) {
        newConversationSidebarBtn.addEventListener('click', async () => {
            try {
                await API.newConversation();
                if (selectedAgentName) {
                    await loadConversations(selectedAgentName);
                }
            } catch (e) {
                console.error('Failed to create new conversation:', e);
            }
        });
    }

    // Initialize resize handles
    initPaneResize('resize-agents', 'agents-pane');
    initPaneResize('resize-conversations', 'conversations-pane');
});
