/**
 * Kestrel Sovereign Console - Identity Module
 * Agent selection, navigation, identity panel, privacy indicator, sidebar
 */

import API from './api.js';
import { state, PRIVACY_MODES, Toast, formatBytes } from './ui.js';

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
// Sidebar
// ============================================================================

export async function loadSidebar() {
    try {
        const health = await API.health();
        document.getElementById('agent-status').innerHTML = `
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="width: 8px; height: 8px; border-radius: 50%; background: ${health.agent_initialized ? 'var(--success)' : 'var(--error)'};"></span>
                <span>${health.agent_initialized ? 'Online' : 'Offline'}</span>
            </div>
        `;
    } catch (e) {
        document.getElementById('agent-status').innerHTML = '<span style="color: var(--error);">Error</span>';
    }

    try {
        const stats = await API.getStorageStats();
        state.storage = stats;
        document.getElementById('storage-summary').innerHTML = `
            <div style="font-size: 0.875rem;">
                <div>${formatBytes(stats.database?.size_bytes || 0)}</div>
                <div style="color: var(--text-secondary);">${stats.conversations?.count || 0} messages</div>
            </div>
        `;
    } catch (e) {
        document.getElementById('storage-summary').innerHTML = '<span style="color: var(--text-secondary);">Unavailable</span>';
    }

    try {
        const wallet = await API.getWallet();
        state.wallet = wallet;
        document.getElementById('wallet-summary').innerHTML = `
            <div style="font-size: 0.875rem;">
                <div>${wallet.total || 0} ${wallet.currency || 'FIL'}</div>
                <div style="color: var(--text-secondary);">Balance</div>
            </div>
        `;
    } catch (e) {
        document.getElementById('wallet-summary').innerHTML = '<span style="color: var(--text-secondary);">Unavailable</span>';
    }
}
