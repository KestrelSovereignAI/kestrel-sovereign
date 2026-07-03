/**
 * Kestrel Sovereign Console - Identity Module
 * Agent selection, navigation, identity panel, privacy indicator, sidebar
 */

import API from './api.js';
import { state, PRIVACY_MODES, Toast, loadCommands } from './ui.js';
import { disconnectNotifications, connectNotifications, loadModels, updateContextStatus, updateThinkingIndicator, mountChatPane, wipeAgentChatPane, refreshAgentThinkingDot, stopAgent, renderModelFooterHtml, appendMessagePart, splitContentByParts } from './chat.js';
import { generateIdenticon } from './identicon.js';
import { trashGroupKey, groupTrashBySession } from './trash_grouping.js';
// #2149: the sidebar renders rows through the shared conversation-list
// component so its row markup, kebab menu, and inline rename are the SAME as
// the history slideout and any embedded mount — zero duplicated list logic.
import { buildConversationRow, beginInlineRename } from './conversations.js';
// Voice mounts via the slot registry now (#2038, ticket 04); the only remaining
// named coupling is the model-selector ownership lock, deferred to ticket 09.
import { reapplyActiveSelectorLock } from './voice/ui.js';
import { UI } from './ui-ext/registry.js';
import Panels from './ui-ext/panels.js';
import { loadFeatureUIContributions } from './ui-ext/feature-loader.js';
// #2145: core panels are ui-ext panel-registry contributions now. registerCorePanels
// adopts their in-place index.html tabs/bodies and gates them through each panel's
// `gate`; makePanelActivator / attachDelegatedNav are the shared activation code
// path reused by the embeddable mountPanels host so there is a single activation
// path (`Panels.activate`) for standalone and embed.
import { registerCorePanels, makePanelActivator, attachDelegatedNav } from './ui-ext/mount-panels.js';

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
let loadMetrics = null;
let loadFeatureStore = null;
let loadApprovals = null;

export function setLazyLoaders(loaders) {
    loadConstitution = loaders.loadConstitution;
    loadMemories = loaders.loadMemories;
    loadExports = loaders.loadExports;
    loadTasks = loaders.loadTasks;
    loadResources = loaders.loadResources;
    loadMetrics = loaders.loadMetrics;
    loadFeatureStore = loaders.loadFeatureStore;
    loadApprovals = loaders.loadApprovals;
}

// #2145: core-panel gating now lives on each panel's `gate` in the ui-ext panel
// registry (see core-panels.js / registerCorePanels). The old PANEL_CAPABILITIES
// map + `panelIsEnabled` here were the SECOND gating mechanism for the same tabs;
// keeping both would let them disagree, so gating flows through the registry
// alone. `chat` is the one core tab NOT migrated (it has its own `mount()` in
// chat.js and is not a registry panel), so it is gated inline in initNavigation.

// Activate a nav panel: flip the active tab/panel classes, run the core lazy
// loader (standalone owns loading through these state-guarded/per-activation
// calls), then route through the panel registry so a contributed panel's body
// renders on first show and `panel:shown` fires. Factored through
// makePanelActivator (#2145) so the SAME activation path serves the standalone
// console and the embeddable mountPanels host; the loader dispatch below is the
// `beforeActivate` hook (standalone-only — an embedder wires loading via
// core-panels-runtime instead, so there is no double-load).
const activatePanel = makePanelActivator({
    tabScope: document,
    panelScope: document,
    api: API,
    beforeActivate(panelId) {
        state.currentPanel = panelId;

        if (panelId === 'constitution' && !state.constitution && loadConstitution) loadConstitution();
        if (panelId === 'memories' && !state.memories && loadMemories) loadMemories();
        if (panelId === 'sovereignty' && !state.exports && loadExports) loadExports();
        if (panelId === 'tasks' && loadTasks) loadTasks();
        if (panelId === 'resources' && loadResources) loadResources();
        if (panelId === 'metrics' && loadMetrics) loadMetrics();
        if (panelId === 'features' && loadFeatureStore) loadFeatureStore();
        if (panelId === 'approvals' && loadApprovals) loadApprovals();
    },
});

export function initNavigation() {
    const navEl = document.querySelector('.nav-tabs');
    const hostEl = document.querySelector('.main-content');

    // `chat` is the one core tab NOT migrated onto the registry (it has its own
    // mount() in chat.js). Gate its static tab/panel here on host opt-out (#879);
    // every OTHER core panel is a registry contribution gated by its own `gate`.
    if (!API.hasCapability('chat')) {
        document.querySelector('.nav-tab[data-panel="chat"]')?.remove();
        document.getElementById('panel-chat')?.remove();
    }

    // #2145: register the core panels onto the ui-ext panel registry BEFORE
    // renderNav so the registry adopts their in-place index.html tabs/bodies and
    // gates each through its own `gate` (derived from the retired
    // PANEL_CAPABILITIES). A gated-off panel's tab is removed by renderNav below.
    registerCorePanels({ api: API });

    // Bind the panel registry (ticket 06) to the live nav + panel host BEFORE
    // wiring click handling and sync (gates core panels, renders any already
    // registered feature panels). Feature panel modules load later in boot
    // (loadFeatureUIContributions) and call registerPanel(); with the nav bound
    // here, those registrations insert their tab/container immediately.
    Panels.renderNav({ navEl, hostEl, ctx: { api: API } });

    // Delegated click handling (factored + shared with mountPanels, #2145): a
    // single listener on the nav container activates any `.nav-tab`, including
    // ones inserted AFTER boot by registerPanel — a per-tab listener would miss
    // those. This is what lets a feature-owned panel become clickable without a
    // core edit.
    if (navEl) attachDelegatedNav(navEl, activatePanel);

    // If the default-active "chat" tab was removed because the host
    // opted out, promote the first surviving tab to active so the page
    // doesn't open onto a vanished panel.
    promoteActiveTabIfNeeded();

    // #2041: re-gate live when a feature is enabled/disabled at runtime. The
    // boot prune above is destructive (host opt-out is static and authoritative);
    // runtime feature flips toggle visibility on the surviving tabs instead, so
    // a re-enable can restore a panel without a page reload.
    if (typeof globalThis !== 'undefined' && typeof globalThis.addEventListener === 'function') {
        globalThis.addEventListener('capabilities:changed', onCapabilitiesChanged);
    }
}

// #2048: a feature enabled at runtime may only NOW appear in the
// ``/api/ui/contributions`` manifest — at boot (and while it was disabled) the
// server enabled-filtered it out, so its panel module was never imported and
// re-gating alone could not surface its tab. (Re)load the manifest first so any
// newly-enabled feature's module is imported and self-registers its panel, THEN
// re-gate the nav. A disable is symmetric: the module stays import-cached but
// its gate now returns false, so reconcile/syncNav removes the tab + body.
async function onCapabilitiesChanged() {
    await loadFeatureUIContributions();
    reconcileNavigationCapabilities();
}

// Promote the first visible tab to active when the active one is gone/hidden.
function promoteActiveTabIfNeeded() {
    const visibleTabs = Array.from(document.querySelectorAll('.nav-tab'))
        .filter((t) => t.style.display !== 'none');
    const active = visibleTabs.find((t) => t.classList.contains('active'));
    if (active) return;
    const first = visibleTabs[0];
    if (!first) return;
    first.classList.add('active');
    const panelId = first.dataset.panel;
    document.querySelectorAll('.panel').forEach((p) => p.classList.remove('active'));
    document.getElementById(`panel-${panelId}`)?.classList.add('active');
    state.currentPanel = panelId;
}

// #2041/#2145: non-destructive re-gate driven by the capabilities:changed event.
// Every core panel is registry-owned now (#2145), so gating flows entirely
// through the registry: syncNav re-evaluates each panel's `gate` and adds/removes
// its tab + container when a capability flips at runtime (a feature
// enabled/disabled), no reload. `chat` is gated statically at boot (host opt-out
// is authoritative and not runtime-toggled), so nothing per-tab remains here.
export function reconcileNavigationCapabilities() {
    Panels.syncNav();
    promoteActiveTabIfNeeded();
}

// ============================================================================
// Identity Panel
// ============================================================================

export async function loadIdentity() {
    // #879: deep-link defense — no /api/identity fetch when disabled.
    // identity is the default panel so disabling it is unusual but legal.
    if (!API.hasCapability('identity')) return;
    try {
        const identity = await API.getIdentity();
        state.identity = identity;

        // Update page title with companion name if in multi-agent mode
        if (currentAgentId && identity.name) {
            document.title = `${identity.name} - Sovereign Console`;
        }

        // Build avatar URL from canonical identity payload. Hosts that embed
        // Kestrel UI under a different URL shape are responsible for routing
        // /api/identity/avatar (and the upload/generate variants) to the
        // right backend — see #863.
        let avatarUrl = null;
        if (identity.avatar_url) {
            avatarUrl = identity.avatar_url;
        } else if (identity.avatar_hash) {
            avatarUrl = '/api/identity/avatar';
        }

        // Update nav header avatar and name
        const navIcon = document.getElementById('nav-agent-icon');
        const navAvatar = document.getElementById('nav-agent-avatar');
        const navName = document.getElementById('nav-agent-name');

        // Use DID + name as seed so agents sharing a DID still get unique icons
        const identiconSeed = [identity.did, identity.name].filter(Boolean).join(':');
        const identiconUrl = identiconSeed ? generateIdenticon(identiconSeed, 48) : null;

        if (navAvatar && navIcon) {
            const navSrc = avatarUrl || identiconUrl;
            if (navSrc) {
                navAvatar.src = navSrc;
                navAvatar.style.display = 'block';
                navAvatar.onerror = () => {
                    // Custom avatar failed — fall back to identicon, then kestrel icon
                    if (identiconUrl && navAvatar.src !== identiconUrl) {
                        navAvatar.src = identiconUrl;
                    } else {
                        navAvatar.style.display = 'none';
                        navIcon.style.display = 'inline';
                    }
                };
                navIcon.style.display = 'none';
            } else {
                navAvatar.style.display = 'none';
                navIcon.style.display = 'inline';
            }
        }
        if (navName) {
            navName.textContent = identity.name || 'Unnamed Agent';
        }

        // Avatar: custom image → identicon → kestrel logo
        const fallbackSrc = identiconUrl || '/static/favicon.svg';
        const avatarSrc = avatarUrl || fallbackSrc;

        const card = document.getElementById('identity-card');
        card.innerHTML = `
            <div class="identity-header">
                <div class="identity-avatar-wrapper">
                    <div class="identity-avatar">
                        <img src="${avatarSrc}" alt="Avatar" class="identity-avatar-img"
                             id="identity-avatar-img"
                             onerror="this.src='${fallbackSrc}';">
                    </div>
                    <div class="avatar-actions">
                        <button id="avatar-upload-btn" title="Upload image">Upload</button>
                        <button id="avatar-generate-btn" title="Generate with AI">Generate</button>
                        <input type="file" id="avatar-file-input" accept="image/*" style="display:none;">
                    </div>
                </div>
                <div class="identity-info">
                    <input type="text" class="profile-editor-name" id="profile-name"
                           value="${(identity.name || '').replace(/"/g, '&quot;')}"
                           placeholder="Agent name" maxlength="64">
                    <textarea class="profile-editor-desc" id="profile-desc"
                              placeholder="Add a description..." maxlength="500"
                              rows="1">${identity.description || ''}</textarea>
                    <div class="identity-did" title="${identity.did}">
                        <span class="identity-did-text">${identity.did || 'No DID assigned'}</span>
                        <button onclick="copyToClipboard('${identity.did}')" style="background:none;border:none;cursor:pointer;flex-shrink:0;" title="Copy DID">\u{1F4CB}</button>
                    </div>
                    ${_renderHybridIdentityRow(identity)}
                </div>
            </div>
            <div class="avatar-generate-panel" id="avatar-generate-panel" style="display:none;">
                <input type="text" id="avatar-gen-prompt" placeholder="Describe the avatar you want...">
                <button id="avatar-gen-submit">Generate</button>
                <div class="avatar-options" id="avatar-options"></div>
            </div>
        `;

        // Wire up profile editor events
        _wireProfileEditor(identity);

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
// Profile Editor Wiring
// ============================================================================

function _wireProfileEditor(identity) {
    const nameInput = document.getElementById('profile-name');
    const descInput = document.getElementById('profile-desc');
    const uploadBtn = document.getElementById('avatar-upload-btn');
    const generateBtn = document.getElementById('avatar-generate-btn');
    const fileInput = document.getElementById('avatar-file-input');
    const genPanel = document.getElementById('avatar-generate-panel');
    const genPrompt = document.getElementById('avatar-gen-prompt');
    const genSubmit = document.getElementById('avatar-gen-submit');

    // --- Name save on blur/Enter ---
    let savedName = identity.name || '';
    const saveName = async () => {
        const newName = nameInput.value.trim();
        if (!newName || newName === savedName) return;
        try {
            await API.updateIdentity({ name: newName });
            savedName = newName;
            // Update nav header
            const navName = document.getElementById('nav-agent-name');
            if (navName) navName.textContent = newName;
            Toast.success('Name updated');
        } catch (e) {
            Toast.error(`Failed to update name: ${e.message}`);
            nameInput.value = savedName;
        }
    };
    nameInput.addEventListener('blur', saveName);
    nameInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { e.preventDefault(); nameInput.blur(); }
    });

    // --- Description save on blur ---
    let savedDesc = identity.description || '';
    const saveDesc = async () => {
        const newDesc = descInput.value.trim();
        if (newDesc === savedDesc) return;
        try {
            await API.updateIdentity({ description: newDesc });
            savedDesc = newDesc;
            Toast.success('Description updated');
        } catch (e) {
            Toast.error(`Failed to update description: ${e.message}`);
            descInput.value = savedDesc;
        }
    };
    descInput.addEventListener('blur', saveDesc);

    // --- Avatar upload ---
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', async () => {
        const file = fileInput.files[0];
        if (!file) return;
        try {
            uploadBtn.textContent = 'Uploading...';
            uploadBtn.disabled = true;
            const result = await API.uploadAvatar(file);
            if (result.avatar_url) {
                document.getElementById('identity-avatar-img').src = result.avatar_url;
                // Update nav avatar
                const navAvatar = document.getElementById('nav-agent-avatar');
                const navIcon = document.getElementById('nav-agent-icon');
                if (navAvatar) {
                    navAvatar.src = result.avatar_url;
                    navAvatar.style.display = 'block';
                    if (navIcon) navIcon.style.display = 'none';
                }
            }
            Toast.success('Avatar updated');
        } catch (e) {
            Toast.error(`Failed to upload avatar: ${e.message}`);
        } finally {
            uploadBtn.textContent = 'Upload';
            uploadBtn.disabled = false;
            fileInput.value = '';
        }
    });

    // --- Avatar generate toggle ---
    generateBtn.addEventListener('click', () => {
        const visible = genPanel.style.display !== 'none';
        genPanel.style.display = visible ? 'none' : 'block';
        if (!visible) genPrompt.focus();
    });

    // --- Avatar generate submit ---
    const doGenerate = async () => {
        const desc = genPrompt.value.trim();
        if (!desc) return;
        const optionsEl = document.getElementById('avatar-options');
        try {
            genSubmit.disabled = true;
            genSubmit.textContent = 'Generating...';
            optionsEl.innerHTML = '<span style="font-size:0.75rem;color:var(--text-tertiary);">This may take a moment...</span>';

            const result = await API.generateAvatar(desc);

            if (!result.success) {
                Toast.error(result.error || 'Generation failed');
                optionsEl.innerHTML = '';
                return;
            }

            // Show generated options as clickable thumbnails
            optionsEl.innerHTML = '';
            const urls = result.image_urls || [];
            for (const url of urls) {
                const img = document.createElement('img');
                img.src = url;
                img.alt = 'Generated avatar option';
                img.addEventListener('click', async () => {
                    try {
                        Toast.info('Setting avatar...');
                        const setResult = await API.setAvatarFromUrl(url);
                        if (setResult.avatar_url) {
                            document.getElementById('identity-avatar-img').src = setResult.avatar_url;
                            const navAvatar = document.getElementById('nav-agent-avatar');
                            const navIcon = document.getElementById('nav-agent-icon');
                            if (navAvatar) {
                                navAvatar.src = setResult.avatar_url;
                                navAvatar.style.display = 'block';
                                if (navIcon) navIcon.style.display = 'none';
                            }
                        }
                        Toast.success('Avatar set!');
                        genPanel.style.display = 'none';
                    } catch (e) {
                        Toast.error(`Failed to set avatar: ${e.message}`);
                    }
                });
                optionsEl.appendChild(img);
            }

            // If stored_url is available, the first one is already set as primary
            if (result.stored_url) {
                document.getElementById('identity-avatar-img').src = result.stored_url;
                Toast.info('First option auto-saved. Click another to switch.');
            }
        } catch (e) {
            Toast.error(`Generation failed: ${e.message}`);
            optionsEl.innerHTML = '';
        } finally {
            genSubmit.disabled = false;
            genSubmit.textContent = 'Generate';
        }
    };

    genSubmit.addEventListener('click', doGenerate);
    genPrompt.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { e.preventDefault(); doGenerate(); }
    });
}

// ============================================================================
// Privacy Indicator
// ============================================================================

export async function loadPrivacyMode() {
    // #879: deep-link defense — no /api/agent/privacy-mode fetch when disabled.
    // Hosts that don't expose privacy controls (the chip in the chat header)
    // typically opt out so the indicator doesn't render with a stale value.
    if (!API.hasCapability('privacy')) return;
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
                        ${state.privacyMode === mode ? `<span style="font-size: 0.75rem; color: var(--accent-color);">${kicon('checkmark')} Current</span>` : ''}
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
                const result = await API.setPrivacyMode(mode);
                state.privacyMode = mode;
                updatePrivacyIndicator(mode);
                dropdown.remove();

                // Auto-switch model selector when privacy mode requires local-only
                if (result.model_switched && window._sharedModelSelector) {
                    // Save current cloud selection before switching to local
                    if (result.allows_cloud_llm === false) {
                        const current = window._sharedModelSelector.getSelection();
                        localStorage.setItem('kestrel_saved_cloud_provider', current.provider);
                        localStorage.setItem('kestrel_saved_cloud_model', current.model);
                    }
                    window._sharedModelSelector.setSelection(
                        result.model_switched.vendor,
                        result.model_switched.model
                    );
                    Toast.success(`Privacy: ${PRIVACY_MODES[mode]?.label || mode} — switched to ${result.model_switched.vendor} (local only)`);
                } else if (result.allows_cloud_llm !== false) {
                    // Switching back to cloud-allowing mode — restore saved cloud selection
                    const savedProvider = localStorage.getItem('kestrel_saved_cloud_provider');
                    const savedModel = localStorage.getItem('kestrel_saved_cloud_model');
                    if (savedProvider && savedModel && window._sharedModelSelector) {
                        window._sharedModelSelector.setSelection(savedProvider, savedModel);
                        Toast.success(`Privacy: ${PRIVACY_MODES[mode]?.label || mode} — restored ${savedProvider} model`);
                        localStorage.removeItem('kestrel_saved_cloud_provider');
                        localStorage.removeItem('kestrel_saved_cloud_model');
                    } else {
                        Toast.success(`Privacy mode set to ${PRIVACY_MODES[mode]?.label || mode}`);
                    }
                } else {
                    Toast.success(`Privacy mode set to ${PRIVACY_MODES[mode]?.label || mode}`);
                }
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

/**
 * Render the hybrid-identity row beneath the legacy DID line.
 *
 * Three states the response can carry (from /api/identity, post-#1000):
 *   - `is_hybrid === true`: post-rotation. Show the new did:web URI
 *     and a green "HYBRID" badge with succession-chain depth.
 *   - `is_hybrid === false` AND signing_did === did: pre-ceremony,
 *     legacy-only. Render nothing here (identity-did line above is
 *     enough; the agent hasn't migrated yet).
 *   - field absent (older API or pre-#1000 server): also render
 *     nothing — defensive against stale clients.
 */
function _renderHybridIdentityRow(identity) {
    if (!identity || identity.is_hybrid !== true) {
        return '';
    }
    const signingDid = identity.signing_did || identity.did;
    const chain = identity.succession_chain_length;
    const chainNote = (typeof chain === 'number' && chain > 0)
        ? `<span class="hybrid-chain-depth" title="Succession chain depth (number of rotation ceremonies)">chain depth ${chain}</span>`
        : '';
    return `
        <div class="identity-hybrid-row" title="${escapeHtml(signingDid)}">
            <span class="hybrid-badge" title="Post-rotation hybrid identity (Ed25519 + ML-DSA-65). New artifacts are signed with both classical and post-quantum signatures.">HYBRID</span>
            <span class="identity-signing-did-text">${escapeHtml(signingDid)}</span>
            <button onclick="copyToClipboard('${escapeHtml(signingDid)}')" style="background:none;border:none;cursor:pointer;flex-shrink:0;" title="Copy signing DID">\u{1F4CB}</button>
            ${chainNote}
        </div>
    `;
}

// ============================================================================
// Agents Pane (MultiAgent)
// ============================================================================

let selectedAgentName = null;

/**
 * Render the DEMO MODE banner from the /api/agents response (#868).
 *
 * The banner is the browser-side defence against the routing precondition
 * that let the EPHEMERAL-purge wipe through (#867 root cause): a demo run
 * silently mounting live agents and the page auto-targeting one of them.
 *
 * Three states:
 *   - `server_demo_mode === true` AND every agent is demo-scoped → green
 *     "DEMO MODE" pill naming the targeted agent.  Routine demo run.
 *   - `server_demo_mode === true` AND any agent is NOT demo-scoped → red
 *     mismatch banner.  This is the bug the rail was added to catch.
 *   - `server_demo_mode === false` → no banner; production UI looks normal.
 */
function renderDemoModeBanner({ serverDemoMode, agents, isStandalone }) {
    const id = 'demo-mode-banner';
    document.getElementById(id)?.remove();
    if (!serverDemoMode) return;

    const target = isStandalone
        ? agents[0]
        : agents.find((a) => a.status !== 'offline') || agents[0];
    const targetName = target?.name || '<no agent>';
    const liveAgents = agents.filter((a) => a.is_demo !== true).map((a) => a.name);
    const isMisconfig = liveAgents.length > 0;

    const banner = document.createElement('div');
    banner.id = id;
    banner.setAttribute('role', 'status');
    banner.style.cssText = `
        position: sticky; top: 0; z-index: 50;
        padding: 0.5rem 1rem; font-size: 0.8rem; font-weight: 600;
        font-family: ui-sans-serif, system-ui, sans-serif;
        display: flex; align-items: center; gap: 0.5rem;
        ${isMisconfig
            ? 'background: linear-gradient(90deg, #7f1d1d, #b91c1c); color: white;'
            : 'background: linear-gradient(90deg, #14532d, #16a34a); color: white;'}
    `;
    banner.innerHTML = isMisconfig
        ? `<span>${window.kicon('ban')} DEMO MODE MISCONFIG</span>
           <span style="opacity:.85; font-weight: 400;">— this server is in demo mode but mounted live agents: ${liveAgents.map(escapeHtml).join(', ')}.  Refusing to auto-select.</span>`
        : `<span>${window.kicon('beaker')} DEMO MODE</span>
           <span style="opacity:.85; font-weight: 400;">— targeting <code>${escapeHtml(targetName)}</code> (is_demo=true).  Production data is not at risk.</span>`;

    document.body.insertBefore(banner, document.body.firstChild);
}

export async function loadAgents() {
    // #879: hosts that aren't a multi_agent (e.g. Frinz) don't have an
    // /api/agents endpoint — skip the fetch entirely and hide the agents
    // pane so the user doesn't see a "Failed to load agents" card.
    if (!API.hasCapability('multi_agent')) {
        const pane = document.getElementById('agents-pane');
        if (pane) pane.style.display = 'none';
        return;
    }
    try {
        const data = await API.getAgents();
        const agents = data.agents || [];
        const isStandalone = data.mode === 'standalone';

        // DEMO MODE banner (#868) — visible warning so an operator notices
        // when a demo server has somehow mounted a live agent.  Demo mode
        // is determined server-side at startup (every loaded agent is
        // demo-scoped); the banner names the agent the page would target.
        renderDemoModeBanner({
            serverDemoMode: !!data.server_demo_mode,
            agents,
            isStandalone,
        });

        const container = document.getElementById('agents-list');
        if (agents.length === 0) {
            container.innerHTML = '<p style="color: var(--text-secondary); padding: 1rem; text-align: center;">No agents available</p>';
            return;
        }

        container.innerHTML = '';
        for (const agent of agents) {
            const isOnline = agent.status !== 'offline';
            const isThinking = state.waitingAgents.has(agent.name);
            const item = document.createElement('div');
            item.className = `agent-item${selectedAgentName === agent.name ? ' selected' : ''}${!isOnline ? ' offline' : ''}${isThinking ? ' agent-thinking' : ''}`;
            // Always carry the real agent name — thinking-dot / stop-button
            // lookups (refreshAgentThinkingDot, etc.) depend on it in every mode.
            // The voice controls key off a separate data-voice-agent-key.
            item.dataset.agentName = agent.name;

            // Only enable multi_agent agent selection in non-standalone mode
            if (isOnline && !isStandalone) {
                item.addEventListener('click', () => window.selectAgent(agent.name));
            }

            // Per-agent thinking pulse + stop control. Pulse is driven
            // by `state.waitingAgents` (the same Set sendMessage adds
            // to on dispatch and removes from in finally). The stop
            // button is rendered always but only visible while
            // `.agent-thinking` is on (CSS gate); a click reaches the
            // exact agent's /stop endpoint via stopAgent().
            item.innerHTML = `
                <span class="agent-status-dot ${isOnline ? 'online' : 'offline'}"></span>
                <span class="agent-thinking-dot" title="${escapeHtml(agent.name || 'Agent')} is thinking"></span>
                <div class="agent-info">
                    <div class="agent-name">${escapeHtml(agent.name || 'Unnamed Agent')}</div>
                    <div class="agent-description">${escapeHtml(agent.description || 'No description')}</div>
                </div>
                <button class="agent-stop-btn" title="Stop ${escapeHtml(agent.name || 'agent')}" aria-label="Stop ${escapeHtml(agent.name || 'agent')}">&times;</button>
            `;

            const stopBtn = item.querySelector('.agent-stop-btn');
            if (stopBtn) {
                stopBtn.addEventListener('click', (e) => {
                    e.stopPropagation();  // don't also fire selectAgent
                    stopAgent(agent.name);
                });
            }
            container.appendChild(item);

            // UI extension slot (#2038): per-card actions zone. The anchor is a
            // dedicated container appended to the card; the registry mounts
            // contributions (voice's 🎧/🎤 controls register into this zone as of
            // ticket 04) and tears them down when the card detaches on the next
            // list rebuild.
            const cardActionsAnchor = document.createElement('div');
            cardActionsAnchor.dataset.slot = 'agent-card-actions';
            cardActionsAnchor.className = 'agent-card-actions';
            item.appendChild(cardActionsAnchor);
            UI.renderSlot('agent-card-actions', {
                element: cardActionsAnchor,
                api: API,
                agentName: agent.name,
                standalone: isStandalone,
            });
        }

        // Demo-server misconfig (#868): a server in demo_mode that mounted
        // any non-demo agent is the routing precondition that wiped
        // Meridian.  The banner already names the bad state — but words
        // alone aren't enough; the page MUST also refuse to install a
        // host-agent prefix, otherwise every subsequent UI click still
        // routes to whichever agent ``selectAgent`` would have picked.
        const hasLiveAgent = (data.server_demo_mode === true) && agents.some(
            (a) => a.is_demo !== true,
        );

        // Auto-select first online agent only in multi_agent mode (not standalone)
        // and never when the demo server is in misconfig — the banner
        // explicitly tells the operator the auto-select is disabled.
        if (!selectedAgentName && !isStandalone && !hasLiveAgent) {
            const firstOnline = agents.find(a => a.status !== 'offline');
            if (firstOnline) {
                window.selectAgent(firstOnline.name);
            }
        }

        // Standalone mode: reveal the conversations pane (and its Trash sub-view
        // from #765) without going through selectAgent.  selectAgent installs a
        // host-agent URL prefix that only exists in multi_agent routing — applying
        // it in standalone produces 404s for /api/conversations and /agent/invoke.
        // Standalone has exactly one agent, so just show the pane and let
        // loadConversations() populate the list against the un-prefixed routes.
        // Skip in misconfig — the agent list is not safe to auto-target.
        if (isStandalone && agents.length > 0 && !hasLiveAgent && API.hasCapability('conversations')) {
            // #879: don't reveal the conversations pane (or fire its fetch)
            // when the host opted out — its own loadConversations() guard
            // would hide the pane after we re-show it, but the show/hide
            // race produces a one-frame flicker we can avoid by gating here.
            const conversationsPane = document.getElementById('conversations-pane');
            if (conversationsPane) conversationsPane.style.display = 'flex';
            try { await loadConversations(agents[0].name); } catch (_) { /* best-effort */ }
        }
    } catch (e) {
        const container = document.getElementById('agents-list');
        container.innerHTML = '<p style="color: var(--error); padding: 1rem;">Failed to load agents</p>';
    }
}

window.selectAgent = async function(agentName) {
    const previousAgentName = selectedAgentName;
    selectedAgentName = agentName;

    // Set host agent routing in API layer
    API.setHostAgent(agentName);

    // #2041: derive this agent's capability set now that routing is pinned. At
    // boot in multi-agent host mode no agent is selected yet, so the server
    // could not inject featureCapabilities and the boot-time refresh hit the
    // host's un-prefixed /api/ui/capabilities (no agent context) and was
    // swallowed — leaving feature-backed panels defaulting on. With routing set,
    // /api/ui/capabilities is host-agent-prefixed and resolves; applying it emits
    // capabilities:changed which non-destructively re-gates the nav (the boot
    // prune left every tab present because capsMap was empty). Done before the
    // panel loaders below so each self-guards against the fresh set.
    await API.refreshCapabilities();

    // #2048: (re)load the feature UI-contributions manifest now that routing is
    // pinned. In multi-agent host mode the boot-time call in app.js hit the
    // host's un-prefixed /api/ui/contributions with NO active agent and 503'd, so
    // feature-owned panel modules (the extracted Spawn panel) were never imported
    // and their tabs never appeared. With the host agent set the request resolves
    // and each enabled feature's module imports + self-registers its panel. The
    // capabilities:changed emitted by refreshCapabilities above may already have
    // kicked off this same load — the loader coalesces concurrent runs, and we
    // await here so the panels are registered before the reactivation below.
    await loadFeatureUIContributions();

    // Mount the new agent's chat pane. Streams already in flight
    // against the previous agent's pane keep painting into that
    // (now-detached) pane — when the user switches back, their work
    // is preserved exactly where they left it. Agent switch does NOT
    // bump any generation; only within-agent context changes
    // (clear/new chat, conversation switch, delete) do.
    mountChatPane(agentName);
    // UI extension bus (#2038): the generic agent-switch event drives voice's
    // per-agent card repaint + session policy (voice subscribes on the bus as of
    // ticket 04 — no named core→voice call).
    UI.emit('agent:switch', { prev: previousAgentName, next: agentName });

    // Refresh the chat-input "Thinking…" indicator + send/input disabled
    // state from the new agent's waiting status. If the previous agent
    // was mid-stream, its indicator stays in `state.waitingAgents` for
    // its own bookkeeping but only the *current* agent's status is
    // reflected in the UI.
    updateThinkingIndicator();

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
    prepareConversationsPaneForAgent(agentName);

    // Reset cached panel data so they reload for the new agent. Note
    // that `state.currentSessionId` is per-pane now (its getter reads
    // from the mounted agent's pane) — explicitly NOT nulling it so
    // each agent keeps its own session across switches.
    state.identity = null;
    state.constitution = null;
    state.memories = null;
    state.exports = null;
    state.storage = null;
    state.wallet = null;

    // Reconnect SSE notifications to the new agent.  #879: every
    // chat-adjacent helper here (connectNotifications, loadModels,
    // loadCommands, updateContextStatus) self-guards on the chat
    // capability, so when a host disables chat these become no-ops and
    // the chat-only endpoints (/api/agent/notifications/sse, /api/models,
    // /api/commands, context-status) are never hit on agent select.
    disconnectNotifications();
    connectNotifications();

    // Reload all agent-specific data in parallel.
    await Promise.all([
        loadIdentity(),
        loadPrivacyMode(),
        loadConversations(agentName),
        loadModels(),
        loadCommands(API),
        updateContextStatus(),
    ]);

    // loadModels() above rebuilt the chat-model selector, discarding any lock
    // the agent:switch handler acquired. Re-lock to the now-active agent's live
    // Realtime session (no-op otherwise) so voice keeps owning the model. This
    // is the model-selector ownership lock — the one voice surface still coupled
    // by name, deferred to ticket 09.
    reapplyActiveSelectorLock();

    // Reload the currently active panel (its cached state was cleared above)
    const panel = state.currentPanel;
    if (panel === 'constitution' && loadConstitution) loadConstitution();
    if (panel === 'memories' && loadMemories) loadMemories();
    if (panel === 'sovereignty' && loadExports) loadExports();
    if (panel === 'tasks' && loadTasks) loadTasks();
    if (panel === 'resources' && loadResources) loadResources();
    if (panel === 'features' && loadFeatureStore) loadFeatureStore();
    // Registry-owned panels (#2048, e.g. Spawn) reload their data off the
    // `panel:shown` event the registry emits; re-activate the active one so it
    // refetches for the newly-selected agent (Panels.activate is idempotent —
    // the body render is skipped after the first show).
    if (panel && Panels.panels().some(p => p.panelId === panel)) {
        Panels.activate(panel, { api: API });
    }
};

// ============================================================================
// Conversations Pane
// ============================================================================

let activeConversationId = null;
let conversationListRequestSeq = 0;
const activeConversationIdsByAgent = new Map();

function currentAgentMatches(expectedAgent) {
    return expectedAgent === API.getHostAgent();
}

function hasExpectedAgent(options) {
    return Object.prototype.hasOwnProperty.call(options || {}, 'expectedAgent');
}

function conversationAgentKey(agentName) {
    return agentName === null || agentName === undefined
        ? '__standalone__'
        : String(agentName);
}

function getActiveConversationIdForAgent(agentName) {
    const pane = state.chatPanes.get(agentName);
    return pane?.sessionId || activeConversationIdsByAgent.get(agentName) || null;
}

function prepareConversationsPaneForAgent(agentName) {
    const container = document.getElementById('conversations-list');
    if (!container) return;
    container.dataset.agentKey = conversationAgentKey(agentName);
    activeConversationId = getActiveConversationIdForAgent(agentName);
    container.innerHTML = '<p style="color: var(--text-secondary); padding: 1rem; text-align: center;">Loading conversations...</p>';
}

// Exported for unit test — kept as a pure function so it's trivial to
// verify without standing up DOM + fetch mocks.  Callers should treat
// it as internal (prefix stays ``_``).
export function _pickMostRecentConversation(conversations) {
    // Choose the conversation with the latest `last_message_at` (fallback
    // to `started_at` if the server doesn't surface the former).  Sorted
    // strictly descending; ties broken by later array index.  Ephemeral
    // sessions never appear here because the privacy wrapper doesn't
    // persist them — so we don't need to filter.
    if (!Array.isArray(conversations) || conversations.length === 0) {
        return null;
    }
    let best = null;
    let bestTs = -Infinity;
    for (const conv of conversations) {
        const raw = conv.last_message_at || conv.started_at || 0;
        const ts = new Date(raw).getTime();
        if (!Number.isFinite(ts)) continue;
        if (ts >= bestTs) {
            bestTs = ts;
            best = conv;
        }
    }
    return best;
}

export async function loadConversations(_agentName) {
    // #879: deep-link defense — hosts with their own chat surface (e.g. Frinz)
    // typically disable the conversations sidebar entirely.  Hide the pane and
    // skip the /api/conversations fetch.
    if (!API.hasCapability('conversations')) {
        const pane = document.getElementById('conversations-pane');
        if (pane) pane.style.display = 'none';
        return;
    }
    // Capture the host agent BEFORE the fetch.  The auto-load below
    // pins to whichever agent's list this call actually retrieved.
    // Reading ``selectedAgentName`` (or ``API.getHostAgent()``) AFTER
    // the await would catch the mutated value if a concurrent
    // selectAgent(B) flipped it — that's the codex round-1 race on
    // #1358: the list is A's, but the guard pins to B, and the
    // auto-load fires A's session_id under B's URL.
    //
    // Use the ROUTING key (``API.getHostAgent()``), NOT the display
    // name passed via ``_agentName``.  In standalone single-agent
    // mode ``getHostAgent()`` returns null and the chat pane is
    // keyed by null too; using ``_agentName`` (a display string)
    // would miss the null-keyed pane and ``paneIsCold`` would read
    // false, regressing the standalone auto-load.  Codex round-2 catch.
    const requestAgent = API.getHostAgent();
    const requestSeq = ++conversationListRequestSeq;
    const requestAgentKey = conversationAgentKey(requestAgent);
    prepareConversationsPaneForAgent(requestAgent);
    // Agent routing is handled by API.setHostAgent() — all calls auto-prefix
    try {
        const data = await API.getConversations();
        const conversations = data.conversations || [];

        const container = document.getElementById('conversations-list');
        if (!container
            || !currentAgentMatches(requestAgent)
            || requestSeq !== conversationListRequestSeq
            || container.dataset.agentKey !== requestAgentKey) {
            return;
        }
        activeConversationId = getActiveConversationIdForAgent(requestAgent);
        if (conversations.length === 0) {
            container.innerHTML = '<p style="color: var(--text-secondary); padding: 1rem; text-align: center;">No conversations yet</p>';
            return;
        }

        container.innerHTML = '';
        for (const conv of conversations) {
            // Build the row through the shared component so the kebab menu,
            // inline rename, and markup match every other conversation surface
            // (#2149). The sidebar keeps its own agent-pinning: the row carries
            // ``data-agent-key`` and the click is gated so a stale row can't
            // dispatch under a switched host (#1358 — enforced by the
            // conversation_agent_switch test). Menu actions route through the
            // pane-aware window.* handlers below, which own the chat-state
            // coordination (fresh-session on delete, etc.).
            const item = buildConversationRow(conv, {
                api: API,
                active: activeConversationId === conv.session_id,
                agentKey: requestAgentKey,
                onSelect: (c, rowEl) => {
                    if (rowEl.dataset.agentKey !== conversationAgentKey(API.getHostAgent())) {
                        return;
                    }
                    window.loadConversation(c.session_id, { expectedAgent: requestAgent });
                },
                buildMenuItems: (c, rowEl) => [
                    {
                        label: 'Rename', labelKey: 'conv_menu_rename', action: 'rename',
                        onSelect: () => {
                            const previewEl = rowEl.querySelector
                                ? rowEl.querySelector('.conversation-preview')
                                : null;
                            if (previewEl) beginInlineRename(previewEl, c, { api: API });
                        },
                    },
                    {
                        label: 'Archive', labelKey: 'conv_menu_archive', action: 'archive',
                        onSelect: () => window.archiveConversation(c.session_id, rowEl),
                    },
                    {
                        label: 'Move to Trash', labelKey: 'conv_menu_trash', action: 'trash',
                        onSelect: () => window.deleteConversation(c.session_id, rowEl),
                    },
                    {
                        label: 'Delete Permanently', labelKey: 'conv_menu_delete_permanent',
                        action: 'purge', danger: true, separatorBefore: true,
                        onSelect: () => window.purgeConversation(c.session_id, rowEl),
                    },
                ],
            });
            container.appendChild(item);
        }

        // Auto-load the most recent conversation on agent select (issue #714).
        // Only when the chat pane is truly cold:
        //   - no currentSessionId has been set since selectAgent mounted
        //   - no in-flight stream against this agent
        //   - no DOM activity in this agent's pane (user might have typed
        //     during the parallel /api/conversations fetch)
        //
        // Without all three checks, an auto-load that lands AFTER the user
        // started typing would call wipeAgentChatPane(), bump the pane
        // generation, and gate out the in-flight stream — which surfaces
        // user-side as "the agent keeps stopping mid-answer."
        //
        // Even with the synchronous checks there's a residual race between
        // here and the actual wipe inside loadConversation. The auto path
        // re-checks post-await; see the {auto: true} branch in
        // window.loadConversation below.
        // ``autoTargetAgent`` was previously read from the mutable
        // ``selectedAgentName`` at response time — codex round-1 P2 on
        // #1358 — which captured the LATEST agent rather than the one
        // whose conversation list we actually fetched.  Pin to the
        // request-time agent captured before the await above.
        const autoTargetAgent = requestAgent;
        const autoTargetPane = state.chatPanes.get(autoTargetAgent);
        const paneIsCold = autoTargetPane
            && !autoTargetPane.streamingMsgDiv
            && autoTargetPane.element.children.length === 0;
        if (!state.currentSessionId
            && !state.waitingAgents.has(autoTargetAgent)
            && paneIsCold
            && typeof window.loadConversation === 'function') {
            const mostRecent = _pickMostRecentConversation(conversations);
            if (mostRecent && mostRecent.session_id) {
                // Fire-and-forget; loadConversation is async and handles
                // its own errors via Toast. The {auto: true} flag tells
                // loadConversation to re-check pane coldness after its
                // own fetch resolves and abort the wipe if the user has
                // begun a turn in the meantime.
                //
                // ``expectedAgent`` is pinned to the agent the LIST was
                // fetched for — loadConversation drops the load if
                // ``API.getHostAgent()`` has since switched.  See #1358.
                window.loadConversation(mostRecent.session_id, {
                    auto: true,
                    expectedAgent: autoTargetAgent,
                });
            }
        }
    } catch (e) {
        const container = document.getElementById('conversations-list');
        if (currentAgentMatches(requestAgent)
            && requestSeq === conversationListRequestSeq
            && container
            && container.dataset.agentKey === requestAgentKey) {
            container.innerHTML = '<p style="color: var(--error); padding: 1rem;">Failed to load conversations</p>';
        }
    }
}


// #2149: archive/unarchive a whole session. Archive is a first-class
// tidy-away state distinct from Trash: an archived conversation is hidden
// from the default (active) list but is NOT staged for deletion. These are
// the pane-aware sidebar handlers; the shared component's kebab menu routes
// here so the chat-state coordination (below) is single-sourced.
window.archiveConversation = async function(sessionId, rowEl) {
    try {
        const result = await API.archiveConversation(sessionId);
        const count = result?.archived_count;
        if (rowEl) {
            rowEl.style.transition = 'opacity 0.2s, transform 0.2s';
            rowEl.style.opacity = '0';
            rowEl.style.transform = 'scale(0.97)';
            setTimeout(() => rowEl.remove(), 200);
        }
        Toast.info(
            typeof count === 'number'
                ? `Conversation archived (${count} messages)`
                : 'Conversation archived',
        );
    } catch (e) {
        Toast.error(`Failed to archive conversation: ${e.message}`);
    }
};

window.unarchiveConversation = async function(sessionId, rowEl) {
    try {
        await API.unarchiveConversation(sessionId);
        if (rowEl) {
            rowEl.style.transition = 'opacity 0.2s';
            rowEl.style.opacity = '0';
            setTimeout(() => rowEl.remove(), 200);
        }
        Toast.success('Conversation unarchived');
        if (typeof loadConversations === 'function') {
            try { await loadConversations(selectedAgentName); } catch (_) { /* noop */ }
        }
    } catch (e) {
        Toast.error(`Failed to unarchive conversation: ${e.message}`);
    }
};


// #2081: render an assistant turn that emitted typed component parts (#1914)
// as interleaved bubbles — prose runs and the component cards between them —
// into ``container`` (the owning agent's pane element, which carries
// ``dataset.agent`` so ``appendMessagePart`` pins each card to the RIGHT
// agent). This sidebar/auto-load loader is otherwise a flat markdown renderer;
// without this path a persisted ``channel_link`` card (and any other part)
// would be silently dropped on a hard refresh or sidebar conversation load.
// Mirrors ``history.js::renderAssistantWithParts`` at this loader's simpler
// altitude (no tool-card interleaving): the first rendered bubble anchors the
// message id + delete control + model footer so they aren't duplicated.
function renderAssistantMessageWithParts(msg, container) {
    const renderMd = window.SharedMarkdown?.renderMarkdown;
    const segments = splitContentByParts(msg.content, msg.metadata?.parts);
    let anchored = false;
    const anchor = (node) => {
        if (!node) return;
        if (msg.id) node.dataset.messageId = msg.id;
        if (anchored) return;
        anchored = true;
        if (msg.id) {
            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'msg-delete-btn';
            deleteBtn.title = 'Delete message';
            deleteBtn.textContent = '✕';
            deleteBtn.onclick = (e) => {
                e.stopPropagation();
                if (typeof window.deleteMessage === 'function') {
                    window.deleteMessage(msg.id, node);
                }
            };
            node.appendChild(deleteBtn);
        }
        const footer = renderModelFooterHtml({ model: msg.model, provider: msg.provider });
        if (footer) node.insertAdjacentHTML('beforeend', footer);
    };
    for (const seg of segments) {
        if (seg.kind === 'part') {
            const pnode = appendMessagePart(seg.part.type, seg.part.data, container);
            anchor(pnode);
            continue;
        }
        // Skip an empty prose run (e.g. the gap between two adjacent parts, or a
        // part-only message) so no blank bubble appears.
        if (!String(seg.text || '').trim()) continue;
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message agent-message';
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        if (renderMd) contentDiv.innerHTML = renderMd(seg.text);
        else contentDiv.textContent = seg.text;
        messageDiv.appendChild(contentDiv);
        container.appendChild(messageDiv);
        anchor(messageDiv);
    }
}

window.loadConversation = async function(sessionId, options = {}) {
    // Stale-load guard: when a row load was queued for one agent and the
    // operator has since selectAgent'd to a different host, the queued
    // session_id is no longer addressable on the new agent — running this
    // would issue a 404'ing GET against the new agent's URL with the prior
    // agent's session_id. Drop the load. See #1358 / #1604.
    if (hasExpectedAgent(options) && !currentAgentMatches(options.expectedAgent)) {
        return;
    }

    activeConversationId = sessionId;
    activeConversationIdsByAgent.set(API.getHostAgent(), sessionId);

    // Update selection UI
    document.querySelectorAll('.conversation-item').forEach(item => {
        item.classList.toggle(
            'active',
            item.dataset.agentKey === conversationAgentKey(API.getHostAgent())
                && item.dataset.sessionId === sessionId,
        );
    });

    // Load conversation messages into chat panel
    try {
        const data = await API.getConversation(sessionId);
        const messages = data.messages || [];

        // Post-await stale-agent re-check: the operator may have
        // switched host agents while ``getConversation`` was in flight.
        // Without this, an auto-load that started on agent A would
        // wipe + render A's history into B's pane after the user
        // switched.  The pre-await guard above catches the case where
        // the switch happened BEFORE the fetch; this one catches a
        // switch DURING.  Codex round-3 catch on #1358.
        if (hasExpectedAgent(options) && !currentAgentMatches(options.expectedAgent)) {
            return;
        }

        const currentAgent = API.getHostAgent();

        // Auto-load defense-in-depth: the loadConversations() caller
        // already checked the pane was cold synchronously, but the
        // /api/conversations fetch + this getConversation() fetch above
        // ran across multiple awaits — plenty of time for the user to
        // type and submit. If they did, abort the wipe so we don't bump
        // the pane generation under their in-flight stream. User-
        // explicit clicks (no `auto` flag) skip this guard: the user's
        // intent to switch conversations is overriding.
        if (options.auto) {
            const pane = state.chatPanes.get(currentAgent);
            const paneIsCold = pane
                && !pane.streamingMsgDiv
                && pane.element.children.length === 0;
            const userBusy = state.waitingAgents.has(currentAgent);
            const sessionAlreadySet = !!state.currentSessionId;
            if (!paneIsCold || userBusy || sessionAlreadySet) {
                // User started a turn while auto-load was in flight.
                // Drop the auto-load silently — the in-flight stream is
                // what the user actually wants to see.
                return;
            }
        }

        // Wipe ONLY the visible agent's pane and bump that agent's
        // pane-local generation. A stream still running against the
        // previous conversation on this agent gates out before its
        // chunks can paint the freshly-loaded view; other agents'
        // streams are untouched.
        wipeAgentChatPane(currentAgent);
        state.currentSessionId = sessionId;
        // Append messages directly into this agent's pane element.
        const pane = state.chatPanes.get(currentAgent);
        const chatContainer = pane ? pane.element : document.getElementById('chat-container');

        const renderMd = window.SharedMarkdown?.renderMarkdown;

        for (const msg of messages) {
            // #2081/#1914: an assistant turn carrying typed component parts
            // re-renders as interleaved prose + component bubbles so a persisted
            // card (e.g. the WhatsApp channel_link QR) survives this load path.
            const parts = msg.metadata?.parts;
            if (msg.role === 'assistant' && Array.isArray(parts) && parts.length) {
                renderAssistantMessageWithParts(msg, chatContainer);
                continue;
            }

            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${msg.role === 'user' ? 'user-message' : 'agent-message'}`;

            // Hover-reveal delete control.  Shares CSS (.msg-delete-btn)
            // and the window.deleteMessage handler with history.js — the
            // intent is that every rendered historical message is
            // deletable, regardless of WHICH loader painted it.  Before
            // issue #715 this was only on the history-panel path, so
            // users loading a conversation from the sidebar saw no way
            // to delete anything.
            if (msg.id) {
                messageDiv.dataset.messageId = msg.id;
                const deleteBtn = document.createElement('button');
                deleteBtn.className = 'msg-delete-btn';
                deleteBtn.title = 'Delete message';
                deleteBtn.textContent = '✕';
                deleteBtn.onclick = (e) => {
                    e.stopPropagation();
                    if (typeof window.deleteMessage === 'function') {
                        window.deleteMessage(msg.id, messageDiv);
                    }
                };
                messageDiv.appendChild(deleteBtn);
            }

            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';

            if (msg.role === 'assistant' && renderMd) {
                contentDiv.innerHTML = renderMd(msg.content);
            } else {
                contentDiv.textContent = msg.content;
            }

            messageDiv.appendChild(contentDiv);
            if (msg.role === 'assistant') {
                const footer = renderModelFooterHtml({
                    model: msg.model,
                    provider: msg.provider,
                });
                if (footer) messageDiv.insertAdjacentHTML('beforeend', footer);
            }
            chatContainer.appendChild(messageDiv);
        }

        // Sync scroll on the live viewport (#chat-container is the
        // overflow:auto element; the pane is its child). Only sync if
        // this is the visible agent's pane — detached panes get scroll
        // restored on remount.
        const viewport = document.getElementById('chat-container');
        if (viewport && pane && pane.element.parentNode === viewport) {
            viewport.scrollTop = viewport.scrollHeight;
        } else if (pane) {
            pane.scrollPos = Number.MAX_SAFE_INTEGER;  // snap to bottom on mount
        }

        // Switch to chat panel
        document.querySelector('[data-panel="chat"]')?.click();
    } catch (e) {
        console.error('Failed to load conversation:', e);
    }
};

window.deleteConversation = async function(sessionId, rowEl) {
    // Soft-delete (#763) — moves the conversation to Trash, recoverable
    // via the trash sub-view (#765). Companion to per-message delete
    // (#715). Goes through the privacy wrapper which rejects ephemeral
    // mode.
    if (!confirm(
        'Move this conversation to Trash? You can restore it from the '
        + 'trash view, or delete it permanently from there.'
    )) {
        return;
    }

    try {
        const result = await API.deleteConversation(sessionId);
        const count = result?.deleted_count;

        // Animate the sidebar row out, then remove it so the list
        // doesn't jump underneath the cursor.
        if (rowEl) {
            rowEl.style.transition = 'opacity 0.2s, transform 0.2s';
            rowEl.style.opacity = '0';
            rowEl.style.transform = 'scale(0.97)';
            setTimeout(() => rowEl.remove(), 200);
        }

        // If the user was viewing the conversation they just deleted,
        // immediately create a fresh backend session so downstream state
        // (context-status footer, auto-load logic) doesn't point at a
        // vanished session.
        if (state.currentSessionId === sessionId) {
            activeConversationId = null;
            activeConversationIdsByAgent.delete(API.getHostAgent());
            const fresh = await API.newConversation();
            wipeAgentChatPane(API.getHostAgent(), `
                <div style="text-align: center; padding: 2rem; color: var(--text-secondary);">
                    <span style="font-size: 2rem;">\u{2728}</span>
                    <p style="margin-top: 0.5rem;">New conversation started. Say hello!</p>
                </div>
            `);
            state.currentSessionId = fresh.session_id;
            activeConversationId = fresh.session_id;
            activeConversationIdsByAgent.set(API.getHostAgent(), fresh.session_id);
            if (selectedAgentName) {
                await loadConversations(selectedAgentName);
            }
            if (typeof updateContextStatus === 'function') {
                updateContextStatus();
            }
        }

        Toast.info(
            typeof count === 'number'
                ? `Conversation moved to trash (${count} messages)`
                : 'Conversation moved to trash'
        );
    } catch (e) {
        Toast.error(`Failed to delete conversation: ${e.message}`);
    }
};

window.purgeConversation = async function(sessionId, rowEl) {
    // Permanent delete (#765). Stronger confirm: the word "permanent"
    // appears in bold in the dialog so the user can't muscle-memory
    // through it. Hard SQL DELETE — no recovery possible.
    if (!confirm(
        `Delete this conversation PERMANENTLY?\n\n`
        + `This is a hard delete — every message in the session will be `
        + `removed and CANNOT be restored. Soft-delete first (the regular `
        + `delete button) is the recoverable path.\n\n`
        + `Type the word "permanent" in your head and click OK if you mean it.`
    )) {
        return;
    }

    try {
        const result = await API.purgeConversation(sessionId, 'user-initiated-ui');
        const count = result?.purged_count;

        if (rowEl) {
            rowEl.style.transition = 'opacity 0.2s, transform 0.2s';
            rowEl.style.opacity = '0';
            rowEl.style.transform = 'scale(0.97)';
            setTimeout(() => rowEl.remove(), 200);
        }

        if (state.currentSessionId === sessionId) {
            activeConversationId = null;
            activeConversationIdsByAgent.delete(API.getHostAgent());
            // Wipe ONLY the visible agent's pane and bump that agent's
            // pane-local generation. In-flight streams on other agents
            // are unaffected.
            wipeAgentChatPane(API.getHostAgent());
            state.currentSessionId = null;
            if (typeof updateContextStatus === 'function') {
                updateContextStatus();
            }
        }

        Toast.info(
            typeof count === 'number'
                ? `Conversation permanently deleted (${count} messages)`
                : 'Conversation permanently deleted'
        );
    } catch (e) {
        Toast.error(`Failed to permanently delete: ${e.message}`);
    }
};

window.restoreConversation = async function(sessionId, rowEl) {
    // Pull a soft-deleted session back out of Trash (#765). No confirm
    // needed — restore is a non-destructive action.
    try {
        const result = await API.restoreConversation(sessionId);
        const count = result?.restored_count;

        if (rowEl) {
            rowEl.style.transition = 'opacity 0.2s';
            rowEl.style.opacity = '0';
            setTimeout(() => rowEl.remove(), 200);
        }

        Toast.success(
            typeof count === 'number'
                ? `Conversation restored (${count} messages)`
                : 'Conversation restored'
        );

        // Refresh the regular conversations list so the restored row
        // reappears in the right place (without forcing a full reload).
        if (typeof loadConversations === 'function') {
            try { await loadConversations(); } catch (_) { /* noop */ }
        }
    } catch (e) {
        Toast.error(`Failed to restore: ${e.message}`);
    }
};

// ============================================================================
// Trash sub-view (#765)
// ============================================================================
//
// The Trash button in the conversations pane header swaps the pane's
// content between the live conversations list and a trash list. Both
// share the pane shell so the user sees the toggle as a *view*, not a
// modal — which keeps the multi_agent-and-conversation switcher behavior
// undisturbed.

export async function loadTrash() {
    const container = document.getElementById('conversations-trash');
    if (!container) return;
    container.innerHTML = '<p style="color: var(--text-secondary); padding: 1rem; text-align: center;">Loading trash…</p>';

    try {
        const data = await API.listTrash(500);
        const messages = data.messages || [];
        if (messages.length === 0) {
            container.innerHTML = '<p style="color: var(--text-secondary); padding: 1rem; text-align: center;">Nothing in trash.</p>';
            return;
        }

        const { sessions, orphans } = groupTrashBySession(messages);

        // Combine sessions and orphans into one list, sort by deleted_at desc.
        const items = [
            ...sessions.map((s) => ({ kind: 'session', ...s })),
            ...orphans.map((m) => ({
                kind: 'message',
                message_id: m.id,
                deleted_at: m.deleted_at,
                preview: m.content?.slice(0, 80) || '(empty)',
                role: m.role,
            })),
        ];
        items.sort((a, b) => (b.deleted_at || '').localeCompare(a.deleted_at || ''));

        // Bucket
        const buckets = new Map();
        for (const item of items) {
            const key = trashGroupKey(item.deleted_at);
            if (!buckets.has(key)) buckets.set(key, []);
            buckets.get(key).push(item);
        }

        // Render in the canonical bucket order
        const order = ['Today', 'Yesterday', 'Last 7 days', 'Older'];
        container.innerHTML = '';
        for (const key of order) {
            const list = buckets.get(key);
            if (!list || !list.length) continue;
            const title = document.createElement('div');
            title.className = 'trash-section-title';
            title.textContent = key;
            container.appendChild(title);
            for (const item of list) {
                container.appendChild(_renderTrashItem(item));
            }
        }

        // Retention notice — v1 hardcodes the value the retention janitor
        // (#764) defaults to. When that ticket lands the value comes
        // from the agent's config.
        const notice = document.createElement('div');
        notice.className = 'trash-retention-notice';
        notice.textContent = 'Trash items are automatically deleted after 30 days.';
        container.appendChild(notice);
    } catch (e) {
        container.innerHTML = `<p style="color: var(--error); padding: 1rem;">Failed to load trash: ${e.message}</p>`;
    }
}

function _renderTrashItem(item) {
    const row = document.createElement('div');
    row.className = 'trash-item';

    const preview = document.createElement('div');
    preview.className = 'trash-preview';
    preview.textContent = item.kind === 'session'
        ? (item.preview || '(empty conversation)')
        : `${item.role || 'msg'}: ${item.preview || '(empty)'}`;
    row.appendChild(preview);

    const meta = document.createElement('div');
    meta.className = 'trash-meta';
    const when = item.deleted_at ? new Date(item.deleted_at).toLocaleString() : 'unknown time';
    const sub = item.kind === 'session'
        ? `${item.count} message${item.count === 1 ? '' : 's'}`
        : 'single message';
    meta.innerHTML = `<span>${when}</span><span>${sub}</span>`;
    row.appendChild(meta);

    const actions = document.createElement('div');
    actions.className = 'trash-actions';

    const restore = document.createElement('button');
    restore.className = 'btn-restore';
    restore.textContent = 'Restore';
    restore.addEventListener('click', async () => {
        if (item.kind === 'session') {
            await window.restoreConversation(item.session_id, row);
        } else {
            await _restoreMessageFromTrash(item.message_id, row);
        }
        // Refresh the trash list so counts/buckets stay accurate.
        await loadTrash();
    });

    const purge = document.createElement('button');
    purge.className = 'btn-purge';
    purge.textContent = 'Delete permanently';
    purge.addEventListener('click', async () => {
        if (item.kind === 'session') {
            await window.purgeConversation(item.session_id, row);
        } else {
            await _purgeMessageFromTrash(item.message_id, row);
        }
        await loadTrash();
    });

    actions.appendChild(restore);
    actions.appendChild(purge);
    row.appendChild(actions);
    return row;
}

async function _restoreMessageFromTrash(messageId, rowEl) {
    try {
        await API.restoreMessage(messageId);
        if (rowEl) {
            rowEl.style.transition = 'opacity 0.2s';
            rowEl.style.opacity = '0';
            setTimeout(() => rowEl.remove(), 200);
        }
        Toast.success('Message restored');
    } catch (e) {
        Toast.error(`Failed to restore message: ${e.message}`);
    }
}

async function _purgeMessageFromTrash(messageId, rowEl) {
    if (!confirm(
        `Delete this message PERMANENTLY?\n\n`
        + `This is a hard delete — the message will be removed and CANNOT `
        + `be restored.`
    )) {
        return;
    }
    try {
        await API.purgeMessage(messageId, 'user-initiated-ui');
        if (rowEl) {
            rowEl.style.transition = 'opacity 0.2s';
            rowEl.style.opacity = '0';
            setTimeout(() => rowEl.remove(), 200);
        }
        Toast.info('Message permanently deleted');
    } catch (e) {
        Toast.error(`Failed to permanently delete: ${e.message}`);
    }
}

export function initTrashToggle() {
    const btn = document.getElementById('trash-toggle-btn');
    const title = document.getElementById('conversations-pane-title');
    const list = document.getElementById('conversations-list');
    const trash = document.getElementById('conversations-trash');
    if (!btn || !list || !trash) return;

    btn.addEventListener('click', async () => {
        const showingTrash = btn.dataset.mode === 'trash';
        if (showingTrash) {
            // Switch back to conversations
            btn.dataset.mode = 'conversations';
            btn.title = 'Show Trash';
            btn.classList.remove('active');
            list.style.display = '';
            trash.style.display = 'none';
            if (title) title.textContent = 'Conversations';
        } else {
            btn.dataset.mode = 'trash';
            btn.title = 'Show Conversations';
            btn.classList.add('active');
            list.style.display = 'none';
            trash.style.display = '';
            if (title) title.textContent = 'Trash';
            await loadTrash();
        }
    });
}

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

    // A realtime/pipeline voice session persists its turns server-side as a
    // new conversation, but the sidebar never hears about it (the browser
    // talks straight to OpenAI). voice/ui.js fires this when a call ends with
    // real turns; reload the list so the new conversation shows up without a
    // manual page refresh.
    window.addEventListener('kestrel:conversations-stale', () => {
        if (typeof loadConversations === 'function') {
            loadConversations(selectedAgentName).catch(() => { /* best-effort */ });
        }
    });

    const newConversationSidebarBtn = document.getElementById('new-conversation-sidebar-btn');
    if (newConversationSidebarBtn) {
        newConversationSidebarBtn.addEventListener('click', async () => {
            try {
                // Delegate to the canonical start-new-conversation flow so we
                // update state.currentSessionId, clear the chat, refresh the
                // context-status footer, and reload the sidebar in one shot.
                // Duplicating partial logic here left the footer showing the
                // old conversation's message count and utilization.
                if (typeof window.startNewConversation === 'function') {
                    await window.startNewConversation();
                } else {
                    await API.newConversation();
                }
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
    // Wire the Trash toggle in the conversations pane header (#765).
    initTrashToggle();
});
