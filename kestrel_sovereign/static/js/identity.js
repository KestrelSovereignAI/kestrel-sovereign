/**
 * Kestrel Sovereign Console - Identity Module
 * Agent selection, navigation, identity panel, privacy indicator, sidebar
 */

import API from './api.js';
import { state, PRIVACY_MODES, Toast, Modal, loadCommands } from './ui.js';
import { renderIdentityDangerZone } from './identity-danger-zone.js';
import { disconnectNotifications, connectNotifications, loadModels, updateContextStatus, updateThinkingIndicator, mountChatPane, wipeAgentChatPane, refreshAgentThinkingDot, stopAgent, renderModelFooterHtml, appendMessagePart, splitContentByParts, renderSignalWakeChip } from './chat.js';
import { generateIdenticon } from './identicon.js';
// #2199: the standalone conversations pane is now a `mountConversations`
// consumer — one list orchestrator (fetch / refresh / seq-guard / views /
// trash) shared with the history slideout and any embedded mount. identity.js
// keeps NO bespoke conversation fetching or request-sequencing; it only wires
// the sidebar-specific hooks (agent pinning, #714 auto-load, chat-state
// coordination on delete) through the component's config.
import { mountConversationsPane } from './conversations.js';
// #2278: the standalone agents pane is now a `mountAgentList` consumer — one
// list orchestrator (adapter fetch / render / selection / per-card
// `agent-card-actions` slot / active highlight) shared with embedding hosts
// (Frinz's companion list). identity.js keeps NO bespoke agent loop; it wires
// the console-specific policy (demo banner, demo-misconfig-gated auto-select,
// standalone conversations-pane refresh) through the component's config hooks.
import { mountAgentList, createDefaultAgentAdapter } from './agent_list.js';
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
        // No ?agent= pin — this says nothing about whether the HOST is
        // single- or multi-agent (that's the sidebar/selectAgent flow); it
        // only means the URL didn't pre-select one.
        console.log('Kestrel Sovereign Console: no agent pinned in URL');
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

    // #2229: the standalone console is chat-first, exactly like every embed.
    // The tab strip starts hidden (index.html) and the chat-header "Advanced"
    // toggle reveals the capability-gated strip through the SAME reveal
    // implementation the embeddable mountPanels host uses (Panels.initReveal).
    // Collapsed = chat only; revealing shows the gated strip (Chat first);
    // collapsing returns to the Chat tab; the toggle hides when only one tab is
    // available and tracks runtime gating changes (nav MutationObserver). The
    // revealed state persists across reloads so an operator who lives in
    // Advanced isn't re-collapsed every load.
    const advancedToggle = document.getElementById('advanced-toggle-btn');
    if (navEl && advancedToggle) {
        Panels.initReveal({
            navEl,
            activate: activatePanel,
            anchor: advancedToggle,
            storageKey: 'kestrel:console-advanced',
        });
    } else if (navEl) {
        // A host that disables the `chat` capability had #panel-chat pruned —
        // and the Advanced toggle lives in the chat header, so it went with it.
        // A chat-less console is all-panels: show the strip permanently instead
        // of leaving every remaining panel unreachable behind a hidden nav with
        // no surviving reveal control (codex P2 on #2231).
        navEl.style.display = '';
    }

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

        // #2208: danger-zone section (delete agent + future destructive actions).
        // Rendered last so it sits at the bottom of the panel. The module decides
        // visibility itself — it blanks the container when no host handler and no
        // native capability apply, so re-rendering on every selectAgent is
        // self-clearing and safe.
        renderIdentityDangerZone({
            container: document.getElementById('identity-danger-zone'),
            identity,
            api: API,
            Modal,
            Toast,
        });
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
                let result = await API.setPrivacyMode(mode);

                // A data-destructive downgrade (e.g. PUBLIC→EPHEMERAL) is STAGED,
                // not applied: the server returns {requires_confirmation:true}
                // and the agent stays in its current mode. Do NOT flip the
                // indicator — that would show EPHEMERAL while the agent is still
                // PUBLIC and keeps persisting (split-state). Confirm first, then
                // apply the staged transition and reflect the REAL result.
                if (result && result.requires_confirmation) {
                    dropdown.remove();
                    const proceed = window.confirm(
                        (result.message
                            || `Switching to ${PRIVACY_MODES[mode]?.label || mode} will delete existing data.`)
                        + '\n\nApply this change now?'
                    );
                    if (!proceed) {
                        // Discard the server-side staged transition so a later
                        // confirm (another tab / !confirm-privacy-mode) can't
                        // apply the change the user just declined.
                        try { await API.cancelPrivacyMode(); } catch (_) { /* best-effort */ }
                        Toast.info(
                            `Privacy mode unchanged (still ${PRIVACY_MODES[state.privacyMode]?.label || state.privacyMode}).`
                        );
                        return;
                    }
                    result = await API.confirmPrivacyMode();
                    if (!result || !result.applied) {
                        Toast.error(
                            `Could not apply privacy mode: ${result?.message || 'unknown error'}`
                        );
                        return;
                    }
                    // Confirmed and applied — fall through to reflect it.
                }

                // Reflect the mode the SERVER actually applied, not the mode
                // this tab clicked: a staged transition applies whatever was
                // pending on the agent, which a concurrent tab could have
                // changed. result.mode is the single source of truth (both the
                // direct-apply and confirm responses carry it); fall back to the
                // clicked mode only if the server didn't return one.
                const appliedMode = (result && result.mode) || mode;
                const appliedLabel = PRIVACY_MODES[appliedMode]?.label || appliedMode;
                state.privacyMode = appliedMode;
                updatePrivacyIndicator(appliedMode);
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
                    Toast.success(`Privacy: ${appliedLabel} — switched to ${result.model_switched.vendor} (local only)`);
                } else if (result.allows_cloud_llm !== false) {
                    // Switching back to cloud-allowing mode — restore saved cloud selection
                    const savedProvider = localStorage.getItem('kestrel_saved_cloud_provider');
                    const savedModel = localStorage.getItem('kestrel_saved_cloud_model');
                    if (savedProvider && savedModel && window._sharedModelSelector) {
                        window._sharedModelSelector.setSelection(savedProvider, savedModel);
                        Toast.success(`Privacy: ${appliedLabel} — restored ${savedProvider} model`);
                        localStorage.removeItem('kestrel_saved_cloud_provider');
                        localStorage.removeItem('kestrel_saved_cloud_model');
                    } else {
                        Toast.success(`Privacy mode set to ${appliedLabel}`);
                    }
                } else {
                    Toast.success(`Privacy mode set to ${appliedLabel}`);
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
// #2278: the mounted `mountAgentList` handle + its default `/api/agents`
// adapter. Mounted once (into `#agents-list`) on the first `loadAgents`; later
// calls just `refresh()`. The adapter is retained so `onLoaded` can read the
// response `mode` / `server_demo_mode` back for the demo banner + misconfig gate.
let agentListHandle = null;
let agentListDefaultAdapter = null;

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

// Console-specific policy fired after each agent-list load (#2278). The shared
// component owns fetch/render/selection/slot; this keeps the demo banner, the
// demo-misconfig auto-select gate, and the standalone conversations-pane refresh
// host-side — reading the response `mode` / `server_demo_mode` back off the
// retained default adapter.
function handleAgentsLoaded(items) {
    const isStandalone = agentListDefaultAdapter
        && agentListDefaultAdapter.mode === 'standalone';
    const serverDemoMode = !!(agentListDefaultAdapter && agentListDefaultAdapter.serverDemoMode);
    const rawAgents = items.map((i) => i.raw).filter(Boolean);

    // DEMO MODE banner (#868) — visible warning so an operator notices when a
    // demo server has somehow mounted a live agent.
    renderDemoModeBanner({
        serverDemoMode,
        agents: rawAgents,
        isStandalone,
    });

    // Demo-server misconfig (#868): a server in demo_mode that mounted any
    // non-demo agent is the routing precondition that wiped Meridian. The page
    // MUST refuse to install a host-agent prefix, otherwise every subsequent
    // click still routes to whichever agent auto-select would have picked.
    const hasLiveAgent = serverDemoMode && rawAgents.some((a) => a.is_demo !== true);

    // Auto-select first online agent only in multi_agent mode (not standalone)
    // and never when the demo server is in misconfig — the banner explicitly
    // tells the operator the auto-select is disabled. Driven through the
    // component's selection path (setHostAgent → onSelect → window.selectAgent).
    if (!selectedAgentName && !isStandalone && !hasLiveAgent) {
        const firstOnline = items.find((i) => i.status !== 'offline');
        if (firstOnline && agentListHandle) {
            agentListHandle.select(firstOnline.name);
        }
    }

    // Standalone mode: mount + target the conversations pane so its list
    // populates against the un-prefixed routes (selectAgent installs a
    // host-agent URL prefix that only exists in multi_agent routing — using it
    // in standalone 404s /api/conversations and /agent/invoke). #2216: this does
    // NOT reveal the pane — visibility is the component's persisted collapse
    // state (default hidden). Skip in misconfig — the list is not safe to target.
    if (isStandalone && items.length > 0 && !hasLiveAgent && API.hasCapability('conversations')) {
        try { refreshConversationsPane(); } catch (_) { /* best-effort */ }
    }
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
    const container = document.getElementById('agents-list');
    if (!container) return;

    // Mount the shared list surface once; later loadAgents() calls just refresh.
    // The default `/api/agents` adapter + default console-row renderer make the
    // standalone console behavior-identical to the pre-#2278 hand-rolled loop.
    if (!agentListHandle) {
        agentListDefaultAdapter = createDefaultAgentAdapter(API);
        agentListHandle = mountAgentList(container, {
            api: API,
            adapter: agentListDefaultAdapter,
            selectedName: selectedAgentName,
            // Per-agent thinking pulse is driven by `state.waitingAgents` (the
            // Set sendMessage adds to on dispatch); the stop control aborts that
            // exact agent's stream via stopAgent().
            isThinking: (name) => state.waitingAgents.has(name),
            onStop: (name) => stopAgent(name),
            // Selecting a card runs the full product wiring (capability refresh,
            // chat mount, agent:switch bus emit); the component already pinned
            // routing via setHostAgent before this fires.
            onSelect: (item) => window.selectAgent(item.name),
            onLoaded: (items) => handleAgentsLoaded(items),
            // Auto-select is host policy here (demo-misconfig gated); drive it
            // from onLoaded rather than the component's blanket autoSelectFirst.
            autoLoad: false,
        });
    }
    // Await the load so routing is pinned (setHostAgent runs synchronously
    // inside the onLoaded auto-select) before app.js runs Security.init().
    await agentListHandle.refresh();
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

    // Update selection UI — the agent-list component owns the active highlight
    // (#2278). setActiveName repaints the highlight only (no re-fire), so a
    // direct selectAgent call (or the auto-select path) reconciles the row state
    // without a selection round-trip.
    if (agentListHandle) agentListHandle.setActiveName(agentName);

    // Update chat header with agent name
    const navName = document.getElementById('nav-agent-name');
    if (navName) navName.textContent = agentName;

    // #2216: do NOT auto-reveal the conversations pane on agent select — the
    // component's persisted collapse state is the only thing that decides its
    // visibility. refreshConversationsPane() below mounts + retargets it without
    // forcing it open.

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
    // #2199: retarget the mounted conversation list to the new agent. The
    // component's `retarget` (its refreshSeq-guarded reload) is the ONLY
    // list-refresh path on an agent switch — preserving #1358 pinning without a
    // second request-sequence guard in identity.js.
    refreshConversationsPane();
    await Promise.all([
        loadIdentity(),
        loadPrivacyMode(),
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
const activeConversationIdsByAgent = new Map();

// #2199: the ONE mount handle for the standalone conversations pane. All list
// fetch / refresh / request-sequencing / view filtering lives inside the shared
// component (conversations.js) — identity.js holds only this handle and the
// sidebar-specific hooks below.
let conversationsHandle = null;
// Whether the mounted list has been pointed at an agent at least once. The
// mount uses autoLoad:false, so until refreshConversationsPane() retargets it
// the list has never fetched — hosts without the multi_agent agent-select
// flow (embeds like Frinz) reach the pane ONLY through the history trigger,
// which must therefore trigger the first load itself (codex round-2 P2).
let conversationsPaneTargeted = false;

function currentAgentMatches(expectedAgent) {
    return expectedAgent === API.getHostAgent();
}

function hasExpectedAgent(options) {
    return Object.prototype.hasOwnProperty.call(options || {}, 'expectedAgent');
}

function getActiveConversationIdForAgent(agentName) {
    const pane = state.chatPanes.get(agentName);
    return pane?.sessionId || activeConversationIdsByAgent.get(agentName) || null;
}

// #714 auto-load-most-recent, moved verbatim off the old bespoke loader. Fired
// from the component's `onLoaded` hook for the active view only, pinned to the
// agent the list was fetched for.
function maybeAutoLoadMostRecent(conversations, autoTargetAgent, view) {
    // Only the default (active) list drives auto-load; archived/trash views must
    // never yank the chat pane onto a tidied-away or deleted session.
    if (view && view !== 'active') return;
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
    // user-side as "the agent keeps stopping mid-answer." The {auto: true}
    // flag makes window.loadConversation re-check pane coldness post-await.
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
            // ``expectedAgent`` is pinned to the agent the LIST was fetched for —
            // loadConversation drops the load if the host has since switched
            // (#1358).
            window.loadConversation(mostRecent.session_id, {
                auto: true,
                expectedAgent: autoTargetAgent,
            });
        }
    }
}

// Chat-state coordination for a delete of the CURRENTLY-OPEN conversation.
// Fired from the component's `onMutated` hook so the pane-aware behavior (start
// a fresh session on trash, blank the pane on purge) is single-sourced here
// while the list fetch/render stays in the component.
async function handleSidebarConversationMutation(action, conv) {
    if (action !== 'trash' && action !== 'purge') return;
    if (state.currentSessionId !== conv.session_id) return;
    const host = API.getHostAgent();
    activeConversationId = null;
    activeConversationIdsByAgent.delete(host);
    if (action === 'trash') {
        // Immediately create a fresh backend session so downstream state
        // (context-status footer, auto-load logic) doesn't point at a
        // vanished session.
        const fresh = await API.newConversation();
        wipeAgentChatPane(host, `
            <div style="text-align: center; padding: 2rem; color: var(--text-secondary);">
                <span style="font-size: 2rem;">\u{2728}</span>
                <p style="margin-top: 0.5rem;">New conversation started. Say hello!</p>
            </div>
        `);
        state.currentSessionId = fresh.session_id;
        activeConversationId = fresh.session_id;
        activeConversationIdsByAgent.set(host, fresh.session_id);
    } else {
        wipeAgentChatPane(host);
        state.currentSessionId = null;
    }
    // #2222: keep the shared pane's highlight override in step with the new
    // active-id (the trashed session's tile is gone; a fresh session, if any,
    // becomes current).
    if (conversationsHandle) conversationsHandle.setActiveSessionId(activeConversationId);
    if (typeof updateContextStatus === 'function') updateContextStatus();
}

// #2222: host-side new-conversation wiring, invoked by the component's New
// button (via the pane's `onNewConversation` config). Mints the canonical
// session, wipes the visible agent's chat pane, adopts the new session as the
// current one (both `state.currentSessionId` and our per-agent active-id map so
// the sidebar highlight and the chat pane agree), and refreshes the context
// footer. The component handles the optimistic tile + active highlight from the
// returned session_id — this owns ONLY the chat-side state so there is exactly
// one `API.newConversation()` call.
async function startNewConversationForPane() {
    const host = API.getHostAgent();
    const result = await API.newConversation();
    wipeAgentChatPane(host, `
        <div style="text-align: center; padding: 2rem; color: var(--text-secondary);">
            <span style="font-size: 2rem;">\u{2728}</span>
            <p style="margin-top: 0.5rem;">New conversation started. Say hello!</p>
        </div>
    `);
    const sid = result && result.session_id;
    if (sid) {
        state.currentSessionId = sid;
        activeConversationId = sid;
        activeConversationIdsByAgent.set(host, sid);
        // #2248: anchor the host agent's chat pane to the freshly-minted
        // session EXPLICITLY. wipeAgentChatPane() above just nulled
        // pane.sessionId, and the send path reads pane.sessionId directly
        // (chat.js). Without this the first turn goes up with session_id=null
        // and only lands in the new session via the implicit last-message
        // derive race — any interleaved conversation row would misfile it. Set
        // it so the first turn is unambiguously anchored to the minted session.
        const pane = state.chatPanes.get(host);
        if (pane) pane.sessionId = sid;
    }
    if (typeof updateContextStatus === 'function') updateContextStatus();
    return result;
}

// Mount the shared conversation PANE unit into the sidebar exactly once, then
// reuse the handle. #2199: the standalone console consumes the SAME
// `mountConversationsPane` export as any embedder — one pane implementation
// owns the list plus the collapse rail, drag-resize (min/max + localStorage),
// and the search/view-bar/stats disclosure. identity.js provides only the
// container (`#conversations-pane`, whose static chrome the export adopts) and
// the sidebar-specific hooks: agent pinning, #714 auto-load, and chat-state
// coordination on delete. The header `#trash-toggle-btn` still drives the
// Active↔Trash view (the component's own view bar stays hidden), and the built
// search + stats blocks ride the pane's disclosure toggle.
function ensureConversationsMount() {
    if (conversationsHandle) return conversationsHandle;
    const container = document.getElementById('conversations-pane');
    if (!container) return null;
    conversationsHandle = mountConversationsPane(container, {
        api: API,
        storageKey: 'kestrel:conversations-pane',
        autoLoad: false,
        showViewBar: false,
        showSearch: true,
        showStats: true,
        agentName: API.getHostAgent(),
        getActiveSessionId: () => activeConversationId,
        // #2222: the component owns the New button now; it calls this to mint
        // the session, and does the optimistic tile + active highlight itself.
        // We do the host-side chat wiring (wipe the pane, adopt the new session
        // as current, refresh the context footer) and update our per-agent
        // active-id map so the highlight stays unified across the sidebar and
        // the chat pane. Returning the API result lets the component prepend a
        // tile for the exact minted session_id.
        onNewConversation: () => startNewConversationForPane(),
        onSelect: (conv, ctx) => {
            window.loadConversation(conv.session_id, { expectedAgent: ctx.agentName });
        },
        onLoaded: (conversations, ctx) => {
            maybeAutoLoadMostRecent(conversations, ctx.agentName, ctx.view);
        },
        onMutated: (mutAction, conv) => {
            handleSidebarConversationMutation(mutAction, conv);
        },
    });
    return conversationsHandle;
}

// Repoint the mounted list at the current host agent (or first-load it). The
// component's `retarget` (refreshSeq-guarded reload) is the ONLY list-refresh
// path on an agent switch — no second request-sequence guard here (#1358 /
// #2199). #879: hosts that opt out of conversations get the pane hidden.
export function refreshConversationsPane() {
    // #2199 P2-2: the chat-header history trigger drives this same pane, so its
    // visibility must track the same capability — otherwise a `conversations:
    // false` host still shows a button that can reveal the disabled pane.
    const trigger = document.getElementById('conversations-toggle-btn');
    if (!API.hasCapability('conversations')) {
        const pane = document.getElementById('conversations-pane');
        if (pane) pane.style.display = 'none';
        if (trigger) trigger.style.display = 'none';
        return;
    }
    if (trigger) trigger.style.display = '';
    const handle = ensureConversationsMount();
    if (!handle) return;
    activeConversationId = getActiveConversationIdForAgent(API.getHostAgent());
    // #2222: seed the component's highlight override for this agent so a
    // just-created / previously-active session stays highlighted across the
    // retarget repaint (and clears when switching to an agent with none).
    handle.setActiveSessionId(activeConversationId);
    conversationsPaneTargeted = true;
    handle.retarget(API.getHostAgent());
}

// #2222: bridge for history.js's ``window.startNewConversation`` (the chat-header
// "New Chat" button path). When the shared conversations pane is available for
// this host, route the new-conversation action through the component so the New
// tile appears instantly, becomes the current conversation, and the host-side
// chat wiring runs exactly once (via the pane's ``onNewConversation``). Returns
// the component's promise, or ``null`` when the pane is unavailable (no
// container / capability off) so the caller can fall back to the direct flow.
window.newConversationViaPane = function() {
    if (!API.hasCapability('conversations')) return null;
    const handle = ensureConversationsMount();
    if (!handle) return null;
    conversationsPaneTargeted = true;
    return handle.newConversation();
};

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

    // #2222: keep the shared pane's highlight unified with our active-id. The
    // component owns the highlight now (setActiveSessionId re-renders with the
    // active row marked); the manual class toggle below stays as a cheap
    // fallback for the pre-mount / no-handle case.
    if (conversationsHandle) conversationsHandle.setActiveSessionId(sessionId);

    // Update selection UI. The mounted list re-renders per agent (retarget is
    // the only refresh path on a switch, #2199), so every visible row already
    // belongs to the current host — a session_id match is sufficient.
    document.querySelectorAll('.conversation-item').forEach(item => {
        item.classList.toggle('active', item.dataset.sessionId === sessionId);
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

        // Auto-load defense-in-depth: the maybeAutoLoadMostRecent() caller
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

            // A persisted COGNITION signal wake collapses to a compact
            // "Autonomous wake" chip rather than surfacing the raw internal
            // instruction template as a user message. Shared with history.js
            // so both conversation loaders render it identically.
            if (msg.role === 'user' && msg.metadata && msg.metadata.signal_wake) {
                renderSignalWakeChip(msg, chatContainer);
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

// ============================================================================
// Trash sub-view (#765) — now the mounted component's Trash view (#2199)
// ============================================================================
//
// The Trash button in the conversations pane header no longer swaps two static
// containers; it drives the mounted conversation list's `setView` between the
// Active and Trash views. Restore / Delete Permanently live in each trash row's
// kebab menu (owned by the shared component), so the bespoke trash renderer and
// the pane-aware window.* delete/purge/restore handlers are gone.

export function initTrashToggle() {
    const btn = document.getElementById('trash-toggle-btn');
    const title = document.getElementById('conversations-pane-title');
    if (!btn) return;

    btn.addEventListener('click', () => {
        const handle = ensureConversationsMount();
        if (!handle) return;
        const showingTrash = btn.dataset.mode === 'trash';
        if (showingTrash) {
            // Switch back to the live conversations list.
            btn.dataset.mode = 'conversations';
            btn.title = 'Show Trash';
            btn.classList.remove('active');
            if (title) title.textContent = 'History';
            handle.setView('active');
        } else {
            btn.dataset.mode = 'trash';
            btn.title = 'Show Conversations';
            btn.classList.add('active');
            if (title) title.textContent = 'Trash';
            handle.setView('trash');
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

// #2199: the chat-header `ki-history` trigger (icon-only, tooltip + aria) opens
// or collapses the conversations pane. It drives the SAME mount handle the
// sidebar uses — ensureConversationsMount() so the toggle works even before the
// first agent-select has revealed the pane.
window.toggleConversationsPane = function() {
    // #2199 P2-2: the history trigger drives the same conversations pane that
    // refreshConversationsPane() gates on the `conversations` capability. A host
    // with `conversations: false` gets that pane hidden — so the toggle must be
    // a no-op there too, otherwise it would reveal and drive a disabled pane.
    if (!API.hasCapability('conversations')) return;
    const handle = ensureConversationsMount();
    if (!handle) return;
    // #2216: the component's open/close/toggle is the WHOLE story — no more
    // two-layer "reveal from display:none THEN toggle the rail" dance. close()
    // fully hides the pane; toggle() from that hidden state reopens it.
    //
    // Hosts without the multi_agent agent-select flow (embeds) never hit
    // refreshConversationsPane() before this trigger — the autoLoad:false mount
    // would open EMPTY and never fetch. First use targets the host agent (the
    // #2199 codex round-2 first-load); retarget is refreshSeq-guarded, and hosts
    // already targeted by agent-select skip this entirely.
    if (!conversationsPaneTargeted) refreshConversationsPane();
    handle.toggle();
};

// Setup collapse buttons and resize handles
document.addEventListener('DOMContentLoaded', () => {
    const collapseAgentsBtn = document.getElementById('collapse-agents-btn');
    if (collapseAgentsBtn) {
        collapseAgentsBtn.addEventListener('click', () => togglePane('agents-pane'));
    }

    // #2199: the conversations pane's collapse chevron + drag-resize handle are
    // owned by the `mountConversationsPane` export (it adopts the static
    // `.collapse-btn` / `.resize-handle` inside `#conversations-pane` when it
    // mounts), so identity.js no longer wires them here — that would double-bind.

    // Wire the chat-header history trigger (#2199).
    const historyTriggerBtn = document.getElementById('conversations-toggle-btn');
    if (historyTriggerBtn) {
        historyTriggerBtn.addEventListener('click', () => window.toggleConversationsPane());
    }

    // A realtime/pipeline voice session persists its turns server-side as a
    // new conversation, but the sidebar never hears about it (the browser
    // talks straight to OpenAI). voice/ui.js fires this when a call ends with
    // real turns; reload the list so the new conversation shows up without a
    // manual page refresh.
    window.addEventListener('kestrel:conversations-stale', (event) => {
        // #2254: an ORGANIC session (the user just types) learns its effective
        // session id from the X-Session-Id header onto `pane.sessionId`, but
        // never told identity.js — so `activeConversationId` stayed null and no
        // row highlighted, even after the #2250 turn-end refresh repainted. The
        // turn-end event now carries `{ sessionId, agent }`; adopt it into the
        // per-agent active-id map (and the live highlight when it's the current
        // host agent) BEFORE the refresh repaints, so the matching row survives:
        // organic first message, pane opened later, and companion/agent switch.
        const detail = event && event.detail;
        if (detail && detail.sessionId && detail.agent) {
            activeConversationIdsByAgent.set(detail.agent, detail.sessionId);
            if (currentAgentMatches(detail.agent)) {
                activeConversationId = detail.sessionId;
                if (conversationsHandle) {
                    conversationsHandle.setActiveSessionId(activeConversationId);
                }
            }
        }
        if (conversationsHandle) conversationsHandle.refresh();
    });

    // #2222: the New-conversation button is now component-owned. The pane mount
    // (`mountConversationsPane`) adopts the static `#new-conversation-sidebar-btn`
    // (or builds a `ki-plus` button for bare embed containers) and wires it to
    // the component's own new-conversation action — optimistic tile + active
    // highlight, with the host-side chat wiring supplied via the
    // `onNewConversation` config. identity.js owns no new-conversation wiring.

    // Initialize resize handles. The conversations pane's resize is owned by
    // the mountConversationsPane export (#2199); only the agents pane keeps the
    // bespoke handler here.
    initPaneResize('resize-agents', 'agents-pane');
    // Wire the Trash toggle in the conversations pane header (#765).
    initTrashToggle();
});
