/**
 * Kestrel Sovereign Console - Application Entry Point
 * Main initialization and coordination
 */

import API from './api.js';
import { state, loadCommands } from './ui.js';
import {
    initAgentFromUrl,
    setLazyLoaders,
    initNavigation,
    loadIdentity,
    loadPrivacyMode,
    loadAgents,
    loadConstitution,
    loadMemories,
    loadExports,
    initSovereigntyButtons,
    loadLocalFiles,
    loadIpfsStatus,
} from './panels.js';
import { mount as mountChat, loadModels, connectNotifications, updateContextStatus, subscribeSSE } from './chat.js';
import { UI } from './ui-ext/registry.js';
// Voice UI is loaded via the manifest loader (#2043, ticket 05) — its core-bundled
// manifest entry points at js/voice/boot.js, which imports voice/ui.js; that module
// self-registers its slot contributions on import (#2038, ticket 04). No bare import
// or named init() call from core anymore.
import { Security } from './security.js';
import { initTasks, loadTasks } from './tasks.js';
import { loadResources } from './resources.js';
import { initMetrics, loadMetrics } from './metrics.js';
import { initSpawn, loadSpawn } from './spawn.js';
import { initFeatureStore, loadFeatureStore } from './feature-store.js';
import { initApprovals, loadApprovals } from './approvals.js';
// Import modules with side effects that define window.* functions
import './database.js';  // Defines window.toggleDbExplorer
import './ipfs.js';      // Defines window.toggleIpfsStatus

// ============================================================================
// Feature UI contributions (manifest-driven, out-of-tree asset loading; #2043)
// ============================================================================

// Inject a feature-contributed stylesheet once. Idempotent — the manifest may
// be (re)loaded and a stylesheet must not be appended twice.
function injectFeatureStylesheet(href) {
    if (!href) return;
    if (document.querySelector(`link[data-ui-ext-css="${CSS.escape(href)}"]`)) return;
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = href;
    link.dataset.uiExtCss = href;
    document.head.appendChild(link);
}

// Fetch the merged UI-contributions manifest and dynamically import each enabled
// feature's modules in declared order. The server enabled-filters and rejects
// cross-origin module URLs; we additionally honor the client capability set so a
// host force-off also suppresses loading. A failed import for one feature is
// isolated so it cannot abort the rest of boot.
async function loadFeatureUIContributions() {
    let contributions = [];
    try {
        const data = await API.request('/api/ui/contributions');
        if (data && Array.isArray(data.contributions)) {
            contributions = data.contributions;
        }
    } catch (e) {
        console.warn('[ui-ext] failed to fetch UI contributions manifest:', e);
        return;
    }

    for (const entry of contributions) {
        if (entry.capability && !API.hasCapability(entry.capability)) continue;
        for (const href of entry.css || []) {
            injectFeatureStylesheet(href);
        }
        for (const mod of entry.modules || []) {
            try {
                await import(mod);
            } catch (e) {
                console.error(
                    `[ui-ext] failed to import feature module ${mod} (feature ${entry.feature}):`,
                    e,
                );
            }
        }
    }
}

// ============================================================================
// Application Initialization
// ============================================================================

async function init() {
    console.log('Kestrel Sovereign Console initializing...');

    // Check for agent parameter in URL (multi-agent mode)
    initAgentFromUrl();

    // Initialize authentication first (fetches API key from server)
    await API.init();

    if (!API.hasCapability('chrome')) {
        document.body.classList.add('console-chrome-hidden');
    }

    // #2041: the capability set is derived from enabled features. The standalone
    // server injects window.KESTREL_UI_CONFIG.featureCapabilities at page render
    // so the merge is ready synchronously. In multi-agent host / embed mode where
    // the render could not resolve a single agent, fetch the map now — BEFORE
    // initNavigation prunes panels or any feature registration runs (boot order:
    // config+capabilities → registry/nav → render).
    const bootConfig = (typeof globalThis !== 'undefined' && globalThis.KESTREL_UI_CONFIG) || {};
    if (!bootConfig.featureCapabilities) {
        await API.refreshCapabilities();
    }

    // Set up lazy loaders for navigation (panels loaded on tab click)
    setLazyLoaders({
        loadConstitution,
        loadMemories,
        loadExports,
        loadTasks,
        loadResources,
        loadMetrics,
        loadSpawn,
        loadFeatureStore,
        loadApprovals,
    });

    // Initialize navigation
    initNavigation();

    // Initialize chat component by mounting it into the console's chat panel
    // — the same public entry point (#1597) that external embedders (Frinz)
    // use, so the console dogfoods the extracted component instead of the old
    // initChat() path. Falls back to a document root if the panel is absent.
    mountChat(document.getElementById('panel-chat'));

    // UI extension slots (#2038, ticket 02): mount the single shared overlay
    // root once at boot and render its zone (empty until a feature contributes a
    // modal). A dedicated `#modal-root` scopes modal teardown rather than
    // leaking dialogs onto document.body.
    let modalRoot = document.getElementById('modal-root');
    if (!modalRoot) {
        modalRoot = document.createElement('div');
        modalRoot.id = 'modal-root';
        document.body.appendChild(modalRoot);
    }
    UI.renderSlot('modal-root', { element: modalRoot, api: API });

    // Bridge the server-push SSE channel onto the UI extension bus (#2038):
    // backend `tools_updated` events become generic bus events so contributions
    // re-gate/re-render without subscribing to SSE directly. Voice consumes this
    // on the bus (ticket 04) instead of its own subscribeSSE.
    subscribeSSE('tools_updated', (evt) => {
        let payload = null;
        try {
            payload = evt && evt.data ? JSON.parse(evt.data) : null;
        } catch (_) { /* forward the bare event when the body isn't JSON */ }
        UI.emit('tools_updated', payload);
    });

    // Bridge runtime capability changes (#2041) onto the UI extension bus
    // (#2038). `API.applyServerCapabilities()` dispatches a DOM
    // `capabilities:changed` event on globalThis when a feature is enabled /
    // disabled at runtime; the slot registry only listens on the UI bus, so
    // without this bridge a capability flip would never re-gate slot
    // contributions (e.g. voice mounting/tearing-down its mic + controls).
    // Forward it so any contribution that opts into `capabilities:changed`
    // re-gates without a page reload.
    if (typeof globalThis !== 'undefined' && typeof globalThis.addEventListener === 'function') {
        globalThis.addEventListener('capabilities:changed', (evt) => {
            UI.emit('capabilities:changed', evt && evt.detail ? evt.detail : null);
        });
    }

    // UI extension slots (#2043, ticket 05): load out-of-tree feature frontend
    // assets from the manifest. Each enabled feature's ES modules are imported
    // in declared order; each module calls UI.register(...). This is how a
    // pip-installed feature mounts slot contributions with no edits to core
    // static/ or app.js — including features (voice) whose JS still lives in
    // core today but is loaded via the manifest like any other feature. Voice's
    // module self-registers its shell (🎙️ button, path badge, picker modal,
    // per-agent controls) into the slot zones on import.
    await loadFeatureUIContributions();

    // Initialize tasks component
    initTasks();

    // Initialize metrics component
    initMetrics();

    // Initialize spawn component
    initSpawn();

    // Initialize feature store component
    initFeatureStore();

    // Initialize approvals panel (epic #1290, D2)
    initApprovals();

    // Initialize sovereignty panel buttons (Export/Import)
    initSovereigntyButtons();

    // Load agents first — in multi_agent mode, selectAgent() handles loading
    // all agent-specific data (identity, privacy, models, SSE, context).
    await loadAgents();

    // Initialize security after agent selection so per-agent security routes
    // use /api/agents/{name}/... in multi_agent mode.
    Security.init();

    // In standalone mode (no multi_agent agent selected), load data directly.
    // #879: each loader self-guards against its capability being disabled
    // (chat → connectNotifications/loadModels/updateContextStatus/
    // loadCommands; identity, privacy → their own caps), so we fan them
    // out unconditionally and the no-ops keep this block stable as new
    // caps land.
    if (!API.isMultiAgentMode()) {
        connectNotifications();
        await Promise.all([
            loadIdentity(),
            loadPrivacyMode(),
            loadModels(),
            loadCommands(API),
        ]);
        updateContextStatus();
    }

    console.log('Kestrel Sovereign Console ready');
}

// Start the application when DOM is ready
document.addEventListener('DOMContentLoaded', init);
