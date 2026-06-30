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
// Voice UI self-registers its slot contributions on import (#2038, ticket 04).
// This bare side-effect import stays until ticket 05's manifest loader imports
// voice as an out-of-tree module.
import './voice/ui.js';
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

    // The voice UI shell (🎙️ button, path badge, picker modal, per-agent
    // controls) mounts itself through the slot registry — it self-registered on
    // import above and renders when core renders each zone (chat-input-actions,
    // input-footer-status, modal-root, agent-card-actions).

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
