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
import { initChat, loadModels, connectNotifications, updateContextStatus } from './chat.js';
import { initVoiceUI } from './voice/ui.js';
import { Security } from './security.js';
import { initTasks, loadTasks } from './tasks.js';
import { loadResources } from './resources.js';
import { initMetrics, loadMetrics } from './metrics.js';
import { initSpawn, loadSpawn } from './spawn.js';
import { initFeatureStore, loadFeatureStore } from './feature-store.js';
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
    });

    // Initialize navigation
    initNavigation();

    // Initialize chat component
    initChat();

    // Initialize voice UI shell — adds the 🎙️ button, transcript drawer,
    // voice picker. Auto-fallback Realtime → Pipeline based on the
    // server-side voice path resolver.
    initVoiceUI();

    // Initialize tasks component
    initTasks();

    // Initialize metrics component
    initMetrics();

    // Initialize spawn component
    initSpawn();

    // Initialize feature store component
    initFeatureStore();

    // Initialize security module
    Security.init();

    // Initialize sovereignty panel buttons (Export/Import)
    initSovereigntyButtons();

    // Load agents first — in multi_agent mode, selectAgent() handles loading
    // all agent-specific data (identity, privacy, models, SSE, context).
    await loadAgents();

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
