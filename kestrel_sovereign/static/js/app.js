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
import { Security } from './security.js';
import { initTasks, loadTasks } from './tasks.js';
import { loadResources } from './resources.js';
import { initMetrics, loadMetrics } from './metrics.js';
import { initSpawn, loadSpawn } from './spawn.js';
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
    });

    // Initialize navigation
    initNavigation();

    // Initialize chat component
    initChat();

    // Initialize tasks component
    initTasks();

    // Initialize metrics component
    initMetrics();

    // Initialize spawn component
    initSpawn();

    // Initialize security module
    Security.init();

    // Initialize sovereignty panel buttons (Export/Import)
    initSovereigntyButtons();

    // Load agents first — in rookery mode, selectAgent() handles loading
    // all agent-specific data (identity, privacy, models, SSE, context).
    await loadAgents();

    // In standalone mode (no rookery agent selected), load data directly
    if (!API.isRookeryMode()) {
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
