/**
 * Kestrel Sovereign Console - Application Entry Point
 * Main initialization and coordination
 */

import API from './api.js';
import { state } from './ui.js';
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
import { initChat, loadModels } from './chat.js';
import { Security } from './security.js';
import { initTasks, loadTasks } from './tasks.js';
import { loadResources } from './resources.js';
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
    });

    // Initialize navigation
    initNavigation();

    // Initialize chat component
    initChat();

    // Initialize tasks component
    initTasks();

    // Initialize security module
    Security.init();

    // Initialize sovereignty panel buttons (Export/Import)
    initSovereigntyButtons();

    // Load initial data in parallel
    await Promise.all([
        loadIdentity(),
        loadPrivacyMode(),
        loadAgents(),
        loadModels(),
    ]);

    console.log('Kestrel Sovereign Console ready');
}

// Start the application when DOM is ready
document.addEventListener('DOMContentLoaded', init);
