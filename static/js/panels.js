/**
 * Kestrel Sovereign Console - Panel Components
 * Re-exports all panel functionality from decomposed modules
 *
 * Module structure:
 * - identity.js: Agent selection, Navigation, Identity panel, Privacy indicator, Sidebar
 * - memories.js: Constitution panel, Memories panel
 * - sovereignty.js: Sovereignty exports/imports, Local file browser, IPFS status
 * - explorers.js: Chat history browser, Database explorer
 */

// Identity Module - Agent selection, Navigation, Identity, Privacy, Sidebar
export {
    initAgentFromUrl,
    setLazyLoaders,
    initNavigation,
    loadIdentity,
    loadPrivacyMode,
    updatePrivacyIndicator,
    loadSidebar,
} from './identity.js';

// Memories Module - Constitution, Memories
export {
    loadConstitution,
    loadMemories,
    initMemoryFilter,
} from './memories.js';

// Sovereignty Module - Exports, Local files, IPFS
export {
    loadExports,
    initSovereigntyButtons,
    loadLocalFiles,
    loadIpfsStatus,
} from './sovereignty.js';

// Explorers Module - Chat history, Database
export {
    loadConversationHistory,
    loadDbTables,
} from './explorers.js';
