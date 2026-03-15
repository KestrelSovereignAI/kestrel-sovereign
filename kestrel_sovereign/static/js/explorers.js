/**
 * Kestrel Sovereign Console - Explorers Module
 * Re-exports chat history and database explorer functionality
 *
 * Module structure:
 * - history.js: Chat History Browser
 * - database.js: Database Explorer
 */

// History Module - Chat history browser
export { loadConversationHistory } from './history.js';

// Database Module - Database explorer
export { loadDbTables } from './database.js';
