/**
 * Kestrel Sovereign Console - API Client
 * Handles all HTTP communication with the backend
 */

// Agent ID for multi-agent mode (set by identity.js via setAgentId)
let currentAgentId = null;

/**
 * Get the API base path for the current agent.
 * In standalone mode, returns empty string.
 * In multi-agent mode, returns /api/kestrel/companions/{id}
 */
function getAgentApiBase() {
    if (currentAgentId) {
        return `/api/kestrel/companions/${encodeURIComponent(currentAgentId)}`;
    }
    return '';
}

/**
 * Rewrites standalone Kestrel endpoints to companion-specific endpoints when in multi-agent mode.
 * When accessed with ?agent={companion_id}, routes go through /api/kestrel/companions/{id}/...
 */
function rewriteEndpoint(endpoint) {
    const agentBase = getAgentApiBase();
    if (!agentBase) return endpoint;  // Standalone mode - no rewrite

    // Map standalone endpoints to companion endpoints
    const mappings = {
        // Identity & Constitution
        '/api/identity': '/identity',
        '/api/constitution': '/constitution',
        // Privacy
        '/agent/privacy-mode': '/privacy',
        // Memory & Knowledge
        '/api/memories': '/memory',
        // Sovereignty
        '/api/sovereignty/exports': '/sovereignty/exports',
        '/api/sovereignty/export': '/sovereignty/export',
        '/api/sovereignty/import': '/sovereignty/import',
        // Chat & Invocation
        '/agent/invoke': '/invoke',
        '/agent/stream': '/stream',
        // Models & Commands
        '/api/models': '/models',
        '/api/model/current': '/model/current',
        '/api/commands': '/commands',
        // Storage & Wallet
        '/api/storage/stats': '/storage/stats',
        '/api/wallet': '/wallet',
        // Keys
        '/api/keys': '/keys',
        // Conversations
        '/api/conversations': '/conversations',
        // Tasks
        '/api/tasks': '/tasks',
        // Security
        '/api/security/pending': '/security/pending',
        '/api/security/approve': '/security/approve',
        '/api/security/permissions': '/security/permissions',
        // Database Explorer
        '/api/db/tables': '/db/tables',
        // IPFS Status
        '/api/ipfs/status': '/ipfs/status',
    };

    // Check for exact matches first
    if (mappings[endpoint]) {
        return `${agentBase}${mappings[endpoint]}`;
    }

    // Check for prefix matches (for endpoints with query params or path segments)
    for (const [pattern, suffix] of Object.entries(mappings)) {
        if (endpoint.startsWith(pattern)) {
            return `${agentBase}${suffix}${endpoint.slice(pattern.length)}`;
        }
    }

    // No rewrite - return as-is (e.g., /health stays /health)
    return endpoint;
}

const API = {
    // API key for authenticated requests (standalone mode)
    _apiKey: null,
    // JWT token for authenticated requests (multi-agent mode)
    _jwtToken: null,

    /**
     * Initialize authentication.
     * - In multi-agent mode: Uses JWT token from localStorage
     * - In standalone mode: Fetches API key from /api/auth/key (localhost-only)
     */
    async init() {
        // Check if we're in multi-agent mode (launched with ?agent=...)
        if (currentAgentId) {
            // Use JWT from localStorage/sessionStorage
            this._jwtToken = localStorage.getItem('platform_auth_token')
                          || localStorage.getItem('token')
                          || sessionStorage.getItem('token');
            if (this._jwtToken) {
                console.log('Kestrel UI: Using JWT token from session');
                return;
            }
            console.warn('Kestrel UI: Multi-agent mode but no JWT token found');
        }

        // Standalone mode: Try to get API key from sessionStorage first
        this._apiKey = sessionStorage.getItem('kestrel_api_key');
        if (this._apiKey) {
            console.log('Using cached API key from sessionStorage');
            return;
        }

        // Fetch API key from server (localhost only)
        try {
            const resp = await fetch('/api/auth/key');
            if (resp.ok) {
                const data = await resp.json();
                this._apiKey = data.key;
                sessionStorage.setItem('kestrel_api_key', this._apiKey);
                console.log('API key retrieved and cached');
            } else if (resp.status === 403) {
                console.warn('API key endpoint not accessible (non-local request)');
            } else {
                console.error('Failed to get API key:', resp.status);
            }
        } catch (e) {
            console.error('Failed to initialize authentication:', e);
        }
    },

    async request(endpoint, options = {}, _retried = false) {
        // Rewrite endpoint for multi-agent mode
        const url = rewriteEndpoint(endpoint);

        const headers = {
            'Content-Type': 'application/json',
            ...options.headers
        };

        // Add appropriate auth header based on mode
        if (this._jwtToken) {
            // Multi-agent mode: use JWT Bearer token
            headers['Authorization'] = `Bearer ${this._jwtToken}`;
        } else if (this._apiKey) {
            // Standalone mode: use API key
            headers['X-API-Key'] = this._apiKey;
        }

        const response = await fetch(url, {
            headers,
            ...options,
        });

        // Handle 401 Unauthorized - auto-refresh key and retry once
        if (response.status === 401 && !_retried) {
            console.warn('Authentication failed - refreshing API key...');
            sessionStorage.removeItem('kestrel_api_key');
            this._apiKey = null;

            // Try to get a fresh key
            try {
                const keyResp = await fetch('/api/auth/key');
                if (keyResp.ok) {
                    const data = await keyResp.json();
                    this._apiKey = data.key;
                    sessionStorage.setItem('kestrel_api_key', this._apiKey);
                    console.log('API key refreshed - retrying request');
                    // Retry the original request with new key
                    return this.request(endpoint, options, true);
                }
            } catch (e) {
                console.error('Failed to refresh API key:', e);
            }

            const error = await response.json().catch(() => ({ detail: 'Authentication failed' }));
            throw new Error(error.detail || 'Authentication failed - please refresh the page');
        }

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        return response.json();
    },

    // Core endpoints
    health: () => API.request('/health'),
    getAgentInfo: () => API.request('/agent/info'),
    getIdentity: () => API.request('/api/identity'),
    getConstitution: () => API.request('/api/constitution'),
    getPrivacyMode: () => API.request('/agent/privacy-mode'),
    setPrivacyMode: (mode) => API.request('/agent/privacy-mode', { method: 'POST', body: JSON.stringify({ mode }) }),
    getSessions: (limit = 50) => API.request(`/api/sessions?limit=${limit}`),
    getMemories: (nodeType = null, limit = 100) => {
        let url = `/api/memories?limit=${limit}`;
        if (nodeType) url += `&node_type=${nodeType}`;
        return API.request(url);
    },
    deleteMemory: (nodeId) => API.request(`/api/memories/${encodeURIComponent(nodeId)}`, { method: 'DELETE' }),
    getMemoryDetail: (nodeId) => API.request(`/api/memories/${encodeURIComponent(nodeId)}`),
    getIdentityChain: () => API.request('/api/identity-chain'),
    getStorageStats: () => API.request('/api/storage/stats'),
    getSovereigntyExports: () => API.request('/api/sovereignty/exports'),
    exportSovereignty: (tier, encrypt) => API.request('/api/sovereignty/export', { method: 'POST', body: JSON.stringify({ tier, encrypt }) }),
    importSovereignty: (cid) => API.request('/api/sovereignty/import', { method: 'POST', body: JSON.stringify({ cid }) }),
    // File Browser
    getSovereigntyFiles: () => API.request('/api/sovereignty/files'),
    getSovereigntyFilePreview: (filename) => API.request(`/api/sovereignty/files/${encodeURIComponent(filename)}/preview`),
    // Conversation History
    getConversations: (decrypt = true) => API.request(`/api/conversations?decrypt=${decrypt}`),
    getConversation: (sessionId, decrypt = true) => API.request(`/api/conversations/${encodeURIComponent(sessionId)}?decrypt=${decrypt}`),
    newConversation: () => API.request('/api/conversations/new', { method: 'POST' }),
    // Database Explorer & IPFS Status
    getDbTables: () => API.request('/api/db/tables'),
    queryDbTable: (table, limit = 50, offset = 0, search = null) => {
        let url = `/api/db/tables/${encodeURIComponent(table)}?limit=${limit}&offset=${offset}`;
        if (search) url += `&search=${encodeURIComponent(search)}`;
        return API.request(url);
    },
    getIpfsStatus: () => API.request('/api/ipfs/status'),
    getWallet: () => API.request('/api/wallet'),
    invoke: (input, model = null, sessionId = null) => API.request('/agent/invoke', {
        method: 'POST',
        body: JSON.stringify({ input, model, session_id: sessionId })
    }),
    /**
     * Get available LLM models
     * @param {Object} options - Query options
     * @param {boolean} options.featuredOnly - Only return featured models (default: true)
     * @param {string} options.category - Filter by category (chat, embedding, image, audio)
     * @param {string[]} options.providers - Filter by provider names
     * @param {boolean} options.useCache - Use cached results (default: true)
     */
    getModels: (options = {}) => {
        const params = new URLSearchParams();
        if (options.featuredOnly !== undefined) params.append('featured_only', options.featuredOnly);
        if (options.category) params.append('category', options.category);
        if (options.providers) params.append('providers', options.providers.join(','));
        if (options.useCache !== undefined) params.append('use_cache', options.useCache);
        const queryString = params.toString();
        return API.request(`/api/models${queryString ? '?' + queryString : ''}`);
    },
    streamInvoke: async function*(input, model = null, sessionId = null) {
        // Rewrite endpoint for multi-agent mode
        const url = rewriteEndpoint('/agent/stream');

        const headers = { 'Content-Type': 'application/json' };

        // Add appropriate auth header based on mode
        if (API._jwtToken) {
            headers['Authorization'] = `Bearer ${API._jwtToken}`;
        } else if (API._apiKey) {
            headers['X-API-Key'] = API._apiKey;
        }

        const response = await fetch(url, {
            method: 'POST',
            headers,
            body: JSON.stringify({ input, model, session_id: sessionId })
        });

        // Handle 401 Unauthorized
        if (response.status === 401) {
            console.warn('Streaming auth failed - clearing cached key');
            sessionStorage.removeItem('kestrel_api_key');
            API._apiKey = null;
            throw new Error('Authentication failed - please refresh the page');
        }

        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            yield decoder.decode(value, { stream: true });
        }
    },

    /**
     * Get the current API key (for use with EventSource which can't send headers).
     * @returns {string|null} The API key or null if not authenticated
     */
    getApiKey() {
        return this._apiKey;
    },

    /**
     * Set the agent ID for multi-agent mode.
     * Called by identity.js after parsing the URL parameter.
     * @param {string|null} agentId - The companion/agent ID or null for standalone mode
     */
    setAgentId(agentId) {
        currentAgentId = agentId;
    },

    /**
     * Get the current agent ID.
     * @returns {string|null} The agent ID or null if in standalone mode
     */
    getAgentId() {
        return currentAgentId;
    },

    /**
     * Check if we're in multi-agent mode (launched with ?agent=...)
     * @returns {boolean} True if in multi-agent mode
     */
    isMultiAgentMode() {
        return currentAgentId !== null;
    },

    /**
     * Get context window status including token usage and budget.
     * @param {string|null} sessionId - Optional session ID to get context for
     * @returns {Promise<Object>} Context status with message_count, utilization_percent, etc.
     */
    getContextStatus: (sessionId = null) => {
        const params = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
        return API.request(`/agent/context-status${params}`);
    },
};

export default API;
