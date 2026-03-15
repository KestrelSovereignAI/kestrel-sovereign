/**
 * Kestrel API client for Sovereign Console
 */

const BASE_URL = window.KESTREL_API_BASE || '';

/**
 * Make an API request with error handling
 * @param {string} endpoint - API endpoint
 * @param {object} options - Fetch options
 * @returns {Promise<object>} Response data
 */
async function request(endpoint, options = {}) {
    const url = `${BASE_URL}${endpoint}`;
    const response = await fetch(url, {
        headers: {
            'Content-Type': 'application/json',
            ...options.headers,
        },
        ...options,
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return response.json();
}

/**
 * Kestrel API client
 */
export const KestrelAPI = {
    // Health & Info
    async health() {
        return request('/health');
    },

    async getAgentInfo() {
        return request('/agent/info');
    },

    // Identity
    async getIdentity() {
        return request('/api/identity');
    },

    // Constitution
    async getConstitution() {
        return request('/api/constitution');
    },

    // Privacy
    async getPrivacyMode() {
        return request('/agent/privacy-mode');
    },

    async setPrivacyMode(mode) {
        return request('/agent/privacy-mode', {
            method: 'POST',
            body: JSON.stringify({ mode }),
        });
    },

    // Sessions
    async getSessions(limit = 50) {
        return request(`/api/sessions?limit=${limit}`);
    },

    // Memories
    async getMemories(nodeType = null, limit = 100) {
        let url = `/api/memories?limit=${limit}`;
        if (nodeType) url += `&node_type=${nodeType}`;
        return request(url);
    },

    async deleteMemory(nodeId) {
        return request(`/api/memories/${encodeURIComponent(nodeId)}`, {
            method: 'DELETE',
        });
    },

    // Storage
    async getStorageStats() {
        return request('/api/storage/stats');
    },

    // Sovereignty
    async getSovereigntyExports() {
        return request('/api/sovereignty/exports');
    },

    async exportSovereignty(tier = 'ipfs', encrypt = true) {
        return request('/api/sovereignty/export', {
            method: 'POST',
            body: JSON.stringify({ tier, encrypt }),
        });
    },

    async importSovereignty(cid) {
        return request('/api/sovereignty/import', {
            method: 'POST',
            body: JSON.stringify({ cid }),
        });
    },

    // Wallet
    async getWallet() {
        return request('/api/wallet');
    },

    // Models
    async getModels() {
        return request('/api/models');
    },

    // Chat
    async invoke(input) {
        return request('/agent/invoke', {
            method: 'POST',
            body: JSON.stringify({ input }),
        });
    },
};

export default KestrelAPI;
