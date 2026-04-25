/**
 * Kestrel Sovereign Console - Testable API Client
 * Extracted from api.js so auth and routing behavior can be tested directly.
 */

const ENDPOINT_MAPPINGS = {
    '/api/identity': '/identity',
    '/api/constitution': '/constitution',
    '/agent/privacy-mode': '/privacy',
    '/api/memories': '/memory',
    '/api/sovereignty/exports': '/sovereignty/exports',
    '/api/sovereignty/export': '/sovereignty/export',
    '/api/sovereignty/import': '/sovereignty/import',
    '/agent/invoke': '/invoke',
    '/agent/stream': '/stream',
    '/api/models': '/models',
    '/api/model/current': '/model/current',
    '/api/commands': '/commands',
    '/api/storage/stats': '/storage/stats',
    '/api/wallet': '/wallet',
    '/api/keys': '/keys',
    '/api/conversations': '/conversations',
    '/api/tasks': '/tasks',
    '/api/security/pending': '/security/pending',
    '/api/security/approve': '/security/approve',
    '/api/security/permissions': '/security/permissions',
    '/api/db/tables': '/db/tables',
    '/api/ipfs/status': '/ipfs/status',
};

function getRequiredDependency(name, value) {
    if (!value) {
        throw new Error(`API client requires ${name}`);
    }
    return value;
}

export function isHostLevelEndpoint(endpoint) {
    if (endpoint === '/api/agents' || endpoint.startsWith('/api/agents?')) return true;
    if (/^\/api\/agents\/[^/]+\/(start|stop|status|logs)/.test(endpoint)) return true;
    if (endpoint === '/api/auth/key' || endpoint.startsWith('/api/auth/')) return true;
    if (endpoint === '/health') return true;
    return false;
}

export function rewriteEndpoint(endpoint, { currentAgentId = null, selectedHostAgent = null } = {}) {
    let rewritten = endpoint;

    if (selectedHostAgent && !isHostLevelEndpoint(rewritten)) {
        rewritten = `/api/agents/${encodeURIComponent(selectedHostAgent)}${rewritten}`;
    }

    if (!currentAgentId) {
        return rewritten;
    }

    const agentBase = `/api/kestrel/companions/${encodeURIComponent(currentAgentId)}`;
    if (ENDPOINT_MAPPINGS[rewritten]) {
        return `${agentBase}${ENDPOINT_MAPPINGS[rewritten]}`;
    }

    for (const [pattern, suffix] of Object.entries(ENDPOINT_MAPPINGS)) {
        if (rewritten.startsWith(pattern)) {
            return `${agentBase}${suffix}${rewritten.slice(pattern.length)}`;
        }
    }

    return rewritten;
}

export function createApiClient({
    fetchFn = globalThis.fetch,
    localStorage = globalThis.localStorage,
    sessionStorage = globalThis.sessionStorage,
    location = globalThis.location,
    logger = globalThis.console,
    AbortControllerCtor = globalThis.AbortController,
    TextDecoderCtor = globalThis.TextDecoder,
} = {}) {
    const fetchImpl = getRequiredDependency('fetch', fetchFn);
    const localStore = getRequiredDependency('localStorage', localStorage);
    const sessionStore = getRequiredDependency('sessionStorage', sessionStorage);
    const locationRef = getRequiredDependency('location', location);
    const log = getRequiredDependency('console', logger);
    const AbortCtor = getRequiredDependency('AbortController', AbortControllerCtor);
    const DecoderCtor = getRequiredDependency('TextDecoder', TextDecoderCtor);

    const state = {
        apiKey: null,
        jwtToken: null,
        oauthSession: false,
        bootstrapDisabled: false,
        currentAgentId: null,
        selectedHostAgent: null,
        streamAbortController: null,
        currentStreamRequestId: null,
    };

    const client = {
        async init() {
            if (state.currentAgentId) {
                state.jwtToken = readPlatformJwt();
                if (state.jwtToken) {
                    log.log('Kestrel UI: Using JWT token from session');
                    return;
                }
                log.warn('Kestrel UI: Multi-agent mode but no JWT token found');
            }

            // Allow ?key= query param for convenience (matches dashboard behavior)
            const params = new URLSearchParams(locationRef.search);
            if (params.get('key')) {
                state.apiKey = params.get('key');
                sessionStore.setItem('kestrel_api_key', state.apiKey);
                log.log('API key set from URL parameter');
                return;
            }

            state.apiKey = sessionStore.getItem('kestrel_api_key');
            if (state.apiKey) {
                log.log('Using cached API key from sessionStorage');
                return;
            }

            try {
                const resp = await fetchImpl('/api/auth/key');
                if (resp.ok) {
                    const data = await resp.json();
                    state.apiKey = data.key;
                    sessionStore.setItem('kestrel_api_key', state.apiKey);
                    log.log('API key retrieved and cached');
                    return;
                }
                if (resp.status === 401 || resp.status === 404 || resp.status === 403) {
                    state.bootstrapDisabled = true;
                    log.log('API key bootstrap unavailable — checking OAuth session');
                } else {
                    log.error('Failed to get API key:', resp.status);
                }
            } catch (error) {
                log.error('Failed to initialize authentication:', error);
            }

            try {
                const meResp = await fetchImpl('/auth/me');
                if (meResp.ok) {
                    const user = await meResp.json();
                    state.oauthSession = true;
                    log.log(`Authenticated via OAuth: ${user.email}`);
                    return;
                }
            } catch (error) {
                // OAuth session check failed; init falls through to redirect/no-auth behavior.
            }

            if (!state.apiKey && !state.jwtToken && !state.oauthSession) {
                if (state.bootstrapDisabled) {
                    log.warn('OAuth required — redirecting to login');
                    locationRef.href = '/auth/login';
                } else {
                    log.warn('No authentication available');
                }
            }
        },

        async request(endpoint, options = {}, retried = false) {
            const headers = buildHeaders(options.headers);
            // Let the browser set Content-Type for FormData (multipart boundary)
            if (options.body instanceof FormData) {
                delete headers['Content-Type'];
            }
            const response = await fetchImpl(rewrite(endpoint), {
                ...options,
                headers,
            });

            if (response.status === 401 && !retried) {
                const recovery = await recoverFromUnauthorized();
                if (recovery === 'redirected') return;
                if (recovery === 'refreshed') return this.request(endpoint, options, true);

                const error = await response.json().catch(() => ({ detail: 'Authentication failed' }));
                throw new Error(error.detail || 'Authentication failed - please refresh the page');
            }

            if (!response.ok) {
                const error = await response.json().catch(() => ({ detail: response.statusText }));
                throw new Error(error.detail || `HTTP ${response.status}`);
            }
            return response.json();
        },

        health: () => client.request('/health'),
        getAgentInfo: () => client.request('/agent/info'),
        getAgents: () => client.request('/api/agents'),
        getIdentity: () => client.request('/api/identity'),
        updateIdentity: (data) => client.request('/api/identity', {
            method: 'PATCH',
            body: JSON.stringify(data),
        }),
        uploadAvatar: (file) => {
            const formData = new FormData();
            formData.append('file', file);
            return client.request('/api/identity/avatar', {
                method: 'POST',
                body: formData,
            });
        },
        setAvatarFromUrl: (url) => client.request('/api/identity/avatar', {
            method: 'POST',
            body: JSON.stringify({ url }),
        }),
        generateAvatar: (description, numOutputs = 2) => client.request('/api/identity/avatar/generate', {
            method: 'POST',
            body: JSON.stringify({ description, num_outputs: numOutputs }),
        }),
        getConstitution: () => client.request('/api/constitution'),
        getPrivacyMode: () => client.request('/agent/privacy-mode'),
        setPrivacyMode: (mode) => client.request('/agent/privacy-mode', { method: 'POST', body: JSON.stringify({ mode }) }),
        getSessions: (limit = 50) => client.request(`/api/sessions?limit=${limit}`),
        getMemories: (nodeType = null, limit = 100) => {
            let url = `/api/memories?limit=${limit}`;
            if (nodeType) url += `&node_type=${nodeType}`;
            return client.request(url);
        },
        deleteMemory: (nodeId) => client.request(`/api/memories/${encodeURIComponent(nodeId)}`, { method: 'DELETE' }),
        getMemoryDetail: (nodeId) => client.request(`/api/memories/${encodeURIComponent(nodeId)}`),
        getIdentityChain: () => client.request('/api/identity-chain'),
        getStorageStats: () => client.request('/api/storage/stats'),
        getSovereigntyExports: () => client.request('/api/sovereignty/exports'),
        exportSovereignty: (tier, encrypt) => client.request('/api/sovereignty/export', { method: 'POST', body: JSON.stringify({ tier, encrypt }) }),
        importSovereignty: (cid) => client.request('/api/sovereignty/import', { method: 'POST', body: JSON.stringify({ cid }) }),
        getSovereigntyFiles: () => client.request('/api/sovereignty/files'),
        getSovereigntyFilePreview: (filename) => client.request(`/api/sovereignty/files/${encodeURIComponent(filename)}/preview`),
        getConversations: (decrypt = true) => client.request(`/api/conversations?decrypt=${decrypt}`),
        getConversation: (sessionId, decrypt = true) => client.request(`/api/conversations/${encodeURIComponent(sessionId)}?decrypt=${decrypt}`),
        newConversation: () => client.request('/api/conversations/new', { method: 'POST' }),
        deleteMessage: (messageId) => client.request(`/api/conversations/messages/${encodeURIComponent(messageId)}`, { method: 'DELETE' }),
        deleteConversation: (sessionId) => client.request(`/api/conversations/${encodeURIComponent(sessionId)}`, { method: 'DELETE' }),
        // Soft-delete recovery surface (#763 / #765). Restore brings a row
        // out of Trash; purge hard-deletes with an audit reason. listTrash
        // is the read side that backs the Trash sub-view.
        listTrash: (limit = 200) => client.request(`/api/trash?limit=${encodeURIComponent(limit)}`),
        restoreConversation: (sessionId) => client.request(
            `/api/conversations/${encodeURIComponent(sessionId)}/restore`,
            { method: 'POST' },
        ),
        purgeConversation: (sessionId, reason = 'user-initiated') => client.request(
            `/api/conversations/${encodeURIComponent(sessionId)}/purge`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ reason }),
            },
        ),
        restoreMessage: (messageId) => client.request(
            `/api/conversations/messages/${encodeURIComponent(messageId)}/restore`,
            { method: 'POST' },
        ),
        purgeMessage: (messageId, reason = 'user-initiated') => client.request(
            `/api/conversations/messages/${encodeURIComponent(messageId)}/purge`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ reason }),
            },
        ),
        renameConversation: (sessionId, name) => client.request(
            `/api/conversations/${encodeURIComponent(sessionId)}`,
            {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name }),
            },
        ),
        getDbTables: () => client.request('/api/db/tables'),
        queryDbTable: (table, limit = 50, offset = 0, search = null) => {
            let url = `/api/db/tables/${encodeURIComponent(table)}?limit=${limit}&offset=${offset}`;
            if (search) url += `&search=${encodeURIComponent(search)}`;
            return client.request(url);
        },
        getIpfsStatus: () => client.request('/api/ipfs/status'),
        getWallet: () => client.request('/api/wallet'),
        invoke: (input, model = null, sessionId = null, provider = null) => client.request('/agent/invoke', {
            method: 'POST',
            body: JSON.stringify({ input, model, session_id: sessionId, provider }),
        }),
        stop: (requestId = null) => client.request('/agent/stop', {
            method: 'POST',
            body: JSON.stringify(requestId ? { request_id: requestId } : {}),
        }),
        getModels: (options = {}) => {
            const params = new URLSearchParams();
            if (options.featuredOnly !== undefined) params.append('featured_only', options.featuredOnly);
            if (options.category) params.append('category', options.category);
            if (options.providers) params.append('providers', options.providers.join(','));
            if (options.useCache !== undefined) params.append('use_cache', options.useCache);
            const queryString = params.toString();
            return client.request(`/api/models${queryString ? `?${queryString}` : ''}`);
        },
        getStreamAbortController() {
            return state.streamAbortController;
        },
        getCurrentStreamRequestId() {
            return state.currentStreamRequestId;
        },
        async *streamInvoke(input, model = null, sessionId = null, provider = null, retried = false) {
            const headers = { 'Content-Type': 'application/json' };
            if (state.jwtToken) {
                headers.Authorization = `Bearer ${state.jwtToken}`;
            } else if (state.apiKey) {
                headers['X-API-Key'] = state.apiKey;
            }

            state.streamAbortController = new AbortCtor();
            const signal = state.streamAbortController.signal;

            try {
                const response = await fetchImpl(rewrite('/agent/stream'), {
                    method: 'POST',
                    headers,
                    body: JSON.stringify({ input, model, session_id: sessionId, provider }),
                    signal,
                });

                if (response.status === 401 && !retried) {
                    const recovery = await recoverFromUnauthorized();
                    if (recovery === 'redirected') return;
                    if (recovery === 'refreshed') {
                        yield* client.streamInvoke(input, model, sessionId, provider, true);
                        return;
                    }

                    const error = await response.json().catch(() => ({ detail: 'Authentication failed' }));
                    throw new Error(error.detail || 'Authentication failed - please refresh the page');
                }

                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                state.currentStreamRequestId = response.headers.get('X-Request-ID');
                const reader = response.body.getReader();
                const decoder = new DecoderCtor();

                try {
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        yield decoder.decode(value, { stream: true });
                    }
                } finally {
                    reader.releaseLock();
                }
            } catch (error) {
                if (error.name === 'AbortError') {
                    log.log('Stream aborted by user');
                    return;
                }
                throw error;
            } finally {
                state.streamAbortController = null;
                state.currentStreamRequestId = null;
            }
        },
        getApiKey() {
            return state.apiKey;
        },
        setAgentId(agentId) {
            state.currentAgentId = agentId;
        },
        getAgentId() {
            return state.currentAgentId;
        },
        isMultiAgentMode() {
            return state.currentAgentId !== null;
        },
        setHostAgent(agentName) {
            state.selectedHostAgent = agentName;
        },
        getHostAgent() {
            return state.selectedHostAgent;
        },
        isRookeryMode() {
            return state.selectedHostAgent !== null;
        },
        buildAgentUrl(path) {
            if (state.selectedHostAgent && !isHostLevelEndpoint(path)) {
                return `/api/agents/${encodeURIComponent(state.selectedHostAgent)}${path}`;
            }
            return path;
        },
        getContextStatus: (sessionId = null) => {
            const params = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
            return client.request(`/agent/context-status${params}`);
        },
        _getState() {
            return { ...state };
        },
    };

    function buildHeaders(extraHeaders = {}) {
        const headers = {
            'Content-Type': 'application/json',
            ...extraHeaders,
        };
        if (state.jwtToken) {
            headers.Authorization = `Bearer ${state.jwtToken}`;
        } else if (state.apiKey) {
            headers['X-API-Key'] = state.apiKey;
        }
        return headers;
    }

    async function recoverFromUnauthorized() {
        if (state.jwtToken) {
            log.warn('Platform session expired — clearing JWT and redirecting to login');
            clearPlatformJwt();
            locationRef.href = '/auth/login';
            return 'redirected';
        }

        if (await refreshApiKey()) {
            return 'refreshed';
        }

        if (state.oauthSession) {
            log.warn('OAuth session expired — redirecting to login');
            locationRef.href = '/auth/login';
            return 'redirected';
        }

        return 'failed';
    }

    async function refreshApiKey() {
        log.warn('Authentication failed - refreshing API key...');
        sessionStore.removeItem('kestrel_api_key');
        state.apiKey = null;

        try {
            const keyResp = await fetchImpl('/api/auth/key');
            if (keyResp.ok) {
                const data = await keyResp.json();
                state.apiKey = data.key;
                sessionStore.setItem('kestrel_api_key', state.apiKey);
                log.log('API key refreshed - retrying request');
                return true;
            }
            if (keyResp.status === 401 || keyResp.status === 403 || keyResp.status === 404) {
                if (keyResp.status === 401 || keyResp.status === 404) {
                    state.bootstrapDisabled = true;
                }
                log.warn('API key bootstrap unavailable during refresh');
            }
        } catch (error) {
            log.error('Failed to refresh API key:', error);
        }

        return false;
    }

    function rewrite(endpoint) {
        return rewriteEndpoint(endpoint, {
            currentAgentId: state.currentAgentId,
            selectedHostAgent: state.selectedHostAgent,
        });
    }

    function readPlatformJwt() {
        return localStore.getItem('platform_auth_token')
            || localStore.getItem('token')
            || sessionStore.getItem('token');
    }

    function clearPlatformJwt() {
        state.jwtToken = null;
        localStore.removeItem('platform_auth_token');
        localStore.removeItem('token');
        sessionStore.removeItem('token');
    }

    return client;
}
