/**
 * Kestrel Sovereign Console - Testable API Client
 *
 * Issues calls to canonical Kestrel paths (e.g. /api/identity, /agent/invoke).
 * Hosts that embed Kestrel UI under a different URL shape are responsible for
 * routing those canonical paths back to the right backend (e.g. by mounting
 * Kestrel UI under a path prefix and forwarding /api/* to a per-agent
 * subprocess). See issue #863 for the rationale.
 *
 * Auth is delegated to an `authProvider` so a host can supply its own
 * (e.g. a JWT it minted) without modifying Kestrel. The default provider
 * preserves the standalone Kestrel-server behavior: try /api/auth/key,
 * fall back to /auth/me, redirect to /auth/login if both fail.
 */

const HOST_LEVEL_AGENTS_RE = /^\/api\/agents\/[^/]+\/(start|stop|status|logs)/;

function getRequiredDependency(name, value) {
    if (!value) {
        throw new Error(`API client requires ${name}`);
    }
    return value;
}

export function isHostLevelEndpoint(endpoint) {
    if (endpoint === '/api/agents' || endpoint.startsWith('/api/agents?')) return true;
    if (HOST_LEVEL_AGENTS_RE.test(endpoint)) return true;
    if (endpoint === '/api/auth/key' || endpoint.startsWith('/api/auth/')) return true;
    if (endpoint === '/health') return true;
    return false;
}

export function applyHostAgentPrefix(endpoint, selectedHostAgent) {
    if (!selectedHostAgent || isHostLevelEndpoint(endpoint)) {
        return endpoint;
    }
    return `/api/agents/${encodeURIComponent(selectedHostAgent)}${endpoint}`;
}

export function createKestrelStandaloneAuthProvider({
    fetchFn,
    sessionStorage,
    location,
    logger,
} = {}) {
    const fetchImpl = getRequiredDependency('fetch', fetchFn);
    const sessionStore = getRequiredDependency('sessionStorage', sessionStorage);
    const locationRef = getRequiredDependency('location', location);
    const log = getRequiredDependency('console', logger);

    let apiKey = null;
    let oauthSession = false;
    let bootstrapDisabled = false;

    async function bootstrapApiKey() {
        try {
            const resp = await fetchImpl('/api/auth/key');
            if (resp.ok) {
                const data = await resp.json();
                apiKey = data.key;
                sessionStore.setItem('kestrel_api_key', apiKey);
                return 'ok';
            }
            if (resp.status === 401 || resp.status === 404 || resp.status === 403) {
                bootstrapDisabled = true;
                return 'disabled';
            }
            log.error('Failed to get API key:', resp.status);
            return 'error';
        } catch (error) {
            log.error('Failed to initialize authentication:', error);
            return 'error';
        }
    }

    return {
        async ensureAuthenticated() {
            const params = new URLSearchParams(locationRef.search || '');
            if (params.get('key')) {
                apiKey = params.get('key');
                sessionStore.setItem('kestrel_api_key', apiKey);
                log.log('API key set from URL parameter');
                return;
            }

            apiKey = sessionStore.getItem('kestrel_api_key');
            if (apiKey) {
                log.log('Using cached API key from sessionStorage');
                return;
            }

            const status = await bootstrapApiKey();
            if (status === 'ok') {
                log.log('API key retrieved and cached');
                return;
            }
            if (status === 'disabled') {
                log.log('API key bootstrap unavailable — checking OAuth session');
            }

            try {
                const meResp = await fetchImpl('/auth/me');
                if (meResp.ok) {
                    const user = await meResp.json();
                    oauthSession = true;
                    log.log(`Authenticated via OAuth: ${user.email}`);
                    return;
                }
            } catch (error) {
                // OAuth session check failed; fall through to redirect/no-auth.
            }

            if (!apiKey && !oauthSession) {
                if (bootstrapDisabled) {
                    log.warn('OAuth required — redirecting to login');
                    locationRef.href = '/auth/login';
                } else {
                    log.warn('No authentication available');
                }
            }
        },

        applyAuth(headers) {
            if (apiKey) {
                return { ...headers, 'X-API-Key': apiKey };
            }
            return headers;
        },

        async onUnauthorized() {
            log.warn('Authentication failed - refreshing API key...');
            sessionStore.removeItem('kestrel_api_key');
            apiKey = null;

            const status = await bootstrapApiKey();
            if (status === 'ok') {
                log.log('API key refreshed - retrying request');
                return 'refreshed';
            }

            if (oauthSession) {
                log.warn('OAuth session expired — redirecting to login');
                locationRef.href = '/auth/login';
                return 'redirected';
            }

            return 'failed';
        },

        getApiKey() {
            return apiKey;
        },
    };
}

export function createBearerTokenAuthProvider({
    getToken,
    onUnauthenticated,
    headerName = 'Authorization',
    tokenPrefix = 'Bearer ',
} = {}) {
    if (typeof getToken !== 'function') {
        throw new Error('BearerTokenAuthProvider requires getToken()');
    }

    return {
        async ensureAuthenticated() {
            const token = await getToken();
            if (!token) {
                if (typeof onUnauthenticated === 'function') {
                    await onUnauthenticated();
                }
                throw new Error('Bearer token unavailable');
            }
        },

        async applyAuth(headers) {
            const token = await getToken();
            if (!token) return headers;
            return { ...headers, [headerName]: `${tokenPrefix}${token}` };
        },

        async onUnauthorized() {
            if (typeof onUnauthenticated === 'function') {
                await onUnauthenticated();
                return 'redirected';
            }
            return 'failed';
        },
    };
}

export function createApiClient({
    fetchFn = globalThis.fetch,
    sessionStorage = globalThis.sessionStorage,
    location = globalThis.location,
    logger = globalThis.console,
    AbortControllerCtor = globalThis.AbortController,
    TextDecoderCtor = globalThis.TextDecoder,
    authProvider = null,
} = {}) {
    const fetchImpl = getRequiredDependency('fetch', fetchFn);
    const sessionStore = getRequiredDependency('sessionStorage', sessionStorage);
    const locationRef = getRequiredDependency('location', location);
    const log = getRequiredDependency('console', logger);
    const AbortCtor = getRequiredDependency('AbortController', AbortControllerCtor);
    const DecoderCtor = getRequiredDependency('TextDecoder', TextDecoderCtor);

    const auth = authProvider || createKestrelStandaloneAuthProvider({
        fetchFn: fetchImpl,
        sessionStorage: sessionStore,
        location: locationRef,
        logger: log,
    });

    const state = {
        selectedHostAgent: null,
        streamAbortController: null,
        currentStreamRequestId: null,
    };

    const client = {
        async init() {
            await auth.ensureAuthenticated();
        },

        async request(endpoint, options = {}, retried = false) {
            const headers = await buildHeaders(options.headers);
            // Let the browser set Content-Type for FormData (multipart boundary)
            if (options.body instanceof FormData) {
                delete headers['Content-Type'];
            }
            const url = applyHostAgentPrefix(endpoint, state.selectedHostAgent);
            const response = await fetchImpl(url, { ...options, headers });

            if (response.status === 401 && !retried) {
                const recovery = await auth.onUnauthorized();
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
        getAgentInfo: () => client.request('/api/agent/info'),
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
        getPrivacyMode: () => client.request('/api/agent/privacy-mode'),
        // setPrivacyMode is destructive on live agents (#867 gates the
        // endpoint) — a flip into EPHEMERAL primes the leak-purge for the
        // next exit.  The UI carries the opt-in header so the rail sees an
        // intentional change; scripts must opt in explicitly.
        setPrivacyMode: (mode) => client.request('/api/agent/privacy-mode', {
            method: 'POST',
            headers: { 'X-Kestrel-Allow-Destructive': 'user-initiated-mode-change' },
            body: JSON.stringify({ mode }),
        }),
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
        // Soft-delete moves to Trash, recoverable (#763 / #765).  The
        // X-Kestrel-Allow-Destructive header satisfies the demo-isolation
        // rail (#766) which gates every destructive endpoint behind an
        // explicit opt-in so a stray script can't wipe a live agent.
        deleteMessage: (messageId) => client.request(`/api/conversations/messages/${encodeURIComponent(messageId)}`, {
            method: 'DELETE',
            headers: { 'X-Kestrel-Allow-Destructive': 'user-initiated-ui' },
        }),
        deleteConversation: (sessionId) => client.request(`/api/conversations/${encodeURIComponent(sessionId)}`, {
            method: 'DELETE',
            headers: { 'X-Kestrel-Allow-Destructive': 'user-initiated-ui' },
        }),
        // Trash sub-view surface — list trashed items, restore one out of
        // Trash, or hard-purge with an audit reason.  Purge is irreversible
        // and always carries the destructive header.
        listTrash: (limit = 200) => client.request(`/api/trash?limit=${encodeURIComponent(limit)}`),
        restoreConversation: (sessionId) => client.request(
            `/api/conversations/${encodeURIComponent(sessionId)}/restore`,
            { method: 'POST' },
        ),
        purgeConversation: (sessionId, reason = 'user-initiated') => client.request(
            `/api/conversations/${encodeURIComponent(sessionId)}/purge`,
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Kestrel-Allow-Destructive': 'user-initiated-purge',
                },
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
                headers: {
                    'Content-Type': 'application/json',
                    'X-Kestrel-Allow-Destructive': 'user-initiated-purge',
                },
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
        invoke: (input, model = null, sessionId = null, provider = null) => client.request('/api/agent/invoke', {
            method: 'POST',
            body: JSON.stringify({ input, model, session_id: sessionId, provider }),
        }),
        stop: (requestId = null) => client.request('/api/agent/stop', {
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
            const headers = await buildHeaders({ 'Content-Type': 'application/json' });

            state.streamAbortController = new AbortCtor();
            const signal = state.streamAbortController.signal;

            try {
                const url = applyHostAgentPrefix('/api/agent/stream', state.selectedHostAgent);
                const response = await fetchImpl(url, {
                    method: 'POST',
                    headers,
                    body: JSON.stringify({ input, model, session_id: sessionId, provider }),
                    signal,
                });

                if (response.status === 401 && !retried) {
                    const recovery = await auth.onUnauthorized();
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
            return typeof auth.getApiKey === 'function' ? auth.getApiKey() : null;
        },
        // Sign an arbitrary headers object with whatever the active auth
        // provider attaches. Use this from any code that has to call fetch()
        // directly instead of going through client.request() — most commonly
        // anything that needs Content-Type control (FormData) or a non-JSON
        // protocol (EventSource preflight, etc.).
        async applyAuth(headers = {}) {
            return await auth.applyAuth({ ...headers });
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
            return applyHostAgentPrefix(path, state.selectedHostAgent);
        },
        getContextStatus: (sessionId = null) => {
            const params = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
            return client.request(`/api/agent/context-status${params}`);
        },
        _getState() {
            return { ...state };
        },
    };

    async function buildHeaders(extraHeaders = {}) {
        const headers = {
            'Content-Type': 'application/json',
            ...extraHeaders,
        };
        return await auth.applyAuth(headers);
    }

    return client;
}
