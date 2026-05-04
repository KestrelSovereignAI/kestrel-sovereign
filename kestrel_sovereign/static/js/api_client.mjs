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

// Canonical list of UI capabilities a host can opt out of (#879).  Embedded
// hosts populate ``KESTREL_UI_CONFIG.capabilities`` with a subset of these
// keys set to ``false`` (or to a nested object for partial support); missing
// keys default to ``true`` so standalone Kestrel renders unchanged.  This is
// the single source of truth — when a new panel ships, add its key here and
// have the panel guard its init() with ``API.hasCapability(key)``.
//
// Object-shaped values are supported via dot-paths (e.g. ``keys.agent``):
// any sub-key absent from the host config is treated as ``true``.
export const CAPABILITY_KEYS = Object.freeze({
    chat: true,
    identity: true,
    constitution: true,
    privacy: true,
    memory: true,
    tasks: true,
    sovereignty: true,
    storage: true,
    wallet: true,
    conversations: true,
    keys: true,           // object: { agent, user, platform }
    audit: true,
    permissions: true,
    rookery: true,        // host-level: "Other agents" sidebar (/api/agents)
    spawn: true,
    featureStore: true,
    metrics: true,
});

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

// Resolve a dot-path against the host-supplied capabilities map.  Returns
// ``true`` when the key is absent (default-on), ``false`` when explicitly
// disabled at any segment.  Boolean leaves win immediately; object leaves
// recurse on the next segment.  Exported for unit tests.
export function resolveCapability(capabilities, path) {
    if (!path) return true;
    const segments = String(path).split('.');
    let cursor = capabilities;
    for (const seg of segments) {
        if (cursor === null || cursor === undefined) return true;
        if (typeof cursor === 'boolean') return cursor;
        if (typeof cursor !== 'object') return true;
        if (!(seg in cursor)) return true;
        cursor = cursor[seg];
    }
    if (cursor === null || cursor === undefined) return true;
    if (typeof cursor === 'boolean') return cursor;
    // Object leaf with no matching path segment — feature is enabled overall;
    // its subkeys gate sub-sections.  Treat the parent as enabled.
    if (typeof cursor === 'object') {
        // If every sub-key is explicitly false, the parent is effectively off.
        const values = Object.values(cursor);
        if (values.length > 0 && values.every((v) => v === false)) return false;
        return true;
    }
    return true;
}

export function createApiClient({
    fetchFn = globalThis.fetch,
    sessionStorage = globalThis.sessionStorage,
    location = globalThis.location,
    logger = globalThis.console,
    AbortControllerCtor = globalThis.AbortController,
    TextDecoderCtor = globalThis.TextDecoder,
    authProvider = null,
    capabilities = null,
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

    // Capabilities map (#879).  ``null`` and ``{}`` both mean "host did not
    // opt out of anything"; missing keys default to ``true``.  Stored as a
    // plain object so callers can read it via ``client.getCapabilities()``
    // for diagnostics, but the resolver is the only sanctioned read path.
    const capsMap = capabilities && typeof capabilities === 'object' ? capabilities : {};

    const state = {
        selectedHostAgent: null,
        // Per-agent stream bookkeeping. Keyed by host-agent name (null key
        // for standalone mode). Was a single slot, which clobbered Agent A's
        // controller when a second stream started on Agent B and made the
        // Stop button target the wrong stream.
        streamAbortControllers: new Map(),
        currentStreamRequestIds: new Map(),
        // Effective session_id reported by the most recent /stream or
        // /invoke for each agent. The server resolves this via the 30-
        // min-gap heuristic when the caller passes null, so before this
        // map existed the frontend pane never learned its durable
        // session id and stayed anchored on null indefinitely.
        effectiveSessionIds: new Map(),
    };

    // Single-source the fetch + auth + 401-retry pipeline so both
    // request() and requestForAgent() share it. Without this factor
    // out, an explicit-agent call would either skip auth-refresh or
    // re-implement it (drift risk). The url passed in is FINAL — the
    // caller has already applied host-agent prefixing if needed.
    async function performRequest(url, options = {}, retried = false, retryCb = null) {
        const headers = await buildHeaders(options.headers);
        if (options.body instanceof FormData) {
            delete headers['Content-Type'];
        }
        const response = await fetchImpl(url, { ...options, headers });

        if (response.status === 401 && !retried) {
            const recovery = await auth.onUnauthorized();
            if (recovery === 'redirected') return;
            if (recovery === 'refreshed' && retryCb) return retryCb();

            const error = await response.json().catch(() => ({ detail: 'Authentication failed' }));
            throw new Error(error.detail || 'Authentication failed - please refresh the page');
        }

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        return response.json();
    }

    const client = {
        async init() {
            await auth.ensureAuthenticated();
        },

        async request(endpoint, options = {}, retried = false) {
            const url = applyHostAgentPrefix(endpoint, state.selectedHostAgent);
            return performRequest(url, options, retried, () => this.request(endpoint, options, true));
        },

        // Like request() but pins the host-agent prefix to the explicit
        // `agent` arg instead of state.selectedHostAgent. Used when a
        // caller must reach a specific agent's endpoint while the user
        // is currently viewing a different one — e.g. clicking the
        // sidebar Stop control on Agent A while the chat pane shows
        // Agent B. The caller must pass the canonical /api/agent/...
        // path; routing is applied here so we don't double-prefix.
        async requestForAgent(endpoint, options = {}, agent, retried = false) {
            const url = applyHostAgentPrefix(endpoint, agent);
            return performRequest(url, options, retried, () => this.requestForAgent(endpoint, options, agent, true));
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
        invoke: async (input, model = null, sessionId = null, provider = null) => {
            // Capture dispatchAgent BEFORE the await so the session_id
            // we record on the response is bound to the agent that
            // owned this dispatch — not whichever agent was selected
            // when the response landed.
            const dispatchAgent = state.selectedHostAgent;
            const result = await client.request('/api/agent/invoke', {
                method: 'POST',
                body: JSON.stringify({ input, model, session_id: sessionId, provider }),
            });
            if (result && typeof result === 'object' && result.session_id) {
                state.effectiveSessionIds.set(dispatchAgent, result.session_id);
            }
            return result;
        },
        // Explicit-agent invoke — mirrors stop(id, agent). sendMessage
        // dispatches a chat against a specific agent's pane; if the
        // user switches agents while sendMessage awaits this call, the
        // unprefixed `invoke()` would route to whichever agent is
        // currently selected and the response would land in the wrong
        // pane. invokeForAgent pins the URL to the captured dispatch
        // agent so the request always reaches the agent the chat was
        // sent to.
        invokeForAgent: async (input, model = null, sessionId = null, provider = null, agent) => {
            const opts = {
                method: 'POST',
                body: JSON.stringify({ input, model, session_id: sessionId, provider }),
            };
            const dispatchAgent = agent === undefined ? state.selectedHostAgent : agent;
            const result = agent !== undefined
                ? await client.requestForAgent('/api/agent/invoke', opts, agent)
                : await client.request('/api/agent/invoke', opts);
            if (result && typeof result === 'object' && result.session_id) {
                state.effectiveSessionIds.set(dispatchAgent, result.session_id);
            }
            return result;
        },
        // Two-arg overload: pass `agent` to target a specific agent's
        // /stop endpoint regardless of which agent is currently
        // selected. Without it, a sidebar "Stop Agent A" click while
        // viewing Agent B would route the stop to B's backend (because
        // request() pins to state.selectedHostAgent), aborting client-
        // side but never telling A's server to halt.
        stop: (requestId = null, agent) => {
            const opts = {
                method: 'POST',
                body: JSON.stringify(requestId ? { request_id: requestId } : {}),
            };
            return agent !== undefined
                ? client.requestForAgent('/api/agent/stop', opts, agent)
                : client.request('/api/agent/stop', opts);
        },
        getModels: (options = {}) => {
            const params = new URLSearchParams();
            if (options.featuredOnly !== undefined) params.append('featured_only', options.featuredOnly);
            if (options.category) params.append('category', options.category);
            if (options.providers) params.append('providers', options.providers.join(','));
            if (options.useCache !== undefined) params.append('use_cache', options.useCache);
            const queryString = params.toString();
            return client.request(`/api/models${queryString ? `?${queryString}` : ''}`);
        },
        getStreamAbortController(agent) {
            // Default to the current host agent so existing single-agent
            // call sites keep working unchanged.
            const key = agent === undefined ? state.selectedHostAgent : agent;
            return state.streamAbortControllers.get(key) || null;
        },
        getCurrentStreamRequestId(agent) {
            const key = agent === undefined ? state.selectedHostAgent : agent;
            return state.currentStreamRequestIds.get(key) || null;
        },
        // Effective session_id surfaced by the server's most recent
        // /stream or /invoke for this agent. Returns null until the
        // first response has been received. sendMessage reads this to
        // update pane.sessionId so each agent's pane learns its
        // durable conversation id without relying on the prior
        // pane.sessionId=null + server-side implicit-derive heuristic.
        getEffectiveSessionId(agent) {
            const key = agent === undefined ? state.selectedHostAgent : agent;
            return state.effectiveSessionIds.get(key) || null;
        },
        async *streamInvoke(input, model = null, sessionId = null, provider = null, retried = false, agent) {
            // Pin the dispatch agent. The sixth `agent` parameter lets a
            // caller (sendMessage) capture state.selectedHostAgent at
            // its own dispatch boundary and pass it through, so the user
            // switching agents between sendMessage's capture and
            // streamInvoke's first await can't drift the URL to the
            // wrong backend. The 401 retry path also passes this value
            // to itself instead of recapturing state.selectedHostAgent
            // — that recursion was the original bug: an auth refresh
            // on Agent A's stream could yield to a switched-to Agent B.
            const dispatchAgent = agent === undefined ? state.selectedHostAgent : agent;

            // Build auth headers BEFORE installing the abort controller in
            // the per-agent map. If buildHeaders() throws (auth provider
            // failure, bearer-token unavailable, etc.) we must not leave a
            // stale controller behind for the next Stop click to fire on.
            const headers = await buildHeaders({ 'Content-Type': 'application/json' });

            const controller = new AbortCtor();
            const signal = controller.signal;
            state.streamAbortControllers.set(dispatchAgent, controller);

            try {
                const url = applyHostAgentPrefix('/api/agent/stream', dispatchAgent);
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
                        // Pass the SAME dispatchAgent to the retry — do
                        // NOT let it recapture state.selectedHostAgent.
                        yield* client.streamInvoke(input, model, sessionId, provider, true, dispatchAgent);
                        return;
                    }

                    const error = await response.json().catch(() => ({ detail: 'Authentication failed' }));
                    throw new Error(error.detail || 'Authentication failed - please refresh the page');
                }

                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                state.currentStreamRequestIds.set(dispatchAgent, response.headers.get('X-Request-ID'));
                // Capture the server-resolved session_id BEFORE the body
                // streams. sendMessage reads it via getEffectiveSessionId
                // immediately so pane.sessionId can be set on the very
                // first turn; subsequent turns send it back as an
                // explicit value, anchoring the pane to a durable id.
                const headerSid = response.headers.get('X-Session-Id');
                if (headerSid) {
                    state.effectiveSessionIds.set(dispatchAgent, headerSid);
                }
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
                // Re-throw AbortError so sendMessage can distinguish a
                // user-initiated stop from a clean stream end. The old
                // silent return swallowed the signal — sendMessage then
                // toasted "agent finished responding" on a non-visible
                // agent the user had just stopped from the sidebar.
                if (error.name === 'AbortError') {
                    log.log('Stream aborted by user');
                }
                throw error;
            } finally {
                // Clear only this dispatch's slot — never the *current* agent's,
                // which may belong to a different in-flight stream.
                if (state.streamAbortControllers.get(dispatchAgent) === controller) {
                    state.streamAbortControllers.delete(dispatchAgent);
                }
                state.currentStreamRequestIds.delete(dispatchAgent);
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
        // #879: capability gating.  Hosts supply ``capabilities`` on
        // ``KESTREL_UI_CONFIG`` to opt out of panels that don't apply in
        // their embed.  Default is "everything on" so standalone Kestrel and
        // pre-#879 hosts behave exactly as before.  Pass dot-paths for
        // nested checks (e.g. ``hasCapability('keys.agent')``).
        hasCapability(path) {
            return resolveCapability(capsMap, path);
        },
        getCapabilities() {
            return capsMap;
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
