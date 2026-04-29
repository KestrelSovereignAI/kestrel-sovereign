import test from 'node:test';
import assert from 'node:assert/strict';

import {
    createApiClient,
    createBearerTokenAuthProvider,
    applyHostAgentPrefix,
    isHostLevelEndpoint,
} from '../../kestrel_sovereign/static/js/api_client.mjs';

function createStorage(initial = {}) {
    const store = new Map(Object.entries(initial));
    return {
        getItem(key) {
            return store.has(key) ? store.get(key) : null;
        },
        setItem(key, value) {
            store.set(key, String(value));
        },
        removeItem(key) {
            store.delete(key);
        },
        dump() {
            return Object.fromEntries(store.entries());
        },
    };
}

function jsonResponse(status, body) {
    return {
        ok: status >= 200 && status < 300,
        status,
        statusText: body?.detail || `HTTP ${status}`,
        async json() {
            return body;
        },
    };
}

function streamResponse(chunks, headers = {}) {
    const encoded = chunks.map((chunk) => new TextEncoder().encode(chunk));
    return {
        ok: true,
        status: 200,
        headers: {
            get(name) {
                return headers[name] ?? headers[name.toLowerCase()] ?? null;
            },
        },
        body: {
            getReader() {
                let index = 0;
                return {
                    async read() {
                        if (index >= encoded.length) return { done: true };
                        return { done: false, value: encoded[index++] };
                    },
                    releaseLock() {},
                };
            },
        },
    };
}

function createLogger() {
    const entries = [];
    return {
        entries,
        log(...args) {
            entries.push(['log', ...args]);
        },
        warn(...args) {
            entries.push(['warn', ...args]);
        },
        error(...args) {
            entries.push(['error', ...args]);
        },
    };
}

function createFetchQueue(...responses) {
    const calls = [];
    const fetchFn = async (url, options = {}) => {
        calls.push({ url, options });
        const next = responses.shift();
        if (!next) {
            throw new Error(`Unexpected fetch to ${url}`);
        }
        return typeof next === 'function' ? next(url, options) : next;
    };
    fetchFn.calls = calls;
    return fetchFn;
}

function createClient({ fetchFn, sessionInitial = {}, authProvider = null } = {}) {
    const logger = createLogger();
    const location = { href: '/console', search: '' };
    const sessionStorage = createStorage(sessionInitial);
    const client = createApiClient({
        fetchFn,
        sessionStorage,
        location,
        logger,
        authProvider,
    });
    return { client, logger, location, sessionStorage };
}

test('init caches bootstrap API key when bootstrap succeeds', async () => {
    const fetchFn = createFetchQueue(jsonResponse(200, { key: 'k-secret' }));
    const { client, sessionStorage, location } = createClient({ fetchFn });

    await client.init();

    assert.equal(client.getApiKey(), 'k-secret');
    assert.equal(sessionStorage.getItem('kestrel_api_key'), 'k-secret');
    assert.equal(location.href, '/console');
    assert.deepEqual(fetchFn.calls.map((call) => call.url), ['/api/auth/key']);
});

test('init redirects to login when bootstrap is unavailable and OAuth session is absent', async () => {
    const fetchFn = createFetchQueue(
        jsonResponse(404, { detail: 'disabled' }),
        jsonResponse(401, { detail: 'unauthenticated' }),
    );
    const { client, location } = createClient({ fetchFn });

    await client.init();

    assert.equal(location.href, '/auth/login');
    assert.deepEqual(fetchFn.calls.map((call) => call.url), ['/api/auth/key', '/auth/me']);
});

test('request hits canonical paths — no companion-prefix rewriting (#863)', async () => {
    // Pre-#863, setAgentId() pushed every call through /api/kestrel/companions/{id}/...
    // — a Frinz-shaped URL embedded into Kestrel UI under a "multi-agent mode"
    // misnomer (Kestrel's actual rookery uses /api/agents/{name}/..., a separate
    // feature preserved by setHostAgent). #863 removed setAgentId entirely;
    // hosts now route canonical /api/* paths themselves.
    const fetchFn = createFetchQueue(jsonResponse(200, { models: ['ok'] }));
    const { client } = createClient({
        fetchFn,
        sessionInitial: { kestrel_api_key: 'k-secret' },
    });

    await client.init();
    await client.request('/api/models');

    assert.deepEqual(fetchFn.calls.map((call) => call.url), ['/api/models']);
    assert.equal(client.setAgentId, undefined);
    assert.equal(client.isMultiAgentMode, undefined);
});

test('request with API key refreshes bootstrap key once and retries the original request', async () => {
    const fetchFn = createFetchQueue(
        jsonResponse(401, { detail: 'expired key' }),
        jsonResponse(200, { key: 'fresh-key' }),
        jsonResponse(200, { models: ['ok'] }),
    );
    const { client, sessionStorage } = createClient({
        fetchFn,
        sessionInitial: { kestrel_api_key: 'stale-key' },
    });

    await client.init();
    const result = await client.request('/api/models');

    assert.deepEqual(result, { models: ['ok'] });
    assert.equal(sessionStorage.getItem('kestrel_api_key'), 'fresh-key');
    assert.deepEqual(fetchFn.calls.map((call) => call.url), [
        '/api/models',
        '/api/auth/key',
        '/api/models',
    ]);
    assert.equal(fetchFn.calls[2].options.headers['X-API-Key'], 'fresh-key');
});

test('streamInvoke posts canonical /agent/stream (#863)', async () => {
    const fetchFn = createFetchQueue(
        streamResponse(['hi'], { 'X-Request-ID': 'stream-1' }),
    );
    const { client } = createClient({
        fetchFn,
        sessionInitial: { kestrel_api_key: 'k-secret' },
    });

    await client.init();

    const chunks = [];
    for await (const chunk of client.streamInvoke('hello')) {
        chunks.push(chunk);
    }

    assert.deepEqual(chunks, ['hi']);
    assert.deepEqual(fetchFn.calls.map((call) => call.url), ['/api/agent/stream']);
});

test('streamInvoke with API key refreshes bootstrap key once and retries the stream', async () => {
    const fetchFn = createFetchQueue(
        jsonResponse(401, { detail: 'expired stream key' }),
        jsonResponse(200, { key: 'fresh-key' }),
        streamResponse(['hel', 'lo'], { 'X-Request-ID': 'stream-1' }),
    );
    const { client, sessionStorage } = createClient({
        fetchFn,
        sessionInitial: { kestrel_api_key: 'stale-key' },
    });

    await client.init();

    const chunks = [];
    for await (const chunk of client.streamInvoke('hello')) {
        chunks.push(chunk);
    }

    assert.deepEqual(chunks, ['hel', 'lo']);
    assert.equal(sessionStorage.getItem('kestrel_api_key'), 'fresh-key');
    assert.deepEqual(fetchFn.calls.map((call) => call.url), [
        '/api/agent/stream',
        '/api/auth/key',
        '/api/agent/stream',
    ]);
    assert.equal(fetchFn.calls[0].options.headers['X-API-Key'], 'stale-key');
    assert.equal(fetchFn.calls[2].options.headers['X-API-Key'], 'fresh-key');
});

test('applyHostAgentPrefix preserves host-level routes and prefixes per-agent routes in rookery mode', () => {
    assert.equal(applyHostAgentPrefix('/api/auth/key', 'host-a'), '/api/auth/key');
    assert.equal(
        applyHostAgentPrefix('/api/models?featured_only=true', 'host-a'),
        '/api/agents/host-a/api/models?featured_only=true',
    );
    assert.equal(applyHostAgentPrefix('/api/agent/invoke', null), '/api/agent/invoke');
});

test('isHostLevelEndpoint identifies fleet-wide routes that must not be wrapped', () => {
    assert.equal(isHostLevelEndpoint('/api/agents'), true);
    assert.equal(isHostLevelEndpoint('/api/agents?owner=me'), true);
    assert.equal(isHostLevelEndpoint('/api/agents/claw/start'), true);
    assert.equal(isHostLevelEndpoint('/api/auth/key'), true);
    assert.equal(isHostLevelEndpoint('/health'), true);
    assert.equal(isHostLevelEndpoint('/api/identity'), false);
    assert.equal(isHostLevelEndpoint('/api/agent/invoke'), false);
});

test('buildAgentUrl maps notification SSE paths through selected host agents', () => {
    const { client } = createClient({ fetchFn: createFetchQueue() });

    client.setHostAgent('claw');

    assert.equal(
        client.buildAgentUrl('/api/agent/notifications/sse'),
        '/api/agents/claw/api/agent/notifications/sse',
    );
});

test('applyAuth signs arbitrary header objects via the active provider (#863 direct-fetch seam)', async () => {
    // Code that has to call fetch() directly (FormData uploads, model selector
    // commits, voice realtime mint) uses applyAuth() to honor whichever auth
    // provider is active. Standalone provider with API key:
    const standalone = createClient({
        fetchFn: createFetchQueue(),
        sessionInitial: { kestrel_api_key: 'k-test' },
    });
    await standalone.client.init();
    const standaloneHeaders = await standalone.client.applyAuth({ 'X-Custom': '1' });
    assert.equal(standaloneHeaders['X-API-Key'], 'k-test');
    assert.equal(standaloneHeaders['X-Custom'], '1');

    // BearerToken provider:
    const bearer = createClient({
        fetchFn: createFetchQueue(),
        authProvider: createBearerTokenAuthProvider({ getToken: () => 'jwt-99' }),
    });
    const bearerHeaders = await bearer.client.applyAuth({ 'Content-Type': 'application/json' });
    assert.equal(bearerHeaders.Authorization, 'Bearer jwt-99');
    assert.equal(bearerHeaders['Content-Type'], 'application/json');
    assert.equal(bearerHeaders['X-API-Key'], undefined);
});

test('BearerTokenAuthProvider attaches Authorization header and skips bootstrap fetches (#863 embed contract)', async () => {
    // Embedded-host path: the host already authenticated the user and
    // injects a JWT via getToken(). Kestrel UI must not call /api/auth/key
    // or /auth/me — both would 404 in a host that doesn't expose them.
    const fetchFn = createFetchQueue(jsonResponse(200, { models: ['ok'] }));
    let tokenCalls = 0;
    const provider = createBearerTokenAuthProvider({
        getToken: () => {
            tokenCalls += 1;
            return 'jwt-from-host';
        },
    });
    const { client } = createClient({ fetchFn, authProvider: provider });

    await client.init();
    const result = await client.request('/api/models');

    assert.deepEqual(result, { models: ['ok'] });
    assert.deepEqual(fetchFn.calls.map((call) => call.url), ['/api/models']);
    assert.equal(fetchFn.calls[0].options.headers.Authorization, 'Bearer jwt-from-host');
    assert.ok(tokenCalls >= 1);
});

test('BearerTokenAuthProvider invokes onUnauthenticated when no token is available', async () => {
    let unauthCount = 0;
    const provider = createBearerTokenAuthProvider({
        getToken: () => null,
        onUnauthenticated: () => {
            unauthCount += 1;
        },
    });
    const { client } = createClient({ fetchFn: createFetchQueue(), authProvider: provider });

    await assert.rejects(() => client.init(), /Bearer token unavailable/);
    assert.equal(unauthCount, 1);
});

test('BearerTokenAuthProvider 401 path delegates to onUnauthenticated and stops the request', async () => {
    let redirected = false;
    const provider = createBearerTokenAuthProvider({
        getToken: () => 'expired-jwt',
        onUnauthenticated: () => {
            redirected = true;
        },
    });
    const fetchFn = createFetchQueue(jsonResponse(401, { detail: 'expired' }));
    const { client } = createClient({ fetchFn, authProvider: provider });

    await client.init();
    const result = await client.request('/api/models');

    assert.equal(result, undefined);
    assert.equal(redirected, true);
});
