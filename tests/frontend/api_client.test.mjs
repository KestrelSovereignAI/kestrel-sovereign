import test from 'node:test';
import assert from 'node:assert/strict';

import { createApiClient, rewriteEndpoint } from '../../static/js/api_client.mjs';

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

function createClient({ fetchFn, localInitial = {}, sessionInitial = {} } = {}) {
    const logger = createLogger();
    const location = { href: '/console' };
    const localStorage = createStorage(localInitial);
    const sessionStorage = createStorage(sessionInitial);
    const client = createApiClient({
        fetchFn,
        localStorage,
        sessionStorage,
        location,
        logger,
    });
    return { client, logger, location, localStorage, sessionStorage };
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

test('request with JWT clears stored tokens and redirects instead of bootstrapping API key', async () => {
    const fetchFn = createFetchQueue(jsonResponse(401, { detail: 'expired' }));
    const { client, location, localStorage, sessionStorage } = createClient({
        fetchFn,
        localInitial: { platform_auth_token: 'jwt-123' },
        sessionInitial: { token: 'fallback-jwt' },
    });

    client.setAgentId('did:test:companion');
    await client.init();
    await client.request('/api/models');

    assert.equal(location.href, '/auth/login');
    assert.equal(localStorage.getItem('platform_auth_token'), null);
    assert.equal(sessionStorage.getItem('token'), null);
    assert.deepEqual(fetchFn.calls.map((call) => call.url), ['/api/kestrel/companions/did%3Atest%3Acompanion/models']);
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

test('rewriteEndpoint preserves host-level auth routes and rewrites per-agent routes in rookery mode', () => {
    assert.equal(
        rewriteEndpoint('/api/auth/key', { currentAgentId: 'did:test', selectedHostAgent: 'host-a' }),
        '/api/auth/key',
    );
    assert.equal(
        rewriteEndpoint('/api/models?featured_only=true', { selectedHostAgent: 'host-a' }),
        '/api/agents/host-a/api/models?featured_only=true',
    );
});

test('rewriteEndpoint maps standalone endpoints into companion routes in multi-agent mode', () => {
    assert.equal(
        rewriteEndpoint('/agent/invoke', { currentAgentId: 'did:test:agent' }),
        '/api/kestrel/companions/did%3Atest%3Aagent/invoke',
    );
    assert.equal(
        rewriteEndpoint('/api/conversations/session-1', { currentAgentId: 'did:test:agent' }),
        '/api/kestrel/companions/did%3Atest%3Aagent/conversations/session-1',
    );
});
