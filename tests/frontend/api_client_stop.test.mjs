import test from 'node:test';
import assert from 'node:assert/strict';

import { createApiClient } from '../../kestrel_sovereign/static/js/api_client.mjs';

// API.stop(requestId, agent) must hit the AGENT-PREFIXED endpoint
// regardless of which agent is currently selected. Without this,
// clicking "Stop Agent A" while viewing Agent B would (a) abort A's
// stream client-side via the per-agent AbortController, but (b) hit
// /api/agents/B/agent/stop server-side — telling the wrong backend
// to halt while A continued chewing tokens.

function makeAuthProvider() {
    return {
        async ensureAuthenticated() {},
        applyAuth: async (h) => ({ ...h, 'X-API-Key': 'k' }),
        async onUnauthorized() { return 'failed'; },
    };
}

function fakeFetch(captured) {
    return async (url, opts) => {
        captured.push({ url, opts });
        return {
            ok: true,
            status: 200,
            json: async () => ({}),
            headers: { get: () => null },
        };
    };
}

test('stop(id, agentName) routes to /api/agents/<agent>/agent/stop regardless of selected agent', async () => {
    const calls = [];
    const client = createApiClient({
        fetchFn: fakeFetch(calls),
        sessionStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
        location: { href: '/', search: '' },
        AbortControllerCtor: globalThis.AbortController || class { abort(){} signal={} },
        TextDecoderCtor: globalThis.TextDecoder || class { decode(){return '';} },
        authProvider: makeAuthProvider(),
    });

    client.setHostAgent('viewing-agent-b');

    await client.stop('req-123', 'agent-a');

    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, '/api/agents/agent-a/api/agent/stop',
        'stop(id, agent) must address the explicit agent, not state.selectedHostAgent');
    assert.equal(calls[0].opts.method, 'POST');
});

test('stop(id) with no agent arg routes through the currently-selected agent (today\'s behavior)', async () => {
    const calls = [];
    const client = createApiClient({
        fetchFn: fakeFetch(calls),
        sessionStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
        location: { href: '/', search: '' },
        AbortControllerCtor: globalThis.AbortController || class { abort(){} signal={} },
        TextDecoderCtor: globalThis.TextDecoder || class { decode(){return '';} },
        authProvider: makeAuthProvider(),
    });

    client.setHostAgent('selected-agent');

    await client.stop('req-456');

    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, '/api/agents/selected-agent/api/agent/stop',
        'no-arg stop must continue to honor state.selectedHostAgent');
});

test('stop(id, null) targets standalone (un-prefixed) endpoint', async () => {
    // null is the standalone-mode key — applyHostAgentPrefix returns
    // the endpoint untouched when selectedHostAgent is null. The same
    // contract must hold for the explicit-agent path.
    const calls = [];
    const client = createApiClient({
        fetchFn: fakeFetch(calls),
        sessionStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
        location: { href: '/', search: '' },
        AbortControllerCtor: globalThis.AbortController || class { abort(){} signal={} },
        TextDecoderCtor: globalThis.TextDecoder || class { decode(){return '';} },
        authProvider: makeAuthProvider(),
    });

    client.setHostAgent('an-agent');

    await client.stop('req-789', null);

    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, '/api/agent/stop',
        'stop(id, null) must NOT prefix — null is the standalone-mode signal');
});
