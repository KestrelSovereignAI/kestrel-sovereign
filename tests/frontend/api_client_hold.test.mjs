import test from 'node:test';
import assert from 'node:assert/strict';

import { createApiClient } from '../../kestrel_sovereign/static/js/api_client.mjs';

function makeAuthProvider() {
    return {
        async ensureAuthenticated() {},
        applyAuth: async (headers) => ({ ...headers, 'X-API-Key': 'k' }),
        async onUnauthorized() { return 'failed'; },
    };
}

function clientWithCalls(calls) {
    return createApiClient({
        fetchFn: async (url, opts) => {
            calls.push({ url, opts });
            return {
                ok: true,
                status: 200,
                json: async () => ({}),
                headers: { get: () => null },
            };
        },
        sessionStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
        location: { href: '/', search: '' },
        AbortControllerCtor: globalThis.AbortController || class { abort(){} signal={} },
        TextDecoderCtor: globalThis.TextDecoder || class { decode(){return '';} },
        authProvider: makeAuthProvider(),
    });
}

test('holdAgent addresses the host control door and carries exact agent identity', async () => {
    const calls = [];
    const client = clientWithCalls(calls);
    client.setHostAgent('currently-selected-other-agent');

    await client.holdAgent('Emma', 'did:test:emma', 'operator pause', 'hold-op-1');

    const mutation = calls.at(-1);
    assert.equal(mutation.url, '/api/host/holds/agents/Emma');
    assert.deepEqual(JSON.parse(mutation.opts.body), {
        target_agent_id: 'did:test:emma',
        reason: 'operator pause',
        operation_id: 'hold-op-1',
    });
});

test('resumeAgentHold sends the exact observed latch receipt', async () => {
    const calls = [];
    const client = clientWithCalls(calls);

    await client.resumeAgentHold(
        'Emma',
        'did:test:emma',
        'hold-receipt-1',
        'operator resume',
        'resume-op-1',
    );

    const mutation = calls.at(-1);
    assert.equal(mutation.url, '/api/host/holds/agents/Emma/release');
    assert.deepEqual(JSON.parse(mutation.opts.body), {
        target_agent_id: 'did:test:emma',
        expected_hold_receipt_id: 'hold-receipt-1',
        reason: 'operator resume',
        operation_id: 'resume-op-1',
    });
});
