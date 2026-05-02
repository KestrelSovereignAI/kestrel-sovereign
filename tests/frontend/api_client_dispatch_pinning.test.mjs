import test from 'node:test';
import assert from 'node:assert/strict';

import { createApiClient } from '../../kestrel_sovereign/static/js/api_client.mjs';

// Three routing-pin invariants that prevent the wrong-agent class of
// bug after a chat is dispatched:
//
//   1. streamInvoke with explicit `agent` arg pins the URL even when
//      state.selectedHostAgent changes between dispatch and fetch.
//   2. A 401 retry inside streamInvoke must reuse the original
//      dispatch agent — recapturing state.selectedHostAgent at retry
//      time was the original bug (auth refresh + agent switch =
//      retry hits the wrong backend).
//   3. invokeForAgent pins the non-streaming URL to the explicit
//      agent; the unprefixed invoke() still routes via the currently
//      selected agent (existing behavior preserved).

function makeAuthProvider({ on401 = 'failed' } = {}) {
    let attempts = 0;
    return {
        async ensureAuthenticated() {},
        applyAuth: async (h) => ({ ...h, 'X-API-Key': `k${attempts}` }),
        async onUnauthorized() { attempts++; return on401; },
    };
}

class FakeReader {
    constructor(chunks = []) { this._chunks = chunks; }
    async read() {
        if (this._chunks.length === 0) return { done: true, value: undefined };
        return { done: false, value: this._chunks.shift() };
    }
    releaseLock() {}
}

function fakeFetchSequence(responses, captured) {
    let i = 0;
    return async (url, opts) => {
        captured.push({ url, opts });
        const r = responses[i++];
        if (typeof r === 'function') return r();
        return r;
    };
}

function okStreamResponse(body = []) {
    return {
        ok: true,
        status: 200,
        headers: { get: () => 'req-1' },
        body: {
            getReader: () => new FakeReader(body.map((s) => new Uint8Array(Buffer.from(s)))),
        },
    };
}

class StubAbort { abort() {} signal = {}; }
class StubDecoder { decode(buf) { return Buffer.from(buf).toString(); } }

test('streamInvoke pins the URL to the explicit `agent` arg even after state.selectedHostAgent changes', async () => {
    const calls = [];
    const client = createApiClient({
        fetchFn: fakeFetchSequence([okStreamResponse(['hello'])], calls),
        sessionStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
        location: { href: '/', search: '' },
        AbortControllerCtor: StubAbort,
        TextDecoderCtor: StubDecoder,
        authProvider: makeAuthProvider(),
    });

    client.setHostAgent('selected-B');

    // Dispatch was for Agent A; pass it explicitly. Then simulate the
    // user switching to B before the fetch lands.
    const it = client.streamInvoke('hi', null, null, null, false, 'dispatch-A');

    // Move state.selectedHostAgent to B AFTER the call but before
    // iteration consumes the URL build (build happens at first await).
    client.setHostAgent('B-now-selected');

    // Drain.
    for await (const _ of it) { /* noop */ }

    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, '/api/agents/dispatch-A/api/agent/stream',
        'URL must be pinned to the explicit dispatch agent, not state.selectedHostAgent');
});

test('streamInvoke 401 retry must reuse the original dispatch agent', async () => {
    const calls = [];
    const r401 = { ok: false, status: 401, json: async () => ({ detail: 'auth' }), headers: { get: () => null } };
    const client = createApiClient({
        fetchFn: fakeFetchSequence([r401, okStreamResponse(['ok'])], calls),
        sessionStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
        location: { href: '/', search: '' },
        AbortControllerCtor: StubAbort,
        TextDecoderCtor: StubDecoder,
        authProvider: makeAuthProvider({ on401: 'refreshed' }),
    });

    client.setHostAgent('Alice');

    const it = client.streamInvoke('hi', null, null, null, false, 'Alice');
    // User switches DURING the auth refresh — between the 401 and the
    // retry fetch. The retry MUST still hit Alice's stream, not Bob's.
    setTimeout(() => client.setHostAgent('Bob'), 0);

    for await (const _ of it) { /* noop */ }

    assert.equal(calls.length, 2, 'first 401 then retry');
    assert.equal(calls[0].url, '/api/agents/Alice/api/agent/stream', 'first call hits Alice');
    assert.equal(calls[1].url, '/api/agents/Alice/api/agent/stream',
        'retry must reuse the captured dispatch agent — NOT recapture state.selectedHostAgent');
});

test('streamInvoke without explicit agent still defaults to state.selectedHostAgent (no regression)', async () => {
    const calls = [];
    const client = createApiClient({
        fetchFn: fakeFetchSequence([okStreamResponse(['x'])], calls),
        sessionStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
        location: { href: '/', search: '' },
        AbortControllerCtor: StubAbort,
        TextDecoderCtor: StubDecoder,
        authProvider: makeAuthProvider(),
    });

    client.setHostAgent('default-target');

    const it = client.streamInvoke('hi');
    for await (const _ of it) { /* noop */ }

    assert.equal(calls[0].url, '/api/agents/default-target/api/agent/stream');
});

test('invokeForAgent pins the non-streaming POST to the explicit agent', async () => {
    const calls = [];
    const client = createApiClient({
        fetchFn: fakeFetchSequence([{
            ok: true, status: 200, json: async () => ({ response: 'r' }), headers: { get: () => null },
        }], calls),
        sessionStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
        location: { href: '/', search: '' },
        AbortControllerCtor: StubAbort,
        TextDecoderCtor: StubDecoder,
        authProvider: makeAuthProvider(),
    });

    client.setHostAgent('viewing-B');

    await client.invokeForAgent('hi', null, null, null, 'dispatch-A');

    assert.equal(calls[0].url, '/api/agents/dispatch-A/api/agent/invoke',
        'invokeForAgent must address the explicit agent regardless of selectedHostAgent');
});

test('invokeForAgent without explicit agent falls back to current selected (preserves invoke() behavior)', async () => {
    const calls = [];
    const client = createApiClient({
        fetchFn: fakeFetchSequence([{
            ok: true, status: 200, json: async () => ({ response: 'r' }), headers: { get: () => null },
        }], calls),
        sessionStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
        location: { href: '/', search: '' },
        AbortControllerCtor: StubAbort,
        TextDecoderCtor: StubDecoder,
        authProvider: makeAuthProvider(),
    });

    client.setHostAgent('selected');

    await client.invokeForAgent('hi');

    assert.equal(calls[0].url, '/api/agents/selected/api/agent/invoke');
});

test('streamInvoke propagates AbortError instead of swallowing it (sendMessage relies on this)', async () => {
    const calls = [];
    const client = createApiClient({
        fetchFn: async () => {
            const e = new Error('aborted'); e.name = 'AbortError'; throw e;
        },
        sessionStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
        location: { href: '/', search: '' },
        AbortControllerCtor: StubAbort,
        TextDecoderCtor: StubDecoder,
        authProvider: makeAuthProvider(),
    });
    client.setHostAgent('A');

    const it = client.streamInvoke('hi', null, null, null, false, 'A');
    let caught = null;
    try {
        for await (const _ of it) { /* noop */ }
    } catch (e) { caught = e; }

    assert.ok(caught, 'AbortError must propagate to the consumer');
    assert.equal(caught.name, 'AbortError');
});
