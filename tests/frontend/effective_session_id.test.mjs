import test from 'node:test';
import assert from 'node:assert/strict';

import { createApiClient } from '../../kestrel_sovereign/static/js/api_client.mjs';

// The api_client must surface the server's resolved session_id so
// chat.js can adopt it onto pane.sessionId. Two paths:
//   - streamInvoke captures X-Session-Id from the response headers
//     before yielding the first body chunk.
//   - invoke / invokeForAgent capture session_id from the JSON body.
//
// In both cases, the captured value is keyed by the dispatch agent
// (NOT state.selectedHostAgent at the time the response landed) so a
// switch-mid-flight doesn't write Agent A's session_id onto Agent B.

class StubAbort { abort() {} signal = {}; }
class StubDecoder { decode(buf) { return Buffer.from(buf || []).toString(); } }

function authProvider() {
    return {
        async ensureAuthenticated() {},
        applyAuth: async (h) => ({ ...h, 'X-API-Key': 'k' }),
        async onUnauthorized() { return 'failed'; },
    };
}

class FakeReader {
    constructor(chunks = []) { this._c = chunks; }
    async read() {
        if (!this._c.length) return { done: true, value: undefined };
        return { done: false, value: this._c.shift() };
    }
    releaseLock() {}
}

function streamResp({ sessionId = null, body = ['hi'] } = {}) {
    const headers = new Map();
    if (sessionId) headers.set('X-Session-Id', sessionId);
    return {
        ok: true, status: 200,
        headers: { get: (k) => headers.get(k) || null },
        body: { getReader: () => new FakeReader(body.map((s) => new Uint8Array(Buffer.from(s)))) },
    };
}

function jsonResp(payload) {
    return {
        ok: true, status: 200,
        headers: { get: () => null },
        json: async () => payload,
    };
}

function newClient(fetchImpl) {
    return createApiClient({
        fetchFn: fetchImpl,
        sessionStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
        location: { href: '/', search: '' },
        AbortControllerCtor: StubAbort,
        TextDecoderCtor: StubDecoder,
        authProvider: authProvider(),
    });
}

test('streamInvoke captures X-Session-Id header before the body streams', async () => {
    let capturedHeaderRead = null;
    const fetchImpl = async () => streamResp({ sessionId: 'sess-from-server-1' });
    const client = newClient(fetchImpl);

    client.setHostAgent('A');
    // Drain the iterator.
    const it = client.streamInvoke('hi', null, null, null, false, 'A');
    for await (const _ of it) { /* noop */ }

    assert.equal(client.getEffectiveSessionId('A'), 'sess-from-server-1',
        'session_id from X-Session-Id header must be exposed via getEffectiveSessionId');
});

test('streamInvoke writes session_id under the DISPATCH agent, not selectedHostAgent', async () => {
    // The user dispatched a chat to Agent A and then switched to B
    // before the response landed. The session_id belongs to A's
    // conversation; writing it onto B would corrupt B's pane.
    const fetchImpl = async () => streamResp({ sessionId: 'A-sess' });
    const client = newClient(fetchImpl);

    client.setHostAgent('A');
    const it = client.streamInvoke('hi', null, null, null, false, 'A');
    client.setHostAgent('B');  // user switched
    for await (const _ of it) { /* noop */ }

    assert.equal(client.getEffectiveSessionId('A'), 'A-sess',
        "A must own the session_id");
    assert.equal(client.getEffectiveSessionId('B'), null,
        "B must NOT have inherited A's session_id");
});

test('streamInvoke without X-Session-Id header leaves the map unchanged (no clobber)', async () => {
    // Older servers / EPHEMERAL mode may omit the header. The map
    // value must persist from any prior turn that DID populate it.
    const calls = [];
    let i = 0;
    const fetchImpl = async () => {
        calls.push(i);
        return i++ === 0
            ? streamResp({ sessionId: 'first-sess' })
            : streamResp({ sessionId: null });  // no header
    };
    const client = newClient(fetchImpl);

    client.setHostAgent('A');

    let it = client.streamInvoke('first', null, null, null, false, 'A');
    for await (const _ of it) { /* noop */ }
    assert.equal(client.getEffectiveSessionId('A'), 'first-sess');

    it = client.streamInvoke('second', null, null, null, false, 'A');
    for await (const _ of it) { /* noop */ }
    assert.equal(client.getEffectiveSessionId('A'), 'first-sess',
        'absent header on the second turn must not wipe the value learned on the first');
});

test('invoke captures session_id from JSON response body', async () => {
    const fetchImpl = async () => jsonResp({ response: 'r', session_id: 'json-sess-1' });
    const client = newClient(fetchImpl);

    client.setHostAgent('A');
    const result = await client.invoke('hi');

    assert.equal(result.session_id, 'json-sess-1');
    assert.equal(client.getEffectiveSessionId('A'), 'json-sess-1');
});

test('invokeForAgent with explicit agent writes session_id under that agent', async () => {
    const fetchImpl = async () => jsonResp({ response: 'r', session_id: 'forA-sess' });
    const client = newClient(fetchImpl);

    client.setHostAgent('B');  // viewing B
    await client.invokeForAgent('hi', null, null, null, 'A');  // dispatch to A

    assert.equal(client.getEffectiveSessionId('A'), 'forA-sess',
        'invokeForAgent must record session_id under the dispatch agent');
    assert.equal(client.getEffectiveSessionId('B'), null,
        "B must NOT have inherited A's session_id");
});

test('getEffectiveSessionId returns null for unknown agent', () => {
    const fetchImpl = async () => jsonResp({});
    const client = newClient(fetchImpl);
    assert.equal(client.getEffectiveSessionId('never-seen'), null);
});
