/**
 * Trash UI API client tests (#765).
 *
 * Verifies the new soft-delete recovery methods on the API client:
 *   listTrash, restoreConversation, purgeConversation,
 *   restoreMessage, purgeMessage
 *
 * These methods exercise the endpoints introduced by #763 — the wire
 * format and HTTP verbs are part of the recovery contract, so a thin
 * unit test here catches accidental regressions before they ship to
 * the browser.
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import { createApiClient } from '../../kestrel_sovereign/static/js/api_client.mjs';

function createStorage(initial = {}) {
    const store = new Map(Object.entries(initial));
    return {
        getItem: (k) => (store.has(k) ? store.get(k) : null),
        setItem: (k, v) => store.set(k, String(v)),
        removeItem: (k) => store.delete(k),
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

function createFetchQueue(...responses) {
    const calls = [];
    const fetchFn = async (url, options = {}) => {
        calls.push({ url, options });
        const next = responses.shift();
        if (!next) throw new Error(`Unexpected fetch to ${url}`);
        return typeof next === 'function' ? next(url, options) : next;
    };
    fetchFn.calls = calls;
    return fetchFn;
}

function makeClient({ fetchFn }) {
    const localStorage = createStorage();
    const sessionStorage = createStorage({ kestrel_api_key: 'k-test' });
    return createApiClient({
        fetchFn,
        localStorage,
        sessionStorage,
        location: { href: '/console' },
        logger: console,
    });
}


test('listTrash GETs /api/trash with the configured limit', async () => {
    const fetchFn = createFetchQueue(jsonResponse(200, {
        messages: [
            { id: 1, role: 'user', content: 'hi', deleted_at: '2026-04-25T10:00:00', metadata: {} },
        ],
        total: 1,
    }));
    const client = makeClient({ fetchFn });
    await client.init();

    const result = await client.listTrash(123);

    assert.equal(fetchFn.calls.length, 1);
    assert.equal(fetchFn.calls[0].url, '/api/trash?limit=123');
    assert.equal(fetchFn.calls[0].options.method ?? 'GET', 'GET');
    assert.equal(result.total, 1);
    assert.equal(result.messages[0].id, 1);
});


test('listTrash defaults to limit=200 when none provided', async () => {
    const fetchFn = createFetchQueue(jsonResponse(200, { messages: [], total: 0 }));
    const client = makeClient({ fetchFn });
    await client.init();

    await client.listTrash();
    assert.equal(fetchFn.calls[0].url, '/api/trash?limit=200');
});


test('restoreConversation POSTs to /restore with the URL-encoded session id', async () => {
    const fetchFn = createFetchQueue(jsonResponse(200, {
        success: true,
        session_id: 'sess uid',
        restored_count: 4,
    }));
    const client = makeClient({ fetchFn });
    await client.init();

    const result = await client.restoreConversation('sess uid');

    assert.equal(fetchFn.calls[0].url, '/api/conversations/sess%20uid/restore');
    assert.equal(fetchFn.calls[0].options.method, 'POST');
    assert.equal(result.restored_count, 4);
});


test('purgeConversation POSTs JSON body with the reason', async () => {
    const fetchFn = createFetchQueue(jsonResponse(200, {
        success: true,
        session_id: 'sess-1',
        purged_count: 7,
        reason: 'user-initiated-ui',
    }));
    const client = makeClient({ fetchFn });
    await client.init();

    await client.purgeConversation('sess-1', 'user-initiated-ui');

    const call = fetchFn.calls[0];
    assert.equal(call.url, '/api/conversations/sess-1/purge');
    assert.equal(call.options.method, 'POST');
    assert.equal(call.options.headers['Content-Type'], 'application/json');
    const body = JSON.parse(call.options.body);
    assert.equal(body.reason, 'user-initiated-ui');
});


test('purgeConversation defaults reason to user-initiated when omitted', async () => {
    const fetchFn = createFetchQueue(jsonResponse(200, { success: true, purged_count: 1 }));
    const client = makeClient({ fetchFn });
    await client.init();

    await client.purgeConversation('sess-2');

    const body = JSON.parse(fetchFn.calls[0].options.body);
    assert.equal(body.reason, 'user-initiated');
});


test('restoreMessage POSTs to /restore with the message id encoded', async () => {
    const fetchFn = createFetchQueue(jsonResponse(200, { success: true, message_id: 42 }));
    const client = makeClient({ fetchFn });
    await client.init();

    await client.restoreMessage(42);

    assert.equal(fetchFn.calls[0].url, '/api/conversations/messages/42/restore');
    assert.equal(fetchFn.calls[0].options.method, 'POST');
});


test('purgeMessage POSTs JSON body with the reason', async () => {
    const fetchFn = createFetchQueue(jsonResponse(200, {
        success: true, message_id: 7, reason: 'gdpr',
    }));
    const client = makeClient({ fetchFn });
    await client.init();

    await client.purgeMessage(7, 'gdpr');

    const call = fetchFn.calls[0];
    assert.equal(call.url, '/api/conversations/messages/7/purge');
    assert.equal(call.options.method, 'POST');
    assert.equal(call.options.headers['Content-Type'], 'application/json');
    const body = JSON.parse(call.options.body);
    assert.equal(body.reason, 'gdpr');
});


test('all trash methods carry the X-API-Key auth header', async () => {
    // Each call is a separate fetch — load up a queue large enough.
    const fetchFn = createFetchQueue(
        jsonResponse(200, { messages: [], total: 0 }),
        jsonResponse(200, { success: true, restored_count: 0 }),
        jsonResponse(200, { success: true, purged_count: 0 }),
        jsonResponse(200, { success: true, message_id: 1 }),
        jsonResponse(200, { success: true, message_id: 1 }),
    );
    const client = makeClient({ fetchFn });
    await client.init();

    await client.listTrash();
    await client.restoreConversation('s');
    await client.purgeConversation('s');
    await client.restoreMessage(1);
    await client.purgeMessage(1);

    for (const call of fetchFn.calls) {
        assert.equal(call.options.headers['X-API-Key'], 'k-test',
            `${call.url} did not carry the API key header`);
    }
});
