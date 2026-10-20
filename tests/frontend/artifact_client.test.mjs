import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { ArtifactDraft, createArtifactClient, publicationDigests, sha256 } from '../../kestrel_sovereign/static/js/artifact_client.mjs';
import { createApiClient, createKestrelStandaloneAuthProvider } from '../../kestrel_sovereign/static/js/api_client.mjs';

const scope = { schema_version: 1, tenant_id: 'tenant', principal_id: 'person', agent_id: 'agent' };
const ref = { schema_version: 1, artifact_id: 'artifact', revision_id: 'revision1' };
async function fixture(content = '# Original') {
    const bytes = new TextEncoder().encode(content);
    const file = { schema_version: 1, path: 'notes.md', media_type: 'text/markdown', size_bytes: bytes.length, sha256: await sha256(bytes) };
    const revision = { schema_version: 1, ref, scope, title: 'Notes', parent_revision_id: null,
        manifest: { schema_version: 1, entrypoint: file.path, files: [file] },
        provenance: { schema_version: 1, session_id: null, run_id: null, tool_name: null }, committed_at: '2026-10-20T12:00:00Z' };
    const staged = [], committed = [], statuses = [];
    const client = {
        get: async () => structuredClone(revision), read: async () => bytes,
        stage: async (key, file, content) => { staged.push({ key, file: structuredClone(file), content: Uint8Array.from(content) }); return file; },
        commit: async request => {
            committed.push(structuredClone(request));
            return { schema_version: 1, ref: { ...ref, revision_id: 'revision2' }, scope,
                idempotency_key: request.idempotency_key, ...await publicationDigests(request),
                committed_at: '2026-10-20T12:01:00Z', storage_id: 'local' };
        },
        status: async key => { statuses.push(key); return null; },
        history: async () => ({ items: [revision], next_cursor: null }),
    };
    const draft = new ArtifactDraft(client); await draft.open(ref);
    return { draft, client, revision, bytes, staged, committed, statuses };
}

test('publication digest is interoperable with independently generated SDK 0.41 fixture', async () => {
    const fixture = JSON.parse(await readFile(new URL('../fixtures/artifacts/publication-digest.json', import.meta.url)));
    assert.deepEqual(await publicationDigests(fixture.request), {
        manifest_sha256: fixture.manifest_sha256, request_sha256: fixture.request_sha256,
    });
});

test('saved state requires a matching durable receipt and advances exact revision', async () => {
    const { draft, staged, committed } = await fixture();
    assert.equal(draft.state, 'saved'); draft.edit('# Edited');
    await draft.save();
    assert.equal(draft.state, 'saved'); assert.equal(draft.revision.ref.revision_id, 'revision2');
    assert.equal(draft.revision.parent_revision_id, 'revision1');
    assert.equal(new TextDecoder().decode(draft.bytes), '# Edited');
    assert.equal(staged[0].key, committed[0].idempotency_key);
    assert.equal(committed[0].expected_revision_id, 'revision1'); draft.close();
});

test('lost commit acknowledgement reconciles same key without duplicate publication', async () => {
    const { draft, client, committed } = await fixture();
    const commit = client.commit; let receipt;
    client.commit = async request => { receipt = await commit(request); throw new TypeError('network disconnected'); };
    draft.edit('unsaved draft'); await draft.save();
    assert.equal(draft.state, 'unknown'); assert.equal(draft.text, 'unsaved draft');
    const key = draft.pendingKey;
    client.status = async candidate => { assert.equal(candidate, key); return receipt; };
    await draft.retry();
    assert.equal(draft.state, 'saved'); assert.equal(committed.length, 1); draft.close();
});

test('null status retains pending content and retries only identical commit', async () => {
    const { draft, client, committed, staged } = await fixture();
    const commit = client.commit;
    client.commit = async request => { committed.push(structuredClone(request)); throw new Error('lost'); };
    draft.edit('keep me'); await draft.save(); const key = draft.pendingKey;
    await draft.reconcile(); assert.equal(draft.state, 'unknown'); assert.equal(draft.pendingKey, key);
    assert.equal(draft.edit('replacement'), false);
    client.commit = commit; await draft.retry();
    assert.deepEqual(committed[0], committed[1]); assert.equal(staged.length, 1); draft.close();
});

test('missing staging is restaged with original bytes and key, not reported erased', async () => {
    const { draft, client, staged } = await fixture();
    const commit = client.commit; let first = true;
    client.commit = async request => { if (first) { first = false; throw { status: 404 }; } return commit(request); };
    draft.edit('retained'); await draft.save(); assert.equal(draft.state, 'unknown');
    const key = draft.pendingKey; await draft.retry();
    assert.equal(draft.state, 'saved'); assert.equal(staged.length, 2);
    assert.equal(staged[1].key, key); assert.deepEqual(staged[0].content, staged[1].content); draft.close();
});

test('erased publication is terminal and cannot automatically republish', async () => {
    const { draft, client, staged } = await fixture();
    client.commit = async () => { throw new Error('lost'); };
    draft.edit('keep'); await draft.save();
    client.status = async () => { throw { status: 404, code: 'artifact_not_found' }; };
    await draft.retry(); assert.equal(draft.state, 'erased');
    await draft.retry(); await draft.save(); assert.equal(staged.length, 1); draft.close();
});

test('a mismatched receipt never marks an edit saved', async () => {
    for (const change of [{ scope: { ...scope, principal_id: 'other' } }, { request_sha256: '0'.repeat(64) },
        { manifest_sha256: '0'.repeat(64) }, { idempotency_key: 'another-save' }, { schema_version: 2 },
        { ref: { ...ref, artifact_id: 'other', revision_id: 'revision2' } }, { ref }]) {
        const { draft, client } = await fixture(); const commit = client.commit;
        client.commit = async request => ({ ...await commit(request), ...change });
        draft.edit('private'); await draft.save();
        assert.equal(draft.state, 'unknown'); assert.equal(draft.revision.ref.revision_id, 'revision1'); draft.close();
    }
});

test('conflicting edit is preserved and requires explicit comparison before rebasing', async () => {
    const { draft, client, revision, committed } = await fixture();
    const commit = client.commit; client.commit = async () => { throw { status: 409 }; };
    draft.edit('my draft'); await draft.save(); assert.equal(draft.state, 'conflict');
    assert.equal(draft.useComparedRevision(), false);
    const latestBytes = new TextEncoder().encode('someone else');
    const latestFile = { ...revision.manifest.files[0], sha256: await sha256(latestBytes), size_bytes: latestBytes.length };
    const latest = { ...revision, ref: { ...ref, revision_id: 'concurrent' }, manifest: { ...revision.manifest, files: [latestFile] } };
    client.get = async () => latest; client.read = async () => latestBytes;
    client.history = async () => ({ items: [latest] });
    await draft.compareLatest(); assert.equal(draft.comparison.text, 'someone else');
    assert.equal(draft.text, 'my draft'); const key = draft.pendingKey;
    assert.equal(draft.useComparedRevision(), true);
    assert.equal(new TextDecoder().decode(draft.bytes), 'someone else');
    client.commit = commit; await draft.save();
    assert.equal(committed[0].expected_revision_id, 'concurrent');
    assert.notEqual(committed[0].idempotency_key, key); draft.close();
});

test('a conflict after an unknown attempt cannot authorize a new save key', async () => {
    const { draft, client } = await fixture();
    client.commit = async () => { throw new Error('unknown'); };
    draft.edit('keep'); await draft.save(); const key = draft.pendingKey;
    client.commit = async () => { throw { status: 409 }; };
    await draft.retry(); assert.equal(draft.state, 'unknown');
    await draft.compareLatest(); assert.equal(draft.useComparedRevision(), false);
    assert.equal(draft.pendingKey, key); draft.close();
});

test('closing while publication awaits cannot publish a saved state into a stale panel', async () => {
    const { client } = await fixture(); let finish; const states = [];
    const draft = new ArtifactDraft(client, value => states.push(value.state)); await draft.open(ref);
    const commit = client.commit; let entered;
    const pending = new Promise(resolve => { entered = resolve; });
    client.commit = async request => { entered(); await new Promise(resolve => { finish = resolve; }); return commit(request); };
    draft.edit('edit'); const saving = draft.save(); await pending;
    draft.close(); finish(); await saving;
    assert.notEqual(states.at(-1), 'saved'); assert.equal(draft.bytes, null); assert.equal(draft.text, '');
});

test('bound API requests pin original agent and refuse changed authentication before upload', async () => {
    let identity = 'original'; const calls = [];
    const api = createApiClient({
        authProvider: { ensureAuthenticated: async () => {}, applyAuth: async headers => ({ ...headers, Authorization: `Bearer ${identity}` }) },
        fetchFn: async (url, options) => { calls.push({ url, options }); return new Response('{}', { headers: { 'Content-Type': 'application/json' } }); },
        sessionStorage: { getItem: () => null, setItem() {}, removeItem() {} },
        location: { href: 'http://test/', search: '' },
    });
    api.setHostAgent('original-agent'); const bound = await api.bindAgentRequests();
    api.setHostAgent('other-agent'); await bound.request('/api/artifacts');
    assert.match(calls[0].url, /original-agent/);
    assert.equal(calls[0].options.credentials, 'omit');
    identity = 'new-principal';
    await assert.rejects(bound.request('/api/artifacts/staging', { method: 'POST', body: 'private' }), error => error.code === 'artifact_auth_changed');
    assert.equal(calls.length, 1);
});

test('bound binary request uses shared authentication and preserves exact bytes', async () => {
    const calls = [];
    const api = createApiClient({
        authProvider: { ensureAuthenticated: async () => {}, applyAuth: async headers => ({ ...headers, Authorization: 'Bearer original' }) },
        fetchFn: async (url, options) => { calls.push({ url, options }); return new Response(new Uint8Array([0, 255, 7])); },
        sessionStorage: { getItem: () => null, setItem() {}, removeItem() {} }, location: { href: 'http://test/', search: '' },
    });
    const bound = await api.bindAgentRequests(null);
    assert.deepEqual(new Uint8Array(await bound.readBytes('/api/artifacts/file')), new Uint8Array([0, 255, 7]));
    assert.equal(calls[0].options.headers.Authorization, 'Bearer original');
    assert.equal(calls[0].options.credentials, 'omit');
});

test('401 refresh cannot resend a bound upload under replacement credentials', async () => {
    let identity = 'original', refreshes = 0; const calls = [];
    const api = createApiClient({
        authProvider: { ensureAuthenticated: async () => {},
            applyAuth: async headers => ({ ...headers, Authorization: `Bearer ${identity}` }),
            onUnauthorized: async () => { refreshes++; identity = 'replacement'; return 'refreshed'; } },
        fetchFn: async (url, options) => { calls.push({ url, options }); return new Response('{"detail":"expired"}', { status: 401 }); },
        sessionStorage: { getItem: () => null, setItem() {}, removeItem() {} }, location: { href: 'http://test/', search: '' },
    });
    const bound = await api.bindAgentRequests('agent');
    await assert.rejects(bound.request('/api/artifacts/staging', { method: 'POST', body: 'private' }), error => error.status === 401);
    assert.equal(refreshes, 1); assert.equal(calls.length, 1);
    assert.equal(calls[0].options.headers.Authorization, 'Bearer original');
});

test('closing during digest preparation clears the draft and never stages bytes', async () => {
    const { draft, staged } = await fixture(); draft.edit('private');
    const saving = draft.save(); draft.close(); await saving;
    assert.equal(staged.length, 0); assert.equal(draft.pendingKey, null); assert.equal(draft.text, '');
});

test('a second concurrent open cannot replace a panel revision', async () => {
    const { client } = await fixture(); const draft = new ArtifactDraft(client);
    const opening = draft.open(ref);
    await assert.rejects(draft.open({ ...ref, revision_id: 'other' }), /new panel/);
    await opening; assert.equal(draft.revision.ref.revision_id, 'revision1'); draft.close();
});

test('standalone OAuth cookie authentication refuses to bind an artifact draft', async () => {
    const calls = [], sessionStorage = { getItem: () => null, setItem() {}, removeItem() {} };
    const location = { href: 'http://test/', search: '' };
    const fetchFn = async url => { calls.push(url); return url === '/api/auth/key'
        ? new Response('{}', { status: 404 }) : new Response('{"email":"owner@example.invalid"}'); };
    const auth = createKestrelStandaloneAuthProvider({ fetchFn, sessionStorage, location, requestApiKey: null,
        logger: { log() {}, warn() {}, error() {} } });
    const api = createApiClient({ authProvider: auth, fetchFn, sessionStorage, location });
    await api.init();
    await assert.rejects(api.bindAgentRequests('agent'), error => error.code === 'artifact_auth_unbound');
    assert.deepEqual(calls, ['/api/auth/key', '/auth/me']);
});

test('bound request omits cookies even when a caller tries to include them', async () => {
    const calls = [];
    const api = createApiClient({ authProvider: { ensureAuthenticated: async () => {},
        applyAuth: async headers => ({ ...headers, 'X-API-Key': 'explicit' }) },
        fetchFn: async (url, options) => { calls.push(options); return new Response('{}'); },
        sessionStorage: { getItem: () => null, setItem() {}, removeItem() {} }, location: { href: 'http://test/', search: '' } });
    const bound = await api.bindAgentRequests('agent');
    await bound.request('/api/artifacts/staging', { method: 'POST', body: 'private', credentials: 'include' });
    assert.equal(calls[0].credentials, 'omit');
});

test('a stable declared principal cannot conceal a changed bearer credential', async () => {
    let token = 'A'; const calls = [];
    const api = createApiClient({ authProvider: { ensureAuthenticated: async () => {},
        getPrincipalIdentity: async () => 'stable', applyAuth: async headers => ({ ...headers, Authorization: `Bearer ${token}` }) },
        fetchFn: async (url, options) => { calls.push(options); return new Response('{}'); },
        sessionStorage: { getItem: () => null, setItem() {}, removeItem() {} }, location: { href: 'http://test/', search: '' } });
    const bound = await api.bindAgentRequests('agent'); token = 'B';
    await assert.rejects(bound.request('/api/artifacts/staging', { method: 'POST', body: 'private' }), error => error.code === 'artifact_auth_changed');
    assert.equal(calls.length, 0);
});

test('mixed line endings stay read-only and retain original source bytes', async () => {
    const { draft, bytes } = await fixture('first\r\nsecond\nthird\r');
    assert.equal(draft.editable, false); assert.equal(draft.edit('replacement'), false);
    assert.deepEqual(draft.bytes, bytes); draft.close();
});


test('receipt recovery distinguishes host 404 from artifact erasure through bound transport', async () => {
    for (const body of [{ error: { code: 'agent_not_found', message: 'Agent unavailable' } }, { detail: 'Not Found' },
        { error: { code: 'artifact_not_found', message: 'Publication erased' } }]) {
        const { draft, client, committed } = await fixture();
        const commit = client.commit; let receipt;
        client.commit = async request => { receipt = await commit(request); throw new Error('lost acknowledgement'); };
        draft.edit('recoverable'); await draft.save(); const key = draft.pendingKey;
        const responses = [new Response(JSON.stringify(body), { status: 404 }), new Response(JSON.stringify(receipt))];
        const calls = [];
        const api = createApiClient({ authProvider: { ensureAuthenticated: async () => {},
            applyAuth: async headers => ({ ...headers, Authorization: 'Bearer original' }) },
            fetchFn: async url => { calls.push(url); return responses.shift(); },
            sessionStorage: { getItem: () => null, setItem() {}, removeItem() {} },
            location: { href: 'http://test/', search: '' } });
        client.status = (await createArtifactClient(api, 'agent')).status;
        await draft.reconcile();
        if (body.error?.code === 'artifact_not_found') {
            assert.equal(draft.state, 'erased'); await draft.retry();
            assert.equal(calls.length, 1); draft.close(); continue;
        }
        assert.equal(draft.state, 'unknown'); assert.equal(draft.pendingKey, key);
        assert.equal(draft.text, 'recoverable');
        await draft.retry();
        assert.equal(draft.state, 'saved'); assert.equal(committed.length, 1);
        assert.equal(calls.length, 2); assert.equal(calls[0], calls[1]);
        assert.match(calls[0], new RegExp(`/publications/${key}$`)); draft.close();
    }
});
