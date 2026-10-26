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
        head: async () => structuredClone(revision),
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
    client.head = async () => latest; client.history = async () => ({ items: [latest] });
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


test('an upload failure before any commit still permits confirmed conflict recovery', async () => {
    const { draft, client, committed } = await fixture(); const stage = client.stage;
    client.stage = async () => { throw new TypeError('Failed to fetch'); };
    draft.edit('retained draft'); await draft.save();
    assert.equal(committed.length, 0); assert.equal(draft.state, 'unknown');
    client.stage = stage; client.commit = async () => { throw { status: 409, code: 'artifact_conflict' }; };
    await draft.retry(); assert.equal(draft.state, 'conflict');
    await draft.compareLatest(); assert.equal(draft.comparison.text, '# Original');
    assert.equal(draft.useComparedRevision(), true); assert.equal(draft.text, 'retained draft'); draft.close();
});

test('opening without permission never creates a draft to discard', async () => {
    for (const error of [{ status: 401 }, { status: 403 }, { code: 'artifact_auth_changed' }]) {
        const draft = new ArtifactDraft({ get: async () => { throw error; } });
        await draft.open(ref); assert.equal(draft.state, 'denied');
        assert.equal(draft.hasUnsavedChanges, false); draft.close();
    }
});


test('a failed restage cannot forget an earlier uncertain commit', async () => {
    const { draft, client } = await fixture(); const stage = client.stage;
    client.commit = async () => { throw { status: 404 }; };
    draft.edit('keep'); await draft.save(); const key = draft.pendingKey;
    client.stage = async () => { throw new Error('stage lost'); };
    await draft.retry(); assert.equal(draft.state, 'unknown');
    client.stage = stage; client.commit = async () => { throw { status: 409 }; };
    await draft.retry(); assert.equal(draft.state, 'unknown');
    await draft.compareLatest(); assert.equal(draft.useComparedRevision(), false);
    assert.equal(draft.pendingKey, key); draft.close();
});

test('explicit rebase adopts the latest source line endings for retained and subsequent edits', async () => {
    for (const ending of ['\r\n', '\r']) {
        const { draft, client, revision, staged } = await fixture('a\nb\n');
        const commit = client.commit; client.commit = async () => { throw { status: 409 }; };
        draft.edit('a\nb\nmine\n'); await draft.save();
        const bytes = new TextEncoder().encode(`new${ending}base${ending}`);
        const file = { ...revision.manifest.files[0], size_bytes: bytes.length, sha256: await sha256(bytes) };
        const latest = { ...revision, ref: { ...ref, revision_id: 'new-base' }, manifest: { ...revision.manifest, files: [file] } };
        client.head = async () => latest; client.history = async () => ({ items: [latest] }); client.get = async () => latest; client.read = async () => bytes;
        await draft.compareLatest(); assert.equal(draft.useComparedRevision(), true);
        assert.equal(draft.text, `a${ending}b${ending}mine${ending}`);
        draft.edit('a\nb\nmine\nmore\n'); client.commit = commit; await draft.save();
        assert.equal(new TextDecoder().decode(staged.at(-1).content), `a${ending}b${ending}mine${ending}more${ending}`);
        draft.close();
    }
});

test('opening rejects mismatched references, invalid metadata and corrupt content', async t => {
    const cases = {
        reference: (r) => { r.ref.revision_id = 'unexpected'; },
        schema: (r) => { r.schema_version = 2; },
        tenant: (r) => { delete r.scope.tenant_id; },
        principal: (r) => { delete r.scope.principal_id; },
        title: (r) => { r.title = {}; },
        entrypoint: (r) => { r.manifest.entrypoint = 'missing'; },
        fractionalSize: (r) => { r.manifest.files[0].size_bytes = 0.5; },
        negativeSize: (r) => { r.manifest.files[0].size_bytes = -1; },
        viewLimit: (r) => { r.manifest.files[0].size_bytes = 16 * 1024 * 1024 + 1; },
        invalidHash: (r) => { r.manifest.files[0].sha256 = 'bad'; },
        mediaType: (r) => { r.manifest.files[0].media_type = {}; },
        sizeMismatch: (r) => { r.manifest.files[0].size_bytes++; },
        hashMismatch: (r) => { r.manifest.files[0].sha256 = '0'.repeat(64); },
    };
    for (const [name, corrupt] of Object.entries(cases)) await t.test(name, async () => {
        const item = await fixture(); item.draft.close(); const revision = structuredClone(item.revision); corrupt(revision);
        let reads = 0; item.client.get = async () => revision; item.client.read = async () => { reads++; return item.bytes; };
        const draft = new ArtifactDraft(item.client); await draft.open(ref);
        assert.equal(draft.state, 'unavailable'); assert.equal(draft.bytes, null); assert.equal(draft.edit('x'), false);
        if (name === 'viewLimit') assert.equal(reads, 0, 'oversized source must be refused before download');
        draft.close();
    });
});

test('single-file edit and byte limits prevent destructive bundle saves and oversized uploads', async () => {
    const item = await fixture(); item.draft.close();
    item.revision.manifest.files.push({ ...item.revision.manifest.files[0], path: 'second.md' });
    const bundle = new ArtifactDraft(item.client); await bundle.open(ref);
    assert.equal(bundle.state, 'saved'); assert.equal(bundle.editable, false); assert.equal(bundle.edit('drop files'), false);
    await bundle.save(); assert.equal(item.staged.length, 0); bundle.close();
    const large = await fixture('x'.repeat(1024 * 1024 + 1));
    assert.equal(large.draft.state, 'saved'); assert.equal(large.draft.editable, false); large.draft.close();
    const exact = await fixture('x'.repeat(1024 * 1024)); assert.equal(exact.draft.editable, true);
    exact.draft.edit('é'.repeat(512 * 1024 + 1)); await exact.draft.save();
    assert.equal(exact.draft.state, 'failed'); assert.equal(exact.staged.length, 0); assert.equal(exact.committed.length, 0);
    exact.draft.close();
});

test('staging must confirm the exact bytes and manifest before any commit', async () => {
    for (const change of [{ sha256: '0'.repeat(64) }, { size_bytes: 0 }, { path: 'other' }, { media_type: 'text/plain' }]) {
        const { draft, client, committed } = await fixture();
        client.stage = async (key, file) => ({ ...file, ...change });
        draft.edit('keep this'); await draft.save();
        assert.equal(committed.length, 0); assert.equal(draft.state, 'unknown'); assert.equal(draft.text, 'keep this'); draft.close();
    }
});

test('comparison refuses a different owner, source structure or noneditable content', async t => {
    const cases = {
        tenant: (r) => { r.scope.tenant_id = 'other'; },
        principal: (r) => { r.scope.principal_id = 'other'; },
        agent: (r) => { r.scope.agent_id = 'other'; },
        path: (r) => { r.manifest.entrypoint = r.manifest.files[0].path = 'other.md'; },
        type: (r) => { r.manifest.files[0].media_type = 'text/plain'; },
        bundle: (r) => { r.manifest.files.push({ ...r.manifest.files[0], path: 'second.md' }); },
        binary: (r) => { r.manifest.files[0].media_type = 'application/octet-stream'; },
        invalidUtf8: () => {},
        mixedLines: () => {},
    };
    for (const [name, change] of Object.entries(cases)) await t.test(name, async () => {
        const { draft, client, revision } = await fixture(); client.commit = async () => { throw { status: 409 }; };
        draft.edit('retained'); await draft.save(); const key = draft.pendingKey;
        const latest = structuredClone(revision); latest.ref.revision_id = 'new-base';
        const bytes = name === 'invalidUtf8' ? new Uint8Array([0xff])
            : new TextEncoder().encode(name === 'mixedLines' ? 'a\nb\r\nc' : 'new source');
        Object.assign(latest.manifest.files[0], { size_bytes: bytes.length, sha256: await sha256(bytes) }); change(latest);
        client.head = async () => latest; client.history = async () => ({ items: [latest] }); client.get = async () => latest; client.read = async () => bytes;
        await draft.compareLatest(); assert.equal(draft.comparison, null); assert.equal(draft.useComparedRevision(), false);
        assert.equal(draft.pendingKey, key); assert.equal(draft.text, 'retained'); draft.close();
    });
});

test('bound binary download refuses changed credentials before reading another owner source', async () => {
    let identity = 'original'; const calls = [];
    const api = createApiClient({
        authProvider: { ensureAuthenticated: async () => {}, applyAuth: async headers => ({ ...headers, Authorization: `Bearer ${identity}` }) },
        fetchFn: async (url, options) => { calls.push({ url, options }); return new Response('private source'); },
        sessionStorage: { getItem: () => null, setItem() {}, removeItem() {} }, location: { href: 'http://test/', search: '' },
    });
    const bound = await api.bindAgentRequests('original-agent'); identity = 'other';
    await assert.rejects(bound.readBytes('/api/artifacts/file'), error => error.code === 'artifact_auth_changed');
    assert.equal(calls.length, 0);
});

test('confirmed validation or quota rejection retains an editable draft without pretending a commit is pending', async () => {
    for (const error of [{ status: 400, code: 'invalid_artifact_request' }, { status: 413, code: 'artifact_quota_exceeded' }]) {
        const { draft, client } = await fixture(); client.commit = async () => { throw error; };
        draft.edit('retained'); await draft.save(); assert.equal(draft.state, 'rejected');
        assert.equal(draft.pendingKey, null); assert.equal(draft.hasUnsavedChanges, true);
        assert.equal(draft.edit('corrected'), true); assert.equal(draft.state, 'dirty'); draft.close();
    }
});

test('known expired staging permits a later conflict while a deleted artifact becomes terminal', async () => {
    for (const deleted of [false, true]) {
        const { draft, client } = await fixture();
        client.commit = async () => { throw { status: 404, code: 'artifact_not_found' }; };
        if (deleted) client.get = async () => { throw { status: 404, code: 'artifact_not_found' }; };
        draft.edit('retained'); await draft.save();
        assert.equal(draft.state, deleted ? 'erased' : 'unknown');
        if (!deleted) {
            client.commit = async () => { throw { status: 409, code: 'artifact_conflict' }; };
            await draft.retry(); assert.equal(draft.state, 'conflict');
        }
        assert.equal(draft.text, 'retained'); draft.close();
    }
});

test('later typed rejections never clear an earlier unknown commit or permit a new key', async () => {
    for (const error of [{ status: 400, code: 'invalid_artifact_request' }, { status: 413, code: 'artifact_quota_exceeded' },
        { status: 404, code: 'artifact_not_found' }]) {
        const { draft, client } = await fixture(); client.commit = async () => { throw new Error('lost'); };
        draft.edit('retained'); await draft.save(); const key = draft.pendingKey;
        client.commit = async () => { throw error; }; await draft.retry();
        assert.equal(draft.state, 'unknown'); assert.equal(draft.pendingKey, key); assert.equal(draft.edit('new'), false);
        draft.close();
    }
});

test('conflict comparison reads the committed head independently of history order', async () => {
    const { draft, client, revision } = await fixture();
    client.commit = async () => { throw { status: 409 }; }; draft.edit('mine'); await draft.save();
    const latest = structuredClone(revision); latest.ref.revision_id = 'committed-head';
    latest.committed_at = '2026-10-19T00:00:00Z';
    client.head = async id => { assert.equal(id, ref.artifact_id); return latest; };
    client.history = async () => { throw new Error('history cannot select a head'); };
    client.get = async requested => { assert.equal(requested.revision_id, 'committed-head'); return latest; };
    await draft.compareLatest();
    assert.equal(draft.comparison.revision.ref.revision_id, 'committed-head');
    assert.equal(draft.useComparedRevision(), true); assert.equal(draft.revision.ref.revision_id, 'committed-head');
    assert.equal(draft.text, 'mine'); draft.close();
});

test('head lookup uses the original bound agent and authenticated artifact route', async () => {
    const { revision, draft } = await fixture(); draft.close(); const calls=[];
    const api = createApiClient({
        authProvider: { ensureAuthenticated: async () => {}, applyAuth: async headers => ({ ...headers, Authorization: 'Bearer owner' }) },
        fetchFn: async (url, options) => { calls.push({url, options}); return new Response(JSON.stringify(revision), {headers:{'Content-Type':'application/json'}}); },
        sessionStorage: { getItem: () => null, setItem() {}, removeItem() {} }, location: { href:'http://test/', search:'' },
    });
    api.setHostAgent('original-agent'); const client=await createArtifactClient(api);
    api.setHostAgent('another-agent'); assert.deepEqual(await client.head('artifact'), revision);
    assert.match(calls[0].url, /original-agent/); assert.ok(calls[0].url.endsWith('/api/artifacts/artifact/head'));
    assert.equal(calls[0].options.credentials, 'omit'); assert.equal(calls[0].options.cache, 'no-store');
});

test('unsupported reference versions are refused before fetching metadata', async () => {
    const item=await fixture(); item.draft.close(); let reads=0;
    item.client.get=async()=>{reads++;return item.revision;};
    const draft=new ArtifactDraft(item.client);await draft.open({...ref,schema_version:2});
    assert.equal(draft.state,'unavailable');assert.equal(reads,0);draft.close();
});

test('rebasing onto a single line retains the draft line-ending convention', async () => {
    for(const ending of ['\r\n','\r']){
        const {draft,client,revision,staged}=await fixture(`a${ending}b${ending}`);const commit=client.commit;
        client.commit=async()=>{throw {status:409};};draft.edit('a\nb\nmine\n');await draft.save();
        const bytes=new TextEncoder().encode('single line');const latest=structuredClone(revision);latest.ref.revision_id='single-line';
        Object.assign(latest.manifest.files[0],{size_bytes:bytes.length,sha256:await sha256(bytes)});
        client.head=async()=>latest;client.get=async()=>latest;client.read=async()=>bytes;
        await draft.compareLatest();assert.equal(draft.useComparedRevision(),true);
        assert.equal(draft.text,`a${ending}b${ending}mine${ending}`);client.commit=commit;await draft.save();
        assert.equal(new TextDecoder().decode(staged.at(-1).content),`a${ending}b${ending}mine${ending}`);draft.close();
    }
});

test('a failed comparison clears obsolete source and can retry without losing draft or save key', async () => {
    const {draft,client,revision}=await fixture();client.commit=async()=>{throw {status:409};};
    draft.edit('retained');await draft.save();const key=draft.pendingKey;
    const latest=structuredClone(revision);latest.ref.revision_id='second';client.head=async()=>latest;client.get=async()=>latest;
    await draft.compareLatest();assert.equal(draft.comparison.revision.ref.revision_id,'second');
    client.head=async()=>{throw {status:404,code:'agent_not_found'};};await draft.compareLatest();
    assert.equal(draft.state,'compare_failed');assert.equal(draft.comparison,null);assert.equal(draft.useComparedRevision(),false);
    assert.equal(draft.hasUnsavedChanges,true);assert.equal(draft.text,'retained');assert.equal(draft.pendingKey,key);
    client.head=async()=>latest;await draft.compareLatest();assert.equal(draft.state,'conflict');
    assert.equal(draft.useComparedRevision(),true);assert.equal(draft.text,'retained');draft.close();
});

test('comparison reports typed deletion and revoked access without retaining an older comparison', async () => {
    for(const error of [{status:404,code:'artifact_not_found'},{status:403},{code:'artifact_auth_changed'}]){
        const {draft,client}=await fixture();client.commit=async()=>{throw {status:409};};draft.edit('retained');await draft.save();
        await draft.compareLatest();assert.ok(draft.comparison);client.head=async()=>{throw error;};await draft.compareLatest();
        assert.equal(draft.state,error.status===404?'erased':'denied');assert.equal(draft.comparison,null);
        assert.equal(draft.useComparedRevision(),false);assert.equal(draft.text,'retained');assert.equal(draft.hasUnsavedChanges,true);draft.close();
    }
});

test('a new upload shares durable receipt verification and same-key recovery with edits', async () => {
    const item = await fixture(); item.draft.close();
    const draft = new ArtifactDraft(item.client);
    let saved, first = true;
    const original = item.client.commit;
    item.client.commit = async request => {
        const reply = await original(request); reply.ref.artifact_id = 'new-artifact'; saved = reply;
        if (first) { first = false; throw new Error('lost acknowledgement'); } return reply;
    };
    await draft.prepare({ title: 'Uploaded notes', path: 'notes.md', media_type: 'text/markdown', bytes: item.bytes });
    assert.equal(draft.state, 'dirty'); assert.equal(draft.hasUnsavedChanges, true);
    draft.edit('New document'); await draft.save();
    assert.equal(draft.state, 'unknown'); assert.equal(item.committed[0].artifact_id, null);
    const key = draft.pendingKey;
    item.client.status = async value => { assert.equal(value, key); return saved; };
    await draft.reconcile();
    assert.equal(draft.state, 'saved'); assert.equal(draft.revision.ref.artifact_id, 'new-artifact');
    assert.equal(draft.revision.title, 'Uploaded notes'); assert.equal(draft.text, 'New document');
    assert.equal(item.committed.length, 1); draft.close();
});

test('a binary upload keeps exact source bytes, refuses oversized source, and never needs a decoder', async () => {
    const item = await fixture(); item.draft.close();
    const draft = new ArtifactDraft(item.client), bytes = new Uint8Array([0, 255, 1, 13, 10]);
    item.client.commit = async request => ({ schema_version: 1, ref: { ...ref, artifact_id: 'uploaded' }, scope,
        idempotency_key: request.idempotency_key, ...await publicationDigests(request) });
    await draft.prepare({ title: 'Image', path: 'image.png', media_type: 'image/png', bytes });
    assert.equal(draft.editable, false); await draft.save();
    assert.equal(draft.state, 'saved'); assert.deepEqual(item.staged[0].content, bytes); draft.close();
    const large = new ArtifactDraft(item.client);
    await large.prepare({ title: 'Large', path: 'large', media_type: 'application/octet-stream', bytes: new Uint8Array(16 * 1024 * 1024 + 1) });
    assert.equal(large.state, 'unavailable'); assert.equal(large.hasUnsavedChanges, false); large.close();
});

test('creation rejects mismatched receipt digests and never compares an unknown new artifact', async () => {
    const item = await fixture(); item.draft.close();
    const draft = new ArtifactDraft(item.client);
    await draft.prepare({ title: 'New', path: 'new.md', media_type: 'text/markdown', bytes: item.bytes });
    item.client.commit = async () => { throw { status: 409 }; };
    await draft.save(); assert.equal(draft.state, 'unknown');
    const key = draft.pendingKey; await draft.compareLatest(); assert.equal(draft.pendingKey, key);
    draft.close();
});

test('creation cannot claim saved after receiving an unrelated or malformed receipt', async () => {
    for (const change of [{ request_sha256:'0'.repeat(64) }, { idempotency_key:'other' }, { scope: {tenant_id:17,principal_id:'owner'} }, { ref: {artifact_id:'https://foreign',revision_id:'r'} }]) {
        const item=await fixture();item.draft.close();const draft=new ArtifactDraft(item.client),commit=item.client.commit;
        item.client.commit=async request=>({...await commit(request),...change});
        await draft.prepare({title:'New',path:'notes.md',media_type:'text/markdown',bytes:item.bytes});await draft.save();
        assert.equal(draft.state,'unknown');assert.ok(draft.pendingKey);assert.equal(draft.text,'# Original');draft.close();
    }
});

const previewBase64='iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC';
async function previewFixture(change={}) {
    const item=await fixture();item.draft.close();item.revision.manifest.files[0].media_type='image/png';
    const bytes=Uint8Array.from(atob(previewBase64), c=>c.charCodeAt(0));
    item.client.preview=async()=>({source:ref,media_type:'image/png',width:1,height:1,content_base64:previewBase64,sha256:await sha256(bytes),...change});
    const draft=new ArtifactDraft(item.client);await draft.open(ref);return {draft,item};
}
test('only a verified bounded revision-bound generated RGB PNG becomes a preview',async()=>{
    const {draft}=await previewFixture();assert.equal(draft.state,'saved');assert.equal(draft.previewState,'ready');
    assert.deepEqual([draft.preview.width,draft.preview.height],[1,1]);draft.close();
    for(const change of [{source:{...ref,revision_id:'other'}},{media_type:'image/svg+xml'},{width:100000},{height:0},
        {sha256:'0'.repeat(64)},{content_base64:btoa('<script>')},{width:2},{content_base64:'A'.repeat(6*1024*1024+1)}]) {
        const {draft}=await previewFixture(change);assert.equal(draft.state,'saved');assert.equal(draft.preview,null);assert.ok(draft.bytes);draft.close();
    }
});
