/** Revision-bound transport and recoverable working copies. No browser storage. */

const TOKEN = /^[A-Za-z0-9](?:[A-Za-z0-9._:@-]{0,254}[A-Za-z0-9])?$/;
const MAX_VIEW_BYTES = 16 * 1024 * 1024;
export const MAX_PREVIEW_PIXELS = 1024 * 1024;
const PREVIEW_TYPES = new Set(['application/pdf', 'image/png', 'image/jpeg', 'image/webp', 'image/gif']);
export const MAX_EDIT_BYTES = 1024 * 1024;
const EDIT_TYPES = new Set(['text/plain', 'text/markdown', 'text/x-python', 'text/javascript',
    'text/css', 'application/json', 'application/javascript', 'application/xml', 'text/xml']);

function token(value) {
    if (typeof value !== 'string' || !TOKEN.test(value)) throw new Error('Invalid artifact reference.');
    return value;
}
export function artifactRef(value) {
    if (!value || (value.schema_version !== undefined && value.schema_version !== 1)) {
        throw new Error('Unsupported artifact reference.');
    }
    return Object.freeze({ schema_version: 1, artifact_id: token(value.artifact_id), revision_id: token(value.revision_id) });
}
function revisionPath(ref) {
    ref = artifactRef(ref);
    return `/api/artifacts/${encodeURIComponent(ref.artifact_id)}/revisions/${encodeURIComponent(ref.revision_id)}`;
}
function canonical(value) {
    if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
    if (value && typeof value === 'object') {
        return `{${Object.keys(value).sort().map(key => `${canonical(key)}:${canonical(value[key])}`).join(',')}}`;
    }
    return JSON.stringify(value).replace(/[\u007f-\uffff]/g, char => `\\u${char.charCodeAt(0).toString(16).padStart(4, '0')}`);
}
function codePointOrder(a, b) {
    const left = Array.from(a, char => char.codePointAt(0)), right = Array.from(b, char => char.codePointAt(0));
    for (let i = 0; i < Math.min(left.length, right.length); i++) if (left[i] !== right[i]) return left[i] - right[i];
    return left.length - right.length;
}
export async function sha256(bytes) {
    return [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))]
        .map(byte => byte.toString(16).padStart(2, '0')).join('');
}
async function digest(value) { return sha256(new TextEncoder().encode(canonical(value))); }
export async function publicationDigests(request) {
    // Python's SDK sorts Unicode code points; JavaScript's default order sorts
    // UTF-16 units, which differs for supplementary-plane filenames.
    const manifest = { ...request.manifest, files: [...request.manifest.files].sort((a, b) => codePointOrder(a.path, b.path)) };
    const manifest_sha256 = await digest(manifest);
    return { manifest_sha256, request_sha256: await digest({ ...request, manifest: manifest_sha256 }) };
}
function sameScope(a, b) {
    return a?.tenant_id === b?.tenant_id && a?.principal_id === b?.principal_id
        && (a?.agent_id ?? null) === (b?.agent_id ?? null);
}
function sourceLineEnding(text) {
    const endings = new Set(text.match(/\r\n|\r|\n/g) || []);
    return endings.size > 1 ? null : [...endings][0] || '';
}
function base64(bytes) {
    let encoded = '';
    for (let start = 0; start < bytes.length; start += 8192) encoded += String.fromCharCode(...bytes.subarray(start, start + 8192));
    return btoa(encoded);
}

export async function createArtifactClient(api, ownerAgent = api.getHostAgent()) {
    const bound = await api.bindAgentRequests(ownerAgent);
    const json = (path, body, signal) => bound.request(path, { method: 'POST', body: JSON.stringify(body), signal, cache: 'no-store' });
    return Object.freeze({
        list: (query = '', cursor = null, signal) => bound.request(`/api/artifacts?limit=50&q=${encodeURIComponent(query)}${cursor ? `&cursor=${encodeURIComponent(cursor)}` : ''}`, { signal, cache: 'no-store' }),
        get: (ref, signal) => bound.request(revisionPath(ref), { signal, cache: 'no-store' }),
        head: (id, signal) => bound.request(`/api/artifacts/${encodeURIComponent(token(id))}/head`, { signal, cache: 'no-store' }),
        read: (ref, path, signal) => bound.readBytes(`${revisionPath(ref)}/file?path=${encodeURIComponent(path)}`, { signal, cache: 'no-store' }),
        preview: (ref, signal) => bound.request(`${revisionPath(ref)}/preview`, { signal, cache: 'no-store' }),
        history: (id, cursor = null, signal) => bound.request(`/api/artifacts/${encodeURIComponent(token(id))}/history?limit=50${cursor ? `&cursor=${encodeURIComponent(cursor)}` : ''}`, { signal, cache: 'no-store' }),
        stage: (key, file, bytes, signal) => json('/api/artifacts/staging', {
            publication_key: key, path: file.path, media_type: file.media_type, content_base64: base64(bytes),
        }, signal),
        commit: (request, signal) => json('/api/artifacts/publications', request, signal),
        status: (key, signal) => bound.request(`/api/artifacts/publications/${encodeURIComponent(token(key))}`, { signal, cache: 'no-store' }),
    });
}

export class ArtifactDraft {
    #client; #changed; #pending = null; #closed = false; #abort = new AbortController();
    #revision = null; #file = null; #bytes = null; #text = ''; #state = 'loading';
    #editable = false; #comparison = null; #opened = false; #lineEnding = '\n'; #newTitle = null; #preview = null; #previewState = 'unavailable';
    constructor(client, changed = () => {}) { this.#client = client; this.#changed = changed; }
    get state() { return this.#state; }
    get revision() { return this.#revision; }
    get title() { return this.#revision?.title ?? this.#newTitle; }
    get file() { return this.#file; }
    get bytes() { return this.#bytes; }
    get text() { return this.#text; }
    get preview() { return this.#preview; }
    get previewState() { return this.#previewState; }
    get editable() { return this.#editable; }
    get pendingKey() { return this.#pending?.request.idempotency_key ?? null; }
    get comparison() { return this.#comparison; }
    get hasUnsavedChanges() { return (!!this.#revision || this.#newTitle !== null) && ['dirty', 'saving', 'unknown', 'conflict', 'compare_failed', 'denied', 'failed', 'rejected', 'erased'].includes(this.#state); }
    #set(state) {
        if (!this.#closed) {
            if (['erased', 'denied', 'unavailable'].includes(state)) { this.#comparison = null; this.#preview = null; }
            this.#state = state; this.#changed(this);
        }
    }
    async #loadRevision(ref) {
        ref = artifactRef(ref);
        const revision = await this.#client.get(ref, this.#abort.signal);
        if (revision.schema_version !== 1 || canonical(artifactRef(revision.ref)) !== canonical(ref)
            || !revision.scope?.tenant_id || !revision.scope?.principal_id || typeof revision.title !== 'string') {
            throw new Error('Artifact metadata does not match the requested revision.');
        }
        const file = revision.manifest?.files?.find(item => item.path === revision.manifest.entrypoint);
        if (!file || !Number.isSafeInteger(file.size_bytes) || file.size_bytes < 0 || file.size_bytes > MAX_VIEW_BYTES
            || !/^[a-f0-9]{64}$/.test(file.sha256) || typeof file.media_type !== 'string') {
            throw new Error('Artifact is too large or cannot be verified for this viewer.');
        }
        const bytes = new Uint8Array(await this.#client.read(ref, file.path, this.#abort.signal));
        if (bytes.length !== file.size_bytes || await sha256(bytes) !== file.sha256) {
            throw new Error('Artifact content failed verification.');
        }
        let text = null;
        if (EDIT_TYPES.has(file.media_type) && file.size_bytes <= MAX_EDIT_BYTES) {
            try { text = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(bytes); } catch (_) { /* Binary source remains downloadable. */ }
        }
        const lineEnding = text === null ? null : sourceLineEnding(text);
        return { revision, file, bytes, text, lineEnding };
    }
    async prepare({ title, path, media_type, bytes }) {
        if (this.#closed || this.#opened) throw new Error('Create a new panel for each source.');
        this.#opened = true;
        if (!globalThis.crypto?.subtle || typeof globalThis.crypto.randomUUID !== 'function') {
            this.#set('secure_context_required'); return;
        }
        bytes = new Uint8Array(bytes);
        if (bytes.length > MAX_VIEW_BYTES || typeof title !== 'string' || !title.trim()) {
            this.#set('unavailable'); return;
        }
        const file = { schema_version: 1, path, media_type, size_bytes: bytes.length, sha256: await sha256(bytes) };
        if (this.#closed) return;
        this.#bytes = bytes; this.#file = file; this.#newTitle = title;
        if (EDIT_TYPES.has(media_type) && bytes.length <= MAX_EDIT_BYTES) {
            try {
                this.#text = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(bytes);
                const ending = sourceLineEnding(this.#text);
                this.#editable = ending !== null; this.#lineEnding = ending || '\n';
            } catch (_) { /* Keep binary source unchanged and downloadable. */ }
        }
        this.#set('dirty');
    }
    async open(ref) {
        if (this.#closed || this.#opened) throw new Error('Create a new panel to open another revision.');
        this.#opened = true;
        if (!globalThis.crypto?.subtle || typeof globalThis.crypto.randomUUID !== 'function') {
            this.#set('secure_context_required'); return;
        }
        try {
            const loaded = await this.#loadRevision(ref);
            if (this.#closed) return;
            this.#revision = loaded.revision; this.#file = loaded.file; this.#bytes = loaded.bytes;
            this.#text = loaded.text ?? '';
            this.#editable = loaded.text !== null && loaded.lineEnding !== null && loaded.revision.manifest.files.length === 1;
            this.#lineEnding = loaded.lineEnding || '\n';
            this.#set('saved');
            await this.#fetchPreview();
        } catch (error) { this.#set([401, 403].includes(error.status) || error.code === 'artifact_auth_changed' ? 'denied' : 'unavailable'); }
    }
    edit(text) {
        if (!this.#editable || this.#pending || this.#closed || this.#state === 'saving') return false;
        // Textareas normalize CRLF/CR to LF. Restore the source convention;
        // mixed-ending files remain read-only rather than being rewritten.
        this.#text = String(text).replace(/\r\n|\r|\n/g, this.#lineEnding); this.#set('dirty'); return true;
    }
    async save() {
        if (this.#closed || this.#pending || (!this.#editable && this.#newTitle === null) || this.#state !== 'dirty') return;
        this.#set('saving');
        const bytes = this.#editable ? new TextEncoder().encode(this.#text) : this.#bytes;
        if (bytes.length > (this.#editable ? MAX_EDIT_BYTES : MAX_VIEW_BYTES)) { this.#set('failed'); return; }
        try {
            const original = this.#file, revision = this.#revision;
            const file = { schema_version: 1, path: original.path, media_type: original.media_type,
                size_bytes: bytes.length, sha256: await sha256(bytes) };
            const request = { schema_version: 1, idempotency_key: crypto.randomUUID(),
                artifact_id: revision?.ref.artifact_id ?? null, expected_revision_id: revision?.ref.revision_id ?? null,
                title: this.title, manifest: { schema_version: 1, files: [file], entrypoint: file.path },
                provenance: { schema_version: 1, session_id: null, run_id: null, tool_name: null } };
            const digests = await publicationDigests(request);
            if (this.#closed) return;
            this.#pending = { request, file, bytes, digests, phase: 'stage', uncertain: false };
            await this.#attempt();
        } catch (_) { this.#set('failed'); }
    }
    async #accepted(receipt) {
        const pending = this.#pending;
        if (!receipt || receipt.schema_version !== 1 || (this.#revision ? !sameScope(receipt.scope, this.#revision.scope) :
                typeof receipt.scope?.tenant_id !== 'string' || !receipt.scope.tenant_id
                    || typeof receipt.scope?.principal_id !== 'string' || !receipt.scope.principal_id)
            || receipt.idempotency_key !== pending.request.idempotency_key
            || (pending.request.artifact_id !== null && receipt.ref?.artifact_id !== pending.request.artifact_id)
            || receipt.ref?.revision_id === pending.request.expected_revision_id
            || receipt.manifest_sha256 !== pending.digests.manifest_sha256
            || receipt.request_sha256 !== pending.digests.request_sha256) {
            throw new Error('Save receipt could not be verified.');
        }
        const ref = artifactRef(receipt.ref);
        if (this.#closed) return;
        this.#revision = { schema_version: 1, ...this.#revision, scope: receipt.scope, title: pending.request.title, ref, parent_revision_id: pending.request.expected_revision_id,
            manifest: pending.request.manifest, provenance: pending.request.provenance, committed_at: receipt.committed_at };
        this.#bytes = pending.bytes; this.#file = pending.file; this.#newTitle = null; this.#pending = null; this.#comparison = null;
        this.#set('saved');
        await this.#fetchPreview();
    }
    async #fetchPreview() {
        if (!PREVIEW_TYPES.has(this.#file?.media_type) || !this.#client.preview || this.#closed) return;
        const ref = this.#revision.ref;
        this.#previewState = 'loading'; this.#changed(this);
        try {
            const result = await this.#client.preview(ref, this.#abort.signal);
            if (this.#closed) return;
            if (canonical(artifactRef(result.source)) !== canonical(ref) || result.media_type !== 'image/png'
                || !Number.isInteger(result.width) || !Number.isInteger(result.height)
                || result.width < 1 || result.width > 1024 || result.height < 1 || result.height > 1024
                || result.width * result.height > MAX_PREVIEW_PIXELS
                || typeof result.content_base64 !== 'string' || result.content_base64.length > 6 * 1024 * 1024) throw new Error('Invalid preview');
            const bytes = Uint8Array.from(atob(result.content_base64), char => char.charCodeAt(0));
            const view = new DataView(bytes.buffer);
            if (bytes.length < 33 || bytes.length > 4 * 1024 * 1024
                || [...bytes.subarray(0, 8)].join(',') !== '137,80,78,71,13,10,26,10'
                || view.getUint32(8) !== 13 || view.getUint32(12) !== 0x49484452
                || view.getUint32(16) !== result.width || view.getUint32(20) !== result.height
                || bytes[24] !== 8 || bytes[25] !== 2 || bytes[26] !== 0 || bytes[27] !== 0 || bytes[28] !== 0
                || await sha256(bytes) !== result.sha256) throw new Error('Invalid preview');
            if (this.#closed) return;
            this.#preview = { bytes, width: result.width, height: result.height };
            this.#previewState = 'ready'; this.#changed(this);
        } catch (error) {
            if (this.#closed) return;
            this.#preview = null; this.#previewState = 'unavailable';
            if ([401, 403].includes(error.status) || error.code === 'artifact_auth_changed') this.#set('denied');
            else if (error.code === 'artifact_not_found') this.#set('erased');
            else this.#changed(this); // Refusal preserves authorized source download.
        }
    }
    async #attempt() {
        const pending = this.#pending;
        this.#set('saving');
        try {
            if (pending.phase === 'stage') {
                const staged = await this.#client.stage(pending.request.idempotency_key, pending.file, pending.bytes, this.#abort.signal);
                if (canonical(staged) !== canonical(pending.file)) throw new Error('Staging did not confirm this exact file.');
                pending.phase = 'commit';
            }
            if (!this.#closed) await this.#accepted(await this.#client.commit(pending.request, this.#abort.signal));
        } catch (error) {
            if (this.#closed) return;
            if ([401, 403].includes(error.status) || error.code === 'artifact_auth_changed') this.#set('denied');
            else if (!pending.uncertain && ((error.status === 400 && error.code === 'invalid_artifact_request')
                || (error.status === 413 && error.code === 'artifact_quota_exceeded'))) {
                // These typed service errors confirm rollback. A prior unknown
                // attempt still owns its key even if a later attempt rejects.
                this.#pending = null; this.#set('rejected');
            }
            else if (error.status === 404 && error.code === 'artifact_not_found'
                && pending.phase === 'commit' && !pending.uncertain) {
                // The service rejected this attempt before commit. Determine
                // whether its staging expired or the artifact was deleted.
                pending.phase = 'stage';
                if (!this.#revision) { this.#set('unknown'); return; }
                try {
                    await this.#client.get(this.#revision.ref, this.#abort.signal);
                    this.#set('unknown');
                } catch (lookup) {
                    this.#set(lookup.status === 404 && lookup.code === 'artifact_not_found' ? 'erased'
                        : [401, 403].includes(lookup.status) || lookup.code === 'artifact_auth_changed' ? 'denied' : 'unknown');
                }
            }
            else if (error.status === 404 && pending.phase === 'commit') {
                // Missing staging may have expired. Only an artifact_not_found status response is
                // terminal erasure. Retry retains the exact key/request/bytes.
                pending.phase = 'stage'; pending.uncertain = true; this.#set('unknown');
            }
            else if (error.status === 409 && pending.phase === 'commit' && !pending.uncertain && this.#revision) this.#set('conflict');
            else { pending.uncertain ||= pending.phase === 'commit'; this.#set('unknown'); }
        }
    }
    async reconcile() {
        if (this.#closed || !this.#pending || this.#state === 'saving') return;
        const previous = this.#state;
        this.#set('saving');
        try {
            const receipt = await this.#client.status(this.#pending.request.idempotency_key, this.#abort.signal);
            if (receipt) await this.#accepted(receipt);
            else this.#set(['conflict', 'compare_failed'].includes(previous) ? previous : 'unknown');
        } catch (error) {
            this.#set(error.status === 404 && error.code === 'artifact_not_found' ? 'erased' : [401, 403].includes(error.status) || error.code === 'artifact_auth_changed' ? 'denied' : 'unknown');
        }
    }
    async retry() {
        if (this.#state !== 'unknown' || !this.#pending || this.#closed) return;
        await this.reconcile();
        if (this.#state === 'unknown' && this.#pending) await this.#attempt();
    }
    async compareLatest() {
        if (!['conflict', 'compare_failed'].includes(this.#state) || this.#pending?.uncertain || this.#closed) return;
        await this.reconcile();
        if (!['conflict', 'compare_failed'].includes(this.#state)) return;
        this.#comparison = null;
        this.#set('saving');
        try {
            const latest = await this.#client.head(this.#revision.ref.artifact_id, this.#abort.signal);
            if (!latest || latest.ref.artifact_id !== this.#revision.ref.artifact_id) throw new Error('No current revision.');
            const loaded = await this.#loadRevision(latest.ref);
            if (!sameScope(loaded.revision.scope, this.#revision.scope) || loaded.text === null || loaded.lineEnding === null
                || loaded.file.path !== this.#file.path || loaded.file.media_type !== this.#file.media_type
                || loaded.revision.manifest.files.length !== 1) throw new Error('The artifact structure changed.');
            if (this.#closed) return;
            this.#comparison = loaded; this.#set('conflict');
        } catch (error) {
            this.#set(error.status === 404 && error.code === 'artifact_not_found' ? 'erased'
                : [401, 403].includes(error.status) || error.code === 'artifact_auth_changed' ? 'denied' : 'compare_failed');
        }
    }
    useComparedRevision() {
        if (!this.#comparison || this.#state !== 'conflict' || this.#pending?.uncertain) return false;
        // Explicit user choice after a confirmed rejected attempt and comparison.
        // Keep the draft; the next Save stages it under a fresh key and exact CAS.
        this.#revision = this.#comparison.revision;
        this.#file = this.#comparison.file; this.#bytes = this.#comparison.bytes;
        this.#lineEnding = this.#comparison.lineEnding || this.#lineEnding;
        this.#text = this.#text.replace(/\r\n|\r|\n/g, this.#lineEnding);
        this.#pending = null; this.#comparison = null;
        this.#set('dirty'); return true;
    }
    async history(cursor = null) {
        return this.#client.history(this.#revision.ref.artifact_id, cursor, this.#abort.signal);
    }
    close() {
        this.#closed = true; this.#abort.abort(); this.#pending = null; this.#bytes = null;
        this.#text = ''; this.#newTitle = null; this.#preview = null; this.#comparison = null; this.#revision = null; this.#file = null;
    }
}
