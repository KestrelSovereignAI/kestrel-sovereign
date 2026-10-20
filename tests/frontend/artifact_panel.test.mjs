import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { createArtifactPanel, artifactPartRenderer } from '../../kestrel_sovereign/static/js/artifact_panel.mjs';
import { publicationDigests, sha256 } from '../../kestrel_sovereign/static/js/artifact_client.mjs';

const ref = { schema_version: 1, artifact_id: 'artifact', revision_id: 'revision1' };
const scope = { schema_version: 1, tenant_id: 'tenant', principal_id: 'person', agent_id: 'agent' };
async function fixture({ getError = null, content = '<script>alert(1)</script>', type = 'text/markdown', title = '<img src=x onerror=alert(1)>', confirmDiscard = () => true, setupWindow = () => {} } = {}) {
    const dom = new JSDOM('<!doctype html><body></body>', { url: 'http://localhost/' });
    setupWindow(dom.window);
    const bytes = new TextEncoder().encode(content);
    const file = { schema_version: 1, path: 'source.md', media_type: type, sha256: await sha256(bytes), size_bytes: bytes.length };
    const revision = { schema_version: 1, ref, scope, title,
        manifest: { schema_version: 1, files: [file], entrypoint: file.path },
        provenance: { schema_version: 1, session_id: null, run_id: null, tool_name: null }, committed_at: '2026-10-20T12:00:00Z' };
    const client = {
        get: async () => { if (getError) throw getError; return structuredClone(revision); }, read: async () => bytes,
        stage: async (key, file) => file,
        commit: async request => ({ schema_version: 1, ref: { ...ref, revision_id: 'revision2' }, scope,
            idempotency_key: request.idempotency_key, ...await publicationDigests(request), committed_at: revision.committed_at, storage_id: 'local' }),
        status: async () => null, head: async () => structuredClone(revision),
        history: async () => ({ items: [revision], next_cursor: null }),
    };
    const opened = [];
    const panel = createArtifactPanel({ client, ref, document: dom.window.document, confirmDiscard,
        onOpenRevision: value => opened.push(value) });
    dom.window.document.body.append(panel.element); await panel.ready;
    return { dom, panel, client, opened, close() { panel.dispose(); dom.window.close(); } };
}
function button(panel, label) { return [...panel.element.querySelectorAll('button')].find(node => node.textContent === label); }
function edit(panel, window, text) {
    const editor = panel.element.querySelector('textarea'); editor.value = text;
    editor.dispatchEvent(new window.Event('input', { bubbles: true }));
}
const tick = () => new Promise(resolve => setTimeout(resolve, 0));
async function until(check) {
    const deadline = Date.now() + 2000;
    while (!check() && Date.now() < deadline) await tick();
    assert.ok(check(), 'asynchronous panel operation completed');
}

test('artifact title and Markdown source cannot inject active DOM', async () => {
    const item = await fixture();
    assert.equal(item.panel.element.querySelector('h3').textContent, '<img src=x onerror=alert(1)>');
    assert.equal(item.panel.element.querySelector('textarea').value, '<script>alert(1)</script>');
    assert.equal(item.panel.element.querySelectorAll('script,img').length, 0);
    assert.equal(button(item.panel, 'Save revision').disabled, true); item.close();
});

test('panel displays a new revision only after durable confirmation', async () => {
    const item = await fixture();
    edit(item.panel, item.dom.window, 'New source');
    assert.equal(button(item.panel, 'Save revision').disabled, false);
    button(item.panel, 'Save revision').click(); await until(() => item.panel.draft.state === 'saved');
    assert.equal(item.panel.draft.state, 'saved');
    assert.match(item.panel.element.querySelector('.artifact-revision').textContent, /revision2/);
    assert.equal(item.panel.element.querySelector('textarea').value, 'New source'); item.close();
});

test('unknown save preserves draft and disables edits while reconciliation is pending', async () => {
    const item = await fixture(); item.client.commit = async () => { throw new Error('lost connection'); };
    edit(item.panel, item.dom.window, 'Keep this draft');
    button(item.panel, 'Save revision').click(); await until(() => item.panel.draft.state === 'unknown');
    assert.equal(item.panel.draft.state, 'unknown');
    assert.equal(item.panel.element.querySelector('textarea').value, 'Keep this draft');
    assert.equal(item.panel.element.querySelector('textarea').readOnly, true);
    assert.equal(button(item.panel, 'Check save').hidden, false);
    assert.equal(button(item.panel, 'Retry same save').hidden, false);
    button(item.panel, 'Check save').click(); await tick();
    assert.equal(item.panel.draft.state, 'unknown'); item.close();
});

test('closing a dirty panel requires the user to accept discarding its draft', async () => {
    let accepted = false, asked = 0;
    const item = await fixture({ confirmDiscard: () => { asked++; return accepted; } });
    edit(item.panel, item.dom.window, 'Draft'); button(item.panel, 'Close').click(); await tick();
    assert.equal(asked, 1); assert.equal(item.panel.element.isConnected, true);
    accepted = true; button(item.panel, 'Close').click(); await tick();
    assert.equal(item.panel.element.isConnected, false); assert.equal(item.panel.draft.text, ''); item.close();
});

test('SVG, HTML and PDF remain downloadable without decoding active documents in the draft workspace', async () => {
    for (const type of ['image/svg+xml', 'text/html', 'application/pdf']) {
        const item = await fixture({ type });
        assert.equal(item.panel.element.querySelectorAll('script,object,embed,img,iframe,canvas').length, 0);
        assert.equal(button(item.panel, 'Download source file').disabled, false);
        if (type === 'application/pdf') assert.match(item.panel.element.textContent, /Download this PDF to view it/);
        item.close();
    }
});

test('history links select an exact revision and never replace the unsaved draft', async () => {
    const item = await fixture(); edit(item.panel, item.dom.window, 'Keep');
    button(item.panel, 'Revision history').click(); await tick();
    const historyLink = [...item.panel.element.querySelectorAll('button')].find(node => node.textContent.includes('revision1'));
    historyLink.click(); assert.deepEqual(item.opened, [ref]);
    assert.equal(item.panel.draft.text, 'Keep'); assert.equal(item.panel.draft.state, 'dirty'); item.close();
});

test('artifact card does not interpret an untrusted title or a URL as a reference', () => {
    const dom = new JSDOM('<!doctype html><body></body>'); const before = globalThis.document;
    globalThis.document = dom.window.document;
    try {
        const api = { getHostAgent: () => 'owner' };
        const card = artifactPartRenderer({ ref, title: '<svg onload=alert(1)>' }, {}, api);
        assert.equal(card.querySelector('strong').textContent, '<svg onload=alert(1)>');
        assert.equal(card.querySelectorAll('svg').length, 0);
        const invalid = artifactPartRenderer({ ref: { artifact_id: 'https://evil', revision_id: 'r' } }, {}, api);
        assert.equal(invalid.textContent, 'Invalid artifact reference');
    } finally { globalThis.document = before; dom.window.close(); }
});

test('editing textarea source retains CRLF and CR byte conventions', async () => {
    for (const ending of ['\r\n', '\r']) {
        const source = `line1${ending}line2${ending}`, item = await fixture({ content: source });
        let staged;
        item.client.stage = async (key, file, bytes) => { staged = bytes; return file; };
        edit(item.panel, item.dom.window, item.panel.element.querySelector('textarea').value + 'X');
        button(item.panel, 'Save revision').click(); await until(() => item.panel.draft.state === 'saved');
        assert.equal(new TextDecoder().decode(staged), source + 'X'); item.close();
    }
});

test('unknown save remains reconcilable after its conversation card is removed', async () => {
    const item = await fixture(), before = globalThis.document, beforeConfirm = globalThis.confirm;
    globalThis.document = item.dom.window.document;
    const { revision } = item.panel.draft, bytes = item.panel.draft.bytes;
    item.panel.dispose(); item.panel.element.remove();
    let publicationKey, checkedKey;
    try {
        const transport = {
            request: async (path, options = {}) => {
                if (path.endsWith('/staging')) {
                    const request = JSON.parse(options.body); publicationKey = request.publication_key;
                    const content = Uint8Array.from(atob(request.content_base64), c => c.charCodeAt(0));
                    return { schema_version: 1, path: request.path, media_type: request.media_type,
                        size_bytes: content.length, sha256: await sha256(content) };
                }
                if (path.endsWith('/publications')) throw new Error('lost response');
                if (path.includes('/publications/')) { checkedKey = path.split('/').at(-1); return null; }
                return revision;
            }, readBytes: async () => bytes,
        };
        const api = { getHostAgent: () => 'agent', bindAgentRequests: async () => transport };
        const pane = item.dom.window.document.createElement('div'); item.dom.window.document.body.append(pane);
        const card = artifactPartRenderer({ ref, title: 'Notes' }, { paneElement: pane }, api); pane.append(card);
        card.querySelector('button').click(); await until(() => item.dom.window.document.querySelector('textarea'));
        const root = item.dom.window.document.querySelector('.artifact-panel');
        edit({ element: root }, item.dom.window, 'Unresolved draft');
        button({ element: root }, 'Save revision').click();
        await until(() => root.querySelector('[role="status"]').textContent.includes('unknown'));
        pane.replaceChildren(); await tick();
        button({ element: root }, 'Check save').click(); await until(() => checkedKey);
        assert.equal(checkedKey, publicationKey); assert.equal(root.querySelector('textarea').value, 'Unresolved draft');
        globalThis.confirm = () => true; button({ element: root }, 'Close').click(); await tick();
    } finally { globalThis.document = before; globalThis.confirm = beforeConfirm; item.close(); }
});

test('conversation replacement keeps a dirty artifact panel and its unload guard alive', async () => {
    const item = await fixture(); const before = globalThis.document, beforeObserver = globalThis.MutationObserver;
    const beforeConfirm = globalThis.confirm;
    globalThis.document = item.dom.window.document; globalThis.MutationObserver = item.dom.window.MutationObserver;
    const { revision } = item.panel.draft; const bytes = item.panel.draft.bytes;
    item.panel.dispose(); item.panel.element.remove();
    try {
        const transport = { request: async path => path.endsWith('/publications') ? Promise.reject(new Error('lost')) :
            path.endsWith('/staging') ? revision.manifest.files[0] : revision,
            readBytes: async () => bytes };
        const api = { getHostAgent: () => 'agent', bindAgentRequests: async () => transport };
        const pane = item.dom.window.document.createElement('div'); item.dom.window.document.body.append(pane);
        const card = artifactPartRenderer({ ref, title: 'Notes' }, { paneElement: pane, workspaceRoot: pane }, api); pane.append(card); await tick();
        card.querySelector('button').click(); await until(() => item.dom.window.document.querySelector('textarea'));
        const editor = item.dom.window.document.querySelector('textarea');
        editor.value = 'Retain after conversation switch'; editor.dispatchEvent(new item.dom.window.Event('input'));
        pane.replaceChildren(); await tick();
        assert.ok(editor.isConnected, 'artifact working copy must outlive its conversation card');
        assert.equal(editor.value, 'Retain after conversation switch');
        let guarded = false; const event = new item.dom.window.Event('beforeunload', { cancelable: true });
        item.dom.window.dispatchEvent(event); guarded = event.defaultPrevented;
        assert.equal(guarded, true, 'retained draft keeps its unload warning');
        const close = [...item.dom.window.document.querySelectorAll('button')].find(node => node.textContent === 'Close');
        globalThis.confirm = () => true; close.click(); await tick();
    } finally { globalThis.document = before; globalThis.MutationObserver = beforeObserver; globalThis.confirm = beforeConfirm; item.close(); }
});


test('a denied open can close and leave without a discard confirmation', async () => {
    let asked = 0;
    const item = await fixture({ getError: { status: 403 }, confirmDiscard: () => { asked++; return false; } });
    const event = new item.dom.window.Event('beforeunload', { cancelable: true });
    item.dom.window.dispatchEvent(event); assert.equal(event.defaultPrevented, false);
    button(item.panel, 'Close').click(); await tick();
    assert.equal(asked, 0); assert.equal(item.panel.element.isConnected, false); item.close();
});

test('a confirmed quota rejection keeps the editor and discard protection available', async () => {
    const item = await fixture({ confirmDiscard: () => false });
    item.client.commit = async () => { throw { status: 413, code: 'artifact_quota_exceeded' }; };
    edit(item.panel, item.dom.window, 'Retained draft'); button(item.panel, 'Save revision').click();
    await until(() => item.panel.draft.state === 'rejected');
    assert.equal(item.panel.element.querySelector('textarea').readOnly, false);
    assert.equal(button(item.panel, 'Download draft').hidden, false);
    const event = new item.dom.window.Event('beforeunload', { cancelable: true });
    item.dom.window.dispatchEvent(event); assert.equal(event.defaultPrevented, true);
    edit(item.panel, item.dom.window, 'Smaller draft'); assert.equal(button(item.panel, 'Save revision').disabled, false);
    item.close();
});

test('insecure-context artifact opening explains HTTPS or localhost without making a request', async () => {
    const original=Object.getOwnPropertyDescriptor(globalThis,'crypto'), cryptoValue=globalThis.crypto;
    for(const crypto of [{randomUUID:cryptoValue.randomUUID},{subtle:cryptoValue.subtle}]) {
        const item=await fixture();item.panel.dispose();item.panel.element.remove();let reads=0;
        item.client.get=async()=>{reads++;throw new Error('must not fetch');};
        try {
            Object.defineProperty(globalThis,'crypto',{value:crypto,configurable:true});
            const panel=createArtifactPanel({client:item.client,ref,document:item.dom.window.document});await panel.ready;
            assert.equal(panel.draft.state,'secure_context_required');assert.equal(reads,0);
            assert.match(panel.element.querySelector('[role="status"]').textContent,/require HTTPS or localhost/);
            assert.equal(panel.draft.hasUnsavedChanges,false);panel.dispose();
        } finally {Object.defineProperty(globalThis,'crypto',original);item.close();}
    }
});

test('deletion before commit does not claim that the draft was previously saved', async () => {
    const item=await fixture();
    item.client.commit=async()=>{throw {status:404,code:'artifact_not_found'};};
    item.client.get=async()=>{throw {status:404,code:'artifact_not_found'};};
    edit(item.panel,item.dom.window,'Never saved draft');button(item.panel,'Save revision').click();
    await until(()=>item.panel.draft.state==='erased');
    assert.equal(item.panel.element.querySelector('[role="status"]').textContent,'The artifact was deleted. This draft cannot be republished automatically.');
    assert.equal(button(item.panel,'Download draft').hidden,false);assert.equal(item.panel.draft.text,'Never saved draft');item.close();
});

test('downloads use an inert MIME type and disposal releases URLs and the unload guard', async () => {
    const create=URL.createObjectURL,revoke=URL.revokeObjectURL;const made=[],revoked=[];
    URL.createObjectURL=blob=>{const url=create(blob);made.push({url,blob});return url;};
    URL.revokeObjectURL=url=>{revoked.push(url);revoke(url);};
    try {
        for(const type of ['text/html','image/svg+xml','application/pdf','text/markdown']) {
            const listeners=new Set();
            const item=await fixture({type,setupWindow:window=>{
                const add=window.addEventListener.bind(window),remove=window.removeEventListener.bind(window);
                window.addEventListener=(name,listener,...options)=>{if(name==='beforeunload')listeners.add(listener);return add(name,listener,...options);};
                window.removeEventListener=(name,listener,...options)=>{if(name==='beforeunload')listeners.delete(listener);return remove(name,listener,...options);};
            }});item.dom.window.HTMLAnchorElement.prototype.click=function(){};
            assert.equal(listeners.size,1);
            try {
                const initial=made.length;button(item.panel,'Download source file').click();await until(()=>made.length>initial);
                assert.equal(made.at(-1).blob.type,'application/octet-stream');
                assert.deepEqual(new Uint8Array(await made.at(-1).blob.arrayBuffer()),item.panel.draft.bytes);
                if(item.panel.draft.editable) {
                    edit(item.panel,item.dom.window,'Private draft');const before=made.length;
                    button(item.panel,'Download draft').click();await until(()=>made.length>before);
                    assert.equal(made.at(-1).blob.type,'application/octet-stream');
                    const event=new item.dom.window.Event('beforeunload',{cancelable:true});item.dom.window.dispatchEvent(event);assert.equal(event.defaultPrevented,true);
                }
                item.panel.dispose();
                assert.equal(listeners.size,0,'closing releases the window-owned reference to the panel');
                for(const {url} of made.slice(initial))assert.ok(revoked.includes(url));
                const after=new item.dom.window.Event('beforeunload',{cancelable:true});item.dom.window.dispatchEvent(after);assert.equal(after.defaultPrevented,false);
            } finally {item.close();}
        }
    } finally {URL.createObjectURL=create;URL.revokeObjectURL=revoke;}
});

test('failed comparison removes stale rebase controls and offers a visible retry', async () => {
    const item=await fixture();const latest=structuredClone(item.panel.draft.revision);latest.ref.revision_id='latest';
    item.client.get=async()=>latest;item.client.head=async()=>latest;item.client.commit=async()=>{throw {status:409};};
    edit(item.panel,item.dom.window,'Keep draft');button(item.panel,'Save revision').click();await until(()=>item.panel.draft.state==='conflict');
    button(item.panel,'Compare latest').click();await until(()=>item.panel.draft.comparison);
    assert.equal(button(item.panel,'Use latest as save base').hidden,false);
    item.client.head=async()=>{throw new Error('unavailable');};button(item.panel,'Compare latest').click();await until(()=>item.panel.draft.state==='compare_failed');
    assert.match(item.panel.element.querySelector('[role="status"]').textContent,/could not be compared/);
    assert.equal(item.panel.element.querySelector('h4'),null);assert.equal(button(item.panel,'Use latest as save base').hidden,true);
    assert.equal(button(item.panel,'Compare latest').hidden,false);assert.equal(button(item.panel,'Download draft').hidden,false);
    item.client.head=async()=>latest;button(item.panel,'Compare latest').click();await until(()=>item.panel.draft.comparison);
    assert.equal(button(item.panel,'Use latest as save base').hidden,false);assert.equal(item.panel.draft.text,'Keep draft');item.close();
});
