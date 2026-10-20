import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { createArtifactPanel, artifactPartRenderer } from '../../kestrel_sovereign/static/js/artifact_panel.mjs';
import { publicationDigests, sha256 } from '../../kestrel_sovereign/static/js/artifact_client.mjs';

const ref = { schema_version: 1, artifact_id: 'artifact', revision_id: 'revision1' };
const scope = { schema_version: 1, tenant_id: 'tenant', principal_id: 'person', agent_id: 'agent' };
async function fixture({ content = '<script>alert(1)</script>', type = 'text/markdown', title = '<img src=x onerror=alert(1)>', confirmDiscard = () => true } = {}) {
    const dom = new JSDOM('<!doctype html><body></body>', { url: 'http://localhost/' });
    const bytes = new TextEncoder().encode(content);
    const file = { schema_version: 1, path: 'source.md', media_type: type, sha256: await sha256(bytes), size_bytes: bytes.length };
    const revision = { schema_version: 1, ref, scope, title,
        manifest: { schema_version: 1, files: [file], entrypoint: file.path },
        provenance: { schema_version: 1, session_id: null, run_id: null, tool_name: null }, committed_at: '2026-10-20T12:00:00Z' };
    const client = {
        get: async () => structuredClone(revision), read: async () => bytes,
        stage: async (key, file) => file,
        commit: async request => ({ schema_version: 1, ref: { ...ref, revision_id: 'revision2' }, scope,
            idempotency_key: request.idempotency_key, ...await publicationDigests(request), committed_at: revision.committed_at, storage_id: 'local' }),
        status: async () => null, history: async () => ({ items: [revision], next_cursor: null }),
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

test('SVG and HTML have no active preview and PDF uses a canvas without active document elements', async () => {
    for (const type of ['image/svg+xml', 'text/html', 'application/pdf']) {
        const item = await fixture({ type });
        assert.equal(item.panel.element.querySelectorAll('script,object,embed,img,iframe').length, 0);
        assert.equal(!!item.panel.element.querySelector('canvas'), type === 'application/pdf');
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
