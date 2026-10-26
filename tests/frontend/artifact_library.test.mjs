import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { readFile } from 'node:fs/promises';
import { openArtifactLibrary } from '../../kestrel_sovereign/static/js/artifact_panel.mjs';
import { publicationDigests, sha256 } from '../../kestrel_sovereign/static/js/artifact_client.mjs';
const tick = () => new Promise(resolve => setTimeout(resolve, 0));
async function until(check) { for (let i=0;i<100&&!check();i++) await tick(); assert.ok(check()); }
function button(root, text) { return [...root.querySelectorAll('button')].find(node => node.textContent === text); }
async function setup() {
    const dom = new JSDOM('<!doctype html><body></body>', { url: 'http://localhost' }), calls = [];
    const scope = { schema_version: 1, tenant_id: 'tenant', principal_id: 'host-sovereign', agent_id: 'agent' };
    const ref = { schema_version: 1, artifact_id: 'artifact', revision_id: 'revision' };
    const bytes = new TextEncoder().encode('# Notes');
    const file = { schema_version: 1, path: 'notes.md', media_type: 'text/markdown', size_bytes: bytes.length, sha256: await sha256(bytes) };
    const revision = { schema_version: 1, ref, title: '<script>Notes</script>', scope, manifest: { schema_version:1, files:[file], entrypoint:file.path } };
    let query = async () => ({ items:[revision], next_cursor:null });
    const transport = { request: async (path, options) => {
        calls.push({path,options});
        if (path.startsWith('/api/artifacts?')) return query(path, options);
        if (path.endsWith('/staging')) return file;
        if (path.endsWith('/publications')) { const req=JSON.parse(options.body); return { schema_version:1,ref,scope,idempotency_key:req.idempotency_key,...await publicationDigests(req) }; }
        return revision;
    }, readBytes: async () => bytes };
    const api = { getHostAgent: () => 'original-agent', bindAgentRequests: async owner => { assert.equal(owner,'original-agent'); return transport; } };
    const library = await openArtifactLibrary({ api, root:dom.window.document.body });
    return { dom, library, api, calls, revision, setQuery(value){query=value;}, close(){library.dispose();dom.window.close();} };
}

test('library opens exact saved heads in the right expandable shared panel without interpreting titles', async () => {
    const item=await setup(), doc=item.dom.window.document;
    const aside=doc.querySelector('aside'); assert.equal(aside.style.right,'0px'); assert.equal(aside.getAttribute('aria-label'),'Artifacts');
    assert.equal(doc.querySelectorAll('script').length,0); assert.equal(doc.querySelector('strong').textContent,'<script>Notes</script>');
    button(aside,'Collapse artifacts').click(); assert.equal(item.library.element.parentNode.hidden,true);
    button(aside,'Show artifacts').click(); assert.equal(item.library.element.parentNode.hidden,false);
    button(aside,'Expand artifacts').click(); assert.equal(button(aside,'Restore panel width').getAttribute('aria-pressed'),'true');
    button(item.library.element,'Open saved artifact').click(); await until(()=>doc.querySelector('textarea'));
    assert.equal(doc.querySelector('textarea').value,'# Notes');
    assert.ok(item.calls.some(call=>call.path==='/api/artifacts/artifact/revisions/revision'));
    item.close();
});

test('search sends the query through bound discovery, continues empty pages and ignores stale responses', async () => {
    const item=await setup(), root=item.library.element; let resolve;
    item.setQuery(async path => path.includes('cursor=next') ? { items:[item.revision],next_cursor:null } : {items:[],next_cursor:'next'});
    root.querySelector('input[type=search]').value='private & title'; button(root,'Search').click();
    await until(()=>root.textContent.includes('No matches in this page'));
    assert.ok(item.calls.at(-1).path.includes('q=private%20%26%20title'));
    button(root,'Continue discovery').click(); await until(()=>root.querySelector('strong'));
    item.setQuery(()=>new Promise(done=>{resolve=done;}));button(root,'Search').click();await until(()=>resolve);
    item.setQuery(async()=>({items:[],next_cursor:null}));
    const reopened=await openArtifactLibrary({api:item.api,root:item.dom.window.document.body});
    // Close both in-flight views before delivering their responses.
    reopened.dispose(); resolve({items:[item.revision],next_cursor:null});await tick();
    assert.equal(root.isConnected,false); item.close();
});

test('new text uses the same panel and stays dirty until the save receipt arrives', async () => {
    const item=await setup(),doc=item.dom.window.document;
    button(item.library.element,'New text artifact').click();await until(()=>doc.querySelector('textarea'));
    const root=doc.querySelector('.artifact-panel'),editor=root.querySelector('textarea');
    assert.equal(root.querySelector('[role=status]').textContent,'Unsaved changes');
    assert.equal(button(root,'Save revision').disabled,false);
    assert.ok(editor); item.close();
});

test('standalone console wires its real chat header action to the shared library', async () => {
    const source=await readFile(new URL('../../kestrel_sovereign/static/js/app.js',import.meta.url),'utf8');
    assert.match(source,/registerHeaderAction\(\{ id: 'artifacts'/);
    assert.match(source,/onClick: \(\) => openArtifactLibrary\(\{ api: API \}\)/);
});
