/** Shared artifact card/viewer. All producer-controlled values use DOM text. */
import { ArtifactDraft, artifactRef, createArtifactClient } from './artifact_client.mjs';

const STATUS = {
    loading: 'Opening saved revision…', saved: 'Saved revision', dirty: 'Unsaved changes',
    saving: 'Saving…', unknown: 'Save status is unknown. Check or retry the same save.',
    conflict: 'Another revision conflicts with this draft. Compare before saving again.',
    compare_failed: 'The current revision could not be compared. Your draft is retained; try again or download it.',
    denied: 'Access is unavailable. Reopen after authentication or permissions change.',
    erased: 'The artifact was deleted. This draft cannot be republished automatically.',
    failed: 'The draft cannot be saved. Keep text edits within 1 MiB.',
    rejected: 'The save was rejected. Your draft is retained; edit and try again, or download it.',
    unavailable: 'This revision is unavailable or cannot be verified for viewing.',
    secure_context_required: 'Artifact viewing and editing require HTTPS or localhost.',
};
const IMAGES = new Set(['image/png', 'image/jpeg', 'image/gif', 'image/webp']);
const workspaces = new WeakMap();

function element(doc, tag, text, className) {
    const node = doc.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
}

export function createArtifactPanel({ client, ref, document: doc = globalThis.document,
    onOpenRevision = () => {}, onClose = () => {}, confirmDiscard = () => globalThis.confirm('Discard the local draft and any unresolved save state?') }) {
    const root = element(doc, 'section', undefined, 'artifact-panel');
    root.setAttribute('aria-label', 'Saved artifact');
    const title = element(doc, 'h3', 'Artifact');
    const identity = element(doc, 'p', '', 'artifact-revision text-muted');
    const status = element(doc, 'p', STATUS.loading); status.setAttribute('role', 'status');
    const toolbar = element(doc, 'div', undefined, 'row-flex');
    const viewer = element(doc, 'div');
    const comparison = element(doc, 'div');
    const history = element(doc, 'div');
    const note = element(doc, 'p', 'Drafts stay in this panel until saved. Download a draft before closing if its save is unresolved.', 'text-muted');
    root.append(title, identity, status, toolbar, viewer, comparison, history, note);
    let disposed = false, previewURL = null, shownFile = null, editor = null, comparisonRef = null;
    const urls = new Set();
    const objectURL = (bytes, type) => {
        const url = URL.createObjectURL(new Blob([bytes], { type })); urls.add(url); return url;
    };
    const releasePreview = () => {
        if (previewURL) { URL.revokeObjectURL(previewURL); urls.delete(previewURL); previewURL = null; }
    };
    const download = (bytes, name) => {
        if (!bytes || disposed) return;
        const link = element(doc, 'a');
        link.href = objectURL(bytes, 'application/octet-stream'); link.download = name;
        root.append(link); link.click(); link.remove();
        // Keep the URL only for this panel's lifetime; revoke on disposal.
    };
    const button = (label, handler) => {
        const node = element(doc, 'button', label, 'btn btn-secondary'); node.type = 'button';
        node.addEventListener('click', () => Promise.resolve().then(handler).catch(() => {
            if (!disposed) status.textContent = 'The operation could not finish. Your draft remains in this panel.';
        }));
        toolbar.append(node); return node;
    };
    const draft = new ArtifactDraft(client, render);
    const save = button('Save revision', () => draft.save());
    const check = button('Check save', () => draft.reconcile());
    const retry = button('Retry same save', () => draft.retry());
    const compare = button('Compare latest', () => draft.compareLatest());
    const rebase = button('Use latest as save base', () => draft.useComparedRevision());
    const exportSaved = button('Download source file', () => download(draft.bytes, draft.file?.path.split('/').pop()));
    const exportDraft = button('Download draft', () => download(new TextEncoder().encode(draft.text), `draft-${draft.file?.path.split('/').pop() || 'notes.txt'}`));
    const historyButton = button('Revision history', async () => {
        const page = await draft.history();
        if (disposed) return;
        history.replaceChildren(element(doc, 'h4', 'Saved revisions'));
        showHistory(page);
    });
    const close = button('Close', () => {
        if (draft.hasUnsavedChanges && !confirmDiscard()) return;
        dispose(); root.remove(); onClose();
    });
    function showHistory(page) {
        for (const revision of page.items || []) {
            const ref = artifactRef(revision.ref);
            if (ref.artifact_id !== draft.revision?.ref.artifact_id) continue;
            const item = element(doc, 'button', `${revision.committed_at} · ${ref.revision_id}`, 'btn btn-secondary');
            item.type = 'button';
            item.addEventListener('click', () => onOpenRevision(ref)); history.append(item);
        }
        if (page.next_cursor) {
            const more = element(doc, 'button', 'More revisions', 'btn btn-secondary'); more.type = 'button';
            more.addEventListener('click', async () => {
                more.disabled = true;
                try {
                    const next = await draft.history(page.next_cursor);
                    if (!disposed) { more.remove(); showHistory(next); }
                } catch (_) { if (!disposed) more.disabled = false; }
            }); history.append(more);
        }
    }
    function render() {
        if (disposed) return;
        status.textContent = STATUS[draft.state] || 'Artifact unavailable';
        title.textContent = draft.revision?.title || 'Artifact';
        identity.textContent = draft.revision ? `Revision ${draft.revision.ref.revision_id}` : '';
        save.disabled = draft.state !== 'dirty';
        check.hidden = !['unknown', 'conflict', 'compare_failed'].includes(draft.state);
        retry.hidden = draft.state !== 'unknown'; compare.hidden = !['conflict', 'compare_failed'].includes(draft.state);
        rebase.hidden = !draft.comparison || draft.state !== 'conflict';
        exportSaved.disabled = !draft.bytes;
        exportDraft.hidden = !draft.editable || !draft.hasUnsavedChanges;
        historyButton.disabled = !draft.revision || draft.state === 'saving';
        if (draft.file && draft.file !== shownFile) {
            releasePreview(); viewer.replaceChildren(); editor = null; shownFile = draft.file;
            if (draft.editable) {
                editor = element(doc, 'textarea'); editor.value = draft.text;
                editor.setAttribute('aria-label', 'Artifact source'); editor.spellcheck = false;
                editor.style.cssText = 'width:100%;min-height:18rem;box-sizing:border-box;font-family:monospace;';
                editor.addEventListener('input', () => draft.edit(editor.value)); viewer.append(editor);
            } else if (IMAGES.has(draft.file.media_type)) {
                previewURL = objectURL(draft.bytes, draft.file.media_type);
                const image = element(doc, 'img'); image.alt = draft.revision.title; image.src = previewURL;
                image.style.maxWidth = '100%'; viewer.append(image);
            } else if (draft.file.media_type === 'application/pdf') {
                viewer.append(element(doc, 'p', 'Download this PDF to view it.'));
            } else if (draft.text) {
                viewer.append(element(doc, 'p', 'This source has mixed line endings or belongs to a bundle. It is read-only here.'),
                    element(doc, 'pre', draft.text));
            } else {
                viewer.append(element(doc, 'p', 'Preview is unavailable for this file or bundle. Download the saved source to open it.'));
            }
        }
        if (editor) editor.readOnly = !['saved', 'dirty', 'failed', 'rejected'].includes(draft.state) || !!draft.pendingKey;
        if (draft.comparison?.revision.ref.revision_id !== comparisonRef) {
            comparison.replaceChildren(); comparisonRef = draft.comparison?.revision.ref.revision_id;
            if (draft.comparison) {
                comparison.append(element(doc, 'h4', 'Latest saved source — your draft is kept above'),
                    element(doc, 'pre', draft.comparison.text));
            }
        }
    }
    function beforeUnload(event) {
        if (draft.hasUnsavedChanges) { event.preventDefault(); event.returnValue = ''; }
    }
    doc.defaultView?.addEventListener('beforeunload', beforeUnload);
    function dispose() {
        if (disposed) return;
        disposed = true; draft.close();
        releasePreview();
        doc.defaultView?.removeEventListener('beforeunload', beforeUnload);
        for (const url of urls) URL.revokeObjectURL(url);
        urls.clear(); viewer.replaceChildren(); comparison.replaceChildren(); history.replaceChildren();
    }
    render();
    const ready = draft.open(ref);
    return { element: root, draft, ready, dispose };
}

/** Open working copies belong to the app workspace, independently of chat cards. */
export function openArtifactPanel({ client, ref, owner, root = globalThis.document.body }) {
    const doc = root.ownerDocument;
    let workspace = workspaces.get(root);
    if (!workspace || !workspace.element.isConnected) {
        const container = element(doc, 'aside', undefined, 'artifact-workspace');
        container.setAttribute('aria-label', 'Open artifacts');
        container.style.cssText = 'position:fixed;right:1rem;top:4rem;width:min(44rem,calc(100vw - 2rem));max-height:calc(100vh - 5rem);overflow:auto;z-index:1500;padding:1rem;background:var(--bg-secondary);color:var(--text-primary);border:1px solid var(--border-color);border-radius:12px;';
        const heading = element(doc, 'h2', 'Open artifacts'), list = element(doc, 'div');
        const toggle = element(doc, 'button', 'Collapse artifacts', 'btn btn-secondary'); toggle.type = 'button';
        toggle.addEventListener('click', () => { list.hidden = !list.hidden; toggle.textContent = list.hidden ? 'Show artifacts' : 'Collapse artifacts'; });
        container.append(heading, toggle, list); root.append(container);
        workspace = { element: container, list, toggle }; workspaces.set(root, workspace);
    }
    workspace.list.hidden = false; workspace.toggle.textContent = 'Collapse artifacts';
    const panel = createArtifactPanel({ client, ref, document: doc,
        onOpenRevision: next => openArtifactPanel({ client, ref: next, owner, root }),
        onClose: () => {
            if (!workspace.list.childElementCount) { workspace.element.remove(); workspaces.delete(root); }
        } });
    if (owner) panel.element.prepend(element(doc, 'p', `Agent: ${owner}`, 'text-muted'));
    workspace.list.append(panel.element); panel.element.scrollIntoView?.({ block: 'nearest' });
    return panel;
}

export function artifactPartRenderer(data, context = {}, api) {
    const doc = globalThis.document;
    const card = element(doc, 'article', undefined, 'artifact-card');
    let ref;
    try { ref = artifactRef(data?.ref); } catch (_) { card.textContent = 'Invalid artifact reference'; return card; }
    const title = element(doc, 'strong', typeof data.title === 'string' ? data.title : 'Saved artifact');
    const open = element(doc, 'button', 'Open saved artifact', 'btn btn-secondary'); open.type = 'button';
    card.append(title, open);
    const owner = context.ownerAgent === undefined ? api.getHostAgent() : context.ownerAgent;
    let opening = false;
    async function show(ref) {
        if (opening) return;
        opening = true; open.disabled = true;
        try {
            const client = await createArtifactClient(api, owner);
            // The overlay root is supplied by the embedder (the standalone
            // console uses body). Never let a chat transcript own the draft.
            const candidate = context.workspaceRoot;
            const root = candidate?.isConnected && !context.paneElement?.contains(candidate) ? candidate : doc.body;
            const panel = openArtifactPanel({ client, ref, owner, root });
            await panel.ready;
        } catch (error) {
            open.textContent = error.code === 'artifact_auth_unbound'
                ? 'Artifact access requires an API key or bearer credential' : 'Artifact access is unavailable';
        }
        finally { opening = false; open.disabled = false; }
    }
    open.addEventListener('click', () => show(ref));
    return card;
}
