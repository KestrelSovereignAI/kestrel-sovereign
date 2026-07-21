import test, { afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const dom = new JSDOM('<!doctype html><html><head></head><body></body></html>', {
    url: 'http://localhost/',
});
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.Node = dom.window.Node;
globalThis.HTMLElement = dom.window.HTMLElement;
globalThis.location = dom.window.location;
globalThis.sessionStorage = dom.window.sessionStorage;
globalThis.localStorage = dom.window.localStorage;
Object.defineProperty(globalThis, 'navigator', {
    value: dom.window.navigator,
    configurable: true,
});
globalThis.fetch = async () => ({ ok: false, status: 500 });
globalThis.CSS = dom.window.CSS || { escape: (value) => String(value) };
globalThis.window.kicon = (name) => `<span class="ki ki-${name}"></span>`;
globalThis.window.KI_PATHS = {};
globalThis.kicon = globalThis.window.kicon;
globalThis.window.SharedMarkdown = {
    renderMarkdown: () => '',
    renderStreamingMarkdown: () => '',
    highlightCodeBlocks: () => {},
    renderMermaidDiagrams: () => {},
    finalizeMarkdown: async () => {},
};

const API = (await import('../../kestrel_sovereign/static/js/api.js')).default;
const { Modal, Toast, setOverlayRoot } = await import('../../kestrel_sovereign/static/js/ui.js');
await import('../../kestrel_sovereign/static/js/files.js');
await import('../../kestrel_sovereign/static/js/feature-store.js');
const chat = await import('../../kestrel_sovereign/static/js/chat.js');
const sovereignty = await import('../../kestrel_sovereign/static/js/sovereignty.js');

function deferred() {
    let resolve;
    let reject;
    const promise = new Promise((res, rej) => {
        resolve = res;
        reject = rej;
    });
    return { promise, resolve, reject };
}

function showProtectedReplacement() {
    let closes = 0;
    Modal.show({
        title: 'Permission Required',
        content: '<p id="protected-replacement">approval</p>',
        onClose: () => { closes += 1; },
    });
    return () => closes;
}

function assertProtectedReplacement(getCloses) {
    assert.ok(document.getElementById('protected-replacement'));
    assert.equal(getCloses(), 0, 'stale async work did not close or replace the protected dialog');
}

function tick() {
    return new Promise((resolve) => setTimeout(resolve, 0));
}

function delay(milliseconds) {
    return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

afterEach(() => {
    Modal.hide();
    setOverlayRoot(null);
    document.body.replaceChildren();
});

test('a delayed real file preview cannot replace a newer modal', async () => {
    const originalPreview = API.getSovereigntyFilePreview;
    const request = deferred();
    API.getSovereigntyFilePreview = () => request.promise;
    try {
        const previewing = window.previewFile('late.txt');
        assert.equal(document.querySelector('.modal-header h3').textContent, 'File Preview');
        const getCloses = showProtectedReplacement();

        request.resolve({
            filename: 'late.txt',
            size: 4,
            content_type: 'text/plain',
            is_text: true,
            content: 'late',
            truncated: false,
        });
        await previewing;

        assertProtectedReplacement(getCloses);
    } finally {
        API.getSovereigntyFilePreview = originalPreview;
    }
});

test('a delayed real feature detail response cannot replace a newer modal', async () => {
    const originalRequest = API.request;
    const request = deferred();
    API.request = () => request.promise;
    try {
        const loading = window.FeatureStore.showDetail('delayed');
        assert.equal(document.querySelector('.modal-header h3').textContent, 'Feature Details');
        const getCloses = showProtectedReplacement();

        request.resolve({
            name: 'delayed',
            description: 'Loaded too late',
            status: 'available',
        });
        await loading;

        assertProtectedReplacement(getCloses);
    } finally {
        API.request = originalRequest;
    }
});

test('a delayed real feature config save cannot close a newer modal', async () => {
    const originalRequest = API.request;
    const saveRequest = deferred();
    API.request = (path, options = {}) => {
        if (options.method === 'PATCH') return saveRequest.promise;
        assert.match(path, /\/config$/);
        return Promise.resolve({
            config_schema: {
                properties: {
                    enabled: { type: 'boolean', title: 'Enabled' },
                },
            },
            config: { enabled: true },
        });
    };
    try {
        await window.FeatureStore.showConfigForm('configurable');
        const save = [...document.querySelectorAll('.modal-btn')]
            .find((button) => button.textContent === 'Save');
        assert.ok(save, 'the real configuration modal rendered its Save action');
        save.click();

        const getCloses = showProtectedReplacement();
        saveRequest.resolve({ ok: true });
        await tick();

        assertProtectedReplacement(getCloses);
    } finally {
        API.request = originalRequest;
    }
});

test('a delayed real context breakdown cannot replace a newer modal', async () => {
    const request = deferred();
    chat.setChatDeps({
        state: { currentSessionId: 'session-2648' },
        api: { getContextStatus: () => request.promise },
    });
    try {
        const loading = window.openContextBreakdownPopup();
        assert.equal(document.querySelector('.modal-body').textContent.trim(), 'Loading…');
        const getCloses = showProtectedReplacement();

        request.resolve({ breakdown: null, compaction_recommended: false });
        await loading;

        assertProtectedReplacement(getCloses);
    } finally {
        chat.setChatDeps({ state: null, api: null });
    }
});

test('a delayed clipboard rejection cannot paint into a replacement modal lifecycle', async () => {
    const clipboardRead = deferred();
    Object.defineProperty(navigator, 'clipboard', {
        value: { readText: () => clipboardRead.promise },
        configurable: true,
    });
    const originalToastError = Toast.error;
    const errors = [];
    Toast.error = (message) => errors.push(message);
    try {
        document.body.innerHTML = '<button id="btn-import">Import</button>';
        sovereignty.initSovereigntyButtons();
        document.getElementById('btn-import').click();
        await delay(60);

        document.getElementById('paste-cid-btn').click();
        const getCloses = showProtectedReplacement();
        clipboardRead.reject(new Error('clipboard denied too late'));
        await tick();

        assertProtectedReplacement(getCloses);
        assert.deepEqual(errors, [], 'the stale import lifecycle cannot emit a clipboard toast');
    } finally {
        Toast.error = originalToastError;
        delete navigator.clipboard;
    }
});

test('IME composition Enter does not submit the sovereignty import dialog', async () => {
    const originalImport = API.importSovereignty;
    const originalGetExports = API.getSovereigntyExports;
    const imports = [];
    API.importSovereignty = async (cid) => {
        imports.push(cid);
        return { message: 'Imported' };
    };
    API.getSovereigntyExports = async () => ({ exports: [] });
    try {
        document.body.innerHTML = '<button id="btn-import">Import</button><div id="exports-list"></div>';
        sovereignty.initSovereigntyButtons();
        document.getElementById('btn-import').click();
        await delay(60);
        const input = document.getElementById('import-cid-input');
        input.value = 'bafycomposition';

        input.dispatchEvent(new dom.window.KeyboardEvent('keydown', {
            key: 'Enter',
            isComposing: true,
            bubbles: true,
            cancelable: true,
        }));
        await tick();
        assert.deepEqual(imports, []);
        assert.ok(document.querySelector('.modal-overlay'));

        input.dispatchEvent(new dom.window.KeyboardEvent('keydown', {
            key: 'Enter',
            bubbles: true,
            cancelable: true,
        }));
        await tick();
        assert.deepEqual(imports, ['bafycomposition']);
        assert.equal(document.querySelector('.modal-overlay'), null);
    } finally {
        API.importSovereignty = originalImport;
        API.getSovereigntyExports = originalGetExports;
    }
});

test('a validated import still starts when modal focus restoration fails', async () => {
    const originalImport = API.importSovereignty;
    const originalGetExports = API.getSovereigntyExports;
    const imports = [];
    API.importSovereignty = async (cid) => {
        imports.push(cid);
        return { message: 'Imported' };
    };
    API.getSovereigntyExports = async () => ({ exports: [] });
    try {
        document.body.innerHTML = '<button id="btn-import">Import</button><div id="exports-list"></div>';
        sovereignty.initSovereigntyButtons();
        const opener = document.getElementById('btn-import');
        opener.focus();
        opener.click();
        await delay(60);
        document.getElementById('import-cid-input').value = 'bafyclosefailure';
        opener.focus = () => { throw new Error('import opener focus failed'); };

        const reportedError = new Promise((resolve) => {
            window.addEventListener('error', (event) => {
                event.preventDefault();
                resolve(event.error);
            }, { once: true });
        });
        [...document.querySelectorAll('.modal-btn')]
            .find((button) => button.textContent === 'Import').click();

        assert.match((await reportedError).message, /import opener focus failed/);
        assert.deepEqual(imports, ['bafyclosefailure'],
            'the validated operation is not lost after modal teardown');
        assert.equal(document.querySelector('.modal-overlay'), null);
    } finally {
        API.importSovereignty = originalImport;
        API.getSovereigntyExports = originalGetExports;
    }
});

test('sovereignty import and export read controls inside a closed shadow overlay', async () => {
    const originalExport = API.exportSovereignty;
    const originalImport = API.importSovereignty;
    const originalGetExports = API.getSovereigntyExports;
    const exports = [];
    const imports = [];
    API.exportSovereignty = async (tier, encrypt) => {
        exports.push({ tier, encrypt });
        return { message: 'Exported' };
    };
    API.importSovereignty = async (cid) => {
        imports.push(cid);
        return { message: 'Imported' };
    };
    API.getSovereigntyExports = async () => ({ exports: [] });

    const host = document.createElement('div');
    const shadow = host.attachShadow({ mode: 'closed' });
    const mount = document.createElement('div');
    shadow.appendChild(mount);
    document.body.appendChild(host);
    setOverlayRoot(mount);
    try {
        document.body.insertAdjacentHTML('beforeend', `
            <button id="btn-export-ipfs">Export</button>
            <button id="btn-import">Import</button>
            <div id="export-list"></div>
        `);
        sovereignty.initSovereigntyButtons();

        document.getElementById('btn-export-ipfs').click();
        const filecoin = mount.querySelector('input[value="FILECOIN"]');
        const encrypt = mount.querySelector('#export-encrypt');
        filecoin.checked = true;
        encrypt.checked = false;
        [...mount.querySelectorAll('.modal-btn')]
            .find((button) => button.textContent === 'Export').click();
        await tick();
        assert.deepEqual(exports, [{ tier: 'FILECOIN', encrypt: false }]);

        document.getElementById('btn-import').click();
        await delay(60);
        const cidInput = mount.querySelector('#import-cid-input');
        cidInput.value = 'bafyshadow';
        [...mount.querySelectorAll('.modal-btn')]
            .find((button) => button.textContent === 'Import').click();
        await tick();
        assert.deepEqual(imports, ['bafyshadow']);
    } finally {
        API.exportSovereignty = originalExport;
        API.importSovereignty = originalImport;
        API.getSovereigntyExports = originalGetExports;
        Modal.hide();
        setOverlayRoot(null);
        host.remove();
    }
});

test('feature configuration saves fields from a closed shadow overlay', async () => {
    const originalRequest = API.request;
    const requests = [];
    API.request = async (path, options = {}) => {
        requests.push({ path, options });
        if (options.method === 'PATCH') return { ok: true };
        return {
            config_schema: {
                properties: {
                    label: { type: 'string', title: 'Label' },
                },
            },
            config: { label: 'before' },
        };
    };

    const host = document.createElement('div');
    const shadow = host.attachShadow({ mode: 'closed' });
    const mount = document.createElement('div');
    shadow.appendChild(mount);
    document.body.appendChild(host);
    setOverlayRoot(mount);
    try {
        await window.FeatureStore.showConfigForm('shadow-config');
        const field = mount.querySelector('[data-config-key="label"]');
        field.value = 'after';
        [...mount.querySelectorAll('.modal-btn')]
            .find((button) => button.textContent === 'Save').click();
        await tick();

        const patch = requests.find(({ options }) => options.method === 'PATCH');
        assert.deepEqual(JSON.parse(patch.options.body), { config: { label: 'after' } });
        assert.equal(mount.querySelector('.modal-overlay'), null);
    } finally {
        API.request = originalRequest;
        Modal.hide();
        setOverlayRoot(null);
        host.remove();
    }
});
