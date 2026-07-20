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
const { Modal, Toast } = await import('../../kestrel_sovereign/static/js/ui.js');
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
