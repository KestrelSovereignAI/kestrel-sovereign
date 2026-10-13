/**
 * Sovereign Console export tiers against the endpoint's contract (#2876).
 *
 * The export modal sent the StorageTier enum names (LOCAL_ONLY, IPFS,
 * FILECOIN), and POST /api/sovereignty/export accepts only its ALLOWED_TIERS,
 * so every console export failed with HTTP 400. The tests that covered the
 * modal mocked the API and kept the wrong spelling. These submit the real
 * modal through the real API client and check each request body against
 * ALLOWED_TIERS as the endpoint itself defines it.
 */
import test, { afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { JSDOM } from 'jsdom';

const ENDPOINT_SOURCE = new URL('../../kestrel_sovereign/endpoints/sovereignty.py', import.meta.url);

// The tiers POST /api/sovereignty/export accepts, read from the endpoint's own
// ALLOWED_TIERS literal rather than copied here, so the console and the
// endpoint cannot drift apart unnoticed again. A definition this reader does
// not understand fails the tests instead of being skipped.
function readEndpointAllowedTiers() {
    const source = readFileSync(ENDPOINT_SOURCE, 'utf8');
    const match = source.match(/^ALLOWED_TIERS\s*=\s*\{([^}]*)\}[ \t]*(?:#.*)?$/m);
    assert.ok(match, 'kestrel_sovereign/endpoints/sovereignty.py no longer defines '
        + 'ALLOWED_TIERS as a set literal; update this reader to match it');
    const tiers = match[1].split(',').map((item) => item.trim()).filter(Boolean).map((item) => {
        const quoted = item.match(/^(["'])([^"'\\]+)\1$/);
        assert.ok(quoted, `ALLOWED_TIERS item ${item} is not a plain string literal; `
            + 'update this reader to match it');
        return quoted[2];
    });
    assert.ok(tiers.length > 0, 'ALLOWED_TIERS is empty');
    return new Set(tiers);
}

const ALLOWED_TIERS = readEndpointAllowedTiers();

function jsonResponse(status, body) {
    return {
        ok: status >= 200 && status < 300,
        status,
        statusText: `HTTP ${status}`,
        async json() {
            return body;
        },
    };
}

// Answers the two sovereignty routes the modal reaches the way the endpoint
// does: a tier outside ALLOWED_TIERS is an HTTP 400.
const exportBodies = [];
const unexpectedRequests = [];
async function sovereigntyEndpoint(url, options = {}) {
    if (url === '/api/sovereignty/export' && options.method === 'POST') {
        const body = JSON.parse(options.body);
        exportBodies.push(body);
        if (!ALLOWED_TIERS.has(body.tier)) {
            return jsonResponse(400, {
                detail: `Invalid tier '${body.tier}'. Must be one of: ${[...ALLOWED_TIERS].sort().join(', ')}`,
            });
        }
        return jsonResponse(200, { success: true, message: `Exported to ${body.tier}` });
    }
    if (url === '/api/sovereignty/exports') {
        return jsonResponse(200, { exports: [] });
    }
    unexpectedRequests.push(url);
    return jsonResponse(404, { detail: 'Not Found' });
}

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
globalThis.fetch = sovereigntyEndpoint;
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
// The console's own API client (api.js), configured before it loads, so the
// modal's request goes through the real exportSovereignty to this endpoint.
globalThis.KESTREL_UI_CONFIG = {
    bearerToken: { getToken: () => 'test-token' },
    fetchFn: sovereigntyEndpoint,
};

const {
    createApiClient,
    createBearerTokenAuthProvider,
    normalizeSovereigntyExportTier,
} = await import('../../kestrel_sovereign/static/js/api_client.mjs');
const { Modal, Toast } = await import('../../kestrel_sovereign/static/js/ui.js');
const sovereignty = await import('../../kestrel_sovereign/static/js/sovereignty.js');

afterEach(() => {
    Modal.hide();
    document.body.replaceChildren();
    exportBodies.length = 0;
    unexpectedRequests.length = 0;
});

function mountSovereigntyPanel() {
    document.body.insertAdjacentHTML('beforeend', `
        <button id="btn-export-ipfs">Export</button>
        <button id="btn-import">Import</button>
        <div id="export-list"></div>
    `);
    sovereignty.initSovereigntyButtons();
}

function tierRadios() {
    return [...document.querySelectorAll('input[name="export-tier"]')];
}

function withTimeout(promise, message, milliseconds = 5000) {
    let timer;
    const timeout = new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(message)), milliseconds);
    });
    return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

// Open the export modal, let `choose` set its controls, press Export, and
// return the toast the console showed once the export settled.
async function submitExportModal(choose) {
    const originalSuccess = Toast.success;
    const originalError = Toast.error;
    let settle;
    const settled = new Promise((resolve) => { settle = resolve; });
    Toast.success = (message) => settle({ kind: 'success', message });
    Toast.error = (message) => settle({ kind: 'error', message });
    try {
        document.getElementById('btn-export-ipfs').click();
        choose();
        [...document.querySelectorAll('.modal-btn')]
            .find((button) => button.textContent === 'Export').click();
        const toast = await withTimeout(settled, 'the export reported no outcome');
        // A successful export refreshes the export list. The endpoint answers
        // without timers, so one macrotask lets that refresh finish before the
        // next test replaces the DOM.
        await new Promise((resolve) => setTimeout(resolve, 0));
        if (toast.kind === 'success') {
            assert.match(document.getElementById('export-list').textContent, /No exports yet/);
        }
        return toast;
    } finally {
        Toast.success = originalSuccess;
        Toast.error = originalError;
    }
}

test('the endpoint contract reader discriminates the old enum spellings', () => {
    // Guards the reader itself: were it to return everything, every
    // membership check below would pass vacuously.
    for (const enumName of ['LOCAL_ONLY', 'local_only', 'IPFS', 'FILECOIN']) {
        assert.equal(ALLOWED_TIERS.has(enumName), false, `${enumName} is not an endpoint tier`);
    }
});

test('every export modal option submits a tier the endpoint accepts', async () => {
    mountSovereigntyPanel();
    document.getElementById('btn-export-ipfs').click();
    const offered = tierRadios().map((radio) => radio.value);
    Modal.hide();
    assert.ok(offered.length > 0, 'the export modal offers no storage tier');

    for (const value of offered) {
        exportBodies.length = 0;
        const toast = await submitExportModal(() => {
            tierRadios().find((radio) => radio.value === value).checked = true;
        });

        assert.equal(exportBodies.length, 1, `choosing ${value} posted one export`);
        const { tier } = exportBodies[0];
        assert.ok(ALLOWED_TIERS.has(tier),
            `choosing ${value} sent tier ${tier}, outside ALLOWED_TIERS ${[...ALLOWED_TIERS].join(', ')}`);
        assert.equal(tier, value, 'the radio value is the tier the endpoint receives');
        assert.equal(toast.kind, 'success', `choosing ${value}: ${toast.message}`);
    }
    assert.equal(new Set(offered).size, offered.length, 'each option sends a distinct tier');
    assert.deepEqual(unexpectedRequests, []);
});

test('the default export modal selection submits a tier the endpoint accepts', async () => {
    mountSovereigntyPanel();
    const toast = await submitExportModal(() => {});

    assert.equal(exportBodies.length, 1);
    assert.ok(ALLOWED_TIERS.has(exportBodies[0].tier), `default sent ${exportBodies[0].tier}`);
    assert.equal(exportBodies[0].encrypt, true, 'encryption stays on by default');
    assert.equal(toast.kind, 'success', toast.message);
});

test('the export modal fallback tier, with nothing selected, is one the endpoint accepts', async () => {
    mountSovereigntyPanel();
    const toast = await submitExportModal(() => {
        for (const radio of tierRadios()) radio.checked = false;
    });

    assert.equal(exportBodies.length, 1);
    assert.ok(ALLOWED_TIERS.has(exportBodies[0].tier), `fallback sent ${exportBodies[0].tier}`);
    assert.equal(toast.kind, 'success', toast.message);
});

// --- normalizeSovereigntyExportTier ---------------------------------------

test('normalizeSovereigntyExportTier maps every known spelling to the endpoint tier', () => {
    const cases = [
        ['local', 'local'],
        ['local_only', 'local'],
        ['LOCAL_ONLY', 'local'],
        ['ipfs', 'ipfs'],
        ['IPFS', 'ipfs'],
        ['filecoin', 'filecoin'],
        ['FILECOIN', 'filecoin'],
    ];
    for (const [input, expected] of cases) {
        const normalized = normalizeSovereigntyExportTier(input);
        assert.equal(normalized, expected, `${input} -> ${expected}`);
        assert.ok(ALLOWED_TIERS.has(normalized), `${input} normalizes to an endpoint tier`);
    }
});

test('normalizeSovereigntyExportTier rejects an unknown tier, naming it', () => {
    const unknown = [
        'LOCAL', 'Local', 'Ipfs', 'Filecoin', 'local-only', 'cloud_hot', 'cloud_cold',
        '', ' ipfs', 'ipfs ', 'toString', '__proto__', undefined, null, 0, {},
    ];
    for (const tier of unknown) {
        assert.throws(
            () => normalizeSovereigntyExportTier(tier),
            /Unknown sovereignty export tier .*; expected one of: local, ipfs, filecoin/,
            `${String(tier)} is rejected`,
        );
    }
    assert.throws(() => normalizeSovereigntyExportTier('cloud_hot'), /"cloud_hot"/);
    assert.throws(() => normalizeSovereigntyExportTier(undefined), /of type undefined/);
});

function makeClient(fetchFn) {
    return createApiClient({
        fetchFn,
        sessionStorage: dom.window.sessionStorage,
        location: { href: '/console', search: '' },
        logger: console,
        authProvider: createBearerTokenAuthProvider({ getToken: () => 'test-token' }),
    });
}

test('exportSovereignty sends the normalized tier', async () => {
    const api = makeClient(sovereigntyEndpoint);

    await api.exportSovereignty('LOCAL_ONLY', false);

    assert.deepEqual(exportBodies, [{ tier: 'local', encrypt: false }]);
});

test('exportSovereignty rejects an unknown tier without sending a request', async () => {
    const calls = [];
    const api = makeClient(async (url, options) => {
        calls.push(url);
        return sovereigntyEndpoint(url, options);
    });

    await assert.rejects(
        api.exportSovereignty('local_cache', true),
        /Unknown sovereignty export tier "local_cache"/,
    );
    assert.deepEqual(calls, []);
});
