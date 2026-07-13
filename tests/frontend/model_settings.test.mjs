/**
 * Unified Model-settings popover contract tests (#2264).
 *
 * Covers the acceptance scenarios named in the issue:
 *   - per-route model repopulation (route-scoped discovery, #2262);
 *   - the meta-provider "Upstream" facet filters the model combo (display only);
 *   - Embeddings "Auto — follow chat" default + explicit selection round-trip (#2263);
 *   - the dimension-mismatch warning renders.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const selectorSource = fs.readFileSync(
    new URL('../../kestrel_sovereign/static/shared/model-selector/index.js', import.meta.url),
    'utf8',
);
const embeddingsSource = fs.readFileSync(
    new URL('../../kestrel_sovereign/static/shared/model-selector/embeddings.js', import.meta.url),
    'utf8',
);
const popoverSource = fs.readFileSync(
    new URL('../../kestrel_sovereign/static/shared/model-selector/popover.js', import.meta.url),
    'utf8',
);

function createSelect() {
    const handlers = {};
    return {
        value: '',
        innerHTML: '',
        style: {},
        options: [],
        addEventListener(type, fn) { (handlers[type] = handlers[type] || []).push(fn); },
        _fire(type) { (handlers[type] || []).forEach(fn => fn()); },
    };
}

// ---------------------------------------------------------------------------
// Chat section: route-scoped repopulation + upstream facet
// ---------------------------------------------------------------------------

function loadSelector({ fetchImpl } = {}) {
    const providerSelect = createSelect();
    const modelSelect = createSelect();
    const routeSelect = createSelect();
    const upstreamSelect = createSelect();
    const storage = new Map();
    const commits = [];

    const context = {
        console: { warn() {}, error() {}, log() {}, debug() {} },
        window: {},
        document: {
            getElementById(id) {
                if (id === 'provider-selector') return providerSelect;
                if (id === 'model-selector') return modelSelect;
                if (id === 'route-selector') return routeSelect;
                if (id === 'upstream-selector') return upstreamSelect;
                return null;
            },
        },
        localStorage: {
            getItem: k => (storage.has(k) ? storage.get(k) : null),
            setItem: (k, v) => storage.set(k, String(v)),
            removeItem: k => storage.delete(k),
        },
        fetch: fetchImpl || (async () => { throw new Error('unexpected fetch'); }),
        setTimeout, clearTimeout,
    };

    vm.runInNewContext(selectorSource, context, { filename: 'model-selector/index.js' });
    const ModelSelector = context.window.SharedModelSelector;
    const selector = new ModelSelector({
        providerSelectId: 'provider-selector',
        modelSelectId: 'model-selector',
        routeSelectId: 'route-selector',
        upstreamSelectId: 'upstream-selector',
        apiEndpoint: '/api/models',
        storagePrefix: 'test',
        onModelChange: (...a) => commits.push(a),
    });
    return { selector, providerSelect, modelSelect, routeSelect, upstreamSelect, storage, commits };
}

test('model combo repopulates from the selected route\'s discovery', async () => {
    // The plan route serves opus-only; the api route serves a broader set. A
    // route change must re-fetch and repopulate from THAT route's catalog.
    const perRoute = {
        'anthropic::plan': [{ id: 'claude-opus-4-7', provider: 'anthropic', is_featured: true }],
        'anthropic::api': [
            { id: 'claude-sonnet-4-6', provider: 'anthropic', is_featured: true },
            { id: 'claude-haiku-4-5', provider: 'anthropic', is_featured: true },
        ],
    };
    const { selector, providerSelect, modelSelect } = loadSelector({
        fetchImpl: async (url) => {
            const route = /route=([^&]+)/.exec(url)?.[1] || '';
            const key = `anthropic::${route}`;
            return { ok: true, json: async () => ({ by_vendor: { anthropic: perRoute[key] } }) };
        },
    });

    selector.allModelsData = {
        by_vendor: { anthropic: perRoute['anthropic::plan'] },
        routes: [
            { vendor: 'anthropic', route: 'plan' },
            { vendor: 'anthropic', route: 'api' },
        ],
    };
    providerSelect.value = 'anthropic';
    selector.selectedRoute = 'plan';
    selector._populateModels();
    assert.ok(modelSelect.innerHTML.includes('claude-opus-4-7'));
    assert.ok(!modelSelect.innerHTML.includes('claude-haiku-4-5'));

    // Switch to the api route and let the route-scoped fetch land.
    selector.selectedRoute = 'api';
    await selector._refreshRouteScopedModels();

    assert.ok(modelSelect.innerHTML.includes('claude-sonnet-4-6'));
    assert.ok(modelSelect.innerHTML.includes('claude-haiku-4-5'));
    assert.ok(!modelSelect.innerHTML.includes('claude-opus-4-7'));
});

test('upstream facet appears for meta-provider catalogs and filters the model combo', () => {
    const { selector, providerSelect, modelSelect, upstreamSelect } = loadSelector();
    selector.allModelsData = {
        by_vendor: {
            openrouter: [
                { id: 'anthropic/claude-sonnet-4-6', provider: 'openrouter', underlying_provider: 'anthropic', is_featured: true },
                { id: 'openai/gpt-5.5', provider: 'openrouter', underlying_provider: 'openai', is_featured: true },
                { id: 'meta/llama-4', provider: 'openrouter', underlying_provider: 'meta', is_featured: true },
            ],
        },
        routes: [{ vendor: 'openrouter', route: 'api' }],
    };
    providerSelect.value = 'openrouter';
    selector._populateModels();

    // Facet is shown with distinct upstreams + an "All" default.
    assert.notEqual(upstreamSelect.style.display, 'none');
    assert.ok(upstreamSelect.innerHTML.includes('All'));
    assert.ok(upstreamSelect.innerHTML.includes('anthropic'));
    assert.ok(upstreamSelect.innerHTML.includes('openai'));
    assert.equal(selector.selectedUpstream, 'All');
    // All models visible under "All".
    assert.ok(modelSelect.innerHTML.includes('gpt-5.5'));
    assert.ok(modelSelect.innerHTML.includes('claude-sonnet-4-6'));

    // Pick a specific upstream — model combo filters, no commit fires.
    upstreamSelect.value = 'openai';
    selector._handleUpstreamChange();
    assert.ok(modelSelect.innerHTML.includes('gpt-5.5'));
    assert.ok(!modelSelect.innerHTML.includes('llama-4'));
    assert.ok(!modelSelect.innerHTML.includes('claude-sonnet-4-6'));
});

test('upstream facet stays hidden for a plain vendor catalog', () => {
    const { selector, providerSelect, upstreamSelect } = loadSelector();
    selector.allModelsData = {
        by_vendor: {
            openai: [{ id: 'gpt-5.5', provider: 'openai', is_featured: true }],
        },
        routes: [{ vendor: 'openai', route: 'api' }],
    };
    providerSelect.value = 'openai';
    selector._populateModels();
    assert.equal(upstreamSelect.style.display, 'none');
    assert.equal(selector.selectedUpstream, 'All');
});

test('upstream filter is never committed as a routing change', () => {
    const { selector, providerSelect, upstreamSelect, commits } = loadSelector();
    selector.isInitialLoad = false;
    selector._lastSyncedSelection = { vendor: 'openrouter', model: 'openai/gpt-5.5', route: 'api' };
    selector.allModelsData = {
        by_vendor: {
            openrouter: [
                { id: 'openai/gpt-5.5', provider: 'openrouter', underlying_provider: 'openai', is_featured: true },
                { id: 'meta/llama-4', provider: 'openrouter', underlying_provider: 'meta', is_featured: true },
            ],
        },
        routes: [{ vendor: 'openrouter', route: 'api' }],
    };
    providerSelect.value = 'openrouter';
    selector._populateModels();
    upstreamSelect.value = 'meta';
    selector._handleUpstreamChange();
    assert.equal(commits.length, 0, 'upstream is a display filter, not a model-set');
});

// ---------------------------------------------------------------------------
// Embeddings section
// ---------------------------------------------------------------------------

function loadEmbeddings({ settings, routes, fetchImpl } = {}) {
    const modeSelect = createSelect();
    const routeSelect = createSelect();
    const dimReadout = { textContent: '', style: {} };
    const warningEl = { textContent: '', style: {} };
    const sharedSpaceEl = { textContent: '', style: {} };
    const reindexButton = { textContent: '', title: '', disabled: false, style: {}, addEventListener() {} };
    const reindexStatus = { textContent: '', style: {} };
    const posts = [];

    const defaultFetch = async (url, opts) => {
        if (opts && opts.method === 'POST') {
            posts.push(JSON.parse(opts.body));
            const route = JSON.parse(opts.body).embedding_route;
            // Echo back a resolved settings object.
            const resolved = route
                ? { embedding_route: route, resolved_route: route, embedding_model: 'text-embedding-3-small', embedding_dim: 1536, kestrel_embedding_dim: 1536 }
                : { embedding_route: null, resolved_route: 'ollama:local', embedding_model: 'nomic-embed-text', embedding_dim: 768, kestrel_embedding_dim: 768 };
            return { ok: true, json: async () => ({ success: true, ...resolved }) };
        }
        return { ok: true, json: async () => settings };
    };

    const context = {
        console: { warn() {}, error() {}, log() {} },
        window: {},
        document: {
            getElementById(id) {
                if (id === 'embedding-mode-selector') return modeSelect;
                if (id === 'embedding-route-selector') return routeSelect;
                if (id === 'embedding-dim-readout') return dimReadout;
                if (id === 'embedding-dim-warning') return warningEl;
                if (id === 'embedding-shared-space') return sharedSpaceEl;
                if (id === 'embedding-reindex-button') return reindexButton;
                if (id === 'embedding-reindex-status') return reindexStatus;
                return null;
            },
        },
        fetch: fetchImpl || defaultFetch,
        setTimeout, clearTimeout,
    };

    vm.runInNewContext(embeddingsSource, context, { filename: 'model-selector/embeddings.js' });
    const EmbeddingSelector = context.window.EmbeddingSelector;
    const embeddings = new EmbeddingSelector({
        settingsEndpoint: '/api/embedding/settings',
        modeSelectId: 'embedding-mode-selector',
        routeSelectId: 'embedding-route-selector',
        dimReadoutId: 'embedding-dim-readout',
        warningId: 'embedding-dim-warning',
        sharedSpaceId: 'embedding-shared-space',
        reindexButtonId: 'embedding-reindex-button',
        reindexStatusId: 'embedding-reindex-status',
        reindexEndpoint: '/api/embedding/reindex',
        confirm: () => true,
        getEmbeddingRoutes: () => routes || [],
    });
    return { embeddings, modeSelect, routeSelect, dimReadout, warningEl, sharedSpaceEl, reindexButton, reindexStatus, posts };
}

test('embeddings defaults to Auto — follow chat provider', async () => {
    const { embeddings, modeSelect, routeSelect } = loadEmbeddings({
        settings: {
            embedding_route: null,
            resolved_route: 'ollama:local',
            embedding_model: 'nomic-embed-text',
            embedding_dim: 768,
            kestrel_embedding_dim: 768,
        },
        routes: [{ vendor: 'openai', route: 'api' }, { vendor: 'ollama', route: 'local' }],
    });
    await embeddings.init();

    assert.equal(embeddings.mode, 'auto');
    assert.equal(modeSelect.value, 'auto');
    // Explicit route select hidden while in auto.
    assert.equal(routeSelect.style.display, 'none');
});

test('embeddings render a verified shared local/cloud space as one entry (#2290)', async () => {
    const { embeddings, sharedSpaceEl } = loadEmbeddings({
        settings: {
            embedding_route: null,
            resolved_route: 'ollama:local',
            embedding_model: 'qwen3-embedding-0.6b',
            embedding_dim: 768,
            kestrel_embedding_dim: 768,
            shared_space: {
                name: 'qwen3',
                space_id: 'qwen3-embedding-0.6b@768',
                model: 'qwen3-embedding-0.6b',
                dim: 768,
                members: ['ollama:local', 'openrouter:api'],
                verified: true,
                parity: { passed: true, min_cosine: 0.994 },
            },
        },
        routes: [{ vendor: 'ollama', route: 'local' }, { vendor: 'openrouter', route: 'api' }],
    });
    await embeddings.init();

    assert.equal(sharedSpaceEl.style.display, '');
    assert.equal(sharedSpaceEl.textContent, 'qwen3-embedding-0.6b — local + cloud (shared)');
});

test('embeddings mark an unverified shared space as parity unverified (#2290)', async () => {
    const { embeddings, sharedSpaceEl } = loadEmbeddings({
        settings: {
            embedding_route: null,
            resolved_route: 'ollama:local',
            embedding_model: 'qwen3-embedding-0.6b',
            embedding_dim: 768,
            kestrel_embedding_dim: 768,
            shared_space: {
                name: 'qwen3',
                space_id: 'qwen3-embedding-0.6b@768',
                model: 'qwen3-embedding-0.6b',
                dim: 768,
                members: ['ollama:local', 'openrouter:api'],
                verified: false,
                parity: null,
            },
        },
        routes: [{ vendor: 'ollama', route: 'local' }, { vendor: 'openrouter', route: 'api' }],
    });
    await embeddings.init();

    assert.equal(sharedSpaceEl.style.display, '');
    assert.equal(sharedSpaceEl.textContent, 'qwen3-embedding-0.6b — local + cloud (parity unverified)');
});

test('embeddings hide the shared-space entry when no pin covers the route (#2290)', async () => {
    const { embeddings, sharedSpaceEl } = loadEmbeddings({
        settings: {
            embedding_route: null,
            resolved_route: 'ollama:local',
            embedding_model: 'nomic-embed-text',
            embedding_dim: 768,
            kestrel_embedding_dim: 768,
            shared_space: null,
        },
        routes: [{ vendor: 'ollama', route: 'local' }],
    });
    await embeddings.init();

    assert.equal(sharedSpaceEl.style.display, 'none');
    assert.equal(sharedSpaceEl.textContent, '');
});

test('re-embed button is hidden when stale_rows is 0 (#2336)', async () => {
    const { embeddings, reindexButton } = loadEmbeddings({
        settings: {
            embedding_route: null,
            resolved_route: 'ollama:local',
            embedding_model: 'nomic-embed-text',
            embedding_dim: 768,
            kestrel_embedding_dim: 768,
            stale_rows: 0,
        },
        routes: [{ vendor: 'ollama', route: 'local' }],
    });
    await embeddings.init();

    assert.equal(reindexButton.style.display, 'none');
});

test('re-embed button shows "Re-embed N memories" when stale_rows > 0 (#2336)', async () => {
    const { embeddings, reindexButton } = loadEmbeddings({
        settings: {
            embedding_route: 'openai:api',
            resolved_route: 'openai:api',
            embedding_model: 'text-embedding-3-small',
            embedding_dim: 1536,
            kestrel_embedding_dim: 1536,
            stale_rows: 42,
        },
        routes: [{ vendor: 'openai', route: 'api' }],
    });
    await embeddings.init();

    assert.equal(reindexButton.style.display, '');
    assert.equal(reindexButton.textContent, 'Re-embed 42 memories');
    assert.equal(reindexButton.disabled, false);
});

test('re-embed button is singular for a single stale memory (#2336)', async () => {
    const { embeddings, reindexButton } = loadEmbeddings({
        settings: {
            embedding_route: 'openai:api',
            resolved_route: 'openai:api',
            embedding_model: 'text-embedding-3-small',
            embedding_dim: 1536,
            kestrel_embedding_dim: 1536,
            stale_rows: 1,
        },
        routes: [{ vendor: 'openai', route: 'api' }],
    });
    await embeddings.init();

    assert.equal(reindexButton.textContent, 'Re-embed 1 memory');
});

test('re-embed button is disabled when embeddings are off (#2336)', async () => {
    const { embeddings, reindexButton } = loadEmbeddings({
        settings: {
            embedding_route: 'none',
            resolved_route: null,
            embedding_model: null,
            embedding_dim: null,
            kestrel_embedding_dim: 768,
            stale_rows: 5,
        },
        routes: [{ vendor: 'ollama', route: 'local' }],
    });
    await embeddings.init();

    // Shown (there ARE stale rows) but disabled — nothing to re-embed to.
    assert.equal(reindexButton.style.display, '');
    assert.equal(reindexButton.disabled, true);
});

test('re-embed button dry-runs, confirms, executes, and refreshes (#2336)', async () => {
    let staleRows = 3;
    const calls = [];
    const fetchImpl = async (url, opts) => {
        const method = (opts && opts.method) || 'GET';
        calls.push({ url, method });
        if (url === '/api/embedding/reindex' && method === 'POST') {
            const body = JSON.parse(opts.body);
            if (body.dry_run) {
                return { ok: true, json: async () => ({ dry_run: true, total_stale: 3 }) };
            }
            // Execute completes inline (empty/small corpus path).
            staleRows = 0;
            return { ok: true, json: async () => ({ dry_run: false, status: 'done', total_stale: 3, total_reembedded: 3 }) };
        }
        // GET /api/embedding/settings
        return {
            ok: true,
            json: async () => ({
                embedding_route: 'openai:api',
                resolved_route: 'openai:api',
                embedding_model: 'text-embedding-3-small',
                embedding_dim: 1536,
                kestrel_embedding_dim: 1536,
                stale_rows: staleRows,
            }),
        };
    };
    const { embeddings, reindexButton } = loadEmbeddings({
        routes: [{ vendor: 'openai', route: 'api' }],
        fetchImpl,
    });
    await embeddings.init();
    assert.equal(reindexButton.textContent, 'Re-embed 3 memories');

    await embeddings._handleReindexClick();

    // A dry-run POST and an execute POST both happened.
    const posts = calls.filter(c => c.method === 'POST' && c.url === '/api/embedding/reindex');
    assert.equal(posts.length, 2);
    // After the refresh load, stale_rows dropped to 0 → button hidden again.
    assert.equal(reindexButton.style.display, 'none');
});

test('embeddings explicit selection round-trips to the API', async () => {
    const { embeddings, modeSelect, routeSelect, posts } = loadEmbeddings({
        settings: {
            embedding_route: null,
            resolved_route: 'ollama:local',
            embedding_model: 'nomic-embed-text',
            embedding_dim: 768,
            kestrel_embedding_dim: 768,
        },
        routes: [{ vendor: 'openai', route: 'api' }, { vendor: 'ollama', route: 'local' }],
    });
    await embeddings.init();

    // Operator expands to explicit and the shown route commits.
    modeSelect.value = 'explicit';
    routeSelect.value = 'openai:api';
    modeSelect._fire('change');
    // allow the async commit to settle
    await new Promise(r => setTimeout(r, 0));

    assert.equal(posts.length, 1);
    assert.equal(posts[0].embedding_route, 'openai:api');
    assert.equal(embeddings.mode, 'explicit');

    // Switching back to Auto clears the pin (embedding_route: null).
    modeSelect.value = 'auto';
    modeSelect._fire('change');
    await new Promise(r => setTimeout(r, 0));
    assert.equal(posts.length, 2);
    assert.equal(posts[1].embedding_route, null);
    assert.equal(embeddings.mode, 'auto');
});

test('embeddings Off — keyword search only commits the "none" sentinel (#2287)', async () => {
    const offPosts = [];
    const offFetch = async (url, opts) => {
        if (opts && opts.method === 'POST') {
            const route = JSON.parse(opts.body).embedding_route;
            offPosts.push({ embedding_route: route });
            // Server echoes the off state: route "none", null resolved fields,
            // but the deployment dim is still reported.
            const resolved = route === 'none'
                ? { embedding_route: 'none', resolved_route: null, embedding_model: null, embedding_dim: null, kestrel_embedding_dim: 768 }
                : { embedding_route: route || null, resolved_route: 'ollama:local', embedding_model: 'nomic-embed-text', embedding_dim: 768, kestrel_embedding_dim: 768 };
            return { ok: true, json: async () => ({ success: true, ...resolved }) };
        }
        return {
            ok: true,
            json: async () => ({
                embedding_route: null,
                resolved_route: 'ollama:local',
                embedding_model: 'nomic-embed-text',
                embedding_dim: 768,
                kestrel_embedding_dim: 768,
            }),
        };
    };
    const { embeddings, modeSelect, routeSelect, dimReadout } = loadEmbeddings({
        routes: [{ vendor: 'openai', route: 'api' }, { vendor: 'ollama', route: 'local' }],
        fetchImpl: offFetch,
    });
    await embeddings.init();
    assert.equal(embeddings.mode, 'auto');

    // Operator picks "Off — keyword search only".
    modeSelect.value = 'off';
    modeSelect._fire('change');
    await new Promise(r => setTimeout(r, 0));

    assert.equal(offPosts.length, 1);
    assert.equal(offPosts[0].embedding_route, 'none');
    assert.equal(embeddings.mode, 'off');
    // Explicit route select stays hidden in off mode.
    assert.equal(routeSelect.style.display, 'none');
    // Readout communicates the deliberate off state, not degradation.
    assert.ok(/Embeddings off/.test(dimReadout.textContent));
});

test('embeddings loads directly into Off mode when route is "none" (#2287)', async () => {
    const { embeddings, modeSelect } = loadEmbeddings({
        settings: {
            embedding_route: 'none',
            resolved_route: null,
            embedding_model: null,
            embedding_dim: null,
            kestrel_embedding_dim: 768,
        },
        routes: [{ vendor: 'openai', route: 'api' }],
    });
    await embeddings.init();

    assert.equal(embeddings.mode, 'off');
    assert.equal(modeSelect.value, 'off');
});

test('embeddings renders a dimension-mismatch warning', async () => {
    const { embeddings, warningEl, dimReadout } = loadEmbeddings({
        settings: {
            embedding_route: 'openai:api',
            resolved_route: 'openai:api',
            embedding_model: 'text-embedding-3-large',
            embedding_dim: 3072,          // resolved provider dim
            kestrel_embedding_dim: 768,   // stored-vector dim
        },
        routes: [{ vendor: 'openai', route: 'api' }],
    });
    await embeddings.init();

    assert.notEqual(warningEl.style.display, 'none');
    assert.ok(/keyword search until re-embedded/.test(warningEl.textContent));
    assert.ok(/3072/.test(dimReadout.textContent));
});

test('embeddings shows no warning when dimensions match', async () => {
    const { embeddings, warningEl } = loadEmbeddings({
        settings: {
            embedding_route: 'openai:api',
            resolved_route: 'openai:api',
            embedding_model: 'text-embedding-3-small',
            embedding_dim: 1536,
            kestrel_embedding_dim: 1536,
        },
        routes: [{ vendor: 'openai', route: 'api' }],
    });
    await embeddings.init();
    assert.equal(warningEl.style.display, 'none');
});

// ---------------------------------------------------------------------------
// Featured "Universal" option + per-route model picker + tradeoff labels (#2337)
// ---------------------------------------------------------------------------

function createButton() {
    const handlers = {};
    return {
        textContent: '', title: '', disabled: false, style: {},
        addEventListener(t, fn) { (handlers[t] = handlers[t] || []).push(fn); },
        _fire(t) { (handlers[t] || []).forEach(fn => fn()); },
    };
}

function loadEmbeddingsUniversal({ settings, routes, catalog, fetchImpl } = {}) {
    const modeSelect = createSelect();
    const routeSelect = createSelect();
    const modelSelect = createSelect();
    const universalEl = createButton();
    const setupStatus = { textContent: '', style: {} };
    const dimReadout = { textContent: '', style: {} };
    const warningEl = { textContent: '', style: {} };
    const sharedSpaceEl = { textContent: '', style: {} };
    const reindexButton = createButton();
    const reindexStatus = { textContent: '', style: {} };
    const calls = [];

    const defaultFetch = async (url, opts) => {
        const method = (opts && opts.method) || 'GET';
        calls.push({ url, method, body: opts && opts.body ? JSON.parse(opts.body) : null });
        if (url.includes('/api/embedding/models')) {
            return { ok: true, json: async () => catalog || { all: [], universal: [] } };
        }
        if (url.includes('/api/embedding/route-model')) {
            return { ok: true, json: async () => ({ success: true, ...settings }) };
        }
        if (url.includes('/api/embedding/space/verify')) {
            return { ok: true, json: async () => ({ success: true, results: { qwen3: { passed: true, min_cosine: 0.98 } } }) };
        }
        // GET /api/embedding/settings
        return { ok: true, json: async () => settings };
    };

    const context = {
        console: { warn() {}, error() {}, log() {} },
        window: {},
        document: {
            getElementById(id) {
                if (id === 'embedding-mode-selector') return modeSelect;
                if (id === 'embedding-route-selector') return routeSelect;
                if (id === 'embedding-model-selector') return modelSelect;
                if (id === 'embedding-universal') return universalEl;
                if (id === 'embedding-setup-status') return setupStatus;
                if (id === 'embedding-dim-readout') return dimReadout;
                if (id === 'embedding-dim-warning') return warningEl;
                if (id === 'embedding-shared-space') return sharedSpaceEl;
                if (id === 'embedding-reindex-button') return reindexButton;
                if (id === 'embedding-reindex-status') return reindexStatus;
                return null;
            },
        },
        fetch: fetchImpl || defaultFetch,
        setTimeout, clearTimeout,
    };

    vm.runInNewContext(embeddingsSource, context, { filename: 'model-selector/embeddings.js' });
    const EmbeddingSelector = context.window.EmbeddingSelector;
    const embeddings = new EmbeddingSelector({
        settingsEndpoint: '/api/embedding/settings',
        modelsEndpoint: '/api/embedding/models',
        routeModelEndpoint: '/api/embedding/route-model',
        verifyEndpoint: '/api/embedding/space/verify',
        modeSelectId: 'embedding-mode-selector',
        routeSelectId: 'embedding-route-selector',
        modelSelectId: 'embedding-model-selector',
        universalId: 'embedding-universal',
        setupStatusId: 'embedding-setup-status',
        dimReadoutId: 'embedding-dim-readout',
        warningId: 'embedding-dim-warning',
        sharedSpaceId: 'embedding-shared-space',
        reindexButtonId: 'embedding-reindex-button',
        reindexStatusId: 'embedding-reindex-status',
        reindexEndpoint: '/api/embedding/reindex',
        confirm: () => true,
        getEmbeddingRoutes: () => routes || [],
    });
    return { embeddings, modeSelect, routeSelect, modelSelect, universalEl, setupStatus, sharedSpaceEl, warningEl, dimReadout, calls };
}

const UNIVERSAL_CATALOG = {
    all: [
        { id: 'qwen3-embedding-0.6b', provider: 'ollama', route: 'ollama:local', display_name: 'qwen3-embedding-0.6b', native_dim: 768 },
        { id: 'qwen/qwen3-embedding-0.6b', provider: 'openrouter', route: 'openrouter:api', display_name: 'qwen3-embedding-0.6b', native_dim: 768 },
        { id: 'text-embedding-3-small', provider: 'openai', route: 'openai:api', display_name: 'text-embedding-3-small', native_dim: 1536 },
    ],
    universal: [
        {
            model: 'qwen3-embedding-0.6b',
            display_name: 'qwen3-embedding-0.6b',
            dim: 768,
            dim_options: [768],
            members: [
                { route: 'ollama:local', model: 'qwen3-embedding-0.6b', provider: 'ollama', is_local: true, native_dim: 768 },
                { route: 'openrouter:api', model: 'qwen/qwen3-embedding-0.6b', provider: 'openrouter', is_local: false, native_dim: 768 },
            ],
        },
    ],
};

test('featured Universal option renders "needs setup" when not configured (#2337)', async () => {
    const { embeddings, universalEl } = loadEmbeddingsUniversal({
        settings: {
            embedding_route: null, resolved_route: 'ollama:local',
            embedding_model: 'nomic-embed-text', embedding_dim: 768, kestrel_embedding_dim: 768,
            shared_space: null,
        },
        routes: [{ vendor: 'ollama', route: 'local', is_local: true }, { vendor: 'openrouter', route: 'api', is_local: false }],
        catalog: UNIVERSAL_CATALOG,
    });
    await embeddings.init();

    assert.equal(universalEl.style.display, '');
    assert.ok(/Universal — qwen3-embedding-0.6b/.test(universalEl.textContent));
    assert.ok(/one search space/.test(universalEl.textContent));
    assert.ok(/needs setup/.test(universalEl.textContent));
});

test('featured Universal option reads "active" when configured + verified (#2337)', async () => {
    const { embeddings, universalEl } = loadEmbeddingsUniversal({
        settings: {
            embedding_route: null, resolved_route: 'ollama:local',
            embedding_model: 'qwen3-embedding-0.6b', embedding_dim: 768, kestrel_embedding_dim: 768,
            shared_space: { name: 'qwen3', model: 'qwen3-embedding-0.6b', dim: 768, members: ['ollama:local', 'openrouter:api'], verified: true },
        },
        routes: [{ vendor: 'ollama', route: 'local', is_local: true }, { vendor: 'openrouter', route: 'api', is_local: false }],
        catalog: UNIVERSAL_CATALOG,
    });
    await embeddings.init();
    assert.ok(/active/.test(universalEl.textContent));
});

test('clicking Universal pins both member routes then verifies parity (#2337)', async () => {
    const { embeddings, universalEl, calls, setupStatus } = loadEmbeddingsUniversal({
        settings: {
            embedding_route: null, resolved_route: 'ollama:local',
            embedding_model: 'nomic-embed-text', embedding_dim: 768, kestrel_embedding_dim: 768,
            shared_space: null,
        },
        routes: [{ vendor: 'ollama', route: 'local', is_local: true }, { vendor: 'openrouter', route: 'api', is_local: false }],
        catalog: UNIVERSAL_CATALOG,
    });
    await embeddings.init();

    await embeddings._handleUniversalClick();

    const routeModelPosts = calls.filter(c => c.method === 'POST' && c.url.includes('/route-model'));
    assert.equal(routeModelPosts.length, 2);
    // Each member pinned with its OWN slug.
    const byRoute = Object.fromEntries(routeModelPosts.map(p => [p.body.route, p.body.embedding_model]));
    assert.equal(byRoute['ollama:local'], 'qwen3-embedding-0.6b');
    assert.equal(byRoute['openrouter:api'], 'qwen/qwen3-embedding-0.6b');
    // Parity probe ran.
    assert.ok(calls.some(c => c.method === 'POST' && c.url.includes('/space/verify')));
    assert.ok(/parity verified/.test(setupStatus.textContent));
});

test('guided setup fails loudly when a member route probe rejects the slug (#2337)', async () => {
    const settings = {
        embedding_route: null, resolved_route: 'ollama:local',
        embedding_model: 'nomic-embed-text', embedding_dim: 768, kestrel_embedding_dim: 768, shared_space: null,
    };
    const fetchImpl = async (url, opts) => {
        const method = (opts && opts.method) || 'GET';
        if (url.includes('/api/embedding/models')) return { ok: true, json: async () => UNIVERSAL_CATALOG };
        if (url.includes('/route-model')) {
            // openrouter's slug is dead upstream → 400.
            const body = JSON.parse(opts.body);
            if (body.route === 'openrouter:api') {
                return { ok: false, json: async () => ({ detail: 'live embedding probe failed' }) };
            }
            return { ok: true, json: async () => ({ success: true, ...settings }) };
        }
        if (url.includes('/space/verify')) throw new Error('verify must not run after a failed pin');
        return { ok: true, json: async () => settings };
    };
    const { embeddings, setupStatus, calls } = loadEmbeddingsUniversal({
        settings,
        routes: [{ vendor: 'ollama', route: 'local', is_local: true }, { vendor: 'openrouter', route: 'api', is_local: false }],
        catalog: UNIVERSAL_CATALOG,
        fetchImpl,
    });
    await embeddings.init();
    await embeddings._handleUniversalClick();

    assert.ok(/Setup failed on openrouter:api/.test(setupStatus.textContent));
    assert.ok(/live embedding probe failed/.test(setupStatus.textContent));
    // Never reached the parity probe.
    assert.ok(!calls.some(c => c.url.includes('/space/verify')));
});

test('non-universal cloud route is labeled with its tradeoff (#2337)', async () => {
    const { embeddings, modeSelect, routeSelect } = loadEmbeddingsUniversal({
        settings: {
            embedding_route: null, resolved_route: 'ollama:local',
            embedding_model: 'nomic-embed-text', embedding_dim: 768, kestrel_embedding_dim: 768, shared_space: null,
        },
        routes: [
            { vendor: 'openai', route: 'api', is_local: false },
            { vendor: 'ollama', route: 'local', is_local: true },
            { vendor: 'openrouter', route: 'api', is_local: false },
        ],
        catalog: UNIVERSAL_CATALOG,
    });
    await embeddings.init();

    // openai:api is cloud-only and NOT a universal member → tradeoff labeled.
    assert.ok(/openai:api — cloud only — private\/local sessions fall back to keyword search/.test(routeSelect.innerHTML));
    // openrouter:api IS a universal member → no tradeoff warning appended.
    assert.ok(!/openrouter:api — cloud only/.test(routeSelect.innerHTML));
});

test('per-route model picker lists discovered models and pins on change (#2337)', async () => {
    const settings = {
        embedding_route: 'openai:api', resolved_route: 'openai:api',
        embedding_model: 'text-embedding-3-small', embedding_dim: 1536, kestrel_embedding_dim: 1536,
        shared_space: null, route_embedding_models: {},
    };
    const { embeddings, modeSelect, routeSelect, modelSelect, calls } = loadEmbeddingsUniversal({
        settings,
        routes: [{ vendor: 'openai', route: 'api', is_local: false }],
        catalog: UNIVERSAL_CATALOG,
    });
    await embeddings.init();

    // In explicit mode with a discovered catalog, the model picker is populated.
    embeddings.mode = 'explicit';
    routeSelect.value = 'openai:api';
    embeddings._renderModelPicker();
    assert.equal(modelSelect.style.display, '');
    assert.ok(/text-embedding-3-small/.test(modelSelect.innerHTML));

    // Changing the model commits a per-route pin.
    modelSelect.value = 'text-embedding-3-small';
    await embeddings._handleModelChange();
    const pins = calls.filter(c => c.method === 'POST' && c.url.includes('/route-model'));
    assert.equal(pins.length, 1);
    assert.equal(pins[0].body.route, 'openai:api');
    assert.equal(pins[0].body.embedding_model, 'text-embedding-3-small');
    assert.equal(pins[0].body.embedding_dim, 1536);
});

test('dim-incompatible route is marked "needs migration" before selection (#2417)', async () => {
    const { embeddings, routeSelect } = loadEmbeddingsUniversal({
        settings: {
            embedding_route: null, resolved_route: 'ollama:local',
            embedding_model: 'nomic-embed-text', embedding_dim: 768, kestrel_embedding_dim: 768, shared_space: null,
        },
        routes: [
            // openai:api resolves to 1536 but the column is 768 → needs migration.
            { vendor: 'openai', route: 'api', is_local: false, embedding_dim: 1536 },
            // ollama:local matches the 768 column → no migration marker.
            { vendor: 'ollama', route: 'local', is_local: true, embedding_dim: 768 },
        ],
        catalog: UNIVERSAL_CATALOG,
    });
    await embeddings.init();

    assert.ok(/openai:api.*1536-dim, needs migration/.test(routeSelect.innerHTML));
    assert.ok(!/ollama:local.*needs migration/.test(routeSelect.innerHTML));
});

test('model picker marks a 1536-dim model on a 768 column as needs migration (#2417)', async () => {
    const settings = {
        embedding_route: 'openai:api', resolved_route: 'openai:api',
        embedding_model: 'text-embedding-3-small', embedding_dim: 768, kestrel_embedding_dim: 768,
        shared_space: null, route_embedding_models: {},
    };
    const { embeddings, routeSelect, modelSelect } = loadEmbeddingsUniversal({
        settings,
        routes: [{ vendor: 'openai', route: 'api', is_local: false, embedding_dim: 1536 }],
        catalog: UNIVERSAL_CATALOG,
    });
    await embeddings.init();

    embeddings.mode = 'explicit';
    routeSelect.value = 'openai:api';
    embeddings._renderModelPicker();
    // text-embedding-3-small is native 1536 → flagged against the 768 column.
    assert.ok(/text-embedding-3-small.*1536-dim, needs migration/.test(modelSelect.innerHTML));
});

test('write-blocked agent surfaces "memory vectors paused" status (#2417)', async () => {
    const { embeddings, warningEl } = loadEmbeddingsUniversal({
        settings: {
            embedding_route: 'openai:api', resolved_route: 'openai:api',
            embedding_model: 'text-embedding-3-small', embedding_dim: 1536, kestrel_embedding_dim: 768,
            dim_write_blocked: true,
            dim_write_status: 'selected provider cannot write — memory vectors paused (resolves 1536-dim, columns are 768-dim)',
            shared_space: null,
        },
        routes: [{ vendor: 'openai', route: 'api', is_local: false, embedding_dim: 1536 }],
        catalog: UNIVERSAL_CATALOG,
    });
    await embeddings.init();

    assert.equal(warningEl.style.display, '');
    assert.ok(/memory vectors paused/.test(warningEl.textContent));
});

// ---------------------------------------------------------------------------
// Popover open/close
// ---------------------------------------------------------------------------

function loadPopover() {
    const listeners = {};
    const button = {
        style: {}, attrs: {},
        addEventListener(t, fn) { (listeners[t] = listeners[t] || []).push(fn); },
        setAttribute(k, v) { this.attrs[k] = v; },
        getBoundingClientRect: () => ({ bottom: 40, left: 10, top: 20, right: 100 }),
        contains: () => false,
        _fire(t, e) { (listeners[t] || []).forEach(fn => fn(e || { stopPropagation() {} })); },
    };
    const panel = {
        style: { display: 'none' }, parentNode: null,
        addEventListener() {},
        contains: () => false,
    };
    const root = { appendChild(el) { el.parentNode = root; } };
    const docListeners = {};
    const context = {
        console: { warn() {}, error() {}, log() {} },
        window: {},
        document: {
            getElementById(id) {
                if (id === 'model-settings-button') return button;
                if (id === 'model-settings-panel') return panel;
                return null;
            },
            addEventListener(t, fn) { (docListeners[t] = docListeners[t] || []).push(fn); },
            removeEventListener() {},
        },
        setTimeout: (fn) => fn(),
        clearTimeout,
    };
    vm.runInNewContext(popoverSource, context, { filename: 'model-selector/popover.js' });
    const ModelSettingsPopover = context.window.ModelSettingsPopover;
    const popover = new ModelSettingsPopover({
        buttonId: 'model-settings-button',
        panelId: 'model-settings-panel',
        getOverlayRoot: () => root,
    });
    return { popover, button, panel, root };
}

test('popover toggles open/closed and re-homes the panel under the overlay root', () => {
    const { popover, button, panel, root } = loadPopover();
    assert.equal(popover.isOpen, false);

    button._fire('click');
    assert.equal(popover.isOpen, true);
    assert.equal(panel.style.display, '');
    assert.equal(panel.parentNode, root, 'panel re-homed under overlay root');
    assert.equal(button.attrs['aria-expanded'], 'true');

    button._fire('click');
    assert.equal(popover.isOpen, false);
    assert.equal(panel.style.display, 'none');
    assert.equal(button.attrs['aria-expanded'], 'false');
});

test('route switch that drops the selected model COMMITS the coerced model (codex P2)', async () => {
    // plan serves opus; api does NOT. Switching plan→api coerces the model
    // combo to the first valid api model — that coerced selection must be
    // committed, or the UI shows sonnet while the server stays on opus/api.
    const perRoute = {
        'anthropic::plan': [{ id: 'claude-opus-4-7', provider: 'anthropic', is_featured: true }],
        'anthropic::api': [
            { id: 'claude-sonnet-4-6', provider: 'anthropic', is_featured: true },
        ],
    };
    const commits = [];
    const { selector, providerSelect, routeSelect } = loadSelector({
        fetchImpl: async (url) => {
            const route = /route=([^&]+)/.exec(url)?.[1] || '';
            const key = `anthropic::${route}`;
            return { ok: true, json: async () => ({ by_vendor: { anthropic: perRoute[key] } }) };
        },
    });
    selector.onModelChange = (vendor, model, _c, route) => { commits.push([vendor, model, route]); };

    selector.allModelsData = {
        by_vendor: { anthropic: perRoute['anthropic::plan'] },
        routes: [
            { vendor: 'anthropic', route: 'plan' },
            { vendor: 'anthropic', route: 'api' },
        ],
    };
    selector.isInitialLoad = false;
    selector._lastSyncedSelection = { vendor: 'anthropic', model: 'claude-opus-4-7', route: 'plan' };
    providerSelect.value = 'anthropic';
    selector.selectedProvider = 'anthropic';
    selector.selectedModel = 'claude-opus-4-7';
    selector.selectedRoute = 'plan';
    selector._populateRoutes();
    selector._populateModels();

    routeSelect.value = 'api';
    selector._handleRouteChange();
    // Let the async route-scoped fetch + repopulate land.
    await new Promise((r) => setTimeout(r, 20));

    const last = commits[commits.length - 1];
    assert.ok(last, 'a commit fired');
    assert.equal(last[1], 'claude-sonnet-4-6', 'the COERCED model was committed to the server');
    assert.equal(last[2], 'api');
});
