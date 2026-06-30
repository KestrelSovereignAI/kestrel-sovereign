import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const source = fs.readFileSync(
    new URL('../../kestrel_sovereign/static/shared/model-selector/index.js', import.meta.url),
    'utf8',
);

function createSelect() {
    return {
        value: '',
        innerHTML: '',
        style: {},
        addEventListener() {},
    };
}

function loadModelSelector({ fetchImpl } = {}) {
    const providerSelect = createSelect();
    const modelSelect = createSelect();
    const routeSelect = createSelect();
    routeSelect.options = [];
    const storage = new Map();

    const context = {
        console: {
            warn() {},
            error() {},
            log() {},
        },
        window: {},
        document: {
            getElementById(id) {
                if (id === 'provider-selector') return providerSelect;
                if (id === 'model-selector') return modelSelect;
                if (id === 'route-selector') return routeSelect;
                return null;
            },
        },
        localStorage: {
            getItem(key) {
                return storage.has(key) ? storage.get(key) : null;
            },
            setItem(key, value) {
                storage.set(key, String(value));
            },
            removeItem(key) {
                storage.delete(key);
            },
        },
        fetch: fetchImpl || (async () => {
            throw new Error('unexpected fetch');
        }),
        setTimeout,
        clearTimeout,
    };

    vm.runInNewContext(source, context, { filename: 'model-selector/index.js' });
    return {
        ModelSelector: context.window.SharedModelSelector,
        providerSelect,
        modelSelect,
        routeSelect,
        storage,
    };
}

test('checkForModelChange parses embedded marker with trailing text', () => {
    const { ModelSelector, providerSelect, modelSelect, storage } = loadModelSelector();
    const selector = new ModelSelector({
        providerSelectId: 'provider-selector',
        modelSelectId: 'model-selector',
        storagePrefix: 'test',
    });

    selector.allModelsData = {
        by_vendor: {
            anthropic: [
                { id: 'claude-sonnet-4-6', provider: 'anthropic', display_name: 'Claude Sonnet 4.6' },
            ],
            openai: [
                { id: 'gpt-5.4', provider: 'openai', display_name: 'GPT-5.4' },
            ],
        },
    };
    selector.selectedProvider = 'openai';
    selector.selectedModel = 'gpt-5.4';

    const changed = selector.checkForModelChange(
        '✓ Model set to: anthropic/claude-sonnet-4-6\n\n' +
        'MODEL_CHANGED:{"model":"anthropic/claude-sonnet-4-6","provider":"anthropic"}\n' +
        '\nAnything after the marker should be ignored.',
    );

    assert.equal(changed, true);
    assert.equal(providerSelect.value, 'anthropic');
    assert.equal(modelSelect.value, 'claude-sonnet-4-6');
    assert.equal(selector.selectedProvider, 'anthropic');
    assert.equal(selector.selectedModel, 'claude-sonnet-4-6');
    assert.equal(storage.get('test_selected_provider'), 'anthropic');
    assert.equal(storage.get('test_selected_model'), 'claude-sonnet-4-6');
});

test('checkForModelChange returns false when marker payload is malformed', () => {
    const { ModelSelector } = loadModelSelector();
    const selector = new ModelSelector({
        providerSelectId: 'provider-selector',
        modelSelectId: 'model-selector',
        storagePrefix: 'test',
    });

    const changed = selector.checkForModelChange('MODEL_CHANGED:{not valid json');

    assert.equal(changed, false);
});


// ----------------------------------------------------------------------
// Regressions for the dropdown seed (post #702 fixes)
//
// _serverDefaultSelection() reads /api/models' "default" + "routes" so the
// dropdown opens on the server's actual mandate-respecting default rather
// than alphabetical-first vendor (which would land on "anthropic" or
// "gpt-3.5-turbo" regardless of routing). Without these tests, future
// refactors can quietly revert to the alphabetical-fallback behavior.
// ----------------------------------------------------------------------

test('_serverDefaultSelection picks server default and matches its route', () => {
    const { ModelSelector } = loadModelSelector();
    const selector = new ModelSelector({
        providerSelectId: 'provider-selector',
        modelSelectId: 'model-selector',
        storagePrefix: 'test',
    });

    selector.allModelsData = {
        default: 'claude-opus-4-7',
        by_vendor: {
            anthropic: [
                { id: 'claude-opus-4-7' },
                { id: 'claude-sonnet-4-6' },
            ],
            openai: [{ id: 'gpt-5-mini' }],
        },
        routes: [
            { vendor: 'anthropic', route: 'plan', model: 'claude-opus-4-7' },
            { vendor: 'anthropic', route: 'api', model: 'auto' },
            { vendor: 'openai', route: 'api', model: 'gpt-5-mini' },
        ],
    };

    const seed = selector._serverDefaultSelection();
    // Field-by-field — the seed is built inside the vm sandbox so its
    // Object.prototype differs from the test realm's. ``assert/strict``
    // aliases ``deepEqual`` to ``deepStrictEqual``, which rejects that.
    assert.equal(seed.vendor, 'anthropic');
    assert.equal(seed.model, 'claude-opus-4-7');
    assert.equal(seed.route, 'plan');
});

test('_serverDefaultSelection falls back to first priority route when no default', () => {
    const { ModelSelector } = loadModelSelector();
    const selector = new ModelSelector({
        providerSelectId: 'provider-selector',
        modelSelectId: 'model-selector',
        storagePrefix: 'test',
    });

    selector.allModelsData = {
        default: null,
        by_vendor: {
            anthropic: [{ id: 'claude-opus-4-7' }],
            openai: [{ id: 'gpt-5-mini' }],
        },
        routes: [
            { vendor: 'anthropic', route: 'plan', model: 'claude-opus-4-7' },
            { vendor: 'openai', route: 'api', model: 'gpt-5-mini' },
        ],
    };

    const seed = selector._serverDefaultSelection();
    assert.equal(seed.vendor, 'anthropic');
    assert.equal(seed.model, 'claude-opus-4-7');
    assert.equal(seed.route, 'plan');
});

test('_serverDefaultSelection returns null when allModelsData is null', () => {
    const { ModelSelector } = loadModelSelector();
    const selector = new ModelSelector({
        providerSelectId: 'provider-selector',
        modelSelectId: 'model-selector',
        storagePrefix: 'test',
    });

    selector.allModelsData = null;
    assert.equal(selector._serverDefaultSelection(), null);
});

test('_serverDefaultSelection handles default whose vendor route has model="auto"', () => {
    // Routes can declare model="auto" — the discovery-resolved default still
    // names a real id. The seed should land on the real id; route comes from
    // the matching vendor.
    const { ModelSelector } = loadModelSelector();
    const selector = new ModelSelector({
        providerSelectId: 'provider-selector',
        modelSelectId: 'model-selector',
        storagePrefix: 'test',
    });

    selector.allModelsData = {
        default: 'gpt-5-mini',
        by_vendor: { openai: [{ id: 'gpt-5-mini' }] },
        routes: [{ vendor: 'openai', route: 'api', model: 'auto' }],
    };

    const seed = selector._serverDefaultSelection();
    assert.equal(seed.vendor, 'openai');
    assert.equal(seed.model, 'gpt-5-mini');
    assert.equal(seed.route, 'api');
});

test('_populateProviders seeds vendor from server default rather than alphabetical first', () => {
    // The bug: with no saved selection, the old code did `vendors[0]` which
    // sorts to "anthropic" alphabetically — even when the server's default
    // belongs to a different vendor. Verify _populateProviders now respects
    // the server default.
    const { ModelSelector, providerSelect } = loadModelSelector();
    const selector = new ModelSelector({
        providerSelectId: 'provider-selector',
        modelSelectId: 'model-selector',
        storagePrefix: 'test',
    });

    // No saved selection — fresh user, fresh agent.
    selector.selectedProvider = '';
    selector.selectedModel = '';
    selector.allModelsData = {
        default: 'gpt-5-mini',
        by_vendor: {
            anthropic: [{ id: 'claude-sonnet-4-6' }],
            openai: [{ id: 'gpt-5-mini' }],
        },
        routes: [
            { vendor: 'openai', route: 'api', model: 'gpt-5-mini' },
            { vendor: 'anthropic', route: 'api', model: 'auto' },
        ],
    };

    selector._populateProviders();

    // Old behavior would have set this to 'anthropic' (sorts first).
    assert.equal(providerSelect.value, 'openai');
    assert.equal(selector.selectedProvider, 'openai');
});

// ----------------------------------------------------------------------
// Featured-only default + "Show all" expander (#2015)
// ----------------------------------------------------------------------

function _openaiMixedData() {
    return {
        default: null,
        by_vendor: {
            // Server order: featured-first, recency-ranked. gpt-3.5-turbo is a
            // non-featured stale model that must NOT seed or show by default.
            openai: [
                { id: 'gpt-5.5', provider: 'openai', display_name: 'GPT-5.5', is_featured: true },
                { id: 'gpt-5.4', provider: 'openai', display_name: 'GPT-5.4', is_featured: true },
                { id: 'gpt-3.5-turbo', provider: 'openai', display_name: 'GPT-3.5 Turbo', is_featured: false },
                { id: 'gpt-4', provider: 'openai', display_name: 'GPT-4', is_featured: false },
            ],
        },
        routes: [{ vendor: 'openai', route: 'api', model: 'auto' }],
    };
}

test('_populateModels defaults to featured set, seeds best featured, offers Show all', () => {
    const { ModelSelector, providerSelect, modelSelect } = loadModelSelector();
    const selector = new ModelSelector({
        providerSelectId: 'provider-selector',
        modelSelectId: 'model-selector',
        storagePrefix: 'test',
    });
    selector.allModelsData = _openaiMixedData();
    providerSelect.value = 'openai';
    selector.selectedModel = '';

    selector._populateModels();

    // Featured-only by default: stale models hidden, expander present.
    assert.ok(modelSelect.innerHTML.includes('gpt-5.5'));
    assert.ok(!modelSelect.innerHTML.includes('"gpt-3.5-turbo"'));
    assert.ok(modelSelect.innerHTML.includes('__show_all__'));
    // Seed is the best featured model — never the alphabetical gpt-3.5-turbo.
    assert.equal(modelSelect.value, 'gpt-5.5');
    assert.equal(selector.selectedModel, 'gpt-5.5');
});

test('selecting the Show all sentinel reveals every model without committing', () => {
    const commits = [];
    const { ModelSelector, providerSelect, modelSelect } = loadModelSelector();
    const selector = new ModelSelector({
        providerSelectId: 'provider-selector',
        modelSelectId: 'model-selector',
        storagePrefix: 'test',
        onModelChange: (...a) => commits.push(a),
    });
    selector.allModelsData = _openaiMixedData();
    providerSelect.value = 'openai';
    selector.isInitialLoad = false;
    selector._populateModels();         // featured-only
    selector.selectedModel = 'gpt-5.5';

    // Operator picks the "Show all" sentinel.
    modelSelect.value = '__show_all__';
    selector._handleModelChange();

    assert.equal(selector.showAllModels, true);
    assert.ok(modelSelect.innerHTML.includes('"gpt-3.5-turbo"'));
    // Sentinel must not be treated as a model selection — no commit fired.
    assert.equal(commits.length, 0);
    assert.equal(selector.selectedModel, 'gpt-5.5');
});

test('checkForModelChange renders a non-featured current model under the collapsed view', () => {
    // Regression (#2015): switching the agent to a deprecated/non-featured model
    // must still show it, even though the dropdown defaults to featured-only.
    const { ModelSelector, providerSelect, modelSelect } = loadModelSelector();
    const selector = new ModelSelector({
        providerSelectId: 'provider-selector',
        modelSelectId: 'model-selector',
        storagePrefix: 'test',
    });
    selector.allModelsData = {
        by_vendor: {
            openai: [
                { id: 'gpt-5.5', provider: 'openai', display_name: 'GPT-5.5', is_featured: true },
                { id: 'gpt-4', provider: 'openai', display_name: 'GPT-4', is_featured: false },
            ],
        },
        routes: [{ vendor: 'openai', route: 'api', model: 'auto' }],
    };
    providerSelect.value = 'openai';
    selector.selectedModel = 'gpt-5.5';
    selector._populateModels();                     // collapsed: gpt-4 not rendered
    assert.ok(!modelSelect.innerHTML.includes('"gpt-4"'));

    const changed = selector.checkForModelChange(
        'MODEL_CHANGED:{"vendor":"openai","model":"openai/gpt-4","model_name":"gpt-4"}');

    assert.equal(changed, true);
    assert.ok(modelSelect.innerHTML.includes('"gpt-4"'));  // now rendered
    assert.equal(modelSelect.value, 'gpt-4');
    assert.equal(selector.selectedModel, 'gpt-4');
});

test('vendor switch resets back to the featured view', () => {
    const { ModelSelector, providerSelect } = loadModelSelector();
    const selector = new ModelSelector({
        providerSelectId: 'provider-selector',
        modelSelectId: 'model-selector',
        storagePrefix: 'test',
    });
    selector.allModelsData = _openaiMixedData();
    selector.showAllModels = true;
    providerSelect.value = 'openai';
    selector.isInitialLoad = false;

    selector._handleProviderChange();

    assert.equal(selector.showAllModels, false);
});

test('_populateProviders prefers saved selection over server default', () => {
    // localStorage > server default > alphabetical. Saved must win.
    const { ModelSelector, providerSelect } = loadModelSelector();
    const selector = new ModelSelector({
        providerSelectId: 'provider-selector',
        modelSelectId: 'model-selector',
        storagePrefix: 'test',
    });

    selector.selectedProvider = 'anthropic';  // user previously chose anthropic
    selector.allModelsData = {
        default: 'gpt-5-mini',  // server says default to openai
        by_vendor: {
            anthropic: [{ id: 'claude-sonnet-4-6' }],
            openai: [{ id: 'gpt-5-mini' }],
        },
        routes: [
            { vendor: 'openai', route: 'api', model: 'gpt-5-mini' },
            { vendor: 'anthropic', route: 'api', model: 'auto' },
        ],
    };

    selector._populateProviders();

    // Saved choice wins — user agency over server default.
    assert.equal(providerSelect.value, 'anthropic');
    assert.equal(selector.selectedProvider, 'anthropic');
});

test('syncWithServer reflects an agent-driven model change (#2068)', async () => {
    // #2068: when the LLM calls the set_model tool, the MODEL_CHANGED marker
    // lands in the tool-result card, not in the streamed assistant text, so
    // checkForModelChange never fires. chat.js detects the set_model tool
    // event and re-syncs from /api/model/current; this pins that the re-sync
    // mechanism actually moves the selector to the new server-side model.
    let fetched = 0;
    const { ModelSelector, providerSelect, modelSelect } = loadModelSelector({
        fetchImpl: async (url) => {
            fetched += 1;
            return {
                ok: true,
                json: async () => ({
                    model: 'openai:api/gpt-5.4-mini',
                    model_name: 'gpt-5.4-mini',
                    vendor: 'openai',
                    route: 'api',
                }),
            };
        },
    });
    const selector = new ModelSelector({
        providerSelectId: 'provider-selector',
        modelSelectId: 'model-selector',
        storagePrefix: 'test',
    });

    // Selector currently shows Anthropic (what the user manually picked).
    selector.allModelsData = {
        by_vendor: {
            anthropic: [{ id: 'claude-haiku-4-5', provider: 'anthropic' }],
            openai: [{ id: 'gpt-5.4-mini', provider: 'openai' }],
        },
    };
    providerSelect.value = 'anthropic';
    modelSelect.value = 'claude-haiku-4-5';
    selector.selectedProvider = 'anthropic';
    selector.selectedModel = 'claude-haiku-4-5';

    await selector.syncWithServer();

    assert.equal(fetched, 1);
    assert.equal(providerSelect.value, 'openai');
    assert.equal(modelSelect.value, 'gpt-5.4-mini');
    assert.equal(selector.selectedProvider, 'openai');
    assert.equal(selector.selectedModel, 'gpt-5.4-mini');
});
