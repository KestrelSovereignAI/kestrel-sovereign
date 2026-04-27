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

function loadModelSelector() {
    const providerSelect = createSelect();
    const modelSelect = createSelect();
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
        fetch: async () => {
            throw new Error('unexpected fetch');
        },
        setTimeout,
        clearTimeout,
    };

    vm.runInNewContext(source, context, { filename: 'model-selector/index.js' });
    return {
        ModelSelector: context.window.SharedModelSelector,
        providerSelect,
        modelSelect,
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
