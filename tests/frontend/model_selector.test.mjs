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
        by_provider: {
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
