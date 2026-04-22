/**
 * Contract tests for the vendor/route/model selector UI.
 *
 * Regression tests cover:
 *   - Route selector visibility (hidden when ≤1 route, visible when >1).
 *   - Vendor change must NOT auto-commit (the bug that sent
 *     !model-set anthropic haiku before the user picked opus).
 *   - Model change DOES commit.
 *   - Route change DOES commit and keeps the current model.
 *   - syncWithServer populates the route selector from server state.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const source = fs.readFileSync(
    new URL('../../kestrel_sovereign/static/shared/model-selector/index.js', import.meta.url),
    'utf8',
);

function createSelect() {
    const handlers = { change: [] };
    return {
        value: '',
        innerHTML: '',
        style: {},
        options: [],
        addEventListener(type, fn) {
            (handlers[type] = handlers[type] || []).push(fn);
        },
        _fire(type) {
            (handlers[type] || []).forEach(fn => fn());
        },
        _setOptions(values) {
            this.options = values.map(v => ({ value: v }));
        },
    };
}

function loadSelector({ withRouteSelect = true } = {}) {
    const providerSelect = createSelect();
    const modelSelect = createSelect();
    const routeSelect = withRouteSelect ? createSelect() : null;
    const storage = new Map();
    const commits = [];  // captures calls to the onModelChange callback

    const context = {
        console: { warn() {}, error() {}, log() {}, debug() {} },
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
            getItem: k => (storage.has(k) ? storage.get(k) : null),
            setItem: (k, v) => storage.set(k, String(v)),
            removeItem: k => storage.delete(k),
        },
        fetch: async () => { throw new Error('unexpected fetch'); },
        setTimeout, clearTimeout,
    };

    vm.runInNewContext(source, context, { filename: 'model-selector/index.js' });

    const ModelSelector = context.window.SharedModelSelector;
    const selector = new ModelSelector({
        providerSelectId: 'provider-selector',
        modelSelectId: 'model-selector',
        routeSelectId: withRouteSelect ? 'route-selector' : undefined,
        storagePrefix: 'test',
        onModelChange: (...args) => commits.push(args),
    });

    return { selector, providerSelect, modelSelect, routeSelect, storage, commits };
}


test('route selector is hidden when vendor has only one route', () => {
    const { selector, providerSelect, routeSelect } = loadSelector();
    selector.allModelsData = {
        by_vendor: {
            ollama: [{ id: 'llama3.2', provider: 'ollama', is_featured: true }],
        },
        routes: [{ vendor: 'ollama', route: 'local', is_local: true, model: 'llama3.2' }],
    };
    providerSelect.value = 'ollama';
    selector._populateRoutes();
    assert.equal(routeSelect.style.display, 'none');
    assert.equal(routeSelect.innerHTML, '');
});


test('route selector is visible with options when vendor has multiple routes', () => {
    const { selector, providerSelect, routeSelect } = loadSelector();
    selector.allModelsData = {
        by_vendor: {
            anthropic: [{ id: 'claude-opus-4-7', provider: 'anthropic', is_featured: true }],
        },
        routes: [
            { vendor: 'anthropic', route: 'plan', is_local: false },
            { vendor: 'anthropic', route: 'api',  is_local: false },
        ],
    };
    providerSelect.value = 'anthropic';
    selector._populateRoutes();
    assert.notEqual(routeSelect.style.display, 'none');
    assert.ok(routeSelect.innerHTML.includes('Plan'), `expected "Plan" option, got: ${routeSelect.innerHTML}`);
    assert.ok(routeSelect.innerHTML.includes('Api'), `expected "Api" option, got: ${routeSelect.innerHTML}`);
    // First route becomes default when no selection is remembered.
    assert.equal(selector.selectedRoute, 'plan');
});


test('REGRESSION: vendor change does NOT fire onModelChange', () => {
    // The bug that caused this test: when the user changed the vendor dropdown
    // to Anthropic, the component auto-populated the model list with the first
    // featured model (haiku, alphabetical on canonical aliases) and fired
    // onModelChange immediately. The chat then sent `!model-set anthropic haiku`
    // before the user had a chance to pick opus.
    const { selector, providerSelect, modelSelect, commits } = loadSelector();
    selector.allModelsData = {
        by_vendor: {
            anthropic: [
                { id: 'claude-haiku-4-5', provider: 'anthropic', is_featured: true },
                { id: 'claude-opus-4-7',  provider: 'anthropic', is_featured: true },
            ],
            openai: [{ id: 'gpt-5', provider: 'openai', is_featured: true }],
        },
        routes: [{ vendor: 'anthropic', route: 'api' }, { vendor: 'openai', route: 'api' }],
    };
    providerSelect.value = 'anthropic';
    selector._handleProviderChange();
    assert.equal(commits.length, 0, 'vendor change must NOT commit — wait for user to pick a model');
});


test('model change DOES fire onModelChange (user commit)', () => {
    const { selector, providerSelect, modelSelect, commits } = loadSelector();
    selector.selectedProvider = 'anthropic';
    modelSelect.value = 'claude-opus-4-7';
    selector._handleModelChange();
    assert.equal(commits.length, 1);
    const [vendor, model, , route] = commits[0];
    assert.equal(vendor, 'anthropic');
    assert.equal(model, 'claude-opus-4-7');
});


test('route change DOES fire onModelChange and keeps the current model', () => {
    const { selector, providerSelect, modelSelect, routeSelect, commits } = loadSelector();
    selector.selectedProvider = 'anthropic';
    selector.selectedModel = 'claude-opus-4-7';
    routeSelect.value = 'plan';
    selector._handleRouteChange();
    assert.equal(commits.length, 1);
    const [vendor, model, , route] = commits[0];
    assert.equal(vendor, 'anthropic');
    assert.equal(model, 'claude-opus-4-7', 'route change must preserve the current model');
    assert.equal(route, 'plan');
});


test('getSelection returns vendor, route, and model', () => {
    const { selector } = loadSelector();
    selector.selectedProvider = 'anthropic';
    selector.selectedRoute = 'plan';
    selector.selectedModel = 'claude-opus-4-7';
    const s = selector.getSelection();
    assert.equal(s.vendor, 'anthropic');
    assert.equal(s.route, 'plan');
    assert.equal(s.model, 'claude-opus-4-7');
    // Legacy alias preserved.
    assert.equal(s.provider, 'anthropic');
});


test('MODEL_CHANGED marker with {vendor, route, model_name} updates all three dropdowns', () => {
    const { selector, providerSelect, modelSelect, routeSelect } = loadSelector();
    selector.allModelsData = {
        by_vendor: {
            anthropic: [
                { id: 'claude-opus-4-7', provider: 'anthropic', is_featured: true },
            ],
        },
        routes: [
            { vendor: 'anthropic', route: 'plan' },
            { vendor: 'anthropic', route: 'api' },
        ],
    };
    const content = 'OK\nMODEL_CHANGED:{"vendor":"anthropic","route":"plan","model_name":"claude-opus-4-7","model":"anthropic:plan/claude-opus-4-7"}';
    const changed = selector.checkForModelChange(content);
    assert.equal(changed, true);
    assert.equal(providerSelect.value, 'anthropic');
    assert.equal(modelSelect.value, 'claude-opus-4-7');
    assert.equal(routeSelect.value, 'plan');
    assert.equal(selector.selectedRoute, 'plan');
});


test('REGRESSION: MODEL_CHANGED marker with JSON-escaped quotes still parses', () => {
    // The agent response sometimes arrives with the MODEL_CHANGED payload
    // already JSON-stringified (quotes escaped as \"). This happened in
    // production: Jason saw `SyntaxError: Expected property name or '}' in
    // JSON at position 1` when JSON.parse hit `{\"model":...}` verbatim.
    const { selector, providerSelect, modelSelect, routeSelect } = loadSelector();
    selector.allModelsData = {
        by_vendor: {
            anthropic: [{ id: 'claude-opus-4-7', provider: 'anthropic', is_featured: true }],
        },
        routes: [
            { vendor: 'anthropic', route: 'plan' },
            { vendor: 'anthropic', route: 'api' },
        ],
    };
    // Escape-stringified payload: quotes are \" and newlines are \n.
    const escapedContent =
        '✓ Model set.\\n\\nMODEL_CHANGED:{\\"vendor\\":\\"anthropic\\",\\"route\\":\\"plan\\",\\"model_name\\":\\"claude-opus-4-7\\",\\"model\\":\\"anthropic:plan/claude-opus-4-7\\"}';
    const changed = selector.checkForModelChange(escapedContent);
    assert.equal(changed, true, 'escape-stringified payload must still parse');
    assert.equal(providerSelect.value, 'anthropic');
    assert.equal(modelSelect.value, 'claude-opus-4-7');
    assert.equal(routeSelect.value, 'plan');
});
