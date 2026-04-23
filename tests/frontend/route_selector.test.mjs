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


test('vendor change with single-model vendor commits immediately', () => {
    // REGRESSION: Jason picked llama_cpp, Kimi auto-populated as the only
    // model and local auto-selected as the only route, but because the old
    // handler refused to fire onModelChange on vendor changes, the server
    // never learned about it. Clicking away and back showed the prior
    // server state (anthropic:plan/opus), not llama_cpp/Kimi.
    const { selector, providerSelect, modelSelect, routeSelect, commits } = loadSelector();
    selector.allModelsData = {
        by_vendor: {
            anthropic: [
                { id: 'claude-opus-4-7', provider: 'anthropic', is_featured: true },
            ],
            llama_cpp: [
                { id: 'Kimi-K2.5.gguf', provider: 'llama_cpp', is_featured: true },
            ],
        },
        routes: [
            { vendor: 'anthropic', route: 'plan' },
            { vendor: 'anthropic', route: 'api' },
            { vendor: 'llama_cpp', route: 'local', is_local: true },
        ],
    };
    // Simulate the sync that ran when the agent was first viewed.
    selector.isInitialLoad = false;
    selector._lastSyncedSelection = {
        vendor: 'anthropic', model: 'claude-opus-4-7', route: 'plan',
    };

    providerSelect.value = 'llama_cpp';
    selector._handleProviderChange();

    assert.equal(commits.length, 1, 'single-model vendor switch must commit');
    const [vendor, model, , route] = commits[0];
    assert.equal(vendor, 'llama_cpp');
    assert.equal(model, 'Kimi-K2.5.gguf');
    assert.equal(route, 'local');
});


test('vendor change does NOT double-commit when state already matches server', () => {
    // User picks a vendor they're already on — no redundant POST.
    const { selector, providerSelect, commits } = loadSelector();
    selector.allModelsData = {
        by_vendor: {
            anthropic: [
                { id: 'claude-opus-4-7', provider: 'anthropic', is_featured: true },
            ],
        },
        routes: [{ vendor: 'anthropic', route: 'plan' }],
    };
    selector.isInitialLoad = false;
    selector._lastSyncedSelection = {
        vendor: 'anthropic', model: 'claude-opus-4-7', route: 'plan',
    };

    providerSelect.value = 'anthropic';
    selector._handleProviderChange();
    assert.equal(commits.length, 0, 'no commit when dropdowns settle to server state');
});


test('isInitialLoad suppresses auto-commit during constructor/sync', () => {
    // During the initial page render, syncWithServer may adjust the
    // dropdowns to match server state. Those adjustments must not trigger
    // their own commits back.
    const { selector, providerSelect, commits } = loadSelector();
    selector.allModelsData = {
        by_vendor: {
            anthropic: [
                { id: 'claude-opus-4-7', provider: 'anthropic', is_featured: true },
            ],
        },
        routes: [{ vendor: 'anthropic', route: 'plan' }],
    };
    // isInitialLoad defaults to true in the constructor.
    providerSelect.value = 'anthropic';
    selector._handleProviderChange();
    assert.equal(commits.length, 0, 'no commits during initial load');
});


test('model change commits with new model but same vendor', () => {
    const { selector, providerSelect, modelSelect, commits } = loadSelector();
    selector.isInitialLoad = false;
    selector._lastSyncedSelection = {
        vendor: 'anthropic', model: 'claude-sonnet-4-6', route: 'api',
    };
    selector.selectedProvider = 'anthropic';
    selector.selectedRoute = 'api';
    modelSelect.value = 'claude-opus-4-7';
    selector._handleModelChange();
    assert.equal(commits.length, 1);
    const [vendor, model, , route] = commits[0];
    assert.equal(vendor, 'anthropic');
    assert.equal(model, 'claude-opus-4-7');
    assert.equal(route, 'api');
});


test('route change commits, keeps current model', () => {
    const { selector, routeSelect, commits } = loadSelector();
    selector.isInitialLoad = false;
    selector._lastSyncedSelection = {
        vendor: 'anthropic', model: 'claude-opus-4-7', route: 'api',
    };
    selector.selectedProvider = 'anthropic';
    selector.selectedModel = 'claude-opus-4-7';
    routeSelect.value = 'plan';
    selector._handleRouteChange();
    assert.equal(commits.length, 1);
    const [vendor, model, , route] = commits[0];
    assert.equal(vendor, 'anthropic');
    assert.equal(model, 'claude-opus-4-7', 'route change must preserve the model');
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
