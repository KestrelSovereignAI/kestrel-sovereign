import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

// Same browser stand-in the other frontend tests use: feature-store.js reads
// window/document at import time.
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

const API = (await import('../../kestrel_sovereign/static/js/api.js')).default;
const { Modal } = await import('../../kestrel_sovereign/static/js/ui.js');
const { renderActionButton, loadFeatureStore } = await import('../../kestrel_sovereign/static/js/feature-store.js');

const modalButtons = () => [...document.querySelectorAll('.modal-btn')].map((b) => b.textContent.trim());

// #3234: every card action is a mutation the server gates on sovereign
// authority; a host-scope row cannot be disabled by anyone.
const STATUSES = ['enabled', 'disabled', 'installed', 'available'];

test('a caller who cannot manage features gets no control, whatever the status', () => {
    for (const status of STATUSES) {
        const html = renderActionButton({ name: 'x', status, core: false }, false);
        assert.equal(html, '', status);
    }
});

test('a managing caller gets the action for each status', () => {
    const expected = {
        enabled: 'Disable',
        disabled: 'Enable',
        installed: 'Enable',
        available: 'Install',
    };
    for (const [status, label] of Object.entries(expected)) {
        const html = renderActionButton({ name: 'x', status, core: false }, true);
        assert.ok(html.includes(`>${label}</button>`), `${status} -> ${label}: ${html}`);
    }
});

test('a row the server refuses to disable renders no Disable; an enabled core row still does', () => {
    for (const refused of [
        { name: 'restart_coordinator', status: 'enabled', core: true, host_scope: true, disable_refusal: 'Host-scope features cannot be disabled per agent' },
        { name: 'identity', status: 'enabled', core: true, host_scope: false, disable_refusal: 'Mandatory sovereignty features cannot be disabled' },
    ]) {
        assert.equal(renderActionButton(refused, true), '', refused.name);
    }
    const core = renderActionButton(
        { name: 'web_search', status: 'enabled', core: true, host_scope: false, disable_refusal: null },
        true,
    );
    assert.ok(core.includes('>Disable</button>'), core);
});

test('the feature grid draws no action control for a caller who cannot manage features', async () => {
    const grid = document.createElement('div');
    grid.id = 'feature-grid';
    document.body.appendChild(grid);
    const originalHas = API.hasCapability;
    const originalRequest = API.request;
    API.hasCapability = () => true;
    try {
        for (const [canManage, expectButtons] of [[false, 0], [true, 2]]) {
            API.request = async () => ({
                features: [
                    { name: 'web_search', status: 'enabled', core: true, disable_refusal: null, description: 'x' },
                    { name: 'voice', status: 'available', core: false, disable_refusal: null, description: 'y' },
                ],
                count: 2,
                can_manage_features: canManage,
            });
            await loadFeatureStore();
            const buttons = grid.querySelectorAll('.feature-action-btn');
            assert.equal(buttons.length, expectButtons, `can_manage_features=${canManage}`);
        }
    } finally {
        API.hasCapability = originalHas;
        API.request = originalRequest;
        grid.remove();
    }
});

test('the detail modal offers no mutation to a caller who cannot manage features', async () => {
    const originalRequest = API.request;
    API.request = async () => ({ features: [], count: 0, can_manage_features: false });
    const grid = document.createElement('div');
    grid.id = 'feature-grid';
    document.body.appendChild(grid);
    const originalHas = API.hasCapability;
    API.hasCapability = () => true;
    try {
        await loadFeatureStore(); // publishes can_manage_features=false
        API.request = async () => ({
            name: 'web_search', status: 'enabled', core: true, disable_refusal: null, description: 'x',
        });
        await window.FeatureStore.showDetail('web_search');
        const labels = modalButtons();
        assert.ok(!labels.includes('Disable') && !labels.includes('Remove') && !labels.includes('Enable') && !labels.includes('Install'), labels);
    } finally {
        Modal.hide();
        API.request = originalRequest;
        API.hasCapability = originalHas;
        grid.remove();
    }
});

test('the detail modal follows the server disable answer for a managing caller', async () => {
    const originalRequest = API.request;
    const originalHas = API.hasCapability;
    const grid = document.createElement('div');
    grid.id = 'feature-grid';
    document.body.appendChild(grid);
    API.hasCapability = () => true;
    try {
        API.request = async () => ({ features: [], count: 0, can_manage_features: true });
        await loadFeatureStore(); // publishes can_manage_features=true
        const cases = [
            [{ name: 'identity', status: 'enabled', core: true, disable_refusal: 'Mandatory sovereignty features cannot be disabled' }, { Disable: false, Remove: false }],
            [{ name: 'restart_coordinator', status: 'enabled', core: true, disable_refusal: 'Host-scope features cannot be disabled per agent' }, { Disable: false, Remove: false }],
            [{ name: 'web_search', status: 'enabled', core: true, disable_refusal: null }, { Disable: true, Remove: false }],
            [{ name: 'voice', status: 'enabled', core: false, disable_refusal: null }, { Disable: true, Remove: true }],
        ];
        for (const [detail, expected] of cases) {
            API.request = async () => ({ description: 'x', ...detail });
            await window.FeatureStore.showDetail(detail.name);
            const labels = modalButtons();
            for (const [label, present] of Object.entries(expected)) {
                assert.equal(labels.includes(label), present, `${detail.name}: ${label} in ${labels}`);
            }
            Modal.hide();
        }
    } finally {
        Modal.hide();
        API.request = originalRequest;
        API.hasCapability = originalHas;
        grid.remove();
    }
});

test('the config form offers no Save when the config read says the caller cannot manage', async () => {
    const originalRequest = API.request;
    API.request = async (path) => {
        assert.match(path, /\/config$/);
        return {
            config_schema: { properties: { enabled: { type: 'boolean', title: 'Enabled' } } },
            config: { enabled: true },
            can_manage_features: false,
        };
    };
    try {
        await window.FeatureStore.showConfigForm('configurable');
        const labels = modalButtons();
        assert.ok(labels.includes('Cancel') && !labels.includes('Save'), labels);
    } finally {
        Modal.hide();
        API.request = originalRequest;
    }
});

test('the default authority is whatever the last catalog read published', async () => {
    const grid = document.createElement('div');
    grid.id = 'feature-grid';
    document.body.appendChild(grid);
    const originalHas = API.hasCapability;
    const originalRequest = API.request;
    API.hasCapability = () => true;
    try {
        for (const canManage of [true, false, true]) {
            API.request = async () => ({ features: [], count: 0, can_manage_features: canManage });
            await loadFeatureStore();
            const html = renderActionButton({ name: 'x', status: 'available', core: false });
            assert.equal(html !== '', canManage, `after a read publishing ${canManage}`);
        }
    } finally {
        API.hasCapability = originalHas;
        API.request = originalRequest;
        grid.remove();
    }
});
