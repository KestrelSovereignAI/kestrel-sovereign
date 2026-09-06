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

await import('../../kestrel_sovereign/static/js/api.js');
await import('../../kestrel_sovereign/static/js/ui.js');
const { renderActionButton } = await import('../../kestrel_sovereign/static/js/feature-store.js');

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

test('an enabled host-scope row renders no Disable, an enabled core row still does', () => {
    const hostScope = renderActionButton(
        { name: 'restart_coordinator', status: 'enabled', core: true, host_scope: true },
        true,
    );
    assert.equal(hostScope, '');
    const core = renderActionButton(
        { name: 'web_search', status: 'enabled', core: true, host_scope: false },
        true,
    );
    assert.ok(core.includes('>Disable</button>'), core);
});

test('the default authority is what the last catalog read published (false before any read)', () => {
    assert.equal(renderActionButton({ name: 'x', status: 'available', core: false }), '');
});
