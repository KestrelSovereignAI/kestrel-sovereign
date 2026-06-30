// Panel teardown-on-disable (#2048, epic #2038 ticket 10).
//
// When a registry-owned panel gates off at runtime (its feature disabled) while
// it is the ACTIVE panel, the registry detaches the `#panel-<id>` node. A bare
// detach fires no `active`-class mutation, so panel code that keys teardown off
// losing `active` (e.g. Spawn's auto-refresh MutationObserver) would never stop
// its work — the interval keeps issuing hidden /api/spawn/children requests.
//
// The registry must run the deactivation path BEFORE detaching: strip the
// `active` class (drives the observer path) AND emit `panel:hidden` (the
// deterministic teardown path spawn.js subscribes to). These tests assert both.

import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.Node = dom.window.Node;
globalThis.MutationObserver = dom.window.MutationObserver;
if (!globalThis.CSS || typeof globalThis.CSS.escape !== 'function') {
    globalThis.CSS = { escape: (s) => String(s).replace(/[^a-zA-Z0-9_-]/g, '\\$&') };
}

const Panels = (await import('../../kestrel_sovereign/static/js/ui-ext/panels.js')).default;
const bus = (await import('../../kestrel_sovereign/static/js/ui-ext/bus.js')).default;

function freshNav() {
    document.body.innerHTML = '<div id="nav"></div><div id="host"></div>';
    return {
        navEl: document.getElementById('nav'),
        hostEl: document.getElementById('host'),
    };
}

test('disabling an active registry panel fires panel:hidden so its work stops', () => {
    Panels._reset();
    const { navEl, hostEl } = freshNav();

    let gateOpen = true;
    Panels.registerPanel({ panelId: 'spawn', label: 'Spawn', gate: () => gateOpen });
    Panels.renderNav({ navEl, hostEl, ctx: {} });

    const panel = document.getElementById('panel-spawn');
    assert.ok(panel, 'registry created the #panel-spawn container');

    // Simulate the panel being viewed: it is `active` and has live work running
    // that a `panel:hidden` teardown (exactly what spawn.js wires) must stop.
    panel.classList.add('active');
    let working = true;
    const off = bus.on('panel:hidden', (p) => {
        if (p && p.panelId === 'spawn') working = false;
    });

    // Feature disabled -> gate flips off -> registry re-gates.
    gateOpen = false;
    Panels.syncNav();

    assert.equal(working, false, 'panel:hidden fired so the panel stopped its work');
    assert.equal(document.getElementById('panel-spawn'), null, 'panel node detached');
    assert.equal(navEl.querySelector('.nav-tab[data-panel="spawn"]'), null, 'tab removed');
    off();
});

test('the active class is stripped before detach (drives the observer teardown path)', async () => {
    Panels._reset();
    const { navEl, hostEl } = freshNav();

    let gateOpen = true;
    Panels.registerPanel({ panelId: 'spawn', label: 'Spawn', gate: () => gateOpen });
    Panels.renderNav({ navEl, hostEl, ctx: {} });

    const panel = document.getElementById('panel-spawn');
    panel.classList.add('active');

    // Mirror spawn.js setupAutoRefresh(): a MutationObserver that stops work the
    // moment the panel loses `active`. A bare detach would never trip this.
    let observedStop = false;
    const observer = new MutationObserver((mutations) => {
        for (const m of mutations) {
            if (m.attributeName === 'class' && !panel.classList.contains('active')) {
                observedStop = true;
            }
        }
    });
    observer.observe(panel, { attributes: true });

    gateOpen = false;
    Panels.syncNav();

    // MutationObserver callbacks are microtasks; let them drain.
    await Promise.resolve();
    await new Promise((r) => setTimeout(r, 0));

    assert.equal(observedStop, true, 'class mutation to non-active was observed before detach');
    observer.disconnect();
});
