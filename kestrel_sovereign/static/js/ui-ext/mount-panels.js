// ============================================================================
// Mountable panel host (issue #2145, epic #2038 ticket 06 north star)
// ============================================================================
//
// Chat already exports an embed contract: a host (Frinz) mounts the chat pane
// same-document via `chat.js` `mount()`. This module gives the tab strip +
// lazily-rendered panel bodies the SAME contract via `mountPanels`.
//
// `mountPanels(containerEl, config)`:
//   - creates a `.nav-tabs` nav element and a `.main-content` panel host inside
//     `containerEl`,
//   - registers the core panels onto the `ui-ext` panel registry (their gates
//     come from the live `API.hasCapability`, so an embedder's
//     `KESTREL_UI_CONFIG.capabilities` opt-outs remove tabs unchanged),
//   - loads out-of-tree feature UI contributions (so feature panels — e.g. the
//     extracted Spawn panel — appear alongside core panels),
//   - renders the gated nav via `Panels.renderNav`,
//   - wires the SAME delegated tab-activation click handling core uses,
//   - returns a small API: `{ activate(panelId), destroy() }`.
//
// The delegated click handler + activation are factored here (`makePanelActivator`
// / `attachDelegatedNav`) and reused by core's `identity.js` boot so there is a
// single activation code path — `Panels.activate()` — for standalone and embed.
// ============================================================================

import API from '../api.js';
import Panels from './panels.js';
import { loadFeatureUIContributions } from './feature-loader.js';
import { CORE_PANEL_DEFS, buildCorePanelBody } from './core-panels.js';

function _cssEscape(s) {
    if (typeof CSS !== 'undefined' && typeof CSS.escape === 'function') return CSS.escape(s);
    return String(s).replace(/[^a-zA-Z0-9_-]/g, '\\$&');
}

// Track whether the core panels have been registered onto the shared registry
// singleton, so a second mount (or the core boot registering first) does not
// re-register — `registerPanel` replaces in place, so this is a cheap guard, not
// a correctness requirement.
let _coreRegistered = false;

/**
 * Register every core panel as a `registerPanel` contribution. Idempotent:
 * re-registering the same `panelId` replaces the prior contribution in place.
 * The gate is bound to the live `api` (default: the core `API` singleton) so
 * capability opt-outs flow through `api.hasCapability` unchanged. The `render`
 * builds the panel body ONLY into a registry-created (empty) container — in
 * standalone `index.html` the body pre-exists and `buildCorePanelBody` no-ops,
 * so the loader path in `identity.js` stays the single owner of standalone data
 * loading and there is no double render.
 *
 * @param {{api?: object}} [opts]
 */
export function registerCorePanels({ api = API } = {}) {
    for (const def of CORE_PANEL_DEFS) {
        Panels.registerPanel({
            panelId: def.panelId,
            label: def.label,
            labelKey: def.labelKey,
            icon: def.icon,
            before: def.before,
            gate: (ctx) => def.gate((ctx && ctx.api) || api),
            render: (bodyEl) => { buildCorePanelBody(bodyEl, def); },
        });
    }
    _coreRegistered = true;
}

/** True once `registerCorePanels` has run (test/boot introspection). */
export function coreePanelsRegistered() {
    return _coreRegistered;
}

/**
 * Build a scoped `activate(panelId)` that flips the active tab/panel classes
 * within the given scopes and routes through `Panels.activate()` (the single
 * activation path: lazy body render + `panel:shown` + `panel-section` zone).
 *
 * `tabScope`/`panelScope` are anything with `querySelectorAll` (a container or
 * `document`). `panelScope.getElementById` is used when present (document),
 * else a scoped `#panel-<id>` query.
 *
 * @param {object} opts
 * @param {ParentNode} opts.tabScope
 * @param {ParentNode} opts.panelScope
 * @param {object} opts.api
 * @param {(panelId: string) => void} [opts.beforeActivate] - runs after the
 *        class flip, before `Panels.activate` (core uses it for its lazy
 *        loaders + `state.currentPanel`).
 */
export function makePanelActivator({ tabScope, panelScope, api, beforeActivate }) {
    return function activate(panelId) {
        if (!panelId) return;
        tabScope.querySelectorAll('.nav-tab').forEach((t) => {
            t.classList.toggle('active', t.dataset.panel === panelId);
        });
        panelScope.querySelectorAll('.panel').forEach((p) => p.classList.remove('active'));
        const panel = typeof panelScope.getElementById === 'function'
            ? panelScope.getElementById(`panel-${panelId}`)
            : panelScope.querySelector(`#panel-${_cssEscape(panelId)}`);
        if (panel) panel.classList.add('active');
        if (typeof beforeActivate === 'function') beforeActivate(panelId);
        Panels.activate(panelId, { api });
    };
}

/**
 * Attach a single delegated click listener to the nav container so any
 * `.nav-tab` (including tabs inserted later by `registerPanel`) activates via
 * `activate`. Returns a detach function.
 *
 * @param {HTMLElement} navEl
 * @param {(panelId: string) => void} activate
 * @returns {() => void} detach
 */
export function attachDelegatedNav(navEl, activate) {
    if (!navEl) return () => {};
    const handler = (e) => {
        const tab = e.target && e.target.closest && e.target.closest('.nav-tab');
        if (!tab || !navEl.contains(tab)) return;
        activate(tab.dataset.panel);
    };
    navEl.addEventListener('click', handler);
    return () => navEl.removeEventListener('click', handler);
}

/**
 * Mount the full capability-gated nav-tab strip + lazily-rendered panel bodies
 * into `containerEl`, same-document — the embed counterpart to chat's `mount()`.
 *
 * @param {HTMLElement} containerEl
 * @param {object} [config]
 * @param {object} [config.api]           - API-like object (default: core singleton)
 * @param {boolean} [config.loadFeatures] - fetch/import feature UI contributions
 *                                           (default true) so feature panels mount
 * @param {boolean} [config.activateFirst]- activate the first visible tab after
 *                                           render (default true)
 * @param {boolean} [config.wireRuntime]  - wire embed-side data loading/init via
 *                                           core-panels-runtime (default true)
 * @returns {Promise<{activate(panelId: string): void, destroy(): void}>}
 */
export async function mountPanels(containerEl, config = {}) {
    if (!containerEl) throw new Error('mountPanels: containerEl is required');
    const api = config.api || API;
    const loadFeatures = config.loadFeatures !== false;
    const activateFirst = config.activateFirst !== false;
    const wireRuntime = config.wireRuntime !== false;

    const navEl = document.createElement('div');
    navEl.className = 'nav-tabs';
    const hostEl = document.createElement('div');
    hostEl.className = 'main-content';
    containerEl.appendChild(navEl);
    containerEl.appendChild(hostEl);

    // Register core panels BEFORE renderNav so their tabs + containers appear in
    // the first sync, gated by the live capabilities.
    registerCorePanels({ api });

    // Import out-of-tree feature UI contributions so feature panels (Spawn, …)
    // register their tabs alongside core. Best-effort: a manifest fetch failure
    // (e.g. no server in a test) must not abort the mount.
    if (loadFeatures) {
        try { await loadFeatureUIContributions(); } catch (_) { /* best-effort */ }
    }

    // Render the gated nav + panel containers into the mounted elements.
    Panels.renderNav({ navEl, hostEl, ctx: { api } });

    const activate = makePanelActivator({ tabScope: navEl, panelScope: hostEl, api });
    const detachNav = attachDelegatedNav(navEl, activate);

    // Wire embed-side data loading + event handlers for core panels. Standalone
    // owns loading through identity.js; embed has no such boot, so a runtime
    // module subscribes each core panel's loader/init to `panel:shown`. Loaded
    // lazily + best-effort so a heavy import failure never breaks the mount.
    // Capture its reset counterpart so destroy() tears the wiring down — the
    // runtime's init/load guards are per-mount, so a later remount must re-wire
    // against fresh DOM (else panels never re-init / bodies stay on "Loading…").
    let resetRuntime = null;
    if (wireRuntime) {
        try {
            const runtime = await import('./core-panels-runtime.js');
            if (runtime && typeof runtime.wireCorePanelRuntime === 'function') {
                runtime.wireCorePanelRuntime({ api });
            }
            if (runtime && typeof runtime.resetCorePanelRuntime === 'function') {
                resetRuntime = runtime.resetCorePanelRuntime;
            }
        } catch (_) { /* embed data-loading is best-effort */ }
    }

    if (activateFirst) {
        const first = navEl.querySelector('.nav-tab');
        if (first) activate(first.dataset.panel);
    }

    return {
        activate,
        destroy() {
            detachNav();
            navEl.remove();
            hostEl.remove();
            // Reset the embed runtime so a subsequent mountPanels re-wires +
            // reloads against fresh DOM (best-effort — never let teardown throw).
            if (resetRuntime) {
                try { resetRuntime(); } catch (_) { /* best-effort */ }
            }
        },
    };
}

export default mountPanels;
