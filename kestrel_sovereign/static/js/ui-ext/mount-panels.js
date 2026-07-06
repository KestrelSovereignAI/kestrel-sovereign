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
 * @param {Array<{panelId: string, label?: string, labelKey?: string, icon?: string, element: HTMLElement, before?: string}>} [config.hostTabs]
 *        - host-provided tabs (e.g. Frinz's own Chat surface). Each entry
 *          registers a normal registry tab whose body ADOPTS the host's live
 *          `element` (moved in on first activation, shown/hidden via the panel
 *          `active` class — never cloned or detached, so active SSE/event
 *          listeners survive tab switches). `destroy()` returns each element to
 *          its original parent/position. Ordering follows `before` (as in
 *          registerPanel); a host tab with no `before` registered first lands
 *          before the core panels (Chat-first).
 * @param {boolean|{toggleLabel?: string, anchor?: HTMLElement, scopeEl?: HTMLElement, onReveal?: (revealed: boolean) => void}} [config.reveal]
 *        - opt-in collapsed/"Advanced" mode (#2211). Omitted/falsy = today's
 *          always-visible nav strip (standalone console, unchanged). When set,
 *          the mount renders COLLAPSED: the nav strip is hidden (no tab headers)
 *          and only the first (leading/Chat) tab's body is visible. A single
 *          capability-gated "Advanced" toggle reveals the full gated strip
 *          (Chat first); `aria-pressed` is managed by the component. The toggle
 *          is either component-owned (a `<button>` inserted before the nav) or,
 *          when `reveal.anchor` is a host-provided element, that element wired in
 *          place. `reveal.toggleLabel` sets the component-owned button's text
 *          (default `'Advanced'`). If everything is gated off so only the first
 *          tab remains, there is nothing to reveal — no toggle is emitted (chat
 *          only). The returned handle gains `toggleReveal(next?)` and a
 *          `revealed` getter so a host can restore the revealed/collapsed state
 *          (alongside the active tab) across a `destroy()`/remount on agent
 *          switch: capture `handle.revealed` + the active tab before destroy,
 *          then after remount call `handle.toggleReveal(wasRevealed)` and
 *          `handle.activate(savedPanelId)`.
 * @returns {Promise<{activate(panelId: string): void, destroy(): void, toggleReveal(next?: boolean): boolean, revealed: boolean}>}
 */
export async function mountPanels(containerEl, config = {}) {
    if (!containerEl) throw new Error('mountPanels: containerEl is required');
    const api = config.api || API;
    const loadFeatures = config.loadFeatures !== false;
    const activateFirst = config.activateFirst !== false;
    const wireRuntime = config.wireRuntime !== false;
    const hostTabs = Array.isArray(config.hostTabs) ? config.hostTabs : [];
    const revealCfg = config.reveal;
    const revealEnabled = !!revealCfg;
    const revealOpts = (revealCfg && typeof revealCfg === 'object') ? revealCfg : {};

    const navEl = document.createElement('div');
    navEl.className = 'nav-tabs';
    const hostEl = document.createElement('div');
    hostEl.className = 'main-content';
    containerEl.appendChild(navEl);
    containerEl.appendChild(hostEl);

    // Register core panels BEFORE renderNav so their tabs + containers appear in
    // the first sync, gated by the live capabilities.
    registerCorePanels({ api });

    // Register host-provided tabs AFTER core so an explicit `before` anchoring a
    // core tab resolves (that tab already exists at sync time). A host tab with
    // no `before` should lead the strip (Chat-first for Frinz); multiple such
    // tabs keep their registration order. We do NOT anchor a no-`before` host tab
    // on a fixed core `panelId` — that core panel may be capability-gated off by
    // the embedder (e.g. Frinz opts out of `identity`), and `insertBefore` with a
    // missing anchor APPENDS, silently dropping Chat to the end. Instead we track
    // the no-`before` host ids in `_leadingHostIds` and explicitly move them to
    // the front of the nav AFTER render, regardless of which core panels survive
    // gating. Each host tab is an ordinary registry contribution (always gated
    // on) whose lazy `render` ADOPTS the host's live element — moved into the
    // registry-created body once and thereafter only shown/hidden via the panel
    // `active` class. We record the element's original placement so destroy() can
    // return it verbatim; the element is never cloned or detached, so its live
    // listeners persist across tab switches and remounts.
    const _adopted = [];
    const _leadingHostIds = [];
    for (const ht of hostTabs) {
        if (!ht || typeof ht.panelId !== 'string') continue;
        const element = ht.element || null;
        if (!ht.before) _leadingHostIds.push(ht.panelId);
        _adopted.push({
            panelId: ht.panelId,
            element,
            parent: element ? element.parentNode : null,
            nextSibling: element ? element.nextSibling : null,
            display: element ? element.style.display : '',
        });
        Panels.registerPanel({
            panelId: ht.panelId,
            label: ht.label,
            labelKey: ht.labelKey,
            icon: ht.icon,
            before: ht.before,
            gate: () => true,
            render: (bodyEl) => {
                // Move the host element in (same node — no clone). If it is
                // already here (a prior render), appendChild is a safe no-op.
                // Preserve the host's inline layout display (e.g. Frinz's
                // chat mount is inline `display:flex`) — tab visibility is
                // governed by the panel container's `active` class, so
                // clearing the element's own display would break host
                // layouts while mounted (codex P2 on #2164). Only a captured
                // `none` is cleared, so a host-hidden element still shows
                // inside its tab.
                if (element && bodyEl) {
                    const orig = _adopted.find((a) => a.element === element);
                    const d = orig ? orig.display : '';
                    element.style.display = d && d !== 'none' ? d : '';
                    bodyEl.appendChild(element);
                }
            },
        });
    }

    // Import out-of-tree feature UI contributions so feature panels (Spawn, …)
    // register their tabs alongside core. Best-effort: a manifest fetch failure
    // (e.g. no server in a test) must not abort the mount.
    if (loadFeatures) {
        try { await loadFeatureUIContributions(); } catch (_) { /* best-effort */ }
    }

    // Render the gated nav + panel containers into the mounted elements.
    Panels.renderNav({ navEl, hostEl, ctx: { api } });

    // Host tabs with no explicit `before` must LEAD the strip regardless of which
    // core panels the embedder gated off (else Chat falls to the end when the
    // first core panel — e.g. identity — is capability-disabled). Move them to
    // the front in registration order (reverse-iterate so the first stays first).
    for (let i = _leadingHostIds.length - 1; i >= 0; i--) {
        const tab = navEl.querySelector(`.nav-tab[data-panel="${_cssEscape(_leadingHostIds[i])}"]`);
        if (tab) navEl.insertBefore(tab, navEl.firstChild);
    }

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

    // ---- reveal mode (#2211) ------------------------------------------------
    // Opt-in collapsed/"Advanced" mode: chat-only by default, a capability-gated
    // toggle reveals the full nav strip. Omitted `reveal` leaves everything above
    // exactly as-is (standalone console unchanged).
    let _revealed = false;
    let _toggleEl = null;
    let _toggleOwned = false;
    let _togglePrevAria = null;
    let _detachToggle = () => {};

    function _applyRevealState() {
        // Collapsed: hide the whole nav strip so the embed shows chat-only (no
        // tab headers). Revealed: show the gated strip (Chat first).
        navEl.style.display = _revealed ? '' : 'none';
        if (_toggleEl) _toggleEl.setAttribute('aria-pressed', _revealed ? 'true' : 'false');
        // Host observability (#2211 addendum): host chrome OUTSIDE the mount
        // (e.g. an agent banner's management buttons) shows/hides with the
        // reveal state. Two zero-JS-friendly hooks, both fired/applied on every
        // state application including the initial one:
        //  - `panels-revealed` class on `reveal.scopeEl` (default: the mount
        //    container) so pure-CSS hosts can gate `[data-advanced-only]`.
        //  - `reveal.onReveal(revealed)` callback for hosts that need JS.
        const scope = (revealOpts.scopeEl && typeof revealOpts.scopeEl === 'object')
            ? revealOpts.scopeEl : containerEl;
        if (scope && scope.classList) scope.classList.toggle('panels-revealed', _revealed);
        if (typeof revealOpts.onReveal === 'function') {
            try { revealOpts.onReveal(_revealed); } catch (_) { /* host bug must not wedge the toggle */ }
        }
    }

    function _setRevealed(next) {
        _revealed = !!next;
        _applyRevealState();
        // With the strip hidden the user can't see or change the active tab, so
        // collapsing returns the visible body to the leading (Chat) tab.
        if (!_revealed) {
            const first = navEl.querySelector('.nav-tab');
            if (first) activate(first.dataset.panel);
        }
    }

    if (revealEnabled) {
        // Always start collapsed on the first (leading/Chat) tab, regardless of
        // the caller's `activateFirst`.
        const first = navEl.querySelector('.nav-tab');
        if (first) activate(first.dataset.panel);

        // The toggle only makes sense when there is MORE than the leading tab to
        // reveal. Everything-gated-off (chat only) ⇒ nothing worth revealing ⇒ no
        // toggle (acceptance: chat only, no toggle).
        if (navEl.querySelectorAll('.nav-tab').length > 1) {
            if (revealOpts.anchor && typeof revealOpts.anchor === 'object') {
                // Host-provided anchor: wire it in place, never move/create it.
                _toggleEl = revealOpts.anchor;
                _togglePrevAria = _toggleEl.getAttribute('aria-pressed');
            } else {
                _toggleEl = document.createElement('button');
                _toggleEl.type = 'button';
                _toggleEl.className = 'nav-advanced-toggle';
                _toggleEl.textContent = revealOpts.toggleLabel || 'Advanced';
                _toggleOwned = true;
                containerEl.insertBefore(_toggleEl, navEl);
            }
            const onToggle = () => _setRevealed(!_revealed);
            _toggleEl.addEventListener('click', onToggle);
            _detachToggle = () => _toggleEl.removeEventListener('click', onToggle);
        }
        _applyRevealState();
    }

    return {
        activate,
        get revealed() { return _revealed; },
        toggleReveal(next) {
            if (!revealEnabled) return _revealed;
            _setRevealed(next === undefined ? !_revealed : !!next);
            return _revealed;
        },
        destroy() {
            detachNav();
            _detachToggle();
            // Clear the reveal scope class so host chrome gated on it (e.g.
            // advanced-only banner buttons) doesn't stay revealed after the
            // mount is gone. onReveal is NOT fired here — destroy is teardown,
            // not a state change the host should react to (remount re-applies).
            if (revealEnabled) {
                const scope = (revealOpts.scopeEl && typeof revealOpts.scopeEl === 'object')
                    ? revealOpts.scopeEl : containerEl;
                if (scope && scope.classList) scope.classList.remove('panels-revealed');
            }
            if (_toggleEl) {
                if (_toggleOwned) {
                    _toggleEl.remove();
                } else if (_togglePrevAria === null) {
                    // Host-provided anchor: restore its prior aria-pressed state.
                    _toggleEl.removeAttribute('aria-pressed');
                } else {
                    _toggleEl.setAttribute('aria-pressed', _togglePrevAria);
                }
            }
            // Return each adopted host element to its original parent/position
            // and display BEFORE the panel host is removed, so the host keeps a
            // live element (same node, listeners intact) to use outside the
            // mount. Unregister the host tab so the registry drops no stale
            // reference.
            for (const rec of _adopted) {
                Panels.unregisterPanel(rec.panelId);
                const el = rec.element;
                if (!el) continue;
                el.style.display = rec.display;
                if (rec.parent) {
                    if (rec.nextSibling && rec.nextSibling.parentNode === rec.parent) {
                        rec.parent.insertBefore(el, rec.nextSibling);
                    } else {
                        rec.parent.appendChild(el);
                    }
                } else if (el.parentNode) {
                    // Had no original parent (freshly created, detached) — just
                    // detach it from our host so removing hostEl doesn't take it.
                    el.parentNode.removeChild(el);
                }
            }
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
