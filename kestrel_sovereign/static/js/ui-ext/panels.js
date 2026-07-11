// ============================================================================
// Panel contribution registry (ticket 06, epic #2038)
// ============================================================================
//
// Data-drives the two coarse-grained zones the positional `UIContribution`
// contract (registry.js) deliberately does NOT cover: whole `nav-tabs` and
// their lazily-rendered `panel-root` bodies. Today nav tabs are a static list in
// `index.html` and a hardcoded `setLazyLoaders({...})` dispatch in
// `identity.js`. This registry makes them a single data-driven code path so a
// FEATURE can add a nav panel with no core edits — and so at least one core
// panel (Metrics) runs through the same path instead of a hardcoded special
// case.
//
// A panel contribution declares:
//
//   registerPanel({
//     panelId,            // stable id; the panel DOM is `#panel-<panelId>`
//     label, labelKey,    // nav tab text (labelKey drives i18n re-hydration)
//     icon,               // optional icon class for the nav tab
//     before,             // optional: insert the tab before this panelId's tab
//     gate?(ctx),         // capability gate; an ungated-false panel shows no tab
//     render?(bodyEl, ctx)// lazily invoked ONCE on first activation
//   })
//
// `render` preserves the existing lazy-load semantics (`setLazyLoaders`): a
// panel's body is rendered the first time its tab is activated, not at boot.
// For a core panel migrated IN PLACE the `#panel-<id>` body already exists in
// `index.html`, so `render` just calls the panel's existing loader; for an
// out-of-tree FEATURE panel the registry creates the `#panel-<id>` container in
// the panel host and `render` fills it.
//
// `panel-section` (the finer-grained, per-sub-capability sub-section zone the
// Resources composite panel needs) is governed by the positional registry
// (registry.js): on activation this module calls `UI.renderSlot('panel-section',
// {element, panelId, api})` so section contributions mount into the active
// panel, gated by `ctx.panelId`. A `panel:shown` bus event is emitted so
// section contributions re-render lazily on first reveal.
// ============================================================================

import { UI } from './registry.js';
import bus from './bus.js';
import { storeGet, storeSet } from '../ui_state.mjs';

/** @type {Map<string, object>} */
const _panels = new Map();
/** @type {Set<string>} contributed panels whose body `render` has already run. */
const _rendered = new Set();

let _navEl = null;
let _hostEl = null;
let _ctx = null;

function _safe(fn, ...args) {
    try {
        return fn(...args);
    } catch (err) {
        console.error('[ui-ext panels] contribution callback threw:', err);
        return undefined;
    }
}

function _escapeAttr(s) {
    return String(s == null ? '' : s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function _cssEscape(s) {
    if (typeof CSS !== 'undefined' && typeof CSS.escape === 'function') return CSS.escape(s);
    return String(s).replace(/[^a-zA-Z0-9_-]/g, '\\$&');
}

function _tabFor(panelId) {
    if (!_navEl) return null;
    return _navEl.querySelector(`.nav-tab[data-panel="${_cssEscape(panelId)}"]`);
}

function _gateOk(def) {
    if (typeof def.gate !== 'function') return true;
    return !!_safe(def.gate, _ctx || {});
}

function _buildTab(def) {
    const btn = document.createElement('button');
    btn.className = 'nav-tab';
    btn.dataset.panel = def.panelId;
    // Mark registry-owned tabs so core's PANEL_CAPABILITIES re-gate
    // (reconcileNavigationCapabilities) leaves them to the registry's own gate.
    btn.dataset.panelRegistry = 'true';
    const labelKey = def.labelKey || '';
    const iconHtml = def.icon ? `<span class="${_escapeAttr(def.icon)}"></span> ` : '';
    const keyAttr = labelKey ? ` data-label-key="${_escapeAttr(labelKey)}"` : '';
    btn.innerHTML = `${iconHtml}<span${keyAttr}>${_escapeAttr(def.label || def.panelId)}</span>`;
    if (labelKey) btn.dataset.labelKey = labelKey;
    return btn;
}

function _ensurePanelContainer(def) {
    if (!_hostEl) return;
    if (document.getElementById(`panel-${def.panelId}`)) return; // in-place core panel
    const panel = document.createElement('div');
    panel.className = 'panel';
    panel.id = `panel-${def.panelId}`;
    // Mark registry-created containers so a gated-off panel can drop its body
    // (an in-place core panel declared in index.html is never removed here).
    panel.dataset.registryOwned = 'true';
    const content = document.createElement('div');
    content.className = 'panel-content';
    panel.appendChild(content);
    _hostEl.appendChild(panel);
}

// Build/refresh nav tabs and panel containers for every registered panel.
// Tabs already present (built earlier, or a core panel still declared in
// index.html) are left in place; a gated-off panel's tab is removed.
function _syncNav() {
    if (!_navEl) return;
    for (const def of _panels.values()) {
        const tab = _tabFor(def.panelId);
        // #2145 (P2-1): adopt an in-place core panel body (declared in
        // index.html, so NOT registry-created) as removable. Pre-#2145 the
        // `panelIsEnabled` prune in initNavigation removed a gated-off core
        // panel's tab AND its `#panel-<id>` body; once gating moved to the
        // registry, only `registryOwned` bodies were dropped, stranding a
        // gated-off core panel's in-place body in the DOM. Marking it here (on
        // every sync, while the gate is still being evaluated) restores the
        // contract: the gate-off branch below removes it, and a re-enable
        // recreates + re-renders it from CORE_PANEL_DEFS via buildCorePanelBody.
        const existing = document.getElementById(`panel-${def.panelId}`);
        if (existing && existing.dataset.registryOwned !== 'true') {
            existing.dataset.registryAdopted = 'true';
        }
        if (!_gateOk(def)) {
            if (tab) tab.remove();
            // Drop the panel body when the panel gates off at runtime (feature
            // disabled) or at boot (host opt-out), and forget its rendered state
            // so a re-enable re-renders fresh. Both registry-created bodies and
            // adopted in-place core bodies are removable.
            const panel = existing;
            if (panel) {
                // Run the panel's deactivation path BEFORE detaching/hiding it.
                // Panel code keys teardown off losing the `active` class (e.g.
                // Spawn's auto-refresh MutationObserver), and a detached node
                // fires no class mutation — so a gated-off panel viewed live
                // would keep doing work (polling /api/spawn/children) forever.
                // Strip `active` (drives the observer path) and fire an explicit
                // `panel:hidden` teardown event (the deterministic path).
                if (panel.classList.contains('active')) {
                    panel.classList.remove('active');
                }
                bus.emit('panel:hidden', { panelId: def.panelId });
                if (panel.dataset.registryOwned === 'true'
                    || panel.dataset.registryAdopted === 'true') {
                    panel.remove();
                }
            }
            _rendered.delete(def.panelId);
            continue;
        }
        if (!tab) {
            const built = _buildTab(def);
            const ref = def.before ? _tabFor(def.before) : null;
            _navEl.insertBefore(built, ref || null);
        } else {
            // Adopt an in-place tab (a core panel still declared in index.html):
            // mark it registry-owned so core's PANEL_CAPABILITIES re-gate
            // (reconcileNavigationCapabilities) leaves gating to this registry's
            // own `gate` — a single gating mechanism per tab (#2145).
            tab.dataset.panelRegistry = 'true';
        }
        _ensurePanelContainer(def);
    }
}

/**
 * Register a panel contribution. Re-registering the same `panelId` replaces the
 * prior contribution. If the nav has already been rendered, the new panel's tab
 * is inserted immediately (so a late/feature registration appears without a
 * reload).
 *
 * @param {object} def
 */
export function registerPanel(def) {
    if (!def || typeof def.panelId !== 'string') {
        console.error('[ui-ext panels] registerPanel: contribution needs a string `panelId`');
        return;
    }
    _panels.set(def.panelId, def);
    _rendered.delete(def.panelId);
    if (_navEl) _syncNav();
}

/**
 * Remove a panel contribution by id, dropping its tab (and a registry-created
 * panel container).
 * @param {string} panelId
 */
export function unregisterPanel(panelId) {
    if (!_panels.has(panelId)) return;
    _panels.delete(panelId);
    _rendered.delete(panelId);
    const tab = _tabFor(panelId);
    if (tab) tab.remove();
}

/**
 * Core boot hook: bind the nav-tab container and the panel host, then render
 * every registered panel's tab/container. Must run BEFORE `initNavigation`
 * wires click handlers, so registry-built tabs get the same generic activation.
 *
 * @param {{navEl: HTMLElement, hostEl: HTMLElement, ctx?: object}} opts
 */
export function renderNav({ navEl, hostEl, ctx } = {}) {
    _navEl = navEl || null;
    _hostEl = hostEl || null;
    _ctx = ctx || {};
    _syncNav();
}

/**
 * Re-evaluate every registered panel's gate against the bound nav/host and
 * add/remove its tab + container accordingly. Core calls this when capabilities
 * change at runtime (a feature enabled/disabled) so registry panels appear or
 * disappear without a page reload. No-op until `renderNav` has bound the nav.
 */
export function syncNav() {
    _syncNav();
}

/**
 * Activate a panel: lazily render a contributed panel's body the first time it
 * is shown (idempotent), then emit `panel:shown` and render the `panel-section`
 * zone for this panel. Safe to call for panels NOT registered here (core panels
 * still declared in index.html) — it skips the body render and just fires the
 * section/event path so `panel-section` contributions and bus subscribers work
 * uniformly across all panels.
 *
 * @param {string} panelId
 * @param {object} [ctx]  - merged over the boot ctx (e.g. `{ api }`).
 */
export function activate(panelId, ctx = {}) {
    const merged = { ...(_ctx || {}), ...ctx, panelId };
    const def = _panels.get(panelId);
    const root = document.getElementById(`panel-${panelId}`);
    if (def && !_rendered.has(panelId)) {
        _rendered.add(panelId);
        const body = (root && root.querySelector('.panel-content')) || root;
        if (typeof def.render === 'function' && body) {
            _safe(def.render, body, merged);
        }
    }
    // Mount per-(panelId) sub-sections into the active panel's content. Gated by
    // ctx.panelId inside each contribution (PanelSectionContext).
    const sectionAnchor = (root && root.querySelector('.panel-content')) || root;
    if (sectionAnchor) {
        UI.renderSlot('panel-section', { element: sectionAnchor, panelId, api: merged.api });
    }
    bus.emit('panel:shown', { panelId });
}

/** The registered panels, in registration order (read-only snapshot; tests). */
export function panels() {
    return [..._panels.values()];
}

/** Test/teardown affordance: forget all panels and unbind the nav/host. */
export function _reset() {
    _panels.clear();
    _rendered.clear();
    _navEl = null;
    _hostEl = null;
    _ctx = null;
}

// ============================================================================
// Reveal state machine (#2211 / #2229)
// ============================================================================
//
// The single, shared "chat-first / Advanced toggle" reveal implementation used
// by BOTH the embeddable `mountPanels` host and the standalone console boot
// (identity.js). Extracted here (#2229) so there is exactly one reveal code path
// — collapse hides the whole nav strip (chat only), an `aria-pressed` toggle
// reveals the gated strip (Chat first), collapsing returns to the leading tab,
// and the toggle is hidden whenever only a single tab is available (tracked live
// via a nav MutationObserver, so a late-registered feature panel wires the toggle
// in and gating back down to chat-only hides it again).
//
// Both callers differ only in DOM plumbing, which is fully parameterized:
//   - `navEl`      the nav-tab strip to show/hide (created by mountPanels;
//                  the static `.nav-tabs` in index.html for standalone).
//   - `activate`   the panel activator (collapse re-activates the leading tab).
//   - `anchor`     a host-provided toggle element (standalone's chat-header
//                  button, or an embedder's own button). When absent, a
//                  component-owned `<button.nav-advanced-toggle>` is created and
//                  placed via `mountToggle`.
//   - `scopeEl`    element that carries the `panels-revealed` class (embeds gate
//                  host chrome on it). Defaults to `navEl`.
//   - `onReveal`   host callback fired on every state application (incl. initial).
//   - `storageKey` localStorage key persisting the revealed state across reloads
//                  (default on for both callers). `false`/`null` disables it.

// Reveal-state persistence delegates to the shared ui_state.mjs raw helpers
// (#2298). The on-disk format ('1'/'0', with legacy 'true' still read) is
// unchanged, so no stored `panels-revealed` state migrates or breaks.

/**
 * Wire the collapsed/"Advanced" reveal behavior against an existing nav strip.
 *
 * @param {object} opts
 * @param {HTMLElement} opts.navEl              nav-tab strip to reveal/collapse
 * @param {(panelId: string) => void} opts.activate  activator (collapse returns to leading tab)
 * @param {HTMLElement} [opts.anchor]           host-provided toggle element
 * @param {string} [opts.toggleLabel]           component-owned button text (default 'Advanced')
 * @param {string} [opts.toggleClassName]       component-owned button class (default 'nav-advanced-toggle')
 * @param {(btn: HTMLElement) => void} [opts.mountToggle]  place a component-owned button
 * @param {HTMLElement} [opts.scopeEl]          element carrying `panels-revealed` (default navEl)
 * @param {(revealed: boolean) => void} [opts.onReveal]    host callback
 * @param {string|false|null} [opts.storageKey] localStorage persistence key (default off unless provided)
 * @returns {{revealed: boolean, setRevealed(next?: boolean): boolean, toggleReveal(next?: boolean): boolean, syncToggle(): void, destroy(): void} | null}
 */
export function initReveal(opts = {}) {
    const {
        navEl,
        activate,
        anchor = null,
        toggleLabel = 'Advanced',
        toggleClassName = 'nav-advanced-toggle',
        mountToggle = null,
        scopeEl = null,
        onReveal = null,
        storageKey = null,
    } = opts;
    if (!navEl) return null;

    const _anchor = (anchor && typeof anchor === 'object') ? anchor : null;
    const _scope = (scopeEl && typeof scopeEl === 'object') ? scopeEl : navEl;
    const _key = (typeof storageKey === 'string' && storageKey) ? storageKey : null;

    let _revealed = false;
    let _toggleEl = null;
    let _toggleOwned = false;
    let _togglePrevAria = null;
    let _detachToggle = () => {};
    let _navObserver = null;

    function _readPersisted() {
        if (!_key) return null;
        const v = storeGet(_key);
        if (v === null || v === undefined) return null;
        return v === '1' || v === 'true';
    }

    function _writePersisted(val) {
        if (!_key) return;
        storeSet(_key, val ? '1' : '0');
    }

    function _applyRevealState() {
        // Collapsed: hide the whole nav strip (chat-only, no tab headers).
        // Revealed: show the gated strip (Chat first).
        navEl.style.display = _revealed ? '' : 'none';
        if (_toggleEl) _toggleEl.setAttribute('aria-pressed', _revealed ? 'true' : 'false');
        // Host observability: a `panels-revealed` class on `scopeEl` (zero-JS
        // hosts gate CSS on it) plus an `onReveal` callback for JS hosts. Both
        // fire on every application, including the initial one.
        if (_scope && _scope.classList) _scope.classList.toggle('panels-revealed', _revealed);
        if (typeof onReveal === 'function') {
            try { onReveal(_revealed); } catch (_) { /* host bug must not wedge the toggle */ }
        }
    }

    function _setRevealed(next, { persist = true } = {}) {
        _revealed = !!next;
        _applyRevealState();
        if (persist) _writePersisted(_revealed);
        // With the strip hidden the user can't see or change the active tab, so
        // collapsing returns the visible body to the leading (Chat) tab.
        if (!_revealed) {
            const first = navEl.querySelector('.nav-tab');
            if (first && typeof activate === 'function') activate(first.dataset.panel);
        }
    }

    function _ensureToggleWired() {
        if (_toggleEl) return;
        if (_anchor) {
            // Host-provided anchor: wire it in place, never move/create it.
            _toggleEl = _anchor;
            _togglePrevAria = _toggleEl.getAttribute('aria-pressed');
        } else {
            _toggleEl = document.createElement('button');
            _toggleEl.type = 'button';
            _toggleEl.className = toggleClassName;
            _toggleEl.textContent = toggleLabel || 'Advanced';
            _toggleOwned = true;
            if (typeof mountToggle === 'function') mountToggle(_toggleEl);
        }
        const onToggle = () => _setRevealed(!_revealed);
        _toggleEl.addEventListener('click', onToggle);
        _detachToggle = () => _toggleEl.removeEventListener('click', onToggle);
    }

    // The toggle only makes sense when there is MORE than the leading tab to
    // reveal. A host-provided anchor is wired eagerly (it pre-exists in the
    // host's DOM, so we must be able to hide it when there is nothing to
    // reveal); a component-owned button is only CREATED once a second tab
    // exists (so a chat-only mount emits no button at all). Tab availability is
    // DYNAMIC — feature panels register after their manifest loads — so this is
    // re-evaluated on every nav mutation, not once at init.
    function _syncToggle() {
        const multi = navEl.querySelectorAll('.nav-tab').length > 1;
        if (_anchor || multi) _ensureToggleWired();
        if (_toggleEl) _toggleEl.style.display = multi ? '' : 'none';
        // Regating down to chat-only collapses a then-meaningless revealed strip,
        // but must NOT clobber the persisted preference — returning tabs (or a
        // reload) should honor "operator lives in Advanced".
        if (!multi && _revealed) _setRevealed(false, { persist: false });
        // ...and the inverse (codex P3 on #2231): when tabs ARRIVE — a second
        // tab registers after boot, or regated tabs return — a persisted
        // "revealed" preference must be reapplied, not just the toggle shown.
        // Without this, an operator whose advanced tabs load late (feature
        // manifests) boots collapsed forever despite the stored preference.
        if (multi && !_revealed && _readPersisted() === true) {
            _setRevealed(true, { persist: false });
        }
    }

    // Start collapsed on the leading (Chat) tab, then wire the toggle + observer.
    const first = navEl.querySelector('.nav-tab');
    if (first && typeof activate === 'function') activate(first.dataset.panel);
    _syncToggle();
    _navObserver = new MutationObserver(_syncToggle);
    _navObserver.observe(navEl, { childList: true });

    // Restore the persisted revealed state, but only when there is something to
    // reveal (the toggle is present + visible). Otherwise apply the collapsed
    // state (which also fires the initial onReveal / scope-class application).
    const persisted = _readPersisted();
    if (persisted === true && _toggleEl && _toggleEl.style.display !== 'none') {
        _setRevealed(true, { persist: false });
    } else {
        _applyRevealState();
    }

    return {
        get revealed() { return _revealed; },
        setRevealed(next) { _setRevealed(next); return _revealed; },
        toggleReveal(next) {
            _setRevealed(next === undefined ? !_revealed : !!next);
            return _revealed;
        },
        syncToggle: _syncToggle,
        destroy() {
            _detachToggle();
            if (_navObserver) { _navObserver.disconnect(); _navObserver = null; }
            // Clear the reveal scope class so host chrome gated on it doesn't
            // stay revealed after teardown. onReveal is NOT fired here — destroy
            // is teardown, not a state change (a remount re-applies).
            if (_scope && _scope.classList) _scope.classList.remove('panels-revealed');
            if (_toggleEl) {
                if (_toggleOwned) {
                    _toggleEl.remove();
                } else {
                    // Host-provided anchor: restore its prior aria-pressed state
                    // and undo any display:none applied while gated down.
                    if (_togglePrevAria === null) {
                        _toggleEl.removeAttribute('aria-pressed');
                    } else {
                        _toggleEl.setAttribute('aria-pressed', _togglePrevAria);
                    }
                    _toggleEl.style.display = '';
                }
            }
        },
    };
}

export const Panels = {
    registerPanel,
    unregisterPanel,
    renderNav,
    syncNav,
    activate,
    panels,
    initReveal,
    _reset,
};

export default Panels;
