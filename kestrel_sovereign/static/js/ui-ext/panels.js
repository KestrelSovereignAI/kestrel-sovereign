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
        if (!_gateOk(def)) {
            if (tab) tab.remove();
            // Drop a registry-created panel body when the panel gates off at
            // runtime (feature disabled), and forget its rendered state so a
            // re-enable re-renders fresh. In-place core panels are left alone.
            const panel = document.getElementById(`panel-${def.panelId}`);
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
                if (panel.dataset.registryOwned === 'true') panel.remove();
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

export const Panels = {
    registerPanel,
    unregisterPanel,
    renderNav,
    syncNav,
    activate,
    panels,
    _reset,
};

export default Panels;
