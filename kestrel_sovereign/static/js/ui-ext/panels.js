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
//     icon,               // icon class for the nav tab; omitted falls back to
//                         // DEFAULT_TAB_ICON so no contribution renders bare
//     before,             // optional: insert the tab before this panelId's tab
//     gate?(ctx),         // capability gate; an ungated-false panel shows no tab
//     render?(bodyEl, ctx),// lazily invoked ONCE on first activation
//     viewState?          // optional {key?, getState(), setState(s)} provider
//                         // persisting the panel's own view state (#2802)
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
import { normalizeProvider, saveViewState, restoreViewState } from './view-state.js';

/** @type {Map<string, object>} */
const _panels = new Map();
/** @type {Set<string>} contributed panels whose body `render` has already run. */
const _rendered = new Set();

let _navEl = null;
let _hostEl = null;
let _ctx = null;
/** Panel id most recently passed to `activate` — the view-state snapshot source. */
let _activePanelId = null;

function _safe(fn, ...args) {
    try {
        return fn(...args);
    } catch (err) {
        console.error('[ui-ext panels] contribution callback threw:', err);
        return undefined;
    }
}

// `_escapeAttr` retired with the innerHTML tab builder (#2822): `_reconcileTab`
// composes tabs through DOM APIs (className / textContent / setAttribute), so a
// descriptor's label or icon can no longer reach an HTML parser at all.

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

// ---- Panel view-state provider lifecycle (#2802) ---------------------------
//
// Strictly additive: everything below is a no-op for a panel that declares no
// `viewState` provider, and no existing event/subscriber observes any change.
// The save side runs at the three moments the panel's live DOM is about to stop
// being the source of truth — deactivation (`activate` switching away), a
// re-gate that drops the body, and page unload — and the restore side runs once,
// on the panel's first (re)render, which is exactly the remount boundary.

let _unloadHookInstalled = false;

/** The validated provider for a registered panel, or null. */
function _providerFor(panelId) {
    return normalizeProvider(_panels.get(panelId));
}

/**
 * Snapshot a panel's view state into `ui_state.mjs`. Guarded on `_rendered` so a
 * panel whose body never rendered cannot overwrite a good stored value with its
 * uninitialized default.
 */
function _saveView(panelId) {
    if (!panelId || !_rendered.has(panelId)) return false;
    const provider = _providerFor(panelId);
    if (!provider) return false;
    return saveViewState(panelId, provider);
}

/** Reapply a panel's persisted view state after its body (re)rendered. */
function _restoreView(panelId) {
    const provider = _providerFor(panelId);
    if (!provider) return false;
    return restoreViewState(panelId, provider);
}

// `activate()` does not run on unload, so a full page reload would otherwise
// lose whatever the active panel accumulated since the last tab switch. Wired
// lazily — only once some panel actually registers a provider — so a console
// with no view-state consumer attaches no listeners at all. `pagehide` is the
// bfcache-safe spelling of unload; `visibilitychange`->hidden covers Safari/iOS,
// which may background a tab without ever firing `pagehide`.
function _ensureUnloadHook() {
    if (_unloadHookInstalled) return;
    try {
        if (typeof window === 'undefined' || !window || typeof window.addEventListener !== 'function') return;
        const flush = () => { _saveView(_activePanelId); };
        window.addEventListener('pagehide', flush);
        window.addEventListener('visibilitychange', () => {
            if (typeof document !== 'undefined' && document.visibilityState === 'hidden') flush();
        });
        _unloadHookInstalled = true;
    } catch (err) {
        // A host without a usable window (SSR, worker) simply gets no unload
        // flush; the deactivation + re-gate snapshots still work.
        console.error('[ui-ext panels] could not wire the view-state unload flush:', err);
    }
}

// ---- Nav tab chrome (#2821, #2822) -----------------------------------------
//
// The descriptor is the single source of truth for a tab's icon and label. Both
// tab paths — a registry-BUILT tab and an in-place tab declared in `index.html`
// that the registry ADOPTS — run through `_reconcileTab`, so one edit to a
// descriptor reaches the standalone console and every embed alike. Before this,
// adoption only stamped the gating flag and never looked at the descriptor's
// chrome, so the two surfaces silently diverged (the reason the `ki-check` typo
// had to be fixed in two files).

/**
 * Icon for a contribution that declares none. Without a default, an
 * out-of-tree panel (Observability today; any future feature panel) renders as
 * bare text in an otherwise iconed strip — the ragged nav #2821 set out to fix,
 * reintroduced from outside the repo. `puzzle` already reads as "extension"
 * here (it is the Features tab's own icon).
 *
 * An embedder contributing `hostTabs` should declare a meaningful `icon`;
 * this is the floor, not a recommendation.
 */
const DEFAULT_TAB_ICON = 'ki ki-puzzle';

function _iconFor(def) {
    return (def && def.icon) || DEFAULT_TAB_ICON;
}

/** Direct text-node content of an element, ignoring child elements (badges). */
function _ownText(el) {
    return Array.from(el.childNodes)
        .filter((n) => n.nodeType === 3)
        .map((n) => n.textContent)
        .join('')
        .trim();
}

/**
 * Bring a tab's children in line with its descriptor: an icon span, a label
 * span, then whatever else the tab already carried. Idempotent — it is the
 * shared body of both the build and the adopt path.
 *
 * Two things it must not break:
 *
 *   - Badge elements (`#tasks-badge`, `#security-pending-badge`,
 *     `#approvals-pending-badge`) live inside the button and are re-queried by
 *     id from tasks.js / security.js / approvals.js. They are preserved BY
 *     NODE, not re-created, so a count rendered before the first nav sync
 *     survives.
 *   - An already-hydrated (translated) label. The descriptor's `label` is the
 *     English fallback; when a tab already carries label text, that text wins.
 *     Only a tab with no label at all falls back to the descriptor.
 */
function _reconcileTab(tab, def) {
    const labelKey = def.labelKey || '';

    // `span.ki` / `span[data-label-key]` match the pre-#2821 in-place markup
    // shape; the `nav-tab-*` classes match anything this function has already
    // produced. A badge span has neither, so it is never mistaken for a label.
    let icon = tab.querySelector('.nav-tab-icon, span.ki');
    let label = tab.querySelector('.nav-tab-label, span[data-label-key]');
    const preserved = Array.from(tab.children).filter((el) => el !== icon && el !== label);

    if (!icon) icon = document.createElement('span');
    // `nav-tab-icon` / `nav-tab-label` are the stable hooks the display-mode
    // preference (#2823) toggles visibility on. They are structural, not
    // cosmetic — the icon class itself is what carries the glyph.
    icon.className = `nav-tab-icon ${_iconFor(def)}`;
    // Decorative: the label span alongside it is the button's accessible name.
    icon.setAttribute('aria-hidden', 'true');

    if (!label) {
        label = document.createElement('span');
        // The plainest in-place shape is `<button data-label-key="tab_x">X</button>`
        // — text directly on the button. Adopt that text so a locale already
        // hydrated onto the button is not reverted to the English descriptor.
        label.textContent = _ownText(tab) || def.label || def.panelId;
    }
    label.classList.add('nav-tab-label');
    if (labelKey) label.setAttribute('data-label-key', labelKey);

    // A `data-label-key` on the BUTTON is load-bearing damage once a tab has
    // children: KestrelTheme._hydrate() assigns `el.textContent` for every
    // `[data-label-key]`, which on a button replaces its whole subtree — icon
    // and badge included. The key belongs on the label span alone.
    tab.removeAttribute('data-label-key');

    tab.replaceChildren(icon, document.createTextNode(' '), label, ...preserved);
}

function _buildTab(def) {
    const btn = document.createElement('button');
    btn.className = 'nav-tab';
    btn.dataset.panel = def.panelId;
    // Mark registry-owned tabs so core's PANEL_CAPABILITIES re-gate
    // (reconcileNavigationCapabilities) leaves them to the registry's own gate.
    btn.dataset.panelRegistry = 'true';
    _reconcileTab(btn, def);
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
            // Snapshot the panel's view state BEFORE its body is dropped (#2802):
            // a re-gate is a remount, and re-enabling re-renders from scratch.
            _saveView(def.panelId);
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
            // ...and bring its chrome in line with the descriptor (#2822), so
            // adoption covers icon/label the way it already covers gating. A
            // descriptor-only icon change previously reached embeds and not the
            // standalone console, with nothing failing to say so.
            _reconcileTab(tab, def);
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
 * A contribution MAY declare an optional `viewState` provider
 * ({@link import('./contract.js').PanelViewStateProvider}) — `{key?, getState(),
 * setState(state)}` — and the registry persists its state through
 * `ui_state.mjs`, restoring it on the panel's next render (#2802). Omitting it
 * leaves the panel behaving exactly as before.
 *
 * @param {object} def
 */
export function registerPanel(def) {
    if (!def || typeof def.panelId !== 'string') {
        console.error('[ui-ext panels] registerPanel: contribution needs a string `panelId`');
        return;
    }
    // Re-registration is the remount boundary (mountPanels re-registers every
    // core panel on each mount), so snapshot the outgoing contribution's view
    // state before `_rendered` is cleared and the body re-renders from scratch.
    _saveView(def.panelId);
    _panels.set(def.panelId, def);
    _rendered.delete(def.panelId);
    if (def.viewState) _ensureUnloadHook();
    if (_navEl) _syncNav();
}

/**
 * Remove a panel contribution by id, dropping its tab (and a registry-created
 * panel container).
 * @param {string} panelId
 */
export function unregisterPanel(panelId) {
    if (!_panels.has(panelId)) return;
    // Unregistration is a teardown boundary (a host destroying its mount), so
    // snapshot before the contribution — and with it its provider — is dropped.
    _saveView(panelId);
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
    // Deactivation snapshot (#2802): the outgoing panel's live DOM stops being
    // the source of truth here. This is deliberately NOT `panel:hidden` — that
    // event is a teardown signal (spawn.js/database.js stop polling on it) and
    // must keep firing only on a re-gate, so an ordinary tab switch stays
    // behaviour-identical for every existing subscriber.
    if (_activePanelId && _activePanelId !== panelId) _saveView(_activePanelId);
    _activePanelId = panelId;
    const def = _panels.get(panelId);
    const root = document.getElementById(`panel-${panelId}`);
    if (def && !_rendered.has(panelId)) {
        _rendered.add(panelId);
        const body = (root && root.querySelector('.panel-content')) || root;
        if (typeof def.render === 'function' && body) {
            _safe(def.render, body, merged);
        }
        // First render of this mount == the remount boundary, so reapply the
        // persisted view state now that the panel's DOM exists. Subsequent
        // activations skip this: the live body already holds the state.
        _restoreView(panelId);
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

/**
 * Persist a panel's view state right now (#2802). The registry already
 * snapshots on deactivation, re-gate, re-registration, and page unload; this is
 * the explicit escape hatch for a host that tears the whole mount down by
 * another route (e.g. `mountPanels().destroy()` on agent switch).
 *
 * @param {string} [panelId] - defaults to the currently active panel.
 * @returns {boolean} true when a value was written.
 */
export function flushViewState(panelId) {
    return _saveView(panelId || _activePanelId);
}

/** Test/teardown affordance: forget all panels and unbind the nav/host. */
export function _reset() {
    _panels.clear();
    _rendered.clear();
    _navEl = null;
    _hostEl = null;
    _ctx = null;
    _activePanelId = null;
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
    flushViewState,
    initReveal,
    _reset,
};

export default Panels;
