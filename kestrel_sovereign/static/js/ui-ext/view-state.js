// ============================================================================
// Panel view-state persistence (#2802)
// ============================================================================
//
// The panel registry already persists a panel's *reveal/active* state
// (`panels.js` `initReveal` + the host's `handle.activate(savedPanelId)`), but a
// panel had no sanctioned way to persist its own *view* state — sub-tab,
// zoom/pan, scroll offset, drill/selection — across a body remount or a full
// page reload. Features that wanted it hand-rolled raw `localStorage`, which is
// precisely the bespoke-per-feature persistence #2298 retired.
//
// This module owns that one mechanism. A panel contribution declares an
// OPTIONAL provider on `registerPanel`:
//
//   registerPanel({
//     panelId: 'observability',
//     render: (body, ctx) => { ... },
//     viewState: {
//       key: 'timeline',                       // optional sub-namespace
//       getState: () => ({ tab, zoom, pan }),  // called on deactivate/unload
//       setState: (s) => applyView(s),         // called on the panel's first render
//     },
//   })
//
// Storage goes through `ui_state.mjs` ONLY (`uiStateGet`/`uiStateSet`, prefix
// `kestrel:ui:`) — there is no second raw-localStorage path. The framework, not
// the feature, composes the key:
//
//   kestrel:ui:panel:<panelId>:<providerKey>
//
// so two panels cannot collide however they name their provider, and
// `providerKey` is a per-panel sub-namespace (a panel that wants several
// independent slices, or genuinely per-agent state, folds that distinction into
// its own `key`). Keys are deliberately NOT agent-scoped: the panels consuming
// this are fleet-wide views whose zoom/sub-tab should survive an agent switch.
//
// Everything degrades: a missing key, a corrupt value, an unavailable
// `localStorage`, or a throwing provider all leave the panel on its own default
// and never throw into the activation path.
// ============================================================================

import { UI_STATE_PREFIX, uiStateGet, uiStateSet, uiStateRemove } from '../ui_state.mjs';

/** Sub-namespace used when a provider declares no `key` of its own. */
export const DEFAULT_VIEW_STATE_KEY = 'view';

// Distinguishes "absent or corrupt" from a legitimately stored `null`, so a
// panel that persists `null` still gets `setState(null)` while a missing/garbage
// value leaves it on its own default.
const MISSING = Symbol('view-state:missing');

/**
 * Compose the framework-owned storage key for a panel's view state. Collisions
 * are structurally impossible: the `panelId` segment is the registry's own
 * unique id.
 *
 * @param {string} panelId
 * @param {string} [providerKey]
 * @returns {string}
 */
export function viewStateKey(panelId, providerKey = DEFAULT_VIEW_STATE_KEY) {
    const sub = (typeof providerKey === 'string' && providerKey) ? providerKey : DEFAULT_VIEW_STATE_KEY;
    return `${UI_STATE_PREFIX}panel:${panelId}:${sub}`;
}

/**
 * Validate a contribution's optional `viewState` provider. A contribution that
 * declares none — or declares a malformed one — yields `null`, and the caller
 * behaves exactly as it did before this hook existed.
 *
 * @param {object} [def] - a panel contribution (`registerPanel` argument).
 * @returns {{key: string, getState: () => any, setState: (state: any) => void} | null}
 */
export function normalizeProvider(def) {
    const vs = def && def.viewState;
    if (!vs || typeof vs !== 'object') return null;
    if (typeof vs.getState !== 'function' || typeof vs.setState !== 'function') {
        console.error(
            '[ui-ext view-state] `viewState` needs both getState() and setState(state); ignoring provider for',
            def && def.panelId,
        );
        return null;
    }
    const key = (typeof vs.key === 'string' && vs.key) ? vs.key : DEFAULT_VIEW_STATE_KEY;
    return { key, getState: vs.getState.bind(vs), setState: vs.setState.bind(vs) };
}

/**
 * Snapshot a provider's state and persist it. A provider that throws, or that
 * returns `undefined` (nothing to persist yet), writes nothing — so a bad
 * snapshot never clobbers a good stored value.
 *
 * @param {string} panelId
 * @param {{key: string, getState: () => any}} provider
 * @returns {boolean} true when a value was written.
 */
export function saveViewState(panelId, provider) {
    if (!panelId || !provider) return false;
    let state;
    try {
        state = provider.getState();
    } catch (err) {
        console.error(`[ui-ext view-state] getState() threw for panel "${panelId}":`, err);
        return false;
    }
    if (state === undefined) return false;
    return uiStateSet(viewStateKey(panelId, provider.key), state);
}

/**
 * Restore a provider's persisted state. Missing storage, a corrupt value, or a
 * disabled `localStorage` all leave the panel on its own default: `setState` is
 * simply never called. A throwing `setState` is isolated, never propagated.
 *
 * @param {string} panelId
 * @param {{key: string, setState: (state: any) => void}} provider
 * @returns {boolean} true when `setState` was invoked with a stored value.
 */
export function restoreViewState(panelId, provider) {
    if (!panelId || !provider) return false;
    const stored = uiStateGet(viewStateKey(panelId, provider.key), MISSING);
    if (stored === MISSING) return false;
    try {
        provider.setState(stored);
    } catch (err) {
        console.error(`[ui-ext view-state] setState() threw for panel "${panelId}":`, err);
        return false;
    }
    return true;
}

/**
 * Forget a panel's persisted view state (a panel resetting its own view).
 *
 * @param {string} panelId
 * @param {{key?: string}} [provider]
 */
export function clearViewState(panelId, provider) {
    if (!panelId) return;
    uiStateRemove(viewStateKey(panelId, provider && provider.key));
}
