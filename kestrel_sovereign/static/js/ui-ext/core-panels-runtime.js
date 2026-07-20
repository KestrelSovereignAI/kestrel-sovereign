// ============================================================================
// Core panel runtime wiring for the embeddable panel host (issue #2145)
// ============================================================================
//
// `mount-panels.js` builds the nav + panel bodies for an embedder same-document,
// but the panel BODIES need their event handlers wired and their data fetched.
// In the standalone console `identity.js`/`app.js` own that (init at boot, load
// on tab activation). An embedder that calls `mountPanels` has no such boot, so
// this module reproduces it: it subscribes each core panel's init + loader to
// the `panel:shown` event the registry emits from `Panels.activate()`.
//
// This is the SAME lazy semantics the standalone console has — init once (the
// first time a panel is shown, right after its body is built), then load on each
// show. It is imported LAZILY and best-effort by `mountPanels` (a heavy import
// failure must not break the mount), and is NOT imported by the standalone boot
// path, so standalone loading stays owned by `identity.js` — there is no
// double-load.
// ============================================================================

import bus from './bus.js';
import API from '../api.js';

import { loadIdentity } from '../identity.js';
import { loadConstitution, loadMemories, initMemoryFilter } from '../memories.js';
import { loadExports, initSovereigntyButtons } from '../sovereignty.js';
import { initDatabaseExplorer } from '../database.js';
import { initTasks, loadTasks } from '../tasks.js';
import { loadResources } from '../resources.js';
import { initMetrics, loadMetrics } from '../metrics.js';
import { initFeatureStore, loadFeatureStore } from '../feature-store.js';
import { initApprovals, loadApprovals } from '../approvals.js';
import { Security } from '../security.js';

// Mount-scoped wiring state (#2145 P2-2). These are reset by
// `resetCorePanelRuntime` (called from `mountPanels().destroy()`), NOT keyed off
// module lifetime — an embedder that destroys then remounts gets fresh DOM, so
// the runtime must re-init the new panels and re-run their first-show loaders.
// The "load once" set is deliberately module-local here rather than the shared
// `state.*` flags the standalone console uses: `state.identity`/`state.exports`
// persist across a remount, which would leave a fresh panel body stuck on its
// "Loading…" placeholder because the guarded load was skipped.
let _wired = false;
let _off = null;
let _offDatabaseExplorer = null;
const _inited = new Set();
const _loaded = new Set();

function _safe(fn) {
    try { if (typeof fn === 'function') fn(); } catch (err) {
        console.error('[core-panels-runtime] handler threw:', err);
    }
}

function _initOnce(panelId, fn) {
    if (_inited.has(panelId)) return;
    _inited.add(panelId);
    _safe(fn);
}

// Run a panel's data loader once per mount (mirrors the standalone console's
// state-guarded "load once" semantics, but scoped to this mount so a remount
// reloads the fresh DOM instead of skipping on a stale cache).
function _loadOnce(panelId, fn) {
    if (_loaded.has(panelId)) return;
    _loaded.add(panelId);
    _safe(fn);
}

/**
 * Wire core-panel init + data loading to `panel:shown`. Idempotent — subsequent
 * calls are no-ops so a second `mountPanels` does not double-subscribe. Pair with
 * `resetCorePanelRuntime` on `destroy()` so a later remount re-wires + reloads.
 *
 * @param {{api?: object, root?: ParentNode}} [opts]
 */
export function wireCorePanelRuntime({ api = API, root = document } = {}) {
    if (_wired) return;
    _wired = true;
    _offDatabaseExplorer = initDatabaseExplorer({ api, root });

    const handler = (payload) => {
        const panelId = payload && payload.panelId;
        if (!panelId) return;
        switch (panelId) {
            case 'identity':
                // Load once per mount — identity has no per-activation refetch in
                // the standalone console either.
                _loadOnce('identity', loadIdentity);
                break;
            case 'constitution':
                _loadOnce('constitution', loadConstitution);
                break;
            case 'memories':
                _initOnce('memories', initMemoryFilter);
                _loadOnce('memories', loadMemories);
                break;
            case 'tasks':
                _initOnce('tasks', initTasks);
                _safe(loadTasks);
                break;
            case 'sovereignty':
                _initOnce('sovereignty', initSovereigntyButtons);
                _loadOnce('sovereignty', loadExports);
                break;
            case 'resources':
                _safe(loadResources);
                break;
            case 'metrics':
                _initOnce('metrics', initMetrics);
                _safe(loadMetrics);
                break;
            case 'features':
                _initOnce('features', initFeatureStore);
                _safe(loadFeatureStore);
                break;
            case 'approvals':
                _initOnce('approvals', initApprovals);
                _safe(loadApprovals);
                break;
            case 'security':
                _initOnce('security', () => Security.init());
                _safe(() => Security.loadPendingApprovals && Security.loadPendingApprovals());
                break;
            default:
                break;
        }
    };
    bus.on('panel:shown', handler);
    _off = () => bus.off('panel:shown', handler);
}

/**
 * Tear down the runtime wiring so a later `mountPanels` re-wires against fresh
 * DOM. Unsubscribes the `panel:shown` handler and clears the per-mount init +
 * load state; without this a remount's fresh panels never re-init (controls
 * unwired) and their loaders are skipped (bodies stuck on "Loading…").
 */
export function resetCorePanelRuntime() {
    if (_off) { _safe(_off); _off = null; }
    if (_offDatabaseExplorer) {
        _safe(_offDatabaseExplorer);
        _offDatabaseExplorer = null;
    }
    _wired = false;
    _inited.clear();
    _loaded.clear();
}

export default wireCorePanelRuntime;
