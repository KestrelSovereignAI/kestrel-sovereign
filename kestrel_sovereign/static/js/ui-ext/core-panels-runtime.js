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
import { state } from '../ui.js';
import API from '../api.js';

import { loadIdentity } from '../identity.js';
import { loadConstitution, loadMemories, initMemoryFilter } from '../memories.js';
import { loadExports, initSovereigntyButtons } from '../sovereignty.js';
import { initTasks, loadTasks } from '../tasks.js';
import { loadResources } from '../resources.js';
import { initMetrics, loadMetrics } from '../metrics.js';
import { initFeatureStore, loadFeatureStore } from '../feature-store.js';
import { initApprovals, loadApprovals } from '../approvals.js';
import { Security } from '../security.js';

let _wired = false;
const _inited = new Set();

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

/**
 * Wire core-panel init + data loading to `panel:shown`. Idempotent — subsequent
 * calls are no-ops so a second `mountPanels` does not double-subscribe.
 *
 * @param {{api?: object}} [opts]
 */
export function wireCorePanelRuntime({ api = API } = {}) {
    if (_wired) return;
    _wired = true;

    bus.on('panel:shown', (payload) => {
        const panelId = payload && payload.panelId;
        if (!panelId) return;
        switch (panelId) {
            case 'identity':
                // Load once — identity has no per-activation refetch in the
                // standalone console either.
                if (!state.identity) _safe(loadIdentity);
                break;
            case 'constitution':
                if (!state.constitution) _safe(loadConstitution);
                break;
            case 'memories':
                _initOnce('memories', initMemoryFilter);
                if (!state.memories) _safe(loadMemories);
                break;
            case 'tasks':
                _initOnce('tasks', initTasks);
                _safe(loadTasks);
                break;
            case 'sovereignty':
                _initOnce('sovereignty', initSovereigntyButtons);
                if (!state.exports) _safe(loadExports);
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
    });
}

export default wireCorePanelRuntime;
