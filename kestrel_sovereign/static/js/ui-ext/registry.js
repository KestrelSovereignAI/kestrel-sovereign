// ============================================================================
// UI extension slot registry (ticket 02, epic #2038)
// ============================================================================
//
// The runtime that contributions register into and that core renders zones
// through. Implements the contract documented in `contract.js` /
// `docs/proposals/ui-extension-slots/SLOTS.md`.
//
// This registry SUPERSEDES the abandoned `registerHeaderAction`
// (`chat.js`). The four reasons that API failed its first feature consumer are
// this module's hard requirements:
//
//   1. Many named zones with per-zone anchor semantics (not header-only). The
//      anchor element is supplied by core via `renderSlot(slot, { element })`;
//      the registry appends one container per contribution into it, in `order`.
//   2. Per-contribution teardown + re-render. Updating contribution A re-gates
//      and re-renders ONLY A and calls ONLY A's teardown — sibling B's DOM,
//      listeners, and closures are untouched. (`registerHeaderAction` did
//      `slot.innerHTML = ''` on every call, destroying live state.)
//   3. Per-context state: each zone instance carries its own `ctx` (e.g.
//      `agentName`), retained so an event-driven refresh re-renders without the
//      caller re-supplying it.
//   4. Stable identity across re-gates: a contribution's container/closures
//      survive re-renders; teardown fires only on actual removal/unmount.
//
// This module is generic — it contains NO feature-name strings. Zone ids and
// event names are passed through verbatim.
// ============================================================================

import bus from './bus.js';

// Registered contributions, keyed by slot id, preserving registration order
// (the stable tie-breaker for equal `order`).
/** @type {Map<string, Array<import('./contract.js').UIContribution>>} */
const _contribs = new Map();

// Live mounted zone instances, keyed by slot id. A single slot can have many
// instances (e.g. `agent-card-actions`, one per card). Each instance is keyed
// by its anchor element.
/**
 * @typedef {Object} InstanceRecord
 * @property {HTMLElement} anchor   - the zone anchor (ctx.element).
 * @property {object} ctx           - last ctx seen for this instance.
 * @property {Map<import('./contract.js').UIContribution, {container: HTMLElement, teardown: (() => void) | null}>} mounted
 */
/** @type {Map<string, InstanceRecord[]>} */
const _instances = new Map();

// Bus events we've wired a forwarding handler for (one per distinct event name).
/** @type {Set<string>} */
const _busWired = new Set();

function _safe(fn, ...args) {
    try {
        return fn(...args);
    } catch (err) {
        console.error('[ui-ext registry] contribution callback threw:', err);
        return undefined;
    }
}

function _sorted(slot) {
    const list = _contribs.get(slot) || [];
    // Decorate-sort-undecorate keeps the sort stable on `order` ties by falling
    // back to registration index.
    return list
        .map((c, i) => ({ c, i }))
        .sort((a, b) => {
            const oa = typeof a.c.order === 'number' ? a.c.order : 100;
            const ob = typeof b.c.order === 'number' ? b.c.order : 100;
            return oa - ob || a.i - b.i;
        })
        .map((x) => x.c);
}

function _instancesFor(slot) {
    let arr = _instances.get(slot);
    if (!arr) {
        arr = [];
        _instances.set(slot, arr);
    }
    return arr;
}

// Drop instance records whose anchor has left the DOM, firing teardowns so a
// re-rendered zone (e.g. the agents list rebuilt via innerHTML) doesn't leak
// listeners/closures. Only prunes when the host actually reports detachment —
// environments without `isConnected` (some test mocks) never false-positive.
function _pruneDetached(slot) {
    const arr = _instances.get(slot);
    if (!arr) return;
    for (let i = arr.length - 1; i >= 0; i--) {
        const inst = arr[i];
        if (inst.anchor && inst.anchor.isConnected === false) {
            _teardownInstance(inst);
            arr.splice(i, 1);
        }
    }
}

function _teardownInstance(inst) {
    for (const rec of inst.mounted.values()) {
        if (rec.teardown) _safe(rec.teardown);
        _removeContainer(rec.container);
    }
    inst.mounted.clear();
}

function _removeContainer(container) {
    if (!container) return;
    if (typeof container.remove === 'function') {
        container.remove();
    } else if (container.parentNode && typeof container.parentNode.removeChild === 'function') {
        container.parentNode.removeChild(container);
    }
}

function _getInstance(slot, anchor) {
    const arr = _instancesFor(slot);
    let inst = arr.find((r) => r.anchor === anchor);
    if (!inst) {
        inst = { anchor, ctx: null, mounted: new Map() };
        arr.push(inst);
    }
    return inst;
}

// Mount, re-render, or unmount a single contribution within one instance,
// honoring its gate. Teardown fires before every re-render and before unmount.
function _applyContribution(slot, inst, contribution) {
    // Item-provider contributions (menu slots) have no DOM to mount; they are
    // pulled on demand by `collectItems`, never mounted into a zone anchor.
    if (typeof contribution.render !== 'function') return;
    const ctx = inst.ctx || {};
    const rec = inst.mounted.get(contribution);
    const gated = contribution.gate ? !!_safe(contribution.gate, ctx) : true;

    if (!gated) {
        if (rec) {
            if (rec.teardown) _safe(rec.teardown);
            _removeContainer(rec.container);
            inst.mounted.delete(contribution);
        }
        return;
    }

    let container;
    if (rec) {
        // Re-render in place: tear down prior render, clear, reuse container so
        // the contribution's position (and siblings) are untouched.
        if (rec.teardown) _safe(rec.teardown);
        rec.teardown = null;
        container = rec.container;
        container.innerHTML = '';
    } else {
        container = document.createElement('div');
        container.dataset.uiExtSlot = slot;
        if (contribution.id) container.dataset.uiExtId = contribution.id;
        inst.anchor.appendChild(container);
        inst.mounted.set(contribution, { container, teardown: null });
    }

    const ret = _safe(contribution.render, container, ctx);
    inst.mounted.get(contribution).teardown = typeof ret === 'function' ? ret : null;
}

// Reorder this instance's registry-owned containers to match `_sorted(slot)`.
// `_applyContribution` always appends a newly mounted container to the end of
// the anchor, so a contribution registered AFTER its lower-`order` siblings were
// already mounted would otherwise sit out of order. This moves the existing
// container nodes (identity/listeners/closures preserved — they are moved, never
// recreated) and leaves any non-registry siblings in the anchor untouched.
function _reorderInstance(slot, inst) {
    const containers = [];
    for (const c of _sorted(slot)) {
        const rec = inst.mounted.get(c);
        if (rec && rec.container) containers.push(rec.container);
    }
    for (let k = 1; k < containers.length; k++) {
        const cur = containers[k];
        const prev = containers[k - 1];
        const parent = cur.parentNode;
        if (!parent || prev.parentNode !== parent) continue;
        if (typeof parent.insertBefore !== 'function') continue;
        if (prev.nextSibling !== cur) parent.insertBefore(cur, prev.nextSibling);
    }
}

// React to a bus event: re-gate + re-render only the contributions that opted
// into it, across every live instance — siblings are left alone.
function _onBusEvent(event) {
    for (const [slot, list] of _contribs.entries()) {
        const subscribed = list.filter((c) => Array.isArray(c.events) && c.events.includes(event));
        if (subscribed.length === 0) continue;
        _pruneDetached(slot);
        for (const inst of _instancesFor(slot)) {
            for (const c of subscribed) {
                // Only touch contributions already mounted in this instance, or
                // newly gated-in ones — `_applyContribution` handles both.
                _applyContribution(slot, inst, c);
            }
        }
    }
}

function _wireBus(events) {
    if (!Array.isArray(events)) return;
    for (const ev of events) {
        if (typeof ev !== 'string' || _busWired.has(ev)) continue;
        _busWired.add(ev);
        bus.on(ev, () => _onBusEvent(ev));
    }
}

export const UI = {
    /**
     * Register a contribution into a zone. Re-registering the same `id`
     * replaces the prior contribution (dedupe). Returns nothing; mounting
     * happens when core next calls `renderSlot`/`refreshSlot` for the zone.
     *
     * @param {import('./contract.js').UIContribution} contribution
     */
    register(contribution) {
        if (!contribution || typeof contribution.slot !== 'string') {
            console.error('[ui-ext registry] register: contribution needs a string `slot`');
            return;
        }
        // A contribution is either DOM-shaped (`render`) or item-provider-shaped
        // (`items`, for menu slots like `chat-message-actions`). Exactly one is
        // required; item-provider contributions are never mounted by renderSlot,
        // they are pulled by `collectItems`.
        if (typeof contribution.render !== 'function'
            && typeof contribution.items !== 'function') {
            console.error('[ui-ext registry] register: contribution needs a `render` or `items` function');
            return;
        }
        const list = _contribs.get(contribution.slot) || [];
        let previous = null;
        if (contribution.id) {
            const i = list.findIndex((c) => c.id === contribution.id);
            if (i >= 0) {
                previous = list[i];
                list[i] = contribution;
            } else {
                list.push(contribution);
            }
        } else {
            list.push(contribution);
        }
        _contribs.set(contribution.slot, list);
        _wireBus(contribution.events);

        // Mount into any already-rendered instances of this zone so a late
        // registration appears without waiting for a second core renderSlot.
        // `modal-root` in particular has no retrigger event, so a contribution
        // registered after boot would otherwise never mount at all.
        _pruneDetached(contribution.slot);
        for (const inst of _instancesFor(contribution.slot)) {
            if (!inst.ctx) continue; // anchor known only after a first renderSlot
            if (previous && previous !== contribution) {
                const rec = inst.mounted.get(previous);
                if (rec) {
                    if (rec.teardown) _safe(rec.teardown);
                    _removeContainer(rec.container);
                    inst.mounted.delete(previous);
                }
            }
            _applyContribution(contribution.slot, inst, contribution);
            _reorderInstance(contribution.slot, inst);
        }
    },

    /**
     * Tear down and forget a single contribution by id, across every live
     * instance of the zone. Siblings untouched.
     *
     * @param {string} slot
     * @param {string} id
     */
    unregister(slot, id) {
        const list = _contribs.get(slot);
        if (!list) return;
        const contribution = list.find((c) => c.id === id);
        _contribs.set(slot, list.filter((c) => c.id !== id));
        if (!contribution) return;
        for (const inst of _instancesFor(slot)) {
            const rec = inst.mounted.get(contribution);
            if (rec) {
                if (rec.teardown) _safe(rec.teardown);
                _removeContainer(rec.container);
                inst.mounted.delete(contribution);
            }
        }
    },

    /**
     * First-pass render of a zone instance: sort by `order`, gate, and mount
     * every contribution into its own container under `ctx.element`. Idempotent
     * — calling again reuses containers and re-renders in place. The instance's
     * `ctx` is retained for later event-driven refreshes.
     *
     * @param {string} slot
     * @param {object} ctx  - must carry `element` (the zone anchor).
     */
    renderSlot(slot, ctx) {
        const anchor = ctx && ctx.element;
        if (!anchor) {
            console.error(`[ui-ext registry] renderSlot("${slot}"): ctx.element (anchor) is required`);
            return;
        }
        _pruneDetached(slot);
        const inst = _getInstance(slot, anchor);
        inst.ctx = ctx;
        const sorted = _sorted(slot);
        for (const c of sorted) {
            _applyContribution(slot, inst, c);
        }
        // Unmount contributions that are mounted here but no longer registered.
        for (const c of [...inst.mounted.keys()]) {
            if (!sorted.includes(c)) {
                const rec = inst.mounted.get(c);
                if (rec.teardown) _safe(rec.teardown);
                _removeContainer(rec.container);
                inst.mounted.delete(c);
            }
        }
        // Containers carried over from a prior render keep their old DOM position;
        // realign the registry-owned containers to the sorted order.
        _reorderInstance(slot, inst);
    },

    /**
     * Re-gate + re-render ONE contribution across every live instance of the
     * zone, leaving siblings' DOM/listeners/closures intact.
     *
     * @param {string} slot
     * @param {string} id
     */
    refreshContribution(slot, id) {
        const list = _contribs.get(slot);
        if (!list) return;
        const contribution = list.find((c) => c.id === id);
        if (!contribution) return;
        _pruneDetached(slot);
        for (const inst of _instancesFor(slot)) {
            _applyContribution(slot, inst, contribution);
        }
    },

    /**
     * Re-evaluate a whole zone against its last-known ctx, for every live
     * instance. Used sparingly — granular refreshes are the norm.
     *
     * @param {string} slot
     */
    refreshSlot(slot) {
        _pruneDetached(slot);
        for (const inst of _instancesFor(slot)) {
            if (inst.ctx) this.renderSlot(slot, inst.ctx);
        }
    },

    /**
     * Publish a bus event (convenience pass-through so core call sites import
     * one module). Equivalent to `bus.emit`.
     *
     * @param {string} event
     * @param {any} [payload]
     */
    emit(event, payload) {
        bus.emit(event, payload);
    },

    /**
     * The contributions registered for a zone, in registration order. Read-only
     * snapshot (a copy) — primarily for core "is this zone empty?" checks and
     * tests.
     *
     * @param {string} slot
     * @returns {import('./contract.js').UIContribution[]}
     */
    contributions(slot) {
        return [...(_contribs.get(slot) || [])];
    },

    /**
     * Collect menu items from an item-provider slot (e.g. `chat-message-actions`).
     * Unlike `render`-based zones, these contributions supply menu ITEMS, not DOM:
     * each is `{ slot, order?, gate?, items: (ctx) => [{label, danger?, ...}] }`.
     * Contributions are gated by `gate(ctx)`, ordered by `order`, and each item
     * provider is error-isolated — a throwing provider yields no items instead of
     * killing the caller's base items. Returns a flat, ordered array.
     *
     * @param {string} slot
     * @param {object} ctx
     * @returns {Array<object>}
     */
    collectItems(slot, ctx) {
        const out = [];
        for (const c of _sorted(slot)) {
            if (typeof c.items !== 'function') continue;
            if (c.gate && !_safe(c.gate, ctx)) continue;
            const items = _safe(c.items, ctx);
            if (Array.isArray(items)) {
                for (const it of items) if (it) out.push(it);
            }
        }
        return out;
    },

    /** Test/teardown affordance: forget all contributions, instances, and bus wiring. */
    _reset() {
        for (const arr of _instances.values()) {
            for (const inst of arr) _teardownInstance(inst);
        }
        _contribs.clear();
        _instances.clear();
        _busWired.clear();
        bus._reset();
    },
};

export default UI;
