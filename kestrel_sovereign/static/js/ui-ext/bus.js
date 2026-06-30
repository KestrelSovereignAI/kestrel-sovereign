// ============================================================================
// UI extension event bus — minimal pub/sub (ticket 02, epic #2038)
// ============================================================================
//
// A tiny synchronous publish/subscribe channel paired with the slot registry
// (`registry.js`). Core emits the event vocabulary defined in the slot contract
// (`contract.js` / SLOTS.md §4); the registry subscribes so that a contribution
// re-gates + re-renders when one of its declared `events` fires — without each
// feature reinventing bespoke refresh plumbing.
//
// This module is deliberately generic: it knows nothing about any specific
// feature or zone. Event NAMES are passed through verbatim; no name is special.
// ============================================================================

/** @type {Map<string, Set<(payload?: any) => void>>} */
const _handlers = new Map();

/**
 * Subscribe to a named event.
 *
 * @param {string} event   - event name (see {@link import('./contract.js').SlotEvent}).
 * @param {(payload?: any) => void} fn - called synchronously on each `emit`.
 * @returns {() => void} an unsubscribe function (convenience over `off`).
 */
export function on(event, fn) {
    if (typeof event !== 'string' || typeof fn !== 'function') return () => {};
    let set = _handlers.get(event);
    if (!set) {
        set = new Set();
        _handlers.set(event, set);
    }
    set.add(fn);
    return () => off(event, fn);
}

/**
 * Unsubscribe a previously-registered handler.
 *
 * @param {string} event
 * @param {(payload?: any) => void} fn
 */
export function off(event, fn) {
    const set = _handlers.get(event);
    if (!set) return;
    set.delete(fn);
    if (set.size === 0) _handlers.delete(event);
}

/**
 * Publish an event to every subscribed handler. A throwing handler is isolated
 * (logged, skipped) so one bad subscriber cannot starve the others — the same
 * error-isolation posture the registry applies to a contribution's `render`.
 *
 * @param {string} event
 * @param {any} [payload]
 */
export function emit(event, payload) {
    const set = _handlers.get(event);
    if (!set) return;
    // Snapshot so a handler that (un)subscribes during dispatch doesn't mutate
    // the set we're iterating.
    for (const fn of [...set]) {
        try {
            fn(payload);
        } catch (err) {
            console.error(`[ui-ext bus] handler for "${event}" threw:`, err);
        }
    }
}

/** Remove all subscribers. Test/teardown affordance; not used in normal flow. */
export function _reset() {
    _handlers.clear();
}

export const bus = { on, off, emit, _reset };
export default bus;
