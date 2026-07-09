/**
 * ui_state.mjs — the ONE shared UI view-state persistence surface (#2298).
 *
 * Before this module the console shipped FOUR independent ad-hoc localStorage
 * implementations (theme.js, conversations.js, agent_list.js, ui-ext/panels.js)
 * plus one-off stashes in identity.js — each re-deriving the same
 * "best-effort, degrade-to-no-op when storage is unavailable" try/catch dance.
 * This module owns that logic once so persistence behaves identically across
 * embeds, sandboxed iframes, and jsdom (where `localStorage` may be absent or
 * throw on access).
 *
 * Two layers:
 *
 *   - Raw string helpers — `storeGet` / `storeSet` / `storeRemove`. These are
 *     the exact best-effort `storeGet`/`storeSet` promoted out of
 *     conversations.js / agent_list.js verbatim (no value transform), so the
 *     existing consumers keep their on-disk format byte-for-byte and no user's
 *     stored state migrates or breaks.
 *
 *   - JSON view-state API — `uiStateGet(key, fallback)` / `uiStateSet(key,
 *     value)` / `uiStateRemove(key)`. JSON-serialized values for NEW keys
 *     (namespaced under the `kestrel:ui:` prefix), so callers can persist
 *     booleans / numbers / strings / objects without hand-rolling
 *     serialization.
 *
 * Everything is guarded: a disabled/throwing localStorage degrades to a no-op
 * (writes silently drop, reads return the fallback) rather than throwing.
 */

/** Prefix for NEW ui-state keys introduced by this module (#2298). */
export const UI_STATE_PREFIX = 'kestrel:ui:';

/**
 * Resolve a usable localStorage, or null. Access itself can throw under strict
 * sandboxing (cross-origin iframes, cookies-blocked), so even reading the
 * global is guarded.
 */
function storage() {
    try {
        if (typeof localStorage !== 'undefined' && localStorage) return localStorage;
    } catch (_) { /* access can throw under strict sandboxing */ }
    try {
        if (typeof globalThis !== 'undefined' && globalThis.localStorage) return globalThis.localStorage;
    } catch (_) { /* ditto */ }
    return null;
}

// ---- Raw string helpers (promoted verbatim from the pane modules) ----------

/** Read a raw string value, or `null` when absent/unavailable. */
export function storeGet(key) {
    const s = storage();
    if (!s) return null;
    try { return s.getItem(key); } catch (_) { return null; }
}

/** Write a raw string value; a no-op when storage is unavailable/full. */
export function storeSet(key, value) {
    const s = storage();
    if (!s) return;
    try { s.setItem(key, value); } catch (_) { /* quota / disabled — ignore */ }
}

/** Remove a key; a no-op when storage is unavailable. */
export function storeRemove(key) {
    const s = storage();
    if (!s) return;
    try { s.removeItem(key); } catch (_) { /* disabled — ignore */ }
}

// ---- JSON view-state API (for new keys) ------------------------------------

/**
 * Read a JSON-serialized value, returning `fallback` when the key is absent,
 * storage is unavailable, or the stored value is not valid JSON (e.g. a stale
 * raw value from before this module) — never throws.
 */
export function uiStateGet(key, fallback = null) {
    const raw = storeGet(key);
    if (raw === null || raw === undefined) return fallback;
    try { return JSON.parse(raw); } catch (_) { return fallback; }
}

/**
 * JSON-serialize and store a value. Returns true on success, false when
 * storage is unavailable or the write throws (quota/disabled).
 */
export function uiStateSet(key, value) {
    const s = storage();
    if (!s) return false;
    try { s.setItem(key, JSON.stringify(value)); return true; }
    catch (_) { return false; }
}

/** Remove a ui-state key; a no-op when storage is unavailable. */
export function uiStateRemove(key) {
    storeRemove(key);
}

// Also expose on a global so non-module (plain-script) consumers like theme.js
// — loaded before the module graph runs but only *calling* these helpers at
// DOMContentLoaded and later — can delegate to the same one implementation.
try {
    if (typeof globalThis !== 'undefined') {
        globalThis.KestrelUIState = {
            UI_STATE_PREFIX,
            storeGet,
            storeSet,
            storeRemove,
            uiStateGet,
            uiStateSet,
            uiStateRemove,
        };
    }
} catch (_) { /* environments without a writable global — module exports still work */ }
