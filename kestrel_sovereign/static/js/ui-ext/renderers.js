// ============================================================================
// Feature tool-output renderer registry (ticket 06, epic #2038)
// ============================================================================
//
// The SECOND, lower-trust half of the chat-renderer story. The first half —
// `registerPartRenderer` in `chat.js` — is the HOST/embedder contract: the host
// owns the markup AND its sanitization, writing it via `innerHTML`. That is the
// documented, intentional trust model for code the embedder ships (e.g. Frinz's
// selfie `<img>`). It is left exactly as-is.
//
// This module adds the contract for FEATURE renderers, which render tool/model
// PAYLOAD — data that is NOT author-trusted. Their output MUST flow through core
// sanitization, so the contract is enforceable *by construction*:
//
//   registerRenderer({
//     id?,                              // stable id (dedupe / unregister)
//     match: { tool?, contentType? },   // dispatch key (string | RegExp | fn)
//     order?,                           // ascending; first match wins
//     gate?(ctx) => boolean,            // capability gate, like every contribution
//     render(payload, ctx) => string,   // INERT HTML STRING ONLY — never a Node
//     mount?(rootEl, ctx) => (void | () => void),  // imperative behavior, post-sanitize
//   })
//
// `render` returns a plain HTML STRING and nothing else. Core (chat.js) parses
// and sanitizes that string through the SAME DOMPurify path as core markdown,
// then builds the live DOM. The renderer is never handed the live target
// element in `render`, so there is no code path by which it can write
// unsanitized markup. A renderer that returns a `Node`/`DocumentFragment` (which
// could carry `addEventListener` listeners DOMPurify cannot strip) is REJECTED.
//
// Imperative behavior (canvas, chart, click handlers) happens exclusively in the
// optional `mount(rootEl, ctx)` hook, which core invokes AFTER it has
// parsed+sanitized the markup and created `rootEl`. `mount` operates only on
// `rootEl` and its descendants and may return a teardown fn.
//
// This module is generic — no feature-name strings. It holds NO DOM and does NOT
// import the markdown module; core supplies the `sanitize` function so the
// registry stays pure and unit-testable.
// ============================================================================

/**
 * @typedef {Object} RendererMatch
 * @property {string|RegExp|((name: string, ctx: object) => boolean)} [tool]
 *           - match against a tool card's name.
 * @property {string|RegExp|((type: string, ctx: object) => boolean)} [contentType]
 *           - match against a message-part content type.
 */

/**
 * @typedef {Object} FeatureRenderer
 * @property {string}        [id]
 * @property {RendererMatch} match
 * @property {number}        [order=100]
 * @property {(ctx: object) => boolean} [gate]
 * @property {(payload: any, ctx: object) => string} render
 * @property {(rootEl: HTMLElement, ctx: object) => (void | (() => void))} [mount]
 */

/** @type {FeatureRenderer[]} */
const _renderers = [];

// Monotonic counter for internal renderer ids. Every registered renderer gets a
// stable `_uid` (emitted into the wrapper and used by `mountRenderers` to find
// it again) regardless of whether the author supplied a public `id`. Without
// this, an anonymous renderer would render but its `mount` hook could never be
// located — see ticket 06 P2.
let _uidSeq = 0;

function _matchOne(spec, value, ctx) {
    if (spec == null || value == null) return false;
    if (typeof spec === 'string') return spec === value;
    if (spec instanceof RegExp) return spec.test(String(value));
    if (typeof spec === 'function') {
        try {
            return !!spec(value, ctx);
        } catch (err) {
            console.error('[ui-ext renderers] match predicate threw:', err);
            return false;
        }
    }
    return false;
}

function _gateOk(r, ctx) {
    if (typeof r.gate !== 'function') return true;
    try {
        return !!r.gate(ctx);
    } catch (err) {
        console.error('[ui-ext renderers] gate threw:', err);
        return false;
    }
}

function _sorted() {
    return _renderers
        .map((r, i) => ({ r, i }))
        .sort((a, b) => {
            const oa = typeof a.r.order === 'number' ? a.r.order : 100;
            const ob = typeof b.r.order === 'number' ? b.r.order : 100;
            return oa - ob || a.i - b.i;
        })
        .map((x) => x.r);
}

/**
 * Register a feature tool-output renderer. Re-registering the same `id` replaces
 * the prior renderer (dedupe). A renderer with neither `match.tool` nor
 * `match.contentType` is rejected — it could never dispatch.
 *
 * @param {FeatureRenderer} def
 */
export function registerRenderer(def) {
    if (!def || typeof def.render !== 'function') {
        console.error('[ui-ext renderers] registerRenderer: needs a `render` function');
        return;
    }
    const match = def.match || {};
    if (match.tool == null && match.contentType == null) {
        console.error('[ui-ext renderers] registerRenderer: `match` needs `tool` or `contentType`');
        return;
    }
    if (def.id) {
        const i = _renderers.findIndex((r) => r.id === def.id);
        if (i >= 0) {
            // Preserve the existing internal uid so a re-register doesn't orphan
            // already-mounted markup that points at the old uid.
            def._uid = _renderers[i]._uid || `ui-ext-r-${_uidSeq++}`;
            _renderers[i] = def;
            return;
        }
    }
    // Every renderer — anonymous or not — gets a stable internal uid so its
    // `mount` hook is always discoverable (ticket 06 P2).
    def._uid = `ui-ext-r-${_uidSeq++}`;
    _renderers.push(def);
}

/**
 * Remove a renderer by id.
 * @param {string} id
 */
export function unregisterRenderer(id) {
    const i = _renderers.findIndex((r) => r.id === id);
    if (i >= 0) _renderers.splice(i, 1);
}

/** First renderer whose `match.tool` matches `name` and whose gate passes. */
export function findToolRenderer(name, ctx = {}) {
    return _sorted().find((r) => r.match && _matchOne(r.match.tool, name, ctx) && _gateOk(r, ctx)) || null;
}

/** First renderer whose `match.contentType` matches `type` and whose gate passes. */
export function findContentTypeRenderer(type, ctx = {}) {
    return _sorted().find((r) => r.match && _matchOne(r.match.contentType, type, ctx) && _gateOk(r, ctx)) || null;
}

/**
 * Run a matched renderer and return SANITIZED, mount-discoverable markup — or
 * `null` when no renderer matches or the renderer misbehaves (so the caller
 * falls back to the core default path). `sanitize` is the core DOMPurify entry
 * (`window.SharedMarkdown.sanitizeHtml`); the registry never sanitizes itself,
 * but it ALWAYS routes the renderer's string through whatever `sanitize` core
 * supplies — there is no branch that returns un-sanitized renderer output.
 *
 * The returned markup is wrapped in a `[data-ui-ext-renderer]` element so
 * `mountRenderers` can later find the live root and invoke the renderer's
 * optional `mount` hook.
 *
 * @param {FeatureRenderer} renderer
 * @param {any}   payload
 * @param {object} ctx
 * @param {(html: string) => string} sanitize
 * @param {(s: string) => string} escapeAttr  - attribute-escaper for the wrapper id.
 * @returns {string|null}
 */
export function renderToSafeHtml(renderer, payload, ctx, sanitize, escapeAttr) {
    if (!renderer) return null;
    let out;
    try {
        out = renderer.render(payload, ctx);
    } catch (err) {
        console.error('[ui-ext renderers] render threw:', err);
        return null;
    }
    // Inert-string contract: anything that is not a string (Node, fragment,
    // object) is rejected — those can carry pre-attached listeners DOMPurify
    // cannot remove, defeating sanitization by construction.
    if (typeof out !== 'string') {
        if (out != null) {
            console.error('[ui-ext renderers] render must return a string, got', typeof out, '- rejecting');
        }
        return null;
    }
    const safe = typeof sanitize === 'function' ? sanitize(out) : '';
    const esc = typeof escapeAttr === 'function' ? escapeAttr : (s) => String(s);
    // Always emit the internal uid (assigned at registration) so `mountRenderers`
    // can find this renderer again — even when the author supplied no public id.
    const idAttr = renderer._uid ? ` data-ui-ext-renderer-id="${esc(renderer._uid)}"` : '';
    return `<div class="ui-ext-tool-render" data-ui-ext-renderer${idAttr}>${safe}</div>`;
}

/**
 * After core has built live DOM from sanitized markup, find every
 * renderer-owned root inside `containerEl` and invoke its `mount(rootEl, ctx)`
 * hook. Returns a single teardown fn that calls every collected teardown (or a
 * no-op). Idempotent per element via a `data-ui-ext-mounted` marker so a
 * re-render of the same container does not double-mount.
 *
 * @param {HTMLElement} containerEl
 * @param {object} ctx
 * @returns {() => void}
 */
export function mountRenderers(containerEl, ctx = {}) {
    if (!containerEl || typeof containerEl.querySelectorAll !== 'function') return () => {};
    const teardowns = [];
    const roots = containerEl.querySelectorAll('[data-ui-ext-renderer]');
    for (const root of roots) {
        if (root.dataset && root.dataset.uiExtMounted === '1') continue;
        const id = root.dataset ? root.dataset.uiExtRendererId : undefined;
        const renderer = id ? _renderers.find((r) => r._uid === id) : null;
        if (root.dataset) root.dataset.uiExtMounted = '1';
        if (!renderer || typeof renderer.mount !== 'function') continue;
        try {
            const ret = renderer.mount(root, ctx);
            if (typeof ret === 'function') teardowns.push(ret);
        } catch (err) {
            console.error('[ui-ext renderers] mount threw:', err);
        }
    }
    return () => {
        for (const t of teardowns) {
            try {
                t();
            } catch (err) {
                console.error('[ui-ext renderers] teardown threw:', err);
            }
        }
    };
}

/** The registered renderers, in registration order (read-only snapshot; tests). */
export function renderers() {
    return [..._renderers];
}

/** Test/teardown affordance: forget all renderers. */
export function _reset() {
    _renderers.length = 0;
}

export const Renderers = {
    registerRenderer,
    unregisterRenderer,
    findToolRenderer,
    findContentTypeRenderer,
    renderToSafeHtml,
    mountRenderers,
    renderers,
    _reset,
};

export default Renderers;
