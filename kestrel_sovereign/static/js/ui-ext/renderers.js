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
// then builds the live DOM. This contract sanitizes the returned markup; it
// does not sandbox installed same-origin feature JavaScript. Context can carry
// the owning pane for routing, and feature code can already access the DOM.
// A renderer that returns a `Node`/`DocumentFragment` (which
// could carry `addEventListener` listeners DOMPurify cannot strip) is REJECTED.
//
// Imperative behavior (canvas, chart, click handlers) happens exclusively in the
// optional `mount(rootEl, ctx)` hook, which core invokes AFTER it has
// parsed+sanitized the markup and created `rootEl`. `mount` should confine its
// changes to `rootEl` and its descendants and may return a teardown fn. The
// supplied owner/pane context is routing information, not an authorization grant.
//
// This module is generic — no feature-name strings. It retains mounted roots
// only for their active resource lifetime and does not import markdown; core
// supplies the sanitizer and ends each mount before discarding its DOM.
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
const _mounts = new Map();
// Physical DOM lifetime is separate from a reusable mount's lifetime. Weak
// keys let discarded subtrees stay retired without retaining their DOM.
const _domLifetimes = new WeakMap();

function domLifetime(node) {
    let lifetime = _domLifetimes.get(node);
    if (!lifetime) {
        lifetime = { retired: false, retiring: 0 };
        _domLifetimes.set(node, lifetime);
    }
    return lifetime;
}

function mountIsRetired(root) {
    for (let node = root; node; node = node.parentNode) {
        const lifetime = _domLifetimes.get(node);
        if (lifetime?.retired || lifetime?.retiring) return true;
    }
    return false;
}

function retireRendererMounts(renderer) {
    for (const mounted of [..._mounts.values()]) {
        if (mounted.renderer === renderer) mounted.dispose();
    }
}

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
            const previous = _renderers[i];
            def._uid = previous._uid || `ui-ext-r-${_uidSeq++}`;
            _renderers[i] = def;
            retireRendererMounts(previous);
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
    if (i >= 0) {
        const [renderer] = _renderers.splice(i, 1);
        retireRendererMounts(renderer);
    }
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
 * no-op). The registry owns mount identity; the DOM marker is diagnostic.
 * Disposing a mount permits the same DOM to mount again.
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
        if (mountIsRetired(root)) continue;
        if (_mounts.has(root)) continue;
        const id = root.dataset ? root.dataset.uiExtRendererId : undefined;
        const renderer = id ? _renderers.find((r) => r._uid === id) : null;
        if (!renderer || typeof renderer.mount !== 'function') continue;
        let active = true;
        let teardown;
        const release = () => {
            const callback = teardown;
            teardown = null;
            if (typeof callback !== 'function') return;
            try {
                callback();
            } catch (err) {
                console.error('[ui-ext renderers] teardown threw:', err);
            }
        };
        const dispose = () => {
            if (!active) return;
            active = false;
            _mounts.delete(root);
            if (root.dataset) delete root.dataset.uiExtMounted;
            release();
        };
        // Claim before invoking feature code, including a reentrant mount or
        // synchronous unregister. A teardown returned after retirement still
        // runs once, without disturbing a newer mount of the same root.
        _mounts.set(root, { renderer, dispose });
        if (root.dataset) root.dataset.uiExtMounted = '1';
        try {
            teardown = renderer.mount(root, ctx);
            if (!active) release();
        } catch (err) {
            dispose();
            console.error('[ui-ext renderers] mount threw:', err);
        }
        teardowns.push(dispose);
    }
    return () => {
        for (const dispose of teardowns) dispose();
    };
}

/**
 * End mounts; an ordinary unmount permits explicit remounting later.
 * `retire: 'self'` permanently retires discarded DOM, while `'children'`
 * retires a container's outgoing children and keeps the container reusable.
 * Reentrant cleanup cannot mount resources inside an operation in progress.
 */
export function unmountRenderers(containerEl, { retire = false } = {}) {
    if (![false, 'self', 'children'].includes(retire)) {
        throw new TypeError('Renderer retirement must be self, children, or false');
    }
    if (!containerEl || typeof containerEl.querySelectorAll !== 'function') return;
    const lifetime = domLifetime(containerEl);
    const retireChildren = () => {
        for (const child of containerEl.querySelectorAll(':scope > *')) {
            domLifetime(child).retired = true;
        }
    };
    lifetime.retiring += 1;
    if (retire === 'self') lifetime.retired = true;
    if (retire === 'children') retireChildren();
    try {
        const roots = [containerEl, ...containerEl.querySelectorAll('[data-ui-ext-renderer]')];
        for (const root of roots) _mounts.get(root)?.dispose();
    } finally {
        // Cleanup may append DOM. It cannot mount while `retiring` is set,
        // and any replacement children it created are outgoing DOM too.
        if (retire === 'children') retireChildren();
        lifetime.retiring -= 1;
    }
}

/** The registered renderers, in registration order (read-only snapshot; tests). */
export function renderers() {
    return [..._renderers];
}

/** Test/teardown affordance: forget all renderers. */
export function _reset() {
    for (const mounted of [..._mounts.values()]) mounted.dispose();
    _renderers.length = 0;
}

export const Renderers = {
    registerRenderer,
    unregisterRenderer,
    findToolRenderer,
    findContentTypeRenderer,
    renderToSafeHtml,
    mountRenderers,
    unmountRenderers,
    renderers,
    _reset,
};

export default Renderers;
