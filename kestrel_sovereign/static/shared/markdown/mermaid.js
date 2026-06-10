/**
 * Mermaid Diagram Rendering
 * Converts mermaid code blocks to rendered diagrams.
 *
 * Three properties this module guarantees:
 *   1. Lazy-load — the (heavy) mermaid ESM bundle is fetched only when a
 *      diagram actually appears, not on every page load. A host that eagerly
 *      sets window.mermaid still works (we use it as-is).
 *   2. Content-hash cache — identical diagram source renders once; later
 *      occurrences (e.g. a chat pane remounting) reuse the cached SVG instead
 *      of re-running mermaid's parse/layout and replacing the node.
 *   3. Collision-safe reuse — every injected SVG gets a fresh id namespace, so
 *      two instances of the same cached diagram on one page never clash on the
 *      internal ids mermaid uses for markers/gradients/clip-paths.
 */

const _MERMAID_CDN = 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';

// Source string -> rendered SVG string (pre-uniquification). Bounded so a long
// session can't grow it without limit.
const _svgCache = new Map();
const _SVG_CACHE_MAX = 100;

function _cacheGet(source) {
    return _svgCache.get(source);
}
function _cacheSet(source, svg) {
    if (_svgCache.size >= _SVG_CACHE_MAX) {
        // Evict the oldest entry (Map preserves insertion order).
        _svgCache.delete(_svgCache.keys().next().value);
    }
    _svgCache.set(source, svg);
}

// Per-injection id namespacing. mermaid emits internal ids (markers, gradients,
// clip-paths) referenced via url(#id) / href="#id"; injecting the same SVG
// twice would duplicate those ids and break the second instance. Rewriting all
// ids (and their references) with a unique suffix on every injection keeps each
// instance self-contained.
let _injectionCounter = 0;
function _uniquifySvgIds(svg) {
    const suffix = `i${++_injectionCounter}`;
    const ids = new Set();
    let out = svg.replace(/\bid="([^"]+)"/g, (_m, id) => {
        ids.add(id);
        return `id="${id}-${suffix}"`;
    });
    if (ids.size === 0) return out;

    // Reference forms that point at an id with '#': url(#id), href="#id", and —
    // crucially — the scoped CSS selectors mermaid emits in its inline <style>
    // (e.g. `#mermaid-123 .node{…}`). The negative lookahead bounds the match so
    // a short id can't partially rewrite a longer one (#a vs #ab).
    ids.forEach((id) => {
        const esc = id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        out = out.replace(new RegExp(`#${esc}(?![\\w-])`, 'g'), `#${id}-${suffix}`);
    });

    // Bare id references (no '#'): aria-labelledby / aria-describedby token lists.
    out = out.replace(
        /(aria-(?:labelledby|describedby)=")([^"]*)(")/g,
        (_m, pre, val, post) =>
            pre + val.split(/\s+/).map((t) => (ids.has(t) ? `${t}-${suffix}` : t)).join(' ') + post,
    );
    return out;
}

function _escapeText(text) {
    return String(text).replace(/[&<>]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[ch]));
}

// Indirection seam: the dynamic import and the wait cap live here so tests can
// simulate a slow/failed load without network access.
const _loader = {
    maxWait: 3000,
    import: (url) => import(url),
};

let _mermaidLoadPromise = null;
/**
 * Resolve the mermaid library, lazy-loading it on first need.
 *
 * Returns null if the load hasn't finished within the wait cap (so message
 * finalization is never blocked on a slow CDN). When that happens the load is
 * still in flight — callers should retry once _mermaidLoadPromise resolves
 * rather than dropping the diagram.
 * @returns {Promise<object|null>} Mermaid library or null
 */
async function ensureMermaid() {
    if (typeof window !== 'undefined' && window.mermaid) return window.mermaid;
    if (typeof mermaid !== 'undefined') return mermaid;

    if (!_mermaidLoadPromise) {
        _mermaidLoadPromise = (async () => {
            try {
                const mod = await _loader.import(_MERMAID_CDN);
                const lib = (mod && mod.default) || mod;
                if (lib && typeof lib.initialize === 'function') {
                    lib.initialize({ startOnLoad: false, theme: 'default' });
                }
                if (typeof window !== 'undefined') window.mermaid = lib;
                return lib;
            } catch (e) {
                console.warn('[Mermaid] Lazy-load failed:', e);
                _mermaidLoadPromise = null; // allow a later diagram to retry
                return null;
            }
        })();
    }

    // Bound the wait so a hung CDN never blocks message finalization forever.
    return await Promise.race([
        _mermaidLoadPromise,
        new Promise((r) => setTimeout(
            () => r((typeof window !== 'undefined' && window.mermaid) || null),
            _loader.maxWait,
        )),
    ]);
}

/**
 * Render mermaid diagrams in an element
 * @param {HTMLElement} element - Container element with mermaid code blocks
 */
async function renderMermaidDiagrams(element) {
    if (!element) return;

    const mermaidBlocks = element.querySelectorAll('code.language-mermaid');
    if (mermaidBlocks.length === 0) return;

    const mermaidLib = await ensureMermaid();
    if (!mermaidLib) {
        // The load timed out but may still be in flight (slow/cold CDN). Render
        // this element once it resolves instead of dropping the diagram. The
        // code blocks are only replaced on success, so they're still here on
        // the retry; the cache keeps it from being wasted work.
        if (_mermaidLoadPromise) {
            _mermaidLoadPromise.then((lib) => { if (lib) renderMermaidDiagrams(element); });
        } else {
            console.warn('[Mermaid] Library not available');
        }
        return;
    }

    for (const block of mermaidBlocks) {
        const code = block.textContent;
        const pre = block.parentElement;

        const wrapper = document.createElement('div');
        wrapper.className = 'mermaid-wrapper';

        try {
            let svg = _cacheGet(code);
            if (svg === undefined) {
                const id = `mermaid-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
                ({ svg } = await mermaidLib.render(id, code));
                _cacheSet(code, svg);
            }
            // Namespace ids per injection so coexisting copies never collide.
            wrapper.innerHTML = _uniquifySvgIds(svg);
            wrapper.classList.add('mermaid-success');
        } catch (e) {
            // Show the error with the original (escaped) code. The message can
            // echo diagram source, so escape it too.
            wrapper.innerHTML = `<div class="mermaid-error">
                <strong>Mermaid Error:</strong> ${_escapeText(e && e.message ? e.message : 'Failed to render')}
                <pre><code>${_escapeText(code)}</code></pre>
            </div>`;
            wrapper.classList.add('mermaid-failed');
            console.warn('[Mermaid] Render error:', e);
        }

        if (pre && pre.tagName === 'PRE') {
            pre.replaceWith(wrapper);
        } else {
            block.replaceWith(wrapper);
        }
    }
}
