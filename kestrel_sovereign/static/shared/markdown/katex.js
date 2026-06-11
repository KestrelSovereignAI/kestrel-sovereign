/**
 * KaTeX Math Rendering
 *
 * Renders LaTeX math ($…$, $$…$$, \(…\), \[…\]) in a finalized message, as a
 * POST-sanitize DOM pass — mirrors mermaid.js:
 *   1. Lazy-load — KaTeX + its auto-render extension + CSS are fetched only
 *      when a message actually contains math delimiters. A host that eagerly
 *      sets window.renderMathInElement is used as-is.
 *   2. Finalize-only — runs on the completed bubble, never per chunk, so a
 *      partial "$x +" mid-stream can't mis-render.
 *   3. Sanitizer-safe — KaTeX mutates the DOM directly (renderMathInElement),
 *      so its markup never passes through DOMPurify (#1658's HTML-only profile
 *      would otherwise strip it) — the same boundary mermaid uses.
 */

// Pinned to the last 0.16.x — battle-tested and ubiquitous on jsDelivr. All
// three artifacts below are verified present (HTTP 206); the fail-graceful
// path degrades to raw delimiters rather than crashing if a fetch ever fails.
const _KATEX_VERSION = '0.16.11';
const _KATEX_CSS = `https://cdn.jsdelivr.net/npm/katex@${_KATEX_VERSION}/dist/katex.min.css`;
const _KATEX_AUTORENDER = `https://cdn.jsdelivr.net/npm/katex@${_KATEX_VERSION}/dist/contrib/auto-render.mjs`;

// Deliberately NO bare `$…$` inline delimiter: chat prose routinely contains
// currency and shell vars ("$5 today and $10 tomorrow", "$PATH"), which a
// single-dollar matcher would swallow as math. Inline math uses \(…\); display
// uses $$…$$ or \[…\]. Display delimiters listed first so the longer ones win.
const _MATH_DELIMITERS = [
    { left: '$$', right: '$$', display: true },
    { left: '\\[', right: '\\]', display: true },
    { left: '\\(', right: '\\)', display: false },
];
// Never render math inside code/pre (a `$` in a shell snippet is a prompt) or
// inside an svg — mermaid injects its diagram SVG before this pass runs, and
// auto-render would otherwise rewrite SVG text nodes and break the diagram.
const _MATH_IGNORED_TAGS = ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'option', 'svg'];

// Cheap guard: only pay the lazy-load when a real delimiter is present ($$,
// \(, \[ — NOT a bare $). After a successful render the delimiters are gone
// (replaced by KaTeX spans), so this also makes a re-run idempotent.
function _hasMath(text) {
    return /\$\$|\\\(|\\\[/.test(String(text || ''));
}

// Indirection seam so tests can simulate a slow/failed load without network.
const _katexLoader = {
    maxWait: 3000,
    import: (url) => import(url),
};

function _ensureKatexCss() {
    if (typeof document === 'undefined') return;
    if (document.querySelector('link[data-katex-css]')) return;
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = _KATEX_CSS;
    link.setAttribute('data-katex-css', '1');
    document.head.appendChild(link);
}

let _katexLoadPromise = null;
/**
 * Resolve KaTeX's auto-render function, lazy-loading on first need. Returns
 * null if the load hasn't finished within the wait cap (so finalization is
 * never blocked); callers should retry once _katexLoadPromise resolves.
 * @returns {Promise<Function|null>}
 */
async function ensureKatex() {
    if (typeof window !== 'undefined' && window.renderMathInElement) {
        return window.renderMathInElement;
    }
    if (!_katexLoadPromise) {
        _katexLoadPromise = (async () => {
            try {
                _ensureKatexCss();
                const mod = await _katexLoader.import(_KATEX_AUTORENDER);
                const fn = (mod && mod.default) || mod;
                if (typeof window !== 'undefined') window.renderMathInElement = fn;
                return fn;
            } catch (e) {
                console.warn('[KaTeX] Lazy-load failed:', e);
                _katexLoadPromise = null; // allow a later message to retry
                return null;
            }
        })();
    }
    return await Promise.race([
        _katexLoadPromise,
        new Promise((r) => setTimeout(
            () => r((typeof window !== 'undefined' && window.renderMathInElement) || null),
            _katexLoader.maxWait,
        )),
    ]);
}

/**
 * Render LaTeX math in an element (in place).
 * @param {HTMLElement} element - Container element to render math into
 */
async function renderMath(element) {
    if (!element || !_hasMath(element.textContent)) return;

    const render = await ensureKatex();
    if (!render) {
        // Load timed out but may still be in flight — render once it resolves
        // rather than leaving math as raw "$…$" text. The _hasMath guard makes
        // the retry idempotent (a successful render removes the delimiters).
        if (_katexLoadPromise) {
            _katexLoadPromise.then((fn) => { if (fn) renderMath(element); });
        } else {
            console.warn('[KaTeX] Library not available');
        }
        return;
    }

    try {
        render(element, {
            delimiters: _MATH_DELIMITERS,
            ignoredTags: _MATH_IGNORED_TAGS,
            // Show the offending source in red instead of throwing — a bad
            // expression must never break the rest of the message.
            throwOnError: false,
        });
    } catch (e) {
        console.warn('[KaTeX] render error:', e);
    }
}
