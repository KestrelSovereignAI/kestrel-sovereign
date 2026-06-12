/**
 * Markdown Parsing Utilities
 * Core parsing functions using marked.js
 */

// Escape attribute values inserted into rendered HTML.
function _escapeAttr(value) {
    return String(value).replace(/[&<>"']/g, (ch) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
    }[ch]));
}

// ---------------------------------------------------------------------------
// HTML sanitization (DOMPurify, hardened).
//
// The markdown stream is NOT trusted authorship: it carries web_search
// results, tool output, and echoed user content, and frinz is multi-tenant.
// So every HTML string produced by marked is run through DOMPurify before it
// reaches innerHTML. Sanitizing the *string* here (not the live DOM) is the
// right boundary: hljs spans and mermaid SVG are injected into the DOM AFTER
// this step (highlightCodeBlocks / renderMermaidDiagrams), so they are never
// stripped by the sanitizer.
//
// Hardening beyond DOMPurify's safe defaults (which already block <script>,
// inline event handlers, and javascript: URLs):
//   - allowDataImages:false  — drop data: image sources (exfil/HTML-smuggling)
//   - force rel="noopener noreferrer" on every target="_blank" anchor, so the
//     guarantee holds even for links DOMPurify keeps but our marked renderer
//     didn't emit (e.g. raw <a> in the source).
// ---------------------------------------------------------------------------
const _SANITIZE_CONFIG = {
    // Markdown only ever produces HTML — restrict to the HTML profile so the
    // SVG and MathML namespaces are dropped entirely. That closes the whole
    // class of SVG data-URI / script vectors (e.g. <svg><image href="data:…">)
    // rather than chasing each sub-element. Mermaid renders its SVG into the
    // DOM AFTER this step (renderMermaidDiagrams), so it is never sanitized
    // here and is unaffected.
    USE_PROFILES: { html: true },
    // marked emits class="language-xxx" (hljs + mermaid detection rely on it);
    // target is needed for external-link new-tab behavior.
    ADD_ATTR: ['target'],
    ALLOW_DATA_ATTR: false,
    // Keep the contents of a removed tag (e.g. strip a stray <foo> but keep its
    // text) rather than dropping the whole subtree.
    KEEP_CONTENT: true,
};

let _purifyHooksInstalled = false;
function _installPurifyHooks() {
    if (_purifyHooksInstalled) return;
    if (typeof DOMPurify === 'undefined' || typeof DOMPurify.addHook !== 'function') return;
    DOMPurify.addHook('afterSanitizeAttributes', (node) => {
        const tag = node.tagName ? node.tagName.toLowerCase() : '';
        // Block data: image sources (allowDataImages:false).
        if (tag === 'img' || tag === 'source') {
            const src = node.getAttribute('src') || '';
            if (/^data:/i.test(src.trim())) node.removeAttribute('src');
            const srcset = node.getAttribute('srcset') || '';
            if (/data:/i.test(srcset)) node.removeAttribute('srcset');
        }
        // Any anchor opening a new tab must be reverse-tabnabbing safe. The
        // _blank browsing-context keyword is case-insensitive, so normalize
        // before comparing (target="_BLANK" must not slip through).
        if (tag === 'a') {
            const target = (node.getAttribute('target') || '').trim().toLowerCase();
            if (target === '_blank') node.setAttribute('rel', 'noopener noreferrer');
        }
    });
    _purifyHooksInstalled = true;
}

let _sanitizerWarned = false;
/**
 * Sanitize a rendered-HTML string before it is assigned to innerHTML.
 *
 * Fails CLOSED when DOMPurify is absent (CDN blocked, CSP, offline deploy, or
 * the node test sandbox): rather than passing raw marked output to innerHTML —
 * an XSS hole in exactly the failure mode where the sanitizer is missing — the
 * HTML is escaped to inert text. A degraded-but-safe render beats executing
 * `<img onerror=…>`. A one-time warning makes the missing sanitizer visible.
 * @param {string} html - HTML produced by marked
 * @returns {string} Sanitized HTML (or escaped text when no sanitizer is available)
 */
function sanitizeHtml(html) {
    if (typeof DOMPurify === 'undefined' || typeof DOMPurify.sanitize !== 'function') {
        if (!_sanitizerWarned) {
            _sanitizerWarned = true;
            if (typeof console !== 'undefined' && console.warn) {
                console.warn('[SharedMarkdown] DOMPurify not loaded — failing closed, markdown shown as escaped text');
            }
        }
        return _escapeAttr(html);
    }
    _installPurifyHooks();
    return DOMPurify.sanitize(html, _SANITIZE_CONFIG);
}

// Configure marked once, on first load, to render external links with
// target="_blank" rel="noopener noreferrer". In-page (#anchor) links keep
// default behavior so internal jump links still work in the same view.
let _markedLinkRendererInstalled = false;
function _installMarkedLinkRenderer() {
    if (_markedLinkRendererInstalled) return;
    if (typeof marked === 'undefined' || typeof marked.use !== 'function') return;

    // marked v11 passes a token object {href, title, text}; older versions
    // pass positional args (href, title, text). Handle both.
    function renderLink(hrefOrToken, titleArg, textArg) {
        let href, title, text;
        if (typeof hrefOrToken === 'object' && hrefOrToken !== null) {
            href = hrefOrToken.href;
            title = hrefOrToken.title;
            // marked passes inner tokens; if rendered text is on the token, use it,
            // otherwise fall back to the raw text.
            text = hrefOrToken.text;
            if (hrefOrToken.tokens && this && typeof this.parser?.parseInline === 'function') {
                try {
                    text = this.parser.parseInline(hrefOrToken.tokens);
                } catch (_e) {
                    // fall back to token.text
                }
            }
        } else {
            href = hrefOrToken;
            title = titleArg;
            text = textArg;
        }
        if (href == null) href = '';
        const isInPage = typeof href === 'string' && href.startsWith('#');
        const titleAttr = title ? ` title="${_escapeAttr(title)}"` : '';
        const hrefAttr = _escapeAttr(href);
        if (isInPage) {
            return `<a href="${hrefAttr}"${titleAttr}>${text}</a>`;
        }
        return `<a href="${hrefAttr}"${titleAttr} target="_blank" rel="noopener noreferrer">${text}</a>`;
    }

    marked.use({ renderer: { link: renderLink } });
    _markedLinkRendererInstalled = true;
}

/**
 * Normalize excessive newlines in text.
 * Collapses 3+ consecutive newlines to exactly 2 (single paragraph break).
 * This prevents LLMs outputting excessive blank lines from creating too much spacing.
 * @param {string} text - Raw text to normalize
 * @returns {string} Text with normalized newlines
 */
function normalizeNewlines(text) {
    if (!text) return '';
    return text.replace(/\n{3,}/g, '\n\n');
}

// Math-span protection (#1661 KaTeX). marked would escape `\(`/`\[` to `(`/`[`
// (so the KaTeX post-pass never sees those delimiters) and could split a
// `$$…$$` block across markdown constructs. So we lift math spans OUT before
// marked, leave an inert private-use placeholder, then restore them AFTER —
// HTML-escaped, so the special chars inside math (`<`, `&`, …) are safe through
// innerHTML while KaTeX still reads the real characters from textContent.
// Display ($$, \[) before inline (\() so the longer delimiters win. No bare
// `$…$`: chat prose has too much currency / shell-var noise.
const _MATH_SPAN_RE = /\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)/g;
const _MATH_PLACEHOLDER = '\uE000'; // BMP private-use char; passes through marked as inert text

function _protectMath(text) {
    const spans = [];
    const out = text.replace(_MATH_SPAN_RE, (m) => {
        spans.push(m);
        return `${_MATH_PLACEHOLDER}${spans.length - 1}${_MATH_PLACEHOLDER}`;
    });
    return { out, spans };
}

function _restoreMath(html, spans) {
    if (!spans.length) return html;
    return html.replace(
        new RegExp(`${_MATH_PLACEHOLDER}(\\d+)${_MATH_PLACEHOLDER}`, 'g'),
        (_m, i) => _escapeAttr(spans[Number(i)] || ''),
    );
}

/**
 * Render markdown text to HTML
 * @param {string} text - Raw text that may contain markdown
 * @returns {string} HTML string with rendered markdown
 */
function renderMarkdown(text) {
    if (!text) return '';

    // Normalize excessive newlines before parsing
    const normalized = normalizeNewlines(text);

    if (typeof marked !== 'undefined') {
        _installMarkedLinkRenderer();
        const { out, spans } = _protectMath(normalized);
        const parsed = marked.parse(out, {
            breaks: true,
            gfm: true,
            headerIds: false,
            mangle: false
        });
        // Restore (HTML-escaped) math BEFORE sanitize — the escaped delimiters
        // are inert text to DOMPurify; KaTeX's post-pass reads the real chars
        // from textContent and renders them.
        return sanitizeHtml(_restoreMath(parsed, spans));
    }

    return sanitizeHtml(normalized.replace(/\n/g, '<br>'));
}

// ===== #1660 block-based streaming markdown (Streamdown-style) =====
//
// Two pieces working together:
//  1. Split the streamed text into a STABLE prefix (completed top-level blocks,
//     separated by blank lines) and the still-growing TAIL block. The stable
//     prefix's parse is memoized — it only re-parses when a new block finalizes,
//     not on every chunk.
//  2. Complete unclosed inline constructs (code/links/bold/math) on the TAIL
//     BLOCK ONLY, so formatting renders immediately mid-stream. Scoping the
//     synthetic closer to the tail block is what makes this flicker-free where
//     #1547's whole-content completion was not: a completed-then-reverted
//     construct can only repaint the actively-streaming tail block, never flip
//     the entire bubble bold/italic for a frame.

const _STREAM_STABLE_CACHE = new Map();
const _STREAM_STABLE_CACHE_CAP = 64;

function _streamMarkedParse(text) {
    return sanitizeHtml(marked.parse(text, {
        breaks: true, gfm: true, headerIds: false, mangle: false,
    }));
}

// Render (and memoize) the stable prefix. Keyed by the exact prefix string, so
// concurrently-streaming panes never collide and a growing message reuses the
// prior render until a new block boundary forms.
function _renderStreamStable(stable) {
    if (!stable) return '';
    const hit = _STREAM_STABLE_CACHE.get(stable);
    if (hit !== undefined) return hit;
    const html = _streamMarkedParse(stable);
    _STREAM_STABLE_CACHE.set(stable, html);
    if (_STREAM_STABLE_CACHE.size > _STREAM_STABLE_CACHE_CAP) {
        // Evict oldest (insertion order) — the prior, now-superseded prefixes.
        _STREAM_STABLE_CACHE.delete(_STREAM_STABLE_CACHE.keys().next().value);
    }
    return html;
}

const _FENCE_RE = /^[ \t]*(```+|~~~+)/;
const _LIST_ITEM_RE = /^[ \t]*([-*+]|\d+[.)])\s/;

// Return the marker char ('`' or '~') of a still-OPEN fence in `text`, else
// null. A fence only closes with the SAME marker character it opened with, so a
// ``` block containing a `~~~` line stays open (CommonMark).
function _openFenceMarker(text) {
    let marker = null;
    for (const line of text.split('\n')) {
        const m = line.match(_FENCE_RE);
        if (!m) continue;
        const c = m[1][0];
        if (marker === null) marker = c;
        else if (marker === c) marker = null;
        // a different marker while inside a fence is literal code — ignore
    }
    return marker;
}

// Fence-aware split into {stable, tail}. The boundary is the last blank line
// that is NOT inside a fenced code block AND not a loose-list gap; an open
// (unclosed) fence keeps its whole region in the tail. Blank-line-separated
// top-level blocks are independent in markdown, so rendering stable and tail
// separately is safe — EXCEPT a blank line between list items (a loose list is
// one list), so we never finalize across a blank line whose next content is a
// list item.
function _splitStreamingTail(text) {
    const lines = text.split('\n');
    let fenceMarker = null;
    let tailStartLine = 0;
    for (let i = 0; i < lines.length; i++) {
        const m = lines[i].match(_FENCE_RE);
        if (m) {
            const c = m[1][0];
            if (fenceMarker === null) fenceMarker = c;
            else if (fenceMarker === c) fenceMarker = null;
            continue;
        }
        if (fenceMarker === null && lines[i].trim() === '') {
            // Peek the next non-blank line: if it's a list item this blank may
            // be a loose-list gap, so keep the prefix in the tail (don't split
            // a multi-item list into separate lists mid-stream).
            let j = i + 1;
            while (j < lines.length && lines[j].trim() === '') j++;
            if (j < lines.length && _LIST_ITEM_RE.test(lines[j])) continue;
            tailStartLine = i + 1;
        }
    }
    return {
        stable: lines.slice(0, tailStartLine).join('\n'),
        tail: lines.slice(tailStartLine).join('\n'),
    };
}

// Complete unclosed constructs in the tail block. Conservative by design:
// completes only the constructs that can be detected unambiguously, so it can't
// reintroduce the #1547 mis-count flicker. Single-`*` italic and bare `$`
// (currency) are deliberately NOT completed — they render fine unclosed until
// their real closer streams in.
function _completeStreamingInline(tail) {
    let t = tail;
    // Open fenced code block → close it with the SAME marker that opened it;
    // leave everything inside untouched.
    const openMarker = _openFenceMarker(t);
    if (openMarker) {
        return t + (openMarker === '~' ? '\n~~~' : '\n```');
    }
    // Inline code: odd backtick count (fences are balanced here) → close.
    if (((t.match(/`/g) || []).length) % 2 !== 0) t += '`';
    // Unclosed link/image target: `[text](url` / `![alt](url` with no `)`.
    if (/!?\[[^\]\n]*\]\([^)\n]*$/.test(t)) t += ')';
    // Bold ** (rarely a bullet/operator) → close if unbalanced.
    if (((t.match(/\*\*/g) || []).length) % 2 !== 0) t += '**';
    // Display math $$ and bracket/paren math \[ \] , \( \) — close if open.
    if (((t.match(/\$\$/g) || []).length) % 2 !== 0) t += '$$';
    if (((t.match(/\\\[/g) || []).length) > ((t.match(/\\\]/g) || []).length)) t += '\\]';
    if (((t.match(/\\\(/g) || []).length) > ((t.match(/\\\)/g) || []).length)) t += '\\)';
    return t;
}

/**
 * Render markdown for streaming content with implied closing tags.
 * Memoizes completed top-level blocks and only re-renders the growing tail
 * block, completing its unclosed inline constructs so formatting appears
 * immediately (#1660).
 * @param {string} content - Partial markdown content being streamed
 * @returns {string} HTML string safe for incomplete markdown
 */
function renderStreamingMarkdown(content) {
    if (typeof marked === 'undefined') {
        return sanitizeHtml(normalizeNewlines(content).replace(/\n/g, '<br>'));
    }

    _installMarkedLinkRenderer();

    const processedContent = normalizeNewlines(content);
    try {
        const { stable, tail } = _splitStreamingTail(processedContent);
        const stableHtml = _renderStreamStable(stable);
        const tailHtml = tail ? _streamMarkedParse(_completeStreamingInline(tail)) : '';
        return stableHtml + tailHtml;
    } catch (e) {
        return sanitizeHtml(content.replace(/\n/g, '<br>'));
    }
}
