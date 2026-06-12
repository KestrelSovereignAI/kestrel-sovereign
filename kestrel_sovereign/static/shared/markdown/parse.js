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
    // Mirror the finalize path (renderMarkdown): protect balanced math spans
    // from marked (which would consume `\(`/`\[` backslashes), restore them
    // HTML-escaped, then sanitize. Completing the tail's math closers (above)
    // is what makes a mid-stream span balanced enough for _protectMath to catch
    // it, so streaming and finalize agree on the delimiters the KaTeX pass reads.
    const { out, spans } = _protectMath(text);
    const parsed = marked.parse(out, {
        breaks: true, gfm: true, headerIds: false, mangle: false,
    });
    return sanitizeHtml(_restoreMath(parsed, spans));
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

const _FENCE_RE = /^[ \t]*(`{3,}|~{3,})/;
const _LIST_ITEM_RE = /^[ \t]*([-*+]|\d+[.)])\s/;
// A reference-style link/image definition line: `[label]: url`.
const _REF_DEF_RE = /^[ \t]{0,3}\[[^\]\n]+\]:[ \t]*\S/m;

// True when `text` ends inside an OPEN display-math region ($$…$$ or \[…\])
// — used so a blank line inside multi-line display math isn't read as a block
// boundary (which would split the math across stable/tail).
function _hasOpenMath(text) {
    if (((text.match(/\$\$/g) || []).length) % 2 !== 0) return true;
    if (((text.match(/\\\[/g) || []).length) > ((text.match(/\\\]/g) || []).length)) return true;
    return false;
}

// Walk the fence state of `text` line by line, invoking `onLine(openMarkerOrNull,
// lineIndex, line)` after each line's fence transition is applied. Returns the
// final open fence marker STRING (e.g. "```", "````", "~~~"), or null. Per
// CommonMark a fence closes only with the SAME marker char and a run AT LEAST AS
// LONG as the opener — so a ```` block containing a ``` line stays open, and a
// ``` block containing a ~~~ line stays open.
const _FENCE_CLOSE_RE = /^[ \t]*(`{3,}|~{3,})[ \t]*$/;  // closer: whitespace-only after
function _walkFences(text, onLine) {
    let open = null; // {char, len}
    const lines = text.split('\n');
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        let transitioned = false;
        if (open === null) {
            // Opener: may carry an info string (e.g. ```js).
            const m = line.match(_FENCE_RE);
            if (m) { open = { char: m[1][0], len: m[1].length }; transitioned = true; }
        } else {
            // Closer: same char, run AT LEAST AS LONG, and NOTHING but
            // whitespace after it (CommonMark) — so ```js inside a ``` block is
            // literal code, not a close.
            const m = line.match(_FENCE_CLOSE_RE);
            if (m && m[1][0] === open.char && m[1].length >= open.len) {
                open = null;
                transitioned = true;
            }
        }
        if (!transitioned && onLine) onLine(open, i, line);
    }
    return open ? open.char.repeat(open.len) : null;
}

function _openFence(text) {
    return _walkFences(text, null);
}

// Remove CLOSED fenced code regions (opener line through closer line). Callers
// run this only after confirming there is no still-OPEN fence, so every fence
// here is balanced. Code contents (which may contain `, **, [], $) must not
// count as inline markdown.
function _stripFences(text) {
    const lines = text.split('\n');
    const kept = [];
    let open = null;
    for (const line of lines) {
        if (open === null) {
            const m = line.match(_FENCE_RE);
            if (m) { open = { char: m[1][0], len: m[1].length }; continue; }
            kept.push(line);
        } else {
            const m = line.match(_FENCE_CLOSE_RE);
            if (m && m[1][0] === open.char && m[1].length >= open.len) open = null;
            // drop the line (fenced code body or the closer)
        }
    }
    return kept.join('\n');
}

// Drop inline code spans so their delimiters (a `**` in `` `**` `` can't open
// emphasis) don't count. Removes closed spans; an UNCLOSED span swallows to
// end-of-line (its tail is code).
function _stripInlineCode(text) {
    let s = text.replace(/(`+)[\s\S]*?\1/g, '');
    const tick = s.lastIndexOf('`');
    if (tick !== -1) {
        const nl = s.indexOf('\n', tick);
        s = s.slice(0, tick) + (nl !== -1 ? s.slice(nl) : '');
    }
    return s;
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
    let tailStartLine = 0;
    _walkFences(text, (openFence, i) => {
        if (openFence !== null) return;  // inside a fenced block — never split
        if (lines[i].trim() !== '') return;
        // Inside an open display-math region ($$…$$ or \[…\] crossing the
        // blank)? Keep it whole in the tail so the balanced span is protected.
        if (_hasOpenMath(lines.slice(0, i).join('\n'))) return;
        // Suppress this blank as a boundary ONLY for a genuine loose-list gap —
        // i.e. the block BEFORE the blank is list-ish AND the content AFTER it
        // is too (a new item or an indented continuation). A paragraph→list
        // transition is a real boundary, so the paragraph still memoizes and
        // tail completion can't bold across into the list (block-scoped flicker
        // prevention). "List-ish" = a list-item marker or an indented line.
        const listish = (s) => s !== undefined
            && (_LIST_ITEM_RE.test(s) || /^[ \t]/.test(s));
        let j = i + 1;
        while (j < lines.length && lines[j].trim() === '') j++;
        let k = i - 1;
        while (k >= 0 && lines[k].trim() === '') k--;
        if (listish(lines[j]) && listish(lines[k])) return;
        tailStartLine = i + 1;
    });
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
    // Open fenced code block → close it with the SAME marker+length that opened
    // it (CommonMark); leave everything inside untouched.
    const openFence = _openFence(t);
    if (openFence) {
        return t + '\n' + openFence;
    }
    // Strip CLOSED fenced regions first so code bodies (which may contain
    // backticks, **, [], $) never count as inline markdown.
    const noFence = _stripFences(t);
    // Inline code: remove BALANCED spans (delimiter-run aware, so a multi-
    // backtick span like `` ` `` that contains a literal backtick is balanced),
    // then any remaining backtick run is an unclosed opener — close it with a
    // matching run.
    const balanced = noFence.replace(/(`+)[\s\S]*?\1/g, '');
    const openTick = balanced.match(/`+/);
    if (openTick) t += openTick[0];
    // The remaining inline constructs must also ignore inline code spans.
    const code = _stripInlineCode(noFence);
    // Unclosed link/image target: `[text](url` / `![alt](url` with no `)`.
    if (/!?\[[^\]\n]*\]\([^)\n]*$/.test(code)) t += ')';
    // Bold **: complete only a genuine emphasis opener — odd ** count AND the
    // last ** immediately followed by a word char. Globs/operators (**/*.py),
    // a trailing **, or ** before punctuation are left literal, matching what
    // the finalized markdown render would do.
    if (((code.match(/\*\*/g) || []).length) % 2 !== 0) {
        const after = code[code.lastIndexOf('**') + 2];
        if (after && /\w/.test(after)) t += '**';
    }
    // Display math $$ and bracket/paren math \[ \] , \( \) — close if open.
    if (((code.match(/\$\$/g) || []).length) % 2 !== 0) t += '$$';
    if (((code.match(/\\\[/g) || []).length) > ((code.match(/\\\]/g) || []).length)) t += '\\]';
    if (((code.match(/\\\(/g) || []).length) > ((code.match(/\\\)/g) || []).length)) t += '\\)';
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
        // Reference-style link definitions (`[label]: url`) are document-global:
        // the definition and the `[label]` that uses it can live in different
        // blank-line blocks. Parsing stable/tail independently would lose that
        // context, so for these (uncommon) messages fall back to a single
        // whole-content parse (fence-close only) — references resolve, at the
        // cost of memoization + immediate inline formatting for just this turn.
        if (_REF_DEF_RE.test(processedContent)) {
            const openFence = _openFence(processedContent);
            return _streamMarkedParse(openFence ? processedContent + '\n' + openFence : processedContent);
        }
        const { stable, tail } = _splitStreamingTail(processedContent);
        const stableHtml = _renderStreamStable(stable);
        const tailHtml = tail ? _streamMarkedParse(_completeStreamingInline(tail)) : '';
        return stableHtml + tailHtml;
    } catch (e) {
        return sanitizeHtml(content.replace(/\n/g, '<br>'));
    }
}
