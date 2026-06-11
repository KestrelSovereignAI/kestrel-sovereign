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

/**
 * Render markdown for streaming content with implied closing tags.
 * Handles incomplete code blocks, bold, italic gracefully during streaming.
 * @param {string} content - Partial markdown content being streamed
 * @returns {string} HTML string safe for incomplete markdown
 */
function renderStreamingMarkdown(content) {
    if (typeof marked === 'undefined') {
        return sanitizeHtml(normalizeNewlines(content).replace(/\n/g, '<br>'));
    }

    _installMarkedLinkRenderer();

    // Normalize excessive newlines before processing
    let processedContent = normalizeNewlines(content);

    // Close unclosed *fenced code blocks* (```) only.
    //
    // A fence is a BLOCK construct: until its closer arrives, marked
    // treats everything after the opening ``` as code to the end of
    // input, so the rest of the bubble renders as one giant code block.
    // Appending a synthetic closer keeps it a bounded <pre> that simply
    // grows as more lines stream in — stable, no flicker.
    const codeBlockMatches = processedContent.match(/```/g) || [];
    if (codeBlockMatches.length % 2 !== 0) {
        processedContent += '\n```';
    }

    // NOTE: we deliberately do NOT synthesize closers for inline
    // emphasis (`**`, `*`) or inline code (`) anymore (#1547). Those
    // were counted with naive regexes that can't tell an emphasis
    // delimiter from a list bullet (`* item`), a multiplication sign,
    // or a stray asterisk. An odd count wrapped a synthetic delimiter
    // around a large span, so the whole bubble flipped bold/italic for
    // a frame and then reverted when the next chunk balanced the count
    // — the "all the text gets bigger/bold then snaps back" flicker.
    // Inline constructs render fine unclosed: marked emits the literal
    // characters until the real closer streams in, then re-resolves to
    // emphasis. That's the standard, non-flickering streaming behavior.

    try {
        // Match the finalize-path options (renderMarkdown above) so the
        // streamed bubble lays out the same way during stream as it does
        // once finalized. The defaults here would collapse single `\n`
        // into a space (CommonMark), scrunching chat lines — including
        // the inline tool-activity markers — into one paragraph until
        // the stream ended. The catch-fallback below already preserves
        // line breaks via `\n` → `<br>`, so the no-`breaks` `try` path
        // was the inconsistent branch.
        return sanitizeHtml(marked.parse(processedContent, {
            breaks: true,
            gfm: true,
            headerIds: false,
            mangle: false,
        }));
    } catch (e) {
        return sanitizeHtml(content.replace(/\n/g, '<br>'));
    }
}
