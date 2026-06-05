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
        return marked.parse(normalized, {
            breaks: true,
            gfm: true,
            headerIds: false,
            mangle: false
        });
    }

    return normalized.replace(/\n/g, '<br>');
}

/**
 * Render markdown for streaming content with implied closing tags.
 * Handles incomplete code blocks, bold, italic gracefully during streaming.
 * @param {string} content - Partial markdown content being streamed
 * @returns {string} HTML string safe for incomplete markdown
 */
function renderStreamingMarkdown(content) {
    if (typeof marked === 'undefined') {
        return normalizeNewlines(content).replace(/\n/g, '<br>');
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
        return marked.parse(processedContent, {
            breaks: true,
            gfm: true,
            headerIds: false,
            mangle: false,
        });
    } catch (e) {
        return content.replace(/\n/g, '<br>');
    }
}
