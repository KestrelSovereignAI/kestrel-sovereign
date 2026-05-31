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

    // Close unclosed code blocks (```)
    const codeBlockMatches = processedContent.match(/```/g) || [];
    if (codeBlockMatches.length % 2 !== 0) {
        processedContent += '\n```';
    }

    // Close unclosed inline code (single `)
    const inlineCodeCount = (processedContent.match(/(?<!`)`(?!`)/g) || []).length;
    if (inlineCodeCount % 2 !== 0) {
        processedContent += '`';
    }

    // Close unclosed bold (**)
    const boldMatches = processedContent.match(/\*\*/g) || [];
    if (boldMatches.length % 2 !== 0) {
        processedContent += '**';
    }

    // Close unclosed italic (single * not part of **)
    const italicMatches = processedContent.match(/(?<!\*)\*(?!\*)/g) || [];
    if (italicMatches.length % 2 !== 0) {
        processedContent += '*';
    }

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
