/**
 * Markdown Parsing Utilities
 * Core parsing functions using marked.js
 */

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
        return marked.parse(processedContent);
    } catch (e) {
        return content.replace(/\n/g, '<br>');
    }
}
