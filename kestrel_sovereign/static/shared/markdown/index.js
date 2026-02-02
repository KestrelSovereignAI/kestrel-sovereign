/**
 * Shared Markdown Rendering Utilities
 *
 * Load order: parse.js, highlight.js, mermaid.js, then this file
 */

/**
 * Render markdown and apply syntax highlighting to an element
 * @param {HTMLElement} element - Container element to render into
 * @param {string} text - Markdown text to render
 */
async function renderMarkdownInto(element, text) {
    element.innerHTML = renderMarkdown(text);
    highlightCodeBlocks(element);
    await renderMermaidDiagrams(element);
}

/**
 * Render streaming markdown into an element with highlighting
 * @param {HTMLElement} element - Container element to render into
 * @param {string} content - Streaming markdown content
 */
function renderStreamingMarkdownInto(element, content) {
    element.innerHTML = renderStreamingMarkdown(content);
    highlightCodeBlocks(element, true);
}

/**
 * Finalize a streaming message - full render with mermaid support
 * @param {HTMLElement} element - Container element to render into
 * @param {string} content - Final complete markdown content
 */
async function finalizeMarkdown(element, content) {
    element.innerHTML = renderMarkdown(content);
    highlightCodeBlocks(element);
    await renderMermaidDiagrams(element);
}

// Export globally for script tag usage
window.SharedMarkdown = {
    renderMarkdown,
    renderStreamingMarkdown,
    highlightCodeBlocks,
    renderMermaidDiagrams,
    renderMarkdownInto,
    renderStreamingMarkdownInto,
    finalizeMarkdown
};
