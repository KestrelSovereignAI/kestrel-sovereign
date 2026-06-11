/**
 * Shared Markdown Rendering Utilities
 *
 * Load order: parse.js, highlight.js, mermaid.js, katex.js, then this file
 */

// katex.js loads just before this file; guard so a host that hasn't added it
// to its script list yet (e.g. an external /kestrel-ui consumer mid-rollout)
// degrades to no-math instead of a ReferenceError. `typeof` on an undeclared
// identifier is safe — it returns 'undefined' rather than throwing.
const _renderMathSafe = (typeof renderMath === 'function')
    ? renderMath
    : async () => {};

/**
 * Render markdown and apply syntax highlighting to an element
 * @param {HTMLElement} element - Container element to render into
 * @param {string} text - Markdown text to render
 */
async function renderMarkdownInto(element, text) {
    element.innerHTML = renderMarkdown(text);
    highlightCodeBlocks(element);
    await renderMermaidDiagrams(element);
    await _renderMathSafe(element);
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
 * Finalize a streaming message - full render with mermaid + math support
 * @param {HTMLElement} element - Container element to render into
 * @param {string} content - Final complete markdown content
 */
async function finalizeMarkdown(element, content) {
    element.innerHTML = renderMarkdown(content);
    highlightCodeBlocks(element);
    await renderMermaidDiagrams(element);
    // Math runs last (post-sanitize DOM pass, ignores code/pre).
    await _renderMathSafe(element);
}

// Export globally for script tag usage
window.SharedMarkdown = {
    renderMarkdown,
    renderStreamingMarkdown,
    highlightCodeBlocks,
    renderMermaidDiagrams,
    renderMath: _renderMathSafe,
    renderMarkdownInto,
    renderStreamingMarkdownInto,
    finalizeMarkdown
};
