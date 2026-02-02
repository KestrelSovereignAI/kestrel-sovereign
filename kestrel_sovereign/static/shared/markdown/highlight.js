/**
 * Code Syntax Highlighting
 * Uses highlight.js for syntax highlighting
 */

/**
 * Apply syntax highlighting to code blocks in an element
 * @param {HTMLElement} element - Container element with code blocks
 * @param {boolean} onlyNew - Only highlight blocks not already highlighted
 */
function highlightCodeBlocks(element, onlyNew = false) {
    if (typeof hljs === 'undefined') return;

    const selector = onlyNew ? 'pre code:not(.hljs)' : 'pre code';
    element.querySelectorAll(selector).forEach((block) => {
        // Skip mermaid blocks (and partial names during streaming)
        const classList = Array.from(block.classList);
        const isMermaid = classList.some(c => c.startsWith('language-mer'));
        if (isMermaid) return;

        // Get the language from class
        const langClass = classList.find(c => c.startsWith('language-'));
        if (langClass) {
            const lang = langClass.replace('language-', '');
            // Only highlight if hljs knows the language
            if (!hljs.getLanguage(lang)) return;
        }

        hljs.highlightElement(block);
    });
}
