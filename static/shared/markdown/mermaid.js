/**
 * Mermaid Diagram Rendering
 * Converts mermaid code blocks to rendered diagrams
 */

/**
 * Wait for mermaid library to be available
 * @param {number} maxWait - Maximum time to wait in ms
 * @returns {Promise<object|null>} Mermaid library or null
 */
async function waitForMermaid(maxWait = 3000) {
    const start = Date.now();
    while ((Date.now() - start) < maxWait) {
        if (window.mermaid || typeof mermaid !== 'undefined') {
            return window.mermaid || mermaid;
        }
        await new Promise(r => setTimeout(r, 100));
    }
    return null;
}

/**
 * Render mermaid diagrams in an element
 * @param {HTMLElement} element - Container element with mermaid code blocks
 */
async function renderMermaidDiagrams(element) {
    if (!element) return;

    const mermaidBlocks = element.querySelectorAll('code.language-mermaid');
    if (mermaidBlocks.length === 0) return;

    // Wait for mermaid to be available
    const mermaidLib = await waitForMermaid();
    if (!mermaidLib) {
        console.warn('[Mermaid] Library not available');
        return;
    }

    console.log(`[Mermaid] Rendering ${mermaidBlocks.length} diagram(s)`);

    for (const block of mermaidBlocks) {
        const code = block.textContent;
        const pre = block.parentElement;

        // Create wrapper for the diagram
        const wrapper = document.createElement('div');
        wrapper.className = 'mermaid-wrapper';

        try {
            // Generate unique ID for this diagram
            const id = `mermaid-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
            const { svg } = await mermaidLib.render(id, code);
            wrapper.innerHTML = svg;
            wrapper.classList.add('mermaid-success');
            console.log(`[Mermaid] Rendered diagram ${id}`);
        } catch (e) {
            // Show error with original code
            wrapper.innerHTML = `<div class="mermaid-error">
                <strong>Mermaid Error:</strong> ${e.message || 'Failed to render'}
                <pre><code>${code.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</code></pre>
            </div>`;
            wrapper.classList.add('mermaid-failed');
            console.warn('[Mermaid] Render error:', e);
        }

        // Replace the pre>code block with the wrapper
        if (pre && pre.tagName === 'PRE') {
            pre.replaceWith(wrapper);
        } else {
            block.replaceWith(wrapper);
        }
    }
}
