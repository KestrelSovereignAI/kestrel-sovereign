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
    // Syntax highlighting needs hljs; the copy-button pass does not. Gate only
    // the highlighting on hljs so copy buttons still appear if hljs failed to
    // load (#1574).
    if (typeof hljs !== 'undefined') {
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

    addCodeCopyButtons(element);
}

/**
 * Add a hover-reveal "Copy" button to each fenced code block in a container.
 *
 * Scoped to `pre > code` blocks (real fenced blocks) so inline `code` chips
 * and tool-activity cards (which never contain `<pre>`) are untouched. Safe to
 * call on every streaming re-render: a `<pre>` that already carries a button is
 * skipped via the `data-copy-ready` flag, so buttons never stack.
 *
 * @param {HTMLElement} element - Container element with rendered code blocks
 */
function addCodeCopyButtons(element) {
    if (!element || typeof document === 'undefined') return;

    element.querySelectorAll('pre > code').forEach((code) => {
        const pre = code.parentElement;
        if (!pre || pre.dataset.copyReady === '1') return;
        // Defensive: never decorate code that lives inside a tool-activity card.
        if (pre.closest('.tool-activity-container')) return;
        pre.dataset.copyReady = '1';

        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'code-copy-btn';
        btn.setAttribute('aria-label', 'Copy code');
        btn.textContent = 'Copy';

        btn.addEventListener('click', async (event) => {
            event.preventDefault();
            event.stopPropagation();
            // Read the raw source at click time — the block may have been
            // re-highlighted or finalized since the button was injected.
            const text = code.textContent || '';
            const ok = await copyTextToClipboard(text);
            if (btn.dataset.resetTimer) {
                clearTimeout(Number(btn.dataset.resetTimer));
            }
            const label = ok ? 'Copied' : 'Failed';
            btn.textContent = label;
            // Keep the accessible name in sync with the visible label so screen
            // readers announce the result, not the stale "Copy code".
            btn.setAttribute('aria-label', `${label} code`);
            btn.classList.toggle('code-copy-btn--ok', ok);
            const timer = setTimeout(() => {
                btn.textContent = 'Copy';
                btn.setAttribute('aria-label', 'Copy code');
                btn.classList.remove('code-copy-btn--ok');
                delete btn.dataset.resetTimer;
            }, 1500);
            btn.dataset.resetTimer = String(timer);
        });

        pre.appendChild(btn);
    });
}

/**
 * Copy text to the clipboard, falling back to execCommand on insecure
 * contexts (e.g. plain-http LAN access). Returns true on success.
 * @param {string} text
 * @returns {Promise<boolean>}
 */
async function copyTextToClipboard(text) {
    try {
        if (navigator.clipboard && window.isSecureContext) {
            await navigator.clipboard.writeText(text);
            return true;
        }
    } catch (_err) {
        // Fall through to the legacy path below.
    }
    const ta = document.createElement('textarea');
    try {
        ta.value = text;
        ta.setAttribute('readonly', '');
        ta.style.position = 'absolute';
        ta.style.left = '-9999px';
        document.body.appendChild(ta);
        ta.select();
        return document.execCommand('copy');
    } catch (_err) {
        return false;
    } finally {
        // Ensure the temp textarea never lingers in the DOM, even if select()
        // or execCommand() throws on a restricted insecure context.
        if (ta.parentNode) ta.parentNode.removeChild(ta);
    }
}
