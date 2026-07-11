/**
 * Unified Model-settings popover (#2264).
 *
 * One toolbar button toggles a single panel that holds two labeled sections:
 *   • Chat       — vendor / route / [upstream] / model combos (SharedModelSelector)
 *   • Embeddings — Auto-follow default + explicit provider:route (EmbeddingSelector)
 *
 * This controller owns only the OPEN/CLOSE + placement behavior. The section
 * contents are driven by the two existing controllers, which it does not
 * construct — chat.js wires those and hands their elements in via the DOM.
 *
 * Scoped-embed safe (#2233): the panel is re-homed under the configured
 * overlay root on open so it escapes ``overflow:hidden`` toolbars and inherits
 * the scoped console's CSS variables, then positioned under the trigger button
 * with fixed coordinates. In the standalone console the overlay root defaults
 * to ``document.body``.
 */

class ModelSettingsPopover {
    /**
     * @param {Object} options
     * @param {string} options.buttonId  Trigger button element id.
     * @param {string} options.panelId   Panel element id (contains both sections).
     * @param {Function} [options.getOverlayRoot] Returns the element the panel
     *        should be appended to while open. Defaults to ``document.body``.
     * @param {Function} [options.onOpen]  Called when the panel opens.
     * @param {Function} [options.onClose] Called when the panel closes.
     */
    constructor(options = {}) {
        this.button = document.getElementById(options.buttonId);
        this.panel = document.getElementById(options.panelId);
        this.getOverlayRoot = options.getOverlayRoot
            || (() => (typeof document !== 'undefined' ? document.body : null));
        this.onOpen = options.onOpen || (() => {});
        this.onClose = options.onClose || (() => {});
        this.isOpen = false;

        this._onDocClick = (e) => this._handleDocClick(e);
        this._onKeydown = (e) => this._handleKeydown(e);

        this._bind();
    }

    _bind() {
        if (this.button) {
            this.button.addEventListener('click', (e) => {
                e.stopPropagation();
                this.toggle();
            });
            this.button.setAttribute('aria-haspopup', 'true');
            this.button.setAttribute('aria-expanded', 'false');
        }
        // Clicks inside the panel must not bubble up to the document handler
        // that closes it.
        if (this.panel) {
            this.panel.addEventListener('click', (e) => e.stopPropagation());
        }
    }

    toggle() {
        if (this.isOpen) this.close();
        else this.open();
    }

    open() {
        if (!this.panel || this.isOpen) return;
        this.isOpen = true;

        // Re-home under the overlay root so the panel isn't clipped by the
        // toolbar and picks up the scoped console's CSS (#2233).
        const root = this.getOverlayRoot();
        if (root && this.panel.parentNode !== root) {
            root.appendChild(this.panel);
        }

        this.panel.style.display = '';
        this._position();

        if (this.button) this.button.setAttribute('aria-expanded', 'true');

        // Defer wiring the dismiss handlers so the opening click doesn't
        // immediately close the panel.
        if (typeof document !== 'undefined') {
            setTimeout(() => {
                document.addEventListener('click', this._onDocClick);
                document.addEventListener('keydown', this._onKeydown);
            }, 0);
        }
        this.onOpen();
    }

    close() {
        if (!this.isOpen) return;
        this.isOpen = false;
        if (this.panel) this.panel.style.display = 'none';
        if (this.button) this.button.setAttribute('aria-expanded', 'false');
        if (typeof document !== 'undefined') {
            document.removeEventListener('click', this._onDocClick);
            document.removeEventListener('keydown', this._onKeydown);
        }
        this.onClose();
    }

    _position() {
        if (!this.panel || !this.button || typeof this.button.getBoundingClientRect !== 'function') {
            return;
        }
        const rect = this.button.getBoundingClientRect();
        this.panel.style.position = 'fixed';
        this.panel.style.top = `${Math.round(rect.bottom + 6)}px`;
        this.panel.style.left = `${Math.round(rect.left)}px`;
        this.panel.style.zIndex = '3000';
    }

    _handleDocClick(e) {
        if (!this.panel) return;
        const target = e.target;
        if (this.panel.contains && this.panel.contains(target)) return;
        if (this.button && this.button.contains && this.button.contains(target)) return;
        this.close();
    }

    _handleKeydown(e) {
        if (e.key === 'Escape' || e.keyCode === 27) this.close();
    }
}

// Export for ES modules / node --test
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ModelSettingsPopover };
}

// Export globally for script-tag usage
if (typeof window !== 'undefined') {
    window.ModelSettingsPopover = ModelSettingsPopover;
}
