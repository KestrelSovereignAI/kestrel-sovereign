/**
 * Kestrel Sovereign Console - Kebab Menu Primitive
 *
 * A small, reusable overflow-menu ("⋯") primitive (#2149). One module, no
 * framework. Other rows/panels will want the same affordance, so the menu
 * building + open/close/keyboard lifecycle lives here rather than baked into
 * any one list.
 *
 * Contract:
 *   - `createKebabButton(getItems, opts)` returns a real <button> the caller
 *     appends into a row. Clicking it (or calling `openMenuAt`) opens the menu
 *     built from `getItems()` — evaluated lazily so item labels/handlers can
 *     reflect current state (e.g. Archive vs Unarchive).
 *   - `openMenuAt(items, position)` opens the same menu from an arbitrary
 *     accelerator — e.g. a row `contextmenu` (right-click) handler — so
 *     right-click is never the ONLY path to the actions.
 *
 * Menu items: `{ label, labelKey?, danger?, separatorBefore?, onSelect }`.
 * Item buttons are real, focusable buttons; the menu is keyboard-accessible
 * (ArrowUp/Down move focus, Escape closes, Enter/click activates). Labels are
 * i18n-tagged via `data-label-key` like the existing nav labels.
 */

let openMenuEl = null;
let cleanupFns = [];

function escapeHtmlSafe(str) {
    // Prefer the shared helper if present so escaping matches the rest of the
    // console; fall back to a minimal inline escape in bare test contexts.
    const sm = typeof window !== 'undefined' && window.escapeHtml;
    if (typeof sm === 'function') return sm(str);
    return String(str == null ? '' : str)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
}

/** Tear down any open menu and its outside-click/Escape listeners. */
export function closeKebabMenu() {
    if (openMenuEl) {
        openMenuEl.remove();
        openMenuEl = null;
    }
    for (const fn of cleanupFns) {
        try { fn(); } catch (_) { /* best-effort */ }
    }
    cleanupFns = [];
}

function focusableItems() {
    if (!openMenuEl || typeof openMenuEl.querySelectorAll !== 'function') return [];
    return Array.from(openMenuEl.querySelectorAll('.kebab-menu-item'));
}

/**
 * Open a menu built from `items` at the given viewport position. Returns the
 * menu element (mostly for tests). Any already-open menu is closed first so at
 * most one menu is live at a time.
 */
export function openMenuAt(items, position = {}) {
    closeKebabMenu();

    const menu = document.createElement('div');
    menu.className = 'kebab-menu';
    menu.setAttribute('role', 'menu');
    menu.style.cssText = `
        position: fixed;
        top: ${Math.max(0, position.y || 0)}px;
        left: ${Math.max(0, position.x || 0)}px;
        min-width: 180px;
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        box-shadow: 0 6px 24px rgba(0,0,0,0.2);
        z-index: 2000;
        padding: 0.25rem;
        overflow: hidden;
    `;

    (items || []).filter(Boolean).forEach((item) => {
        if (item.separatorBefore) {
            const sep = document.createElement('div');
            sep.className = 'kebab-menu-separator';
            sep.setAttribute('role', 'separator');
            sep.style.cssText = 'height: 1px; margin: 0.25rem 0; background: var(--border-color);';
            menu.appendChild(sep);
        }
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = `kebab-menu-item${item.danger ? ' kebab-menu-item-danger' : ''}`;
        btn.setAttribute('role', 'menuitem');
        if (item.labelKey) btn.dataset.labelKey = item.labelKey;
        btn.dataset.action = item.action || '';
        btn.style.cssText = `
            display: block;
            width: 100%;
            text-align: left;
            padding: 0.4rem 0.6rem;
            background: none;
            border: none;
            border-radius: 6px;
            font-size: 0.8rem;
            cursor: pointer;
            color: ${item.danger ? 'var(--error, #ef4444)' : 'var(--text-primary)'};
        `;
        btn.innerHTML = escapeHtmlSafe(item.label);
        btn.addEventListener('mouseover', () => { btn.style.background = 'var(--bg-tertiary)'; });
        btn.addEventListener('mouseout', () => { btn.style.background = 'none'; });
        btn.addEventListener('click', (e) => {
            if (e && typeof e.stopPropagation === 'function') e.stopPropagation();
            closeKebabMenu();
            if (typeof item.onSelect === 'function') item.onSelect(e);
        });
        menu.appendChild(btn);
    });

    document.body.appendChild(menu);
    openMenuEl = menu;

    // Keyboard: Arrow navigation between items, Escape closes.
    const onKeyDown = (e) => {
        const items2 = focusableItems();
        if (!items2.length) return;
        const idx = items2.indexOf(e.target);
        if (e.key === 'Escape') {
            e.preventDefault();
            closeKebabMenu();
        } else if (e.key === 'ArrowDown') {
            e.preventDefault();
            const next = items2[(idx + 1 + items2.length) % items2.length] || items2[0];
            next.focus && next.focus();
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            const prev = items2[(idx - 1 + items2.length) % items2.length] || items2[items2.length - 1];
            prev.focus && prev.focus();
        }
    };
    menu.addEventListener('keydown', onKeyDown);

    // Outside-click / global-Escape close. Deferred a tick so the opening
    // click/contextmenu that spawned the menu doesn't immediately close it.
    const onDocClick = (e) => {
        if (openMenuEl && !openMenuEl.contains(e.target)) closeKebabMenu();
    };
    const onDocKey = (e) => { if (e.key === 'Escape') closeKebabMenu(); };
    setTimeout(() => {
        document.addEventListener('click', onDocClick);
        document.addEventListener('keydown', onDocKey);
    }, 0);
    cleanupFns.push(() => document.removeEventListener('click', onDocClick));
    cleanupFns.push(() => document.removeEventListener('keydown', onDocKey));

    // Focus the first item so keyboard users land inside the menu.
    const first = focusableItems()[0];
    if (first && typeof first.focus === 'function') first.focus();

    return menu;
}

/** Position helper: derive viewport coords from a click/contextmenu event. */
export function positionFromEvent(event) {
    if (!event) return { x: 0, y: 0 };
    if (typeof event.clientX === 'number' && typeof event.clientY === 'number'
        && (event.clientX || event.clientY)) {
        return { x: event.clientX, y: event.clientY };
    }
    const target = event.currentTarget || event.target;
    if (target && typeof target.getBoundingClientRect === 'function') {
        const rect = target.getBoundingClientRect();
        return { x: rect.left, y: rect.bottom };
    }
    return { x: 0, y: 0 };
}

/**
 * Build the "⋯" kebab button. `getItems` is called each time the menu opens so
 * items reflect current state. The button is a real, focusable button with an
 * accessible label; menu items are focusable too (keyboard-accessible per
 * #2149). Clicks stop propagation so the owning row's click handler (which
 * usually loads the conversation) does not also fire.
 */
export function createKebabButton(getItems, opts = {}) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `kebab-btn${opts.className ? ' ' + opts.className : ''}`;
    btn.setAttribute('aria-haspopup', 'menu');
    btn.setAttribute('aria-label', opts.ariaLabel || 'Conversation actions');
    btn.title = opts.title || 'More actions';
    const icon = typeof window !== 'undefined' && typeof window.kicon === 'function'
        ? window.kicon('ellipsis-vertical')
        : '⋯';
    btn.innerHTML = icon || '⋯';
    btn.addEventListener('click', (e) => {
        if (e && typeof e.stopPropagation === 'function') e.stopPropagation();
        if (e && typeof e.preventDefault === 'function') e.preventDefault();
        openMenuAt(getItems(), positionFromEvent(e));
    });
    return btn;
}
