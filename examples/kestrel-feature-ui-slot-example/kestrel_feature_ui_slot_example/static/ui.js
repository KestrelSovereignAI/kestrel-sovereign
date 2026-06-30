/**
 * ui.js — the reference feature's frontend entry module (#2043).
 *
 * Served from THIS package at /features/ui-slot-example/static/ui.js (NOT core
 * static/). The boot loader imports it; it registers one button into the
 * `chat-input-actions` slot.
 *
 * Note the imports: an out-of-tree module reaches core modules by their
 * absolute, same-origin URLs (`/js/...`), since a relative `./registry.js`
 * would resolve under this feature's own mount, not core's.
 */

import UI from '/js/ui-ext/registry.js';

UI.register({
    slot: 'chat-input-actions',
    id: 'ui-slot-example-button',
    order: 50,
    render(el) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'ui-slot-example-btn';
        btn.title = 'UI slot example (out-of-tree feature)';
        btn.setAttribute('aria-label', 'UI slot example');
        btn.textContent = '★';
        const onClick = () => {
            // eslint-disable-next-line no-alert
            window.alert('Hello from an out-of-tree feature — loaded via the UI manifest.');
        };
        btn.addEventListener('click', onClick);
        el.appendChild(btn);
        // Teardown: the registry calls this before re-render/unmount.
        return () => btn.removeEventListener('click', onClick);
    },
});
