/**
 * Example: embedding the Kestrel chat component and extending it.
 *
 * Demonstrates the public embedding API shipped by the chat-extract epic
 * (#1597): mount the component into any container, register a custom
 * message-part renderer, and add a header action. This is the same surface
 * the Kestrel console itself uses (app.js mounts via `mount()`), and that
 * external hosts (e.g. Frinz) use to embed chat.
 *
 * See docs/frontend/embedding-chat.md for the full hook reference.
 */
import {
    mount,
    registerPartRenderer,
    registerHeaderAction,
} from '../js/chat.js';

/**
 * A custom renderer for `image` message parts. `data` is `{ src, alt }`.
 *
 * ALWAYS return a DOM Node for rich parts — a returned Node is appended
 * as-is, whereas a returned string is set as innerHTML (trusted-markup
 * contract). Building the <img> as a Node keeps host data out of the HTML
 * parser entirely.
 */
export function imagePartRenderer(data) {
    const img = document.createElement('img');
    img.src = (data && data.src) || '';
    img.alt = (data && data.alt) || '';
    img.className = 'chat-image-part';
    img.style.maxWidth = '320px';
    img.style.borderRadius = '8px';
    return img;
}

/**
 * Mount the chat component into `container` and wire the example extensions:
 *   - an `image` part renderer (renders `appendMessagePart('image', …)`), and
 *   - a header action that appends an example image part on click.
 *
 * Returns the public chat component API from `mount()`.
 */
export function setupEmbeddedChat(container) {
    const api = mount(container, {});

    registerPartRenderer('image', imagePartRenderer);

    registerHeaderAction({
        id: 'example-image',
        title: 'Insert an example image',
        icon: '🖼️',
        label: 'Image',
        onClick: () =>
            api.appendMessagePart('image', {
                src: '/static/img/example.png',
                alt: 'example image part',
            }),
    });

    return api;
}
