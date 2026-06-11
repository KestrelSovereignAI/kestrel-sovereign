# Embedding the Kestrel chat component

The chat UI (`kestrel_sovereign/static/js/chat.js`) is an embeddable component
with a small public API (chat-extract epic #1597). The Kestrel console itself
mounts it the same way an external host does — `app.js` calls `mount()` — so the
console is the reference consumer alongside external embedders (e.g. Frinz).

A runnable example lives at
[`kestrel_sovereign/static/examples/embed-chat-example.mjs`](../../kestrel_sovereign/static/examples/embed-chat-example.mjs).

## Mounting

```js
import { mount, createChatComponent } from './js/chat.js';

// Mount into a container and initialize. Returns the public component API.
const chat = mount(containerEl, { deps });

// Or: configure without mounting (apply deps/container, get the API back).
const api = createChatComponent({ deps, container: containerEl });
```

- **`mount(containerEl, config)`** — sets the chat root to `containerEl`
  (falls back to `document` when null), applies `config.deps` if given, runs
  `initChat()`, and returns the component API. All element lookups are scoped
  to `containerEl`, so multiple concerns can coexist on one page.
- **`createChatComponent(config)`** — applies `config.deps` / `config.container`
  and returns the API **without** mounting/initializing.
- `config.deps` is an optional dependency override (api client, toast,
  markdown renderer, …); omit it to use the built-in singletons.

> Floating overlays (command-autocomplete, toasts) are intentionally appended
> to `document.body` and positioned `fixed`, so they are not scoped to the
> mount container.

## Markdown rendering & required scripts

Assistant messages render through the shared markdown module
(`kestrel_sovereign/static/shared/markdown/`). An embedder that serves these
files itself (e.g. Frinz via a `/kestrel-ui/` mount) must load them — and their
libraries — in this order, **before** `chat.js` runs:

```html
<!-- libraries -->
<script src=".../marked@11/marked.min.js"></script>
<!-- DOMPurify MUST precede parse.js (see below) -->
<script src=".../dompurify@3/dist/purify.min.js"></script>
<script src=".../highlight.js@11/highlight.min.js"></script>

<!-- shared module, in order -->
<script src="shared/markdown/parse.js"></script>
<script src="shared/markdown/highlight.js"></script>
<script src="shared/markdown/mermaid.js"></script>
<script src="shared/markdown/katex.js"></script>
<script src="shared/markdown/index.js"></script>
```

- **Sanitization (required).** All rendered markdown is run through DOMPurify
  (hardened: HTML-profile only, `data:` images blocked, `rel=noopener` forced
  on `target=_blank`). If DOMPurify is **not** loaded before `parse.js`, the
  renderer **fails closed** — markdown is shown as escaped text, not raw HTML,
  with a one-time console warning. Always load DOMPurify.
- **Mermaid** diagrams (` ```mermaid `) and **KaTeX** math (`$$…$$`, `\[…\]`,
  `\(…\)` — no bare `$…$`) are **lazy-loaded** on first use by their shared
  modules; you don't load mermaid/KaTeX yourself. If a host omits `katex.js`,
  `index.js` degrades to no-math (no error). Math runs only on finalized
  messages; both render as a post-sanitize DOM pass and are never re-sanitized.
- Tool activity renders from typed in-band stream sentinels into collapsible
  cards — no host wiring needed.

## Hooks

### `registerPartRenderer(type, fn)`

Register a renderer for a custom message-part `type`. `fn(data)` is called when
`appendMessagePart(type, data)` runs.

```js
registerPartRenderer('image', (data) => {
    const img = document.createElement('img');
    img.src = data.src;
    img.alt = data.alt || '';
    return img;            // return a DOM Node for rich parts
});

api.appendMessagePart('image', { src: '/img/selfie.png', alt: 'selfie' });
```

- **Return a DOM `Node`** for rich/untrusted content — it is appended as-is.
  A returned **string** is set as `innerHTML` (trusted-markup contract); never
  build that string from untrusted data.
- A renderer that throws is isolated: `appendMessagePart` logs it and degrades
  to escaped text rather than breaking the whole conversation.
- With no renderer registered for a type, the part degrades to escaped text.

### `registerHeaderAction({ id, icon, title, label, onClick })`

Add a button to the chat header.

```js
registerHeaderAction({
    id: 'example-image',
    title: 'Insert an example image',  // tooltip (always plain text)
    icon: '🖼️',                        // string glyph, or a DOM Node for SVG
    label: 'Image',                    // always escaped as text
    onClick: () => api.appendMessagePart('image', { src: '/img/x.png' }),
});
```

- `id` dedupes — re-registering the same `id` replaces the action.
- `label` is **always** HTML-escaped. `icon` may be a DOM **Node** (appended
  safely) or a string treated as embedder-trusted markup (for simple glyphs);
  never pass untrusted data as an `icon` string.
- `title` is set as a DOM property (safe).
