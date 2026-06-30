/**
 * boot.js — Voice feature manifest entry module (#2043).
 *
 * Loaded by the UI-contributions boot loader (`app.js`
 * `loadFeatureUIContributions`) instead of being imported directly by
 * `app.js`. The voice JS still lives in core `static/` today (its assets are
 * slated to move into the `kestrel-feature-voice` package later); until then
 * this thin entry is what the core-bundled manifest entry points at, so voice
 * loads through the same path as any out-of-tree feature.
 *
 * Importing this module initializes the voice UI shell. As of #2042 (ticket 04)
 * voice/ui.js self-registers its slot contributions at import time, so the bare
 * side-effect import below is all that's needed — there is no `initVoiceUI()` to
 * call anymore.
 */

import './ui.js';
