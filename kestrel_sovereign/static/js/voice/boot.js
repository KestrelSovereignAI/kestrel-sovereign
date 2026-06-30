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
 * Importing this module initializes the voice UI shell exactly as the old
 * `initVoiceUI()` call in `app.js` did.
 */

import { initVoiceUI } from './ui.js';

initVoiceUI();
