// ============================================================================
// Feature UI contributions loader (manifest-driven, out-of-tree assets; #2043)
// ============================================================================
//
// Fetches the merged ``/api/ui/contributions`` manifest and dynamically
// ``import()``s each enabled feature's modules (and injects its stylesheets) so
// a feature can mount slot/panel contributions with no edits to core static/.
//
// Why this lives in its own module (#2048): in multi-agent host mode no agent is
// selected at boot, so the boot-time call in ``app.js`` hits the host's
// un-prefixed ``/api/ui/contributions`` which 503s (``get_agent`` has no active
// agent) and imports nothing — the extracted Spawn panel would never load. The
// loader must therefore also run AFTER an agent is selected (``selectAgent`` in
// ``identity.js``) and when capabilities flip at runtime (a feature enabled
// after boot), at which point the request is host-agent-prefixed and resolves.
//
// Re-running is safe and idempotent: ES module ``import()`` caches by URL so an
// already-loaded module is not re-executed, ``registerPanel`` replaces a prior
// registration in place, and ``injectFeatureStylesheet`` de-dupes by href.
// Concurrent invocations are coalesced onto a single in-flight fetch.

import API from '../api.js';

// Inject a feature-contributed stylesheet once. Idempotent — the manifest may
// be (re)loaded and a stylesheet must not be appended twice.
function injectFeatureStylesheet(href) {
    if (!href) return;
    if (document.querySelector(`link[data-ui-ext-css="${CSS.escape(href)}"]`)) return;
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = href;
    link.dataset.uiExtCss = href;
    document.head.appendChild(link);
}

let _inFlight = null;

// Fetch the merged UI-contributions manifest and dynamically import each enabled
// feature's modules in declared order. The server enabled-filters and rejects
// cross-origin module URLs; we additionally honor the client capability set so a
// host force-off also suppresses loading. A failed import for one feature is
// isolated so it cannot abort the rest of boot. Concurrent callers share the
// same in-flight run so a boot call + an agent-select call don't double-fetch.
export async function loadFeatureUIContributions() {
    if (_inFlight) return _inFlight;
    _inFlight = (async () => {
        let contributions = [];
        try {
            const data = await API.request('/api/ui/contributions');
            if (data && Array.isArray(data.contributions)) {
                contributions = data.contributions;
            }
        } catch (e) {
            console.warn('[ui-ext] failed to fetch UI contributions manifest:', e);
            return;
        }

        for (const entry of contributions) {
            if (entry.capability && !API.hasCapability(entry.capability)) continue;
            for (const href of entry.css || []) {
                injectFeatureStylesheet(href);
            }
            for (const mod of entry.modules || []) {
                try {
                    await import(mod);
                } catch (e) {
                    console.error(
                        `[ui-ext] failed to import feature module ${mod} (feature ${entry.feature}):`,
                        e,
                    );
                }
            }
        }
    })().finally(() => { _inFlight = null; });
    return _inFlight;
}
