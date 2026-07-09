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

// Pin a feature-static asset URL to the selected agent in multi-agent host mode.
//
// The manifest is fetched per selected agent (host-agent-prefixed), but the
// module/css URLs it carries are root-relative — ``/features/{slug}/static/…``.
// Imported as-is, that root-relative URL would be served by the HOST, which owns
// no feature mounts; the host would have to guess which backing agent serves the
// asset, and the first-configured agent is the WRONG one whenever agents have
// heterogeneous feature sets (the selected agent may have the feature enabled
// while the first does not) (#2048). Pinning the URL to the selected agent makes
// the existing ``/api/agents/{id}/…`` proxy route it to the agent whose manifest
// actually declared the contribution — no host-side guessing, no homogeneity
// assumption.
//
// Only ``/features/…`` (feature-shipped, static_dir-backed) URLs are pinned.
// Core-bundled assets (``/js/…``) are served by the host directly and must stay
// un-prefixed. In standalone mode ``buildAgentUrl`` is a no-op (no selected
// agent), so the URL is returned unchanged and the server's own
// ``/features/{slug}/static/`` mount serves it.
function pinFeatureAssetUrl(url) {
    if (
        typeof url === 'string'
        && url.startsWith('/features/')
        && typeof API.buildAgentUrl === 'function'
    ) {
        return API.buildAgentUrl(url);
    }
    return url;
}

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

// Canonical (pre-pin) identities of assets already imported, so the same feature
// asset executes exactly once even across agent switches. In multi-agent host
// mode pinFeatureAssetUrl() yields a different per-agent URL for the same module
// (e.g. /api/agents/A/features/.../spawn.js vs …/B/…); the browser module cache
// treats those as distinct modules and would re-run the module's top-level side
// effects (slot registration, bus.on('panel:shown') handlers), causing duplicate
// loads/renders. Keying dedupe on the unprefixed `feature::asset` identity avoids
// that while the first fetch still routes through the selected agent.
const _importedModules = new Set();

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
                // injectFeatureStylesheet is idempotent via a DOM <link> check.
                injectFeatureStylesheet(pinFeatureAssetUrl(href));
            }
            for (const mod of entry.modules || []) {
                const modKey = `${entry.feature}::${mod}`;
                if (_importedModules.has(modKey)) continue;
                try {
                    await import(pinFeatureAssetUrl(mod));
                    // Mark imported only on success so a failed import can retry
                    // on the next run (e.g. after the feature is re-enabled).
                    _importedModules.add(modKey);
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

// ============================================================================
// Host-scoped UI contributions loader (#2293)
// ============================================================================
//
// The multi-agent host serves a DISTINCT host-scoped UI surface at
// ``/api/host/ui/contributions`` (parallel to the per-agent
// ``/api/ui/contributions`` above). Host-feature panels contribute here.
//
// Why a separate loader: in multi-agent host mode boot intentionally skips the
// per-agent loader until an agent is selected (that path 503s with no agent).
// Host-scoped panels are NOT agent-bound — their manifest and their static
// assets (``/host/features/{slug}/static/…``) are served by the host itself, so
// they must be loaded at host-console boot with NO agent pinning. Fetched via
// ``API.requestHost`` so the call reaches the host root (never host-agent-
// prefixed) and carries no CSRF token (this is a safe GET).
const _hostImportedModules = new Set();
let _hostInFlight = null;

export async function loadHostFeatureUIContributions() {
    if (_hostInFlight) return _hostInFlight;
    _hostInFlight = (async () => {
        let contributions = [];
        try {
            const requestHost = typeof API.requestHost === 'function'
                ? API.requestHost.bind(API)
                : API.request.bind(API);
            const data = await requestHost('/api/host/ui/contributions');
            if (data && Array.isArray(data.contributions)) {
                contributions = data.contributions;
            }
        } catch (e) {
            console.warn('[ui-ext] failed to fetch host UI contributions manifest:', e);
            return;
        }

        for (const entry of contributions) {
            if (entry.capability && !API.hasCapability(entry.capability)) continue;
            for (const href of entry.css || []) {
                // Host assets are host-served and root-relative — no agent pin.
                injectFeatureStylesheet(href);
            }
            for (const mod of entry.modules || []) {
                const modKey = `${entry.feature}::${mod}`;
                if (_hostImportedModules.has(modKey)) continue;
                try {
                    await import(mod);
                    _hostImportedModules.add(modKey);
                } catch (e) {
                    console.error(
                        `[ui-ext] failed to import host feature module ${mod} (feature ${entry.feature}):`,
                        e,
                    );
                }
            }
        }
    })().finally(() => { _hostInFlight = null; });
    return _hostInFlight;
}
