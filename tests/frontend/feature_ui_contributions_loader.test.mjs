// #2048: the feature UI-contributions loader must run AGAIN after agent
// selection (multi-agent host mode) and on a runtime capability flip — not only
// once at boot. In multi-agent mode the boot call hits the host's un-prefixed
// /api/ui/contributions with no active agent and 503s, so a feature-owned panel
// module (the extracted Spawn panel) is never imported and its tab never
// appears. These tests exercise the loader's behavior (boot-503 → re-run on
// select; disabled-at-boot → re-run on runtime enable) plus the identity.js /
// app.js wiring that drives those re-runs.

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { JSDOM } from 'jsdom';

const dom = new JSDOM('<!doctype html><html><head></head><body></body></html>', {
    url: 'http://localhost/',
});
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.Node = dom.window.Node;
// api.js constructs the singleton API client at import time; it requires these
// globals to be present (they're read as createApiClient defaults). The loader
// only ever calls API.request / API.hasCapability, both stubbed below, so the
// real fetch/location/sessionStorage are never exercised — they just need to
// exist so the constructor doesn't throw.
globalThis.location = dom.window.location;
globalThis.sessionStorage = dom.window.sessionStorage;
globalThis.fetch = async () => { throw new Error('fetch should be stubbed via API.request'); };
// jsdom does not always implement CSS.escape; the loader uses it to dedupe
// injected <link> tags. A passthrough is sufficient for the test.
if (!globalThis.CSS || typeof globalThis.CSS.escape !== 'function') {
    globalThis.CSS = { escape: (s) => String(s).replace(/[^a-zA-Z0-9_-]/g, '\\$&') };
}

const API = (await import('../../kestrel_sovereign/static/js/api.js')).default;
const { loadFeatureUIContributions } = await import(
    '../../kestrel_sovereign/static/js/ui-ext/feature-loader.js'
);

// Each enabled module the manifest declares is a data: URL whose side effect is
// to push a tag onto globalThis.__featLoaded — a stand-in for spawn.js calling
// registerPanel(). data: imports are cached by URL, so every test uses a unique
// tag to keep imports observable across cases.
function loaderModule(tag) {
    const body = `globalThis.__featLoaded=(globalThis.__featLoaded||[]);globalThis.__featLoaded.push(${JSON.stringify(tag)});`;
    return 'data:text/javascript,' + encodeURIComponent(body);
}

function installApiStub({ manifestFor, caps = {} }) {
    API.request = async (path) => {
        assert.equal(path, '/api/ui/contributions');
        return manifestFor();
    };
    API.hasCapability = (cap) => caps[cap] !== false;
}

test.beforeEach(() => {
    globalThis.__featLoaded = [];
    document.head.innerHTML = '';
});

test('multi-agent boot 503 imports nothing; re-run after agent selection imports the panel module', async () => {
    // Simulate boot in multi-agent host mode: no active agent → 503.
    let hasAgent = false;
    installApiStub({
        manifestFor: () => {
            if (!hasAgent) {
                const err = new Error('Agent not initialized.');
                err.status = 503;
                throw err;
            }
            return {
                contributions: [
                    { feature: 'spawnfeature', capability: 'spawn', modules: [loaderModule('spawn-after-select')], css: [] },
                ],
            };
        },
        caps: { spawn: true },
    });

    // Boot call (app.js): 503 swallowed, nothing imported — the Spawn tab would
    // never appear if this were the only call.
    await loadFeatureUIContributions();
    assert.deepEqual(globalThis.__featLoaded, [], 'boot-time 503 must import no feature modules');

    // selectAgent() pins routing; the re-run now resolves and imports the module.
    hasAgent = true;
    await loadFeatureUIContributions();
    assert.deepEqual(
        globalThis.__featLoaded,
        ['spawn-after-select'],
        'after agent selection the Spawn panel module must be imported exactly once',
    );
});

test('feature disabled at boot then enabled at runtime mounts without reload', async () => {
    // Feature is enabled server-side here; gating is what flips. Before the
    // runtime enable the capability is off, so the loader skips the import; after
    // it flips on, the same manifest now imports the module.
    const caps = { spawn: false };
    installApiStub({
        manifestFor: () => ({
            contributions: [
                { feature: 'spawnfeature', capability: 'spawn', modules: [loaderModule('spawn-runtime-enable')], css: [] },
            ],
        }),
        caps,
    });

    await loadFeatureUIContributions();
    assert.deepEqual(globalThis.__featLoaded, [], 'a capability-gated-off module must not import');

    // Runtime enable (onCapabilitiesChanged → loadFeatureUIContributions).
    caps.spawn = true;
    await loadFeatureUIContributions();
    assert.deepEqual(
        globalThis.__featLoaded,
        ['spawn-runtime-enable'],
        'enabling the feature at runtime must import + mount its module with no reload',
    );
});

test('multi-agent host mode pins feature-static URLs to the selected agent; leaves core-bundled /js/ URLs alone (#2048)', async () => {
    // The manifest carries root-relative feature-static URLs (/features/…) and a
    // core-bundled URL (/js/…). In host mode the loader must pin ONLY the feature
    // assets to the selected agent so the /api/agents/{id} proxy reaches the agent
    // that declared them; /js/ assets are host-served and must stay un-prefixed.
    const buildCalls = [];
    const originalBuild = API.buildAgentUrl;
    API.buildAgentUrl = (url) => {
        buildCalls.push(url);
        return `/api/agents/agentB${url}`;
    };
    try {
        installApiStub({
            manifestFor: () => ({
                contributions: [
                    {
                        feature: 'spawnfeature',
                        capability: 'spawn',
                        modules: ['/features/spawnfeature/static/spawn.js', '/js/voice/boot.js'],
                        css: ['/features/spawnfeature/static/spawn.css'],
                    },
                ],
            }),
            caps: { spawn: true },
        });
        // import() of /api/agents/... would 404 in jsdom; the loader isolates a
        // failed import, so we only need to observe which URLs were pinned.
        await loadFeatureUIContributions();

        // The feature module + css were routed through buildAgentUrl; the bundled
        // /js/ module was NOT.
        assert.ok(
            buildCalls.includes('/features/spawnfeature/static/spawn.js'),
            'feature-static module must be pinned to the selected agent',
        );
        assert.ok(
            buildCalls.includes('/features/spawnfeature/static/spawn.css'),
            'feature-static stylesheet must be pinned to the selected agent',
        );
        assert.ok(
            !buildCalls.includes('/js/voice/boot.js'),
            'core-bundled /js/ asset must NOT be agent-pinned (host serves it directly)',
        );

        // The injected stylesheet href is the agent-pinned URL (raw attribute,
        // not jsdom's document-resolved absolute form).
        const link = document.head.querySelector('link[data-ui-ext-css]');
        assert.ok(link, 'feature stylesheet must be injected');
        assert.equal(
            link.getAttribute('href'),
            '/api/agents/agentB/features/spawnfeature/static/spawn.css',
        );
    } finally {
        API.buildAgentUrl = originalBuild;
    }
});

test('injects a feature stylesheet once even across repeated loads', async () => {
    installApiStub({
        manifestFor: () => ({
            contributions: [
                { feature: 'spawnfeature', capability: 'spawn', modules: [], css: ['/features/spawnfeature/static/spawn.css'] },
            ],
        }),
        caps: { spawn: true },
    });

    await loadFeatureUIContributions();
    await loadFeatureUIContributions();
    const links = document.head.querySelectorAll('link[data-ui-ext-css]');
    assert.equal(links.length, 1, 'stylesheet must be injected exactly once across re-runs');
});

test('concurrent invocations are coalesced onto a single fetch', async () => {
    let fetches = 0;
    API.request = async () => {
        fetches += 1;
        return { contributions: [{ feature: 'f', capability: 'spawn', modules: [loaderModule('coalesced')], css: [] }] };
    };
    API.hasCapability = () => true;

    await Promise.all([loadFeatureUIContributions(), loadFeatureUIContributions()]);
    assert.equal(fetches, 1, 'overlapping loads must share one in-flight manifest fetch');
});

// --- wiring assertions: the re-runs are actually invoked ---

const identitySrc = fs.readFileSync(
    new URL('../../kestrel_sovereign/static/js/identity.js', import.meta.url),
    'utf8',
);
const appSrc = fs.readFileSync(
    new URL('../../kestrel_sovereign/static/js/app.js', import.meta.url),
    'utf8',
);

test('identity.js imports the shared loader', () => {
    assert.match(
        identitySrc,
        /import \{ loadFeatureUIContributions \} from '\.\/ui-ext\/feature-loader\.js'/,
    );
});

test('selectAgent re-runs the loader after refreshCapabilities (multi-agent path)', () => {
    const selectIdx = identitySrc.indexOf('window.selectAgent = async function');
    assert.notEqual(selectIdx, -1, 'selectAgent must exist');
    const tail = identitySrc.slice(selectIdx);
    const refreshIdx = tail.indexOf('await API.refreshCapabilities();');
    const loadIdx = tail.indexOf('await loadFeatureUIContributions();');
    assert.ok(refreshIdx !== -1, 'selectAgent must refresh capabilities');
    assert.ok(loadIdx !== -1, 'selectAgent must re-run the feature loader');
    assert.ok(
        refreshIdx < loadIdx,
        'the loader re-run must follow refreshCapabilities so routing + caps are set first',
    );
});

test('capabilities:changed re-runs the loader before re-gating the nav (runtime-enable path)', () => {
    assert.match(
        identitySrc,
        /globalThis\.addEventListener\('capabilities:changed', onCapabilitiesChanged\)/,
        'capabilities:changed must be handled by onCapabilitiesChanged',
    );
    const fnIdx = identitySrc.indexOf('async function onCapabilitiesChanged()');
    assert.notEqual(fnIdx, -1, 'onCapabilitiesChanged must exist');
    const body = identitySrc.slice(fnIdx, fnIdx + 400);
    const loadIdx = body.indexOf('await loadFeatureUIContributions();');
    const reconcileIdx = body.indexOf('reconcileNavigationCapabilities();');
    assert.ok(loadIdx !== -1, 'onCapabilitiesChanged must re-run the loader');
    assert.ok(reconcileIdx !== -1, 'onCapabilitiesChanged must re-gate the nav');
    assert.ok(loadIdx < reconcileIdx, 'import the newly-enabled module BEFORE re-gating');
});

test('app.js imports the shared loader rather than defining it inline', () => {
    assert.match(
        appSrc,
        /import \{ loadFeatureUIContributions \} from '\.\/ui-ext\/feature-loader\.js'/,
    );
    assert.doesNotMatch(
        appSrc,
        /async function loadFeatureUIContributions/,
        'the loader definition must live in the shared module, not app.js',
    );
});
