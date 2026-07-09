// #2232: host tier-gate ("upgrade_required") rendering. Covers the API error
// enrichment (status + parsed body survive on the thrown Error) and the pure
// rendering helpers that turn the structured 403 envelope into an upsell.

import test from 'node:test';
import assert from 'node:assert/strict';

import { createApiClient } from '../../kestrel_sovereign/static/js/api_client.mjs';

function createStorage(initial = {}) {
    const store = new Map(Object.entries(initial));
    return {
        getItem(key) { return store.has(key) ? store.get(key) : null; },
        setItem(key, value) { store.set(key, String(value)); },
        removeItem(key) { store.delete(key); },
    };
}

function jsonResponse(status, body) {
    return {
        ok: status >= 200 && status < 300,
        status,
        statusText: body?.detail || `HTTP ${status}`,
        async json() { return body; },
    };
}

function createLogger() {
    return { log() {}, warn() {}, error() {} };
}

const UPGRADE_BODY = {
    code: 'upgrade_required',
    action: 'session',
    required_tier: 'premium',
    current_tier: 'free',
    message: 'Session approvals need Premium.',
    upgrade_href: 'https://frinz.example/upgrade',
};

// --- API error enrichment -------------------------------------------------

test('performRequest attaches status + parsed body to the thrown error', async () => {
    let call = 0;
    const fetchFn = async () => {
        call += 1;
        return jsonResponse(403, UPGRADE_BODY);
    };
    const client = createApiClient({
        fetchFn,
        sessionStorage: createStorage({ kestrel_api_key: 'k' }),
        location: { href: 'http://localhost/', pathname: '/', search: '' },
        logger: createLogger(),
        authProvider: {
            async ensureAuthenticated() {},
            async applyAuth(h) { return h; },
            async onUnauthorized() { return 'failed'; },
        },
    });

    await assert.rejects(
        () => client.request('/api/security/approve', { method: 'POST', body: '{}' }),
        (err) => {
            assert.equal(err.status, 403);
            assert.ok(err.body, 'error carries the parsed body');
            assert.equal(err.body.code, 'upgrade_required');
            assert.equal(err.body.upgrade_href, 'https://frinz.example/upgrade');
            // message still flattens to `detail`-or-fallback for legacy callers
            assert.match(err.message, /HTTP 403|upgrade/i);
            return true;
        },
    );
    assert.equal(call, 1);
});

// --- Pure rendering helpers ----------------------------------------------
// upgrade-prompt.js imports api.js at module load, which builds an API client
// from globals; seed them (and a proactive gate map) before importing.

async function loadHelper() {
    globalThis.sessionStorage = createStorage();
    globalThis.location = { href: 'http://localhost/', pathname: '/', search: '' };
    globalThis.fetch = async () => jsonResponse(200, {});
    globalThis.KESTREL_UI_CONFIG = {};
    return import('../../kestrel_sovereign/static/js/upgrade-prompt.js');
}

// api.js is a module singleton, so getApprovalScopeGates reads whatever
// API.getCapabilities() returns at call time. Stub it to exercise both the
// gated-host and standalone paths without depending on import order.
async function withCapabilities(capsMap, fn) {
    globalThis.sessionStorage = createStorage();
    globalThis.location = { href: 'http://localhost/', pathname: '/', search: '' };
    globalThis.fetch = async () => jsonResponse(200, {});
    globalThis.KESTREL_UI_CONFIG = {};
    const { default: API } = await import('../../kestrel_sovereign/static/js/api.js');
    const original = API.getCapabilities;
    API.getCapabilities = () => capsMap;
    try {
        return await fn();
    } finally {
        API.getCapabilities = original;
    }
}

test('extractUpgradeRequired recognizes the tier-gate envelope', async () => {
    const { extractUpgradeRequired } = await loadHelper();
    const err = Object.assign(new Error('HTTP 403'), { status: 403, body: UPGRADE_BODY });
    const upgrade = extractUpgradeRequired(err);
    assert.ok(upgrade);
    assert.equal(upgrade.requiredTier, 'premium');
    assert.equal(upgrade.currentTier, 'free');
    assert.equal(upgrade.message, 'Session approvals need Premium.');
    assert.equal(upgrade.upgradeHref, 'https://frinz.example/upgrade');
});

test('extractUpgradeRequired returns null for unrelated errors', async () => {
    const { extractUpgradeRequired } = await loadHelper();
    assert.equal(extractUpgradeRequired(null), null);
    assert.equal(extractUpgradeRequired(new Error('boom')), null);
    // wrong status
    assert.equal(extractUpgradeRequired(Object.assign(new Error('x'), { status: 500, body: UPGRADE_BODY })), null);
    // wrong code
    assert.equal(extractUpgradeRequired(Object.assign(new Error('x'), { status: 403, body: { code: 'nope' } })), null);
});

test('getApprovalScopeGates reads the host proactive gate map', async () => {
    await withCapabilities({ approvalScopes: { session: 'premium', always: 'sovereign' } }, async () => {
        const { getApprovalScopeGates } = await import('../../kestrel_sovereign/static/js/upgrade-prompt.js');
        assert.deepEqual(getApprovalScopeGates(), { session: 'premium', always: 'sovereign' });
    });
});

test('getApprovalScopeGates is empty for a standalone console (no gate map)', async () => {
    await withCapabilities({}, async () => {
        const { getApprovalScopeGates } = await import('../../kestrel_sovereign/static/js/upgrade-prompt.js');
        assert.deepEqual(getApprovalScopeGates(), {});
    });
});

test('upgradeBannerHtml renders the message and a link to upgrade_href', async () => {
    const { extractUpgradeRequired, upgradeBannerHtml } = await loadHelper();
    const upgrade = extractUpgradeRequired(Object.assign(new Error(), { status: 403, body: UPGRADE_BODY }));
    const html = upgradeBannerHtml(upgrade);
    assert.match(html, /Session approvals need Premium\./);
    assert.match(html, /href="https:\/\/frinz\.example\/upgrade"/);
    assert.match(html, /Upgrade to premium/);
});

test('upgradeBannerHtml escapes untrusted text', async () => {
    const { upgradeBannerHtml } = await loadHelper();
    const html = upgradeBannerHtml({ message: '<img src=x onerror=alert(1)>', requiredTier: 'p', upgradeHref: 'javascript:evil' });
    assert.doesNotMatch(html, /<img src=x/);
    assert.match(html, /&lt;img/);
});

test('tierBadgeHtml renders a badge with the tier text', async () => {
    const { tierBadgeHtml } = await loadHelper();
    assert.match(tierBadgeHtml('premium'), /premium/);
    assert.equal(tierBadgeHtml(null), '');
});

test('unsafe upgrade_href schemes are stripped before rendering (codex P2)', async () => {
    const { extractUpgradeRequired, upgradeBannerHtml, upgradeToastHtml, sanitizeUpgradeHref } =
        await import('../../kestrel_sovereign/static/js/upgrade-prompt.js');

    for (const bad of ['javascript:alert(1)', 'data:text/html,x', 'vbscript:x', '//evil.example/upgrade', '  javascript:alert(1)']) {
        const upgrade = extractUpgradeRequired({
            status: 403,
            body: { code: 'upgrade_required', message: 'Needs premium.', upgrade_href: bad },
        });
        assert.equal(upgrade.upgradeHref, null, `href must be stripped: ${bad}`);
        assert.ok(!upgradeBannerHtml(upgrade).includes('<a '), 'banner renders no link');
        assert.ok(!upgradeToastHtml(upgrade).includes('<a '), 'toast renders no link');
        // Message still shows — the nudge survives, only the link is dropped.
        assert.ok(upgradeBannerHtml(upgrade).includes('Needs premium.'));
    }

    for (const good of ['/upgrade.html', 'https://frinz.ai/upgrade', 'http://localhost:7779/upgrade.html']) {
        assert.equal(sanitizeUpgradeHref(good), good, `href must survive: ${good}`);
    }
});
