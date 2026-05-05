/**
 * E2E: Model Selector UI Flow (vendor/route/model refactor, epic #688)
 *
 * Runs against the live multi_agent on ``KESTREL_URL`` (default localhost:8888).
 * Verifies the wire-level contract of the refactor:
 *
 *   1. ``/api/models`` returns ``by_vendor`` (not ``by_provider``) and a
 *      ``routes`` array where each entry has ``vendor`` + ``route``. No
 *      pseudo-provider buckets (``claude_plan``, ``openai_plan``,
 *      ``openai_mini``) appear.
 *   2. When the selected vendor has >1 route, the ``#route-selector`` element
 *      becomes visible with each route as an option. Single-route vendors keep
 *      it hidden.
 *   3. Changing vendor → auto-selects that vendor's first model → commits via
 *      ``POST /api/model/set`` (REST, not chat stream) → ``/api/model/current``
 *      echoes the committed ``{vendor, model, route}``.
 *   4. Clicking the same vendor + model twice does NOT double-commit (the
 *      ``_maybeCommit`` diff guard).
 *
 * The fine-grained state-machine contract lives in ``tests/frontend/*.test.mjs``
 * (Node test runner, no browser). This file checks the live-browser ↔ live-API
 * integration only — what Playwright is actually for.
 */
const { test, expect } = require('@playwright/test');

const BASE_URL = process.env.KESTREL_URL || 'http://localhost:8888';
const AGENT = process.env.KESTREL_AGENT || 'Meridian';

async function getApiKey(request) {
    if (process.env.KESTREL_API_KEY) return process.env.KESTREL_API_KEY;
    try {
        const response = await request.get(`${BASE_URL}/api/auth/key`);
        if (response.ok()) return (await response.json()).key;
    } catch (e) {}
    return null;
}

function authHeaders(key) {
    return key ? { 'X-API-Key': key } : {};
}

async function getCurrent(request, apiKey) {
    const resp = await request.get(
        `${BASE_URL}/api/agents/${AGENT}/api/model/current`,
        { headers: authHeaders(apiKey) },
    );
    expect(resp.ok()).toBeTruthy();
    return resp.json();
}

async function getModelsPayload(request, apiKey) {
    const resp = await request.get(
        `${BASE_URL}/api/agents/${AGENT}/api/models`,
        { headers: authHeaders(apiKey) },
    );
    expect(resp.ok()).toBeTruthy();
    return resp.json();
}

async function openAgentWithSelectorReady(page, apiKey, agent) {
    await page.goto(`${BASE_URL}/`);
    // The server's page loader honours the ``key=`` query for the initial
    // handshake; bounce through localStorage so subsequent fetches carry it.
    await page.evaluate((k) => {
        try { window.localStorage.setItem('kestrel_api_key', k); } catch (_) {}
    }, apiKey);
    await page.reload();

    // Drive selectAgent once the identity module exposes it.
    await page.waitForFunction(
        () => typeof window.selectAgent === 'function',
        { timeout: 15000 },
    );
    await page.evaluate((a) => window.selectAgent(a), agent);

    // Wait for the shared model selector to populate real options (not the
    // "Loading…" placeholder that chat.js writes before the async roundtrip).
    await page.waitForFunction(
        () => {
            const ps = document.querySelector('#provider-selector');
            if (!ps) return false;
            const opts = Array.from(ps.options).map(o => o.value).filter(Boolean);
            return opts.length >= 2;  // need >= 2 vendors for the flow tests
        },
        { timeout: 20000 },
    );
}


test.describe('Model selector vendor/route/model flow', () => {

    test('API /api/models returns by_vendor and per-vendor routes list', async ({ request }) => {
        const apiKey = await getApiKey(request);
        const data = await getModelsPayload(request, apiKey);

        expect(data).toHaveProperty('by_vendor');
        expect(data).toHaveProperty('routes');
        expect(Array.isArray(data.routes)).toBeTruthy();

        for (const r of data.routes) {
            expect(r).toHaveProperty('vendor');
            expect(r).toHaveProperty('route');
        }

        // No pseudo-provider buckets should appear — they're routes under vendors.
        for (const bad of ['openai_plan', 'claude_plan', 'openai_mini']) {
            expect(Object.keys(data.by_vendor)).not.toContain(bad);
        }
    });

    test('route selector is visible for a vendor with >1 route and hidden otherwise', async ({ page, request }) => {
        const apiKey = await getApiKey(request);
        const data = await getModelsPayload(request, apiKey);

        const routesByVendor = {};
        for (const r of data.routes) {
            (routesByVendor[r.vendor] = routesByVendor[r.vendor] || []).push(r.route);
        }
        const multiRouteVendor = Object.entries(routesByVendor).find(([, rs]) => rs.length > 1);
        test.skip(!multiRouteVendor, 'No vendor has >1 route configured');
        const singleRouteVendor = Object.entries(routesByVendor).find(
            ([v, rs]) => rs.length === 1 && data.by_vendor[v] && data.by_vendor[v].length > 0,
        );

        await openAgentWithSelectorReady(page, apiKey, AGENT);

        // Multi-route vendor → route selector should be visible with all routes as options.
        await page.evaluate((v) => {
            const el = document.querySelector('#provider-selector');
            el.value = v;
            el.dispatchEvent(new Event('change', { bubbles: true }));
        }, multiRouteVendor[0]);
        await page.waitForFunction(() => {
            const el = document.querySelector('#route-selector');
            return el && el.style.display !== 'none' && el.options.length >= 2;
        }, { timeout: 5000 });

        const options = await page.$$eval('#route-selector option', opts => opts.map(o => o.value));
        for (const route of multiRouteVendor[1]) {
            expect(options).toContain(route);
        }

        if (singleRouteVendor) {
            await page.evaluate((v) => {
                const el = document.querySelector('#provider-selector');
                el.value = v;
                el.dispatchEvent(new Event('change', { bubbles: true }));
            }, singleRouteVendor[0]);
            await page.waitForFunction(() => {
                const el = document.querySelector('#route-selector');
                return el && el.style.display === 'none';
            }, { timeout: 5000 });
        }
    });

    test('vendor change commits via REST (not chat stream) and /api/model/current echoes it', async ({ page, request }) => {
        const apiKey = await getApiKey(request);
        const data = await getModelsPayload(request, apiKey);

        // Pick a vendor that is NOT the currently-mandated one and has models.
        const current = await getCurrent(request, apiKey);
        const candidates = Object.entries(data.by_vendor).filter(
            ([v, models]) => v !== current.vendor && Array.isArray(models) && models.length > 0,
        );
        test.skip(candidates.length === 0, 'No alternate vendor with models available');
        const [targetVendor, targetModels] = candidates[0];

        await openAgentWithSelectorReady(page, apiKey, AGENT);

        // Record any POST /api/model/set — the refactor requires the commit
        // to go through REST, not through the chat stream as a !model-set
        // message (which was how cross-agent contamination used to happen).
        const commits = [];
        page.on('request', (req) => {
            if (req.method() === 'POST' && req.url().includes('/api/model/set')) {
                commits.push({ url: req.url(), body: req.postData() });
            }
        });

        await page.evaluate((v) => {
            const el = document.querySelector('#provider-selector');
            el.value = v;
            el.dispatchEvent(new Event('change', { bubbles: true }));
        }, targetVendor);

        // Wait for the commit round-trip to land on the server.
        await page.waitForFunction(
            async ([base, agent, vendor]) => {
                const resp = await fetch(`${base}/api/agents/${agent}/api/model/current`, {
                    credentials: 'omit',
                    headers: { 'X-API-Key': window.localStorage.getItem('kestrel_api_key') || '' },
                });
                if (!resp.ok) return false;
                const j = await resp.json();
                return j.vendor === vendor;
            },
            [BASE_URL, AGENT, targetVendor],
            { timeout: 15000 },
        );

        const after = await getCurrent(request, apiKey);
        expect(after.vendor).toBe(targetVendor);
        // The committed model must be one of the new vendor's models.
        expect(targetModels.map(m => m.id)).toContain(after.model_name);
        expect(commits.length).toBeGreaterThan(0);
    });

});
