/**
 * Sovereign Console smoke — the Playwright subset pull-request CI runs (#2682).
 *
 * Proves, through a real browser, that the shipped console boots, bootstraps
 * its own API key, routes to the right agent, and renders a reply end to end.
 * Nothing here calls an LLM: the only chat turn is `!status`, a non-cognitive
 * command, and the instance has no paid provider configured.
 *
 * Launch it ONLY through `kestrel demo smoke` (CI does), which creates a fresh
 * isolated instance on a non-live port and records its origin in
 * console-smoke-instance.json. Run raw, it refuses before any network call.
 *
 *   uv run kestrel demo smoke
 */
const path = require('path');
const { test, expect } = require('@playwright/test');

const {
  loadSmokeManifest,
  assertSmokeInstance,
  assertServedAgentRoster,
  foreignOriginRequests,
} = require('./console_smoke_proof.cjs');

const BASE_URL = process.env.KESTREL_URL;
const CHECKOUT_ROOT = path.resolve(__dirname, '..', '..');
const CONSOLE_READY = 'Kestrel Sovereign Console ready';

/** @type {Record<string, any>} */
let manifest;

test.beforeAll(() => {
  manifest = loadSmokeManifest();
  assertSmokeInstance(manifest, {
    baseUrl: BASE_URL,
    dbPath: process.env.KESTREL_DB_PATH,
    checkoutRoot: CHECKOUT_ROOT,
  });
});

test('console boots, authenticates, and answers !status from the fresh agent', async ({ page }) => {
  const pageErrors = [];
  page.on('pageerror', (error) => pageErrors.push(error.message));
  const apiRequests = [];
  page.on('request', (request) => {
    if (['fetch', 'xhr', 'eventsource'].includes(request.resourceType())) {
      apiRequests.push(request.url());
    }
  });

  const pathIs = (pathname) => (response) => new URL(response.url()).pathname === pathname;
  const consoleReady = page.waitForEvent('console', {
    predicate: (message) => message.text() === CONSOLE_READY,
  });
  const authKeyResponse = page.waitForResponse(pathIs('/api/auth/key'));
  const agentsResponse = page.waitForResponse(pathIs('/api/agents'));
  // Each is awaited below; these handlers only stop an early failure in
  // one step from also reporting the others as unhandled rejections.
  for (const pending of [consoleReady, authKeyResponse, agentsResponse]) {
    pending.catch(() => {});
  }

  await test.step('the instance is healthy', async () => {
    const health = await page.request.get(`${BASE_URL}/health`);
    expect(health.status()).toBe(200);
    expect(await health.json()).toMatchObject({ status: 'ok', agent_initialized: true });
  });

  await test.step('the console loads from the isolated instance', async () => {
    const response = await page.goto(BASE_URL);
    expect(response?.status()).toBe(200);
    expect(new URL(page.url()).origin).toBe(new URL(manifest.base_url).origin);
    await expect(page.locator('#message-input')).toBeVisible();
  });

  let apiKey;
  await test.step('the console bootstraps its own API key and finishes booting', async () => {
    const response = await authKeyResponse;
    expect(response.status()).toBe(200);
    apiKey = (await response.json()).key;
    expect(apiKey).toBeTruthy();
    await consoleReady;
    expect(await page.evaluate(() => sessionStorage.getItem('kestrel_api_key'))).toBe(apiKey);
  });

  await test.step('the console routes to exactly the freshly created agent', async () => {
    const response = await agentsResponse;
    expect(response.request().headers()['x-api-key']).toBe(apiKey);
    assertServedAgentRoster(
      { ok: response.ok(), status: response.status(), data: await response.json() },
      manifest,
    );
    await expect(page.locator('#agents-pane .agent-item')).toHaveCount(1);
    const banner = page.locator('#demo-mode-banner');
    await expect(banner).toContainText('DEMO MODE');
    await expect(banner).not.toContainText('MISCONFIG');
  });

  await test.step('the Advanced toggle reveals the panel tabs', async () => {
    // The console is chat-first (#2229/#2350): the tab strip starts hidden
    // and only this toggle reveals it. A fresh browser context has no
    // persisted reveal state, so it must start collapsed.
    const advanced = page.locator('#advanced-toggle-btn');
    await expect(advanced).toHaveAttribute('aria-pressed', 'false');
    await expect(page.locator('nav .nav-tabs')).toBeHidden();
    await advanced.click();
    await expect(advanced).toHaveAttribute('aria-pressed', 'true');
    await expect(page.locator('nav .nav-tabs')).toBeVisible();
  });

  await test.step('the Identity panel shows the fresh agent DID', async () => {
    await page.locator('.nav-tab[data-panel="identity"]').click();
    await expect(page.locator('#identity-card .identity-did-text')).toHaveText(manifest.agent_did);
  });

  await test.step('!status answers through the chat without an LLM', async () => {
    await page.locator('.nav-tab[data-panel="chat"]').click();
    const input = page.locator('#message-input');
    await input.fill('!status');
    await input.press('Escape'); // close the command autocomplete
    await page.locator('#send-button').click();
    await expect(
      page.locator('.agent-message', { hasText: `Agent ID: ${manifest.agent_did}` }).first(),
    ).toBeVisible({ timeout: 30_000 });
  });

  await test.step('every API call stayed on the instance and nothing threw', async () => {
    expect(foreignOriginRequests(apiRequests, manifest.base_url)).toEqual([]);
    expect(pageErrors).toEqual([]);
  });
});
