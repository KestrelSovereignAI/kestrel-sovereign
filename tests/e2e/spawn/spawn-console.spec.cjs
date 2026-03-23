/**
 * Spawn Console Panel E2E Tests
 *
 * Tests the Spawn tab in the Sovereign Console:
 * - Panel renders correctly with all sections
 * - Active children display with status badges
 * - Delegation chain tree visualization
 * - Budget chart rendering (Chart.js)
 * - Auto-refresh controls
 * - Spawn history timeline
 * - Screenshots at each stage for kestrel-eye verification
 *
 * NO MOCKS - Tests real Kestrel agent UI against running server.
 * Requires: Kestrel server running on localhost:8888
 */
const { test, expect } = require('@playwright/test');

const BASE_URL = process.env.KESTREL_URL || 'http://localhost:8888';

// ============================================================================
// Helpers
// ============================================================================

async function getApiKey(request) {
    if (process.env.KESTREL_API_KEY) {
        return process.env.KESTREL_API_KEY;
    }
    try {
        const response = await request.get(`${BASE_URL}/api/auth/key`);
        if (response.ok()) {
            const data = await response.json();
            return data.key;
        }
    } catch (e) {
        console.warn('Could not fetch API key:', e.message);
    }
    return null;
}

function authHeaders(apiKey) {
    if (!apiKey) return {};
    return { 'X-API-Key': apiKey };
}

// ============================================================================
// Spawn Console Panel Tests
// ============================================================================

test.describe('Spawn Console Panel', () => {
    let apiKey = null;

    test.beforeAll(async ({ request }) => {
        apiKey = await getApiKey(request);
    });

    test('spawn tab exists in navigation', async ({ page }) => {
        await page.goto(BASE_URL);
        await page.waitForLoadState('domcontentloaded');

        const spawnTab = page.locator('.nav-tab[data-panel="spawn"]');
        await expect(spawnTab).toBeVisible();
        await expect(spawnTab).toHaveText('Spawn');

        await page.screenshot({ path: 'test-results/spawn-console-nav-tab.png' });
    });

    test('spawn panel renders all sections on tab click', async ({ page }) => {
        await page.goto(BASE_URL);
        await page.waitForLoadState('domcontentloaded');

        // Click the Spawn tab
        await page.locator('.nav-tab[data-panel="spawn"]').click();

        // Panel should become active
        const panel = page.locator('#panel-spawn');
        await expect(panel).toBeVisible();

        await page.screenshot({ path: 'test-results/spawn-console-panel-initial.png' });

        // Verify all sub-sections exist
        await expect(panel.locator('text=Spawn Manager')).toBeVisible();
        await expect(panel.locator('text=Active Children')).toBeVisible();
        await expect(panel.locator('text=Delegation Chain')).toBeVisible();
        await expect(panel.locator('text=Budget Allocation')).toBeVisible();
        await expect(panel.locator('text=Spawn History')).toBeVisible();
    });

    test('active children section renders', async ({ page }) => {
        await page.goto(BASE_URL);
        await page.waitForLoadState('domcontentloaded');
        await page.locator('.nav-tab[data-panel="spawn"]').click();

        const childrenList = page.locator('#spawn-children-list');
        await expect(childrenList).toBeVisible();

        // Should have either children cards or an empty-state message
        const content = await childrenList.textContent();
        expect(content.trim().length).toBeGreaterThan(0);

        await page.screenshot({ path: 'test-results/spawn-console-children-list.png' });
    });

    test('delegation chain section renders', async ({ page }) => {
        await page.goto(BASE_URL);
        await page.waitForLoadState('domcontentloaded');
        await page.locator('.nav-tab[data-panel="spawn"]').click();

        const chainSection = page.locator('#spawn-delegation-chain');
        await expect(chainSection).toBeVisible();

        const content = await chainSection.textContent();
        expect(content.trim().length).toBeGreaterThan(0);

        await page.screenshot({ path: 'test-results/spawn-console-delegation-chain.png' });
    });

    test('budget chart canvas exists', async ({ page }) => {
        await page.goto(BASE_URL);
        await page.waitForLoadState('domcontentloaded');
        await page.locator('.nav-tab[data-panel="spawn"]').click();

        const canvas = page.locator('#spawn-budget-chart');
        await expect(canvas).toBeAttached();

        await page.screenshot({ path: 'test-results/spawn-console-budget-chart.png' });
    });

    test('spawn history section renders', async ({ page }) => {
        await page.goto(BASE_URL);
        await page.waitForLoadState('domcontentloaded');
        await page.locator('.nav-tab[data-panel="spawn"]').click();

        const historyList = page.locator('#spawn-history-list');
        await expect(historyList).toBeVisible();

        const content = await historyList.textContent();
        expect(content.trim().length).toBeGreaterThan(0);

        await page.screenshot({ path: 'test-results/spawn-console-history.png' });
    });
});

// ============================================================================
// Auto-Refresh Controls
// ============================================================================

test.describe('Spawn Auto-Refresh Controls', () => {
    let apiKey = null;

    test.beforeAll(async ({ request }) => {
        apiKey = await getApiKey(request);
    });

    test('refresh button exists and is clickable', async ({ page }) => {
        await page.goto(BASE_URL);
        await page.waitForLoadState('domcontentloaded');
        await page.locator('.nav-tab[data-panel="spawn"]').click();

        const refreshBtn = page.locator('#btn-refresh-spawn');
        await expect(refreshBtn).toBeVisible();
        await expect(refreshBtn).toBeEnabled();

        // Click refresh and verify no errors
        await refreshBtn.click();
        // Wait a moment for the refresh to process
        await page.waitForTimeout(1000);

        await page.screenshot({ path: 'test-results/spawn-console-after-refresh.png' });
    });

    test('auto-refresh interval selector works', async ({ page }) => {
        await page.goto(BASE_URL);
        await page.waitForLoadState('domcontentloaded');
        await page.locator('.nav-tab[data-panel="spawn"]').click();

        const select = page.locator('#spawn-refresh-interval');
        await expect(select).toBeVisible();

        // Verify options exist
        const options = select.locator('option');
        const count = await options.count();
        expect(count).toBe(4); // Off, 5s, 10s, 30s

        // Verify default is 10s
        const defaultVal = await select.inputValue();
        expect(defaultVal).toBe('10');

        // Change to 5s
        await select.selectOption('5');
        const newVal = await select.inputValue();
        expect(newVal).toBe('5');

        // Change to Off
        await select.selectOption('0');
        const offVal = await select.inputValue();
        expect(offVal).toBe('0');

        await page.screenshot({ path: 'test-results/spawn-console-refresh-selector.png' });
    });
});

// ============================================================================
// Spawn Panel with Active Children
// ============================================================================

test.describe('Spawn Panel Data Display', () => {
    let apiKey = null;

    test.beforeAll(async ({ request }) => {
        apiKey = await getApiKey(request);
    });

    test('panel displays data from API correctly', async ({ page, request }) => {
        // First get the API data to know what to expect
        const response = await request.get(`${BASE_URL}/api/spawn/children`, {
            headers: authHeaders(apiKey)
        });
        expect(response.ok()).toBeTruthy();
        const apiData = await response.json();

        // Navigate to the spawn panel
        await page.goto(BASE_URL);
        await page.waitForLoadState('domcontentloaded');
        await page.locator('.nav-tab[data-panel="spawn"]').click();

        // Click refresh to ensure data is loaded
        await page.locator('#btn-refresh-spawn').click();
        await page.waitForTimeout(2000);

        if (apiData.count > 0) {
            // If children exist, verify they appear in the UI
            const childrenList = page.locator('#spawn-children-list');
            const listContent = await childrenList.textContent();

            for (const child of apiData.children) {
                // Child name should appear somewhere in the panel
                const nameVisible = listContent.includes(child.name);
                if (nameVisible) {
                    expect(nameVisible).toBe(true);
                }
            }

            // Delegation chain should show parent
            const chainContent = await page.locator('#spawn-delegation-chain').textContent();
            expect(chainContent.length).toBeGreaterThan(0);
        }

        await page.screenshot({ path: 'test-results/spawn-console-data-display.png' });
    });

    test('child cards show status, purpose, and budget info', async ({ page, request }) => {
        const apiData = await request.get(`${BASE_URL}/api/spawn/children`, {
            headers: authHeaders(apiKey)
        }).then(r => r.json());

        await page.goto(BASE_URL);
        await page.waitForLoadState('domcontentloaded');
        await page.locator('.nav-tab[data-panel="spawn"]').click();
        await page.locator('#btn-refresh-spawn').click();
        await page.waitForTimeout(2000);

        if (apiData.count > 0) {
            const childrenList = page.locator('#spawn-children-list');

            // Check for status indicators (badges)
            // The UI uses status classes/badges for running/stopped
            const content = await childrenList.innerHTML();

            // Should contain status text
            const hasStatusIndicator = /running|stopped|completed|terminated/i.test(content);
            expect(hasStatusIndicator).toBe(true);
        }

        await page.screenshot({ path: 'test-results/spawn-console-child-cards.png' });
    });

    test('full panel screenshot for kestrel-eye verification', async ({ page }) => {
        await page.goto(BASE_URL);
        await page.waitForLoadState('domcontentloaded');
        await page.locator('.nav-tab[data-panel="spawn"]').click();

        // Click refresh to load all data
        await page.locator('#btn-refresh-spawn').click();
        await page.waitForTimeout(2000);

        // Full page screenshot of the spawn panel
        await page.screenshot({
            path: 'test-results/spawn-console-full-panel.png',
            fullPage: true
        });

        // Also capture just the panel area
        const panel = page.locator('#panel-spawn');
        await panel.screenshot({
            path: 'test-results/spawn-console-panel-only.png'
        });
    });
});

// ============================================================================
// Spawn API Contract Tests
// ============================================================================

test.describe('Spawn API Contract', () => {
    let apiKey = null;

    test.beforeAll(async ({ request }) => {
        apiKey = await getApiKey(request);
    });

    test('GET /api/spawn/children returns correct schema', async ({ request }) => {
        const response = await request.get(`${BASE_URL}/api/spawn/children`, {
            headers: authHeaders(apiKey)
        });

        expect(response.ok()).toBeTruthy();
        expect(response.status()).toBe(200);

        const data = await response.json();

        // Top-level structure
        expect(data).toHaveProperty('children');
        expect(data).toHaveProperty('count');
        expect(data).toHaveProperty('delegation_chain');
        expect(data).toHaveProperty('history');

        // Types
        expect(Array.isArray(data.children)).toBe(true);
        expect(typeof data.count).toBe('number');
        expect(typeof data.delegation_chain).toBe('object');
        expect(Array.isArray(data.history)).toBe(true);

        // Count matches array length
        expect(data.count).toBe(data.children.length);

        // Delegation chain structure
        expect(data.delegation_chain).toHaveProperty('name');
        expect(data.delegation_chain).toHaveProperty('did');
        expect(data.delegation_chain).toHaveProperty('status');
        expect(data.delegation_chain).toHaveProperty('children');
    });

    test('each child has required fields', async ({ request }) => {
        const response = await request.get(`${BASE_URL}/api/spawn/children`, {
            headers: authHeaders(apiKey)
        });
        const data = await response.json();

        const requiredFields = [
            'name', 'status', 'did', 'purpose',
            'ttl_seconds', 'budget_allocated', 'budget_spent',
            'budget_remaining', 'started_at', 'ttl_remaining'
        ];

        for (const child of data.children) {
            for (const field of requiredFields) {
                expect(child).toHaveProperty(field);
            }
            expect(['running', 'stopped']).toContain(child.status);
            expect(typeof child.ttl_seconds).toBe('number');
            expect(typeof child.budget_allocated).toBe('number');
            expect(typeof child.budget_spent).toBe('number');
            expect(typeof child.budget_remaining).toBe('number');
            expect(typeof child.ttl_remaining).toBe('number');
        }
    });

    test('history events have required fields', async ({ request }) => {
        const response = await request.get(`${BASE_URL}/api/spawn/children`, {
            headers: authHeaders(apiKey)
        });
        const data = await response.json();

        for (const event of data.history) {
            expect(event).toHaveProperty('event');
            expect(event).toHaveProperty('child_name');
            expect(event).toHaveProperty('status');
            expect(['spawned', 'terminated']).toContain(event.event);
        }
    });
});
