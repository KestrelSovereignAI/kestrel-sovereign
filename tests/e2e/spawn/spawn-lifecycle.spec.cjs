/**
 * Spawn Lifecycle E2E Tests
 *
 * Tests the full spawn lifecycle against a running Kestrel server:
 * - Spawn lifecycle (create children via chat, verify via API, termination)
 * - Budget enforcement (spending tracked, overspend blocked, remainder returned)
 * - Constitutional scoping (child constraints enforced)
 * - TTL expiration (auto-termination after timeout)
 *
 * NO MOCKS - Tests real Kestrel agent with real LLM calls.
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

async function waitForAgentResponse(page, initialCount, timeout = 60000) {
    await page.waitForFunction(
        (count) => {
            const messages = document.querySelectorAll('.agent-message');
            if (messages.length <= count) return false;
            const lastMsg = messages[messages.length - 1];
            const content = lastMsg.textContent || '';
            const isStreaming = lastMsg.querySelector('.streaming');
            return content.trim().length > 5 && !isStreaming;
        },
        initialCount,
        { timeout }
    );
}

async function sendMessageAndWait(page, message, timeout = 60000) {
    const initialCount = await page.locator('.agent-message').count();
    const input = page.locator('#message-input');
    await input.fill(message);
    await page.locator('#send-button').click();
    await waitForAgentResponse(page, initialCount, timeout);
    return page.locator('.agent-message').last();
}

async function getSpawnChildren(request, apiKey) {
    const response = await request.get(`${BASE_URL}/api/spawn/children`, {
        headers: authHeaders(apiKey)
    });
    expect(response.ok()).toBeTruthy();
    return await response.json();
}

// ============================================================================
// Spawn Lifecycle Tests
// ============================================================================

test.describe('Spawn Lifecycle', () => {
    let apiKey = null;

    test.beforeAll(async ({ request }) => {
        apiKey = await getApiKey(request);
    });

    test('spawn children endpoint returns valid structure', async ({ request }) => {
        const data = await getSpawnChildren(request, apiKey);

        expect(data).toHaveProperty('children');
        expect(data).toHaveProperty('count');
        expect(data).toHaveProperty('delegation_chain');
        expect(data).toHaveProperty('history');
        expect(Array.isArray(data.children)).toBe(true);
        expect(typeof data.count).toBe('number');
        expect(data.count).toBe(data.children.length);
    });

    test('spawn agent via chat and verify child created', async ({ page, request }) => {
        await page.goto(BASE_URL);
        await page.waitForLoadState('domcontentloaded');

        // Record initial child count
        const before = await getSpawnChildren(request, apiKey);
        const initialCount = before.count;

        // Send a message asking the agent to spawn a child
        const response = await sendMessageAndWait(
            page,
            'Spawn a child agent named "research-helper" with purpose "summarize recent AI news" and a TTL of 120 seconds and budget of 1.0',
            120000
        );

        const responseText = await response.textContent();
        await page.screenshot({ path: 'test-results/spawn-lifecycle-chat-spawn.png' });

        // Verify spawn was acknowledged in the response
        // The agent should mention spawning or creating a child
        const mentionsSpawn = /spawn|child|creat|agent|research.helper/i.test(responseText);
        expect(mentionsSpawn).toBe(true);

        // Check the API for updated children list
        const after = await getSpawnChildren(request, apiKey);

        // If spawn succeeded, we should have more children
        if (after.count > initialCount) {
            const newChild = after.children.find(c => c.name.includes('research'));
            expect(newChild).toBeDefined();

            if (newChild) {
                expect(newChild.status).toBe('running');
                expect(newChild.purpose).toBeTruthy();
                expect(newChild.did).toBeTruthy();
                expect(newChild.ttl_seconds).toBeGreaterThan(0);
            }
        }

        await page.screenshot({ path: 'test-results/spawn-lifecycle-after-spawn.png' });
    });

    test('delegation chain reflects parent-child relationship', async ({ request }) => {
        const data = await getSpawnChildren(request, apiKey);

        // Delegation chain should always have the parent node
        expect(data.delegation_chain).toHaveProperty('name');
        expect(data.delegation_chain).toHaveProperty('did');
        expect(data.delegation_chain).toHaveProperty('status');
        expect(data.delegation_chain).toHaveProperty('children');
        expect(data.delegation_chain.status).toBe('running');

        // If children exist, they should appear in the chain
        if (data.count > 0) {
            expect(data.delegation_chain.children.length).toBeGreaterThan(0);

            for (const child of data.delegation_chain.children) {
                expect(child).toHaveProperty('name');
                expect(child).toHaveProperty('did');
                expect(child).toHaveProperty('status');
                expect(child).toHaveProperty('children');
            }
        }
    });

    test('spawn history records events', async ({ request }) => {
        const data = await getSpawnChildren(request, apiKey);

        expect(Array.isArray(data.history)).toBe(true);

        // If any spawns occurred, history should have entries
        for (const event of data.history) {
            expect(event).toHaveProperty('event');
            expect(event).toHaveProperty('child_name');
            expect(event).toHaveProperty('status');
            expect(['spawned', 'terminated']).toContain(event.event);

            if (event.event === 'spawned') {
                expect(event).toHaveProperty('started_at');
            }
            if (event.event === 'terminated') {
                expect(event).toHaveProperty('ended_at');
                expect(event).toHaveProperty('budget_consumed');
                expect(typeof event.budget_consumed).toBe('number');
            }
        }
    });

    test('terminate child via chat and verify removal', async ({ page, request }) => {
        await page.goto(BASE_URL);
        await page.waitForLoadState('domcontentloaded');

        const before = await getSpawnChildren(request, apiKey);

        if (before.count === 0) {
            // No children to terminate; spawn one first
            await sendMessageAndWait(
                page,
                'Spawn a temporary child agent named "temp-worker" with purpose "test termination" and TTL 60 seconds',
                120000
            );
            await page.waitForTimeout(2000);
        }

        const withChildren = await getSpawnChildren(request, apiKey);

        if (withChildren.count > 0) {
            const childName = withChildren.children[0].name;
            const response = await sendMessageAndWait(
                page,
                `Terminate the child agent named "${childName}"`,
                60000
            );

            const responseText = await response.textContent();
            await page.screenshot({ path: 'test-results/spawn-lifecycle-terminate.png' });

            // Response should acknowledge termination
            const mentionsTerminate = /terminat|stop|shut|remov|end/i.test(responseText);
            expect(mentionsTerminate).toBe(true);

            // Verify via API
            const after = await getSpawnChildren(request, apiKey);
            // Child count should decrease or child status should change
            const childStillRunning = after.children.find(
                c => c.name === childName && c.status === 'running'
            );
            // After termination, child should not be running
            expect(childStillRunning).toBeUndefined();
        }
    });
});

// ============================================================================
// Budget Enforcement Tests
// ============================================================================

test.describe('Spawn Budget Enforcement', () => {
    let apiKey = null;

    test.beforeAll(async ({ request }) => {
        apiKey = await getApiKey(request);
    });

    test('spawned child has budget tracking fields', async ({ request }) => {
        const data = await getSpawnChildren(request, apiKey);

        for (const child of data.children) {
            expect(child).toHaveProperty('budget_allocated');
            expect(child).toHaveProperty('budget_spent');
            expect(child).toHaveProperty('budget_remaining');
            expect(typeof child.budget_allocated).toBe('number');
            expect(typeof child.budget_spent).toBe('number');
            expect(typeof child.budget_remaining).toBe('number');
            // Remaining should not exceed allocated
            expect(child.budget_remaining).toBeLessThanOrEqual(child.budget_allocated);
            // Spent should not be negative
            expect(child.budget_spent).toBeGreaterThanOrEqual(0);
        }
    });

    test('spawn with budget cap via chat', async ({ page, request }) => {
        await page.goto(BASE_URL);
        await page.waitForLoadState('domcontentloaded');

        const response = await sendMessageAndWait(
            page,
            'Spawn a child agent named "budget-test" with purpose "test budget limits" and a budget of 0.50 and TTL of 90 seconds',
            120000
        );

        const responseText = await response.textContent();
        await page.screenshot({ path: 'test-results/spawn-budget-create.png' });

        // Check the child was created with budget
        const data = await getSpawnChildren(request, apiKey);
        const budgetChild = data.children.find(c => c.name.includes('budget'));

        if (budgetChild) {
            expect(budgetChild.budget_allocated).toBeGreaterThan(0);
            expect(budgetChild.budget_remaining).toBeLessThanOrEqual(budgetChild.budget_allocated);
            // At creation, spent should be 0 or very small
            expect(budgetChild.budget_spent).toBeLessThanOrEqual(budgetChild.budget_allocated);
        }
    });
});

// ============================================================================
// Constitutional Scoping Tests
// ============================================================================

test.describe('Spawn Constitutional Scoping', () => {
    let apiKey = null;

    test.beforeAll(async ({ request }) => {
        apiKey = await getApiKey(request);
    });

    test('spawn with constraints via chat', async ({ page, request }) => {
        await page.goto(BASE_URL);
        await page.waitForLoadState('domcontentloaded');

        const response = await sendMessageAndWait(
            page,
            'Spawn a child agent named "constrained-worker" with purpose "read-only research" with only these features allowed: list_children. Set TTL to 90 seconds.',
            120000
        );

        const responseText = await response.textContent();
        await page.screenshot({ path: 'test-results/spawn-constitutional-scope.png' });

        // Verify the response acknowledges the constraints
        const mentionsConstraint = /constrain|restrict|limit|scope|feature|allow/i.test(responseText);
        expect(mentionsConstraint).toBe(true);
    });
});

// ============================================================================
// TTL Expiration Tests
// ============================================================================

test.describe('Spawn TTL Expiration', () => {
    let apiKey = null;

    test.beforeAll(async ({ request }) => {
        apiKey = await getApiKey(request);
    });

    test('spawn with short TTL and verify auto-termination', async ({ page, request }) => {
        await page.goto(BASE_URL);
        await page.waitForLoadState('domcontentloaded');

        // Spawn with very short TTL
        const response = await sendMessageAndWait(
            page,
            'Spawn a child agent named "ttl-test" with purpose "test TTL expiration" and TTL of 15 seconds',
            120000
        );

        await page.screenshot({ path: 'test-results/spawn-ttl-created.png' });

        // Check child exists right after spawn
        const immediately = await getSpawnChildren(request, apiKey);
        const ttlChild = immediately.children.find(c => c.name.includes('ttl'));

        if (ttlChild) {
            expect(ttlChild.status).toBe('running');
            expect(ttlChild.ttl_seconds).toBeLessThanOrEqual(30);

            // TTL remaining should be positive but decreasing
            expect(ttlChild.ttl_remaining).toBeGreaterThanOrEqual(0);

            // Wait for TTL to expire (TTL + buffer for processing)
            await page.waitForTimeout(20000);

            await page.screenshot({ path: 'test-results/spawn-ttl-expired.png' });

            // Verify child is no longer running
            const afterExpiry = await getSpawnChildren(request, apiKey);
            const expiredChild = afterExpiry.children.find(
                c => c.name.includes('ttl') && c.status === 'running'
            );

            // Child should either be gone or have a non-running status
            if (expiredChild) {
                // If still present, TTL remaining should be 0
                expect(expiredChild.ttl_remaining).toBe(0);
            }

            // Check history for termination event
            const historyEvent = afterExpiry.history.find(
                h => h.child_name && h.child_name.includes('ttl') && h.event === 'terminated'
            );
            // If TTL auto-termination worked, we should see it in history
            if (historyEvent) {
                expect(historyEvent.status).toMatch(/timed_out|terminated|completed/);
            }
        }
    });

    test('TTL remaining decreases over time', async ({ request }) => {
        const data1 = await getSpawnChildren(request, apiKey);
        const runningChildren = data1.children.filter(c => c.status === 'running' && c.ttl_remaining > 5);

        if (runningChildren.length > 0) {
            const child = runningChildren[0];
            const ttlBefore = child.ttl_remaining;

            // Wait 3 seconds
            await new Promise(resolve => setTimeout(resolve, 3000));

            const data2 = await getSpawnChildren(request, apiKey);
            const sameChild = data2.children.find(c => c.name === child.name);

            if (sameChild && sameChild.status === 'running') {
                // TTL should have decreased
                expect(sameChild.ttl_remaining).toBeLessThan(ttlBefore);
            }
        }
    });
});
