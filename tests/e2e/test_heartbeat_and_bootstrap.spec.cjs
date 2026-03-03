/**
 * Heartbeat System & Bootstrap File Convention E2E Tests
 *
 * Tests for:
 * - Heartbeat API endpoints (GET /agent/heartbeat/status, POST /agent/heartbeat/trigger)
 * - !heartbeat command via chat
 * - !reload-context command via chat
 * - Bootstrap file loading verification via context-status
 *
 * NO MOCKS - Tests real Kestrel agent with real API calls.
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

// Helper to wait for agent response with actual content
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

// Helper to send a message and wait for response
async function sendMessageAndWait(page, message, timeout = 60000) {
    const initialCount = await page.locator('.agent-message').count();
    const input = page.locator('#message-input');
    await input.fill(message);
    await page.locator('#send-button').click();
    await waitForAgentResponse(page, initialCount, timeout);
    return page.locator('.agent-message').last();
}

// ============================================================================
// Heartbeat API Tests
// ============================================================================

test.describe('Heartbeat API', () => {
    let apiKey = null;

    test.beforeAll(async ({ request }) => {
        apiKey = await getApiKey(request);
    });

    test('GET /agent/heartbeat/status returns heartbeat config', async ({ request }) => {
        const response = await request.get(`${BASE_URL}/agent/heartbeat/status`, {
            headers: authHeaders(apiKey),
        });
        expect(response.ok()).toBeTruthy();

        const data = await response.json();
        // Should have enabled field (may be true or false depending on config)
        expect(data).toHaveProperty('enabled');

        // If heartbeat is configured, check additional fields
        if (data.enabled !== undefined && !data.message) {
            expect(data).toHaveProperty('running');
            expect(data).toHaveProperty('interval_seconds');
            expect(data).toHaveProperty('target');
            expect(data).toHaveProperty('history_count');
            expect(typeof data.interval_seconds).toBe('number');
        }
    });

    test('POST /agent/heartbeat/trigger executes a heartbeat', async ({ request }) => {
        const response = await request.post(`${BASE_URL}/agent/heartbeat/trigger`, {
            headers: authHeaders(apiKey),
        });

        // May be 404 if heartbeat not configured, or 200 if it runs
        if (response.status() === 404) {
            const data = await response.json();
            expect(data.detail).toContain('not configured');
        } else {
            expect(response.ok()).toBeTruthy();
            const data = await response.json();
            expect(data).toHaveProperty('status');
            expect(data).toHaveProperty('timestamp');
            expect(data).toHaveProperty('duration_ms');
            // Status should be one of the valid values
            expect(['ok', 'alert', 'skipped', 'error']).toContain(data.status);
        }
    });

    test('heartbeat status reflects history after trigger', async ({ request }) => {
        // Trigger a heartbeat first
        const triggerResp = await request.post(`${BASE_URL}/agent/heartbeat/trigger`, {
            headers: authHeaders(apiKey),
        });

        if (triggerResp.status() === 404) {
            test.skip();
            return;
        }

        // Then check status shows history
        const statusResp = await request.get(`${BASE_URL}/agent/heartbeat/status`, {
            headers: authHeaders(apiKey),
        });
        expect(statusResp.ok()).toBeTruthy();

        const data = await statusResp.json();
        expect(data.history_count).toBeGreaterThan(0);
        expect(data.last_result).not.toBeNull();
        expect(data.last_result).toHaveProperty('status');
    });
});

// ============================================================================
// Heartbeat Command Tests (via Chat UI)
// ============================================================================

test.describe('Heartbeat Command', () => {
    let apiKey = null;

    test.beforeAll(async ({ request }) => {
        apiKey = await getApiKey(request);
    });

    test('!heartbeat command returns result via API', async ({ request }) => {
        const response = await request.post(`${BASE_URL}/agent/invoke`, {
            headers: {
                ...authHeaders(apiKey),
                'Content-Type': 'application/json',
            },
            data: { input: '!heartbeat' },
        });
        expect(response.ok()).toBeTruthy();

        const data = await response.json();
        const text = data.response || '';
        // Should contain heartbeat-related output
        expect(
            text.includes('Heartbeat') || text.includes('heartbeat') || text.includes('not configured')
        ).toBeTruthy();
    });

    test('!reload-context command returns result via API', async ({ request }) => {
        const response = await request.post(`${BASE_URL}/agent/invoke`, {
            headers: {
                ...authHeaders(apiKey),
                'Content-Type': 'application/json',
            },
            data: { input: '!reload-context' },
        });
        expect(response.ok()).toBeTruthy();

        const data = await response.json();
        const text = data.response || '';
        // Should mention bootstrap files or "No bootstrap files" or "reloaded"
        expect(
            text.includes('reloaded') ||
            text.includes('bootstrap') ||
            text.includes('No bootstrap') ||
            text.includes('SOUL.md') ||
            text.includes('Context builder')
        ).toBeTruthy();
    });
});

// ============================================================================
// Bootstrap Context Tests (via Agent Info API)
// ============================================================================

test.describe('Bootstrap Context', () => {
    let apiKey = null;

    test.beforeAll(async ({ request }) => {
        apiKey = await getApiKey(request);
    });

    test('agent info endpoint returns initialized agent', async ({ request }) => {
        const response = await request.get(`${BASE_URL}/agent/info`, {
            headers: authHeaders(apiKey),
        });
        expect(response.ok()).toBeTruthy();

        const data = await response.json();
        expect(data).toHaveProperty('agent_id');
        expect(data).toHaveProperty('features');
        expect(Array.isArray(data.features)).toBeTruthy();
    });

    test('context-status endpoint shows token usage', async ({ request }) => {
        const response = await request.get(`${BASE_URL}/agent/context-status`, {
            headers: authHeaders(apiKey),
        });
        expect(response.ok()).toBeTruthy();

        const data = await response.json();
        expect(data).toHaveProperty('model');
        expect(data).toHaveProperty('context_limit');
        expect(data).toHaveProperty('total_budget');
        expect(data).toHaveProperty('status');
        // Context limit should be positive
        expect(data.context_limit).toBeGreaterThan(0);
    });
});

// ============================================================================
// Chat UI Integration Tests
// ============================================================================

test.describe('Heartbeat & Bootstrap via Chat UI', () => {
    let apiKey = null;

    test.beforeAll(async ({ request }) => {
        apiKey = await getApiKey(request);
    });

    test('!help lists heartbeat and reload-context commands', async ({ request }) => {
        const response = await request.post(`${BASE_URL}/agent/invoke`, {
            headers: {
                ...authHeaders(apiKey),
                'Content-Type': 'application/json',
            },
            data: { input: '!help' },
        });
        expect(response.ok()).toBeTruthy();

        const data = await response.json();
        const text = data.response || '';
        expect(text).toContain('!heartbeat');
        expect(text).toContain('!reload-context');
    });
});
