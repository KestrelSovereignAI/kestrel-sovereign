/**
 * Core-Only E2E Tests
 *
 * Verifies the Kestrel agent boots and works correctly with ONLY core features.
 * Non-core features should be disabled via KESTREL_DISABLED_FEATURES env var
 * on the running server before executing these tests.
 *
 * NO MOCKS - Tests real Kestrel agent with real API calls.
 * Requires: Kestrel server running on localhost:8888 with non-core features disabled.
 *
 * Parent issue: #462 (Open Source Core/Feature Split)
 */
const { test, expect } = require('@playwright/test');

const BASE_URL = process.env.KESTREL_URL || 'http://localhost:8888';

// Non-core features that should be disabled on the test server
const DISABLED_FEATURES = [
    'AuditAnchorFeature',
    'BridgeFeature',
    'ChannelFeature',
    'CodeEditFeature',
    'ComputeFeature',
    'ConsentFeature',
    'CouncilFeature',
    'DeliveryFeature',
    'DeployFeature',
    'GCPComputeFeature',
    'GitHubFeature',
    'KeyManagementFeature',
    'MCPAgent',
    'MemoryAgencyFeature',
    'ObservabilityFeature',
    'ReflectionFeature',
    'ResponseAuditFeature',
    'RunPodFeature',
    'SchedulerFeature',
    'SpawnFeature',
    'StateOfMindFeature',
    'StrategicMemoryFeature',
    'VastAIFeature',
    'VisualIdentityFeature',
    'VoiceFeature',
    'WalletFeature',
    'WebSearchFeature',
    'WebhookFeature',
    'WellnessFeature',
];

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
// Agent Boot & Health
// ============================================================================

test.describe('Core-Only: Boot & Health', () => {
    let apiKey = null;

    test.beforeAll(async ({ request }) => {
        apiKey = await getApiKey(request);
    });

    test('GET /health returns healthy', async ({ request }) => {
        const response = await request.get(`${BASE_URL}/health`);
        expect(response.ok()).toBeTruthy();

        const data = await response.json();
        expect(data.status).toBe('ok');
        expect(data.agent_initialized).toBe(true);
    });

    test('GET /agent/info returns agent with features list', async ({ request }) => {
        const response = await request.get(`${BASE_URL}/agent/info`, {
            headers: authHeaders(apiKey),
        });
        expect(response.ok()).toBeTruthy();

        const data = await response.json();
        expect(data).toHaveProperty('agent_id');
        expect(data).toHaveProperty('features');
        expect(Array.isArray(data.features)).toBeTruthy();

        // None of the disabled features should be loaded
        for (const disabledName of DISABLED_FEATURES) {
            expect(data.features).not.toContain(disabledName);
        }
    });
});

// ============================================================================
// Core Endpoints
// ============================================================================

test.describe('Core-Only: Core Endpoints', () => {
    let apiKey = null;

    test.beforeAll(async ({ request }) => {
        apiKey = await getApiKey(request);
    });

    test('GET /api/conversations returns conversations', async ({ request }) => {
        const response = await request.get(`${BASE_URL}/api/conversations`, {
            headers: authHeaders(apiKey),
        });
        expect(response.ok()).toBeTruthy();

        const data = await response.json();
        expect(data).toHaveProperty('conversations');
        expect(Array.isArray(data.conversations)).toBeTruthy();
    });

    test('GET /api/memories returns memory nodes', async ({ request }) => {
        const response = await request.get(`${BASE_URL}/api/memories`, {
            headers: authHeaders(apiKey),
        });
        expect(response.ok()).toBeTruthy();

        const data = await response.json();
        expect(data).toHaveProperty('nodes');
        expect(Array.isArray(data.nodes)).toBeTruthy();
    });

    test('GET /api/constitution returns verified constitution', async ({ request }) => {
        const response = await request.get(`${BASE_URL}/api/constitution`, {
            headers: authHeaders(apiKey),
        });
        expect(response.ok()).toBeTruthy();

        const data = await response.json();
        expect(data).toHaveProperty('text');
        expect(data.text).not.toBeNull();
        expect(data.text.length).toBeGreaterThan(0);
        expect(data.verified).toBe(true);
    });

    test('GET /api/identity returns agent DID', async ({ request }) => {
        const response = await request.get(`${BASE_URL}/api/identity`, {
            headers: authHeaders(apiKey),
        });
        expect(response.ok()).toBeTruthy();

        const data = await response.json();
        expect(data).toHaveProperty('did');
        expect(data.did).not.toBeNull();
    });

    test('GET /api/models returns available models', async ({ request }) => {
        const response = await request.get(`${BASE_URL}/api/models`, {
            headers: authHeaders(apiKey),
        });
        expect(response.ok()).toBeTruthy();
    });

    test('GET /api/storage/stats returns storage info', async ({ request }) => {
        const response = await request.get(`${BASE_URL}/api/storage/stats`, {
            headers: authHeaders(apiKey),
        });
        expect(response.ok()).toBeTruthy();

        const data = await response.json();
        expect(typeof data).toBe('object');
    });

    test('GET /api/commands lists available commands', async ({ request }) => {
        const response = await request.get(`${BASE_URL}/api/commands`, {
            headers: authHeaders(apiKey),
        });
        expect(response.ok()).toBeTruthy();

        const data = await response.json();
        expect(data).toHaveProperty('commands');
        expect(Array.isArray(data.commands)).toBeTruthy();
    });
});

// ============================================================================
// Privacy Mode
// ============================================================================

test.describe('Core-Only: Privacy Mode', () => {
    let apiKey = null;

    test.beforeAll(async ({ request }) => {
        apiKey = await getApiKey(request);
    });

    test('GET /agent/privacy-mode returns current mode', async ({ request }) => {
        const response = await request.get(`${BASE_URL}/agent/privacy-mode`, {
            headers: authHeaders(apiKey),
        });
        expect(response.ok()).toBeTruthy();

        const data = await response.json();
        expect(data).toHaveProperty('privacy_mode');
    });

    test('POST /agent/privacy-mode sets and retrieves mode', async ({ request }) => {
        // Set to NORMAL
        const setResponse = await request.post(`${BASE_URL}/agent/privacy-mode`, {
            headers: {
                ...authHeaders(apiKey),
                'Content-Type': 'application/json',
            },
            data: { mode: 'normal' },
        });
        expect(setResponse.ok()).toBeTruthy();

        // Verify it was set
        const getResponse = await request.get(`${BASE_URL}/agent/privacy-mode`, {
            headers: authHeaders(apiKey),
        });
        expect(getResponse.ok()).toBeTruthy();

        const data = await getResponse.json();
        expect(data.privacy_mode).toBe('normal');
    });

    test('POST /agent/privacy-mode rejects invalid mode', async ({ request }) => {
        const response = await request.post(`${BASE_URL}/agent/privacy-mode`, {
            headers: {
                ...authHeaders(apiKey),
                'Content-Type': 'application/json',
            },
            data: { mode: 'invalid_mode' },
        });
        expect(response.status()).toBe(400);
    });
});

// ============================================================================
// Memory Storage
// ============================================================================

test.describe('Core-Only: Memory Storage', () => {
    let apiKey = null;

    test.beforeAll(async ({ request }) => {
        apiKey = await getApiKey(request);
    });

    test('GET /api/identity-chain returns identity chain', async ({ request }) => {
        const response = await request.get(`${BASE_URL}/api/identity-chain`, {
            headers: authHeaders(apiKey),
        });
        expect(response.ok()).toBeTruthy();

        const data = await response.json();
        expect(typeof data).toBe('object');
    });

    test('freshly-incepted agent has memory nodes', async ({ request }) => {
        const response = await request.get(`${BASE_URL}/api/memories`, {
            headers: authHeaders(apiKey),
        });
        expect(response.ok()).toBeTruthy();

        const data = await response.json();
        expect(data).toHaveProperty('nodes');
        expect(Array.isArray(data.nodes)).toBeTruthy();
        // A freshly-incepted agent has at least the agent node
        expect(data.nodes.length).toBeGreaterThanOrEqual(1);
    });
});

// ============================================================================
// Agent Invoke
// ============================================================================

test.describe('Core-Only: Agent Invoke', () => {
    let apiKey = null;

    test.beforeAll(async ({ request }) => {
        apiKey = await getApiKey(request);
    });

    test('POST /agent/invoke processes a message', async ({ request }) => {
        // Skip if no LLM keys configured
        if (
            !process.env.OPENAI_API_KEY &&
            !process.env.ANTHROPIC_API_KEY &&
            !process.env.OPENROUTER_API_KEY
        ) {
            test.skip();
            return;
        }

        const response = await request.post(`${BASE_URL}/agent/invoke`, {
            headers: {
                ...authHeaders(apiKey),
                'Content-Type': 'application/json',
            },
            data: { input: 'Say hello in exactly 3 words.' },
        });
        expect(response.ok()).toBeTruthy();

        const data = await response.json();
        expect(data).toHaveProperty('response');
        expect(typeof data.response).toBe('string');
        expect(data.response.length).toBeGreaterThan(0);
    });
});

// ============================================================================
// Disabled Feature Endpoints
// ============================================================================

test.describe('Core-Only: Disabled Features Not Mounted', () => {
    let apiKey = null;

    test.beforeAll(async ({ request }) => {
        apiKey = await getApiKey(request);
    });

    test('voice endpoints return 404/503 when VoiceFeature disabled', async ({ request }) => {
        const response = await request.get(`${BASE_URL}/voice/voices`, {
            headers: authHeaders(apiKey),
        });
        // Route not mounted (404/405) or feature disabled (503)
        expect([404, 405, 503]).toContain(response.status());
    });

    test('spawn endpoints return 404/503 when SpawnFeature disabled', async ({ request }) => {
        const response = await request.get(`${BASE_URL}/api/spawn/children`, {
            headers: authHeaders(apiKey),
        });
        expect([404, 405, 503]).toContain(response.status());
    });

    test('observability endpoints return 404/503 when disabled', async ({ request }) => {
        const response = await request.get(`${BASE_URL}/api/observability/events`, {
            headers: authHeaders(apiKey),
        });
        expect([404, 405, 503]).toContain(response.status());
    });
});
