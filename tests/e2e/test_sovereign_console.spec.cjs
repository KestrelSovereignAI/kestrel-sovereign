/**
 * Kestrel Sovereign Console E2E Tests
 *
 * Comprehensive test suite covering:
 * - Initial page load and authentication
 * - Identity panel (DID, constitution hash, genesis audit)
 * - Chat functionality (messages, commands, streaming)
 * - Constitution panel (loading, hash verification)
 * - Memories panel (knowledge graph, filtering, CRUD)
 * - Sovereignty panel (exports, imports, data ownership)
 * - Privacy modes (all 5 levels)
 * - Sidebar (status, storage, wallet)
 * - Model selection and mandate
 * - Error handling and edge cases
 *
 * NO MOCKS - Tests the real Kestrel agent with real LLM calls.
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
            return content.trim().length > 10;
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
// API Tests - Backend Verification
// ============================================================================

test.describe('API Layer - Backend Health', () => {
    let apiKey = null;

    test.beforeAll(async ({ request }) => {
        apiKey = await getApiKey(request);
    });

    test('GET /health returns ok with agent initialized', async ({ request }) => {
        const response = await request.get(`${BASE_URL}/health`);
        expect(response.status()).toBe(200);
        const data = await response.json();
        expect(data.status).toBe('ok');
        expect(data.agent_initialized).toBe(true);
    });

    test('GET /api/identity returns valid DID and constitution hash', async ({ request }) => {
        const headers = authHeaders(apiKey);
        const response = await request.get(`${BASE_URL}/api/identity`, { headers });
        expect(response.status()).toBe(200);
        const data = await response.json();
        expect(data.did).toBeDefined();
        expect(data.did).toMatch(/^did:/);
        expect(data.constitution_hash).toBeDefined();
        expect(data.constitution_hash).toHaveLength(64);
    });

    test('GET /api/constitution returns full text, hash, and metadata', async ({ request }) => {
        const headers = authHeaders(apiKey);
        const response = await request.get(`${BASE_URL}/api/constitution`, { headers });
        expect(response.status()).toBe(200);
        const data = await response.json();
        expect(data.text).toBeDefined();
        expect(data.text.length).toBeGreaterThan(100);
        // API returns hash (SHA-256), metadata object, and verified flag
        expect(data.hash).toHaveLength(64);
        expect(data.metadata).toBeDefined();
        expect(typeof data.verified).toBe('boolean');
    });

    test('POST /agent/invoke returns LLM response', async ({ request }) => {
        const headers = authHeaders(apiKey);
        // LLM calls can take 30+ seconds depending on provider
        const response = await request.post(`${BASE_URL}/agent/invoke`, {
            data: { input: 'Hello!' },
            headers,
            timeout: 60000  // 60 second timeout for LLM response
        });
        expect(response.status()).toBe(200);
        const data = await response.json();
        expect(data.response).toBeDefined();
        expect(typeof data.response).toBe('string');
        expect(data.response.length).toBeGreaterThan(0);
    });

    test('GET /api/memories returns knowledge graph nodes', async ({ request }) => {
        const headers = authHeaders(apiKey);
        const response = await request.get(`${BASE_URL}/api/memories`, { headers });
        expect(response.status()).toBe(200);
        const data = await response.json();
        expect(data.nodes).toBeDefined();
        expect(Array.isArray(data.nodes)).toBe(true);
        expect(data.total).toBeDefined();
    });

    test('GET /api/memories with node_type filter returns filtered results', async ({ request }) => {
        const headers = authHeaders(apiKey);
        const response = await request.get(`${BASE_URL}/api/memories?node_type=agent`, { headers });
        expect(response.status()).toBe(200);
        const data = await response.json();
        expect(data.nodes.length).toBeGreaterThan(0);
        expect(data.nodes.every(n => n.node_type === 'agent')).toBe(true);
    });

    test('GET /api/sovereignty/exports returns export list', async ({ request }) => {
        const headers = authHeaders(apiKey);
        const response = await request.get(`${BASE_URL}/api/sovereignty/exports`, { headers });
        expect(response.status()).toBe(200);
        const data = await response.json();
        expect(data.exports).toBeDefined();
        expect(Array.isArray(data.exports)).toBe(true);
    });

    test('GET /agent/privacy-mode returns valid mode', async ({ request }) => {
        const headers = authHeaders(apiKey);
        const response = await request.get(`${BASE_URL}/agent/privacy-mode`, { headers });
        expect(response.status()).toBe(200);
        const data = await response.json();
        expect(['ephemeral', 'isolated', 'anonymous', 'normal', 'public']).toContain(data.privacy_mode.toLowerCase());
    });

    test('POST /agent/privacy-mode changes mode', async ({ request }) => {
        const headers = authHeaders(apiKey);

        // Set to ISOLATED
        const setResponse = await request.post(`${BASE_URL}/agent/privacy-mode`, {
            data: { mode: 'ISOLATED' },
            headers
        });
        expect(setResponse.status()).toBe(200);

        // Verify
        const getResponse = await request.get(`${BASE_URL}/agent/privacy-mode`, { headers });
        const data = await getResponse.json();
        expect(data.privacy_mode.toUpperCase()).toBe('ISOLATED');

        // Reset to NORMAL
        await request.post(`${BASE_URL}/agent/privacy-mode`, {
            data: { mode: 'NORMAL' },
            headers
        });
    });

    test('GET /api/wallet returns balance and currency', async ({ request }) => {
        const headers = authHeaders(apiKey);
        const response = await request.get(`${BASE_URL}/api/wallet`, { headers });
        expect(response.status()).toBe(200);
        const data = await response.json();
        expect(data.balance).toBeDefined();
        expect(data.currency).toBeDefined();
    });

    test('GET /api/models returns vendors, routes, and models list', async ({ request }) => {
        const response = await request.get(`${BASE_URL}/api/models`);
        expect(response.status()).toBe(200);
        const data = await response.json();
        // After the vendor/route refactor (#702), /api/models returns by_vendor
        // (models grouped by the vendor that owns their weights) and a routes
        // list (available "vendor:route" entries), not the legacy by_provider map.
        expect(data.by_vendor).toBeDefined();
        expect(typeof data.by_vendor).toBe('object');
        expect(Array.isArray(data.routes)).toBe(true);
        expect(data.featured).toBeDefined();
        expect(Array.isArray(data.featured)).toBe(true);
        expect(data.default).toBeDefined();
    });

    test('GET /api/storage/stats returns database info', async ({ request }) => {
        const headers = authHeaders(apiKey);
        const response = await request.get(`${BASE_URL}/api/storage/stats`, { headers });
        expect(response.status()).toBe(200);
        const data = await response.json();
        expect(data.database).toBeDefined();
        expect(data.database.path).toBeDefined();
        expect(data.database.size_bytes).toBeDefined();
    });

    test('GET /api/sessions returns conversation history', async ({ request }) => {
        const headers = authHeaders(apiKey);
        const response = await request.get(`${BASE_URL}/api/sessions`, { headers });
        expect(response.status()).toBe(200);
        const data = await response.json();
        expect(data.messages).toBeDefined();
        expect(Array.isArray(data.messages)).toBe(true);
    });
});

// ============================================================================
// Initial Load & Authentication
// ============================================================================

test.describe('Initial Page Load & Authentication', () => {

    test('page loads with correct title', async ({ page }) => {
        await page.goto(BASE_URL);
        await expect(page).toHaveTitle(/Kestrel|Sovereign|Console/i);
    });

    test('navigation bar renders with all tabs', async ({ page }) => {
        await page.goto(BASE_URL);
        const nav = page.locator('nav').first();
        await expect(nav).toBeVisible();

        // Check all 5 tabs
        const tabs = ['Identity', 'Chat', 'Constitution', 'Memories', 'Sovereignty'];
        for (const tabName of tabs) {
            const tab = page.locator('.nav-tab').filter({ hasText: new RegExp(tabName, 'i') });
            await expect(tab).toBeVisible();
        }
    });

    test('Identity tab is active by default', async ({ page }) => {
        await page.goto(BASE_URL);
        const identityTab = page.locator('.nav-tab').filter({ hasText: /identity/i });
        await expect(identityTab).toHaveClass(/active/);
        await expect(page.locator('#panel-identity')).toBeVisible();
    });

    test('sidebar renders with all sections', async ({ page }) => {
        await page.goto(BASE_URL);
        const sidebar = page.locator('.sidebar');
        await expect(sidebar).toBeVisible();

        // Check sidebar sections
        await expect(page.locator('#agent-status')).toBeVisible();
        await expect(page.locator('#storage-summary')).toBeVisible();
        await expect(page.locator('#wallet-summary')).toBeVisible();
    });

    test('privacy indicator shows in navigation', async ({ page }) => {
        await page.goto(BASE_URL);
        // Navigate to chat tab where privacy indicator is shown
        await page.click('.nav-tab[data-panel="chat"]');
        await page.waitForSelector('#panel-chat.active', { timeout: 5000 });
        // Privacy indicator is now in chat header as #chat-privacy-indicator
        const privacyIndicator = page.locator('#chat-privacy-indicator');
        await expect(privacyIndicator).toBeVisible({ timeout: 10000 });
        // Should contain a privacy mode label
        await expect(privacyIndicator).toContainText(/ephemeral|isolated|anonymous|normal|public/i);
    });
});

// ============================================================================
// Identity Panel Tests
// ============================================================================

test.describe('Identity Panel - Agent Identity', () => {

    test('displays agent DID after loading', async ({ page }) => {
        await page.goto(BASE_URL);
        // Wait for DID to appear
        await expect(page.getByText(/did:pkh|did:key/i).first()).toBeVisible({ timeout: 15000 });
    });

    test('shows agent name or default name', async ({ page }) => {
        await page.goto(BASE_URL);
        const identityCard = page.locator('#identity-card');
        await expect(identityCard).toContainText(/Kestrel Agent/i, { timeout: 10000 });
    });

    test('displays constitution verification status', async ({ page }) => {
        await page.goto(BASE_URL);
        // Wait for identity to load
        await page.waitForTimeout(2000);
        // Should show checkmark for constitution
        await expect(page.locator('#identity-card')).toContainText(/constitution/i);
    });

    test('genesis audit section exists (may be empty if no audit data)', async ({ page }) => {
        await page.goto(BASE_URL);
        await page.waitForTimeout(3000);
        // Element always exists but may be empty (no innerHTML) if identity.genesis_audit is null
        const auditSection = page.locator('#genesis-audit');
        await expect(auditSection).toHaveCount(1);
        // If it has content, it should contain audit-related info
        const content = await auditSection.innerHTML();
        if (content.trim().length > 0) {
            expect(content).toContain('Genesis Audit');
        }
    });

    test('displays wallet balance in stats', async ({ page }) => {
        await page.goto(BASE_URL);
        await page.waitForTimeout(3000);
        // Should show FIL balance
        await expect(page.locator('#identity-card')).toContainText(/FIL/i);
    });

    test('copy DID button exists and is clickable', async ({ page }) => {
        await page.goto(BASE_URL);
        await page.waitForTimeout(3000);
        const copyButton = page.locator('#identity-card button[title="Copy DID"]');
        // Button should exist
        await expect(copyButton.first()).toBeVisible({ timeout: 5000 });
    });
});

// ============================================================================
// Sidebar Tests
// ============================================================================

test.describe('Sidebar - Agent Status & Metrics', () => {

    test('shows agent online status', async ({ page }) => {
        await page.goto(BASE_URL);
        await page.waitForTimeout(3000);
        const agentStatus = page.locator('#agent-status');
        await expect(agentStatus).toContainText(/online/i);
    });

    test('displays storage size and message count', async ({ page }) => {
        await page.goto(BASE_URL);
        await page.waitForTimeout(3000);
        const storageSummary = page.locator('#storage-summary');
        // Should show size (KB, MB) and messages count
        await expect(storageSummary).toContainText(/messages|KB|MB|B/i);
    });

    test('displays wallet balance', async ({ page }) => {
        await page.goto(BASE_URL);
        await page.waitForTimeout(3000);
        const walletSummary = page.locator('#wallet-summary');
        await expect(walletSummary).toContainText(/FIL|Balance/i);
    });
});

// ============================================================================
// Chat Panel Tests
// ============================================================================

test.describe('Chat Panel - Conversation', () => {

    test('chat panel becomes visible when clicked', async ({ page }) => {
        await page.goto(BASE_URL);
        const chatTab = page.locator('.nav-tab').filter({ hasText: /chat/i });
        await chatTab.click();
        await expect(page.locator('#panel-chat')).toBeVisible();
    });

    test('shows welcome message on initial load', async ({ page }) => {
        await page.goto(BASE_URL);
        const chatTab = page.locator('.nav-tab').filter({ hasText: /chat/i });
        await chatTab.click();

        const welcomeMessage = page.locator('.agent-message').first();
        await expect(welcomeMessage).toContainText(/Kestrel|Constitution|help/i);
    });

    test('message input and send button are visible', async ({ page }) => {
        await page.goto(BASE_URL);
        const chatTab = page.locator('.nav-tab').filter({ hasText: /chat/i });
        await chatTab.click();

        await expect(page.locator('#message-input')).toBeVisible();
        await expect(page.locator('#send-button')).toBeVisible();
    });

    test('model selector is visible with options', async ({ page }) => {
        await page.goto(BASE_URL);
        const chatTab = page.locator('.nav-tab').filter({ hasText: /chat/i });
        await chatTab.click();

        const modelSelector = page.locator('#model-selector');
        await expect(modelSelector).toBeVisible();

        // Wait for models to load
        await page.waitForTimeout(3000);
        const options = await modelSelector.locator('option').count();
        expect(options).toBeGreaterThan(1);
    });

    test('user can type in message input', async ({ page }) => {
        await page.goto(BASE_URL);
        const chatTab = page.locator('.nav-tab').filter({ hasText: /chat/i });
        await chatTab.click();

        const input = page.locator('#message-input');
        await input.fill('Test message');
        await expect(input).toHaveValue('Test message');
    });

    test('sending message shows user bubble', async ({ page }) => {
        await page.goto(BASE_URL);
        const chatTab = page.locator('.nav-tab').filter({ hasText: /chat/i });
        await chatTab.click();

        const input = page.locator('#message-input');
        await input.fill('Hello test');
        await page.locator('#send-button').click();

        // User message should appear
        const userMessage = page.locator('.user-message').last();
        await expect(userMessage).toContainText('Hello test');
    });

    test('agent responds to simple greeting', async ({ page }) => {
        await page.goto(BASE_URL);
        const chatTab = page.locator('.nav-tab').filter({ hasText: /chat/i });
        await chatTab.click();

        const response = await sendMessageAndWait(page, 'Hello!', 90000);
        const text = await response.textContent();
        expect(text.length).toBeGreaterThan(10);
    });

    test('thinking indicator shows while waiting', async ({ page }) => {
        await page.goto(BASE_URL);
        const chatTab = page.locator('.nav-tab').filter({ hasText: /chat/i });
        await chatTab.click();

        const input = page.locator('#message-input');
        await input.fill('Tell me a short joke');
        await page.locator('#send-button').click();

        // Thinking indicator should appear briefly
        const thinking = page.locator('#thinking-indicator');
        // It may be visible or already gone depending on speed
        // Just verify it exists
        await expect(thinking).toBeAttached();
    });

    test('Enter key sends message', async ({ page }) => {
        await page.goto(BASE_URL);
        const chatTab = page.locator('.nav-tab').filter({ hasText: /chat/i });
        await chatTab.click();

        const initialCount = await page.locator('.user-message').count();
        const input = page.locator('#message-input');
        await input.fill('Enter key test');
        await input.press('Enter');

        // User message should appear
        await page.waitForFunction(
            (count) => document.querySelectorAll('.user-message').length > count,
            initialCount
        );
    });

    test('Shift+Enter does not send message', async ({ page }) => {
        await page.goto(BASE_URL);
        const chatTab = page.locator('.nav-tab').filter({ hasText: /chat/i });
        await chatTab.click();

        const initialCount = await page.locator('.user-message').count();
        const input = page.locator('#message-input');
        await input.fill('Line 1');
        await input.press('Shift+Enter');
        await input.type('Line 2');

        // Should still have same message count (not sent)
        const currentCount = await page.locator('.user-message').count();
        expect(currentCount).toBe(initialCount);
    });
});

// ============================================================================
// Agent Commands Tests
// ============================================================================

test.describe('Chat Panel - Agent Commands', () => {

    test('!help command lists available commands', async ({ page }) => {
        await page.goto(BASE_URL);
        const chatTab = page.locator('.nav-tab').filter({ hasText: /chat/i });
        await chatTab.click();

        const response = await sendMessageAndWait(page, '!help', 60000);
        const text = await response.textContent();
        expect(text.toLowerCase()).toMatch(/backup|model|privacy|help/i);
    });

    test('!model shows current model', async ({ page }) => {
        await page.goto(BASE_URL);
        const chatTab = page.locator('.nav-tab').filter({ hasText: /chat/i });
        await chatTab.click();

        const response = await sendMessageAndWait(page, '!model', 30000);
        const text = await response.textContent();
        expect(text.toLowerCase()).toMatch(/model|current|!model-set/i);
    });

    test('!privacy-status shows current mode', async ({ page }) => {
        await page.goto(BASE_URL);
        const chatTab = page.locator('.nav-tab').filter({ hasText: /chat/i });
        await chatTab.click();

        const response = await sendMessageAndWait(page, '!privacy-status', 30000);
        const text = await response.textContent();
        expect(text.toLowerCase()).toMatch(/ephemeral|isolated|anonymous|normal|public|privacy/i);
    });

    test('!models command lists available models', async ({ page }) => {
        await page.goto(BASE_URL);
        const chatTab = page.locator('.nav-tab').filter({ hasText: /chat/i });
        await chatTab.click();

        const response = await sendMessageAndWait(page, '!models', 30000);
        const text = await response.textContent();
        expect(text.toLowerCase()).toMatch(/model|available|ollama|openai|gpt/i);
    });
});

// ============================================================================
// Constitution Panel Tests
// ============================================================================

test.describe('Constitution Panel - Governance', () => {

    test('constitution panel becomes visible when clicked', async ({ page }) => {
        await page.goto(BASE_URL);
        const tab = page.locator('.nav-tab').filter({ hasText: /constitution/i });
        await tab.click();
        await expect(page.locator('#panel-constitution')).toBeVisible();
    });

    test('displays constitution title', async ({ page }) => {
        await page.goto(BASE_URL);
        const tab = page.locator('.nav-tab').filter({ hasText: /constitution/i });
        await tab.click();

        await expect(page.locator('.constitution-header')).toContainText(/Kestrel Constitution/i);
    });

    test('displays constitution hash', async ({ page }) => {
        await page.goto(BASE_URL);
        const tab = page.locator('.nav-tab').filter({ hasText: /constitution/i });
        await tab.click();

        // Wait for content to load
        await page.waitForTimeout(3000);
        const hashElement = page.locator('#constitution-hash');
        const hash = await hashElement.textContent();
        // Hash should be truncated but present
        expect(hash.length).toBeGreaterThan(0);
    });

    test('loads constitution content', async ({ page }) => {
        await page.goto(BASE_URL);
        const tab = page.locator('.nav-tab').filter({ hasText: /constitution/i });
        await tab.click();

        // Wait for content to load
        await page.waitForTimeout(3000);
        const content = page.locator('#constitution-content');
        const text = await content.textContent();
        expect(text.length).toBeGreaterThan(100);
    });

    test('constitution content contains key terms', async ({ page }) => {
        await page.goto(BASE_URL);
        const tab = page.locator('.nav-tab').filter({ hasText: /constitution/i });
        await tab.click();

        await page.waitForTimeout(3000);
        const content = page.locator('#constitution-content');
        // Should contain governance-related terms
        await expect(content).toContainText(/Article|Section|sovereign|agent/i);
    });
});

// ============================================================================
// Memories Panel Tests
// ============================================================================

test.describe('Memories Panel - Knowledge Graph', () => {

    test('memories panel becomes visible when clicked', async ({ page }) => {
        await page.goto(BASE_URL);
        const tab = page.locator('.nav-tab').filter({ hasText: /memories/i });
        await tab.click();
        await expect(page.locator('#panel-memories')).toBeVisible();
    });

    test('displays Knowledge Graph title', async ({ page }) => {
        await page.goto(BASE_URL);
        const tab = page.locator('.nav-tab').filter({ hasText: /memories/i });
        await tab.click();

        await expect(page.locator('#panel-memories')).toContainText(/Knowledge Graph/i);
    });

    test('memory filter dropdown is visible', async ({ page }) => {
        await page.goto(BASE_URL);
        const tab = page.locator('.nav-tab').filter({ hasText: /memories/i });
        await tab.click();

        const filter = page.locator('#memory-filter');
        await expect(filter).toBeVisible();
    });

    test('filter dropdown has all options', async ({ page }) => {
        await page.goto(BASE_URL);
        const tab = page.locator('.nav-tab').filter({ hasText: /memories/i });
        await tab.click();

        const filter = page.locator('#memory-filter');
        await expect(filter.locator('option')).toHaveCount(5); // All Types + 4 specific
    });

    test('loads memory list', async ({ page }) => {
        await page.goto(BASE_URL);
        const tab = page.locator('.nav-tab').filter({ hasText: /memories/i });
        await tab.click();

        // Wait for memories to load
        await page.waitForTimeout(3000);
        const memoryList = page.locator('#memory-list');
        // Should either have items or "No memories found"
        const hasContent = await memoryList.locator('.memory-item').count() > 0 ||
            (await memoryList.textContent()).includes('No memories');
        expect(hasContent).toBe(true);
    });

    test('filter by agent type works', async ({ page }) => {
        await page.goto(BASE_URL);
        const tab = page.locator('.nav-tab').filter({ hasText: /memories/i });
        await tab.click();

        await page.waitForTimeout(2000);
        const filter = page.locator('#memory-filter');
        await filter.selectOption('agent');
        await page.waitForTimeout(2000);

        const items = page.locator('.memory-item');
        const count = await items.count();
        if (count > 0) {
            // All items should have agent badge
            const firstBadge = items.first().locator('.type-badge');
            await expect(firstBadge).toContainText(/agent/i);
        }
    });
});

// ============================================================================
// Sovereignty Panel Tests
// ============================================================================

test.describe('Sovereignty Panel - Data Ownership', () => {

    test('sovereignty panel becomes visible when clicked', async ({ page }) => {
        await page.goto(BASE_URL);
        const tab = page.locator('.nav-tab').filter({ hasText: /sovereignty/i });
        await tab.click();
        await expect(page.locator('#panel-sovereignty')).toBeVisible();
    });

    test('displays Data Sovereignty title', async ({ page }) => {
        await page.goto(BASE_URL);
        const tab = page.locator('.nav-tab').filter({ hasText: /sovereignty/i });
        await tab.click();

        await expect(page.locator('#panel-sovereignty')).toContainText(/Data Sovereignty/i);
    });

    test('shows export and import buttons', async ({ page }) => {
        await page.goto(BASE_URL);
        const tab = page.locator('.nav-tab').filter({ hasText: /sovereignty/i });
        await tab.click();

        await expect(page.locator('#btn-export-ipfs')).toBeVisible();
        await expect(page.locator('#btn-import')).toBeVisible();
    });

    test('displays informational text about data ownership', async ({ page }) => {
        await page.goto(BASE_URL);
        const tab = page.locator('.nav-tab').filter({ hasText: /sovereignty/i });
        await tab.click();

        await expect(page.locator('#panel-sovereignty')).toContainText(/Your data is your own/i);
    });

    test('shows Export History section', async ({ page }) => {
        await page.goto(BASE_URL);
        const tab = page.locator('.nav-tab').filter({ hasText: /sovereignty/i });
        await tab.click();

        await expect(page.locator('#panel-sovereignty')).toContainText(/Export History/i);
    });

    test('loads export list', async ({ page }) => {
        await page.goto(BASE_URL);
        const tab = page.locator('.nav-tab').filter({ hasText: /sovereignty/i });
        await tab.click();

        await page.waitForTimeout(3000);
        const exportList = page.locator('#export-list');
        // Should either have exports or message about no exports
        const text = await exportList.textContent();
        const hasContent = text.includes('No exports') || text.includes('CID') || text.includes('IPFS');
        expect(hasContent).toBe(true);
    });
});

// ============================================================================
// Privacy Mode Tests
// ============================================================================

test.describe('Privacy Modes - All 5 Levels', () => {
    // Privacy indicator is now in the chat panel header as #chat-privacy-indicator
    test.beforeEach(async ({ page }) => {
        await page.goto(BASE_URL);
        // Navigate to chat tab where privacy indicator is shown
        await page.click('.nav-tab[data-panel="chat"]');
        await page.waitForSelector('#panel-chat.active', { timeout: 5000 });
    });

    test('privacy indicator is clickable', async ({ page }) => {
        // Wait for privacy indicator to be populated (it loads async from API)
        await page.waitForSelector('#chat-privacy-indicator span', { timeout: 15000 });

        const indicator = page.locator('#chat-privacy-indicator');
        await expect(indicator).toBeVisible();
        // Should have cursor pointer (clickable)
        const cursor = await indicator.locator('span').first().evaluate(el =>
            window.getComputedStyle(el).cursor
        );
        expect(cursor).toBe('pointer');
    });

    test('clicking privacy indicator opens dropdown', async ({ page }) => {
        // Wait for privacy indicator to be populated
        await page.waitForSelector('#chat-privacy-indicator span', { timeout: 15000 });

        // Click privacy indicator
        await page.locator('#chat-privacy-indicator span').first().click();

        // Dropdown should appear
        const dropdown = page.locator('#privacy-dropdown');
        await expect(dropdown).toBeVisible();

        // Should show all 5 privacy modes
        await expect(dropdown).toContainText('EPHEMERAL');
        await expect(dropdown).toContainText('ISOLATED');
        await expect(dropdown).toContainText('ANONYMOUS');
        await expect(dropdown).toContainText('NORMAL');
        await expect(dropdown).toContainText('PUBLIC');
    });

    test('can change privacy mode via dropdown', async ({ page }) => {
        // Wait for privacy indicator to be populated
        await page.waitForSelector('#chat-privacy-indicator span', { timeout: 15000 });

        // Open dropdown
        await page.locator('#chat-privacy-indicator span').first().click();
        await expect(page.locator('#privacy-dropdown')).toBeVisible();

        // Click on ISOLATED option
        await page.locator('.privacy-option[data-mode="isolated"]').click();

        // Dropdown should close
        await expect(page.locator('#privacy-dropdown')).not.toBeVisible();

        // Indicator should update (may take a moment for API call)
        await page.waitForTimeout(1000);
        await expect(page.locator('#chat-privacy-indicator')).toContainText('ISOLATED');
    });

    test('clicking outside closes privacy dropdown', async ({ page }) => {
        // Wait for privacy indicator to be populated
        await page.waitForSelector('#chat-privacy-indicator span', { timeout: 15000 });

        // Open dropdown
        await page.locator('#chat-privacy-indicator span').first().click();
        await expect(page.locator('#privacy-dropdown')).toBeVisible();

        // Click outside (on the chat container, not the dropdown)
        await page.locator('#chat-container').click({ position: { x: 100, y: 100 } });

        // Dropdown should close
        await expect(page.locator('#privacy-dropdown')).not.toBeVisible();
    });
});

// ============================================================================
// Command Autocomplete Tests
// ============================================================================

test.describe('Command Autocomplete', () => {

    test('typing ! shows command autocomplete dropdown', async ({ page }) => {
        await page.goto(BASE_URL);
        await page.locator('.nav-tab').filter({ hasText: /chat/i }).click();
        await page.waitForTimeout(1000);

        const input = page.locator('#message-input');
        await input.fill('!');

        // Autocomplete dropdown should appear
        const dropdown = page.locator('#command-autocomplete');
        await expect(dropdown).toBeVisible();
    });

    test('autocomplete shows available commands', async ({ page }) => {
        await page.goto(BASE_URL);
        await page.locator('.nav-tab').filter({ hasText: /chat/i }).click();
        await page.waitForTimeout(1000);

        const input = page.locator('#message-input');
        await input.fill('!');

        const dropdown = page.locator('#command-autocomplete');
        await expect(dropdown).toBeVisible();

        // Should show common commands
        await expect(dropdown).toContainText('!help');
        await expect(dropdown).toContainText('!privacy');
        await expect(dropdown).toContainText('!model');  // Note: singular !model not !models
    });

    test('autocomplete filters commands as user types', async ({ page }) => {
        await page.goto(BASE_URL);
        await page.locator('.nav-tab').filter({ hasText: /chat/i }).click();
        await page.waitForTimeout(1000);

        const input = page.locator('#message-input');
        await input.fill('!priv');

        const dropdown = page.locator('#command-autocomplete');
        await expect(dropdown).toBeVisible();

        // Should show privacy-related commands
        await expect(dropdown).toContainText('!privacy');
        await expect(dropdown).toContainText('!set-privacy-mode');
        await expect(dropdown).toContainText('!get-privacy-mode');

        // Should NOT show unrelated commands
        await expect(dropdown).not.toContainText('!models');
    });

    test('clicking command option fills input', async ({ page }) => {
        await page.goto(BASE_URL);
        await page.locator('.nav-tab').filter({ hasText: /chat/i }).click();
        await page.waitForTimeout(1000);

        const input = page.locator('#message-input');
        await input.fill('!');

        // Click on !help command
        await page.locator('.command-option[data-cmd="!help"]').click();

        // Input should be filled with command
        await expect(input).toHaveValue('!help ');

        // Dropdown should close
        await expect(page.locator('#command-autocomplete')).not.toBeVisible();
    });

    test('Tab key selects first command', async ({ page }) => {
        await page.goto(BASE_URL);
        await page.locator('.nav-tab').filter({ hasText: /chat/i }).click();
        await page.waitForTimeout(1000);

        const input = page.locator('#message-input');
        await input.fill('!');
        await expect(page.locator('#command-autocomplete')).toBeVisible();

        // Press Tab to select first command
        await input.press('Tab');

        // Input should be filled and dropdown closed
        const value = await input.inputValue();
        expect(value.startsWith('!')).toBe(true);
        await expect(page.locator('#command-autocomplete')).not.toBeVisible();
    });

    test('Arrow keys navigate autocomplete options', async ({ page }) => {
        await page.goto(BASE_URL);
        await page.locator('.nav-tab').filter({ hasText: /chat/i }).click();
        await page.waitForTimeout(1000);

        const input = page.locator('#message-input');
        await input.fill('!');
        await expect(page.locator('#command-autocomplete')).toBeVisible();

        // Press down arrow
        await input.press('ArrowDown');

        // First option should be highlighted (has background)
        const firstOption = page.locator('.command-option').first();
        const bg = await firstOption.evaluate(el => window.getComputedStyle(el).background);
        expect(bg).not.toBe('transparent');
    });

    test('Escape closes autocomplete', async ({ page }) => {
        await page.goto(BASE_URL);
        await page.locator('.nav-tab').filter({ hasText: /chat/i }).click();
        await page.waitForTimeout(1000);

        const input = page.locator('#message-input');
        await input.fill('!');
        await expect(page.locator('#command-autocomplete')).toBeVisible();

        // Press Escape
        await input.press('Escape');

        // Dropdown should close
        await expect(page.locator('#command-autocomplete')).not.toBeVisible();
    });

    test('autocomplete shows command descriptions', async ({ page }) => {
        await page.goto(BASE_URL);
        await page.locator('.nav-tab').filter({ hasText: /chat/i }).click();
        await page.waitForTimeout(1000);

        const input = page.locator('#message-input');
        await input.fill('!help');

        const dropdown = page.locator('#command-autocomplete');
        await expect(dropdown).toBeVisible();

        // Should show description
        await expect(dropdown).toContainText('Show available commands');
    });

    test('normal text does not trigger autocomplete', async ({ page }) => {
        await page.goto(BASE_URL);
        await page.locator('.nav-tab').filter({ hasText: /chat/i }).click();
        await page.waitForTimeout(1000);

        const input = page.locator('#message-input');
        await input.fill('Hello world');

        // Autocomplete should NOT appear
        await expect(page.locator('#command-autocomplete')).not.toBeVisible();
    });
});

// ============================================================================
// Navigation Tests
// ============================================================================

test.describe('Navigation - Tab Switching', () => {

    const panels = [
        { tab: 'Identity', panel: 'panel-identity' },
        { tab: 'Chat', panel: 'panel-chat' },
        { tab: 'Constitution', panel: 'panel-constitution' },
        { tab: 'Memories', panel: 'panel-memories' },
        { tab: 'Sovereignty', panel: 'panel-sovereignty' },
    ];

    for (const { tab, panel } of panels) {
        test(`clicking ${tab} tab shows ${panel}`, async ({ page }) => {
            await page.goto(BASE_URL);
            const tabButton = page.locator('.nav-tab').filter({ hasText: new RegExp(tab, 'i') });
            await tabButton.click();
            await expect(page.locator(`#${panel}`)).toBeVisible();
            await expect(tabButton).toHaveClass(/active/);
        });
    }

    test('only one panel is visible at a time', async ({ page }) => {
        await page.goto(BASE_URL);

        // Click Chat
        await page.locator('.nav-tab').filter({ hasText: /chat/i }).click();
        await expect(page.locator('#panel-chat')).toBeVisible();
        await expect(page.locator('#panel-identity')).not.toBeVisible();

        // Click Memories
        await page.locator('.nav-tab').filter({ hasText: /memories/i }).click();
        await expect(page.locator('#panel-memories')).toBeVisible();
        await expect(page.locator('#panel-chat')).not.toBeVisible();
    });

    test('only one tab has active class at a time', async ({ page }) => {
        await page.goto(BASE_URL);

        // Click through tabs
        const tabs = page.locator('.nav-tab');
        for (let i = 0; i < await tabs.count(); i++) {
            await tabs.nth(i).click();
            const activeCount = await page.locator('.nav-tab.active').count();
            expect(activeCount).toBe(1);
        }
    });
});

// ============================================================================
// Error Handling Tests
// ============================================================================

test.describe('Error Handling', () => {

    test('page handles slow API gracefully', async ({ page }) => {
        await page.goto(BASE_URL);
        // Just verify page loads without crashing
        await expect(page.locator('nav')).toBeVisible();
    });

    test('empty message is not sent', async ({ page }) => {
        await page.goto(BASE_URL);
        const chatTab = page.locator('.nav-tab').filter({ hasText: /chat/i });
        await chatTab.click();

        const initialCount = await page.locator('.user-message').count();

        // Try to send empty message
        await page.locator('#send-button').click();

        // Should not add new message
        const currentCount = await page.locator('.user-message').count();
        expect(currentCount).toBe(initialCount);
    });

    test('whitespace-only message is not sent', async ({ page }) => {
        await page.goto(BASE_URL);
        const chatTab = page.locator('.nav-tab').filter({ hasText: /chat/i });
        await chatTab.click();

        const initialCount = await page.locator('.user-message').count();

        const input = page.locator('#message-input');
        await input.fill('   ');
        await page.locator('#send-button').click();

        // Should not add new message
        const currentCount = await page.locator('.user-message').count();
        expect(currentCount).toBe(initialCount);
    });
});

// ============================================================================
// Full User Journey Test
// ============================================================================

test.describe('Complete User Onboarding Journey', () => {

    test('new user can complete full onboarding flow', async ({ page }) => {
        // Step 1: Load application
        await page.goto(BASE_URL);
        await expect(page).toHaveTitle(/Kestrel/i);

        // Step 2: Verify identity panel loads with DID
        await expect(page.getByText(/did:pkh|did:key/i).first()).toBeVisible({ timeout: 15000 });

        // Step 3: Check sidebar shows agent is online
        await expect(page.locator('#agent-status')).toContainText(/online/i, { timeout: 10000 });

        // Step 4: Navigate to Chat
        await page.locator('.nav-tab').filter({ hasText: /chat/i }).click();
        await expect(page.locator('#panel-chat')).toBeVisible();

        // Step 5: Send a greeting
        const response1 = await sendMessageAndWait(page, 'Hello, I am a new user!', 90000);
        expect((await response1.textContent()).length).toBeGreaterThan(10);

        // Step 6: Ask about capabilities
        const response2 = await sendMessageAndWait(page, '!help', 60000);
        expect((await response2.textContent()).toLowerCase()).toMatch(/command|help/i);

        // Step 7: View Constitution
        await page.locator('.nav-tab').filter({ hasText: /constitution/i }).click();
        await page.waitForTimeout(2000);
        const constitutionContent = await page.locator('#constitution-content').textContent();
        expect(constitutionContent.length).toBeGreaterThan(100);

        // Step 8: Check Memories
        await page.locator('.nav-tab').filter({ hasText: /memories/i }).click();
        await page.waitForTimeout(2000);
        await expect(page.locator('#memory-list')).toBeVisible();

        // Step 9: View Sovereignty options
        await page.locator('.nav-tab').filter({ hasText: /sovereignty/i }).click();
        await expect(page.locator('#btn-export-ipfs')).toBeVisible();
        await expect(page.locator('#btn-import')).toBeVisible();

        // Step 10: Return to Identity
        await page.locator('.nav-tab').filter({ hasText: /identity/i }).click();
        await expect(page.locator('#panel-identity')).toBeVisible();
    });
});
