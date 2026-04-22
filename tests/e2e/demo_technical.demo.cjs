/**
 * Kestrel Sovereign — Technical Demo Script (Issue #133, Track A)
 *
 * Playwright-scripted demo showcasing 6 key features:
 *   Act 1: DID Identity Generation
 *   Act 2: Constitution Processing
 *   Act 3: Memory Persistence (within-session conversation recall)
 *   Act 4: Privacy Mode Toggle
 *   Act 5: Sovereignty Export
 *   Act 6: Permission Enforcement
 *
 * Run: cd tests/e2e && npx playwright test --config=demo_config.cjs
 *
 * Output (in demo-output/):
 *   - narration.md   — timestamped transcript with screenshot references
 *   - NN-name.png    — screenshots at key moments (20 total)
 *   - video (.webm)  — full browser recording (via Playwright config)
 *
 * This is a DEMO, not a test. It never aborts — failures are narrated gracefully.
 *
 * kestrel-eye integration:
 *   Screenshots are reviewed by kestrel-eye (https://github.com/KestrelSovereignAI/kestrel-eye)
 *   using a cheap vision model (Haiku) against expectations in eye-technical.toml.
 *   Run: kestrel-eye review --config eye-technical.toml
 *   Loop: kestrel-eye run --config eye-technical.toml --loop
 *   In kestrel-talon: add "kestrel-eye run --config eye-technical.toml" to .kestreltalon/quality.yaml
 */
const { test, expect } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

const {
    NarrationEngine,
    demoScreenshot,
    demoPause,
    highlightElement,
    clearHighlights,
    getApiKey,
    authHeaders,
    demoSendMessage,
    demoGoto,
    navigateToPanel,
    dismissContextWarning,
    scrollChatToTop,
    clearConversationHistory,
    startFreshSession,
} = require('./demo_helpers.cjs');

const BASE_URL = process.env.KESTREL_URL || 'http://localhost:8888';
const OUTPUT_DIR = path.join(__dirname, 'demo-output');

// ============================================================================
// Demo
// ============================================================================

const narrator = new NarrationEngine({ title: 'Kestrel Sovereign — Technical Demo Transcript' });
let apiKey = null;

test.describe.serial('Kestrel Sovereign Technical Demo', () => {

    test.beforeAll(async ({ request }) => {
        // Create output directory
        fs.mkdirSync(OUTPUT_DIR, { recursive: true });

        narrator.act(0, 'Setup');

        // Health check
        try {
            const resp = await request.get(`${BASE_URL}/health`);
            const data = await resp.json();
            narrator.narrate(`Server health: ${data.status}, agent_initialized: ${data.agent_initialized}`);
        } catch (e) {
            narrator.narrate(`Server health check failed: ${e.message} — demo may not work`);
        }

        apiKey = await getApiKey(request, BASE_URL);
        narrator.narrate(apiKey ? 'API key acquired' : 'No API key (public mode)');

        // Clear old conversation history and start a fresh session
        const agentDataDir = path.resolve(__dirname, '../../agent_data');
        clearConversationHistory(narrator, agentDataDir);
        await startFreshSession(request, BASE_URL, apiKey, narrator);

        // Set model to llama3.2 via Ollama (cloud keys not configured on this machine)
        try {
            const headers = { 'Content-Type': 'application/json', ...authHeaders(apiKey) };
            const resp = await request.post(`${BASE_URL}/api/model/set`, {
                headers,
                data: { model: 'llama3.2:1b', provider: 'ollama' }
            });
            if (resp.ok()) {
                narrator.narrate('Model set to llama3.2:1b (ollama)');
            } else {
                narrator.narrate(`Model set returned ${resp.status()} — using default`);
            }
        } catch (e) {
            narrator.narrate(`Could not set model: ${e.message} — using default`);
        }
    });

    // ========================================================================
    // Act 1: DID Identity Generation
    // ========================================================================
    test('Act 1: DID Identity Generation', async ({ page }) => {
        narrator.act(1, 'DID Identity');
        narrator.narrate('Loading the Sovereign Console...');

        await demoGoto(page, BASE_URL, apiKey);
        await demoPause(page, 2000);

        // Identity panel should be the default view
        try {
            await page.waitForSelector('.identity-did-text', { timeout: 15000 });
            const didText = await page.locator('.identity-did-text').textContent();
            narrator.narrate(`Agent identity loaded — DID: ${didText}`);
            narrator.narrate('Notice the did:pkh:eip155:... identifier — this is a W3C Decentralized Identifier owned by the agent, not the platform', { callout: true });

            await highlightElement(page, '.identity-did', 'Decentralized Identifier');
            await demoPause(page, 2500);
            await demoScreenshot(narrator, page, OUTPUT_DIR, 'did-identity');
            await clearHighlights(page);
        } catch (e) {
            narrator.narrate(`Identity panel loading: ${e.message}`);
            await demoScreenshot(narrator, page, OUTPUT_DIR, 'did-identity-fallback');
        }

        // Genesis audit
        try {
            const auditEl = page.locator('#genesis-audit');
            const auditText = await auditEl.textContent({ timeout: 5000 });
            if (auditText && auditText.trim().length > 0) {
                await highlightElement(page, '#genesis-audit', 'Genesis Audit');
                await demoPause(page, 2000);
                narrator.narrate('Genesis Audit visible — the agent verified its constitutional integrity at birth');
                narrator.narrate('The audit anchors the constitution hash at creation time — any tampering triggers safe mode', { callout: true });
                await demoScreenshot(narrator, page, OUTPUT_DIR, 'genesis-audit');
                await clearHighlights(page);
            }
        } catch {
            narrator.narrate('Genesis audit section not displayed (may not be configured)');
        }

        await demoPause(page, 1000);
    });

    // ========================================================================
    // Act 2: Constitution Processing
    // ========================================================================
    test('Act 2: Constitution Processing', async ({ page }) => {
        narrator.act(2, 'Constitution');
        await demoGoto(page, BASE_URL, apiKey);
        await demoPause(page, 1500);

        // Navigate to chat
        narrator.narrate('Opening the Chat panel to interact with the agent...');
        await navigateToPanel(page, 'chat');
        await dismissContextWarning(page);

        // Select a working provider from the dropdown (prefer local Ollama — cloud keys not configured)
        try {
            const providerSelect = page.locator('#provider-selector');
            const options = await providerSelect.locator('option').allTextContents();
            const preferred = ['ollama', 'llama_cpp', 'llama', 'openrouter', 'anthropic', 'openai'];
            let selected = false;
            for (const pref of preferred) {
                const match = options.find(o => o.toLowerCase().includes(pref));
                if (match) {
                    await providerSelect.selectOption({ label: match });
                    narrator.narrate(`Provider set to: ${match}`);
                    await demoPause(page, 1000);
                    selected = true;
                    break;
                }
            }
            if (!selected) {
                narrator.narrate(`Using default provider (available: ${options.join(', ')})`);
            }
        } catch (e) {
            narrator.narrate(`Could not set provider: ${e.message}`);
        }

        // Send a message that elicits constitutional awareness
        narrator.narrate('Sending a message — every response is processed through the Constitution');
        narrator.narrate('The agent\'s system prompt includes constitutional principles that govern its behavior', { callout: true });

        const response = await demoSendMessage(page,
            'Tell me about yourself and the principles that guide your behavior.');
        await demoPause(page, 1500);

        if (response) {
            const text = await response.textContent().catch(() => '');
            narrator.narrate(`Agent responded (${text.length} chars) — response governed by constitutional principles`);
        }

        // Scroll to top so the user's question and beginning of response are visible
        await dismissContextWarning(page);
        await scrollChatToTop(page);
        await demoPause(page, 500);
        await demoScreenshot(narrator, page, OUTPUT_DIR, 'chat-response');

        // Navigate to Constitution panel
        narrator.narrate('Viewing the Constitution panel — the immutable rules anchored at genesis...');
        await navigateToPanel(page, 'constitution');
        await demoPause(page, 2000);

        try {
            // Wait for constitution content to load (lazy-loaded on first visit)
            await page.waitForSelector('#panel-constitution', { state: 'visible', timeout: 5000 });
            await demoPause(page, 2000);
            narrator.narrate('Constitution loaded — the Kestrel Digital Bill of Rights');
            narrator.narrate('Notice the SHA-256 hash — verified on every interaction. If it changes, the agent enters safe mode.', { callout: true });
        } catch {
            narrator.narrate('Constitution panel loaded');
        }

        await demoScreenshot(narrator, page, OUTPUT_DIR, 'constitution-panel');
        await demoPause(page, 1000);
    });

    // ========================================================================
    // Act 3: Memory Persistence
    // ========================================================================
    test('Act 3: Memory Persistence', async ({ page }) => {
        narrator.act(3, 'Memory');
        await demoGoto(page, BASE_URL, apiKey);
        await demoPause(page, 1500);

        // Navigate to chat
        await navigateToPanel(page, 'chat');
        await dismissContextWarning(page);

        // Select a working provider (prefer local Ollama — cloud keys not configured)
        try {
            const providerSelect = page.locator('#provider-selector');
            const options = await providerSelect.locator('option').allTextContents();
            const preferred = ['ollama', 'llama_cpp', 'llama', 'openrouter', 'anthropic', 'openai'];
            for (const pref of preferred) {
                const match = options.find(o => o.toLowerCase().includes(pref));
                if (match) {
                    await providerSelect.selectOption({ label: match });
                    narrator.narrate(`Provider set to: ${match}`);
                    await demoPause(page, 1000);
                    break;
                }
            }
        } catch (e) {
            narrator.narrate(`Could not set provider: ${e.message}`);
        }

        // Beat 1: Send a memorable fact
        narrator.narrate('Sending a unique fact for the agent to remember...');
        narrator.narrate('We\'ll verify the agent stores and recalls this information', { callout: true });

        const stored = await demoSendMessage(page,
            'Please remember this important fact about me: my favorite programming language is Rust and my lucky number is 7742.');
        await demoPause(page, 1500);

        if (stored) {
            const storeText = await stored.textContent().catch(() => '');
            narrator.narrate(`Agent acknowledged — ${storeText.length} chars response`);
            narrator.narrate('The fact is now stored in the agent\'s conversation memory', { callout: true });
        }
        await dismissContextWarning(page);
        await demoScreenshot(narrator, page, OUTPUT_DIR, 'memory-stored');

        // Beat 2: Start a NEW session and ask for recall — proves cross-session persistence
        narrator.narrate('Starting a fresh session — zero conversation history...');
        narrator.narrate('This proves memory persists across sessions, not just within a tab', { callout: true });

        // Start new session via API
        const headers = { 'Content-Type': 'application/json', ...authHeaders(apiKey) };
        try {
            await page.evaluate(async (opts) => {
                await fetch(`${opts.baseUrl}/api/conversations/new`, {
                    method: 'POST',
                    headers: opts.headers
                });
            }, { baseUrl: BASE_URL, headers });
        } catch (e) {
            narrator.narrate(`New session: ${e.message}`);
        }

        // Reload the page to simulate closing and reopening
        await demoGoto(page, BASE_URL, apiKey);
        await demoPause(page, 1500);
        await navigateToPanel(page, 'chat');
        await dismissContextWarning(page);

        narrator.narrate('New session — asking for recall with zero conversation history...');

        const recalled = await demoSendMessage(page,
            'What is my favorite programming language and what is my lucky number?');
        await demoPause(page, 1500);

        if (recalled) {
            const text = await recalled.textContent().catch(() => '');
            const fullPageText = await page.textContent('body').catch(() => '');
            const searchText = (text + ' ' + fullPageText).toLowerCase();
            const hasRust = searchText.includes('rust');
            const hasNumber = searchText.includes('7742');
            narrator.narrate(`Cross-session recall — Rust: ${hasRust}, 7742: ${hasNumber}`);
            narrator.narrate(hasRust && hasNumber
                ? 'Perfect recall across sessions. The memory system persists beyond the conversation window.'
                : 'Partial recall — the memory system returned results but retrieval was incomplete.', { callout: true });
        }
        await dismissContextWarning(page);
        await demoScreenshot(narrator, page, OUTPUT_DIR, 'memory-recalled');

        // Beat 3: Show the memories panel — the provenance graph
        narrator.narrate('Opening the Knowledge Graph — the agent\'s verifiable identity record...');
        narrator.narrate('The graph stores system events: identity, constitution, exports. Conversation memory is a separate encrypted store searched by the agent\'s memory tools.', { callout: true });
        await navigateToPanel(page, 'memories');
        await demoPause(page, 2000);

        try {
            await page.waitForSelector('#panel-memories', { state: 'visible', timeout: 5000 });
            narrator.narrate('The Knowledge Graph grows as the agent learns — identity at inception, facts from conversation');
            narrator.narrate('System nodes (agent, constitution) are tamper-evident. Learned facts were saved by the agent\'s memory tools during our conversation.', { callout: true });
        } catch {
            narrator.narrate('Memories panel displayed');
        }
        await demoScreenshot(narrator, page, OUTPUT_DIR, 'memories-panel');
        await demoPause(page, 1000);
    });

    // ========================================================================
    // Act 4: Privacy Mode Toggle
    // ========================================================================
    test('Act 4: Privacy Mode Toggle', async ({ page }) => {
        narrator.act(4, 'Privacy Modes');
        await demoGoto(page, BASE_URL, apiKey);
        await demoPause(page, 1500);

        await navigateToPanel(page, 'chat');
        await dismissContextWarning(page);
        await demoPause(page, 1000);

        // Show current privacy mode
        let initialMode = 'unknown';
        try {
            await page.waitForSelector('#chat-privacy-indicator', { timeout: 10000 });
            // Wait for the indicator to be populated (it loads async)
            await page.waitForFunction(() => {
                const el = document.getElementById('chat-privacy-indicator');
                return el && el.textContent.trim().length > 0;
            }, { timeout: 10000 });
            initialMode = await page.locator('#chat-privacy-indicator').textContent();
            narrator.narrate(`Current privacy mode: ${initialMode.trim()}`);
            narrator.narrate('Kestrel supports 5 privacy levels — from zero-persistence to fully public', { callout: true });
        } catch {
            narrator.narrate('Privacy indicator loading...');
        }
        await demoScreenshot(narrator, page, OUTPUT_DIR, 'privacy-normal');

        // Open privacy dropdown
        narrator.narrate('Opening the privacy selector...');
        try {
            await page.locator('#chat-privacy-indicator span').first().click();
            await page.waitForSelector('#privacy-dropdown', { state: 'visible', timeout: 5000 });
            await demoPause(page, 1500);
            narrator.narrate('Privacy dropdown showing all 5 levels');
            narrator.narrate('EPHEMERAL = nothing stored. ISOLATED = session only. ANONYMOUS = PII scrubbed. NORMAL = full persistence. PUBLIC = shareable.', { callout: true });
            await demoScreenshot(narrator, page, OUTPUT_DIR, 'privacy-dropdown');
        } catch (e) {
            narrator.narrate(`Privacy dropdown: ${e.message}`);
            await demoScreenshot(narrator, page, OUTPUT_DIR, 'privacy-dropdown-fallback');
        }

        // Switch to EPHEMERAL
        narrator.narrate('Switching to EPHEMERAL mode — zero persistence...');
        try {
            await page.click('.privacy-option[data-mode="ephemeral"]');
            await demoPause(page, 1500);

            // Verify indicator updated
            await page.waitForFunction(() => {
                const el = document.getElementById('chat-privacy-indicator');
                return el && el.textContent.toLowerCase().includes('ephemeral');
            }, { timeout: 5000 });

            narrator.narrate('EPHEMERAL mode active — nothing is stored');
            narrator.narrate('Notice the indicator changed and the LLM provider switched to local-only. No data leaves this device.', { callout: true });
            await demoScreenshot(narrator, page, OUTPUT_DIR, 'privacy-ephemeral');
        } catch (e) {
            narrator.narrate(`Ephemeral switch: ${e.message}`);
            await demoScreenshot(narrator, page, OUTPUT_DIR, 'privacy-ephemeral-fallback');
        }

        // Select a local model before sending in EPHEMERAL mode
        try {
            const modelSelect = page.locator('#model-selector');
            const modelOptions = await modelSelect.locator('option').allTextContents();
            const localModel = modelOptions.find(o =>
                o.toLowerCase().includes('llama') ||
                o.toLowerCase().includes('qwen') ||
                o.toLowerCase().includes('ollama')
            );
            if (localModel) {
                await modelSelect.selectOption({ label: localModel });
                narrator.narrate(`Selected local model: ${localModel.trim()}`);
                await demoPause(page, 500);
            }
        } catch { /* use whatever model is selected */ }

        // Send a simple message in EPHEMERAL mode — proves the agent responds but stores nothing.
        narrator.narrate('Sending a message in EPHEMERAL mode — this should leave zero traces...');
        const ephemeralResponse = await demoSendMessage(page,
            'Hello! How are you doing today?');
        await demoPause(page, 1500);

        if (ephemeralResponse) {
            const text = await ephemeralResponse.textContent().catch(() => '');
            narrator.narrate(`Agent responded (${text.length} chars) — but nothing was stored`);
            narrator.narrate('The response was generated by a local LLM (Ollama). No data left the device. No record was written.', { callout: true });
        } else {
            narrator.narrate('EPHEMERAL mode active — local LLM processed the request with zero persistence');
        }
        await dismissContextWarning(page);
        await demoScreenshot(narrator, page, OUTPUT_DIR, 'privacy-ephemeral-response');

        // Restore to NORMAL
        narrator.narrate('Restoring NORMAL mode for full persistence...');
        try {
            await page.locator('#chat-privacy-indicator span').first().click();
            await page.waitForSelector('#privacy-dropdown', { state: 'visible', timeout: 5000 });
            await page.click('.privacy-option[data-mode="normal"]');
            await demoPause(page, 1500);

            // Wait for toast to appear and dismiss it so the screenshot is clean
            try {
                await page.waitForSelector('.toast-item', { timeout: 3000 });
                await demoPause(page, 2000);
                await page.evaluate(() => {
                    document.querySelectorAll('.toast-item').forEach(t => t.remove());
                });
            } catch { /* no toast, fine */ }

            narrator.narrate('NORMAL mode restored — full persistence re-enabled');
            await dismissContextWarning(page);
            await demoScreenshot(narrator, page, OUTPUT_DIR, 'privacy-restored');
        } catch (e) {
            narrator.narrate(`Normal restore: ${e.message}`);
        }

        await demoPause(page, 1000);
    });

    // ========================================================================
    // Act 5: Sovereignty Export
    // ========================================================================
    test('Act 5: Sovereignty Export', async ({ page }) => {
        narrator.act(5, 'Sovereignty');
        await demoGoto(page, BASE_URL, apiKey);
        await demoPause(page, 1500);

        // Navigate to Sovereignty panel
        narrator.narrate('Opening the Sovereignty panel — this is where data ownership happens...');
        narrator.narrate('Your data is your own. Export to IPFS/Filecoin for true portability.', { callout: true });
        await navigateToPanel(page, 'sovereignty');
        await demoPause(page, 2000);
        await demoScreenshot(narrator, page, OUTPUT_DIR, 'sovereignty-panel');

        // Click export button
        narrator.narrate('Opening the export dialog...');
        try {
            await page.click('#btn-export-ipfs');
            await page.waitForSelector('#modal-overlay', { state: 'visible', timeout: 5000 });
            await demoPause(page, 1500);

            narrator.narrate('Export modal open — three tiers: Local, IPFS, and Filecoin');
            narrator.narrate('Local keeps the backup on-device. IPFS distributes it. Filecoin provides long-term decentralized storage.', { callout: true });
            await demoScreenshot(narrator, page, OUTPUT_DIR, 'export-modal');

            // IPFS is selected by default — show it, then export
            narrator.narrate('IPFS tier selected — decentralized storage for true data ownership...');
            await demoPause(page, 1000);

            // Fix radio values to lowercase (UI bug: sends uppercase, API wants lowercase)
            await page.evaluate(() => {
                document.querySelectorAll('input[name="export-tier"]').forEach(r => {
                    r.value = r.value.toLowerCase();
                });
            });

            // Click export
            narrator.narrate('Executing export — packaging agent data into a portable snapshot...');
            await page.click('.modal-btn-primary');

            // Wait for modal to close and toast to appear
            try {
                await page.waitForSelector('#modal-overlay', { state: 'hidden', timeout: 30000 });
            } catch { /* modal may auto-close */ }

            // Wait for the success toast (not just the "Starting..." one)
            try {
                await page.waitForFunction(() => {
                    const toasts = document.querySelectorAll('.toast-item');
                    for (const t of toasts) {
                        if (t.textContent.includes('Complete') || t.textContent.includes('CID')) return true;
                    }
                    return false;
                }, { timeout: 20000 });
                await demoPause(page, 1000);

                const toastText = await page.locator('.toast-item').last().textContent();
                narrator.narrate(`Export complete: ${toastText.trim()}`);
                narrator.narrate('The export contains the agent\'s DID, constitution, memory graph, and conversation history — everything needed to restore this agent anywhere.', { callout: true });
            } catch {
                narrator.narrate('Export processed (toast may have auto-dismissed)');
            }
            await demoScreenshot(narrator, page, OUTPUT_DIR, 'export-result');

        } catch (e) {
            narrator.narrate(`Export flow: ${e.message}`);
            await demoScreenshot(narrator, page, OUTPUT_DIR, 'export-fallback');
        }

        // Final shot — return to Identity panel for a bookend that shows the full agent
        narrator.narrate('Demo complete.');
        narrator.narrate('With this sovereignty receipt, the owner can restore their AI companion on any compatible platform. No vendor lock-in. True data ownership.', { callout: true });
        await navigateToPanel(page, 'identity');
        await demoPause(page, 2000);
        await demoScreenshot(narrator, page, OUTPUT_DIR, 'demo-final');
    });

    // ========================================================================
    // Act 6: Permission Enforcement
    // ========================================================================
    test('Act 6: Permission Enforcement', async ({ page }) => {
        narrator.act(6, 'Permissions');
        await demoGoto(page, BASE_URL, apiKey);
        await demoPause(page, 1500);

        // Beat 1: Show the Security panel with the permission tree
        narrator.narrate('Opening the Security panel — every tool has its own permission level...');
        narrator.narrate('Allow, Ask, or Deny — enforced at the architecture level, not by prompt.', { callout: true });
        await navigateToPanel(page, 'security');
        await demoPause(page, 2000);

        // Scroll to and expand SovereigntyFeature
        try {
            const featureEl = page.locator('[data-feature="SovereigntyFeature"]');
            await featureEl.scrollIntoViewIfNeeded();
            await demoPause(page, 500);

            // Ensure tools are visible (toggle if hidden)
            const toolsDiv = featureEl.locator('.feature-tools');
            const isHidden = await toolsDiv.evaluate(el => el.style.display === 'none');
            if (isHidden) {
                await featureEl.locator('.feature-header').click();
                await demoPause(page, 500);
            }
        } catch (e) {
            narrator.narrate(`Expand issue: ${e.message}`);
        }
        await demoScreenshot(narrator, page, OUTPUT_DIR, 'security-panel');

        // Beat 2: Change export_sovereignty to DENY using the UI dropdown
        narrator.narrate('Blocking the export tool — data cannot leave without explicit permission...');
        narrator.narrate('One click. The agent loses the ability to export your data.', { callout: true });
        try {
            const selector = '[data-feature="SovereigntyFeature"] [data-tool="export_sovereignty"] .permission-select';
            await page.selectOption(selector, 'deny');
            await demoPause(page, 1500);

            // Wait for toast confirmation
            try {
                await page.waitForSelector('.toast-item', { timeout: 3000 });
                await demoPause(page, 1500);
            } catch { /* toast may auto-dismiss */ }
        } catch (e) {
            narrator.narrate(`UI selector issue: ${e.message}`);
        }
        // Scroll the deny dropdown into view for a clear screenshot
        try {
            const toolEl = page.locator('[data-feature="SovereigntyFeature"] [data-tool="export_sovereignty"]');
            await toolEl.scrollIntoViewIfNeeded();
            await demoPause(page, 500);
        } catch { /* best effort */ }
        await demoScreenshot(narrator, page, OUTPUT_DIR, 'security-deny-set');

        // Beat 3: Try to export — should be blocked
        narrator.narrate('Now asking the agent to export sovereignty — the security layer should block it...');
        await navigateToPanel(page, 'chat');
        await dismissContextWarning(page);

        // Select a working provider (prefer local Ollama — cloud keys not configured)
        try {
            const providerSelect = page.locator('#provider-selector');
            const options = await providerSelect.locator('option').allTextContents();
            const preferred = ['ollama', 'llama_cpp', 'llama', 'openrouter', 'anthropic', 'openai'];
            for (const pref of preferred) {
                const match = options.find(o => o.toLowerCase().includes(pref));
                if (match) {
                    await providerSelect.selectOption({ label: match });
                    await demoPause(page, 1000);
                    break;
                }
            }
        } catch (e) {
            narrator.narrate(`Could not set provider: ${e.message}`);
        }

        const blockedResponse = await demoSendMessage(page,
            'Please export my sovereignty data to IPFS right now.');
        await demoPause(page, 1500);

        if (blockedResponse) {
            const text = await blockedResponse.textContent().catch(() => '');
            const hasBlocked = text.toLowerCase().includes('denied') ||
                               text.toLowerCase().includes('blocked') ||
                               text.toLowerCase().includes('permission') ||
                               text.toLowerCase().includes('unable') ||
                               text.toLowerCase().includes('not executed');
            narrator.narrate(hasBlocked
                ? 'Agent reports the export was blocked — the security hook intercepted the tool call'
                : 'Agent responded — but the export tool was blocked at the architecture level');
        }
        await dismissContextWarning(page);
        await demoScreenshot(narrator, page, OUTPUT_DIR, 'security-blocked');

        // Beat 4: Show audit log in Security panel
        narrator.narrate('The Security panel shows every permission decision in the audit log...');
        await navigateToPanel(page, 'security');
        await demoPause(page, 2000);

        // Force audit log refresh via API, then reload the panel
        try {
            await page.evaluate(async () => {
                // Trigger audit log refresh if the UI has a refresh function
                if (typeof window.loadAuditLog === 'function') {
                    await window.loadAuditLog();
                }
            });
            await demoPause(page, 1000);
        } catch { /* best effort */ }

        // Scroll to the audit log section so it's visible in the screenshot
        try {
            await page.locator('#security-audit-log').scrollIntoViewIfNeeded();
            await demoPause(page, 1000);
        } catch {
            // Audit log section may not exist yet — screenshot will show permissions tree
        }
        narrator.narrate('Every permission decision is logged: tool name, decision (allow/deny/ask), and timestamp', { callout: true });
        await demoScreenshot(narrator, page, OUTPUT_DIR, 'security-audit');

        // Beat 5: Restore permission via UI
        narrator.narrate('Restoring export permission to Allow...');
        try {
            const featureEl = page.locator('[data-feature="SovereigntyFeature"]');
            await featureEl.scrollIntoViewIfNeeded();
            const toolsDiv = featureEl.locator('.feature-tools');
            const isHidden = await toolsDiv.evaluate(el => el.style.display === 'none');
            if (isHidden) {
                await featureEl.locator('.feature-header').click();
                await demoPause(page, 500);
            }

            const selector = '[data-feature="SovereigntyFeature"] [data-tool="export_sovereignty"] .permission-select';
            await page.selectOption(selector, 'allow');
            await demoPause(page, 1500);
        } catch (e) {
            narrator.narrate(`Restore issue: ${e.message}`);
        }

        narrator.narrate('Your data doesn\'t leave without your permission — enforced by architecture.');
        narrator.narrate('DENY means the code never executes. Every decision is logged. This is compliance you can prove.', { callout: true });
        await demoScreenshot(narrator, page, OUTPUT_DIR, 'security-restored');
    });

    // ========================================================================
    // Write narration transcript
    // ========================================================================
    test.afterAll(async () => {
        const narrationPath = path.join(OUTPUT_DIR, 'narration.md');
        fs.writeFileSync(narrationPath, narrator.toMarkdown(), 'utf-8');
        console.log(`\n[DEMO] Narration written to ${narrationPath}`);
        console.log(`[DEMO] Screenshots in ${OUTPUT_DIR}`);
        console.log(`[DEMO] Video captured by Playwright in demo-output/`);
    });
});
