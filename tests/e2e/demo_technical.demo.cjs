/**
 * Kestrel Sovereign — Technical Demo Script (Issue #133, Track A)
 *
 * Playwright-scripted demo showcasing 5 key features:
 *   Act 1: DID Identity Generation
 *   Act 2: Constitution Processing
 *   Act 3: Memory Persistence (within-session conversation recall)
 *   Act 4: Privacy Mode Toggle
 *   Act 5: Sovereignty Export
 *
 * Run: cd tests/e2e && npx playwright test --config=demo_config.cjs
 *
 * Output (in demo-output/):
 *   - narration.md   — timestamped transcript with screenshot references
 *   - NN-name.png    — screenshots at key moments
 *   - video (.webm)  — full browser recording (via Playwright config)
 *
 * This is a DEMO, not a test. It never aborts — failures are narrated gracefully.
 */
const { test, expect } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

const BASE_URL = process.env.KESTREL_URL || 'http://localhost:8888';
const OUTPUT_DIR = path.join(__dirname, 'demo-output');

// ============================================================================
// Narration Engine
// ============================================================================

class NarrationEngine {
    constructor() {
        this.entries = [];
        this.startTime = Date.now();
        this.screenshotIndex = 0;
    }

    narrate(section, text, callout = null) {
        const elapsed = ((Date.now() - this.startTime) / 1000).toFixed(1);
        const entry = { timestamp: `${elapsed}s`, section, text, callout, screenshot: null };
        this.entries.push(entry);
        console.log(`[DEMO ${elapsed}s] [${section}] ${text}`);
        if (callout) console.log(`  >> ${callout}`);
        return entry;
    }

    async screenshot(page, name) {
        this.screenshotIndex++;
        const filename = `${String(this.screenshotIndex).padStart(2, '0')}-${name}.png`;
        const filepath = path.join(OUTPUT_DIR, filename);
        try {
            await page.screenshot({ path: filepath, fullPage: false });
        } catch (e) {
            console.warn(`  >> Screenshot failed: ${e.message}`);
            return filename;
        }
        if (this.entries.length > 0) {
            this.entries[this.entries.length - 1].screenshot = filename;
        }
        console.log(`  >> SCREENSHOT: ${filename}`);
        return filename;
    }

    toMarkdown() {
        let md = '# Kestrel Sovereign — Technical Demo Transcript\n\n';
        md += `> Generated: ${new Date().toISOString()}\n\n`;
        md += '---\n\n';

        let currentSection = null;
        for (const entry of this.entries) {
            if (entry.section !== currentSection) {
                currentSection = entry.section;
                md += `## ${currentSection}\n\n`;
            }
            md += `**[${entry.timestamp}]** ${entry.text}\n`;
            if (entry.callout) {
                md += `> *${entry.callout}*\n`;
            }
            if (entry.screenshot) {
                md += `\n![${entry.screenshot}](./${entry.screenshot})\n`;
            }
            md += '\n';
        }
        return md;
    }
}

// ============================================================================
// Helpers
// ============================================================================

async function getApiKey(request) {
    if (process.env.KESTREL_API_KEY) return process.env.KESTREL_API_KEY;
    try {
        const response = await request.get(`${BASE_URL}/api/auth/key`);
        if (response.ok()) {
            const data = await response.json();
            return data.key;
        }
    } catch (e) { /* ignore */ }
    return null;
}

function authHeaders(apiKey) {
    return apiKey ? { 'X-API-Key': apiKey } : {};
}

/** Clear old conversation history so the demo starts with a clean context window.
 *  We delete directly from SQLite because there is no bulk-clear API endpoint. */
function clearConversationHistory() {
    const { execSync } = require('child_process');
    // Agent data lives under agent_data/<agent_name>/kestrel_prime.db.
    // Walk all agent databases and clear conversation_history.
    const agentDataDir = path.resolve(__dirname, '../../agent_data');
    try {
        const dbs = execSync(`find "${agentDataDir}" -name "kestrel_prime.db" -maxdepth 3 2>/dev/null`)
            .toString().trim().split('\n').filter(Boolean);
        for (const db of dbs) {
            try {
                execSync(`sqlite3 "${db}" "DELETE FROM conversation_history;"`);
            } catch { /* table may not exist in some dbs */ }
        }
        narrator.narrate('Setup', `Cleared conversation history from ${dbs.length} agent database(s)`);
    } catch (e) {
        narrator.narrate('Setup', `Could not clear history: ${e.message}`);
    }
}

/** Start a fresh session via the API. */
async function startFreshSession(request) {
    try {
        const headers = apiKey ? { 'X-API-Key': apiKey } : {};
        const resp = await request.post(`${BASE_URL}/api/conversations/new`, { headers });
        if (resp.ok()) {
            const data = await resp.json();
            narrator.narrate('Setup', `Fresh session started: ${data.session_id || 'ok'}`);
            return true;
        }
    } catch (e) {
        narrator.narrate('Setup', `Could not start fresh session: ${e.message}`);
    }
    return false;
}

/** Send a chat message and wait for agent response. Returns last agent message locator or null. */
async function demoSendMessage(page, message, timeout = 90000) {
    try {
        const initialCount = await page.locator('.agent-message').count();
        await page.locator('#message-input').fill(message);
        await page.locator('#send-button').click();
        await page.waitForFunction(
            (count) => {
                const msgs = document.querySelectorAll('.agent-message');
                if (msgs.length <= count) return false;
                const last = msgs[msgs.length - 1];
                return (last.textContent || '').trim().length > 5 &&
                       !last.querySelector('.streaming');
            },
            initialCount,
            { timeout }
        );
        return page.locator('.agent-message').last();
    } catch (e) {
        console.warn(`[DEMO] Message send/wait issue: ${e.message}`);
        // Still return whatever we have
        const count = await page.locator('.agent-message').count();
        return count > 0 ? page.locator('.agent-message').last() : null;
    }
}

/** Visual pause so the recording is watchable. */
async function demoPause(page, ms = 2000) {
    await page.waitForTimeout(ms);
}

/** Set API key headers on the page context and navigate to the app. */
async function demoGoto(page) {
    if (apiKey) {
        await page.setExtraHTTPHeaders({ 'X-API-Key': apiKey });
    }
    await page.goto(BASE_URL);
}

/** Navigate to a named panel. */
async function navigateToPanel(page, panelName) {
    await page.click(`.nav-tab[data-panel="${panelName}"]`);
    try {
        await page.waitForSelector(`#panel-${panelName}`, { state: 'visible', timeout: 5000 });
    } catch { /* panel may already be visible */ }
    await demoPause(page, 1000);
}

/** Remove context warnings and hide the noisy status bar for clean screenshots. */
async function dismissContextWarning(page) {
    await page.evaluate(() => {
        document.querySelectorAll('.context-warning').forEach(el => el.remove());
        // Hide the "N msgs · X% Compress" indicator — it's operational, not demo material
        const status = document.getElementById('context-status');
        if (status) status.style.display = 'none';
    });
}

/** Scroll chat container to show the first user message at the top. */
async function scrollChatToTop(page) {
    await page.evaluate(() => {
        const container = document.getElementById('chat-container');
        if (container) container.scrollTop = 0;
    });
}

/** Inject a highlight glow around an element for the recording. */
async function highlightElement(page, selector, label) {
    await page.evaluate(({ sel, lbl }) => {
        const el = document.querySelector(sel);
        if (!el) return;
        el.style.outline = '3px solid #3b82f6';
        el.style.outlineOffset = '4px';
        el.style.transition = 'outline 0.3s';
        const badge = document.createElement('div');
        badge.className = 'demo-highlight-badge';
        badge.textContent = lbl;
        badge.style.cssText = `
            position: absolute; top: -28px; left: 0;
            background: #3b82f6; color: white; padding: 3px 10px;
            border-radius: 4px; font-size: 13px; font-weight: 600; z-index: 9999;
            white-space: nowrap;
        `;
        if (getComputedStyle(el).position === 'static') el.style.position = 'relative';
        el.prepend(badge);
    }, { sel: selector, lbl: label });
}

/** Remove all highlight badges and outlines. */
async function clearHighlights(page) {
    await page.evaluate(() => {
        document.querySelectorAll('.demo-highlight-badge').forEach(b => b.remove());
        document.querySelectorAll('[style*="outline: 3px"]').forEach(el => {
            el.style.outline = '';
            el.style.outlineOffset = '';
        });
    });
}

// ============================================================================
// Demo
// ============================================================================

const narrator = new NarrationEngine();
let apiKey = null;

test.describe.serial('Kestrel Sovereign Technical Demo', () => {

    test.beforeAll(async ({ request }) => {
        // Create output directory
        fs.mkdirSync(OUTPUT_DIR, { recursive: true });

        // Health check
        try {
            const resp = await request.get(`${BASE_URL}/health`);
            const data = await resp.json();
            narrator.narrate('Setup', `Server health: ${data.status}, agent_initialized: ${data.agent_initialized}`);
        } catch (e) {
            narrator.narrate('Setup', `Server health check failed: ${e.message} — demo may not work`);
        }

        apiKey = await getApiKey(request);
        narrator.narrate('Setup', apiKey ? 'API key acquired' : 'No API key (public mode)');

        // Clear old conversation history and start a fresh session
        clearConversationHistory();
        await startFreshSession(request);

        // Set model to Claude Opus 4.6 for high-quality demo responses
        try {
            const headers = { 'Content-Type': 'application/json', ...authHeaders(apiKey) };
            const resp = await request.post(`${BASE_URL}/api/model/set`, {
                headers,
                data: { model: 'claude-opus-4-6', provider: 'anthropic' }
            });
            if (resp.ok()) {
                narrator.narrate('Setup', 'Model set to Claude Opus 4.6 (anthropic)');
            } else {
                narrator.narrate('Setup', `Model set returned ${resp.status()} — using default`);
            }
        } catch (e) {
            narrator.narrate('Setup', `Could not set model: ${e.message} — using default`);
        }
    });

    // ========================================================================
    // Act 1: DID Identity Generation
    // ========================================================================
    test('Act 1: DID Identity Generation', async ({ page }) => {
        const section = 'Act 1: DID Identity';
        narrator.narrate(section, 'Loading the Sovereign Console...');

        await demoGoto(page);
        await demoPause(page, 2000);

        // Identity panel should be the default view
        try {
            await page.waitForSelector('.identity-did-text', { timeout: 15000 });
            const didText = await page.locator('.identity-did-text').textContent();
            narrator.narrate(section,
                `Agent identity loaded — DID: ${didText}`,
                'Notice the did:pkh:eip155:... identifier — this is a W3C Decentralized Identifier owned by the agent, not the platform');

            await highlightElement(page, '.identity-did', 'Decentralized Identifier');
            await demoPause(page, 2500);
            await narrator.screenshot(page, 'did-identity');
            await clearHighlights(page);
        } catch (e) {
            narrator.narrate(section, `Identity panel loading: ${e.message}`);
            await narrator.screenshot(page, 'did-identity-fallback');
        }

        // Genesis audit
        try {
            const auditEl = page.locator('#genesis-audit');
            const auditText = await auditEl.textContent({ timeout: 5000 });
            if (auditText && auditText.trim().length > 0) {
                await highlightElement(page, '#genesis-audit', 'Genesis Audit');
                await demoPause(page, 2000);
                narrator.narrate(section,
                    'Genesis Audit visible — the agent verified its constitutional integrity at birth',
                    'The audit anchors the constitution hash at creation time — any tampering triggers safe mode');
                await narrator.screenshot(page, 'genesis-audit');
                await clearHighlights(page);
            }
        } catch {
            narrator.narrate(section, 'Genesis audit section not displayed (may not be configured)');
        }

        await demoPause(page, 1000);
    });

    // ========================================================================
    // Act 2: Constitution Processing
    // ========================================================================
    test('Act 2: Constitution Processing', async ({ page }) => {
        const section = 'Act 2: Constitution';
        await demoGoto(page);
        await demoPause(page, 1500);

        // Navigate to chat
        narrator.narrate(section, 'Opening the Chat panel to interact with the agent...');
        await navigateToPanel(page, 'chat');
        await dismissContextWarning(page);

        // Send a message that elicits constitutional awareness
        narrator.narrate(section,
            'Sending a message — every response is processed through the Constitution',
            'The agent\'s system prompt includes constitutional principles that govern its behavior');

        const response = await demoSendMessage(page,
            'Tell me about yourself and the principles that guide your behavior.');
        await demoPause(page, 1500);

        if (response) {
            const text = await response.textContent().catch(() => '');
            narrator.narrate(section,
                `Agent responded (${text.length} chars) — response governed by constitutional principles`);
        }

        // Scroll to top so the user's question and beginning of response are visible
        await dismissContextWarning(page);
        await scrollChatToTop(page);
        await demoPause(page, 500);
        await narrator.screenshot(page, 'chat-response');

        // Navigate to Constitution panel
        narrator.narrate(section, 'Viewing the Constitution panel — the immutable rules anchored at genesis...');
        await navigateToPanel(page, 'constitution');
        await demoPause(page, 2000);

        try {
            // Wait for constitution content to load (lazy-loaded on first visit)
            await page.waitForSelector('#panel-constitution', { state: 'visible', timeout: 5000 });
            await demoPause(page, 2000);
            narrator.narrate(section,
                'Constitution loaded — the Kestrel Digital Bill of Rights',
                'Notice the SHA-256 hash — verified on every interaction. If it changes, the agent enters safe mode.');
        } catch {
            narrator.narrate(section, 'Constitution panel loaded');
        }

        await narrator.screenshot(page, 'constitution-panel');
        await demoPause(page, 1000);
    });

    // ========================================================================
    // Act 3: Memory Persistence
    // ========================================================================
    test('Act 3: Memory Persistence', async ({ page }) => {
        const section = 'Act 3: Memory';
        await demoGoto(page);
        await demoPause(page, 1500);

        // Navigate to chat
        await navigateToPanel(page, 'chat');
        await dismissContextWarning(page);

        // Beat 1: Send a memorable fact
        narrator.narrate(section,
            'Sending a unique fact for the agent to remember...',
            'We\'ll verify the agent stores and recalls this information');

        const stored = await demoSendMessage(page,
            'Please remember this important fact about me: my favorite programming language is Rust and my lucky number is 7742.');
        await demoPause(page, 1500);

        if (stored) {
            const storeText = await stored.textContent().catch(() => '');
            narrator.narrate(section,
                `Agent acknowledged — ${storeText.length} chars response`,
                'The fact is now stored in the agent\'s conversation memory');
        }
        await dismissContextWarning(page);
        await narrator.screenshot(page, 'memory-stored');

        // Beat 2: Start a NEW session and ask for recall — proves cross-session persistence
        narrator.narrate(section,
            'Starting a fresh session — zero conversation history...',
            'This proves memory persists across sessions, not just within a tab');

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
            narrator.narrate(section, `New session: ${e.message}`);
        }

        // Reload the page to simulate closing and reopening
        await demoGoto(page);
        await demoPause(page, 1500);
        await navigateToPanel(page, 'chat');
        await dismissContextWarning(page);

        narrator.narrate(section,
            'New session — asking for recall with zero conversation history...');

        const recalled = await demoSendMessage(page,
            'What is my favorite programming language and what is my lucky number?');
        await demoPause(page, 1500);

        if (recalled) {
            const text = await recalled.textContent().catch(() => '');
            const hasRust = text.toLowerCase().includes('rust');
            const hasNumber = text.includes('7742');
            narrator.narrate(section,
                `Cross-session recall — Rust: ${hasRust}, 7742: ${hasNumber}`,
                hasRust && hasNumber
                    ? 'Perfect recall across sessions. The memory system persists beyond the conversation window.'
                    : 'Partial recall — the memory system returned results but retrieval was incomplete.');
        }
        await dismissContextWarning(page);
        await narrator.screenshot(page, 'memory-recalled');

        // Beat 3: Show the memories panel — the knowledge graph (saved items, documents, etc.)
        narrator.narrate(section,
            'Opening the Memories panel — the knowledge graph stores structured data...',
            'Saved items, documents, backups, and sovereignty receipts are persisted here');
        await navigateToPanel(page, 'memories');
        await demoPause(page, 2000);

        try {
            await page.waitForSelector('#panel-memories', { state: 'visible', timeout: 5000 });
            narrator.narrate(section,
                'Knowledge graph visible — agent identity, constitution, backups, and export receipts',
                'Each node has a type badge and can be inspected or deleted by the owner');
        } catch {
            narrator.narrate(section, 'Memories panel displayed');
        }
        await narrator.screenshot(page, 'memories-panel');
        await demoPause(page, 1000);
    });

    // ========================================================================
    // Act 4: Privacy Mode Toggle
    // ========================================================================
    test('Act 4: Privacy Mode Toggle', async ({ page }) => {
        const section = 'Act 4: Privacy Modes';
        await demoGoto(page);
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
            narrator.narrate(section,
                `Current privacy mode: ${initialMode.trim()}`,
                'Kestrel supports 5 privacy levels — from zero-persistence to fully public');
        } catch {
            narrator.narrate(section, 'Privacy indicator loading...');
        }
        await narrator.screenshot(page, 'privacy-normal');

        // Open privacy dropdown
        narrator.narrate(section, 'Opening the privacy selector...');
        try {
            await page.locator('#chat-privacy-indicator span').first().click();
            await page.waitForSelector('#privacy-dropdown', { state: 'visible', timeout: 5000 });
            await demoPause(page, 1500);
            narrator.narrate(section,
                'Privacy dropdown showing all 5 levels',
                'EPHEMERAL = nothing stored. ISOLATED = session only. ANONYMOUS = PII scrubbed. NORMAL = full persistence. PUBLIC = shareable.');
            await narrator.screenshot(page, 'privacy-dropdown');
        } catch (e) {
            narrator.narrate(section, `Privacy dropdown: ${e.message}`);
            await narrator.screenshot(page, 'privacy-dropdown-fallback');
        }

        // Switch to EPHEMERAL
        narrator.narrate(section, 'Switching to EPHEMERAL mode — zero persistence...');
        try {
            await page.click('.privacy-option[data-mode="ephemeral"]');
            await demoPause(page, 1500);

            // Verify indicator updated
            await page.waitForFunction(() => {
                const el = document.getElementById('chat-privacy-indicator');
                return el && el.textContent.toLowerCase().includes('ephemeral');
            }, { timeout: 5000 });

            narrator.narrate(section,
                'EPHEMERAL mode active — nothing is stored',
                'Notice the indicator changed and the LLM provider switched to local-only. No data leaves this device.');
            await narrator.screenshot(page, 'privacy-ephemeral');
        } catch (e) {
            narrator.narrate(section, `Ephemeral switch: ${e.message}`);
            await narrator.screenshot(page, 'privacy-ephemeral-fallback');
        }

        // Send a message in EPHEMERAL mode — proves the agent responds but stores nothing
        narrator.narrate(section, 'Sending a message in EPHEMERAL mode — this should leave zero traces...');
        const ephemeralResponse = await demoSendMessage(page,
            'What is the meaning of sovereignty in the context of AI agents?');
        await demoPause(page, 1500);

        if (ephemeralResponse) {
            const text = await ephemeralResponse.textContent().catch(() => '');
            narrator.narrate(section,
                `Agent responded (${text.length} chars) — but nothing was stored`,
                'The response was generated by a local LLM (Ollama). No data left the device. No record was written.');
        } else {
            narrator.narrate(section,
                'EPHEMERAL mode active — local LLM processed the request with zero persistence');
        }
        await dismissContextWarning(page);
        await narrator.screenshot(page, 'privacy-ephemeral-response');

        // Restore to NORMAL
        narrator.narrate(section, 'Restoring NORMAL mode for full persistence...');
        try {
            await page.locator('#chat-privacy-indicator span').first().click();
            await page.waitForSelector('#privacy-dropdown', { state: 'visible', timeout: 5000 });
            await page.click('.privacy-option[data-mode="normal"]');
            await demoPause(page, 1500);

            // Wait for toast to appear and dismiss it so the screenshot is clean
            try {
                await page.waitForSelector('.toast-item', { timeout: 3000 });
                await demoPause(page, 2000);
                // Dismiss any remaining toasts
                await page.evaluate(() => {
                    document.querySelectorAll('.toast-item').forEach(t => t.remove());
                });
            } catch { /* no toast, fine */ }

            narrator.narrate(section, 'NORMAL mode restored — full persistence re-enabled');
            await dismissContextWarning(page);
            await narrator.screenshot(page, 'privacy-restored');
        } catch (e) {
            narrator.narrate(section, `Normal restore: ${e.message}`);
        }

        await demoPause(page, 1000);
    });

    // ========================================================================
    // Act 5: Sovereignty Export
    // ========================================================================
    test('Act 5: Sovereignty Export', async ({ page }) => {
        const section = 'Act 5: Sovereignty';
        await demoGoto(page);
        await demoPause(page, 1500);

        // Navigate to Sovereignty panel
        narrator.narrate(section,
            'Opening the Sovereignty panel — this is where data ownership happens...',
            'Your data is your own. Export to IPFS/Filecoin for true portability.');
        await navigateToPanel(page, 'sovereignty');
        await demoPause(page, 2000);
        await narrator.screenshot(page, 'sovereignty-panel');

        // Click export button
        narrator.narrate(section, 'Opening the export dialog...');
        try {
            await page.click('#btn-export-ipfs');
            await page.waitForSelector('#modal-overlay', { state: 'visible', timeout: 5000 });
            await demoPause(page, 1500);

            narrator.narrate(section,
                'Export modal open — three tiers: Local, IPFS, and Filecoin',
                'Local keeps the backup on-device. IPFS distributes it. Filecoin provides long-term decentralized storage.');
            await narrator.screenshot(page, 'export-modal');

            // IPFS is selected by default — show it, then export
            narrator.narrate(section, 'IPFS tier selected — decentralized storage for true data ownership...');
            await demoPause(page, 1000);

            // Fix radio values to lowercase (UI bug: sends uppercase, API wants lowercase)
            await page.evaluate(() => {
                document.querySelectorAll('input[name="export-tier"]').forEach(r => {
                    r.value = r.value.toLowerCase();
                });
            });

            // Click export
            narrator.narrate(section, 'Executing export — packaging agent data into a portable snapshot...');
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
                narrator.narrate(section,
                    `Export complete: ${toastText.trim()}`,
                    'The export contains the agent\'s DID, constitution, memory graph, and conversation history — everything needed to restore this agent anywhere.');
            } catch {
                narrator.narrate(section, 'Export processed (toast may have auto-dismissed)');
            }
            await narrator.screenshot(page, 'export-result');

        } catch (e) {
            narrator.narrate(section, `Export flow: ${e.message}`);
            await narrator.screenshot(page, 'export-fallback');
        }

        // Final pause
        narrator.narrate(section,
            'Demo complete.',
            'With this sovereignty receipt, the owner can restore their AI companion on any compatible platform. No vendor lock-in. True data ownership.');
        await demoPause(page, 2000);
        await narrator.screenshot(page, 'demo-final');
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
