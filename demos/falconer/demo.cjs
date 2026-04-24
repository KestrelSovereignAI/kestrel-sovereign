/**
 * Kestrel Falconer — Product Demo Script
 *
 * Playwright-scripted demo showcasing the Falconer operating model:
 *   Act 1: The Falconer Vision (product narrative page)
 *   Act 2: Meet the Sovereign Flock (multi-agent console)
 *   Act 3: Morning Signal Briefing (Claws scans GitHub)
 *   Act 4: Strategic Dispatch (Claws → Talon mesh handoff)
 *   Act 5: Talon Autonomous Execution (the strike)
 *   Act 6: Constitutional Governance (trust architecture)
 *
 * Run: cd demos/falconer && npx playwright test --config=config.cjs
 *
 * Output (in demo-output/):
 *   - narration.md   — timestamped transcript with screenshot references
 *   - NN-name.png    — screenshots at key moments (~20 total)
 *   - video (.webm)  — full browser recording
 *
 * Prerequisites:
 *   - Server running in multi-agent mode: uv run uvicorn server:app --port 8888
 *     (rookery.toml must be present for multi-agent mode)
 *   - GITHUB_TOKEN in .env (for morning signal to scan repos)
 *   - KESTREL_API_KEY in .env
 *   - OPENROUTER_API_KEY in .env (or another working LLM provider)
 *
 * kestrel-eye integration:
 *   Run: kestrel-eye review --config eye-falconer.toml
 */
const { test, expect } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

const BASE_URL = process.env.KESTREL_URL || 'http://localhost:8888';
const OUTPUT_DIR = path.join(__dirname, 'demo-output');

/** Build a console URL with ?key= param if API key is available. */
function consoleUrl(pathSuffix = '') {
    const base = `${BASE_URL}${pathSuffix}`;
    return apiKey ? `${base}?key=${encodeURIComponent(apiKey)}` : base;
}

/**
 * Inject API key into sessionStorage before any page JS runs.
 * Works even if the deployed api_client.mjs doesn't support ?key= yet,
 * since api_client.init() checks sessionStorage before /api/auth/key.
 */
async function injectApiKey(page) {
    if (apiKey) {
        await page.addInitScript((key) => {
            sessionStorage.setItem('kestrel_api_key', key);
        }, apiKey);
    }
}

// ============================================================================
// Narration Engine (shared pattern with demo_technical)
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

    async screenshotFullPage(page, name) {
        this.screenshotIndex++;
        const filename = `${String(this.screenshotIndex).padStart(2, '0')}-${name}.png`;
        const filepath = path.join(OUTPUT_DIR, filename);
        try {
            await page.screenshot({ path: filepath, fullPage: true });
        } catch (e) {
            console.warn(`  >> Full-page screenshot failed: ${e.message}`);
            return filename;
        }
        if (this.entries.length > 0) {
            this.entries[this.entries.length - 1].screenshot = filename;
        }
        console.log(`  >> SCREENSHOT: ${filename}`);
        return filename;
    }

    toMarkdown() {
        let md = '# Kestrel Falconer — Product Demo Transcript\n\n';
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

function getApiKey() {
    if (process.env.KESTREL_API_KEY) return process.env.KESTREL_API_KEY;
    return null;
}

function authHeaders(apiKey) {
    return apiKey ? { 'X-API-Key': apiKey } : {};
}

/** Send a chat message and wait for agent response. */
async function demoSendMessage(page, message, timeout = 120000) {
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
        const count = await page.locator('.agent-message').count();
        return count > 0 ? page.locator('.agent-message').last() : null;
    }
}

/** Visual pause for recording clarity. */
async function demoPause(page, ms = 2000) {
    await page.waitForTimeout(ms);
}

/** Select a working LLM provider from the dropdown. */
async function selectProvider(page, narrator) {
    const providerSelector = page.locator('#provider-selector');
    if (!await providerSelector.isVisible().catch(() => false)) return;

    const options = await providerSelector.locator('option').allTextContents();
    const preferred = ['openrouter', 'anthropic', 'openai'];
    for (const pref of preferred) {
        const match = options.find(o => o.toLowerCase().includes(pref));
        if (match) {
            await providerSelector.selectOption({ label: match });
            narrator.narrate('Setup', `Provider set to: ${match}`);
            await demoPause(page, 1000);
            return;
        }
    }
}

// ============================================================================
// Shared state
// ============================================================================

const narrator = new NarrationEngine();
let apiKey = null;

// ============================================================================
// Setup
// ============================================================================

test.beforeAll(async ({ request }) => {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });

    // Verify server is alive
    let isRookery = false;
    try {
        const resp = await request.get(`${BASE_URL}/health`);
        const data = await resp.json();
        isRookery = data.agents && typeof data.agents === 'object' && Object.keys(data.agents).length > 0;
        narrator.narrate('Setup', `Server health: ${data.status}, agents: ${Object.keys(data.agents || {}).length || '?'}`);
    } catch (e) {
        narrator.narrate('Setup', `Server not reachable: ${e.message}`);
    }

    apiKey = getApiKey();
    if (!apiKey) {
        // Fetch bootstrap key from server (same as Sovereign Console)
        try {
            const keyResp = await request.get(`${BASE_URL}/api/auth/key`);
            if (keyResp.ok()) {
                const keyData = await keyResp.json();
                apiKey = keyData.api_key || keyData.key || null;
            }
        } catch (e) { /* server may not expose key */ }
    }
    if (apiKey) {
        narrator.narrate('Setup', 'API key acquired');
    }

    // Skip bootstrap/discovery on the primary agent so bang-commands work
    // immediately.  After a cold start the agent enters discovery mode and
    // intercepts all non-bootstrap commands, sending them to the LLM instead.
    if (apiKey) {
        const invokeUrl = isRookery
            ? `${BASE_URL}/api/agents/Kestrel/agent/invoke`
            : `${BASE_URL}/agent/invoke`;
        try {
            const statusResp = await request.post(invokeUrl, {
                headers: { 'X-API-Key': apiKey, 'Content-Type': 'application/json' },
                data: { input: '!bootstrap-status' },
            });
            const statusData = await statusResp.json();
            if (statusData.response && statusData.response.includes('discovery')) {
                await request.post(invokeUrl, {
                    headers: { 'X-API-Key': apiKey, 'Content-Type': 'application/json' },
                    data: { input: '!skip-discovery' },
                });
                narrator.narrate('Setup', 'Skipped agent discovery (cold-start bootstrap)');
            }
        } catch (e) {
            // Non-fatal: commands may work fine if bootstrap already completed
            narrator.narrate('Setup', `Bootstrap check skipped: ${e.message}`);
        }
    }
});

test.afterAll(async () => {
    const md = narrator.toMarkdown();
    fs.writeFileSync(path.join(OUTPUT_DIR, 'narration.md'), md, 'utf-8');
    console.log(`\n[DEMO] Narration written to ${path.join(OUTPUT_DIR, 'narration.md')}`);
    console.log(`[DEMO] Screenshots in ${OUTPUT_DIR}`);
    console.log(`[DEMO] Video captured by Playwright in demo-output/`);
});

// ============================================================================
// Act 1: The Falconer Vision
// ============================================================================

test('Act 1: The Falconer Vision', async ({ page }) => {
    narrator.narrate('Act 1: The Falconer Vision', 'Loading the Falconer product page...');

    await page.goto(`${BASE_URL}/static/kestrel-falconer-v2.html`);
    await page.waitForLoadState('networkidle');
    await demoPause(page, 2000);

    // Hero section
    const hero = page.locator('.hero');
    await expect(hero).toBeVisible();
    narrator.narrate('Act 1: The Falconer Vision',
        'Kestrel Falconer — the AI-native operating model where humans lead and sovereign AI agents execute.',
        'This is not autocomplete. This is a complete product organization run by 4 autonomous AI agents under human governance.');
    await narrator.screenshot(page, 'falconer-hero');

    // Scroll to metaphor (Old Way vs Falconer Way)
    await page.locator('.metaphor').scrollIntoViewIfNeeded();
    await demoPause(page, 1500);
    narrator.narrate('Act 1: The Falconer Vision',
        'The Old Way vs The Falconer Way — Copilot gave developers a faster typewriter. The Falconer replaces the entire process.',
        '"The AI was never the bottleneck — the process is." Nobody replaced the PM, QA lead, or business owner. Until now.');
    await narrator.screenshot(page, 'falconer-metaphor');

    // Scroll to Flock diagram
    await page.locator('.diagram-section').scrollIntoViewIfNeeded();
    await demoPause(page, 2000);
    narrator.narrate('Act 1: The Falconer Vision',
        'The Flock — One human falconer, four sovereign birds, one constitutional foundation.',
        'Claws (PM), Talon (Developer), Eye (QA), Flight (Business PO). Each is a sovereign AI agent with its own DID and constitution.');
    await narrator.screenshot(page, 'falconer-flock-diagram');

    // Scroll to role-mapping section (what each bird replaces)
    await page.locator('.role-mapping').scrollIntoViewIfNeeded();
    await demoPause(page, 1500);
    narrator.narrate('Act 1: The Falconer Vision',
        'What Each Bird Replaces — Claws replaces PM/TPM, Talon replaces developers, Eye replaces QA leads, Flight replaces business owners.',
        'Traditional roles mapped to sovereign AI agents. The falconer always stays in command.');
    await narrator.screenshot(page, 'falconer-birds');

    // Scroll to daily workflow loop
    const loopSection = page.locator('.loop-section');
    if (await loopSection.isVisible().catch(() => false)) {
        await loopSection.scrollIntoViewIfNeeded();
        await demoPause(page, 1500);
        narrator.narrate('Act 1: The Falconer Vision',
            'A Day in the Falconer\'s Life — the autonomous daily workflow loop.',
            '8:00 AM Morning Signal → Claws picks top issue → Talon executes → Eye validates → Flight narrates. The falconer reviews at the end of the day.');
        await narrator.screenshot(page, 'falconer-daily-loop');
    }
});

// ============================================================================
// Act 2: Meet the Sovereign Flock
// ============================================================================

test('Act 2: Meet the Sovereign Flock', async ({ page }) => {
    narrator.narrate('Act 2: The Sovereign Flock', 'Loading the Sovereign Console — the command center...');

    await injectApiKey(page);
    await page.goto(consoleUrl());
    await page.waitForLoadState('domcontentloaded');
    await demoPause(page, 3000);

    // Check for multi-agent sidebar
    const agentsPane = page.locator('#agents-pane');
    const agentsList = page.locator('#agents-list');

    if (await agentsPane.isVisible().catch(() => false)) {
        // Wait for agents to load
        await page.waitForFunction(() => {
            const list = document.querySelector('#agents-list');
            return list && list.querySelectorAll('.agent-item').length > 0;
        }, { timeout: 15000 }).catch(() => {});

        const agentCount = await page.locator('.agent-item').count();
        const onlineCount = await page.locator('.agent-status-dot.online').count();

        narrator.narrate('Act 2: The Sovereign Flock',
            `${agentCount} agents in the rookery — ${onlineCount} online. Each is a sovereign agent with its own DID, constitution, and memory.`,
            'This is multi-agent mode. Every agent in the sidebar is an independent sovereign entity running on its own port.');
        await narrator.screenshot(page, 'flock-agents-list');
    } else {
        narrator.narrate('Act 2: The Sovereign Flock',
            'Sovereign Console loaded (single-agent mode — enable rookery.toml for full flock view).');
        await narrator.screenshot(page, 'console-single-agent');
    }

    // Show the Identity panel (DID)
    const identityTab = page.locator('nav a, nav button, [data-tab]').filter({ hasText: /identity/i }).first();
    if (await identityTab.isVisible().catch(() => false)) {
        await identityTab.click();
        await demoPause(page, 2000);

        // Capture DID
        const didText = await page.textContent('body');
        const didMatch = didText.match(/did:pkh:eip155:\d+:0x[a-fA-F0-9]+/);
        if (didMatch) {
            narrator.narrate('Act 2: The Sovereign Flock',
                `Agent identity: ${didMatch[0]}`,
                'W3C Decentralized Identifier — cryptographically owned by the agent, not the platform. This is the anchor for everything: constitution, memory, sovereignty.');
        }
        await narrator.screenshot(page, 'flock-agent-identity');
    }

    // Navigate to Chat and set provider
    const chatTab = page.locator('nav a, nav button, [data-tab]').filter({ hasText: /chat/i }).first();
    if (await chatTab.isVisible().catch(() => false)) {
        await chatTab.click();
        await demoPause(page, 1500);
        await selectProvider(page, narrator);
        narrator.narrate('Act 2: The Sovereign Flock',
            'Chat panel ready — this is where the falconer commands the flock.',
            'Every command (!morning, !dispatch, !talon) flows through the constitutional agent with full audit trail.');
        await narrator.screenshot(page, 'flock-chat-ready');
    }
});

// ============================================================================
// Act 3: Morning Signal Briefing (Claws)
// ============================================================================

test('Act 3: Morning Signal Briefing', async ({ page }) => {
    narrator.narrate('Act 3: Morning Signal', 'Opening the Falconer Dashboard — the operational command center...');

    // --- Part 1: Falconer Dashboard ---
    // Pass API key via query param so the dashboard authenticates automatically
    // Use the module-level apiKey (fetched in beforeAll), not getApiKey() which only reads env vars
    const dashUrl = apiKey
        ? `${BASE_URL}/static/falconer-dashboard.html?key=${encodeURIComponent(apiKey)}`
        : `${BASE_URL}/static/falconer-dashboard.html`;
    await page.goto(dashUrl);
    await page.waitForLoadState('domcontentloaded');
    await demoPause(page, 3000);

    // KPI grid — the top-level metrics a falconer checks every morning
    const kpiGrid = page.locator('.kpi-grid, #kpiGrid');
    if (await kpiGrid.isVisible().catch(() => false)) {
        narrator.narrate('Act 3: Morning Signal',
            'The Falconer Dashboard — KPIs at a glance: agents online, active tasks, mesh messages, context health.',
            'This is the operational nerve center. Before the Morning Signal even runs, the falconer can see flock health in real time.');
    } else {
        narrator.narrate('Act 3: Morning Signal',
            'The Falconer Dashboard — the operational view of the sovereign flock.');
    }
    await narrator.screenshot(page, 'morning-signal-dashboard');

    // Flock status cards
    const flockGrid = page.locator('.flock-grid, #flockGrid');
    if (await flockGrid.isVisible().catch(() => false)) {
        await flockGrid.scrollIntoViewIfNeeded();
        await demoPause(page, 1500);
        narrator.narrate('Act 3: Morning Signal',
            'Flock status — each sovereign agent has its own card with health, context usage, and active jobs.',
            'Claws (PM), Talon (Developer), Eye (QA), Flight (Business PO) — all sovereign, all monitored.');
        await narrator.screenshot(page, 'morning-signal-flock');
    }

    // Tasks & Jobs tab — shows scheduled Morning Signal
    const tasksTab = page.locator('.tab').filter({ hasText: /task/i }).first();
    if (await tasksTab.isVisible().catch(() => false)) {
        await tasksTab.click();
        await demoPause(page, 2000);

        narrator.narrate('Act 3: Morning Signal',
            'Tasks & Scheduled Jobs — the Morning Signal runs automatically at 7:00 AM every day.',
            'Claws scans all GitHub repos, scores milestones, detects blockers, and generates a prioritized briefing. No human triggers needed — the flock works autonomously.');
        await narrator.screenshot(page, 'morning-signal-tasks');
    }

    // Governance tab — constitutional anchor
    const govTab = page.locator('.tab').filter({ hasText: /governance/i }).first();
    if (await govTab.isVisible().catch(() => false)) {
        await govTab.click();
        await demoPause(page, 2000);

        narrator.narrate('Act 3: Morning Signal',
            'Governance — every agent action, including the Morning Signal, is governed by the Kestrel Constitution.',
            'SHA-256 hash verified on every interaction. DID identity, privacy modes, and tool permissions — all visible, all auditable.');
        await narrator.screenshot(page, 'morning-signal-governance');
    }

    // --- Part 2: Daily Loop on the Product Page ---
    narrator.narrate('Act 3: Morning Signal', 'Now visualizing the daily workflow — where the Morning Signal fits...');
    await page.goto(`${BASE_URL}/static/kestrel-falconer-v2.html`);
    await page.waitForLoadState('networkidle');
    await demoPause(page, 1500);

    // Scroll to the Daily Loop section
    const loopSection = page.locator('.loop-section');
    if (await loopSection.isVisible().catch(() => false)) {
        await loopSection.scrollIntoViewIfNeeded();
        await demoPause(page, 2000);
        narrator.narrate('Act 3: Morning Signal',
            'A Day in the Falconer\'s Life — 7:00 AM: Claws generates Morning Signal → 8:00 AM: Falconer reviews → Talon executes all day → Eye validates → Flight reports.',
            'The Morning Signal replaces standups, triage meetings, and status emails. One briefing, generated autonomously from live GitHub data across all repos.');
        await narrator.screenshot(page, 'morning-signal-daily-loop');
    }
});

// ============================================================================
// Act 4: Strategic Dispatch (Claws → Talon)
// ============================================================================

test('Act 4: Strategic Dispatch', async ({ page }) => {
    narrator.narrate('Act 4: Strategic Dispatch', 'Now the falconer dispatches — Claws picks the top priority and hands off to Talon...');

    await injectApiKey(page);
    await page.goto(consoleUrl());
    await page.waitForLoadState('domcontentloaded');
    await demoPause(page, 2000);

    // Navigate to Chat
    const chatTab = page.locator('nav a, nav button, [data-tab]').filter({ hasText: /chat/i }).first();
    if (await chatTab.isVisible().catch(() => false)) {
        await chatTab.click();
        await demoPause(page, 1000);
    }
    await selectProvider(page, narrator);

    // Step 1: Suggest — show what Claws recommends
    narrator.narrate('Act 4: Strategic Dispatch',
        'Asking Claws to suggest the highest-priority issue from strategic memory...',
        '!dispatch suggest — Claws examines all milestones, weighs urgency, risk, and dependencies to pick the ONE issue that moves the needle most.');

    const suggestResponse = await demoSendMessage(page, '!dispatch suggest', 120000);

    if (suggestResponse) {
        const text = await suggestResponse.textContent().catch(() => '');
        const issueMatch = text.match(/#\d+/);
        const priorityMatch = text.match(/priority[:\s]*(\w+)/i);

        narrator.narrate('Act 4: Strategic Dispatch',
            `Claws recommends: ${issueMatch ? issueMatch[0] : 'top issue'} (${priorityMatch ? priorityMatch[1] : 'prioritized'})`,
            'This is AI-native project management. No sprint planning. No estimation poker. Claws scans GitHub, scores by impact, and recommends.');
    }
    await narrator.screenshot(page, 'dispatch-suggest');

    // Step 2: Execute dispatch via mesh
    await demoPause(page, 2000);
    narrator.narrate('Act 4: Strategic Dispatch',
        'Executing the dispatch — Claws sends an ASSIGN message to Talon via the Agent Mesh Protocol...',
        'The mesh is how sovereign agents communicate. Typed messages (ASSIGN, COMPLETE, REJECT) with correlation IDs for full traceability.');

    const dispatchResponse = await demoSendMessage(page, '!dispatch', 120000);

    if (dispatchResponse) {
        const text = await dispatchResponse.textContent().catch(() => '');
        const issueRef = text.match(/#(\d+)/);
        const hasResult = text.length > 20;

        narrator.narrate('Act 4: Strategic Dispatch',
            hasResult
                ? `Dispatch complete${issueRef ? ` — issue ${issueRef[0]} handed off to Talon` : ''}`
                : 'Dispatch sent to Talon via mesh',
            'Claws → Talon handoff complete. The ASSIGN message travels through the Agent Mesh Protocol. Talon will claim the issue and start coding autonomously.');
    }
    await narrator.screenshot(page, 'dispatch-execute');
});

// ============================================================================
// Act 5: Talon Autonomous Execution
// ============================================================================

test('Act 5: Talon Autonomous Execution', async ({ page, request }) => {
    narrator.narrate('Act 5: Talon Execution', 'Checking Talon\'s workload — the autonomous developer agent...');

    await injectApiKey(page);
    await page.goto(consoleUrl());
    await page.waitForLoadState('domcontentloaded');
    await demoPause(page, 2000);

    // Navigate to Chat
    const chatTab = page.locator('nav a, nav button, [data-tab]').filter({ hasText: /chat/i }).first();
    if (await chatTab.isVisible().catch(() => false)) {
        await chatTab.click();
        await demoPause(page, 1000);
    }
    await selectProvider(page, narrator);

    // Check Talon status
    narrator.narrate('Act 5: Talon Execution',
        'Checking Talon status — how many jobs are running, completed, or failed...',
        '!talon status — The falconer monitors Talon\'s workload. Running jobs, completed PRs, failed attempts. All visible, all auditable.');

    const statusResponse = await demoSendMessage(page, '!talon status', 60000);

    if (statusResponse) {
        const text = await statusResponse.textContent().catch(() => '');
        narrator.narrate('Act 5: Talon Execution',
            `Talon status: ${text.substring(0, 200).replace(/\n/g, ' ')}`,
            'Talon is the autonomous developer. It claims issues, writes code, runs tests, self-corrects on failures, and opens PRs. The falconer sets direction — Talon executes.');
    }
    await narrator.screenshot(page, 'talon-status');

    // Check mesh inbox via API
    try {
        const headers = authHeaders(apiKey);
        const inboxResp = await request.get(`${BASE_URL}/agent/mesh/inbox?limit=5`, { headers });
        if (inboxResp.ok()) {
            const inbox = await inboxResp.json();
            const messages = inbox.messages || inbox || [];
            const count = Array.isArray(messages) ? messages.length : 0;
            narrator.narrate('Act 5: Talon Execution',
                `Mesh inbox: ${count} message(s) — the communication backbone of the flock`,
                'Agent Mesh Protocol: typed messages (ASSIGN, COMPLETE, REJECT, STATUS_UPDATE) flow between sovereign agents. Every message has a correlation ID for full trace.');
        }
    } catch (e) {
        narrator.narrate('Act 5: Talon Execution', `Mesh inbox check: ${e.message}`);
    }

    // Show the kill switch
    narrator.narrate('Act 5: Talon Execution',
        'The falconer has a kill switch — !talon pause stops all autonomous execution immediately.',
        'Trust requires control. One command stops Talon from claiming any new work. Resume when ready. The human is always the falconer.');
    await narrator.screenshot(page, 'talon-kill-switch');
});

// ============================================================================
// Act 6: Constitutional Governance
// ============================================================================

test('Act 6: Constitutional Governance', async ({ page }) => {
    narrator.narrate('Act 6: Constitutional Governance', 'The trust architecture — why enterprises can deploy this...');

    await injectApiKey(page);
    await page.goto(consoleUrl());
    await page.waitForLoadState('domcontentloaded');
    await demoPause(page, 2000);

    // Constitution tab
    const constitutionTab = page.locator('nav a, nav button, [data-tab]').filter({ hasText: /constitution/i }).first();
    if (await constitutionTab.isVisible().catch(() => false)) {
        await constitutionTab.click();
        await demoPause(page, 3000);

        narrator.narrate('Act 6: Constitutional Governance',
            'The Kestrel Digital Bill of Rights — immutable rules anchored at agent genesis.',
            'SHA-256 hash verified on every interaction. If the constitution changes, the agent enters safe mode. This is compliance you can prove to auditors.');
        await narrator.screenshot(page, 'constitution-panel');
    }

    // Security tab — permissions
    const securityTab = page.locator('nav a, nav button, [data-tab]').filter({ hasText: /security/i }).first();
    if (await securityTab.isVisible().catch(() => false)) {
        await securityTab.click();
        await demoPause(page, 2000);

        narrator.narrate('Act 6: Constitutional Governance',
            'Tool-level permissions — Allow, Ask, or Deny. Enforced at the architecture level, not by prompt.',
            'Every tool the agent can use (export, memory, web search) has its own permission. DENY = the code never executes. Every decision is logged in the audit trail.');
        await narrator.screenshot(page, 'security-permissions');

        // Show that permissions are per-tool
        const permissionRows = await page.locator('.permission-row, [class*="permission"], tr').count();
        if (permissionRows > 0) {
            narrator.narrate('Act 6: Constitutional Governance',
                `${permissionRows} tool permissions configured — granular control over every agent capability.`,
                'The falconer doesn\'t just set direction — they control exactly what each bird CAN and CANNOT do. Architecture-enforced, not prompt-enforced.');
        }
        await narrator.screenshot(page, 'security-tool-permissions');
    }

    // Privacy mode indicator
    const privacyIndicator = page.locator('#privacy-mode, [class*="privacy"]').first();
    if (await privacyIndicator.isVisible().catch(() => false)) {
        narrator.narrate('Act 6: Constitutional Governance',
            '5 privacy levels — EPHEMERAL (zero persistence) to PUBLIC (shareable). The falconer chooses.',
            'EPHEMERAL mode: nothing stored, local LLM only, zero data leaves the device. Enterprise-grade privacy without configuration complexity.');
        await narrator.screenshot(page, 'privacy-governance');
    }

    // Final — navigate back to Falconer page for the closing shot
    await page.goto(`${BASE_URL}/static/kestrel-falconer-v2.html`);
    await page.waitForLoadState('networkidle');
    await demoPause(page, 1500);

    // Scroll to proof section
    const proofSection = page.locator('.proof-section');
    if (await proofSection.isVisible().catch(() => false)) {
        await proofSection.scrollIntoViewIfNeeded();
        await demoPause(page, 1500);
        narrator.narrate('Act 6: Constitutional Governance',
            'Living proof — Gabi, Jason, and Noel are already using Falconer daily. This is not a demo. This is production.',
            'Three falconers, each with their own sovereign flock. Daily morning signals, autonomous ticket execution, constitutional governance. The future of product development.');
        await narrator.screenshot(page, 'falconer-proof');
    }

    // Final hero shot
    await page.locator('.hero').scrollIntoViewIfNeeded();
    await demoPause(page, 1000);
    narrator.narrate('Act 6: Constitutional Governance',
        'Kestrel Falconer — Humans lead. AI executes. No exceptions.',
        'Constitutional AI. Sovereign agents. Cryptographic identity. Provable compliance. The operating model that makes sprint ceremonies obsolete.');
    await narrator.screenshot(page, 'falconer-finale');
});
