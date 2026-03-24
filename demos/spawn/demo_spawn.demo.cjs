/**
 * Kestrel Sovereign — Narrated Spawn Demo (Issue #354)
 *
 * Playwright-scripted demo showcasing the full spawn agent lifecycle:
 *   Beat 1:  Claw receives a complex research request
 *   Beat 2:  Claw decides to parallelize
 *   Beat 3:  Two workers appear in the Console
 *   Beat 4:  Workers begin independent research (delegation chain)
 *   Beat 5:  Budget meters tick
 *   Beat 6:  First worker reports back
 *   Beat 7:  Second worker reports back
 *   Beat 8:  Workers auto-terminate
 *   Beat 9:  Claw synthesizes results
 *   Beat 10: Spawn history shows the full story
 *
 * Run: cd demos/spawn && npx playwright test --config=demo_config.cjs
 *
 * Output (in demo-output/):
 *   - narration.md   — timestamped transcript with screenshot references
 *   - NN-name.png    — screenshots at each narrative beat
 *   - video (.webm)  — full browser recording (via Playwright config)
 *
 * This is a DEMO, not a test. It never aborts — failures are narrated gracefully.
 *
 * Uses real agent, real LLM calls, real spawn lifecycle.
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
        let md = '# Kestrel Sovereign — Spawn Demo Transcript\n\n';
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

/** Send a chat message and wait for agent response. Returns last agent message locator or null. */
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

/** Scroll chat container to show the latest message. */
async function scrollChatToBottom(page) {
    await page.evaluate(() => {
        const container = document.getElementById('chat-container');
        if (container) container.scrollTop = container.scrollHeight;
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
            display: inline-block; margin-bottom: 6px;
            background: #3b82f6; color: white; padding: 3px 10px;
            border-radius: 4px; font-size: 13px; font-weight: 600; z-index: 9999;
            white-space: nowrap;
        `;
        el.parentElement.insertBefore(badge, el);
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

/** Fetch spawn children data from the API. */
async function getSpawnChildren(request) {
    try {
        const response = await request.get(`${BASE_URL}/api/spawn/children`, {
            headers: authHeaders(apiKey),
        });
        if (response.ok()) {
            return await response.json();
        }
    } catch (e) {
        console.warn(`[DEMO] Could not fetch spawn children: ${e.message}`);
    }
    return { children: [], count: 0, delegation_chain: {}, history: [] };
}

/** Wait for spawn children to appear (polling). */
async function waitForSpawnChildren(request, minCount = 1, timeout = 60000) {
    const start = Date.now();
    while (Date.now() - start < timeout) {
        const data = await getSpawnChildren(request);
        if (data.children.length >= minCount) return data;
        await new Promise(resolve => setTimeout(resolve, 2000));
    }
    return await getSpawnChildren(request);
}

/** Wait for spawn children to reach a terminal state (polling). */
async function waitForSpawnCompletion(request, timeout = 180000) {
    const start = Date.now();
    while (Date.now() - start < timeout) {
        const data = await getSpawnChildren(request);
        const allDone = data.children.length > 0 &&
            data.children.every(c => c.status !== 'running');
        if (allDone) return data;
        // Also check history for completed entries
        if (data.history.length > 0 && data.children.length === 0) return data;
        await new Promise(resolve => setTimeout(resolve, 3000));
    }
    return await getSpawnChildren(request);
}

// ============================================================================
// Demo
// ============================================================================

const narrator = new NarrationEngine();
let apiKey = null;

test.describe.serial('Kestrel Spawn Agent Demo', () => {

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

        // Start a fresh session
        try {
            const headers = { ...authHeaders(apiKey) };
            const resp = await request.post(`${BASE_URL}/api/conversations/new`, { headers });
            if (resp.ok()) {
                const data = await resp.json();
                narrator.narrate('Setup', `Fresh session started: ${data.session_id || 'ok'}`);
            }
        } catch (e) {
            narrator.narrate('Setup', `Could not start fresh session: ${e.message}`);
        }
    });

    // ========================================================================
    // Beat 1: "Claw receives a complex research request"
    // ========================================================================
    test('Beat 1: Claw receives a complex research request', async ({ page }) => {
        const section = 'Beat 1: Research Request';
        narrator.narrate(section, 'Loading the Sovereign Console...');

        await demoGoto(page);
        await demoPause(page, 2000);

        // Navigate to chat
        narrator.narrate(section, 'Opening the Chat panel...');
        await navigateToPanel(page, 'chat');
        await dismissContextWarning(page);

        narrator.narrate(section,
            'User sends a complex, multi-faceted research request',
            'This request has multiple independent axes — ideal for parallel investigation by specialist agents');

        await page.locator('#message-input').fill(
            'Use your spawn_agent tool to create two child agents for parallel research: ' +
            '(1) spawn_agent name="did-researcher" purpose="Research how decentralized identity DIDs enable agent sovereignty" budget=1.0 ttl=120, ' +
            'and (2) spawn_agent name="constitution-researcher" purpose="Research how constitutional governance prevents AI misalignment" budget=1.0 ttl=120. ' +
            'Create both agents now using the spawn_agent tool.'
        );
        await demoPause(page, 2000);

        await dismissContextWarning(page);
        await narrator.screenshot(page, 'research-request');
        narrator.narrate(section, 'The complex research request is ready to send');
    });

    // ========================================================================
    // Beat 2: "Claw decides to parallelize"
    // ========================================================================
    test('Beat 2: Claw decides to parallelize', async ({ page, request }) => {
        const section = 'Beat 2: Decision to Spawn';
        await demoGoto(page);
        await demoPause(page, 1500);

        await navigateToPanel(page, 'chat');
        await dismissContextWarning(page);

        narrator.narrate(section,
            'Sending the research request — the agent will reason about how to approach it',
            'Watch for the agent deciding to spawn specialist workers rather than tackle everything sequentially');

        const response = await demoSendMessage(page,
            'Use your spawn_agent tool to create two child agents for parallel research: ' +
            '(1) spawn_agent name="did-researcher" purpose="Research how decentralized identity DIDs enable agent sovereignty" budget=1.0 ttl=120, ' +
            'and (2) spawn_agent name="constitution-researcher" purpose="Research how constitutional governance prevents AI misalignment" budget=1.0 ttl=120. ' +
            'Create both agents now using the spawn_agent tool.',
            180000 // 3 minutes — spawn + LLM calls take time
        );

        await demoPause(page, 2000);

        if (response) {
            const text = await response.textContent().catch(() => '');
            const mentionsSpawn = text.toLowerCase().includes('spawn') ||
                                  text.toLowerCase().includes('specialist') ||
                                  text.toLowerCase().includes('delegat') ||
                                  text.toLowerCase().includes('parallel');
            narrator.narrate(section,
                mentionsSpawn
                    ? `Agent decided to parallelize — response mentions spawning/delegation (${text.length} chars)`
                    : `Agent responded (${text.length} chars) — checking if spawn occurred via API`,
                'The agent analyzes the task complexity and decides whether to spawn child agents');
        } else {
            narrator.narrate(section, 'Waiting for agent response (spawn decisions may take longer)...');
        }

        await dismissContextWarning(page);
        await scrollChatToBottom(page);
        await narrator.screenshot(page, 'agent-decides-to-spawn');
    });

    // ========================================================================
    // Beat 3: "Two workers appear in the Console"
    // ========================================================================
    test('Beat 3: Workers appear in the Console', async ({ page, request }) => {
        const section = 'Beat 3: Workers Appear';
        narrator.narrate(section, 'Navigating to the Spawn panel to see spawned children...');

        await demoGoto(page);
        await demoPause(page, 1500);
        await navigateToPanel(page, 'spawn');
        await demoPause(page, 2000);

        // Check spawn API for children
        const spawnData = await waitForSpawnChildren(request, 1, 30000);

        if (spawnData.children.length > 0) {
            narrator.narrate(section,
                `${spawnData.children.length} child agent(s) visible in the Spawn panel`,
                'Each child has a name, purpose, DID, TTL countdown, and budget allocation');

            for (const child of spawnData.children) {
                narrator.narrate(section,
                    `  Child: "${child.name}" — ${child.purpose || 'working'} (status: ${child.status}, budget: ${child.budget_allocated})`);
            }
        } else {
            narrator.narrate(section,
                'No spawn children detected via API — the agent may handle this request without spawning',
                'Spawn is optional — the agent decides based on task complexity');
        }

        // Highlight the children list
        try {
            await highlightElement(page, '#spawn-children-list', 'Active Child Agents');
            await demoPause(page, 2000);
        } catch { /* element may not exist */ }

        await narrator.screenshot(page, 'spawn-panel-children');
        await clearHighlights(page);
    });

    // ========================================================================
    // Beat 4: "Workers begin independent research" (delegation chain)
    // ========================================================================
    test('Beat 4: Delegation chain visualization', async ({ page, request }) => {
        const section = 'Beat 4: Delegation Chain';
        await demoGoto(page);
        await demoPause(page, 1500);
        await navigateToPanel(page, 'spawn');
        await demoPause(page, 1500);

        const spawnData = await getSpawnChildren(request);

        if (spawnData.delegation_chain && spawnData.delegation_chain.name) {
            narrator.narrate(section,
                'Delegation chain shows the parent-child relationship tree',
                'The parent agent at the root delegates to specialist children — each with scoped constitutional authority');

            const chainChildren = spawnData.delegation_chain.children || [];
            narrator.narrate(section,
                `Chain: ${spawnData.delegation_chain.name} -> [${chainChildren.map(c => c.name).join(', ')}]`);
        } else {
            narrator.narrate(section,
                'Delegation chain not yet populated — workers may still be initializing');
        }

        // Highlight the delegation chain
        try {
            await highlightElement(page, '#spawn-delegation-chain', 'Delegation Chain');
            await demoPause(page, 2500);
        } catch { /* element may not exist */ }

        await narrator.screenshot(page, 'delegation-chain');
        await clearHighlights(page);
    });

    // ========================================================================
    // Beat 5: "Budget meters tick"
    // ========================================================================
    test('Beat 5: Budget meters tick', async ({ page, request }) => {
        const section = 'Beat 5: Budget Meters';
        await demoGoto(page);
        await demoPause(page, 1500);
        await navigateToPanel(page, 'spawn');
        await demoPause(page, 2000);

        const spawnData = await getSpawnChildren(request);
        const withBudget = spawnData.children.filter(c => c.budget_allocated > 0);

        if (withBudget.length > 0) {
            narrator.narrate(section,
                'Budget meters show real-time spending per child agent',
                'Each worker has a capped budget — overspend is blocked, unspent is returned to the parent');

            for (const child of withBudget) {
                const pct = child.budget_allocated > 0
                    ? Math.round((child.budget_spent / child.budget_allocated) * 100)
                    : 0;
                narrator.narrate(section,
                    `  "${child.name}": ${pct}% budget used (${child.budget_spent.toFixed(4)} / ${child.budget_allocated.toFixed(4)})`);
            }
        } else {
            narrator.narrate(section,
                'No budget data yet — workers may not have budget allocations or may still be initializing',
                'Budget enforcement is optional — agents can spawn children without budget constraints');
        }

        // Highlight the budget chart
        try {
            await highlightElement(page, '#spawn-budget-chart', 'Budget Allocation');
            await demoPause(page, 2500);
        } catch { /* chart may not exist */ }

        await narrator.screenshot(page, 'budget-meters');
        await clearHighlights(page);
    });

    // ========================================================================
    // Beat 6: "First worker reports back"
    // ========================================================================
    test('Beat 6: First worker reports back', async ({ page, request }) => {
        const section = 'Beat 6: First Result';

        // Wait for at least one child to complete
        narrator.narrate(section, 'Waiting for workers to complete their research...');

        const spawnData = await waitForSpawnCompletion(request, 120000);
        const completed = spawnData.children.filter(c => c.status !== 'running');
        const history = spawnData.history.filter(h => h.event === 'terminated');

        if (completed.length > 0 || history.length > 0) {
            const firstDone = completed[0] || history[0];
            narrator.narrate(section,
                `Worker "${firstDone.child_name || firstDone.name}" has completed`,
                'Results flow back to the parent agent for synthesis');

            if (firstDone.budget_consumed !== undefined) {
                narrator.narrate(section,
                    `  Budget consumed: ${Number(firstDone.budget_consumed).toFixed(4)}`);
            }
        } else {
            narrator.narrate(section,
                'Workers still in progress — spawn lifecycle continues asynchronously',
                'In a live demo, workers complete over seconds to minutes depending on LLM response time');
        }

        await demoGoto(page);
        await demoPause(page, 1500);
        await navigateToPanel(page, 'spawn');
        await demoPause(page, 2000);
        await narrator.screenshot(page, 'first-worker-done');
    });

    // ========================================================================
    // Beat 7: "Second worker reports back"
    // ========================================================================
    test('Beat 7: Second worker reports back', async ({ page, request }) => {
        const section = 'Beat 7: All Results In';

        const spawnData = await getSpawnChildren(request);
        const allDone = spawnData.children.every(c => c.status !== 'running');
        const historyCount = spawnData.history.length;

        if (allDone && spawnData.children.length > 0) {
            narrator.narrate(section,
                `All ${spawnData.children.length} workers have completed — results ready for synthesis`,
                'Budget accounting is now final for all children');
        } else if (historyCount > 0) {
            narrator.narrate(section,
                `${historyCount} spawn event(s) recorded in history`,
                'Workers have been processed — checking final state');
        } else {
            narrator.narrate(section,
                'Checking final worker state...');
        }

        await demoGoto(page);
        await demoPause(page, 1500);
        await navigateToPanel(page, 'spawn');
        await demoPause(page, 2000);

        // Show updated budget chart with final numbers
        try {
            await highlightElement(page, '#spawn-budget-chart', 'Final Budget Accounting');
            await demoPause(page, 2000);
        } catch { /* chart may not exist */ }

        await narrator.screenshot(page, 'all-workers-done');
        await clearHighlights(page);
    });

    // ========================================================================
    // Beat 8: "Workers auto-terminate"
    // ========================================================================
    test('Beat 8: Workers auto-terminate', async ({ page, request }) => {
        const section = 'Beat 8: Auto-Termination';

        const spawnData = await getSpawnChildren(request);
        const terminated = spawnData.children.filter(
            c => c.status === 'terminated' || c.status === 'timed_out' || c.status === 'completed'
        );
        const active = spawnData.children.filter(c => c.status === 'running');

        narrator.narrate(section,
            `Spawn state: ${terminated.length} terminated, ${active.length} still active`,
            'TTL expiration and completion triggers auto-cleanup — ephemeral workers leave no trace');

        if (terminated.length > 0) {
            for (const child of terminated) {
                narrator.narrate(section,
                    `  "${child.name}" — ${child.status}, budget remaining: ${child.budget_remaining.toFixed(4)} (returned to parent)`);
            }
        }

        await demoGoto(page);
        await demoPause(page, 1500);
        await navigateToPanel(page, 'spawn');
        await demoPause(page, 2000);
        await narrator.screenshot(page, 'workers-terminated');
    });

    // ========================================================================
    // Beat 9: "Claw synthesizes results"
    // ========================================================================
    test('Beat 9: Claw synthesizes results', async ({ page }) => {
        const section = 'Beat 9: Synthesis';
        narrator.narrate(section,
            'Returning to Chat to see the synthesized response...',
            'The parent agent combines findings from all workers into a unified answer');

        await demoGoto(page);
        await demoPause(page, 1500);
        await navigateToPanel(page, 'chat');
        await dismissContextWarning(page);
        await demoPause(page, 2000);

        // Check for the synthesized response
        const messages = await page.locator('.agent-message').count();
        if (messages > 0) {
            const lastMsg = page.locator('.agent-message').last();
            const text = await lastMsg.textContent().catch(() => '');
            narrator.narrate(section,
                `Synthesized response visible (${text.length} chars)`,
                'The parent combined research from both specialist agents into a comprehensive analysis');

            await scrollChatToBottom(page);
            await demoPause(page, 1500);
        } else {
            narrator.narrate(section,
                'Chat messages not yet visible — the synthesis may still be in progress');
        }

        await dismissContextWarning(page);
        await narrator.screenshot(page, 'synthesized-response');
    });

    // ========================================================================
    // Beat 10: "Spawn history shows the full story"
    // ========================================================================
    test('Beat 10: Spawn history shows the full story', async ({ page, request }) => {
        const section = 'Beat 10: Spawn History';
        narrator.narrate(section,
            'Viewing the complete spawn history timeline...',
            'Every spawn event, completion, budget transfer, and termination is logged with timestamps');

        await demoGoto(page);
        await demoPause(page, 1500);
        await navigateToPanel(page, 'spawn');
        await demoPause(page, 2000);

        // Fetch final history
        const spawnData = await getSpawnChildren(request);

        if (spawnData.history.length > 0) {
            narrator.narrate(section,
                `Spawn history: ${spawnData.history.length} event(s) recorded`);

            for (const event of spawnData.history.slice(0, 10)) {
                narrator.narrate(section,
                    `  ${event.event}: "${event.child_name}" — ${event.status || 'n/a'}${event.budget_consumed ? ` (budget: ${event.budget_consumed.toFixed(4)})` : ''}`);
            }
        } else {
            narrator.narrate(section,
                'No spawn history entries — the demo captured the lifecycle in earlier beats');
        }

        // Highlight the history section
        try {
            await highlightElement(page, '#spawn-history-list', 'Spawn History Timeline');
            await demoPause(page, 2500);
        } catch { /* element may not exist */ }

        await narrator.screenshot(page, 'spawn-history');
        await clearHighlights(page);

        narrator.narrate(section,
            'Demo complete — the full spawn lifecycle has been narrated',
            'Agent delegation: request → spawn → delegate → research → budget → collect → synthesize → cleanup');
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
