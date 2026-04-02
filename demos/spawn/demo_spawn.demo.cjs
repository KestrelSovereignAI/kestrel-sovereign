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
    scrollChatToBottom,
    getSpawnChildren,
    waitForSpawnChildren,
    waitForSpawnCompletion,
} = require('../../tests/e2e/demo_helpers.cjs');

const BASE_URL = process.env.KESTREL_URL || 'http://localhost:8888';
const OUTPUT_DIR = path.join(__dirname, 'demo-output');

// ============================================================================
// Demo
// ============================================================================

const narrator = new NarrationEngine({ title: 'Kestrel Sovereign — Spawn Demo Transcript' });
let apiKey = null;

test.describe.serial('Kestrel Spawn Agent Demo', () => {

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

        // Start a fresh session
        try {
            const headers = { ...authHeaders(apiKey) };
            const resp = await request.post(`${BASE_URL}/api/conversations/new`, { headers });
            if (resp.ok()) {
                const data = await resp.json();
                narrator.narrate(`Fresh session started: ${data.session_id || 'ok'}`);
            }
        } catch (e) {
            narrator.narrate(`Could not start fresh session: ${e.message}`);
        }
    });

    // ========================================================================
    // Beat 1: "Claw receives a complex research request"
    // ========================================================================
    test('Beat 1: Claw receives a complex research request', async ({ page }) => {
        narrator.act(1, 'Research Request');
        narrator.narrate('Loading the Sovereign Console...');

        await demoGoto(page, BASE_URL, apiKey);
        await demoPause(page, 2000);

        // Navigate to chat
        narrator.narrate('Opening the Chat panel...');
        await navigateToPanel(page, 'chat');
        await dismissContextWarning(page);

        narrator.narrate('User sends a complex, multi-faceted research request');
        narrator.narrate('This request has multiple independent axes — ideal for parallel investigation by specialist agents', { callout: true });

        await page.locator('#message-input').fill(
            'I need a comprehensive analysis of sovereign AI architecture. ' +
            'Research two areas in parallel: (1) How decentralized identity (DIDs) enables agent sovereignty, ' +
            'and (2) How constitutional governance prevents AI misalignment. ' +
            'Spawn specialist agents to research each area independently, then synthesize their findings.'
        );
        await demoPause(page, 2000);

        await dismissContextWarning(page);
        await demoScreenshot(narrator, page, OUTPUT_DIR, 'research-request');
        narrator.narrate('The complex research request is ready to send');
    });

    // ========================================================================
    // Beat 2: "Claw decides to parallelize"
    // Send message and poll spawn API in parallel — catch children alive
    // ========================================================================
    test('Beat 2: Claw decides to parallelize', async ({ page, request }) => {
        narrator.act(2, 'Decision to Spawn');
        await demoGoto(page, BASE_URL, apiKey);
        await demoPause(page, 1500);

        await navigateToPanel(page, 'chat');
        await dismissContextWarning(page);

        narrator.narrate('Sending the research request — the agent will reason about how to approach it');
        narrator.narrate('Watch for the agent deciding to spawn specialist workers rather than tackle everything sequentially', { callout: true });

        // Send message WITHOUT waiting — we need to poll spawn API while streaming
        await page.locator('#message-input').fill(
            'I need a comprehensive analysis of sovereign AI architecture. ' +
            'Research two areas in parallel: (1) How decentralized identity (DIDs) enables agent sovereignty, ' +
            'and (2) How constitutional governance prevents AI misalignment. ' +
            'Spawn specialist agents to research each area independently, then synthesize their findings.'
        );
        await page.locator('#send-button').click();

        // Poll spawn API while agent is streaming — catch children alive
        narrator.narrate('Message sent — polling spawn API to catch children as they appear...');
        narrator.narrate('The agent will call spawn_agent during its response cycle', { callout: true });

        const spawnData = await waitForSpawnChildren(request, BASE_URL, apiKey, 1, 120000);

        if (spawnData.children.length > 0) {
            narrator.narrate(`${spawnData.children.length} child agent(s) detected while agent is still processing!`);
            narrator.narrate('Children are alive — switching to Spawn panel to screenshot them', { callout: true });

            // Switch to spawn panel to screenshot live children
            await navigateToPanel(page, 'spawn');
            await demoPause(page, 2000);
            await clearHighlights(page);
            await highlightElement(page, '#spawn-children-list', 'Active Child Agents');
            await demoScreenshot(narrator, page, OUTPUT_DIR, 'agent-decides-to-spawn');

            // Switch back to chat and wait for response to complete
            await navigateToPanel(page, 'chat');
        } else {
            narrator.narrate('Children spawned and completed before we could catch them — checking history');
            await demoScreenshot(narrator, page, OUTPUT_DIR, 'agent-decides-to-spawn');
        }

        // Now wait for the agent response to finish
        try {
            await page.waitForFunction(
                () => {
                    const msgs = document.querySelectorAll('.agent-message');
                    if (msgs.length === 0) return false;
                    const last = msgs[msgs.length - 1];
                    return (last.textContent || '').trim().length > 5 &&
                           !last.querySelector('.streaming');
                },
                { timeout: 180000 }
            );
        } catch (e) {
            // Response may have already completed while we were on spawn panel
        }

        await demoPause(page, 1000);
        await dismissContextWarning(page);
        await scrollChatToBottom(page);
    });

    // ========================================================================
    // Beat 3: "Two workers appear in the Console"
    // ========================================================================
    test('Beat 3: Workers appear in the Console', async ({ page, request }) => {
        narrator.act(3, 'Workers Appear');
        narrator.narrate('Navigating to the Spawn panel to see spawned children...');

        await demoGoto(page, BASE_URL, apiKey);
        await demoPause(page, 1500);
        await navigateToPanel(page, 'spawn');
        await demoPause(page, 2000);

        // Check spawn API — children may be alive or in history by now
        const spawnData = await getSpawnChildren(request, BASE_URL, apiKey);

        if (spawnData.children.length > 0) {
            narrator.narrate(`${spawnData.children.length} child agent(s) visible in the Spawn panel`);
            narrator.narrate('Each child has a name, purpose, DID, TTL countdown, and budget allocation', { callout: true });

            for (const child of spawnData.children) {
                narrator.narrate(`  Child: "${child.name}" — ${child.purpose || 'working'} (status: ${child.status}, budget: ${child.budget_allocated})`);
            }
        } else if (spawnData.history.length > 0) {
            narrator.narrate(`Children already completed — ${spawnData.history.length} event(s) in spawn history`);
            narrator.narrate('Children lived and died during the agent response cycle', { callout: true });
            for (const event of spawnData.history) {
                narrator.narrate(`  ${event.event}: "${event.child_name}" — ${event.status || 'done'}`);
            }
        } else {
            narrator.narrate('No spawn children detected via API — the agent may handle this request without spawning');
            narrator.narrate('Spawn is optional — the agent decides based on task complexity', { callout: true });
        }

        // Highlight the children list
        try {
            await highlightElement(page, '#spawn-children-list', 'Active Child Agents');
            await demoPause(page, 2000);
        } catch { /* element may not exist */ }

        await demoScreenshot(narrator, page, OUTPUT_DIR, 'spawn-panel-children');
        await clearHighlights(page);
    });

    // ========================================================================
    // Beat 4: "Workers begin independent research" (delegation chain)
    // ========================================================================
    test('Beat 4: Delegation chain visualization', async ({ page, request }) => {
        narrator.act(4, 'Delegation Chain');
        await demoGoto(page, BASE_URL, apiKey);
        await demoPause(page, 1500);
        await navigateToPanel(page, 'spawn');
        await demoPause(page, 1500);

        const spawnData = await getSpawnChildren(request, BASE_URL, apiKey);

        if (spawnData.delegation_chain && spawnData.delegation_chain.name) {
            narrator.narrate('Delegation chain shows the parent-child relationship tree');
            narrator.narrate('The parent agent at the root delegates to specialist children — each with scoped constitutional authority', { callout: true });

            const chainChildren = spawnData.delegation_chain.children || [];
            narrator.narrate(`Chain: ${spawnData.delegation_chain.name} -> [${chainChildren.map(c => c.name).join(', ')}]`);
        } else {
            narrator.narrate('Delegation chain not yet populated — workers may still be initializing');
        }

        // Highlight the delegation chain
        try {
            await highlightElement(page, '#spawn-delegation-chain', 'Delegation Chain');
            await demoPause(page, 2500);
        } catch { /* element may not exist */ }

        await demoScreenshot(narrator, page, OUTPUT_DIR, 'delegation-chain');
        await clearHighlights(page);
    });

    // ========================================================================
    // Beat 5: "Budget meters tick"
    // ========================================================================
    test('Beat 5: Budget meters tick', async ({ page, request }) => {
        narrator.act(5, 'Budget Meters');
        await demoGoto(page, BASE_URL, apiKey);
        await demoPause(page, 1500);
        await navigateToPanel(page, 'spawn');
        await demoPause(page, 2000);

        const spawnData = await getSpawnChildren(request, BASE_URL, apiKey);
        const withBudget = spawnData.children.filter(c => c.budget_allocated > 0);

        if (withBudget.length > 0) {
            narrator.narrate('Budget meters show real-time spending per child agent');
            narrator.narrate('Each worker has a capped budget — overspend is blocked, unspent is returned to the parent', { callout: true });

            for (const child of withBudget) {
                const pct = child.budget_allocated > 0
                    ? Math.round((child.budget_spent / child.budget_allocated) * 100)
                    : 0;
                narrator.narrate(`  "${child.name}": ${pct}% budget used (${child.budget_spent.toFixed(4)} / ${child.budget_allocated.toFixed(4)})`);
            }
        } else {
            narrator.narrate('No budget data yet — workers may not have budget allocations or may still be initializing');
            narrator.narrate('Budget enforcement is optional — agents can spawn children without budget constraints', { callout: true });
        }

        // Highlight the budget chart
        try {
            await highlightElement(page, '#spawn-budget-chart', 'Budget Allocation');
            await demoPause(page, 2500);
        } catch { /* chart may not exist */ }

        await demoScreenshot(narrator, page, OUTPUT_DIR, 'budget-meters');
        await clearHighlights(page);
    });

    // ========================================================================
    // Beat 6: "First worker reports back"
    // ========================================================================
    test('Beat 6: First worker reports back', async ({ page, request }) => {
        narrator.act(6, 'First Result');

        // Wait for at least one child to complete
        narrator.narrate('Waiting for workers to complete their research...');

        const spawnData = await waitForSpawnCompletion(request, BASE_URL, apiKey, 120000);
        const completed = spawnData.children.filter(c => c.status !== 'running');
        const history = spawnData.history.filter(h => h.event === 'terminated');

        if (completed.length > 0 || history.length > 0) {
            const firstDone = completed[0] || history[0];
            narrator.narrate(`Worker "${firstDone.child_name || firstDone.name}" has completed`);
            narrator.narrate('Results flow back to the parent agent for synthesis', { callout: true });

            if (firstDone.budget_consumed !== undefined) {
                narrator.narrate(`  Budget consumed: ${Number(firstDone.budget_consumed).toFixed(4)}`);
            }
        } else {
            narrator.narrate('Workers still in progress — spawn lifecycle continues asynchronously');
            narrator.narrate('In a live demo, workers complete over seconds to minutes depending on LLM response time', { callout: true });
        }

        await demoGoto(page, BASE_URL, apiKey);
        await demoPause(page, 1500);
        await navigateToPanel(page, 'spawn');
        await demoPause(page, 2000);
        await demoScreenshot(narrator, page, OUTPUT_DIR, 'first-worker-done');
    });

    // ========================================================================
    // Beat 7: "Second worker reports back"
    // ========================================================================
    test('Beat 7: Second worker reports back', async ({ page, request }) => {
        narrator.act(7, 'All Results In');

        const spawnData = await getSpawnChildren(request, BASE_URL, apiKey);
        const allDone = spawnData.children.every(c => c.status !== 'running');
        const historyCount = spawnData.history.length;

        if (allDone && spawnData.children.length > 0) {
            narrator.narrate(`All ${spawnData.children.length} workers have completed — results ready for synthesis`);
            narrator.narrate('Budget accounting is now final for all children', { callout: true });
        } else if (historyCount > 0) {
            narrator.narrate(`${historyCount} spawn event(s) recorded in history`);
            narrator.narrate('Workers have been processed — checking final state', { callout: true });
        } else {
            narrator.narrate('Checking final worker state...');
        }

        await demoGoto(page, BASE_URL, apiKey);
        await demoPause(page, 1500);
        await navigateToPanel(page, 'spawn');
        await demoPause(page, 2000);

        // Show updated budget chart with final numbers
        try {
            await highlightElement(page, '#spawn-budget-chart', 'Final Budget Accounting');
            await demoPause(page, 2000);
        } catch { /* chart may not exist */ }

        await demoScreenshot(narrator, page, OUTPUT_DIR, 'all-workers-done');
        await clearHighlights(page);
    });

    // ========================================================================
    // Beat 8: "Workers auto-terminate"
    // ========================================================================
    test('Beat 8: Workers auto-terminate', async ({ page, request }) => {
        narrator.act(8, 'Auto-Termination');

        const spawnData = await getSpawnChildren(request, BASE_URL, apiKey);
        const terminated = spawnData.children.filter(
            c => c.status === 'terminated' || c.status === 'timed_out' || c.status === 'completed'
        );
        const active = spawnData.children.filter(c => c.status === 'running');

        narrator.narrate(`Spawn state: ${terminated.length} terminated, ${active.length} still active`);
        narrator.narrate('TTL expiration and completion triggers auto-cleanup — ephemeral workers leave no trace', { callout: true });

        if (terminated.length > 0) {
            for (const child of terminated) {
                narrator.narrate(`  "${child.name}" — ${child.status}, budget remaining: ${child.budget_remaining.toFixed(4)} (returned to parent)`);
            }
        }

        await demoGoto(page, BASE_URL, apiKey);
        await demoPause(page, 1500);
        await navigateToPanel(page, 'spawn');
        await demoPause(page, 2000);
        await demoScreenshot(narrator, page, OUTPUT_DIR, 'workers-terminated');
    });

    // ========================================================================
    // Beat 9: "Claw synthesizes results"
    // ========================================================================
    test('Beat 9: Claw synthesizes results', async ({ page }) => {
        narrator.act(9, 'Synthesis');
        narrator.narrate('Returning to Chat to see the synthesized response...');
        narrator.narrate('The parent agent combines findings from all workers into a unified answer', { callout: true });

        await demoGoto(page, BASE_URL, apiKey);
        await demoPause(page, 1500);
        await navigateToPanel(page, 'chat');
        await dismissContextWarning(page);
        await demoPause(page, 2000);

        // Check for the synthesized response
        const messages = await page.locator('.agent-message').count();
        if (messages > 0) {
            const lastMsg = page.locator('.agent-message').last();
            const text = await lastMsg.textContent().catch(() => '');
            narrator.narrate(`Synthesized response visible (${text.length} chars)`);
            narrator.narrate('The parent combined research from both specialist agents into a comprehensive analysis', { callout: true });

            await scrollChatToBottom(page);
            await demoPause(page, 1500);
        } else {
            narrator.narrate('Chat messages not yet visible — the synthesis may still be in progress');
        }

        await dismissContextWarning(page);
        await demoScreenshot(narrator, page, OUTPUT_DIR, 'synthesized-response');
    });

    // ========================================================================
    // Beat 10: "Spawn history shows the full story"
    // ========================================================================
    test('Beat 10: Spawn history shows the full story', async ({ page, request }) => {
        narrator.act(10, 'Spawn History');
        narrator.narrate('Viewing the complete spawn history timeline...');
        narrator.narrate('Every spawn event, completion, budget transfer, and termination is logged with timestamps', { callout: true });

        await demoGoto(page, BASE_URL, apiKey);
        await demoPause(page, 1500);
        await navigateToPanel(page, 'spawn');
        await demoPause(page, 2000);

        // Fetch final history
        const spawnData = await getSpawnChildren(request, BASE_URL, apiKey);

        if (spawnData.history.length > 0) {
            narrator.narrate(`Spawn history: ${spawnData.history.length} event(s) recorded`);

            for (const event of spawnData.history.slice(0, 10)) {
                narrator.narrate(`  ${event.event}: "${event.child_name}" — ${event.status || 'n/a'}${event.budget_consumed ? ` (budget: ${event.budget_consumed.toFixed(4)})` : ''}`);
            }
        } else {
            narrator.narrate('No spawn history entries — the demo captured the lifecycle in earlier beats');
        }

        // Highlight the history section
        try {
            await highlightElement(page, '#spawn-history-list', 'Spawn History Timeline');
            await demoPause(page, 2500);
        } catch { /* element may not exist */ }

        await demoScreenshot(narrator, page, OUTPUT_DIR, 'spawn-history');
        await clearHighlights(page);

        narrator.narrate('Demo complete — the full spawn lifecycle has been narrated');
        narrator.narrate('Agent delegation: request → spawn → delegate → research → budget → collect → synthesize → cleanup', { callout: true });
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
