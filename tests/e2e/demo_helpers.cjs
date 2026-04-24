// @ts-check
const path = require('path');
const { execSync } = require('child_process');

// Core demo infrastructure from @kestrel/flight
const {
  NarrationEngine,
  demoScreenshot,
  demoPause,
  highlightElement,
  clearHighlights,
} = require('@kestrel/flight');

/**
 * Kestrel Sovereign Demo Helpers
 *
 * Sovereign-specific demo utilities built on @kestrel/flight.
 * Core infrastructure (NarrationEngine, screenshots, highlights, pause)
 * comes from kestrel-flight. This file adds sovereign auth, chat interaction,
 * panel navigation, and spawn API helpers.
 *
 * kestrel-eye integration:
 *   Screenshots are reviewed by kestrel-eye using a cheap vision model (Haiku)
 *   against expectations in eye-*.toml configs.
 *   Run: kestrel-eye review --config eye-technical.toml
 *   Loop: kestrel-eye run --config eye-technical.toml --loop
 */

// ---------------------------------------------------------------------------
// Auth helpers
// ---------------------------------------------------------------------------

/**
 * Fetch API key from env or server.
 * @param {import('@playwright/test').APIRequestContext} request
 * @param {string} baseUrl
 * @returns {Promise<string|null>}
 */
async function getApiKey(request, baseUrl) {
  if (process.env.KESTREL_API_KEY) return process.env.KESTREL_API_KEY;
  try {
    const response = await request.get(`${baseUrl}/api/auth/key`);
    if (response.ok()) {
      const data = await response.json();
      return data.key;
    }
  } catch (e) { /* ignore */ }
  return null;
}

/**
 * Build X-API-Key header object.
 * @param {string|null} apiKey
 * @returns {Record<string, string>}
 */
function authHeaders(apiKey) {
  return apiKey ? { 'X-API-Key': apiKey } : {};
}

// ---------------------------------------------------------------------------
// UI interaction helpers (Sovereign-specific selectors)
// ---------------------------------------------------------------------------

/**
 * Send a chat message and wait for agent response.
 * Returns last agent message locator or null.
 * @param {import('@playwright/test').Page} page
 * @param {string} message
 * @param {number} [timeout=90000]
 */
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
    const count = await page.locator('.agent-message').count();
    return count > 0 ? page.locator('.agent-message').last() : null;
  }
}

/**
 * Set API key headers on the page context and navigate to the app.
 * @param {import('@playwright/test').Page} page
 * @param {string} baseUrl
 * @param {string|null} apiKey
 */
async function demoGoto(page, baseUrl, apiKey) {
  if (apiKey) {
    await page.setExtraHTTPHeaders({ 'X-API-Key': apiKey });
  }
  await page.goto(baseUrl);
}

/**
 * Navigate to a named panel.
 * @param {import('@playwright/test').Page} page
 * @param {string} panelName
 */
async function navigateToPanel(page, panelName) {
  await page.click(`.nav-tab[data-panel="${panelName}"]`);
  try {
    await page.waitForSelector(`#panel-${panelName}`, { state: 'visible', timeout: 5000 });
  } catch { /* panel may already be visible */ }
  await demoPause(page, 1000);
}

/**
 * Remove context warnings and hide the noisy status bar for clean screenshots.
 * @param {import('@playwright/test').Page} page
 */
async function dismissContextWarning(page) {
  await page.evaluate(() => {
    document.querySelectorAll('.context-warning').forEach(el => el.remove());
    const status = document.getElementById('context-status');
    if (status) status.style.display = 'none';
  });
}

/**
 * Scroll chat container to show the first user message at the top.
 * @param {import('@playwright/test').Page} page
 */
async function scrollChatToTop(page) {
  await page.evaluate(() => {
    const container = document.getElementById('chat-container');
    if (container) container.scrollTop = 0;
  });
}

/**
 * Scroll chat container to show the latest message.
 * @param {import('@playwright/test').Page} page
 */
async function scrollChatToBottom(page) {
  await page.evaluate(() => {
    const container = document.getElementById('chat-container');
    if (container) container.scrollTop = container.scrollHeight;
  });
}

// ---------------------------------------------------------------------------
// Provider selection
// ---------------------------------------------------------------------------

/**
 * Default preferred provider order for technical demos — local first so the
 * demo runs without cloud API keys, falling back to configured cloud vendors.
 * Matches the order that was duplicated across Acts 2/3/6 in
 * demo_technical.demo.cjs before consolidation.
 */
const DEFAULT_DEMO_PROVIDER_ORDER = [
  'ollama',
  'llama_cpp',
  'llama',
  'openrouter',
  'anthropic',
  'openai',
];

/**
 * Select the first available provider from `#provider-selector` that matches
 * one of the `preferred` substrings (case-insensitive).  Logs the outcome via
 * `narrator` when provided.
 *
 * Returns the matching option label, or `null` if no preference matched (the
 * page's default stays selected).
 *
 * @param {import('@playwright/test').Page} page
 * @param {object} [opts]
 * @param {import('@kestrel/flight').NarrationEngine} [opts.narrator]
 *        Optional narrator for progress narration.  When null, stays silent.
 * @param {string[]} [opts.preferred]
 *        Substrings to look for, in priority order.  Defaults to the
 *        local-first order used by the technical demos.
 * @param {number} [opts.pauseAfterSelect=1000]
 *        ms to pause after a successful selection so the UI settles before
 *        the next action.
 * @param {boolean} [opts.narrateFallback=false]
 *        If true, list the options available when no preference matched.
 * @param {boolean} [opts.narrateSelection=true]
 *        If false, stay silent on successful selection (but still narrate
 *        errors via `narrator`).  Use for acts where the user's focus is on
 *        something other than provider choice.
 * @returns {Promise<string|null>} the label chosen, or null when none matched.
 */
async function selectDemoProvider(page, opts = {}) {
  const {
    narrator = null,
    preferred = DEFAULT_DEMO_PROVIDER_ORDER,
    pauseAfterSelect = 1000,
    narrateFallback = false,
    narrateSelection = true,
  } = opts;

  try {
    const providerSelect = page.locator('#provider-selector');
    const options = await providerSelect.locator('option').allTextContents();
    for (const pref of preferred) {
      const match = options.find((o) => o.toLowerCase().includes(pref));
      if (match) {
        await providerSelect.selectOption({ label: match });
        if (narrator && narrateSelection) narrator.narrate(`Provider set to: ${match}`);
        if (pauseAfterSelect > 0) await demoPause(page, pauseAfterSelect);
        return match;
      }
    }
    if (narrator && narrateFallback) {
      narrator.narrate(`Using default provider (available: ${options.join(', ')})`);
    }
    return null;
  } catch (e) {
    if (narrator) narrator.narrate(`Could not set provider: ${e.message}`);
    return null;
  }
}

// ---------------------------------------------------------------------------
// Session management
// ---------------------------------------------------------------------------

/**
 * Clear old conversation history so the demo starts with a clean context window.
 * Deletes directly from SQLite because there is no bulk-clear API endpoint.
 * @param {import('@kestrel/flight').NarrationEngine} narrator
 * @param {string} agentDataDir - absolute path to agent_data/
 */
function clearConversationHistory(narrator, agentDataDir) {
  try {
    const dbs = execSync(`find "${agentDataDir}" -name "kestrel_prime.db" -maxdepth 3 2>/dev/null`)
      .toString().trim().split('\n').filter(Boolean);
    for (const db of dbs) {
      try {
        execSync(`sqlite3 "${db}" "DELETE FROM conversation_history;"`);
      } catch { /* table may not exist in some dbs */ }
      try {
        execSync(`sqlite3 "${db}" "DELETE FROM graph_nodes WHERE node_type IN ('backup_artifact', 'sovereignty_receipt');"`);
      } catch { /* table may not exist */ }
      try {
        execSync(`sqlite3 "${db}" "DELETE FROM security_audit_log;"`);
      } catch { /* table may not exist */ }
      try {
        execSync(`sqlite3 "${db}" "UPDATE security_permissions SET level = 'ask' WHERE level != 'ask';"`);
      } catch { /* table may not exist */ }
    }
    narrator.narrate(`Cleared conversation history, exports, and permissions from ${dbs.length} agent database(s)`);
  } catch (e) {
    narrator.narrate(`Could not clear history: ${e.message}`);
  }
}

/**
 * Start a fresh session via the API.
 * @param {import('@playwright/test').APIRequestContext} request
 * @param {string} baseUrl
 * @param {string|null} apiKey
 * @param {import('@kestrel/flight').NarrationEngine} narrator
 */
async function startFreshSession(request, baseUrl, apiKey, narrator) {
  try {
    const headers = apiKey ? { 'X-API-Key': apiKey } : {};
    const resp = await request.post(`${baseUrl}/api/conversations/new`, { headers });
    if (resp.ok()) {
      const data = await resp.json();
      narrator.narrate(`Fresh session started: ${data.session_id || 'ok'}`);
      return true;
    }
  } catch (e) {
    narrator.narrate(`Could not start fresh session: ${e.message}`);
  }
  return false;
}

// ---------------------------------------------------------------------------
// Spawn API helpers
// ---------------------------------------------------------------------------

/**
 * Fetch spawn children data from the API.
 * @param {import('@playwright/test').APIRequestContext} request
 * @param {string} baseUrl
 * @param {string|null} apiKey
 */
async function getSpawnChildren(request, baseUrl, apiKey) {
  try {
    const response = await request.get(`${baseUrl}/api/spawn/children`, {
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

/**
 * Wait for spawn children to appear (polling).
 * @param {import('@playwright/test').APIRequestContext} request
 * @param {string} baseUrl
 * @param {string|null} apiKey
 * @param {number} [minCount=1]
 * @param {number} [timeout=60000]
 */
async function waitForSpawnChildren(request, baseUrl, apiKey, minCount = 1, timeout = 60000) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    const data = await getSpawnChildren(request, baseUrl, apiKey);
    if (data.children.length >= minCount) return data;
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
  return await getSpawnChildren(request, baseUrl, apiKey);
}

/**
 * Wait for spawn children to reach a terminal state (polling).
 * @param {import('@playwright/test').APIRequestContext} request
 * @param {string} baseUrl
 * @param {string|null} apiKey
 * @param {number} [timeout=180000]
 */
async function waitForSpawnCompletion(request, baseUrl, apiKey, timeout = 180000) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    const data = await getSpawnChildren(request, baseUrl, apiKey);
    const allDone = data.children.length > 0 &&
      data.children.every(c => c.status !== 'running');
    if (allDone) return data;
    if (data.history.length > 0 && data.children.length === 0) return data;
    await new Promise(resolve => setTimeout(resolve, 3000));
  }
  return await getSpawnChildren(request, baseUrl, apiKey);
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

module.exports = {
  // Re-exported from @kestrel/flight
  NarrationEngine,
  demoScreenshot,
  demoPause,
  highlightElement,
  clearHighlights,
  // Sovereign-specific
  getApiKey,
  authHeaders,
  demoSendMessage,
  demoGoto,
  navigateToPanel,
  dismissContextWarning,
  scrollChatToTop,
  scrollChatToBottom,
  clearConversationHistory,
  startFreshSession,
  // Provider selection
  selectDemoProvider,
  DEFAULT_DEMO_PROVIDER_ORDER,
  // Spawn helpers
  getSpawnChildren,
  waitForSpawnChildren,
  waitForSpawnCompletion,
};
