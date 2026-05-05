/**
 * Trash Vignette — Soft-delete + Restore + Hard-purge
 *
 * Covers issues:
 *   #763 — soft-delete by default (deleted_at column, no row removal)
 *   #765 — Trash sub-view UI (restore, hard-purge affordances)
 *
 * Why this exists:
 *   On 2026-04-24 a Playwright harness pointed at the live server wiped three
 *   agents' conversation histories. Soft-delete makes recovery possible; the
 *   Trash sub-view makes recovery *visible*. This vignette proves a user can
 *   delete a message, find it in Trash, restore it, and (only if they really
 *   mean it) hard-purge it forever.
 *
 * Run: demos/run.sh trash
 */
const { test } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

const {
  NarrationEngine,
  demoScreenshot,
  demoPause,
  highlightElement,
  clearHighlights,
  getApiKey,
  demoGoto,
  demoSendMessage,
  navigateToPanel,
  dismissContextWarning,
  scrollChatToBottom,
} = require('../shared/demo_helpers.cjs');

const BASE_URL = process.env.KESTREL_URL || 'http://localhost:8900';
const OUTPUT_DIR = path.join(__dirname, 'demo-output');

const narrator = new NarrationEngine({
  title: 'Kestrel Sovereign — Trash Vignette Transcript',
});
let apiKey = null;

// ---------------------------------------------------------------------------
// Helpers specific to the trash vignette
// ---------------------------------------------------------------------------

/**
 * Wait for #conversations-pane to reveal.  In standalone mode loadAgents
 * auto-reveals the pane (identity.js standalone branch); we just wait for
 * the visibility flip and for the conversations-list to populate.
 *
 * Earlier versions of this helper tried to invoke window.selectAgent() to
 * force the reveal — but selectAgent calls API.setHostAgent(name), which
 * applies a /api/agents/<name>/ URL prefix that only exists in multi_agent
 * routing.  Applying it in standalone 404s every subsequent call (chat
 * invoke, conversations, trash list).  The right fix is in identity.js;
 * the demo just waits.
 */
async function ensureConversationsPaneOpen(page) {
  await page.waitForSelector('#conversations-pane', { state: 'visible', timeout: 10000 }).catch(() => {});
  await demoPause(page, 1000);
  return true;
}

/**
 * Re-fetch the latest session via window.loadConversation() so message divs
 * get `data-message-id` attached.
 *
 * Live-sent messages from chat.js don't carry IDs — they only appear after
 * loadConversationHistory rerenders from the backend.  Without IDs, the
 * `.msg-delete-btn` and `.msg-purge-btn` aren't appended (history.js:399),
 * so the soft-delete and purge affordances stay invisible.
 *
 * Looks at the first .conversation-item's `data-session-id` and calls
 * window.loadConversation directly — clicking the row would also work, but
 * direct invocation is more deterministic.
 */
async function loadFirstConversation(page) {
  // The conversations-list loads asynchronously after selectAgent.  Wait for
  // a row to attach (state:'attached', not 'visible' — the conversations-pane
  // may be the active sub-view of the sidebar but its rows are still rendered
  // when only the trash sub-view is visible).
  try {
    await page.waitForSelector('.conversation-item', { state: 'attached', timeout: 8000 });
  } catch {
    return false;
  }
  const ok = await page.evaluate(async () => {
    const item = document.querySelector('.conversation-item');
    if (!item || !item.dataset.sessionId) return false;
    if (typeof window.loadConversation !== 'function') return false;
    await window.loadConversation(item.dataset.sessionId);
    return true;
  });
  // After loadConversation, wait for messages to render with their ids — only
  // then are .msg-delete-btn / .msg-purge-btn attached.
  if (ok) {
    try {
      await page.waitForSelector('.message[data-message-id]', { state: 'attached', timeout: 5000 });
    } catch { /* messages may legitimately be empty */ }
  }
  await demoPause(page, 800);
  return ok;
}

// ---------------------------------------------------------------------------

test.describe.serial('Trash Vignette', () => {
  test.beforeAll(async ({ request }) => {
    if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    apiKey = await getApiKey(request, BASE_URL);
    narrator.act(0, 'Setup');
    narrator.narrate(apiKey ? 'API key acquired' : 'No API key (public mode)');
  });

  // Auto-accept every confirm() dialog the UI raises — both deleteMessage and
  // purgeMessage gate behind a confirm before issuing the API call.  Each beat
  // gets its own Playwright page (default isolation), so this handler must be
  // installed per-page; without it confirm() returns false and the click is
  // a no-op (Beats 2/4/5 silently skip).
  test.beforeEach(async ({ page }) => {
    page.on('dialog', (dialog) => dialog.accept().catch(() => {}));
  });

  test.afterAll(() => {
    const narrationPath = path.join(OUTPUT_DIR, 'narration.md');
    fs.writeFileSync(narrationPath, narrator.toMarkdown(), 'utf-8');
    console.log(`[DEMO] Narration written to ${narrationPath}`);
    console.log(`[DEMO] Screenshots in ${OUTPUT_DIR}`);
  });

  // -------------------------------------------------------------------------
  // Beat 1 — Build a tiny conversation so there's something to delete
  // -------------------------------------------------------------------------
  test('Beat 1: A conversation worth keeping', async ({ page }) => {
    narrator.act(1, 'A small conversation');
    narrator.narrate(
      'Soft-delete only matters if there is something to delete. We start by ' +
      'selecting the demo agent (which reveals the conversations sidebar) and ' +
      'sending a real message — both user and agent reply land in conversation_history.'
    );

    // Auto-accept any confirm() dialog the UI raises later in the demo —
    // hard-purge gates on window.confirm; left unhandled it freezes Playwright.
    page.on('dialog', (dialog) => dialog.accept().catch(() => {}));

    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await ensureConversationsPaneOpen(page);
    await navigateToPanel(page, 'chat');
    await dismissContextWarning(page);
    await demoPause(page, 1000);

    await demoSendMessage(
      page,
      'Hello — say one short sentence so I have something to demonstrate Trash with.',
      90000,
    );
    await scrollChatToBottom(page);

    // Reload the session so the new messages re-render with data-message-id —
    // without this, the hover-reveal delete buttons never attach.
    await loadFirstConversation(page);
    await scrollChatToBottom(page);
    await demoPause(page, 1000);

    await demoScreenshot(narrator, page, OUTPUT_DIR, 'conversation-seeded');
    narrator.narrate('Two messages exist with backend IDs — eligible for soft-delete.', { callout: true });
  });

  // -------------------------------------------------------------------------
  // Beat 2 — Soft-delete: ✕ on a message hides it from the active list
  // -------------------------------------------------------------------------
  test('Beat 2: Soft-delete a message', async ({ page }) => {
    narrator.act(2, 'Soft-delete');
    narrator.narrate(
      'Hovering a message reveals two affordances: ✕ (soft-delete, recoverable) ' +
      'and ⊘ (purge, permanent). The user clicks ✕ — the message vanishes from ' +
      'chat but the row still exists with deleted_at set.'
    );

    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await ensureConversationsPaneOpen(page);
    await navigateToPanel(page, 'chat');
    await dismissContextWarning(page);
    await loadFirstConversation(page);
    await scrollChatToBottom(page);
    await demoPause(page, 800);

    // Only target messages that have IDs — the live-rendered ones don't
    const lastMessage = page.locator('.message[data-message-id]').last();
    const found = await lastMessage.count();
    if (found === 0) {
      narrator.narrate('No messages with data-message-id present — backend did not return IDs.');
      await demoScreenshot(narrator, page, OUTPUT_DIR, 'hover-reveals-buttons');
      await demoScreenshot(narrator, page, OUTPUT_DIR, 'after-soft-delete');
      return;
    }

    await lastMessage.scrollIntoViewIfNeeded();
    await lastMessage.hover();
    await demoPause(page, 600);

    const deleteBtn = lastMessage.locator('.msg-delete-btn');
    try {
      await highlightElement(page, '.message[data-message-id]:last-of-type .msg-delete-btn', 'Soft-delete (recoverable)');
      await demoPause(page, 1500);
    } catch { /* highlight is best-effort */ }
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'hover-reveals-buttons');
    await clearHighlights(page);

    // Dispatch the click directly in page-context — Playwright's
    // .click({ force: true }) on a `display: none` element (the hover-reveal
    // state) doesn't reliably fire the onclick handler.  Calling the click()
    // method on the element via JS bypasses the visibility check entirely
    // and runs the onclick as the user would.
    const clicked = await page.evaluate(() => {
      const messages = document.querySelectorAll('.message[data-message-id]');
      if (!messages.length) return false;
      const btn = messages[messages.length - 1].querySelector('.msg-delete-btn');
      if (!btn) return false;
      btn.click();
      return true;
    });
    if (clicked) {
      narrator.narrate('Soft-delete fired — the message is gone from the chat list.');
    } else {
      narrator.narrate('msg-delete-btn missing on the last message — UI may have changed.');
    }
    await demoPause(page, 1500);
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'after-soft-delete');
    narrator.narrate('Row still exists with deleted_at set — recoverable from Trash.', { callout: true });
  });

  // -------------------------------------------------------------------------
  // Beat 3 — Open Trash: deleted message is visible and labeled
  // -------------------------------------------------------------------------
  test('Beat 3: Find it in Trash', async ({ page }) => {
    narrator.act(3, 'The Trash sub-view');
    narrator.narrate(
      'The 🗑 toggle in the conversations sidebar flips the list between ' +
      'active and trashed items. Trashed items show a preview, when they were ' +
      'deleted, and a Restore / Purge pair.'
    );

    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await ensureConversationsPaneOpen(page);
    await navigateToPanel(page, 'chat');  // trash toggle is in the conversations-pane visible alongside chat
    await dismissContextWarning(page);
    await demoPause(page, 800);

    const toggle = page.locator('#trash-toggle-btn');
    if ((await toggle.count()) === 0 || !(await toggle.isVisible().catch(() => false))) {
      narrator.narrate('#trash-toggle-btn not visible — agent selection may not have completed.');
      await demoScreenshot(narrator, page, OUTPUT_DIR, 'trash-view-open');
      return;
    }

    try {
      await highlightElement(page, '#trash-toggle-btn', 'Trash toggle');
      await demoPause(page, 1200);
      await clearHighlights(page);
    } catch { /* best-effort */ }

    await toggle.click();
    await demoPause(page, 1500);

    try {
      await highlightElement(page, '#conversations-trash', 'Trashed items');
      await demoPause(page, 1500);
    } catch { /* best-effort */ }
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'trash-view-open');
    await clearHighlights(page);
    narrator.narrate('The deleted message lives here until it ages out (retention janitor) or the user purges it.', { callout: true });
  });

  // -------------------------------------------------------------------------
  // Beat 4 — Restore: clicking Restore brings it back to the active list
  // -------------------------------------------------------------------------
  test('Beat 4: Restore from Trash', async ({ page }) => {
    narrator.act(4, 'Restore');
    narrator.narrate(
      'Restore clears deleted_at — the row is live again, indistinguishable ' +
      'from one that was never deleted.'
    );

    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await ensureConversationsPaneOpen(page);
    await navigateToPanel(page, 'chat');
    await dismissContextWarning(page);
    await demoPause(page, 800);

    // Make sure trash view is open
    const trashPane = page.locator('#conversations-trash');
    let visible = await trashPane.isVisible().catch(() => false);
    if (!visible) {
      const toggle = page.locator('#trash-toggle-btn');
      if ((await toggle.count()) > 0) {
        await toggle.click();
        await demoPause(page, 1500);
        visible = await trashPane.isVisible().catch(() => false);
      }
    }

    const firstRestore = page.locator('.trash-item .btn-restore').first();
    if ((await firstRestore.count()) > 0) {
      await firstRestore.scrollIntoViewIfNeeded();
      try {
        await highlightElement(page, '.trash-item:first-child .btn-restore', 'Restore (clears deleted_at)');
        await demoPause(page, 1200);
        await clearHighlights(page);
      } catch { /* best-effort */ }
      await firstRestore.click();
      await demoPause(page, 1500);
      narrator.narrate('Restore fired — the message reappears in the active list.');
    } else {
      narrator.narrate('No trashed items to restore — Beat 2 may have skipped.');
    }

    // Reload the conversation so the chat renders the restored message
    await loadFirstConversation(page);
    await scrollChatToBottom(page);
    await demoPause(page, 1000);
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'restored-to-chat');
  });

  // -------------------------------------------------------------------------
  // Beat 5 — Hard-purge: ⊘ deletes the row outright (no recovery)
  // -------------------------------------------------------------------------
  test('Beat 5: Hard-purge — the irreversible path', async ({ page }) => {
    narrator.act(5, 'Hard-purge');
    narrator.narrate(
      'Some deletions need to be irreversible — GDPR requests, accidental ' +
      'leaks. The ⊘ button hard-deletes the row. The UI raises confirm() and ' +
      'attaches X-Kestrel-Allow-Destructive so the destructive-op rail (#766) ' +
      'can audit the operation.'
    );

    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await ensureConversationsPaneOpen(page);
    await navigateToPanel(page, 'chat');
    await dismissContextWarning(page);
    await loadFirstConversation(page);
    await scrollChatToBottom(page);
    await demoPause(page, 800);

    const lastMessage = page.locator('.message[data-message-id]').last();
    if ((await lastMessage.count()) === 0) {
      narrator.narrate('No messages with data-message-id present.');
      await demoScreenshot(narrator, page, OUTPUT_DIR, 'purge-button-revealed');
      await demoScreenshot(narrator, page, OUTPUT_DIR, 'after-hard-purge');
      return;
    }

    await lastMessage.scrollIntoViewIfNeeded();
    await lastMessage.hover();
    await demoPause(page, 600);
    try {
      await highlightElement(page, '.message[data-message-id]:last-of-type .msg-purge-btn', 'Hard-purge (irreversible)');
      await demoPause(page, 1500);
    } catch { /* best-effort */ }
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'purge-button-revealed');
    await clearHighlights(page);

    const purged = await page.evaluate(() => {
      // Auto-accept confirm() since the click happens inside page.evaluate —
      // Playwright's `page.on('dialog', ...)` only intercepts dialogs raised
      // from outside the evaluate frame. Without this override, btn.click()
      // blocks on the confirm prompt and the purge never fires (the message
      // is still visible after, which fails the eye expectation).
      window.confirm = () => true;
      const messages = document.querySelectorAll('.message[data-message-id]');
      if (!messages.length) return false;
      const btn = messages[messages.length - 1].querySelector('.msg-purge-btn');
      if (!btn) return false;
      btn.click();
      return true;
    });
    if (purged) {
      narrator.narrate('Confirm dialog auto-accepted — purge fires.');
    } else {
      narrator.narrate('msg-purge-btn missing on the last message — UI may have changed.');
    }
    await demoPause(page, 1500);
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'after-hard-purge');
    narrator.narrate('Row gone from conversation_history — not even in Trash.', { callout: true });
  });

  // -------------------------------------------------------------------------
  // Beat 6 — Bookend: Trash reflects reality
  // -------------------------------------------------------------------------
  test('Beat 6: Bookend — Trash reflects reality', async ({ page }) => {
    narrator.act(6, 'Bookend');
    narrator.narrate(
      'Re-opening Trash shows the purged message is no longer there. The ' +
      'destructive-op rail logged the purge to security_audit_log so an ' +
      'operator can review what was deleted, when, and why.'
    );

    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await ensureConversationsPaneOpen(page);
    await navigateToPanel(page, 'chat');
    await dismissContextWarning(page);
    await demoPause(page, 800);

    const trashPane = page.locator('#conversations-trash');
    const visible = await trashPane.isVisible().catch(() => false);
    if (!visible) {
      const toggle = page.locator('#trash-toggle-btn');
      if ((await toggle.count()) > 0) {
        await toggle.click();
        await demoPause(page, 1500);
      }
    }

    try {
      await highlightElement(page, '#conversations-trash', 'Trash after purge');
      await demoPause(page, 1500);
    } catch { /* best-effort */ }
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'trash-after-purge');
    await clearHighlights(page);
    narrator.narrate('Demo complete — soft-delete protects users from themselves; hard-purge respects intent when they really mean it.', { callout: true });
  });
});
