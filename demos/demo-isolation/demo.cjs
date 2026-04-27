/**
 * Demo Isolation Vignette — the destructive-op rail (#766)
 *
 * Why this exists:
 *   On 2026-04-24 a Playwright harness pointed at the live server wiped
 *   three agents' conversation histories.  The rail this vignette
 *   demonstrates would have refused that wipe.  It's a server-side prevention
 *   layer: every destructive endpoint sits behind enforce_destructive_op,
 *   which requires the X-Kestrel-Allow-Destructive header on a live agent.
 *
 * Honesty note:
 *   `demos/run.sh` runs against a demo-scoped agent (is_demo=True).  The
 *   rail INTENTIONALLY allows destructive ops on demo-scoped targets — that's
 *   what makes demos safe to run.  So this vignette can't actually trigger
 *   the 403 path against the demo agent.  Instead it RENDERS the
 *   refusal/allowed banners that the rail produces against a *live* target
 *   so the operator can see the visible story.  The 403 path is exercised
 *   by tests/unit/test_demo_isolation.py.
 *
 * Beats:
 *   1. Healthy state — a live conversation (the thing the rail protects).
 *   2. The refusal — what a misbehaving script sees against a live agent.
 *   3. The allow — what the production UI sees when it carries the header.
 *   4. Bookend — the security audit log surface.
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
  title: 'Kestrel Sovereign — Demo-Isolation Rail Vignette Transcript',
});
let apiKey = null;

async function renderBanner(page, opts) {
  await page.evaluate(({ kind, status, title, body, footer }) => {
    document.getElementById('demo-isolation-banner')?.remove();
    const banner = document.createElement('div');
    banner.id = 'demo-isolation-banner';
    const isRefused = kind === 'refused';
    const grad = isRefused
      ? 'linear-gradient(135deg, #fee2e2, #fecaca)'
      : 'linear-gradient(135deg, #dcfce7, #bbf7d0)';
    const border = isRefused ? '#dc2626' : '#16a34a';
    const text = isRefused ? '#7f1d1d' : '#14532d';
    const shadow = isRefused
      ? '0 8px 32px rgba(220, 38, 38, 0.25)'
      : '0 8px 32px rgba(22, 163, 74, 0.25)';
    banner.style.cssText = `
      position: fixed; top: 80px; left: 50%; transform: translateX(-50%);
      max-width: 640px; padding: 1.25rem 1.5rem;
      background: ${grad}; border: 2px solid ${border};
      border-radius: 12px; color: ${text};
      font-family: ui-monospace, monospace; font-size: 0.85rem;
      box-shadow: ${shadow}; z-index: 9999; line-height: 1.5;
    `;
    banner.innerHTML = `
      <div style="font-weight: 700; font-size: 1rem; margin-bottom: 0.5rem;">
        ${isRefused ? '⛔' : '✅'} ${status} — ${title}
      </div>
      <div style="opacity: 0.9;">${body}</div>
      ${footer ? `<div style="margin-top: 0.75rem; font-size: 0.7rem; opacity: 0.75;">${footer}</div>` : ''}
    `;
    document.body.appendChild(banner);
  }, opts);
}

test.describe.serial('Demo Isolation Vignette', () => {
  test.beforeAll(async ({ request }) => {
    if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    apiKey = await getApiKey(request, BASE_URL);
    narrator.act(0, 'Setup');
    narrator.narrate(apiKey ? 'API key acquired' : 'No API key (public mode)');
  });

  test.afterAll(() => {
    const narrationPath = path.join(OUTPUT_DIR, 'narration.md');
    fs.writeFileSync(narrationPath, narrator.toMarkdown(), 'utf-8');
    console.log(`[DEMO] Narration written to ${narrationPath}`);
  });

  // -------------------------------------------------------------------------
  // Beat 1 — A real conversation exists.
  // -------------------------------------------------------------------------
  test('Beat 1: Healthy state — there is something to lose', async ({ page }) => {
    narrator.act(1, 'A live conversation');
    narrator.narrate(
      'The 2026-04-24 incident happened because a Playwright harness pointed ' +
      'at the live server and called destructive APIs against three agents.  ' +
      'Their conversation histories were wiped.  The rail in #766 is the ' +
      'server-side prevention layer that would have refused those calls.'
    );

    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await navigateToPanel(page, 'chat');
    await dismissContextWarning(page);
    await demoPause(page, 1000);
    await demoSendMessage(page, 'Say one short sentence about Kestrel.', 90000);
    await scrollChatToBottom(page);
    await demoPause(page, 1500);
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'live-agent-with-data');
    narrator.narrate('Real persistence — exactly the surface the rail protects.', { callout: true });
  });

  // -------------------------------------------------------------------------
  // Beat 2 — The refusal banner.
  // -------------------------------------------------------------------------
  test('Beat 2: Refusal — destructive call without the opt-in header', async ({ page }) => {
    narrator.act(2, 'Refusal');
    narrator.narrate(
      'Against a *live* agent, a DELETE without X-Kestrel-Allow-Destructive ' +
      'is refused.  The rail returns 403 and writes an audit row to ' +
      'security_audit_log with the caller IP, the endpoint, the redacted ' +
      'headers, and the decision "refused-no-destructive-header".'
    );
    narrator.narrate(
      'This demo runs against a demo-scoped agent (is_demo=True), where the ' +
      'rail allows destructive ops by design — that\'s what makes demos ' +
      'safe to run.  The screenshot below shows the banner the rail emits ' +
      'when it does refuse.  Unit tests in tests/unit/test_demo_isolation.py ' +
      'exercise the actual 403 path.'
    );

    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await navigateToPanel(page, 'chat');
    await dismissContextWarning(page);
    await demoPause(page, 800);

    await renderBanner(page, {
      kind: 'refused',
      status: '403',
      title: 'Destructive op refused',
      body: 'Destructive operations on live agents require the <code>X-Kestrel-Allow-Destructive</code> header carrying a reason.',
      footer: 'audit row written: decision=refused-no-destructive-header',
    });
    await demoPause(page, 1500);
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'refused-without-header');
    await page.evaluate(() => document.getElementById('demo-isolation-banner')?.remove());
  });

  // -------------------------------------------------------------------------
  // Beat 3 — The allowed banner.
  // -------------------------------------------------------------------------
  test('Beat 3: Allowed — header carries an explicit reason', async ({ page }) => {
    narrator.act(3, 'Allowed with intent');
    narrator.narrate(
      'The production UI attaches X-Kestrel-Allow-Destructive automatically ' +
      'with a free-text reason.  Scripts that legitimately need destructive ' +
      'ops opt in explicitly.  Either way the rail records the reason so an ' +
      'operator can review what fired and why.'
    );

    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await navigateToPanel(page, 'chat');
    await dismissContextWarning(page);
    await demoPause(page, 800);

    await renderBanner(page, {
      kind: 'allowed',
      status: '200',
      title: 'Destructive op allowed',
      body: 'Header present: <code>X-Kestrel-Allow-Destructive: user-initiated-ui</code>',
      footer: 'audit row written: decision=allowed-with-header, reason=user-initiated-ui',
    });
    await demoPause(page, 1500);
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'allowed-with-header');
    await page.evaluate(() => document.getElementById('demo-isolation-banner')?.remove());
  });

  // -------------------------------------------------------------------------
  // Beat 4 — The audit trail surface.
  // -------------------------------------------------------------------------
  test('Beat 4: Bookend — the audit trail', async ({ page }) => {
    narrator.act(4, 'Bookend');
    narrator.narrate(
      'Every refusal and every header-carrying allow lands in ' +
      'security_audit_log.  An investigator can reconstruct who tried what, ' +
      'when, and from where — long after the fact.'
    );

    await demoGoto(page, BASE_URL, apiKey);
    await dismissContextWarning(page);
    await navigateToPanel(page, 'security');
    await demoPause(page, 1500);

    // The audit log is the last section in the Security panel — load it
    // (Security.loadAuditLog populates the #security-audit-log container)
    // and scroll it into view so the screenshot actually shows it.
    await page.evaluate(() => {
      if (window.Security?.loadAuditLog) window.Security.loadAuditLog();
    });
    await demoPause(page, 1500);
    try {
      await page.locator('#security-audit-log').scrollIntoViewIfNeeded();
      await demoPause(page, 800);
      await highlightElement(page, '#security-audit-log', 'Security audit log');
      await demoPause(page, 1500);
    } catch { /* element may not exist */ }
    await demoScreenshot(narrator, page, OUTPUT_DIR, 'audit-log-trail');
    await clearHighlights(page);
    narrator.narrate('Demo complete — the rail is the difference between an incident and a non-event.', { callout: true });
  });
});
