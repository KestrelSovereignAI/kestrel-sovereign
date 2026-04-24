/**
 * #748 — Security approval popup end-to-end browser test.
 *
 * Verifies the client-side fix: security.js subscribes to the SSE stream
 * via subscribeSSE() (not the never-assigned window.eventSource), so an
 * `approval_request` event pushed through the notifications stream opens
 * the Modal-based approval dialog (DOM modal, not native browser popup)
 * and that clicking a button POSTs the decision to /api/security/approve.
 *
 * Runs against a live server (see KESTREL_URL). The SSE endpoint is
 * intercepted via page.route so we can deliver a synthetic approval event
 * without needing an LLM or a live compute script.
 */

const { test, expect } = require('@playwright/test');

const KESTREL_URL = process.env.KESTREL_URL || 'http://localhost:9088';
const API_KEY = process.env.KESTREL_API_KEY || 'test-key-748';

const APPROVAL_ID = 'test-approval-748';
const APPROVAL_PAYLOAD = {
    id: APPROVAL_ID,
    feature: 'ComputeFeature',
    tool: 'run_script',
    args: { script_id: 's-test-748', purpose: 'verify popup wire' },
    timestamp: '2026-04-24T12:00:00',
};

function sseBody(events) {
    // Concatenate SSE events into a single response body. The browser's
    // EventSource parses each event: / data: block separated by \n\n.
    return events
        .map((e) => `event: ${e.event}\ndata: ${JSON.stringify(e.data)}\n\n`)
        .join('');
}

test.describe('#748 security approval popup', () => {
    test('approval_request SSE event opens the Modal popup and submits decision', async ({ page }) => {
        // Deliver the approval event on the first SSE connect, then close the
        // stream. Subsequent reconnects return a connected-only stream so the
        // UI doesn't re-open the modal from a duplicate event.
        let sseCallCount = 0;
        await page.route('**/agent/notifications/sse**', async (route) => {
            sseCallCount += 1;
            const events =
                sseCallCount === 1
                    ? [
                          { event: 'connected', data: { status: 'connected' } },
                          { event: 'approval_request', data: APPROVAL_PAYLOAD },
                      ]
                    : [{ event: 'connected', data: { status: 'connected' } }];
            await route.fulfill({
                status: 200,
                headers: {
                    'Content-Type': 'text/event-stream',
                    'Cache-Control': 'no-cache',
                    Connection: 'keep-alive',
                },
                body: sseBody(events),
            });
        });

        // Spy on the approve POST so we can assert the payload without
        // hitting the real approval queue (no such pending request exists).
        const approveCalls = [];
        await page.route('**/api/security/approve', async (route, request) => {
            approveCalls.push(JSON.parse(request.postData() || '{}'));
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({ success: true, approved: true, scope: 'session' }),
            });
        });

        await page.goto(`${KESTREL_URL}/?key=${encodeURIComponent(API_KEY)}`);

        // Approval modal should appear. Uses the Modal component — DOM only,
        // not a native dialog — so Playwright can query it directly.
        await expect(page.locator('#modal-overlay')).toBeVisible({ timeout: 10000 });
        await expect(page.locator('.modal-header h3')).toContainText('Permission Required');

        const modalBody = page.locator('.modal-body');
        await expect(modalBody).toContainText('ComputeFeature');
        await expect(modalBody).toContainText('run_script');
        await expect(modalBody).toContainText('s-test-748');

        // All four decision buttons must be present.
        await expect(page.locator('.modal-btn:has-text("Deny")')).toBeVisible();
        await expect(page.locator('.modal-btn:has-text("This Time")')).toBeVisible();
        await expect(page.locator('.modal-btn:has-text("This Session")')).toBeVisible();
        await expect(page.locator('.modal-btn:has-text("Always")')).toBeVisible();

        // Click "This Session" and verify the decision is posted.
        await page.click('.modal-btn:has-text("This Session")');

        await expect.poll(() => approveCalls.length, { timeout: 5000 }).toBeGreaterThan(0);
        expect(approveCalls[0]).toMatchObject({
            approval_id: APPROVAL_ID,
            approved: true,
            scope: 'session',
        });

        // Modal should close after the decision is submitted.
        await expect(page.locator('#modal-overlay')).not.toBeVisible({ timeout: 5000 });
    });

    test('denying from the modal posts approved=false', async ({ page }) => {
        let sseCallCount = 0;
        await page.route('**/agent/notifications/sse**', async (route) => {
            sseCallCount += 1;
            const events =
                sseCallCount === 1
                    ? [
                          { event: 'connected', data: { status: 'connected' } },
                          {
                              event: 'approval_request',
                              data: { ...APPROVAL_PAYLOAD, id: `${APPROVAL_ID}-deny` },
                          },
                      ]
                    : [{ event: 'connected', data: { status: 'connected' } }];
            await route.fulfill({
                status: 200,
                headers: {
                    'Content-Type': 'text/event-stream',
                    'Cache-Control': 'no-cache',
                    Connection: 'keep-alive',
                },
                body: sseBody(events),
            });
        });

        const approveCalls = [];
        await page.route('**/api/security/approve', async (route, request) => {
            approveCalls.push(JSON.parse(request.postData() || '{}'));
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({ success: true, approved: false, scope: 'once' }),
            });
        });

        await page.goto(`${KESTREL_URL}/?key=${encodeURIComponent(API_KEY)}`);

        await expect(page.locator('#modal-overlay')).toBeVisible({ timeout: 10000 });
        await page.click('.modal-btn:has-text("Deny")');

        await expect.poll(() => approveCalls.length, { timeout: 5000 }).toBeGreaterThan(0);
        expect(approveCalls[0]).toMatchObject({
            approval_id: `${APPROVAL_ID}-deny`,
            approved: false,
            scope: 'once',
        });

        await expect(page.locator('#modal-overlay')).not.toBeVisible({ timeout: 5000 });
    });
});
