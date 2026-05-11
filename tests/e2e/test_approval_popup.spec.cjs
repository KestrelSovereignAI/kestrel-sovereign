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

        // All decision buttons must be present, including global Auto.
        await expect(page.locator('.modal-btn:has-text("Deny")')).toBeVisible();
        await expect(page.locator('.modal-btn:has-text("This Time")')).toBeVisible();
        await expect(page.locator('.modal-btn:has-text("This Session")')).toBeVisible();
        await expect(page.locator('.modal-btn:has-text("Always")')).toBeVisible();
        await expect(page.locator('.modal-btn:has-text("Enable Auto Mode")')).toBeVisible();
        await expect(modalBody).toContainText('Auto Mode approves this request');
        expect((await page.locator('.modal-btn').allTextContents()).map((text) => text.trim())).toEqual([
            'Deny',
            'This Time',
            'This Session',
            'Always',
            'Enable Auto Mode',
        ]);

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

    test('Enable Auto turns on global Auto mode and approves the active request once', async ({ page }) => {
        let sseCallCount = 0;
        await page.route('**/agent/notifications/sse**', async (route) => {
            sseCallCount += 1;
            const events =
                sseCallCount === 1
                    ? [
                          { event: 'connected', data: { status: 'connected' } },
                          { event: 'approval_request', data: { ...APPROVAL_PAYLOAD, id: 'auto-1' } },
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
                body: JSON.stringify({ success: true, approved: true, scope: 'once' }),
            });
        });

        const autoModeCalls = [];
        await page.route('**/api/security/auto-mode', async (route, request) => {
            if (request.method() !== 'POST') {
                await route.fulfill({
                    status: 200,
                    contentType: 'application/json',
                    body: JSON.stringify({
                        enabled: false,
                        warning: 'Auto Mode is off.',
                    }),
                });
                return;
            }

            autoModeCalls.push(JSON.parse(request.postData() || '{}'));
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({
                    enabled: true,
                    warning: 'Auto Mode enabled for this session. Constitutional and honesty checks still run first.',
                }),
            });
        });

        await page.goto(`${KESTREL_URL}/?key=${encodeURIComponent(API_KEY)}`);

        await expect(page.locator('#chat-auto-mode-btn')).toContainText('Auto Mode: Off');
        await expect(page.locator('#modal-overlay')).toBeVisible({ timeout: 10000 });
        await page.click('.modal-btn:has-text("Enable Auto Mode")');

        await expect.poll(() => autoModeCalls.length, { timeout: 5000 }).toBe(1);
        expect(autoModeCalls[0]).toMatchObject({ enabled: true });
        await expect(page.locator('#chat-auto-mode-btn')).toContainText('Auto Mode: On');

        await expect.poll(() => approveCalls.length, { timeout: 5000 }).toBe(1);
        expect(approveCalls[0]).toMatchObject({
            approval_id: 'auto-1',
            approved: true,
            scope: 'once',
        });

        await expect(page.locator('#modal-overlay')).not.toBeVisible({ timeout: 5000 });
        await expect(page.locator('.toast-item').filter({ hasText: 'Auto Mode enabled' })).toBeVisible();
        await expect(page.locator('.toast-item').filter({ hasText: 'Approved (once)' })).toHaveCount(0);
        await expect(page.locator('.toast-item').filter({ hasText: 'Failed to submit decision' })).toHaveCount(0);
    });

    test('Enable Auto keeps approval-submit failure toast silent after mode is enabled', async ({ page }) => {
        let sseCallCount = 0;
        await page.route('**/agent/notifications/sse**', async (route) => {
            sseCallCount += 1;
            const events =
                sseCallCount === 1
                    ? [
                          { event: 'connected', data: { status: 'connected' } },
                          { event: 'approval_request', data: { ...APPROVAL_PAYLOAD, id: 'auto-submit-fails' } },
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

        await page.route('**/api/security/approve', async (route) => {
            await route.fulfill({
                status: 500,
                contentType: 'application/json',
                body: JSON.stringify({ detail: 'synthetic approval submit failure' }),
            });
        });

        await page.route('**/api/security/auto-mode', async (route, request) => {
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({
                    enabled: request.method() === 'POST',
                    warning: 'Auto Mode enabled for this session. Constitutional and honesty checks still run first.',
                }),
            });
        });

        await page.goto(`${KESTREL_URL}/?key=${encodeURIComponent(API_KEY)}`);

        await expect(page.locator('#modal-overlay')).toBeVisible({ timeout: 10000 });
        await page.click('.modal-btn:has-text("Enable Auto Mode")');

        await expect(page.locator('#chat-auto-mode-btn')).toContainText('Auto Mode: On');
        await expect(page.locator('#modal-overlay')).not.toBeVisible({ timeout: 5000 });
        await expect(page.locator('.toast-item').filter({ hasText: 'Auto Mode enabled' })).toBeVisible();
        await expect(page.locator('.toast-item').filter({ hasText: 'Failed to submit decision' })).toHaveCount(0);
        await expect(page.locator('.toast-item').filter({ hasText: 'Approval was withdrawn' })).toHaveCount(0);
    });

    test('Auto mode silently approves later approval requests without another popup', async ({ page }) => {
        await page.route('**/agent/notifications/sse**', async (route) => {
            await route.fulfill({
                status: 200,
                headers: {
                    'Content-Type': 'text/event-stream',
                    'Cache-Control': 'no-cache',
                    Connection: 'keep-alive',
                },
                body: sseBody([{ event: 'connected', data: { status: 'connected' } }]),
            });
        });

        const approveCalls = [];
        await page.route('**/api/security/approve', async (route, request) => {
            approveCalls.push(JSON.parse(request.postData() || '{}'));
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({ success: true, approved: true, scope: 'once' }),
            });
        });

        let autoEnabled = false;
        await page.route('**/api/security/auto-mode', async (route, request) => {
            if (request.method() === 'POST') {
                autoEnabled = Boolean(JSON.parse(request.postData() || '{}').enabled);
            }

            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({
                    enabled: autoEnabled,
                    warning: autoEnabled
                        ? 'Auto Mode enabled for this session. Constitutional and honesty checks still run first.'
                        : 'Auto Mode is off.',
                }),
            });
        });

        await page.goto(`${KESTREL_URL}/?key=${encodeURIComponent(API_KEY)}`);
        await page.getByRole('button', { name: 'Chat' }).click();
        await page.click('#chat-auto-mode-btn');
        await expect(page.locator('#chat-auto-mode-btn')).toContainText('Auto Mode: On');

        await page.evaluate((payload) => {
            window.Security.handleApprovalRequest(payload);
        }, { ...APPROVAL_PAYLOAD, id: 'auto-later-1' });

        await expect.poll(() => approveCalls.length, { timeout: 5000 }).toBe(1);
        expect(approveCalls[0]).toMatchObject({
            approval_id: 'auto-later-1',
            approved: true,
            scope: 'once',
        });
        await expect(page.locator('#modal-overlay')).not.toBeVisible({ timeout: 5000 });
        await expect(page.locator('.toast-item').filter({ hasText: 'Permission Required' })).toHaveCount(0);
        await expect(page.locator('.toast-item').filter({ hasText: 'Approved (once)' })).toHaveCount(0);
        await expect(page.locator('.toast-item').filter({ hasText: 'Failed to submit decision' })).toHaveCount(0);
    });

    test('chat header Auto toggle enables mode without a second confirmation popup', async ({ page }) => {
        await page.route('**/agent/notifications/sse**', async (route) => {
            await route.fulfill({
                status: 200,
                headers: {
                    'Content-Type': 'text/event-stream',
                    'Cache-Control': 'no-cache',
                    Connection: 'keep-alive',
                },
                body: sseBody([{ event: 'connected', data: { status: 'connected' } }]),
            });
        });

        const autoModeCalls = [];
        let autoEnabled = false;
        await page.route('**/api/security/auto-mode', async (route, request) => {
            if (request.method() === 'POST') {
                const body = JSON.parse(request.postData() || '{}');
                autoModeCalls.push(body);
                autoEnabled = Boolean(body.enabled);
            }

            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({
                    enabled: autoEnabled,
                    warning: autoEnabled
                        ? 'Auto Mode enabled for this session. Constitutional and honesty checks still run first.'
                        : 'Auto Mode is off.',
                }),
            });
        });

        await page.goto(`${KESTREL_URL}/?key=${encodeURIComponent(API_KEY)}`);
        await page.getByRole('button', { name: 'Chat' }).click();

        await expect(page.locator('#chat-auto-mode-btn')).toContainText('Auto Mode: Off');
        await page.click('#chat-auto-mode-btn');

        await expect.poll(() => autoModeCalls.length, { timeout: 5000 }).toBe(1);
        expect(autoModeCalls[0]).toMatchObject({ enabled: true });
        await expect(page.locator('#chat-auto-mode-btn')).toContainText('Auto Mode: On');
        await expect(page.locator('#modal-overlay')).not.toBeVisible();
    });

    test('multiple approval events are serialized — one modal at a time, no stacking', async ({ page }) => {
        // Deliver THREE approval events on the first SSE connect. The client
        // must queue them and show one modal at a time, not stack overlays.
        let sseCallCount = 0;
        await page.route('**/agent/notifications/sse**', async (route) => {
            sseCallCount += 1;
            const events =
                sseCallCount === 1
                    ? [
                          { event: 'connected', data: { status: 'connected' } },
                          {
                              event: 'approval_request',
                              data: { ...APPROVAL_PAYLOAD, id: 'queue-1', args: { script_id: 'one' } },
                          },
                          {
                              event: 'approval_request',
                              data: { ...APPROVAL_PAYLOAD, id: 'queue-2', args: { script_id: 'two' } },
                          },
                          {
                              event: 'approval_request',
                              data: { ...APPROVAL_PAYLOAD, id: 'queue-3', args: { script_id: 'three' } },
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
                body: JSON.stringify({ success: true, approved: true, scope: 'session' }),
            });
        });

        await page.goto(`${KESTREL_URL}/?key=${encodeURIComponent(API_KEY)}`);

        // Wait for the first modal to appear.
        await expect(page.locator('#modal-overlay')).toBeVisible({ timeout: 10000 });

        // Only ONE overlay must be in the DOM at any given moment — stacking
        // two overlays is exactly the bug this test guards against.
        await expect(page.locator('#modal-overlay')).toHaveCount(1);
        await expect(page.locator('.modal-body')).toContainText('"script_id": "one"');

        // Approve the first → queue should advance to the second.
        await page.click('.modal-btn:has-text("This Session")');
        await expect(page.locator('.modal-body')).toContainText('"script_id": "two"', { timeout: 5000 });
        await expect(page.locator('#modal-overlay')).toHaveCount(1);

        await page.click('.modal-btn:has-text("This Session")');
        await expect(page.locator('.modal-body')).toContainText('"script_id": "three"', { timeout: 5000 });
        await expect(page.locator('#modal-overlay')).toHaveCount(1);

        await page.click('.modal-btn:has-text("Deny")');
        await expect(page.locator('#modal-overlay')).not.toBeVisible({ timeout: 5000 });

        // All three decisions should have been posted in order.
        await expect.poll(() => approveCalls.length, { timeout: 5000 }).toBe(3);
        expect(approveCalls.map((c) => c.approval_id)).toEqual(['queue-1', 'queue-2', 'queue-3']);
        expect(approveCalls.map((c) => c.approved)).toEqual([true, true, false]);
    });

    test('duplicate approval_request events with the same id do not prompt twice', async ({ page }) => {
        let sseCallCount = 0;
        await page.route('**/agent/notifications/sse**', async (route) => {
            sseCallCount += 1;
            const events =
                sseCallCount === 1
                    ? [
                          { event: 'connected', data: { status: 'connected' } },
                          { event: 'approval_request', data: { ...APPROVAL_PAYLOAD, id: 'dup-1' } },
                          { event: 'approval_request', data: { ...APPROVAL_PAYLOAD, id: 'dup-1' } },
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
                body: JSON.stringify({ success: true, approved: true, scope: 'once' }),
            });
        });

        await page.goto(`${KESTREL_URL}/?key=${encodeURIComponent(API_KEY)}`);

        await expect(page.locator('#modal-overlay')).toBeVisible({ timeout: 10000 });
        await expect(page.locator('#modal-overlay')).toHaveCount(1);

        await page.click('.modal-btn:has-text("This Time")');
        await expect(page.locator('#modal-overlay')).not.toBeVisible({ timeout: 5000 });

        // Give the deduper a moment to discard any redelivered event.
        await page.waitForTimeout(500);
        await expect(page.locator('#modal-overlay')).not.toBeVisible();

        expect(approveCalls.length).toBe(1);
        expect(approveCalls[0].approval_id).toBe('dup-1');
    });

    test('expired approval (server returns 404) shows a friendly toast, not a raw error', async ({ page }) => {
        let sseCallCount = 0;
        await page.route('**/agent/notifications/sse**', async (route) => {
            sseCallCount += 1;
            const events =
                sseCallCount === 1
                    ? [
                          { event: 'connected', data: { status: 'connected' } },
                          {
                              event: 'approval_request',
                              data: { ...APPROVAL_PAYLOAD, id: 'expired-1' },
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

        // Simulate the server-side request_approval having already timed out.
        await page.route('**/api/security/approve', async (route) => {
            await route.fulfill({
                status: 404,
                contentType: 'application/json',
                body: JSON.stringify({ detail: "Request 'expired-1' not found or expired" }),
            });
        });

        await page.goto(`${KESTREL_URL}/?key=${encodeURIComponent(API_KEY)}`);

        await expect(page.locator('#modal-overlay')).toBeVisible({ timeout: 10000 });
        await page.click('.modal-btn:has-text("This Session")');

        // User should see a clear warning toast — not a raw error stack.
        const warning = page.locator('.toast-item').filter({ hasText: 'withdrawn' });
        await expect(warning).toBeVisible({ timeout: 5000 });

        // Modal should still close.
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
