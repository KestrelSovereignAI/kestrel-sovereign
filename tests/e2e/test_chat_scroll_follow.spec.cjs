/**
 * Chat stick-to-bottom — real-browser regression tests (#2909)
 *
 * The jsdom suite (`tests/frontend/chat_scroll_follow.test.mjs`) covers the
 * controller's logic and its wiring into chat.js. It cannot cover the half
 * of this feature that IS layout: jsdom implements no layout engine, so
 * `scrollHeight` / `clientHeight` are 0, assigning `scrollTop` fires no
 * `scroll` event, and `position: sticky` does nothing. Those tests define
 * a fake scroll box to stand in for all of it.
 *
 * This spec exercises the same controller in Chromium against the real
 * `index.css`, and pins the three things only a browser can answer:
 *
 *   1. The self-healing assumption. `chat_scroll.js` deliberately keeps no
 *      "ignore my own scroll" bookkeeping, on the grounds that a
 *      programmatic scroll-to-bottom fires `scroll` and the handler will
 *      recompute the same answer. But `scroll` is ASYNC. If one fires after
 *      content grew and before the next scroll-to-bottom, the handler reads
 *      a gap over threshold and disengages following — and once off, later
 *      appends stop scrolling and raise the pill. That is a self-inflicted
 *      detach in the middle of a reply, and it is invisible to jsdom.
 *   2. That the sticky pill actually pins inside the scrollport. Its only
 *      sibling is a `display: contents` pane, which contributes no box.
 *   3. That the 48px threshold behaves against real fractional layout.
 *
 * Standalone by design: it serves `kestrel_sovereign/static/` itself on an
 * ephemeral port rather than driving a running console. The controller has
 * no server dependency, so binding this to the :8888 host fixture would buy
 * nothing and inherit its startup flake.
 */
const { test, expect } = require('@playwright/test');
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');

const STATIC_DIR = path.resolve(__dirname, '../../kestrel_sovereign/static');
const HARNESS_PATH = '/__scroll_harness.html';

const MIME = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.mjs': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
};

/**
 * Reproduces the real chat column: `#chat-container` is the scroll box
 * (`overflow-y: auto`, bounded height as a flex child) holding a single
 * `.chat-container-pane`, which is `display: contents`. Both classes come
 * from the real index.css — only the outer shell is synthetic, standing in
 * for the console chrome that gives the container its height.
 */
const HARNESS_HTML = `<!doctype html>
<html><head><meta charset="utf-8">
<link rel="stylesheet" href="./index.css">
<style>
  html, body { height: 100%; margin: 0; }
  #shell { display: flex; flex-direction: column; height: 100vh; }
  #chat-container { flex: 1; overflow-y: auto; }
</style></head>
<body>
<div id="shell">
  <div class="chat-container" id="chat-container">
    <div class="chat-container-pane" id="pane" data-agent="Harness"></div>
  </div>
</div>
<script type="module">
import {
    installScrollFollow, maybeScrollToBottom, forceScrollToBottom,
    getFollowState, isAtBottom, JUMP_TO_LATEST_CLASS,
} from './js/chat_scroll.js';

const container = document.getElementById('chat-container');
const pane = document.getElementById('pane');
installScrollFollow(container);

// Record every follow-state transition, so a disengage that self-corrects
// a frame later is still caught. Sampling the flag at the end would miss it.
const transitions = [];
let last = getFollowState(container).following;
container.addEventListener('scroll', () => {
    const now = getFollowState(container).following;
    if (now !== last) { transitions.push(now); last = now; }
}, { passive: true });

function bubble(text, cls) {
    const d = document.createElement('div');
    d.className = 'message ' + (cls || 'agent-message');
    d.textContent = text;
    pane.appendChild(d);
    return d;
}

window.H = {
    seed(n) {
        for (let i = 0; i < n; i++) bubble('seed ' + i + ' ' + 'x'.repeat(60));
        forceScrollToBottom(container);
        transitions.length = 0;
    },
    append(text) { bubble(text); maybeScrollToBottom(container); },
    token(chunk) {
        if (pane.lastElementChild) pane.lastElementChild.textContent += chunk;
        maybeScrollToBottom(container);
    },
    userAppend(text) { bubble(text, 'user-message'); forceScrollToBottom(container); },
    state: () => getFollowState(container),
    transitions: () => transitions.slice(),
    resetTransitions: () => { transitions.length = 0; },
    atBottom: () => isAtBottom(container),
    metrics: () => ({
        scrollTop: container.scrollTop,
        scrollHeight: container.scrollHeight,
        clientHeight: container.clientHeight,
        gap: container.scrollHeight - container.scrollTop - container.clientHeight,
    }),
    pill() {
        const p = container.querySelector('.' + JUMP_TO_LATEST_CLASS);
        if (!p) return { exists: false };
        const r = p.getBoundingClientRect();
        const c = container.getBoundingClientRect();
        return {
            exists: true,
            hidden: p.hidden,
            visible: r.height > 0 && r.width > 0,
            insideViewport: r.bottom <= c.bottom + 1 && r.top >= c.top - 1,
            text: p.textContent,
            position: getComputedStyle(p).position,
        };
    },
    clickPill() {
        const p = container.querySelector('.' + JUMP_TO_LATEST_CLASS);
        if (p) p.click();
    },
    scrollTo(px) { container.scrollTop = px; },
};
window.HARNESS_READY = true;
</script>
</body></html>`;

let server;
let origin;

test.beforeAll(async () => {
    server = http.createServer((req, res) => {
        const url = new URL(req.url, 'http://127.0.0.1');
        if (url.pathname === HARNESS_PATH) {
            res.writeHead(200, { 'Content-Type': MIME['.html'] });
            res.end(HARNESS_HTML);
            return;
        }
        // Resolve against the real static dir, refusing anything that escapes it.
        const target = path.resolve(STATIC_DIR, '.' + url.pathname);
        if (!target.startsWith(STATIC_DIR) || !fs.existsSync(target) || fs.statSync(target).isDirectory()) {
            res.writeHead(404);
            res.end('not found');
            return;
        }
        res.writeHead(200, { 'Content-Type': MIME[path.extname(target)] || 'application/octet-stream' });
        fs.createReadStream(target).pipe(res);
    });
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    origin = `http://127.0.0.1:${server.address().port}`;
});

test.afterAll(async () => {
    if (server) await new Promise((resolve) => server.close(resolve));
});

/**
 * Scroll the box and wait for the controller to have SEEN it.
 *
 * `scrollTop = x` is synchronous but the follow flag is written by the
 * `scroll` handler, which the browser dispatches later. Reading the state
 * straight after the assignment races that dispatch. Waiting on the
 * observable outcome (rather than a sleep) keeps this deterministic.
 */
async function scrollAndSettle(page, px, wantFollowing) {
    await page.evaluate((p) => window.H.scrollTo(p), px);
    await page.waitForFunction(
        (want) => window.H.state().following === want,
        wantFollowing,
    );
}

/** Let any pending `scroll` dispatches drain before sampling geometry. */
async function settle(page) {
    await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
}

/** Load the harness with a scrollable backlog already in place. */
async function openHarness(page) {
    const errors = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    await page.goto(origin + HARNESS_PATH);
    await page.waitForFunction(() => window.HARNESS_READY === true);
    await page.evaluate(() => window.H.seed(40));
    const m = await page.evaluate(() => window.H.metrics());
    // Guard the harness itself: if the box does not overflow there is no
    // scrolling to observe and every assertion below would pass vacuously.
    expect(m.scrollHeight, 'harness box must actually overflow').toBeGreaterThan(m.clientHeight);
    expect(m.clientHeight).toBeGreaterThan(0);
    return errors;
}

test('the harness can observe a yank (guards against vacuous passes)', async ({ page }) => {
    await openHarness(page);
    await page.evaluate(() => window.H.scrollTo(200));
    // Force is unconditional by contract, so this is the yank the feature
    // suppresses — if the harness cannot see THIS move, it cannot see a bug.
    await page.evaluate(() => window.H.userAppend('forced'));
    const m = await page.evaluate(() => window.H.metrics());
    expect(m.scrollTop).toBeGreaterThan(200);
});

test('rapid streaming at the tail never spuriously disengages following', async ({ page }) => {
    const errors = await openHarness(page);
    await page.evaluate(async () => {
        window.H.resetTransitions();
        window.H.append('streaming reply: ');
        for (let i = 0; i < 400; i++) {
            window.H.token('token' + i + ' lorem ipsum dolor sit amet '.repeat(2));
            if (i % 40 === 0) await new Promise((r) => requestAnimationFrame(r));
        }
    });
    const state = await page.evaluate(() => window.H.state());
    const transitions = await page.evaluate(() => window.H.transitions());
    expect(transitions.filter((t) => t === false), 'no self-inflicted detach mid-stream').toEqual([]);
    expect(state.following).toBe(true);
    const m = await page.evaluate(() => window.H.metrics());
    expect(m.gap).toBeLessThanOrEqual(48);
    expect(errors).toEqual([]);
});

test('scrolled up, the viewport does not move while content streams in', async ({ page }) => {
    await openHarness(page);
    await scrollAndSettle(page, 200, false);
    await page.evaluate(async () => {
        for (let i = 0; i < 120; i++) {
            window.H.token('more streamed text while the reader is scrolled up ');
            if (i % 20 === 0) await new Promise((r) => requestAnimationFrame(r));
        }
    });
    const m = await page.evaluate(() => window.H.metrics());
    expect(Math.abs(m.scrollTop - 200), 'viewport must stay put').toBeLessThan(1);
});

test('the "Jump to latest" pill appears, pins inside the scrollport, and clears on click', async ({ page }) => {
    await openHarness(page);
    await scrollAndSettle(page, 200, false);
    await page.evaluate(() => window.H.append('content arriving below the reader'));

    const pill = await page.evaluate(() => window.H.pill());
    expect(pill.exists).toBe(true);
    expect(pill.hidden).toBe(false);
    expect(pill.visible, 'pill must have a real box').toBe(true);
    expect(pill.position).toBe('sticky');
    // The pane beside it is `display: contents`, so sticky has no sibling box
    // to resolve against — pin it against the scrollport for real.
    expect(pill.insideViewport, 'pill must pin inside the visible scrollport').toBe(true);

    await page.evaluate(() => window.H.clickPill());
    await settle(page);
    const m = await page.evaluate(() => window.H.metrics());
    const state = await page.evaluate(() => window.H.state());
    expect(state.following).toBe(true);
    expect(m.gap).toBeLessThanOrEqual(48);
    expect((await page.evaluate(() => window.H.pill())).hidden).toBe(true);
});

test('dragging back to the bottom re-engages following with no click', async ({ page }) => {
    await openHarness(page);
    await scrollAndSettle(page, 150, false);
    await page.evaluate(() => window.H.append('content while away'));

    await scrollAndSettle(page, 1e9, true);
    const state = await page.evaluate(() => window.H.state());
    expect(state.following, 'returning to the tail re-engages').toBe(true);
    expect(state.unseenTail, 'and acknowledges what arrived').toBe(false);

    const before = (await page.evaluate(() => window.H.metrics())).scrollTop;
    await page.evaluate(() => window.H.append('after re-engaging'));
    await settle(page);
    const m = await page.evaluate(() => window.H.metrics());
    expect(m.scrollTop, 'following resumed, so this append scrolls').toBeGreaterThan(before);
    expect(m.gap).toBeLessThanOrEqual(48);
});

test('a few pixels off the bottom still counts as following (48px threshold)', async ({ page }) => {
    await openHarness(page);
    await page.evaluate(() => {
        const m = window.H.metrics();
        window.H.scrollTo(m.scrollTop - 20);
    });
    expect((await page.evaluate(() => window.H.state())).following).toBe(true);
    expect(await page.evaluate(() => window.H.atBottom())).toBe(true);
});

test('a user-originated append forces to the tail from far up the scrollback', async ({ page }) => {
    await openHarness(page);
    await scrollAndSettle(page, 100, false);
    await page.evaluate(() => window.H.userAppend('my own message'));
    await settle(page);
    const state = await page.evaluate(() => window.H.state());
    const m = await page.evaluate(() => window.H.metrics());
    expect(state.following).toBe(true);
    expect(m.gap).toBeLessThanOrEqual(48);
});
