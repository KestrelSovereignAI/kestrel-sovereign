// #2909: the chat viewport must stick to the bottom ONLY while the reader is
// at the bottom. It used to force `scrollTop = scrollHeight` on all 13 append
// sites unconditionally, so scrolling up to re-read something during a
// streaming turn yanked you back to the tail on the very next token.
//
// The follow decision now has one owner — `chat_scroll.js` — and `chat.js`
// routes every append through it (`maybeScrollToBottom` for agent/system
// content, `forceScrollToBottom` for user-originated actions). These tests
// drive the REAL chat.js entry points against a fake scroll box, so they pin
// the wiring, not just the controller.
//
// Harness note: jsdom implements no layout. `scrollHeight` / `clientHeight`
// are always 0, and assigning `scrollTop` neither sticks nor fires `scroll`.
// `makeScrollBox` below installs real, settable properties that clamp like a
// browser's scrolling box and emit `scroll` when the position actually
// changes — which is exactly the mechanism the controller's self-healing
// follow flag depends on.

import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';

const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.Node = dom.window.Node;
globalThis.Element = dom.window.Element;
globalThis.HTMLElement = dom.window.HTMLElement;
globalThis.Event = dom.window.Event;
globalThis.location = dom.window.location;
globalThis.CSS = { escape: (s) => String(s).replace(/[^a-zA-Z0-9_-]/g, '\\$&') };
globalThis.sessionStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.fetch = async () => ({ ok: false, status: 500, json: async () => ({}) });
globalThis.kicon = () => '';
globalThis.window.kicon = globalThis.kicon;
globalThis.window.SharedMarkdown = {
    renderMarkdown: (s) => String(s),
    renderStreamingMarkdown: (s) => String(s),
    highlightCodeBlocks: () => {},
    renderMermaidDiagrams: () => {},
    finalizeMarkdown: async (el, content) => { el.textContent = content; },
};

for (const id of ['chat-container', 'thinking-indicator']) {
    const el = document.createElement('div');
    el.id = id;
    document.body.appendChild(el);
}
for (const id of ['message-input', 'send-button']) {
    const el = document.createElement(id === 'send-button' ? 'button' : 'textarea');
    el.id = id;
    document.body.appendChild(el);
}

const apiModule = await import('../../kestrel_sovereign/static/js/api.js');
const API = apiModule.default;
API.hasCapability = () => true;

const { state, getOrCreateChatPane } = await import('../../kestrel_sovereign/static/js/ui.js');
const chatModule = await import('../../kestrel_sovereign/static/js/chat.js');
const {
    initChat,
    setChatDeps,
    mountChatPane,
    wipeAgentChatPane,
    addMessage,
    addMessageStreaming,
    updateStreamingMessage,
    handleRestartStatus,
    connectNotifications,
} = chatModule;
const {
    forceScrollToBottom,
    installScrollFollow,
    isFollowing,
    setFollowState,
    STICK_TO_BOTTOM_THRESHOLD_PX,
    JUMP_TO_LATEST_CLASS,
} = await import('../../kestrel_sovereign/static/js/chat_scroll.js');

const container = document.getElementById('chat-container');

/**
 * Give `el` a real scrolling box. jsdom has none: it reports
 * scrollHeight/clientHeight as 0 and silently drops scrollTop writes, so
 * without this every assertion below would pass vacuously (0 === 0 forever).
 *
 * Models the two browser behaviours the controller actually leans on:
 *   * scrollTop is CLAMPED to [0, scrollHeight - clientHeight];
 *   * a `scroll` event fires only when the position really changes — which is
 *     what lets a programmatic scroll-to-bottom re-arm the follow flag.
 */
function makeScrollBox(el, { clientHeight = 400, scrollHeight = 400 } = {}) {
    let _scrollTop = 0;
    let _scrollHeight = scrollHeight;
    Object.defineProperty(el, 'clientHeight', { configurable: true, get: () => clientHeight });
    Object.defineProperty(el, 'scrollHeight', { configurable: true, get: () => _scrollHeight });
    Object.defineProperty(el, 'scrollTop', {
        configurable: true,
        get: () => _scrollTop,
        set(v) {
            const max = Math.max(0, _scrollHeight - clientHeight);
            const next = Math.max(0, Math.min(max, Number(v) || 0));
            if (next === _scrollTop) return;
            _scrollTop = next;
            el.dispatchEvent(new dom.window.Event('scroll'));
        },
    });
    return {
        /** Content appended: the box gets taller, the viewport does not move. */
        grow(px = 500) { _scrollHeight += px; return this; },
        get maxScroll() { return Math.max(0, _scrollHeight - clientHeight); },
        /** A user-driven scroll — same path a real wheel/drag takes. */
        userScrollTo(px) { el.scrollTop = px; },
        /** Reset to a fresh, non-overflowing box pinned at the top. */
        reset() { _scrollHeight = clientHeight; _scrollTop = 0; return this; },
    };
}

const box = makeScrollBox(container);

initChat();

/** The pill is built lazily, so "absent" and "hidden" are both "not shown". */
function pillShown() {
    const pill = container.querySelector('.' + JUMP_TO_LATEST_CLASS);
    return !!pill && pill.hidden === false;
}

/**
 * Put the harness in a known state: one mounted pane, box at the bottom,
 * following engaged. Every test starts here so they can run in any order.
 */
function freshPane(agentName) {
    API.setHostAgent(agentName);
    const pane = getOrCreateChatPane(agentName);
    mountChatPane(agentName);
    wipeAgentChatPane(agentName);
    box.reset();
    forceScrollToBottom(container);
    return pane;
}

/** Simulate an agent appending content: the box grows, then chat.js scrolls. */
function agentAppend(pane, msgDiv, text) {
    box.grow(200);
    updateStreamingMessage(msgDiv, text, pane.element);
}

test('harness: the fake scroll box can actually observe a yank', () => {
    // Guards against every assertion below passing vacuously. If scrollTop
    // could never move, "it did not move" would prove nothing at all.
    const pane = freshPane('harness-agent');
    const msgDiv = addMessageStreaming('agent', pane.element);
    box.grow(1000);
    box.userScrollTo(0);
    assert.equal(container.scrollTop, 0);
    forceScrollToBottom(container);
    assert.ok(container.scrollTop > 0, 'the box must be able to move; otherwise test 2 is vacuous');
    assert.equal(container.scrollTop, box.maxScroll);
    assert.ok(msgDiv);
});

test('1. at the bottom, an agent append keeps scrolling the viewport', () => {
    const pane = freshPane('follow-agent');
    const msgDiv = addMessageStreaming('agent', pane.element);
    assert.equal(isFollowing(container), true);

    agentAppend(pane, msgDiv, 'first chunk');
    assert.equal(container.scrollTop, box.maxScroll, 'append at the tail must follow the tail');
    assert.ok(container.scrollTop > 0);

    const afterFirst = container.scrollTop;
    agentAppend(pane, msgDiv, 'first chunk + second chunk');
    assert.ok(container.scrollTop > afterFirst, 'each append keeps advancing while following');
    assert.equal(container.scrollTop, box.maxScroll);
});

test('2. scrolled up, successive streaming updates do NOT move the viewport', () => {
    const pane = freshPane('stream-agent');
    const msgDiv = addMessageStreaming('agent', pane.element);
    box.grow(2000);

    // The reader scrolls up to re-read something mid-turn.
    box.userScrollTo(150);
    assert.equal(container.scrollTop, 150);
    assert.equal(isFollowing(container), false, 'scrolling up must disengage following');

    // The turn keeps streaming. This is the regression that #2909 is about:
    // before the fix, EVERY one of these yanked the reader to the bottom.
    let text = '';
    for (let i = 0; i < 5; i++) {
        text += `chunk-${i} `;
        agentAppend(pane, msgDiv, text);
        assert.equal(container.scrollTop, 150, `streaming update ${i} must not move the viewport`);
    }
    assert.equal(isFollowing(container), false);

    // ...and the box was genuinely capable of moving the whole time.
    forceScrollToBottom(container);
    assert.equal(container.scrollTop, box.maxScroll);
});

test('3. scrolling back to the bottom re-engages following, with no click', () => {
    const pane = freshPane('reengage-agent');
    const msgDiv = addMessageStreaming('agent', pane.element);
    box.grow(2000);

    box.userScrollTo(120);
    assert.equal(isFollowing(container), false);
    agentAppend(pane, msgDiv, 'while away');
    assert.equal(container.scrollTop, 120, 'still detached');

    // The reader scrolls back down to the tail themselves.
    box.userScrollTo(box.maxScroll);
    assert.equal(isFollowing(container), true, 'returning to the tail must re-engage');

    // The NEXT append follows again.
    agentAppend(pane, msgDiv, 'while away + back at the tail');
    assert.equal(container.scrollTop, box.maxScroll, 'append after re-engaging must follow');
});

test('4. scrolled up, sending a message snaps to the bottom and re-engages', async () => {
    const pane = freshPane('send-agent');
    box.grow(2000);
    box.userScrollTo(90);
    assert.equal(isFollowing(container), false);

    // A user bubble is the user's own send: never leave them in scrollback.
    await addMessage('user', 'hello from the user', pane.element);
    assert.equal(container.scrollTop, box.maxScroll, 'a user send must snap to the bottom');
    assert.equal(isFollowing(container), true, 'a user send must re-engage following');

    // An AGENT bubble in the same position must NOT force.
    box.userScrollTo(90);
    assert.equal(isFollowing(container), false);
    await addMessage('agent', 'agent reply', pane.element);
    assert.equal(container.scrollTop, 90, 'an agent bubble must respect the flag');
});

test('5. a few pixels off the bottom still counts as following', () => {
    // The slack must be real, not exact equality: device-pixel rounding,
    // fractional line heights and reflow while a bubble streams all leave a
    // few pixels between `scrollTop + clientHeight` and `scrollHeight`.
    assert.ok(
        STICK_TO_BOTTOM_THRESHOLD_PX >= 32 && STICK_TO_BOTTOM_THRESHOLD_PX <= 64,
        `threshold must stay in the documented 32-64px range, got ${STICK_TO_BOTTOM_THRESHOLD_PX}`,
    );

    const pane = freshPane('threshold-agent');
    const msgDiv = addMessageStreaming('agent', pane.element);
    box.grow(2000);

    // Literal gaps, deliberately NOT derived from the constant: deriving them
    // would make this test pass for ANY threshold, including 0, which is the
    // exact thing it exists to rule out.
    box.userScrollTo(box.maxScroll - 24);
    assert.equal(isFollowing(container), true, '24px off the bottom must still follow');
    agentAppend(pane, msgDiv, 'near the tail');
    assert.equal(container.scrollTop, box.maxScroll, 'a within-threshold append must follow');

    // Well beyond it: a deliberate scroll away, which must detach.
    box.userScrollTo(box.maxScroll - 400);
    assert.equal(isFollowing(container), false, '400px off the bottom must detach');
    const parked = container.scrollTop;
    agentAppend(pane, msgDiv, 'near the tail + more');
    assert.equal(container.scrollTop, parked, 'a beyond-threshold append must not follow');
});

test('6. follow state survives a pane detach / remount round-trip', () => {
    const paneA = freshPane('roundtrip-A');
    const msgA = addMessageStreaming('agent', paneA.element);
    box.grow(2000);

    // Reader scrolls up in A's conversation.
    box.userScrollTo(200);
    assert.equal(isFollowing(container), false);

    // Switch to B — B is a fresh pane, so B follows its own tail.
    freshPane('roundtrip-B');
    assert.equal(isFollowing(container), true, "B's own conversation follows normally");

    // ...and back to A. The saved followLive must come back with scrollPos,
    // or returning to a scrolled-up conversation silently re-engages.
    API.setHostAgent('roundtrip-A');
    mountChatPane('roundtrip-A');
    assert.equal(paneA.followLive, false, 'followLive must persist on the pane object');
    assert.equal(isFollowing(container), false, 'a scrolled-up pane comes back scrolled up');

    const parked = container.scrollTop;
    agentAppend(paneA, msgA, 'content after remount');
    assert.equal(container.scrollTop, parked, 'an append after remount must not yank');
});

test('6b. an unacknowledged "Jump to latest" survives the same round-trip', () => {
    // Follow state is TWO facts, and both belong to the conversation: whether
    // the reader is on the tail, and whether anything landed below them that
    // they have not seen. Persisting only the first means A can announce
    // "content arrived", the reader visits B to answer something, and comes
    // back to A detached with the announcement silently retracted — the
    // content is still unread, but nothing says so.
    const paneA = freshPane('pill-roundtrip-A');
    const msgA = addMessageStreaming('agent', paneA.element);
    box.grow(2000);

    box.userScrollTo(180);
    agentAppend(paneA, msgA, 'arrived while the reader was scrolled up');
    assert.equal(pillShown(), true, 'A announced unread content before the switch');

    // B is a fresh conversation: its own tail, nothing unread. Critically,
    // A's pending announcement must not leak into B either.
    freshPane('pill-roundtrip-B');
    assert.equal(pillShown(), false, "A's unread tail is not B's news");

    // Back to A, whose long conversation fills the box again.
    box.grow(2000);
    API.setHostAgent('pill-roundtrip-A');
    mountChatPane('pill-roundtrip-A');

    assert.equal(paneA.unseenTail, true, 'the unread tail must persist on the pane');
    assert.equal(isFollowing(container), false, 'still detached from the tail');
    assert.equal(pillShown(), true, 'the unread tail is still unread on return');

    // And clicking it still works after the remount.
    container.querySelector('.' + JUMP_TO_LATEST_CLASS)
        .dispatchEvent(new dom.window.MouseEvent('click', { bubbles: true }));
    assert.equal(container.scrollTop, box.maxScroll);
    assert.equal(pillShown(), false);
});

test('7. wipeAgentChatPane re-engages following (a cleared pane is fresh)', () => {
    const pane = freshPane('wipe-agent');
    const msgDiv = addMessageStreaming('agent', pane.element);
    box.grow(2000);
    box.userScrollTo(175);
    assert.equal(isFollowing(container), false);
    agentAppend(pane, msgDiv, 'unread content behind the reader');
    assert.equal(pillShown(), true);

    wipeAgentChatPane('wipe-agent');
    assert.equal(pane.followLive, true, 'a wiped pane records that it follows');
    assert.equal(isFollowing(container), true, 'a wiped pane follows its tail');
    assert.equal(pillShown(), false,
        'the wipe removed the very content the pill pointed at');

    // The wipe emptied the pane; a fresh conversation appends at the tail.
    const fresh = addMessageStreaming('agent', pane.element);
    agentAppend(pane, fresh, 'new conversation');
    assert.equal(container.scrollTop, box.maxScroll, 'a wiped pane follows new content');
    assert.ok(msgDiv);
});

test('7b. wiping a DETACHED pane clears its saved unread tail too', () => {
    // A pane's two follow fields must stay coherent even when the wipe cannot
    // reach the controller (clear chat / conversation switch / delete on an
    // agent the reader is not currently looking at). Restoring a pane whose
    // saved halves disagree is how a stale announcement comes back from the
    // dead — the same defect this issue is about, one layer down.
    const paneA = freshPane('wipe-detached-A');
    const msgA = addMessageStreaming('agent', paneA.element);
    box.grow(2000);
    box.userScrollTo(150);
    agentAppend(paneA, msgA, 'unread content');
    assert.equal(pillShown(), true);

    freshPane('wipe-detached-B');
    assert.equal(paneA.followLive, false, 'A detached scrolled up...');
    assert.equal(paneA.unseenTail, true, '...with content it had announced');

    wipeAgentChatPane('wipe-detached-A');
    assert.equal(paneA.followLive, true, 'a wiped pane follows again');
    assert.equal(paneA.unseenTail, false, 'and has nothing left to announce');

    // And returning to it lands at the tail, silently.
    box.grow(2000);
    API.setHostAgent('wipe-detached-A');
    mountChatPane('wipe-detached-A');
    assert.equal(isFollowing(container), true);
    assert.equal(pillShown(), false);
});

test('8. the "Jump to latest" pill appears only when detached AND content arrived', () => {
    const pane = freshPane('pill-agent');
    const msgDiv = addMessageStreaming('agent', pane.element);
    box.grow(2000);

    assert.equal(pillShown(), false, 'no pill while following');

    // Detached in an IDLE conversation: nothing has arrived, so there is
    // nothing to announce and no pill.
    box.userScrollTo(100);
    assert.equal(isFollowing(container), false);
    assert.equal(pillShown(), false, 'no pill merely because the reader scrolled up');

    // Content arrives while detached — now the reader needs to be told.
    agentAppend(pane, msgDiv, 'something arrived');
    assert.equal(pillShown(), true, 'pill appears when content lands off-tail');
    assert.equal(container.scrollTop, 100, 'showing the pill must not move the viewport');

    // Clicking it jumps to the tail, re-engages, and dismisses itself.
    const pill = container.querySelector('.' + JUMP_TO_LATEST_CLASS);
    pill.dispatchEvent(new dom.window.MouseEvent('click', { bubbles: true }));
    assert.equal(container.scrollTop, box.maxScroll, 'the pill jumps to the tail');
    assert.equal(isFollowing(container), true, 'the pill re-engages following');
    assert.equal(pillShown(), false, 'the pill hides once the reader is back at the tail');

    // Scrolling back to the tail by hand also dismisses it.
    box.userScrollTo(100);
    agentAppend(pane, msgDiv, 'something arrived again');
    assert.equal(pillShown(), true);
    box.userScrollTo(box.maxScroll);
    assert.equal(pillShown(), false, 'returning to the tail dismisses the pill');
});

test('9. forcing where the viewport cannot move still re-engages following', () => {
    // The follow flag mostly self-heals: a programmatic scroll fires `scroll`
    // and the listener recomputes. But only when the position actually
    // CHANGES — no movement, no event. A short, non-overflowing conversation
    // restored with following disengaged is exactly that case, so
    // `forceScrollToBottom` has to write the flag itself rather than trust
    // the echo. (In a real browser it is weaker still: `scroll` is delivered
    // asynchronously, so the flag is stale for the rest of the current task.
    // This harness dispatches synchronously, which gives the self-heal every
    // chance to mask the bug — and it is still caught here.)
    const solo = document.createElement('div');
    const soloBox = makeScrollBox(solo, { clientHeight: 300, scrollHeight: 300 });
    installScrollFollow(solo);

    // What a pane restore does for a scrolled-up pane.
    setFollowState(solo, { following: false });
    assert.equal(isFollowing(solo), false);
    assert.equal(soloBox.maxScroll, 0, 'nothing to scroll: the content fits the box');

    forceScrollToBottom(solo);
    assert.equal(solo.scrollTop, 0, 'no movement was possible, so no scroll event fired');
    assert.equal(isFollowing(solo), true, 'force must write the flag, not rely on the echo');
});

test('10. a restart status painted outside the viewport does not move it', () => {
    // `handleRestartStatus` has two targets that are NOT the visible pane: an
    // explicit `targetEl` (the #1816 history repaint) and the notification
    // agent's pane, which is detached whenever the reader is looking at
    // another agent. Either one telling the controller would scroll the
    // visible conversation — or raise "Jump to latest" over content that
    // isn't in it.
    const visible = freshPane('restart-visible');
    const msgDiv = addMessageStreaming('agent', visible.element);
    box.grow(2000);
    box.userScrollTo(160);
    assert.equal(isFollowing(container), false);
    assert.equal(pillShown(), false);

    const detached = document.createElement('div');
    handleRestartStatus({ request_id: 'req-detached', status: 'pending' }, detached);
    assert.equal(detached.querySelectorAll('.restart-status-message').length, 1,
        'the bubble must still land in the target it was given');
    assert.equal(container.scrollTop, 160, 'content outside the viewport must not move it');
    assert.equal(pillShown(), false,
        'nor announce itself as this conversation\'s unread content');

    // The guard must DISCRIMINATE, not just disable: the same call against the
    // pane the reader is actually looking at still registers as new content.
    handleRestartStatus({ request_id: 'req-visible', status: 'pending' }, visible.element);
    assert.equal(visible.element.querySelectorAll('.restart-status-message').length, 1);
    assert.equal(container.scrollTop, 160, 'still no yank — the reader is scrolled up');
    assert.equal(pillShown(), true, 'the visible pane announces its own new content');
    assert.ok(msgDiv);
});

test('11. a backgrounded agent\'s task notification does not touch the viewport', () => {
    // The notification SSE stream is pinned to the agent it was opened for.
    // Switch agents mid-turn and its task notifications keep landing in that
    // agent's now-detached pane — which must have no effect on the pane the
    // reader is watching.
    setChatDeps({ toast: { show: () => {} } });   // the toast is not under test
    const handlers = new Map();
    class FakeES {
        constructor(_url) { FakeES.last = this; }
        addEventListener(name, h) {
            const list = handlers.get(name) || [];
            list.push(h);
            handlers.set(name, list);
        }
        close() {}
    }
    const origES = globalThis.EventSource;
    globalThis.EventSource = FakeES;

    try {
        // Open the stream for the background agent, then walk away to another.
        const background = freshPane('notify-background');
        connectNotifications();
        const notify = handlers.get('task_notification') || [];
        assert.ok(notify.length >= 1, 'connectNotifications must register the listener');
        const fire = (message) => {
            for (const h of notify) h({ data: JSON.stringify({ message, type: 'completed' }) });
        };

        const visible = freshPane('notify-visible');
        box.grow(2000);
        box.userScrollTo(140);
        assert.equal(isFollowing(container), false);
        assert.equal(pillShown(), false);

        fire('background task finished');
        assert.equal(
            background.element.querySelectorAll('.notification-message').length, 1,
            'the notification still paints into its own agent\'s pane');
        assert.equal(visible.element.querySelectorAll('.notification-message').length, 0);
        assert.equal(container.scrollTop, 140,
            'another agent\'s notification must not move this conversation');
        assert.equal(pillShown(), false,
            'nor claim this conversation has unread content');

        // Discriminating half: mounted pane, same code path, and it registers.
        API.setHostAgent('notify-background');
        mountChatPane('notify-background');
        box.grow(2000);
        box.userScrollTo(120);
        assert.equal(isFollowing(container), false);
        fire('second background task finished');
        assert.equal(container.scrollTop, 120, 'still no yank while scrolled up');
        assert.equal(pillShown(), true,
            'a notification in the MOUNTED pane is this conversation\'s new content');
    } finally {
        chatModule.disconnectNotifications();
        globalThis.EventSource = origES;
        setChatDeps({ toast: null });
    }
});

test('the pill is styled in index.css, never inline', async () => {
    const { readFileSync } = await import('node:fs');
    const css = readFileSync(
        new URL('../../kestrel_sovereign/static/index.css', import.meta.url),
        'utf8',
    );
    assert.ok(css.includes('.' + JUMP_TO_LATEST_CLASS),
        'the pill must have a rule in index.css');
    // `hidden` has to beat `display: flex`, or the pill is permanently visible.
    assert.ok(css.includes('.' + JUMP_TO_LATEST_CLASS + '[hidden]'),
        'the [hidden] rule must override the display rule');

    const src = readFileSync(
        new URL('../../kestrel_sovereign/static/js/chat_scroll.js', import.meta.url),
        'utf8',
    );
    assert.equal((src.match(/\.style\./g) || []).length, 0,
        'the controller must not carry inline styles');
});

test('chat.js owns no bare tail-follow assignment (one owner, not 13 conditionals)', async () => {
    const { readFileSync } = await import('node:fs');
    const src = readFileSync(
        new URL('../../kestrel_sovereign/static/js/chat.js', import.meta.url),
        'utf8',
    );
    const bare = src.match(/scrollTop\s*=\s*[A-Za-z_$][A-Za-z0-9_$]*\.scrollHeight/g) || [];
    assert.deepEqual(bare, [],
        'every tail-follow in chat.js must route through chat_scroll.js');
    assert.ok(src.includes("from './chat_scroll.js'"),
        'chat.js must import the controller');
});

test('state is per scroll box and defaults to following for an unknown container', () => {
    // Embedders mount their own container; a box nobody has scrolled yet must
    // follow, otherwise a fresh console would never track a stream at all.
    const other = document.createElement('div');
    assert.equal(isFollowing(other), true);
    assert.equal(isFollowing(null), true, 'a missing container must not throw');
});
