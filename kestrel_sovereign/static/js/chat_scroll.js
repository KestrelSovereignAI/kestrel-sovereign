/**
 * Kestrel Sovereign Console - Chat stick-to-bottom controller (#2909)
 *
 * The chat viewport used to force `scrollTop = scrollHeight` on every
 * append, so scrolling up to re-read something during a streaming turn
 * yanked the reader back to the tail on the next token. This module is
 * the single owner of that decision: `chat.js` no longer assigns
 * `scrollTop` for tail-following at all, it routes every append through
 * `maybeScrollToBottom` (agent/system originated) or
 * `forceScrollToBottom` (user originated).
 *
 * Follow state is per scroll box and derived from geometry: a `scroll`
 * listener recomputes "am I at the bottom?" on every event and stores
 * it. Because a programmatic scroll-to-bottom also fires `scroll`, the
 * flag self-heals — there is no "ignore my own scroll" bookkeeping. The
 * two writers outside that listener are deliberate and explicit:
 * `forceScrollToBottom` (user action re-engages) and `setFollowing`
 * (a remounted pane restoring its saved `followLive`).
 */

// Slack for "at the bottom". Exact equality is not usable: device-pixel
// rounding, fractional line heights, and reflow while a bubble streams
// all leave a few pixels between `scrollTop + clientHeight` and
// `scrollHeight`.
export const STICK_TO_BOTTOM_THRESHOLD_PX = 48;

/** Class of the "Jump to latest" affordance; styled in index.css. */
export const JUMP_TO_LATEST_CLASS = 'chat-jump-to-latest';

// Live per-container state. WeakMap so a torn-down container (embedded
// console remount) doesn't leak.
const _boxes = new WeakMap();

function stateFor(container) {
    let s = _boxes.get(container);
    if (!s) {
        s = { following: true, pendingContent: false, pill: null, installed: false };
        _boxes.set(container, s);
    }
    return s;
}

/**
 * Geometry predicate: is this scroll box at (or within the threshold of)
 * its bottom? A box that cannot scroll (no overflow) is always "at the
 * bottom", which is what keeps a short conversation following.
 */
export function isAtBottom(container) {
    if (!container) return true;
    const gap = container.scrollHeight - container.scrollTop - container.clientHeight;
    return gap <= STICK_TO_BOTTOM_THRESHOLD_PX;
}

/** Is tail-following currently engaged for this scroll box? */
export function isFollowing(container) {
    if (!container) return true;
    return stateFor(container).following !== false;
}

/**
 * Seed the follow flag without moving the viewport. Only the pane
 * mount/restore path uses this — everything else goes through the
 * scroll listener or forceScrollToBottom.
 */
export function setFollowing(container, following) {
    if (!container) return;
    const s = stateFor(container);
    s.following = !!following;
    if (s.following) s.pendingContent = false;
    renderJumpToLatest(container);
}

/**
 * Install the `scroll` listener that owns the follow flag. Idempotent —
 * safe to call from every wiring point that can resolve the container.
 */
export function installScrollFollow(container) {
    if (!container || typeof container.addEventListener !== 'function') return;
    const s = stateFor(container);
    if (s.installed) return;
    s.installed = true;
    container.addEventListener('scroll', () => {
        const atBottom = isAtBottom(container);
        s.following = atBottom;
        // Coming back to the tail acknowledges whatever arrived while
        // the reader was away, so the pill has nothing left to announce.
        if (atBottom) s.pendingContent = false;
        renderJumpToLatest(container);
    });
}

/**
 * Agent- or system-originated append: scroll only while following is
 * engaged. When it isn't, record that content arrived so the "Jump to
 * latest" pill can say so. Returns whether it scrolled.
 */
export function maybeScrollToBottom(container) {
    if (!container) return false;
    const s = stateFor(container);
    if (s.following === false) {
        s.pendingContent = true;
        renderJumpToLatest(container);
        return false;
    }
    container.scrollTop = container.scrollHeight;
    renderJumpToLatest(container);
    return true;
}

/**
 * User-originated action (send, queued follow-up, pill click, freshly
 * loaded history): re-engage following and snap to the tail.
 */
export function forceScrollToBottom(container) {
    if (!container) return;
    const s = stateFor(container);
    s.following = true;
    s.pendingContent = false;
    container.scrollTop = container.scrollHeight;
    renderJumpToLatest(container);
}

/**
 * Reset a scroll box to a fresh conversation's state: following, with
 * nothing pending. Used by the pane wipe path.
 */
export function resetScrollFollow(container) {
    if (!container) return;
    const s = stateFor(container);
    s.following = true;
    s.pendingContent = false;
    renderJumpToLatest(container);
}

/** The pill element for this container, or null if none has been built. */
export function getJumpToLatestPill(container) {
    if (!container) return null;
    return stateFor(container).pill;
}

/**
 * Build (or re-attach) the "Jump to latest" pill and set its visibility.
 *
 * The pill is a direct child of the scroll box and `position: sticky`
 * in CSS, so it pins to the bottom of the scrollport without assuming
 * anything about the container's ancestors (embedders supply their own
 * chat chrome). `mountChatPane` empties the container on every agent
 * switch, hence the re-attach.
 *
 * It shows ONLY when following is disengaged AND content has appended
 * since — drifting off the tail in an idle conversation is not news.
 */
function renderJumpToLatest(container) {
    const s = stateFor(container);
    if (typeof document === 'undefined' || !container.appendChild) return;
    if (!s.pill) {
        const pill = document.createElement('button');
        pill.type = 'button';
        pill.className = JUMP_TO_LATEST_CLASS;
        pill.textContent = 'Jump to latest';
        pill.setAttribute('aria-label', 'Jump to the latest message');
        pill.addEventListener('click', () => forceScrollToBottom(container));
        s.pill = pill;
    }
    if (s.pill.parentNode !== container) container.appendChild(s.pill);
    s.pill.hidden = !(s.following === false && s.pendingContent);
}
