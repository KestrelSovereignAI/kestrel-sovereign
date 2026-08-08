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
 * `forceScrollToBottom` (a user action re-engages) and `setFollowState`
 * (a pane re-seeding state on mount or wipe).
 *
 * The state is keyed on the scroll BOX because that is what the live
 * listener can observe, but it describes a CONVERSATION — so both
 * fields (`following` and `unseenTail`) travel with the pane across an
 * agent switch, exactly like `scrollPos` does. `getFollowState` /
 * `setFollowState` are that seam; see `mountChatPane` in chat.js.
 *
 * Which means the box is only the authority while the conversation is
 * mounted. Panes keep streaming after they are detached, so the same
 * two decisions exist with no viewport to act on: `notePaneAppend` and
 * `notePaneUserAction` apply them to the pane record directly. Routing a
 * detached write through the container instead would answer for the
 * wrong conversation — scrolling, or announcing unread content, in
 * whichever pane the reader happens to be watching.
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
        s = { following: true, unseenTail: false, pill: null, installed: false };
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

/**
 * Read the full follow state so a pane can save it on detach. Returns a
 * plain snapshot, not the live record — the caller stores these on the
 * pane object next to `scrollPos`. An unknown box follows and has
 * nothing unread: a console nobody has scrolled yet must track its
 * stream.
 */
export function getFollowState(container) {
    if (!container) return { following: true, unseenTail: false };
    const s = stateFor(container);
    return { following: s.following !== false, unseenTail: s.unseenTail === true };
}

/** Is tail-following currently engaged for this scroll box? */
export function isFollowing(container) {
    return getFollowState(container).following;
}

/**
 * Seed the follow state without moving the viewport. Used by the pane
 * mount/restore path (restoring a saved `followLive` / `unseenTail`)
 * and by the pane wipe (a cleared pane is a fresh conversation that
 * follows its tail with nothing unread behind it).
 *
 * `unseenTail` must be passed explicitly and defaults to false, because
 * the *default* answer at a re-seed is "nothing has appended since the
 * reader got here" — in particular whatever the PREVIOUS pane had
 * queued up is not this conversation's news. The mount path overrides
 * it with the value this pane itself saved on detach, so a pill the
 * reader never acknowledged is still waiting when they come back.
 */
export function setFollowState(container, { following, unseenTail = false } = {}) {
    if (!container) return;
    const s = stateFor(container);
    s.following = !!following;
    s.unseenTail = !!unseenTail;
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
        if (atBottom) s.unseenTail = false;
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
        s.unseenTail = true;
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
    s.unseenTail = false;
    container.scrollTop = container.scrollHeight;
    renderJumpToLatest(container);
}

/**
 * Agent- or system-originated append into a pane the reader is NOT
 * looking at (a backgrounded agent still streaming, a history repaint
 * into a switched-away conversation). The scroll box belongs to a
 * different conversation, so it must not be touched at all — but this
 * one still has to remember that content landed below its reader, or it
 * comes back detached with the tail silently unannounced.
 *
 * A pane that was following needs no record: its mount path snaps to the
 * tail, so the new content is on screen the moment it returns.
 */
export function notePaneAppend(pane) {
    if (!pane) return;
    if (pane.followLive === false) pane.unseenTail = true;
}

/**
 * User-originated write into a detached pane (their own send or queued
 * follow-up landing in a backgrounded agent). Same contract as
 * `forceScrollToBottom`, minus the scroll there is no viewport for: the
 * pane re-engages, so returning to it lands on what the user just did.
 */
export function notePaneUserAction(pane) {
    if (!pane) return;
    pane.followLive = true;
    pane.unseenTail = false;
}

/**
 * Reflect follow state in the "Jump to latest" affordance.
 *
 * The pill shows ONLY when following is disengaged AND content has
 * appended since it disengaged — drifting off the tail in an idle
 * conversation is not news, and a chat that silently stops following
 * with nothing to announce needs no affordance.
 *
 * It is built lazily, at the moment it first has to be visible, and
 * never merely to be hidden. That is not just an optimization: the
 * hidden case is the streaming hot path (every token calls
 * `maybeScrollToBottom`), and constructing a view in order to
 * immediately hide it would put DOM work on it, and would add a
 * permanent non-pane child to a container whose contract elsewhere in
 * `chat.js` is "holds the mounted pane".
 *
 * The pill is a child of the scroll box and `position: sticky` in CSS,
 * so it pins to the bottom of the scrollport without requiring a
 * positioned ancestor — this module cannot assume anything about
 * `#chat-container`'s ancestors, since embedders (`chat.mount()`)
 * supply their own chat chrome. `mountChatPane` empties the container
 * on every agent switch, hence the re-attach on show.
 */
function renderJumpToLatest(container) {
    const s = stateFor(container);
    const shouldShow = s.following === false && s.unseenTail;
    if (!shouldShow) {
        if (s.pill) s.pill.hidden = true;
        return;
    }
    if (typeof document === 'undefined' || typeof container.appendChild !== 'function') return;
    if (!s.pill) {
        // No aria-label: the visible text is the button's accessible name.
        const pill = document.createElement('button');
        pill.type = 'button';
        pill.className = JUMP_TO_LATEST_CLASS;
        pill.textContent = 'Jump to latest';
        pill.addEventListener('click', () => forceScrollToBottom(container));
        s.pill = pill;
    }
    if (s.pill.parentNode !== container) container.appendChild(s.pill);
    s.pill.hidden = false;
}
