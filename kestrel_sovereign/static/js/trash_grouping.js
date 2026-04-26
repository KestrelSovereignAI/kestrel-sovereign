/**
 * Trash list grouping helpers (#765).
 *
 * Pure JS — no DOM, no browser APIs. The Trash sub-view in identity.js
 * renders against these results; pulling them into a standalone module
 * keeps the unit-test surface small and lets the helpers run under
 * Node's built-in test runner.
 */

/**
 * Bucket a deleted_at timestamp into one of the four buckets the
 * design discussion settled on:
 *
 *   Today / Yesterday / Last 7 days / Older
 *
 * Anything unparseable falls into "Older" — better to surface than
 * to drop it.
 *
 * The "now" injection is for tests; production code passes nothing
 * and the function uses the current wall clock.
 */
export function trashGroupKey(deletedAt, now = new Date()) {
    if (!deletedAt) return 'Older';
    const d = new Date(deletedAt);
    if (Number.isNaN(d.getTime())) return 'Older';

    const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const startOfYesterday = new Date(startOfToday);
    startOfYesterday.setDate(startOfYesterday.getDate() - 1);
    const sevenDaysAgo = new Date(startOfToday);
    sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);

    if (d >= startOfToday) return 'Today';
    if (d >= startOfYesterday) return 'Yesterday';
    if (d >= sevenDaysAgo) return 'Last 7 days';
    return 'Older';
}

/**
 * Roll a flat list of trashed messages up to one entry per session.
 *
 * The /api/trash endpoint returns soft-deleted messages at the row
 * level. Users think in conversations ("I deleted that thread about
 * X"), so the UI needs to group rows that share a metadata.session_id
 * into a single Trash item with a count and a preview.
 *
 * Messages without a session_id (orphans, system markers) are
 * surfaced individually so the user can still recover them.
 */
export function groupTrashBySession(messages) {
    const sessions = new Map();
    const orphans = [];
    for (const msg of messages || []) {
        const sid = msg?.metadata?.session_id;
        if (!sid) {
            orphans.push(msg);
            continue;
        }
        if (!sessions.has(sid)) {
            sessions.set(sid, {
                session_id: sid,
                deleted_at: msg.deleted_at,
                preview: (msg.content || '').slice(0, 80) || '(empty)',
                count: 0,
                latest_id: msg.id,
            });
        }
        const entry = sessions.get(sid);
        entry.count += 1;
        // Track the latest deleted_at within the session so the
        // bucket assignment matches the user's mental model.
        if ((msg.deleted_at || '') > (entry.deleted_at || '')) {
            entry.deleted_at = msg.deleted_at;
            entry.latest_id = msg.id;
        }
        // Prefer a user-role message for the preview — those are the
        // most recognizable to the operator.
        if (msg.role === 'user' && msg.content) {
            entry.preview = msg.content.slice(0, 80);
        }
    }
    return { sessions: Array.from(sessions.values()), orphans };
}
