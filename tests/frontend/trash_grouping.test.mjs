/**
 * Trash list grouping helpers (#765).
 *
 * These pure helpers drive the Trash sub-view's bucketing and
 * session rollup. The bucket logic matters for retention readability
 * (the user shouldn't see the same row labeled "Today" five seconds
 * before midnight and "Yesterday" five seconds after); the rollup
 * matters because /api/trash returns rows at the message level but
 * users think in conversations.
 */
import test from 'node:test';
import assert from 'node:assert/strict';

import { trashGroupKey, groupTrashBySession } from '../../kestrel_sovereign/static/js/trash_grouping.js';


// --- trashGroupKey -----------------------------------------------------------


// Day boundaries are computed in local time (that's what users see)
// so the tests use bare YYYY-MM-DD strings that the Date constructor
// parses as local midnight rather than UTC. Avoids timezone flakiness
// when the test runs in a non-UTC environment.

test('trashGroupKey: row deleted earlier today buckets as Today', () => {
    const now = new Date(2026, 3, 25, 15, 0, 0);  // local: 2026-04-25 15:00
    const earlier = new Date(2026, 3, 25, 3, 0, 0).toISOString();
    assert.equal(trashGroupKey(earlier, now), 'Today');
});


test('trashGroupKey: row deleted yesterday buckets as Yesterday', () => {
    const now = new Date(2026, 3, 25, 15, 0, 0);
    const yesterday = new Date(2026, 3, 24, 20, 0, 0).toISOString();
    assert.equal(trashGroupKey(yesterday, now), 'Yesterday');
});


test('trashGroupKey: row deleted 5 days ago buckets as Last 7 days', () => {
    const now = new Date(2026, 3, 25, 15, 0, 0);
    const fiveDaysAgo = new Date(2026, 3, 20, 12, 0, 0).toISOString();
    assert.equal(trashGroupKey(fiveDaysAgo, now), 'Last 7 days');
});


test('trashGroupKey: row deleted 30 days ago buckets as Older', () => {
    const now = new Date(2026, 3, 25, 15, 0, 0);
    const longAgo = new Date(2026, 2, 26, 12, 0, 0).toISOString();
    assert.equal(trashGroupKey(longAgo, now), 'Older');
});


test('trashGroupKey: missing deleted_at falls into Older', () => {
    assert.equal(trashGroupKey(null), 'Older');
    assert.equal(trashGroupKey(''), 'Older');
    assert.equal(trashGroupKey(undefined), 'Older');
});


test('trashGroupKey: unparseable timestamp falls into Older without crashing', () => {
    assert.equal(trashGroupKey('not-a-date'), 'Older');
});


// --- groupTrashBySession -----------------------------------------------------


test('groupTrashBySession: messages sharing a session_id roll up to a single entry', () => {
    const messages = [
        {
            id: 1, role: 'user', content: 'first',
            deleted_at: '2026-04-25T10:00:00',
            metadata: { session_id: 'sess-A' },
        },
        {
            id: 2, role: 'assistant', content: 'reply',
            deleted_at: '2026-04-25T10:00:01',
            metadata: { session_id: 'sess-A' },
        },
        {
            id: 3, role: 'user', content: 'second',
            deleted_at: '2026-04-25T10:00:02',
            metadata: { session_id: 'sess-A' },
        },
    ];

    const { sessions, orphans } = groupTrashBySession(messages);
    assert.equal(orphans.length, 0);
    assert.equal(sessions.length, 1);
    assert.equal(sessions[0].session_id, 'sess-A');
    assert.equal(sessions[0].count, 3);
});


test('groupTrashBySession: session deleted_at tracks the latest message', () => {
    // Session-level "when was this trashed" should pick the most
    // recent deleted_at across the session's rows. That's how the
    // user thinks about it.
    const messages = [
        {
            id: 1, role: 'user', content: 'old',
            deleted_at: '2026-04-20T08:00:00',
            metadata: { session_id: 'sess-B' },
        },
        {
            id: 2, role: 'user', content: 'newer',
            deleted_at: '2026-04-25T15:00:00',
            metadata: { session_id: 'sess-B' },
        },
    ];

    const { sessions } = groupTrashBySession(messages);
    assert.equal(sessions[0].deleted_at, '2026-04-25T15:00:00');
    assert.equal(sessions[0].latest_id, 2);
});


test('groupTrashBySession: prefers a user-role message for the preview', () => {
    const messages = [
        {
            id: 1, role: 'system', content: 'session start',
            deleted_at: '2026-04-25T10:00:00',
            metadata: { session_id: 'sess-C' },
        },
        {
            id: 2, role: 'assistant', content: 'first reply text',
            deleted_at: '2026-04-25T10:00:01',
            metadata: { session_id: 'sess-C' },
        },
        {
            id: 3, role: 'user', content: 'what the user actually said',
            deleted_at: '2026-04-25T10:00:02',
            metadata: { session_id: 'sess-C' },
        },
    ];

    const { sessions } = groupTrashBySession(messages);
    assert.equal(sessions[0].preview, 'what the user actually said');
});


test('groupTrashBySession: messages without session_id surface as orphans', () => {
    const messages = [
        {
            id: 99, role: 'system', content: 'standalone',
            deleted_at: '2026-04-25T10:00:00',
            metadata: {},
        },
        {
            id: 100, role: 'user', content: 'no metadata at all',
            deleted_at: '2026-04-25T10:00:01',
        },
    ];

    const { sessions, orphans } = groupTrashBySession(messages);
    assert.equal(sessions.length, 0);
    assert.equal(orphans.length, 2);
    assert.equal(orphans[0].id, 99);
    assert.equal(orphans[1].id, 100);
});


test('groupTrashBySession: empty / null input yields empty buckets without crashing', () => {
    let result = groupTrashBySession([]);
    assert.deepEqual(result, { sessions: [], orphans: [] });

    result = groupTrashBySession(null);
    assert.deepEqual(result, { sessions: [], orphans: [] });

    result = groupTrashBySession(undefined);
    assert.deepEqual(result, { sessions: [], orphans: [] });
});


test('groupTrashBySession: previews truncate to 80 chars', () => {
    const long = 'x'.repeat(200);
    const messages = [{
        id: 1, role: 'user', content: long,
        deleted_at: '2026-04-25T10:00:00',
        metadata: { session_id: 'sess-D' },
    }];

    const { sessions } = groupTrashBySession(messages);
    assert.equal(sessions[0].preview.length, 80);
});
