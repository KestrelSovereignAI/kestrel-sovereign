import test from 'node:test';
import assert from 'node:assert/strict';

// identity.js imports DOM / API-client singletons that aren't available in
// a plain Node context; we only need the pure picker helper, so we read
// the source and evaluate just that export in isolation.  Keeps this unit
// test hermetic without standing up jsdom.
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const IDENTITY_SRC = path.resolve(
    HERE,
    '../../kestrel_sovereign/static/js/identity.js',
);

async function loadPicker() {
    const src = await readFile(IDENTITY_SRC, 'utf-8');
    const match = src.match(
        /export function _pickMostRecentConversation[\s\S]*?\n\}/,
    );
    assert.ok(match, 'identity.js must export _pickMostRecentConversation');
    // Evaluate the function body in a fresh scope.  Drop the `export`
    // keyword so it's a plain declaration in the wrapper.
    const body = match[0].replace(/^export\s+/, '');
    const make = new Function(`${body}; return _pickMostRecentConversation;`);
    return make();
}


test('auto-load picks the conversation with the latest last_message_at', async () => {
    const pick = await loadPicker();
    const result = pick([
        { session_id: 'a', started_at: '2026-01-01T00:00:00Z', last_message_at: '2026-01-01T12:00:00Z' },
        { session_id: 'b', started_at: '2026-02-01T00:00:00Z', last_message_at: '2026-02-15T08:00:00Z' },
        { session_id: 'c', started_at: '2026-01-15T00:00:00Z', last_message_at: '2026-01-20T10:00:00Z' },
    ]);
    assert.equal(result.session_id, 'b');
});


test('auto-load falls back to started_at when last_message_at is missing', async () => {
    const pick = await loadPicker();
    const result = pick([
        { session_id: 'a', started_at: '2026-01-01T00:00:00Z' },
        { session_id: 'b', started_at: '2026-03-10T00:00:00Z' },
        { session_id: 'c', started_at: '2026-02-15T00:00:00Z' },
    ]);
    assert.equal(result.session_id, 'b');
});


test('auto-load returns null for an empty list', async () => {
    const pick = await loadPicker();
    assert.equal(pick([]), null);
});


test('auto-load returns null for non-array input (defensive)', async () => {
    const pick = await loadPicker();
    assert.equal(pick(null), null);
    assert.equal(pick(undefined), null);
});


test('auto-load skips entries with unparseable timestamps', async () => {
    const pick = await loadPicker();
    const result = pick([
        { session_id: 'bogus', started_at: 'not-a-date' },
        { session_id: 'good', started_at: '2026-01-01T00:00:00Z' },
    ]);
    // Only the good one has a finite timestamp, so it wins.
    assert.equal(result.session_id, 'good');
});


test('auto-load breaks ties by later array index (stable-ish)', async () => {
    const pick = await loadPicker();
    const ts = '2026-03-10T00:00:00Z';
    const result = pick([
        { session_id: 'first', started_at: ts, last_message_at: ts },
        { session_id: 'second', started_at: ts, last_message_at: ts },
    ]);
    // `>=` in the implementation means later array index wins on ties;
    // pin that behavior so a future refactor doesn't silently flip it.
    assert.equal(result.session_id, 'second');
});
