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
        { session_id: 'a', has_human_turn: true, started_at: '2026-01-01T00:00:00Z', last_message_at: '2026-01-01T12:00:00Z' },
        { session_id: 'b', has_human_turn: true, started_at: '2026-02-01T00:00:00Z', last_message_at: '2026-02-15T08:00:00Z' },
        { session_id: 'c', has_human_turn: true, started_at: '2026-01-15T00:00:00Z', last_message_at: '2026-01-20T10:00:00Z' },
    ]);
    assert.equal(result.session_id, 'b');
});


test('auto-load falls back to started_at when last_message_at is missing', async () => {
    const pick = await loadPicker();
    const result = pick([
        { session_id: 'a', has_human_turn: true, started_at: '2026-01-01T00:00:00Z' },
        { session_id: 'b', has_human_turn: true, started_at: '2026-03-10T00:00:00Z' },
        { session_id: 'c', has_human_turn: true, started_at: '2026-02-15T00:00:00Z' },
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
        { session_id: 'bogus', has_human_turn: true, started_at: 'not-a-date' },
        { session_id: 'good', has_human_turn: true, started_at: '2026-01-01T00:00:00Z' },
    ]);
    // Only the good one has a finite timestamp, so it wins.
    assert.equal(result.session_id, 'good');
});


test('auto-load breaks ties by later array index (stable-ish)', async () => {
    const pick = await loadPicker();
    const ts = '2026-03-10T00:00:00Z';
    const result = pick([
        { session_id: 'first', has_human_turn: true, started_at: ts, last_message_at: ts },
        { session_id: 'second', has_human_turn: true, started_at: ts, last_message_at: ts },
    ]);
    // `>=` in the implementation means later array index wins on ties;
    // pin that behavior so a future refactor doesn't silently flip it.
    assert.equal(result.session_id, 'second');
});


// #3653: an agent that chains its own work keeps one autonomous session per
// chain (#3429), and that session is nearly always its newest. Auto-load must
// open the operator's conversation, not the agent's wake log.

test('auto-load skips a newer wake-only session for an older one a person took part in', async () => {
    const pick = await loadPicker();
    const result = pick([
        {
            session_id: 'chain',
            has_human_turn: false,
            preview: 'Autonomous wake — talon.job_complete',
            last_message_at: '2026-10-15T15:06:00Z',
        },
        {
            session_id: 'operator',
            has_human_turn: true,
            preview: 'can you check the deploy',
            last_message_at: '2026-10-14T13:40:00Z',
        },
    ]);
    assert.equal(result.session_id, 'operator');
});


test('auto-load picks nothing when no session has a human turn', async () => {
    const pick = await loadPicker();
    const result = pick([
        { session_id: 'chain-a', has_human_turn: false, last_message_at: '2026-10-15T15:06:00Z' },
        { session_id: 'chain-b', has_human_turn: false, last_message_at: '2026-10-09T08:00:00Z' },
    ]);
    // No fallback to an autonomous session: the pane stays empty.
    assert.equal(result, null);
});


test('auto-load reads the server flag, never the preview wording', async () => {
    const pick = await loadPicker();
    // A preview that reads like a typed turn does not make a session eligible
    // when the server says no person took part (or says nothing at all) ...
    assert.equal(pick([
        { session_id: 'flag-false', has_human_turn: false, preview: 'hello', last_message_at: '2026-10-15T00:00:00Z' },
        { session_id: 'flag-absent', preview: 'hello', last_message_at: '2026-10-15T00:00:00Z' },
    ]), null);
    // ... and a label that reads autonomous does not disqualify one it says a
    // person took part in.
    const result = pick([
        {
            session_id: 'flag-true',
            has_human_turn: true,
            preview: 'Autonomous wake — talon.job_complete',
            last_message_at: '2026-10-15T00:00:00Z',
        },
    ]);
    assert.equal(result.session_id, 'flag-true');
});
