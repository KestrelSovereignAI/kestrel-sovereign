import test from 'node:test';
import assert from 'node:assert/strict';

// Dynamic thinking-status: the chat indicator replaces the static
// "Thinking…" with a one-word description of the agent's current stream
// phase — "Searching…" while a search tool runs, "Reading…", "Running…",
// "Writing…" once answer prose flows. These tests pin the PURE derivation
// helpers (`toolStatusPhase`, `statusPhaseForChunk`) that map the wire
// signals the renderer already parses (tool markers, prose) onto a phase
// key. The indicator wiring (pane.statusPhase → updateThinkingIndicator →
// i18n word) is integration; the mapping is the part worth pinning here.

// chat.js touches a handful of globals at import time; stub the minimum.
globalThis.window = globalThis.window || {};
globalThis.window.SharedMarkdown = {
    renderMarkdown: () => '',
    renderStreamingMarkdown: (s) => s,
    highlightCodeBlocks: () => {},
    renderMermaidDiagrams: () => {},
    finalizeMarkdown: async (el, content) => { el.textContent = content; },
};
const stubNode = () => ({
    classList: { add() {}, remove() {} },
    style: {},
    dataset: {},
    appendChild() {},
    setAttribute() {},
    addEventListener() {},
});
globalThis.document = {
    getElementById: () => null,
    createElement: () => stubNode(),
    head: stubNode(),
    body: stubNode(),
    addEventListener() {},
    querySelector() { return null; },
    querySelectorAll() { return []; },
};
globalThis.sessionStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
globalThis.location = { href: '/', search: '' };
globalThis.fetch = async () => ({ ok: false, status: 500 });
globalThis.CSS = { escape: (s) => String(s) };
globalThis.kicon = () => '';

const { toolStatusPhase, statusPhaseForChunk } = await import(
    '../../kestrel_sovereign/static/js/chat.js'
);

test('toolStatusPhase maps common tool names to functional verbs', () => {
    assert.equal(toolStatusPhase('web_search'), 'searching');
    assert.equal(toolStatusPhase('read_file'), 'reading');
    assert.equal(toolStatusPhase('write_file'), 'writing');
    assert.equal(toolStatusPhase('run_shell'), 'running');
    assert.equal(toolStatusPhase('save_fact'), 'remembering');
    assert.equal(toolStatusPhase('analyze_image'), 'looking');
    assert.equal(toolStatusPhase('ask_agent'), 'consulting');
});

test('toolStatusPhase resolves namespaced and detail-suffixed names', () => {
    assert.equal(toolStatusPhase('github.create_issue'), 'pushing');
    assert.equal(toolStatusPhase('web_search: kestrel falcons'), 'searching');
});

test('toolStatusPhase falls back to the honest generic for unknown tools', () => {
    assert.equal(toolStatusPhase('frobnicate_widget'), 'working');
    assert.equal(toolStatusPhase(''), 'working');
    assert.equal(toolStatusPhase(null), 'working');
});

// Regression pins for two real-inventory bugs the first cut had:
test('toolStatusPhase does not mistake webhooks_* for web search', () => {
    // Bare "web" used to flag these as "searching".
    assert.equal(toolStatusPhase('webhooks_list'), 'reading');
    assert.equal(toolStatusPhase('webhooks_register'), 'writing');
    assert.equal(toolStatusPhase('webhooks_remove'), 'writing');
    assert.equal(toolStatusPhase('search_documents'), 'searching');
});

test('toolStatusPhase does not mistake *_task for "ask" (consulting)', () => {
    // Bare "ask" matched the "ask" inside "task"; task orchestration is
    // now intentionally consulting, but via real tokens.
    assert.equal(toolStatusPhase('cancel_task'), 'consulting');
    assert.equal(toolStatusPhase('ask_agent'), 'consulting');
    assert.equal(toolStatusPhase('deploy_agent'), 'consulting');
    assert.equal(toolStatusPhase('council_convene'), 'consulting');
});

test('toolStatusPhase resolves underscore-separated verbs (no leading-\\b miss)', () => {
    // These all sit after an underscore, which kills a leading \\b.
    assert.equal(toolStatusPhase('fs_list'), 'reading');
    assert.equal(toolStatusPhase('schedule_list'), 'reading');
    assert.equal(toolStatusPhase('health_check'), 'reading');
    assert.equal(toolStatusPhase('strategy_add_decision'), 'writing');
    assert.equal(toolStatusPhase('channels_send'), 'writing');
});

test('toolStatusPhase maps the real read/write/recall/voice families', () => {
    assert.equal(toolStatusPhase('git_status'), 'reading');
    assert.equal(toolStatusPhase('list_models'), 'reading');
    assert.equal(toolStatusPhase('fs_write'), 'writing');
    assert.equal(toolStatusPhase('set_model'), 'writing');
    assert.equal(toolStatusPhase('recall_recent'), 'remembering');
    assert.equal(toolStatusPhase('run_script'), 'running');
    assert.equal(toolStatusPhase('speak'), 'speaking');
    assert.equal(toolStatusPhase('transcribe'), 'listening');
});

// #1659: statusPhaseForChunk is now prose-only. Tool activity is structured
// (typed TOOL sentinels), so markers never appear in the visible stream — the
// in-flight tool's verb is set by the streaming loop via toolStatusPhase on
// the start sentinel, not by scanning text here.

test('statusPhaseForChunk maps answer prose to writing', () => {
    assert.equal(statusPhaseForChunk('The answer is 42 because'), 'writing');
    assert.equal(statusPhaseForChunk('Based on the results, here is'), 'writing');
});

test('statusPhaseForChunk ignores the bare --- wire delimiter and whitespace', () => {
    assert.equal(statusPhaseForChunk('---'), null);
    assert.equal(statusPhaseForChunk('\n  \n'), null);
});

test('statusPhaseForChunk is null for an empty chunk (caller keeps prior phase)', () => {
    assert.equal(statusPhaseForChunk(''), null);
    assert.equal(statusPhaseForChunk(null), null);
});

test('statusPhaseForChunk no longer derives tool verbs from text (#1659)', () => {
    // A leftover glyph in the visible stream reads as ordinary prose now; the
    // tool verb comes from toolStatusPhase on the typed start sentinel.
    assert.equal(statusPhaseForChunk('\u{1F527} Calling web_search...'), 'writing');
    assert.equal(toolStatusPhase('web_search'), 'searching');
});
