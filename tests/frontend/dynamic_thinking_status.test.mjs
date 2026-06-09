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

// 🔧 Calling <name>... start marker.
const start = (name) => `\u{1F527} Calling ${name}...`;
// ✓ <name> complete done marker.
const done = (name) => `✓ ${name} complete`;
// ❌ <name> failed error marker.
const fail = (name) => `❌ ${name} failed: boom`;

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

test('statusPhaseForChunk reads an in-flight tool start as that tool verb', () => {
    assert.equal(statusPhaseForChunk(start('web_search')), 'searching');
    assert.equal(statusPhaseForChunk(start('read_file')), 'reading');
});

test('statusPhaseForChunk returns the in-flight verb even with prose before the start', () => {
    // The prose BEFORE a 🔧 Calling marker is the prior step's output, so
    // an open tool call still wins — don't flip to "Writing…".
    assert.equal(statusPhaseForChunk(`Here is the plan.\n${start('run_shell')}`), 'running');
});

test('statusPhaseForChunk drops back to thinking when a tool completes', () => {
    assert.equal(statusPhaseForChunk(done('web_search')), 'thinking');
    assert.equal(statusPhaseForChunk(fail('run_shell')), 'thinking');
});

test('statusPhaseForChunk reads plain answer prose as writing', () => {
    assert.equal(statusPhaseForChunk('The answer is 42 because'), 'writing');
});

test('statusPhaseForChunk treats post-completion prose as writing', () => {
    // Tool finished, then the model resumes composing → "Writing…".
    assert.equal(statusPhaseForChunk(`${done('web_search')}\nBased on the results,`), 'writing');
});

test('statusPhaseForChunk ignores the bare --- wire delimiter as a phase signal', () => {
    assert.equal(statusPhaseForChunk('---'), null);
    assert.equal(statusPhaseForChunk('\n  \n'), null);
});

test('statusPhaseForChunk is null for an empty chunk (caller keeps prior phase)', () => {
    assert.equal(statusPhaseForChunk(''), null);
    assert.equal(statusPhaseForChunk(null), null);
});

test('statusPhaseForChunk resolves a completed marker embedded in cumulative content', () => {
    // The loop feeds the cumulative tail, so a marker that arrived split
    // across packets ("🔧 Calling web_" + "search...") resolves once whole,
    // even with prior prose and a finished earlier tool ahead of it.
    const cumulative = `Let me look that up.\n${done('read_file')}\nOk.\n${start('web_search')}`;
    assert.equal(statusPhaseForChunk(cumulative), 'searching');
});

test('statusPhaseForChunk keeps the in-flight tool verb past an earlier completed tool', () => {
    const cumulative = `${start('read_file')}\n${done('read_file')}\n${start('run_shell')}`;
    assert.equal(statusPhaseForChunk(cumulative), 'running');
});

test('statusPhaseForChunk does not corrupt TOOL_MARKER_TOKEN lastIndex across calls', () => {
    // The shared global regex is used via .match/.replace only; two calls
    // in a row must be independent (a stale lastIndex would drop a match).
    const chunk = start('web_search');
    assert.equal(statusPhaseForChunk(chunk), 'searching');
    assert.equal(statusPhaseForChunk(chunk), 'searching');
});
