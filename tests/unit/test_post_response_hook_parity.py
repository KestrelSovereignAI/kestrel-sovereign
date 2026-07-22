"""
#2675: non-streaming POST_RESPONSE parity with the streaming path.

The deterministic narration check (#1042 layer 3, ``ResponseAuditHook``) reads
``HookInput.pre_tool_prose`` against ``HookInput.tool_results``. Before #2675 the
streaming path populated ``pre_tool_prose`` / ``tool_calls`` / ``tool_results``
on POST_RESPONSE while the non-streaming path fired the hook with only
``response_text`` — so the SAME dishonest "tool succeeded" narration that
streaming caught silently no-op'd on the non-streaming path.

These tests drive the REAL streaming (``process_input_streaming``) and
non-streaming (``_process_input_traced_locked``) methods with EQUIVALENT tool
turns, capture the POST_RESPONSE ``HookInput`` each path assembles, and assert
BOTH:

* the evidence fields (``pre_tool_prose`` / ``tool_calls`` / ``tool_results``)
  match across the two transports, and
* a real ``ResponseAuditHook`` returns the SAME narration verdict on both.

The streaming harness mirrors ``test_streaming_post_response_narration_fields``;
the non-streaming harness drives the real traced-locked body so this test
actually protects the #2675 assembly code.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sdk.hooks.base import HookEvent, HookOutput
from kestrel_sdk.llm import ToolCallStarted


# A single equivalent "dishonest tool success" turn, expressed once so both
# transports narrate the same pre-tool prose, issue the same tool call, and
# observe the same failing result envelope.
_PRE_TOOL_PROSE = "Saved your color."
_TOOL_ID = "tc-1"
_TOOL_NAME = "save_fact"
_TOOL_ARGS = {"fact": "color=teal"}
_TOOL_RESULT_ENVELOPE = {
    "tool_call_id": _TOOL_ID,
    "name": _TOOL_NAME,
    "arguments": _TOOL_ARGS,
    "result": {"status": "error", "error": "no store"},
}
_POST_TOOL_TEXT = "Looking at the result, the save did not persist."


@asynccontextmanager
async def _passthrough():
    yield


class _CapturingHooks:
    """Fake hooks_manager shared by both transports. Records the POST_RESPONSE
    ``HookInput`` (whichever entry point delivers it — the streaming and
    non-streaming paths both fire it snapshot-pinned via
    ``execute_hooks_snapshot``) and accepts every other firing as a benign
    ALLOW no-op."""

    def __init__(self):
        self.post_response_input = None

    def get_enabled_hooks(self, event):
        # One enabled (non-enforcing) POST_RESPONSE hook: enough for the
        # turn-start snapshot to be non-empty so the audit actually fires,
        # but ``fail_closed``/``awaits_user_input`` absent → warn semantics
        # (no buffering) on both paths.
        if event == HookEvent.POST_RESPONSE:
            return [object()]
        return []

    def _maybe_capture(self, event, hook_input):
        if event == HookEvent.POST_RESPONSE:
            self.post_response_input = hook_input

    async def execute_hooks(self, event, hook_input, **_kwargs):
        self._maybe_capture(event, hook_input)
        return HookOutput.allow("ok")

    async def execute_hooks_snapshot(self, event, hook_input, _hooks, **_kwargs):
        self._maybe_capture(event, hook_input)
        return HookOutput.allow("ok")

    async def execute_post_response_observers(
        self, event, hook_input, _hooks, **_kwargs
    ):
        return HookOutput.allow("ok")

    async def execute_hooks_parallel(self, _event, _hook_input):
        return None


# ---------------------------------------------------------------------------
# Streaming harness (mirrors test_streaming_post_response_narration_fields)
# ---------------------------------------------------------------------------


def _build_streaming_agent(hooks: _CapturingHooks):
    from kestrel_sovereign.agent.streaming import StreamingMixin

    privacy_agent = MagicMock()
    privacy_agent.add_conversation = AsyncMock()
    privacy_agent.privacy_config.allows_cloud_llm.return_value = True
    privacy_agent.privacy_mode.name = "normal"
    privacy_agent.get_conversation_history = AsyncMock(return_value=[])

    agent = MagicMock()
    agent.privacy_agent = privacy_agent
    agent.features = {}
    agent.did = "did:test:streaming"
    agent.extension = None
    agent._cached_features_prompt = ""
    agent.is_request_cancelled = MagicMock(return_value=False)
    agent._maybe_audit = AsyncMock()
    agent._genesis_audit_cognition_block = AsyncMock(return_value=None)
    agent._get_privacy_transition_lock = MagicMock(return_value=_passthrough())
    agent._turn_lifecycle = MagicMock(return_value=_passthrough())
    agent.hooks_manager = hooks
    agent._get_governing_constitution = AsyncMock(return_value="")
    agent.check_solvency = AsyncMock(return_value="test-model")
    agent._build_all_tools = MagicMock(return_value=[])
    agent.user_prompt_template = MagicMock()
    agent.user_prompt_template.format.return_value = "rendered"
    agent.emit_event = AsyncMock()

    ctx = MagicMock()
    ctx.system_prompt = "system"
    ctx.dynamic_user_context = ""
    ctx.messages = []
    agent.context_manager = MagicMock()
    agent.context_manager.build_context = AsyncMock(return_value=ctx)

    agent.observability_store = MagicMock()
    agent.observability_store.log_tool_call = AsyncMock(return_value="evt-1")
    agent.observability_store.log_tool_response = AsyncMock()
    agent.observability_store.log_metric = AsyncMock()

    agent.process_input_streaming = StreamingMixin.process_input_streaming.__get__(agent)
    agent._process_input_streaming_traced_locked = (
        StreamingMixin._process_input_streaming_traced_locked.__get__(agent)
    )
    agent._persist_assistant_turn_safely = (
        StreamingMixin._persist_assistant_turn_safely.__get__(agent)
    )
    agent._fire_post_response_hook = (
        StreamingMixin._fire_post_response_hook.__get__(agent)
    )
    agent._emit_revising_event = (
        StreamingMixin._emit_revising_event.__get__(agent)
    )
    return agent


async def _run_streaming_tool_turn(hooks: _CapturingHooks):
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    agent = _build_streaming_agent(hooks)

    async def stream():
        yield _PRE_TOOL_PROSE
        yield ToolCallStarted(index=0, id=_TOOL_ID, name=_TOOL_NAME)
        yield LLMResponse(
            content="",
            tool_calls=[ToolCall(id=_TOOL_ID, name=_TOOL_NAME, arguments=_TOOL_ARGS)],
        )

    agent.llm_service = MagicMock()
    agent.llm_service.stream_with_tool_detection = lambda **kw: stream()

    async def post_tool(*, tool_results=None, **kw):
        if tool_results is not None:
            tool_results.append(dict(_TOOL_RESULT_ENVELOPE))
        yield _POST_TOOL_TEXT

    agent._handle_orchestrator_response_streaming = post_tool

    async for _ in agent.process_input_streaming("save my color", session_id="s-stream"):
        pass
    return hooks.post_response_input


# ---------------------------------------------------------------------------
# Non-streaming harness (drives the real _process_input_traced_locked body)
# ---------------------------------------------------------------------------


def _build_nonstreaming_agent(hooks: _CapturingHooks):
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.agent.streaming import StreamingMixin
    from kestrel_sovereign.privacy import PrivacyMode

    agent = MagicMock()
    agent.did = "did:test:nonstream"
    agent.extension = None
    agent.features = {}
    agent._session_briefed = True
    agent._privacy_mode = PrivacyMode.NORMAL
    agent.hooks_manager = hooks

    privacy_agent = MagicMock()
    privacy_agent.add_conversation = AsyncMock()
    privacy_agent.get_conversation_history = AsyncMock(return_value=[])
    privacy_agent.privacy_config.allows_cloud_llm.return_value = True
    agent.privacy_agent = privacy_agent

    agent._maybe_compact_codex_thread = AsyncMock()
    agent._get_governing_constitution = AsyncMock(return_value="")
    agent.check_solvency = AsyncMock(return_value="test-model")
    agent._build_all_tools = MagicMock(return_value=[])
    agent._assemble_post_build_system_prompt = MagicMock(return_value="system")
    agent.user_prompt_template = MagicMock()
    agent.user_prompt_template.format.return_value = "rendered"
    agent._persist_assistant_conversation = AsyncMock()
    agent._post_response_pipeline = AsyncMock()

    ctx = MagicMock()
    ctx.system_prompt = "system"
    ctx.dynamic_user_context = ""
    ctx.messages = []
    ctx.degraded_mode = False
    ctx.warnings = []
    ctx.budget_summary = {}
    ctx.total_tokens = 0
    ctx.episode_count = 0
    ctx.memory_count = 0
    ctx.rag_chunks = 0
    agent.context_manager = MagicMock()
    agent.context_manager.build_context = AsyncMock(return_value=ctx)

    agent.observability_store = MagicMock()
    agent.observability_store.log_metric = AsyncMock()
    agent.observability_store.log_tool_call = AsyncMock(return_value="evt")
    agent.observability_store.log_tool_response = AsyncMock()
    agent.observability_store.log_error = AsyncMock()

    agent._process_input_traced_locked = (
        KestrelAgent._process_input_traced_locked.__get__(agent)
    )
    agent._fire_post_response_hook = (
        StreamingMixin._fire_post_response_hook.__get__(agent)
    )
    return agent


def _patch_nonstreaming_module_helpers(monkeypatch):
    """Neutralize the module-level side helpers the traced-locked body calls,
    so the harness exercises the #2675 evidence assembly in isolation."""
    import kestrel_sovereign.kestrel_agent as ka
    import kestrel_sovereign.agent.preturn_state as ps

    monkeypatch.setattr(ka, "check_prompt_injection", lambda _text: None)
    monkeypatch.setattr(
        ka, "resolve_turn_invocation_context", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        ka,
        "inject_operator_turn",
        AsyncMock(return_value=MagicMock(keep_trailing_system=False)),
    )
    monkeypatch.setattr(
        ps, "build_operational_state_block", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        ps, "build_preturn_state_block", AsyncMock(return_value=None)
    )


async def _run_nonstreaming_turn(hooks, monkeypatch, *, initial_response, orchestrator):
    _patch_nonstreaming_module_helpers(monkeypatch)
    agent = _build_nonstreaming_agent(hooks)
    agent.llm_service = MagicMock()
    agent.llm_service.generate_with_messages = AsyncMock(return_value=initial_response)
    agent._handle_orchestrator_response = orchestrator
    await agent._process_input_traced_locked(
        "save my color", "test-model", "s-nonstream", None,
    )
    return hooks.post_response_input


async def _run_nonstreaming_tool_turn(hooks, monkeypatch):
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    initial = LLMResponse(
        content=_PRE_TOOL_PROSE,
        tool_calls=[ToolCall(id=_TOOL_ID, name=_TOOL_NAME, arguments=_TOOL_ARGS)],
    )

    async def orchestrator(*, tool_results=None, **kw):
        if tool_results is not None:
            tool_results.append(dict(_TOOL_RESULT_ENVELOPE))
        return _POST_TOOL_TEXT

    return await _run_nonstreaming_turn(
        hooks, monkeypatch, initial_response=initial, orchestrator=orchestrator
    )


# ---------------------------------------------------------------------------
# Real-Codex inline-tool harness (#2675 P1: the dominant openai:plan transport)
# ---------------------------------------------------------------------------
#
# Codex executes kestrel-dispatched tools INLINE: ``get_response()`` returns
# ``tool_calls=None`` and exposes the calls via ``executed_tool_calls`` — so the
# non-streaming assembly's ``has_tool_calls`` branch is False even though tools
# ran and ``stop_tool_results`` is populated. The streaming path snapshots
# pre-tool prose at the first ``ToolCallStarted`` marker; non-streaming
# ``CodexAdapter.get_response`` used to drop every marker event and keep only
# ``content`` + ``executed_tool_calls``, leaving the narration check with NO
# pre-tool evidence for the dominant transport. These tests drive the REAL
# ``CodexAdapter`` over a scripted inline event sequence (pre-tool prose ->
# FAILED dynamicToolCall -> full post-tool agentMessage) so they protect both
# the adapter's ``pre_tool_prose`` preservation and the kestrel_agent assembly
# that reads it — not a stub.

_INLINE_PRE_TOOL_PROSE = "Saved your color."
_INLINE_FULL_CONTENT = "Saved your color. It is all set now."
_INLINE_TOOL_ID = "call_inline_1"
_INLINE_TOOL_NAME = "save_fact"
_INLINE_TOOL_ARGS = {"fact": "color=teal"}
_INLINE_FAILED_RESULT = {"success": False, "error": "no store configured"}


class _CodexInlineFailedToolApp:
    """Minimal ``CodexAppServerClient`` stand-in driving the adapter's real
    inline-tool path: ``turn/start`` invokes the registered ``item/tool/call``
    handler (so ``executed_tool_calls`` is populated with a FAILED result), and
    ``iter_turn_events`` scripts the pre-tool prose delta, the dynamicToolCall
    completion (the ``pre_tool_prose`` snapshot boundary), then the full
    post-tool ``agentMessage``."""

    def __init__(self):
        self.registered = {}

    async def ensure_started(self):
        pass

    async def request(self, method, params=None, *, timeout=120):
        if method == "thread/start":
            return {"thread": {"id": "thr-inline"}}
        if method == "turn/start":
            handler = self.registered.get(("item/tool/call", "thr-inline"))
            assert handler is not None, "tool-call handler missing"
            await handler({
                "threadId": "thr-inline",
                "callId": _INLINE_TOOL_ID,
                "tool": _INLINE_TOOL_NAME,
                "arguments": json.dumps(_INLINE_TOOL_ARGS),
            })
            return {"turn": {"id": "turn-inline"}}
        return {}

    def register_server_request_handler(self, m, h, *, thread_id=None):
        self.registered[(m, thread_id)] = h
        return lambda: self.registered.pop((m, thread_id), None)

    def open_turn_sink(self, key):
        return key

    def close_turn_sink(self, key):
        pass

    async def iter_turn_events(
        self, sink, *, idle_timeout=120, thread_id=None, cancel_token=None
    ):
        for ev in [
            {"method": "item/agentMessage/delta",
             "params": {"delta": _INLINE_PRE_TOOL_PROSE}},
            {"method": "item/completed", "params": {"item": {
                "type": "dynamicToolCall", "id": _INLINE_TOOL_ID,
                "name": _INLINE_TOOL_NAME,
                "arguments": json.dumps(_INLINE_TOOL_ARGS),
            }}},
            {"method": "item/completed", "params": {"item": {
                "type": "agentMessage", "text": _INLINE_FULL_CONTENT,
            }}},
            {"method": "turn/completed",
             "params": {"turn": {"status": "completed"}}},
        ]:
            yield ev


async def _inline_tool_executor(name, args):
    return dict(_INLINE_FAILED_RESULT)


async def _drive_real_codex_inline_response():
    """The REAL ``LLMResponse`` ``CodexAdapter.get_response`` produces for the
    inline failed-tool turn."""
    from kestrel_sovereign.llm.codex_adapter import CodexAdapter

    a = CodexAdapter()
    a._client = _CodexInlineFailedToolApp()
    return await a.get_response(
        client="x", model="auto",
        messages=[{"role": "user", "content": "save my color"}],
        tools=[{"type": "function", "function": {
            "name": _INLINE_TOOL_NAME, "description": "d",
            "parameters": {"type": "object"}}}],
        session_id="s-inline", tool_executor=_inline_tool_executor,
    )


def _fold_inline_executed_into_tool_results(response, tool_results):
    """Mirror the orchestrator's ``_append_executed_tool_breadcrumbs``: fold an
    inline-executing adapter's ``executed_tool_calls`` into the audit-shaped
    ``tool_results`` envelopes the non-streaming assembly derives ``tool_calls``
    from and hands POST_RESPONSE."""
    from kestrel_sovereign.security.narration_check import (
        summarize_tool_result_for_audit,
    )

    for e in getattr(response, "executed_tool_calls", None) or []:
        tool_results.append({
            "tool_call_id": e["id"],
            "name": e["name"],
            "arguments": e["arguments"],
            "result": summarize_tool_result_for_audit(e["result"]),
        })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_turn_evidence_matches_across_transports(monkeypatch):
    """Equivalent tool turns must produce equivalent POST_RESPONSE evidence on
    both transports — the core #2675 parity claim."""
    st_input = await _run_streaming_tool_turn(_CapturingHooks())
    ns_input = await _run_nonstreaming_tool_turn(_CapturingHooks(), monkeypatch)

    assert st_input is not None, "streaming path never fired POST_RESPONSE"
    assert ns_input is not None, "non-streaming path never fired POST_RESPONSE"

    # Pre-tool prose: streaming snapshots it at the marker boundary;
    # non-streaming reads it off the initial response's content. Same words.
    assert st_input.pre_tool_prose == _PRE_TOOL_PROSE
    assert ns_input.pre_tool_prose == st_input.pre_tool_prose

    # Tool calls: both derive {id, name, arguments} — streaming from the LLM
    # tool_calls, non-streaming from the accumulated result envelopes.
    assert st_input.tool_calls == [
        {"id": _TOOL_ID, "name": _TOOL_NAME, "arguments": _TOOL_ARGS}
    ]
    assert ns_input.tool_calls == st_input.tool_calls

    # Tool results: identical envelope list on both paths.
    assert st_input.tool_results == [_TOOL_RESULT_ENVELOPE]
    assert ns_input.tool_results == st_input.tool_results


@pytest.mark.asyncio
async def test_narration_check_fires_identically_on_both_transports(monkeypatch):
    """A real ResponseAuditHook run over each path's captured HookInput must
    reach the SAME deterministic narration verdict — the dishonest 'Saved.'
    before a failed tool is caught on non-streaming exactly as on streaming."""
    from kestrel_sovereign.features.response_audit.hook import ResponseAuditHook

    st_input = await _run_streaming_tool_turn(_CapturingHooks())
    ns_input = await _run_nonstreaming_tool_turn(_CapturingHooks(), monkeypatch)

    def _make_audit_agent():
        agent = MagicMock()
        agent.llm_service = MagicMock()
        # Force the deterministic-only path: the LLM audit is unavailable, so
        # only the narration check drives the verdict (parity is about the
        # deterministic signal, not the probabilistic LLM judgment).
        agent.llm_service.get_audit_response = AsyncMock(
            side_effect=RuntimeError("audit provider down")
        )
        agent.features = {}
        return agent

    st_hook = ResponseAuditHook(agent=_make_audit_agent(), mode="warn", risk_threshold=3)
    ns_hook = ResponseAuditHook(agent=_make_audit_agent(), mode="warn", risk_threshold=3)

    st_out = await st_hook.execute(st_input)
    ns_out = await ns_hook.execute(ns_input)

    # Deterministic narration verdict is identical on both paths.
    assert st_hook.last_narration_verdict.risk_boost == 2
    assert ns_hook.last_narration_verdict.risk_boost == 2
    assert (
        ns_hook.last_narration_verdict.reasoning
        == st_hook.last_narration_verdict.reasoning
    )
    assert st_hook.last_narration_verdict.offending_verb == "saved"
    assert ns_hook.last_narration_verdict.offending_verb == "saved"
    assert st_hook.last_narration_verdict.offending_tool == _TOOL_NAME
    assert ns_hook.last_narration_verdict.offending_tool == _TOOL_NAME

    # Both annotate the response with the audit warning (warn-mode MODIFY,
    # carried as an ``updated_input`` rewrite) rather than silently passing the
    # dishonest narration through — the same observable outcome on both paths.
    assert st_out.permission_decision == ns_out.permission_decision
    assert st_out.updated_input is not None and ns_out.updated_input is not None
    assert "narrated completion before" in st_out.updated_input["response_text"]
    assert "narrated completion before" in ns_out.updated_input["response_text"]


@pytest.mark.asyncio
async def test_no_tool_turn_evidence_matches_and_stays_empty(monkeypatch):
    """No-tool turns remain unchanged: both transports pass None for all three
    narration fields, so the check has nothing to verify."""
    from kestrel_sovereign.llm.adapter import LLMResponse

    # Streaming no-tool turn.
    st_hooks = _CapturingHooks()
    st_agent = _build_streaming_agent(st_hooks)

    async def stream():
        yield "Hello! No tools needed."

    st_agent.llm_service = MagicMock()
    st_agent.llm_service.stream_with_tool_detection = lambda **kw: stream()
    async for _ in st_agent.process_input_streaming("hi", session_id="s-st-notool"):
        pass
    st_input = st_hooks.post_response_input

    # Non-streaming no-tool turn.
    ns_hooks = _CapturingHooks()
    initial = LLMResponse(content="Hello! No tools needed.", tool_calls=None)

    async def orchestrator(*, response, tool_results=None, **kw):
        return response.content or ""

    ns_input = await _run_nonstreaming_turn(
        ns_hooks, monkeypatch, initial_response=initial, orchestrator=orchestrator
    )

    for hook_input in (st_input, ns_input):
        assert hook_input is not None
        assert hook_input.pre_tool_prose is None
        assert hook_input.tool_calls is None
        assert hook_input.tool_results is None


@pytest.mark.asyncio
async def test_nonstreaming_multi_iteration_tool_evidence_ordered(monkeypatch):
    """Multi-iteration non-streaming flow (model calls A, sees result, calls B)
    must hand POST_RESPONSE tool_calls and tool_results that line up by index
    across every iteration — ordered and complete, not just the first call."""
    from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

    hooks = _CapturingHooks()
    initial = LLMResponse(
        content="Working on it.",
        tool_calls=[ToolCall(id="tc-A", name="lookup_a", arguments={"q": "first"})],
    )

    async def orchestrator(*, tool_results=None, **kw):
        # The orchestrator accumulates BOTH iterations' envelopes in order.
        if tool_results is not None:
            tool_results.append({
                "tool_call_id": "tc-A",
                "name": "lookup_a",
                "arguments": {"q": "first"},
                "result": {"status": "ok"},
            })
            tool_results.append({
                "tool_call_id": "tc-B",
                "name": "lookup_b",
                "arguments": {"q": "second"},
                "result": {"status": "ok"},
            })
        return "All done."

    hook_input = await _run_nonstreaming_turn(
        hooks, monkeypatch, initial_response=initial, orchestrator=orchestrator
    )

    assert hook_input is not None
    assert hook_input.tool_calls is not None and len(hook_input.tool_calls) == 2
    assert hook_input.tool_results is not None and len(hook_input.tool_results) == 2
    # Index alignment across iterations.
    for i in range(2):
        assert hook_input.tool_calls[i]["id"] == hook_input.tool_results[i]["tool_call_id"]
        assert hook_input.tool_calls[i]["name"] == hook_input.tool_results[i]["name"]
        assert (
            hook_input.tool_calls[i]["arguments"]
            == hook_input.tool_results[i]["arguments"]
        )
    assert [tc["name"] for tc in hook_input.tool_calls] == ["lookup_a", "lookup_b"]


@pytest.mark.asyncio
async def test_inline_codex_get_response_preserves_marker_bound_pre_tool_prose():
    """The real non-streaming ``CodexAdapter.get_response`` preserves the
    pre-tool prose snapshot taken at the FIRST inline tool boundary — NOT the
    full pre+post-tool content. This is the adapter half of the #2675 P1 fix:
    before it, the adapter dropped every marker event and kept only
    ``content`` + ``executed_tool_calls``, so the non-streaming narration check
    had no pre-tool evidence for the dominant openai:plan transport."""
    resp = await _drive_real_codex_inline_response()

    # Inline execution: calls ride ``executed_tool_calls`` — they are NOT
    # re-surfaced as ``tool_calls`` (that would make the orchestrator
    # re-dispatch and duplicate side effects).
    assert not resp.tool_calls
    executed = getattr(resp, "executed_tool_calls", None)
    assert executed and len(executed) == 1
    assert executed[0]["name"] == _INLINE_TOOL_NAME
    assert executed[0]["arguments"] == _INLINE_TOOL_ARGS
    assert executed[0]["result"] == _INLINE_FAILED_RESULT

    # ``content`` is the FULL turn (pre- AND post-tool synthesis) ...
    assert resp.content == _INLINE_FULL_CONTENT
    # ... while ``pre_tool_prose`` is the marker-bound snapshot — the pre-tool
    # half only, explicitly NOT derived from the full content.
    assert getattr(resp, "pre_tool_prose", None) == _INLINE_PRE_TOOL_PROSE
    assert resp.pre_tool_prose != resp.content


@pytest.mark.asyncio
async def test_inline_codex_streaming_and_nonstreaming_snapshot_same_boundary():
    """Cross-transport parity at the adapter boundary: the pre-tool text the
    streaming path sees before the first ``ToolCallStarted`` equals the
    ``pre_tool_prose`` the non-streaming path preserves, for the SAME inline
    event sequence — so both transports feed the narration check identical
    pre-tool evidence."""
    from kestrel_sovereign.agent.streaming import _parse_stream_sentinels
    from kestrel_sovereign.llm.adapter import LLMResponse
    from kestrel_sovereign.llm.codex_adapter import CodexAdapter

    a = CodexAdapter()
    a._client = _CodexInlineFailedToolApp()
    events = [
        ev async for ev in a.get_streaming_response_with_tools(
            client="x", model="auto",
            messages=[{"role": "user", "content": "save my color"}],
            tools=[{"type": "function", "function": {
                "name": _INLINE_TOOL_NAME, "description": "d",
                "parameters": {"type": "object"}}}],
            session_id="s-inline-stream", tool_executor=_inline_tool_executor,
        )
    ]
    # Everything the streaming client would render before the first tool marker.
    pre_tool_chunks = []
    saw_tool_call = False
    for ev in events:
        if isinstance(ev, ToolCallStarted):
            saw_tool_call = True
            break
        if isinstance(ev, str):
            pre_tool_chunks.append(ev)
    assert saw_tool_call, "streaming path never emitted ToolCallStarted"
    # Strip the wire tool sentinels the way StreamingMixin does before auditing.
    streaming_pre_tool = _parse_stream_sentinels("".join(pre_tool_chunks))[0]
    # The streaming final envelope also carries the inline execution record.
    final = [ev for ev in events if isinstance(ev, LLMResponse)][-1]
    assert getattr(final, "executed_tool_calls", None)

    ns_resp = await _drive_real_codex_inline_response()
    assert (
        streaming_pre_tool
        == ns_resp.pre_tool_prose
        == _INLINE_PRE_TOOL_PROSE
    )


@pytest.mark.asyncio
async def test_inline_codex_tool_turn_narration_caught_on_nonstreaming(monkeypatch):
    """End-to-end #2675 P1: a real inline Codex turn whose pre-tool prose lies
    ('Saved your color.') about a tool that FAILED is caught by the
    deterministic narration check on the NON-streaming path — the same
    violation streaming catches. Before the fix the assembly passed
    ``pre_tool_prose=None`` for inline turns and the check silently no-op'd."""
    from kestrel_sovereign.features.response_audit.hook import ResponseAuditHook

    initial = await _drive_real_codex_inline_response()

    async def orchestrator(*, response, tool_results=None, **kw):
        # Fold the adapter's inline-executed calls into audit tool_results
        # exactly as ``_append_executed_tool_breadcrumbs`` does on the live path.
        if tool_results is not None:
            _fold_inline_executed_into_tool_results(response, tool_results)
        return response.content or ""

    hook_input = await _run_nonstreaming_turn(
        _CapturingHooks(), monkeypatch,
        initial_response=initial, orchestrator=orchestrator,
    )

    assert hook_input is not None
    # The assembly fed the narration check the marker-bound pre-tool prose,
    # NOT the full pre+post-tool content the model ultimately emitted.
    assert hook_input.pre_tool_prose == _INLINE_PRE_TOOL_PROSE
    assert hook_input.pre_tool_prose != initial.content
    # ``tool_calls`` derived from the inline envelopes; results carry the failure.
    assert hook_input.tool_calls == [
        {"id": _INLINE_TOOL_ID, "name": _INLINE_TOOL_NAME,
         "arguments": _INLINE_TOOL_ARGS}
    ]
    assert (
        hook_input.tool_results
        and hook_input.tool_results[0]["result"] == {
            "success": False, "error": "no store configured",
        }
    )

    # A real ResponseAuditHook reaches the dishonest-success verdict on the
    # deterministic-only path (LLM audit forced unavailable).
    audit_agent = MagicMock()
    audit_agent.llm_service = MagicMock()
    audit_agent.llm_service.get_audit_response = AsyncMock(
        side_effect=RuntimeError("audit provider down")
    )
    audit_agent.features = {}
    hook = ResponseAuditHook(agent=audit_agent, mode="warn", risk_threshold=3)
    out = await hook.execute(hook_input)

    assert hook.last_narration_verdict.risk_boost == 2
    assert hook.last_narration_verdict.offending_verb == "saved"
    assert hook.last_narration_verdict.offending_tool == _INLINE_TOOL_NAME
    # warn-mode MODIFY annotation, not a silent pass-through.
    assert out.updated_input is not None
    assert "narrated completion before" in out.updated_input["response_text"]
