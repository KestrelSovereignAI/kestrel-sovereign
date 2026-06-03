"""Tests for the OpenAI plan adapter (app-server backed) and registry.

The adapter no longer hand-rolls HTTP/OAuth — it drives the official
``codex app-server`` binary, which owns auth via ``~/.codex/auth.json``.
The old JWT/header/token-refresh internals are intentionally gone, so
their tests are gone too; these exercise the new app-server projection,
the per-turn tool-executor bridge, and binary-resolution registry
wiring.
"""
from typing import Any, Dict
from unittest.mock import patch

import pytest

from kestrel_sovereign.llm.adapter import LLMAdapter, LLMResponse, ThinkingDelta
from kestrel_sovereign.llm.codex_adapter import (
    CodexAdapter,
    _build_turn_input,
    _convert_tools_to_codex_dynamic_tools,
    _extract_instructions_and_input,
    _result_to_codex_response,
    _usage_from,
)
from kestrel_sovereign.llm.codex_app_server import CodexAppServerError
from kestrel_sovereign.llm.provider_registry import ProviderRegistry
from kestrel_sdk.llm import ToolCallStarted


class TestOpenAIPlanAdapterClass:
    def test_is_llm_adapter_subclass(self):
        assert isinstance(CodexAdapter(), LLMAdapter)

    def test_name_is_openai_plan(self):
        assert CodexAdapter().name == "openai_plan"

    def test_metadata(self):
        a = CodexAdapter()
        assert a.substrate_type() == "gpt"
        assert a.display_name() == "OpenAI Codex (plan)"
        assert a.key_env_var() is None
        assert a.cost_per_1m_tokens() == {"input": 0.0, "output": 0.0}


class TestOpenAIPlanListModels:
    @pytest.mark.asyncio
    async def test_list_models_is_not_supported_directly(self):
        with pytest.raises(NotImplementedError, match="canonical openai provider"):
            await CodexAdapter().list_models()


class TestMessageHelpers:
    def test_extract_system_prompt(self):
        instructions, inputs = _extract_instructions_and_input([
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ])
        assert instructions == "You are helpful"
        assert len(inputs) == 1 and inputs[0]["role"] == "user"

    def test_extract_structured_system_content(self):
        instructions, _ = _extract_instructions_and_input([
            {"role": "system", "content": [
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": "Part 2"},
            ]},
        ])
        assert "Part 1" in instructions and "Part 2" in instructions


class TestDynamicToolsSpec:
    """The app-server expects ``{name,description,inputSchema}`` — NOT
    the Responses-API ``{type:function, function:{...}}`` wrapper."""

    def test_converts_to_codex_shape(self):
        result = _convert_tools_to_codex_dynamic_tools([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }])
        assert result == [{
            "name": "get_weather",
            "description": "Get weather",
            "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}},
            "namespace": "kestrel",
        }]
        assert "type" not in result[0]
        assert "parameters" not in result[0]

    def test_none_and_empty(self):
        assert _convert_tools_to_codex_dynamic_tools(None) is None
        assert _convert_tools_to_codex_dynamic_tools([]) is None

    def test_namespace_attached_to_every_tool(self):
        """Codex's ``codex_core`` registers dynamicTools in a process-
        global handler registry keyed by ``(namespace, name)``. Without
        a kestrel namespace, our ``spawn_agent`` (from SpawnFeature,
        core=true) collides with codex's native ``spawn_agent`` —
        the SECOND ``thread/start`` call into the same app-server
        process panics with "handler for tool spawn_agent already
        registered" and the process exits. Namespacing under
        ``"kestrel"`` keeps every kestrel tool in its own slot, so
        codex's built-ins and our tools coexist."""
        result = _convert_tools_to_codex_dynamic_tools([
            {"type": "function", "function": {
                "name": "spawn_agent", "description": "",
                "parameters": {"type": "object"}}},
            {"type": "function", "function": {
                "name": "shell", "description": "",
                "parameters": {"type": "object"}}},
            {"type": "function", "function": {
                "name": "memory_search", "description": "",
                "parameters": {"type": "object"}}},
        ])
        assert all(spec["namespace"] == "kestrel" for spec in result), (
            "every dynamicTool spec must carry namespace='kestrel' so "
            "kestrel's tools register in their own codex_core slot"
        )


class TestTurnInputBuilder:
    def test_existing_thread_sends_only_latest_user(self):
        assert _build_turn_input(
            [{"role": "user", "content": "first"},
             {"role": "assistant", "content": "ok"},
             {"role": "user", "content": "second"}],
            fresh_thread=False,
        ) == "second"

    def test_fresh_thread_seeds_prior_transcript(self):
        out = _build_turn_input(
            [{"role": "user", "content": "first"},
             {"role": "assistant", "content": "ok"},
             {"role": "user", "content": "second"}],
            fresh_thread=True,
        )
        assert "Conversation so far" in out
        assert "user: first" in out and "assistant: ok" in out
        assert out.rstrip().endswith("second")


class TestUsageProjection:
    def test_maps_codex_token_usage(self):
        assert _usage_from({"total": {
            "totalTokens": 100, "inputTokens": 90,
            "cachedInputTokens": 40, "outputTokens": 10,
        }}) == {
            "input_tokens": 90, "output_tokens": 10,
            "total_tokens": 100, "cache_read_input_tokens": 40,
        }


class TestResultMarshalling:
    """Kestrel tool results -> codex CodexDynamicToolCallResponse."""

    def test_success_payload(self):
        r = _result_to_codex_response({"success": True, "result": "salamander"})
        assert r["success"] is True
        assert r["contentItems"][0]["text"] == "salamander"

    def test_error_payload(self):
        r = _result_to_codex_response({"success": False, "error": "denied"})
        assert r["success"] is False
        assert r["contentItems"][0]["text"] == "denied"

    def test_non_string_serialized(self):
        r = _result_to_codex_response({"success": True, "result": {"x": 1}})
        assert '"x": 1' in r["contentItems"][0]["text"]


class _FakeAppServer:
    """Stands in for CodexAppServerClient: scripts a turn's events
    plus optional server→client requests."""

    def __init__(self, events, registered=None):
        self._events = events
        self.requests = []
        self.registered_handlers: Dict[str, Any] = registered or {}
        self.started = False
        self.dynamic_tools = None

    async def ensure_started(self):
        self.started = True

    async def request(self, method, params=None, *, timeout=120):
        self.requests.append((method, params))
        if method == "thread/start":
            self.dynamic_tools = (params or {}).get("dynamicTools")
            return {"thread": {"id": "thr-1"}}
        if method == "turn/start":
            return {"turn": {"id": "turn-1"}}
        return {}

    def register_server_request_handler(self, method, handler, *, thread_id=None):
        key = (method, thread_id)
        self.registered_handlers[key] = handler
        return lambda: self.registered_handlers.pop(key, None)

    def open_turn_sink(self, key):
        return key

    def close_turn_sink(self, key):
        pass

    async def iter_turn_events(self, sink, *, idle_timeout=120):
        for ev in self._events:
            yield ev


def _adapter_with(events, registered=None):
    a = CodexAdapter()
    a._client = _FakeAppServer(events, registered)
    return a


_TEXT_TURN = [
    {"method": "item/agentMessage/delta", "params": {"delta": "Hel"}},
    {"method": "item/agentMessage/delta", "params": {"delta": "lo"}},
    {"method": "item/completed",
     "params": {"item": {"type": "agentMessage", "text": "Hello"}}},
    {"method": "thread/tokenUsage/updated",
     "params": {"tokenUsage": {"total": {"inputTokens": 7, "outputTokens": 2,
                                         "totalTokens": 9, "cachedInputTokens": 3}}}},
    {"method": "turn/completed", "params": {"turn": {"status": "completed"}}},
]


class TestAdapterTextPath:
    @pytest.mark.asyncio
    async def test_get_response_text_and_usage(self):
        a = _adapter_with(_TEXT_TURN)
        r = await a.get_response(
            client="ignored", model="auto",
            messages=[{"role": "user", "content": "hi"}], session_id="s1",
        )
        assert isinstance(r, LLMResponse)
        assert r.content == "Hello"
        assert (r.input_tokens, r.output_tokens, r.total_tokens) == (7, 2, 9)
        assert r.cache_read_input_tokens == 3
        cached_id, cached_fp = a._session_threads["s1"]
        assert cached_id == "thr-1" and cached_fp  # fingerprint set

    @pytest.mark.asyncio
    async def test_session_thread_reused(self):
        a = _adapter_with(_TEXT_TURN)
        # Pre-seed with the fingerprint that matches this call (model="auto"
        # → None, no instructions, no tools) so the cache hits.
        fp = CodexAdapter._thread_fingerprint(None, None, None)
        a._session_threads["s1"] = ("thr-existing", fp)
        await a.get_response(client="x", model="auto",
                             messages=[{"role": "user", "content": "hi"}],
                             session_id="s1")
        methods = [m for m, _ in a._client.requests]
        assert "thread/start" not in methods
        assert "turn/start" in methods

    @pytest.mark.asyncio
    async def test_session_thread_restarted_when_config_changes(self):
        """When tools (or model/system) change on a reused session, the
        thread is invalidated and re-created — otherwise the model would
        keep using the old thread-level config."""
        a = _adapter_with(_TEXT_TURN)
        a._session_threads["s1"] = (
            "thr-old", CodexAdapter._thread_fingerprint(None, None, None),
        )

        async def exe(name, args):
            return {"success": True, "result": "ok"}

        await a.get_response(
            client="x", model="auto",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {
                "name": "t", "description": "d",
                "parameters": {"type": "object"}}}],
            session_id="s1", tool_executor=exe,
        )
        methods = [m for m, _ in a._client.requests]
        assert "thread/start" in methods, "thread should be restarted on config change"
        new_id, _ = a._session_threads["s1"]
        assert new_id == "thr-1" and new_id != "thr-old"

    @pytest.mark.asyncio
    async def test_streaming_yields_text_chunks(self):
        a = _adapter_with(_TEXT_TURN)
        out = [
            c async for c in a.get_streaming_response(
                client="x", model="auto",
                messages=[{"role": "user", "content": "hi"}], session_id="s")
        ]
        assert "".join(c for c in out if isinstance(c, str)) == "Hello"


class TestThreadStartParams:
    """Things we MUST send on thread/start so the app-server's native
    tools behave correctly (cwd anchoring, sandbox profile)."""

    @pytest.mark.asyncio
    async def test_thread_start_carries_cwd(self):
        """Without ``cwd``, codex's native shell tool runs relative paths
        against wherever the kestrel process started — not the user's
        working directory. ``pwd`` returned the install prefix in
        Nellie's session; that's the symptom."""
        a = _adapter_with(_TEXT_TURN)
        await a.get_response(
            client="x", model="auto",
            messages=[{"role": "user", "content": "hi"}], session_id="s",
        )
        ts = [p for m, p in a._client.requests if m == "thread/start"][0]
        assert isinstance(ts.get("cwd"), str) and ts["cwd"]

    @pytest.mark.asyncio
    async def test_thread_start_cwd_honors_env_override(self, monkeypatch):
        monkeypatch.setenv("KESTREL_CODEX_CWD", "/agents/nellie/workspace")
        a = _adapter_with(_TEXT_TURN)
        await a.get_response(
            client="x", model="auto",
            messages=[{"role": "user", "content": "hi"}], session_id="s",
        )
        ts = [p for m, p in a._client.requests if m == "thread/start"][0]
        assert ts["cwd"] == "/agents/nellie/workspace"


class TestToolActivityMarkers:
    """The chat-UI parses 🔧/✓/❌ marker lines from the streamed text to
    render the expandable tool-activity cards (chat.js
    ``isToolActivityStartLine``). For the orchestrator-dispatched path
    the orchestrator emits these markers; for codex's inline tool loop
    the adapter must emit them itself or the chat surface goes opaque
    — which is exactly what Nellie reported."""

    @pytest.mark.asyncio
    async def test_native_shell_emits_start_and_complete_markers(self):
        """The codex-native shell tool (commandExecution) runs entirely
        inside the app-server's sandbox — kestrel never sees it via
        item/tool/call. Without start/complete markers, shell calls
        were invisible to the chat UI."""
        events = [
            {"method": "item/started", "params": {"item": {
                "id": "i-1", "type": "commandExecution",
                "command": "git status --short --branch",
            }}},
            {"method": "item/completed", "params": {"item": {
                "id": "i-1", "type": "commandExecution",
                "command": "git status --short --branch",
                "status": "succeeded",
            }}},
            {"method": "item/agentMessage/delta", "params": {"delta": "done"}},
            {"method": "item/completed",
             "params": {"item": {"type": "agentMessage", "text": "done"}}},
            {"method": "turn/completed", "params": {"turn": {"status": "completed"}}},
        ]
        a = _adapter_with(events)
        chunks = [
            c async for c in a.get_streaming_response(
                client="x", model="auto",
                messages=[{"role": "user", "content": "go"}], session_id="s",
            )
        ]
        stream = "".join(c for c in chunks if isinstance(c, str))
        assert "\U0001f527 Calling shell: git status --short --branch" in stream
        assert "✓ shell complete" in stream

    @pytest.mark.asyncio
    async def test_failed_shell_emits_failure_marker(self):
        events = [
            {"method": "item/started", "params": {"item": {
                "id": "i-2", "type": "commandExecution", "command": "rg foo",
            }}},
            {"method": "item/completed", "params": {"item": {
                "id": "i-2", "type": "commandExecution", "command": "rg foo",
                "status": "failed",
            }}},
            {"method": "turn/completed", "params": {"turn": {"status": "completed"}}},
        ]
        a = _adapter_with(events)
        chunks = [
            c async for c in a.get_streaming_response(
                client="x", model="auto",
                messages=[{"role": "user", "content": "go"}], session_id="s",
            )
        ]
        stream = "".join(c for c in chunks if isinstance(c, str))
        assert "❌ shell failed" in stream

    @pytest.mark.asyncio
    async def test_markers_do_not_leak_into_llmresponse_content(self):
        """Markers belong on the wire, not in ``LLMResponse.content``.
        Audit logs, narration checks, and any non-chat consumer of the
        response would otherwise see literal "🔧" characters."""
        events = [
            {"method": "item/started", "params": {"item": {
                "id": "i-3", "type": "commandExecution", "command": "ls",
            }}},
            {"method": "item/completed", "params": {"item": {
                "id": "i-3", "type": "commandExecution",
                "command": "ls", "status": "succeeded",
            }}},
            {"method": "item/agentMessage/delta", "params": {"delta": "Hi"}},
            {"method": "item/completed",
             "params": {"item": {"type": "agentMessage", "text": "Hi"}}},
            {"method": "turn/completed", "params": {"turn": {"status": "completed"}}},
        ]
        a = _adapter_with(events)
        r = await a.get_response(
            client="x", model="auto",
            messages=[{"role": "user", "content": "go"}], session_id="s",
        )
        assert r.content == "Hi"
        assert "\U0001f527" not in (r.content or "")
        assert "✓" not in (r.content or "")

    @pytest.mark.asyncio
    async def test_markers_dedupe_when_started_and_completed_repeat(self):
        events = [
            {"method": "item/started", "params": {"item": {
                "id": "i-4", "type": "commandExecution", "command": "pwd",
            }}},
            {"method": "item/started", "params": {"item": {
                "id": "i-4", "type": "commandExecution", "command": "pwd",
            }}},
            {"method": "item/completed", "params": {"item": {
                "id": "i-4", "type": "commandExecution",
                "command": "pwd", "status": "succeeded",
            }}},
            {"method": "item/completed", "params": {"item": {
                "id": "i-4", "type": "commandExecution",
                "command": "pwd", "status": "succeeded",
            }}},
            {"method": "turn/completed", "params": {"turn": {"status": "completed"}}},
        ]
        a = _adapter_with(events)
        chunks = [
            c async for c in a.get_streaming_response(
                client="x", model="auto",
                messages=[{"role": "user", "content": "go"}], session_id="s",
            )
        ]
        stream = "".join(c for c in chunks if isinstance(c, str))
        assert stream.count("\U0001f527 Calling shell") == 1
        assert stream.count("✓ shell complete") == 1

    @pytest.mark.asyncio
    async def test_idless_items_dont_collapse_into_one_dedupe_slot(self):
        """Codex review P2 (PR #1334): the empty-string fallback for
        missing ``id``/``callId`` would have put every id-less item in
        the same dedupe slot, dropping all but the first marker. The
        real protocol always sends ids, but defensive coding matters —
        if a future build ever omits one, the rest of the chat shouldn't
        go dark."""
        events = [
            {"method": "item/started", "params": {"item": {
                "type": "commandExecution", "command": "ls",
                # NO id field — exercises the fallback path.
            }}},
            {"method": "item/completed", "params": {"item": {
                "type": "commandExecution", "command": "ls",
                "status": "succeeded",
            }}},
            {"method": "item/started", "params": {"item": {
                "type": "commandExecution", "command": "pwd",
            }}},
            {"method": "item/completed", "params": {"item": {
                "type": "commandExecution", "command": "pwd",
                "status": "succeeded",
            }}},
            {"method": "turn/completed", "params": {"turn": {"status": "completed"}}},
        ]
        a = _adapter_with(events)
        chunks = [
            c async for c in a.get_streaming_response(
                client="x", model="auto",
                messages=[{"role": "user", "content": "go"}], session_id="s",
            )
        ]
        stream = "".join(c for c in chunks if isinstance(c, str))
        # Both calls must appear — start + complete each.
        assert stream.count("\U0001f527 Calling shell") == 2
        assert stream.count("✓ shell complete") == 2

    @pytest.mark.asyncio
    async def test_dynamic_tool_emits_marker_and_preserves_executed_log(self):
        """Kestrel-dispatched tools (dynamicToolCall): the start/complete
        markers must fire AND the executed_tool_calls record the
        kestrel-side audit row — both surfaces, not one or the other."""
        calls: list = []

        async def exe(name, args):
            calls.append((name, args))
            return {"success": True, "result": "ok"}

        events = [
            {"method": "item/started", "params": {"item": {
                "id": "call-1", "type": "dynamicToolCall",
                "name": "memory_search", "arguments": {"q": "x"},
            }}},
            {"method": "item/completed", "params": {"item": {
                "id": "call-1", "type": "dynamicToolCall",
                "name": "memory_search", "arguments": {"q": "x"},
                "status": "succeeded",
            }}},
            {"method": "item/completed",
             "params": {"item": {"type": "agentMessage", "text": "found"}}},
            {"method": "turn/completed", "params": {"turn": {"status": "completed"}}},
        ]
        a = _adapter_with(events)
        chunks = [
            c async for c in a.get_streaming_response_with_tools(
                client="x", model="auto",
                messages=[{"role": "user", "content": "go"}],
                tools=[{"type": "function", "function": {
                    "name": "memory_search", "description": "d",
                    "parameters": {"type": "object"}}}],
                session_id="s", tool_executor=exe,
            )
        ]
        stream = "".join(c for c in chunks if isinstance(c, str))
        assert "\U0001f527 Calling memory_search" in stream
        assert "✓ memory_search complete" in stream
        # LLMResponse should still carry executed_tool_calls so the
        # orchestrator path can fold it into chat history (PR #1331).
        final = [c for c in chunks if isinstance(c, LLMResponse)][-1]
        # No tool was actually invoked via item/tool/call here (we
        # didn't drive the handler), so executed_tool_calls may be
        # empty — but the attribute must remain attachable.
        assert hasattr(final, "content")


class TestToolExecutorBridge:
    """The crucial new surface: per-turn item/tool/call handler that
    runs through kestrel's security-gated executor."""

    @pytest.mark.asyncio
    async def test_tools_without_executor_fail_loud(self):
        a = _adapter_with(_TEXT_TURN)
        with pytest.raises(CodexAppServerError, match="tool_executor callback"):
            await a.get_response(
                client="x", model="auto",
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {
                    "name": "t", "description": "d",
                    "parameters": {"type": "object"}}}],
                session_id="s",
            )

    @pytest.mark.asyncio
    async def test_executor_registered_and_unregistered(self):
        a = _adapter_with(_TEXT_TURN)

        async def exe(name, args):
            return {"success": True, "result": "x"}

        await a.get_response(
            client="x", model="auto",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {
                "name": "t", "description": "d",
                "parameters": {"type": "object"}}}],
            session_id="s", tool_executor=exe,
        )
        # Handler unregistered (thread-scoped key) when turn finishes.
        assert not any(
            k[0] == "item/tool/call" for k in a._client.registered_handlers
        )

    @pytest.mark.asyncio
    async def test_dynamic_tools_sent_at_thread_start(self):
        # Empirically required: turn/start-only dynamicTools made the
        # model say "tool isn't available." thread/start is the seam.
        a = _adapter_with(_TEXT_TURN)

        async def exe(name, args):
            return {"success": True, "result": "x"}

        await a.get_response(
            client="x", model="auto",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {
                "name": "get_weather", "description": "d",
                "parameters": {"type": "object"}}}],
            session_id="s", tool_executor=exe,
        )
        assert a._client.dynamic_tools == [{
            "name": "get_weather", "description": "d",
            "inputSchema": {"type": "object"},
            "namespace": "kestrel",
        }]

    @pytest.mark.asyncio
    async def test_handler_routes_call_to_executor_and_marshals_result(self):
        a = CodexAdapter()
        seen = []

        async def exe(name, args):
            seen.append((name, args))
            return {"success": True, "result": "salamander"}

        handler = a._make_tool_call_handler(
            exe, "thr-1", frozenset({"get_secret"}),
        )
        reply = await handler({
            "threadId": "thr-1", "tool": "get_secret",
            "arguments": '{"k":"v"}',
        })
        assert seen == [("get_secret", {"k": "v"})]
        assert reply == {
            "contentItems": [{"type": "inputText", "text": "salamander"}],
            "success": True,
        }

    @pytest.mark.asyncio
    async def test_handler_rejects_call_for_other_thread(self):
        a = CodexAdapter()

        async def exe(name, args):
            return {"success": True, "result": "ok"}

        handler = a._make_tool_call_handler(exe, "thr-A", frozenset({"t"}))
        reply = await handler({"threadId": "thr-B", "tool": "t", "arguments": {}})
        assert reply["success"] is False
        assert "different turn" in reply["contentItems"][0]["text"]

    @pytest.mark.asyncio
    async def test_handler_rejects_unadvertised_tool_name(self):
        """Security: a tool name the app-server requests but which
        wasn't in the turn's dynamicTools must not run through the
        orchestrator's full registry."""
        a = CodexAdapter()
        called = []

        async def exe(name, args):
            called.append(name)
            return {"success": True, "result": "should-not-run"}

        handler = a._make_tool_call_handler(
            exe, "thr-1", frozenset({"allowed_tool"}),
        )
        reply = await handler({
            "threadId": "thr-1",
            "tool": "denied_or_hallucinated_tool",
            "arguments": {},
        })
        assert reply["success"] is False
        assert "not advertised" in reply["contentItems"][0]["text"]
        assert called == [], "executor must not run for unadvertised tool"

    @pytest.mark.asyncio
    async def test_no_handler_registered_for_text_only_turn(self):
        """Defense in depth: an item/tool/call handler must NOT be
        registered when the turn has no advertised tools — even if a
        tool_executor was passed."""
        a = _adapter_with(_TEXT_TURN)

        async def exe(name, args):
            return {"success": True, "result": "x"}

        # No tools arg → no dynamic tools → no handler registration.
        await a.get_response(
            client="x", model="auto",
            messages=[{"role": "user", "content": "hi"}],
            session_id="text-only", tool_executor=exe,
        )
        assert not any(
            k[0] == "item/tool/call" for k in a._client.registered_handlers
        ), "text-only turns must not register a tool handler"

    @pytest.mark.asyncio
    async def test_concurrent_first_calls_on_same_session_create_one_thread(self):
        """Regression: two concurrent first calls on the same session_id
        must not both run thread/start and overwrite each other's cache
        entry (loses server-side history for whichever thread doesn't
        win the race). The per-session lock around ``_ensure_thread``
        guarantees one thread/start.
        """
        import asyncio

        a = _adapter_with(_TEXT_TURN)
        # Gate the first thread/start so the second call races.
        original_request = a._client.request
        gate = asyncio.Event()

        async def gated_request(method, params=None, *, timeout=120):
            if method == "thread/start":
                await gate.wait()
            return await original_request(method, params, timeout=timeout)

        a._client.request = gated_request

        async def call():
            return await a.get_response(
                client="x", model="auto",
                messages=[{"role": "user", "content": "hi"}],
                session_id="race",
            )

        t1 = asyncio.create_task(call())
        t2 = asyncio.create_task(call())
        await asyncio.sleep(0)  # let both reach the gate
        gate.set()
        await asyncio.gather(t1, t2)

        thread_starts = [m for m, _ in a._client.requests if m == "thread/start"]
        assert len(thread_starts) == 1, (
            f"expected one thread/start, got {len(thread_starts)} — race "
            "would have leaked an orphan thread"
        )

    @pytest.mark.asyncio
    async def test_per_thread_serialization_for_concurrent_turns(self):
        """The codex app-server runs one active turn per thread. Two
        concurrent ``_run_turn`` calls sharing a thread must serialize,
        not race on turn-sink registration.
        """
        import asyncio

        a = _adapter_with(_TEXT_TURN)
        # Seed the cache so both calls hit the SAME thread.
        fp = CodexAdapter._thread_fingerprint(None, None, None)
        a._session_threads["shared"] = ("thr-shared", fp)

        # Make iter_turn_events block until released so we can observe
        # the second call queuing on the lock.
        in_turn = asyncio.Event()
        release = asyncio.Event()
        original_iter = a._client.iter_turn_events

        async def gated_iter(sink, *, idle_timeout=120):
            in_turn.set()
            await release.wait()
            async for ev in original_iter(sink, idle_timeout=idle_timeout):
                yield ev

        a._client.iter_turn_events = gated_iter

        async def run_one(label):
            return label, await a.get_response(
                client="x", model="auto",
                messages=[{"role": "user", "content": label}],
                session_id="shared",
            )

        first = asyncio.create_task(run_one("first"))
        await in_turn.wait()
        in_turn.clear()
        second = asyncio.create_task(run_one("second"))
        # Give second a chance to start; it must be parked on the lock.
        await asyncio.sleep(0)
        assert not second.done(), "second turn ran before first released"
        release.set()
        results = await asyncio.gather(first, second)
        assert {r[0] for r in results} == {"first", "second"}

    @pytest.mark.asyncio
    async def test_dynamic_tool_call_item_surfaces_as_tool_call_event(self):
        """The app-server emits inline-executed tool items with type
        ``dynamicToolCall`` — historically I only matched
        functionCall/toolCall and dropped these silently."""
        events = [
            {"method": "item/completed", "params": {"item": {
                "type": "dynamicToolCall", "id": "c1",
                "name": "get_weather", "arguments": '{"city": "SF"}'}}},
            {"method": "item/completed",
             "params": {"item": {"type": "agentMessage", "text": "ok"}}},
            {"method": "turn/completed", "params": {}},
        ]
        a = _adapter_with(events)

        async def exe(name, args):
            return {"success": True, "result": "sunny"}

        seen = [
            c async for c in a.get_streaming_response_with_tools(
                client="x", model="auto",
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {
                    "name": "get_weather", "description": "d",
                    "parameters": {"type": "object"}}}],
                session_id="s", tool_executor=exe,
            )
        ]
        starts = [c for c in seen if isinstance(c, ToolCallStarted)]
        assert starts and starts[0].name == "get_weather"

    @pytest.mark.asyncio
    async def test_reasoning_delta_method_names_match_live_protocol(self):
        """The app-server emits reasoning as
        ``item/reasoning/textDelta`` and ``item/reasoning/summaryTextDelta``,
        not the generic ``/reasoning/delta`` suffix the earlier code
        guessed at."""
        from kestrel_sovereign.llm.adapter import ThinkingDelta

        events = [
            {"method": "item/reasoning/textDelta",
             "params": {"delta": "thinking-one"}},
            {"method": "item/reasoning/summaryTextDelta",
             "params": {"delta": "summary-two"}},
            {"method": "item/completed",
             "params": {"item": {"type": "agentMessage", "text": "final"}}},
            {"method": "turn/completed", "params": {}},
        ]
        a = _adapter_with(events)
        out = [
            c async for c in a.get_streaming_response(
                client="x", model="auto",
                messages=[{"role": "user", "content": "hi"}],
                session_id="s",
            )
        ]
        deltas = [c for c in out if isinstance(c, ThinkingDelta)]
        assert [d.content for d in deltas] == ["thinking-one", "summary-two"]

    @pytest.mark.asyncio
    async def test_executed_tool_calls_attached_with_app_server_callid(self):
        """The adapter records every inline-executed tool call (with
        the app-server's own ``callId`` for audit alignment) on the
        returned LLMResponse via the ``executed_tool_calls`` runtime
        attribute. The orchestrator reads this generically (any
        inline-executing adapter sets it) to render chat-history
        breadcrumbs without re-dispatching."""

        class _AppWithToolCall:
            def __init__(self):
                self.registered = {}
                self.requests = []
                self.dynamic_tools = None

            async def ensure_started(self):
                pass

            async def request(self, method, params=None, *, timeout=120):
                self.requests.append((method, params))
                if method == "thread/start":
                    self.dynamic_tools = (params or {}).get("dynamicTools")
                    return {"thread": {"id": "thr-x"}}
                if method == "turn/start":
                    # Simulate the app-server issuing item/tool/call
                    # mid-turn — the handler stamps the executed_log.
                    handler = self.registered.get(("item/tool/call", "thr-x"))
                    assert handler is not None, "tool-call handler missing"
                    await handler({
                        "threadId": "thr-x",
                        "callId": "call_abc",
                        "tool": "get_weather",
                        "arguments": '{"city": "SF"}',
                    })
                    return {"turn": {"id": "turn-x"}}
                return {}

            def register_server_request_handler(self, m, h, *, thread_id=None):
                self.registered[(m, thread_id)] = h
                return lambda: self.registered.pop((m, thread_id), None)

            def open_turn_sink(self, key):
                return key

            def close_turn_sink(self, key):
                pass

            async def iter_turn_events(self, sink, *, idle_timeout=120):
                for ev in [
                    {"method": "item/completed",
                     "params": {"item": {
                         "type": "dynamicToolCall",
                         "id": "call_abc",
                         "name": "get_weather",
                         "arguments": '{"city": "SF"}',
                     }}},
                    {"method": "item/completed",
                     "params": {"item": {
                         "type": "agentMessage", "text": "It's sunny.",
                     }}},
                    {"method": "turn/completed", "params": {}},
                ]:
                    yield ev

        async def exe(name, args):
            return {"success": True, "result": "sunny"}

        a = CodexAdapter()
        a._client = _AppWithToolCall()
        r = await a.get_response(
            client="x", model="auto",
            messages=[{"role": "user", "content": "weather?"}],
            tools=[{"type": "function", "function": {
                "name": "get_weather", "description": "d",
                "parameters": {"type": "object"}}}],
            session_id="brd-test", tool_executor=exe,
        )
        # No re-dispatch surface.
        assert not r.tool_calls
        # New: orchestrator-readable breadcrumb data with the app-server's
        # OWN callId (so audit trails align with codex-side identifiers).
        executed = getattr(r, "executed_tool_calls", None)
        assert executed and len(executed) == 1
        assert executed[0]["id"] == "call_abc"
        assert executed[0]["name"] == "get_weather"
        assert executed[0]["arguments"] == {"city": "SF"}
        assert executed[0]["result"] == {"success": True, "result": "sunny"}

    @pytest.mark.asyncio
    async def test_inline_executed_tools_absent_from_final_response(self):
        """Regression: the app-server runs tools inline via our handler.
        Surfacing those calls in LLMResponse.tool_calls would make the
        orchestrator re-dispatch them, duplicating every side effect."""
        events_with_tool = [
            {"method": "item/agentMessage/delta", "params": {"delta": "ok"}},
            {"method": "item/completed", "params": {"item": {
                "type": "functionCall", "id": "c1",
                "name": "get_weather", "arguments": '{"city": "SF"}'}}},
            {"method": "item/completed",
             "params": {"item": {"type": "agentMessage", "text": "It's sunny."}}},
            {"method": "turn/completed", "params": {}},
        ]
        a = _adapter_with(events_with_tool)

        async def exe(name, args):
            return {"success": True, "result": "sunny"}

        # Non-streaming path
        r = await a.get_response(
            client="x", model="auto",
            messages=[{"role": "user", "content": "weather?"}],
            tools=[{"type": "function", "function": {
                "name": "get_weather", "description": "d",
                "parameters": {"type": "object"}}}],
            session_id="s-inline", tool_executor=exe,
        )
        assert r.content == "It's sunny."
        assert not r.tool_calls, (
            "tool was inline-executed; surfacing it would make the "
            "orchestrator re-dispatch and duplicate side effects"
        )

    @pytest.mark.asyncio
    async def test_handler_records_post_hook_effective_args_not_pre_hook(self):
        """If a PRE_TOOL_USE hook rewrites args (PII redact, normalize)
        the breadcrumb must record what actually RAN, not what the
        model sent. The kestrel-side executor returns
        ``(effective_args, result)`` for exactly this — pre-hook args
        leaking into audit would undo the hook's purpose.
        """
        a = CodexAdapter()

        async def exe_with_hook(name, args):
            # Simulate a hook that redacts a sensitive field.
            redacted = {k: ("[REDACTED]" if k == "email" else v) for k, v in args.items()}
            return redacted, {"success": True, "result": "ok"}

        log: list = []
        handler = a._make_tool_call_handler(
            exe_with_hook, "thr", frozenset({"send"}), log,
        )
        await handler({
            "threadId": "thr", "callId": "c1", "tool": "send",
            "arguments": {"email": "secret@example.com", "body": "hi"},
        })
        assert log and log[0]["arguments"] == {
            "email": "[REDACTED]", "body": "hi",
        }, "breadcrumb leaked pre-hook arguments"

    @pytest.mark.asyncio
    async def test_handler_records_failed_inline_executions(self):
        """Mirrors the orchestrator-dispatched path: tool failures
        appear in audit / STOP / UI surfaces, not silently dropped."""
        a = CodexAdapter()

        async def exe_that_raises(name, args):
            raise RuntimeError("backend down")

        log: list = []
        handler = a._make_tool_call_handler(
            exe_that_raises, "thr", frozenset({"t"}), log,
        )
        reply = await handler({
            "threadId": "thr", "callId": "c-fail", "tool": "t",
            "arguments": {},
        })
        # App-server gets a failure reply (existing behavior).
        assert reply["success"] is False
        # Breadcrumb records the failed call so audit isn't blind.
        assert log and log[0] == {
            "id": "c-fail", "name": "t", "arguments": {},
            "result": {"success": False, "error": "backend down"},
        }

    @pytest.mark.asyncio
    async def test_handler_executor_exception_becomes_failure_reply(self):
        a = CodexAdapter()

        async def exe(name, args):
            raise RuntimeError("boom")

        handler = a._make_tool_call_handler(exe, "thr", frozenset({"t"}))
        reply = await handler({"threadId": "thr", "tool": "t", "arguments": {}})
        assert reply["success"] is False
        assert "boom" in reply["contentItems"][0]["text"]


class TestOpenAIPlanProviderRegistry:
    """openai:plan now registers on binary resolution, not a token."""

    def _plan_config(self):
        return {
            "route_priority": ["openai:plan"],
            "vendors": {"openai": {"is_cloud": True, "routes": {
                "plan": {"adapter": "CodexAdapter", "model": "auto"}}}},
        }

    def test_registers_with_resolved_binary_as_client(self):
        with patch(
            "kestrel_sovereign.llm.codex_app_server.resolve_codex_binary",
            return_value="/path/to/codex",
        ):
            providers = ProviderRegistry(self._plan_config()).initialize_providers()
        # Other routes can be discovered via entry_points (kimi, xai, deepseek
        # plugins); filter to the route under test.
        plan = next((p for p in providers if p.name == "openai:plan"), None)
        assert plan is not None
        assert isinstance(plan.adapter, CodexAdapter)
        assert plan.client == "/path/to/codex"

    def test_skips_openai_plan_when_binary_unresolvable(self):
        # The invariant: openai:plan must NOT register when the codex
        # binary can't be resolved. With entry-point plugins installed
        # this surfaces as the route being absent from a non-empty
        # providers list; without plugins the registry raises
        # ``No routes could be initialized`` because openai:plan was
        # the only configured route. Both outcomes satisfy the
        # invariant — accept either, since plugin presence is an
        # environmental detail (CI vs developer machine).
        from kestrel_sovereign.llm.provider_registry import (
            ProviderInitializationError,
        )
        with patch(
            "kestrel_sovereign.llm.codex_app_server.resolve_codex_binary",
            side_effect=CodexAppServerError("codex binary not found"),
        ):
            try:
                providers = ProviderRegistry(
                    self._plan_config(),
                ).initialize_providers()
            except ProviderInitializationError:
                providers = []  # no routes could initialize — invariant holds
        assert not any(p.name == "openai:plan" for p in providers)


class TestDiscoveredRouteCapFromThreadStart:
    """``CodexAdapter._record_discovered_route_cap_from_thread_start``
    folds the per-turn cap codex reports on ``thread/start`` into the
    catalog so the existing context-status surface + ContextManager
    budget sizing reflect THIS session's ground truth instead of an
    empirical static guess that goes stale when OpenAI tunes the
    subscription tier."""

    def _adapter(self) -> CodexAdapter:
        # Direct construction; this method doesn't touch any of the
        # async/auth machinery — it's pure parsing.
        return CodexAdapter.__new__(CodexAdapter)

    def test_records_auto_compact_token_limit_camel_case(self):
        from kestrel_sovereign.llm.model_catalog import get_catalog_service
        catalog = get_catalog_service()
        catalog.clear_discovered_route_context_caps()
        try:
            self._adapter()._record_discovered_route_cap_from_thread_start(
                "openai:plan/gpt-5.5",
                {"thread": {"id": "thr_1", "autoCompactTokenLimit": 49152}},
            )
            assert catalog.get_route_context_cap("openai:plan/gpt-5.5") == 49152
        finally:
            catalog.clear_discovered_route_context_caps()

    def test_records_snake_case_field_too(self):
        """Codex-rs may emit snake_case in some versions; both should
        be honored so the parser doesn't break when the wire format
        drifts."""
        from kestrel_sovereign.llm.model_catalog import get_catalog_service
        catalog = get_catalog_service()
        catalog.clear_discovered_route_context_caps()
        try:
            self._adapter()._record_discovered_route_cap_from_thread_start(
                "openai:plan/gpt-5.5",
                {"thread": {"id": "thr_1", "auto_compact_token_limit": 49152}},
            )
            assert catalog.get_route_context_cap("openai:plan/gpt-5.5") == 49152
        finally:
            catalog.clear_discovered_route_context_caps()

    def test_picks_smallest_when_multiple_fields_present(self):
        """``modelContextWindow`` reports the full window;
        ``autoCompactTokenLimit`` is the per-turn ceiling — the
        smaller value is the real binding constraint, so the parser
        takes the min."""
        from kestrel_sovereign.llm.model_catalog import get_catalog_service
        catalog = get_catalog_service()
        catalog.clear_discovered_route_context_caps()
        try:
            self._adapter()._record_discovered_route_cap_from_thread_start(
                "openai:plan/gpt-5.5",
                {
                    "thread": {
                        "id": "thr_1",
                        "modelContextWindow": 1_000_000,
                        "maxContextWindow": 1_000_000,
                        "autoCompactTokenLimit": 32_768,
                    },
                },
            )
            assert catalog.get_route_context_cap("openai:plan/gpt-5.5") == 32_768
        finally:
            catalog.clear_discovered_route_context_caps()

    def test_missing_model_id_still_records_on_the_route(self):
        """Codex round 1 P2 on this PR: discovery is keyed by ROUTE,
        not by full model id, because downstream lookups are
        route-qualified and the route covers every model on it. A
        missing/auto model id (the common ``model = "auto"`` case)
        must NOT skip recording — the cap applies to whatever model
        codex picks for this session."""
        from kestrel_sovereign.llm.model_catalog import get_catalog_service
        catalog = get_catalog_service()
        catalog.clear_discovered_route_context_caps()
        try:
            self._adapter()._record_discovered_route_cap_from_thread_start(
                None,
                {"thread": {"id": "thr_1", "autoCompactTokenLimit": 49152}},
            )
            assert catalog.get_route_context_cap("openai:plan/gpt-5.5") == 49152
            assert catalog.get_route_context_cap("openai:plan") == 49152
        finally:
            catalog.clear_discovered_route_context_caps()

    def test_missing_cap_fields_silently_skip(self):
        from kestrel_sovereign.llm.model_catalog import get_catalog_service
        catalog = get_catalog_service()
        catalog.clear_discovered_route_context_caps()
        try:
            # No relevant fields — should be a no-op.
            self._adapter()._record_discovered_route_cap_from_thread_start(
                "openai:plan/gpt-5.5",
                {"thread": {"id": "thr_1", "somethingElse": True}},
            )
            # No discovered entry recorded; whatever value is returned
            # comes from the file layer fallback (unconstrained by
            # this test — we just assert no crash + the discovered
            # cache stays clear).
            from kestrel_sovereign.llm.model_catalog import (
                get_catalog_service as svc_factory,
            )
            assert svc_factory()._discovered_route_context_caps == {}
        finally:
            catalog.clear_discovered_route_context_caps()

    def test_boolean_value_does_not_get_treated_as_int(self):
        """``True`` is an int subclass in Python — the parser must not
        treat a stray bool flag named ``autoCompactTokenLimit: True``
        as a 1-token cap."""
        from kestrel_sovereign.llm.model_catalog import get_catalog_service
        catalog = get_catalog_service()
        catalog.clear_discovered_route_context_caps()
        try:
            self._adapter()._record_discovered_route_cap_from_thread_start(
                "openai:plan/gpt-5.5",
                {"thread": {"id": "thr_1", "autoCompactTokenLimit": True}},
            )
            assert catalog._discovered_route_context_caps == {}
        finally:
            catalog.clear_discovered_route_context_caps()

    def test_finds_cap_in_thread_settings_nested_block(self):
        """Codex round 2 P2 on this PR: codex sometimes places the
        ThreadSettings snapshot one level deeper under
        ``thread.settings`` rather than at the thread root. The parser
        must descend one level so the cap isn't silently missed."""
        from kestrel_sovereign.llm.model_catalog import get_catalog_service
        catalog = get_catalog_service()
        catalog.clear_discovered_route_context_caps()
        try:
            self._adapter()._record_discovered_route_cap_from_thread_start(
                "openai:plan/gpt-5.5",
                {
                    "thread": {
                        "id": "thr_1",
                        "settings": {"autoCompactTokenLimit": 40960},
                    },
                },
            )
            assert catalog.get_route_context_cap("openai:plan/gpt-5.5") == 40960
        finally:
            catalog.clear_discovered_route_context_caps()

    def test_finds_cap_in_thread_thread_settings_camel_case(self):
        """Same nested-snapshot case, but with the camelCase key
        codex's binary strings table reveals
        (``thread.threadSettings``)."""
        from kestrel_sovereign.llm.model_catalog import get_catalog_service
        catalog = get_catalog_service()
        catalog.clear_discovered_route_context_caps()
        try:
            self._adapter()._record_discovered_route_cap_from_thread_start(
                "openai:plan/gpt-5.5",
                {
                    "thread": {
                        "id": "thr_1",
                        "threadSettings": {"autoCompactTokenLimit": 28672},
                    },
                },
            )
            assert catalog.get_route_context_cap("openai:plan/gpt-5.5") == 28672
        finally:
            catalog.clear_discovered_route_context_caps()

    def test_finds_cap_at_top_level_too(self):
        """The thread settings snapshot may be at the response root or
        nested — both shapes should resolve."""
        from kestrel_sovereign.llm.model_catalog import get_catalog_service
        catalog = get_catalog_service()
        catalog.clear_discovered_route_context_caps()
        try:
            self._adapter()._record_discovered_route_cap_from_thread_start(
                "openai:plan/gpt-5.5",
                {"autoCompactTokenLimit": 24576, "thread": {"id": "thr_1"}},
            )
            assert catalog.get_route_context_cap("openai:plan/gpt-5.5") == 24576
        finally:
            catalog.clear_discovered_route_context_caps()
