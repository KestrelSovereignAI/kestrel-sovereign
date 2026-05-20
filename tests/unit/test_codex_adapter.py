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
        }]
        assert "type" not in result[0]
        assert "parameters" not in result[0]

    def test_none_and_empty(self):
        assert _convert_tools_to_codex_dynamic_tools(None) is None
        assert _convert_tools_to_codex_dynamic_tools([]) is None


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
        }]

    @pytest.mark.asyncio
    async def test_handler_routes_call_to_executor_and_marshals_result(self):
        a = CodexAdapter()
        seen = []

        async def exe(name, args):
            seen.append((name, args))
            return {"success": True, "result": "salamander"}

        handler = a._make_tool_call_handler(exe, "thr-1")
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

        handler = a._make_tool_call_handler(exe, "thr-A")
        reply = await handler({"threadId": "thr-B", "tool": "t", "arguments": {}})
        assert reply["success"] is False
        assert "different turn" in reply["contentItems"][0]["text"]

    @pytest.mark.asyncio
    async def test_handler_executor_exception_becomes_failure_reply(self):
        a = CodexAdapter()

        async def exe(name, args):
            raise RuntimeError("boom")

        handler = a._make_tool_call_handler(exe, "thr")
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
        # Other routes (entry_point plugins) may still register; the
        # openai:plan route itself must be absent. The registry only
        # raises "No routes could be initialized" when EVERY route fails.
        with patch(
            "kestrel_sovereign.llm.codex_app_server.resolve_codex_binary",
            side_effect=CodexAppServerError("codex binary not found"),
        ):
            providers = ProviderRegistry(self._plan_config()).initialize_providers()
        assert not any(p.name == "openai:plan" for p in providers)
