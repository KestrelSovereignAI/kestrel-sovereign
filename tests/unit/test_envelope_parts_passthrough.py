"""First-class typed-part passthrough on subagent dispatch (#2641).

``emit_part`` delivery used to depend on the per-turn ContextVar collector
propagating into wherever the tool actually ran. That contract silently broke
on the subagent-dispatch path when the transport executed the tool on a task
whose frozen context predated the turn (the codex app-server reader spawns
each ``item/tool/call`` handler that way) — the Frinz ``/dashboard.html``
selfie flow logged ``[SUBAGENT-TOOL] generate_selfie result …`` end-to-end but
the ``selfie_finished`` typed part never reached the outbound stream.

#2641 makes delivery an envelope contract instead:

- ``emit_part`` with no turn collector buffers onto the tool result under
  construction (bound by ``DynamicTool.execute``); the wrapper attaches the
  buffered parts to the serialized envelope as its ``parts`` field.
- Explicit ``ToolResult.parts`` (SDK forward-compat, read via ``getattr``)
  rides the same field.
- ``execute_as_subagent`` returns ``{success, result, parts: [...]}``.
- The orchestrator's dispatch sites re-emit envelope-carried parts into the
  parent turn's collector, so the PART sentinels land exactly where they
  always did — right after the dispatching tool.

These tests pin every hop, ending with the acceptance passthrough: a
subagent-emitted ``selfie_finished`` part reaching the outbound stream through
the REAL dispatch chain.
"""
from __future__ import annotations

import asyncio
import contextvars
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sdk.hooks.base import PermissionDecision
from kestrel_sdk.llm.response import LLMResponse, ToolCall
from kestrel_sdk.tools.result import ToolResult

from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.agent.parts import (
    MAX_PART_PAYLOAD_BYTES,
    build_part_sentinel,
    drain_parts,
    emit_part,
    part_collector,
    sanitize_part,
    tool_result_parts_buffer,
)
from kestrel_sovereign.agent.streaming import _parse_stream_sentinels
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sdk.tools.base import ToolCategory


# --------------------------------------------------------------------------
# emit_part fallback: tool-result-under-construction buffer
# --------------------------------------------------------------------------

def test_emit_part_falls_back_to_tool_result_buffer():
    # No turn collector bound: the part must land on the bound tool-result
    # buffer instead of being dropped (pre-#2641 it returned False here).
    with tool_result_parts_buffer() as buf:
        assert emit_part("selfie_finished", {"url": "u"}, part_id="s1") is True
        assert buf == [{"type": "selfie_finished", "data": {"url": "u"}, "id": "s1"}]
    # Outside both scopes: still a no-op, never an error.
    assert emit_part("selfie_finished", {"url": "u"}) is False


def test_emit_part_prefers_turn_collector_over_buffer():
    # 100% back-compat: when the turn collector IS bound, it wins and the
    # envelope buffer stays empty — no double delivery is possible.
    with part_collector():
        with tool_result_parts_buffer() as buf:
            assert emit_part("todo", {"t": 1}) is True
            assert buf == []
        assert drain_parts() == [{"type": "todo", "data": {"t": 1}}]


def test_emit_part_buffer_still_enforces_sanitization():
    # The fallback buffer applies the SAME type/size rules as the collector.
    with tool_result_parts_buffer() as buf:
        assert emit_part("", {"x": 1}) is False
        assert emit_part("to\x1edo", {"x": 1}) is False
        assert emit_part("todo", {"body": "z" * (MAX_PART_PAYLOAD_BYTES + 100)}) is False
        assert buf == []


# --------------------------------------------------------------------------
# sanitize_part — the dispatcher-side gate for envelope-carried entries
# --------------------------------------------------------------------------

def test_sanitize_part_valid_entry_strips_extra_keys():
    clean = sanitize_part({
        "type": "selfie_finished",
        "data": {"url": "u"},
        "id": "x" * 200,
        "evil": "smuggled",  # never reaches the wire payload
    })
    assert clean == {
        "type": "selfie_finished",
        "data": {"url": "u"},
        "id": "x" * 128,  # id truncated exactly like emit_part does
    }
    # id is optional — absent stays absent.
    assert sanitize_part({"type": "t", "data": 1}) == {"type": "t", "data": 1}


def test_sanitize_part_rejects_invalid_entries():
    assert sanitize_part("not a dict") is None
    assert sanitize_part(None) is None
    assert sanitize_part({"data": 1}) is None  # no type
    assert sanitize_part({"type": "to\x1edo", "data": 1}) is None
    assert sanitize_part({"type": "x" * 65, "data": 1}) is None
    assert sanitize_part({"type": "t", "data": object()}) is None
    assert sanitize_part(
        {"type": "t", "data": {"body": "z" * (MAX_PART_PAYLOAD_BYTES + 100)}}
    ) is None
    assert sanitize_part({"type": "t", "data": float("nan")}) is None


# --------------------------------------------------------------------------
# DynamicTool.execute — envelope attachment at the wrap site
# --------------------------------------------------------------------------

class _SelfieFeature(Feature):
    """Probe feature whose tools mirror the Frinz selfie flow shapes."""

    @property
    def tool_description(self) -> str:
        return "visual identity probe"

    async def initialize(self):
        return None

    @tool(
        name="generate_selfie",
        description="Render a selfie and emit pending/finished parts",
        category=ToolCategory.UTILITY,
    )
    async def generate_selfie(self) -> ToolResult:
        """Render a selfie."""
        emit_part("selfie_pending", {"scene": "window"}, part_id="sf-1")
        emit_part("selfie_finished", {"url": "https://img/x.png"}, part_id="sf-1")
        return ToolResult.ok(
            "Selfie rendered", data={"url": "https://img/x.png"},
        )

    @tool(
        name="legacy_selfie",
        description="Pre-migration dict-returning tool that emits a part",
        category=ToolCategory.UTILITY,
    )
    async def legacy_selfie(self) -> dict:
        """Emit a part, return a legacy dict."""
        emit_part("selfie_finished", {"url": "https://img/y.png"})
        return {"url": "https://img/y.png"}

    @tool(
        name="explicit_parts_selfie",
        description="Returns ToolResult carrying explicit parts (SDK forward-compat)",
        category=ToolCategory.UTILITY,
    )
    async def explicit_parts_selfie(self) -> ToolResult:
        """Attach parts explicitly on the ToolResult."""
        result = ToolResult.ok("Selfie ready", data={"url": "https://img/z.png"})
        # The pinned SDK's frozen ToolResult predates the ``parts`` field;
        # setting the instance attribute is exactly what the shipped field
        # will look like to ``getattr(result, "parts", None)``.
        object.__setattr__(result, "parts", [
            {"type": "selfie_finished", "data": {"url": "https://img/z.png"}, "id": "sf-3"},
            {"type": "bad\x1etype", "data": 1},  # must be dropped by sanitize
        ])
        return result

    @tool(
        name="failing_selfie",
        description="Emits a pending part then raises",
        category=ToolCategory.UTILITY,
    )
    async def failing_selfie(self) -> ToolResult:
        """Emit then fail."""
        emit_part("selfie_pending", {"scene": "beach"}, part_id="sf-4")
        raise RuntimeError("render backend down")


def _make_selfie_feature(llm_service=None) -> _SelfieFeature:
    agent = SimpleNamespace(llm_service=llm_service, did="did:test:selfie")
    return _SelfieFeature(agent)


def _tool(feature: Feature, name: str):
    return {t.name: t for t in feature.get_tools()}[name]


@pytest.mark.asyncio
async def test_dynamic_tool_attaches_buffered_parts_when_no_collector():
    feature = _make_selfie_feature()
    envelope = await _tool(feature, "generate_selfie").execute()
    assert envelope["success"] is True
    assert envelope["status"] == "ok"
    assert envelope["parts"] == [
        {"type": "selfie_pending", "data": {"scene": "window"}, "id": "sf-1"},
        {"type": "selfie_finished", "data": {"url": "https://img/x.png"}, "id": "sf-1"},
    ]


@pytest.mark.asyncio
async def test_dynamic_tool_legacy_dict_envelope_carries_parts():
    feature = _make_selfie_feature()
    envelope = await _tool(feature, "legacy_selfie").execute()
    assert envelope["success"] is True
    assert envelope["result"] == {"url": "https://img/y.png"}
    assert envelope["parts"] == [
        {"type": "selfie_finished", "data": {"url": "https://img/y.png"}},
    ]


@pytest.mark.asyncio
async def test_dynamic_tool_honors_explicit_toolresult_parts():
    # SDK forward-compat: an explicit ``ToolResult.parts`` list is honored,
    # sanitized entry-by-entry (the smuggled control-char type is dropped).
    feature = _make_selfie_feature()
    envelope = await _tool(feature, "explicit_parts_selfie").execute()
    assert envelope["success"] is True
    assert envelope["parts"] == [
        {"type": "selfie_finished", "data": {"url": "https://img/z.png"}, "id": "sf-3"},
    ]


@pytest.mark.asyncio
async def test_dynamic_tool_error_path_still_carries_parts():
    # A *_pending part emitted before the failure still travels — matching
    # the collector path, where emit_part delivers immediately regardless of
    # how the tool call ends.
    feature = _make_selfie_feature()
    envelope = await _tool(feature, "failing_selfie").execute()
    assert envelope["success"] is False
    assert "render backend down" in envelope["error"]
    assert envelope["parts"] == [
        {"type": "selfie_pending", "data": {"scene": "beach"}, "id": "sf-4"},
    ]


@pytest.mark.asyncio
async def test_dynamic_tool_envelope_unchanged_when_collector_bound():
    # Back-compat: with a live turn collector the envelope is byte-identical
    # to the pre-#2641 shape (no ``parts`` key) and the collector delivers.
    feature = _make_selfie_feature()
    with part_collector():
        envelope = await _tool(feature, "generate_selfie").execute()
        drained = drain_parts()
    assert "parts" not in envelope
    assert [p["type"] for p in drained] == ["selfie_pending", "selfie_finished"]


# --------------------------------------------------------------------------
# execute_as_subagent — envelope returns parts by contract
# --------------------------------------------------------------------------

class _ToolCallOnceLLM:
    """Anthropic/OpenAI-style flow: generate() returns one tool call; the
    subagent loop executes it, then the continuation returns plain text."""

    def __init__(self, tool_name: str):
        self._tool_name = tool_name

    async def generate(self, **kwargs):
        return LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="c1", name=self._tool_name, arguments={})],
        )

    async def generate_with_messages(self, **kwargs):
        return LLMResponse(content="selfie done", tool_calls=None)


class _CodexInlineLLM:
    """Codex app-server-style flow: the adapter executes the tool INSIDE the
    LLM turn via the threaded ``tool_executor`` — on a task whose context was
    frozen BEFORE the turn (no collector, no tool-result buffer). This is the
    exact topology that dropped the Frinz selfie parts."""

    def __init__(self, frozen_ctx: contextvars.Context, tool_name: str):
        self._frozen_ctx = frozen_ctx
        self._tool_name = tool_name

    async def generate(self, **kwargs):
        executor = kwargs.get("tool_executor")
        assert executor is not None
        loop = asyncio.get_running_loop()
        # Reader-spawned handler task: inherits the reader's frozen context,
        # not the turn's.
        await loop.create_task(
            executor(self._tool_name, {}), context=self._frozen_ctx.copy(),
        )
        return LLMResponse(content="selfie done", tool_calls=None)

    async def generate_with_messages(self, **kwargs):
        return LLMResponse(content="selfie done", tool_calls=None)


@pytest.mark.asyncio
async def test_execute_as_subagent_returns_parts_from_tool_loop():
    feature = _make_selfie_feature(_ToolCallOnceLLM("generate_selfie"))
    envelope = await feature.execute_as_subagent(task="take a selfie")
    assert envelope["success"] is True
    assert [p["type"] for p in envelope["parts"]] == [
        "selfie_pending", "selfie_finished",
    ]


@pytest.mark.asyncio
async def test_execute_as_subagent_returns_parts_via_codex_inline_executor():
    # Freeze a pre-turn context (no collector, no buffer) — the reader task's
    # view of the world in the live codex app-server.
    frozen_ctx = contextvars.copy_context()
    feature = _make_selfie_feature(_CodexInlineLLM(frozen_ctx, "generate_selfie"))
    envelope = await feature.execute_as_subagent(task="take a selfie")
    assert envelope["success"] is True, envelope
    assert [p["type"] for p in envelope["parts"]] == [
        "selfie_pending", "selfie_finished",
    ], (
        "parts emitted on a frozen-context inline-executor task must ride the "
        "subagent envelope — this is the exact Frinz dashboard selfie gap"
    )


@pytest.mark.asyncio
async def test_execute_as_subagent_no_parts_keeps_envelope_shape():
    # No parts produced → no ``parts`` key; existing callers see the
    # pre-#2641 envelope byte-for-byte.
    class _PlainLLM:
        async def generate(self, **kwargs):
            return LLMResponse(content="just words", tool_calls=None)

        async def generate_with_messages(self, **kwargs):
            return LLMResponse(content="just words", tool_calls=None)

    feature = _make_selfie_feature(_PlainLLM())
    envelope = await feature.execute_as_subagent(task="chat only")
    assert envelope["success"] is True
    assert "parts" not in envelope


# --------------------------------------------------------------------------
# Orchestrator dispatch — envelope parts re-emitted into the parent turn
# --------------------------------------------------------------------------

def _allow_hook_output():
    return SimpleNamespace(
        permission_decision=PermissionDecision.ALLOW,
        permission_reason=None,
        updated_input=None,
    )


class _Orchestrator(OrchestratorEngineMixin):
    """Minimal host running the REAL dispatch methods over stubbed
    governance/observability collaborators."""

    def __init__(self):
        self.hooks_manager = MagicMock()
        self.hooks_manager.execute_hooks = AsyncMock(
            return_value=_allow_hook_output(),
        )
        self.hooks_manager.execute_hooks_parallel = AsyncMock(return_value=None)
        self.observability_store = MagicMock()
        self.observability_store.log_tool_response = AsyncMock()
        self._direct_tools = {}
        self._tool_to_feature = {}
        self.features = {}

    async def _get_denied_tools(self, feature_name):
        return set()

    def _register_explored_feature_tools(self, feature):
        return None


def test_reemit_envelope_parts_off_turn_is_noop():
    # No collector bound: nothing to deliver, the envelope stays the
    # caller's to consume — and nothing raises.
    OrchestratorEngineMixin._reemit_envelope_parts(
        {"success": True, "parts": [{"type": "t", "data": 1}]},
    )
    assert drain_parts() == []


def test_reemit_envelope_parts_sanitizes_and_delivers():
    with part_collector():
        OrchestratorEngineMixin._reemit_envelope_parts({
            "success": True,
            "parts": [
                {"type": "selfie_finished", "data": {"url": "u"}, "id": "i1"},
                {"type": "bad\x1etype", "data": 1},   # dropped
                "not-a-dict",                          # dropped
            ],
        })
        # Non-dict envelopes and partsless envelopes are no-ops.
        OrchestratorEngineMixin._reemit_envelope_parts("string result")
        OrchestratorEngineMixin._reemit_envelope_parts({"success": True})
        drained = drain_parts()
    assert drained == [
        {"type": "selfie_finished", "data": {"url": "u"}, "id": "i1"},
    ]


@pytest.mark.asyncio
async def test_subagent_selfie_part_reaches_outbound_stream():
    """Acceptance (#2641): a subagent-emitted ``selfie_finished`` part reaches
    the outbound stream through the REAL chat-path dispatch chain —
    ``_dispatch_feature_tool`` → ``execute_as_subagent`` → subagent tool loop
    → envelope → re-emit → turn collector → PART sentinel → stream parser."""
    orch = _Orchestrator()
    feature = _make_selfie_feature(_ToolCallOnceLLM("generate_selfie"))

    with part_collector():
        result = await orch._dispatch_feature_tool(
            SimpleNamespace(name="selfie_feature", id="tc-1"),
            feature,
            {"task": "take a selfie"},
            time.time(),
            "evt-1",
            "please take a selfie",
            tool_events=[],
            streaming=True,
            session_id="sess-1",
        )
        drained = drain_parts()

    # The envelope carried the parts...
    assert result["success"] is True
    assert [p["type"] for p in result["parts"]] == [
        "selfie_pending", "selfie_finished",
    ]
    # ...and the dispatch site re-emitted each into the turn collector
    # exactly once (no duplicates from the collector/envelope dual paths).
    assert [p["type"] for p in drained] == ["selfie_pending", "selfie_finished"]

    # Prove stream delivery: the drained part serializes to a PART sentinel
    # the streaming parser recovers — the outbound-stream wire contract.
    finished = next(p for p in drained if p["type"] == "selfie_finished")
    sentinel = build_part_sentinel(finished)
    clean, _tools, parsed = _parse_stream_sentinels("pre " + sentinel + "post")
    assert clean == "pre post"
    assert parsed[0]["type"] == "selfie_finished"
    assert parsed[0]["data"] == {"url": "https://img/x.png"}


@pytest.mark.asyncio
async def test_codex_subagent_selfie_part_reaches_outbound_stream():
    """Same acceptance flow on the codex topology: the subagent's tool runs on
    a frozen-context task (no collector anywhere in sight), yet the part
    still lands on the parent turn via the envelope contract."""
    orch = _Orchestrator()
    frozen_ctx = contextvars.copy_context()  # captured BEFORE the turn
    feature = _make_selfie_feature(_CodexInlineLLM(frozen_ctx, "generate_selfie"))

    with part_collector():
        result = await orch._dispatch_feature_tool(
            SimpleNamespace(name="selfie_feature", id="tc-1"),
            feature,
            {"task": "take a selfie"},
            time.time(),
            "evt-1",
            "please take a selfie",
            tool_events=[],
            streaming=True,
            session_id="sess-1",
        )
        drained = drain_parts()

    assert result["success"] is True
    assert [p["type"] for p in drained] == ["selfie_pending", "selfie_finished"], (
        "codex inline-executor subagent parts must reach the parent turn — "
        "pre-#2641 these were silently dropped and the selfie degraded to an "
        "inline markdown image"
    )


@pytest.mark.asyncio
async def test_direct_tool_dispatch_reemits_envelope_parts():
    """After progressive disclosure, an explored feature's tools re-register
    as DIRECT tools — the second selfie call in a conversation dispatches via
    ``_dispatch_direct_tool``, which must honor the same envelope contract."""
    orch = _Orchestrator()
    feature = _make_selfie_feature()
    orch._direct_tools = {"explicit_parts_selfie": _tool(feature, "explicit_parts_selfie")}

    with part_collector():
        result = await orch._dispatch_direct_tool(
            SimpleNamespace(name="explicit_parts_selfie", id="tc-2"),
            "explicit_parts_selfie",
            {},
            time.time(),
            "evt-2",
            tool_events=[],
            streaming=True,
            session_id="sess-2",
        )
        drained = drain_parts()

    assert result["success"] is True
    assert [p["type"] for p in drained] == ["selfie_finished"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
