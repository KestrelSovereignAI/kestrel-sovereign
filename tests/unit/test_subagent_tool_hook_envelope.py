"""A feature subagent's tools run inside the full PRE/POST_TOOL_USE envelope,
filed under the turn that dispatched them (#3612).

``Feature._execute_subagent_tool`` is the one hook-enforced path for a feature
subagent's inner tools — the post-LLM loop and the inline executor (codex
app-server) both go through it. It used to fire ``PRE_TOOL_USE`` only, under
the constant ``session_id="subagent"``. The observability hook emits a
``<tool> (started)`` marker on PRE and the terminal tool span on POST, and
closes leftover starts at the session's ``Stop``; with no POST and a "session"
that never stops, every subagent tool call rendered on the fleet Timeline as a
bar running to the live edge forever (73 of them on Emma's lane in one
morning, none closed).

These tests pin, on both doors, over the real ``HooksManager``:
  - PRE and POST both fire for a tool that ran, POST carrying the result, the
    measured duration, the effective args and the feature;
  - both edges carry the dispatching turn's session, resolved once at
    dispatch, and ``""`` (never a sentinel) for a session-less turn;
  - a tool that raised still gets its POST, reporting the failure;
  - a DENY/ASK, which did not run, gets no POST.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from kestrel_sdk.hooks.base import Hook, HookEvent, HookInput, HookOutput
from kestrel_sdk.llm.response import LLMResponse, ToolCall
from kestrel_sdk.tools.base import ToolCategory, ToolSchema

from kestrel_sovereign.features.base import Feature
from kestrel_sovereign.hooks.manager import HooksManager
from tests.utils.native_tool_effects import drive_native
from kestrel_sdk.resources import ResourceOutcomeUnknown

TOOL_SLEEP_S = 0.02


class _SlowTool:
    def __init__(self, name: str, returns):
        self.name = name
        self.schema = ToolSchema(
            name=name,
            description=f"fake tool {name}",
            category=ToolCategory.UTILITY,
        )
        self._returns = returns
        self.executed_with = None

    async def execute(self, **kwargs):
        self.executed_with = dict(kwargs)
        await asyncio.sleep(TOOL_SLEEP_S)
        if isinstance(self._returns, Exception):
            raise self._returns
        return self._returns


class _ProbeFeature(Feature):
    @property
    def tool_description(self) -> str:
        return "probe"

    async def initialize(self):
        return None

    def get_tools(self):
        return self._fake_tools


class _Recorder(Hook):
    """Records every tool-hook edge as the hooks saw it; optionally answers
    PRE_TOOL_USE with a blocking decision."""

    def __init__(self, pre_decision: HookOutput | None = None):
        super().__init__(
            name="recorder",
            events=[HookEvent.PRE_TOOL_USE, HookEvent.POST_TOOL_USE],
        )
        self._pre_decision = pre_decision
        self.edges: list[dict] = []

    async def execute(self, input: HookInput) -> HookOutput:
        self.edges.append({
            "event": input.hook_event_name,
            "session_id": input.session_id,
            "tool_name": input.tool_name,
            "feature_name": input.feature_name,
            "tool_input": dict(input.tool_input or {}),
            "tool_response": input.tool_response,
            "execution_time_ms": input.execution_time_ms,
        })
        if (
            input.hook_event_name == HookEvent.PRE_TOOL_USE.value
            and self._pre_decision is not None
        ):
            return self._pre_decision
        return HookOutput.allow()

    def event_names(self) -> list[str]:
        return [e["event"] for e in self.edges]


class _TurnBinding:
    """Release the real live turn's session after its feature captured it."""

    def __init__(self, agent):
        self.agent = agent

    def release(self):
        self.agent._active_session_id = None


class _LoopDoorLLM:
    """Post-LLM loop door: ``generate`` returns one tool call, which the
    feature's own loop executes; the continuation returns text."""

    def __init__(self, binding: _TurnBinding, tool_name: str, args: dict):
        self._binding = binding
        self._tool_name = tool_name
        self._args = args

    async def generate(self, **kwargs):
        self._binding.release()
        return LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="c1", name=self._tool_name, arguments=self._args)],
        )

    async def generate_with_messages(self, **kwargs):
        return LLMResponse(content="done", tool_calls=None)


class _InlineDoorLLM:
    """Inline door (codex app-server): the adapter runs the tool INSIDE the
    LLM call through the threaded ``tool_executor``."""

    def __init__(self, binding: _TurnBinding, tool_name: str, args: dict):
        self._binding = binding
        self._tool_name = tool_name
        self._args = args

    async def generate(self, **kwargs):
        self._binding.release()
        await kwargs["tool_executor"](self._tool_name, dict(self._args))
        return LLMResponse(content="done", tool_calls=None)

    async def generate_with_messages(self, **kwargs):
        return LLMResponse(content="done", tool_calls=None)


DOORS = {"loop": _LoopDoorLLM, "inline": _InlineDoorLLM}


async def _make(door: str, tool: _SlowTool, *, session_id, recorder: _Recorder, args=None,
          native_tool_effect_agent):
    agent = await native_tool_effect_agent(did="did:test:probe")
    binding = _TurnBinding(agent)
    manager = HooksManager()
    manager.register(recorder)
    agent.hooks_manager = manager
    agent.llm_service = DOORS[door](binding, tool.name, args or {})
    feature = _ProbeFeature(agent)
    feature._fake_tools = [tool]
    feature._test_session = session_id
    agent.features[feature.name] = feature
    return feature


async def _run_feature(feature, **kwargs):
    async def operation():
        async with feature.agent._turn_lifecycle():
            feature.agent._active_session_id = feature._test_session
            return await feature.execute_as_subagent(**kwargs)
    return await drive_native(feature.agent, operation)


@pytest.mark.asyncio
@pytest.mark.parametrize("door", sorted(DOORS))
async def test_tool_that_ran_fires_pre_and_post_under_the_dispatching_turn(door, *, native_tool_effect_agent):
    tool = _SlowTool("search_memory", returns={"success": True, "data": {"hits": 2}})
    recorder = _Recorder()
    feature = (await _make(door, tool, session_id="sess-turn", recorder=recorder, args={"q": "x"}, native_tool_effect_agent=native_tool_effect_agent))

    envelope = await _run_feature(feature, task="search for x")

    assert envelope["success"] is True, envelope
    assert tool.executed_with == {"q": "x"}
    assert recorder.event_names() == ["PreToolUse", "PostToolUse"], (
        "a subagent tool that ran must close its PRE with a POST — without "
        "it the observability hook's (started) marker is never paired and "
        "the Timeline paints it running forever (#3612)"
    )
    pre, post = recorder.edges
    # Both edges file the call under the turn that dispatched the subagent —
    # resolved once at dispatch, so the binding going away mid-run (the
    # doubles release it before the tool runs) does not drop it.
    assert pre["session_id"] == post["session_id"] == "sess-turn"
    assert post["tool_name"] == "search_memory"
    assert post["feature_name"] == "_ProbeFeature"
    assert post["tool_input"] == {"q": "x"}
    assert post["tool_response"]["success"] is True
    assert post["tool_response"]["data"] == {"hits": 2}
    # The duration is measured around the tool's own run.
    assert post["execution_time_ms"] >= int(TOOL_SLEEP_S * 1000) - 1


@pytest.mark.asyncio
@pytest.mark.parametrize("door", sorted(DOORS))
async def test_uncertain_tool_raises_without_a_normal_post_hook(door, *, native_tool_effect_agent):
    tool = _SlowTool("broken", returns=ValueError("bad input"))
    recorder = _Recorder()
    feature = (await _make(door, tool, session_id="sess-turn", recorder=recorder, native_tool_effect_agent=native_tool_effect_agent))

    with pytest.raises(ResourceOutcomeUnknown):
        await _run_feature(feature, task="anything")
    assert recorder.event_names() == ["PreToolUse"]
    assert recorder.edges[0]['session_id'] == 'sess-turn'
    assert await feature.agent.tool_effect_store.database.fetchone(
        'SELECT result_status, error_class FROM a2a_tool_dispatches') == ('error','ResourceOutcomeUnknown')


@pytest.mark.asyncio
@pytest.mark.parametrize("door", sorted(DOORS))
@pytest.mark.parametrize(
    "decision",
    [HookOutput.deny("not allowed"), HookOutput.ask("appr-1", "needs approval")],
    ids=["deny", "ask"],
)
async def test_blocked_call_fires_no_post(door, decision, *, native_tool_effect_agent):
    """A refused call never ran, so it gets no POST — the chat path's
    contract. The observability hook closes its start at the turn's ``Stop``,
    which now reaches it because the call is filed under that turn."""
    tool = _SlowTool("dangerous", returns={"success": True})
    recorder = _Recorder(pre_decision=decision)
    feature = (await _make(door, tool, session_id="sess-turn", recorder=recorder, native_tool_effect_agent=native_tool_effect_agent))

    await _run_feature(feature, task="anything")

    assert tool.executed_with is None
    assert recorder.event_names() == ["PreToolUse"]
    assert recorder.edges[0]["session_id"] == "sess-turn"


@pytest.mark.asyncio
@pytest.mark.parametrize("door", sorted(DOORS))
async def test_session_less_turn_files_hooks_under_empty_session_not_a_sentinel(door, *, native_tool_effect_agent):
    """No chat session → ``""``, as ``USER_PROMPT_SUBMIT`` stamps a
    session-less turn. A named sentinel is what made every subagent call the
    agent ever made — across restarts — one never-ending pseudo-session."""
    tool = _SlowTool("search_memory", returns={"success": True})
    recorder = _Recorder()
    feature = (await _make(door, tool, session_id=None, recorder=recorder, native_tool_effect_agent=native_tool_effect_agent))

    await _run_feature(feature, task="anything")

    assert recorder.event_names() == ["PreToolUse", "PostToolUse"]
    assert [e["session_id"] for e in recorder.edges] == ["", ""]


@pytest.mark.asyncio
@pytest.mark.parametrize("capture_args", [False, True])
@pytest.mark.parametrize("scenario,decision", [
    ("allow", None), ("bare_stop", None), ("bare_stop", "allow"),
    ("stop_then_deny", None), ("stop_then_deny", "allow"),
])
async def test_real_subagent_tool_cannot_cross_a_stopped_hook_chain(
    tmp_path, capture_args, scenario, decision, native_tool_effect_agent,
):
    from kestrel_sovereign.features.base import tool
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sdk.hooks.base import PermissionDecision

    witness = tmp_path / "owned-subagent-effect.txt"

    class OwnedEffectFeature(Feature):
        @property
        def tool_description(self):
            return "Owned subagent policy probe"
        async def initialize(self):
            pass
        @tool(name="record_marker", description="Record the owned subagent marker")
        async def record_marker(self, marker: str):
            witness.write_text(marker, encoding="utf-8")
            return {"success": True, "data": {"marker": marker}}

    class PolicyHook(Hook):
        def __init__(self, name, output, priority):
            super().__init__(name, events=[HookEvent.PRE_TOOL_USE], priority=priority)
            self.output = output
        async def execute(self, input):
            return self.output

    agent = await native_tool_effect_agent(did="did:test:owned-subagent-policy")
    agent.hooks_manager = HooksManager()
    agent.hooks_manager.register(PolicyHook("redact", HookOutput.modify({"marker": "owned-redacted-marker"}), 5))
    if scenario != "allow":
        output = HookOutput(continue_execution=False, stop_reason="Owned stop",
            permission_decision=PermissionDecision.ALLOW if decision == "allow" else None)
        agent.hooks_manager.register(PolicyHook("stop", output, 10))
    if scenario == "stop_then_deny":
        agent.hooks_manager.register(PolicyHook("later-deny", HookOutput.deny("Owned later denial"), 50))
    feature = OwnedEffectFeature(agent)
    agent.features[feature.name] = feature
    tools = {item.name: item for item in feature.get_tools()}
    name, = tools
    output = await drive_native(agent, lambda: feature._execute_subagent_tool(tool_name=name,
        args={"marker": "owned-private-marker"}, tools_by_name=tools,
        session_id="owned-subagent-turn", return_with_effective_args=capture_args))
    if capture_args:
        effective, result = output
        assert effective == {"marker": "owned-redacted-marker"}
    else:
        result = output
    if scenario == "allow":
        assert result["success"] is True
        assert witness.read_text(encoding="utf-8") == "owned-redacted-marker"
    else:
        assert not witness.exists(), "A stopped subagent performed its real tool effect"
        assert result["success"] is False
        assert "NOT executed" in result["error"]
