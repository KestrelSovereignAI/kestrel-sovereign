"""Built-in and SDK feature tools bind a call's arguments identically (#3921).

A built-in feature's ``@tool`` methods run through Core's ``DynamicTool``
wrapper; an external feature's run through the SDK's. Before #3921 they
disagreed twice:

* Only the SDK wrapper coerced a model's arguments to the declared types, so
  ``web_search(max_results="30")`` reached Tavily's ``min(max_results, 10)``
  as a string and raised ``TypeError``.
* Only Core's wrapper re-raised a host execution-control error, so an SDK
  feature's tool turned an ``ExecutionAuthorityError`` into a failed result.

One contract, run over a built-in feature and an SDK feature through the
command, promoted direct, governed binding, named, subagent and scheduler
paths. Expectations
never depend on the feature's kind, so a divergence fails one of the two.
Numeric and boolean strings arrive coerced; an uncoercible value fails with the
same named error; an ``ExecutionAuthorityError`` propagates as the same object;
an ordinary exception becomes the same failure envelope. The arguments
PRE_TOOL_USE authorized, the method ran with, POST_TOOL_USE saw and the
dispatch row recorded agree, including after a PRE_TOOL_USE hook rewrote them:
a rewrite the tool's wrapper would still coerce does not run.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from kestrel_sdk.features.base import Feature as SdkFeature
from kestrel_sdk.features.host_errors import (
    is_host_control_error,
    register_control_error_predicate,
)
from kestrel_sdk.hooks.base import Hook, HookEvent, HookInput, HookOutput
from kestrel_sdk.llm.response import LLMResponse, ToolCall
from kestrel_sdk.native_a2a.types import AgentSkill, TaskState
from kestrel_sdk.tools.base import ToolCategory, ToolParameter

from kestrel_sovereign.a2a.projections import embedded_json_value
from kestrel_sovereign.a2a.scope import NativeRecipientScope
from kestrel_sovereign.a2a.stores.unified.observability_store import ObservabilityStore
from kestrel_sovereign.a2a.task_manager import create_task_manager
from kestrel_sovereign.agent.direct_tool_arguments import (
    coerce_declared_tool_arguments,
    hook_rewrite_argument_error,
    normalize_direct_tool_arguments,
)
from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.agent.tool_registry import ToolRegistryMixin
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    is_execution_control_error,
)
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.scheduler.feature import SchedulerFeature
from kestrel_sovereign.features.subagent_dispatch import ensure_subagent_dispatch
from kestrel_sovereign.features.web_search.feature import WebSearchFeature
from kestrel_sovereign.features.web_search.tool import WebSearchTool
from kestrel_sovereign.hooks.manager import HooksManager
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

SESSION = "sess-3921"
UNCOERCIBLE_ERROR = (
    "probe: parameter 'limit' expects type 'integer', but received 'abc' (str), "
    "which is not a valid integer"
)


class _ProbeTools:
    """The tool bodies both probe features share: only the base differs."""

    tool_name = "probe_feature"

    def __init__(self, agent):
        super().__init__(agent)
        self.calls: list[dict] = []
        self.error: BaseException | None = None

    @property
    def tool_description(self) -> str:
        return "Probe how a tool call's arguments are bound."

    async def initialize(self) -> None:
        return None

    @tool(
        name="probe",
        description="Record the arguments the call was bound with.",
        category=ToolCategory.DATA_ACCESS,
        command_prefix="!probe",
    )
    async def probe(self, limit: int = 5, exact: bool = False) -> dict:
        self.calls.append({"limit": limit, "exact": exact})
        return {"limit": limit, "exact": exact}

    @tool(
        name="probe_raise",
        description="Raise the configured error.",
        category=ToolCategory.DATA_ACCESS,
        command_prefix="!probe-raise",
    )
    async def probe_raise(self) -> dict:
        raise self.error


class _BuiltinProbe(_ProbeTools, Feature):
    """A built-in feature: Core's ``DynamicTool`` wraps its tools."""


class _SdkProbe(_ProbeTools, SdkFeature):
    """An external feature: the SDK's ``DynamicTool`` wraps its tools."""


KINDS = {
    "builtin": _BuiltinProbe,
    "sdk": ensure_subagent_dispatch(_SdkProbe),
}


def _typed(arguments: dict) -> list:
    """``arguments`` with each value's type: ``"30" != 30`` but ``1 == True``."""
    return sorted((name, type(value).__name__, value) for name, value in arguments.items())


class _Recorder(Hook):
    """Records the arguments PRE_TOOL_USE authorized and POST_TOOL_USE saw."""

    def __init__(self):
        super().__init__(
            name="authorized-spy",
            events=[HookEvent.PRE_TOOL_USE, HookEvent.POST_TOOL_USE],
        )
        self.authorized: list[dict] = []
        self.reported: list[dict] = []

    async def execute(self, input: HookInput) -> HookOutput:
        seen = self.authorized
        if input.hook_event_name == HookEvent.POST_TOOL_USE.value:
            seen = self.reported
        seen.append(dict(input.tool_input or {}))
        return HookOutput.allow()


def _hooks() -> tuple[HooksManager, _Recorder]:
    manager = HooksManager()
    recorder = _Recorder()
    manager.register(recorder)
    return manager, recorder


class _Agent(OrchestratorEngineMixin, ToolRegistryMixin):
    def __init__(self, store, hooks):
        self.did = "did:test:binding-parity"
        self.features = {}
        self._direct_tools, self._tool_to_feature, self._explored_features = {}, {}, {}
        self._direct_tool_defs, self._pinned_features = [], set()
        self.hooks_manager, self.observability_store = hooks, store


async def _recorded_arguments(backend) -> list:
    rows = await backend.fetch_all("SELECT args_redacted FROM a2a_tool_dispatches")
    return [json.loads(row[0]) for row in rows]


class _SubagentLLM:
    """A subagent's model: one tool call, then a final answer.

    ``inline`` runs the call the way the codex app-server does, inside the
    model call through ``tool_executor``, and keeps the ``(effective_args,
    result)`` pair the adapter records in ``a2a_tool_dispatches``.
    """

    def __init__(self, tool_name: str, arguments: dict, *, inline: bool):
        self.tool_name = tool_name
        self.arguments = arguments
        self.inline = inline
        self.executed: list[tuple] = []
        self.continuations: list[list] = []

    async def generate(self, **kwargs):
        if self.inline:
            self.executed.append(
                await kwargs["tool_executor"](self.tool_name, dict(self.arguments))
            )
            return LLMResponse(content="done", tool_calls=None)
        return LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="c1", name=self.tool_name, arguments=dict(self.arguments))],
        )

    async def generate_with_messages(self, **kwargs):
        self.continuations.append(list(kwargs["messages"]))
        return LLMResponse(content="done", tool_calls=None)

    @property
    def tool_messages(self) -> list[dict]:
        """The tool results the loop sent back with its first continuation."""
        first = self.continuations[0] if self.continuations else []
        return [message for message in first if message["role"] == "tool"]


class _Path:
    """One way a model's or user's call reaches a feature's tool."""

    def __init__(self, kind: str, tmp_path):
        self.kind = kind
        self.tmp_path = tmp_path
        self.recorder: _Recorder | None = None
        self.feature = None
        #: What ``probe_raise`` raises.
        self.error: BaseException | None = None
        self._closers: list = []

    def make_feature(self, agent):
        self.feature = KINDS[self.kind](agent)
        self.feature.error = self.error
        return self.feature

    async def close(self):
        for close in reversed(self._closers):
            await close()

    async def call(self, tool_name: str, arguments: dict, command: str):
        raise NotImplementedError

    def recorded(self) -> list:
        """The bindings this path's audit recorded, when it records one."""
        return []


class _CommandPath(_Path):
    """``!probe 30 true`` through ``TaskManager.execute_command``."""

    async def _manager(self):
        manager = await create_task_manager(
            str(self.tmp_path / "command.db"),
            scope=NativeRecipientScope("did:test:binding-parity"),
            include_memory=False,
            include_feedback=False,
        )
        self._closers.append(manager.close)
        manager.hooks_manager, self.recorder = _hooks()
        feature = self.make_feature(SimpleNamespace(did="did:test:binding-parity"))
        manager.register_provider(
            "probe_feature",
            [
                AgentSkill(id=name, name=name, description=name, tags=["probe"])
                for name in ("probe", "probe_raise")
            ],
            feature,
            command_prefixes={"!probe": "probe", "!probe-raise": "probe_raise"},
        )
        return manager

    async def call(self, tool_name, arguments, command):
        return await (await self._manager()).execute_command(command)


class _SkillPath(_CommandPath):
    """``TaskManager.execute_skill`` with JSON arguments, as a workflow step
    (``run_workflow``) sends them: no command parser coerces them first."""

    async def call(self, tool_name, arguments, command):
        manager = await self._manager()
        task = await manager.execute_skill(
            agent_id="probe_feature", skill_id=tool_name, args=dict(arguments), sync=True,
        )
        if task.status.state == TaskState.TASK_STATE_COMPLETED:
            [artifact] = task.artifacts
            recognized, value = embedded_json_value(artifact.parts[0])
            assert recognized
            return value
        assert task.status.state == TaskState.TASK_STATE_FAILED, task.status
        return {"success": False, "error": task.status.message.parts[0].text}


class _OrchestratorPath(_Path):
    async def _agent(self):
        backend = SQLiteBackend(str(self.tmp_path / "dispatch.db"))
        await backend.connect()
        store = ObservabilityStore(backend)
        await store.initialize()
        self._closers.append(store.close)
        self.backend = backend
        hooks, self.recorder = _hooks()
        agent = _Agent(store, hooks)
        feature = self.make_feature(agent)
        agent.features[type(feature).__name__] = feature
        return agent

    async def recorded_rows(self) -> list:
        return await _recorded_arguments(self.backend)


class _DirectPath(_OrchestratorPath):
    """A promoted tool called by the model in the chat loop."""

    async def call(self, tool_name, arguments, command):
        agent = await self._agent()
        agent._register_explored_feature_tools(self.feature)
        assert tool_name in agent._direct_tools
        return await agent._dispatch_tool_call(
            ToolCall(id="tc-1", name=tool_name, arguments=dict(arguments)),
            features_by_tool_name={},
            known_tools={tool_name},
            messages=[],
            iteration=0,
            user_message="",
            session_id=SESSION,
        )


class _BoundPath(_OrchestratorPath):
    """A tool mounted with ``register_dynamic_tools``, called through its
    governed binding (``bound_tool``), as an MCP App calls one."""

    async def call(self, tool_name, arguments, command):
        agent = await self._agent()
        [tool] = [t for t in self.feature.get_tools() if t.name == tool_name]
        bindings = []
        agent.register_dynamic_tools(
            "probe_feature", [tool],
            on_registered=lambda _mounted, binding: bindings.append(binding),
        )
        [binding] = bindings
        return await binding.invoke(dict(arguments), session_id=SESSION, source="parity")


class _NamedPath(_OrchestratorPath):
    """``execute_named_tool`` through the codex inline executor."""

    async def call(self, tool_name, arguments, command):
        agent = await self._agent()
        executor = agent._make_inline_tool_executor(SESSION)
        effective, result = await executor(tool_name, dict(arguments))
        await agent._append_executed_tool_breadcrumbs(
            [],
            [{"id": "tc-1", "name": tool_name, "arguments": effective, "result": result}],
            session_id=SESSION,
        )
        return result


class _SubagentPath(_Path):
    """A feature subagent's own model calling one of the feature's tools."""

    inline = False

    async def call(self, tool_name, arguments, command):
        hooks, self.recorder = _hooks()
        self.llm = _SubagentLLM(tool_name, arguments, inline=self.inline)
        agent = SimpleNamespace(
            llm_service=self.llm,
            did="did:test:binding-parity",
            hooks_manager=hooks,
            get_turn_bound_session_id=lambda: SESSION,
        )
        envelope = await self.make_feature(agent).execute_as_subagent(task="probe it")
        assert envelope["success"] is True, envelope
        if self.inline:
            [(_effective, result)] = self.llm.executed
            return result
        [message] = self.llm.tool_messages
        return json.loads(message["content"])

    def recorded(self) -> list:
        return [effective for effective, _result in self.llm.executed]


class _SubagentInlinePath(_SubagentPath):
    inline = True


class _SchedulerPath(_Path):
    """A scheduled tick naming the tool, through the scheduler's hook gate."""

    async def call(self, tool_name, arguments, command):
        hooks, self.recorder = _hooks()
        agent = SimpleNamespace(did="did:test:binding-parity", features={}, hooks_manager=hooks)
        feature = self.make_feature(agent)
        agent.features[type(feature).__name__] = feature
        scheduler = SchedulerFeature(agent)
        return await scheduler._lookup_raw_tool_result(tool_name, dict(arguments))


PATHS = {
    "bound": _BoundPath,
    "command": _CommandPath,
    "direct": _DirectPath,
    "named": _NamedPath,
    "scheduler": _SchedulerPath,
    "skill": _SkillPath,
    "subagent": _SubagentPath,
    "subagent_inline": _SubagentInlinePath,
}


@pytest_asyncio.fixture
async def make_path(tmp_path):
    made: list[_Path] = []

    def _make(path: str, kind: str) -> _Path:
        made.append(PATHS[path](kind, tmp_path))
        return made[-1]

    yield _make
    for path in made:
        await path.close()


def _success_envelope(limit, exact) -> dict:
    return {"success": True, "result": {"limit": limit, "exact": exact}, "tool": "probe"}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("path", sorted(PATHS))
async def test_numeric_and_boolean_strings_reach_the_method_coerced(make_path, path, kind):
    probe = make_path(path, kind)

    result = await probe.call("probe", {"limit": "30", "exact": "true"}, "!probe 30 true")

    assert result == _success_envelope(30, True)
    assert [_typed(call) for call in probe.feature.calls] == [
        _typed({"limit": 30, "exact": True})
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("path", sorted(PATHS))
async def test_the_authorized_executed_and_recorded_arguments_agree(make_path, path, kind):
    """The PRE_TOOL_USE spy, the method and the dispatch row see one binding."""
    probe = make_path(path, kind)

    await probe.call("probe", {"limit": "30", "exact": "true"}, "!probe 30 true")

    bound = [_typed({"limit": 30, "exact": True})]
    await _assert_one_binding(probe, bound, ran=True)


async def _assert_one_binding(probe, bound: list, *, ran: bool) -> None:
    """Every record of the call names ``bound``; the method ran with it iff ``ran``."""
    assert [_typed(args) for args in probe.recorder.authorized] == bound
    assert [_typed(call) for call in probe.feature.calls] == (bound if ran else [])
    assert [_typed(args) for args in probe.recorder.reported] == (bound if ran else [])
    if isinstance(probe, _OrchestratorPath):
        assert [_typed(row) for row in await probe.recorded_rows()] == bound
    elif isinstance(probe, _SubagentInlinePath):
        assert [_typed(args) for args in probe.recorded()] == bound


class _Rewrite(Hook):
    """A PRE_TOOL_USE hook that rewrites ``limit``, or denies the call."""

    def __init__(self, decision: HookOutput, *, priority: int = 100):
        super().__init__(name="rewrite", events=[HookEvent.PRE_TOOL_USE], priority=priority)
        self.decision = decision

    async def execute(self, input: HookInput) -> HookOutput:
        return self.decision


def _rewritten_before_the_spy(rewrite: dict):
    """``_hooks`` with a MODIFY hook that runs before the authorization spy,
    so the spy authorizes the rewrite, as any later hook in a chain would."""

    def hooks():
        manager = HooksManager()
        manager.register(_Rewrite(HookOutput.modify(rewrite), priority=10))
        recorder = _Recorder()
        manager.register(recorder)
        return manager, recorder

    return patch(f"{__name__}._hooks", hooks)


def _refusal_text(result) -> str:
    """Why a path refused the call: every path's failure carries ``error``."""
    assert result.get("success") is not True and result.get("status") != "ok", result
    return result["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("path", sorted(PATHS))
async def test_a_hook_rewrite_to_another_type_does_not_run(make_path, path, kind):
    """``limit: "7"`` from a MODIFY hook is what every later hook authorizes;
    the wrapper would run the method with ``7``. So it does not run, and the
    records name the binding the hooks authorized."""
    probe = make_path(path, kind)
    rewrite = {"limit": "7", "exact": "false"}

    with _rewritten_before_the_spy(rewrite):
        result = await probe.call("probe", {"limit": "30", "exact": "true"}, "!probe 30 true")

    assert (
        "probe: a PRE_TOOL_USE hook rewrote 'limit' (declared 'integer'), "
        "'exact' (declared 'boolean') to a value of another type"
    ) in _refusal_text(result)
    await _assert_one_binding(probe, [_typed(rewrite)], ran=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("path", sorted(PATHS))
async def test_a_hook_rewrite_its_type_refuses_fails_with_the_named_error(make_path, path, kind):
    probe = make_path(path, kind)

    with _rewritten_before_the_spy({"limit": "abc"}):
        result = await probe.call("probe", {"limit": "30", "exact": "true"}, "!probe 30 true")

    assert f"{UNCOERCIBLE_ERROR}. A PRE_TOOL_USE hook rewrote the arguments" in _refusal_text(
        result
    )
    await _assert_one_binding(probe, [_typed({"limit": "abc"})], ran=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("path", sorted(PATHS))
async def test_a_hook_rewrite_in_the_declared_types_runs_as_authorized(make_path, path, kind):
    probe = make_path(path, kind)
    rewrite = {"limit": 7, "exact": False}

    with _rewritten_before_the_spy(rewrite):
        result = await probe.call("probe", {"limit": "30", "exact": "true"}, "!probe 30 true")

    assert result == _success_envelope(7, False)
    await _assert_one_binding(probe, [_typed(rewrite)], ran=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", sorted(KINDS))
async def test_a_denied_direct_call_records_the_binding_it_was_refused(make_path, kind):
    probe = make_path("direct", kind)
    original_hooks = _hooks

    def hooks_with_denial():
        manager, recorder = original_hooks()
        manager.register(_Rewrite(HookOutput.deny("not today")))
        return manager, recorder

    with patch(f"{__name__}._hooks", hooks_with_denial):
        await probe.call("probe", {"limit": "30", "exact": "true"}, "!probe 30 true")

    assert probe.feature.calls == []
    assert [_typed(row) for row in await probe.recorded_rows()] == [
        _typed({"limit": 30, "exact": True})
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", sorted(KINDS))
async def test_a_direct_call_a_hook_rewrote_records_the_binding_that_ran(make_path, kind):
    probe = make_path("direct", kind)
    original_hooks = _hooks

    def hooks_with_rewrite():
        manager, recorder = original_hooks()
        manager.register(_Rewrite(HookOutput.modify({"limit": 7, "exact": False})))
        return manager, recorder

    with patch(f"{__name__}._hooks", hooks_with_rewrite):
        await probe.call("probe", {"limit": "30", "exact": "true"}, "!probe 30 true")

    ran = [_typed({"limit": 7, "exact": False})]
    assert [_typed(call) for call in probe.feature.calls] == ran
    assert [_typed(row) for row in await probe.recorded_rows()] == ran


#: How each path reports a value its declared type refuses.
UNCOERCIBLE_RESULTS = {
    "bound": {"success": False, "error": UNCOERCIBLE_ERROR},
    "command": {"success": False, "error": UNCOERCIBLE_ERROR},
    "scheduler": {"status": "error", "error": UNCOERCIBLE_ERROR, "tool": "probe"},
    "skill": {"success": False, "error": UNCOERCIBLE_ERROR},
    "direct": {"success": False, "error": UNCOERCIBLE_ERROR},
    "named": {"success": False, "error": f"Invalid tool arguments: {UNCOERCIBLE_ERROR}"},
    "subagent": {"status": "error", "error": UNCOERCIBLE_ERROR, "tool": "probe"},
    "subagent_inline": {"status": "error", "error": UNCOERCIBLE_ERROR, "tool": "probe"},
}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("path", sorted(PATHS))
async def test_an_uncoercible_value_fails_with_the_same_named_error(make_path, path, kind):
    probe = make_path(path, kind)

    result = await probe.call("probe", {"limit": "abc"}, "!probe abc")

    expected = UNCOERCIBLE_RESULTS[path]
    assert {key: result.get(key) for key in expected} == expected
    assert probe.feature.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("path", sorted(PATHS))
async def test_an_execution_authority_error_propagates_as_the_same_object(
    make_path, path, kind,
):
    probe = make_path(path, kind)
    denial = ExecutionAuthorityError("original admission retired")
    probe.error = denial

    with pytest.raises(ExecutionAuthorityError) as raised:
        await probe.call("probe_raise", {}, "!probe-raise")

    assert raised.value is denial


#: How each path reports a tool that raised an ordinary exception.
RAISED_RESULTS = {
    "bound": {"success": False, "error": "lookup failed", "tool": "probe_raise"},
    "command": {"success": False, "error": "lookup failed"},
    "scheduler": {"success": False, "error": "lookup failed", "tool": "probe_raise"},
    "skill": {"success": False, "error": "lookup failed"},
    "direct": {"success": False, "error": "lookup failed", "tool": "probe_raise"},
    "named": {"success": False, "error": "lookup failed", "tool": "probe_raise"},
    "subagent": {"success": False, "error": "lookup failed", "tool": "probe_raise"},
    "subagent_inline": {"success": False, "error": "lookup failed", "tool": "probe_raise"},
}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("path", sorted(PATHS))
async def test_an_ordinary_exception_becomes_the_same_failure_envelope(
    make_path, path, kind,
):
    probe = make_path(path, kind)
    probe.error = LookupError("lookup failed")

    result = await probe.call("probe_raise", {}, "!probe-raise")

    assert result == RAISED_RESULTS[path]


# --- the wrappers themselves -------------------------------------------------


def _probe_tool(kind: str, name: str = "probe"):
    feature = KINDS[kind](SimpleNamespace(did="did:test:binding-parity"))
    [found] = [candidate for candidate in feature.get_tools() if candidate.name == name]
    return feature, found


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", sorted(KINDS))
async def test_each_wrapper_coerces_a_call_it_is_handed_directly(kind):
    """A caller that reaches ``execute`` without a dispatch path still gets
    the declared types: the coercion is the wrapper's, not only the path's."""
    feature, probe = _probe_tool(kind)

    assert await probe.execute(limit="30", exact="false") == _success_envelope(30, False)
    assert [_typed(call) for call in feature.calls] == [_typed({"limit": 30, "exact": False})]

    refused = await probe.execute(limit="abc")
    assert {key: refused[key] for key in ("status", "error", "tool")} == {
        "status": "error", "error": UNCOERCIBLE_ERROR, "tool": "probe",
    }
    assert "success" not in refused
    assert len(feature.calls) == 1


@pytest.mark.asyncio
async def test_both_wrappers_refuse_an_uncoercible_value_identically():
    _, builtin = _probe_tool("builtin")
    _, sdk = _probe_tool("sdk")

    assert await builtin.execute(limit="abc") == await sdk.execute(limit="abc")


class _HandWrittenTool:
    """An ``AgentTool`` that is not a native ``@tool`` method: its ``execute``
    receives what it is sent, whatever its parameter list declares."""

    name = "hand_written"
    schema = SimpleNamespace(
        parameters=[ToolParameter("limit", "integer", "Limit", required=False)],
    )


@pytest.mark.parametrize("value", ["30", None, {"nested": [3]}], ids=["string", "null", "object"])
def test_a_tool_whose_wrapper_does_not_coerce_is_not_coerced_before_authorization(value):
    """Coercing it here would change what it runs with, not only when: a
    non-nullable ``ToolParameter`` would refuse a ``null`` it has always taken."""
    supplied = {"limit": value}

    for check in (coerce_declared_tool_arguments, normalize_direct_tool_arguments):
        arguments, error = check("hand_written", _HandWrittenTool(), supplied)
        assert error is None
        assert arguments is supplied
    # Nor is a hook's rewrite of it refused: it runs with what the hooks saw.
    assert hook_rewrite_argument_error("hand_written", _HandWrittenTool(), supplied) is None


def test_core_registers_its_control_error_predicate_with_the_sdk():
    """Importing ``execution_custody`` installed Core's predicate, so the SDK
    wrapper re-raises exactly what Core's wrapper re-raises."""
    assert is_host_control_error(ExecutionAuthorityError("retired")) is True
    assert is_host_control_error(LookupError("ordinary")) is False
    # Registering the same function again is the documented no-op.
    register_control_error_predicate(is_execution_control_error)
    assert is_host_control_error(ExecutionAuthorityError("retired")) is True


# --- web_search, the reported case -----------------------------------------------


def _tavily_response():
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"answer": "a", "results": [{"title": "t", "url": "u"}]}
    return response


@pytest.mark.asyncio
async def test_web_search_max_results_as_a_string_reaches_tavily_as_an_integer():
    """``max_results: "30"`` used to reach ``min(max_results, 10)`` as a string."""
    feature = WebSearchFeature(SimpleNamespace(did="did:test:binding-parity"))
    feature.tool = WebSearchTool(api_key="test-key")
    [web_search] = [t for t in feature.get_tools() if t.name == "web_search"]

    with patch("httpx.AsyncClient") as client_class:
        client = AsyncMock()
        client.post.return_value = _tavily_response()
        client.__aenter__.return_value = client
        client.__aexit__.return_value = None
        client_class.return_value = client

        result = await web_search.execute(query="kestrel", max_results="30")

    assert result["status"] == "ok", result
    assert result["success"] is True
    assert client.post.await_args.kwargs["json"]["max_results"] == 10


# --- #3919's SDK route: nothing a call carried reaches a log ---------------------

MARKER = "PRIVATE-MARKER-3921-7f3a"


@pytest.mark.asyncio
@pytest.mark.parametrize("inline", [False, True], ids=["loop", "inline"])
@pytest.mark.parametrize(
    "tool_name, arguments",
    [("probe_raise", {}), ("probe", {"limit": MARKER})],
    ids=["raised", "coercion-rejected"],
)
async def test_an_sdk_subagent_logs_nothing_a_call_carried(caplog, inline, tool_name, arguments):
    """An SDK feature adopted through ``ensure_subagent_dispatch``: a tool that
    raises with a marker, and a value its type refuses that carries one, leave
    the marker in no captured log record, at DEBUG."""
    hooks, _ = _hooks()
    llm = _SubagentLLM(tool_name, arguments, inline=inline)
    feature = KINDS["sdk"](
        SimpleNamespace(
            llm_service=llm,
            did="did:test:binding-parity",
            hooks_manager=hooks,
            get_turn_bound_session_id=lambda: SESSION,
        )
    )
    feature.error = LookupError(f"could not read {MARKER}")

    with caplog.at_level(logging.DEBUG):
        envelope = await feature.execute_as_subagent(task="probe it")

    assert envelope["success"] is True, envelope
    # The model still learns what failed: only the logs are content-free.
    if inline:
        [(_effective, result)] = llm.executed
        reported = json.dumps(result)
    else:
        [message] = llm.tool_messages
        reported = message["content"]
    assert MARKER in reported
    assert caplog.records, "the subagent logged nothing; the check below is vacuous"
    leaked = [record.getMessage() for record in caplog.records if MARKER in record.getMessage()]
    assert leaked == []
