"""Feature-subagent diagnostics log non-content facts only (#3919).

A subagent's inner tools mostly read the store (``search_memory``,
``read_message_by_id``, ``list_conversations``), which keeps that content
encrypted. Their results, the subagent's final answer, the task it was handed,
a hook's refusal reason and an exception's message were all logged as text or
as a 100-500 character prefix, at INFO, into a plaintext host log. Each now
logs only its length and a process-keyed digest (``content_log_summary``),
plus names, decisions and timing. A refused call's argument error names the
arguments the caller sent, so its log lines get the same summary while the
model still gets the whole error. These are Core's log sites; an extracted
feature's tools run through the SDK's own wrapper, whose records the SDK owns
(kestrel-sovereign-sdk#105).

Every test captures at DEBUG, because DEBUG is a supported operator level and
its records persist as well, and looks for the markers in every field of every
record (message, arguments, exception, stack), not only the rendered message.
"""

from __future__ import annotations

import logging
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kestrel_sdk.hooks.base import (
    Hook,
    HookEvent,
    HookInput,
    HookOutput,
    PermissionDecision,
)
from kestrel_sdk.llm.response import LLMResponse, ToolCall
from kestrel_sdk.native_a2a.helpers.proto_helpers import new_task, new_text_message
from kestrel_sdk.native_a2a.types import TaskState
from kestrel_sdk.tools.base import ToolCategory, ToolSchema

from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.hooks.manager import HooksManager
from kestrel_sovereign.logging_config import (
    exception_frames,
    exception_log_summary,
)
from tests.unit.test_execute_named_tool import (
    _FakeFeature,
    _FakeHooksManager,
    _FakeTool,
    _make_orchestrator,
    _SubagentFeature,
)
from tests.unit.test_tool_allowlist_registry import (  # noqa: F401
    _make_feature,
    _make_tool,
    _tool_call,
    agent,
    dispatch_agent,
)
from tests.utils.log_records import records_containing
from tests.utils.native_tool_effects import drive_native
from kestrel_sdk.resources import ResourceOutcomeUnknown
from uuid import uuid4

# None of these letters is a hex digit, so no digest can spell one by chance.
TOOL_MARKER = "SECRET-MARKER-qzXjvw"
ANSWER_MARKER = "ANSWER-MARKER-ruKpYt"
TASK_MARKER = "TASK-MARKER-wLmQzs"
REASON_MARKER = "REASON-MARKER-yRtNvk"
ERROR_MARKER = "ERROR-MARKER-hJxWqp"
# An argument name the caller made up, as a model does (#3396).
ARGUMENT_MARKER = "PRIVATE_ARGUMENT_sGtLnz"
MARKERS = (
    TOOL_MARKER, ANSWER_MARKER, TASK_MARKER, REASON_MARKER, ERROR_MARKER,
    ARGUMENT_MARKER,
)

SUMMARY = re.compile(r"chars=\d+ digest=[0-9a-f]+")


def _leaks(records):
    # SQLite worker SQL diagnostics are tracked separately in Core #3998.
    records = [r for r in records if r.name == "root" or r.name.startswith("kestrel")]
    return {
        marker: [
            f"{r.levelname} {r.name} {r.pathname}:{r.lineno}"
            for r in records_containing(records, marker)
        ]
        for marker in MARKERS
        if records_containing(records, marker)
    }


def _lines(records, prefix):
    return [r.getMessage() for r in records if r.getMessage().startswith(prefix)]


class _Tool:
    def __init__(self, name, returns):
        self.name = name
        self.schema = ToolSchema(
            name=name, description=f"fake tool {name}", category=ToolCategory.UTILITY,
        )
        self._returns = returns

    async def execute(self, **kwargs):
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


class _DenyHook(Hook):
    def __init__(self, decision: HookOutput):
        super().__init__(name="deny", events=[HookEvent.PRE_TOOL_USE])
        self._decision = decision

    async def execute(self, input: HookInput) -> HookOutput:
        return self._decision


class _LoopDoorLLM:
    """The feature's own loop runs the tool; the continuation answers."""

    def __init__(self, tool_name):
        self._tool_name = tool_name

    async def generate(self, **kwargs):
        return LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="c1", name=self._tool_name, arguments={"q": "x"})],
        )

    async def generate_with_messages(self, **kwargs):
        return LLMResponse(content=f"Found it: {ANSWER_MARKER}", tool_calls=None)


class _InlineDoorLLM:
    """The adapter runs the tool inside the call (codex app-server door)."""

    def __init__(self, tool_name):
        self._tool_name = tool_name

    async def generate(self, **kwargs):
        await kwargs["tool_executor"](self._tool_name, {"q": "x"})
        return LLMResponse(content=f"Found it: {ANSWER_MARKER}", tool_calls=None)

    async def generate_with_messages(self, **kwargs):
        return LLMResponse(content=f"Found it: {ANSWER_MARKER}", tool_calls=None)


DOORS = {"loop": _LoopDoorLLM, "inline": _InlineDoorLLM}


async def _feature(llm, tool_, hooks=(), *, native_tool_effect_agent):
    manager = HooksManager()
    for hook in hooks:
        manager.register(hook)
    host = await native_tool_effect_agent(did="did:test:probe")
    host.llm_service = llm
    host.hooks_manager = manager
    feature = _ProbeFeature(host)
    feature._fake_tools = [tool_]
    host.features[feature.name] = feature
    return feature


async def _run_feature(feature, **kwargs):
    return await drive_native(feature.agent, lambda: feature.execute_as_subagent(**kwargs))


async def _named_host(features, hooks_manager, *, native_tool_effect_agent):
    return await _make_orchestrator(features, hooks_manager,
        native_tool_effect_agent=native_tool_effect_agent)


async def _named(host, name, args, **kwargs):
    return await drive_native(host, lambda: host.execute_named_tool(name, args, **kwargs))


@pytest.fixture
def debug_caplog(caplog):
    caplog.set_level(logging.DEBUG)
    return caplog


@pytest.mark.asyncio
async def test_tool_result_task_and_final_answer_are_logged_as_summaries(debug_caplog, *, native_tool_effect_agent):
    tool_ = _Tool(
        "search_memory",
        returns={"success": True, "data": {"text": f"the heron {TOOL_MARKER}"}},
    )
    feature = (await _feature(_LoopDoorLLM(tool_.name), tool_, native_tool_effect_agent=native_tool_effect_agent))

    envelope = await _run_feature(feature, task=f"find {TASK_MARKER}")

    # The content itself still reaches the orchestrator; only the log is blind.
    assert envelope["success"] is True, envelope
    assert ANSWER_MARKER in envelope["result"]
    leaks = _leaks(debug_caplog.records)
    assert not leaks, leaks

    [result_line] = _lines(debug_caplog.records, "[SUBAGENT-TOOL] search_memory result")
    assert re.fullmatch(
        r"\[SUBAGENT-TOOL\] search_memory result \(\d+ms\): chars=\d+ digest=[0-9a-f]+",
        result_line,
    ), result_line
    [answer_line] = _lines(debug_caplog.records, "[SUBAGENT-RESULT]")
    assert SUMMARY.search(answer_line), answer_line
    [task_line] = [
        line for line in _lines(debug_caplog.records, "Feature ")
        if "executing subagent task" in line
    ]
    assert SUMMARY.search(task_line), task_line


@pytest.mark.asyncio
async def test_text_only_subagent_answer_is_not_previewed_at_debug(debug_caplog, *, native_tool_effect_agent):
    class _TextOnlyLLM:
        async def generate(self, **kwargs):
            return LLMResponse(content=f"No tools needed: {ANSWER_MARKER}", tool_calls=None)

        async def generate_with_messages(self, **kwargs):
            return LLMResponse(content=f"No tools needed: {ANSWER_MARKER}", tool_calls=None)

    feature = (await _feature(_TextOnlyLLM(), _Tool("search_memory", returns={"success": True}), native_tool_effect_agent=native_tool_effect_agent))

    envelope = await _run_feature(feature, task=f"answer {TASK_MARKER}")

    assert envelope["success"] is True, envelope
    leaks = _leaks(debug_caplog.records)
    assert not leaks, leaks
    assert any(
        "LLM returned text (no tool calls)" in line and SUMMARY.search(line)
        for line in _lines(debug_caplog.records, "Feature ")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("door", sorted(DOORS))
async def test_tool_exception_is_logged_by_class_and_summary(door, debug_caplog, *, native_tool_effect_agent):
    tool_ = _Tool("read_message_by_id", returns=ValueError(f"row {ERROR_MARKER}"))
    feature = (await _feature(DOORS[door](tool_.name), tool_, native_tool_effect_agent=native_tool_effect_agent))

    with pytest.raises(ResourceOutcomeUnknown) as caught:
        await _run_feature(feature, task='read it')
    assert caught.value.__cause__ is not None

    leaks = _leaks(debug_caplog.records)
    assert not leaks, leaks
    assert ERROR_MARKER not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("door", sorted(DOORS))
@pytest.mark.parametrize(
    "decision, label",
    [
        (HookOutput.deny(f"arg q={REASON_MARKER} refused"), "PERMISSION DENIED"),
        (HookOutput.ask("appr-1", f"approve q={REASON_MARKER}"), "APPROVAL REQUIRED"),
    ],
    ids=["deny", "ask"],
)
async def test_hook_refusal_reason_is_logged_as_summary(door, decision, label, debug_caplog, *, native_tool_effect_agent):
    """A hook's reason can echo the tool's arguments (#3873)."""
    tool_ = _Tool("shell", returns={"success": True})
    feature = (await _feature(DOORS[door](tool_.name), tool_, hooks=[_DenyHook(decision)], native_tool_effect_agent=native_tool_effect_agent))

    await _run_feature(feature, task="run it")

    leaks = _leaks(debug_caplog.records)
    assert not leaks, leaks
    [blocked] = _lines(debug_caplog.records, "[SUBAGENT-TOOL] shell blocked")
    assert blocked.startswith(f"[SUBAGENT-TOOL] shell blocked ({label}): reason chars=")


@pytest.mark.asyncio
async def test_subagent_failure_is_logged_by_class_and_summary(debug_caplog, *, native_tool_effect_agent):
    class _FailingLLM:
        async def generate(self, **kwargs):
            raise RuntimeError(f"provider echoed {ERROR_MARKER}")

    feature = (await _feature(_FailingLLM(), _Tool("search_memory", returns={"success": True}), native_tool_effect_agent=native_tool_effect_agent))

    envelope = await _run_feature(feature, task="anything")

    assert envelope["success"] is False
    leaks = _leaks(debug_caplog.records)
    assert not leaks, leaks
    assert any(
        "subagent execution failed: RuntimeError chars=" in line
        for line in _lines(debug_caplog.records, "Feature ")
    )


class _RaisingToolFeature(Feature):
    @property
    def tool_description(self) -> str:
        return "raising"

    async def initialize(self):
        return None

    @tool(name="fs_read", description="Raise with the content it read")
    async def fs_read(self) -> dict:
        raise OSError(f"could not parse {ERROR_MARKER}")


@pytest.mark.asyncio
async def test_decorated_tool_exception_is_logged_by_class_and_summary(debug_caplog):
    [wrapped] = _RaisingToolFeature(SimpleNamespace()).get_tools()

    result = await wrapped.execute()

    assert result["success"] is False and ERROR_MARKER in result["error"]
    leaks = _leaks(debug_caplog.records)
    assert not leaks, leaks
    assert "Error executing tool fs_read: OSError chars=" in debug_caplog.text


@pytest.mark.asyncio
async def test_native_task_failure_logs_frames_without_the_message(debug_caplog, native_tool_effect_agent):
    host = await native_tool_effect_agent(did="did:test:probe")
    feature = _ProbeFeature(host)
    host.features[feature.name] = feature
    feature._fake_tools = [_Tool("fs_read", returns=OSError(f"bad {ERROR_MARKER}"))]
    task = new_task(
        "task-3919", "ctx-3919", TaskState.TASK_STATE_SUBMITTED,
        history=[new_text_message("request")],
    )
    task.metadata.update({"skill": "fs_read", "args": {}})

    completed = []
    async def exercise():
        completed.append(await feature.handle_task(task))
    with pytest.raises(ResourceOutcomeUnknown):
        await drive_native(host, exercise)
    [result] = completed

    assert result.status.state == TaskState.TASK_STATE_FAILED
    leaks = _leaks(debug_caplog.records)
    assert not leaks, leaks
    [failed] = [r for r in debug_caplog.records if "task handling failed" in r.getMessage()]
    assert failed.exc_info is None
    assert "ToolEffectOutcomeUnconfirmed chars=" in failed.getMessage()
    assert "in execute_tool_effect" in failed.getMessage(), "the native boundary retains its location"


@pytest.mark.asyncio
@pytest.mark.parametrize("log_traceback", [False, True])
async def test_orchestrator_feature_error_is_logged_by_class_and_summary(
    log_traceback, debug_caplog,
):
    class _Store:
        async def log_tool_response(self, **kwargs):
            return None

    class _Hooks:
        async def execute_hooks_parallel(self, *args, **kwargs):
            return None

    host = SimpleNamespace(observability_store=_Store(), hooks_manager=_Hooks())
    # Called from inside the handler, as ``_dispatch_feature_tool`` does, so a
    # logging call that asked for ``exc_info`` would find this exception.
    try:
        raise KeyError(f"missing {ERROR_MARKER}")
    except KeyError as error:
        result = await OrchestratorEngineMixin._handle_feature_error(
            host, error, "memory_agent", "MemoryFeature", {}, 0.0, "evt-1",
            log_traceback=log_traceback,
        )

    assert ERROR_MARKER in result["error"]
    leaks = _leaks(debug_caplog.records)
    assert not leaks, leaks
    [line] = _lines(debug_caplog.records, "Feature memory_agent execution failed")
    assert "KeyError chars=" in line
    assert ("test_orchestrator_feature_error" in line) is log_traceback


def test_exception_summary_names_the_class_and_none_of_the_message():
    summary = exception_log_summary(ValueError(f"{ERROR_MARKER} the heron"))

    assert re.fullmatch(r"ValueError chars=\d+ digest=[0-9a-f]+", summary), summary
    assert ERROR_MARKER not in summary and "heron" not in summary


def test_exception_frames_carry_location_but_not_the_message():
    def _fail():
        raise RuntimeError(f"{ERROR_MARKER} the heron")

    try:
        _fail()
    except RuntimeError as error:
        frames = exception_frames(error)

    assert "_fail" in frames and __file__ in frames
    assert ERROR_MARKER not in frames
    assert "RuntimeError:" not in frames


# ---------------------------------------------------------------------------
# Orchestrator dispatch sites: hook refusals and execution failures
# ---------------------------------------------------------------------------


class _DecidingHooks:
    """PRE hooks answer ``decisions[event]`` (allow when absent); POST hooks
    are recorded and ignored."""

    def __init__(self, decisions=None):
        self.decisions = decisions or {}

    async def execute_hooks(self, event, hook_input):
        return self.decisions.get(event, HookOutput.allow())

    async def execute_hooks_parallel(self, event, hook_input):
        return None


async def _chat_host(hooks, *, native_tool_effect_agent):
    host = await native_tool_effect_agent(did="did:test:probe")
    host.hooks_manager = hooks
    return host


def _subagent_feature(side_effect=None):
    feature = MagicMock()
    feature.tool_name = "memory_feature"
    feature.name = "MemoryFeature"
    feature.enabled = True
    feature.execute_as_subagent = AsyncMock(
        return_value={"success": True}, side_effect=side_effect,
    )
    return feature


async def _dispatch_feature(host, feature):
    feature.agent = host
    host.features[feature.name] = feature
    return await drive_native(host, lambda: host._dispatch_feature_tool(
        tool_call=SimpleNamespace(name=feature.tool_name),
        feature=feature,
        args={"task": "find it"},
        dispatch_start=0.0,
        dispatch_event_id="evt-feature",
        user_message="find it",
        session_id="sess-3919",
    ))


async def _dispatch_direct(host, tool_, name="send_email", args=None):
    assert tool_.name == name
    host.register_dynamic_tools("privacy-probe", [tool_])
    return await drive_native(host, lambda: host._dispatch_direct_tool(
        tool_call=SimpleNamespace(name=name),
        tool_name=name,
        args=args or {},
        dispatch_start=0.0,
        dispatch_event_id="evt-direct",
        session_id="sess-3919",
    ))

@pytest.mark.asyncio
async def test_chat_path_subagent_refusal_reason_is_logged_as_summary(debug_caplog, *, native_tool_effect_agent):
    hooks = _DecidingHooks({
        HookEvent.PRE_SUBAGENT_CALL: HookOutput.deny(f"task={REASON_MARKER}"),
    })
    feature = _subagent_feature()

    result = await _dispatch_feature((await _chat_host(hooks, native_tool_effect_agent=native_tool_effect_agent)), feature)

    assert result["success"] is False
    feature.execute_as_subagent.assert_not_awaited()
    leaks = _leaks(debug_caplog.records)
    assert not leaks, leaks
    [line] = _lines(debug_caplog.records, "[HOOKS] Subagent blocked")
    assert "reason chars=" in line, line


@pytest.mark.asyncio
async def test_chat_path_tool_refusal_reason_is_logged_as_summary(debug_caplog, *, native_tool_effect_agent):
    hooks = _DecidingHooks({
        HookEvent.PRE_TOOL_USE: HookOutput.deny(f"to={REASON_MARKER}"),
    })
    tool_ = MagicMock()
    tool_.name = "send_email"
    tool_.execute = AsyncMock(return_value={"success": True})

    result = await _dispatch_direct((await _chat_host(hooks, native_tool_effect_agent=native_tool_effect_agent)), tool_)

    assert result["success"] is False
    tool_.execute.assert_not_awaited()
    leaks = _leaks(debug_caplog.records)
    assert not leaks, leaks
    [line] = _lines(debug_caplog.records, "[HOOKS] Tool blocked")
    assert "reason chars=" in line, line


@pytest.mark.asyncio
async def test_chat_path_direct_tool_failure_is_logged_by_class(debug_caplog, *, native_tool_effect_agent):
    tool_ = MagicMock()
    tool_.name = "send_email"
    tool_.execute = AsyncMock(side_effect=ValueError(f"bad {ERROR_MARKER}"))

    with pytest.raises(ResourceOutcomeUnknown) as caught:
        await _dispatch_direct((await _chat_host(_DecidingHooks(), native_tool_effect_agent=native_tool_effect_agent)), tool_)
    assert caught.value.__cause__ is not None

    leaks = _leaks(debug_caplog.records)
    assert not leaks, leaks
    assert ERROR_MARKER not in str(caught.value)


@pytest.mark.asyncio
async def test_chat_path_subagent_failure_keeps_frames_without_the_message(debug_caplog, *, native_tool_effect_agent):
    feature = _subagent_feature(side_effect=RuntimeError(f"echoed {ERROR_MARKER}"))

    with pytest.raises(ResourceOutcomeUnknown) as caught:
        await _dispatch_feature((await _chat_host(_DecidingHooks(), native_tool_effect_agent=native_tool_effect_agent)), feature)
    assert caught.value.__cause__ is not None

    leaks = _leaks(debug_caplog.records)
    assert not leaks, leaks
    assert ERROR_MARKER not in str(caught.value)


@pytest.mark.asyncio
async def test_governed_refusal_reasons_are_logged_as_summaries(debug_caplog, *, native_tool_effect_agent):
    host = (await _named_host(
        features={
            "email": _FakeFeature("email", [_FakeTool("send_email", returns={"ok": 1})]),
            "memory": _SubagentFeature("memory_feature", returns={"ok": 1}),
        },
        hooks_manager=_FakeHooksManager(
            pre_decision=PermissionDecision.DENY,
            pre_reason=f"arg={REASON_MARKER}",
        ),
    native_tool_effect_agent=native_tool_effect_agent))

    tool_result = await _named(host,
        "send_email", {}, session_id="s", source="codex_app_server",
    )
    subagent_result = await _named(host,
        "memory_feature", {"task": "x"}, session_id="s", source="codex_app_server",
    )

    assert tool_result["success"] is False and subagent_result["success"] is False
    leaks = _leaks(debug_caplog.records)
    assert not leaks, leaks
    blocked = _lines(debug_caplog.records, "[GOVERNED-DISPATCH] ")
    assert any("blocked (deny) source=codex_app_server tool=send_email reason chars=" in b
               for b in blocked), blocked
    assert any("subagent blocked (deny) source=codex_app_server feature=_SubagentFeature "
               "reason chars=" in b for b in blocked), blocked


@pytest.mark.asyncio
async def test_governed_subagent_failure_is_logged_by_class(debug_caplog, *, native_tool_effect_agent):
    host = (await _named_host(
        features={
            "memory": _SubagentFeature(
                "memory_feature", side_effect=RuntimeError(f"echoed {ERROR_MARKER}"),
            ),
        },
        hooks_manager=_FakeHooksManager(),
    native_tool_effect_agent=native_tool_effect_agent))

    with pytest.raises(ResourceOutcomeUnknown) as caught:
        await _named(host, 'memory_feature', {'task': 'x'}, session_id='s', source='codex_app_server')
    assert caught.value.__cause__ is not None

    leaks = _leaks(debug_caplog.records)
    assert not leaks, leaks
    assert ERROR_MARKER not in str(caught.value)


# ---------------------------------------------------------------------------
# Refused arguments: the error names what the caller sent
# ---------------------------------------------------------------------------


class _NotesFeature(Feature):
    """A real ``@tool``: its schema lists the parameters it accepts."""

    @property
    def tool_description(self) -> str:
        return "notes"

    async def initialize(self):
        return None

    @tool(name="search_notes", description="Search the notes")
    async def search_notes(self, query: str, limit: int = 5) -> dict:
        return {"success": True, "query": query, "limit": limit}


# What ``normalize_direct_tool_arguments`` tells the model, unchanged (#3396).
ARGUMENT_REFUSAL = (
    f"search_notes does not accept argument '{ARGUMENT_MARKER}'. "
    "Valid parameters: query, limit."
)
REFUSED_ARGS = {"query": "heron", ARGUMENT_MARKER: "x"}


class _AuditStore:
    def __init__(self, error=None):
        self.rows = []
        self._error = error

    async def log_decision(self, **row):
        if self._error is not None:
            raise self._error
        self.rows.append(row)


async def _notes_orchestrator(store=None, *, native_tool_effect_agent):
    features = {"notes": _NotesFeature(SimpleNamespace())}
    if store is not None:
        features["SecurityFeature"] = SimpleNamespace(
            permission_store=store, get_tools=lambda: [],
        )
    return (await _named_host(features=features, hooks_manager=_FakeHooksManager(), native_tool_effect_agent=native_tool_effect_agent))


@pytest.mark.asyncio
async def test_governed_argument_refusal_names_the_argument_to_the_model_only(
    debug_caplog,*, native_tool_effect_agent
):
    """``execute_named_tool`` logged the argument error itself and again
    through ``record_tool_rejection``."""
    store = _AuditStore()
    host = (await _notes_orchestrator(store, native_tool_effect_agent=native_tool_effect_agent))

    result = await _named(host,
        "search_notes", dict(REFUSED_ARGS), session_id="s", source="codex_app_server",
    )

    assert result == {
        "success": False, "error": f"Invalid tool arguments: {ARGUMENT_REFUSAL}",
    }
    assert host.hooks_manager.pre_calls == [], "refused before PRE_TOOL_USE"
    # The audit row is unchanged: it records the refusal as the model got it.
    [row] = store.rows
    assert row["args_summary"].startswith(ARGUMENT_REFUSAL), row
    leaks = _leaks(debug_caplog.records)
    assert not leaks, leaks
    [check] = _lines(debug_caplog.records, "[GOVERNED-DISPATCH] argument check failed")
    assert re.fullmatch(
        r"\[GOVERNED-DISPATCH\] argument check failed source=codex_app_server "
        r"tool=search_notes err=ToolArgumentError chars=\d+ digest=[0-9a-f]+",
        check,
    ), check
    [refused] = _lines(debug_caplog.records, "[TOOL-GUARDRAIL] tool_validation refused")
    assert re.fullmatch(
        r"\[TOOL-GUARDRAIL\] tool_validation refused tool 'search_notes': "
        r"reason chars=\d+ digest=[0-9a-f]+ \| allowlist .*",
        refused,
    ), refused
    # One summary of one error: the two lines correlate without its text.
    assert SUMMARY.search(check).group() == SUMMARY.search(refused).group()


@pytest.mark.asyncio
async def test_failed_refusal_audit_is_logged_as_summaries(debug_caplog, *, native_tool_effect_agent):
    host = (await _notes_orchestrator(_AuditStore(error=RuntimeError(f"row {ERROR_MARKER}")), native_tool_effect_agent=native_tool_effect_agent))

    result = await _named(host,
        "search_notes", dict(REFUSED_ARGS), session_id="s", source="codex_app_server",
    )

    assert ARGUMENT_MARKER in result["error"]
    leaks = _leaks(debug_caplog.records)
    assert not leaks, leaks
    [failed] = _lines(debug_caplog.records, "[TOOL-GUARDRAIL] failed to record refusal")
    assert re.fullmatch(
        r"\[TOOL-GUARDRAIL\] failed to record refusal of 'search_notes' "
        r"\(RuntimeError chars=\d+ digest=[0-9a-f]+\); reason chars=\d+ digest=[0-9a-f]+",
        failed,
    ), failed


@pytest.mark.asyncio
async def test_chat_path_argument_refusal_names_the_argument_to_the_model_only(
    debug_caplog,*, native_tool_effect_agent
):
    [search_notes] = _NotesFeature(SimpleNamespace()).get_tools()

    result = await _dispatch_direct(
        (await _chat_host(_DecidingHooks(), native_tool_effect_agent=native_tool_effect_agent)), search_notes,
        name="search_notes", args=dict(REFUSED_ARGS),
    )

    assert result == {"success": False, "error": ARGUMENT_REFUSAL}
    leaks = _leaks(debug_caplog.records)
    assert not leaks, leaks
    [refused] = _lines(debug_caplog.records, "[DIRECT-TOOL] search_notes refused")
    assert re.fullmatch(
        r"\[DIRECT-TOOL\] search_notes refused: ToolArgumentError "
        r"chars=\d+ digest=[0-9a-f]+",
        refused,
    ), refused


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name, args",
    [
        ("search_notes", {"query": "heron", ARGUMENT_MARKER: {1}}),
        ("memory_feature", {"task": "x", ARGUMENT_MARKER: {1}}),
    ],
    ids=["tool", "subagent"],
)
async def test_governed_argument_validation_is_logged_as_summary(
    name, args, debug_caplog,*, native_tool_effect_agent
):
    """The input guardrail's error names an argument of the wrong type."""
    host = (await _notes_orchestrator(native_tool_effect_agent=native_tool_effect_agent))
    host.features["memory"] = _SubagentFeature("memory_feature", returns={"ok": 1})

    result = await _named(host,
        name, args, session_id="s", source="codex_app_server",
    )

    assert result["success"] is False
    assert f"argument '{ARGUMENT_MARKER}' has invalid type set" in result["error"]
    # ``validate_tool_arguments`` also logs the error at its own site, outside
    # this change; this checks every record the dispatcher itself writes.
    dispatcher = [
        r for r in debug_caplog.records
        if r.name != "kestrel_sovereign.security.input_guardrails"
    ]
    leaks = _leaks(dispatcher)
    assert not leaks, leaks
    [failed] = [m for m in _lines(dispatcher, "[GOVERNED-DISPATCH] ") if "validation" in m]
    assert "err=ToolValidationError chars=" in failed, failed


@pytest.mark.asyncio
async def test_chat_path_argument_validation_is_logged_as_summary(
    dispatch_agent, debug_caplog,  # noqa: F811 - the imported fixture
):
    dispatch_agent.features = {
        "MemoryFeature": _make_feature("memory_feature", [_make_tool("recall_a")], agent=dispatch_agent),
    }

    result = await drive_native(dispatch_agent, lambda: dispatch_agent._dispatch_tool_call(
        _tool_call("recall_a", {"query": "q", ARGUMENT_MARKER: {1}}),
        dispatch_agent._visible_features_by_tool_name(),
        dispatch_agent._known_tool_names(),
        [],
        0,
        None,
    ))

    assert f"argument '{ARGUMENT_MARKER}' has invalid type set" in str(result), result
    dispatcher = [
        r for r in debug_caplog.records
        if r.name != "kestrel_sovereign.security.input_guardrails"
    ]
    leaks = _leaks(dispatcher)
    assert not leaks, leaks
    [failed] = [r.getMessage() for r in dispatcher
        if r.name == "root" and r.getMessage().startswith("[ORCHESTRATOR] Tool validation failed")]
    assert re.search(
        r"Tool validation failed: tool='recall_a' err=ToolValidationError "
        r"chars=\d+ digest=[0-9a-f]+$",
        failed,
    ), failed
