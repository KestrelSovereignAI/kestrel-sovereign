"""Feature-subagent diagnostics log non-content facts only (#3919).

A subagent's inner tools mostly read the store (``search_memory``,
``read_message_by_id``, ``list_conversations``), which keeps that content
encrypted. Their results, the subagent's final answer, the task it was handed,
a hook's refusal reason and an exception's message were all logged as text or
as a 100-500 character prefix, at INFO, into a plaintext host log. Each now
logs only its length and a process-keyed digest (``content_log_summary``),
plus names, decisions and timing. An extracted feature's tools run through the
SDK's own wrapper, whose records Core summarizes the same way when it adopts
the feature.

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

from kestrel_sdk.features import base as sdk_feature_base
from kestrel_sdk.features.base import Feature as SdkFeature, tool as sdk_tool
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
from kestrel_sovereign.features.subagent_dispatch import (
    SdkToolLogSummary,
    ensure_subagent_dispatch,
)
from kestrel_sovereign.hooks.manager import HooksManager
from kestrel_sovereign.logging_config import (
    exception_frames,
    exception_log_summary,
)
from tests.unit.test_execute_named_tool import (
    _FakeFeature,
    _FakeHooksManager,
    _FakeTool,
    _MinimalOrchestrator,
    _SubagentFeature,
)
from tests.utils.log_records import records_containing

# None of these letters is a hex digit, so no digest can spell one by chance.
TOOL_MARKER = "SECRET-MARKER-qzXjvw"
ANSWER_MARKER = "ANSWER-MARKER-ruKpYt"
TASK_MARKER = "TASK-MARKER-wLmQzs"
REASON_MARKER = "REASON-MARKER-yRtNvk"
ERROR_MARKER = "ERROR-MARKER-hJxWqp"
ARGUMENT_MARKER = "ARGUMENT-MARKER-sGtLnz"
MARKERS = (
    TOOL_MARKER, ANSWER_MARKER, TASK_MARKER, REASON_MARKER, ERROR_MARKER,
    ARGUMENT_MARKER,
)
SDK_FEATURE_LOGGER = sdk_feature_base.logger

SUMMARY = re.compile(r"chars=\d+ digest=[0-9a-f]+")


def _leaks(records):
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


def _feature(llm, tool_, hooks=()):
    manager = HooksManager()
    for hook in hooks:
        manager.register(hook)
    agent = SimpleNamespace(
        llm_service=llm,
        did="did:test:probe",
        hooks_manager=manager,
        get_turn_bound_session_id=lambda: "sess-3919",
    )
    feature = _ProbeFeature(agent)
    feature._fake_tools = [tool_]
    return feature


@pytest.fixture
def debug_caplog(caplog):
    caplog.set_level(logging.DEBUG)
    return caplog


@pytest.mark.asyncio
async def test_tool_result_task_and_final_answer_are_logged_as_summaries(debug_caplog):
    tool_ = _Tool(
        "search_memory",
        returns={"success": True, "data": {"text": f"the heron {TOOL_MARKER}"}},
    )
    feature = _feature(_LoopDoorLLM(tool_.name), tool_)

    envelope = await feature.execute_as_subagent(task=f"find {TASK_MARKER}")

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
async def test_text_only_subagent_answer_is_not_previewed_at_debug(debug_caplog):
    class _TextOnlyLLM:
        async def generate(self, **kwargs):
            return LLMResponse(content=f"No tools needed: {ANSWER_MARKER}", tool_calls=None)

        async def generate_with_messages(self, **kwargs):
            return LLMResponse(content=f"No tools needed: {ANSWER_MARKER}", tool_calls=None)

    feature = _feature(_TextOnlyLLM(), _Tool("search_memory", returns={"success": True}))

    envelope = await feature.execute_as_subagent(task=f"answer {TASK_MARKER}")

    assert envelope["success"] is True, envelope
    leaks = _leaks(debug_caplog.records)
    assert not leaks, leaks
    assert any(
        "LLM returned text (no tool calls)" in line and SUMMARY.search(line)
        for line in _lines(debug_caplog.records, "Feature ")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("door", sorted(DOORS))
async def test_tool_exception_is_logged_by_class_and_summary(door, debug_caplog):
    tool_ = _Tool("read_message_by_id", returns=ValueError(f"row {ERROR_MARKER}"))
    feature = _feature(DOORS[door](tool_.name), tool_)

    await feature.execute_as_subagent(task="read it")

    leaks = _leaks(debug_caplog.records)
    assert not leaks, leaks
    [raised] = _lines(debug_caplog.records, "[SUBAGENT-TOOL] read_message_by_id raised")
    assert re.fullmatch(
        r"\[SUBAGENT-TOOL\] read_message_by_id raised ValueError "
        r"chars=\d+ digest=[0-9a-f]+",
        raised,
    ), raised


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
async def test_hook_refusal_reason_is_logged_as_summary(door, decision, label, debug_caplog):
    """A hook's reason can echo the tool's arguments (#3873)."""
    tool_ = _Tool("shell", returns={"success": True})
    feature = _feature(DOORS[door](tool_.name), tool_, hooks=[_DenyHook(decision)])

    await feature.execute_as_subagent(task="run it")

    leaks = _leaks(debug_caplog.records)
    assert not leaks, leaks
    [blocked] = _lines(debug_caplog.records, "[SUBAGENT-TOOL] shell blocked")
    assert blocked.startswith(f"[SUBAGENT-TOOL] shell blocked ({label}): reason chars=")


@pytest.mark.asyncio
async def test_subagent_failure_is_logged_by_class_and_summary(debug_caplog):
    class _FailingLLM:
        async def generate(self, **kwargs):
            raise RuntimeError(f"provider echoed {ERROR_MARKER}")

    feature = _feature(_FailingLLM(), _Tool("search_memory", returns={"success": True}))

    envelope = await feature.execute_as_subagent(task="anything")

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
async def test_native_task_failure_logs_frames_without_the_message(debug_caplog):
    feature = _ProbeFeature(SimpleNamespace())
    feature._fake_tools = [_Tool("fs_read", returns=OSError(f"bad {ERROR_MARKER}"))]
    task = new_task(
        "task-3919", "ctx-3919", TaskState.TASK_STATE_SUBMITTED,
        history=[new_text_message("request")],
    )
    task.metadata.update({"skill": "fs_read", "args": {}})

    result = await feature.handle_task(task)

    assert result.status.state == TaskState.TASK_STATE_FAILED
    leaks = _leaks(debug_caplog.records)
    assert not leaks, leaks
    [failed] = [r for r in debug_caplog.records if "task handling failed" in r.getMessage()]
    assert failed.exc_info is None
    assert "OSError chars=" in failed.getMessage()
    assert "in execute" in failed.getMessage(), "the frames still say where it failed"


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


class _ChatPathAgent(OrchestratorEngineMixin):
    def __init__(self, hooks):
        self.hooks_manager = hooks
        self._direct_tools = {}
        self._tool_to_feature = {}
        self.features = {}
        self.observability_store = MagicMock()
        self.observability_store.log_tool_response = AsyncMock()
        self._get_denied_tools = AsyncMock(return_value=set())
        self._register_explored_feature_tools = MagicMock()


def _subagent_feature(side_effect=None):
    feature = MagicMock()
    feature.tool_name = "memory_feature"
    feature.execute_as_subagent = AsyncMock(
        return_value={"success": True}, side_effect=side_effect,
    )
    return feature


async def _dispatch_feature(agent, feature):
    return await agent._dispatch_feature_tool(
        tool_call=SimpleNamespace(name=feature.tool_name),
        feature=feature,
        args={"task": "find it"},
        dispatch_start=0.0,
        dispatch_event_id="evt-feature",
        user_message="find it",
        session_id="sess-3919",
    )


async def _dispatch_direct(agent, tool_):
    agent._direct_tools["send_email"] = tool_
    return await agent._dispatch_direct_tool(
        tool_call=SimpleNamespace(name="send_email"),
        tool_name="send_email",
        args={},
        dispatch_start=0.0,
        dispatch_event_id="evt-direct",
        session_id="sess-3919",
    )


@pytest.mark.asyncio
async def test_chat_path_subagent_refusal_reason_is_logged_as_summary(debug_caplog):
    hooks = _DecidingHooks({
        HookEvent.PRE_SUBAGENT_CALL: HookOutput.deny(f"task={REASON_MARKER}"),
    })
    feature = _subagent_feature()

    result = await _dispatch_feature(_ChatPathAgent(hooks), feature)

    assert result["success"] is False
    feature.execute_as_subagent.assert_not_awaited()
    leaks = _leaks(debug_caplog.records)
    assert not leaks, leaks
    [line] = _lines(debug_caplog.records, "[HOOKS] Subagent blocked")
    assert "reason chars=" in line, line


@pytest.mark.asyncio
async def test_chat_path_tool_refusal_reason_is_logged_as_summary(debug_caplog):
    hooks = _DecidingHooks({
        HookEvent.PRE_TOOL_USE: HookOutput.deny(f"to={REASON_MARKER}"),
    })
    tool_ = MagicMock()
    tool_.execute = AsyncMock(return_value={"success": True})

    result = await _dispatch_direct(_ChatPathAgent(hooks), tool_)

    assert result["success"] is False
    tool_.execute.assert_not_awaited()
    leaks = _leaks(debug_caplog.records)
    assert not leaks, leaks
    [line] = _lines(debug_caplog.records, "[HOOKS] Tool blocked")
    assert "reason chars=" in line, line


@pytest.mark.asyncio
async def test_chat_path_direct_tool_failure_is_logged_by_class(debug_caplog):
    tool_ = MagicMock()
    tool_.execute = AsyncMock(side_effect=ValueError(f"bad {ERROR_MARKER}"))

    result = await _dispatch_direct(_ChatPathAgent(_DecidingHooks()), tool_)

    assert result["success"] is False and ERROR_MARKER in result["error"]
    leaks = _leaks(debug_caplog.records)
    assert not leaks, leaks
    [line] = _lines(debug_caplog.records, "[DIRECT-TOOL] send_email failed")
    assert "ValueError chars=" in line, line


@pytest.mark.asyncio
async def test_chat_path_subagent_failure_keeps_frames_without_the_message(debug_caplog):
    feature = _subagent_feature(side_effect=RuntimeError(f"echoed {ERROR_MARKER}"))

    result = await _dispatch_feature(_ChatPathAgent(_DecidingHooks()), feature)

    assert result["success"] is False and ERROR_MARKER in result["error"]
    leaks = _leaks(debug_caplog.records)
    assert not leaks, leaks
    [line] = _lines(debug_caplog.records, "Feature memory_feature execution failed")
    assert "RuntimeError chars=" in line and "File " in line, line


@pytest.mark.asyncio
async def test_governed_refusal_reasons_are_logged_as_summaries(debug_caplog):
    agent = _MinimalOrchestrator(
        features={
            "email": _FakeFeature("email", [_FakeTool("send_email", returns={"ok": 1})]),
            "memory": _SubagentFeature("memory_feature", returns={"ok": 1}),
        },
        hooks_manager=_FakeHooksManager(
            pre_decision=PermissionDecision.DENY,
            pre_reason=f"arg={REASON_MARKER}",
        ),
    )

    tool_result = await agent.execute_named_tool(
        "send_email", {}, session_id="s", source="codex_app_server",
    )
    subagent_result = await agent.execute_named_tool(
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
async def test_governed_subagent_failure_is_logged_by_class(debug_caplog):
    agent = _MinimalOrchestrator(
        features={
            "memory": _SubagentFeature(
                "memory_feature", side_effect=RuntimeError(f"echoed {ERROR_MARKER}"),
            ),
        },
        hooks_manager=_FakeHooksManager(),
    )

    result = await agent.execute_named_tool(
        "memory_feature", {"task": "x"}, session_id="s", source="codex_app_server",
    )

    assert result["success"] is False and ERROR_MARKER in result["error"]
    leaks = _leaks(debug_caplog.records)
    assert not leaks, leaks
    [line] = _lines(debug_caplog.records, "[GOVERNED-DISPATCH] subagent execute raised")
    assert "err=RuntimeError chars=" in line and "File " in line, line


# ---------------------------------------------------------------------------
# Extracted features: the SDK's own tool wrapper
# ---------------------------------------------------------------------------


class _SdkRaisingFeature(SdkFeature):
    """Built on the public SDK base, as a pip-installed feature is."""

    tool_name = "sdk_probe"

    @property
    def tool_description(self) -> str:
        return "sdk probe"

    async def initialize(self) -> None:
        return None

    @sdk_tool(name="fs_read", description="Raise with the content it read")
    async def fs_read(self, q: str = "", limit: int = 10) -> dict:
        raise OSError(f"could not parse {ERROR_MARKER}")


@pytest.fixture
def bare_sdk_logger():
    """The SDK feature logger as a fresh process has it: no summary filter.

    The filter is process-wide once installed, so without this an earlier test
    could install it and hide a door that never does.
    """
    ours = [f for f in SDK_FEATURE_LOGGER.filters if isinstance(f, SdkToolLogSummary)]
    for installed in ours:
        SDK_FEATURE_LOGGER.removeFilter(installed)
    yield SDK_FEATURE_LOGGER
    for installed in [
        f for f in SDK_FEATURE_LOGGER.filters if isinstance(f, SdkToolLogSummary)
    ]:
        SDK_FEATURE_LOGGER.removeFilter(installed)
    for installed in ours:
        SDK_FEATURE_LOGGER.addFilter(installed)


def _sdk_feature(llm):
    cls = ensure_subagent_dispatch(_SdkRaisingFeature)
    return cls(SimpleNamespace(
        llm_service=llm,
        did="did:test:probe",
        hooks_manager=HooksManager(),
        get_turn_bound_session_id=lambda: "sess-3919",
    ))


@pytest.mark.asyncio
@pytest.mark.parametrize("door", sorted(DOORS))
async def test_sdk_feature_tool_exception_is_logged_by_class_and_summary(
    door, bare_sdk_logger, debug_caplog,
):
    """An extracted feature keeps the SDK's ``get_tools()`` wrapper, which
    catches the tool's exception and logs it itself, before Core sees it."""
    feature = _sdk_feature(DOORS[door]("fs_read"))

    await feature.execute_as_subagent(task="read it")

    leaks = _leaks(debug_caplog.records)
    assert not leaks, leaks
    [line] = _lines(debug_caplog.records, "Error executing tool fs_read")
    assert re.fullmatch(
        r"Error executing tool fs_read: OSError chars=\d+ digest=[0-9a-f]+", line,
    ), line


@pytest.mark.asyncio
async def test_sdk_feature_tool_results_are_the_sdks(bare_sdk_logger, debug_caplog):
    """Only the log records change: the model still gets the SDK's envelopes."""
    [wrapped] = _sdk_feature(None).get_tools()

    failed = await wrapped.execute()
    rejected = await wrapped.execute(limit=f"{ARGUMENT_MARKER} the heron")

    assert failed == {
        "success": False,
        "error": f"could not parse {ERROR_MARKER}",
        "tool": "fs_read",
    }
    assert rejected["tool"] == "fs_read" and ARGUMENT_MARKER in rejected["error"]
    leaks = _leaks(debug_caplog.records)
    assert not leaks, leaks
    [line] = _lines(debug_caplog.records, "Rejected tool call")
    assert re.fullmatch(
        r"Rejected tool call — fs_read: parameter 'limit' expects type 'integer', "
        r"but received chars=\d+ digest=[0-9a-f]+",
        line,
    ), line


def _sdk_wrapper_record(msg, args=()):
    return logging.LogRecord(
        SDK_FEATURE_LOGGER.name, logging.ERROR, "base.py", 1, msg, args, None,
        func="execute",
    )


@pytest.mark.parametrize(
    "msg, args",
    [
        # No active exception, so nothing says where the tool name ends.
        (f"Error executing tool fs_read: {ERROR_MARKER}", ()),
        # A rejection the SDK words differently than the filter expects.
        ("Rejected tool call — %s", (f"fs_read: {ARGUMENT_MARKER} is not a list",)),
        # Any other record from the wrapper.
        ("Coerced %s", (ARGUMENT_MARKER,)),
    ],
    ids=["error-without-exception", "rejection-reworded", "unknown"],
)
def test_sdk_wrapper_record_of_unknown_shape_is_withheld(msg, args):
    record = _sdk_wrapper_record(msg, args)

    SdkToolLogSummary().filter(record)

    assert not _leaks([record])
    assert re.fullmatch(
        r"SDK tool wrapper record withheld: chars=\d+ digest=[0-9a-f]+",
        record.getMessage(),
    ), record.getMessage()


def test_sdk_wrapper_summary_is_idempotent():
    """Filtering twice (two installs, or a handler copy) keeps the summary."""
    summary = SdkToolLogSummary()
    try:
        raise ValueError(f"{ERROR_MARKER} the heron")
    except ValueError as error:
        record = _sdk_wrapper_record(f"Error executing tool fs_read: {error}")
        summary.filter(record)
        once = record.getMessage()
        summary.filter(record)

    assert record.getMessage() == once
    assert once.startswith("Error executing tool fs_read: ValueError chars=")


def test_sdk_wrapper_exception_text_leaves_record_frames():
    """A wrapper that logged with ``exc_info`` would render ``Type: message``."""
    try:
        raise ValueError(f"{ERROR_MARKER} the heron")
    except ValueError as error:
        record = _sdk_wrapper_record(f"Error executing tool fs_read: {error}")
        record.exc_info = (type(error), error, error.__traceback__)
        SdkToolLogSummary().filter(record)

    assert record.exc_info is None and record.exc_text is None
    assert not _leaks([record])
    assert "test_sdk_wrapper_exception_text_leaves_record_frames" in record.getMessage()


def test_adopting_sdk_features_installs_one_summary_filter(bare_sdk_logger):
    ensure_subagent_dispatch(_SdkRaisingFeature)
    ensure_subagent_dispatch(_SdkRaisingFeature)

    assert [
        type(f) for f in bare_sdk_logger.filters if isinstance(f, SdkToolLogSummary)
    ] == [SdkToolLogSummary]
