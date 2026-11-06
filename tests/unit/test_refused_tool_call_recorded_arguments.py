"""A refused call is recorded with the arguments its hooks refused (#3953).

When a PRE_TOOL_USE chain ends in DENY or ASK, the call's audit record names
the arguments as they stood when the deciding hook ran: the hook input's
``tool_input`` after any earlier MODIFY rewrite or in-place mutation,
overridden by the deciding output's own ``updated_input``. That is the binding
an executed call is already recorded with. Before #3953 the chat dispatch row
(direct, named-by-registry, governed and feature-dispatcher calls) fell back to
the call's pre-hook arguments, so a rewrite the authorization spy saw was not
the one recorded, and a redaction an earlier hook made reappeared in
``a2a_tool_dispatches.args_redacted``.

Each path is the #3921 parity harness's (temp SQLite, a real ``HooksManager``
chain), plus the chat paths it does not drive: a feature tool resolved by name,
and a feature's subagent dispatcher called from the chat loop and by name.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import pytest_asyncio
from kestrel_sdk.features.base import Feature as SdkFeature
from kestrel_sdk.hooks.base import Hook, HookEvent, HookInput, HookOutput
from kestrel_sdk.llm.response import ToolCall
from kestrel_sdk.storage.database.interface import QueryError
from kestrel_sdk.tools.base import ToolCategory

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.subagent_dispatch import ensure_subagent_dispatch
from kestrel_sovereign.hooks.manager import HooksManager
from tests.unit import test_tool_argument_binding_parity as parity

SECRET = "SECRET-3953-c41e"
REDACTED = "[REDACTED]"


class _NoteTools(parity._ProbeTools):
    """The parity probe, plus a tool whose argument a redactor can redact."""

    @tool(
        name="probe_note",
        description="Record a note.",
        category=ToolCategory.DATA_ACCESS,
        command_prefix="!probe-note",
    )
    async def probe_note(self, note: str = "") -> dict:
        self.calls.append({"note": note})
        return {"note": note}


class _BuiltinNote(_NoteTools, Feature):
    """A built-in feature: Core's ``DynamicTool`` wraps its tools."""


class _SdkNote(_NoteTools, SdkFeature):
    """An external feature: the SDK's ``DynamicTool`` wraps its tools."""


KINDS = {"builtin": _BuiltinNote, "sdk": ensure_subagent_dispatch(_SdkNote)}


class _Notes:
    """Build the path's feature from ``KINDS`` above."""

    def make_feature(self, agent):
        self.feature = KINDS[self.kind](agent)
        self.feature.error = self.error
        return self.feature


class _Direct(_Notes, parity._DirectPath):
    """A promoted tool called by the model in the chat loop."""


class _Bound(_Notes, parity._BoundPath):
    """A governed binding (``register_dynamic_tools``), as an MCP App calls one."""


class _Named(_Notes, parity._NamedPath):
    """``execute_named_tool`` through the codex inline executor."""


class _SubagentInline(_Notes, parity._SubagentInlinePath):
    """A feature subagent's model calling one of its tools inline."""


class _Registered(_Notes, parity._OrchestratorPath):
    """A feature tool the chat loop resolves by name: not promoted (#2929)."""

    async def call(self, tool_name, arguments, command):
        agent = await self._agent()
        assert tool_name not in agent._direct_tools
        return await agent._dispatch_tool_call(
            ToolCall(id="tc-1", name=tool_name, arguments=dict(arguments)),
            features_by_tool_name={},
            known_tools={tool_name},
            messages=[],
            iteration=0,
            user_message="",
            session_id=parity.SESSION,
        )


class _Dispatcher(_Notes, parity._OrchestratorPath):
    """A feature's subagent dispatcher called by the model in the chat loop."""

    async def _agent(self):
        agent = await super()._agent()
        # Only an allowed call reaches the subagent's model.
        agent.llm_service = parity._SubagentLLM("probe", {"limit": 1}, inline=False)
        agent.get_turn_bound_session_id = lambda: parity.SESSION
        return agent

    async def call(self, tool_name, arguments, command):
        agent = await self._agent()
        return await agent._dispatch_tool_call(
            ToolCall(id="tc-1", name=tool_name, arguments=dict(arguments)),
            features_by_tool_name={tool_name: self.feature},
            known_tools={tool_name},
            messages=[],
            iteration=0,
            user_message="",
            session_id=parity.SESSION,
        )


class _NamedDispatcher(_Dispatcher):
    """A feature's subagent dispatcher called through ``execute_named_tool``."""

    call = parity._NamedPath.call


class _UncapturedNamedDispatcher(_Dispatcher):
    """``execute_named_tool`` with no ``_capture``, as voice, MCP and A2A call it.

    The named path writes no dispatch row for such a caller, refused or
    allowed: its transport records the call. Only the completion hook it
    fires is this path's own audit."""

    async def call(self, tool_name, arguments, command):
        agent = await self._agent()
        return await agent.execute_named_tool(
            tool_name, dict(arguments), session_id=parity.SESSION, source="mcp",
        )


TOOL_PATHS = {
    "bound": _Bound,
    "direct": _Direct,
    "named": _Named,
    "registered": _Registered,
    "subagent_inline": _SubagentInline,
}
DISPATCHER_PATHS = {"dispatcher": _Dispatcher, "named_dispatcher": _NamedDispatcher}
#: Paths whose completion hook reports a refused call, with or without a row.
COMPLETION_PATHS = {
    **DISPATCHER_PATHS,
    "uncaptured_named_dispatcher": _UncapturedNamedDispatcher,
}

DISPATCH = {"task": f"look up {SECRET}", "context": "a note"}


@pytest_asyncio.fixture
async def make_probe(tmp_path):
    made: list = []

    def _make(path: str, kind: str):
        made.append({**TOOL_PATHS, **COMPLETION_PATHS}[path](kind, tmp_path))
        return made[-1]

    yield _make
    for made_path in made:
        await made_path.close()


class _Spy(Hook):
    """Records the arguments each hook call of ``event`` saw, as it saw them."""

    def __init__(self, event: HookEvent):
        super().__init__(name="authorized-spy", events=[event])
        self.authorized: list[dict] = []

    async def execute(self, input: HookInput) -> HookOutput:
        self.authorized.append(json.loads(json.dumps(input.tool_input)))
        return HookOutput.allow()


class _Completion(Hook):
    """Records the arguments POST_SUBAGENT_CALL reports, as an audit hook would."""

    def __init__(self):
        super().__init__(name="completion-audit", events=[HookEvent.POST_SUBAGENT_CALL])
        self.reported: list[dict] = []

    async def execute(self, input: HookInput) -> HookOutput:
        self.reported.append(json.loads(json.dumps(input.tool_input)))
        return HookOutput.allow()


class _Rewrite(Hook):
    """Rewrites ``changes`` into the arguments, by MODIFY or in place."""

    def __init__(self, changes: dict, *, in_place: bool, event: HookEvent):
        super().__init__(name="rewrite", events=[event], priority=10)
        self.changes, self.in_place = changes, in_place

    async def execute(self, input: HookInput) -> HookOutput:
        if self.in_place:
            input.tool_input.update(self.changes)
            return HookOutput.allow()
        return HookOutput.modify({**input.tool_input, **self.changes})


class _Decide(Hook):
    """Returns ``decision``, after the spy."""

    def __init__(self, decision: HookOutput, *, event: HookEvent):
        super().__init__(name="decide", events=[event], priority=200)
        self.decision = decision

    async def execute(self, input: HookInput) -> HookOutput:
        return self.decision


def _chain(*hooks_after_spy: Hook, before: Hook | None = None,
           event: HookEvent = HookEvent.PRE_TOOL_USE):
    """The path's hooks: ``before``, the authorization spy, then the rest."""

    def hooks():
        manager = HooksManager()
        if before is not None:
            manager.register(before)
        spy = _Spy(event)
        manager.register(spy)
        for hook in hooks_after_spy:
            manager.register(hook)
        return manager, spy

    return patch.object(parity, "_hooks", hooks)


async def _recorded(probe) -> list:
    """The arguments the path's audit recorded for the call."""
    if isinstance(probe, parity._SubagentInlinePath):
        return [json.loads(json.dumps(args)) for args in probe.recorded()]
    return await probe.recorded_rows()


async def _recorded_text(probe) -> str:
    """Everything the path's audit stored for the call's arguments."""
    if isinstance(probe, parity._SubagentInlinePath):
        return json.dumps(probe.recorded())
    rows = await probe.backend.fetch_all("SELECT args_redacted FROM a2a_tool_dispatches")
    return " ".join(row[0] for row in rows)


#: A fresh output per call: the manager stamps the deciding hook onto it.
REFUSALS = {
    "deny": lambda: HookOutput.deny("not today"),
    "ask": lambda: HookOutput.ask("approval-3953", "needs a person"),
}


def _refused(result) -> None:
    error = parity._refusal_text(result)
    assert "not today" in error or "needs a person" in error, error


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", sorted(REFUSALS))
@pytest.mark.parametrize("in_place", [False, True], ids=["modify", "in-place"])
@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("path", sorted(TOOL_PATHS))
async def test_a_refused_call_records_the_rewrite_the_spy_authorized(
    make_probe, path, kind, in_place, decision,
):
    """MODIFY, then the spy, then DENY or ASK: the record is what the spy saw."""
    probe = make_probe(path, kind)

    with _chain(
        _Decide(REFUSALS[decision](), event=HookEvent.PRE_TOOL_USE),
        before=_Rewrite({"limit": 7}, in_place=in_place, event=HookEvent.PRE_TOOL_USE),
    ):
        _refused(await probe.call("probe", {"limit": "30", "exact": "true"}, ""))

    assert probe.feature.calls == []
    assert probe.recorder.authorized == [{"limit": 7, "exact": True}]
    assert await _recorded(probe) == probe.recorder.authorized


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", sorted(REFUSALS))
@pytest.mark.parametrize("in_place", [False, True], ids=["modify", "in-place"])
@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("path", sorted(TOOL_PATHS))
async def test_a_redaction_a_later_hook_refused_stays_redacted(
    make_probe, path, kind, in_place, decision,
):
    probe = make_probe(path, kind)

    with _chain(
        _Decide(REFUSALS[decision](), event=HookEvent.PRE_TOOL_USE),
        before=_Rewrite({"note": REDACTED}, in_place=in_place, event=HookEvent.PRE_TOOL_USE),
    ):
        _refused(await probe.call("probe_note", {"note": SECRET}, ""))

    assert probe.feature.calls == []
    assert await _recorded(probe) == [{"note": REDACTED}]
    assert SECRET not in await _recorded_text(probe)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("path", sorted(TOOL_PATHS))
async def test_the_deciding_hooks_own_rewrite_is_what_it_refused(make_probe, path, kind):
    """A deciding output that carries ``updated_input`` refused that input."""
    probe = make_probe(path, kind)
    denial = HookOutput.deny("not today")
    denial.updated_input = {"note": REDACTED}

    with _chain(_Decide(denial, event=HookEvent.PRE_TOOL_USE)):
        _refused(await probe.call("probe_note", {"note": SECRET}, ""))

    assert await _recorded(probe) == [{"note": REDACTED}]
    assert SECRET not in await _recorded_text(probe)


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", sorted(REFUSALS))
@pytest.mark.parametrize("in_place", [False, True], ids=["modify", "in-place"])
@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("path", sorted(DISPATCHER_PATHS))
async def test_a_refused_subagent_dispatch_records_what_its_hooks_refused(
    make_probe, path, kind, in_place, decision,
):
    """The same rule for a feature's dispatcher, whose PRE_TOOL_USE chain
    runs inside its PRE_SUBAGENT_CALL envelope."""
    probe = make_probe(path, kind)

    with _chain(
        _Decide(REFUSALS[decision](), event=HookEvent.PRE_TOOL_USE),
        before=_Rewrite({"task": REDACTED}, in_place=in_place, event=HookEvent.PRE_TOOL_USE),
    ):
        _refused(await probe.call("probe_feature", dict(DISPATCH), ""))

    assert probe.feature.calls == []
    assert probe.recorder.authorized == [{**DISPATCH, "task": REDACTED}]
    assert await _recorded(probe) == probe.recorder.authorized
    assert SECRET not in await _recorded_text(probe)


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", sorted(REFUSALS))
@pytest.mark.parametrize("in_place", [False, True], ids=["modify", "in-place"])
@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("path", sorted(COMPLETION_PATHS))
async def test_a_refused_subagent_dispatch_completes_with_what_its_hooks_refused(
    make_probe, path, kind, in_place, decision,
):
    """Its POST_SUBAGENT_CALL completion hook reports the refused arguments
    too, whether or not the named path's caller captures them."""
    probe = make_probe(path, kind)
    completion = _Completion()

    with _chain(
        _Decide(REFUSALS[decision](), event=HookEvent.PRE_TOOL_USE),
        completion,
        before=_Rewrite({"task": REDACTED}, in_place=in_place, event=HookEvent.PRE_TOOL_USE),
    ):
        _refused(await probe.call("probe_feature", dict(DISPATCH), ""))

    assert probe.feature.calls == []
    assert probe.recorder.authorized == [{**DISPATCH, "task": REDACTED}]
    assert completion.reported == probe.recorder.authorized


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", sorted(REFUSALS))
@pytest.mark.parametrize("in_place", [False, True], ids=["modify", "in-place"])
@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("path", sorted(DISPATCHER_PATHS))
async def test_a_refused_subagent_call_records_what_its_hooks_refused(
    make_probe, path, kind, in_place, decision,
):
    """A PRE_SUBAGENT_CALL refusal is recorded by the same rule, in exactly
    one row."""
    probe = make_probe(path, kind)
    completion = _Completion()

    with _chain(
        _Decide(REFUSALS[decision](), event=HookEvent.PRE_SUBAGENT_CALL),
        completion,
        before=_Rewrite(
            {"task": REDACTED}, in_place=in_place, event=HookEvent.PRE_SUBAGENT_CALL,
        ),
        event=HookEvent.PRE_SUBAGENT_CALL,
    ):
        _refused(await probe.call("probe_feature", dict(DISPATCH), ""))

    assert probe.feature.calls == []
    assert probe.recorder.authorized == [{**DISPATCH, "task": REDACTED}]
    assert await _recorded(probe) == probe.recorder.authorized
    assert SECRET not in await _recorded_text(probe)
    assert SECRET not in json.dumps(completion.reported)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision", "error_class"), [("ask", "ApprovalRequired"), ("deny", "PermissionDenied")],
)
@pytest.mark.parametrize("kind", sorted(KINDS))
async def test_a_subagent_call_the_chat_loop_refused_is_recorded_as_policy_denied(
    make_probe, kind, decision, error_class,
):
    """The table has no approval status: an ASK is a ``policy_denied`` row
    that names ``ApprovalRequired``. Recorded as ``approval_required``, the
    row failed the table's CHECK and was never written (#3953)."""
    probe = make_probe("dispatcher", kind)

    with _chain(
        _Decide(REFUSALS[decision](), event=HookEvent.PRE_SUBAGENT_CALL),
        event=HookEvent.PRE_SUBAGENT_CALL,
    ):
        _refused(await probe.call("probe_feature", dict(DISPATCH), ""))

    rows = await probe.backend.fetch_all(
        "SELECT result_status, error_class FROM a2a_tool_dispatches"
    )
    assert [tuple(row) for row in rows] == [("policy_denied", error_class)]


ALL_PATHS = sorted({**TOOL_PATHS, **DISPATCHER_PATHS})


def _call_for(path: str) -> tuple[str, dict, dict, list]:
    """The tool a path calls, its arguments, the binding they record, and
    every binding the path's audit records.

    A dispatcher's subagent then makes its own ``probe`` call, recorded ahead
    of the dispatcher's envelope, which is written once the subagent returns
    (#3778).
    """
    if path in DISPATCHER_PATHS:
        return "probe_feature", dict(DISPATCH), dict(DISPATCH), [{"limit": 1}, dict(DISPATCH)]
    bound = {"limit": 30, "exact": True}
    return "probe", {"limit": "30", "exact": "true"}, bound, [bound]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("path", ALL_PATHS)
async def test_an_allowed_call_records_the_arguments_it_ran_with(make_probe, path, kind):
    """The ALLOW path is unchanged: the spy allows, the call runs, and the
    record names the binding it ran with."""
    probe = make_probe(path, kind)
    tool_name, arguments, bound, recorded = _call_for(path)

    with _chain():
        result = await probe.call(tool_name, arguments, "")

    assert result.get("success") is True, result
    # A dispatcher's subagent then authorizes its own tool call too.
    assert probe.recorder.authorized[0] == bound
    assert await _recorded(probe) == recorded


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("path", ALL_PATHS)
async def test_a_call_no_hook_sees_records_its_own_arguments(make_probe, path, kind):
    probe = make_probe(path, kind)
    tool_name, arguments, _bound, recorded = _call_for(path)

    with patch.object(parity, "_hooks", lambda: (HooksManager(), _Spy(HookEvent.PRE_TOOL_USE))):
        result = await probe.call(tool_name, arguments, "")

    assert result.get("success") is True, result
    assert await _recorded(probe) == recorded


class _BoundResponseLogFails(_Bound):
    """A governed binding whose observability response log is unavailable,
    so the binding's own fallback recorder writes the call's row."""

    async def _agent(self):
        agent = await super()._agent()

        async def unavailable(**_kwargs):
            raise RuntimeError("response log unavailable")

        agent.observability_store.log_tool_response = unavailable
        return agent


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", sorted(KINDS))
async def test_a_governed_refusal_the_dispatch_could_not_finish_keeps_its_arguments(
    tmp_path, kind,
):
    """The binding's fallback row names what the hooks refused, too."""
    probe = _BoundResponseLogFails(kind, tmp_path)
    try:
        with _chain(
            _Decide(REFUSALS["deny"](), event=HookEvent.PRE_TOOL_USE),
            before=_Rewrite({"note": REDACTED}, in_place=False, event=HookEvent.PRE_TOOL_USE),
        ):
            with pytest.raises(RuntimeError, match="response log unavailable"):
                await probe.call("probe_note", {"note": SECRET}, "")

        assert probe.feature.calls == []
        assert await _recorded(probe) == [{"note": REDACTED}]
        assert SECRET not in await _recorded_text(probe)
    finally:
        await probe.close()


#: A first response write that fails, by the two handlers the chat loop's
#: dispatcher recovers through: a backend error and a builtin one.
RESPONSE_WRITE_FAILURES = {
    "backend": lambda: QueryError("response log unavailable"),
    "builtin": lambda: ConnectionError("response log unavailable"),
}


class _DispatcherResponseLogFailsOnce(_Dispatcher):
    """A dispatcher whose first observability response write fails, so the
    chat loop's error recovery reports the refused call."""

    failure = "backend"

    async def _agent(self):
        agent = await super()._agent()
        store = agent.observability_store
        log_tool_response = store.log_tool_response
        failures = [RESPONSE_WRITE_FAILURES[self.failure]()]

        async def fails_once(**kwargs):
            if failures:
                raise failures.pop()
            return await log_tool_response(**kwargs)

        store.log_tool_response = fails_once
        return agent


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", sorted(RESPONSE_WRITE_FAILURES))
@pytest.mark.parametrize("decision", sorted(REFUSALS))
@pytest.mark.parametrize("in_place", [False, True], ids=["modify", "in-place"])
@pytest.mark.parametrize("kind", sorted(KINDS))
async def test_a_refused_dispatch_whose_response_write_failed_completes_with_what_was_refused(
    tmp_path, kind, in_place, decision, failure,
):
    """The error recovery's completion hook reports the refused arguments,
    not the call's: the redaction an earlier hook made stays made."""
    probe = _DispatcherResponseLogFailsOnce(kind, tmp_path)
    probe.failure = failure
    completion = _Completion()
    try:
        with _chain(
            _Decide(REFUSALS[decision](), event=HookEvent.PRE_TOOL_USE),
            completion,
            before=_Rewrite({"task": REDACTED}, in_place=in_place, event=HookEvent.PRE_TOOL_USE),
        ):
            result = await probe.call("probe_feature", dict(DISPATCH), "")

        assert "response log unavailable" in parity._refusal_text(result), result
        assert probe.feature.calls == []
        assert probe.recorder.authorized == [{**DISPATCH, "task": REDACTED}]
        assert completion.reported == probe.recorder.authorized
        assert await _recorded(probe) == probe.recorder.authorized
        assert SECRET not in await _recorded_text(probe)
    finally:
        await probe.close()
