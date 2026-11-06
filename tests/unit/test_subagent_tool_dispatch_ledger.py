"""A feature subagent's own tool calls are in the dispatch log (#3778).

``a2a_tool_dispatches`` (read by ``kestrel tool-dispatches``) recorded a
feature-as-subagent dispatch only as its envelope. The calls the subagent made
inside it reached ``security_audit_log`` and nothing else, so when Claw's Talon
coordinator subagent ran a ``talon_claim`` and then a ``talon_cancel``, two
people searching the dispatch log concluded the job had been cancelled by
nobody.

Every call a subagent makes now writes its own row, whatever its outcome, and
names the envelope that delegated to it: its ``parent_dispatch_id`` is the
envelope row's ``dispatch_id``. Covered on both ways a subagent runs its tools
(its own post-LLM loop, and the inline executor the codex app-server calls) and
every way an envelope is dispatched: the chat loop, and ``execute_named_tool``,
which the codex inline executor and the other transports call. A named
envelope writes its own row, so it exists without an inline caller and when
the codex turn fails before recording its executed calls; the inline recorder
does not write it twice. A cancellation after a call started is recorded, as an
interrupted call, on the call and on its envelope.

A write the ledger refuses is never taken for a row. Native storage refuses
one once execution authority is revoked, and nothing grants it authority to
write after that. The refusal is reported once, as a WARNING naming the row,
the row is not attempted again, and the original error is raised unchanged.
A write whose COMMIT outcome is unknown, or that left a native resource
unretired, is not taken for a row either, but its evidence is never dropped:
the caller receives it, with the original error or cancellation retained in
it, even for a best-effort row.
"""

from __future__ import annotations

import asyncio
import copy
import json
import pickle
import re
from types import SimpleNamespace

import pytest
import pytest_asyncio
from kestrel_sdk.hooks.base import Hook, HookEvent, HookInput, HookOutput
from kestrel_sdk.llm.response import LLMResponse, ToolCall
from kestrel_sdk.tools import DeferredToolResult, ToolResult
from kestrel_sdk.tools.base import ToolCategory

from kestrel_sovereign.a2a.stores.unified.observability_store import (
    ObservabilityStore,
    ToolDispatchEntry,
)
from kestrel_sovereign.agent import orchestrator_engine
from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.agent.tool_registry import ToolRegistryMixin
from kestrel_sovereign.agent.turn_lifecycle import TurnLifecycleMixin
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    ExecutionCommitOutcomeError,
    ExecutionCustody,
    ToolAuditUnconfirmedError,
    bind_execution_custody_snapshot,
    execution_cleanup_debts,
    execution_commit_outcome,
    execution_evidence_carriers,
    is_execution_control_error,
)
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.hooks.manager import HooksManager
from kestrel_sovereign.llm.adapter import InlineToolOutcome, inline_tool_dispatch_id
from kestrel_sovereign.llm.codex_adapter import CodexAdapter
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

SESSION = "sess-3778"
AGENT_DID = "did:test:claw"
DISPATCHER = "talon_coordinator_feature"
CLAIM = ("talon_claim", {"repo": "KestrelSovereignAI/kestrel-sovereign", "issue": 3664})
CANCEL = ("talon_cancel", {"job_id": "692b13dd"})


class _TalonProbe(Feature):
    """A feature whose subagent launches a job and then cancels it."""

    tool_name = DISPATCHER

    def __init__(self, agent):
        super().__init__(agent)
        self.calls: list[tuple] = []
        self.error: BaseException | None = None
        self.hold_cancel = False
        self.cancel_entered = asyncio.Event()
        # What cancelling the job does, when a test needs a real effect.
        self.cancel_effect = None

    @property
    def tool_description(self) -> str:
        return "Coordinate Talon jobs."

    async def initialize(self) -> None:
        return None

    @tool(name="talon_claim", description="Claim an issue.", category=ToolCategory.UTILITY)
    async def talon_claim(self, repo: str, issue: int) -> dict:
        self.calls.append(("talon_claim", repo, issue))
        if self.error is not None:
            raise self.error
        return {"job_id": "692b13dd"}

    @tool(name="talon_cancel", description="Cancel a job.", category=ToolCategory.UTILITY)
    async def talon_cancel(self, job_id: str) -> dict:
        self.calls.append(("talon_cancel", job_id))
        if self.cancel_effect is not None:
            await self.cancel_effect(job_id)
        if self.hold_cancel:
            # The job is cancelled; the call has not returned yet.
            self.cancel_entered.set()
            await asyncio.Event().wait()
        return {"cancelled": job_id}


class _Spy(Hook):
    """Records every tool-use edge, and the rows already written at POST."""

    def __init__(self, backend):
        super().__init__(
            name="edge-spy", events=[HookEvent.PRE_TOOL_USE, HookEvent.POST_TOOL_USE],
        )
        self.backend = backend
        self.edges: list[tuple[str, str]] = []
        self.rows_at_post: dict[str, int] = {}

    async def execute(self, input: HookInput) -> HookOutput:
        self.edges.append((input.hook_event_name, input.tool_name))
        if input.hook_event_name == HookEvent.POST_TOOL_USE.value:
            row = await self.backend.fetch_one(
                "SELECT COUNT(*) FROM a2a_tool_dispatches WHERE tool_name = ?",
                (input.tool_name,),
            )
            self.rows_at_post[input.tool_name] = row[0]
        return HookOutput.allow()


class _Decide(Hook):
    """A PRE_TOOL_USE hook answering one tool's call with ``decision``."""

    def __init__(self, tool_name: str, decision: HookOutput):
        super().__init__(name="decide", events=[HookEvent.PRE_TOOL_USE], priority=10)
        self.tool_name = tool_name
        self.decision = decision

    async def execute(self, input: HookInput) -> HookOutput:
        if input.tool_name == self.tool_name:
            return self.decision
        return HookOutput.allow()


class _Agent(OrchestratorEngineMixin, ToolRegistryMixin, TurnLifecycleMixin):
    def __init__(self, store, hooks):
        self.did = AGENT_DID
        self.features = {}
        self._direct_tools, self._tool_to_feature, self._explored_features = {}, {}, {}
        self._direct_tool_defs, self._pinned_features = [], set()
        self.hooks_manager, self.observability_store = hooks, store

    def get_turn_bound_session_id(self):
        return SESSION


class _SubagentLLM:
    """The subagent's model: ``calls`` in one response, then a final answer.

    ``inline`` runs them as the codex app-server does, inside the model call
    through ``tool_executor``.
    """

    def __init__(self, calls, *, inline: bool):
        self.calls = calls
        self.inline = inline

    async def generate(self, **kwargs):
        if self.inline:
            for name, arguments in self.calls:
                await kwargs["tool_executor"](name, dict(arguments))
            return LLMResponse(content="done", tool_calls=None)
        return LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(id=f"c{index}", name=name, arguments=dict(arguments))
                for index, (name, arguments) in enumerate(self.calls)
            ],
        )

    async def generate_with_messages(self, **kwargs):
        return LLMResponse(content="done", tool_calls=None)


class _Host:
    def __init__(self, agent, feature, backend, spy):
        self.agent, self.feature, self.backend, self.spy = agent, feature, backend, spy

    async def rows(self) -> list[dict]:
        columns = (
            "tool_name", "adapter", "result_status", "error_class", "error_message",
            "dispatch_id", "parent_dispatch_id", "turn_id", "session_id",
            "args_redacted",
        )
        rows = await self.backend.fetch_all(
            f"SELECT {', '.join(columns)} FROM a2a_tool_dispatches ORDER BY id"
        )
        return [dict(zip(columns, row)) for row in rows]


@pytest_asyncio.fixture
async def make_host(tmp_path):
    stores = []

    async def _make(*, calls=(CLAIM, CANCEL), inline=False, decide=None) -> _Host:
        backend = SQLiteBackend(str(tmp_path / f"dispatch-{len(stores)}.db"))
        await backend.connect()
        store = ObservabilityStore(backend)
        await store.initialize()
        stores.append(store)
        hooks = HooksManager()
        if decide is not None:
            hooks.register(_Decide(*decide))
        spy = _Spy(backend)
        hooks.register(spy)
        agent = _Agent(store, hooks)
        agent.llm_service = _SubagentLLM(calls, inline=inline)
        feature = _TalonProbe(agent)
        agent.features[type(feature).__name__] = feature
        return _Host(agent, feature, backend, spy)

    yield _make
    for store in stores:
        await store.close()


def _envelope_and_inner(rows: list[dict]) -> tuple[dict, list[dict]]:
    [envelope] = [row for row in rows if row["tool_name"] == DISPATCHER]
    inner = [row for row in rows if row["tool_name"] != DISPATCHER]
    return envelope, inner


def _assert_inner_calls_name_their_envelope(rows, envelope, inner, spy):
    assert envelope["parent_dispatch_id"] is None
    assert envelope["dispatch_id"]
    assert [row["tool_name"] for row in inner] == ["talon_claim", "talon_cancel"]
    for row in inner:
        assert row["parent_dispatch_id"] == envelope["dispatch_id"]
        assert row["adapter"] == f"subagent:_TalonProbe.{row['tool_name']}"
        assert row["result_status"] == "success"
        assert row["turn_id"] == envelope["turn_id"]
        assert row["session_id"] == SESSION
    assert json.loads(inner[0]["args_redacted"]) == CLAIM[1]
    assert json.loads(inner[1]["args_redacted"]) == CANCEL[1]
    # Every row is its own: a parent id names exactly one row.
    assert len({row["dispatch_id"] for row in rows}) == len(rows)
    # #3612's pairing stands, and the row follows POST_TOOL_USE.
    for name in ("talon_claim", "talon_cancel"):
        assert spy.edges.count((HookEvent.PRE_TOOL_USE.value, name)) == 1
        assert spy.edges.count((HookEvent.POST_TOOL_USE.value, name)) == 1
        assert spy.rows_at_post[name] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("inline", [False, True], ids=["loop", "inline"])
async def test_chat_loop_envelope_names_the_calls_its_subagent_made(make_host, inline):
    host = await make_host(inline=inline)
    async with host.agent._turn_lifecycle():
        result = await host.agent._dispatch_tool_call(
            ToolCall(id="tc-1", name=DISPATCHER, arguments={"task": "claim, then cancel"}),
            features_by_tool_name={DISPATCHER: host.feature},
            known_tools={DISPATCHER},
            messages=[],
            iteration=0,
            user_message="",
            session_id=SESSION,
        )

    assert result["success"] is True, result
    assert host.feature.calls == [
        ("talon_claim", CLAIM[1]["repo"], 3664), ("talon_cancel", "692b13dd"),
    ]
    rows = await host.rows()
    envelope, inner = _envelope_and_inner(rows)
    assert envelope["adapter"] == "_TalonProbe.execute_as_subagent"
    _assert_inner_calls_name_their_envelope(rows, envelope, inner, host.spy)


@pytest.mark.asyncio
async def test_two_envelopes_in_one_turn_each_name_their_own_calls(make_host):
    host = await make_host(calls=(CLAIM,))
    async with host.agent._turn_lifecycle():
        for call_id in ("tc-1", "tc-2"):
            await host.agent._dispatch_tool_call(
                ToolCall(id=call_id, name=DISPATCHER, arguments={"task": "claim"}),
                features_by_tool_name={DISPATCHER: host.feature},
                known_tools={DISPATCHER},
                messages=[],
                iteration=0,
                user_message="",
                session_id=SESSION,
            )

    rows = await host.rows()
    assert [row["tool_name"] for row in rows] == [
        "talon_claim", DISPATCHER, "talon_claim", DISPATCHER,
    ]
    first_claim, first_envelope, second_claim, second_envelope = rows
    assert first_envelope["dispatch_id"] != second_envelope["dispatch_id"]
    assert first_claim["parent_dispatch_id"] == first_envelope["dispatch_id"]
    assert second_claim["parent_dispatch_id"] == second_envelope["dispatch_id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("inline", [False, True], ids=["loop", "inline"])
async def test_codex_inline_envelope_names_the_calls_its_subagent_made(make_host, inline):
    """The issue's case: the envelope itself ran inside a codex turn, and its
    row is written from the adapter's executed-call log after the turn."""
    host = await make_host(inline=inline)
    executed: list[dict] = []
    async with host.agent._turn_lifecycle():
        handler = CodexAdapter()._make_tool_call_handler(
            host.agent._make_inline_tool_executor(SESSION),
            "t", frozenset({DISPATCHER}), executed_log=executed,
        )
        response = await handler({
            "threadId": "t", "callId": "call-1", "name": DISPATCHER,
            "arguments": {"task": "claim, then cancel"},
        })
        await host.agent._append_executed_tool_breadcrumbs(
            [], executed, session_id=SESSION,
        )

    assert response["success"] is True, response
    rows = await host.rows()
    envelope, inner = _envelope_and_inner(rows)
    # The named path writes it, as it writes a codex call's deferred row.
    assert envelope["adapter"] == "named:codex_app_server:_TalonProbe.execute_as_subagent"
    [entry] = executed
    assert entry["dispatch_id"] == envelope["dispatch_id"]
    _assert_inner_calls_name_their_envelope(rows, envelope, inner, host.spy)



async def _dispatch(host, door, arguments):
    """Dispatch the envelope through ``door`` inside a turn.

    ``chat`` is the chat loop; ``named`` is ``execute_named_tool`` as a
    transport with no inline caller (voice, MCP) calls it.
    """
    async with host.agent._turn_lifecycle():
        if door == "chat":
            return await host.agent._dispatch_tool_call(
                ToolCall(id="tc-1", name=DISPATCHER, arguments=arguments),
                features_by_tool_name={DISPATCHER: host.feature},
                known_tools={DISPATCHER},
                messages=[],
                iteration=0,
                user_message="",
                session_id=SESSION,
            )
        return await host.agent.execute_named_tool(
            DISPATCHER, arguments, session_id=SESSION, source="voice_realtime",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("inline", [False, True], ids=["loop", "inline"])
async def test_a_named_envelope_with_no_inline_caller_records_itself(make_host, inline):
    """Before, the subagent's rows named a parent no row had: nothing records
    a named call when no inline adapter is there to."""
    host = await make_host(inline=inline)

    result = await _dispatch(host, "named", {"task": "claim, then cancel"})

    # Callers outside the inline executor get the plain envelope.
    assert type(result) is dict and result["success"] is True, result
    rows = await host.rows()
    envelope, inner = _envelope_and_inner(rows)
    assert envelope["adapter"] == "named:voice_realtime:_TalonProbe.execute_as_subagent"
    assert envelope["result_status"] == "success"
    _assert_inner_calls_name_their_envelope(rows, envelope, inner, host.spy)


@pytest.mark.asyncio
async def test_a_codex_envelope_is_recorded_when_its_turn_fails_before_its_breadcrumbs(
    make_host,
):
    """The codex inline executor ran the envelope, then the provider failed
    before returning its response, so the adapter's executed-call log is
    never recorded. The envelope's row is already written."""
    host = await make_host()
    executed: list[dict] = []
    async with host.agent._turn_lifecycle():
        handler = CodexAdapter()._make_tool_call_handler(
            host.agent._make_inline_tool_executor(SESSION),
            "t", frozenset({DISPATCHER}), executed_log=executed,
        )
        await handler({
            "threadId": "t", "callId": "call-1", "name": DISPATCHER,
            "arguments": {"task": "claim, then cancel"},
        })

    rows = await host.rows()
    envelope, inner = _envelope_and_inner(rows)
    [entry] = executed
    assert entry["dispatch_id"] == envelope["dispatch_id"]
    _assert_inner_calls_name_their_envelope(rows, envelope, inner, host.spy)


class _DecideSubagent(Hook):
    """A PRE_SUBAGENT_CALL hook refusing every subagent dispatch."""

    def __init__(self, decision: HookOutput):
        super().__init__(name="decide-subagent", events=[HookEvent.PRE_SUBAGENT_CALL])
        self.decision = decision

    async def execute(self, input: HookInput) -> HookOutput:
        return self.decision


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "refuse", "expected"),
    [
        pytest.param(
            {"task": "x" * 10_001}, None,
            ("error", "ToolValidationError"), id="invalid-arguments",
        ),
        pytest.param(
            {"task": "claim"}, ("subagent", HookOutput.deny("no delegation")),
            ("policy_denied", "PermissionDenied"), id="subagent-denied",
        ),
        pytest.param(
            {"task": "claim"}, ("tool", HookOutput.ask("approval-1", "ask first")),
            ("policy_denied", "ApprovalRequired"), id="tool-asked",
        ),
    ],
)
async def test_a_refused_codex_envelope_is_recorded_once(make_host, arguments, refuse, expected):
    host = await make_host()
    if refuse is not None:
        gate, decision = refuse
        host.agent.hooks_manager.register(
            _DecideSubagent(decision) if gate == "subagent"
            else _Decide(DISPATCHER, decision)
        )
    executed: list[dict] = []
    async with host.agent._turn_lifecycle():
        handler = CodexAdapter()._make_tool_call_handler(
            host.agent._make_inline_tool_executor(SESSION),
            "t", frozenset({DISPATCHER}), executed_log=executed,
        )
        await handler({
            "threadId": "t", "callId": "call-1", "name": DISPATCHER,
            "arguments": arguments,
        })
        await host.agent._append_executed_tool_breadcrumbs(
            [], executed, session_id=SESSION,
        )

    [row] = await host.rows()
    assert row["tool_name"] == DISPATCHER
    assert (row["result_status"], row["error_class"]) == expected
    assert row["dispatch_id"] == executed[0]["dispatch_id"]
    assert host.feature.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("inline", [False, True], ids=["loop", "inline"])
@pytest.mark.parametrize("door", ["chat", "named"])
async def test_a_call_cancelled_after_its_effect_is_recorded_under_its_envelope(
    make_host, door, inline,
):
    """The reviewer's case: the job is cancelled, then the turn is, before the
    call returns. The call and its envelope are both recorded as interrupted,
    and the cancellation still ends the dispatch."""
    host = await make_host(inline=inline)
    host.feature.hold_cancel = True

    dispatch = asyncio.create_task(_dispatch(host, door, {"task": "claim, then cancel"}))
    await asyncio.wait_for(host.feature.cancel_entered.wait(), 5)
    dispatch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await dispatch

    assert host.feature.calls[-1] == ("talon_cancel", "692b13dd")
    envelope, inner = _envelope_and_inner(await host.rows())
    claim, cancel = inner
    assert (claim["tool_name"], claim["result_status"]) == ("talon_claim", "success")
    assert (cancel["tool_name"], cancel["result_status"], cancel["error_class"]) == (
        "talon_cancel", "error", "CancelledError",
    )
    assert "while the call was running" in cancel["error_message"]
    assert json.loads(cancel["args_redacted"]) == CANCEL[1]
    for row in inner:
        assert row["parent_dispatch_id"] == envelope["dispatch_id"]
        assert row["session_id"] == SESSION
    assert (envelope["result_status"], envelope["error_class"]) == ("error", "CancelledError")


class _HoldPost(Hook):
    """Holds one tool's POST_TOOL_USE: the call returned, its row is not yet
    written."""

    def __init__(self, tool_name: str):
        super().__init__(name="hold-post", events=[HookEvent.POST_TOOL_USE])
        self.tool_name = tool_name
        self.entered = asyncio.Event()

    async def execute(self, input: HookInput) -> HookOutput:
        if input.tool_name == self.tool_name:
            self.entered.set()
            await asyncio.Event().wait()
        return HookOutput.allow()


@pytest.mark.asyncio
async def test_a_call_cancelled_after_it_returned_says_so(make_host):
    host = await make_host(calls=(CANCEL,))
    hold = _HoldPost("talon_cancel")
    host.agent.hooks_manager.register(hold)

    dispatch = asyncio.create_task(_dispatch(host, "named", {"task": "cancel"}))
    await asyncio.wait_for(hold.entered.wait(), 5)
    dispatch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await dispatch

    envelope, [cancel] = _envelope_and_inner(await host.rows())
    assert (cancel["result_status"], cancel["error_class"]) == ("error", "CancelledError")
    assert "after the call returned" in cancel["error_message"]
    assert cancel["parent_dispatch_id"] == envelope["dispatch_id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("refused", [False, True], ids=["written", "refused"])
@pytest.mark.parametrize("target", ["talon_cancel", DISPATCHER], ids=["call", "envelope"])
async def test_a_cancellation_during_a_rows_write_leaves_exactly_that_row(
    make_host, monkeypatch, caplog, target, refused,
):
    """The write is owned: the cancellation waits for it, so the call is not
    recorded a second time as interrupted. A refused write was the row's one
    attempt too: it is reported once, and nothing records it again."""
    host = await make_host()
    store = host.agent.observability_store
    write = store.log_tool_dispatch
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow(entry, **kwargs):
        if entry.tool_name == target and not entered.is_set():
            entered.set()
            await release.wait()
            if refused:
                raise OSError("disk I/O error")
        await write(entry, **kwargs)

    monkeypatch.setattr(store, "log_tool_dispatch", slow)
    interrupted = _spy_interrupted_records(monkeypatch)
    dispatch = asyncio.create_task(_dispatch(host, "named", {"task": "claim, then cancel"}))
    await asyncio.wait_for(entered.wait(), 5)
    dispatch.cancel()
    for _ in range(5):
        await asyncio.sleep(0)
    assert not dispatch.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await dispatch

    rows = await host.rows()
    assert target not in [entry["tool_name"] for entry in interrupted]
    reports = _unwritten_reports(caplog)
    assert [report["tool_name"] for report in reports] == ([target] if refused else [])
    statuses = [(row["tool_name"], row["result_status"]) for row in rows]
    if target == "talon_cancel":
        assert statuses[:-1] == [("talon_claim", "success")] + (
            [] if refused else [("talon_cancel", "success")]
        )
        envelope = rows[-1]
        assert (envelope["tool_name"], envelope["result_status"], envelope["error_class"]) == (
            DISPATCHER, "error", "CancelledError",
        )
    else:
        assert statuses == [("talon_claim", "success"), ("talon_cancel", "success")] + (
            [] if refused else [(DISPATCHER, "success")]
        )


@pytest.mark.asyncio
async def test_a_repeated_cancellation_does_not_cut_the_interrupted_row_short(
    make_host, monkeypatch,
):
    host = await make_host()
    host.feature.hold_cancel = True
    store = host.agent.observability_store
    write = store.log_tool_dispatch
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow(entry, **kwargs):
        if entry.tool_name == "talon_cancel":
            entered.set()
            await release.wait()
        await write(entry, **kwargs)

    monkeypatch.setattr(store, "log_tool_dispatch", slow)
    dispatch = asyncio.create_task(_dispatch(host, "named", {"task": "claim, then cancel"}))
    await asyncio.wait_for(host.feature.cancel_entered.wait(), 5)
    dispatch.cancel()
    await asyncio.wait_for(entered.wait(), 5)
    dispatch.cancel()
    for _ in range(5):
        await asyncio.sleep(0)
    assert not dispatch.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await dispatch

    envelope, [_claim, cancel] = _envelope_and_inner(await host.rows())
    assert (cancel["result_status"], cancel["error_class"]) == ("error", "CancelledError")
    assert cancel["parent_dispatch_id"] == envelope["dispatch_id"]


class _LiveRuntimeFence:
    """The runtime fence a SQLite host's execution custody is bound to.

    Live until its ``ExecutionCustody`` is revoked: native SQLite storage then
    refuses every write made under that custody.
    """

    backend_type = "sqlite"

    def require_work(self) -> None:
        return None

    async def lock_and_validate(self, _connection) -> None:
        self.require_work()


_UNWRITTEN = re.compile(
    r"a2a_tool_dispatches row not written: dispatch_id=(\S+) tool=(\S+) "
    r"parent_dispatch_id=(\S+) cause=(\w+)"
)


def _unwritten_reports(caplog) -> list[dict]:
    """Every WARNING that a dispatch row was not written, parsed."""
    reports = []
    for record in caplog.records:
        match = _UNWRITTEN.search(record.getMessage())
        if match is not None:
            assert record.levelname == "WARNING"
            dispatch_id, tool_name, parent, cause = match.groups()
            reports.append({
                "dispatch_id": dispatch_id, "tool_name": tool_name,
                "parent_dispatch_id": None if parent == "None" else parent,
                "cause": cause,
            })
    return reports


def _spy_interrupted_records(monkeypatch) -> list[dict]:
    """Record what each interrupted-call record was given and raised, and
    whether it reported its row written (``on_written``)."""
    records = []
    record = orchestrator_engine._record_interrupted_dispatch

    async def spy(host, error, **kwargs):
        entry = {"tool_name": kwargs["tool_name"], "error": error, "written": []}
        on_written = kwargs.get("on_written")
        if on_written is not None:
            def _written():
                entry["written"].append(True)
                on_written()
            kwargs["on_written"] = _written
        entry["raised"] = await record(host, error, **kwargs)
        records.append(entry)
        return entry["raised"]

    monkeypatch.setattr(orchestrator_engine, "_record_interrupted_dispatch", spy)
    return records


async def _revoke_after_effect(host):
    """``talon_cancel`` cancels the job in SQLite, then execution authority
    is revoked, as a Stop revokes it, before the call's row is written."""
    runtime = ExecutionCustody(_LiveRuntimeFence())
    await host.backend.execute("CREATE TABLE talon_jobs (job_id TEXT, state TEXT)")

    async def cancel_effect(job_id):
        await host.backend.execute(
            "INSERT INTO talon_jobs (job_id, state) VALUES (?, 'cancelled')", (job_id,),
        )
        runtime.revoke("Stop arrived after talon_cancel took effect")

    host.feature.cancel_effect = cancel_effect
    return runtime


@pytest.mark.asyncio
@pytest.mark.parametrize("inline", [False, True], ids=["loop", "inline"])
@pytest.mark.parametrize("door", ["chat", "named"])
async def test_a_write_refused_once_authority_is_revoked_is_reported_not_recorded(
    make_host, monkeypatch, caplog, capsys, door, inline,
):
    """The reviewer's case on real SQLite under native custody: the effect is
    done, authority is revoked, and native storage refuses the interrupted
    call's row. Nothing is marked recorded, one WARNING names each refused
    row, and the original authority error propagates unchanged. No authority
    to write after revocation is added: the fenced owner does not write."""
    host = await make_host(inline=inline)
    runtime = await _revoke_after_effect(host)
    interrupted = _spy_interrupted_records(monkeypatch)

    with bind_execution_custody_snapshot((runtime,)):
        with pytest.raises(ExecutionAuthorityError) as raised:
            await _dispatch(host, door, {"task": "claim, then cancel"})

    # The effect is committed, and only the call before the revocation has a
    # row: neither the cancel nor its envelope is recorded.
    assert await host.backend.fetch_all("SELECT job_id, state FROM talon_jobs") == [
        ("692b13dd", "cancelled"),
    ]
    [claim] = await host.rows()
    assert claim["tool_name"] == "talon_claim"

    # Both interrupted records ran, neither was reported written, and each
    # raised the error it was given: the one the caller sees.
    assert [entry["tool_name"] for entry in interrupted] == ["talon_cancel", DISPATCHER]
    for entry in interrupted:
        assert entry["raised"] is entry["error"]
        assert entry["written"] == []
    assert interrupted[0]["error"] is interrupted[1]["error"] is raised.value
    assert type(raised.value) is ExecutionAuthorityError
    assert "talon_cancel took effect" in str(raised.value)
    # Unchanged: the refusal is not attached to it as its cause.
    assert raised.value.__cause__ is None

    # One WARNING for each refused row, naming it, its tool and its envelope.
    envelope_id = claim["parent_dispatch_id"]
    cancel_report, envelope_report = _unwritten_reports(caplog)
    assert envelope_report == {
        "dispatch_id": envelope_id, "tool_name": DISPATCHER,
        "parent_dispatch_id": None, "cause": "ExecutionAuthorityError",
    }
    assert cancel_report["tool_name"] == "talon_cancel"
    assert cancel_report["parent_dispatch_id"] == envelope_id
    assert cancel_report["dispatch_id"] not in (envelope_id, claim["dispatch_id"])
    assert cancel_report["cause"] == "ExecutionAuthorityError"
    # Reported by the writer alone, not also by the ledger's own fallback.
    assert "a2a_tool_dispatches" not in capsys.readouterr().err


class _RevokeAtPost(Hook):
    """Revokes execution authority during one tool's POST_TOOL_USE: the call
    returned, its row is not yet written."""

    def __init__(self, tool_name: str, runtime: ExecutionCustody):
        super().__init__(name="revoke-at-post", events=[HookEvent.POST_TOOL_USE])
        self.tool_name = tool_name
        self.runtime = runtime

    async def execute(self, input: HookInput) -> HookOutput:
        if input.tool_name == self.tool_name:
            self.runtime.revoke("Stop arrived during talon_cancel's POST_TOOL_USE")
        return HookOutput.allow()


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["chat", "named"])
async def test_a_refused_row_is_not_attempted_again_as_interrupted(
    make_host, monkeypatch, caplog, door,
):
    """The call's own row is refused, then the revocation ends the call. The
    refused write settled the row: no second attempt, no second report."""
    host = await make_host()
    runtime = ExecutionCustody(_LiveRuntimeFence())
    host.agent.hooks_manager.register(_RevokeAtPost("talon_cancel", runtime))
    interrupted = _spy_interrupted_records(monkeypatch)

    with bind_execution_custody_snapshot((runtime,)):
        with pytest.raises(ExecutionAuthorityError) as raised:
            await _dispatch(host, door, {"task": "claim, then cancel"})

    [claim] = await host.rows()
    assert host.feature.calls[-1] == ("talon_cancel", "692b13dd")
    # Only the envelope is recorded as interrupted; the cancel's refused row
    # was its one attempt.
    assert [entry["tool_name"] for entry in interrupted] == [DISPATCHER]
    assert interrupted[0]["raised"] is raised.value
    reports = _unwritten_reports(caplog)
    assert [report["tool_name"] for report in reports] == ["talon_cancel", DISPATCHER]
    assert reports[0]["parent_dispatch_id"] == claim["parent_dispatch_id"]
    assert reports[1]["dispatch_id"] == claim["parent_dispatch_id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("required", [False, True], ids=["best-effort", "required"])
async def test_a_failed_write_never_reports_its_row_written(make_host, monkeypatch, caplog,
                                                           required):
    """``on_written`` is the receipt for a row; only a written row gets one.
    A best-effort row's failure leaves the caller's outcome as it was; a
    required row's failure is raised, as before."""
    host = await make_host()
    failure = OSError("disk I/O error")

    async def fail(entry, **kwargs):
        raise failure

    monkeypatch.setattr(host.agent.observability_store, "log_tool_dispatch", fail)
    written, settled = [], []
    write = orchestrator_engine._write_tool_dispatch(
        host.agent, tool_name="talon_cancel", adapter="subagent:_TalonProbe.talon_cancel",
        args=dict(CANCEL[1]), result={"cancelled": "692b13dd"}, session_id=SESSION,
        dispatch_start=0.0, parent_dispatch_id="envelope-1",
        on_written=lambda: written.append(True), on_settled=lambda: settled.append(True),
        **({"required": True} if required else {}),
    )
    if required:
        with pytest.raises(OSError) as raised:
            await write
        assert raised.value is failure
    else:
        assert await write is None
    assert (written, settled) == ([], [True])
    [report] = _unwritten_reports(caplog)
    assert (report["tool_name"], report["parent_dispatch_id"], report["cause"]) == (
        "talon_cancel", "envelope-1", "OSError",
    )
    assert re.fullmatch(r"[0-9a-f]{32}", report["dispatch_id"])

    # A written row is.
    monkeypatch.undo()
    await orchestrator_engine._write_tool_dispatch(
        host.agent, tool_name="talon_cancel", adapter="subagent:_TalonProbe.talon_cancel",
        args=dict(CANCEL[1]), result={"cancelled": "692b13dd"}, session_id=SESSION,
        dispatch_start=0.0, on_written=lambda: written.append(True),
    )
    assert written == [True]


@pytest.mark.asyncio
async def test_a_host_with_no_ledger_reports_no_row_written(make_host, caplog):
    """No ledger, no row: nothing is marked recorded, and nothing refused it."""
    host = await make_host()
    host.agent.observability_store = None
    written, settled = [], []
    await orchestrator_engine._write_tool_dispatch(
        host.agent, tool_name="talon_cancel", adapter="subagent:_TalonProbe.talon_cancel",
        args=dict(CANCEL[1]), result={"cancelled": "692b13dd"}, session_id=SESSION,
        dispatch_start=0.0,
        on_written=lambda: written.append(True), on_settled=lambda: settled.append(True),
    )
    assert (written, settled) == ([], [True])
    assert _unwritten_reports(caplog) == []


class _DeferredTool:
    """A subagent tool whose result is deferred: its row is the required one
    its completion writes."""

    name = "talon_view"
    schema = SimpleNamespace(parameters=[])

    def __init__(self):
        self.published: list[str] = []

    def _publish(self):
        self.published.append("published")
        return []

    async def execute(self, **kwargs):
        return DeferredToolResult(ToolResult.ok("job view").to_dict(), publish=self._publish)


@pytest.mark.asyncio
async def test_a_deferred_calls_failed_row_is_not_attempted_again(
    make_host, monkeypatch, caplog,
):
    """The completion's required row fails: the call is reported once and
    refused as unconfirmed, not recorded a second time as interrupted."""
    host = await make_host()
    tool = _DeferredTool()
    store = host.agent.observability_store

    async def fail(entry, **kwargs):
        raise OSError("disk I/O error")

    monkeypatch.setattr(store, "log_tool_dispatch", fail)
    interrupted = _spy_interrupted_records(monkeypatch)
    with pytest.raises(ToolAuditUnconfirmedError):
        await host.feature._execute_subagent_tool(
            tool_name=tool.name, args={}, tools_by_name={tool.name: tool},
            session_id=SESSION, parent_dispatch_id="envelope-1",
        )

    assert interrupted == []
    [report] = _unwritten_reports(caplog)
    assert (report["tool_name"], report["parent_dispatch_id"]) == ("talon_view", "envelope-1")
    # Its UI is withheld, not published, once the row could not be written.
    assert tool.published == []


class _DeferredEnvelope(Feature):
    """A custom feature whose subagent dispatch itself returns a deferred
    result: the envelope's row is the required one its completion writes."""

    tool_name = "talon_view_feature"

    def __init__(self, agent):
        super().__init__(agent)
        self.tool = _DeferredTool()

    @property
    def tool_description(self) -> str:
        return "View a Talon job."

    async def initialize(self) -> None:
        return None

    async def execute_as_subagent(self, **kwargs):
        return await self.tool.execute()


@pytest.mark.asyncio
async def test_a_chat_envelopes_failed_deferred_row_is_not_attempted_again(
    make_host, monkeypatch, caplog,
):
    host = await make_host()
    feature = _DeferredEnvelope(host.agent)
    store = host.agent.observability_store

    async def fail(entry, **kwargs):
        raise OSError("disk I/O error")

    monkeypatch.setattr(store, "log_tool_dispatch", fail)
    interrupted = _spy_interrupted_records(monkeypatch)
    with pytest.raises(ToolAuditUnconfirmedError):
        async with host.agent._turn_lifecycle():
            await host.agent._dispatch_tool_call(
                ToolCall(id="tc-1", name=feature.tool_name, arguments={"task": "view"}),
                features_by_tool_name={feature.tool_name: feature},
                known_tools={feature.tool_name},
                messages=[],
                iteration=0,
                user_message="",
                session_id=SESSION,
            )

    assert interrupted == []
    assert [report["tool_name"] for report in _unwritten_reports(caplog)] == [
        feature.tool_name,
    ]
    assert feature.tool.published == []


class _DeferredTalonProbe(_TalonProbe):
    """Runs its subagent's calls, then returns a deferred result: the
    envelope's row is the required one its completion writes, and the calls
    name that row as their parent."""

    def __init__(self, agent):
        super().__init__(agent)
        self.published: list[str] = []

    def _publish(self):
        self.published.append("published")
        return []

    async def execute_as_subagent(self, **kwargs):
        await super().execute_as_subagent(**kwargs)
        return DeferredToolResult(
            ToolResult.ok("claimed, then cancelled").to_dict(), publish=self._publish,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("inline", [False, True], ids=["loop", "inline"])
async def test_a_deferred_envelope_is_written_under_the_id_its_calls_name(make_host, inline):
    """The reviewer's case: the completion wrote the envelope's row under a
    fresh id, so ``talon_claim`` and ``talon_cancel`` named a parent no row
    had, and the correctly named row was suppressed as already audited."""
    host = await make_host(inline=inline)
    feature = _DeferredTalonProbe(host.agent)

    async with host.agent._turn_lifecycle():
        result = await host.agent._dispatch_tool_call(
            ToolCall(id="tc-1", name=DISPATCHER, arguments={"task": "claim, then cancel"}),
            features_by_tool_name={DISPATCHER: feature},
            known_tools={DISPATCHER},
            messages=[],
            iteration=0,
            user_message="",
            session_id=SESSION,
        )

    assert result["status"] == "ok", result
    assert feature.published == ["published"]
    rows = await host.rows()
    envelope, inner = _envelope_and_inner(rows)
    assert (envelope["adapter"], envelope["result_status"]) == (
        "_DeferredTalonProbe.execute_as_subagent", "success",
    )
    assert [row["tool_name"] for row in inner] == ["talon_claim", "talon_cancel"]
    for row in inner:
        assert row["parent_dispatch_id"] == envelope["dispatch_id"]
    assert len({row["dispatch_id"] for row in rows}) == len(rows) == 3


@pytest.mark.asyncio
async def test_a_direct_tools_deferred_row_is_written_under_its_dispatch(
    make_host, monkeypatch,
):
    """Every positioned dispatch's row is written under its ``ledger_id``,
    including the one a direct tool's deferred completion writes."""
    host = await make_host()
    tool = _DeferredTool()
    host.agent._direct_tools[tool.name] = tool
    contexts = []
    begin = orchestrator_engine.begin_tool_call

    def _begin(agent, known_tools):
        context = begin(agent, known_tools)
        contexts.append(context)
        return context

    monkeypatch.setattr(orchestrator_engine, "begin_tool_call", _begin)
    async with host.agent._turn_lifecycle():
        result = await host.agent._dispatch_tool_call(
            ToolCall(id="tc-1", name=tool.name, arguments={}),
            features_by_tool_name={},
            known_tools={tool.name},
            messages=[],
            iteration=0,
            user_message="",
            session_id=SESSION,
        )

    assert result["status"] == "ok", result
    assert tool.published == ["published"]
    [context] = contexts
    [row] = await host.rows()
    assert (row["tool_name"], row["dispatch_id"]) == (tool.name, context.ledger_id)


def _lose_commit_ack(monkeypatch, host, tool_name):
    """The next COMMIT of ``tool_name``'s row lands, then its acknowledgement
    is lost: native SQLite cannot say whether the row was written."""
    store = host.agent.observability_store
    write = store.log_tool_dispatch
    connection = host.backend._connection
    commit = connection.commit
    armed = []

    async def _write(entry, **kwargs):
        if entry.tool_name == tool_name and not armed:
            armed.append(entry)
        await write(entry, **kwargs)

    async def _commit():
        await commit()
        if armed and armed[-1] is not None:
            armed.append(None)
            raise OSError("native COMMIT acknowledgement lost")

    monkeypatch.setattr(store, "log_tool_dispatch", _write)
    monkeypatch.setattr(connection, "commit", _commit)


_UNCONFIRMED = re.compile(
    r"a2a_tool_dispatches row write outcome (\w+): dispatch_id=(\S+) tool=(\S+) "
)


def _unconfirmed_reports(caplog) -> list[tuple[str, str]]:
    """``(tool, commit outcome)`` of every WARNING that a row's write may
    have landed."""
    reports = []
    for record in caplog.records:
        match = _UNCONFIRMED.search(record.getMessage())
        if match is not None:
            assert record.levelname == "WARNING"
            outcome, _dispatch_id, tool_name = match.groups()
            reports.append((tool_name, outcome))
    return reports


@pytest.mark.asyncio
@pytest.mark.parametrize("inline", [False, True], ids=["loop", "inline"])
@pytest.mark.parametrize("door", ["chat", "named"])
async def test_an_interrupted_rows_lost_commit_ack_reaches_the_caller(
    make_host, monkeypatch, caplog, door, inline,
):
    """The reviewer's case on real SQLite under native custody: the job is
    cancelled, then the turn is, and the interrupted call's row COMMITs and
    loses its acknowledgement. The caller receives that commit uncertainty,
    with the cancellation retained in it, not a plain ``CancelledError``."""
    host = await make_host(inline=inline)
    host.feature.hold_cancel = True
    runtime = ExecutionCustody(_LiveRuntimeFence())
    _lose_commit_ack(monkeypatch, host, "talon_cancel")
    interrupted = _spy_interrupted_records(monkeypatch)

    with bind_execution_custody_snapshot((runtime,)):
        dispatch = asyncio.create_task(_dispatch(host, door, {"task": "claim, then cancel"}))
    await asyncio.wait_for(host.feature.cancel_entered.wait(), 5)
    dispatch.cancel()
    with pytest.raises(ExecutionCommitOutcomeError) as raised:
        await dispatch

    assert is_execution_control_error(raised.value)
    assert execution_commit_outcome(raised.value) == "unknown"
    cancel_record, envelope_record = interrupted
    assert cancel_record["tool_name"] == "talon_cancel"
    assert type(cancel_record["error"]) is asyncio.CancelledError
    # The cancellation is retained in what the caller receives.
    assert any(
        carrier is cancel_record["error"]
        for carrier in execution_evidence_carriers(raised.value)
    )
    assert envelope_record["tool_name"] == DISPATCHER
    assert envelope_record["raised"] is raised.value
    # Neither row is taken for written: the cancel's may have landed, and the
    # envelope's was refused by the custody that latched that uncertainty.
    assert cancel_record["written"] == envelope_record["written"] == []
    assert _unconfirmed_reports(caplog) == [
        ("talon_cancel", "unknown"), (DISPATCHER, "unknown"),
    ]
    rows = await host.rows()
    assert [(row["tool_name"], row["error_class"]) for row in rows] == [
        ("talon_claim", None), ("talon_cancel", "CancelledError"),
    ]


class _UnretiredWrite(OSError):
    """A write failure that left a native resource it could not release."""


def _unretired_write_error():
    error = _UnretiredWrite("native connection reset failed")
    owner = asyncio.get_running_loop().create_future()
    owner.set_exception(OSError("reset failed"))
    owner.exception()
    error._kestrel_native_retirement_debts = ((object(), owner),)
    return error


@pytest.mark.asyncio
@pytest.mark.parametrize("required", [False, True], ids=["best-effort", "required"])
@pytest.mark.parametrize("evidence", ["commit-unknown", "unretired"])
async def test_a_writes_commit_or_retirement_evidence_is_never_dropped(
    make_host, monkeypatch, caplog, required, evidence,
):
    """A refused or failed best-effort row leaves its caller's outcome alone
    (``test_a_failed_write_never_reports_its_row_written``). A write that may
    have committed, or that left a resource unretired, does not: its error
    is raised even for a best-effort row, and its row is not taken for
    written."""
    host = await make_host()
    if evidence == "commit-unknown":
        _lose_commit_ack(monkeypatch, host, "talon_cancel")
    else:
        failure = _unretired_write_error()

        async def fail(entry, **kwargs):
            raise failure

        monkeypatch.setattr(host.agent.observability_store, "log_tool_dispatch", fail)
    written, settled = [], []

    with pytest.raises(Exception) as raised:
        await orchestrator_engine._write_tool_dispatch(
            host.agent, tool_name="talon_cancel",
            adapter="subagent:_TalonProbe.talon_cancel", args=dict(CANCEL[1]),
            result={"cancelled": "692b13dd"}, session_id=SESSION, dispatch_start=0.0,
            parent_dispatch_id="envelope-1",
            on_written=lambda: written.append(True),
            on_settled=lambda: settled.append(True),
            **({"required": True} if required else {}),
        )

    assert (written, settled) == ([], [True])
    if evidence == "commit-unknown":
        assert type(raised.value) is ExecutionCommitOutcomeError
        assert execution_commit_outcome(raised.value) == "unknown"
        assert _unconfirmed_reports(caplog) == [("talon_cancel", "unknown")]
        # It did land: native SQLite could not say so.
        [row] = await host.rows()
        assert row["tool_name"] == "talon_cancel"
    else:
        assert raised.value is failure
        assert execution_cleanup_debts(raised.value)
        [report] = _unwritten_reports(caplog)
        assert (report["tool_name"], report["cause"]) == ("talon_cancel", "_UnretiredWrite")


@pytest.mark.asyncio
@pytest.mark.parametrize("inline", [False, True], ids=["loop", "inline"])
@pytest.mark.parametrize(
    ("calls", "decide", "error", "expected"),
    [
        pytest.param(
            (("talon_launch", {}),), None, None,
            ("talon_launch", "error", "UnknownToolError"), id="not-in-palette",
        ),
        pytest.param(
            (("talon_claim", {"repo": "r", "issue": "abc"}),), None, None,
            ("talon_claim", "error", "ToolArgumentError"), id="uncoercible",
        ),
        pytest.param(
            (CANCEL,), ("talon_cancel", HookOutput.deny("no cancels")), None,
            ("talon_cancel", "policy_denied", "PermissionDenied"), id="deny",
        ),
        pytest.param(
            (CANCEL,), ("talon_cancel", HookOutput.ask("approval-1", "ask first")), None,
            ("talon_cancel", "policy_denied", "ApprovalRequired"), id="ask",
        ),
        pytest.param(
            (CLAIM,), ("talon_claim", HookOutput.modify({"repo": "r", "issue": "7"})), None,
            ("talon_claim", "error", "ToolArgumentRewriteError"), id="rewrite-refused",
        ),
        # The ``@tool`` wrapper returns what its method raised as a failed
        # result, which is recorded as the result reports it.
        pytest.param(
            (CLAIM,), None, RuntimeError("talon unreachable"),
            ("talon_claim", "error", "InlineToolFailure"), id="raised",
        ),
    ],
)
async def test_a_call_that_did_not_succeed_is_recorded_under_its_envelope(
    make_host, inline, calls, decide, error, expected,
):
    host = await make_host(calls=calls, inline=inline, decide=decide)
    host.feature.error = error
    async with host.agent._turn_lifecycle():
        await host.agent._dispatch_tool_call(
            ToolCall(id="tc-1", name=DISPATCHER, arguments={"task": "try it"}),
            features_by_tool_name={DISPATCHER: host.feature},
            known_tools={DISPATCHER},
            messages=[],
            iteration=0,
            user_message="",
            session_id=SESSION,
        )

    envelope, inner = _envelope_and_inner(await host.rows())
    tool_name, status, error_class = expected
    [row] = inner
    assert (row["tool_name"], row["result_status"], row["error_class"]) == (
        tool_name, status, error_class,
    )
    assert row["parent_dispatch_id"] == envelope["dispatch_id"]
    assert row["error_message"]
    ran = error is not None
    assert bool(host.feature.calls) is ran


class _RaisingTool:
    """A tool that is not a native ``@tool`` method: what it raises reaches
    the subagent's own ``except``, not a wrapper."""

    name = "talon_status"
    schema = SimpleNamespace(parameters=[])

    async def execute(self, **kwargs):
        raise RuntimeError("talon unreachable")


@pytest.mark.asyncio
async def test_a_tool_that_raises_is_recorded_with_its_error(make_host):
    host = await make_host()
    tool = _RaisingTool()

    result = await host.feature._execute_subagent_tool(
        tool_name=tool.name, args={}, tools_by_name={tool.name: tool},
        session_id=SESSION, parent_dispatch_id="envelope-1",
    )

    assert result == {"success": False, "error": "RuntimeError: talon unreachable"}
    [row] = await host.rows()
    assert (row["tool_name"], row["result_status"], row["error_class"]) == (
        "talon_status", "error", "RuntimeError",
    )
    assert row["error_message"] == "talon unreachable"
    assert row["parent_dispatch_id"] == "envelope-1"
    assert host.spy.edges == [
        (HookEvent.PRE_TOOL_USE.value, "talon_status"),
        (HookEvent.POST_TOOL_USE.value, "talon_status"),
    ]


@pytest.mark.asyncio
async def test_a_subagent_run_outside_any_dispatch_records_no_parent(make_host):
    host = await make_host()
    envelope = await host.feature.execute_as_subagent(task="claim, then cancel")

    assert envelope["success"] is True, envelope
    rows = await host.rows()
    assert [row["tool_name"] for row in rows] == ["talon_claim", "talon_cancel"]
    assert [row["parent_dispatch_id"] for row in rows] == [None, None]


@pytest.mark.asyncio
async def test_the_ledger_adds_its_dispatch_columns_to_an_existing_table(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "legacy.db"))
    await backend.connect()
    await backend.execute_script("""
        CREATE TABLE a2a_tool_dispatches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_did TEXT NOT NULL,
            session_id TEXT,
            turn_id TEXT NOT NULL,
            ts TEXT,
            tool_name TEXT NOT NULL,
            adapter TEXT NOT NULL,
            args_redacted TEXT NOT NULL,
            result_status TEXT NOT NULL,
            error_class TEXT,
            error_message TEXT,
            latency_ms INTEGER NOT NULL,
            result_size_bytes INTEGER
        )
    """)
    await backend.execute(
        "INSERT INTO a2a_tool_dispatches (agent_did, turn_id, tool_name, adapter, "
        "args_redacted, result_status, latency_ms) "
        "VALUES (?, 'turn_old', 'github', 'GithubFeature.execute_as_subagent', '{}', "
        "'error', 3)",
        (AGENT_DID,),
    )
    store = ObservabilityStore(backend)
    await store.initialize()
    try:
        common = dict(
            agent_did=AGENT_DID, session_id=SESSION, turn_id="turn_new",
            args_redacted={}, result_status="error", error_class="RuntimeError",
            error_message="boom", latency_ms=1, result_size_bytes=None,
        )
        await store.log_tool_dispatch(ToolDispatchEntry(
            tool_name=DISPATCHER, adapter="_TalonProbe.execute_as_subagent",
            dispatch_id="envelope-1", **common,
        ))
        await store.log_tool_dispatch(ToolDispatchEntry(
            tool_name="talon_cancel", adapter="subagent:_TalonProbe.talon_cancel",
            parent_dispatch_id="envelope-1", **common,
        ))

        failures = {row["tool_name"]: row for row in await store.recent_failures(AGENT_DID)}
        assert failures["github"]["dispatch_id"] is None
        assert failures["github"]["parent_dispatch_id"] is None
        assert failures[DISPATCHER]["dispatch_id"] == "envelope-1"
        assert failures[DISPATCHER]["parent_dispatch_id"] is None
        # A row given no id is named by the writer.
        assert failures["talon_cancel"]["dispatch_id"]
        assert failures["talon_cancel"]["dispatch_id"] != "envelope-1"
        assert failures["talon_cancel"]["parent_dispatch_id"] == "envelope-1"
    finally:
        await store.close()


def test_an_inline_outcome_is_still_the_pair_every_reader_unpacks():
    outcome = InlineToolOutcome({"job_id": "j"}, {"success": True}, "d-1")

    effective_args, result = outcome
    assert (effective_args, result) == ({"job_id": "j"}, {"success": True})
    assert outcome == ({"job_id": "j"}, {"success": True})
    assert inline_tool_dispatch_id(outcome) == "d-1"
    for duplicate in (copy.copy(outcome), copy.deepcopy(outcome),
                      pickle.loads(pickle.dumps(outcome))):
        assert duplicate == outcome
        assert inline_tool_dispatch_id(duplicate) == "d-1"
    assert inline_tool_dispatch_id(({"job_id": "j"}, {"success": True})) is None
    assert inline_tool_dispatch_id(InlineToolOutcome({}, {})) is None


@pytest.mark.asyncio
async def test_codex_records_a_dispatch_id_only_when_the_executor_names_one():
    executed: list[dict] = []

    async def plain(name, args):
        return dict(args), {"success": True}

    async def named(name, args):
        return InlineToolOutcome(dict(args), {"success": True}, "d-1")

    for executor in (plain, named):
        handler = CodexAdapter()._make_tool_call_handler(
            executor, "t", frozenset({"talon_status"}), executed_log=executed,
        )
        await handler({"threadId": "t", "callId": "c", "name": "talon_status",
                       "arguments": {}})

    assert "dispatch_id" not in executed[0]
    assert executed[1]["dispatch_id"] == "d-1"


def test_recent_failures_text_names_the_dispatch_a_call_ran_inside(monkeypatch, capsys):
    from kestrel_sovereign import cli

    async def rows(args, query_name):
        assert query_name == "recent-failures"
        common = {"ts": "2026-10-21T14:43:59Z", "result_status": "error",
                  "error_class": "InlineToolFailure", "error_message": "job gone"}
        return [
            {"id": 2, "tool_name": "talon_cancel", "parent_dispatch_id": "envelope-1", **common},
            {"id": 1, "tool_name": "talon_status", "parent_dispatch_id": None, **common},
        ]

    monkeypatch.setattr(cli, "_query_tool_dispatches", rows)
    assert cli.cmd_tool_log_recent_failures(SimpleNamespace(json=False)) == 0

    cancel, status = capsys.readouterr().out.splitlines()
    assert cancel.endswith("job gone [inside dispatch envelope-1]")
    assert status.endswith("job gone")
