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
"""

from __future__ import annotations

import asyncio
import copy
import json
import pickle
from types import SimpleNamespace

import pytest
import pytest_asyncio
from kestrel_sdk.hooks.base import Hook, HookEvent, HookInput, HookOutput
from kestrel_sdk.llm.response import LLMResponse, ToolCall
from kestrel_sdk.tools.base import ToolCategory

from kestrel_sovereign.a2a.stores.unified.observability_store import (
    ObservabilityStore,
    ToolDispatchEntry,
)
from kestrel_sovereign.agent.orchestrator_engine import OrchestratorEngineMixin
from kestrel_sovereign.agent.tool_registry import ToolRegistryMixin
from kestrel_sovereign.agent.turn_lifecycle import TurnLifecycleMixin
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
async def test_a_cancellation_during_a_rows_write_leaves_exactly_that_row(
    make_host, monkeypatch,
):
    """The write is owned: the cancellation waits for it, so the call is not
    recorded a second time as interrupted."""
    host = await make_host()
    store = host.agent.observability_store
    write = store.log_tool_dispatch
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow(entry, **kwargs):
        if entry.tool_name == "talon_cancel" and not entered.is_set():
            entered.set()
            await release.wait()
        await write(entry, **kwargs)

    monkeypatch.setattr(store, "log_tool_dispatch", slow)
    dispatch = asyncio.create_task(_dispatch(host, "named", {"task": "claim, then cancel"}))
    await asyncio.wait_for(entered.wait(), 5)
    dispatch.cancel()
    for _ in range(5):
        await asyncio.sleep(0)
    assert not dispatch.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await dispatch

    envelope, inner = _envelope_and_inner(await host.rows())
    assert [(row["tool_name"], row["result_status"]) for row in inner] == [
        ("talon_claim", "success"), ("talon_cancel", "success"),
    ]
    assert (envelope["result_status"], envelope["error_class"]) == ("error", "CancelledError")


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
