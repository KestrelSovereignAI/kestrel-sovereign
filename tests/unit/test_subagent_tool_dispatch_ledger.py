"""A feature subagent's own tool calls are in the dispatch log (#3778).

``a2a_tool_dispatches`` (read by ``kestrel tool-dispatches``) recorded a
feature-as-subagent dispatch only as its envelope. The calls the subagent made
inside it reached ``security_audit_log`` and nothing else, so when Claw's Talon
coordinator subagent ran a ``talon_claim`` and then a ``talon_cancel``, two
people searching the dispatch log concluded the job had been cancelled by
nobody.

Every call a subagent completes now writes its own row, whatever its outcome,
and names the envelope that delegated to it: its ``parent_dispatch_id`` is the
envelope row's ``dispatch_id``. Covered on both ways a subagent runs its tools
(its own post-LLM loop, and the inline executor the codex app-server calls) and
every way an envelope is dispatched: the chat loop, and ``execute_named_tool``,
which the codex inline executor and the other transports call. A named
envelope writes its own row, so it exists without an inline caller and when
the codex turn fails before recording its executed calls; the inline recorder
does not write it twice, and writes it when the envelope's own write failed.
The row names the arguments the subagent ran with, after an inner
``PRE_TOOL_USE`` rewrite.

Rows are written as a top-level call's row is: in the calling task, once the
call completes. A call made inside the caller's transaction therefore writes
its row in that transaction, and a call that a cancellation ends before then
records nothing, as a top-level call does (#3978 owns recording it). A
``ToolResult`` a subagent returns keeps its status, in the row and in what the
caller receives.

A write the ledger refuses is never taken for a row. Native storage refuses
one once execution authority is revoked, and nothing grants it authority to
write after that. The refusal is reported once, as a WARNING naming the row,
and the original error is raised unchanged. A write whose COMMIT outcome is
unknown, or that left a native resource unretired, is not taken for a row
either, but its evidence is raised, not dropped.
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
)
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.hooks.manager import HooksManager
from kestrel_sovereign.llm.adapter import InlineToolOutcome, inline_tool_dispatch_id
from kestrel_sovereign.llm.codex_adapter import CodexAdapter
from kestrel_sovereign.storage.db.sqlite import SQLiteBackend
from tests.utils.native_tool_effects import drive_native
from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown
from kestrel_sovereign.execution_custody import execution_evidence_carriers

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
        # What ``talon_claim`` returns, when a test needs another result.
        self.claim_result = None
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
        if self.claim_result is not None:
            return self.claim_result
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


class _ReportingTalonProbe(_TalonProbe):
    """Runs its subagent's calls, then returns what ``outcome()`` makes of
    them: a custom subagent's own result, in place of the default envelope."""

    def __init__(self, agent, outcome):
        super().__init__(agent)
        self.outcome = outcome

    async def execute_as_subagent(self, **kwargs):
        await super().execute_as_subagent(**kwargs)
        return self.outcome()


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

    def use(self, feature):
        """Dispatch ``feature`` as the envelope instead."""
        self.agent.features = {type(feature).__name__: feature}
        self.feature = feature
        return feature

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
async def make_host(native_tool_effect_agent):
    async def _make(*, calls=(CLAIM, CANCEL), inline=False, decide=None) -> _Host:
        agent = await native_tool_effect_agent(did=AGENT_DID)
        backend = agent._raw_storage.db.backend
        hooks = HooksManager()
        if decide is not None:
            hooks.register(_Decide(*decide))
        spy = _Spy(backend)
        hooks.register(spy)
        agent.hooks_manager = hooks
        agent.llm_service = _SubagentLLM(calls, inline=inline)
        feature = _TalonProbe(agent)
        agent.features[type(feature).__name__] = feature
        return _Host(agent, feature, backend, spy)

    return _make


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


def _spy_audited_marks(monkeypatch) -> list[str]:
    """The status of every dispatch marked audited (its row written)."""
    marks = []
    mark = orchestrator_engine._mark_tool_dispatch_audited

    def spy(dispatch_meta, bound_execution=None, *, status):
        marks.append(status)
        mark(dispatch_meta, bound_execution, status=status)

    monkeypatch.setattr(orchestrator_engine, "_mark_tool_dispatch_audited", spy)
    return marks


@pytest.mark.asyncio
@pytest.mark.parametrize("inline", [False, True], ids=["loop", "inline"])
async def test_chat_loop_envelope_names_the_calls_its_subagent_made(make_host, inline):
    host = await make_host(inline=inline)
    result = None
    async def native_turn_operation():
        nonlocal result
        async with host.agent._turn_lifecycle():
            host.agent._active_session_id = SESSION
            result = await host.agent._dispatch_tool_call(
                ToolCall(id="tc-1", name=DISPATCHER, arguments={"task": "claim, then cancel"}),
                features_by_tool_name={DISPATCHER: host.feature},
                known_tools={DISPATCHER},
                messages=[],
                iteration=0,
                user_message="",
                session_id=SESSION,
            )

    await drive_native(host.agent, native_turn_operation)

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
    call_id = None
    async def native_turn_operation():
        nonlocal call_id
        async with host.agent._turn_lifecycle():
            host.agent._active_session_id = SESSION
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

    await drive_native(host.agent, native_turn_operation)

    rows = await host.rows()
    assert [row["tool_name"] for row in rows] == [
        "talon_claim", DISPATCHER, "talon_claim", DISPATCHER,
    ]
    first_claim, first_envelope, second_claim, second_envelope = rows
    assert first_envelope["dispatch_id"] != second_envelope["dispatch_id"]
    assert first_claim["parent_dispatch_id"] == first_envelope["dispatch_id"]
    assert second_claim["parent_dispatch_id"] == second_envelope["dispatch_id"]


async def _codex_dispatch(host, arguments, *, record=True, before_record=None):
    """Run the envelope as the codex app-server does, inside a turn.

    Returns the codex reply and the adapter's executed-call log. ``record``
    records that log afterwards, as a turn that returned its response does;
    ``before_record`` is awaited, in the turn, before it is.
    """
    executed: list[dict] = []
    handler = None
    response = None
    async def native_turn_operation():
        nonlocal handler, response
        async with host.agent._turn_lifecycle():
            host.agent._active_session_id = SESSION
            handler = CodexAdapter()._make_tool_call_handler(
                host.agent._make_inline_tool_executor(SESSION),
                "t", frozenset({DISPATCHER}), executed_log=executed,
            )
            response = await handler({
                "threadId": "t", "callId": "call-1", "name": DISPATCHER,
                "arguments": arguments,
            })
            if before_record is not None:
                await before_record()
            if record:
                await host.agent._append_executed_tool_breadcrumbs(
                    [], executed, session_id=SESSION,
                )

    await drive_native(host.agent, native_turn_operation)
    return response, executed


async def _dispatch(host, door, arguments):
    """Dispatch the envelope through ``door`` inside a turn.

    ``chat`` is the chat loop; ``named`` is ``execute_named_tool`` as a
    transport with no inline caller (voice, MCP) calls it; ``codex`` is the
    codex app-server's inline call, and returns its reply.
    """
    if door == "codex":
        response, _executed = await _codex_dispatch(host, arguments)
        return response
    async def native_turn_operation():
        async with host.agent._turn_lifecycle():
            host.agent._active_session_id = SESSION
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

    return await drive_native(host.agent, native_turn_operation)


@pytest.mark.asyncio
@pytest.mark.parametrize("inline", [False, True], ids=["loop", "inline"])
async def test_codex_inline_envelope_names_the_calls_its_subagent_made(make_host, inline):
    """The issue's case: the envelope itself ran inside a codex turn."""
    host = await make_host(inline=inline)

    response, executed = await _codex_dispatch(host, {"task": "claim, then cancel"})

    assert response["success"] is True, response
    rows = await host.rows()
    envelope, inner = _envelope_and_inner(rows)
    # The named path writes it, as it writes a codex call's deferred row, and
    # the executed-call recorder does not write it again.
    assert envelope["adapter"] == "named:codex_app_server:_TalonProbe.execute_as_subagent"
    [entry] = executed
    assert entry["dispatch_id"] == envelope["dispatch_id"]
    _assert_inner_calls_name_their_envelope(rows, envelope, inner, host.spy)


@pytest.mark.asyncio
@pytest.mark.parametrize("inline", [False, True], ids=["loop", "inline"])
async def test_a_named_envelope_with_no_inline_caller_records_itself(
    make_host, monkeypatch, inline,
):
    """Before, the subagent's rows named a parent no row had: nothing records
    a named call when no inline adapter is there to."""
    host = await make_host(inline=inline)
    audited = _spy_audited_marks(monkeypatch)

    result = await _dispatch(host, "named", {"task": "claim, then cancel"})

    # Callers outside the inline executor get the subagent's own result.
    assert type(result) is dict and result["success"] is True, result
    rows = await host.rows()
    envelope, inner = _envelope_and_inner(rows)
    assert envelope["adapter"] == "named:voice_realtime:_TalonProbe.execute_as_subagent"
    assert envelope["result_status"] == "success"
    # The dispatch is marked audited once its row is written.
    assert audited == ["success"]
    _assert_inner_calls_name_their_envelope(rows, envelope, inner, host.spy)


@pytest.mark.asyncio
async def test_a_codex_envelope_is_recorded_when_its_turn_fails_before_its_breadcrumbs(
    make_host,
):
    """The codex inline executor ran the envelope, then the provider failed
    before returning its response, so the adapter's executed-call log is
    never recorded. The envelope's row is already written."""
    host = await make_host()

    _response, executed = await _codex_dispatch(
        host, {"task": "claim, then cancel"}, record=False,
    )

    rows = await host.rows()
    envelope, inner = _envelope_and_inner(rows)
    [entry] = executed
    assert entry["dispatch_id"] == envelope["dispatch_id"]
    _assert_inner_calls_name_their_envelope(rows, envelope, inner, host.spy)


@pytest.mark.asyncio
async def test_native_codex_envelope_audit_refusal_withholds_reply_and_breadcrumbs(make_host, monkeypatch):
    """A refused native audit cannot be healed or replayed by a breadcrumb."""
    host = await make_host()
    await host.backend.execute(
        "CREATE TRIGGER refuse_envelope BEFORE INSERT ON a2a_tool_dispatches "
        f"WHEN NEW.tool_name = '{DISPATCHER}' "
        "BEGIN SELECT RAISE(ABORT, 'envelope refused'); END"
    )
    original = host.agent.observability_store.log_tool_dispatch
    attempted = []
    async def write(entry, **kwargs):
        attempted.append(entry.tool_name)
        return await original(entry, **kwargs)
    monkeypatch.setattr(host.agent.observability_store, 'log_tool_dispatch', write)
    healed = []
    async def heal():
        healed.append(True)
    with pytest.raises(ResourceOutcomeUnknown):
        await _codex_dispatch(host, {"task": "claim, then cancel"}, before_record=heal)
    assert healed == [], 'An unconfirmed native audit reached normal delivery'
    assert attempted == ['talon_claim', 'talon_cancel', DISPATCHER], 'An original audit was attempted twice'
    assert [row['tool_name'] for row in await host.rows()] == ['talon_claim', 'talon_cancel']


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["named", "codex"])
async def test_an_envelope_records_the_arguments_its_inner_hook_rewrote(
    make_host, monkeypatch, door,
):
    """The reviewer's case: an inner ``PRE_TOOL_USE`` redactor rewrites the
    delegation, then allows it. The subagent runs the rewritten task, and the
    envelope's row, the codex caller and the completion hooks name it, not
    the original the redactor removed."""
    host = await make_host()
    host.agent.hooks_manager.register(_Decide(
        DISPATCHER, HookOutput.modify(updated_input={"task": "safe delegation"}),
    ))
    seen_after = []

    class _Completion(Hook):
        def __init__(self):
            super().__init__(
                name="completion-spy",
                events=[HookEvent.POST_TOOL_USE, HookEvent.POST_SUBAGENT_CALL],
            )

        async def execute(self, input: HookInput) -> HookOutput:
            if input.tool_name == DISPATCHER:
                seen_after.append((input.hook_event_name, dict(input.tool_input)))
            return HookOutput.allow()

    host.agent.hooks_manager.register(_Completion())
    received = []
    run = host.feature.execute_as_subagent

    async def execute_as_subagent(**kwargs):
        received.append(kwargs["task"])
        return await run(**kwargs)

    monkeypatch.setattr(host.feature, "execute_as_subagent", execute_as_subagent)
    original = {"task": "private original delegation"}

    if door == "codex":
        response, executed = await _codex_dispatch(host, dict(original))
        [entry] = executed
        assert entry["arguments"] == {"task": "safe delegation"}
    else:
        response = await _dispatch(host, door, dict(original))

    assert response["success"] is True, response
    assert received == ["safe delegation"]
    rows = await host.rows()
    envelope, inner = _envelope_and_inner(rows)
    assert json.loads(envelope["args_redacted"]) == {"task": "safe delegation"}
    assert all("private" not in row["args_redacted"] for row in rows)
    assert sorted(event for event, _args in seen_after) == sorted([
        HookEvent.POST_TOOL_USE.value, HookEvent.POST_SUBAGENT_CALL.value,
    ])
    assert all(args == {"task": "safe delegation"} for _event, args in seen_after)
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

    response, executed = await _codex_dispatch(host, arguments)

    assert response["success"] is False, response
    [row] = await host.rows()
    assert row["tool_name"] == DISPATCHER
    assert (row["result_status"], row["error_class"]) == expected
    assert row["dispatch_id"] == executed[0]["dispatch_id"]
    assert host.feature.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["chat", "named", "codex"])
async def test_native_subagent_effects_refuse_the_callers_transaction(make_host, door):
    """STARTED must commit durably before a subagent can perform its effect."""
    host = await make_host()
    await host.backend.execute("CREATE TABLE talon_jobs (job_id TEXT, state TEXT)")
    async def cancel_effect(job_id):
        await host.backend.execute("INSERT INTO talon_jobs VALUES (?, 'cancelled')", (job_id,))
    host.feature.cancel_effect = cancel_effect
    async def operation():
        async with host.agent._turn_lifecycle():
            host.agent._active_session_id = SESSION
            async with host.backend.transaction():
                if door == 'chat':
                    result = await host.agent._dispatch_tool_call(
                        ToolCall(id='transaction-call', name=DISPATCHER, arguments={'task':'claim, then cancel'}),
                        {DISPATCHER:host.feature}, {DISPATCHER}, [], 0, '', session_id=SESSION)
                    assert result['success'] is False and 'own durable transaction' in result['error']
                elif door == 'named':
                    with pytest.raises(ResourceConflict, match='own durable transaction'):
                        await host.agent.execute_named_tool(DISPATCHER, {'task':'claim, then cancel'},
                            session_id=SESSION, source='voice_realtime')
                else:
                    handler = CodexAdapter()._make_tool_call_handler(
                        host.agent._make_inline_tool_executor(SESSION), 'transaction-thread',
                        frozenset({DISPATCHER}), executed_log=[])
                    response = await handler({'threadId':'transaction-thread', 'callId':'transaction-call',
                        'name':DISPATCHER, 'arguments':{'task':'claim, then cancel'}})
                    assert response['success'] is False and 'own durable transaction' in str(response)
    async with asyncio.timeout(5):
        await drive_native(host.agent, operation)
    assert host.feature.calls == [], 'An effect started in a rollbackable transaction'
    assert await host.backend.fetch_all('SELECT * FROM talon_jobs') == []
    assert await host.backend.fetch_all('SELECT state FROM tool_effect_claims') == []
    rows = await host.rows()
    assert [row['tool_name'] for row in rows] == ([DISPATCHER] if door == 'chat' else [])
    if rows:
        assert rows[0]['error_class'] == 'ResourceConflict'


@pytest.mark.asyncio
async def test_native_subagent_leaf_refuses_the_callers_transaction(make_host):
    host = await make_host()
    await host.backend.execute("CREATE TABLE talon_jobs (job_id TEXT, state TEXT)")
    async def cancel_effect(job_id):
        await host.backend.execute("INSERT INTO talon_jobs VALUES (?, 'cancelled')", (job_id,))
    host.feature.cancel_effect = cancel_effect
    tools = {tool.name: tool for tool in host.feature.get_tools()}
    async def operation():
        async with host.backend.transaction():
            with pytest.raises(ResourceConflict, match='own durable transaction'):
                await host.feature._execute_subagent_tool(
                    tool_name='talon_cancel', args=dict(CANCEL[1]), tools_by_name=tools,
                    session_id=SESSION, parent_dispatch_id='envelope-1')
    async with asyncio.timeout(5):
        await drive_native(host.agent, operation)
    assert host.feature.calls == []
    assert await host.backend.fetch_all('SELECT * FROM talon_jobs') == []
    assert await host.backend.fetch_all('SELECT state FROM tool_effect_claims') == []
    assert await host.rows() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("door", ["chat", "named", "codex"])
@pytest.mark.parametrize(
    "outcome",
    [
        pytest.param(lambda: ToolResult.failed("talon unreachable"), id="failed"),
        pytest.param(
            lambda: ToolResult.partial("claimed", "cancel unconfirmed"), id="partial",
        ),
    ],
)
async def test_a_subagents_tool_result_keeps_its_status(make_host, door, outcome):
    """The reviewer's case: a named envelope's row and its codex reply took a
    failed ``ToolResult`` for a success. Its status reaches the row and the
    caller on every door; the row records a failure's error."""
    host = await make_host()
    host.use(_ReportingTalonProbe(host.agent, outcome))
    expected = outcome()

    result = await _dispatch(host, door, {"task": "claim, then cancel"})

    if door == "codex":
        assert result["success"] is False, result
        assert expected.error in result["contentItems"][0]["text"]
    else:
        # The caller receives the subagent's result itself.
        assert isinstance(result, ToolResult) and result.status == expected.status
    rows = await host.rows()
    envelope, inner = _envelope_and_inner(rows)
    assert envelope["result_status"] == "error"
    if door != "chat" and expected.status.value == "error":
        assert (envelope["error_class"], envelope["error_message"]) == (
            "InlineToolFailure", "talon unreachable",
        )
    for row in inner:
        assert row["parent_dispatch_id"] == envelope["dispatch_id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("inline", [False, True], ids=["loop", "inline"])
@pytest.mark.parametrize("door", ["chat", "named"])
async def test_an_interrupted_call_records_nothing_and_its_cancellation_propagates(
    make_host, caplog, door, inline,
):
    """The job is cancelled, then the turn is, before the call returns. As for
    a top-level call, the interrupted call and its envelope have no row: only
    the call that completed before them does, and the cancellation ends the
    dispatch as it arrived. Recording the interrupted call is #3978's."""
    host = await make_host(inline=inline)
    host.feature.hold_cancel = True

    dispatch = asyncio.create_task(_dispatch(host, door, {"task": "claim, then cancel"}))
    await asyncio.wait_for(host.feature.cancel_entered.wait(), 5)
    dispatch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(dispatch, 5)

    assert host.feature.calls[-1] == ("talon_cancel", "692b13dd")
    [claim] = await host.rows()
    assert (claim["tool_name"], claim["result_status"]) == ("talon_claim", "success")
    assert claim["parent_dispatch_id"]
    assert _unwritten_reports(caplog) == []


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


class _RevokeAtPost(Hook):
    """Revokes execution authority during one tool's POST_TOOL_USE: the call
    took effect and returned, its row is not yet written."""

    def __init__(self, tool_name: str, runtime: ExecutionCustody):
        super().__init__(name="revoke-at-post", events=[HookEvent.POST_TOOL_USE])
        self.tool_name = tool_name
        self.runtime = runtime

    async def execute(self, input: HookInput) -> HookOutput:
        if input.tool_name == self.tool_name:
            self.runtime.revoke(f"Stop arrived after {self.tool_name} took effect")
        return HookOutput.allow()


def _spy_store_refusals(host) -> list[BaseException]:
    """Every error the ledger's writer raised."""
    refusals = []
    store = host.agent.observability_store
    write = store.log_tool_dispatch

    async def spy(entry, **kwargs):
        try:
            await write(entry, **kwargs)
        except Exception as error:
            refusals.append(error)
            raise

    store.log_tool_dispatch = spy
    return refusals


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("door", "inline", "target"),
    [
        pytest.param("chat", False, "talon_cancel", id="chat-loop-call"),
        pytest.param("chat", True, "talon_cancel", id="chat-inline-call"),
        pytest.param("named", False, "talon_cancel", id="named-loop-call"),
        pytest.param("named", True, "talon_cancel", id="named-inline-call"),
        pytest.param("named", False, DISPATCHER, id="named-envelope"),
    ],
)
async def test_a_write_refused_once_authority_is_revoked_is_reported_not_recorded(
    make_host, monkeypatch, caplog, capsys, door, inline, target,
):
    """Ruling 1's case on real SQLite under native custody: the call takes
    effect, authority is revoked, and native storage refuses its row. Nothing
    is marked recorded, one WARNING names the refused row, and the original
    authority error propagates unchanged. No authority to write after
    revocation is added: the fenced owner does not write."""
    host = await make_host(inline=inline)
    await host.backend.execute("CREATE TABLE talon_jobs (job_id TEXT, state TEXT)")

    async def cancel_effect(job_id):
        await host.backend.execute(
            "INSERT INTO talon_jobs (job_id, state) VALUES (?, 'cancelled')", (job_id,),
        )

    host.feature.cancel_effect = cancel_effect
    runtime = ExecutionCustody(_LiveRuntimeFence())
    host.agent.hooks_manager.register(_RevokeAtPost(target, runtime))
    audited = _spy_audited_marks(monkeypatch)
    refusals = _spy_store_refusals(host)

    with bind_execution_custody_snapshot((runtime,)):
        with pytest.raises(ExecutionAuthorityError) as raised:
            await _dispatch(host, door, {"task": "claim, then cancel"})

    # The effect is committed, and only the calls written before the
    # revocation have rows.
    assert await host.backend.fetch_all("SELECT job_id, state FROM talon_jobs") == [
        ("692b13dd", "cancelled"),
    ]
    rows = await host.rows()
    expected = ["talon_claim"] + (["talon_cancel"] if target == DISPATCHER else [])
    assert [row["tool_name"] for row in rows] == expected
    envelope_id = rows[0]["parent_dispatch_id"]
    assert all(row["parent_dispatch_id"] == envelope_id for row in rows)
    # Nothing is marked audited: the envelope's row was never written.
    assert audited == []

    # The original Stop is checked before an audit write is admitted.
    assert _unwritten_reports(caplog) == []
    [refusal] = refusals
    assert isinstance(refusal, ExecutionAuthorityError)
    assert raised.value is refusal, 'The original authority denial was replaced'

    assert type(raised.value) is ExecutionAuthorityError
    assert f"Stop arrived after {target} took effect" in str(raised.value)
    assert raised.value.__cause__ is None
    # Reported by the writer alone, not also by the ledger's own fallback.
    assert "a2a_tool_dispatches" not in capsys.readouterr().err


@pytest.mark.asyncio
async def test_a_failed_write_never_reports_its_row_written(make_host, monkeypatch, caplog):
    """``on_written`` is the receipt for a row; only a written row gets one.
    A best-effort row's failure leaves the caller's outcome as it was."""
    host = await make_host()
    failure = OSError("disk I/O error")

    async def fail(entry, **kwargs):
        raise failure

    monkeypatch.setattr(host.agent.observability_store, "log_tool_dispatch", fail)
    written = []
    assert await orchestrator_engine._write_tool_dispatch(
        host.agent, tool_name="talon_cancel", adapter="subagent:_TalonProbe.talon_cancel",
        args=dict(CANCEL[1]), result={"cancelled": "692b13dd"}, session_id=SESSION,
        dispatch_start=0.0, parent_dispatch_id="envelope-1",
        on_written=lambda: written.append(True),
    ) is None
    assert written == []
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
    [row] = await host.rows()
    assert row["tool_name"] == "talon_cancel"


@pytest.mark.asyncio
async def test_a_host_with_no_ledger_reports_no_row_written(make_host, caplog):
    """No ledger, no row: nothing is marked recorded, and nothing refused it."""
    host = await make_host()
    host.agent.observability_store = None
    written = []
    await orchestrator_engine._write_tool_dispatch(
        host.agent, tool_name="talon_cancel", adapter="subagent:_TalonProbe.talon_cancel",
        args=dict(CANCEL[1]), result={"cancelled": "692b13dd"}, session_id=SESSION,
        dispatch_start=0.0, on_written=lambda: written.append(True),
    )
    assert written == []
    assert _unwritten_reports(caplog) == []


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
@pytest.mark.parametrize("evidence", ["commit-unknown", "unretired"])
async def test_a_writes_commit_or_retirement_evidence_is_never_dropped(
    make_host, monkeypatch, caplog, evidence,
):
    """A refused or failed best-effort row leaves its caller's outcome alone
    (``test_a_failed_write_never_reports_its_row_written``). A write that may
    have committed, or that left a resource unretired, does not: its error
    is raised, and its row is not taken for written."""
    host = await make_host()
    if evidence == "commit-unknown":
        _lose_commit_ack(monkeypatch, host, "talon_cancel")
    else:
        failure = _unretired_write_error()

        async def fail(entry, **kwargs):
            raise failure

        monkeypatch.setattr(host.agent.observability_store, "log_tool_dispatch", fail)
    written = []

    with pytest.raises(Exception) as raised:
        await orchestrator_engine._write_tool_dispatch(
            host.agent, tool_name="talon_cancel",
            adapter="subagent:_TalonProbe.talon_cancel", args=dict(CANCEL[1]),
            result={"cancelled": "692b13dd"}, session_id=SESSION, dispatch_start=0.0,
            parent_dispatch_id="envelope-1",
            on_written=lambda: written.append(True),
        )

    assert written == []
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
async def test_a_subagents_deferred_call_is_written_under_its_parent(make_host):
    host = await make_host()
    tool = _DeferredTool()

    result = await drive_native(host.agent, lambda: host.feature._execute_subagent_tool(
        tool_name=tool.name, args={}, tools_by_name={tool.name: tool},
        session_id=SESSION, parent_dispatch_id="envelope-1",
    ))

    assert result["status"] == "ok", result
    assert tool.published == ["published"]
    [row] = await host.rows()
    assert (row["tool_name"], row["adapter"], row["parent_dispatch_id"]) == (
        "talon_view", "subagent:_TalonProbe.talon_view", "envelope-1",
    )


@pytest.mark.asyncio
async def test_a_subagents_deferred_call_whose_row_fails_withholds_its_ui(
    make_host, monkeypatch,
):
    """Its row is required: a failed write refuses the call as unconfirmed."""
    host = await make_host()
    tool = _DeferredTool()

    async def fail(entry, **kwargs):
        raise OSError("disk I/O error")

    monkeypatch.setattr(host.agent.observability_store, "log_tool_dispatch", fail)
    with pytest.raises(ToolAuditUnconfirmedError):
        await drive_native(host.agent, lambda: host.feature._execute_subagent_tool(
            tool_name=tool.name, args={}, tools_by_name={tool.name: tool},
            session_id=SESSION, parent_dispatch_id="envelope-1",
        ))

    assert tool.published == []


@pytest.mark.asyncio
@pytest.mark.parametrize("inline", [False, True], ids=["loop", "inline"])
@pytest.mark.parametrize("door", ["chat", "named", "codex"])
async def test_a_deferred_envelope_is_written_under_the_id_its_calls_name(
    make_host, door, inline,
):
    """The reviewer's case: the completion wrote the envelope's row under a
    fresh id, so ``talon_claim`` and ``talon_cancel`` named a parent no row
    had, and the correctly named row was suppressed as already audited."""
    host = await make_host(inline=inline)
    published: list[str] = []

    def _publish():
        published.append("published")
        return []

    host.use(_ReportingTalonProbe(host.agent, lambda: DeferredToolResult(
        ToolResult.ok("claimed, then cancelled").to_dict(), publish=_publish,
    )))

    result = await _dispatch(host, door, {"task": "claim, then cancel"})

    if door == "codex":
        assert result["success"] is True, result
    else:
        assert result["status"] == "ok", result
    if door == "chat":
        # Completing a named envelope's deferred result is #3922's.
        assert published == ["published"]
    rows = await host.rows()
    envelope, inner = _envelope_and_inner(rows)
    assert envelope["result_status"] == "success"
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
    result = None
    async def native_turn_operation():
        nonlocal result
        async with host.agent._turn_lifecycle():
            host.agent._active_session_id = SESSION
            result = await host.agent._dispatch_tool_call(
                ToolCall(id="tc-1", name=tool.name, arguments={}),
                features_by_tool_name={},
                known_tools={tool.name},
                messages=[],
                iteration=0,
                user_message="",
                session_id=SESSION,
            )

    await drive_native(host.agent, native_turn_operation)

    assert result["status"] == "ok", result
    assert tool.published == ["published"]
    [context] = contexts
    [row] = await host.rows()
    assert (row["tool_name"], row["dispatch_id"]) == (tool.name, context.ledger_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("inline", [False, True], ids=["loop", "inline"])
@pytest.mark.parametrize(
    ("calls", "decide", "error", "returned", "expected"),
    [
        pytest.param(
            (("talon_launch", {}),), None, None, None,
            ("talon_launch", "error", "UnknownToolError"), id="not-in-palette",
        ),
        pytest.param(
            (("talon_claim", {"repo": "r", "issue": "abc"}),), None, None, None,
            ("talon_claim", "error", "ToolArgumentError"), id="uncoercible",
        ),
        pytest.param(
            (CANCEL,), ("talon_cancel", HookOutput.deny("no cancels")), None, None,
            ("talon_cancel", "policy_denied", "PermissionDenied"), id="deny",
        ),
        pytest.param(
            (CANCEL,), ("talon_cancel", HookOutput.ask("approval-1", "ask first")), None,
            None, ("talon_cancel", "policy_denied", "ApprovalRequired"), id="ask",
        ),
        pytest.param(
            (CLAIM,), ("talon_claim", HookOutput.modify({"repo": "r", "issue": "7"})), None,
            None, ("talon_claim", "error", "ToolArgumentRewriteError"), id="rewrite-refused",
        ),
        # The ``@tool`` wrapper returns what its method raised as a failed
        # result, which is recorded as the result reports it.
        pytest.param(
            (CLAIM,), None, RuntimeError("talon unreachable"), None,
            ("talon_claim", "error", "InlineToolFailure"), id="raised",
        ),
        pytest.param(
            (CLAIM,), None, None, ToolResult.failed("issue 3664 is closed"),
            ("talon_claim", "error", "InlineToolFailure"), id="failed-result",
        ),
    ],
)
async def test_a_call_that_did_not_succeed_is_recorded_under_its_envelope(
    make_host, inline, calls, decide, error, returned, expected,
):
    host = await make_host(calls=calls, inline=inline, decide=decide)
    host.feature.error = error
    host.feature.claim_result = returned
    async def native_turn_operation():
        async with host.agent._turn_lifecycle():
            host.agent._active_session_id = SESSION
            await host.agent._dispatch_tool_call(
                ToolCall(id="tc-1", name=DISPATCHER, arguments={"task": "try it"}),
                features_by_tool_name={DISPATCHER: host.feature},
                known_tools={DISPATCHER},
                messages=[],
                iteration=0,
                user_message="",
                session_id=SESSION,
            )

    if error is not None:
        with pytest.raises(ResourceOutcomeUnknown) as caught:
            await drive_native(host.agent, native_turn_operation)
        assert any(carrier is error for carrier in execution_evidence_carriers(caught.value))
    else:
        await drive_native(host.agent, native_turn_operation)

    envelope, inner = _envelope_and_inner(await host.rows())
    tool_name, status, error_class = expected
    if error is not None:
        error_class = "ResourceOutcomeUnknown"
    [row] = inner
    assert (row["tool_name"], row["result_status"], row["error_class"]) == (
        tool_name, status, error_class,
    )
    assert row["parent_dispatch_id"] == envelope["dispatch_id"]
    assert row["error_message"]
    if returned is not None:
        assert row["error_message"] == returned.error
    ran = error is not None or returned is not None
    assert bool(host.feature.calls) is ran


class _RaisingTool:
    """A tool that is not a native ``@tool`` method: what it raises reaches
    the subagent's own ``except``, not a wrapper."""

    name = "talon_status"
    schema = SimpleNamespace(parameters=[])

    def __init__(self):
        self.error = RuntimeError("talon unreachable")
        self.calls = 0

    async def execute(self, **kwargs):
        self.calls += 1
        raise self.error


@pytest.mark.asyncio
async def test_a_tool_that_raises_retains_uncertainty_and_its_parent_audit(make_host):
    host = await make_host()
    tool = _RaisingTool()
    with pytest.raises(ResourceOutcomeUnknown) as caught:
        await drive_native(host.agent, lambda: host.feature._execute_subagent_tool(
            tool_name=tool.name, args={}, tools_by_name={tool.name: tool},
            session_id=SESSION, parent_dispatch_id="envelope-1"))
    assert any(carrier is tool.error for carrier in execution_evidence_carriers(caught.value))
    assert tool.calls == 1
    [row] = await host.rows()
    assert (row["tool_name"], row["result_status"], row["error_class"]) == (
        "talon_status", "error", "ResourceOutcomeUnknown")
    assert "unconfirmed" in row["error_message"] and "talon unreachable" not in row["error_message"]
    assert row["parent_dispatch_id"] == "envelope-1"
    assert host.spy.edges == [(HookEvent.PRE_TOOL_USE.value, "talon_status")]
    assert await host.backend.fetch_all('SELECT state FROM tool_effect_claims') == [('UNCONFIRMED',)]


@pytest.mark.asyncio
async def test_a_subagent_run_outside_any_dispatch_records_no_parent(make_host):
    host = await make_host()
    envelope = await drive_native(host.agent, lambda: host.feature.execute_as_subagent(task="claim, then cancel"))

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
