"""signal_dispatch delegates execution to the host's Talon dispatcher (#3838).

The fleet host names one agent its only Talon dispatcher (``[talon.policy]
dispatcher = "Claw"``, kestrel-feature-talon#57). Emma owns the queue and
selects; Claw runs the work. These tests pin both halves:

* which agent executes is what Emma's own Talon reports through
  ``talon_get_config``, the policy its launch gate loads;
* execute sends Claw a ``kestrel.work_order/v1`` A2A task, records its
  dispatch and arms ``wait("a2a:<dispatch_id>", mode="signal")`` instead of
  launching anything on Emma;
* the #3398 guard and workspace readiness read Claw's answers through Emma's
  outbound audit, never Emma's own (empty) Talon registry;
* an unsettled dispatch -- reserved, uncertain, unanswered -- holds the lane,
  and neither it nor a blocking answer is lost behind more than one listing
  page of newer dispatches.

Claw is reached only through the tools Emma calls by name. Its tasks are real
native ``Task`` values turned into ``get_peer_task_result``'s answer by Peers'
own ``_native_peer_result``, so the order Emma sent is read back through the
same protobuf round trip a real peer read takes.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from google.protobuf.json_format import ParseDict
from kestrel_sdk.native_a2a.types import Message, Part, Role, Task, TaskState, TaskStatus
from kestrel_sdk.tools.result import ToolResult, ToolResultStatus

from kestrel_sovereign.a2a.lifecycle import new_task_status, task_state_label
from kestrel_sovereign.a2a.stores.unified.outbound_store import (
    DispatchState,
    OutboundDispatch,
)
from kestrel_sovereign.agent.orchestrator_engine import ToolNotRegisteredError
from kestrel_sovereign.features.peers.feature import PeersFeature
from kestrel_sovereign.features.strategic_memory import StrategicMemoryFeature
from kestrel_sovereign.features.strategic_memory.feature import _SaveOutcome
from kestrel_sovereign.features.strategic_memory.issue_selection import (
    describe_exclusion,
)
from kestrel_sovereign.features.strategic_memory.ledger import (
    BLOCKERS_KEY,
    PATTERNS_KEY,
    StrategyLedger,
)
from kestrel_sovereign.features.strategic_memory.run_history import (
    LIST_OUTBOUND_TOOL,
    ORDER_REFUSED,
    OUTBOUND_PAGE_SIZE,
    READ_PEER_TASK_TOOL,
    WORK_ORDER_SCHEMA,
    WORK_REPORT_SCHEMA,
    RunHistoryUnreadable,
    read_executor_run_history,
)
from kestrel_sovereign.waits.engine import WaitRegistry

#: When Claw answered the orders below.
ANSWERED = datetime(2026, 10, 24, 9, 56, 4, tzinfo=timezone.utc)
TALON_JOB = "802b93fef2824f53944bdac49167a613"
POLICY_TOOL = "talon_get_config"


def _order(issue=3093, repo="o/r", order_id=None):
    return {
        "schema": WORK_ORDER_SCHEMA,
        "order_id": order_id or f"wo-{issue}-1",
        "repo": repo,
        "issue": issue,
        "action": "claim",
        "talon": {},
        "verify": [],
        "context": "Fresh claim.",
    }


def _report(disposition, issue=3093, repo="o/r", job=TALON_JOB):
    return json.dumps({
        "schema": WORK_REPORT_SCHEMA,
        "order_id": f"wo-{issue}-1",
        "repo": repo,
        "issue": issue,
        "talon_job_id": job,
        "disposition": disposition,
        "pr": None,
    })


def _talon_policy(dispatcher):
    return {
        "allowed_backends": ["claude", "codex", "opencode"],
        "allow_api_billing": False,
        "require_worktree": True,
        "require_sandboxed_workspace": True,
        "allow_background_jobs": True,
        "dispatcher": dispatcher,
    }


def _row(
    dispatch_id,
    state=DispatchState.ACKNOWLEDGED,
    *,
    verb="task",
    tool="send_a2a_task",
    parent=None,
    response_kind="task",
):
    """One row of Emma's outbound audit, as ``list_outbound_a2a_tasks`` lists it."""
    return OutboundDispatch(
        dispatch_id=dispatch_id,
        sender_id="did:pkh:emma",
        sender_tenant="emma",
        recipient_id="did:pkh:claw",
        tenant="claw",
        signing_did="did:web:emma",
        interface_url="http://127.0.0.1/a2a/claw",
        protocol_binding="JSONRPC",
        message_id=f"m-{dispatch_id}",
        requested_context_id="",
        request_digest="r",
        message_digest="m",
        dispatch_tool=tool,
        verb=verb,
        created_at="2026-10-24T07:56:10+00:00",
        state=state,
        revision=1,
        remote_task_id=None if state is DispatchState.RESERVED else f"task-{dispatch_id}",
        remote_context_id="ctx",
        remote_task_state=None,
        response_digest=None,
        response_kind=response_kind,
        updated_at="2026-10-24T07:56:10+00:00",
        operation_id="op",
        cancel_operation_id=None,
        cancel_request_digest=None,
        cancel_state=None,
        root_dispatch_id=parent or dispatch_id,
        parent_dispatch_id=parent,
    )


def _read(dispatch_id, parts, state, *, answer=None, at=ANSWERED):
    """``get_peer_task_result``'s answer for Claw's task, built by Peers itself.

    ``parts`` is ``None`` for a history that shows no message from Emma, and
    ``at`` is ``None`` for an answer with no status time.
    """
    sent = [
        Message(
            message_id=f"u-{dispatch_id}",
            role=Role.ROLE_USER,
            parts=[ParseDict(part, Part()) for part in parts],
        )
    ] if parts is not None else []
    reply = (
        None
        if answer is None
        else Message(
            message_id=f"a-{dispatch_id}", role=Role.ROLE_AGENT, parts=[Part(text=answer)]
        )
    )
    task = Task(
        id=f"task-{dispatch_id}",
        context_id="ctx",
        status=(
            new_task_status(state, at=at, message=reply)
            if at is not None
            else TaskStatus(state=state, message=reply)
        ),
        history=sent,
    )
    return PeersFeature._native_peer_result(
        SimpleNamespace(
            dispatch_id=dispatch_id,
            remote_task_id=task.id,
            remote_task_state=task_state_label(state),
            response_kind="task",
        ),
        task,
    )


class _Claw:
    """The tools Emma calls by name: her Talon's ``talon_get_config``, and the
    Peers and wait tools over Claw's tasks and her outbound audit.

    ``rows`` is newest first, as the audit lists it, and the listing pages by
    ``before`` cursor exactly as the real store does, never returning more
    than ``limit`` rows. ``send_a2a_task`` reserves a row and yields before it
    acknowledges, so an unserialized read-then-send would interleave.
    """

    def __init__(self, dispatcher="Claw"):
        self.rows = []
        self.reads = {}
        self.calls = []
        self.policy_result = ToolResult.ok(
            "Talon config loaded",
            data={"success": True, "policy": _talon_policy(dispatcher), "preference": {}},
        )
        self.send_result = None
        self.wait_result = None
        self.unregistered = set()
        self.raises = {}

    def answered(self, dispatch_id, state, answer=None, *, order=None, at=ANSWERED):
        self.rows.insert(0, _row(dispatch_id))
        self.reads[dispatch_id] = _read(
            dispatch_id, [{"data": order or _order()}], state, answer=answer, at=at
        )

    def pending(self, dispatch_id, state=TaskState.TASK_STATE_WORKING, *, order=None):
        self.rows.insert(0, _row(dispatch_id))
        self.reads[dispatch_id] = _read(dispatch_id, [{"data": order or _order()}], state)

    def unrelated(self, count, *, verb="message"):
        for n in range(count):
            self.rows.insert(0, _row(f"other-{verb}-{n}", DispatchState.COMPLETE, verb=verb))

    def names(self):
        return [name for name, _ in self.calls]

    def args(self, name):
        [args] = [args for called, args in self.calls if called == name]
        return args

    def peer_reads(self):
        return [a["dispatch_id"] for n, a in self.calls if n == READ_PEER_TASK_TOOL]

    async def execute_named_tool(self, name, args, *, session_id, source):
        await asyncio.sleep(0)
        assert session_id == "chat-7"
        assert source == "strategic_memory.signal_dispatch"
        self.calls.append((name, dict(args)))
        if name in self.unregistered:
            raise ToolNotRegisteredError(f"Tool '{name}' not registered")
        if name in self.raises:
            raise self.raises[name]
        if name == POLICY_TOOL:
            return self.policy_result
        if name == LIST_OUTBOUND_TOOL:
            assert args["recipient"] == "Claw"
            start = 0
            if args.get("before"):
                [start] = [
                    n + 1 for n, row in enumerate(self.rows)
                    if row.dispatch_id == args["before"]
                ]
            rows = [asdict(row) for row in self.rows[start: start + args["limit"]]]
            return ToolResult.ok("listed", data={"rows": rows, "count": len(rows)})
        if name == READ_PEER_TASK_TOOL:
            return self.reads[args["dispatch_id"]]
        if name == "send_a2a_task":
            if self.send_result is not None:
                return self.send_result
            dispatch_id = f"{len(self.rows) + 1:032x}"
            # Peers reserves the dispatch before it submits anything.
            self.rows.insert(0, _row(dispatch_id, DispatchState.RESERVED))
            await asyncio.sleep(0)
            self.rows[0] = _row(dispatch_id)
            self.reads[dispatch_id] = _read(
                dispatch_id,
                [{"text": args["message"]}, *args["parts"]],
                TaskState.TASK_STATE_SUBMITTED,
            )
            return ToolResult.ok(
                f"A2A task sent to {args['recipient']}",
                data={
                    "sent": True,
                    "recipient": args["recipient"],
                    "dispatch_id": dispatch_id,
                    "remote_task_id": f"task-{dispatch_id}",
                    "state": "submitted",
                    "response_kind": "task",
                },
            )
        if name == "wait":
            if self.wait_result is not None:
                return self.wait_result
            return ToolResult.ok(
                f"Watching {args['target']}",
                data={"ref": args["target"], "mode": "signal", "watching": True},
            )
        if name == "workflow_run":
            return ToolResult.ok("started", data={"run_id": "run-42", "status": "pending"})
        raise AssertionError(f"signal_dispatch called an unexpected tool: {name}")


class _EmmasOwnTalon:
    """Emma's ``talon`` provider: installed, but Claw runs every job. A
    delegating dispatch must not read it."""

    kind = "talon"
    signal = None

    def __init__(self):
        self.reads = []

    async def finished_runs(self):
        self.reads.append("finished_runs")
        return {"complete": True, "runs": [], "reason": ""}

    async def workspace_readiness(self, repo):
        self.reads.append(f"workspace_readiness:{repo}")
        return {"repo": repo, "provisioned": True, "workspace": f"/srv/{repo}"}


def _agent(claw, name="Emma"):
    agent = SimpleNamespace(
        agent_name=name,
        execute_named_tool=claw.execute_named_tool,
        get_turn_bound_session_id=lambda: "chat-7",
        operator_registry=SimpleNamespace(
            get_workflow_registration=lambda workflow: SimpleNamespace(owner="feature:x")
        ),
        wait_registry=WaitRegistry(),
        talon=_EmmasOwnTalon(),
    )
    agent.wait_registry.register(agent.talon)
    return agent


def _feature(agent, *issues):
    feature = StrategicMemoryFeature(agent=agent)
    feature._data = {"morning_signal_config": {"scan_repos": ["o/r"]}}
    feature._strategy_path = Path("/tmp/kestrel-test/STRATEGY.yaml")
    feature._save = MagicMock(return_value=_SaveOutcome(persisted=True))
    feature._ledger = StrategyLedger(Path("/tmp/kestrel-test/STRATEGY_LEDGER.yaml"))
    feature._ledger.data[BLOCKERS_KEY] = [
        {"severity": "high", "issue": f"o/r#{number}", "title": f"issue {number}"}
        for number in (issues or (3093,))
    ]
    feature._ledger.data[PATTERNS_KEY] = []
    feature._ledger.normalize()
    feature._ledger.save = MagicMock(return_value=None)
    return feature


def _github(monkeypatch, *issues, timeline=()):
    """GitHub as the selector sees it: each issue open and ``agent-ready``,
    no PR on it, and ``timeline`` on each since its last run."""
    from kestrel_sovereign.features.strategic_memory import issue_selection

    monkeypatch.setattr(issue_selection, "get_github_token", lambda: "token")
    numbers = set(issues or (3093,))

    async def fake_get(path, token):
        for number in numbers:
            if path == f"/repos/o/r/issues/{number}":
                return {
                    "number": number,
                    "title": f"issue {number}",
                    "state": "open",
                    "repository_url": "https://api.github.com/repos/o/r",
                    "labels": [{"name": "agent-ready"}],
                }
        if "/issues?" in path:
            return []
        raise RuntimeError(f"404 {path}")

    async def fake_post(path, token, body):
        if "timelineItems" in body["query"]:
            return {"data": {"repository": {"issue": {
                "lastEditedAt": None,
                "timelineItems": {"nodes": list(timeline)},
            }}}}
        return {"data": {"repository": {"issue": {
            "closedByPullRequestsReferences": {"nodes": []},
        }}}}

    monkeypatch.setattr(issue_selection, "github_api_get", fake_get)
    monkeypatch.setattr(issue_selection, "github_api_post", fake_post)


# ---------------------------------------------------------------------------
# 1. Delegated execute
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_sends_the_dispatcher_a_work_order_and_arms_its_wake(monkeypatch):
    _github(monkeypatch)
    claw = _Claw()
    agent = _agent(claw)
    feature = _feature(agent)

    result = await feature.signal_dispatch(mode="execute")

    assert result.status is ToolResultStatus.OK, result.error
    assert result.data["dispatched"] is True
    assert result.data["executor"] == "Claw"
    assert "workflow_run" not in claw.names()
    send = claw.args("send_a2a_task")
    assert send["recipient"] == "Claw"
    [part] = send["parts"]
    order = part["data"]
    assert order["schema"] == WORK_ORDER_SCHEMA
    assert (order["repo"], order["issue"], order["action"]) == ("o/r", 3093, "claim")
    assert order["order_id"] == result.data["order_id"]
    assert order["order_id"].startswith("wo-3093-")
    dispatch_id = result.data["dispatch_id"]
    assert dispatch_id and result.data["remote_task_id"] == f"task-{dispatch_id}"
    assert claw.args("wait") == {
        "target": f"a2a:{dispatch_id}",
        "mode": "signal",
        "reason": f"Claw's report on work order {order['order_id']}",
    }
    assert result.data["wake"] == f"a2a:{dispatch_id}"
    assert f"as dispatch {dispatch_id}" in result.confirmation
    # Emma's own Talon is never asked: its registry and workspaces are not
    # where the work runs.
    assert agent.talon.reads == []
    assert claw.names() == [
        POLICY_TOOL, LIST_OUTBOUND_TOOL, "send_a2a_task", "wait",
    ]


@pytest.mark.asyncio
async def test_the_dispatcher_itself_still_launches_locally(monkeypatch):
    _github(monkeypatch)
    claw = _Claw(dispatcher="Claw")
    agent = _agent(claw, name="Claw")
    feature = _feature(agent)

    result = await feature.signal_dispatch(mode="execute")

    assert result.status is ToolResultStatus.OK, result.error
    assert result.data["dispatched"] is True
    assert "executor" not in result.data
    assert claw.names() == [POLICY_TOOL, "workflow_run"]
    assert agent.talon.reads == ["finished_runs", "workspace_readiness:o/r"]


@pytest.mark.asyncio
async def test_a_policy_without_a_dispatcher_launches_locally(monkeypatch):
    _github(monkeypatch)
    claw = _Claw(dispatcher=None)
    feature = _feature(_agent(claw))

    result = await feature.signal_dispatch(mode="execute")

    assert result.data["dispatched"] is True
    assert claw.names() == [POLICY_TOOL, "workflow_run"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["execute", "suggest"])
@pytest.mark.parametrize(
    "broken",
    [
        ToolResult.failed("talon.policy.dispatcher must be the agent_name ..."),
        ToolResult.ok("loaded", data={"success": True}),
        ToolResult.ok("loaded", data={"success": True, "policy": _talon_policy("  ")}),
        ToolNotRegisteredError("Tool 'talon_get_config' not registered"),
        RuntimeError("config unreadable"),
    ],
)
async def test_an_unconfirmed_dispatcher_policy_dispatches_nothing(mode, broken, monkeypatch):
    """Launching here could be refused as TALON_NOT_DISPATCHER every day, and
    delegating could hand the work to an agent Talon does not name."""
    _github(monkeypatch)
    claw = _Claw()
    if isinstance(broken, Exception):
        if isinstance(broken, ToolNotRegisteredError):
            claw.unregistered.add(POLICY_TOOL)
        else:
            claw.raises[POLICY_TOOL] = broken
    else:
        claw.policy_result = broken
    agent = _agent(claw)
    feature = _feature(agent)

    result = await feature.signal_dispatch(mode=mode)

    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["reason_code"] == "DISPATCHER_POLICY_UNCONFIRMED"
    assert result.data["dispatched"] is False
    assert claw.names() == [POLICY_TOOL]
    assert agent.talon.reads == []


@pytest.mark.asyncio
async def test_suggest_names_the_executor_and_sends_nothing(monkeypatch):
    _github(monkeypatch)
    claw = _Claw()
    feature = _feature(_agent(claw))

    result = await feature.signal_dispatch(mode="suggest")

    assert result.status is ToolResultStatus.OK
    assert result.data["issue"]["issue_number"] == 3093
    assert result.data["dispatched"] is False
    assert "send Claw a `kestrel.work_order/v1` work order" in result.confirmation
    assert claw.names() == [POLICY_TOOL, LIST_OUTBOUND_TOOL]


@pytest.mark.asyncio
async def test_a_refused_send_is_rejected_and_arms_nothing(monkeypatch):
    _github(monkeypatch)
    claw = _Claw()
    claw.send_result = ToolResult.failed(
        "Could not send A2A task: peer not found",
        data={"sent": False, "dispatch_id": "d-1", "delivery_state": "not-sent"},
    )
    feature = _feature(_agent(claw))

    result = await feature.signal_dispatch(mode="execute")

    assert result.status is ToolResultStatus.ERROR
    assert result.data["reason_code"] == "WORK_ORDER_REJECTED"
    assert result.data["dispatched"] is False
    assert "wait" not in claw.names()


@pytest.mark.asyncio
async def test_an_uncertain_send_is_unconfirmed_and_still_arms_its_wake(monkeypatch):
    """The order may have reached Claw: its wait reconciles the dispatch, so
    whatever Claw makes of it still wakes Emma."""
    _github(monkeypatch)
    claw = _Claw()
    claw.send_result = ToolResult.failed(
        "A2A delivery is uncertain; reconcile the dispatch before sending again",
        data={"sent": None, "dispatch_id": "d-1", "delivery_state": "unknown"},
    )
    feature = _feature(_agent(claw))

    result = await feature.signal_dispatch(mode="execute")

    assert result.status is ToolResultStatus.ERROR
    assert result.data["reason_code"] == "WORK_ORDER_UNCONFIRMED"
    assert result.data["dispatched"] is None
    assert result.data["dispatch_id"] == "d-1"
    assert claw.args("wait")["target"] == "a2a:d-1"
    assert "Do not send it again" in result.error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "send_result, dispatched, code",
    [
        (
            {"success": False, "error": "Permission denied: send_a2a_task"},
            False,
            "WORK_ORDER_REJECTED",
        ),
        (
            ToolResult.failed("A2A delivery was acknowledged", data={"sent": True}),
            True,
            "WORK_ORDER_UNCONFIRMED",
        ),
        (
            ToolResult.failed("A2A delivery is uncertain", data={"sent": None}),
            None,
            "WORK_ORDER_UNCONFIRMED",
        ),
        (ToolResult.ok("sent", data={"sent": True}), True, "WORK_ORDER_UNCONFIRMED"),
    ],
)
async def test_a_send_without_a_dispatch_is_classified_by_what_peers_reported(
    send_result, dispatched, code, monkeypatch
):
    _github(monkeypatch)
    claw = _Claw()
    claw.send_result = send_result
    feature = _feature(_agent(claw))

    result = await feature.signal_dispatch(mode="execute")

    assert result.status is ToolResultStatus.ERROR
    assert (result.data["dispatched"], result.data["reason_code"]) == (dispatched, code)
    assert "wait" not in claw.names()


@pytest.mark.asyncio
async def test_an_unarmed_wake_is_reported_with_the_order_it_belongs_to(monkeypatch):
    _github(monkeypatch)
    claw = _Claw()
    claw.wait_result = ToolResult.failed("wait engine unavailable")
    feature = _feature(_agent(claw))

    result = await feature.signal_dispatch(mode="execute")

    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["reason_code"] == "WORK_ORDER_WAKE_UNARMED"
    assert result.data["dispatched"] is True
    assert result.data["wake"] is None
    assert "wait engine unavailable" in result.error
    assert f"arm wait(target='a2a:{result.data['dispatch_id']}'" in result.confirmation


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", [LIST_OUTBOUND_TOOL, "send_a2a_task"])
async def test_without_the_peers_tools_no_order_is_sent(tool, monkeypatch):
    _github(monkeypatch)
    claw = _Claw()
    claw.unregistered.add(tool)
    feature = _feature(_agent(claw))

    result = await feature.signal_dispatch(mode="execute")

    assert result.status is ToolResultStatus.ERROR
    assert result.data["reason_code"] == "WORK_ORDER_UNAVAILABLE"
    assert result.data["dispatched"] is False
    assert "wait" not in claw.names()


@pytest.mark.asyncio
async def test_concurrent_dispatches_send_one_order(monkeypatch):
    """A scheduled dispatch and a turn's cannot both find nothing outstanding."""
    _github(monkeypatch, 3093, 3094)
    claw = _Claw()
    feature = _feature(_agent(claw), 3093, 3094)

    first, second = await asyncio.gather(
        feature.signal_dispatch(mode="execute"),
        feature.signal_dispatch(mode="execute"),
    )

    assert claw.names().count("send_a2a_task") == 1
    assert sorted([first.data["dispatched"], second.data["dispatched"]]) == [False, True]
    busy = first if first.data["dispatched"] is False else second
    assert [order["dispatch_id"] for order in busy.data["outstanding"]] == [
        claw.rows[0].dispatch_id
    ]


# ---------------------------------------------------------------------------
# 2. The guards read the executor's reports
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("disposition", ["clarifying", "blocked"])
@pytest.mark.parametrize("mode", ["execute", "suggest"])
async def test_a_report_that_asked_a_question_is_not_redispatched(
    disposition, mode, monkeypatch
):
    """#3398 through the executor: Claw reported #3093 clarifying and nothing
    on the issue since authorizes a retry, so it is skipped, with why."""
    _github(monkeypatch)
    claw = _Claw()
    claw.answered("d1", TaskState.TASK_STATE_COMPLETED, _report(disposition))
    agent = _agent(claw)
    feature = _feature(agent)

    result = await feature.signal_dispatch(mode=mode)

    assert result.status is ToolResultStatus.OK
    assert result.data["dispatched"] is False
    assert result.data["issue"] is None
    [skip] = result.data["skipped"]
    assert skip["reason"] == "run_history"
    assert (skip["executor"], skip["order_id"], skip["dispatch_id"]) == (
        "Claw", "wo-3093-1", "d1",
    )
    assert skip["job_id"] == TALON_JOB
    assert (
        f"skipped o/r#3093 -- Claw reported work order wo-3093-1 {disposition} at "
        "2026-10-24 09:56 UTC without a PR (Talon job 802b93fe), and nothing "
        "since authorizes a retry"
    ) in result.confirmation
    assert "send_a2a_task" not in claw.names()
    assert agent.talon.reads == []


@pytest.mark.asyncio
async def test_a_comment_after_the_report_authorizes_the_retry(monkeypatch):
    _github(monkeypatch, timeline=[{
        "__typename": "IssueComment",
        "createdAt": "2026-10-24T12:00:00Z",
        "authorAssociation": "MEMBER",
        "body": "Answered the question above; go ahead.",
    }])
    claw = _Claw()
    claw.answered("d1", TaskState.TASK_STATE_COMPLETED, _report("clarifying"))
    feature = _feature(_agent(claw))

    result = await feature.signal_dispatch(mode="execute")

    assert result.data["dispatched"] is True
    assert claw.args("send_a2a_task")["parts"][0]["data"]["issue"] == 3093


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state", [TaskState.TASK_STATE_FAILED, TaskState.TASK_STATE_REJECTED]
)
async def test_an_order_the_executor_refused_waits_for_new_input(state, monkeypatch):
    """The executor's refusal is its workspace readiness: core cannot ask its
    Talon, and the same order against unchanged input is refused again."""
    _github(monkeypatch)
    claw = _Claw()
    claw.answered("d1", state, "No talon workspace exists for o/r.")
    agent = _agent(claw)
    feature = _feature(agent)

    result = await feature.signal_dispatch(mode="execute")

    assert result.data["dispatched"] is False
    [skip] = result.data["skipped"]
    assert (skip["reason"], skip["disposition"]) == ("run_history", ORDER_REFUSED)
    assert "skipped o/r#3093 -- Claw refused work order wo-3093-1 at 2026-10-24 09:56 UTC" in (
        result.confirmation
    )
    assert "send_a2a_task" not in claw.names()
    assert agent.talon.reads == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer, state",
    [
        (_report("completed"), TaskState.TASK_STATE_COMPLETED),
        (_report("failed"), TaskState.TASK_STATE_COMPLETED),
        ("not a report", TaskState.TASK_STATE_COMPLETED),
        (None, TaskState.TASK_STATE_CANCELED),
    ],
)
async def test_an_answer_that_asked_nothing_holds_nothing_back(answer, state, monkeypatch):
    _github(monkeypatch)
    claw = _Claw()
    claw.answered("d1", state, answer)
    feature = _feature(_agent(claw))

    result = await feature.signal_dispatch(mode="execute")

    assert result.data["dispatched"] is True
    assert result.data["skipped"] == []


@pytest.mark.asyncio
async def test_the_newest_order_on_an_issue_decides_it(monkeypatch):
    """#3093 was re-ordered after new input and Claw opened a PR: the older
    clarifying answer no longer holds it back."""
    _github(monkeypatch)
    claw = _Claw()
    claw.answered("old", TaskState.TASK_STATE_COMPLETED, _report("clarifying"))
    claw.answered("new", TaskState.TASK_STATE_COMPLETED, _report("completed"))
    feature = _feature(_agent(claw))

    result = await feature.signal_dispatch(mode="suggest")

    assert result.data["issue"]["issue_number"] == 3093
    assert result.data["skipped"] == []


@pytest.mark.asyncio
async def test_a_run_is_keyed_by_the_issue_the_order_named(monkeypatch):
    """A report that names another issue cannot hold that other issue back,
    and does not release the one its order named."""
    _github(monkeypatch, 3093, 3094)
    claw = _Claw()
    claw.answered(
        "d1", TaskState.TASK_STATE_COMPLETED, _report("clarifying", issue=3094),
        order=_order(3093),
    )
    feature = _feature(_agent(claw), 3093, 3094)

    result = await feature.signal_dispatch(mode="suggest")

    assert result.data["issue"]["issue_number"] == 3094
    assert [(s["issue_number"], s["reason"]) for s in result.data["skipped"]] == [
        (3093, "run_history"),
    ]


@pytest.mark.asyncio
async def test_settled_dispatches_are_read_from_the_executor_once(monkeypatch):
    """A terminal task cannot change, so it is read once. A live task that
    carries no order is read again: a later read may show more history."""
    _github(monkeypatch, 3093, 3094)
    claw = _Claw()
    claw.answered("d1", TaskState.TASK_STATE_COMPLETED, _report("clarifying"))
    claw.rows.insert(0, _row("plain"))
    claw.reads["plain"] = _read(
        "plain", [{"text": "how are you"}], TaskState.TASK_STATE_COMPLETED, answer="fine"
    )
    claw.rows.insert(0, _row("chat"))
    claw.reads["chat"] = _read("chat", [{"text": "still there?"}], TaskState.TASK_STATE_WORKING)
    feature = _feature(_agent(claw), 3093, 3094)

    await feature.signal_dispatch(mode="suggest")
    await feature.signal_dispatch(mode="suggest")

    assert claw.peer_reads() == ["chat", "plain", "d1", "chat"]


@pytest.mark.asyncio
async def test_an_order_sent_as_json_text_is_recognized(monkeypatch):
    """An order sent by hand as the JSON object in a text part is one too."""
    _github(monkeypatch)
    claw = _Claw()
    claw.rows.insert(0, _row("manual"))
    claw.reads["manual"] = _read(
        "manual", [{"text": json.dumps(_order())}], TaskState.TASK_STATE_WORKING
    )
    feature = _feature(_agent(claw))

    result = await feature.signal_dispatch(mode="execute")

    assert result.data["dispatched"] is False
    [order] = result.data["outstanding"]
    assert (order["dispatch_id"], order["issue_number"]) == ("manual", 3093)


@pytest.mark.asyncio
async def test_a_task_that_shows_no_message_from_this_agent_may_be_an_order(monkeypatch):
    _github(monkeypatch)
    claw = _Claw()
    claw.rows.insert(0, _row("bare"))
    claw.reads["bare"] = _read("bare", None, TaskState.TASK_STATE_WORKING)
    feature = _feature(_agent(claw))

    result = await feature.signal_dispatch(mode="execute")

    assert result.data["dispatched"] is False
    [order] = result.data["outstanding"]
    assert order["dispatch_id"] == "bare" and order["issue_number"] is None
    assert "shows no message from this agent" in order["detail"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["execute", "suggest"])
async def test_a_refusal_no_issue_can_be_placed_on_is_reported_not_applied(mode, monkeypatch):
    """Review finding: a terminal answer whose task names no issue can hold
    none back, and failing every later selection on it would stop delegated
    dispatch for good. It is reported with every outcome instead."""
    _github(monkeypatch)
    claw = _Claw()
    claw.rows.insert(0, _row("bare"))
    claw.reads["bare"] = _read("bare", None, TaskState.TASK_STATE_FAILED, answer="no")
    feature = _feature(_agent(claw))

    result = await feature.signal_dispatch(mode=mode)

    assert result.status is ToolResultStatus.OK
    assert result.data["issue"]["issue_number"] == 3093
    assert result.data["dispatched"] is (mode == "execute")
    [note] = result.data["unplaced_answers"]
    assert "Claw answered dispatch bare refused" in note
    assert "which issue it holds back is unknown" in note


@pytest.mark.asyncio
async def test_an_undated_question_is_dated_when_it_was_first_read(monkeypatch):
    """Review finding: an answer with no status time still holds its issue
    back, dated no earlier than the answer -- when this agent read it -- so a
    comment before that cannot lift it, and nothing else is held back."""
    _github(monkeypatch, 3093, 3094, timeline=[{
        "__typename": "IssueComment",
        "createdAt": "2026-10-24T12:00:00Z",
        "authorAssociation": "MEMBER",
        "body": "Answered.",
    }])
    claw = _Claw()
    claw.answered("d1", TaskState.TASK_STATE_COMPLETED, _report("blocked"), at=None)
    feature = _feature(_agent(claw), 3093, 3094)
    before = datetime.now(timezone.utc)

    result = await feature.signal_dispatch(mode="execute")

    assert result.data["dispatched"] is True
    assert result.data["issue"]["issue_number"] == 3094
    [skip] = result.data["skipped"]
    assert (skip["issue_number"], skip["reason"]) == (3093, "run_history")
    assert datetime.fromisoformat(skip["completed_at"]) >= before


@pytest.mark.asyncio
async def test_a_read_with_neither_task_nor_message_holds_the_lane(monkeypatch):
    _github(monkeypatch)
    claw = _Claw()
    claw.rows.insert(0, _row("odd"))
    claw.reads["odd"] = ToolResult.ok("Read native peer response", data={"dispatch_id": "odd"})
    feature = _feature(_agent(claw))

    result = await feature.signal_dispatch(mode="execute")

    assert result.data["dispatched"] is False
    [order] = result.data["outstanding"]
    assert order["detail"] == "its read returned neither a task nor a message"


@pytest.mark.asyncio
async def test_a_superseded_undated_answer_does_not_block_forever(monkeypatch):
    """Review finding: an old answer that cannot be dated is irrelevant once a
    newer order on the same issue has been answered."""
    _github(monkeypatch)
    claw = _Claw()
    claw.answered("old", TaskState.TASK_STATE_FAILED, "refused", at=None)
    claw.answered("new", TaskState.TASK_STATE_COMPLETED, _report("completed"))
    feature = _feature(_agent(claw))

    result = await feature.signal_dispatch(mode="execute")

    assert result.data["dispatched"] is True, result.confirmation


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["execute", "suggest"])
async def test_an_unreadable_settled_answer_dispatches_nothing(mode, monkeypatch):
    _github(monkeypatch)
    claw = _Claw()
    claw.rows.insert(0, _row("d1", DispatchState.COMPLETE))
    claw.reads["d1"] = ToolResult.failed("Could not read native peer task: 503")
    feature = _feature(_agent(claw))

    result = await feature.signal_dispatch(mode=mode)

    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["reason_code"] == "RUN_HISTORY_UNCONFIRMED"
    assert result.data["executor"] == "Claw"
    assert "Could not confirm Claw's work reports" in result.confirmation
    assert "503" in result.confirmation
    assert "send_a2a_task" not in claw.names()


# ---------------------------------------------------------------------------
# 3. Unsettled dispatches hold the lane; no listing page hides anything
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["execute", "suggest"])
async def test_a_reserved_dispatch_is_in_flight_not_idle(mode, monkeypatch):
    """Review P1 on #3758: a send reserved but not yet submitted can still
    reach Claw, and nothing can say which issue it carries."""
    _github(monkeypatch)
    claw = _Claw()
    claw.rows.insert(0, _row("reserved", DispatchState.RESERVED))
    feature = _feature(_agent(claw))

    result = await feature.signal_dispatch(mode=mode)

    assert result.status is ToolResultStatus.OK
    assert result.data["dispatched"] is False
    [order] = result.data["outstanding"]
    assert (order["dispatch_id"], order["delivery_state"]) == ("reserved", "reserved")
    assert "Claw has not settled dispatch reserved" in result.confirmation
    assert "send_a2a_task" not in claw.names()
    # Its content cannot be read until it is submitted, so it is not asked.
    assert claw.peer_reads() == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state",
    [
        TaskState.TASK_STATE_SUBMITTED,
        TaskState.TASK_STATE_WORKING,
        TaskState.TASK_STATE_INPUT_REQUIRED,
    ],
)
async def test_an_unanswered_order_holds_the_lane(state, monkeypatch):
    _github(monkeypatch)
    claw = _Claw()
    claw.pending("d1", state)
    feature = _feature(_agent(claw))

    result = await feature.signal_dispatch(mode="execute")

    assert result.data["dispatched"] is False
    [order] = result.data["outstanding"]
    assert (order["order_id"], order["repo"], order["issue_number"]) == (
        "wo-3093-1", "o/r", 3093,
    )
    assert order["task_state"] == task_state_label(state)
    assert "work order wo-3093-1 on o/r#3093 (dispatch d1): Claw has not answered it" in (
        result.confirmation
    )
    assert "send_a2a_task" not in claw.names()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "delivery", [DispatchState.SUBMITTING, DispatchState.UNKNOWN, DispatchState.ACKNOWLEDGED]
)
async def test_a_dispatch_whose_task_cannot_be_read_holds_the_lane(delivery, monkeypatch):
    _github(monkeypatch)
    claw = _Claw()
    claw.rows.insert(0, _row("d1", delivery))
    claw.reads["d1"] = ToolResult.failed(
        "Delivery is still uncertain; no matching authorized peer Task was found",
        data={"dispatch_id": "d1", "delivery_state": "unknown"},
    )
    feature = _feature(_agent(claw))

    result = await feature.signal_dispatch(mode="execute")

    assert result.data["dispatched"] is False
    [order] = result.data["outstanding"]
    assert order["dispatch_id"] == "d1"
    assert "its task could not be read" in order["detail"]
    assert "send_a2a_task" not in claw.names()


@pytest.mark.asyncio
async def test_dispatches_that_cannot_carry_an_order_do_not_hold_the_lane(monkeypatch):
    """Messages and ``ask_agent`` send text only, a continuation is more input
    to its root's task, a proven non-delivery reached nobody, and a task that
    carries no work order is not one."""
    _github(monkeypatch)
    claw = _Claw()
    claw.rows.insert(0, _row("msg", DispatchState.RESERVED, verb="message", tool="send_a2a_message"))
    claw.rows.insert(0, _row("ask", DispatchState.RESERVED, tool="ask_agent"))
    claw.rows.insert(0, _row("cont", DispatchState.RESERVED, parent="root"))
    claw.rows.insert(0, _row("unsent", DispatchState.NOT_SENT))
    claw.rows.insert(0, _row("instant", DispatchState.COMPLETE, response_kind="message"))
    claw.rows.insert(0, _row("plain"))
    claw.reads["plain"] = _read("plain", [{"text": "hello"}], TaskState.TASK_STATE_WORKING)
    feature = _feature(_agent(claw))

    result = await feature.signal_dispatch(mode="execute")

    assert result.data["dispatched"] is True, result.confirmation
    assert claw.peer_reads() == ["plain"]


@pytest.mark.asyncio
@pytest.mark.parametrize("newer", [60, OUTBOUND_PAGE_SIZE + 50])
async def test_a_blocking_answer_is_not_lost_behind_newer_dispatches(newer, monkeypatch):
    """Review finding on #3758: one ``list_recent(50)`` page loses an older
    clarifying answer once 50 newer dispatches exist. The read pages to the
    end of the audit."""
    _github(monkeypatch)
    claw = _Claw()
    claw.answered("d1", TaskState.TASK_STATE_COMPLETED, _report("clarifying"))
    claw.unrelated(newer)
    feature = _feature(_agent(claw))

    result = await feature.signal_dispatch(mode="execute")

    assert result.data["dispatched"] is False
    assert [skip["dispatch_id"] for skip in result.data["skipped"]] == ["d1"]
    listings = [args for name, args in claw.calls if name == LIST_OUTBOUND_TOOL]
    assert all(args["limit"] == OUTBOUND_PAGE_SIZE for args in listings)
    assert len(listings) == newer // OUTBOUND_PAGE_SIZE + 1


@pytest.mark.asyncio
async def test_an_unanswered_order_is_not_lost_behind_newer_dispatches(monkeypatch):
    _github(monkeypatch)
    claw = _Claw()
    claw.pending("d1")
    claw.unrelated(OUTBOUND_PAGE_SIZE * 2)
    feature = _feature(_agent(claw))

    result = await feature.signal_dispatch(mode="execute")

    assert result.data["dispatched"] is False
    assert [order["dispatch_id"] for order in result.data["outstanding"]] == ["d1"]
    assert "send_a2a_task" not in claw.names()


# ---------------------------------------------------------------------------
# The reader on its own
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_listing_that_repeats_a_row_is_unreadable():
    claw = _Claw()
    claw.unrelated(OUTBOUND_PAGE_SIZE)
    claw.rows.append(claw.rows[0])

    with pytest.raises(RunHistoryUnreadable, match="twice"):
        await read_executor_run_history("Claw", lambda n, a: _call(claw, n, a))


@pytest.mark.asyncio
@pytest.mark.parametrize("issue", ["", "three", None])
async def test_a_question_on_an_order_that_names_no_issue_is_reported(issue):
    claw = _Claw()
    claw.answered(
        "d1", TaskState.TASK_STATE_COMPLETED, _report("blocked"),
        order={**_order(), "issue": issue},
    )

    history = await read_executor_run_history("Claw", lambda n, a: _call(claw, n, a))

    assert history.runs == {}
    [note] = history.unplaced
    assert "does not identify an issue" in note


@pytest.mark.asyncio
async def test_the_order_is_read_back_through_the_native_round_trip():
    """A data part comes back as a protobuf Value: the issue is a double."""
    claw = _Claw()
    claw.answered("d1", TaskState.TASK_STATE_COMPLETED, _report("clarifying"))
    assert claw.reads["d1"].data["task"]["history"][0]["parts"][0]["data"]["issue"] == 3093.0

    history = await read_executor_run_history("Claw", lambda n, a: _call(claw, n, a))

    run = history.latest("O/R", 3093)
    assert (run.disposition, run.completed_at, run.executor) == ("clarifying", ANSWERED, "Claw")
    assert run.awaits_new_input
    assert history.outstanding == ()
    assert describe_exclusion({
        "repo": "o/r", "issue_number": 3093, "reason": "run_history",
        "job_id": run.job_id, "disposition": run.disposition,
        "completed_at": run.completed_at.isoformat(), "executor": "Claw",
        "order_id": run.order_id, "dispatch_id": run.dispatch_id,
    }).startswith("skipped o/r#3093 -- Claw reported work order wo-3093-1 clarifying")


async def _call(claw, name, args):
    return await claw.execute_named_tool(
        name, args, session_id="chat-7", source="strategic_memory.signal_dispatch"
    )
