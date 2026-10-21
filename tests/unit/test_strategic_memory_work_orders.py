"""Delegated execution for signal_dispatch (#3758).

The queue owner picks; another agent runs the pick. With
``morning_signal_config.dispatch_executor`` set, execute sends the pick as a
``kestrel.work_order/v1`` A2A task, records its task id and arms a wake on the
report, and the selection guards read the orders sent to that executor
instead of the selecting agent's own (empty) Talon registry.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from kestrel_sdk.a2a.types import (
    Artifact,
    DataPart,
    Message,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)
from kestrel_sdk.tools.result import ToolResult, ToolResultStatus
from kestrel_sovereign.agent.orchestrator_engine import ToolNotRegisteredError
from kestrel_sovereign.features.strategic_memory import issue_selection, work_orders
from kestrel_sovereign.features.strategic_memory.feature import (
    SIGNAL_DISPATCH_REASON_CODES,
)
from kestrel_sovereign.features.strategic_memory.run_history import (
    QUESTION_DISPOSITIONS,
    TalonRun,
)
from kestrel_sovereign.features.strategic_memory.work_orders import (
    AWAITING_INPUT_DISPOSITIONS,
    RESERVATION_LEASE,
    InvalidDispatchExecutor,
    OrderedRun,
    WorkOrderJournal,
    WorkOrderNotReleasable,
    WorkOrdersUnreadable,
    attempt_outcome,
    build_work_order,
    dispatch_executor,
    open_work_order_journal,
    parse_work_report,
    read_work_orders,
    work_order_message,
    work_order_window,
)
from kestrel_sovereign.endpoints.agent import _task_http_payload
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.db import SQLiteBackend
from tests.unit.test_a2a_outbound_store import (
    _async_client_with,
    _make_feature as _make_peers,
    _mock_post_response,
)
from tests.utils.postgres_schema import postgres_test_url
from tests.unit.test_strategic_memory_toolresult import (
    _TalonJobs,
    _agent_ready_issue,
    _make_feature,
    _talon_run,
    _with_talon,
)

_REGISTER_WATCH = (
    "kestrel_sovereign.features.strategic_memory.feature.register_wait_watch"
)
_PEERS_HTTP = "kestrel_sovereign.features.peers.feature.httpx.AsyncClient"
NOW = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)
DID = "did:emma"


@pytest_asyncio.fixture
async def db(tmp_path):
    """The queue owner's own database, where the journal is kept."""
    raw = SQLiteBackend(str(tmp_path / "work-orders.db"))
    await raw.connect()
    database = AsyncDatabase(raw)
    try:
        yield database
    finally:
        await database.close()


@pytest_asyncio.fixture
async def journal(db):
    journal = WorkOrderJournal(db, DID)
    await journal.ensure()
    return journal


@pytest_asyncio.fixture
async def shared_journal(db_backend):
    """A journal on each backend, under a DID no other test or run uses: the
    PostgreSQL database outlives the run and other cases share it, so the
    case removes exactly the rows it wrote."""
    database = AsyncDatabase(db_backend)
    journal = WorkOrderJournal(database, f"did:test:{uuid.uuid4().hex}")
    await journal.ensure()
    try:
        yield journal
    finally:
        await database.execute(
            f"DELETE FROM {work_orders.JOURNAL_TABLE} WHERE agent_id = ?",
            (journal.agent_id,),
        )


async def _second_connection(db_backend):
    """A second, independently connected writer on the same database."""
    if hasattr(db_backend, "db_path"):
        backend = SQLiteBackend(db_backend.db_path)
    else:
        from kestrel_sovereign.storage.db.postgres import PostgresBackend

        backend = PostgresBackend(postgres_test_url())
    await backend.connect()
    return AsyncDatabase(backend), backend


def _ago(now=None, **delta):
    return ((now or datetime.now(timezone.utc)) - timedelta(**delta)).isoformat()


def _row(task_id, *, sent, recipient="Claw", verb="task", terminal_state=None,
         terminal_at=None, session_id=None, route_state="routable", error=None):
    """One sender-side outbound row, as ``list_outbound_a2a_tasks`` returns it."""
    return {
        "id": f"row-{task_id}",
        "agent_id": DID,
        "task_id": task_id,
        "recipient": recipient,
        "verb": verb,
        "session_id": session_id or f"s-{task_id}",
        "skill_id": None,
        "dispatch_tool": "send_a2a_task",
        "message_summary": f"Work order for {task_id}",
        "created_at": sent,
        "route_state": route_state,
        "terminal_state": terminal_state,
        "terminal_at": terminal_at,
        "error": error,
    }


def _order(order_id, issue, repo="o/r"):
    return {
        "schema": "kestrel.work_order/v1",
        "order_id": order_id,
        "repo": repo,
        "issue": issue,
        "action": "claim",
        "talon": {},
        "verify": [],
        "context": "context",
    }


def _report(order_id, issue, disposition, repo="o/r", **extra):
    """A ``kestrel.work_report/v1`` as the executor answers an order."""
    return {
        "schema": "kestrel.work_report/v1",
        "order_id": order_id,
        "repo": repo,
        "issue": issue,
        "talon_job_id": f"job-{order_id}",
        "disposition": disposition,
        "pr": None,
        "question": None,
        "blockers": [],
        **extra,
    }


def _task(order=None, *, state="submitted", report=None):
    """The executor's task, as ``get_peer_task_result`` returns it."""
    if isinstance(report, dict):
        report = json.dumps(report)
    return {
        "state": state,
        "reply_text": report or "",
        "artifacts": (
            [{"name": "work_order", "parts": [{"type": "data", "data": order}], "index": 0}]
            if order is not None
            else []
        ),
    }


class _Peers:
    """PeersFeature's two reads, over a fixed outbound record and executor."""

    def __init__(self, rows=(), tasks=None, *, unreachable=(), list_error=None):
        self.rows = list(rows)
        self.tasks = dict(tasks or {})
        self.unreachable = set(unreachable)
        self.list_error = list_error
        self.fetched = []

    async def list_outbound_a2a_tasks(self, limit=50, recipient=""):
        if self.list_error is not None:
            return ToolResult.failed(self.list_error, data={"rows": [], "count": 0})
        return ToolResult.ok(
            f"Outbound A2A audit: {len(self.rows)} row(s)",
            data={"rows": list(self.rows), "count": len(self.rows)},
        )

    async def get_peer_task_result(self, recipient, task_id):
        self.fetched.append((recipient, task_id))
        if task_id in self.unreachable:
            return ToolResult.failed(
                f"Could not reach peer '{recipient}' for task {task_id}",
                data={"recipient": recipient, "task_id": task_id},
            )
        return ToolResult.ok(
            f"Fetched peer task {task_id[:8]}", data=dict(self.tasks[task_id])
        )


def _agent_with(peers):
    return SimpleNamespace(
        get_feature=lambda name: peers if name == "PeersFeature" else None
    )


# ---------------------------------------------------------------------------
# The contract the executor speaks
# ---------------------------------------------------------------------------


def test_the_work_order_vocabulary_matches_the_fleet_contract():
    """The work-order contract the queue owner and executor agreed
    (2026-10-20): a ``work_order`` data artifact of schema
    ``kestrel.work_order/v1``, answered once with a ``kestrel.work_report/v1``
    whose disposition is Talon's processing summary. Change both or neither."""
    assert work_orders.WORK_ORDER_SCHEMA == "kestrel.work_order/v1"
    assert work_orders.WORK_ORDER_ARTIFACT == "work_order"
    assert work_orders.WORK_REPORT_SCHEMA == "kestrel.work_report/v1"
    assert work_orders.REPORT_DISPOSITIONS == frozenset(
        {"completed", "clarifying", "blocked", "failed", "cancelled"}
    )
    assert work_orders.DISPATCH_EXECUTOR_KEY == "dispatch_executor"


def test_an_order_waits_for_new_input_after_a_question_a_cancel_or_a_refusal():
    """#3398's rule, widened for an executor: a question, a cancelled job,
    or an order the executor did not run all leave nothing new to run."""
    assert QUESTION_DISPOSITIONS <= AWAITING_INPUT_DISPOSITIONS
    assert {"cancelled", "order_failed", "order_canceled"} <= AWAITING_INPUT_DISPOSITIONS
    assert "completed" not in AWAITING_INPUT_DISPOSITIONS
    # A failed Talon run is labelled agent-failed, as a local one is.
    assert "failed" not in AWAITING_INPUT_DISPOSITIONS


def test_a_local_run_still_waits_only_after_a_question():
    def run(disposition):
        return TalonRun("j", "o/r", 1, disposition, NOW)

    assert run("clarifying").awaits_new_input
    assert run("blocked").awaits_new_input
    assert not run("cancelled").awaits_new_input
    assert not run("completed").awaits_new_input


def test_every_delegation_code_is_declared():
    assert {
        "INVALID_DISPATCH_EXECUTOR",
        "WORK_ORDERS_UNCONFIRMED",
        "WORK_ORDER_TRANSPORT_UNAVAILABLE",
        "WORK_ORDER_SEND_FAILED",
        "WORK_ORDER_REJECTED",
        "WORK_ORDER_WAKE_UNARMED",
    } <= SIGNAL_DISPATCH_REASON_CODES


@pytest.mark.parametrize(
    "config, expected",
    [({}, None), ({"dispatch_executor": None}, None),
     ({"dispatch_executor": "  "}, None), ({"dispatch_executor": " Claw "}, "Claw")],
)
def test_the_executor_is_an_explicit_opt_in(config, expected):
    assert dispatch_executor(config) == expected


@pytest.mark.parametrize("value", [["Claw"], 7, True, {"name": "Claw"}])
def test_an_executor_that_names_no_agent_is_refused(value):
    with pytest.raises(InvalidDispatchExecutor):
        dispatch_executor({"dispatch_executor": value})


def test_the_window_defaults_and_refuses_what_is_not_a_day_count():
    assert work_order_window({}) == timedelta(days=30)
    assert work_order_window({"work_order_window_days": 7}) == timedelta(days=7)
    for bad in (0, -1, "7", 1.5, True):
        with pytest.raises(WorkOrdersUnreadable):
            work_order_window({"work_order_window_days": bad})


def test_a_work_order_is_a_bounded_fresh_claim():
    issue = {
        "repo": "o/r", "issue_number": 42, "issue_title": "Repair\nthe boundary",
        "priority": "high", "context": " ".join(["word"] * 400),
    }

    order = build_work_order(issue)

    assert order["schema"] == "kestrel.work_order/v1"
    assert order["order_id"].startswith("wo-42-")
    assert (order["repo"], order["issue"], order["action"]) == ("o/r", 42, "claim")
    assert order["talon"] == {} and order["verify"] == []
    assert len(order["context"].split()) == 300
    assert order["context"].startswith("Repair the boundary. Selected by signal_dispatch")
    assert build_work_order(issue)["order_id"] != order["order_id"]
    message = work_order_message(order, issue)
    assert "\n" not in message
    assert message == (
        f"Work order {order['order_id']}: claim o/r#42 (Repair the boundary). "
        "See the work_order artifact."
    )


@pytest.mark.parametrize(
    "text, problem",
    [
        pytest.param("", "empty", id="empty"),
        pytest.param("Done, opened PR #5.", "not a JSON object", id="prose"),
        pytest.param("[1, 2]", "not a JSON object", id="array"),
        pytest.param(
            json.dumps({"schema": "other/v1", "disposition": "completed"}), "schema",
            id="other-schema",
        ),
        pytest.param(
            json.dumps(_report("wo-1", 7, "shipped")), "disposition",
            id="unknown-disposition",
        ),
        pytest.param(
            json.dumps(_report("wo-2", 7, "completed")), "order 'wo-2'",
            id="other-order",
        ),
        pytest.param(
            json.dumps(_report("wo-1", 8, "completed")), "issue 8", id="other-issue",
        ),
        pytest.param(
            json.dumps(_report("wo-1", 7, "completed", repo="o/x")),
            "repository 'o/x'",
            id="other-repository",
        ),
        pytest.param("[" * 100_000, "not a JSON object", id="deeply-nested"),
    ],
)
def test_a_report_that_is_not_one_says_why(text, problem):
    with pytest.raises(ValueError, match=problem):
        parse_work_report(text, order_id="wo-1", repo="o/r", issue_number=7)


def test_a_fenced_report_is_read():
    text = "```json\n" + json.dumps(_report("wo-1", 7, "Blocked", repo="O/R")) + "\n```"

    report = parse_work_report(text, order_id="wo-1", repo="o/r", issue_number=7)

    assert report["disposition"] == "blocked"


# ---------------------------------------------------------------------------
# Reading the ledger
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_ledger_reads_every_order_sent_to_the_executor(journal):
    rows = [
        _row("t-open", sent=_ago(NOW, hours=1)),
        _row("t-asked", sent=_ago(NOW, days=2), terminal_state="completed",
             terminal_at=_ago(NOW, days=1)),
        _row("t-refused", sent=_ago(NOW, days=3), terminal_state="failed",
             terminal_at=_ago(NOW, days=3, minutes=-5)),
        # A question, a message, another peer, a failed send, a task that is
        # no work order, and an order older than the window: none is an
        # order in the executor's lane.
        _row("t-question", sent=_ago(NOW, hours=2), verb="question"),
        _row("t-message", sent=_ago(NOW, hours=2), verb="message"),
        _row("t-meridian", sent=_ago(NOW, hours=2), recipient="Meridian"),
        _row("t-lost", sent=_ago(NOW, hours=2), terminal_state="dispatch_failed"),
        _row("t-chat", sent=_ago(NOW, hours=2)),
        _row("t-old", sent=_ago(NOW, days=40)),
    ]
    peers = _Peers(rows, {
        "t-open": _task(_order("wo-1-a", 1), state="working"),
        "t-asked": _task(
            _order("wo-2-a", 2), state="completed",
            report=_report("wo-2-a", 2, "clarifying"),
        ),
        "t-refused": _task(_order("wo-3-a", 3), state="failed", report="wrong sender"),
        "t-chat": _task(None, state="completed", report="hello"),
    })

    ledger = await read_work_orders(
        _agent_with(peers), "claw", journal=journal, window=timedelta(days=30), now=NOW
    )

    assert [order.order_id for order in ledger.orders] == ["wo-1-a", "wo-2-a", "wo-3-a"]
    assert sorted(task for _, task in peers.fetched) == [
        "t-asked", "t-chat", "t-open", "t-refused",
    ]
    assert not ledger.lane_idle
    assert ledger.lane() == {
        "executor": "claw",
        "idle": False,
        "outstanding": [{
            "order_id": "wo-1-a", "a2a_task_id": "t-open", "repo": "o/r",
            "issue_number": 1, "state": "working", "delivery": "sent",
            "sent_at": _ago(NOW, hours=1),
        }],
    }
    history = ledger.run_history()
    asked = history.latest("O/R", 2)
    assert isinstance(asked, OrderedRun)
    assert (asked.disposition, asked.job_id, asked.executor) == (
        "clarifying", "job-wo-2-a", "Claw",
    )
    assert asked.awaits_new_input
    assert asked.completed_at.isoformat() == _ago(NOW, days=1)
    refused = history.latest("o/r", 3)
    assert refused.disposition == "order_failed"
    assert refused.awaits_new_input and not refused.executed
    assert history.latest("o/r", 1) is None


@pytest.mark.asyncio
async def test_an_answer_is_dated_when_this_agent_first_observed_it(journal):
    """The record's ``terminal_at`` dates an answer only when it records that
    answer; otherwise the read that observed it does. Never before the send."""
    rows = [
        _row("t-stamped", sent=_ago(NOW, days=2), terminal_state="completed",
             terminal_at=_ago(NOW, days=1)),
        _row("t-unstamped", sent=_ago(NOW, days=2)),
        _row("t-expired", sent=_ago(NOW, days=2), terminal_state="expired",
             terminal_at=_ago(NOW, days=2, minutes=-1)),
        _row("t-skewed", sent=_ago(NOW, days=2), terminal_state="completed",
             terminal_at=_ago(NOW, days=9)),
    ]
    tasks = {
        task_id: _task(
            _order(f"wo-{n}", n), state="completed",
            report=_report(f"wo-{n}", n, "blocked"),
        )
        for n, task_id in enumerate(
            ["t-stamped", "t-unstamped", "t-expired", "t-skewed"], start=1
        )
    }

    ledger = await read_work_orders(
        _agent_with(_Peers(rows, tasks)), "Claw", journal=journal,
        window=timedelta(days=30), now=NOW,
    )
    history = ledger.run_history()

    assert history.latest("o/r", 1).completed_at.isoformat() == _ago(NOW, days=1)
    assert history.latest("o/r", 2).completed_at == NOW
    assert history.latest("o/r", 3).completed_at == NOW
    assert history.latest("o/r", 4).completed_at.isoformat() == _ago(NOW, days=2)


@pytest.mark.asyncio
async def test_an_unreadable_report_withholds_only_its_issue(journal):
    rows = [_row("t-1", sent=_ago(NOW, hours=3), terminal_state="completed",
                 terminal_at=_ago(NOW, hours=2))]
    peers = _Peers(rows, {
        "t-1": _task(_order("wo-5", 5), state="completed", report="Opened PR #9."),
    })

    ledger = await read_work_orders(
        _agent_with(peers), "Claw", journal=journal, window=timedelta(days=30), now=NOW
    )

    [order] = ledger.orders
    assert "not a JSON object" in order.unreadable
    assert order.run is None and not order.outstanding
    assert ledger.lane_idle
    assert ledger.run_history().latest("o/r", 5) is None


@pytest.mark.asyncio
async def test_an_order_the_executor_cannot_be_asked_about_makes_the_ledger_unreadable(
    journal,
):
    rows = [_row("t-1", sent=_ago(NOW, hours=3))]

    with pytest.raises(WorkOrdersUnreadable, match="Could not reach peer 'Claw'"):
        await read_work_orders(
            _agent_with(_Peers(rows, unreachable={"t-1"})), "Claw", journal=journal,
            window=timedelta(days=30), now=NOW,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "agent, problem",
    [
        (SimpleNamespace(), "PeersFeature is not enabled"),
        (_agent_with(None), "PeersFeature is not enabled"),
        (_agent_with(SimpleNamespace(list_outbound_a2a_tasks=AsyncMock())),
         "no get_peer_task_result"),
        (_agent_with(_Peers(list_error="Outbound audit store unavailable")),
         "Outbound audit store unavailable"),
    ],
)
async def test_without_the_outbound_record_the_ledger_is_unreadable(
    agent, problem, journal
):
    with pytest.raises(WorkOrdersUnreadable, match=problem):
        await read_work_orders(
            agent, "Claw", journal=journal, window=timedelta(days=30), now=NOW
        )


@pytest.mark.asyncio
async def test_a_record_too_long_to_read_whole_is_unreadable(journal):
    rows = [
        _row(f"t-{n}", sent=_ago(NOW, minutes=n), verb="message")
        for n in range(work_orders.OUTBOUND_READ_LIMIT)
    ]

    with pytest.raises(WorkOrdersUnreadable, match="more than one read returns"):
        await read_work_orders(
            _agent_with(_Peers(rows)), "Claw", journal=journal,
            window=timedelta(days=30), now=NOW,
        )


@pytest.mark.asyncio
async def test_the_ledger_reads_the_real_peers_record_and_the_executors_task(tmp_path):
    """Through the real PeersFeature: ``send_a2a_task`` with the work order
    records the outbound row, and the executor's task comes back in the
    shape its host serves (``_task_http_payload``)."""
    peers, db = await _make_peers(tmp_path)
    try:
        peers.agent.get_feature = (
            lambda name: peers if name == "PeersFeature" else None
        )
        order = build_work_order({
            "repo": "o/r", "issue_number": 3093, "issue_title": "epic",
            "priority": "high", "context": "Blocker",
        })
        with patch(
            "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
            return_value=_async_client_with(
                post_resp=_mock_post_response(task_id="wo-task")
            ),
        ):
            sent = await peers.send_a2a_task(
                "claw", "Work order", artifacts=[{"name": "work_order", "data": order}]
            )
        task_id = sent.data["task_id"]
        answered = Task(
            id=task_id,
            status=TaskStatus(
                state=TaskState.COMPLETED,
                message=Message(role="agent", parts=[TextPart(text=json.dumps(
                    _report(order["order_id"], 3093, "clarifying")
                ))]),
            ),
            artifacts=[Artifact(name="work_order", parts=[DataPart(data=order)])],
        )
        peers.refresh_local_host_peer_directory(
            host_url="http://multi_agent",
            transport_key="",
            local_get=AsyncMock(return_value=_task_http_payload(answered)),
        )

        with patch(
            "kestrel_sovereign.features.peers.feature.httpx.AsyncClient",
            return_value=_async_client_with(),
        ):
            ledger = await read_work_orders(
                peers.agent, "Claw", journal=WorkOrderJournal(db, DID),
                window=timedelta(days=30),
            )
    finally:
        await db.close()

    [read] = ledger.orders
    assert (read.order_id, read.task_id, read.repo, read.issue_number) == (
        order["order_id"], task_id, "o/r", 3093,
    )
    assert read.run.disposition == "clarifying"
    assert read.run.awaits_new_input


# ---------------------------------------------------------------------------
# signal_dispatch, delegated
# ---------------------------------------------------------------------------


def _sent(task_id="t-new"):
    return ToolResult.ok(
        f"A2A task {task_id} submitted to Claw",
        data={"sent": True, "task_id": task_id, "state": "submitted", "recipient": "Claw"},
    )


def _delegating_agent(peers, db, *, sent=None, raises=None, send=None):
    """An agent whose picks go to Claw, with a Talon registry of its own
    that must never be read: Claw's runs are not in it. Its journal is kept
    in ``db``."""
    execute_named_tool = AsyncMock(
        side_effect=send or raises,
        return_value=sent if sent is not None else _sent(),
    )
    agent = SimpleNamespace(
        did=DID,
        _raw_storage=SimpleNamespace(db=db),
        execute_named_tool=execute_named_tool,
        get_turn_bound_session_id=lambda: "chat-7",
        get_feature=lambda name: peers if name == "PeersFeature" else None,
        operator_registry=SimpleNamespace(
            get_workflow_registration=lambda name: SimpleNamespace(owner="x")
        ),
    )
    return _with_talon(agent, _TalonJobs([_talon_run("completed", job_id="own")]))


def _delegating_feature(agent, *blockers, executor="Claw"):
    return _make_feature(
        {"morning_signal_config": {
            "scan_repos": ["o/r"], "dispatch_executor": executor,
        }},
        agent=agent,
        blockers=list(blockers) or [{"severity": "high", "issue": "o/r#3093", "title": "epic"}],
    )


def _github(monkeypatch, issues, *, comments=None):
    """GitHub serving ``issues``: no PR links any, and the only post-run
    activity is ``comments`` (number -> ISO time of a maintainer comment).

    Every read yields to the loop, as a network read does, so a dispatch
    running alongside another interleaves with it during selection."""
    monkeypatch.setattr(issue_selection, "get_github_token", lambda: "token")
    reads = []
    comments = comments or {}

    async def fake_get(path, token):
        await asyncio.sleep(0)
        reads.append(path)
        for number, issue in issues.items():
            if path == f"/repos/o/r/issues/{number}":
                return issue
        if "/issues?" in path:
            return []
        raise RuntimeError(f"404 {path}")

    async def fake_post(path, token, body):
        await asyncio.sleep(0)
        reads.append(path)
        number = body["variables"]["number"]
        if "timelineItems" in body["query"]:
            nodes = (
                [{"__typename": "IssueComment", "createdAt": comments[number],
                  "authorAssociation": "OWNER", "body": "Go ahead, scope is X."}]
                if number in comments
                else []
            )
            return {"data": {"repository": {"issue": {
                "lastEditedAt": None, "timelineItems": {"nodes": nodes},
            }}}}
        return {"data": {"repository": {"issue": {
            "closedByPullRequestsReferences": {"nodes": []},
        }}}}

    monkeypatch.setattr(issue_selection, "github_api_get", fake_get)
    monkeypatch.setattr(issue_selection, "github_api_post", fake_post)
    return reads


_BOARD = {3093: _agent_ready_issue(3093, "epic"), 2: _agent_ready_issue(2, "next")}


@pytest.mark.asyncio
async def test_execute_sends_a_work_order_instead_of_launching_locally(monkeypatch, db):
    """#3758 acceptance: the pick goes to the executor as a work order over
    the governed A2A task tool, its task id is recorded, and its report is
    armed as this agent's wake. ``workflow_run`` is never asked, and this
    agent's own Talon registry is never read."""
    _github(monkeypatch, _BOARD)
    peers = _Peers()
    agent = _delegating_agent(peers, db)
    feat = _delegating_feature(agent)
    register = AsyncMock()

    with patch(_REGISTER_WATCH, new=register):
        result = await feat.signal_dispatch(mode="execute")

    assert result.status is ToolResultStatus.OK
    assert result.data["dispatched"] is True
    assert result.data["executor"] == "Claw"
    assert result.data["workflow"] is None
    assert result.data["a2a_task_id"] == "t-new"
    assert result.data["wake"] == {"ref": "a2a:t-new", "armed": True}
    [(name, request), kwargs] = agent.execute_named_tool.await_args
    assert name == "send_a2a_task"
    assert kwargs == {"session_id": "chat-7", "source": "strategic_memory.signal_dispatch"}
    assert request["recipient"] == "Claw"
    [artifact] = request["artifacts"]
    assert artifact["name"] == "work_order"
    order = artifact["data"]
    assert order == result.data["work_order"]
    assert (order["schema"], order["repo"], order["issue"], order["action"]) == (
        "kestrel.work_order/v1", "o/r", 3093, "claim",
    )
    assert request["message"].startswith(f"Work order {order['order_id']}: claim o/r#3093")
    # The attempt's outbound record is found by its session id even when the
    # send returns no task id.
    assert request["session_id"] == order["order_id"]
    assert result.data["delivery"] == "sent"
    register.assert_awaited_once_with(agent, "a2a:t-new", origin_session_id="chat-7")
    assert f"Work order {order['order_id']} for o/r#3093 sent to Claw" in result.confirmation
    own_talon = agent.wait_registry.get("talon")
    assert own_talon.reads == 0
    assert own_talon.workspace_reads == []


@pytest.mark.asyncio
async def test_without_an_executor_nothing_is_delegated(monkeypatch, db):
    """The opt-in is explicit and per agent (#3752): unset, the dispatch runs
    locally as before and the A2A record is not read."""
    _github(monkeypatch, _BOARD)
    peers = _Peers()
    agent = _delegating_agent(peers, db, sent=ToolResult.ok("started", data={"run_id": "r-1"}))
    feat = _delegating_feature(agent, executor=None)

    result = await feat.signal_dispatch(mode="execute")

    assert result.data["executor"] is None
    assert result.data["workflow"] == "fleet_coding_pipeline"
    assert agent.execute_named_tool.await_args.args[0] == "workflow_run"
    assert agent.wait_registry.get("talon").reads == 1
    assert peers.fetched == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["execute", "suggest"])
async def test_an_unanswered_order_is_not_ordered_again(mode, monkeypatch, db):
    """#3398 through the executor's answers: Claw's last order on #3093 ended
    clarifying, and nothing on the issue since is newer. It is not ordered
    again, and the output says why."""
    _github(monkeypatch, {3093: _BOARD[3093]})
    rows = [_row("t-1", sent=_ago(hours=5), terminal_state="completed",
                 terminal_at=_ago(hours=4))]
    peers = _Peers(rows, {"t-1": _task(
        _order("wo-3093-1", 3093), state="completed",
        report=_report("wo-3093-1", 3093, "clarifying"),
    )})
    agent = _delegating_agent(peers, db)
    feat = _delegating_feature(agent)

    with patch(_REGISTER_WATCH, new=AsyncMock()) as register:
        result = await feat.signal_dispatch(mode=mode)

    assert result.status is ToolResultStatus.OK
    assert result.data["dispatched"] is False
    assert result.data["issue"] is None
    [skip] = result.data["skipped"]
    assert skip["reason"] == "run_history"
    assert (skip["order_id"], skip["executor"], skip["a2a_task_id"]) == (
        "wo-3093-1", "Claw", "t-1",
    )
    assert "No actionable issue found." in result.confirmation
    assert "skipped o/r#3093 -- work order wo-3093-1 to Claw ended clarifying" in (
        result.confirmation
    )
    assert "nothing since authorizes a retry" in result.confirmation
    agent.execute_named_tool.assert_not_awaited()
    register.assert_not_awaited()
    assert agent.wait_registry.get("talon").reads == 0


@pytest.mark.asyncio
async def test_a_comment_after_the_answer_authorizes_the_next_order(monkeypatch, db):
    _github(monkeypatch, {3093: _BOARD[3093]}, comments={3093: _ago(hours=1)})
    rows = [_row("t-1", sent=_ago(hours=5), terminal_state="completed",
                 terminal_at=_ago(hours=4))]
    peers = _Peers(rows, {"t-1": _task(
        _order("wo-3093-1", 3093), state="completed",
        report=_report("wo-3093-1", 3093, "blocked"),
    )})
    agent = _delegating_agent(peers, db)

    with patch(_REGISTER_WATCH, new=AsyncMock()):
        result = await _delegating_feature(agent).signal_dispatch(mode="execute")

    assert result.data["dispatched"] is True
    assert result.data["issue"]["issue_number"] == 3093


@pytest.mark.asyncio
async def test_an_order_the_executor_refused_waits_for_new_input(monkeypatch, db):
    _github(monkeypatch, {3093: _BOARD[3093]})
    rows = [_row("t-1", sent=_ago(hours=5), terminal_state="failed",
                 terminal_at=_ago(hours=4))]
    peers = _Peers(rows, {"t-1": _task(
        _order("wo-3093-1", 3093), state="failed", report="sender is not Emma",
    )})
    agent = _delegating_agent(peers, db)

    result = await _delegating_feature(agent).signal_dispatch(mode="execute")

    assert result.data["dispatched"] is False
    [skip] = result.data["skipped"]
    assert skip["disposition"] == "order_failed"
    assert "skipped o/r#3093 -- Claw did not run work order wo-3093-1 (its task ended failed" in (
        result.confirmation
    )
    agent.execute_named_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_suggest_skips_an_issue_the_executor_still_holds(monkeypatch, db):
    """An order waiting in the executor's inbox has no Talon label on the
    issue yet; ordering it again would queue the same issue twice."""
    _github(monkeypatch, _BOARD)
    rows = [_row("t-1", sent=_ago(hours=1))]
    peers = _Peers(rows, {"t-1": _task(_order("wo-3093-1", 3093), state="submitted")})
    agent = _delegating_agent(peers, db)
    feat = _delegating_feature(
        agent,
        {"severity": "critical", "issue": "o/r#3093", "title": "epic"},
        {"severity": "high", "issue": "o/r#2", "title": "next"},
    )

    result = await feat.signal_dispatch(mode="suggest")

    assert result.status is ToolResultStatus.OK
    assert result.data["issue"]["issue_number"] == 2
    [skip] = result.data["skipped"]
    assert skip["reason"] == "work_order_outstanding"
    assert "skipped o/r#3093 -- work order wo-3093-1 to Claw is still submitted" in (
        result.confirmation
    )
    assert result.data["executor_lane"]["idle"] is False
    assert "once its outstanding order's report is in" in result.confirmation
    agent.execute_named_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_holds_while_the_executors_lane_is_busy(monkeypatch, db):
    """One order runs at a time: with an order still outstanding, nothing is
    selected (GitHub is not asked) and nothing is sent."""
    reads = _github(monkeypatch, _BOARD)
    rows = [_row("t-1", sent=_ago(hours=1))]
    peers = _Peers(rows, {"t-1": _task(_order("wo-7-1", 7), state="working")})
    agent = _delegating_agent(peers, db)

    result = await _delegating_feature(agent).signal_dispatch(mode="execute")

    assert result.status is ToolResultStatus.OK
    assert result.data["dispatched"] is False
    assert result.data["issue"] is None
    assert result.data["executor_lane"]["outstanding"][0]["order_id"] == "wo-7-1"
    assert "Claw's lane is busy: work order wo-7-1 (o/r#7) is working" in result.confirmation
    assert reads == []
    agent.execute_named_tool.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["execute", "suggest"])
async def test_an_unreadable_ledger_selects_and_sends_nothing(mode, monkeypatch, db):
    reads = _github(monkeypatch, _BOARD)
    rows = [_row("t-1", sent=_ago(hours=1))]
    agent = _delegating_agent(_Peers(rows, unreachable={"t-1"}), db)

    result = await _delegating_feature(agent).signal_dispatch(mode=mode)

    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["reason_code"] == "WORK_ORDERS_UNCONFIRMED"
    assert result.data["dispatched"] is False
    assert "Could not reach peer 'Claw' for task t-1" in result.confirmation
    assert "not the same as having nothing to do" in result.confirmation
    assert reads == []
    agent.execute_named_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_unreadable_report_is_not_reported_as_nothing_to_do(monkeypatch, db):
    _github(monkeypatch, {3093: _BOARD[3093]})
    rows = [_row("t-1", sent=_ago(hours=5), terminal_state="completed",
                 terminal_at=_ago(hours=4))]
    peers = _Peers(rows, {"t-1": _task(
        _order("wo-3093-1", 3093), state="completed", report="All done!",
    )})
    agent = _delegating_agent(peers, db)

    result = await _delegating_feature(agent).signal_dispatch(mode="execute")

    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["reason_code"] == "BLOCKERS_UNCONFIRMED"
    assert "GitHub or Claw" in result.confirmation
    assert "No actionable issue found" not in result.confirmation
    [skip] = result.data["skipped"]
    assert skip["reason"] == "work_report_unreadable"
    assert "Claw answered work order wo-3093-1 with a report that cannot be read" in (
        result.confirmation
    )
    agent.execute_named_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_executor_that_names_no_agent_refuses_before_reading_anything(monkeypatch, db):
    reads = _github(monkeypatch, _BOARD)
    peers = _Peers()
    agent = _delegating_agent(peers, db)

    result = await _delegating_feature(agent, executor=["Claw"]).signal_dispatch()

    assert result.status is ToolResultStatus.ERROR
    assert result.data["reason_code"] == "INVALID_DISPATCH_EXECUTOR"
    assert reads == [] and peers.fetched == []
    assert agent.wait_registry.get("talon").reads == 0
    agent.execute_named_tool.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sent, raises, code",
    [
        (ToolResult.failed("Peer is not available in the automatic directory"),
         None, "WORK_ORDER_REJECTED"),
        # A governance hook's denial: the tool never ran.
        ({"success": False, "error": "Permission denied: ASK timed out"},
         None, "WORK_ORDER_REJECTED"),
        (None, ToolNotRegisteredError("send_a2a_task"), "WORK_ORDER_TRANSPORT_UNAVAILABLE"),
        (None, RuntimeError("boom"), "WORK_ORDER_SEND_FAILED"),
    ],
)
async def test_an_order_that_did_not_reach_the_executor_arms_no_wake(
    sent, raises, code, monkeypatch, db
):
    _github(monkeypatch, _BOARD)
    agent = _delegating_agent(_Peers(), db, sent=sent, raises=raises)

    with patch(_REGISTER_WATCH, new=AsyncMock()) as register:
        result = await _delegating_feature(agent).signal_dispatch(mode="execute")

    assert result.status is ToolResultStatus.ERROR
    assert result.data["reason_code"] == code
    assert result.data["dispatched"] is False
    assert result.data["delivery"] == "not_sent"
    assert result.data["order_id"] == result.data["work_order"]["order_id"]
    register.assert_not_awaited()
    # Nothing reached Claw, so nothing holds its lane or the issue.
    [entry] = await WorkOrderJournal(db, DID).entries("Claw")
    assert entry.state == "not_sent"
    ledger = await read_work_orders(
        agent, "Claw", journal=WorkOrderJournal(db, DID), window=timedelta(days=30)
    )
    assert ledger.lane_idle and ledger.orders == ()


@pytest.mark.asyncio
async def test_a_sent_order_whose_wake_cannot_be_armed_says_so(monkeypatch, db):
    _github(monkeypatch, _BOARD)
    agent = _delegating_agent(_Peers(), db)

    with patch(
        _REGISTER_WATCH,
        new=AsyncMock(side_effect=ValueError("no wait provider registered for 'a2a'")),
    ):
        result = await _delegating_feature(agent).signal_dispatch(mode="execute")

    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["reason_code"] == "WORK_ORDER_WAKE_UNARMED"
    assert result.data["dispatched"] is True
    assert result.data["a2a_task_id"] == "t-new"
    assert result.data["wake"] == {"ref": "a2a:t-new", "armed": False}
    assert "Its report will not wake this agent" in result.confirmation


# ---------------------------------------------------------------------------
# The journal: the lane reservation, uncertain delivery, retention
# ---------------------------------------------------------------------------


def _recording_send(peers, *, outcome="sent", error=None, raises=None):
    """``execute_named_tool`` over a ``_Peers`` record: each send leaves the
    outbound row PeersFeature would leave for it, found by its session id."""
    sends = []

    async def send(name, args, *, session_id, source):
        assert name == "send_a2a_task"
        await asyncio.sleep(0)
        task_id = f"t-{len(sends) + 1}"
        sends.append(args)
        order = args["artifacts"][0]["data"]
        if outcome == "sent":
            peers.rows.insert(0, _row(task_id, sent=_ago(), session_id=args["session_id"]))
            peers.tasks[task_id] = _task(order, state="submitted")
            if raises is not None:
                raise raises
            return _sent(task_id)
        if outcome == "failed":
            peers.rows.insert(0, _row(
                task_id, sent=_ago(), session_id=args["session_id"],
                route_state="reserved", terminal_state="dispatch_failed",
                terminal_at=_ago(), error=error,
            ))
        if raises is not None:
            raise raises
        return ToolResult.failed(
            "A2A dispatch failed",
            data={"sent": outcome == "delivered-unrecorded", "task_id": task_id},
        )

    return send, sends


async def _journal_order(journal, *, order_id, issue, state, at, task_id=None,
                         disposition=None, a2a_state="completed"):
    """Put one ``signal_dispatch`` order in the journal as of ``at``."""
    reservation = await journal.reserve("Claw", now=at)
    if state == "reserved":
        return reservation
    await journal.begin_send(
        reservation.order_key, order_id=order_id, repo="o/r", issue_number=issue,
        now=at,
    )
    if state == "sending":
        return reservation
    if state == "unconfirmed":
        await journal.record_unconfirmed(
            reservation.order_key, detail="the request timed out", task_id=task_id
        )
        return reservation
    await journal.record_sent(reservation.order_key, task_id=task_id, sent_at=at)
    if state == "settled":
        await journal.settle(
            reservation.order_key, a2a_state=a2a_state,
            answered_at=at + timedelta(hours=1), disposition=disposition,
            job_id=f"job-{order_id}", detail=None,
        )
    return reservation


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_the_lane_admits_one_open_dispatch_order_per_executor(shared_journal):
    journal = shared_journal
    first = await journal.reserve("Claw", now=NOW)
    assert first is not None
    assert await journal.reserve("claw", now=NOW) is None
    # Another executor's lane is its own.
    assert await journal.reserve("Meridian", now=NOW) is not None
    # A reservation that never became an order frees the lane.
    assert await journal.withdraw(first.order_key)
    assert await journal.reserve("Claw", now=NOW) is not None


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_two_writers_cannot_both_reserve_the_lane(db_backend, shared_journal):
    """The reservation is a database compare-and-set, so it holds between
    processes sharing the database, not only between tasks in one."""
    second_db, second_backend = await _second_connection(db_backend)
    try:
        other = WorkOrderJournal(second_db, shared_journal.agent_id)
        results = await asyncio.gather(
            shared_journal.reserve("Claw", now=NOW), other.reserve("Claw", now=NOW)
        )
    finally:
        await second_backend.close()

    assert sorted(result is None for result in results) == [False, True]
    [entry] = await shared_journal.entries("Claw")
    assert entry.state == "reserved"


@pytest.mark.asyncio
async def test_concurrent_dispatches_send_one_order(monkeypatch, db):
    """Finding P2 of the #3758 review: two dispatches running at once both
    read an idle lane, then both selected and sent. The lane is reserved
    before selection, so the second finds it held and sends nothing."""
    _github(monkeypatch, _BOARD)
    peers = _Peers()
    send, sends = _recording_send(peers)
    agent = _delegating_agent(peers, db, send=send)
    feat = _delegating_feature(
        agent,
        {"severity": "critical", "issue": "o/r#3093", "title": "epic"},
        {"severity": "high", "issue": "o/r#2", "title": "next"},
    )

    with patch(_REGISTER_WATCH, new=AsyncMock()):
        results = await asyncio.gather(
            feat.signal_dispatch(mode="execute"), feat.signal_dispatch(mode="execute")
        )

    assert len(sends) == 1
    [dispatched] = [result for result in results if result.data["dispatched"]]
    [held] = [result for result in results if not result.data["dispatched"]]
    assert dispatched.data["issue"]["issue_number"] == 3093
    assert held.status is ToolResultStatus.OK
    assert "Claw's lane is busy" in held.confirmation
    assert held.data["executor_lane"]["idle"] is False
    entries = await WorkOrderJournal(db, DID).entries("Claw")
    assert [entry.state for entry in entries] == ["sent"]


@pytest.mark.asyncio
async def test_a_reservation_is_withdrawn_when_nothing_is_sent(monkeypatch, db):
    """Every way out of a dispatch that sent nothing frees the lane."""
    _github(monkeypatch, {})  # no issue on the board is open
    agent = _delegating_agent(_Peers(), db)

    result = await _delegating_feature(agent).signal_dispatch(mode="execute")

    assert result.data["dispatched"] is False
    assert await WorkOrderJournal(db, DID).entries("Claw") == []
    agent.execute_named_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_reservation_whose_lease_lapsed_cannot_send(monkeypatch, db):
    """A dispatch that outlives its lease loses the lane to the reader that
    withdrew it, rather than sending under a reservation nobody holds."""
    _github(monkeypatch, _BOARD)
    agent = _delegating_agent(_Peers(), db)
    feat = _delegating_feature(agent)
    original = WorkOrderJournal.begin_send

    async def withdrawn_first(self, order_key, **kwargs):
        await self.withdraw(order_key)
        return await original(self, order_key, **kwargs)

    with patch.object(WorkOrderJournal, "begin_send", withdrawn_first):
        result = await feat.signal_dispatch(mode="execute")

    assert result.status is ToolResultStatus.ERROR
    assert result.data["reason_code"] == "WORK_ORDER_LANE_LOST"
    agent.execute_named_tool.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome, error, raises, code, delivery",
    [
        pytest.param(
            "failed", "peer_identity_rekey_failed", None,
            "WORK_ORDER_DELIVERY_UNCONFIRMED", "unconfirmed",
            id="delivered-route-not-kept",
        ),
        pytest.param(
            "failed", "connect_error:Claw", None,
            "WORK_ORDER_DELIVERY_UNCONFIRMED", "unconfirmed", id="timed-out",
        ),
        pytest.param(
            "delivered-unrecorded", None, None,
            "WORK_ORDER_DELIVERY_UNCONFIRMED", "unconfirmed",
            id="sent-but-no-record",
        ),
        pytest.param(
            "failed", "peer_unavailable:Claw", None,
            "WORK_ORDER_REJECTED", "not_sent", id="host-said-unavailable",
        ),
        pytest.param(
            "failed", "signing_failed:no_key", None,
            "WORK_ORDER_REJECTED", "not_sent", id="never-left",
        ),
        pytest.param(
            "refused", None, None, "WORK_ORDER_REJECTED", "not_sent",
            id="refused-before-reserving",
        ),
        pytest.param(
            "sent", None, RuntimeError("post-tool hook failed"),
            None, "sent", id="accepted-then-raised",
        ),
    ],
)
async def test_a_failed_send_is_classified_from_the_outbound_record(
    outcome, error, raises, code, delivery, monkeypatch, db
):
    """Finding P1 of the #3758 review: a send that reported failure can still
    have reached the executor. Only a record that proves non-delivery frees
    the lane; one that does not keeps the order open."""
    _github(monkeypatch, _BOARD)
    peers = _Peers()
    send, sends = _recording_send(peers, outcome=outcome, error=error, raises=raises)
    agent = _delegating_agent(peers, db, send=send)

    with patch(_REGISTER_WATCH, new=AsyncMock()) as register:
        result = await _delegating_feature(agent).signal_dispatch(mode="execute")

    assert result.data["delivery"] == delivery
    assert result.data.get("reason_code") == code
    assert result.data["dispatched"] is (delivery == "sent")
    [entry] = await WorkOrderJournal(db, DID).entries("Claw")
    assert entry.state == delivery
    assert entry.order_id == sends[0]["session_id"]
    if delivery == "sent":
        assert entry.task_id == "t-1"
        register.assert_awaited_once()
    else:
        register.assert_not_awaited()
    if delivery == "unconfirmed":
        assert result.status is ToolResultStatus.PARTIAL
        assert "may have reached Claw" in result.confirmation
        assert "strategy_release_work_order" in result.confirmation


@pytest.mark.asyncio
async def test_an_unconfirmed_order_holds_the_lane_and_its_issue(monkeypatch, db):
    _github(monkeypatch, _BOARD)
    peers = _Peers()
    send, _ = _recording_send(peers, outcome="failed", error="connect_error:Claw")
    agent = _delegating_agent(peers, db, send=send)
    feat = _delegating_feature(
        agent,
        {"severity": "critical", "issue": "o/r#3093", "title": "epic"},
        {"severity": "high", "issue": "o/r#2", "title": "next"},
    )
    with patch(_REGISTER_WATCH, new=AsyncMock()):
        first = await feat.signal_dispatch(mode="execute")
    order_id = first.data["order_id"]
    agent.execute_named_tool.reset_mock()

    again = await feat.signal_dispatch(mode="execute")
    preview = await feat.signal_dispatch(mode="suggest")

    assert again.data["dispatched"] is False
    assert f"work order {order_id} (o/r#3093) may have reached Claw" in again.confirmation
    assert f"strategy_release_work_order('{order_id}', reason)" in again.confirmation
    agent.execute_named_tool.assert_not_awaited()
    assert preview.data["issue"]["issue_number"] == 2
    [skip] = preview.data["skipped"]
    assert skip["reason"] == "work_order_unconfirmed"
    assert "its delivery could not be confirmed" in preview.confirmation


@pytest.mark.asyncio
async def test_a_delivered_order_whose_route_was_not_kept_is_not_ordered_again(
    tmp_path, monkeypatch,
):
    """The review's reproduction, through the real PeersFeature over SQLite:
    the executor accepted the task, then retaining its route failed, so the
    send returned an error with ``sent: true``. The order is kept open; the
    next dispatch sends nothing until it is released."""
    _github(monkeypatch, {3093: _BOARD[3093]})
    peers, db = await _make_peers(tmp_path)
    try:
        agent = peers.agent
        agent.did = DID
        agent.get_feature = lambda name: peers if name == "PeersFeature" else None
        agent.get_turn_bound_session_id = lambda: "chat-7"
        agent.operator_registry = SimpleNamespace(
            get_workflow_registration=lambda name: SimpleNamespace(owner="x")
        )

        async def send(name, args, *, session_id, source):
            return await peers.send_a2a_task(**args)

        agent.execute_named_tool = AsyncMock(side_effect=send)
        _with_talon(agent, _TalonJobs([]))
        feat = _delegating_feature(agent)

        with patch(
            _PEERS_HTTP,
            return_value=_async_client_with(
                post_resp=_mock_post_response(task_id="t-accepted")
            ),
        ), patch(
            "kestrel_sovereign.a2a.outbound_store.rekey_outbound_task",
            AsyncMock(return_value=0),
        ), patch(_REGISTER_WATCH, new=AsyncMock()) as register:
            first = await feat.signal_dispatch(mode="execute")

        assert first.status is ToolResultStatus.PARTIAL
        assert first.data["reason_code"] == "WORK_ORDER_DELIVERY_UNCONFIRMED"
        assert first.data["sender_result"]["data"]["sent"] is True
        register.assert_not_awaited()
        order_id = first.data["order_id"]

        agent.execute_named_tool.reset_mock()
        again = await feat.signal_dispatch(mode="execute")
        assert again.data["dispatched"] is False
        assert "may have reached Claw" in again.confirmation
        agent.execute_named_tool.assert_not_awaited()

        released = await feat.strategy_release_work_order(
            order_id, "Claw confirmed it holds no such task"
        )
        assert released.status is ToolResultStatus.OK
        assert released.data["previous_state"] == "unconfirmed"

        with patch(
            _PEERS_HTTP,
            return_value=_async_client_with(
                post_resp=_mock_post_response(task_id="t-second")
            ),
        ), patch(_REGISTER_WATCH, new=AsyncMock()):
            after = await feat.signal_dispatch(mode="execute")
        assert after.data["dispatched"] is True
        assert after.data["a2a_task_id"] == "t-second"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_a_blocking_report_does_not_expire(monkeypatch, db):
    """Finding P2 of the #3758 review: an order answered ``blocked`` 60 days
    ago, its outbound row long past the window, still withholds the issue.
    Elapsed time is not new input (#3398)."""
    _github(monkeypatch, {3093: _BOARD[3093]})
    long_ago = datetime.now(timezone.utc) - timedelta(days=60)
    await _journal_order(
        WorkOrderJournal(db, DID), order_id="wo-3093-old", issue=3093,
        state="settled", at=long_ago, task_id="t-old", disposition="blocked",
    )
    peers = _Peers([_row(
        "t-old", sent=long_ago.isoformat(), session_id="wo-3093-old",
        terminal_state="completed", terminal_at=(long_ago + timedelta(hours=1)).isoformat(),
    )])
    agent = _delegating_agent(peers, db)

    result = await _delegating_feature(agent).signal_dispatch(mode="execute")

    assert result.data["dispatched"] is False
    [skip] = result.data["skipped"]
    assert (skip["reason"], skip["order_id"], skip["disposition"]) == (
        "run_history", "wo-3093-old", "blocked",
    )
    assert peers.fetched == []  # a journaled answer is not asked for again
    agent.execute_named_tool.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_an_adopted_order_outlives_the_window(shared_journal):
    """An order sent by hand is journaled the first time the scan finds it,
    and kept after its row leaves the window."""
    rows = [_row("t-1", sent=_ago(NOW, days=5), terminal_state="completed",
                 terminal_at=_ago(NOW, days=4))]
    peers = _Peers(rows, {"t-1": _task(
        _order("wo-9", 9), state="completed", report=_report("wo-9", 9, "clarifying"),
    )})
    window = timedelta(days=30)

    first = await read_work_orders(
        _agent_with(peers), "Claw", journal=shared_journal, window=window, now=NOW
    )
    later = await read_work_orders(
        _agent_with(peers), "Claw", journal=shared_journal, window=window,
        now=NOW + timedelta(days=60),
    )

    asked = first.run_history().latest("o/r", 9)
    kept = later.run_history().latest("o/r", 9)
    assert (kept.disposition, kept.order_id) == ("clarifying", "wo-9")
    assert kept.completed_at == asked.completed_at
    assert peers.fetched == [("Claw", "t-1")]


@pytest.mark.asyncio
async def test_an_outstanding_order_past_the_window_keeps_the_lane_busy(journal):
    long_ago = NOW - timedelta(days=60)
    await _journal_order(
        journal, order_id="wo-7-old", issue=7, state="sent", at=long_ago,
        task_id="t-old",
    )
    peers = _Peers([], {"t-old": _task(_order("wo-7-old", 7), state="working")})

    ledger = await read_work_orders(
        _agent_with(peers), "Claw", journal=journal, window=timedelta(days=30), now=NOW
    )

    assert not ledger.lane_idle
    [order] = ledger.outstanding()
    assert (order.order_id, order.state) == ("wo-7-old", "working")
    assert peers.fetched == [("Claw", "t-old")]


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_an_accepted_order_is_settled_once_answered(shared_journal):
    journal = shared_journal
    await _journal_order(
        journal, order_id="wo-5-a", issue=5, state="sent", at=NOW - timedelta(hours=3),
        task_id="t-5",
    )
    peers = _Peers(
        [_row("t-5", sent=_ago(NOW, hours=3), session_id="wo-5-a")],
        {"t-5": _task(
            _order("wo-5-a", 5), state="completed",
            report=_report("wo-5-a", 5, "blocked"),
        )},
    )

    ledger = await read_work_orders(
        _agent_with(peers), "Claw", journal=journal, window=timedelta(days=30), now=NOW
    )
    again = await read_work_orders(
        _agent_with(peers), "Claw", journal=journal, window=timedelta(days=30),
        now=NOW + timedelta(days=1),
    )

    assert ledger.lane_idle
    run = again.run_history().latest("o/r", 5)
    assert (run.disposition, run.job_id, run.completed_at) == ("blocked", "job-wo-5-a", NOW)
    assert peers.fetched == [("Claw", "t-5")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state, at, rows, expected",
    [
        pytest.param("reserved", NOW, [], ["reserved"], id="live-reservation-holds"),
        pytest.param(
            "reserved", NOW - RESERVATION_LEASE - timedelta(seconds=1), [], [],
            id="lapsed-reservation-withdrawn",
        ),
        pytest.param("sending", NOW, [], ["sending"], id="send-in-progress"),
        pytest.param(
            "sending", NOW - RESERVATION_LEASE - timedelta(seconds=1), [],
            ["unconfirmed"], id="sender-stopped-mid-send",
        ),
        pytest.param(
            "sending", NOW,
            [_row("t-9", sent=_ago(NOW, minutes=1), session_id="wo-9-a",
                  route_state="reserved", terminal_state="dispatch_failed",
                  error="peer_unavailable:Claw")],
            ["not_sent"], id="refused-before-delivery",
        ),
        pytest.param(
            "unconfirmed", NOW,
            [_row("t-9", sent=_ago(NOW, minutes=1), session_id="wo-9-a")],
            ["sent"], id="route-confirmed-after-all",
        ),
        pytest.param(
            "unconfirmed", NOW - timedelta(days=90),
            [_row("t-9", sent=_ago(NOW, days=90), session_id="wo-9-a",
                  route_state="reserved", terminal_state="dispatch_failed",
                  error="peer_identity_rekey_failed")],
            ["unconfirmed"], id="unconfirmed-does-not-expire",
        ),
    ],
)
@pytest.mark.dual_backend
async def test_an_attempt_is_reconciled_from_the_outbound_record(
    state, at, rows, expected, shared_journal
):
    journal = shared_journal
    await _journal_order(journal, order_id="wo-9-a", issue=9, state=state, at=at)
    peers = _Peers(rows, {"t-9": _task(_order("wo-9-a", 9), state="working")})

    ledger = await read_work_orders(
        _agent_with(peers), "Claw", journal=journal, window=timedelta(days=30), now=NOW
    )

    assert [entry.state for entry in await journal.entries("Claw")] == expected
    open_states = {"reserved", "sending", "sent", "unconfirmed"}
    assert ledger.lane_idle == (not set(expected) & open_states)


def test_only_a_record_that_proves_non_delivery_frees_an_attempt():
    def failed(error):
        return _row("t", sent=_ago(NOW), route_state="reserved",
                    terminal_state="dispatch_failed", error=error)

    assert attempt_outcome([]) == (None, None)
    assert attempt_outcome([_row("t", sent=_ago(NOW))])[0] == "sent"
    for error in ("peer_not_in_automatic_directory", "peer_unavailable:Claw",
                  "signing_failed:x"):
        assert attempt_outcome([failed(error)])[0] == "not_sent"
    for error in ("connect_error:Claw", "peer_router_error:PeerProtocolError",
                  "peer_identity_rekey_failed", "peer_identity_reservation_failed",
                  None):
        assert attempt_outcome([failed(error)])[0] == "unconfirmed"
    # A send still in flight (reserved, no outcome yet) is not a refusal.
    assert attempt_outcome(
        [_row("t", sent=_ago(NOW), route_state="reserved")]
    )[0] == "unconfirmed"


@pytest.mark.asyncio
async def test_releasing_an_order(monkeypatch, db, journal):
    feat = _delegating_feature(_delegating_agent(_Peers(), db))
    await _journal_order(journal, order_id="wo-6-a", issue=6, state="settled",
                         at=NOW - timedelta(days=2), task_id="t-6",
                         disposition="completed")
    await _journal_order(
        journal, order_id="wo-4-a", issue=4, state="unconfirmed", at=NOW, task_id="t-4",
    )

    no_reason = await feat.strategy_release_work_order("wo-4-a", "  ")
    unknown = await feat.strategy_release_work_order("wo-nope", "gone")
    closed = await feat.strategy_release_work_order("wo-6-a", "gone")
    released = await feat.strategy_release_work_order("t-4", "Claw has no such task")
    twice = await feat.strategy_release_work_order("wo-4-a", "again")

    assert no_reason.status is ToolResultStatus.ERROR
    assert "no work order wo-nope" in unknown.error
    assert "is settled, not open" in closed.error
    assert released.status is ToolResultStatus.OK
    assert (released.data["order_id"], released.data["issue_number"]) == ("wo-4-a", 4)
    assert twice.status is ToolResultStatus.ERROR
    [entry] = [e for e in await journal.entries("Claw") if e.order_id == "wo-4-a"]
    assert (entry.state, entry.detail) == ("released", "released: Claw has no such task")


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_releasing_through_the_journal(shared_journal):
    journal = shared_journal
    await _journal_order(
        journal, order_id="wo-4-a", issue=4, state="unconfirmed", at=NOW, task_id="t-4",
    )

    released = await journal.release("wo-4-a", reason="Claw has no such task")

    assert released.state == "unconfirmed"
    [entry] = await journal.entries("Claw")
    assert (entry.state, entry.detail) == ("released", "released: Claw has no such task")
    with pytest.raises(WorkOrderNotReleasable, match="is released, not open"):
        await journal.release("t-4", reason="again")
    # The lane is free again.
    assert await journal.reserve("Claw", now=NOW) is not None


@pytest.mark.asyncio
async def test_a_running_dispatchs_reservation_cannot_be_released(journal):
    await _journal_order(journal, order_id=None, issue=None, state="reserved", at=NOW)
    [entry] = await journal.entries("Claw")
    with pytest.raises(WorkOrderNotReleasable):
        await journal.release(entry.order_key, reason="x")


@pytest.mark.asyncio
async def test_the_journal_needs_a_database_and_a_did(db):
    with pytest.raises(WorkOrdersUnreadable, match="no database"):
        await open_work_order_journal(SimpleNamespace(did=DID))
    with pytest.raises(WorkOrdersUnreadable, match="cannot be scoped"):
        await open_work_order_journal(
            SimpleNamespace(did="", _raw_storage=SimpleNamespace(db=db))
        )


@pytest.mark.asyncio
@pytest.mark.dual_backend
async def test_the_journal_is_scoped_to_its_agent(db_backend, shared_journal):
    mine = shared_journal
    database = AsyncDatabase(db_backend)
    theirs = WorkOrderJournal(database, f"did:test:{uuid.uuid4().hex}")
    try:
        await _journal_order(mine, order_id="wo-1-a", issue=1, state="sent", at=NOW,
                             task_id="t-1")

        assert await theirs.entries("Claw") == []
        # Their lane is theirs: my open order does not hold it.
        assert await theirs.reserve("Claw", now=NOW) is not None
    finally:
        await database.execute(
            f"DELETE FROM {work_orders.JOURNAL_TABLE} WHERE agent_id = ?",
            (theirs.agent_id,),
        )
