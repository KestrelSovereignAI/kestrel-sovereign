"""signal_dispatch delegates execution to the host's Talon dispatcher (#3758).

The fleet host names one agent its only Talon dispatcher (``[talon.policy]
dispatcher = "Claw"``, kestrel-feature-talon#57). Emma owns the queue and
selects; Claw runs the work. These tests pin both halves of the delegation:

* which agent executes is what Emma's own Talon reports through
  ``talon_get_config``, the policy its launch gate loads, not a TOML table
  core finds on its own;
* execute sends Claw a ``kestrel.work_order/v1`` A2A task, records the
  dispatch and arms its ``a2a:<dispatch_id>`` wake, instead of launching
  anything on Emma, and sends nothing while an order is unanswered;
* the #3398 guard and the workspace gate read Claw's ``kestrel.work_report/v1``
  answers from the whole outbound audit, never Emma's own (empty) Talon
  registry, and never only the newest page of it;
* an order whose wake was never armed, because the dispatch that sent it
  stopped first, has it armed by the next dispatch, and still wakes Emma.

Claw is reached only through the tools Emma calls by name. Its tasks are real
native ``Task`` values, turned into ``get_peer_task_result``'s answer by
Peers' own ``_native_peer_result``, and the order Emma sends is read back
through the same protobuf round trip a real peer read takes. Wakes go
through the real wait lifecycle: ``register_wait_watch`` and the real
``WaitSignalStore`` over SQLite, the real ``A2AWaitable`` over Emma's audit,
and the real reconciler.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from google.protobuf.json_format import ParseDict
from kestrel_sdk.native_a2a.types import Message, Part, Role, Task, TaskState, TaskStatus
from kestrel_sdk.tools.result import ToolResult, ToolResultStatus

from kestrel_sovereign import paths
from kestrel_sovereign.a2a.lifecycle import new_task_status, task_state_label
from kestrel_sovereign.a2a.stores.unified.outbound_store import (
    DispatchState,
    OutboundDispatch,
)
from kestrel_sovereign.agent.orchestrator_engine import ToolNotRegisteredError
from kestrel_sovereign.features.peers.feature import PeersFeature
from kestrel_sovereign.features.peers.wait_provider import A2AWaitable
from kestrel_sovereign.features.strategic_memory import StrategicMemoryFeature, issue_selection
from kestrel_sovereign.features.strategic_memory import feature as strategic_feature
from kestrel_sovereign.features.strategic_memory.feature import _SaveOutcome
from kestrel_sovereign.features.strategic_memory.ledger import (
    BLOCKERS_KEY,
    PATTERNS_KEY,
    StrategyLedger,
)
from kestrel_sovereign.features.strategic_memory.run_history import (
    DELIVERY_UNCERTAIN,
    LIST_OUTBOUND_TOOL,
    ORDER_CANCELLED,
    ORDER_REFUSED,
    READ_PEER_TASK_TOOL,
    WORK_ORDER_PAGE_SIZE,
    WORK_ORDER_SCHEMA,
    WORK_REPORT_SCHEMA,
    RunHistoryUnreadable,
    read_executor_run_history,
)
from kestrel_sovereign.paths import AGENT_DB_PATH_ENV
from kestrel_sovereign.storage.async_wait_signal_store import WaitSignalStore
from kestrel_sovereign.waits.engine import WaitRegistry
from kestrel_sovereign.waits.reconciler import WaitReconciler
from tests.shared.environment_guard import unset_environment
from tests.unit.test_wait_reconciler import _CapturingDispatcher

#: When Claw answered the orders below.
ANSWERED = datetime(2026, 10, 24, 9, 56, 4, tzinfo=timezone.utc)
TALON_JOB = "802b93fef2824f53944bdac49167a613"
EMMA_DID = "did:pkh:emma"
TALON_CONFIG_TOOL = "talon_get_config"


# ---------------------------------------------------------------------------
# Claw, as Emma reaches it through the tools she calls by name
# ---------------------------------------------------------------------------


def _order(issue=3093, repo="o/r", order_id="wo-3093-1"):
    return {
        "schema": WORK_ORDER_SCHEMA,
        "order_id": order_id,
        "repo": repo,
        "issue": issue,
        "action": "claim",
        "talon": {},
        "verify": [],
        "context": "Fresh claim.",
    }


def _report(disposition, issue=3093, job=TALON_JOB):
    return json.dumps({
        "schema": WORK_REPORT_SCHEMA,
        "order_id": f"wo-{issue}-1",
        "repo": "o/r",
        "issue": issue,
        "talon_job_id": job,
        "disposition": disposition,
        "pr": None,
    })


def _talon_policy(dispatcher=None):
    """``talon_get_config``'s ``policy``: ``asdict(TalonPolicy)`` in Talon >=0.2.14."""
    return {
        "allowed_backends": ["claude", "codex", "opencode"],
        "allow_api_billing": False,
        "require_worktree": True,
        "require_sandboxed_workspace": True,
        "allow_background_jobs": True,
        "dispatcher": dispatcher,
    }


def _row(dispatch_id, state=DispatchState.ACKNOWLEDGED, verb="task"):
    """One row of Emma's outbound audit, as ``list_outbound_a2a_tasks`` lists it."""
    return OutboundDispatch(
        dispatch_id=dispatch_id,
        sender_id=EMMA_DID,
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
        dispatch_tool="send_a2a_task",
        verb=verb,
        created_at="2026-10-24T07:56:10+00:00",
        state=state,
        revision=1,
        remote_task_id=f"task-{dispatch_id}",
        remote_context_id="ctx",
        remote_task_state=None,
        response_digest=None,
        response_kind="task",
        updated_at="2026-10-24T07:56:10+00:00",
        operation_id="op",
        cancel_operation_id=None,
        cancel_request_digest=None,
        cancel_state=None,
        root_dispatch_id=dispatch_id,
        parent_dispatch_id=None,
    )


def _task(dispatch_id, parts, state, *, answer=None, at=ANSWERED, artifacts=()):
    """Claw's task for one dispatch: the message Emma sent, and its answer."""
    sent = Message(
        message_id=f"u-{dispatch_id}",
        role=Role.ROLE_USER,
        parts=[ParseDict(part, Part()) for part in parts],
    )
    answer_message = (
        None
        if answer is None
        else Message(
            message_id=f"a-{dispatch_id}", role=Role.ROLE_AGENT, parts=[Part(text=answer)]
        )
    )
    status = (
        new_task_status(state, at=at, message=answer_message)
        if at is not None
        else TaskStatus(state=state, message=answer_message)
    )
    task = Task(id=f"task-{dispatch_id}", context_id="ctx", status=status, history=[sent])
    for artifact in artifacts:
        task.artifacts.add().CopyFrom(artifact)
    return task


def _read(dispatch_id, task):
    """``get_peer_task_result``'s answer for ``task``, built by Peers itself."""
    return PeersFeature._native_peer_result(
        SimpleNamespace(
            dispatch_id=dispatch_id,
            remote_task_id=task.id,
            remote_task_state=task_state_label(task.status.state),
            response_kind="task",
        ),
        task,
    )


class _Claw:
    """The tools Emma calls by name: her own Talon's ``talon_get_config``, and
    the Peers tools over Claw's tasks and her outbound audit.

    ``rows`` is newest first, as the audit lists it, and the listing pages
    through it by ``before`` cursor as the real store does. ``send_a2a_task``
    adds a submitted task carrying exactly the parts it was given, so a later
    read sees the order Emma sent. Every call yields first, as a real tool
    call does, so an unserialized read-then-send would interleave. Each call
    is recorded in ``calls``.
    """

    def __init__(self):
        self.rows = []
        self.reads = {}
        self.calls = []
        self.policy = _talon_policy()
        self.policy_result = None
        self.send_result = None
        self.send_row_state = DispatchState.UNKNOWN
        self.unregistered = set()
        self.raises = {}

    def answered(self, dispatch_id, state, answer=None, *, order=None, at=ANSWERED):
        self._add(dispatch_id, [{"data": order or _order()}], state, answer=answer, at=at)

    def pending(self, dispatch_id, state=TaskState.TASK_STATE_WORKING, *, order=None):
        self._add(dispatch_id, [{"data": order or _order()}], state)

    def uncertain(self, dispatch_id):
        self.rows.insert(0, _row(dispatch_id, DispatchState.UNKNOWN))
        self.reads[dispatch_id] = ToolResult.failed(
            "Delivery is still uncertain; no matching authorized peer Task was found",
            data={"dispatch_id": dispatch_id, "delivery_state": "unknown"},
        )

    def unrelated(self, count, verb="message"):
        """``count`` newer dispatches to Claw that are not work orders."""
        for n in range(count):
            self.rows.insert(0, _row(f"other-{verb}-{n}", verb=verb))

    def _add(self, dispatch_id, parts, state, *, answer=None, at=ANSWERED, row_state=DispatchState.ACKNOWLEDGED):
        self.rows.insert(0, _row(dispatch_id, row_state))
        self.reads[dispatch_id] = _read(
            dispatch_id, _task(dispatch_id, parts, state, answer=answer, at=at)
        )

    def names(self):
        return [name for name, _ in self.calls]

    def args(self, name):
        [args] = [args for called, args in self.calls if called == name]
        return args

    def peer_reads(self):
        return [args["dispatch_id"] for name, args in self.calls if name == READ_PEER_TASK_TOOL]

    async def execute_named_tool(self, name, args, *, session_id, source):
        await asyncio.sleep(0)
        assert source == "strategic_memory.signal_dispatch"
        self.calls.append((name, dict(args)))
        if name in self.unregistered:
            raise ToolNotRegisteredError(f"Tool '{name}' not registered")
        if name in self.raises:
            raise self.raises[name]
        if name == TALON_CONFIG_TOOL:
            if self.policy_result is not None:
                return self.policy_result
            return ToolResult.ok(
                "Talon config loaded (default=claude/opus)",
                data={"success": True, "policy": dict(self.policy), "preference": {}},
            )
        if name == LIST_OUTBOUND_TOOL:
            start = 0
            if args.get("before"):
                [start] = [
                    n + 1 for n, row in enumerate(self.rows)
                    if row.dispatch_id == args["before"]
                ]
            rows = [asdict(row) for row in self.rows[start: start + args["limit"]]]
            return ToolResult.ok(
                f"Found {len(rows)} outbound A2A dispatches",
                data={"rows": rows, "count": len(rows)},
            )
        if name == READ_PEER_TASK_TOOL:
            return self.reads[args["dispatch_id"]]
        if name == "send_a2a_task":
            if self.send_result is not None:
                # Peers reserves the dispatch before it sends anything.
                sent = (
                    self.send_result.data
                    if isinstance(self.send_result, ToolResult)
                    else self.send_result.get("data")
                )
                dispatch_id = (sent or {}).get("dispatch_id")
                if dispatch_id:
                    self.rows.insert(0, _row(dispatch_id, self.send_row_state))
                    self.reads[dispatch_id] = ToolResult.failed(
                        "Delivery is still uncertain",
                        data={"dispatch_id": dispatch_id, "delivery_state": "unknown"},
                    )
                return self.send_result
            dispatch_id = f"{len(self.rows) + 1:032x}"
            self._add(
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
        if name == "workflow_run":
            return ToolResult.ok("started", data={"run_id": "run-42", "status": "pending"})
        raise AssertionError(f"signal_dispatch called an unexpected tool: {name}")


class _EmmasOwnTalon:
    """Emma's own ``talon`` provider: installed, but Claw runs every job, so
    its registry is empty. A delegating dispatch must not read it at all."""

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

    async def active_handles(self):
        raise AssertionError("signal_dispatch must not enumerate active_handles()")

    async def poll(self, handle):
        raise AssertionError("signal_dispatch must not poll: poll() reaps and writes")


class _EmmasPeers:
    """What ``A2AWaitable`` reads on Emma: her audit rows and Claw's tasks."""

    def __init__(self, claw):
        self._claw = claw

    def _native_outbound_store(self):
        claw = self._claw

        class _Audit:
            async def get(self, dispatch_id):
                return next(
                    (row for row in claw.rows if row.dispatch_id == dispatch_id), None
                )

        return _Audit()

    async def get_peer_task_result(self, dispatch_id):
        return self._claw.reads[dispatch_id]


def _agent(claw, name="Emma", *, talon=True, a2a=True):
    agent = SimpleNamespace(
        agent_name=name,
        did=EMMA_DID,
        agent_id=EMMA_DID,
        _raw_storage=SimpleNamespace(db=claw.db),
        execute_named_tool=claw.execute_named_tool,
        get_turn_bound_session_id=lambda: "chat-7",
        operator_registry=SimpleNamespace(
            get_workflow_registration=lambda workflow: SimpleNamespace(owner="feature:x")
        ),
        wait_registry=WaitRegistry(),
    )
    agent.talon = _EmmasOwnTalon()
    if talon:
        agent.wait_registry.register(agent.talon)
    if a2a:
        agent.wait_registry.register(A2AWaitable(_EmmasPeers(claw)))
    return agent


def _feature(agent):
    feature = StrategicMemoryFeature(agent=agent)
    feature._data = {"morning_signal_config": {"scan_repos": ["o/r"]}}
    feature._strategy_path = Path("/tmp/kestrel-test/STRATEGY.yaml")
    feature._save = MagicMock(return_value=_SaveOutcome(persisted=True))
    feature._ledger = StrategyLedger(Path("/tmp/kestrel-test/STRATEGY_LEDGER.yaml"))
    feature._ledger.data[BLOCKERS_KEY] = [
        {"severity": "high", "issue": "o/r#3093", "title": "Repair the boundary"}
    ]
    feature._ledger.data[PATTERNS_KEY] = []
    feature._ledger.normalize()
    feature._ledger.save = MagicMock(return_value=None)
    return feature


def _publish_project_talon_policy(monkeypatch, line):
    """Write a ``[talon.policy]`` table into the project's own ``kestrel.toml``."""
    unset_environment(monkeypatch, AGENT_DB_PATH_ENV)
    home = paths.project_dir()
    home.mkdir(parents=True, exist_ok=True)
    home.joinpath("kestrel.toml").write_text(f"[talon.policy]\n{line}\n", encoding="utf-8")


@pytest.fixture
def claw_dispatches(claw):
    claw.policy = _talon_policy("Claw")


def _github(monkeypatch, *, timeline=()):
    """GitHub as Emma's selector sees it: o/r#3093 open and ``agent-ready``,
    no PR on it, and ``timeline`` as everything on it since. Returns the
    paths read, so a test can show selection never ran."""
    monkeypatch.setattr(issue_selection, "get_github_token", lambda: "token")
    reads = []

    async def fake_get(path, token):
        reads.append(path)
        if path == "/repos/o/r/issues/3093":
            return {
                "number": 3093,
                "title": "Repair the boundary",
                "state": "open",
                "repository_url": "https://api.github.com/repos/o/r",
                "labels": [{"name": "agent-ready"}],
            }
        if "/issues?" in path:
            return []
        raise RuntimeError(f"404 {path}")

    async def fake_post(path, token, body):
        reads.append(body["query"])
        if "timelineItems" in body["query"]:
            return {"data": {"repository": {"issue": {
                "lastEditedAt": "2026-08-24T10:00:00Z",
                "timelineItems": {"nodes": list(timeline)},
            }}}}
        return {"data": {"repository": {"issue": {
            "closedByPullRequestsReferences": {"nodes": []},
        }}}}

    monkeypatch.setattr(issue_selection, "github_api_get", fake_get)
    monkeypatch.setattr(issue_selection, "github_api_post", fake_post)
    return reads


async def _history(claw, settled=None):
    return await read_executor_run_history(
        "Claw", claw.execute_named_tool_for_reader, settled
    )


@pytest.fixture
async def claw(tmp_path, sqlite_database_factory):
    executor = _Claw()
    executor.db = await sqlite_database_factory(tmp_path / "emma.db")
    executor.waits = WaitSignalStore(executor.db, EMMA_DID)

    async def call(name, args):
        return await executor.execute_named_tool(
            name, args, session_id="s", source="strategic_memory.signal_dispatch"
        )

    executor.execute_named_tool_for_reader = call
    return executor


# ---------------------------------------------------------------------------
# Which agent executes: what Emma's own Talon reports
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_executor_is_the_agent_talons_policy_names(
    claw, claw_dispatches, monkeypatch
):
    """Talon loads its policy from the file its feature config names, which
    need not be the project ``kestrel.toml``. Here the project table names
    Emma while Talon's names Claw: the launch gate refuses Emma, so the
    dispatch must go to Claw."""
    _publish_project_talon_policy(monkeypatch, 'dispatcher = "Emma"')
    _github(monkeypatch)

    result = await _feature(_agent(claw)).signal_dispatch(mode="execute")

    assert claw.calls[0] == (TALON_CONFIG_TOOL, {})
    assert result.data["executor"] == "Claw"
    assert claw.args("send_a2a_task")["recipient"] == "Claw"
    assert "workflow_run" not in claw.names()


@pytest.mark.asyncio
async def test_a_project_table_talon_does_not_load_names_no_executor(claw, monkeypatch):
    """The converse: the project table names Claw, but the policy Talon's
    launches load names nobody, so Emma launches her own."""
    _publish_project_talon_policy(monkeypatch, 'dispatcher = "Claw"')
    _github(monkeypatch)

    result = await _feature(_agent(claw)).signal_dispatch(mode="execute")

    assert claw.names() == [TALON_CONFIG_TOOL, "workflow_run"]
    assert result.data["dispatched"] is True
    assert "executor" not in result.data


@pytest.mark.asyncio
async def test_the_dispatcher_itself_still_launches_through_its_workflow(
    claw, claw_dispatches, monkeypatch
):
    _github(monkeypatch)
    agent = _agent(claw, name="Claw")

    result = await _feature(agent).signal_dispatch(mode="execute")

    assert result.status is ToolResultStatus.OK
    assert claw.names() == [TALON_CONFIG_TOOL, "workflow_run"]
    assert result.data["workflow_run_id"] == "run-42"
    assert agent.talon.reads == ["finished_runs", "workspace_readiness:o/r"]


@pytest.mark.asyncio
async def test_without_a_dispatcher_policy_the_agent_launches_its_own(claw, monkeypatch):
    _github(monkeypatch)
    agent = _agent(claw)

    result = await _feature(agent).signal_dispatch(mode="execute")

    assert claw.names() == [TALON_CONFIG_TOOL, "workflow_run"]
    assert result.data["dispatched"] is True


@pytest.mark.asyncio
async def test_a_talon_that_predates_the_dispatcher_key_lets_every_agent_launch(
    claw, monkeypatch
):
    _github(monkeypatch)
    claw.policy = {
        key: value for key, value in _talon_policy("Claw").items() if key != "dispatcher"
    }

    result = await _feature(_agent(claw)).signal_dispatch(mode="execute")

    assert claw.names() == [TALON_CONFIG_TOOL, "workflow_run"]
    assert result.data["dispatched"] is True


@pytest.mark.asyncio
async def test_an_agent_without_talon_asks_no_policy(claw, monkeypatch):
    """No ``talon`` provider: nothing on this agent enforces a dispatcher."""
    _github(monkeypatch)

    result = await _feature(_agent(claw, talon=False)).signal_dispatch(mode="execute")

    assert claw.names() == ["workflow_run"]
    assert result.data["dispatched"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unconfirm, detail",
    [
        pytest.param(
            lambda claw: claw.unregistered.add(TALON_CONFIG_TOOL),
            "no enabled feature exposes 'talon_get_config'",
            id="no-tool",
        ),
        pytest.param(
            lambda claw: claw.raises.__setitem__(TALON_CONFIG_TOOL, RuntimeError("toml locked")),
            "RuntimeError: toml locked",
            id="raised",
        ),
        pytest.param(
            lambda claw: setattr(
                claw, "policy_result",
                ToolResult.failed(
                    "Talon config_path is required for policy and preference operations",
                    data={"success": False},
                ),
            ),
            "Talon config_path is required",
            id="talon-refused",
        ),
        pytest.param(
            lambda claw: setattr(
                claw, "policy_result",
                {"success": False, "error": "Permission denied: talon_get_config"},
            ),
            "Permission denied: talon_get_config",
            id="permission-denied",
        ),
        pytest.param(
            lambda claw: setattr(
                claw, "policy_result", ToolResult.ok("loaded", data={"success": True})
            ),
            "reported no policy",
            id="no-policy",
        ),
    ],
)
@pytest.mark.parametrize("mode", ["execute", "suggest"])
async def test_an_unconfirmed_dispatcher_policy_dispatches_nothing(
    claw, monkeypatch, unconfirm, detail, mode
):
    """Where the work runs is unknown: launching here could be refused every
    day, and delegating could hand it to an agent Talon does not name."""
    reads = _github(monkeypatch)
    unconfirm(claw)

    result = await _feature(_agent(claw)).signal_dispatch(mode=mode)

    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["reason_code"] == "DISPATCHER_POLICY_UNCONFIRMED"
    assert result.data["dispatched"] is False
    assert detail in result.confirmation
    assert claw.names() == [TALON_CONFIG_TOOL]
    assert reads == []


@pytest.mark.asyncio
@pytest.mark.parametrize("dispatcher", ["  ", 7])
@pytest.mark.parametrize("mode", ["execute", "suggest"])
async def test_a_dispatcher_naming_no_agent_dispatches_nothing(claw, monkeypatch, dispatcher, mode):
    """Talon refuses every launch over such a setting; reading it as absent
    would launch here anyway."""
    claw.policy = _talon_policy(dispatcher)
    reads = _github(monkeypatch)

    result = await _feature(_agent(claw)).signal_dispatch(mode=mode)

    assert result.status is ToolResultStatus.ERROR
    assert result.data["reason_code"] == "DISPATCHER_POLICY_INVALID"
    assert claw.names() == [TALON_CONFIG_TOOL] and reads == []


@pytest.mark.asyncio
async def test_without_governed_dispatch_talon_cannot_be_asked(monkeypatch):
    _github(monkeypatch)
    agent = SimpleNamespace(agent_name="Emma", wait_registry=WaitRegistry())
    agent.wait_registry.register(_EmmasOwnTalon())

    result = await _feature(agent).signal_dispatch(mode="suggest")

    assert result.data["reason_code"] == "GOVERNED_DISPATCH_UNAVAILABLE"
    assert result.data["dispatched"] is False


# ---------------------------------------------------------------------------
# Reading Claw's work reports
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_read_lists_only_dispatches_to_the_executor(claw):
    await _history(claw)

    assert claw.args(LIST_OUTBOUND_TOOL) == {
        "limit": WORK_ORDER_PAGE_SIZE,
        "recipient": "Claw",
    }


@pytest.mark.asyncio
async def test_the_read_walks_the_whole_audit(claw):
    """An order guards its issue however old it is, and an unanswered one
    holds the lane however many dispatches came after it (review P2): 50
    newer unrelated dispatches once pushed both out of the listing."""
    claw.answered("blocked", TaskState.TASK_STATE_COMPLETED, _report("blocked"))
    claw.pending("working", order=_order(issue=3094, order_id="wo-3094-1"))
    claw.unrelated(WORK_ORDER_PAGE_SIZE * 2)

    history = await _history(claw)

    assert history.latest("o/r", 3093).disposition == "blocked"
    assert [order.dispatch_id for order in history.outstanding] == ["working"]
    cursors = [args.get("before") for name, args in claw.calls if name == LIST_OUTBOUND_TOOL]
    assert cursors == [
        None,
        claw.rows[WORK_ORDER_PAGE_SIZE - 1].dispatch_id,
        claw.rows[2 * WORK_ORDER_PAGE_SIZE - 1].dispatch_id,
    ]


@pytest.mark.asyncio
async def test_a_listing_that_repeats_a_dispatch_is_unreadable(claw):
    """A cursor that does not advance would page forever, or stop short."""
    claw.unrelated(WORK_ORDER_PAGE_SIZE)
    listing = claw.execute_named_tool

    async def stuck(name, args, **kwargs):
        if name == LIST_OUTBOUND_TOOL:
            args = {key: value for key, value in args.items() if key != "before"}
        return await listing(name, args, **kwargs)

    claw.execute_named_tool = stuck

    with pytest.raises(RunHistoryUnreadable, match="twice"):
        await _history(claw)


@pytest.mark.asyncio
async def test_a_settled_dispatch_is_read_from_the_executor_once(claw):
    """A terminal task cannot change, and neither can the message Emma sent,
    so a growing audit is not re-read from Claw every dispatch. An order
    still running is read every time."""
    claw.answered("d1", TaskState.TASK_STATE_COMPLETED, _report("blocked"))
    claw.pending("d2", order=_order(issue=3094, order_id="wo-3094-1"))
    claw._add("d3", [{"text": "look at CI"}], TaskState.TASK_STATE_WORKING)
    settled = {}

    first = await _history(claw, settled)
    claw.calls.clear()
    second = await _history(claw, settled)

    assert claw.peer_reads() == ["d2"]
    assert second.runs == first.runs
    assert second.outstanding == first.outstanding
    assert second.answered == first.answered == ("d1",)
    assert set(settled) == {"d1", "d3"}


@pytest.mark.asyncio
async def test_a_continuation_settled_earlier_is_still_read_once(claw):
    claw.answered("d1", TaskState.TASK_STATE_COMPLETED, _report("blocked"))
    claw.rows.insert(0, _row("d2"))
    claw.reads["d2"] = claw.reads["d1"]
    settled = {}
    await _history(claw, settled)

    history = await _history(claw, settled)

    assert history.answered == ("d2",)


@pytest.mark.asyncio
async def test_an_answered_order_is_a_run_on_the_issue_the_order_named(claw):
    """The report says issue 9999. The run is the order's issue, 3093: the
    issue an answer guards is the one this agent sent."""
    claw.answered(
        "d1", TaskState.TASK_STATE_COMPLETED, _report("blocked", issue=9999)
    )

    history = await _history(claw)

    run = history.latest("O/R", 3093)
    assert run is not None
    assert history.latest("o/r", 9999) is None
    assert (run.disposition, run.job_id, run.order_id) == ("blocked", TALON_JOB, "wo-3093-1")
    assert (run.executor, run.dispatch_id, run.completed_at) == ("Claw", "d1", ANSWERED)
    assert run.awaits_new_input
    assert history.executor == "Claw"
    assert history.outstanding == ()
    assert history.answered == ("d1",)


@pytest.mark.asyncio
async def test_the_order_survives_the_protobuf_round_trip(claw):
    """A2A carries the data part as a protobuf Value: issue 3758 comes back
    as 3758.0 and must still identify the issue."""
    claw.answered(
        "d1",
        TaskState.TASK_STATE_COMPLETED,
        _report("clarifying", issue=3758),
        order=_order(issue=3758, order_id="wo-3758-2"),
    )
    task = claw.reads["d1"].data["task"]
    assert task["history"][0]["parts"][0]["data"]["issue"] == 3758.0

    run = (await _history(claw)).latest("o/r", 3758)

    assert run is not None and run.disposition == "clarifying"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state", [TaskState.TASK_STATE_FAILED, TaskState.TASK_STATE_REJECTED]
)
async def test_an_order_the_executor_could_not_run_is_a_refusal(claw, state):
    """``failed`` is Claw's "could not execute it" -- no workspace, not
    agent-ready, the wrong sender -- whatever the answer text says."""
    claw.answered("d1", state, _report("completed"))

    run = (await _history(claw)).latest("o/r", 3093)

    assert run.disposition == ORDER_REFUSED
    assert run.job_id == "d1"
    assert run.awaits_new_input and not run.ended_with_question


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state, answer, disposition",
    [
        pytest.param(TaskState.TASK_STATE_CANCELED, None, ORDER_CANCELLED, id="cancelled"),
        pytest.param(TaskState.TASK_STATE_COMPLETED, _report("completed"), "completed", id="pr"),
        pytest.param(TaskState.TASK_STATE_COMPLETED, _report("failed"), "failed", id="talon-failed"),
        pytest.param(TaskState.TASK_STATE_COMPLETED, "Done, see PR", "unknown", id="free-text"),
        pytest.param(TaskState.TASK_STATE_COMPLETED, "{" * 100000, "unknown", id="deep-json"),
        pytest.param(
            TaskState.TASK_STATE_COMPLETED,
            json.dumps({"schema": "other/v1", "disposition": "blocked"}),
            "unknown",
            id="not-a-report",
        ),
    ],
)
async def test_an_answer_that_asks_nothing_does_not_wait_for_input(
    claw, state, answer, disposition
):
    claw.answered("d1", state, answer)

    run = (await _history(claw)).latest("o/r", 3093)

    assert run.disposition == disposition
    assert not run.awaits_new_input


@pytest.mark.asyncio
async def test_an_unanswered_order_is_outstanding(claw):
    claw.pending("d1", TaskState.TASK_STATE_SUBMITTED, order=_order(order_id="wo-3093-2"))
    claw.uncertain("d2")

    history = await _history(claw)

    assert history.runs == {}
    assert [(o.dispatch_id, o.state, o.order_id, o.repo, o.issue_number) for o in history.outstanding] == [
        ("d2", DELIVERY_UNCERTAIN, None, None, None),
        ("d1", "submitted", "wo-3093-2", "o/r", 3093),
    ]


@pytest.mark.asyncio
async def test_an_order_waiting_on_its_sender_is_outstanding(claw):
    claw.pending("d1", TaskState.TASK_STATE_INPUT_REQUIRED)

    [order] = (await _history(claw)).outstanding

    assert order.state == "input-required"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "row",
    [
        pytest.param(_row("d1", DispatchState.RESERVED), id="never-submitted"),
        pytest.param(_row("d1", DispatchState.NOT_SENT), id="proved-not-delivered"),
        pytest.param(_row("d1", verb="message"), id="a-message"),
        pytest.param(_row("d1", verb="question"), id="a-question"),
    ],
)
async def test_a_dispatch_that_is_no_delivered_task_is_not_read(claw, row):
    claw.rows.append(row)

    history = await _history(claw)

    assert history.runs == {} and history.outstanding == ()
    assert READ_PEER_TASK_TOOL not in claw.names()


@pytest.mark.asyncio
async def test_a_task_carrying_no_order_is_not_a_work_order(claw):
    """A text-only task, and one whose order-shaped data is only in its
    artifacts -- the recipient's output, not what the sender sent."""
    from kestrel_sdk.native_a2a.types import Artifact

    claw._add("d1", [{"text": "please look at the CI"}], TaskState.TASK_STATE_WORKING)
    artifact = Artifact(artifact_id="a", name="work_order")
    artifact.parts.append(ParseDict({"data": _order()}, Part()))
    claw.rows.insert(0, _row("d2"))
    claw.reads["d2"] = _read(
        "d2",
        _task("d2", [{"text": "hi"}], TaskState.TASK_STATE_FAILED, artifacts=[artifact]),
    )

    history = await _history(claw)

    assert history.runs == {} and history.outstanding == () and history.answered == ()


@pytest.mark.asyncio
async def test_a_continuation_of_the_same_task_is_read_once(claw):
    claw.pending("d1")
    claw.rows.insert(0, _row("d2"))
    claw.reads["d2"] = claw.reads["d1"]

    assert len((await _history(claw)).outstanding) == 1


@pytest.mark.asyncio
async def test_a_task_answered_with_a_message_leaves_nothing_running(claw):
    claw.rows.insert(0, _row("d1"))
    claw.reads["d1"] = PeersFeature._native_peer_result(
        SimpleNamespace(
            dispatch_id="d1", remote_task_id=None, remote_task_state=None,
            response_kind="message",
        ),
        Message(message_id="m", role=Role.ROLE_AGENT, parts=[Part(text="no")]),
    )

    history = await _history(claw)

    assert history.runs == {} and history.outstanding == ()


@pytest.mark.asyncio
async def test_a_withholding_answer_on_an_order_naming_no_issue_is_unreadable(claw):
    claw.answered(
        "d1", TaskState.TASK_STATE_COMPLETED, _report("blocked"),
        order={**_order(), "issue": "next one"},
    )

    with pytest.raises(RunHistoryUnreadable, match="does not identify an issue"):
        await _history(claw)


@pytest.mark.asyncio
async def test_an_answer_that_withholds_nothing_on_an_order_naming_no_issue_is_skipped(claw):
    claw.answered(
        "d1", TaskState.TASK_STATE_COMPLETED, _report("completed"),
        order={**_order(), "repo": ""},
    )

    history = await _history(claw)

    assert history.runs == {}
    assert history.answered == ("d1",)


@pytest.mark.asyncio
async def test_an_answer_without_a_time_is_unreadable(claw):
    """Nothing could then say whether a later comment came after it."""
    claw.answered("d1", TaskState.TASK_STATE_COMPLETED, _report("completed"), at=None)

    with pytest.raises(RunHistoryUnreadable, match="no readable answer time"):
        await _history(claw)


@pytest.mark.asyncio
async def test_a_task_that_cannot_be_read_is_unreadable(claw):
    claw.rows.insert(0, _row("d1"))
    claw.reads["d1"] = ToolResult.failed(
        "Could not read native peer task: peer unreachable", data={"dispatch_id": "d1"}
    )

    with pytest.raises(RunHistoryUnreadable, match="peer unreachable"):
        await _history(claw)


@pytest.mark.asyncio
async def test_a_listing_a_permission_hook_denied_is_unreadable(claw, monkeypatch):
    async def denied(name, args):
        return {"success": False, "error": "Permission denied: list_outbound_a2a_tasks"}

    with pytest.raises(RunHistoryUnreadable, match="Permission denied"):
        await read_executor_run_history("Claw", denied)


@pytest.mark.asyncio
async def test_a_read_that_raises_is_unreadable(claw):
    claw.raises[LIST_OUTBOUND_TOOL] = RuntimeError("database is locked")

    with pytest.raises(RunHistoryUnreadable, match="database is locked"):
        await _history(claw)


@pytest.mark.asyncio
async def test_without_the_peers_tools_the_read_says_so(claw):
    claw.unregistered.add(LIST_OUTBOUND_TOOL)

    with pytest.raises(ToolNotRegisteredError):
        await _history(claw)


# ---------------------------------------------------------------------------
# signal_dispatch on the queue owner
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_sends_the_executor_a_work_order_and_arms_its_wake(
    claw, claw_dispatches, monkeypatch
):
    _github(monkeypatch)
    agent = _agent(claw)

    result = await _feature(agent).signal_dispatch(mode="execute")

    assert result.status is ToolResultStatus.OK, result.error
    assert claw.names() == [TALON_CONFIG_TOOL, LIST_OUTBOUND_TOOL, "send_a2a_task"]
    send = claw.args("send_a2a_task")
    assert send["recipient"] == "Claw"
    [part] = send["parts"]
    order = part["data"]
    assert order["schema"] == WORK_ORDER_SCHEMA
    assert (order["repo"], order["issue"], order["action"]) == ("o/r", 3093, "claim")
    assert order["order_id"].startswith("wo-3093-")
    assert "Repair the boundary" in order["context"]
    assert len(order["context"].split()) <= 300
    dispatch_id = result.data["dispatch_id"]
    watch = await claw.waits.get("a2a", dispatch_id)
    assert watch.watching
    # The report wakes the session that sent the order, as wait() does.
    assert watch.watch_origin_session_id == "chat-7"
    assert result.data["dispatched"] is True
    assert result.data["executor"] == "Claw"
    assert result.data["order_id"] == order["order_id"]
    assert result.data["remote_task_id"] == f"task-{dispatch_id}"
    assert result.data["wake"] == f"a2a:{dispatch_id}"
    assert dispatch_id in result.confirmation
    # Emma's own Talon is neither the run history nor the workspace answer.
    assert agent.talon.reads == []


@pytest.mark.asyncio
async def test_suggest_names_the_executor_and_sends_nothing(claw, claw_dispatches, monkeypatch):
    _github(monkeypatch)
    agent = _agent(claw)

    result = await _feature(agent).signal_dispatch(mode="suggest")

    assert result.status is ToolResultStatus.OK
    assert result.data["issue"]["issue_number"] == 3093
    assert result.data["executor"] == "Claw"
    assert "Execute mode will send Claw a `kestrel.work_order/v1` work order" in (
        result.confirmation
    )
    assert claw.names() == [TALON_CONFIG_TOOL, LIST_OUTBOUND_TOOL]
    assert agent.talon.reads == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["execute", "suggest"])
async def test_a_blocked_report_withholds_its_issue(claw, claw_dispatches, monkeypatch, mode):
    """#3398 on the queue owner: Claw's last run on #3093 blocked, nothing on
    the issue since authorizes a retry, so the same order is not sent again."""
    _github(monkeypatch)
    claw.answered("d1", TaskState.TASK_STATE_COMPLETED, _report("blocked"))

    result = await _feature(_agent(claw)).signal_dispatch(mode=mode)

    assert result.status is ToolResultStatus.OK
    assert result.data["dispatched"] is False
    assert result.data["issue"] is None
    [skip] = result.data["skipped"]
    assert (skip["reason"], skip["executor"], skip["order_id"]) == (
        "run_history", "Claw", "wo-3093-1",
    )
    assert (
        "skipped o/r#3093 -- Claw reported work order wo-3093-1 blocked at "
        "2026-10-24 09:56 UTC without a PR (Talon job 802b93fe), and nothing "
        "since authorizes a retry"
    ) in result.confirmation
    assert "send_a2a_task" not in claw.names()


@pytest.mark.asyncio
@pytest.mark.parametrize("verb", ["message", "task"])
async def test_a_blocked_report_still_withholds_after_many_newer_dispatches(
    claw, claw_dispatches, monkeypatch, verb
):
    """Review P2's reproduction: 50 newer dispatches to Claw pushed the
    blocked report out of a 50-row listing, and #3093 was ordered again
    against unchanged input."""
    _github(monkeypatch)
    claw.answered("d1", TaskState.TASK_STATE_COMPLETED, _report("blocked"))
    claw.unrelated(WORK_ORDER_PAGE_SIZE + 1, verb="message")
    if verb == "task":
        for n in range(3):
            claw._add(f"chat-{n}", [{"text": "look at CI"}], TaskState.TASK_STATE_COMPLETED)

    result = await _feature(_agent(claw)).signal_dispatch(mode="execute")

    assert result.data["dispatched"] is False
    assert [skip["dispatch_id"] for skip in result.data["skipped"]] == ["d1"]
    assert "send_a2a_task" not in claw.names()


@pytest.mark.asyncio
async def test_an_order_unanswered_behind_many_newer_dispatches_holds_the_lane(
    claw, claw_dispatches, monkeypatch
):
    reads = _github(monkeypatch)
    claw.pending("d1")
    claw.unrelated(WORK_ORDER_PAGE_SIZE + 1)

    result = await _feature(_agent(claw)).signal_dispatch(mode="execute")

    assert result.data["dispatched"] is False
    assert [order["dispatch_id"] for order in result.data["outstanding"]] == ["d1"]
    assert "send_a2a_task" not in claw.names() and reads == []


@pytest.mark.asyncio
async def test_an_order_the_executor_refused_withholds_its_issue(
    claw, claw_dispatches, monkeypatch
):
    """Claw could not run the last order (no workspace, say). Its refusal is
    the workspace answer Emma cannot get from her own Talon."""
    _github(monkeypatch)
    claw.answered(
        "d1", TaskState.TASK_STATE_FAILED, "No Talon workspace exists for o/r."
    )

    result = await _feature(_agent(claw)).signal_dispatch(mode="execute")

    assert result.data["dispatched"] is False
    assert (
        "skipped o/r#3093 -- Claw refused work order wo-3093-1 at 2026-10-24 "
        "09:56 UTC, and nothing since authorizes a retry"
    ) in result.confirmation
    assert "send_a2a_task" not in claw.names()


@pytest.mark.asyncio
async def test_new_input_after_the_report_authorizes_a_new_order(
    claw, claw_dispatches, monkeypatch
):
    _github(monkeypatch, timeline=[{
        "__typename": "IssueComment",
        "createdAt": "2026-10-24T12:00:00Z",
        "authorAssociation": "OWNER",
        "body": "Scope ruled: only the reader. Retry.",
    }])
    claw.answered("d1", TaskState.TASK_STATE_COMPLETED, _report("blocked"))

    result = await _feature(_agent(claw)).signal_dispatch(mode="execute")

    assert result.data["dispatched"] is True
    assert claw.args("send_a2a_task")["parts"][0]["data"]["issue"] == 3093


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outstanding",
    [
        pytest.param(lambda claw: claw.pending("d1"), id="unanswered"),
        pytest.param(lambda claw: claw.uncertain("d1"), id="delivery-uncertain"),
    ],
)
async def test_an_unanswered_order_holds_the_executors_lane(
    claw, claw_dispatches, monkeypatch, outstanding
):
    reads = _github(monkeypatch)
    outstanding(claw)

    result = await _feature(_agent(claw)).signal_dispatch(mode="execute")

    assert result.status is ToolResultStatus.OK
    assert result.data["dispatched"] is False
    [order] = result.data["outstanding"]
    assert order["dispatch_id"] == "d1"
    assert "Claw has not answered" in result.confirmation
    assert "d1" in result.confirmation
    assert "send_a2a_task" not in claw.names()
    assert reads == []


@pytest.mark.asyncio
async def test_suggest_says_execute_waits_for_the_outstanding_order(
    claw, claw_dispatches, monkeypatch
):
    _github(monkeypatch)
    claw.pending("d1", order=_order(issue=3094, order_id="wo-3094-1"))

    result = await _feature(_agent(claw)).signal_dispatch(mode="suggest")

    assert result.data["issue"]["issue_number"] == 3093
    assert "once it has answered work order wo-3094-1 on o/r#3094" in result.confirmation
    assert result.data["outstanding"][0]["order_id"] == "wo-3094-1"


@pytest.mark.asyncio
async def test_the_order_it_sends_holds_the_lane_for_the_next_run(
    claw, claw_dispatches, monkeypatch
):
    _github(monkeypatch)
    feature = _feature(_agent(claw))

    first = await feature.signal_dispatch(mode="execute")
    second = await feature.signal_dispatch(mode="execute")

    assert first.data["dispatched"] is True
    assert second.data["dispatched"] is False
    assert first.data["order_id"] in second.confirmation
    assert claw.names().count("send_a2a_task") == 1


@pytest.mark.asyncio
async def test_concurrent_executes_send_one_order(claw, claw_dispatches, monkeypatch):
    """A scheduled dispatch and a turn's: without the lock both read an idle
    lane before either sends."""
    _github(monkeypatch)
    feature = _feature(_agent(claw))

    results = await asyncio.gather(
        feature.signal_dispatch(mode="execute"),
        feature.signal_dispatch(mode="execute"),
    )

    assert sorted(result.data["dispatched"] for result in results) == [False, True]
    assert claw.names().count("send_a2a_task") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["execute", "suggest"])
async def test_unreadable_reports_dispatch_nothing(claw, claw_dispatches, monkeypatch, mode):
    reads = _github(monkeypatch)
    claw.rows.insert(0, _row("d1"))
    claw.reads["d1"] = ToolResult.failed(
        "Could not read native peer task: peer unreachable", data={"dispatch_id": "d1"}
    )

    result = await _feature(_agent(claw)).signal_dispatch(mode=mode)

    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["reason_code"] == "RUN_HISTORY_UNCONFIRMED"
    assert result.data["dispatched"] is False
    assert "Could not confirm Claw's work reports" in result.confirmation
    assert "peer unreachable" in result.confirmation
    assert "send_a2a_task" not in claw.names()
    assert reads == []


@pytest.mark.asyncio
async def test_without_peers_nothing_is_dispatched(claw, claw_dispatches, monkeypatch):
    _github(monkeypatch)
    claw.unregistered.add(LIST_OUTBOUND_TOOL)

    result = await _feature(_agent(claw)).signal_dispatch(mode="execute")

    assert result.status is ToolResultStatus.ERROR
    assert result.data["reason_code"] == "WORK_ORDER_UNAVAILABLE"
    assert result.data["dispatched"] is False


@pytest.mark.asyncio
async def test_a_refused_send_sends_nothing_and_arms_nothing(claw, claw_dispatches, monkeypatch):
    _github(monkeypatch)
    claw.send_result = ToolResult.failed(
        "Could not send A2A task: peer directory has no agent 'Claw'",
        data={"sent": False, "recipient": "Claw", "dispatch_id": None,
              "delivery_state": "not_submitted"},
    )

    result = await _feature(_agent(claw)).signal_dispatch(mode="execute")

    assert result.status is ToolResultStatus.ERROR
    assert result.data["reason_code"] == "WORK_ORDER_REJECTED"
    assert result.data["dispatched"] is False
    assert await claw.waits.list_watched() == []


@pytest.mark.asyncio
async def test_a_send_a_permission_hook_denied_is_refused(claw, claw_dispatches, monkeypatch):
    _github(monkeypatch)
    claw.send_result = {"success": False, "error": "Permission denied: send_a2a_task"}

    result = await _feature(_agent(claw)).signal_dispatch(mode="execute")

    assert result.data["reason_code"] == "WORK_ORDER_REJECTED"
    assert "Permission denied" in result.error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sent, error, dispatched, row_state",
    [
        pytest.param(
            None, "A2A delivery is uncertain; reconcile the dispatch before sending again",
            None, DispatchState.UNKNOWN, id="uncertain",
        ),
        pytest.param(
            True, "A2A delivery was acknowledged; inspect the retained dispatch before further action",
            True, DispatchState.ACKNOWLEDGED, id="acknowledged",
        ),
    ],
)
async def test_a_send_that_may_have_arrived_is_not_sent_again_and_is_watched(
    claw, claw_dispatches, monkeypatch, sent, error, dispatched, row_state
):
    _github(monkeypatch)
    claw.send_row_state = row_state
    claw.send_result = ToolResult.failed(
        error,
        data={"sent": sent, "recipient": "Claw", "dispatch_id": "d9",
              "remote_task_id": None, "delivery_state": "unknown"},
    )

    result = await _feature(_agent(claw)).signal_dispatch(mode="execute")

    assert result.status is ToolResultStatus.ERROR
    assert result.data["reason_code"] == "WORK_ORDER_UNCONFIRMED"
    assert result.data["dispatched"] is dispatched
    assert result.data["dispatch_id"] == "d9"
    assert (await claw.waits.get("a2a", "d9")).watching
    assert "Do not send it again" in result.error


@pytest.mark.asyncio
async def test_a_send_that_raises_is_unconfirmed(claw, claw_dispatches, monkeypatch):
    _github(monkeypatch)
    claw.raises["send_a2a_task"] = RuntimeError("socket closed")

    result = await _feature(_agent(claw)).signal_dispatch(mode="execute")

    assert result.data["reason_code"] == "WORK_ORDER_UNCONFIRMED"
    assert result.data["dispatched"] is None
    assert "list_outbound_a2a_tasks(recipient='Claw')" in result.error


@pytest.mark.asyncio
async def test_a_send_naming_no_dispatch_is_unconfirmed(claw, claw_dispatches, monkeypatch):
    _github(monkeypatch)
    claw.send_result = ToolResult.ok("A2A task sent to Claw", data={"sent": True})

    result = await _feature(_agent(claw)).signal_dispatch(mode="execute")

    assert result.data["reason_code"] == "WORK_ORDER_UNCONFIRMED"
    assert await claw.waits.list_watched() == []


async def _register_fails(agent, ref, *, origin_session_id=None):
    raise RuntimeError("db closed")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unarm, detail",
    [
        pytest.param(
            lambda monkeypatch: None, "no wait provider for kind 'a2a'", id="no-a2a-provider",
        ),
        pytest.param(
            lambda monkeypatch: monkeypatch.setattr(
                strategic_feature, "register_wait_watch", _register_fails
            ),
            "arming it raised RuntimeError",
            id="raised",
        ),
    ],
)
async def test_an_order_whose_wake_was_not_armed_says_how_to_arm_it(
    claw, claw_dispatches, monkeypatch, unarm, detail
):
    """#3755: nothing else wakes the sender when Claw answers."""
    _github(monkeypatch)
    unarm(monkeypatch)
    a2a = detail != "no wait provider for kind 'a2a'"

    result = await _feature(_agent(claw, a2a=a2a)).signal_dispatch(mode="execute")

    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["reason_code"] == "WORK_ORDER_WAKE_UNARMED"
    assert result.data["dispatched"] is True
    assert result.data["wake"] is None
    dispatch_id = result.data["dispatch_id"]
    assert f"wait(target='a2a:{dispatch_id}', mode='signal')" in result.confirmation
    assert detail in result.confirmation


# ---------------------------------------------------------------------------
# A wake lost between the executor's acceptance and its watch
# ---------------------------------------------------------------------------


async def _wakes(claw, agent):
    """Run the real reconciler over Emma's watches; the wakes it emits."""
    agent.dispatcher = _CapturingDispatcher()
    await WaitReconciler(agent).reconcile()
    return [signal.payload["handle"] for signal in agent.dispatcher.signals]


async def _interrupted_send(claw, monkeypatch, agent):
    """Send the order, then stop before its watch is registered, as a
    process that dies there does."""
    async def stops(agent, ref, *, origin_session_id=None):
        raise asyncio.CancelledError()

    monkeypatch.setattr(strategic_feature, "register_wait_watch", stops)
    with pytest.raises(asyncio.CancelledError):
        await _feature(agent).signal_dispatch(mode="execute")
    monkeypatch.undo()
    [dispatch_id] = [
        row.dispatch_id for row in claw.rows if claw.reads[row.dispatch_id].status
        is ToolResultStatus.OK
    ]
    assert await claw.waits.get("a2a", dispatch_id) is None
    return dispatch_id


def _answer(claw, dispatch_id, state, answer):
    sent = claw.reads[dispatch_id].data["task"]["history"][0]["parts"]
    claw.reads[dispatch_id] = _read(dispatch_id, _task(dispatch_id, sent, state, answer=answer))


@pytest.mark.asyncio
async def test_an_order_answered_after_an_interrupted_watch_still_wakes_its_sender(
    claw, claw_dispatches, monkeypatch
):
    """Review P2's reproduction. The order went out, the dispatch stopped
    before its watch was armed, and Claw answered before the next daily
    dispatch. That dispatch arms the missing watch, so the reconciler wakes
    Emma with Claw's report -- where before it scanned zero handles."""
    _github(monkeypatch)
    agent = _agent(claw)
    dispatch_id = await _interrupted_send(claw, monkeypatch, agent)
    _github(monkeypatch)
    _answer(claw, dispatch_id, TaskState.TASK_STATE_COMPLETED, _report("blocked"))
    assert await _wakes(claw, agent) == []

    result = await _feature(agent).signal_dispatch(mode="execute")

    assert result.data["dispatched"] is False
    assert claw.names().count("send_a2a_task") == 1
    watch = await claw.waits.get("a2a", dispatch_id)
    assert watch.watching
    assert watch.watch_origin_session_id == "chat-7"
    assert await _wakes(claw, agent) == [dispatch_id]


@pytest.mark.asyncio
async def test_an_order_still_running_after_an_interrupted_watch_wakes_when_answered(
    claw, claw_dispatches, monkeypatch
):
    _github(monkeypatch)
    agent = _agent(claw)
    dispatch_id = await _interrupted_send(claw, monkeypatch, agent)
    _github(monkeypatch)

    result = await _feature(agent).signal_dispatch(mode="execute")

    assert result.data["outstanding"][0]["dispatch_id"] == dispatch_id
    assert await _wakes(claw, agent) == []
    _answer(claw, dispatch_id, TaskState.TASK_STATE_COMPLETED, _report("completed"))
    assert await _wakes(claw, agent) == [dispatch_id]


@pytest.mark.asyncio
async def test_recovery_leaves_a_registered_wake_as_it_is(claw, claw_dispatches, monkeypatch):
    """An order whose wake was delivered in another session keeps that
    session and that delivery: recovery neither moves it nor replays it."""
    _github(monkeypatch)
    claw.answered("d1", TaskState.TASK_STATE_COMPLETED, _report("blocked"))
    claw.pending("d2", order=_order(issue=3094, order_id="wo-3094-1"))
    await claw.waits.start_watch("a2a", "d1", origin_session_id="chat-orig")
    await claw.waits.seed_signaled("a2a", "d1", "done")
    await claw.waits.stop_watch("a2a", "d1")
    await claw.waits.start_watch("a2a", "d2", origin_session_id="chat-orig")
    before = {handle: await claw.waits.get("a2a", handle) for handle in ("d1", "d2")}
    agent = _agent(claw)

    await _feature(agent).signal_dispatch(mode="execute")

    for handle, state in before.items():
        assert await claw.waits.get("a2a", handle) == state
    assert await _wakes(claw, agent) == []


@pytest.mark.asyncio
async def test_suggest_arms_no_wake(claw, claw_dispatches, monkeypatch):
    _github(monkeypatch)
    claw.pending("d1", order=_order(issue=3094, order_id="wo-3094-1"))

    await _feature(_agent(claw)).signal_dispatch(mode="suggest")

    assert await claw.waits.get("a2a", "d1") is None


@pytest.mark.asyncio
async def test_a_wake_that_cannot_be_recovered_sends_no_new_order(
    claw, claw_dispatches, monkeypatch
):
    """The wait engine that cannot arm the old order's wake would not arm the
    new one's either."""
    _github(monkeypatch)
    claw.answered("d1", TaskState.TASK_STATE_COMPLETED, _report("completed"))

    result = await _feature(_agent(claw, a2a=False)).signal_dispatch(mode="execute")

    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["reason_code"] == "WORK_ORDER_WAKE_UNARMED"
    assert result.data["dispatched"] is False
    [unarmed] = result.data["unarmed_wakes"]
    assert unarmed.startswith("a2a:d1: no wait provider for kind 'a2a'")
    assert "send_a2a_task" not in claw.names()


@pytest.mark.asyncio
async def test_a_settled_order_is_not_read_again_by_the_next_dispatch(
    claw, claw_dispatches, monkeypatch
):
    _github(monkeypatch)
    claw.answered("d1", TaskState.TASK_STATE_COMPLETED, _report("blocked"))
    feature = _feature(_agent(claw))

    await feature.signal_dispatch(mode="execute")
    claw.calls.clear()
    second = await feature.signal_dispatch(mode="execute")

    assert claw.peer_reads() == []
    assert [skip["dispatch_id"] for skip in second.data["skipped"]] == ["d1"]


def test_the_fake_audit_row_is_a_real_outbound_dispatch():
    """``replace`` keeps the fake honest: every field the store has, no more."""
    row = _row("d1")
    assert replace(row, state=DispatchState.COMPLETE).dispatch_id == "d1"
