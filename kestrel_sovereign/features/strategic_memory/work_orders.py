"""Work orders this agent sent its executor, and the executor's reports (#3758).

The fleet splits picking work from running it. The queue owner (Emma) reads
her ledger and her ``scan_repos`` and picks; one other agent (Claw) is the only
one allowed to start Talon runs (kestrel-feature-talon #57,
``[talon.policy] dispatcher``). ``signal_dispatch`` used to run the pick
through ``workflow_run`` on the agent that picked it, which a non-dispatcher's
Talon now refuses. And the selection guards read the picking agent's own Talon
registry, which is empty there: they could not see that the executor's last
run on an issue stopped to ask a question (#3398).

``morning_signal_config.dispatch_executor`` names the executor. With it set,
``signal_dispatch`` sends the pick as an A2A task carrying a
``kestrel.work_order/v1`` data artifact, and the executor answers that task
once, with a ``kestrel.work_report/v1`` JSON object, when it has the outcome.
It is an explicit, per-agent opt-in: an agent that merely still has a
``signal_dispatch`` row (#3752) does not start sending orders.

The journal
-----------

Every order is kept in a durable, agent-scoped journal (the
``strategic_work_orders`` table), which stands in for the executor's registry.
Core does not reach into another agent's runtime; the journal records what
this agent sent and what the A2A protocol later told it about each order:

* the sender-side outbound record of every task it sent the executor
  (``PeersFeature.list_outbound_a2a_tasks``): the task id, when it was sent,
  whether the executor accepted it (``route_state``) and, for a send that
  failed, why (``error``);
* each of those tasks as the executor holds it
  (``PeersFeature.get_peer_task_result``): its state says whether the order is
  still outstanding, and its answer is the report.

A row is one order. One ``signal_dispatch`` sends moves through:

``reserved``
    The lane is held while the dispatch selects. The row is the reservation:
    a unique index admits one open dispatch row per executor, so two
    dispatches running at once cannot both pass the idle check and send.
    A reservation that never became an order is withdrawn.
``sending``
    An issue was chosen and the send is being attempted. The order's A2A
    ``session_id`` is its ``order_id``, so the outbound record of the attempt
    is found by it even when the send never returned a task id.
``sent``
    The executor accepted the task. Outstanding until its answer is observed.
``unconfirmed``
    The send may have reached the executor, and nothing can say whether it
    did: the transport returned after delivery but could not retain the
    route, the request timed out, or the dispatch stopped mid-send. The
    order keeps the executor's lane busy and its issue withheld until the
    outbound record confirms one way or the other, or an operator who has
    asked the executor releases it (``strategy_release_work_order``).
``settled``
    The executor's answer was observed. Its outcome is kept for good: elapsed
    time never authorizes ordering the issue again (#3398 requires newer
    input for that).
``not_sent``
    The send provably did not reach the executor: refused before any request
    was made, or refused by the peer host before a task was created.
``released``
    An operator released an order this agent could no longer confirm.

Orders sent by hand from a turn are as much in the executor's lane as those
``signal_dispatch`` sent. The outbound record is scanned for them, back as far
as ``morning_signal_config.work_order_window_days``, and each one found is
adopted into the journal, after which it is kept like any other. The window
bounds only how far back that scan looks for orders not yet journaled.

From the journal the ledger answers what the selection guards need:

* an issue with an order still open is in flight on the executor;
* an issue whose last order was answered ``blocked`` or ``clarifying``, was
  ``cancelled``, or was not executed at all (the task ended ``failed`` or
  ``canceled``), is waiting on new input, exactly as #3398 waits after a local
  Talon run that asked a question;
* the executor's lane is idle only when no order is open.

Workspace readiness has no separate read here. The executor checks its own
workspace before it claims, and an order it cannot run comes back not
executed, which withholds the issue until something new authorizes a retry.

When an order was answered is the moment this agent first observed the task
terminal: the outbound record's ``terminal_at`` once stamped, otherwise the
read that observed it. That is sender-owned evidence and never earlier than
the executor's answer, so "something newer than the run" is never judged
against a time the executor chose. It is journaled once and never moves.

A read that cannot be completed raises :class:`WorkOrdersUnreadable`: an
order nobody could read may be the very order on the issue about to be picked.
An order that was read but whose report cannot be parsed is not that: its
issue is known, so only that issue is withheld (see
:attr:`WorkOrder.unreadable`).
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from kestrel_sovereign.execution_custody import is_execution_control_error
from kestrel_sovereign.features.storage_access import (
    AgentIdentityUnavailable,
    resolve_feature_database,
    resolve_scoped_agent_did,
)

from .run_history import QUESTION_DISPOSITIONS, RunHistory, TalonRun
from .timestamps import parse_instant

logger = logging.getLogger(__name__)

#: The data artifact schema a work order carries, and the artifact's name.
WORK_ORDER_SCHEMA = "kestrel.work_order/v1"
WORK_ORDER_ARTIFACT = "work_order"
#: The schema of the JSON object the executor answers an order with.
WORK_REPORT_SCHEMA = "kestrel.work_report/v1"

#: ``morning_signal_config`` key naming the peer agent that runs this agent's
#: picks. Unset, ``signal_dispatch`` runs them locally as before.
DISPATCH_EXECUTOR_KEY = "dispatch_executor"

#: How far back the outbound record is scanned for orders not yet in the
#: journal (orders sent by hand). Once journaled, an order is kept whatever
#: its age. ``morning_signal_config.work_order_window_days`` overrides it.
DEFAULT_WORK_ORDER_WINDOW_DAYS = 30
WORK_ORDER_WINDOW_DAYS_KEY = "work_order_window_days"

#: The work order's ``context`` is bounded so it stays a brief, not a dump.
CONTEXT_WORD_LIMIT = 300

#: The bundled feature that owns A2A sends and the outbound record.
PEERS_FEATURE = "PeersFeature"

#: ``list_outbound_tasks`` clamps one read to this many rows.
OUTBOUND_READ_LIMIT = 1000

#: A2A task states (``kestrel_sdk.a2a.types.TaskState``) after which an order
#: has an answer. Anything else is still in the executor's hands.
COMPLETED_STATE = "completed"
NOT_EXECUTED_STATES = frozenset({"failed", "canceled", "cancelled"})

#: The dispositions a ``kestrel.work_report/v1`` may carry: Talon's processing
#: summary as the executor read it.
REPORT_DISPOSITIONS = frozenset(
    {"completed", "clarifying", "blocked", "failed", "cancelled"}
)
#: A Talon job the operator cancelled. The executor does nothing further with
#: its order, so neither does the dispatch until something new authorizes it.
CANCELLED_DISPOSITION = "cancelled"
#: The disposition an order gets when its task ended without the executor
#: running it: the executor refused it, or it was withdrawn.
ORDER_FAILED_DISPOSITION = "order_failed"
ORDER_CANCELED_DISPOSITION = "order_canceled"
NOT_EXECUTED_DISPOSITIONS = frozenset(
    {ORDER_FAILED_DISPOSITION, ORDER_CANCELED_DISPOSITION}
)
#: Outcomes after which sending the same order against unchanged input is the
#: same order again (#3398).
AWAITING_INPUT_DISPOSITIONS = (
    QUESTION_DISPOSITIONS
    | {CANCELLED_DISPOSITION}
    | NOT_EXECUTED_DISPOSITIONS
)

# ---------------------------------------------------------------------------
# The journal's vocabulary (see the module docstring)
# ---------------------------------------------------------------------------

JOURNAL_TABLE = "strategic_work_orders"

#: Who put the order in the journal: a ``signal_dispatch`` send, or the scan
#: that found an order sent by hand.
ORIGIN_DISPATCH = "dispatch"
ORIGIN_OBSERVED = "observed"
ORIGINS = (ORIGIN_DISPATCH, ORIGIN_OBSERVED)

STATE_RESERVED = "reserved"
STATE_SENDING = "sending"
STATE_SENT = "sent"
STATE_UNCONFIRMED = "unconfirmed"
STATE_SETTLED = "settled"
STATE_NOT_SENT = "not_sent"
STATE_RELEASED = "released"
# Ordered: the CHECK constraint below is generated from it.
JOURNAL_STATES = (
    STATE_RESERVED,
    STATE_SENDING,
    STATE_SENT,
    STATE_UNCONFIRMED,
    STATE_SETTLED,
    STATE_NOT_SENT,
    STATE_RELEASED,
)
#: States in which an order holds the executor's lane. For a dispatch row
#: these are the states the lane index admits only one of per executor.
OPEN_STATES = (STATE_RESERVED, STATE_SENDING, STATE_SENT, STATE_UNCONFIRMED)
#: States that carry no evidence about the issue: nothing reached the
#: executor, or an operator said the order no longer stands.
DISCARDED_STATES = frozenset({STATE_NOT_SENT, STATE_RELEASED})

#: How long a dispatch may hold a ``reserved`` or ``sending`` row before
#: another reader treats its holder as gone. Selection reads GitHub per
#: candidate and the send waits on approval policy, so it is generous; the
#: holder's own transitions are compare-and-set, so a holder that outlives
#: its lease loses the row rather than writing over the reader's verdict.
RESERVATION_LEASE = timedelta(minutes=30)

#: ``a2a_outbound_tasks.error`` values PeersFeature records only for a send
#: that cannot have created a task on the executor: refused before any request
#: was made (``signing_failed:``), or answered 404/403 (not in the directory,
#: not authorized) or 503 (unavailable) by the peer host. Every other failure
#: -- a timeout or connection drop after the request was written, a malformed
#: answer, a route that could not be retained after delivery -- may have
#: delivered the order, and is unconfirmed.
DEFINITE_NON_DELIVERY_ERRORS = frozenset({"peer_not_in_automatic_directory"})
DEFINITE_NON_DELIVERY_PREFIXES = ("peer_unavailable:", "signing_failed:")
#: The outbound record's state for a send that failed, and for a route the
#: executor accepted.
DISPATCH_FAILED_STATE = "dispatch_failed"
ROUTABLE_ROUTE_STATE = "routable"

_STATE_CHECK = "state IN ({})".format(", ".join(f"'{s}'" for s in JOURNAL_STATES))
_ORIGIN_CHECK = "origin IN ({})".format(", ".join(f"'{o}'" for o in ORIGINS))
_OPEN_STATE_LIST = ", ".join(f"'{s}'" for s in OPEN_STATES)

JOURNAL_DDL = f"""
    CREATE TABLE IF NOT EXISTS {JOURNAL_TABLE} (
        agent_id TEXT NOT NULL,
        order_key TEXT NOT NULL,
        order_id TEXT,
        executor TEXT NOT NULL,
        executor_key TEXT NOT NULL,
        origin TEXT NOT NULL CHECK ({_ORIGIN_CHECK}),
        state TEXT NOT NULL CHECK ({_STATE_CHECK}),
        repo TEXT,
        issue_number INTEGER,
        task_id TEXT,
        a2a_state TEXT,
        reserved_at TEXT NOT NULL,
        lease_expires_at TEXT,
        sent_at TEXT,
        answered_at TEXT,
        disposition TEXT,
        job_id TEXT,
        detail TEXT,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (agent_id, order_key)
    )
"""
#: The lane: at most one open ``signal_dispatch`` order per executor. Orders
#: found by the scan are not in it -- adopting one must never fail -- and a
#: dispatch reads them as busy before it reserves.
JOURNAL_LANE_INDEX = f"""
    CREATE UNIQUE INDEX IF NOT EXISTS uq_{JOURNAL_TABLE}_lane
    ON {JOURNAL_TABLE}(agent_id, executor_key)
    WHERE origin = '{ORIGIN_DISPATCH}' AND state IN ({_OPEN_STATE_LIST})
"""
JOURNAL_TASK_INDEX = f"""
    CREATE INDEX IF NOT EXISTS idx_{JOURNAL_TABLE}_task
    ON {JOURNAL_TABLE}(agent_id, task_id)
"""

_COLUMNS = (
    "order_key, order_id, executor, origin, state, repo, issue_number, "
    "task_id, a2a_state, reserved_at, lease_expires_at, sent_at, answered_at, "
    "disposition, job_id, detail"
)


class WorkOrdersUnreadable(Exception):
    """The ledger of orders sent to the executor could not be read whole.

    Raised rather than returning part of it: an order that could not be read
    may be the order on the issue about to be picked. Also raised when the
    journal cannot be written, since an order that is not journaled is an
    order the next dispatch cannot see.
    """


class InvalidDispatchExecutor(ValueError):
    """``morning_signal_config.dispatch_executor`` is set but names no agent."""


class WorkOrderNotReleasable(ValueError):
    """``strategy_release_work_order`` named no single open order."""


@dataclass(frozen=True)
class OrderedRun(TalonRun):
    """The outcome of one answered work order, as a run on its issue."""

    order_id: str
    task_id: str
    executor: str

    @property
    def awaits_new_input(self) -> bool:
        return self.disposition in AWAITING_INPUT_DISPOSITIONS

    @property
    def executed(self) -> bool:
        return self.disposition not in NOT_EXECUTED_DISPOSITIONS


@dataclass(frozen=True)
class JournalEntry:
    """One row of the journal."""

    order_key: str
    order_id: Optional[str]
    executor: str
    origin: str
    state: str
    repo: Optional[str]
    issue_number: Optional[int]
    task_id: Optional[str]
    a2a_state: Optional[str]
    reserved_at: datetime
    lease_expires_at: Optional[datetime]
    sent_at: Optional[datetime]
    answered_at: Optional[datetime]
    disposition: Optional[str]
    job_id: Optional[str]
    detail: Optional[str]

    def lease_expired(self, moment: datetime) -> bool:
        return self.lease_expires_at is None or self.lease_expires_at <= moment


@dataclass(frozen=True)
class WorkOrder:
    """One order in the journal, as the selection guards read it."""

    order_id: Optional[str]
    task_id: Optional[str]
    executor: str
    repo: Optional[str]
    issue_number: Optional[int]
    sent_at: datetime
    #: The journal state (see the module docstring).
    delivery: str
    #: The task's A2A state as the executor last reported it, when known.
    a2a_state: Optional[str] = None
    #: The answered order's outcome, when it could be read.
    run: Optional[OrderedRun] = None
    #: Why an answered order's report could not be read.
    unreadable: Optional[str] = None
    #: Why the order's delivery is unconfirmed, or how it was released.
    detail: Optional[str] = None

    @property
    def outstanding(self) -> bool:
        """Whether the order holds the executor's lane."""
        return self.delivery in OPEN_STATES

    @property
    def unconfirmed(self) -> bool:
        return self.delivery == STATE_UNCONFIRMED

    @property
    def state(self) -> str:
        """The A2A state of an accepted order, otherwise its journal state."""
        if self.delivery == STATE_SENT and self.a2a_state:
            return self.a2a_state
        return self.delivery

    def summary(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "a2a_task_id": self.task_id,
            "repo": self.repo,
            "issue_number": self.issue_number,
            "state": self.state,
            "delivery": self.delivery,
            "sent_at": self.sent_at.isoformat(),
            **({"detail": self.detail} if self.detail else {}),
        }


@dataclass(frozen=True)
class WorkOrderLedger:
    """Every order in the journal for ``executor``, newest first."""

    executor: str
    orders: Tuple[WorkOrder, ...] = ()

    def orders_on(self, repo: str, issue_number: int) -> List[WorkOrder]:
        """The orders on one issue, newest first. Repository names are
        case-insensitive."""
        key = repo.lower()
        return [
            order
            for order in self.orders
            if order.repo is not None
            and order.repo.lower() == key
            and order.issue_number == issue_number
        ]

    def outstanding(self) -> Tuple[WorkOrder, ...]:
        return tuple(order for order in self.orders if order.outstanding)

    @property
    def lane_idle(self) -> bool:
        """Whether the executor holds no open order from this agent."""
        return not self.outstanding()

    def lane(self) -> Dict[str, Any]:
        outstanding = self.outstanding()
        return {
            "executor": self.executor,
            "idle": not outstanding,
            "outstanding": [order.summary() for order in outstanding],
        }

    def run_history(self) -> RunHistory:
        """The most recent answered order per issue, as #3398's history."""
        return RunHistory.from_runs(
            order.run for order in self.orders if order.run is not None
        )


def dispatch_executor(config: Mapping[str, Any]) -> Optional[str]:
    """The executor ``morning_signal_config`` names, or ``None``.

    Raises :class:`InvalidDispatchExecutor` for a value that is set but is not
    a name: falling back to a local launch would be the very launch the
    setting exists to prevent.
    """
    value = config.get(DISPATCH_EXECUTOR_KEY)
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidDispatchExecutor(
            f"morning_signal_config.{DISPATCH_EXECUTOR_KEY} must be an agent "
            f"name, not {value!r}"
        )
    return value.strip() or None


def work_order_window(config: Mapping[str, Any]) -> timedelta:
    """``morning_signal_config.work_order_window_days``, or the default.

    Raises :class:`WorkOrdersUnreadable` for a value that is not a positive
    integer rather than guessing what the operator meant.
    """
    value = config.get(WORK_ORDER_WINDOW_DAYS_KEY)
    if value is None:
        return timedelta(days=DEFAULT_WORK_ORDER_WINDOW_DAYS)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise WorkOrdersUnreadable(
            f"morning_signal_config.{WORK_ORDER_WINDOW_DAYS_KEY} must be a "
            f"positive integer, not {value!r}"
        )
    return timedelta(days=value)


def build_work_order(
    issue: Mapping[str, Any], *, order_id: Optional[str] = None
) -> Dict[str, Any]:
    """The ``kestrel.work_order/v1`` payload for the issue selection picked.

    A fresh claim with no Talon overrides and no verify commands: the
    executor's host preferences decide the run, and it verifies what the diff
    touched.
    """
    number = issue["issue_number"]
    context = " ".join(
        str(part).strip()
        for part in (
            f"{issue.get('issue_title') or ''}.",
            f"Selected by signal_dispatch at priority {issue.get('priority')}.",
            issue.get("context") or "",
        )
        if str(part).strip()
    ).split()
    return {
        "schema": WORK_ORDER_SCHEMA,
        "order_id": order_id or f"wo-{number}-{uuid.uuid4().hex[:8]}",
        "repo": issue["repo"],
        "issue": number,
        "action": "claim",
        "talon": {},
        "verify": [],
        "context": " ".join(context[:CONTEXT_WORD_LIMIT]),
    }


def work_order_message(order: Mapping[str, Any], issue: Mapping[str, Any]) -> str:
    """The one-line task message an order travels with."""
    title = " ".join(str(issue.get("issue_title") or "").split())
    return (
        f"Work order {order['order_id']}: {order['action']} "
        f"{order['repo']}#{order['issue']}"
        + (f" ({title})" if title else "")
        + ". See the work_order artifact."
    )


def executor_key(executor: str) -> str:
    """The case-insensitive key an executor's lane is held under."""
    return executor.strip().lower()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: Optional[datetime]) -> Optional[str]:
    return moment.astimezone(timezone.utc).isoformat() if moment is not None else None


def _is_unique_violation(exc: Exception) -> bool:
    """A backend's unique-index refusal. SQLite says ``UNIQUE constraint
    failed``; PostgreSQL ``duplicate key value violates unique constraint``."""
    message = str(exc).lower()
    return "unique constraint" in message or "duplicate key value" in message


class WorkOrderJournal:
    """The durable journal of work orders, scoped to one agent.

    Every method raises :class:`WorkOrdersUnreadable` when the database does
    not answer; a transition that lost a compare-and-set returns ``False``.
    """

    def __init__(self, db: Any, agent_id: str):
        self._db = db
        self._agent_id = agent_id
        self._ready = False

    @property
    def agent_id(self) -> str:
        """The DID every row of this journal is scoped to."""
        return self._agent_id

    async def ensure(self) -> None:
        if self._ready:
            return
        await self._run("create the work-order journal", self._create)
        self._ready = True

    async def _create(self) -> None:
        await self._db.execute(JOURNAL_DDL)
        await self._db.execute(JOURNAL_LANE_INDEX)
        await self._db.execute(JOURNAL_TASK_INDEX)

    async def _run(self, action: str, operation, *args: Any) -> Any:
        try:
            return await operation(*args)
        except Exception as exc:  # noqa: BLE001 - storage boundary; reported, not swallowed
            if is_execution_control_error(exc):
                raise
            raise WorkOrdersUnreadable(f"could not {action}: {exc}") from exc

    async def entries(self, executor: Optional[str] = None) -> List[JournalEntry]:
        """Every row for ``executor`` (every executor when ``None``)."""
        await self.ensure()
        sql = f"SELECT {_COLUMNS} FROM {JOURNAL_TABLE} WHERE agent_id = ?"
        params: Tuple[Any, ...] = (self._agent_id,)
        if executor is not None:
            sql += " AND executor_key = ?"
            params += (executor_key(executor),)
        rows = await self._run("read the work-order journal", self._db.fetchall, sql, params)
        return [_entry(row) for row in rows]

    async def _update(
        self,
        order_key: str,
        from_states: Sequence[str],
        changes: Dict[str, Any],
        action: str,
        *,
        origin: Optional[str] = None,
    ) -> bool:
        await self.ensure()
        changes = {**changes, "updated_at": _iso(_utc_now())}
        assignments = ", ".join(f"{column} = ?" for column in changes)
        states = ", ".join("?" for _ in from_states)
        sql = (
            f"UPDATE {JOURNAL_TABLE} SET {assignments} "
            f"WHERE agent_id = ? AND order_key = ? AND state IN ({states})"
        )
        params = (*changes.values(), self._agent_id, order_key, *from_states)
        if origin is not None:
            sql += " AND origin = ?"
            params += (origin,)
        affected = await self._run(action, self._db.execute, sql, params)
        return _rowcount(affected) == 1

    async def reserve(
        self, executor: str, *, now: Optional[datetime] = None
    ) -> Optional[JournalEntry]:
        """Hold ``executor``'s lane for one dispatch, or ``None`` when another
        dispatch already holds it."""
        await self.ensure()
        moment = now or _utc_now()
        order_key = f"dispatch:{uuid.uuid4().hex}"
        try:
            await self._db.execute(
                f"INSERT INTO {JOURNAL_TABLE} (agent_id, order_key, executor, "
                "executor_key, origin, state, reserved_at, lease_expires_at, "
                "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self._agent_id, order_key, executor.strip(), executor_key(executor),
                    ORIGIN_DISPATCH, STATE_RESERVED, _iso(moment),
                    _iso(moment + RESERVATION_LEASE), _iso(moment),
                ),
            )
        except Exception as exc:  # noqa: BLE001 - storage boundary
            if is_execution_control_error(exc):
                raise
            if _is_unique_violation(exc):
                return None
            raise WorkOrdersUnreadable(
                f"could not reserve {executor}'s lane in the work-order journal: {exc}"
            ) from exc
        return JournalEntry(
            order_key=order_key, order_id=None, executor=executor.strip(),
            origin=ORIGIN_DISPATCH, state=STATE_RESERVED, repo=None,
            issue_number=None, task_id=None, a2a_state=None, reserved_at=moment,
            lease_expires_at=moment + RESERVATION_LEASE, sent_at=None,
            answered_at=None, disposition=None, job_id=None, detail=None,
        )

    async def withdraw(self, order_key: str) -> bool:
        """Remove a reservation that never became an order."""
        await self.ensure()
        affected = await self._run(
            "withdraw a lane reservation",
            self._db.execute,
            f"DELETE FROM {JOURNAL_TABLE} WHERE agent_id = ? AND order_key = ? "
            "AND state = ?",
            (self._agent_id, order_key, STATE_RESERVED),
        )
        return _rowcount(affected) == 1

    async def begin_send(
        self, order_key: str, *, order_id: str, repo: str, issue_number: int,
        now: Optional[datetime] = None,
    ) -> bool:
        """Name the order a reservation is about to send. ``False`` when the
        reservation lapsed and another reader withdrew it."""
        moment = now or _utc_now()
        return await self._update(
            order_key, (STATE_RESERVED,),
            {
                "state": STATE_SENDING, "order_id": order_id, "repo": repo,
                "issue_number": issue_number,
                "lease_expires_at": _iso(moment + RESERVATION_LEASE),
            },
            "record the order being sent",
        )

    async def record_sent(
        self, order_key: str, *, task_id: str, sent_at: Optional[datetime] = None
    ) -> bool:
        """The executor accepted the order's task."""
        return await self._update(
            order_key, (STATE_SENDING, STATE_UNCONFIRMED),
            {
                "state": STATE_SENT, "task_id": task_id,
                "sent_at": _iso(sent_at or _utc_now()), "lease_expires_at": None,
            },
            "record the sent order",
        )

    async def record_unconfirmed(
        self, order_key: str, *, detail: str, task_id: Optional[str] = None
    ) -> bool:
        """The send may have reached the executor; nothing can say whether."""
        changes: Dict[str, Any] = {
            "state": STATE_UNCONFIRMED, "detail": detail, "lease_expires_at": None,
        }
        if task_id:
            changes["task_id"] = task_id
        return await self._update(
            order_key, (STATE_SENDING,), changes, "record the unconfirmed order"
        )

    async def record_not_sent(self, order_key: str, *, detail: str) -> bool:
        """The send provably did not reach the executor."""
        return await self._update(
            order_key, (STATE_SENDING, STATE_UNCONFIRMED),
            {"state": STATE_NOT_SENT, "detail": detail, "lease_expires_at": None},
            "record the unsent order",
        )

    async def observe(self, order_key: str, *, a2a_state: str) -> bool:
        """Record the A2A state of a still-outstanding order."""
        return await self._update(
            order_key, (STATE_SENT,), {"a2a_state": a2a_state},
            "record the order's state",
        )

    async def settle(
        self, order_key: str, *, a2a_state: str, answered_at: datetime,
        disposition: Optional[str], job_id: Optional[str], detail: Optional[str],
    ) -> bool:
        """Record the executor's answer to an accepted order, once."""
        return await self._update(
            order_key, (STATE_SENT,),
            {
                "state": STATE_SETTLED, "a2a_state": a2a_state,
                "answered_at": _iso(answered_at), "disposition": disposition,
                "job_id": job_id, "detail": detail,
            },
            "record the order's answer",
        )

    async def adopt(
        self, *, executor: str, order_id: str, repo: str, issue_number: int,
        task_id: str, sent_at: datetime, a2a_state: str,
        answered_at: Optional[datetime] = None, disposition: Optional[str] = None,
        job_id: Optional[str] = None, detail: Optional[str] = None,
    ) -> None:
        """Journal an order sent by hand that the scan found."""
        await self.ensure()
        settled = answered_at is not None
        await self._run(
            "journal an order sent by hand",
            self._db.execute,
            f"INSERT INTO {JOURNAL_TABLE} (agent_id, order_key, order_id, executor, "
            "executor_key, origin, state, repo, issue_number, task_id, a2a_state, "
            "reserved_at, sent_at, answered_at, disposition, job_id, detail, "
            "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (agent_id, order_key) DO NOTHING",
            (
                self._agent_id, f"task:{task_id}", order_id, executor,
                executor_key(executor), ORIGIN_OBSERVED,
                STATE_SETTLED if settled else STATE_SENT, repo, issue_number,
                task_id, a2a_state, _iso(sent_at), _iso(sent_at), _iso(answered_at),
                disposition, job_id, detail, _iso(_utc_now()),
            ),
        )

    async def release(self, reference: str, *, reason: str) -> JournalEntry:
        """Release the one open order ``reference`` names (an order id or an
        A2A task id). Raises :class:`WorkOrderNotReleasable` when it names
        none, or more than one."""
        wanted = reference.strip()
        candidates = [
            entry
            for entry in await self.entries()
            if wanted and wanted in (entry.order_id, entry.task_id)
        ]
        open_ones = [entry for entry in candidates if entry.state in OPEN_STATES]
        if not open_ones:
            if candidates:
                raise WorkOrderNotReleasable(
                    f"work order {wanted} is {candidates[0].state}, not open; "
                    "there is nothing to release"
                )
            raise WorkOrderNotReleasable(f"no work order {wanted} is in the journal")
        if len(open_ones) > 1:
            raise WorkOrderNotReleasable(
                f"{len(open_ones)} open work orders match {wanted}; name one by "
                "its A2A task id: "
                + ", ".join(str(entry.task_id) for entry in open_ones)
            )
        [entry] = open_ones
        if entry.state == STATE_RESERVED:
            raise WorkOrderNotReleasable(
                f"work order {wanted} is still being selected by a running "
                "dispatch; it is withdrawn when that dispatch ends"
            )
        released = await self._update(
            entry.order_key, (entry.state,),
            {
                "state": STATE_RELEASED, "detail": f"released: {reason}",
                "lease_expires_at": None,
            },
            "release the work order",
        )
        if not released:
            raise WorkOrderNotReleasable(
                f"work order {wanted} changed while it was being released; "
                "read the lane again"
            )
        return entry


async def open_work_order_journal(agent: Any) -> WorkOrderJournal:
    """The journal in ``agent``'s own database, scoped to its DID.

    Raises :class:`WorkOrdersUnreadable` when the agent has no database or no
    DID to scope it by: without the journal an order cannot be kept.
    """
    db = resolve_feature_database(agent)
    if db is None:
        raise WorkOrdersUnreadable(
            "this agent has no database to keep the work-order journal in"
        )
    try:
        agent_id = resolve_scoped_agent_did(agent)
    except AgentIdentityUnavailable as exc:
        raise WorkOrdersUnreadable(
            f"the work-order journal cannot be scoped to this agent: {exc}"
        ) from exc
    journal = WorkOrderJournal(db, agent_id)
    await journal.ensure()
    return journal


async def read_work_orders(
    agent: Any,
    executor: str,
    *,
    journal: WorkOrderJournal,
    window: timedelta,
    now: Optional[datetime] = None,
) -> WorkOrderLedger:
    """Every work order ``agent`` sent ``executor``, brought up to date.

    Reconciles the journal before it answers: an attempt whose delivery was
    in doubt is resolved from the outbound record where that record can say,
    an accepted order is asked about, its answer is journaled once observed,
    and an order sent by hand within ``window`` is adopted.

    Raises :class:`WorkOrdersUnreadable` when the agent has no A2A transport
    to read through, its outbound record cannot be read or cannot be shown to
    cover the window, an open order cannot be read from the executor, or the
    journal does not answer.
    """
    peers = _peers(agent)
    moment = (now or _utc_now()).astimezone(timezone.utc)
    since = moment - window
    rows = await _outbound_rows(peers)
    if len(rows) >= OUTBOUND_READ_LIMIT:
        oldest = parse_instant(rows[-1].get("created_at"))
        if oldest is None or oldest >= since:
            raise WorkOrdersUnreadable(
                f"the outbound record holds more than {OUTBOUND_READ_LIMIT} "
                "tasks in the window, more than one read returns"
            )
    outbound = _tasks_to(rows, executor)
    by_session: Dict[str, List[Mapping[str, Any]]] = {}
    by_task: Dict[str, Mapping[str, Any]] = {}
    for row in outbound:
        by_session.setdefault(str(row.get("session_id") or ""), []).append(row)
        by_task.setdefault(row["task_id"], row)

    # An attempt with no accepted task id: the outbound record of the attempt
    # (found by its session id, which is the order id) says what became of it.
    for entry in await journal.entries(executor):
        if entry.origin == ORIGIN_DISPATCH and entry.state in (
            STATE_RESERVED, STATE_SENDING, STATE_UNCONFIRMED,
        ):
            await _reconcile_attempt(
                journal, entry, by_session.get(entry.order_id or "", []), moment
            )

    # An accepted order: the executor says whether it is answered. Asked
    # whatever the order's age; the window bounds only the scan below.
    entries = await journal.entries(executor)
    for entry in entries:
        if entry.state == STATE_SENT and entry.task_id:
            await _refresh_accepted(
                journal, peers, entry, by_task.get(entry.task_id), moment
            )

    # Orders sent by hand, not yet journaled.
    known_tasks = {entry.task_id for entry in entries if entry.task_id}
    attempts = {
        entry.order_id
        for entry in entries
        if entry.origin == ORIGIN_DISPATCH and entry.order_id
    }
    for row in outbound:
        task_id = row["task_id"]
        if task_id in known_tasks or str(row.get("session_id") or "") in attempts:
            continue
        sent_at = parse_instant(row.get("created_at"))
        if sent_at < since:
            continue
        if row.get("terminal_state") == DISPATCH_FAILED_STATE:
            # A hand-made send that failed: its sender saw the failure, and
            # its route cannot be read to learn what it ordered.
            continue
        await _adopt_if_order(journal, peers, row, task_id, sent_at, moment)

    orders = [
        _order_view(entry)
        for entry in await journal.entries(executor)
        if entry.state not in DISCARDED_STATES
    ]
    orders.sort(key=lambda order: order.sent_at, reverse=True)
    return WorkOrderLedger(executor=executor, orders=tuple(orders))


def _peers(agent: Any) -> Any:
    get_feature = getattr(agent, "get_feature", None)
    peers = get_feature(PEERS_FEATURE) if callable(get_feature) else None
    if peers is None:
        raise WorkOrdersUnreadable(
            f"{PEERS_FEATURE} is not enabled on this agent, so the work orders "
            "it sent cannot be read"
        )
    for method in ("list_outbound_a2a_tasks", "get_peer_task_result"):
        if not callable(getattr(peers, method, None)):
            raise WorkOrdersUnreadable(
                f"{PEERS_FEATURE} ({type(peers).__name__}) has no {method}()"
            )
    return peers


async def _outbound_rows(peers: Any) -> List[Any]:
    try:
        listing = await peers.list_outbound_a2a_tasks(limit=OUTBOUND_READ_LIMIT)
    except Exception as exc:  # noqa: BLE001 - provider boundary; reported, not swallowed
        if is_execution_control_error(exc):
            raise
        raise WorkOrdersUnreadable(
            f"reading the outbound A2A record failed: {exc}"
        ) from exc
    data = _ok_data(listing)
    if data is None:
        raise WorkOrdersUnreadable(
            f"the outbound A2A record could not be read: {_error_text(listing)}"
        )
    rows = data.get("rows")
    if not isinstance(rows, list):
        raise WorkOrdersUnreadable("the outbound A2A record returned no list of rows")
    return rows


def _tasks_to(rows: List[Any], executor: str) -> List[Mapping[str, Any]]:
    """The outbound task rows to ``executor``, each with a task id and a send
    time, newest first."""
    wanted = executor_key(executor)
    tasks = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise WorkOrdersUnreadable(
                f"the outbound record returned a row that is not a record "
                f"({type(row).__name__})"
            )
        if str(row.get("recipient") or "").strip().lower() != wanted:
            continue
        if str(row.get("verb") or "").strip().lower() != "task":
            continue
        task_id = row.get("task_id")
        if (
            parse_instant(row.get("created_at")) is None
            or not isinstance(task_id, str)
            or not task_id.strip()
        ):
            raise WorkOrdersUnreadable(
                f"an outbound task to {executor} has no readable task id or "
                f"send time ({task_id!r}, {row.get('created_at')!r})"
            )
        tasks.append(row)
    return tasks


def definitely_not_delivered(row: Mapping[str, Any]) -> bool:
    """Whether an outbound row records a send that cannot have created a task
    on the executor (see :data:`DEFINITE_NON_DELIVERY_ERRORS`)."""
    if row.get("terminal_state") != DISPATCH_FAILED_STATE:
        return False
    error = str(row.get("error") or "")
    return error in DEFINITE_NON_DELIVERY_ERRORS or error.startswith(
        DEFINITE_NON_DELIVERY_PREFIXES
    )


def attempt_outcome(
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[Optional[str], Optional[Mapping[str, Any]]]:
    """What the outbound record says became of one send attempt.

    ``(STATE_SENT, row)`` when the executor accepted its task,
    ``(STATE_NOT_SENT, row)`` when the send cannot have reached it,
    ``(STATE_UNCONFIRMED, row)`` when it may have, and ``(None, None)`` when
    the record holds no row for the attempt.
    """
    if not rows:
        return None, None
    for row in rows:
        if (
            row.get("route_state") == ROUTABLE_ROUTE_STATE
            and row.get("terminal_state") != DISPATCH_FAILED_STATE
        ):
            return STATE_SENT, row
    if all(definitely_not_delivered(row) for row in rows):
        return STATE_NOT_SENT, rows[0]
    uncertain = next(row for row in rows if not definitely_not_delivered(row))
    return STATE_UNCONFIRMED, uncertain


async def read_send_attempt(
    agent: Any, executor: str, order_id: str
) -> Tuple[Optional[str], Optional[Mapping[str, Any]]]:
    """:func:`attempt_outcome` for the send of ``order_id``, whose A2A session
    id is its order id. Raises :class:`WorkOrdersUnreadable` when the outbound
    record cannot be read."""
    rows = _tasks_to(await _outbound_rows(_peers(agent)), executor)
    return attempt_outcome(
        [row for row in rows if str(row.get("session_id") or "") == order_id]
    )


def describe_attempt_row(row: Mapping[str, Any]) -> str:
    """A short account of an outbound row for an unconfirmed order."""
    error = row.get("error")
    if error:
        return f"the transport recorded {error!r} for task {row.get('task_id')}"
    return (
        f"task {row.get('task_id')}'s route was never confirmed "
        f"(route {row.get('route_state')!r})"
    )


async def _reconcile_attempt(
    journal: WorkOrderJournal,
    entry: JournalEntry,
    rows: List[Mapping[str, Any]],
    moment: datetime,
) -> None:
    """Resolve a dispatch row with no accepted task id, as far as the outbound
    record can."""
    if entry.state == STATE_RESERVED:
        if entry.lease_expired(moment):
            # Its dispatch stopped before choosing an issue: never an order.
            await journal.withdraw(entry.order_key)
        return
    outcome, row = attempt_outcome(rows)
    if outcome == STATE_SENT:
        await journal.record_sent(
            entry.order_key,
            task_id=row["task_id"],
            sent_at=parse_instant(row.get("created_at")),
        )
    elif outcome == STATE_NOT_SENT:
        await journal.record_not_sent(
            entry.order_key,
            detail=f"the transport refused it before delivery ({row.get('error')})",
        )
    elif entry.state == STATE_SENDING and entry.lease_expired(moment):
        await journal.record_unconfirmed(
            entry.order_key,
            detail=(
                describe_attempt_row(row)
                if row is not None
                else "the dispatch sending it stopped before it recorded the outcome"
            ),
            task_id=row.get("task_id") if row is not None else None,
        )


async def _refresh_accepted(
    journal: WorkOrderJournal,
    peers: Any,
    entry: JournalEntry,
    row: Optional[Mapping[str, Any]],
    moment: datetime,
) -> None:
    """Ask the executor about one accepted order; journal its answer."""
    recipient = str((row or {}).get("recipient") or entry.executor).strip()
    data = await _fetch_task(peers, recipient, entry.task_id)
    state = str(data.get("state") or "unknown").strip().lower()
    if state != COMPLETED_STATE and state not in NOT_EXECUTED_STATES:
        if state != entry.a2a_state:
            await journal.observe(entry.order_key, a2a_state=state)
        return
    disposition, job_id, unreadable = _answer(
        data, state,
        order_id=entry.order_id or entry.task_id,
        repo=entry.repo or "",
        issue_number=entry.issue_number or 0,
    )
    await journal.settle(
        entry.order_key,
        a2a_state=state,
        answered_at=_answered_at(row, state, entry.sent_at or entry.reserved_at, moment),
        disposition=disposition,
        job_id=job_id,
        detail=unreadable,
    )


async def _adopt_if_order(
    journal: WorkOrderJournal,
    peers: Any,
    row: Mapping[str, Any],
    task_id: str,
    sent_at: datetime,
    moment: datetime,
) -> None:
    """Journal the task ``row`` sent, if it is a work order."""
    recipient = str(row.get("recipient")).strip()
    data = await _fetch_task(peers, recipient, task_id)
    order = _order_artifact(data.get("artifacts"))
    if order is None:
        return
    repo = order.get("repo")
    issue_number = _issue_number(order.get("issue"))
    order_id = order.get("order_id")
    if not isinstance(repo, str) or not repo.strip() or issue_number is None:
        # The executor refuses an order that names no issue, so no run on any
        # issue can follow it. Said, not silently dropped.
        logger.warning(
            "Work order task %s to %s names no issue it identifies "
            "(repo=%r, issue=%r); it is not counted against any issue",
            task_id, recipient, repo, order.get("issue"),
        )
        return
    order_id = order_id.strip() if isinstance(order_id, str) and order_id.strip() else task_id
    state = str(data.get("state") or "unknown").strip().lower()
    answered = state == COMPLETED_STATE or state in NOT_EXECUTED_STATES
    disposition = job_id = unreadable = None
    if answered:
        disposition, job_id, unreadable = _answer(
            data, state, order_id=order_id, repo=repo.strip(), issue_number=issue_number
        )
    await journal.adopt(
        executor=recipient,
        order_id=order_id,
        repo=repo.strip(),
        issue_number=issue_number,
        task_id=task_id,
        sent_at=sent_at,
        a2a_state=state,
        answered_at=_answered_at(row, state, sent_at, moment) if answered else None,
        disposition=disposition,
        job_id=job_id,
        detail=unreadable,
    )


async def _fetch_task(peers: Any, recipient: str, task_id: str) -> Mapping[str, Any]:
    try:
        fetched = await peers.get_peer_task_result(recipient, task_id)
    except Exception as exc:  # noqa: BLE001 - provider boundary; reported, not swallowed
        if is_execution_control_error(exc):
            raise
        raise WorkOrdersUnreadable(
            f"reading task {task_id} from {recipient} failed: {exc}"
        ) from exc
    data = _ok_data(fetched)
    if data is None:
        raise WorkOrdersUnreadable(
            f"task {task_id} could not be read from {recipient}: "
            f"{_error_text(fetched)}"
        )
    return data


def _answer(
    data: Mapping[str, Any], state: str, *, order_id: str, repo: str, issue_number: int
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """``(disposition, talon_job_id, unreadable)`` for an answered order."""
    if state in NOT_EXECUTED_STATES:
        disposition = (
            ORDER_FAILED_DISPOSITION if state == "failed" else ORDER_CANCELED_DISPOSITION
        )
        return disposition, None, None
    try:
        report = parse_work_report(
            data.get("reply_text"),
            order_id=order_id,
            repo=repo,
            issue_number=issue_number,
        )
    except ValueError as exc:
        return None, None, str(exc)
    talon_job_id = report.get("talon_job_id")
    job_id = talon_job_id.strip() if isinstance(talon_job_id, str) else None
    return report["disposition"], job_id or None, None


def _answered_at(
    row: Optional[Mapping[str, Any]], state: str, sent_at: datetime, moment: datetime
) -> datetime:
    # The record's stamp dates this answer only when it records this answer:
    # a stamp of another state (a sender-side deadline expiry) predates it.
    stamped = parse_instant((row or {}).get("terminal_at"))
    if _canonical_state((row or {}).get("terminal_state")) != _canonical_state(state):
        stamped = None
    return max(stamped or moment, sent_at)


def _order_view(entry: JournalEntry) -> WorkOrder:
    sent_at = entry.sent_at or entry.reserved_at
    run = None
    unreadable = None
    if entry.state == STATE_SETTLED:
        if entry.disposition is None:
            unreadable = entry.detail or "the executor's answer could not be read"
        elif entry.repo and entry.issue_number is not None:
            run = OrderedRun(
                job_id=entry.job_id or "",
                repo=entry.repo,
                issue_number=entry.issue_number,
                disposition=entry.disposition,
                completed_at=entry.answered_at or sent_at,
                order_id=entry.order_id or entry.task_id or entry.order_key,
                task_id=entry.task_id or "",
                executor=entry.executor,
            )
    return WorkOrder(
        order_id=entry.order_id,
        task_id=entry.task_id,
        executor=entry.executor,
        repo=entry.repo,
        issue_number=entry.issue_number,
        sent_at=sent_at,
        delivery=entry.state,
        a2a_state=entry.a2a_state,
        run=run,
        unreadable=unreadable,
        detail=entry.detail if entry.state != STATE_SETTLED else None,
    )


def _entry(row: Sequence[Any]) -> JournalEntry:
    (
        order_key, order_id, executor, origin, state, repo, issue_number, task_id,
        a2a_state, reserved_at, lease_expires_at, sent_at, answered_at,
        disposition, job_id, detail,
    ) = row
    reserved = parse_instant(reserved_at)
    if reserved is None:
        raise WorkOrdersUnreadable(
            f"work-order journal row {order_key!r} has no readable reservation time"
        )
    return JournalEntry(
        order_key=order_key,
        order_id=order_id,
        executor=executor,
        origin=origin,
        state=state,
        repo=repo,
        issue_number=int(issue_number) if issue_number is not None else None,
        task_id=task_id,
        a2a_state=a2a_state,
        reserved_at=reserved,
        lease_expires_at=parse_instant(lease_expires_at),
        sent_at=parse_instant(sent_at),
        answered_at=parse_instant(answered_at),
        disposition=disposition,
        job_id=job_id,
        detail=detail,
    )


def _rowcount(affected: Any) -> int:
    if isinstance(affected, int):
        return affected
    return int(getattr(affected, "rowcount", 0) or 0)


def parse_work_report(
    text: Any, *, order_id: str, repo: str, issue_number: int
) -> Dict[str, Any]:
    """The ``kestrel.work_report/v1`` object an answered order carries.

    Raises ``ValueError`` naming what is wrong when the answer is not one: not
    JSON, another schema, no known disposition, or a report about another
    order or issue than the one it answers.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("the executor's answer is empty")
    body = text.strip()
    if body.startswith("```"):
        lines = body.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            body = "\n".join(lines[1:-1])
    try:
        report = json.loads(body)
    except (ValueError, RecursionError) as exc:
        raise ValueError(f"the executor's answer is not a JSON object ({exc})") from exc
    if not isinstance(report, dict):
        raise ValueError("the executor's answer is not a JSON object")
    if report.get("schema") != WORK_REPORT_SCHEMA:
        raise ValueError(
            f"the executor's answer is not a {WORK_REPORT_SCHEMA} "
            f"(schema {report.get('schema')!r})"
        )
    disposition = report.get("disposition")
    disposition = disposition.strip().lower() if isinstance(disposition, str) else None
    if disposition not in REPORT_DISPOSITIONS:
        raise ValueError(
            f"the report's disposition {report.get('disposition')!r} is not one "
            f"of {', '.join(sorted(REPORT_DISPOSITIONS))}"
        )
    reported_order = report.get("order_id")
    if reported_order is not None and reported_order != order_id:
        raise ValueError(
            f"the report names order {reported_order!r}, not {order_id!r}"
        )
    reported_repo = report.get("repo")
    if reported_repo is not None and (
        not isinstance(reported_repo, str) or reported_repo.strip().lower() != repo.lower()
    ):
        raise ValueError(f"the report names repository {reported_repo!r}, not {repo!r}")
    reported_issue = report.get("issue")
    if reported_issue is not None and _issue_number(reported_issue) != issue_number:
        raise ValueError(f"the report names issue {reported_issue!r}, not {issue_number}")
    return {**report, "disposition": disposition}


def _order_artifact(artifacts: Any) -> Optional[Mapping[str, Any]]:
    """The ``kestrel.work_order/v1`` data a task's artifacts carry, if any."""
    if not isinstance(artifacts, list):
        return None
    for artifact in artifacts:
        if not isinstance(artifact, Mapping) or artifact.get("name") != WORK_ORDER_ARTIFACT:
            continue
        for part in artifact.get("parts") or ():
            data = part.get("data") if isinstance(part, Mapping) else None
            if isinstance(data, Mapping) and data.get("schema") == WORK_ORDER_SCHEMA:
                return data
    return None


def _canonical_state(value: Any) -> str:
    state = str(value or "").strip().lower()
    return "canceled" if state == "cancelled" else state


def _issue_number(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        text = value.strip().lstrip("#")
        if text.isdigit() and int(text) > 0:
            return int(text)
    return None


def _ok_data(result: Any) -> Optional[Mapping[str, Any]]:
    """A successful tool result's data, or ``None`` when it did not succeed."""
    status = getattr(result, "status", None)
    if getattr(status, "value", status) != "ok":
        return None
    data = getattr(result, "data", None)
    return data if isinstance(data, Mapping) else None


def _error_text(result: Any) -> str:
    error = getattr(result, "error", None)
    return str(error) if error else "no reason given"
