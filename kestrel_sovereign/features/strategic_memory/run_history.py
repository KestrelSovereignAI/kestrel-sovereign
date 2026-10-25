"""What Talon's job registry says about each issue's most recent run (#3398).

``signal_dispatch`` sent #3093 to Talon on 09-16, 09-23 and 09-24. Every run
blocked at iteration 1 with no diff and no PR, each saying the same thing:
this epic has nothing to implement here. Selection only asked GitHub whether
the issue was open, unlabelled and free of an open PR, so a label cleared to
mean "acknowledged" re-armed the same run. The registry already knew how the
last run ended and when; this module is how selection reads it.

Core cannot import kestrel-feature-talon. It reads the registry through the
provider Talon registers for the ``talon`` wait kind on
``agent.wait_registry``, and only through one operation of it:

``finished_runs()``
    Returns ``{"complete": bool, "runs": [run, ...], "reason": str}`` where
    each run is ``{"job_id", "repo", "issue", "disposition",
    "completed_at"}``: one entry per terminal job, ``completed_at`` the
    ISO-8601 moment Talon terminalized it. ``complete`` is ``True`` only when
    every persisted job was read; an absent registry is complete and empty,
    while an unreadable, oversized, corrupt or partly parsed one is not, and
    the optional ``reason`` says why. It must be read-only: no reaping, no
    pushing of preserved work, no registry write.

Two properties of that operation are load-bearing, and the wait provider's
other methods have neither. ``active_handles()`` reloads the registry and
returns every job it holds, but a registry it cannot read logs and returns an
empty list, so "no previous runs" and "could not read the runs" look alike.
That is the #3093 failure again. ``poll()`` reaps jobs, pushes preserved work
to the remote and rewrites the registry, which is right for the reconciler's
tick and wrong for a read that ``signal_dispatch(mode='suggest')`` makes
while previewing. A provider without ``finished_runs()`` therefore has an
unconfirmed history, never an empty one, and neither of the other two is
called here.

``finished_runs()`` first shipped in kestrel-feature-talon 0.2.10, and core
declares no dependency on feature packages (``pyproject.toml`` keeps them out
of its optional extras). :data:`FINISHED_RUNS_REQUIREMENT` is therefore where
that floor is declared: an older provider is refused with
:class:`TalonProviderOutdated`, which names the requirement rather than
reporting an unreadable registry.

Only the ``talon`` kind is read. A sibling provider such as ``A2AWaitable``
spreads peer-returned data into what it reports, and a peer must not be able
to stand down dispatch of an issue by naming it in a payload.

The executor's work reports (#3838)
-----------------------------------

A host can name one agent its only Talon dispatcher (``[talon.policy]
dispatcher``, kestrel-feature-talon#57). A queue owner that is not that agent
still selects, but the executor runs the work, so the queue owner's own
registry holds none of the runs that matter. :func:`read_executor_run_history`
reads the executor's answers instead.

``signal_dispatch`` sends each issue as an A2A task whose message carries a
:data:`WORK_ORDER_SCHEMA` data part. The executor answers the task once, when
it has the outcome: ``completed`` with a :data:`WORK_REPORT_SCHEMA` JSON object
whose ``disposition`` is Talon's, or ``failed`` when it could not execute the
order at all -- no Talon workspace for the repository, an issue it would not
take. That refusal is the executor's workspace readiness: the queue owner
cannot ask another agent's Talon, and sending the same order against
unchanged input gets the same refusal, so it waits for new input as a run
that asked a question does.

The read goes through this agent's own outbound A2A audit, filtered to the
executor, through the Peers tools ``list_outbound_a2a_tasks`` and
``get_peer_task_result``, and walks all of it page by page: the guard an answer
sets lasts until new input lifts it, not until newer dispatches crowd it out of
one listing page. Every run is keyed by the issue the order named, never by the
issue a report names. A dispatch the audit cannot yet settle -- reserved but
not submitted, delivery uncertain, a task the executor has not answered, a task
that cannot be read -- is an :class:`OutstandingOrder`: nothing says it is not
an order for the very issue selection would pick next.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
from collections.abc import AsyncIterator, Mapping, MutableMapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple

from kestrel_sdk.native_a2a.types import TaskState
from kestrel_sdk.tools.result import ToolResult

from kestrel_sovereign.a2a.lifecycle import TERMINAL_STATES, task_state_label
from kestrel_sovereign.a2a.stores.unified.outbound_store import DispatchState
from kestrel_sovereign.execution_custody import is_execution_control_error

from .timestamps import parse_instant

logger = logging.getLogger(__name__)

#: The wait kind kestrel-feature-talon registers (``TalonWaitable.kind``).
TALON_WAIT_KIND = "talon"

#: The provider operation run history is read from (see the module docstring).
FINISHED_RUNS_METHOD = "finished_runs"

#: The first kestrel-feature-talon release whose ``TalonWaitable`` has
#: ``finished_runs()`` (kestrel-feature-talon #37). Earlier releases cannot
#: report a complete history, so every read through them is refused.
FINISHED_RUNS_REQUIREMENT = "kestrel-feature-talon>=0.2.10"

#: kestrel-feature-talon ``run_disposition`` values for a run that stopped to
#: ask a question rather than open a PR: ``BLOCKED`` (Talon also labels the
#: issue ``agent-blocked``) and ``CLARIFYING`` (``agent-clarifying``). Core
#: cannot import that package, so this is a copy of the vocabulary;
#: ``test_the_retry_vocabulary_matches_talons`` pins it.
QUESTION_DISPOSITIONS = frozenset({"blocked", "clarifying"})

#: The data part a queue owner sends its executor as an A2A task, and the JSON
#: object the executor answers that task with (#3838). The fleet's contract:
#: an order carries ``order_id``, ``repo``, ``issue``, ``action``,
#: ``resume_branch``, ``talon``, ``verify`` and ``context``; a report carries
#: ``order_id``, ``repo``, ``issue``, ``talon_job_id`` and ``disposition``
#: among its fields.
WORK_ORDER_SCHEMA = "kestrel.work_order/v1"
WORK_REPORT_SCHEMA = "kestrel.work_report/v1"

#: The disposition of an order the executor answered ``failed`` or
#: ``rejected``: it could not execute it, so no Talon run exists, and the same
#: order against unchanged input is refused again.
ORDER_REFUSED = "refused"

#: The Peers tools the executor's reports are read through, and the one whose
#: dispatches can carry a work order (``send_a2a_task`` is the only peer tool
#: that sends data parts; ``ask_agent`` and messages send text only).
LIST_OUTBOUND_TOOL = "list_outbound_a2a_tasks"
READ_PEER_TASK_TOOL = "get_peer_task_result"
SEND_TASK_TOOL = "send_a2a_task"

#: Rows per page of the outbound audit listing. The read pages to the end of
#: the audit, so this bounds one call, not what is read.
OUTBOUND_PAGE_SIZE = 200

_TERMINAL_TASK_STATES = frozenset(task_state_label(state) for state in TERMINAL_STATES)
_REFUSED_TASK_STATES = frozenset(
    task_state_label(state)
    for state in (TaskState.TASK_STATE_FAILED, TaskState.TASK_STATE_REJECTED)
)
_CANCELED_TASK_STATE = task_state_label(TaskState.TASK_STATE_CANCELED)
#: How a native task read names the sender's own message in its history.
_SENDER_ROLE = "ROLE_USER"

_ISSUE_NUMBER = re.compile(r"#?[0-9]+")


class RunHistoryUnreadable(Exception):
    """Talon's run history is not known to be complete.

    Raised rather than returning a partial history: a job that could not be
    read may be exactly the run that asked the question, and an issue whose
    last run is unknown is not an issue whose last run is known to be fine.
    """

    #: The package requirement that would make the history readable, when the
    #: cause is a provider too old to report it rather than a registry read.
    requirement: Optional[str] = None


class TalonProviderOutdated(RunHistoryUnreadable):
    """The ``talon`` wait provider has no ``finished_runs()`` to read from.

    Not a fault in Talon's registry, and not one a retry clears: every read
    through this provider is unconfirmed until kestrel-feature-talon is
    upgraded, so the refusal names the release that fixes it.
    """

    requirement = FINISHED_RUNS_REQUIREMENT


@dataclass(frozen=True)
class TalonRun:
    """One finished Talon run against one issue.

    A run read from an executor's answer to a work order (#3838) also names
    the ``executor``, the ``order_id`` and the A2A ``dispatch_id`` the order
    went out as. ``job_id`` is then the Talon job the report names, or the
    ``dispatch_id`` when it names none, and ``completed_at`` is the moment the
    executor answered, which is no earlier than the run's end -- or, for an
    answer that carries no time, the moment this agent first read it, which
    is no earlier than the answer.
    """

    job_id: str
    repo: str
    issue_number: int
    disposition: str
    completed_at: datetime
    executor: Optional[str] = None
    order_id: Optional[str] = None
    dispatch_id: Optional[str] = None

    @property
    def ended_with_question(self) -> bool:
        """Whether the run stopped to ask something instead of opening a PR."""
        return self.disposition in QUESTION_DISPOSITIONS

    @property
    def awaits_new_input(self) -> bool:
        """Whether dispatching the issue again, unchanged, repeats this outcome.

        A run that asked a question has said what it needs; an order the
        executor refused is refused again.
        """
        return self.ended_with_question or self.disposition == ORDER_REFUSED


@dataclass(frozen=True)
class OutstandingOrder:
    """A dispatch to the executor that is not settled yet (#3838).

    ``delivery_state`` is the dispatch's state in this agent's outbound audit
    and ``task_state`` the executor's task state last read, ``None`` when no
    task could be read. ``order_id``, ``repo`` and ``issue_number`` are
    ``None`` when the dispatch's content could not be read, and for an order
    that does not identify them. ``detail`` says why it is not settled.
    """

    dispatch_id: str
    delivery_state: str
    task_state: Optional[str] = None
    order_id: Optional[str] = None
    repo: Optional[str] = None
    issue_number: Optional[int] = None
    detail: Optional[str] = None


@dataclass(frozen=True)
class RunHistory:
    """The most recent finished Talon run per issue.

    Keyed by ``(repo.lower(), issue_number)``: GitHub repository names are
    case-insensitive, and the ledger and Talon's registry need not agree on
    case.

    ``executor`` names the agent whose work reports the history was read
    from, or is ``None`` for this agent's own Talon registry, and
    ``outstanding`` lists that executor's unsettled dispatches, newest first.
    ``unplaced`` describes each answer that would hold an issue back but
    comes from a task that does not say which issue: it can hold none back,
    so it is reported rather than applied.
    """

    runs: Dict[Tuple[str, int], TalonRun] = field(default_factory=dict)
    executor: Optional[str] = None
    outstanding: Tuple[OutstandingOrder, ...] = ()
    unplaced: Tuple[str, ...] = ()

    @classmethod
    def from_runs(cls, runs: Iterable[TalonRun]) -> "RunHistory":
        latest: Dict[Tuple[str, int], TalonRun] = {}
        for run in runs:
            key = (run.repo.lower(), run.issue_number)
            current = latest.get(key)
            if current is None or run.completed_at > current.completed_at:
                latest[key] = run
        return cls(latest)

    def latest(self, repo: str, issue_number: int) -> Optional[TalonRun]:
        return self.runs.get((repo.lower(), issue_number))


async def read_run_history(agent: Any) -> RunHistory:
    """The most recent finished Talon run per issue on ``agent``.

    An agent with no ``talon`` wait provider has no Talon registry, so no
    issue on it has a last run: the history is empty. A registered provider
    with no ``finished_runs()`` raises :class:`TalonProviderOutdated`; one
    whose ``finished_runs()`` fails or does not report a complete read raises
    :class:`RunHistoryUnreadable`.
    """
    registry = getattr(agent, "wait_registry", None)
    provider = registry.get(TALON_WAIT_KIND) if registry is not None else None
    if provider is None:
        return RunHistory()
    read = getattr(provider, FINISHED_RUNS_METHOD, None)
    if not callable(read):
        raise TalonProviderOutdated(
            f"the {TALON_WAIT_KIND!r} wait provider "
            f"({type(provider).__name__}) has no read-only "
            f"{FINISHED_RUNS_METHOD}() that reports whether it read every job; "
            f"reading Talon's run history requires {FINISHED_RUNS_REQUIREMENT}, "
            "the first release with it -- upgrade kestrel-feature-talon and "
            "restart the host"
        )
    try:
        report = read()
        if inspect.isawaitable(report):
            report = await report
    except Exception as exc:  # noqa: BLE001 - provider boundary; reported, not swallowed
        raise RunHistoryUnreadable(f"reading Talon's finished runs failed: {exc}") from exc
    if not isinstance(report, Mapping):
        raise RunHistoryUnreadable(
            f"Talon's {FINISHED_RUNS_METHOD}() returned {type(report).__name__}, "
            "not a report"
        )
    if report.get("complete") is not True:
        reason = report.get("reason")
        raise RunHistoryUnreadable(
            "Talon could not read its whole job registry"
            + (f": {reason}" if isinstance(reason, str) and reason.strip() else "")
        )
    entries = report.get("runs")
    if not isinstance(entries, list):
        raise RunHistoryUnreadable(
            f"Talon's {FINISHED_RUNS_METHOD}() reported no list of runs"
        )
    runs = []
    for entry in entries:
        run = _issue_run(entry)
        if run is not None:
            runs.append(run)
    return RunHistory.from_runs(runs)


def _issue_run(entry: Any) -> Optional[TalonRun]:
    """The run on an issue ``entry`` records, or ``None`` when it is not one.

    A job that names no issue (a batch, an ``iterate`` on a PR) is not a run
    on an issue. One that names an issue it does not identify, or a run with
    no completion time, cannot be placed: it may be the very run that asked
    the question, so the history is unreadable.
    """
    if not isinstance(entry, Mapping):
        raise RunHistoryUnreadable(
            f"Talon reported a run that is not a record ({type(entry).__name__})"
        )
    job_id = entry.get("job_id")
    if not isinstance(job_id, str) or not job_id.strip():
        raise RunHistoryUnreadable("Talon reported a run with no job id")
    issue = entry.get("issue")
    if issue is None or issue == "":
        return None
    repo = entry.get("repo")
    issue_number = _issue_number(issue)
    if not isinstance(repo, str) or not repo.strip() or issue_number is None:
        raise RunHistoryUnreadable(
            f"Talon job {job_id} names issue {issue!r} in repository {repo!r}, "
            "which does not identify an issue"
        )
    completed_at = parse_instant(entry.get("completed_at"))
    if completed_at is None:
        # Talon stamps every job it terminalizes. Without the stamp nothing
        # can say whether a later comment or edit came after this run.
        raise RunHistoryUnreadable(
            f"Talon job {job_id} on {repo}#{issue_number} finished with no "
            "readable completion time"
        )
    disposition = entry.get("disposition")
    return TalonRun(
        job_id=job_id,
        repo=repo.strip(),
        issue_number=issue_number,
        disposition=disposition if isinstance(disposition, str) else "unknown",
        completed_at=completed_at,
    )


def _issue_number(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and _ISSUE_NUMBER.fullmatch(value.strip()):
        number = int(value.strip().lstrip("#"))
        return number if number > 0 else None
    return None


#: Calls one tool by name through the agent's governed dispatch
#: (``agent.execute_named_tool``, bound to a session and an audit source) and
#: returns what the tool returned.
NamedToolCall = Callable[[str, Dict[str, Any]], Awaitable[Any]]


def tool_payload(result: Any) -> Dict[str, Any]:
    """A named tool's result as a plain mapping.

    A ``ToolResult`` becomes its ``to_dict()``; a legacy dict, or the
    ``{"success": False, "error": ...}`` envelope a permission hook returns,
    is copied; anything else is wrapped as ``{"result": ...}``.
    """
    if isinstance(result, ToolResult):
        return result.to_dict()
    if isinstance(result, Mapping):
        return dict(result)
    return {"result": str(result)}


def tool_succeeded(payload: Mapping[str, Any]) -> bool:
    """Whether a :func:`tool_payload` reports that the tool did what was asked."""
    return payload.get("status") == "ok" or payload.get("success") is True


@dataclass(frozen=True)
class WorkOrder:
    """What one work order names.

    ``repo`` and ``issue_number`` are ``None`` when the order does not
    identify them; ``raw_repo`` and ``raw_issue`` keep what it said.
    ``readable`` is ``False`` when the task's history shows no message from
    this agent at all: it may have carried an order, for any issue.
    """

    order_id: Optional[str]
    repo: Optional[str]
    issue_number: Optional[int]
    raw_repo: Any = None
    raw_issue: Any = None
    readable: bool = True

    @property
    def key(self) -> Optional[Tuple[str, int]]:
        """The :class:`RunHistory` key of the issue the order names."""
        if self.repo is None or self.issue_number is None:
            return None
        return (self.repo.lower(), self.issue_number)


@dataclass(frozen=True)
class SettledDispatch:
    """What one dispatch to the executor says that holds no lane.

    ``order`` is ``None`` for a dispatch that carried no work order, or that
    never reached the executor. ``run`` is the run an answered order records,
    or ``None`` when its answer records none. ``unplaced`` says why an answer
    that would hold an issue back cannot be placed on one. ``final`` is ``True``
    when nothing about the reading can change -- the dispatch never reached
    the executor, or its task is terminal -- so
    :func:`read_executor_run_history` keeps it in the ``settled`` mapping it
    is given and never reads that dispatch from the executor again.
    """

    order: Optional[WorkOrder] = None
    run: Optional[TalonRun] = None
    unplaced: Optional[str] = None
    final: bool = True


async def read_executor_run_history(
    executor: str,
    call_tool: NamedToolCall,
    settled: Optional[MutableMapping[str, SettledDispatch]] = None,
) -> RunHistory:
    """The most recent finished run per issue, as ``executor``'s answers say.

    Walks this agent's whole audit of ``send_a2a_task`` dispatches to
    ``executor``, newest first and a page of :data:`OUTBOUND_PAGE_SIZE` at a
    time, and reads each one's task. A task whose sender message carries no
    :data:`WORK_ORDER_SCHEMA` object (as a data part, or as a text part that
    is that JSON) is not a work order. The newest order
    on an issue decides it: an answered order is a :class:`TalonRun` dated by
    the moment the executor answered, and an unsettled dispatch of any kind is
    an :class:`OutstandingOrder` in ``outstanding``.

    ``settled`` keeps what each dispatch said once nothing about it can
    change. A caller that passes the same mapping every time reads each
    finished dispatch from the executor once, however long the audit grows.

    Raises :class:`RunHistoryUnreadable` when the audit cannot be listed to
    its end, and when a terminal dispatch's task cannot be read: its answer
    may be the one that should hold selection back. A
    ``ToolNotRegisteredError`` from ``call_tool`` propagates, because no
    retry clears it.
    """
    settled = {} if settled is None else settled
    runs: Dict[Tuple[str, int], TalonRun] = {}
    decided: set = set()
    outstanding: List[OutstandingOrder] = []
    unplaced: List[str] = []
    async for row in _outbound_rows(executor, call_tool):
        if (
            row.get("verb") != "task"
            or row.get("dispatch_tool") != SEND_TASK_TOOL
            or row.get("parent_dispatch_id")
        ):
            # A continuation is more input to its root's task, not a new order.
            continue
        dispatch_id = row["dispatch_id"]
        reading = settled.get(dispatch_id)
        if reading is None:
            reading = await _read_dispatch(executor, row, call_tool)
        if isinstance(reading, OutstandingOrder):
            outstanding.append(reading)
            if reading.repo is not None and reading.issue_number is not None:
                decided.add((reading.repo.lower(), reading.issue_number))
            continue
        if reading.final:
            settled[dispatch_id] = reading
        if reading.unplaced is not None:
            logger.warning("signal_dispatch cannot place an answer: %s", reading.unplaced)
            unplaced.append(reading.unplaced)
        key = reading.order.key if reading.order is not None else None
        if key is None or key in decided:
            continue
        decided.add(key)
        if reading.run is not None:
            runs[key] = reading.run
    return RunHistory(
        runs,
        executor=executor,
        outstanding=tuple(outstanding),
        unplaced=tuple(unplaced),
    )


async def _outbound_rows(
    executor: str, call_tool: NamedToolCall
) -> AsyncIterator[Mapping[str, Any]]:
    """Every row of this agent's audit of dispatches to ``executor``, newest first.

    Pages by ``before`` cursor until a page comes back short. A row without a
    dispatch id, or one listed twice, means the listing cannot be walked to
    its end.
    """
    before: Optional[str] = None
    listed: set = set()
    while True:
        args: Dict[str, Any] = {"limit": OUTBOUND_PAGE_SIZE, "recipient": executor}
        if before is not None:
            args["before"] = before
        listing = tool_payload(await _invoke(call_tool, LIST_OUTBOUND_TOOL, args))
        if not tool_succeeded(listing):
            raise RunHistoryUnreadable(
                f"listing the A2A dispatches sent to {executor} failed: "
                f"{listing.get('error') or 'no reason given'}"
            )
        data = listing.get("data")
        rows = data.get("rows") if isinstance(data, Mapping) else None
        if not isinstance(rows, list):
            raise RunHistoryUnreadable(
                f"{LIST_OUTBOUND_TOOL} reported no list of the dispatches sent "
                f"to {executor}"
            )
        for row in rows:
            dispatch_id = row.get("dispatch_id") if isinstance(row, Mapping) else None
            if not isinstance(dispatch_id, str) or not dispatch_id.strip():
                raise RunHistoryUnreadable(
                    f"{LIST_OUTBOUND_TOOL} reported a dispatch to {executor} "
                    "with no dispatch id"
                )
            if dispatch_id in listed:
                raise RunHistoryUnreadable(
                    f"{LIST_OUTBOUND_TOOL} listed dispatch {dispatch_id} to "
                    f"{executor} twice, so the audit cannot be read to its end"
                )
            listed.add(dispatch_id)
            yield row
        if len(rows) < OUTBOUND_PAGE_SIZE:
            return
        before = rows[-1]["dispatch_id"]


async def _invoke(call_tool: NamedToolCall, name: str, args: Dict[str, Any]) -> Any:
    from kestrel_sovereign.agent.orchestrator_engine import ToolNotRegisteredError

    try:
        return await call_tool(name, args)
    except ToolNotRegisteredError:
        raise
    except Exception as exc:  # noqa: BLE001 - tool boundary; reported, not swallowed
        if is_execution_control_error(exc):
            raise
        raise RunHistoryUnreadable(f"{name} failed: {exc}") from exc


async def _read_dispatch(
    executor: str, row: Mapping[str, Any], call_tool: NamedToolCall
) -> "SettledDispatch | OutstandingOrder":
    """What one root ``send_a2a_task`` dispatch to ``executor`` says now."""
    dispatch_id = row["dispatch_id"]
    delivery = str(row.get("state"))
    if delivery == DispatchState.NOT_SENT.value:
        # The sender proved nothing reached the executor.
        return SettledDispatch()
    if delivery == DispatchState.RESERVED.value:
        # Reserved by a send that has not submitted it yet. That send may
        # still deliver it, and its content cannot be read until it does.
        return OutstandingOrder(
            dispatch_id,
            delivery,
            detail="reserved by a send that has not submitted it yet",
        )
    if (
        delivery == DispatchState.COMPLETE.value
        and row.get("response_kind") == "message"
    ):
        # Answered at once with a message: no task holds an order.
        return SettledDispatch()
    read = tool_payload(
        await _invoke(call_tool, READ_PEER_TASK_TOOL, {"dispatch_id": dispatch_id})
    )
    data = read.get("data")
    data = data if isinstance(data, Mapping) else {}
    if not tool_succeeded(read):
        reason = read.get("error") or "no reason given"
        if delivery == DispatchState.COMPLETE.value:
            # Its answer is final and may be the one that holds an issue back.
            raise RunHistoryUnreadable(
                f"reading dispatch {dispatch_id} to {executor} failed: {reason}"
            )
        return OutstandingOrder(
            dispatch_id,
            delivery,
            task_state=_text(row.get("remote_task_state")),
            detail=f"its task could not be read: {reason}",
        )
    task = data.get("task")
    if not isinstance(task, Mapping):
        if isinstance(data.get("message"), Mapping):
            # Answered with a message, not a task.
            return SettledDispatch()
        return OutstandingOrder(
            dispatch_id,
            delivery,
            task_state=_text(row.get("remote_task_state")),
            detail="its read returned neither a task nor a message",
        )
    order = _work_order(task)
    task_state = _text(data.get("state"))
    terminal = task_state in _TERMINAL_TASK_STATES
    if order is None:
        # Not an order. Kept only once its task is terminal, so a read that
        # showed less of the history than a later one cannot hide an order.
        return SettledDispatch(final=terminal)
    if not terminal:
        return OutstandingOrder(
            dispatch_id,
            delivery,
            task_state=task_state,
            order_id=order.order_id,
            repo=order.repo,
            issue_number=order.issue_number,
            detail=(
                f"{executor} has not answered it"
                if order.readable
                else f"its task shows no message from this agent, so it may "
                f"be an order {executor} has not answered"
            ),
        )
    run, unplaced = _answered_order(
        executor,
        dispatch_id,
        order,
        task_state,
        task,
        data,
        observed_at=datetime.now(timezone.utc),
    )
    return SettledDispatch(order=order, run=run, unplaced=unplaced)


def _work_order(task: Mapping[str, Any]) -> Optional[WorkOrder]:
    """The work order the sender's own message in ``task`` carries, or ``None``.

    Read from the task's history, where the recipient keeps the message it
    was sent, never from its artifacts or status, which are its own output.
    The order is a :data:`WORK_ORDER_SCHEMA` data part, or a text part that is
    that JSON object. ``None`` means the history shows this agent's message
    and it carries no order. A history with no message from this agent cannot
    say, so it is an unreadable :class:`WorkOrder`.
    """
    history = task.get("history")
    sender_seen = False
    for message in history if isinstance(history, list) else ():
        if not isinstance(message, Mapping) or message.get("role") != _SENDER_ROLE:
            continue
        sender_seen = True
        parts = message.get("parts")
        for part in parts if isinstance(parts, list) else ():
            body = _order_body(part)
            if body is None:
                continue
            repo = body.get("repo")
            return WorkOrder(
                order_id=_text(body.get("order_id")),
                repo=_text(repo),
                issue_number=_order_issue_number(body.get("issue")),
                raw_repo=repo,
                raw_issue=body.get("issue"),
            )
    if sender_seen:
        return None
    return WorkOrder(order_id=None, repo=None, issue_number=None, readable=False)


def _order_body(part: Any) -> Optional[Mapping[str, Any]]:
    """The :data:`WORK_ORDER_SCHEMA` object one message part carries, or ``None``."""
    if not isinstance(part, Mapping):
        return None
    body = part.get("data")
    if body is None:
        body = _json_object(part.get("text"))
    if isinstance(body, Mapping) and body.get("schema") == WORK_ORDER_SCHEMA:
        return body
    return None


def _order_issue_number(value: Any) -> Optional[int]:
    # A2A carries a data part as a protobuf Value, whose numbers are doubles:
    # issue 3838 comes back as 3838.0.
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return _issue_number(value)


def _answered_order(
    executor: str,
    dispatch_id: str,
    order: WorkOrder,
    task_state: str,
    task: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    observed_at: datetime,
) -> Tuple[Optional[TalonRun], Optional[str]]:
    """The run an answered order records, and why it cannot be placed.

    ``failed`` or ``rejected`` is a refusal whatever the answer says: the
    executor could not execute the order. ``canceled`` is a cancelled run. A
    ``completed`` answer's disposition is its report's, or ``unknown`` when
    the answer is not a :data:`WORK_REPORT_SCHEMA` report.

    The run is dated by the answer's status time, or by ``observed_at`` --
    when this agent read it, never earlier than the answer -- when it carries
    none: a later date only makes fewer comments and edits count as new
    input. The run is ``None`` when the order does not identify its issue;
    the reason is then set if the answer would hold an issue back, because
    no issue can be held back by it, and the caller reports it instead.
    """
    report = _work_report(result.get("reply_text"))
    if task_state in _REFUSED_TASK_STATES:
        disposition = ORDER_REFUSED
    elif task_state == _CANCELED_TASK_STATE:
        disposition = "cancelled"
    else:
        reported = report.get("disposition")
        disposition = (
            reported.strip().lower()
            if isinstance(reported, str) and reported.strip()
            else "unknown"
        )
        if not report:
            logger.warning(
                "%s answered work order %s completed without a %s report; "
                "its outcome is unknown",
                executor, order.order_id or dispatch_id, WORK_REPORT_SCHEMA,
            )
    holds_back = disposition in QUESTION_DISPOSITIONS or disposition == ORDER_REFUSED
    label = order.order_id or f"dispatch {dispatch_id}"
    if order.key is None:
        if not holds_back:
            return None, None
        if not order.readable:
            return None, (
                f"{executor} answered dispatch {dispatch_id} {disposition}, but "
                "its task shows no message from this agent, so which issue it "
                "holds back is unknown"
            )
        return None, (
            f"{executor} answered work order {label} {disposition}, but the "
            f"order names issue {order.raw_issue!r} in repository "
            f"{order.raw_repo!r}, which does not identify an issue"
        )
    status = task.get("status")
    completed_at = parse_instant(
        status.get("timestamp") if isinstance(status, Mapping) else None
    ) or observed_at
    return TalonRun(
        job_id=_text(report.get("talon_job_id")) or dispatch_id,
        repo=order.repo,
        issue_number=order.issue_number,
        disposition=disposition,
        completed_at=completed_at,
        executor=executor,
        order_id=order.order_id,
        dispatch_id=dispatch_id,
    ), None


def _work_report(text: Any) -> Dict[str, Any]:
    """The :data:`WORK_REPORT_SCHEMA` object ``text`` holds, or ``{}``."""
    report = _json_object(text)
    if report is not None and report.get("schema") == WORK_REPORT_SCHEMA:
        return dict(report)
    return {}


def _json_object(text: Any) -> Optional[Mapping[str, Any]]:
    """The JSON object ``text`` is, or ``None``."""
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        value = json.loads(text)
    except (ValueError, RecursionError):
        # Python 3.11's decoder raises RecursionError on deep nesting.
        return None
    return value if isinstance(value, Mapping) else None


def _text(value: Any) -> Optional[str]:
    return value.strip() if isinstance(value, str) and value.strip() else None
