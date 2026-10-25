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

The executor's work reports (#3758)
-----------------------------------

A host can name one agent its only Talon dispatcher (kestrel-feature-talon
#57). A queue owner that is not that agent still selects, but the executor
runs the work, so the queue owner's own registry is empty: it holds none of
the runs that matter. :func:`read_executor_run_history` reads the executor's
answers instead. ``signal_dispatch`` sends each issue as an A2A task carrying
a :data:`WORK_ORDER_SCHEMA` data part. The executor answers the task once,
when it has the outcome: ``completed`` with a :data:`WORK_REPORT_SCHEMA` JSON
whose ``disposition`` is Talon's, or ``failed`` when it could not run the
order at all (the issue was not ``agent-ready``, its repository had no Talon
workspace, the sender was not one it takes orders from). That refusal is the
executor's workspace readiness: the queue owner cannot ask the executor's
Talon directly.

Only the sender's own outbound A2A audit, filtered to the executor, is read,
through the Peers feature's ``list_outbound_a2a_tasks`` and
``get_peer_task_result``, and all of it: the guard an answer sets lasts until
new input lifts it, not until newer dispatches crowd the answer out of a
listing. The executor is the agent the operator configured to
run Talon, so its report is the run's outcome, as Talon's registry is on the
agent that runs it. Each run is still keyed by the issue the order named,
never by the issue a report names.
"""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import AsyncIterator, Mapping, MutableMapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple

from kestrel_sdk.native_a2a.types import TaskState

from kestrel_sovereign.a2a.lifecycle import TERMINAL_STATES, task_state_label
from kestrel_sovereign.a2a.stores.unified.outbound_store import DispatchState
from kestrel_sovereign.execution_custody import is_execution_control_error

from .timestamps import parse_instant

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
#: the executor answers that task with (#3758). The fleet's contract: the
#: order carries ``order_id``, ``repo``, ``issue``, ``action``, ``talon``,
#: ``verify`` and ``context``; the report carries ``order_id``, ``repo``,
#: ``issue``, ``talon_job_id`` and ``disposition`` among its fields.
WORK_ORDER_SCHEMA = "kestrel.work_order/v1"
WORK_REPORT_SCHEMA = "kestrel.work_report/v1"

#: The disposition of an order the executor answered ``failed`` or
#: ``rejected``: it could not run it, so no Talon run exists. Sending the same
#: order against unchanged input gets the same refusal, so it waits for new
#: input the way a run that asked a question does.
ORDER_REFUSED = "refused"

#: The disposition of an order whose task was cancelled before a report.
ORDER_CANCELLED = "cancelled"

#: How many dispatches to the executor one page of the audit listing holds.
#: The read pages to the end of the audit: an order guards its issue however
#: old it is, and an unanswered one holds the lane however many dispatches
#: came after it. A listing that stopped at a page would forget both.
WORK_ORDER_PAGE_SIZE = 200

#: The Peers feature tools the read goes through.
LIST_OUTBOUND_TOOL = "list_outbound_a2a_tasks"
READ_PEER_TASK_TOOL = "get_peer_task_result"

#: ``OutstandingOrder.state`` for a dispatch whose delivery is uncertain: no
#: task can be read, so nothing says it is not an order the executor holds.
DELIVERY_UNCERTAIN = "delivery_uncertain"

#: Dispatch states in which nothing reached the executor: the send was never
#: submitted, or the sender proved it was not delivered.
_UNDELIVERED = frozenset({DispatchState.RESERVED.value, DispatchState.NOT_SENT.value})
_TERMINAL_TASK_STATES = frozenset(task_state_label(state) for state in TERMINAL_STATES)
_REFUSED_TASK_STATES = frozenset(
    task_state_label(state)
    for state in (TaskState.TASK_STATE_FAILED, TaskState.TASK_STATE_REJECTED)
)
_CANCELLED_TASK_STATE = task_state_label(TaskState.TASK_STATE_CANCELED)
#: How the native task read names the sender's own message in a history.
_USER_ROLE = "ROLE_USER"

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

    A run read from an executor's work report (#3758) also names the
    ``executor``, the ``order_id`` and the A2A ``dispatch_id`` the order went
    out as; ``job_id`` is then the Talon job the report names, or the
    ``dispatch_id`` when it names none. ``completed_at`` is the moment the
    executor answered the order, which is no earlier than the run's end.
    """

    job_id: str
    repo: str
    issue_number: int
    disposition: str
    completed_at: datetime
    order_id: Optional[str] = None
    executor: Optional[str] = None
    dispatch_id: Optional[str] = None

    @property
    def ended_with_question(self) -> bool:
        """Whether the run stopped to ask something instead of opening a PR."""
        return self.disposition in QUESTION_DISPOSITIONS

    @property
    def awaits_new_input(self) -> bool:
        """Whether dispatching the issue again, unchanged, repeats this outcome.

        A run that asked a question has said what it needs; an order the
        executor refused will be refused again.
        """
        return self.ended_with_question or self.disposition == ORDER_REFUSED


@dataclass(frozen=True)
class OutstandingOrder:
    """A dispatch to the executor that it has not answered yet.

    ``state`` is the task state last read, or :data:`DELIVERY_UNCERTAIN` when
    no task could be read at all. ``repo`` and ``issue_number`` are ``None``
    when the order did not name an issue they identify, and for an uncertain
    delivery, whose content cannot be read back.
    """

    dispatch_id: str
    state: str
    order_id: Optional[str] = None
    repo: Optional[str] = None
    issue_number: Optional[int] = None


@dataclass(frozen=True)
class RunHistory:
    """The most recent finished Talon run per issue.

    Keyed by ``(repo.lower(), issue_number)``: GitHub repository names are
    case-insensitive, and the ledger and Talon's registry need not agree on
    case.

    ``executor`` names the agent whose work reports the history was read
    from, or is ``None`` for this agent's own Talon registry. ``outstanding``
    lists the orders that executor has not answered, newest first, and
    ``answered`` the dispatch id of every order it has, whether or not the
    answer guards an issue.
    """

    runs: Dict[Tuple[str, int], TalonRun] = field(default_factory=dict)
    executor: Optional[str] = None
    outstanding: Tuple[OutstandingOrder, ...] = ()
    answered: Tuple[str, ...] = ()

    @classmethod
    def from_runs(
        cls,
        runs: Iterable[TalonRun],
        *,
        executor: Optional[str] = None,
        outstanding: Iterable[OutstandingOrder] = (),
        answered: Iterable[str] = (),
    ) -> "RunHistory":
        latest: Dict[Tuple[str, int], TalonRun] = {}
        for run in runs:
            key = (run.repo.lower(), run.issue_number)
            current = latest.get(key)
            if current is None or run.completed_at > current.completed_at:
                latest[key] = run
        return cls(
            latest,
            executor=executor,
            outstanding=tuple(outstanding),
            answered=tuple(answered),
        )

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
#: (``agent.execute_named_tool``, bound to a session and source) and returns
#: what the tool returned.
NamedToolCall = Callable[[str, Dict[str, Any]], Awaitable[Any]]


def tool_payload(result: Any) -> Dict[str, Any]:
    """A named tool's result as a plain mapping.

    A ``ToolResult`` becomes its ``to_dict()``; a legacy dict, or the
    ``{"success": False, "error": ...}`` envelope a permission hook returns,
    is copied; anything else is wrapped as ``{"result": ...}``.
    """
    from kestrel_sdk.tools.result import ToolResult

    if isinstance(result, ToolResult):
        return result.to_dict()
    if isinstance(result, Mapping):
        return dict(result)
    return {"result": str(result)}


def tool_succeeded(payload: Mapping[str, Any]) -> bool:
    """Whether a :func:`tool_payload` reports that the tool did what was asked."""
    return payload.get("status") == "ok" or payload.get("success") is True


@dataclass(frozen=True)
class _Order:
    """What one work order names. ``repo``/``issue_number`` are ``None`` when
    the order does not identify them; ``raw_*`` keep what it said."""

    order_id: Optional[str]
    repo: Optional[str]
    issue_number: Optional[int]
    raw_repo: Any
    raw_issue: Any


@dataclass(frozen=True)
class SettledDispatch:
    """What one dispatch to the executor said, once nothing about it can change.

    The sender's message in a task is fixed when it is sent, and a terminal
    task's answer is final, so :func:`read_executor_run_history` reads such a
    dispatch from the executor once and keeps this, by dispatch id, in the
    ``settled`` mapping it is given. ``task_id`` is ``None`` when the executor
    answered with a message instead of a task. ``order`` says whether the
    task carried a work order, and ``run`` is the run its answer records, or
    ``None`` when the answer guards nothing.
    """

    task_id: Optional[str]
    order: bool = False
    run: Optional[TalonRun] = None


async def read_executor_run_history(
    executor: str,
    call_tool: NamedToolCall,
    settled: Optional[MutableMapping[str, SettledDispatch]] = None,
) -> RunHistory:
    """The most recent finished run per issue, as ``executor``'s reports say.

    Walks this agent's whole audit of A2A dispatches to ``executor``, a page
    of :data:`WORK_ORDER_PAGE_SIZE` at a time, and reads each one's task. A
    task whose sender message carries no :data:`WORK_ORDER_SCHEMA` data part
    is not a work order and is ignored, as is a dispatch the audit proves
    never reached the executor. An answered order is a :class:`TalonRun`,
    dated by the moment the executor answered it. An unanswered one is an
    :class:`OutstandingOrder`, and so is a dispatch whose delivery is
    uncertain: nothing says it is not an order the executor holds.

    ``settled`` keeps what each dispatch said once nothing about it can
    change (:class:`SettledDispatch`). A caller that passes the same mapping
    every time reads each finished order from the executor once, not once per
    read, however long the audit grows.

    Raises :class:`RunHistoryUnreadable` when the audit or a task cannot be
    read, when an answered order has no readable answer time, and when an
    order whose answer withholds its issue does not identify that issue: any
    of them may be the order still waiting on an answer. A
    ``ToolNotRegisteredError`` from ``call_tool`` propagates, because no
    retry clears it: the Peers feature is not enabled.
    """
    settled = {} if settled is None else settled
    runs: List[TalonRun] = []
    outstanding: List[OutstandingOrder] = []
    answered: List[str] = []
    # A continuation is its own dispatch of the same peer task: read once.
    seen_tasks: set = set()
    async for row in _audit_rows(executor, call_tool):
        if row.get("verb") != "task" or str(row.get("state")) in _UNDELIVERED:
            continue
        dispatch_id = row["dispatch_id"]
        entry = settled.get(dispatch_id)
        if entry is None:
            read = tool_payload(
                await _invoke(call_tool, READ_PEER_TASK_TOOL, {"dispatch_id": dispatch_id})
            )
            result = read.get("data")
            result = result if isinstance(result, Mapping) else {}
            if not tool_succeeded(read):
                if result.get("delivery_state") == DispatchState.UNKNOWN.value:
                    outstanding.append(OutstandingOrder(dispatch_id, DELIVERY_UNCERTAIN))
                    continue
                raise RunHistoryUnreadable(
                    f"reading dispatch {dispatch_id} to {executor} failed: "
                    f"{read.get('error') or 'no reason given'}"
                )
            task = result.get("task")
            if not isinstance(task, Mapping):
                # Answered with a message, not a task: nothing is left running.
                entry = SettledDispatch(task_id=None)
            else:
                task_id = str(task.get("id"))
                order = _work_order(task)
                state = result.get("state")
                state = state if isinstance(state, str) else ""
                if order is None:
                    entry = SettledDispatch(task_id=task_id)
                elif state not in _TERMINAL_TASK_STATES:
                    if task_id not in seen_tasks:
                        seen_tasks.add(task_id)
                        outstanding.append(
                            OutstandingOrder(
                                dispatch_id,
                                state,
                                order_id=order.order_id,
                                repo=order.repo,
                                issue_number=order.issue_number,
                            )
                        )
                    continue
                else:
                    entry = SettledDispatch(
                        task_id=task_id,
                        order=True,
                        run=_answered_order(
                            executor, dispatch_id, order, state, task, result
                        ),
                    )
            settled[dispatch_id] = entry
        if entry.task_id is not None:
            if entry.task_id in seen_tasks:
                continue
            seen_tasks.add(entry.task_id)
        if entry.order:
            answered.append(dispatch_id)
            if entry.run is not None:
                runs.append(entry.run)
    return RunHistory.from_runs(
        runs, executor=executor, outstanding=outstanding, answered=answered
    )


async def _audit_rows(
    executor: str, call_tool: NamedToolCall
) -> AsyncIterator[Mapping[str, Any]]:
    """Every row of this agent's audit of dispatches to ``executor``, newest first.

    Pages until a page comes back short. A row without a dispatch id, or one
    listed twice, means the listing cannot be walked to its end.
    """
    before: Optional[str] = None
    listed: set = set()
    while True:
        args: Dict[str, Any] = {"limit": WORK_ORDER_PAGE_SIZE, "recipient": executor}
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
                f"{LIST_OUTBOUND_TOOL} reported no list of the dispatches sent to {executor}"
            )
        for row in rows:
            if not isinstance(row, Mapping):
                raise RunHistoryUnreadable(
                    f"{LIST_OUTBOUND_TOOL} reported a dispatch to {executor} that is "
                    f"not a record ({type(row).__name__})"
                )
            dispatch_id = row.get("dispatch_id")
            if not isinstance(dispatch_id, str) or not dispatch_id.strip():
                raise RunHistoryUnreadable(
                    f"{LIST_OUTBOUND_TOOL} reported a dispatch to {executor} with no "
                    "dispatch id"
                )
            if dispatch_id in listed:
                raise RunHistoryUnreadable(
                    f"{LIST_OUTBOUND_TOOL} listed dispatch {dispatch_id} to {executor} "
                    "twice, so the audit cannot be read to its end"
                )
            listed.add(dispatch_id)
            yield row
        if len(rows) < WORK_ORDER_PAGE_SIZE:
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


def _work_order(task: Mapping[str, Any]) -> Optional[_Order]:
    """The work order the sender's own message in ``task`` carries, or ``None``.

    Read from the task's history, where the recipient keeps the message it
    was sent, never from its artifacts, which are the recipient's output.
    """
    history = task.get("history")
    if not isinstance(history, list):
        return None
    for message in history:
        if not isinstance(message, Mapping) or message.get("role") != _USER_ROLE:
            continue
        parts = message.get("parts")
        for part in parts if isinstance(parts, list) else ():
            body = part.get("data") if isinstance(part, Mapping) else None
            if not isinstance(body, Mapping) or body.get("schema") != WORK_ORDER_SCHEMA:
                continue
            repo = body.get("repo")
            order_id = body.get("order_id")
            return _Order(
                order_id=order_id.strip()
                if isinstance(order_id, str) and order_id.strip()
                else None,
                repo=repo.strip() if isinstance(repo, str) and repo.strip() else None,
                issue_number=_order_issue_number(body.get("issue")),
                raw_repo=repo,
                raw_issue=body.get("issue"),
            )
    return None


def _order_issue_number(value: Any) -> Optional[int]:
    # A2A carries a data part as a protobuf Value, whose numbers are doubles:
    # issue 3758 comes back as 3758.0.
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return _issue_number(value)


def _answered_order(
    executor: str,
    dispatch_id: str,
    order: _Order,
    state: str,
    task: Mapping[str, Any],
    result: Mapping[str, Any],
) -> Optional[TalonRun]:
    """The run an answered order records, or ``None`` when it guards nothing.

    ``failed``/``rejected`` is a refusal whatever the answer says: the
    executor could not run the order. A ``completed`` answer's disposition is
    the report's, or ``unknown`` when the answer is not a report.
    """
    if state in _REFUSED_TASK_STATES:
        disposition, job_id = ORDER_REFUSED, dispatch_id
    elif state == _CANCELLED_TASK_STATE:
        disposition, job_id = ORDER_CANCELLED, dispatch_id
    else:
        report = _work_report(result.get("reply_text"))
        reported = report.get("disposition")
        disposition = (
            reported.strip() if isinstance(reported, str) and reported.strip() else "unknown"
        )
        talon_job = report.get("talon_job_id")
        job_id = (
            talon_job.strip()
            if isinstance(talon_job, str) and talon_job.strip()
            else dispatch_id
        )
    label = order.order_id or f"dispatch {dispatch_id}"
    if order.repo is None or order.issue_number is None:
        if disposition in QUESTION_DISPOSITIONS or disposition == ORDER_REFUSED:
            raise RunHistoryUnreadable(
                f"{executor} answered work order {label} {disposition}, but the "
                f"order names issue {order.raw_issue!r} in repository "
                f"{order.raw_repo!r}, which does not identify an issue"
            )
        return None
    status = task.get("status")
    completed_at = parse_instant(
        status.get("timestamp") if isinstance(status, Mapping) else None
    )
    if completed_at is None:
        # Without it nothing can say whether a later comment or edit came
        # after this answer.
        raise RunHistoryUnreadable(
            f"{executor} answered work order {label} on "
            f"{order.repo}#{order.issue_number} with no readable answer time"
        )
    return TalonRun(
        job_id=job_id,
        repo=order.repo,
        issue_number=order.issue_number,
        disposition=disposition,
        completed_at=completed_at,
        order_id=order.order_id,
        executor=executor,
        dispatch_id=dispatch_id,
    )


def _work_report(text: Any) -> Dict[str, Any]:
    """The :data:`WORK_REPORT_SCHEMA` object ``text`` holds, or ``{}``."""
    if not isinstance(text, str) or not text.strip():
        return {}
    try:
        report = json.loads(text)
    except (ValueError, RecursionError):
        # Python 3.11's decoder raises RecursionError on deep nesting.
        return {}
    if isinstance(report, Mapping) and report.get("schema") == WORK_REPORT_SCHEMA:
        return dict(report)
    return {}
