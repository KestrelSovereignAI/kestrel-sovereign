"""Deadlines and usable-route probes for tests that call live LLM routes (#3269).

An integration test that drives a real agent through a live route can wait
forever: a route that accepts the connection and never answers, a daemon busy
with someone else's generation, or a tool call parked on the approval queue
all leave the coroutine waiting at 0% CPU. Nothing in the test bounded that,
so the integration tier stopped there until an outer ``timeout`` killed it
with no test named.

Two tools close that:

* :func:`bounded_live_call` puts a deadline on one call. When it fires, the
  test fails naming itself, the command or prompt, the routes the call would
  try, and the await chain it was parked in.
* :func:`require_answering_ollama`, :func:`require_answering_discovery` and
  :func:`require_answering_chat_route` decide whether to run from the call the
  test needs rather than from a daemon's liveness: work that does not finish
  within a few seconds gives a skip naming where it was parked, not a hang.

A test's calls and probes share one :class:`LiveCallBudget`, cut from the
test's own pytest-timeout. Each gets the smaller of its own deadline and what
the timeout has left once setup and earlier calls are paid for. A slow first
call therefore shortens the next call's deadline, instead of letting the
timeout fire first and fail the test without naming the command or route.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Iterable, Iterator, TypeVar

import httpx
import pytest

T = TypeVar("T")

#: The longest one live-route call may take before the test fails naming it.
#: A call gets less when its test's :class:`LiveCallBudget` has less left, so
#: its deadline still fires before pytest-timeout dumps every thread.
LIVE_CALL_DEADLINE_SECONDS = 75.0

#: How long a probe waits for its call to finish before the test is skipped.
#: Long enough for a local model to load cold, short enough that a silent
#: route costs seconds rather than the tier.
ROUTE_PROBE_DEADLINE_SECONDS = 15.0

#: How long a call cancelled at its deadline may take to unwind before the
#: test stops waiting for it.
CANCEL_GRACE_SECONDS = 5.0

#: The time a test keeps, after its last live call, for its fixtures to tear
#: down inside its timeout. These modules' teardowns measured under 0.5s.
TEARDOWN_RESERVE_SECONDS = 10.0

#: pytest-timeout backstop for a module of live-route tests. pytest-timeout
#: lets a marker override ``--timeout``, so these modules run under it whatever
#: the run passes, including a hand-run tier that passes none. It equals the
#: integration tier's ``--timeout=120`` in ci.yml.
TEST_TIMEOUT_SECONDS = 120

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: When a test's pytest-timeout clock started, and how many seconds it runs.
_BACKSTOP = pytest.StashKey[tuple[float, float]]()


def _current_test() -> str:
    """The node id of the running test, as pytest publishes it."""
    current = os.environ.get("PYTEST_CURRENT_TEST", "")
    return current.rsplit(" ", 1)[0] or "<unknown test>"


def _secs(seconds: float) -> str:
    """Render a duration for a message, to the hundredth: ``75``, ``0.5``, ``12.35``."""
    return f"{round(max(seconds, 0.0), 2):g}"


def record_backstop(item: pytest.Item, timeout: float) -> None:
    """Note that ``item``'s pytest-timeout clock of ``timeout`` seconds starts now.

    The integration conftest calls this from pytest-timeout's
    ``pytest_timeout_set_timer`` hook, just before the timer is armed. For
    these modules that is before setup, so setup counts against the budget
    exactly as it counts against the timer.
    """
    item.stash[_BACKSTOP] = (time.monotonic(), float(timeout))


class LiveCallBudget:
    """What one test's pytest-timeout has left for its live-route work.

    Every call and probe in the test draws on the same budget. Each may wait
    at most what the timeout has left after keeping back
    :data:`CANCEL_GRACE_SECONDS` and :data:`TEARDOWN_RESERVE_SECONDS`, so its
    deadline fires, the call unwinds and the fixtures tear down before the
    timeout does.
    """

    def __init__(self, started: float, timeout: float) -> None:
        #: The ``time.monotonic()`` at which the test's timeout clock started.
        self.started = started
        #: The test's timeout, in seconds.
        self.timeout = timeout
        #: Each bounded call or probe so far, and the seconds it took.
        self.spent: list[tuple[str, float]] = []

    @classmethod
    def for_test(cls, item: pytest.Item) -> LiveCallBudget:
        """The budget of ``item``, timed from when its pytest-timeout clock started.

        With no clock started yet (pytest-timeout is disabled, or its clock
        covers only the test body and so starts after this fixture), the budget
        starts now and runs these modules' :data:`TEST_TIMEOUT_SECONDS`.
        """
        started, timeout = item.stash.get(
            _BACKSTOP, (time.monotonic(), float(TEST_TIMEOUT_SECONDS))
        )
        return cls(started, timeout)

    def left(self) -> float:
        """Seconds a call starting now may wait and still beat the timeout."""
        reserve = CANCEL_GRACE_SECONDS + TEARDOWN_RESERVE_SECONDS
        return self.started + self.timeout - time.monotonic() - reserve

    def account(self) -> str:
        """Where the test's timeout has gone so far, for a message."""
        elapsed = time.monotonic() - self.started
        calls = sum(seconds for _, seconds in self.spent)
        parts = [f"{label} {_secs(seconds)}s" for label, seconds in self.spent]
        parts.append(f"setup and the rest of the test {_secs(elapsed - calls)}s")
        reserve = CANCEL_GRACE_SECONDS + TEARDOWN_RESERVE_SECONDS
        return (
            f"{_secs(elapsed)}s of the test's {_secs(self.timeout)}s timeout had "
            f"gone ({', '.join(parts)}), and its last {_secs(reserve)}s is kept "
            "to cancel a call and tear down"
        )


def describe_chat_routes(llm_service: Any) -> str:
    """Name the routes, in order, and the model a turn on ``llm_service`` would try."""
    providers, target_model = llm_service.resolve_provider_routing()
    names = ", ".join(route.get("name") or "?" for route in providers)
    return f"chat routes [{names or 'none'}] model {target_model or 'auto'}"


def describe_discovery_routes(llm_service: Any) -> str:
    """Name the routes model discovery (``!model-list``) contacts."""
    names = ", ".join(
        route.get("name") or vendor
        for vendor, route in llm_service._select_discovery_routes()
    )
    return f"discovery routes [{names or 'none'}]"


def _frames(task: asyncio.Task) -> list[tuple[str, str, int]]:
    """The task's coroutine frames, outermost first, as (name, file, line)."""
    frames = []
    awaited: Any = task.get_coro()
    while awaited is not None:
        frame = getattr(awaited, "cr_frame", None) or getattr(awaited, "gi_frame", None)
        if frame is None:
            break
        frames.append((frame.f_code.co_qualname, frame.f_code.co_filename, frame.f_lineno))
        awaited = getattr(awaited, "cr_await", None) or getattr(awaited, "gi_yieldfrom", None)
    return frames


def _task_tree(
    task: asyncio.Task, depth: int = 0, seen: set | None = None
) -> Iterator[tuple[int, asyncio.Task, list[tuple[str, str, int]], Any, list[asyncio.Task]]]:
    """Yield ``(depth, task, frames, waiter, children)`` for ``task`` and what it waits on.

    ``waiter`` is the future the task is suspended on and ``children`` the
    running tasks behind it: a task it awaits (``process_input`` runs its turn
    in an isolated child task) or the tasks of a ``gather``. asyncio keeps the
    waiter as the task's ``_fut_waiter`` (3.11 through 3.14) and ``gather`` its
    tasks as ``_children``. Neither is public API: if either moves, the walk
    stops at the waiter instead of following it.
    """
    seen = set() if seen is None else seen
    seen.add(task)
    waiter = getattr(task, "_fut_waiter", None)
    behind = [waiter] if isinstance(waiter, asyncio.Task) else getattr(waiter, "_children", ())
    children = [
        child for child in behind
        if isinstance(child, asyncio.Task) and not child.done() and child not in seen
    ]
    yield depth, task, _frames(task), waiter, children
    for child in children:
        yield from _task_tree(child, depth + 1, seen)


def await_chain(task: asyncio.Task) -> str:
    """Render where ``task`` is parked: each task's frames, outermost first.

    The walk follows the tasks it waits on, so the chain ends where the work
    is actually parked rather than at the first task boundary.
    """
    lines = []
    for depth, node, frames, waiter, children in _task_tree(task):
        indent = "  " * (depth + 1)
        lines.append(f"{indent}task {node.get_name()!r}:")
        lines.extend(f"{indent}  {name} ({path}:{line})" for name, path, line in frames)
        if waiter is not None and not children:
            lines.append(f"{indent}  awaiting {waiter!r}")
    return "\n".join(lines)


def parked_in(task: asyncio.Task) -> str:
    """Name the innermost repository code each waiting branch of ``task`` is in."""
    spots = []
    for _, _, frames, _, children in _task_tree(task):
        if children:
            continue
        own = [
            frame for frame in frames
            if Path(frame[1]).is_relative_to(_REPO_ROOT)
            and "site-packages" not in Path(frame[1]).parts
        ]
        if own:
            name, path, line = own[-1]
            spots.append(f"{name} ({Path(path).name}:{line})")
    return ", ".join(dict.fromkeys(spots)) or "an await outside this repository"


async def _settle(tasks: Iterable[asyncio.Task]) -> bool:
    """Cancel ``tasks`` and give them :data:`CANCEL_GRACE_SECONDS` to unwind.

    Returns whether every task finished. A task that ignores cancellation is
    left to the event loop's teardown rather than awaited forever, which is
    the hang this module exists to end.
    """
    pending = [task for task in tasks if not task.done()]
    for task in pending:
        task.cancel()
    if not pending:
        return True
    _, still_running = await asyncio.wait(pending, timeout=CANCEL_GRACE_SECONDS)
    return not still_running


async def _overran(task: asyncio.Task, deadline: float) -> bool:
    """Wait up to ``deadline`` for ``task``; return whether it is still running.

    If the waiting test is itself cancelled, the task is cancelled with it
    rather than left running behind it.
    """
    try:
        await asyncio.wait([task], timeout=deadline)
    except BaseException:
        task.cancel()
        raise
    return not task.done()


@dataclass(frozen=True)
class _Bounded:
    """One call run against a :class:`LiveCallBudget`, and how it ended."""

    task: asyncio.Task
    #: The seconds the call was given.
    deadline: float
    #: Why it was given less than it asked for, or ``""`` if it was not.
    cut: str
    #: Whether it was still running when its deadline fired.
    overran: bool


async def _run_within(
    awaitable: Awaitable[Any],
    *,
    budget: LiveCallBudget,
    label: str,
    requested: float,
    waiting_on: str,
) -> _Bounded:
    """Run ``awaitable`` for at most ``requested`` seconds and what ``budget`` has left.

    Fails the test at once, without running the call, when the budget has
    nothing left for it: every second of the timeout is already spent or kept.
    """
    task = asyncio.ensure_future(awaitable)
    left = budget.left()
    if left <= 0:
        account = budget.account()
        await _settle([task])
        pytest.fail(
            f"{_current_test()}: no time left to run {label}, which would wait on "
            f"{waiting_on}: {account}.",
            pytrace=False,
        )
    cut = ""
    deadline = requested
    if left < requested:
        deadline = left
        cut = f", all that was left of {_secs(requested)}s once {budget.account()}"
    started = time.monotonic()
    overran = await _overran(task, deadline)
    budget.spent.append((label, time.monotonic() - started))
    return _Bounded(task, deadline, cut, overran)


async def bounded_live_call(
    awaitable: Awaitable[T],
    *,
    budget: LiveCallBudget,
    what: str,
    waiting_on: str,
    deadline: float = LIVE_CALL_DEADLINE_SECONDS,
) -> T:
    """Await one live-route call, failing the test by name if it overruns.

    ``budget`` is the test's :class:`LiveCallBudget`, shared by all of its
    calls. ``what`` names the command or prompt, and ``waiting_on`` the routes
    (see :func:`describe_chat_routes`) or service the call depends on. The
    call's own result or exception passes through unchanged.
    """
    run = await _run_within(
        awaitable, budget=budget, label=what, requested=deadline, waiting_on=waiting_on
    )
    if not run.overran:
        return run.task.result()
    chain = await_chain(run.task)
    unwound = await _settle([run.task])
    pytest.fail(
        f"{_current_test()}: {what} did not return within {_secs(run.deadline)}s"
        f"{run.cut}; it was waiting on {waiting_on}. Cancelled at the deadline, it "
        f"{'unwound' if unwound else f'was still running {_secs(CANCEL_GRACE_SECONDS)}s later'}. "
        f"Await chain when the deadline fired:\n{chain}",
        pytrace=False,
    )


async def _require_finishing(
    awaitable: Awaitable[Any],
    *,
    budget: LiveCallBudget,
    label: str,
    waiting_on: str,
    deadline: float,
    skip_reason: str,
) -> None:
    """Skip, naming where it was parked, unless ``awaitable`` finishes in time.

    Only an overrun skips. A call that raised has answered; the test makes
    the same call and reports it.
    """
    run = await _run_within(
        awaitable, budget=budget, label=label, requested=deadline, waiting_on=waiting_on
    )
    if run.overran:
        spot = parked_in(run.task)
        await _settle([run.task])
        pytest.skip(
            f"{skip_reason} within {_secs(run.deadline)}s{run.cut} (parked in {spot})"
        )
    # Retrieve the outcome so an error answer is not logged as never retrieved.
    run.task.exception()


def require_answering_ollama(deadline: float = ROUTE_PROBE_DEADLINE_SECONDS) -> None:
    """Skip unless the Ollama daemon lists its models within ``deadline``."""
    try:
        import ollama
    except ImportError as exc:
        pytest.skip(f"Ollama not available: {exc}")

    try:
        ollama.Client(timeout=deadline).list()
    except (ConnectionError, httpx.HTTPError, ollama.ResponseError) as exc:
        pytest.skip(f"Ollama not available: {type(exc).__name__}: {exc}")


async def require_answering_discovery(
    llm_service: Any,
    *,
    budget: LiveCallBudget,
    deadline: float = ROUTE_PROBE_DEADLINE_SECONDS,
) -> None:
    """Skip unless model discovery, the work behind ``!model-list``, finishes in time.

    Runs the discovery a cold ``!model-list`` runs, uncached, rather than
    asking each route something of its own choosing: discovery also
    reconciles every route's embedding and decision capabilities, so one
    route silent on any of those stalls the listing. Discovery tolerates a
    route that answers with an error, and so does this. The catalog it
    gathers stays cached, so the test's own listing reads it.
    """
    routes = describe_discovery_routes(llm_service)
    await _require_finishing(
        llm_service.discover_all_models(use_cache=False),
        budget=budget,
        label="the model-discovery probe",
        waiting_on=routes,
        deadline=deadline,
        skip_reason=f"!model-list cannot answer here: model discovery ({routes}) did not finish",
    )


async def require_answering_chat_route(
    llm_service: Any,
    *,
    budget: LiveCallBudget,
    deadline: float = ROUTE_PROBE_DEADLINE_SECONDS,
) -> None:
    """Skip unless a one-word prompt through ``llm_service``'s routes returns in time.

    It takes the turn's own route order, so it asks the route the test's turn
    would use.
    """
    routes = describe_chat_routes(llm_service)
    await _require_finishing(
        llm_service.generate(
            system_prompt="Reply with the single word OK.",
            user_prompt="Ping.",
        ),
        budget=budget,
        label="the chat-route probe",
        waiting_on=routes,
        deadline=deadline,
        skip_reason=f"No answer to a one-word prompt from {routes}",
    )
