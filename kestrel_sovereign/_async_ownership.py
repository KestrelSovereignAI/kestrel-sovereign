"""Cancellation-safe ownership of internal asyncio tasks.

Callers that create cleanup, launch, or persistence tasks remain responsible
for reaching a terminal outcome even when their own task is cancelled. This
module provides that host-wide ownership rule without depending on subprocess,
Talon, compute, or training concepts.
"""

from __future__ import annotations

import asyncio
import contextvars
import math
from collections.abc import AsyncIterator, Callable
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Generic, TypeVar, cast
from weakref import WeakKeyDictionary

_T = TypeVar("_T")
_ITERATOR_TERMINAL = object()
_ITERATOR_INTERRUPTED = object()
_TASK_ERROR_UNSET = object()
# Only used when publication of the completion callback itself failed. A
# timer yields to the SAME native owner without another Task/collector or a
# hot zero-delay loop; ordinary joins remain completion-driven.
_OWNED_COMPLETION_RECHECK_SECONDS = 0.01


@contextmanager
def owned_task_join_dependency(target, *, waiter=None):
    """Record an actual must-join relationship, never execution authority.

    Framework coordinators may register their unconditional native joins
    before dispatching children. Ordinary await_owned_task calls use the same
    relationship automatically. Weak, reference-counted waiter edges disappear
    at physical completion, including nested joins and cancellation unwind.
    """
    waiter = waiter if waiter is not None else asyncio.current_task()
    if waiter is None or target is waiter or target.done() or not _has_owned_task_metadata(target):
        yield
        return
    waiters = getattr(target, "_kestrel_owned_join_waiters", None)
    if waiters is None:
        waiters = target._kestrel_owned_join_waiters = WeakKeyDictionary()
    waiters[waiter] = waiters.get(waiter, 0) + 1
    try:
        yield
    finally:
        remaining = waiters.get(waiter, 1) - 1
        if remaining:
            waiters[waiter] = remaining
        else:
            waiters.pop(waiter, None)


def owned_task_join_ancestors(task=None):
    """Physical callers waiting on this Task, not copied-context ancestors.

    Reentrant shutdown cannot join those callers while they depend on its
    return. Their existing lifecycle roots must remain until completion; this
    observation neither settles their ACKs nor grants another admission.
    """
    task = task if task is not None else asyncio.current_task()
    seen, ancestors, pending = {task}, set(), [task]
    while pending:
        current = pending.pop()
        for waiter in tuple(getattr(current, "_kestrel_owned_join_waiters", ())):
            if waiter not in seen and not waiter.done():
                seen.add(waiter)
                ancestors.add(waiter)
                pending.append(waiter)
    return ancestors


def _tracks_owned_task_join(function):
    @wraps(function)
    async def joined(task, *args, **kwargs):
        with owned_task_join_dependency(task):
            return await function(task, *args, **kwargs)
    return joined


@dataclass(slots=True)
class OwnedTaskAdmission:
    """Caller-local receipt of native Task acceptance, before publication.

    This is not a second registry. The synchronous creator writes the exact
    Task here even if a later callback publication fails; its existing owner
    then joins or retains that Task rather than treating failure as absence.
    """
    task: asyncio.Task | None = None


def _has_owned_task_metadata(task) -> bool:
    """Native storage or an EXACT producer-admitted metadata contract.

    Future-like protocol alone grants no private writes or ancestry edges.
    Even an asyncio subclass may be read-only. Core's task-bound admission
    receipt covers its custom factory Tasks; a forwarding facade cannot reuse
    the underlying Task's receipt as its own capability.
    """
    if type(task) in (asyncio.Task, asyncio.Future):
        return True
    admission = getattr(task, "_kestrel_owned_admission", None)
    return type(admission) is OwnedTaskAdmission and admission.task is task


def _retain_owned_admission_error(task, error):
    """Core-only failed start-gate handoff; the payload must remain unstarted."""
    from kestrel_sovereign.execution_custody import execution_observation_error, retained_execution_error
    cached = getattr(task, "_kestrel_owned_terminal_error", None)
    terminal = retained_execution_error(execution_observation_error(cached, error))
    task._kestrel_owned_terminal_error = terminal
    return terminal


def create_owned_task(operation, *, name=None, context=None, retain_originals=False,
                      retirement_resource=None, loop=None, admission=None):
    """Capture the original carrier inside a child before asyncio can erase it.

    Creation retains normal copied-context semantics. The caller still owns
    physical joining or a documented bounded retirement transfer.
    Native epilogues can opt into retaining an original enclosing cancellation
    with its explicit cause. Generic cleanup-error priority stays unchanged.
    Without an explicit admission receipt, a postaccept publication failure is
    delivered through the returned failed-gated Task. True allocation refusal
    still raises synchronously; an explicit receipt retains raise-plus-receipt.
    A physical closer can name its original resource: refusal to allocate that
    closer is explicit unstarted retirement debt, never a successful close.
    An explicit running loop preserves callers' native loop-admission surface;
    a different loop cannot receive ownership of this caller's operation.
    """
    started = False
    private_admission = admission is None
    if admission is None:
        admission = OwnedTaskAdmission()
    gate = None
    async def run():
        nonlocal started
        try:
            if gate is not None:
                opening = await gate
                if opening is not None:
                    raise opening
            started = True
            return await operation
        except BaseException as error:
            from kestrel_sovereign.execution_custody import (
                execution_terminal_error,
                execution_observation_error,
                retained_execution_error,
            )
            task = asyncio.current_task()
            cached = getattr(task, "_kestrel_owned_terminal_error", None)
            terminal = retained_execution_error(
                execution_observation_error(cached, error)
                if cached is not None and cached is not error else
                execution_terminal_error(error, retain_originals=retain_originals)
            )
            try:
                task._kestrel_owned_terminal_error = terminal
            except BaseException as publication:
                raise execution_observation_error(terminal, publication)
            raise
    wrapper = run()
    try:
        if type(admission) is not OwnedTaskAdmission or admission.task is not None:
            raise TypeError("owned admission requires a fresh original receipt")
        gate = asyncio.get_running_loop().create_future()
        if loop is None:
            task = asyncio.create_task(wrapper, name=name, context=context)
        else:
            if loop is not asyncio.get_running_loop():
                raise RuntimeError("owned operation requires the original running loop")
            task = loop.create_task(wrapper, name=name, context=context)
    except BaseException as error:
        wrapper.close()
        close = getattr(operation, "close", None)
        if callable(close):
            close()
        if retirement_resource is not None:
            from kestrel_sovereign.execution_custody import execution_terminal_error
            debt = RuntimeError("original native closer admission refused")
            # This private native receipt owns the actual admission failure,
            # including a genuine source C. Source-only shared projections
            # must retain it even though no accepted Task exists to harvest.
            debt.__cause__ = error
            debt._kestrel_native_retirement_debts = ((retirement_resource, None),)
            raise execution_terminal_error(
                error, debt, retain_originals=True,
                isolate_control_originals=True,
            )
        raise
    unstarted_closed = False
    def close_unstarted(_task):
        nonlocal unstarted_closed
        if not started and not unstarted_closed:
            unstarted_closed = True
            close = getattr(operation, "close", None)
            if callable(close):
                close()
    admission.task = task
    try:
        task._kestrel_owned_admission = admission
        task.add_done_callback(close_unstarted)
    except BaseException as opening:
        if gate is not None:
            # The payload is still behind its failed gate. Retain this actual
            # source before any later consumer can cancel the Task before its
            # first step and replace the opening with a synthetic cancellation.
            from kestrel_sovereign.execution_custody import (
                execution_observation_error, execution_terminal_error,
            )
            terminal = execution_terminal_error(opening, retain_originals=retain_originals)
            try:
                _retain_owned_admission_error(task, terminal)
            except BaseException as publication:
                terminal = execution_observation_error(terminal, publication)
            finally:
                try:
                    close_unstarted(task)
                except BaseException as close_error:
                    terminal = execution_observation_error(terminal, close_error)
                finally:
                    gate.set_result(terminal)
            if private_admission:
                return task
            raise terminal
        raise
    if gate is not None:
        gate.set_result(None)
    return task


def pending_owned_retirement_error(resource, tasks, *, message):
    """Retain actual unfinished original owners, never a retirement ACK.

    A bounded drain may return control, but its resource and exact Tasks
    remain owned until physical completion. No Task is cancelled, replaced,
    or marked complete by this snapshot.
    """
    pending = tuple(dict.fromkeys(task for task in tasks if not task.done()))
    if not pending:
        return None
    error = TimeoutError(message)
    error._kestrel_native_retirement_debts = tuple(
        (resource, task) for task in pending
    )
    return error


async def wait_owned_tasks_until(tasks, *, deadline: float):
    """Observe original Tasks within one absolute budget, without cancelling them.

    The caller retains every pending Task and selects its terminal control after
    this observation. Repeated observer withdrawal neither resets the deadline
    nor allocates a collector Task or interrupts an accepted native producer.
    """
    from kestrel_sovereign.execution_custody import execution_terminal_error

    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
        raise ValueError("owned task deadline must be finite")
    originals = set(tasks)
    pending = {task for task in originals if not task.done()}
    withdrawal = None
    loop = asyncio.get_running_loop()
    while pending:
        pending = {task for task in pending if not task.done()}
        remaining = deadline - loop.time()
        if not pending or remaining <= 0:
            break
        try:
            _, pending = await asyncio.wait(pending, timeout=remaining)
        except asyncio.CancelledError as error:
            withdrawal = execution_terminal_error(
                withdrawal, error, retain_originals=True,
            )
    return OwnedTaskOutcome((originals - pending, pending), None, withdrawal)


async def gather_owned(operations, *, cancel_siblings_on_error=True,
                       retirement_resource=None, retain_originals=False,
                       name_prefix: str | None = None,
                       isolate_control_originals=False,
                       return_when=asyncio.ALL_COMPLETED):
    """Join accepted children even if later admission or observation fails.

    Independent native retirement can disable sibling cancellation: all
    operations are attempted, refused admission retains the original resource,
    and accepted children physically join before terminal control is selected.
    Finite callers can isolate original control carriers from shared source
    receipts. Indexed names are diagnostic only, never a second owner.
    FIRST_COMPLETED races are finite observers: after the first outcome, every
    remaining accepted peer is cancelled and physically joined before return.
    """
    from kestrel_sovereign.execution_custody import (
        execution_terminal_error,
        is_execution_control_error,
    )
    if return_when not in (asyncio.ALL_COMPLETED, asyncio.FIRST_COMPLETED):
        raise ValueError("owned gather requires ALL_COMPLETED or FIRST_COMPLETED")
    children = []
    pending = set()
    primary = None
    first_observed = False
    try:
        for index, operation in enumerate(operations):
            admission = OwnedTaskAdmission()
            try:
                children.append(create_owned_task(
                    operation, retirement_resource=retirement_resource,
                    retain_originals=retain_originals,
                    name=f"{name_prefix}:{index}" if name_prefix is not None else None,
                    admission=admission,
                ))
            except BaseException as error:
                if admission.task is not None:
                    # Publication failed, not acceptance. This exact gated
                    # child must be harvested without cancelling away its
                    # original opening failure.
                    children.append(admission.task)
                    outcome = await await_owned_task(admission.task)
                    error = execution_terminal_error(
                        error, outcome.error, outcome.cancellation,
                        retain_originals=retain_originals,
                        isolate_control_originals=isolate_control_originals,
                    )
                error = owned_task_admission_error(
                    retirement_resource, admission, error,
                )
                if cancel_siblings_on_error:
                    raise error
                primary = execution_terminal_error(
                    primary, error, retain_originals=retain_originals,
                    isolate_control_originals=isolate_control_originals,
                )
        pending = set(children)
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            first_observed = return_when == asyncio.FIRST_COMPLETED
            for child in done:
                error = owned_task_error(child)
                if error is not None:
                    raise error
            if first_observed:
                break
    except BaseException as error:
        primary = execution_terminal_error(
            primary, error, retain_originals=retain_originals,
            isolate_control_originals=isolate_control_originals,
        )
    if first_observed or (primary is not None and cancel_siblings_on_error):
        for child in children:
            if not child.done():
                child.cancel()
    errors = [primary]
    results = []
    for child in children:
        outcome = await await_owned_task(child)
        results.append(outcome.result)
        if outcome.error is not None and (not isinstance(outcome.error, asyncio.CancelledError) or is_execution_control_error(outcome.error)):
            errors.append(outcome.error)
        errors.append(outcome.cancellation)
    terminal = execution_terminal_error(
        *errors, retain_originals=retain_originals,
        isolate_control_originals=isolate_control_originals,
    )
    if terminal is not None:
        raise terminal
    return results


class _ConsumerCloseRecord:
    """Whether the consumer of one owned iterator closed it early.

    The record belongs to exactly ONE task: the producer task its iterator
    owns. It is reached through a context variable only because that is how a
    producer finds its nearest enclosing iterator, but context is inherited by
    every task the producer spawns, and a disposition is not. A child task — a
    background cognition turn a streamed turn enqueued, say — shares this very
    object by reference and would otherwise report the parent's disconnect as
    its own. So the record names its owner task, and only that task may read
    it as set.
    """

    __slots__ = ("closed", "owner", "parents")

    def __init__(self) -> None:
        self.closed = False
        self.owner: asyncio.Task[None] | None = None
        self.parents: tuple[asyncio.Task, ...] = ()

    def closed_for(self, task: asyncio.Task[object] | None) -> bool:
        return self.closed and task is not None and task is self.owner


_CONSUMER_CLOSE: contextvars.ContextVar[_ConsumerCloseRecord | None] = (
    contextvars.ContextVar("kestrel_owned_iterator_consumer_close", default=None)
)


def owned_iterator_parent_tasks() -> tuple[asyncio.Task, ...]:
    """Explicit iterator forwarding lineage, never inherited task authority.

    Observation such as a call watchdog follows its physical source tasks.
    An arbitrary child copied from a producer is not itself that producer,
    and must not claim this lineage or its parent's call-level timing scope.
    """
    record = _CONSUMER_CLOSE.get()
    if record is None:
        return ()
    try:
        current = asyncio.current_task()
    except RuntimeError:
        return ()
    if record.owner is not current:
        return ()
    return record.parents


def owned_consumer_closed() -> bool:
    """Whether the consumer of the enclosing owned iterator closed it early.

    Read from INSIDE the producer. This is a recorded disposition, set by
    :meth:`OwnedAsyncIterator.aclose` before it interrupts the producer, so a
    producer that unwinds with ``CancelledError`` because its reader went away
    can say so. The exception alone cannot: a cooperative Stop and a host
    teardown raise that same ``CancelledError`` (#3159). Only the nearest
    enclosing owned iterator is consulted, and only from that iterator's own
    producer task: a producer outside any owned iterator, or any task that
    merely inherited the producer's context, reports ``False``.
    """

    record = _CONSUMER_CLOSE.get()
    if record is None:
        return False
    try:
        current = asyncio.current_task()
    except RuntimeError:
        # No running loop: nothing here can be an owned producer.
        return False
    return record.closed_for(current)


@dataclass(frozen=True, slots=True)
class OwnedTaskOutcome(Generic[_T]):
    """Terminal result of an owned task plus caller cancellation state."""

    result: _T | None
    error: BaseException | None
    cancellation: asyncio.CancelledError | None


def owned_task_admission_error(resource, admission, failure):
    """Private debt for one failed admission; never decorate the source.

    An accepted gated Task is distinct from a true allocation refusal. Neither
    is a physical retirement ACK. The caller still owns joining the receipt.
    """
    from kestrel_sovereign.execution_custody import (
        execution_native_source_error, execution_terminal_error,
    )
    debt = None
    if resource is not None:
        debt = RuntimeError("original native operation admission failed")
        debt.__cause__ = execution_native_source_error(failure) or failure
        debt._kestrel_native_retirement_debts = ((resource, admission.task),)
    return execution_terminal_error(
        failure, debt, retain_originals=True, isolate_control_originals=True,
    )


async def await_owned_operation(operation, *, retirement_resource=None,
                                retain_originals=False, name=None, context=None,
                                **observation):
    """Admit and physically join one operation, including publication failure.

    This is only for an existing asynchronous join boundary. Bounded or
    synchronous handoffs must instead inventory their explicit receipt in the
    existing lifecycle owner. No registry, replacement Task, retry or new
    observation deadline is introduced here.
    """
    from kestrel_sovereign.execution_custody import execution_terminal_error
    admission = OwnedTaskAdmission()
    opening = None
    try:
        task = create_owned_task(
            operation, retirement_resource=retirement_resource,
            retain_originals=retain_originals, name=name, context=context,
            admission=admission,
        )
    except BaseException as failure:
        opening = owned_task_admission_error(retirement_resource, admission, failure)
        task = admission.task
        if task is None:
            return OwnedTaskOutcome(None, opening, observation.get("pending_cancellation"))
    if opening is None:
        return await await_owned_task(task, **observation)
    # Do not cancel the failed start gate into a substitute CancelledError.
    outcome = await await_owned_task(
        task, pending_cancellation=observation.get("pending_cancellation"),
    )
    return OwnedTaskOutcome(
        outcome.result,
        execution_terminal_error(opening, outcome.error, retain_originals=True,
                                 isolate_control_originals=True),
        outcome.cancellation,
    )


def owned_task_observation_control(outcome):
    """Keep native source facts, not a finite observer's withdrawal wrapper.

    This projects evidence only; it neither authenticates foreign errors nor
    adopts, closes or retries the original resource. A cancellation which is
    itself the sole native carrier cannot be silently discarded.
    """
    from kestrel_sovereign.execution_custody import (
        _explicit_control_chain, execution_terminal_error, is_execution_control_error,
    )
    if outcome is None:
        return None
    controls = [outcome.error]
    withdrawal = outcome.cancellation
    if is_execution_control_error(withdrawal):
        originals = [error for error in _explicit_control_chain(withdrawal)
                     if error is not withdrawal and is_execution_control_error(error)]
        controls.extend(originals or [withdrawal])
    return execution_terminal_error(
        *controls, retain_originals=True, isolate_control_originals=True,
    )


@dataclass(frozen=True, slots=True)
class PendingOwnedTask(Generic[_T]):
    """Abandoned observation, never a terminal native result or commit receipt."""

    task: asyncio.Future[_T]
    cancellation: asyncio.CancelledError

    def raise_abandonment(self):
        from kestrel_sovereign.execution_custody import (
            ExecutionAuthorityError, execution_terminal_error,
        )
        pending = ExecutionAuthorityError("native operation remains pending in its retained lifecycle owner")
        pending.native_owner = self.task
        pending.__cause__ = self.cancellation
        raise execution_terminal_error(self.cancellation, pending)


class OwnedAsyncIterator(Generic[_T]):
    """Keep every interaction with an async iterator in one owner task.

    Async generators may bind ``ContextVar`` tokens while producing their first
    item. Resuming or closing such a generator from another task operates in a
    copied context and makes token reset fail. This adapter owns construction,
    iteration, and closure in one task while its consumer receives items over a
    one-item handshake. ``aclose`` remains cancellation-safe by joining that
    owner with :func:`await_owned_task`.
    """

    def __init__(
        self,
        iterator_factory: Callable[[], AsyncIterator[_T]],
        *,
        operation: str,
        cleanup_requested: Callable[[], bool] | None = None,
        owner_context: contextvars.Context | None = None,
        owner_factory: Callable | None = None,
        owner_started: Callable | None = None,
    ) -> None:
        self._iterator_factory = iterator_factory
        self._operation = operation
        self._cleanup_requested = cleanup_requested
        self._items: asyncio.Queue[tuple[object, _T | None]] = asyncio.Queue()
        self._continue = asyncio.Event()
        self._stop = asyncio.Event()
        self._item_outstanding = False
        self._waiting_for_continue = False
        self._interrupt_after_continue = False
        self._closed = False
        self._interrupted_by_cleanup = False
        self._cleanup_error: BaseException | None = None
        # asyncio discards a cancelled task's original exception after its
        # first retrieval. Both the reader and closer need the same evidence.
        self._terminal_error: BaseException | None = None
        self._terminal_published = False
        self._cancellation_watch_started = False
        self._construction_error = None
        self._cancellation_owner = None
        self._consumer_close = _ConsumerCloseRecord()
        parent = asyncio.current_task()
        self._consumer_close.parents = (
            () if parent is None else (parent, *owned_iterator_parent_tasks())
        )
        # The producer runs in a copy of the constructing context in which
        # this iterator's close record is bound, so the producer can find
        # the record of ITS consumer. The record's owner binding below is
        # what keeps tasks the producer spawns from claiming it.
        owner_context = contextvars.copy_context() if owner_context is None else owner_context
        owner_context.run(_CONSUMER_CLOSE.set, self._consumer_close)
        dispatch = asyncio.Event()
        async def run_owned():
            await dispatch.wait()
            await self._run()
        self._owner = None
        owner_admission, watcher_admission = OwnedTaskAdmission(), OwnedTaskAdmission()
        try:
            self._owner = (
                create_owned_task(run_owned(), name=f"owned_async_iterator:{operation}",
                    context=owner_context, admission=owner_admission, retain_originals=True)
                if owner_factory is None else owner_factory(run_owned, owner_context)
            )
            # All accepted owners and every subsequent publication belong to
            # this one construction transaction, including the first callback.
            self._consumer_close.owner = self._owner
            self._owner.add_done_callback(self._publish_owner_completion)
            if owner_started is not None:
                owner_started(self, self._owner)
            self._cancellation_owner = create_owned_task(
                self._watch_owner_cancellation(),
                name=f"owned_async_iterator_cancel:{operation}",
                admission=watcher_admission, retain_originals=True,
            )
            self._cancellation_owner.add_done_callback(
                self._interrupt_if_watch_never_started
            )
        except BaseException as error:
            self._owner = self._owner if self._owner is not None else owner_admission.task
            self._cancellation_owner = (self._cancellation_owner if self._cancellation_owner is not None
                else watcher_admission.task)
            if self._owner is None:
                raise  # True allocation refusal; no accepted owner to join.
            from kestrel_sovereign.execution_custody import execution_observation_error
            error = execution_observation_error(error, *(
                getattr(task, "_kestrel_owned_terminal_error", None)
                for task in (self._owner, self._cancellation_owner) if task is not None
            ))
            # No source body has crossed dispatch. Retain the exact partially
            # constructed owner so the async boundary can physically join it;
            # the synchronous constructor cannot manufacture that completion.
            self._construction_error = error
            error._kestrel_failed_owned_iterator = self
            error._kestrel_native_retirement_debts = (
                *getattr(error, "_kestrel_native_retirement_debts", ()),
                (self, self._owner),
            )
            self._stop.set()
            self._owner.cancel()
            if self._cancellation_owner is not None:
                self._cancellation_owner.cancel()
            raise error
        dispatch.set()

    def __aiter__(self) -> "OwnedAsyncIterator[_T]":
        return self

    def _publish_owner_completion(self, _owner: asyncio.Task[None]) -> None:
        # A native owner can terminate while waiting for privacy admission,
        # before _run() enters its finally. The blocked reader still needs
        # the original owner's terminal outcome, never a replacement body.
        if not self._terminal_published:
            self._terminal_published = True
            self._items.put_nowait((_ITERATOR_TERMINAL, None))

    @property
    def terminal_error(self) -> BaseException | None:
        """Return the source owner's terminal error once it has settled."""

        if not self._owner.done():
            return None
        if self._terminal_error is not None:
            return self._terminal_error
        try:
            return self._owner.exception()
        except asyncio.CancelledError as error:
            return error

    @property
    def owner_task(self) -> asyncio.Task[None]:
        """Return the task whose cancellation interrupts a blocked producer."""

        return self._cancellation_owner

    @property
    def interrupted_by_cleanup(self) -> bool:
        """Whether requested cancellation interrupted source iteration."""

        return self._interrupted_by_cleanup

    @property
    def cleanup_error(self) -> BaseException | None:
        """Return an error raised while cancellation/closure unwound the source."""

        return self._cleanup_error

    async def __anext__(self) -> _T:
        if self._closed:
            raise StopAsyncIteration
        if self._item_outstanding:
            self._item_outstanding = False
            self._continue.set()
            if self._interrupt_after_continue:
                self._interrupt_after_continue = False
                asyncio.get_running_loop().call_soon(
                    self._cancel_owner_if_running
                )

        marker, item = await self._items.get()
        if marker is _ITERATOR_INTERRUPTED:
            # Do not join here. The consumer may first publish its bounded Stop
            # acknowledgement; its ``finally`` then calls ``aclose`` and owns
            # the source cleanup through terminalization.
            raise StopAsyncIteration
        if marker is _ITERATOR_TERMINAL:
            self._closed = True
            outcome = await self._join_source_owners()
            raise_owned_outcome(outcome, operation=self._operation, retain_originals=True)
            raise StopAsyncIteration

        self._item_outstanding = True
        return cast(_T, item)

    async def _watch_owner_cancellation(self) -> None:
        """Translate lifecycle cancellation without racing a delivered item."""

        self._cancellation_watch_started = True
        try:
            outcome = await await_owned_task(self._owner, defer_cancellation=False)
            if outcome.error is not owned_task_error(self._owner):
                # A borrowed join's publication/removal failure is distinct
                # from the producer's physical result. Its original watcher
                # remains owned by this iterator and is harvested below.
                raise outcome.error
        except asyncio.CancelledError as error:
            from kestrel_sovereign.execution_custody import is_execution_control_error
            if is_execution_control_error(error):
                raise
            if self._owner.done():
                return
            if self._item_outstanding:
                # Let the consumer resume the source once. A source already at
                # EOF can publish its real cleanup result; a provider that
                # blocks in the next ``anext`` is interrupted one loop turn later.
                self._interrupt_after_continue = True
            elif self._waiting_for_continue:
                # The consumer already released the handshake in this event-loop
                # turn. Let the owner publish a synchronous EOF/failure first,
                # then interrupt only if it actually blocks in the next source step.
                asyncio.get_running_loop().call_soon(
                    self._cancel_owner_if_running
                )
            else:
                self._cancel_owner_if_running()
            raise
        except BaseException:
            # Losing the cancellation observer cannot leave an operational
            # producer running without its lifecycle handoff. Interrupt once;
            # the reader/closer still physically joins that SAME producer.
            self._stop.set()
            self._continue.set()
            if not self._owner.done() and not self._owner.cancelling():
                self._owner.cancel()
            raise

    async def _join_source_owners(self):
        """Retain the source AND its original cancellation observer outcomes."""
        from kestrel_sovereign.execution_custody import execution_terminal_error, execution_observation_error, is_execution_control_error
        def compose(*errors):
            if self._construction_error is not None:
                return execution_observation_error(*errors)
            return execution_terminal_error(*errors, retain_originals=True, isolate_control_originals=True)
        outcome = await await_owned_task(self._owner)
        watcher = await await_owned_task(self._cancellation_owner) if self._cancellation_owner is not None else None
        watcher_error = watcher.error if watcher is not None else None
        if isinstance(watcher_error, asyncio.CancelledError) and not is_execution_control_error(watcher_error):
            watcher_error = None  # The watcher's private Stop handoff, not reader withdrawal.
        return OwnedTaskOutcome(
            outcome.result,
            compose(self._terminal_error, outcome.error, watcher_error),
            compose(outcome.cancellation, watcher.cancellation if watcher is not None else None),
        )

    def _interrupt_if_watch_never_started(
        self,
        watcher: asyncio.Task[None],
    ) -> None:
        """Close the pre-first-turn cancellation gap of the watcher task.

        ``Task.cancel()`` can terminally cancel a newly-created task without
        running one byte of its coroutine.  In that case the watcher's
        ``except CancelledError`` cannot translate cancellation to the source
        owner, so do that one missing handoff from a synchronous done callback.
        Once the watcher has started, its handshake-aware body remains the
        sole owner of cancellation ordering.
        """

        if (
            watcher.cancelled()
            and not self._cancellation_watch_started
            and not self._owner.done()
        ):
            self._cancel_owner_if_running()

    def _cancel_owner_if_running(self) -> None:
        if not self._owner.done() and not self._owner.cancelling():
            self._owner.cancel()

    async def _run(self) -> None:
        iterator: AsyncIterator[_T] | None = None
        try:
            iterator = self._iterator_factory()
            try:
                async for item in iterator:
                    if self._stop.is_set():
                        break
                    await self._items.put((self, item))
                    self._waiting_for_continue = True
                    try:
                        await self._continue.wait()
                    finally:
                        self._waiting_for_continue = False
                    self._continue.clear()
                    if self._stop.is_set():
                        break
            except BaseException as error:
                # Import lazily: execution_custody also uses this primitive.
                # A cancellation carrying authority/commit evidence is not
                # our private producer interrupt, even during requested close.
                from kestrel_sovereign.execution_custody import (
                    is_execution_control_error,
                )
                if is_execution_control_error(error):
                    raise
                # An exception raised while a requested close interrupts
                # ``anext`` belongs to generator unwinding, not ordinary
                # source execution. Preserve that distinction for lifecycle
                # acknowledgement at the transport boundary.
                close_requested = self._stop.is_set()
                cancellation_requested = close_requested
                if (
                    not cancellation_requested
                    and self._cleanup_requested is not None
                ):
                    try:
                        cancellation_requested = self._cleanup_requested() is True
                    except Exception:
                        # An observability predicate must not replace the
                        # iterator's actual terminal failure.
                        cancellation_requested = False
                # A Stop marker can race a source's independent terminal
                # failure after its final item. Only an initiated close owns a
                # non-cancellation unwind as cleanup debt; the marker extends
                # ownership solely to cancellation that interrupted the source.
                if close_requested and not isinstance(
                    error,
                    asyncio.CancelledError,
                ):
                    self._cleanup_error = error
                if cancellation_requested and isinstance(
                    error,
                    asyncio.CancelledError,
                ):
                    self._interrupted_by_cleanup = True
                    self._items.put_nowait((_ITERATOR_INTERRUPTED, None))
                    return
                raise
        except BaseException as error:
            self._terminal_error = error
            raise
        finally:
            try:
                if iterator is not None:
                    close_iterator = getattr(iterator, "aclose", None)
                    if callable(close_iterator):
                        try:
                            await close_iterator()
                        except BaseException as error:
                            from kestrel_sovereign.execution_custody import (
                                execution_terminal_error,
                            )
                            self._cleanup_error = error
                            self._terminal_error = execution_terminal_error(
                                self._terminal_error, error,
                                retain_originals=True, isolate_control_originals=True,
                            )
                            raise self._terminal_error
            finally:
                self._publish_owner_completion(self._owner)

    async def aclose(self) -> None:
        """Stop producing and join the source iterator's owner task."""

        if self._closed:
            return
        self._closed = True
        # Record the disposition BEFORE the producer is interrupted, so the
        # producer's unwind reads why it was cancelled rather than guessing
        # from the exception it happens to receive.
        self._consumer_close.closed = True
        self._stop.set()
        self._continue.set()
        owner_cancelled_by_close = False
        if not self._owner.done() and not self._owner.cancelling() and not self._interrupted_by_cleanup:
            owner_cancelled_by_close = self._owner.cancel()
        outcome = await self._join_source_owners()
        from kestrel_sovereign.execution_custody import is_execution_control_error
        if (
            (owner_cancelled_by_close or self._construction_error is not None)
            and isinstance(outcome.error, asyncio.CancelledError)
            and self._cleanup_error is None
            and not is_execution_control_error(outcome.error)
        ):
            # Cancellation is the private interrupt used to wake a producer
            # blocked in ``anext``. Once the owner's finally block has closed
            # the iterator, it is a successful close, while cancellation of
            # the caller itself still takes precedence below.
            outcome = OwnedTaskOutcome(None, None, outcome.cancellation)
        if self._construction_error is not None and outcome.error is None:
            # Reconcile ONLY the original constructor task reservation, after
            # its actual frame/lock cleanup AND watcher have physically joined.
            # A real owner/closer failure keeps this debt and every other debt.
            error = self._construction_error
            error._kestrel_native_retirement_debts = tuple(
                debt for debt in error._kestrel_native_retirement_debts
                if not (debt[0] is self and debt[1] is self._owner)
            )
        raise_owned_outcome(outcome, operation=self._operation, retain_originals=True)


async def create_owned_iterator(iterator_factory, **options):
    """Construct a stream and join any accepted partial owner before failure.

    Public async handoffs must use this boundary, not infer that a failed
    synchronous constructor accepted no producer or cancellation watcher.
    """
    try:
        return OwnedAsyncIterator(iterator_factory, **options)
    except BaseException as opening:
        partial = getattr(opening, "_kestrel_failed_owned_iterator", None)
        if isinstance(partial, OwnedAsyncIterator):
            try:
                await partial.aclose()
            except BaseException as cleanup:
                from kestrel_sovereign.execution_custody import execution_observation_error
                raise execution_observation_error(opening, cleanup)
        raise


def owned_task_error(task: asyncio.Future[_T]) -> BaseException | None:
    """Retain the first terminal error across every owner of the same task.

    Asyncio consumes a cancelled task's original exception on first retrieval.
    A completion callback and a later lifecycle join must share that evidence,
    not manufacture a cause-free cancellation on the second read.

    Protocol-only foreign Futures need no private storage. Their exact finite
    observation belongs to OwnedTaskOutcome; producer-supplied retained error
    remains readable, but this join does not invent cross-owner source caching.
    """
    retained = getattr(task, "_kestrel_owned_terminal_error", _TASK_ERROR_UNSET)
    if retained is not _TASK_ERROR_UNSET:
        if task.done() and not task.cancelled():
            # Capturing inside the child preserves original evidence, but does
            # not acknowledge asyncio's native exception notification. Harvest
            # that completed result as well so a joined task cannot later emit
            # an orphaned "exception was never retrieved" diagnostic.
            task.exception()
        return retained
    if not task.done():
        raise RuntimeError("owned task has not reached terminal state")
    try:
        error = task.exception()
    except asyncio.CancelledError as cancellation:
        error = cancellation
    if _has_owned_task_metadata(task):
        try:
            task._kestrel_owned_terminal_error = error  # type: ignore[attr-defined]
        except BaseException as publication:
            # This producer promised native metadata, then refused its cache.
            # Report that broken contract alongside the physical outcome; a
            # failed publication must never replace UNKNOWN/source C with it.
            from kestrel_sovereign.execution_custody import execution_observation_error
            return execution_observation_error(error, publication)
    return error


@_tracks_owned_task_join
async def await_owned_task(
    task: asyncio.Future[_T],
    pending_cancellation: asyncio.CancelledError | None = None,
    *,
    cancel_on_caller_cancellation: bool | Callable[[asyncio.Future[_T]], None] = False,
    defer_cancellation: bool = True,
) -> OwnedTaskOutcome[_T]:
    """Retrieve ``task`` despite repeated cancellation of its caller.

    The shielded waiter carries only a successful result, never the owned
    task's exception. The owned result is retrieved exactly once after
    terminalization, avoiding orphaned-task and shield-future warnings.

    The waiter is a plain future rather than a task: event-loop teardown
    cancels every task, and a waiter task cancelled while ``task`` still runs
    would make each ``shield`` raise at once, without yielding, so this loop
    would spin and ``task`` could never finish. Nothing but this function
    holds the future, so only ``task`` completing resolves it.

    ``cancel_on_caller_cancellation`` forwards the first caller cancellation
    once to an operation that needs it to begin unwinding. Subsequent caller
    cancellations remain shielded until that same owner physically settles.
    True uses the task's native cancel method. A synchronous callable instead
    routes that first request through the original lifecycle owner's delivery
    policy. A failed forwarder cannot abandon the operation: its error is
    selected with the original task outcome only after physical joining.

    A borrowed observer can set ``defer_cancellation=False``: it neither
    cancels nor adopts the original Task, and actual caller withdrawal escapes
    immediately. The successful completion signal still avoids shield's
    synthetic cancellation when the original Task itself was cancelled.
    """
    if not defer_cancellation and (cancel_on_caller_cancellation or pending_cancellation is not None):
        raise ValueError("borrowed observation cannot forward or own pending cancellation")
    completion: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    def mark_complete(_task: asyncio.Future[_T]) -> None:
        if not completion.done():
            completion.set_result(None)

    cancellation_forwarded = False
    forwarding_error: BaseException | None = None
    publication_errors = []
    try:
        task.add_done_callback(mark_complete)
    except BaseException as error:
        publication_errors.append(error)
    try:
        while not task.done():
            if publication_errors and not defer_cancellation:
                # This is explicitly a borrowed observer, not its native
                # lifecycle owner. Its original owner retains the must-join.
                break
            try:
                if publication_errors:
                    await asyncio.sleep(_OWNED_COMPLETION_RECHECK_SECONDS)
                elif not defer_cancellation:
                    # This Future belongs only to the borrowed observer. Its
                    # cancellation cannot reach the producer, and an extra
                    # shield would delay terminal readers behind close return.
                    await completion
                else:
                    await asyncio.shield(completion)
            except asyncio.CancelledError as error:
                if pending_cancellation is None:
                    pending_cancellation = error
                if not defer_cancellation:
                    # Leave promptly, but only AFTER callback removal has
                    # contributed its independent control to this observer.
                    break
                if (callable(cancel_on_caller_cancellation) or cancel_on_caller_cancellation) and not cancellation_forwarded:
                    # Interrupt the operation once, then own its entire unwind.
                    # Repeated caller cancellation must not interrupt cleanup.
                    cancellation_forwarded = True
                    try:
                        if callable(cancel_on_caller_cancellation):
                            cancel_on_caller_cancellation(task)
                        else:
                            task.cancel()
                    except BaseException as failure:  # noqa: BLE001 - forwarding failure cannot abandon its owner
                        forwarding_error = failure
    finally:
        try:
            task.remove_done_callback(mark_complete)
        except BaseException as error:
            publication_errors.append(error)

    if not defer_cancellation and pending_cancellation is not None:
        from kestrel_sovereign.execution_custody import execution_terminal_error
        raise execution_terminal_error(
            pending_cancellation, *publication_errors, retain_originals=True,
            isolate_control_originals=True,
        )
    if not task.done():
        # Only an explicitly borrowed observer may leave before settlement.
        assert not defer_cancellation and publication_errors
        from kestrel_sovereign.execution_custody import execution_observation_error
        raise execution_observation_error(*publication_errors)
    error = owned_task_error(task)
    if publication_errors:
        from kestrel_sovereign.execution_custody import execution_observation_error
        failures = ([error] if error is not None else []) + publication_errors
        error = execution_observation_error(*failures)
    if forwarding_error is not None:
        from kestrel_sovereign.execution_custody import (
            execution_observation_error, execution_terminal_error, is_execution_control_error,
        )
        error = (execution_observation_error(error, forwarding_error)
                 if is_execution_control_error(error) or is_execution_control_error(forwarding_error)
                 else execution_terminal_error(error, forwarding_error))
        if error is not forwarding_error:
            error.add_note(f"cancellation forwarding also failed: {forwarding_error}")
    if error is not None:
        return OwnedTaskOutcome(None, error, pending_cancellation)
    return OwnedTaskOutcome(task.result(), None, pending_cancellation)


async def wait_owned_or_pending(
    task: asyncio.Future[_T], *, cancellation_grace: float,
    cancel_on_caller_cancellation: bool = True,
) -> OwnedTaskOutcome[_T] | PendingOwnedTask[_T]:
    """Observe a lifecycle-retained task; bound waiting only after cancellation.

    The caller MUST retain and physically join the task elsewhere. One timer
    starts on first cancellation; repeated interruption cannot reset it or
    re-cancel the native unwind. A pending result does not assert settlement.
    """
    if not math.isfinite(cancellation_grace) or cancellation_grace <= 0:
        raise ValueError("cancellation_grace must be finite and positive")
    loop = asyncio.get_running_loop()
    completion = loop.create_future()
    def wake(_task=None):
        if not completion.done():
            completion.set_result(None)
    task.add_done_callback(wake)
    cancellation = None
    timer = None
    try:
        while not task.done() and not completion.done():
            try:
                await asyncio.shield(completion)
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
                    timer = loop.call_later(cancellation_grace, wake)
                    if cancel_on_caller_cancellation:
                        task.cancel()
        if not task.done():
            return PendingOwnedTask(task, cancellation)
        return await await_owned_task(task, cancellation)
    finally:
        task.remove_done_callback(wake)
        if timer is not None:
            timer.cancel()


def raise_owned_outcome(
    outcome: OwnedTaskOutcome[_T],
    *,
    operation: str,
    retain_originals: bool = False,
) -> _T:
    """Preserve control before ordinary cancellation precedence.

    Finite joins may retain original native source and caller evidence without
    modifying the shared source receipt; the default priority is unchanged.
    """

    from kestrel_sovereign.execution_custody import (
        execution_terminal_error,
        is_execution_control_error,
    )
    if any(
        error is not None and is_execution_control_error(error)
        for error in (outcome.error, outcome.cancellation)
    ):
        raise execution_terminal_error(
            outcome.error, outcome.cancellation, retain_originals=retain_originals,
            isolate_control_originals=retain_originals,
        )

    if outcome.cancellation is not None:
        if outcome.error is not None:
            outcome.cancellation.add_note(f"{operation} also failed: {outcome.error}")
            raise outcome.cancellation from outcome.error
        raise outcome.cancellation
    if outcome.error is not None:
        raise outcome.error
    return cast(_T, outcome.result)


async def run_blocking_operation(
    operation: Callable[..., _T],
    *args: object,
) -> _T:
    """Finish a blocking operation before propagating caller cancellation."""

    operation_task = asyncio.create_task(asyncio.to_thread(operation, *args))
    outcome = await await_owned_task(operation_task)
    return raise_owned_outcome(outcome, operation="owned blocking operation")


__all__ = [
    "OwnedAsyncIterator",
    "OwnedTaskOutcome",
    "PendingOwnedTask",
    "await_owned_task",
    "create_owned_task",
    "gather_owned",
    "owned_task_error",
    "raise_owned_outcome",
    "run_blocking_operation",
    "wait_owned_or_pending",
    "wait_owned_tasks_until",
]
