"""Passive observation of original native resource/epilogue lifetimes.

Only backends register resources and their actual native owners. Consumers
can close new admission and join accepted epilogues; neither operation grants
work authority. Notification futures below are classification barriers, NEVER
COMMIT acknowledgements, physical closers, or successful-retirement receipts.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import BuiltinMethodType, FunctionType, MethodType
from typing import Any

from kestrel_sovereign._async_ownership import OwnedTaskOutcome, await_owned_task, owned_task_join_ancestors
from kestrel_sovereign.execution_custody import (
    ExecutionAuthorityError,
    execution_terminal_error,
    is_execution_control_error,
    retain_acknowledged_execution_commit,
)


@dataclass(eq=False)
class _NativePhaseObservation:
    resource: Any
    owner: asyncio.Future | None = None
    classified: asyncio.Future = field(default_factory=lambda: asyncio.get_running_loop().create_future())
    observers: tuple = ()

    def record_native_failure(self, error, *, commit_outcome=None, cancellation=None):
        """Backend-only physical failure/refusal, NOT a transaction body error."""
        for observer in self.observers:
            if observer._trusted_phases_only:
                observer._record_native_failure(self.resource, self.owner, error, commit_outcome, cancellation)

    def record_acknowledged_withdrawal(self, cancellation, receipt):
        """Backend-proven ACK/current withdrawal, NOT failed retirement."""
        for observer in self.observers:
            if not observer._trusted_phases_only:
                continue
            observer.native_failures.append(_NativeFailureFact(
                self.resource, self.owner, cancellation, "committed",
                cancellation, receipt,
            ))
            carrier = RuntimeError("original acknowledged native withdrawal")
            retain_acknowledged_execution_commit(carrier, receipt)
            observer._error = execution_terminal_error(
                observer._error, cancellation, carrier, retain_originals=True,
                isolate_control_originals=True,
            )
            observer.close_admission()


@dataclass(frozen=True, slots=True)
class _NativeFailureFact:
    resource: Any
    owner: asyncio.Future | None
    error: BaseException
    commit_outcome: str | None
    cancellation: BaseException | None = None
    acknowledgment: Any = None


@dataclass(eq=False)
class _NativeResourceObservation:
    resource: Any
    observers: tuple[NativeExecutionObserver, ...]
    body_owner: asyncio.Task = field(default_factory=asyncio.current_task)
    connection: Any = None
    native_connection: Any = None
    body_owners: set[asyncio.Task] = field(default_factory=set)
    phases: set[_NativePhaseObservation] = field(default_factory=set)
    acknowledgments: tuple = ()

    def retain_body_owner(self):
        """An inherited lease's user is not necessarily its checkout task.

        Keep every participating task while it can still use this resource.
        Completed tasks are released immediately; finishing the physical
        checkout removes the remaining callbacks and roots as well.
        """
        owner = asyncio.current_task()
        if owner not in self.body_owners:
            self.body_owners.add(owner)
            owner.add_done_callback(self.body_owners.discard)

    def record_error(self, error):
        if all(observer._trusted_phases_only for observer in self.observers):
            return  # Foreign body attributes are not consulted for provenance.
        if error is not None and is_execution_control_error(error):
            for observer in self.observers:
                if not observer._trusted_phases_only:
                    observer._error = execution_terminal_error(observer._error, error)

    @contextmanager
    def phase(self, resource):
        """Reserve before task creation; the backend supplies its ACTUAL owner."""
        phase = _NativePhaseObservation(resource, observers=self.observers)
        self.phases.add(phase)
        try:
            yield phase
        except BaseException as error:
            # Classification reaches the inventory even if a provider catches
            # it inside a still-open operational-session body and hangs again.
            self.record_error(error)
            raise
        finally:
            self.phases.discard(phase)
            phase.classified.set_result(None)

    def finish(self, error):
        self.record_error(error)
        for owner in self.body_owners:
            owner.remove_done_callback(self.body_owners.discard)
        self.body_owners.clear()
        for observer in self.observers:
            observer._resources.discard(self)


class NativeExecutionObserver:
    """Read-only lifecycle inventory for one original, retiring consumer.

    Successful records disappear promptly. Pending resources stay rooted;
    actual control/debt evidence is aggregated by the canonical classifier,
    not evicted or replaced by a later successful operation.
    """

    def __init__(self, *, owner_task=None, trusted_phases_only=False):
        self._resources: set[_NativeResourceObservation] = set()
        self._error: BaseException | None = None
        self._closed = False
        self._owner_task = owner_task
        self._trusted_phases_only = trusted_phases_only
        self.native_failures: list[_NativeFailureFact] = []

    def _belongs_to_owner(self):
        if not self._trusted_phases_only:
            return True
        current, root = asyncio.current_task(), self._owner_task
        if root is None or root.done():
            return False
        ancestors = owned_task_join_ancestors(current)
        if current is root or root in ancestors:
            return True
        # Direct Task awaits have a physical waiter edge too; mere inherited
        # ContextVars do not enroll a long-lived SDK reader or supervisor.
        seen, pending = set(), [root]
        while pending:
            waiter = pending.pop()
            if waiter in seen:
                continue
            if waiter is current or waiter in ancestors:
                return True
            seen.add(waiter)
            awaited = getattr(waiter, "_fut_waiter", None)
            if isinstance(awaited, asyncio.Task):
                pending.append(awaited)
            elif type(awaited) is asyncio.tasks._GatheringFuture:
                pending.extend(awaited._children)
        # TaskGroup waits on a classification Future rather than directly on
        # each child. Verify the ACTUAL stdlib group's child and parent wait,
        # not a copied context or similarly named foreign attribute.
        pending, group_seen = [current, *ancestors], set()
        while pending:
            child = pending.pop()
            if child in group_seen:
                continue
            group_seen.add(child)
            if child in seen:
                return True
            for callback, _context in getattr(child, "_callbacks", None) or ():
                if type(callback) is BuiltinMethodType and callback.__name__ == "task_wakeup":
                    parent = callback.__self__
                    if (isinstance(parent, asyncio.Task) and not parent.done()
                        and getattr(parent, "_fut_waiter", None) is child):
                        pending.append(parent)
                elif (type(callback) is FunctionType
                      and callback.__code__.co_name == "_done_callback"
                      and any(callback.__code__ is code for code in asyncio.tasks.gather.__code__.co_consts)):
                    # Only the actual stdlib gather callback can introduce
                    # its outer Future; no foreign attribute or closure is
                    # inspected. The Future's real waiter then joins this
                    # child through the same reverse graph as a direct Task.
                    cells = dict(zip(callback.__code__.co_freevars, callback.__closure__ or ()))
                    cell = cells.get("outer")
                    outer = cell.cell_contents if cell is not None else None
                    if type(outer) is asyncio.tasks._GatheringFuture and child in outer._children:
                        pending.append(outer)
                if type(callback) is not MethodType:
                    continue
                group = callback.__self__
                if type(group) is not asyncio.TaskGroup:
                    continue
                parent = group._parent_task
                if (callback.__func__ is asyncio.TaskGroup._on_task_done
                    and group._entered and child in group._tasks
                    and parent is not None and not parent.done()):
                    # Entering the stdlib group already obliges this exact
                    # parent to join its child, including before __aexit__
                    # creates the completion Future. Follow nested groups
                    # transitively; copied context alone contributes no edge.
                    pending.extend((parent, *owned_task_join_ancestors(parent)))
        return False

    def _record_native_failure(self, resource, owner, error, commit_outcome, cancellation=None):
        if any(fact.resource is resource and fact.owner is owner for fact in self.native_failures):
            return
        self.native_failures.append(_NativeFailureFact(resource, owner, error, commit_outcome, cancellation))
        debt = ExecutionAuthorityError("original native phase failed")
        debt._kestrel_native_retirement_debts = ((resource, owner),)
        self._error = execution_terminal_error(
            self._error, error, debt, cancellation, retain_originals=True,
            isolate_control_originals=True,
        )
        self.close_admission()

    def close_admission(self):
        """Irreversibly refuse NEW resources, not cleanup of retained ones."""
        self._closed = True

    async def join_accepted(self) -> OwnedTaskOutcome[None]:
        """Join actual accepted exits/resets; never await arbitrary bodies.

        Open body/acquisition/adapter resources are explicit retirement debt.
        A body that has not produced a native closer is not a finite epilogue,
        and an instantaneously empty owner set is not successful retirement.
        The original provider/resource remains retained for reconciliation.
        """
        self.close_admission()
        cancellation = None
        observations = []
        while True:
            phases = tuple(phase for record in tuple(self._resources) for phase in tuple(record.phases))
            if not phases:
                break
            for phase in phases:
                if phase.owner is not None:
                    outcome = await await_owned_task(phase.owner)
                    observations.append(outcome.error)
                    if is_execution_control_error(outcome.error):
                        self._error = execution_terminal_error(
                            self._error, outcome.error, retain_originals=True,
                            isolate_control_originals=True,
                        )
                    cancellation = execution_terminal_error(cancellation, outcome.cancellation)
                # Classification must finish before the inventory is sampled
                # again. The resource remains rooted across subsequent phases;
                # this notification itself proves no physical retirement.
                classified = await await_owned_task(phase.classified)
                observations.append(classified.error)
                if is_execution_control_error(classified.error):
                    self._error = execution_terminal_error(
                        self._error, classified.error, retain_originals=True,
                        isolate_control_originals=True,
                    )
                cancellation = execution_terminal_error(cancellation, classified.cancellation)
        error = execution_terminal_error(
            self._error, *observations, retain_originals=True,
            isolate_control_originals=True,
        )
        unresolved = tuple(self._resources)
        if unresolved:
            debt = ExecutionAuthorityError("original native resource still held by a cancellable body")
            debt._kestrel_native_retirement_debts = tuple((record, None) for record in unresolved)
            for record in unresolved:
                for receipt in record.acknowledgments:
                    retain_acknowledged_execution_commit(debt, receipt)
            error = execution_terminal_error(
                error, debt, retain_originals=True, isolate_control_originals=True,
            )
        return OwnedTaskOutcome(None, error, cancellation)


_NATIVE_OBSERVERS: ContextVar[tuple[NativeExecutionObserver, ...]] = ContextVar(
    "kestrel_native_retirement_observers", default=(),
)


@contextmanager
def bind_native_execution_observer(observer: NativeExecutionObserver):
    """Add observation to the existing context; NEVER substitute its scopes."""
    if not isinstance(observer, NativeExecutionObserver):
        raise TypeError("native retirement requires an original observer")
    current = _NATIVE_OBSERVERS.get()
    token = _NATIVE_OBSERVERS.set(current if observer in current else (*current, observer))
    try:
        yield
    finally:
        _NATIVE_OBSERVERS.reset(token)


def _begin_native_resource(resource):
    """Backend-only reservation BEFORE an acquisition can suspend."""
    observers = tuple(observer for observer in _NATIVE_OBSERVERS.get() if observer._belongs_to_owner())
    if not observers:
        return None
    if any(observer._closed for observer in observers):
        raise ExecutionAuthorityError("original native retirement observation closed to new resources")
    record = _NativeResourceObservation(resource, observers)
    for observer in observers:
        observer._resources.add(record)
    return record


def _native_observation_requested():
    """Backend fast-path predicate; no owner, resource or authority is minted."""
    return any(observer._belongs_to_owner() for observer in _NATIVE_OBSERVERS.get())


async def _settle_native_observer_enrollment():
    """Let an eager constructor publish its REAL join edge before acquisition.

    TaskGroup membership, direct await/gather waiters and canonical owned-task
    joins appear only after eager Task construction returns. At most one
    cancellable scheduling turn precedes admission; copied context by itself
    still grants nothing. The backend then samples the same authenticated
    relationships and original live scopes, BEFORE selecting any native path.
    No foreign frame, global task scan, task factory patch or new owner exists.
    """
    if any(observer._trusted_phases_only
           and observer._owner_task is not None and not observer._owner_task.done()
           and not observer._belongs_to_owner() for observer in _NATIVE_OBSERVERS.get()):
        await asyncio.sleep(0)


def _require_native_observation_for_connection(connection):
    """Never attach an empty inventory to a borrowed, pre-existing lease.

    Binding must precede the original physical checkout. A connection already
    enrolled for every current observer retains its actual native lifecycle;
    an inherited operational lease cannot silently bypass that inventory.
    """
    for observer in _NATIVE_OBSERVERS.get():
        if not observer._belongs_to_owner():
            continue
        if observer._closed:
            raise ExecutionAuthorityError(
                "original native retirement observation closed to new work on its retained connection"
            )
        records = tuple(record for record in observer._resources if record.connection is connection)
        if not records:
            raise ExecutionAuthorityError(
                "bind native retirement observation before acquiring the original operational/transaction lease"
            )
        for record in records:
            record.retain_body_owner()
