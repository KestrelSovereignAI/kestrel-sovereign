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
from typing import Any

from kestrel_sovereign._async_ownership import OwnedTaskOutcome, await_owned_task
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
        if error is not None and is_execution_control_error(error):
            for observer in self.observers:
                observer._error = execution_terminal_error(observer._error, error)

    @contextmanager
    def phase(self, resource):
        """Reserve before task creation; the backend supplies its ACTUAL owner."""
        phase = _NativePhaseObservation(resource)
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

    def __init__(self):
        self._resources: set[_NativeResourceObservation] = set()
        self._error: BaseException | None = None
        self._closed = False

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
        while True:
            phases = tuple(phase for record in tuple(self._resources) for phase in tuple(record.phases))
            if not phases:
                break
            for phase in phases:
                if phase.owner is not None:
                    outcome = await await_owned_task(phase.owner)
                    cancellation = execution_terminal_error(cancellation, outcome.cancellation)
                # Classification must finish before the inventory is sampled
                # again. The resource remains rooted across subsequent phases;
                # this notification itself proves no physical retirement.
                classified = await await_owned_task(phase.classified)
                cancellation = execution_terminal_error(cancellation, classified.cancellation)
        error = self._error
        unresolved = tuple(self._resources)
        if unresolved:
            debt = ExecutionAuthorityError("original native resource still held by a cancellable body")
            debt._kestrel_native_retirement_debts = tuple((record, None) for record in unresolved)
            for record in unresolved:
                for receipt in record.acknowledgments:
                    retain_acknowledged_execution_commit(debt, receipt)
            error = execution_terminal_error(error, debt)
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
    observers = _NATIVE_OBSERVERS.get()
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
    return bool(_NATIVE_OBSERVERS.get())


def _require_native_observation_for_connection(connection):
    """Never attach an empty inventory to a borrowed, pre-existing lease.

    Binding must precede the original physical checkout. A connection already
    enrolled for every current observer retains its actual native lifecycle;
    an inherited operational lease cannot silently bypass that inventory.
    """
    for observer in _NATIVE_OBSERVERS.get():
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
