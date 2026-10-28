"""Lexical egress custody borrowed from one original visual action RPC.

This is not a session grant resolver. A driver-created gateway job must present
its exact SDK session and current action ID while the original host operation
is active. It borrows that operation's existing privacy/current-authority check;
it never acquires a replacement context or gate on a background task. The
originating RPC joins every borrower before releasing those original leases.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TypeAlias

from kestrel_sdk.computer_use import ComputerSession
from kestrel_sdk.operator import OperatorAuthorizationError

from kestrel_sovereign._async_ownership import (
    await_owned_operation, await_owned_task, pending_owned_retirement_error,
    raise_owned_outcome,
)

CurrentCheck: TypeAlias = Callable[[], None]
EgressGuard: TypeAlias = Callable[[ComputerSession, str], AbstractAsyncContextManager[CurrentCheck]]


@dataclass
class _ActiveCall:
    session: ComputerSession
    action_id: str
    check_original: CurrentCheck
    deadline: datetime
    clock: Callable[[], datetime]
    origin: asyncio.Task
    active: bool = True
    jobs: dict[asyncio.Task, int] = field(default_factory=dict)
    accepted_jobs: set[asyncio.Task] = field(default_factory=set)
    cancellation_requested: set[asyncio.Task] = field(default_factory=set)
    sdk_owner: asyncio.Task | None = None

    def check(self):
        if not self.active or self.clock() >= self.deadline:
            raise OperatorAuthorizationError("originating visual action egress lease has ended")
        self.check_original()

    def revoke(self):
        # No await between closing admission and requesting every accepted
        # borrower's cancellation. A second closer cannot interrupt cleanup by
        # sending another cancellation to the same job.
        self.active = False
        for task in tuple(self.accepted_jobs):
            self.cancel(task)
        # Completed successful borrowers must not hide an unrelated SDK tail.
        # Retain original child cancellation even after its join: a later closer
        # must not interrupt SDK cleanup already unwinding that cancellation.
        if (self.sdk_owner is not None
                and not any(not task.done() for task in self.accepted_jobs)
                and not any(task in self.cancellation_requested for task in self.accepted_jobs)):
            self.cancel(self.sdk_owner)

    def cancel(self, task):
        if not task.done() and task not in self.cancellation_requested:
            self.cancellation_requested.add(task)
            task.cancel()

    def _bind_sdk_owner(self, task):
        if (task is None or task is self.origin or task.get_loop() is not asyncio.get_running_loop()
                or (self.sdk_owner is not None and self.sdk_owner is not task)):
            raise OperatorAuthorizationError("original SDK invocation owner differs")
        self.sdk_owner = task

    def _withdraw_sdk(self, task):
        # Canonical Task admission supplies this exact owner even when caller
        # withdrawal wins before its first step. Never forward cancellation
        # through the SDK parent into a native child's accepted cleanup.
        self._bind_sdk_owner(task)
        self.revoke()

    async def invoke_sdk(self, operation):
        async def invoke():
            self._bind_sdk_owner(asyncio.current_task())
            self.check()
            value = await operation()
            self.check()
            return value
        remaining = max(0, (self.deadline-self.clock()).total_seconds())
        async with asyncio.timeout(remaining):
            outcome = await await_owned_operation(invoke(), retain_originals=True,
                cancel_on_caller_cancellation=self._withdraw_sdk)
            # Keep the retained original timeout cancellation inside its own
            # timeout context, after the canonical native Task join.
            return raise_owned_outcome(outcome, operation="original human SDK invocation", retain_originals=True)

    def _require_external_closer(self):
        task = asyncio.current_task()
        if task is self.sdk_owner or task in self.accepted_jobs:
            self.active = False
            raise pending_owned_retirement_error(self, (task,),
                message="original SDK invocation cannot acknowledge its own retirement")

    async def close(self):
        self._require_external_closer()
        self.revoke()
        withdrawal = None
        terminal = None
        # Native children settle first. A parent which is already unwinding
        # their cancellation must never receive another cancellation during
        # its cleanup. Inline implementations still have one owned Task.
        tasks = (*tuple(self.accepted_jobs), *((self.sdk_owner,) if self.sdk_owner is not None else ()))
        for task in tasks:
            outcome = await await_owned_task(task)
            withdrawal = withdrawal or outcome.cancellation
            error = outcome.error
            from kestrel_sovereign.execution_custody import execution_cleanup_debts, execution_terminal_error
            if error is not None and (execution_cleanup_debts(error)
                    or not isinstance(error, (Exception, asyncio.CancelledError))):
                terminal = execution_terminal_error(terminal, error, retain_originals=True,
                    isolate_control_originals=True)
        if terminal is not None:
            raise terminal
        if withdrawal is not None:
            raise withdrawal


class ActiveCallEgress:
    """Finite active-call borrows, not durable or delegated visual authority."""

    def __init__(self, *, clock=None, max_jobs=16):
        if type(max_jobs) is not int or max_jobs <= 0:
            raise ValueError("active egress job bound must be positive")
        self.clock = clock or (lambda:datetime.now(UTC))
        self.max_jobs = max_jobs
        self._calls = {}

    @staticmethod
    def _key(session, action_id, purpose="action"):
        scope = session.scope
        return (scope.tenant_id, scope.principal_id, scope.agent_id, session.session_id,
                session.lease_id, session.driver_id, session.control_epoch, action_id, purpose)

    @asynccontextmanager
    async def activate(self, session, action_id, check, *, deadline):
        async with self._activate(session, action_id, check, deadline=deadline, purpose="action"):
            yield

    @asynccontextmanager
    async def activate_transfer(self, session, transfer_id, check, *, deadline):
        """Borrow the exact original paused workspace operation; grant nothing."""
        async with self._activate(session, transfer_id, check, deadline=deadline, purpose="workspace"):
            yield

    @asynccontextmanager
    async def activate_human(self, session, identifier, check, *, deadline, metadata=False):
        async with self._activate(session, identifier, check, deadline=deadline,
                purpose="human-metadata" if metadata else "human") as call:
            yield call

    @asynccontextmanager
    async def activate_human_delivery(self, session, identifier, check, *, deadline, metadata=False):
        async with self._activate(session, identifier, check, deadline=deadline,
                purpose="human-metadata-delivery" if metadata else "human-delivery"):
            yield

    @asynccontextmanager
    async def _activate(self, session, action_id, check, *, deadline, purpose):
        valid_session = ((purpose == "action" and session.state == "ready" and session.controller == "agent")
            or (purpose == "workspace" and session.state == "paused" and session.controller == "none")
            or (purpose in {"human", "human-delivery"} and session.state == "paused" and session.controller == "human")
            or purpose in {"human-metadata", "human-metadata-delivery", "session"})
        if (not callable(check) or not valid_session
                or deadline.tzinfo is None or deadline <= self.clock() or (purpose not in {"human-metadata", "human-metadata-delivery", "session"} and deadline > session.expires_at)):
            raise OperatorAuthorizationError("no originating visual action egress authority")
        key = self._key(session, action_id, purpose)
        if key in self._calls:
            raise OperatorAuthorizationError("originating visual action is already active")
        call = _ActiveCall(session, action_id, check, deadline, self.clock, asyncio.current_task())
        call.check()
        self._calls[key] = call
        try:
            yield call
        finally:
            try:
                await call.close()
            finally:
                self._calls.pop(key, None)

    @asynccontextmanager
    async def operation(self, session: ComputerSession, action_id: str):
        """Trusted driver factory callback; untrusted HTTP cannot call it."""
        async with self._operation(session, action_id, purpose="action") as check:
            yield check

    @asynccontextmanager
    async def transfer_operation(self, session: ComputerSession, transfer_id: str):
        """Purpose-separated borrow by an owned workspace transport task."""
        async with self._operation(session, transfer_id, purpose="workspace") as check:
            yield check

    @asynccontextmanager
    async def human_operation(self, session, identifier):
        async with self._operation(session, identifier, purpose="human") as check:
            yield check

    @asynccontextmanager
    async def human_status_operation(self, session, identifier):
        async with self._operation(session, identifier, purpose="human-metadata") as check:
            yield check

    @asynccontextmanager
    async def human_delivery_operation(self, session, identifier, *, metadata=False):
        async with self._operation(session, identifier,
                purpose="human-metadata-delivery" if metadata else "human-delivery") as check:
            yield check

    def cancel_human_delivery(self, session, identifier, task, *, metadata=False):
        """Forward withdrawal once through the same exact delivery owner."""
        purpose = "human-metadata-delivery" if metadata else "human-delivery"
        call = self._calls.get(self._key(session, identifier, purpose))
        if call is None or call.session != session:
            raise OperatorAuthorizationError("original human delivery owner is unavailable")
        call.cancel(task)

    @asynccontextmanager
    async def _operation(self, session, action_id, *, purpose):
        call = self._calls.get(self._key(session, action_id, purpose))
        task = asyncio.current_task()
        if (call is None or session != call.session or task is None or task is call.origin or task is call.sdk_owner
                or (task not in call.accepted_jobs and len(call.accepted_jobs) >= self.max_jobs)):
            raise OperatorAuthorizationError("egress does not belong to an active original action")
        call.check()
        call.accepted_jobs.add(task)
        call.jobs[task] = call.jobs.get(task, 0)+1
        try:
            remaining = (call.deadline-self.clock()).total_seconds()
            if purpose in {"human", "human-metadata", "human-delivery", "human-metadata-delivery", "session"}:
                # Expiry, Stop and HTTP withdrawal must share one cancellation
                # request. A second timer cancellation must not interrupt an
                # already accepted native transport's physical cleanup.
                timer = asyncio.get_running_loop().call_later(max(0, remaining), call.cancel, task)
                try:
                    call.check()
                    yield call.check
                    call.check()
                finally:
                    timer.cancel()
            else:
                async with asyncio.timeout(max(0, remaining)):
                    call.check()
                    yield call.check
                    call.check()
        finally:
            count = call.jobs.get(task, 0)-1
            if count > 0:
                call.jobs[task] = count
            else:
                call.jobs.pop(task, None)

    async def revoke_session(self, scope, session_id):
        """Priority safety path; close every exact owner's active RPC borrow."""
        owner = (scope.tenant_id, scope.principal_id, scope.agent_id, session_id)
        calls = [call for key, call in self._calls.items() if key[:4] == owner]
        for call in calls:
            call.active = False
        for call in calls:
            call._require_external_closer()
        for call in calls:
            call.revoke()
        withdrawal = None
        for call in calls:
            try:
                await call.close()
            except asyncio.CancelledError as error:
                withdrawal = withdrawal or error
        if withdrawal is not None:
            raise withdrawal
