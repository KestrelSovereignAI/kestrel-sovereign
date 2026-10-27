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

from kestrel_sovereign._async_ownership import await_owned_task

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

    def cancel(self, task):
        if not task.done() and task not in self.cancellation_requested:
            self.cancellation_requested.add(task)
            task.cancel()

    async def close(self):
        self.revoke()
        withdrawal = None
        terminal = None
        for task in tuple(self.accepted_jobs):
            outcome = await await_owned_task(task)
            withdrawal = withdrawal or outcome.cancellation
            error = outcome.error
            if error is not None and not isinstance(error, (Exception, asyncio.CancelledError)):
                terminal = terminal or error
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
                purpose="human-metadata" if metadata else "human"):
            yield

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
            or purpose in {"human-metadata", "human-metadata-delivery"})
        if (not callable(check) or not valid_session
                or deadline.tzinfo is None or deadline <= self.clock() or (purpose not in {"human-metadata", "human-metadata-delivery"} and deadline > session.expires_at)):
            raise OperatorAuthorizationError("no originating visual action egress authority")
        key = self._key(session, action_id, purpose)
        if key in self._calls:
            raise OperatorAuthorizationError("originating visual action is already active")
        call = _ActiveCall(session, action_id, check, deadline, self.clock, asyncio.current_task())
        call.check()
        self._calls[key] = call
        try:
            yield
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
        if (call is None or session != call.session or task is None or task is call.origin
                or (task not in call.accepted_jobs and len(call.accepted_jobs) >= self.max_jobs)):
            raise OperatorAuthorizationError("egress does not belong to an active original action")
        call.check()
        call.accepted_jobs.add(task)
        call.jobs[task] = call.jobs.get(task, 0)+1
        try:
            remaining = (call.deadline-self.clock()).total_seconds()
            if purpose in {"human-delivery", "human-metadata-delivery"}:
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
            call.revoke()
        withdrawal = None
        for call in calls:
            try:
                await call.close()
            except asyncio.CancelledError as error:
                withdrawal = withdrawal or error
        if withdrawal is not None:
            raise withdrawal
