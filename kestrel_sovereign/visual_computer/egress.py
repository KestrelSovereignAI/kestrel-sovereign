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
    def _key(session, action_id):
        scope = session.scope
        return (scope.tenant_id, scope.principal_id, scope.agent_id, session.session_id,
                session.lease_id, session.driver_id, session.control_epoch, action_id)

    @asynccontextmanager
    async def activate(self, session, action_id, check, *, deadline):
        if (not callable(check) or session.state != "ready" or session.controller != "agent"
                or deadline.tzinfo is None or deadline <= self.clock() or deadline > session.expires_at):
            raise OperatorAuthorizationError("no originating visual action egress authority")
        key = self._key(session, action_id)
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
        call = self._calls.get(self._key(session, action_id))
        task = asyncio.current_task()
        if (call is None or session != call.session or task is None or task is call.origin
                or (task not in call.accepted_jobs and len(call.accepted_jobs) >= self.max_jobs)):
            raise OperatorAuthorizationError("egress does not belong to an active original action")
        call.check()
        call.accepted_jobs.add(task)
        call.jobs[task] = call.jobs.get(task, 0)+1
        try:
            remaining = (call.deadline-self.clock()).total_seconds()
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
