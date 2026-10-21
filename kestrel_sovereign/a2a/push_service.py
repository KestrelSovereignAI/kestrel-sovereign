"""Managed publication and retirement of the native push resident worker."""

import asyncio
from collections.abc import Callable

from kestrel_sovereign._async_ownership import (
    OwnedTaskAdmission,
    await_owned_task,
    gather_owned,
    owned_task_error,
)
from kestrel_sovereign.a2a.push_worker import NativePushWorker
from kestrel_sovereign.execution_custody import (
    execution_terminal_error,
    is_execution_control_error,
)


class NativePushService:
    """Bind the original worker to its host's existing resident-task publisher.

    The publisher accepts a fresh coroutine, name and ``OwnedTaskAdmission``.
    Agent hosts supply their canonical runtime publisher; standalone owners
    supply ``create_owned_task`` and retain this service through shutdown.
    Acceptance is inventoried before publication can raise. No second worker,
    replacement task or fire-and-forget startup observer is created.
    """

    def __init__(self, worker: NativePushWorker, *, task_factory: Callable):
        if not isinstance(worker, NativePushWorker) or not callable(task_factory):
            raise TypeError(
                "Native push requires its worker and resident-task publisher"
            )
        self.worker = worker
        self._task_factory = task_factory
        self._admission = OwnedTaskAdmission()
        self._attempted = False
        self._stop_requested = False

    @property
    def task(self) -> asyncio.Task | None:
        """The exact accepted owner, including failed publication."""
        return self._admission.task

    def require_ready(self) -> NativePushWorker:
        """Refuse advertisement or execution after the resident owner fails."""
        task = self.task
        if task is not None and not task.done() and self.worker.ready:
            return self.worker
        if task is not None and task.done():
            error = owned_task_error(task)
            if error is not None:
                raise error
        raise RuntimeError("Native push has no ready managed resident worker")

    async def wait_ready(self) -> None:
        """Race the finite startup signal against the original owner's outcome."""
        task = self.task
        if task is None:
            raise RuntimeError("Native push service has not accepted a resident worker")

        async def observe_original_owner():
            # This observer borrows the resident; cancellation of the startup
            # observer must not become ownership of a new producer.
            outcome = await await_owned_task(task, defer_cancellation=False)
            error = execution_terminal_error(
                outcome.error, outcome.cancellation, retain_originals=True
            )
            if error is not None:
                raise error
            raise RuntimeError("Native push resident worker stopped during startup")

        await gather_owned(
            (self.worker.started.wait(), observe_original_owner()),
            return_when=asyncio.FIRST_COMPLETED,
            retain_originals=True,
            retirement_resource=self.worker,
            name_prefix="native-push-service-start",
        )
        self.require_ready()

    async def start(self) -> None:
        """Initialize readiness before accepting the host-owned resident worker."""
        if self._attempted:
            raise RuntimeError("Native push startup requires a fresh managed service")
        self._attempted = True
        try:
            await self.worker.outbox.initialize()
            operation = self.worker.run()
            try:
                received = self._task_factory(
                    operation, name="native-push-service", admission=self._admission
                )
            finally:
                if self.task is None:
                    operation.close()
            if not isinstance(received, asyncio.Task) or received is not self.task:
                raise RuntimeError(
                    "Native push publisher did not return its original accepted owner"
                )
            await self.wait_ready()
        except BaseException as original:
            try:
                await self.close()
            except BaseException as retirement:  # noqa: BLE001 - original physical retirement
                raise execution_terminal_error(
                    original, retirement, retain_originals=True
                )
            raise

    async def close(self) -> None:
        """Cancel and physically join the original resident before storage closes."""
        task = self.task
        if task is None:
            return
        if not task.done():
            self._stop_requested = True
            task.cancel("native push service shutdown")
        outcome = await await_owned_task(task)
        error = outcome.error
        if (
            self._stop_requested
            and isinstance(error, asyncio.CancelledError)
            and not is_execution_control_error(error)
        ):
            error = None
        error = execution_terminal_error(
            outcome.cancellation, error, retain_originals=True, prefer_cancellation=True
        )
        if error is not None:
            raise error
