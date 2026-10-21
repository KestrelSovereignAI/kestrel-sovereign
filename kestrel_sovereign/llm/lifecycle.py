"""Host custody for a provider whose agent never reached publication."""
import asyncio
from dataclasses import dataclass
from typing import Any

from kestrel_sovereign._async_ownership import OwnedTaskAdmission, await_owned_task, create_owned_task, owned_task_error, owned_task_observation_control
from kestrel_sovereign.execution_custody import (
    ExecutionCustody, _explicit_control_chain, bind_execution_cleanup, execution_terminal_error,
    is_execution_control_error,
)


def llm_close_observation_control(outcome):
    """Keep nonlocal native originals without retaining mere observer history.

    Withdrawal may already carry a separate original native receipt as its
    explicit cause. Keep that source, not the caller wrapper. If the actual
    cancellation itself is the sole native receipt, its facts cannot be
    discarded or replaced with a fabricated successful-retirement projection.
    """
    return owned_task_observation_control(outcome)


def start_llm_close_task(owner, key, operation, *, resource=None):
    """Own each native close once and harvest its canonical disposition.

    All clients, including rotated/unpublished private routes, share this
    lifecycle registry. Unknown native effects cannot acquire a new close
    owner just because their caller or route was withdrawn.
    """
    tasks = vars(owner).setdefault("_llm_close_tasks", {})
    refusals = vars(owner).setdefault("_llm_close_admission_errors", {})
    resources = vars(owner).setdefault("_llm_native_retirement_resources", {})
    if key in refusals:
        raise refusals[key]
    if key in tasks:
        return tasks[key]
    if resource is not None:
        # Register the original physical owner BEFORE Task admission. A None
        # success tombstone in tasks cannot represent an unaccepted close.
        resources[key] = resource
    admission = OwnedTaskAdmission()
    try:
        task = create_owned_task(
            operation(), name=f"llm-close:{key}", retain_originals=True,
            retirement_resource=resource, admission=admission,
        )
    except BaseException as error:
        task = admission.task
        if task is None:
            refusals[key] = error
        if resource is not None or is_execution_control_error(error):
            owner._llm_close_control = execution_terminal_error(
                vars(owner).get("_llm_close_control"), error, retain_originals=True,
                isolate_control_originals=True,
            )
        if task is None:
            raise
        # Canonical callback publication failed AFTER Task acceptance. Its
        # start gate prevented the physical operation; retain and join the
        # accepted original, whose resource debt cannot become a close ACK.
    tasks[key] = task

    def completed(done):
        harvest_llm_close_task(owner, key, done)

    try:
        task.add_done_callback(completed)
    except BaseException as error:
        # The Task is already accepted and inventoried. Keep the publication
        # receipt, but return that SAME Task so the caller physically joins
        # it and invokes the canonical harvester even without a callback.
        owner._llm_close_control = execution_terminal_error(
            vars(owner).get("_llm_close_control"), error, retain_originals=True,
            isolate_control_originals=True,
        )
    return task


def harvest_llm_close_task(owner, key, task, *, observation_error=None):
    """Reconcile one ORIGINAL terminal owner before releasing its registry key.

    Eager completion does not run queued callbacks before a direct join returns.
    Both the callback and the original joiner use this same idempotent harvest.
    """
    if not task.done():
        raise RuntimeError("cannot harvest an unfinished original LLM closer")
    # A join's callback failure is not the physical closer's result. Retain
    # its nonlocal control at the existing owner even if a queued completion
    # callback already released the successful physical resource below.
    # Caller withdrawal is deliberately NOT an input to this shared latch.
    if is_execution_control_error(observation_error):
        owner._llm_close_control = execution_terminal_error(
            vars(owner).get("_llm_close_control"), observation_error,
            retain_originals=True, isolate_control_originals=True,
        )
    error = owned_task_error(task)
    tasks = vars(owner).setdefault("_llm_close_tasks", {})
    if tasks.get(key) is not task:
        return error
    resources = vars(owner).setdefault("_llm_native_retirement_resources", {})
    resource = resources.get(key)
    if error is not None and resource is not None:
        # A native closer can return a shared commit/control receipt. Its
        # private retirement debt belongs on ONE canonical finite carrier,
        # not on that source or a new wrapper for every shutdown observer.
        retained = getattr(task, "_kestrel_llm_retirement_error", None)
        if retained is None:
            debt = RuntimeError("original LLM native retirement failed")
            debt._kestrel_native_retirement_debts = ((resource, task),)
            retained = execution_terminal_error(
                error, debt, retain_originals=True, isolate_control_originals=True,
            )
            task._kestrel_llm_retirement_error = retained
        error = retained
    if error is not None and is_execution_control_error(error):
        owner._llm_close_control = execution_terminal_error(
            vars(owner).get("_llm_close_control"), error, retain_originals=True,
            isolate_control_originals=True,
        )
    if error is None:
        tasks[key] = None
        resources.pop(key, None)
    elif key in ("preference-writers", "modality-writers"):
        # This is a writer join, not native retirement. A later wrapper may
        # join the SAME original writers, without repeating any native effect.
        tasks.pop(key)
    # Failed physical closers remain exact original owners, not retryable ACKs.
    return error


@dataclass(slots=True)
class UnpublishedLLMOwner:
    provider: Any
    scopes: tuple[ExecutionCustody, ...]
    task: asyncio.Task | None = None
    task_joins_native: bool = False
    control: BaseException | None = None
    admission_failure: BaseException | None = None
    retired: bool = False

    async def retire(self, *, join_native: bool = False) -> None:
        if self.admission_failure is not None:
            # The original provider is still owned, not a successful bounded
            # attempt that can be upgraded by submitting a different closer.
            raise self.admission_failure
        if self.retired:
            if self.control is not None:
                raise self.control
            return
        if self.task is None:
            self.task_joins_native = join_native
            admission = OwnedTaskAdmission()
            with bind_execution_cleanup(self.provider, self.scopes):
                try:
                    self.task = create_owned_task(
                        self._close(join_native=join_native), name="unpublished-provider:close",
                        retain_originals=True, retirement_resource=self.provider,
                        admission=admission,
                    )
                except BaseException as error:
                    self.task = admission.task
                    self.control = execution_terminal_error(
                        self.control, error, retain_originals=True,
                        isolate_control_originals=True,
                    )
                    if self.task is None:
                        self.admission_failure = self.control
                        raise self.admission_failure
        task, joins_native = self.task, self.task_joins_native
        outcome = await await_owned_task(task)
        if self.task is task and outcome.error is None and self.control is None:
            self.task = None
            self.retired = join_native and joins_native
        terminal = execution_terminal_error(
            outcome.error, outcome.cancellation, retain_originals=True,
            isolate_control_originals=True,
        )
        observed_control = llm_close_observation_control(outcome)
        if self.control is not None or is_execution_control_error(observed_control):
            self.control = execution_terminal_error(
                self.control, observed_control, retain_originals=True,
                isolate_control_originals=True,
            )
        if join_native and not joins_native:
            # A drain may encounter an in-flight bounded startup attempt. Its
            # completion is not the terminal host join; upgrade the SAME owner
            # after that attempt settles, retaining original control evidence.
            # The bounded attempt is physically done even if it failed. A host
            # join needs a new wrapper task around the SAME retained resource
            # owners, not recursive rejoining of this completed attempt.
            self.task = None
            try:
                await self.retire(join_native=True)
            except BaseException as error:
                raise execution_terminal_error(
                    terminal, error, retain_originals=True,
                    isolate_control_originals=True,
                )
            withdrawal = execution_terminal_error(
                *(error for error in _explicit_control_chain(terminal)
                  if isinstance(error, asyncio.CancelledError)),
                retain_originals=True, prefer_cancellation=True,
                isolate_control_originals=True,
            )
            if withdrawal is not None:
                raise withdrawal
            return
        if self.control is not None:
            raise execution_terminal_error(
                self.control, terminal, retain_originals=True,
                isolate_control_originals=True,
            )
        if terminal is not None:
            raise terminal

    async def _close(self, *, join_native: bool) -> None:
        """Startup is bounded; terminal host drain owns the physical join."""
        failure = None
        try:
            await self.provider.close()
        except BaseException as error:
            failure = error
        pending = getattr(type(self.provider), "has_pending_close_owners", None)
        retained = bool(callable(pending) and pending(self.provider))
        waiter = getattr(type(self.provider), "wait_for_close_completion", None)
        if join_native and callable(waiter):
            try:
                await waiter(self.provider)
            except BaseException as error:
                failure = execution_terminal_error(
                    failure, error, retain_originals=True,
                    isolate_control_originals=True,
                )
            if retained:
                # The first failure may only have reported bounded cleanup
                # debt. Query the SAME idempotent owner after physical join;
                # never discard an original authority/commit receipt.
                original_control = (
                    failure if failure is not None and is_execution_control_error(failure)
                    else execution_terminal_error(
                        *(error for error in _explicit_control_chain(failure)
                          if isinstance(error, asyncio.CancelledError)),
                        retain_originals=True, prefer_cancellation=True,
                        isolate_control_originals=True,
                    )
                )
                try:
                    await self.provider.close()
                except BaseException as error:
                    failure = execution_terminal_error(
                        original_control, error, retain_originals=True,
                        isolate_control_originals=True,
                    )
                else:
                    failure = original_control
        if failure is not None:
            raise failure


async def retire_unpublished_llm(
    owners: dict[str, UnpublishedLLMOwner], name: str, provider: Any,
    scopes: tuple[ExecutionCustody, ...],
    *, join_native: bool = False,
) -> None:
    """Register before awaiting; retain pending work or uncertain receipts."""
    owner = owners.get(name)
    if owner is None:
        owner = owners[name] = UnpublishedLLMOwner(provider, scopes)
    elif owner.provider is not provider:
        raise RuntimeError("unpublished provider cleanup still owns this name")
    try:
        await owner.retire(join_native=join_native)
    except BaseException as error:
        pending = getattr(type(provider), "has_pending_close_owners", None)
        if not is_execution_control_error(error) and not (
            callable(pending) and pending(provider)
        ):
            owners.pop(name, None)
        raise
    owners.pop(name, None)


async def drain_unpublished_llm_owners(owners: dict[str, UnpublishedLLMOwner]) -> None:
    """Attempt every independent owner; never replace a failed native close."""
    failures = []
    for name, owner in tuple(owners.items()):
        try:
            await retire_unpublished_llm(
                owners, name, owner.provider, owner.scopes, join_native=True,
            )
        except BaseException as error:
            failures.append(error)
    if failures:
        terminal = execution_terminal_error(
            *failures, retain_originals=True, isolate_control_originals=True,
        )
        assert terminal is not None
        raise terminal
