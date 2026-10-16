"""Host custody for a provider whose agent never reached publication."""
import asyncio
from dataclasses import dataclass
from typing import Any

from kestrel_sovereign._async_ownership import await_owned_task, create_owned_task, owned_task_error
from kestrel_sovereign.execution_custody import (
    ExecutionCustody, bind_execution_cleanup, execution_terminal_error,
    is_execution_control_error,
)


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
    try:
        task = create_owned_task(
            operation(), name=f"llm-close:{key}", retain_originals=True,
            retirement_resource=resource,
        )
    except BaseException as error:
        refusals[key] = error
        if resource is not None or is_execution_control_error(error):
            owner._llm_close_control = execution_terminal_error(
                vars(owner).get("_llm_close_control"), error, retain_originals=True,
            )
        raise
    tasks[key] = task

    def completed(done):
        error = owned_task_error(done)
        if error is not None and resource is not None:
            error._kestrel_native_retirement_debts = (
                *getattr(error, "_kestrel_native_retirement_debts", ()),
                (resource, done),
            )
        if error is not None and is_execution_control_error(error):
            owner._llm_close_control = execution_terminal_error(
                vars(owner).get("_llm_close_control"), error,
            )
        if tasks.get(key) is done:
            if error is None:
                tasks[key] = None
                resources.pop(key, None)
            elif key in ("preference-writers", "modality-writers"):
                # This task joins writers; it is not a native resource close.
                # A bounded join may have been cancelled while its original
                # writers remained live. A later wrapper can join those SAME
                # writers, without repeating a write or native close.
                tasks.pop(key)
            # ANY native failure is not proof that every transport
            # resource retired. Keep the exact failed owner; never replace it
            # with a speculative second close or advertise release. A control
            # receipt describes authority/effect disposition, not retirement.

    task.add_done_callback(completed)
    return task


@dataclass(slots=True)
class UnpublishedLLMOwner:
    provider: Any
    scopes: tuple[ExecutionCustody, ...]
    task: asyncio.Task | None = None
    task_joins_native: bool = False
    control: BaseException | None = None
    admission_failure: BaseException | None = None

    async def retire(self, *, join_native: bool = False) -> None:
        if self.admission_failure is not None:
            # The original provider is still owned, not a successful bounded
            # attempt that can be upgraded by submitting a different closer.
            raise self.admission_failure
        if self.task is None:
            self.task_joins_native = join_native
            with bind_execution_cleanup(self.provider, self.scopes):
                try:
                    self.task = create_owned_task(
                        self._close(join_native=join_native), name="unpublished-provider:close",
                        retain_originals=True, retirement_resource=self.provider,
                    )
                except BaseException as error:
                    self.admission_failure = error
                    self.control = execution_terminal_error(self.control, error, retain_originals=True)
                    raise
        task, joins_native = self.task, self.task_joins_native
        outcome = await await_owned_task(task)
        if self.task is task and outcome.error is None:
            self.task = None
        terminal = execution_terminal_error(outcome.error, outcome.cancellation)
        if terminal is not None and is_execution_control_error(terminal):
            self.control = execution_terminal_error(self.control, terminal)
        if join_native and not joins_native:
            # A drain may encounter an in-flight bounded startup attempt. Its
            # completion is not the terminal host join; upgrade the SAME owner
            # after that attempt settles, retaining original control evidence.
            # The bounded attempt is physically done even if it failed. A host
            # join needs a new wrapper task around the SAME retained resource
            # owners, not recursive rejoining of this completed attempt.
            self.task = None
            await self.retire(join_native=True)
            if outcome.cancellation is not None:
                raise outcome.cancellation
            return
        if self.control is not None:
            raise self.control
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
                failure = execution_terminal_error(failure, error)
            if retained:
                # The first failure may only have reported bounded cleanup
                # debt. Query the SAME idempotent owner after physical join;
                # never discard an original authority/commit receipt.
                original_control = (
                    failure if failure is not None and is_execution_control_error(failure)
                    else None
                )
                try:
                    await self.provider.close()
                except BaseException as error:
                    failure = execution_terminal_error(original_control, error)
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
        terminal = execution_terminal_error(*failures)
        assert terminal is not None
        raise terminal
