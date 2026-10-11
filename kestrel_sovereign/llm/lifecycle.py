"""Host custody for a provider whose agent never reached publication."""
import asyncio
from dataclasses import dataclass
from typing import Any

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import (
    ExecutionCustody, bind_execution_cleanup, execution_terminal_error,
    is_execution_control_error,
)


@dataclass(slots=True)
class UnpublishedLLMOwner:
    provider: Any
    scopes: tuple[ExecutionCustody, ...]
    task: asyncio.Task | None = None
    control: BaseException | None = None

    async def retire(self) -> None:
        if self.task is None:
            with bind_execution_cleanup(self.provider, self.scopes):
                self.task = asyncio.create_task(
                    self.provider.close(), name="unpublished-provider:close",
                )
        outcome = await await_owned_task(self.task)
        self.task = None
        terminal = execution_terminal_error(outcome.error, outcome.cancellation)
        if terminal is not None and is_execution_control_error(terminal):
            self.control = execution_terminal_error(self.control, terminal)
        if self.control is not None:
            raise self.control
        if terminal is not None:
            raise terminal


async def retire_unpublished_llm(
    owners: dict[str, UnpublishedLLMOwner], name: str, provider: Any,
    scopes: tuple[ExecutionCustody, ...],
) -> None:
    """Register before awaiting; retain pending work or uncertain receipts."""
    owner = owners.get(name)
    if owner is None:
        owner = owners[name] = UnpublishedLLMOwner(provider, scopes)
    elif owner.provider is not provider:
        raise RuntimeError("unpublished provider cleanup still owns this name")
    try:
        await owner.retire()
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
            await retire_unpublished_llm(owners, name, owner.provider, owner.scopes)
        except BaseException as error:
            failures.append(error)
    if failures:
        terminal = execution_terminal_error(*failures)
        assert terminal is not None
        raise terminal
