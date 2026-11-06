"""Revocable turn ownership and one acknowledged boundary for tool effects."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
from typing import Any, Awaitable, Callable, Iterator

import rfc8785

from kestrel_sdk.resources import ResourceError, ResourceOutcomeUnknown
from kestrel_sdk.tools.result import ToolResult, ToolResultStatus
from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import (
    execution_terminal_error, require_execution_work,
)
from kestrel_sovereign.tool_effect_store import (
    ToolEffectAttempt, ToolEffectClaim, ToolEffectStore,
)
from kestrel_sovereign.turn_scope import turn_scoped


class ToolEffectJournalUnavailable(ResourceError):
    """A governed effect requires the original initialized durable journal."""


class ToolEffectPrivacyRefused(ResourceError):
    """The current privacy policy cannot retain the evidence needed for effects."""


class ToolEffectOutcomeUnconfirmed(ResourceOutcomeUnknown):
    """A turn observed an uncertain effect and must not narrate success."""


def logical_operation_key(value: object) -> str:
    """Encode opaque transport keys once without narrowing their wire domain."""
    try:
        encoded = rfc8785.dumps(["kestrel-tool-effect-client-key-v1", value])
    except rfc8785.CanonicalizationError as error:
        raise ValueError("Tool operation identity is not canonical JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def require_persistent_tool_effects(owner: Any) -> ToolEffectStore:
    """Refuse before approval prompts as well as immediately before claims."""
    privacy = owner.privacy_config
    if privacy is None or not privacy.allows_persistent_storage():
        raise ToolEffectPrivacyRefused("tool_effect_privacy_mode_refused")
    store = owner.tool_effect_store
    if not isinstance(store, ToolEffectStore) or not store.initialized:
        raise ToolEffectJournalUnavailable("tool_effect_journal_unavailable")
    return store


@dataclass(eq=False)
class ToolEffectTurn:
    """One live generation, shared with owned readers and nested tool calls."""

    owner: Any
    store: ToolEffectStore
    attempt: ToolEffectAttempt
    invocation_id: str
    live: bool = True
    closing: bool = False
    pending: set[asyncio.Future] = field(default_factory=set)
    unconfirmed: list[BaseException] = field(default_factory=list)

    def require_live(self, owner: Any) -> None:
        if self.owner is not owner or not self.live or self.closing:
            raise ToolEffectJournalUnavailable("tool_effect_attempt_retired")

    async def close(self, error: BaseException | None = None) -> None:
        """Join original claimants before closing admission; never cancel them."""
        self.closing = True
        terminal = error
        try:
            while self.pending:
                for completion in tuple(self.pending):
                    outcome = await await_owned_task(completion)
                    terminal = execution_terminal_error(
                        terminal, outcome.error, outcome.cancellation,
                        retain_originals=True,
                    )
            try:
                async with self.owner._get_durable_persistence_gate().shared_lock():
                    privacy = self.owner.privacy_config
                    if privacy is None:
                        raise ToolEffectJournalUnavailable("tool_effect_journal_unavailable")
                    if privacy.allows_persistent_storage():
                        await self.store.finish(self.attempt)
            except BaseException as finish_error:
                terminal = execution_terminal_error(
                    terminal, finish_error, retain_originals=True,
                )
            if self.unconfirmed:
                unknown = ToolEffectOutcomeUnconfirmed("tool_effect_outcome_unconfirmed")
                unknown.__cause__ = self.unconfirmed[0]
                terminal = execution_terminal_error(
                    terminal, unknown, retain_originals=True,
                )
        finally:
            self.live = False
        if terminal is not None:
            raise terminal


_current_turn: ContextVar[ToolEffectTurn | None] = ContextVar(
    "kestrel_tool_effect_turn", default=None,
)
_current_claim: ContextVar[ToolEffectClaim | None] = ContextVar(
    "kestrel_tool_effect_claim", default=None,
)


def current_tool_effect_turn(owner: Any) -> ToolEffectTurn | None:
    turn = _current_turn.get()
    return turn if turn is not None and turn.owner is owner else None


@contextmanager
def bind_tool_effect_turn(
    state: tuple[ToolEffectTurn | None, ToolEffectClaim | None],
) -> Iterator[None]:
    """Re-present the original revocable object; a snapshot cannot revive it."""
    turn_token = _current_turn.set(state[0])
    claim_token = _current_claim.set(state[1])
    try:
        yield
    finally:
        _current_claim.reset(claim_token)
        _current_turn.reset(turn_token)


turn_scoped(
    "tool_effect_attempt",
    variables=(_current_turn, _current_claim),
    capture=lambda owner: (current_tool_effect_turn(owner), _current_claim.get()),
    bind=bind_tool_effect_turn,
)


async def admit_tool_effect_turn(
    owner: Any, *, namespace: object, client_key: object, request: object,
    invocation_id: str,
) -> ToolEffectTurn | None:
    """Admit before model work, retaining no journal rows in volatile modes."""
    async with owner._get_durable_persistence_gate().shared_lock():
        privacy = owner.privacy_config
        if privacy is None:
            raise ToolEffectJournalUnavailable("tool_effect_journal_unavailable")
        if not privacy.allows_persistent_storage():
            return None
        store = require_persistent_tool_effects(owner)
        attempt = await store.admit(
            namespace=namespace, client_key=logical_operation_key(client_key),
            request=request,
        )
        return ToolEffectTurn(owner, store, attempt, invocation_id)


async def execute_tool_effect(
    owner: Any, *, registration: object, arguments: dict,
    execute: Callable[[], Awaitable[Any]],
) -> Any:
    """Claim durably, call once, settle truthfully, and retain every failure."""
    turn = current_tool_effect_turn(owner)
    if turn is None:
        raise ToolEffectJournalUnavailable("tool_effect_attempt_unavailable")
    turn.require_live(owner)
    completion = asyncio.get_running_loop().create_future()
    turn.pending.add(completion)
    try:
        async with owner._get_durable_persistence_gate().shared_lock():
            turn.require_live(owner)
            store = require_persistent_tool_effects(owner)
            if store is not turn.store:
                raise ToolEffectJournalUnavailable("tool_effect_journal_changed")
            require_execution_work(owner)
            claim = await store.claim(
                turn.attempt, registration=registration, arguments=arguments,
                parent=_current_claim.get(),
            )
            started = False
            result = None
            error = None
            token = _current_claim.set(claim)
            try:
                require_execution_work(owner)
                started = True
                result = await execute()
            except BaseException as execution_error:
                error = execution_error
            finally:
                _current_claim.reset(token)
            returned_error = (
                isinstance(result, ToolResult) and result.status == ToolResultStatus.ERROR
            ) or (
                isinstance(result, dict)
                and (result.get("success") is False or result.get("status") == "error")
            )
            state = (
                "NOT_EXECUTED" if not started else
                "UNCONFIRMED" if error is not None else
                "RETURNED_ERROR" if returned_error else "RETURNED_OK"
            )
            if started and error is not None:
                turn.unconfirmed.append(error)
            try:
                # Ordinary SQL authority is required. A revoked runtime may
                # refuse settlement; the original STARTED row then fences
                # replay. Cleanup never fabricates permission to write SQL.
                await store.settle(claim, state=state)
            except BaseException as settlement_error:
                turn.unconfirmed.append(settlement_error)
                error = execution_terminal_error(
                    error, settlement_error, retain_originals=True,
                )
            if error is not None:
                raise error
            return result
    finally:
        turn.pending.discard(completion)
        completion.set_result(None)
