"""Revocable turn ownership and one acknowledged boundary for tool effects."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
import inspect
from typing import Any, Awaitable, Callable, Iterator

import rfc8785

from kestrel_sdk.resources import ResourceConflict, ResourceError, ResourceOutcomeUnknown, observed_outcome_unknown
from kestrel_sdk.tools.result import ToolResult, ToolResultStatus
from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import (
    execution_evidence_carriers, execution_terminal_error, is_execution_control_error,
    require_execution_work,
)
from kestrel_sovereign.tool_effect_store import (
    ToolEffectAttempt, ToolEffectClaim, ToolEffectStore,
)
from kestrel_sovereign.turn_scope import turn_scoped
from kestrel_sovereign.agent.tool_allowlist import registry_generation


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
    if owner is None:
        raise ToolEffectJournalUnavailable("tool_effect_journal_unavailable")
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
    generation: int | None = None
    live: bool = True
    closing: bool = False
    pending: set[asyncio.Future] = field(default_factory=set)
    unconfirmed: list[BaseException] = field(default_factory=list)

    def require_live(self, owner: Any) -> None:
        if (self.owner is not owner or not self.live
                or (self.closing and asyncio.current_task() not in self.pending)):
            raise ToolEffectJournalUnavailable("tool_effect_attempt_retired")

    def require_confirmed(self) -> None:
        """An uncertain effect cannot be followed by ordinary turn output."""
        if self.unconfirmed:
            raise ToolEffectOutcomeUnconfirmed("tool_effect_outcome_unconfirmed") from self.unconfirmed[0]

    async def close(self, error: BaseException | None = None) -> None:
        """Join original claimants before closing admission; never cancel them."""
        self.closing = True
        terminal = error
        try:
            while self.pending:
                for completion in tuple(self.pending):
                    outcome = await await_owned_task(completion)
                    self.pending.discard(completion)
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
            # Preserve an original terminal outcome and cooperative withdrawal.
            # A swallowed uncertain effect still needs a typed root failure;
            # adding that synthetic error to an already typed/control failure
            # would replace or wrap the very evidence the caller must retain.
            protected = (bool(observed_outcome_unknown(terminal))
                or is_execution_control_error(terminal)
                or any(issubclass(type(item), asyncio.CancelledError)
                    for item in execution_evidence_carriers(terminal)))
            if self.unconfirmed and protected:
                terminal = execution_terminal_error(
                    terminal, *self.unconfirmed, retain_originals=True,
                    prefer_cancellation=True,
                )
            if self.unconfirmed and not protected:
                unknown = ToolEffectOutcomeUnconfirmed('tool_effect_outcome_unconfirmed')
                unknown.__cause__ = execution_terminal_error(
                    terminal, *self.unconfirmed, retain_originals=True)
                terminal = unknown
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


@dataclass(frozen=True)
class _ToolEffectRegistration:
    owner: Any
    generation: int
    validate: Callable[[], None]


_current_registration: ContextVar[_ToolEffectRegistration | None] = ContextVar(
    'kestrel_tool_effect_registration', default=None,
)


@contextmanager
def bind_tool_effect_registration(owner, generation, validate):
    """Retain an original native provider through approval and owned handoff."""
    token = _current_registration.set(_ToolEffectRegistration(owner, generation, validate))
    try:
        yield
    finally:
        _current_registration.reset(token)


def current_tool_effect_state(owner: Any) -> tuple[ToolEffectTurn | None, ToolEffectClaim | None]:
    """Capture one owner's turn and its exact parent claim together."""
    turn = _current_turn.get()
    return (turn, _current_claim.get()) if turn is not None and turn.owner is owner else (None, None)


def current_tool_effect_turn(owner: Any) -> ToolEffectTurn | None:
    return current_tool_effect_state(owner)[0]


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
    capture=current_tool_effect_state,
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
    validate_registration: Callable[[], None] | None = None,
    expected_generation: int | None = None,
) -> Any:
    """Claim durably and revalidate the registration selected before hooks.

    Dispatchers retain ``expected_generation`` from selection through awaited
    approval. Component callers without hooks capture it here before admission.
    """
    turn = current_tool_effect_turn(owner)
    if turn is None:
        raise ToolEffectJournalUnavailable("tool_effect_attempt_unavailable")
    turn.require_live(owner)
    turn.require_confirmed()
    prepared = _current_registration.get()
    if prepared is not None and prepared.owner is not owner:
        prepared = None
    generation = expected_generation
    if generation is None:
        generation = prepared.generation if prepared is not None else registry_generation(owner)
    completion = asyncio.get_running_loop().create_future()
    turn.pending.add(completion)
    try:
        # Identity-writing tools reenter the privacy mutex. The transition
        # takes that mutex before the exclusive gate, so every effect must
        # preserve the same mutex -> gate order across its complete dispatch.
        async with owner._get_privacy_transition_lock(), owner._get_durable_persistence_gate().shared_lock():
            turn.require_live(owner)
            store = require_persistent_tool_effects(owner)
            if store is not turn.store:
                raise ToolEffectJournalUnavailable("tool_effect_journal_changed")
            require_execution_work(owner)
            claim = await store.claim(
                turn.attempt, registration=[generation, registration], arguments=arguments,
                parent=_current_claim.get(),
            )
            started = False
            result = None
            error = None
            token = _current_claim.set(claim)
            try:
                require_execution_work(owner)
                turn.require_confirmed()
                if registry_generation(owner) != generation:
                    raise ResourceConflict('Tool registration changed during effect admission')
                validators = ([prepared.validate] if prepared is not None else [])
                if validate_registration is not None:
                    validators.append(validate_registration)
                for validate in validators:
                    validation = validate()
                    if validation is not None:
                        if inspect.iscoroutine(validation):
                            validation.close()
                        raise TypeError('Tool registration validation must complete synchronously')
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
            known_conflict = (issubclass(type(error), ResourceConflict)
                and not observed_outcome_unknown(error)
                and not is_execution_control_error(error))
            state = (
                "NOT_EXECUTED" if not started else
                "RETURNED_ERROR" if known_conflict else
                "UNCONFIRMED" if error is not None else
                "RETURNED_ERROR" if returned_error else "RETURNED_OK"
            )
            if started and error is not None and not known_conflict:
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
