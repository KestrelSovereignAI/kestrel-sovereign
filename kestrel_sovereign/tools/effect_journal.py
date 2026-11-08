"""Revocable turn ownership and one acknowledged boundary for tool effects."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
import inspect
from typing import Any, Awaitable, Callable, Iterator

from kestrel_sovereign.a2a.media import encode_inline_json

from kestrel_sdk.resources import ResourceConflict, ResourceError, ResourceOutcomeUnknown
from kestrel_sdk.tools.result import ToolResult, ToolResultStatus
from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.execution_custody import (
    execution_evidence_carriers, execution_observation_error, require_execution_work, is_tool_execution_terminal_error,
)
from kestrel_sovereign.tool_effect_store import (
    ToolEffectAttempt, ToolEffectClaim, ToolEffectStore,
)
from kestrel_sovereign.tools.execution_outcome import TOOL_OUTCOME_UNCONFIRMED
from kestrel_sovereign.turn_scope import turn_scoped
from kestrel_sovereign.agent.tool_allowlist import registry_generation


class ToolEffectJournalUnavailable(ResourceError):
    """A governed effect requires the original initialized durable journal."""


class ToolEffectPrivacyRefused(ResourceError):
    """The current privacy policy cannot retain the evidence needed for effects."""


class ToolEffectOutcomeUnconfirmed(ResourceOutcomeUnknown):
    """A turn observed an uncertain effect and must not narrate success."""


def _requires_original_terminal_error(error: BaseException | None) -> bool:
    return ((error is not None and not isinstance(error, Exception))
        or is_tool_execution_terminal_error(error)
        or any(issubclass(type(item), asyncio.CancelledError)
            for item in execution_evidence_carriers(error)))


def _record_uncertain_effect(turn, error: BaseException, *,
        message: str = 'tool_effect_outcome_unconfirmed') -> BaseException:
    """Retain the original cause once, independently of acknowledged claims."""
    if not _requires_original_terminal_error(error):
        original = error
        error = ToolEffectOutcomeUnconfirmed(message)
        error.__cause__ = original
    if not any(item is error for item in turn.unconfirmed):
        turn.unconfirmed.append(error)
    return error


def record_tool_effect_completion_failure(owner, error: BaseException) -> BaseException:
    """Fence failed delivery without rewriting an acknowledged effect/audit."""
    turn = current_tool_effect_turn(owner)
    if turn is None:
        raise ToolEffectJournalUnavailable('tool_effect_attempt_unavailable') from error
    return _record_uncertain_effect(turn, error, message=TOOL_OUTCOME_UNCONFIRMED)


def logical_operation_key(value: object) -> str:
    """Encode opaque transport keys once without narrowing their wire domain."""
    encoded = encode_inline_json(["kestrel-tool-effect-client-key-v2", value], sort_keys=True)
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
    pending: dict[asyncio.Future, asyncio.Task] = field(default_factory=dict)
    unconfirmed: list[BaseException] = field(default_factory=list)

    privacy_lock: Any = field(init=False, repr=False)
    privacy_scope: Any = field(init=False, repr=False)
    owns_privacy_scope: bool = field(init=False, repr=False)

    def __post_init__(self):
        self.privacy_lock = self.owner._get_privacy_transition_lock()
        self.privacy_scope, self.owns_privacy_scope = self.privacy_lock.begin_native_scope(self)

    def require_live(self, owner: Any) -> None:
        if self.owner is not owner or not self.live:
            raise ToolEffectJournalUnavailable("tool_effect_attempt_retired")
        if self.closing:
            from kestrel_sovereign.execution_custody import ExecutionAuthorityError
            try:
                self.privacy_lock.require_native_scope(self.privacy_scope,
                    retained_owners=self.pending.values())
            except ExecutionAuthorityError as error:
                raise ToolEffectJournalUnavailable("tool_effect_attempt_retired") from error

    def require_confirmed(self) -> None:
        """An uncertain effect cannot be followed by ordinary turn output."""
        if self.unconfirmed:
            raise ToolEffectOutcomeUnconfirmed("tool_effect_outcome_unconfirmed") from self.unconfirmed[0]

    async def close(self, error: BaseException | None = None) -> None:
        """Join original claimants before closing admission; never cancel them."""
        from kestrel_sovereign._async_ownership import bind_owned_coordinator
        with bind_owned_coordinator(self.privacy_lock):
            self.closing = True
            if self.owns_privacy_scope:
                self.privacy_scope.closing = True
            terminal = error
            try:
                while self.pending:
                    for completion, executor in tuple(self.pending.items()):
                        from kestrel_sovereign._async_ownership import owned_native_wait
                        async with owned_native_wait((executor,)):
                            outcome = await await_owned_task(completion)
                        self.pending.pop(completion, None)
                        terminal = execution_observation_error(
                            terminal, outcome.error, outcome.cancellation,
                            prefer_cancellation=isinstance(terminal, asyncio.CancelledError),
                        )
                try:
                    async with self.owner._get_durable_persistence_gate().shared_lock():
                        privacy = self.owner.privacy_config
                        if privacy is None:
                            raise ToolEffectJournalUnavailable("tool_effect_journal_unavailable")
                        if privacy.allows_persistent_storage():
                            await self.store.finish(self.attempt, unconfirmed=bool(self.unconfirmed))
                except BaseException as finish_error:
                    terminal = execution_observation_error(terminal, finish_error,
                        prefer_cancellation=isinstance(terminal, asyncio.CancelledError))
                # Keep original terminals and uncertain effects in one finite
                # envelope without decorating either source's graph. A synthetic
                # unknown classification is only for turns without a terminal.
                protected = _requires_original_terminal_error(terminal)
                if self.unconfirmed and protected:
                    terminal = execution_observation_error(terminal, *self.unconfirmed,
                        prefer_cancellation=isinstance(terminal, asyncio.CancelledError))
                if self.unconfirmed and not protected:
                    unknown = ToolEffectOutcomeUnconfirmed('tool_effect_outcome_unconfirmed')
                    unknown.__cause__ = execution_observation_error(terminal, *self.unconfirmed,
                        prefer_cancellation=isinstance(terminal, asyncio.CancelledError))
                    terminal = unknown
            finally:
                self.live = False
                if self.owns_privacy_scope:
                    try:
                        await self.privacy_lock.close_native_scope(self.privacy_scope, terminal)
                    except BaseException as retirement_error:
                        terminal = execution_observation_error(terminal, retirement_error,
                            prefer_cancellation=isinstance(terminal, asyncio.CancelledError))
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
    from kestrel_sovereign._async_ownership import bind_owned_coordinator
    with ExitStack() as routing:
        if state[0] is not None:
            routing.enter_context(bind_owned_coordinator(state[0].privacy_lock))
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
    turn.privacy_lock.require_native_scope(turn.privacy_scope)
    prepared = _current_registration.get()
    if prepared is not None and prepared.owner is not owner:
        prepared = None
    generation = expected_generation
    if generation is None:
        generation = prepared.generation if prepared is not None else registry_generation(owner)
    completion = asyncio.get_running_loop().create_future()
    turn.pending[completion] = asyncio.current_task()
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
            returned = False
            returned_error = False
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
                returned = True
                returned_error = (
                    isinstance(result, ToolResult) and result.status == ToolResultStatus.ERROR
                ) or (
                    isinstance(result, dict)
                    and (result.get("success") is False or result.get("status") == "error")
                )
            except BaseException as execution_error:
                error = execution_error
            finally:
                _current_claim.reset(token)
            known_conflict = (not returned and issubclass(type(error), ResourceConflict)
                and not is_tool_execution_terminal_error(error))
            state = (
                "NOT_EXECUTED" if not started else
                "RETURNED_ERROR" if known_conflict else
                "UNCONFIRMED" if error is not None else
                "RETURNED_ERROR" if returned_error else "RETURNED_OK"
            )
            if started and error is not None and not known_conflict:
                error = _record_uncertain_effect(turn, error)
            try:
                # Ordinary SQL authority is required. A revoked runtime may
                # refuse settlement; the original STARTED row then fences
                # replay. Cleanup never fabricates permission to write SQL.
                await store.settle(claim, state=state)
            except BaseException as settlement_error:
                turn.unconfirmed.append(settlement_error)
                error = execution_observation_error(error, settlement_error,
                    prefer_cancellation=isinstance(error, asyncio.CancelledError))
            if error is not None:
                raise error
            return result
    finally:
        turn.pending.pop(completion, None)
        completion.set_result(None)
