"""Durable operation admission inside the canonical live invocation boundary."""

from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import wraps
import inspect

from kestrel_sovereign.agent.invocation import (
    current_invocation_id, current_invocation_provenance,
)
from kestrel_sovereign.auth import CallerRole, current_caller_context
from kestrel_sovereign.execution_custody import execution_terminal_error
from kestrel_sovereign.tools.effect_journal import (
    ToolEffectJournalUnavailable, ToolEffectTurn, admit_tool_effect_turn, bind_tool_effect_turn,
    current_tool_effect_state,
)
from kestrel_sovereign.tool_effect_store import ToolEffectClaim


def _operation_namespace():
    caller = current_caller_context()
    if caller is not None and caller.role != CallerRole.ANONYMOUS:
        return ['caller', caller.role.value, caller.auth_method.value, caller.identity]
    provenance = current_invocation_provenance()
    if provenance is not None:
        if provenance.source_kind == 'http_request':
            return ['service', provenance.actor] if provenance.actor else ['anonymous', 'http_request']
        return [provenance.source_kind, provenance.source_locator]
    return ['internal', 'local']


@dataclass(frozen=True)
class _JournalInvocation:
    turn: ToolEffectTurn | None
    parent_claim: ToolEffectClaim | None
    root: bool

    def bind(self):
        return bind_tool_effect_turn((self.turn, self.parent_claim))

    async def close(self, error=None):
        with self.bind():
            if self.root and self.turn is not None:
                await self.turn.close(error)
            elif error is not None:
                raise error


async def _open_journal_turn(owner, arguments, route):
    invocation_id = current_invocation_id()
    if invocation_id is None:
        raise ToolEffectJournalUnavailable('tool_effect_invocation_unavailable')
    generation = owner._request_generation_for_current_task(invocation_id)
    if generation is None:
        raise ToolEffectJournalUnavailable('tool_effect_generation_unavailable')
    turn, parent_claim = current_tool_effect_state(owner)
    root = turn is None or (turn.invocation_id, turn.generation) != (invocation_id, generation)
    if root:
        provenance = current_invocation_provenance()
        request = dict(
            route=provenance.source_locator if provenance is not None else route,
            input=arguments['user_input'], session_id=arguments.get('session_id'),
            model_provider=arguments.get('model_override'),
            attachments=arguments.get('attachments'),
            include_memories=arguments.get('include_memories'),
        )
        if provenance is not None and provenance.input_digest is not None:
            request = dict(route=provenance.source_locator, input_digest=provenance.input_digest)
        turn = await admit_tool_effect_turn(owner, namespace=_operation_namespace(),
            client_key=invocation_id, request=request, invocation_id=invocation_id)
        if turn is not None:
            turn.generation = generation
    elif turn is not None:
        turn.require_live(owner)
    return _JournalInvocation(turn, None if root else parent_claim, root)


@asynccontextmanager
async def _journal_turn(owner, arguments, route):
    state = await _open_journal_turn(owner, arguments, route)
    with state.bind():
        try:
            yield state.turn
        except BaseException as error:
            await state.close(error)
        else:
            await state.close()


def journal_async_invocation(function):
    """Admit once after lifecycle admission and before the turn body runs.

    The outer invocation decorator owns registration and cancellation. Nested
    calls reuse only that exact live generation. This decorator never starts
    a replacement lifecycle or invents a new authority for journal cleanup.
    """
    signature = inspect.signature(function)

    @wraps(function)
    async def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        async with _journal_turn(args[0], bound.arguments, function.__name__):
            return await function(*args, **kwargs)

    return wrapped


def journal_async_generator_invocation(function):
    """Bind only while advancing or closing, including on a foreign driver."""
    signature = inspect.signature(function)

    @wraps(function)
    async def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        state = await _open_journal_turn(args[0], bound.arguments, function.__name__)
        iterator = function(*args, **kwargs)
        terminal = None
        try:
            while True:
                with state.bind():
                    try:
                        item = await anext(iterator)
                    except StopAsyncIteration:
                        break
                    if state.turn is not None:
                        state.turn.require_confirmed()
                yield item
        except BaseException as error:
            terminal = error
        finally:
            with state.bind():
                try:
                    await iterator.aclose()
                except BaseException as error:
                    terminal = execution_terminal_error(terminal, error, retain_originals=True)
                await state.close(terminal)

    return wrapped
