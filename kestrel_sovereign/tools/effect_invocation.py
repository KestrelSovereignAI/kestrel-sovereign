"""Durable operation admission inside the canonical live invocation boundary."""

from contextlib import asynccontextmanager
from functools import wraps
import inspect

from kestrel_sovereign.agent.invocation import (
    current_invocation_id, current_invocation_provenance,
)
from kestrel_sovereign.auth import CallerRole, current_caller_context
from kestrel_sovereign.tools.effect_journal import (
    ToolEffectJournalUnavailable, admit_tool_effect_turn, bind_tool_effect_turn,
    current_tool_effect_state,
)


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


@asynccontextmanager
async def _journal_turn(owner, arguments, route):
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
        turn = await admit_tool_effect_turn(owner, namespace=_operation_namespace(),
            client_key=invocation_id, request=request, invocation_id=invocation_id)
        if turn is not None:
            turn.generation = generation
    elif turn is not None:
        turn.require_live(owner)
    with bind_tool_effect_turn((turn, None if root else parent_claim)):
        try:
            yield turn
        except BaseException as error:
            if root and turn is not None:
                await turn.close(error)
            raise
        else:
            if root and turn is not None:
                await turn.close()


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
