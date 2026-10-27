"""Closed native terminal-result profiles, never an ordinary SQL admission.

Only Core's two original producer owners can reach these fixed journal paths.
The existing native program, relation, connection and COMMIT seals remain in
force. A profile is not an application-supplied table, query or callback.
"""
from kestrel_sovereign.execution_custody import ExecutionAuthorityError


def _original_result_profile(journal):
    from kestrel_sovereign.signals.outcome_custody import _OriginalSignalOutcome
    from kestrel_sovereign.storage.original_action_result import _OriginalActionResult
    if type(journal) is _OriginalSignalOutcome:
        from kestrel_sovereign.signals.store import _OUTCOME_COLUMNS
        return 'signal_log', {name: ('INTEGER' if name == 'duration_ms' else 'TEXT') for name in _OUTCOME_COLUMNS}
    if type(journal) is _OriginalActionResult:
        return 'visual_computer_actions', {'owner':'TEXT', 'session_key':'TEXT', 'action_id':'TEXT', 'payload':'BLOB', 'pending':'BIGINT'}
    raise ExecutionAuthorityError('Exact original native result owner required')


def _require_original_result(journal, backend, *, admission=False):
    _original_result_profile(journal)
    if admission:
        journal.require_admission(backend)
    else:
        journal.require_original(backend)


async def _drain_original_results(backend):
    """Native close joins original result holds before connection/key retirement.

    This inventory owns resources, not permission. It cannot execute a query.
    The original producer must leave its lexical result unit before closing the
    database itself; joining itself would otherwise deadlock.
    """
    import asyncio
    from kestrel_sovereign._async_ownership import await_owned_operation
    from kestrel_sovereign.execution_custody import execution_observation_error
    backend._original_results_closing = True
    if getattr(backend, '_original_result_holds', None) and backend.owns_open_transaction:
        raise ExecutionAuthorityError('Cannot join original results while holding their database transaction')
    for result, closed in tuple(getattr(backend, '_original_result_holds', {}).items()):
        if result._owner_task is asyncio.current_task():
            raise ExecutionAuthorityError('Original result owner must leave before closing its database')
        async def join_original():
            await asyncio.shield(closed)
        outcome = await await_owned_operation(join_original(), retain_originals=True,
            retirement_resource=backend, name='original-result-close-join')
        error = execution_observation_error(outcome.error, outcome.cancellation)
        if error is not None:
            raise error
