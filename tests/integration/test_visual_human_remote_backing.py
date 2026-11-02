"""Genuine Human/Acquisition inverses on the one original native engine."""
import pytest

from kestrel_sovereign.execution_custody import ExecutionAuthorityError
from kestrel_sovereign.visual_computer.dispatch import original_human_dispatch
from tests.integration.test_visual_custody_backends import SCOPE
from tests.integration.test_visual_custody_backends import custody as custody
from tests.integration.test_visual_human_original_result_backends import prepared_human, dispatch


@pytest.mark.asyncio
async def test_genuine_human_native_result_cannot_borrow_acquisition_or_current_quote(custody, monkeypatch):
    store, _ = custody
    current, session, lease, frame, request, _ = await prepared_human(store)
    backend = store.db.backend
    native = backend._settle_original_action_result
    observed = []
    async def original_human_writer(owner):
        for operation in (backend._settle_original_acquisition_result,
                          backend._commit_current_runpod_quote, backend._commit_current_visual_step):
            with pytest.raises(ExecutionAuthorityError):
                await operation(owner)
        observed.append(owner)
        return await native(owner)
    monkeypatch.setattr(backend, '_settle_original_action_result', original_human_writer)
    with original_human_dispatch(session, lease, request, frame) as origin:
        async with store.original_human_action_result(SCOPE, session, request, guard=lambda: None) as owner:
            await dispatch(store, current, request, owner)
            fact = origin.refuse_unentered(RuntimeError('same original human driver never entered'))
            await owner.not_dispatched(origin, fact)
            assert observed == [owner] and owner.has_acknowledged_commit()
    row = await store.action(SCOPE, session.session_id, request.action.action_id)
    assert row.channel == 'human' and row.control_id == request.control_id
    assert row.receipt.outcome == 'denied' and row.dispatch_started
    assert not backend._original_result_holds and not store._original_results
