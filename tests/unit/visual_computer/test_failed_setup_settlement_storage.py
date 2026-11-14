"""Failed-setup recovery through real Host/SQLite and original Task ownership.

Daemon responses and the retirement BODY are scripted by the existing fixture.
This exercises durable custody and actual original close Task/Future joins;
it does not establish a physical container retirement or native worker ACK.
"""
from uuid import uuid4

import pytest

from kestrel_sovereign._async_ownership import await_owned_task
from kestrel_sovereign.visual_computer.repository import CustodyRecord
from kestrel_sovereign.visual_computer.runtime import VisualAction
from tests.unit.visual_computer.test_local_two_domain_admit import local_setup as local_setup
from tests.unit.visual_computer.test_two_domain_setup_publication import (
    scripted_daemon_maps as scripted_daemon_maps,
)
from tests.unit.visual_computer.test_two_domain_worker_profile import (
    measured_created_browser_profile as measured_created_browser_profile,
    measured_created_broker_profile as measured_created_broker_profile,
)


@pytest.mark.asyncio
async def test_storage_refusal_retains_original_setup_until_authorized_settlement(
    scripted_daemon_maps, monkeypatch,
):
    setup, state = scripted_daemon_maps
    runtime, repository = setup.runtime, setup.runtime.repository
    repository.max_requests = 1
    state['failure'] = 'internal-network'
    execute = repository.db.execute
    blocked = True
    refused_writes = []
    refusal = OSError('SCRIPTED failed-setup settlement storage refusal')

    async def refuse_after_original_close(sql, *args, **kwargs):
        if (blocked and state['closed'] and sql.lstrip().upper().startswith('UPDATE ')
                and 'visual_computer_requests' in sql):
            refused_writes.append(sql)
            raise refusal
        return await execute(sql, *args, **kwargs)

    monkeypatch.setattr(repository.db, 'execute', refuse_after_original_close)
    try:
        async with setup.bound() as context:
            request = setup.request(context, request_id='failed-setup-storage-recovery')
            with pytest.raises(Exception):
                await runtime.admit(context, request)

            resources = state['resources']
            assert resources is not None and state['closed'] == [resources]
            original = resources.original
            task = resources._close_admission.task
            assert task is not None and task is resources._original_close_admission.task
            assert getattr(task, '_kestrel_owned_admission', None) is resources._close_admission
            outcome = await await_owned_task(task)
            assert outcome.error is None and outcome.cancellation is None
            assert await original._finished is None
            assert resources._closed and resources._closing
            saved = await repository.get(request.scope, request_id=request.request_id)
            assert saved.phase == 'acquiring' and saved.lease is None and saved.session is None
            row = await repository.db.fetchone(
                'SELECT released, retired FROM visual_computer_requests WHERE owner = ? AND request_id = ?',
                (repository._owner(request.scope), request.request_id),
            )
            assert tuple(row) == (0, 0)
            assert runtime.egress._calls.get(original._key) is original, (
                'failed durable settlement discarded the SAME original cleanup custody'
            )
            assert refused_writes, 'completed cleanup never attempted durable settlement'
            commands_before_recovery = list(state['commands'])
    finally:
        blocked = False

    # A fresh genuine Host context supplies current authority. The old cleanup
    # Future is retained evidence, not a reusable authorization or create call.
    async with setup.bound() as current_context:
        result = await runtime.reconcile_request(current_context, request.request_id)
        assert result['released'] is True and result['acquired'] is None
        settled = await repository.get(request.scope, request_id=request.request_id)
        assert settled.phase == 'retired' and settled.retired_from == 'acquiring'
        assert settled.lease is None and settled.session is None
        assert runtime.egress._calls.get(original._key) is None
        assert setup.driver._connections == {}
        assert state['commands'] == commands_before_recovery
        assert state['closed'] == [resources]

        fresh = setup.request(current_context, request_id='capacity-after-storage-recovery')
        async with runtime.operation(current_context, VisualAction.ADMIT) as check:
            probe = CustodyRecord(request=fresh, session_id=uuid4().hex,
                                  last_activity_at=runtime._now())
            await repository.admit(fresh.scope, probe, check)
            await repository.retire_unacquired(fresh.scope, probe,
                                             at=runtime._now(), guard=check)
