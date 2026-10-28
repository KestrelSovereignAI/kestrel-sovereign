"""Original authenticated host + native SQLite, controlled SDK worker only.

This is a host record/protocol race. The existing controlled driver is not
Chromium, a Docker physical input worker, or physical-driver revoke ACK proof.
"""
import asyncio
import hashlib
import json
import os
from pathlib import Path

import pytest
from kestrel_sdk.computer_use.human_control import HumanControlLease
from kestrel_sovereign.visual_computer.human import HumanRecord
from tests.unit.visual_computer.test_host import AUTH
from tests.unit.visual_computer.test_host import visual_host as visual_host
from tests.unit.visual_computer.test_human_http import setup


@pytest.mark.asyncio
async def test_status_snapshot_cannot_erase_acknowledged_original_revocation(
    visual_host, harness, monkeypatch, tmp_path,
):
    client, feature, extension, sid, base, body = await setup(visual_host, harness)
    lease = HumanControlLease.model_validate_json(json.dumps(body))
    runtime = feature.runtime
    human, repository = runtime.human, runtime.repository
    scope = lease.request.scope
    cid = lease.request.control_id
    assert scope.tenant_id == 'did:test:kite'
    assert scope.principal_id == 'host-sovereign'
    assert scope.agent_id == 'did:test:kite'
    assert repository.db is harness.db and repository.db.backend_type == 'sqlite'
    owner = repository._owner(scope)
    key = repository._address('private', sid, 'human', cid)
    output = Path(os.environ.get('CURRENT_ROW_PROOF_DIR', tmp_path))
    paused, release = asyncio.Event(), asyncio.Event()
    status_returned = False
    status_owner = None
    stale_task = None
    status_children = []
    original_record = human._record
    original_status = extension.status
    facts = {'scope': scope.model_dump(), 'control_id': cid,
             'controlled_sdk_driver_only': True, 'physical_driver_ack': False}

    def persist():
        (output/'host-race-facts.json').write_text(json.dumps(facts, indent=2)+'\n')

    async def inspect_row(label):
        # Read-only native observation: a fresh HTTP status would mutate/repair
        # the same record and would make this counterexample dishonest.
        row = await repository.db.fetchone(
            'SELECT payload FROM visual_computer_private_records WHERE owner = ? AND record_key = ?',
            (owner, key),
        )
        assert row is not None
        payload = bytes(row[0])
        (output/(label+'.ciphertext')).write_bytes(payload)
        value = repository._open(owner, key, payload, HumanRecord)
        facts[label] = {
            'ciphertext_sha256': hashlib.sha256(payload).hexdigest(),
            'lease_state': value.lease.state if value.lease else None,
            'revocation_id': value.revocation.request_id if value.revocation else None,
            'tombstone': value.tombstone is not None,
            'frame': value.frame is not None,
        }
        persist()
        return value, payload

    async def observed_status(session, request):
        nonlocal status_returned, status_owner
        calls = [call for call in runtime.egress._calls.values()
                 if call.sdk_owner is asyncio.current_task() and call.session == session]
        assert len(calls) == 1
        # The SDK invocation now has its own joined Task. The second record
        # read still belongs to this exact call's original host RPC Task.
        status_owner = calls[0].origin
        value = await original_status(session, request)
        assert value.state == 'active' and value.request == lease.request
        # The production egress owner still retains this original status
        # borrow here. Its ordinary close, not this observer, joins it.
        status_children.extend(calls[0].accepted_jobs)
        status_returned = True
        return value

    async def after_original_read(selected_scope, selected_sid, selected_cid, *, check, accept=None, update=None):
        value = await original_record(selected_scope, selected_sid, selected_cid,
                                      check=check, accept=accept, update=update)
        if (asyncio.current_task() is status_owner and status_returned
                and accept is None and update is None and not paused.is_set()):
            assert selected_scope == scope and selected_sid == sid and selected_cid == cid
            assert value.lease.state == 'active' and value.revocation is None
            # EXACT status second internal read, after original native
            # transaction/COMMIT/guard/connection ownership have exited.
            assert repository.db.owns_open_transaction is False
            assert repository.db.backend._in_transaction is False
            assert repository.db.backend._connection.in_transaction is False
            assert not runtime.egress._calls
            assert status_children and all(child.done() for child in status_children)
            facts['barrier'] = {
                'after_original_status_second_read_commit': True,
                'owns_open_transaction': False,
                'sqlite_in_transaction': False,
                'active_egress_calls': 0,
                'original_status_children': len(set(status_children)),
                'serial': runtime._control_serials[runtime._key(scope, sid)],
                'stale_object_sha256': hashlib.sha256(value.model_dump_json().encode()).hexdigest(),
            }
            persist()
            paused.set()
            await release.wait()
        return value

    monkeypatch.setattr(extension, 'status', observed_status)
    monkeypatch.setattr(human, '_record', after_original_read)
    stale_task = asyncio.create_task(client.get(base+'/human/'+cid, headers=AUTH),
                                     name='original-human-status-stale-read')
    try:
        await asyncio.wait_for(paused.wait(), 5)
        before, _ = await inspect_row('before-revoke')
        assert before.lease.state == 'active' and before.revocation is None
        revoke_id = 'original-acknowledged-revoke'
        revoked = await client.post(base+'/human/'+cid+'/revoke', headers=AUTH,
            json={'request_id': revoke_id, 'control_epoch': lease.request.control_epoch})
        facts['revoke_http_status'] = revoked.status_code
        facts['revoke_http_body'] = revoked.json()
        assert revoked.status_code == 200 and revoked.json() == {'revoked': True}
        assert extension.lease.state == 'revoked'
        committed, committed_payload = await inspect_row('after-revoke-before-resume')
        assert committed.revocation.request_id == revoke_id
        assert committed.lease.state == 'revoked' and committed.lease.request == lease.request
        assert not committed.frame and committed.tombstone is None
        facts['original_revoke_protocol_ack'] = True
        facts['serial_after_revoke'] = runtime._control_serials[runtime._key(scope, sid)]
        persist()
        release.set()
        response = await stale_task
        facts['stale_status_http_status'] = response.status_code
        facts['stale_status_http_body'] = response.json()
        final, final_payload = await inspect_row('after-stale-status')
        facts['after_both'] = {
            'worker_protocol_lease': extension.lease.state,
            'worker_effects': extension.effects,
            'native_transaction_open': repository.db.backend._in_transaction,
            'original_result_holds': len(repository.db.backend._original_result_holds),
            'original_results': len(repository._original_results),
            'active_egress_calls': len(runtime.egress._calls),
            'original_status_children_all_joined': all(child.done() for child in status_children),
            'revoked_ciphertext_unchanged': final_payload == committed_payload,
        }
        persist()
        assert extension.lease.state == 'revoked' and extension.effects == 0
        assert not runtime.egress._calls and not repository._original_results
        assert not repository.db.backend._original_result_holds
        assert all(child.done() for child in status_children)
        # This is the intended invariant, with original raw/ciphertext facts
        # written BEFORE the assertion. No status/observation repair follows.
        assert response.status_code == 409, response.text
        assert final_payload == committed_payload
        assert final.revocation == committed.revocation, 'stale status erased acknowledged original host revocation'
        assert final.lease.state == 'revoked', 'stale status revived host lease metadata'
    finally:
        release.set()
        await asyncio.gather(stale_task, return_exceptions=True)
        facts['original_status_task_physically_joined'] = stale_task.done()
        facts['original_status_children_joined_in_finally'] = all(child.done() for child in status_children)
        persist()


async def _ciphertext(repository, scope, sid, cid):
    owner = repository._owner(scope)
    key = repository._address('private', sid, 'human', cid)
    row = await repository.db.fetchone(
        'SELECT payload FROM visual_computer_private_records WHERE owner = ? AND record_key = ?',
        (owner, key),
    )
    assert row is not None
    return bytes(row[0])


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['revocation', 'target', 'lease', 'frame', 'request', 'subclass'])
async def test_atomic_human_successor_preserves_original_terminal_row(harness, change):
    """Genuine current operation remains required; stale DTOs cannot erase facts."""
    from kestrel_sdk.resources import ResourceConflict
    from kestrel_sovereign.visual_computer.runtime import VisualAction
    from tests.unit.visual_computer.test_human import setup as sdk_setup

    h = harness
    context = h.context()
    with h.bind(context):
        session, active_lease, extension = await sdk_setup(h, context)
        await h.runtime.human.observe(context, session.session_id, 'human')
        async with h.runtime.operation(context, VisualAction.HUMAN_VIEW) as check:
            active = await h.runtime.human._record(h.runtime.scope(context), session.session_id, 'human', check=check)
        assert active.frame is not None
        await h.runtime.human.revoke(context, session.session_id, 'human', 'terminal', session.control_epoch)
        async with h.runtime.operation(context, VisualAction.HUMAN_VIEW) as check:
            scope = h.runtime.scope(context)
            terminal = await h.runtime.human._record(scope, session.session_id, 'human', check=check)
            before = await _ciphertext(h.repository, scope, session.session_id, 'human')
            attempted = terminal.model_copy(deep=True)
            if change == 'revocation':
                attempted.revocation = None
            elif change == 'target':
                attempted.revocation = attempted.revocation.model_copy(update={'control_epoch': session.control_epoch + 1})
            elif change == 'lease':
                attempted.lease = active_lease
            elif change == 'frame':
                attempted.frame = active.frame
            elif change == 'request':
                attempted.request = None
            else:
                class OtherHumanRecord(HumanRecord):
                    def _require_successor(self, original):
                        raise AssertionError('unissued schema callback must never run')
                attempted = OtherHumanRecord.model_validate(attempted.model_dump())
            with pytest.raises(ResourceConflict):
                await h.runtime.human._record(scope, session.session_id, 'human', check=check, update=attempted)
            assert await _ciphertext(h.repository, scope, session.session_id, 'human') == before
            assert await h.runtime.human._record(scope, session.session_id, 'human', check=check) == terminal
        assert extension.lease.state == 'revoked' and extension.effects == 0
        assert not h.repository._original_results and not h.db.backend._original_result_holds


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['drop', 'fenced_at', 'lease'])
async def test_atomic_preopen_tombstone_preserves_fact_and_allows_same_target_request_id(harness, change):
    from datetime import timedelta
    from kestrel_sdk.resources import ResourceConflict
    from kestrel_sovereign.visual_computer.runtime import VisualAction
    from tests.unit.visual_computer.test_human import setup as sdk_setup

    h = harness
    context = h.context()
    with h.bind(context):
        session, active_lease, extension = await sdk_setup(h, context)
        first = await h.runtime.human.revoke(context, session.session_id, 'unopened', 'first', session.control_epoch)
        repeated = await h.runtime.human.revoke(context, session.session_id, 'unopened', 'second', session.control_epoch)
        assert repeated.request.request_id == 'second'
        assert repeated.fenced_at == first.fenced_at
        async with h.runtime.operation(context, VisualAction.HUMAN_VIEW) as check:
            scope = h.runtime.scope(context)
            terminal = await h.runtime.human._record(scope, session.session_id, 'unopened', check=check)
            assert terminal.tombstone == repeated and terminal.request is None and terminal.lease is None
            before = await _ciphertext(h.repository, scope, session.session_id, 'unopened')
            attempted = terminal.model_copy(deep=True)
            if change == 'drop':
                attempted.tombstone = None
            elif change == 'fenced_at':
                attempted.tombstone = repeated.model_copy(update={'fenced_at': repeated.fenced_at + timedelta(seconds=1)})
            else:
                attempted.lease = active_lease
            with pytest.raises(ResourceConflict):
                await h.runtime.human._record(scope, session.session_id, 'unopened', check=check, update=attempted)
            assert await _ciphertext(h.repository, scope, session.session_id, 'unopened') == before
        assert extension.effects == 0


@pytest.mark.asyncio
async def test_nonterminal_human_snapshots_and_original_terminal_status_remain_valid(visual_host, harness):
    client, feature, extension, sid, base, lease = await setup(visual_host, harness)
    hp = base + '/human/control'
    active = await client.get(hp, headers=AUTH)
    assert active.status_code == 200 and active.json()['state'] == 'active'
    frame = await client.get(hp + '/observation', headers=AUTH)
    assert frame.status_code == 200 and frame.json()['frame']['observation_id'].startswith('human-')
    again = await client.get(hp, headers=AUTH)
    assert again.status_code == 200 and again.json()['state'] == 'active'
    for _ in range(2):
        retired = await client.post(hp + '/revoke', headers=AUTH,
            json={'request_id': 'same-original-revoke', 'control_epoch': lease['request']['control_epoch']})
        assert retired.status_code == 200 and retired.json() == {'revoked': True}
        status = await client.get(hp, headers=AUTH)
        assert status.status_code == 200 and status.json()['state'] == 'revoked'
    assert extension.effects == 0


@pytest.mark.asyncio
async def test_post_fence_observe_cannot_replace_terminal_row_or_deliver_frame(
    visual_host, harness, monkeypatch, tmp_path,
):
    """Controlled SDK pixels only; original HTTP response/joins are measured."""
    client, feature, extension, sid, base, body = await setup(visual_host, harness)
    lease = HumanControlLease.model_validate_json(json.dumps(body))
    runtime = feature.runtime
    human, repository = runtime.human, runtime.repository
    scope, cid = lease.request.scope, lease.request.control_id
    output = Path(os.environ.get('CURRENT_ROW_PROOF_DIR', tmp_path))
    fenced, drain, frame_ready, resume = (asyncio.Event() for _ in range(4))
    original_drain, original_record = runtime.egress.revoke_session, human._record
    original_observe = extension.observe
    children = []
    revoke_task = frame_task = None
    facts = {'controlled_sdk_pixels_only': True, 'physical_driver_ack': False}

    def persist():
        (output / 'post-fence-observe-facts.json').write_text(json.dumps(facts, indent=2) + '\n')

    async def after_original_fence(selected_scope, selected_sid):
        assert selected_scope == scope and selected_sid == sid
        facts['after_priority_fence_serial'] = runtime._control_serials[runtime._key(scope, sid)]
        fenced.set()
        await drain.wait()
        await original_drain(selected_scope, selected_sid)

    async def capture_original_child(session, original_lease):
        value = await original_observe(session, original_lease)
        for call in runtime.egress._calls.values():
            children.extend(call.accepted_jobs)
        return value

    async def before_stale_write(selected_scope, selected_sid, selected_cid, *, check, accept=None, update=None):
        if update is not None and update.frame is not None and not frame_ready.is_set():
            assert selected_scope == scope and selected_sid == sid and selected_cid == cid
            check()  # Genuine original checker admits this post-fence caller.
            assert runtime._control_serials[runtime._key(scope, sid)] == facts['after_priority_fence_serial']
            assert not repository.db.owns_open_transaction
            assert not repository.db.backend._connection.in_transaction
            assert not runtime.egress._calls and children and all(child.done() for child in children)
            facts['after_original_observe_rpc'] = {'transaction_open': False, 'active_egress_calls': 0,
                'original_children_joined': True, 'serial': runtime._control_serials[runtime._key(scope, sid)]}
            persist()
            frame_ready.set()
            await resume.wait()
        return await original_record(selected_scope, selected_sid, selected_cid,
            check=check, accept=accept, update=update)

    monkeypatch.setattr(runtime.egress, 'revoke_session', after_original_fence)
    monkeypatch.setattr(extension, 'observe', capture_original_child)
    monkeypatch.setattr(human, '_record', before_stale_write)
    revoke_task = asyncio.create_task(client.post(base + '/human/' + cid + '/revoke', headers=AUTH,
        json={'request_id': 'original-post-fence-revoke', 'control_epoch': lease.request.control_epoch}))
    try:
        await asyncio.wait_for(fenced.wait(), 5)
        frame_task = asyncio.create_task(client.get(base + '/human/' + cid + '/observation', headers=AUTH))
        await asyncio.wait_for(frame_ready.wait(), 5)
        drain.set()
        revoked = await revoke_task
        assert revoked.status_code == 200 and revoked.json() == {'revoked': True}
        assert extension.lease.state == 'revoked'
        terminal_payload = await _ciphertext(repository, scope, sid, cid)
        terminal = repository._open(repository._owner(scope), repository._address('private', sid, 'human', cid), terminal_payload, HumanRecord)
        assert terminal.revocation.request_id == 'original-post-fence-revoke'
        assert terminal.lease.state == 'revoked' and terminal.frame is None
        facts['revoke_before_frame_continuation'] = {'http': 200, 'row_revoked': True,
            'terminal_payload_sha256': hashlib.sha256(terminal_payload).hexdigest()}
        resume.set()
        response = await frame_task
        final_payload = await _ciphertext(repository, scope, sid, cid)
        facts['final'] = {'frame_http_status': response.status_code,
            'response_body': response.text, 'ciphertext_unchanged': final_payload == terminal_payload,
            'worker_protocol_state': extension.lease.state, 'worker_effects': extension.effects,
            'original_children_joined': all(child.done() for child in children)}
        persist()
        assert response.status_code == 409, response.text
        assert 'image_base64' not in response.text and 'private-human-pixels' not in response.text
        assert final_payload == terminal_payload
        assert extension.lease.state == 'revoked' and extension.effects == 0
    finally:
        drain.set()
        resume.set()
        await asyncio.gather(*(task for task in (frame_task, revoke_task) if task is not None), return_exceptions=True)
        facts['original_outer_tasks_joined'] = all(task.done() for task in (frame_task, revoke_task) if task is not None)
        facts['original_children_joined_finally'] = all(child.done() for child in children)
        persist()


@pytest.mark.asyncio
async def test_private_human_schema_never_invokes_an_unrelated_model_callback(harness):
    from kestrel_sdk.resources import ResourceConflict
    from kestrel_sovereign.visual_computer.runtime import VisualAction
    from tests.unit.visual_computer.test_human import setup as sdk_setup

    class UnrelatedModel:
        @classmethod
        def model_validate_json(cls, value):
            raise AssertionError('unrelated model callback must not run')

    h = harness
    context = h.context()
    with h.bind(context):
        session, _, _ = await sdk_setup(h, context)
        async with h.runtime.operation(context, VisualAction.HUMAN_VIEW) as check:
            with pytest.raises(ResourceConflict, match='schema'):
                await h.repository.private_record(h.runtime.scope(context), session.session_id,
                    'human', 'human', UnrelatedModel, guard=check)
