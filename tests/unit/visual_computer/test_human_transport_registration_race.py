"""Actual authenticated host ASGI/SQLite with controlled SDK pixels only."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys

import httpx
import pytest
from kestrel_sdk.computer_use.human_control import HumanControlLease
from kestrel_sovereign.visual_computer.human import HumanRecord
from tests.unit.visual_computer.test_host import AUTH, visual_host as visual_host
from tests.unit.visual_computer.test_human_http import setup
from tests.unit.visual_computer.test_human import original_human_rpc
from tests.unit.visual_computer.test_human_delivery import OuterTransport
from tests.unit.visual_computer.test_human_record_concurrency import _ciphertext


@pytest.mark.asyncio
async def test_original_revoke_ack_prevents_post_fence_transport_snapshot_delivery(
    visual_host, harness, monkeypatch, tmp_path,
):
    client, feature, extension, sid, base, body = await setup(visual_host, harness)
    lease = HumanControlLease.model_validate_json(json.dumps(body))
    runtime = feature.runtime
    human, repository = runtime.human, runtime.repository
    scope, cid = lease.request.scope, lease.request.control_id
    output = Path(os.environ.get('CURRENT_ROW_PROOF_DIR', tmp_path))
    fenced, drain, read_done, resume = (asyncio.Event() for _ in range(4))
    original_drain, original_record = runtime.egress.revoke_session, human._record
    original_observe = extension.observe
    original_transport_code = type(human).transport_delivery.__wrapped__.__code__
    rpc_children = []
    observe_owner = None
    revoke_task = frame_task = None
    path = base + '/human/' + cid + '/observation'
    transport = OuterTransport(visual_host[2], path)
    facts = {'controlled_sdk_pixels_only': True, 'physical_driver_ack': False,
        'corrected_terminal_row_guard_enabled': True}

    def persist():
        (output / 'transport-race-facts.json').write_text(json.dumps(facts, indent=2) + '\n')

    async def after_original_fence(selected_scope, selected_sid):
        assert selected_scope == scope and selected_sid == sid
        facts['after_priority_fence_serial'] = runtime._control_serials[runtime._key(scope, sid)]
        fenced.set()
        await drain.wait()
        await original_drain(selected_scope, selected_sid)

    async def observed_rpc(session, original_lease):
        nonlocal observe_owner
        call = original_human_rpc(runtime, session)
        observe_owner = call.origin
        value = await original_observe(session, original_lease)
        rpc_children.extend(call.accepted_jobs)
        return value

    async def after_original_transport_read(selected_scope, selected_sid, selected_cid, *, check, accept=None, update=None):
        caller_code = sys._getframe(1).f_code
        value = await original_record(selected_scope, selected_sid, selected_cid,
            check=check, accept=accept, update=update)
        if (caller_code is original_transport_code and asyncio.current_task() is observe_owner
                and accept is None and update is None and not read_done.is_set()):
            assert selected_scope == scope and selected_sid == sid and selected_cid == cid
            assert value.frame is not None and value.lease.state == 'active' and value.revocation is None
            check()
            assert not repository.db.owns_open_transaction
            assert not repository.db.backend._connection.in_transaction
            assert not runtime.egress._calls and rpc_children and all(task.done() for task in rpc_children)
            assert runtime._control_serials[runtime._key(scope, sid)] == facts['after_priority_fence_serial']
            facts['transport_read_barrier'] = {'actual_original_transport_code': True,
                'after_real_native_read_commit': True, 'transaction_open': False,
                'egress_calls': 0, 'original_rpc_children_joined': True,
                'snapshot_active': True, 'snapshot_has_frame': True,
                'serial': runtime._control_serials[runtime._key(scope, sid)]}
            persist()
            read_done.set()
            await resume.wait()
        return value

    monkeypatch.setattr(runtime.egress, 'revoke_session', after_original_fence)
    monkeypatch.setattr(extension, 'observe', observed_rpc)
    monkeypatch.setattr(human, '_record', after_original_transport_read)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=transport), base_url='http://test') as delayed:
        revoke_task = asyncio.create_task(client.post(base + '/human/' + cid + '/revoke', headers=AUTH,
            json={'request_id': 'original-transport-race-revoke', 'control_epoch': lease.request.control_epoch}))
        try:
            await asyncio.wait_for(fenced.wait(), 5)
            frame_task = asyncio.create_task(delayed.get(path, headers=AUTH))
            await asyncio.wait_for(read_done.wait(), 5)
            drain.set()
            revoked = await revoke_task
            assert revoked.status_code == 200 and revoked.json() == {'revoked': True}
            assert extension.lease.state == 'revoked'
            terminal_payload = await _ciphertext(repository, scope, sid, cid)
            (output / 'terminal-before-resume.ciphertext').write_bytes(terminal_payload)
            terminal = repository._open(repository._owner(scope), repository._address('private', sid, 'human', cid), terminal_payload, HumanRecord)
            assert terminal.revocation.request_id == 'original-transport-race-revoke'
            assert terminal.lease.state == 'revoked' and terminal.frame is None
            facts['original_revoke_before_delivery_registration'] = {'http': 200,
                'row_revoked': True, 'frame_absent': True,
                'ciphertext_sha256': hashlib.sha256(terminal_payload).hexdigest()}
            persist()
            resume.set()
            entered = asyncio.create_task(transport.entered.wait())
            try:
                done, _ = await asyncio.wait({frame_task, entered}, timeout=5,
                    return_when=asyncio.FIRST_COMPLETED)
                assert done, 'original request neither refused nor reached its send'
                admitted = transport.entered.is_set()
                accepted_delivery_children = [task for call in runtime.egress._calls.values()
                    for task in call.accepted_jobs if task is transport.child]
                facts['actual_outer_send_after_revoke_ack'] = {'entered': admitted,
                    'exact_child_registered_with_original_delivery_owner': bool(accepted_delivery_children)}
                transport.release.set()
            finally:
                if not entered.done():
                    entered.cancel()
                await asyncio.gather(entered, return_exceptions=True)
            response = await frame_task
            final_payload = await _ciphertext(repository, scope, sid, cid)
            (output / 'after-delivery.ciphertext').write_bytes(final_payload)
            facts['final'] = {'http': response.status_code, 'body': response.text,
                'actual_outer_body_bytes': [b.decode() for b in transport.bodies],
                'terminal_ciphertext_unchanged': final_payload == terminal_payload,
                'worker_protocol_state': extension.lease.state, 'worker_effects': extension.effects,
                'actual_delivery_child_joined': transport.child is None or transport.child.done(),
                'original_rpc_children_joined': all(task.done() for task in rpc_children),
                'egress_calls': len(runtime.egress._calls),
                'native_result_owners': len(repository._original_results),
                'native_backend_holds': len(repository.db.backend._original_result_holds)}
            persist()
            assert final_payload == terminal_payload
            # The response was already prepared: preserve the original unknown
            # transport outcome rather than inventing a known negative receipt.
            assert response.status_code == 503, response.text
            assert response.json() == {'error': {'code': 'visual_outcome_unknown'}}
            assert 'image_base64' not in response.text
        finally:
            drain.set()
            resume.set()
            transport.release.set()
            await asyncio.gather(*(task for task in (frame_task, revoke_task) if task is not None), return_exceptions=True)
            facts['original_outer_tasks_joined'] = all(task.done() for task in (frame_task, revoke_task) if task is not None)
            facts['original_rpc_children_joined_finally'] = all(task.done() for task in rpc_children)
            persist()
