"""Genuine registered HTTP/auth/SQLite custody; controlled SDK, not Chromium."""
import asyncio
from datetime import UTC, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path

import pytest
from kestrel_sdk.computer_use.human_control import HumanControlLease, HumanControlRevocationTombstone
from kestrel_sovereign.visual_computer.human import HumanRecord
from tests.unit.visual_computer.test_host import AUTH
from tests.unit.visual_computer.test_host import visual_host as visual_host
from tests.unit.visual_computer.test_human_http import setup


async def observed_original_human(repository, lease, *, cid=None):
    # Existing committed concurrency-test pattern: native READ ONLY observer.
    # No row insertion/update, status repair, context or current-check creation.
    owner = repository._owner(lease.request.scope)
    key = repository._address('private', lease.request.session_id, 'human', cid or lease.request.control_id)
    row = await repository.db.fetchone(
        'SELECT payload FROM visual_computer_private_records WHERE owner = ? AND record_key = ?',
        (owner, key))
    assert row is not None
    payload = bytes(row[0])
    return repository._open(owner, key, payload, HumanRecord), hashlib.sha256(payload).hexdigest()


@pytest.mark.asyncio
@pytest.mark.parametrize('sequence', ['accepted-R', 'unaccepted-R'])
async def test_old_human_revoke_does_not_fence_handed_back_agent(
    visual_host, harness, monkeypatch, tmp_path, sequence,
):
    client, feature, extension, sid, base, body = await setup(visual_host, harness)
    lease = HumanControlLease.model_validate_json(json.dumps(body))
    runtime, repository = feature.runtime, feature.runtime.repository
    assert repository.db is harness.db and repository.db.backend_type == 'sqlite'
    hp = base + '/human/' + lease.request.control_id
    original_request = {'request_id': 'original-client-R', 'control_epoch': lease.request.control_epoch}
    out = Path(os.environ.get('CURRENT_ROW_PROOF_DIR', tmp_path))
    facts = {'sequence': sequence, 'controlled_SDK_only': True, 'physical_browser_proof': False}
    entered, finish = asyncio.Event(), asyncio.Event()
    cancelled, sdk_tasks, calls = [], [], []
    late_agent = revoke_task = None
    original_observe = harness.adapter.observe
    after_human = before_human = None
    fresh = acted = after_status = None
    late_outcome = None
    cleanup_response = None

    async def held_original_agent_observe(session):
        task = asyncio.current_task()
        exact = [call for call in runtime.egress._calls.values()
                 if call.session == session and task in call.accepted_jobs]
        assert len(exact) == 1
        assert task is not exact[0].origin
        sdk_tasks.append(task)
        calls.append(exact[0])
        entered.set()
        try:
            await finish.wait()
        except asyncio.CancelledError as error:
            cancelled.append(error)
            raise
        return await original_observe(session)

    try:
        if sequence == 'accepted-R':
            first = await client.post(hp + '/revoke', headers=AUTH, json=original_request)
            assert first.status_code == 200, first.text
        # The other case models a preissued client request that never arrived.
        # Do not pretend it was accepted or invent a transport acknowledgement.
        resumed = await client.post(base + '/handback', headers=AUTH, json={
            'request_id': 'genuine-original-handback', 'session_id': sid,
            'expected_control_epoch': lease.request.control_epoch, 'operation': 'resume',
            'share_screen': True})
        assert resumed.status_code == 200, resumed.text
        assert resumed.json()['controller'] == 'agent'
        assert resumed.json()['control_epoch'] == lease.request.control_epoch + 1
        before_human, before_cipher = await observed_original_human(repository, lease)
        assert before_human.request == lease.request
        assert before_human.lease.state == 'revoked' and before_human.frame is None
        assert before_human.revocation is not None
        if sequence == 'accepted-R':
            assert before_human.revocation.request_id == original_request['request_id']
        else:
            assert before_human.revocation.request_id != original_request['request_id']
        healthy = await client.get(base + '/observation', headers=AUTH)
        assert healthy.status_code == 200, healthy.text

        # Only a normal SDK body is held; actual original host admission and
        # its canonical Task/borrower registration/checker are unchanged.
        monkeypatch.setattr(harness.adapter, 'observe', held_original_agent_observe)
        late_agent = asyncio.create_task(client.get(base + '/observation', headers=AUTH))
        await asyncio.wait_for(entered.wait(), 5)
        assert sdk_tasks[0] in calls[0].accepted_jobs and not sdk_tasks[0].done()
        revoke_task = asyncio.create_task(client.post(hp + '/revoke', headers=AUTH, json=original_request))
        replay = await asyncio.wait_for(asyncio.shield(revoke_task), 5)
        facts['original_revoke_reply_status'] = replay.status_code
        facts['original_revoke_reply_body'] = replay.json()
        facts['original_agent_borrower_cancelled'] = bool(cancelled)
        finish.set()
        late_outcome, = await asyncio.gather(late_agent, return_exceptions=True)
        monkeypatch.setattr(harness.adapter, 'observe', original_observe)
        after_human, after_cipher = await observed_original_human(repository, lease)
        after_status = await client.get(base, headers=AUTH)
        assert after_status.status_code == 200, after_status.text
        fresh = await client.get(base + '/observation', headers=AUTH)
        if fresh.status_code == 200:
            frame = fresh.json()['frame']
            acted = await client.post(base + '/actions', headers=AUTH, json={
                'action_id': 'fresh-agent-after-old-human-' + sequence,
                'session_id': sid, 'observation_id': frame['observation_id'],
                'control_epoch': frame['control_epoch'],
                'not_after': (datetime.now(UTC) + timedelta(seconds=5)).isoformat(),
                'action': {'kind': 'click', 'x': 1, 'y': 1}})
        facts.update(before_cipher_sha256=before_cipher, after_cipher_sha256=after_cipher,
            original_human_request_preserved=after_human.request == before_human.request,
            original_human_revocation_preserved=after_human.revocation == before_human.revocation,
            original_tombstone_preserved=after_human.tombstone == before_human.tombstone,
            status_after_stale=after_status.json(), fresh_agent_observe=fresh.status_code,
            fresh_agent_act=acted.status_code if acted is not None else None,
            prior_original_agent_response=getattr(late_outcome, 'status_code', None),
            prior_original_agent_error_type=type(late_outcome).__name__,
            original_captured_sdk_Task_count=len(sdk_tasks))
    finally:
        # Every actual returned Task is retained. Open only our real Event;
        # never clear a runtime fence, fabricate a response, or cancel again.
        finish.set()
        monkeypatch.setattr(harness.adapter, 'observe', original_observe)
        originals = [task for task in (late_agent, revoke_task, *sdk_tasks) if task is not None]
        await asyncio.gather(*originals, return_exceptions=True)
        facts['all_original_Tasks_joined'] = all(task.done() for task in originals)
        cleanup_response = await client.post(base + '/release', headers=AUTH)
        facts['real_foreground_release_cleanup_status'] = cleanup_response.status_code
        facts['egress_empty_after_original_release'] = not runtime.egress._calls
        facts['original_result_holds_after_release'] = len(harness.db.backend._original_result_holds)
        (out / ('old-human-revoke-' + sequence + '.json')).write_text(json.dumps(facts, indent=2) + '\n')

    # Original cleanup completes BEFORE the deliberate behavioral oracle.
    assert cleanup_response.status_code == 200, cleanup_response.text
    assert facts['all_original_Tasks_joined'] and not runtime.egress._calls
    assert not harness.db.backend._original_result_holds
    assert fresh is not None and fresh.status_code == 200, facts
    assert acted is not None and acted.status_code == 200, facts
    assert acted.json()['outcome'] == 'completed', facts
    assert not cancelled and getattr(late_outcome, 'status_code', None) == 200, facts
    assert facts['original_revoke_reply_status'] == (200 if sequence == 'accepted-R' else 409), facts
    assert facts['original_human_request_preserved'] and facts['original_human_revocation_preserved'] and facts['original_tombstone_preserved'], facts
    assert extension.effects == 0 and harness.adapter.effects == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('control', ['stop', 'takeover'])
async def test_pending_human_revoke_cannot_adopt_later_control_serial(
    visual_host, harness, monkeypatch, tmp_path, control,
):
    client, feature, extension, sid, base, body = await setup(visual_host, harness)
    lease = HumanControlLease.model_validate_json(json.dumps(body))
    runtime, repository = feature.runtime, feature.runtime.repository
    hp = base + '/human/' + lease.request.control_id
    key = runtime._key(lease.request.scope, sid)
    request_id = 'pending-original-human-revoke'
    entered, resume = asyncio.Event(), asyncio.Event()
    original_record = runtime.human._record
    original_revoke = extension.revoke
    held = False
    original_tasks, original_rpc_requests = [], []
    revoke_task = reply = cleanup = None
    facts = {'control': control, 'controlled_SDK_only': True, 'physical_browser_proof': False}
    out = Path(os.environ.get('CURRENT_ROW_PROOF_DIR', tmp_path))

    async def after_original_intent_commit(*args, **kwargs):
        nonlocal held
        value = await original_record(*args, **kwargs)
        proposed = kwargs.get('update')
        if (not held and proposed is not None and proposed.revocation is not None
                and proposed.revocation.request_id == request_id):
            # Original encrypted row transaction/COMMIT has actually returned.
            # Hold only the calling test schedule, never a SQL or authority door.
            assert harness.db.backend._in_transaction is False
            held = True
            original_tasks.append(asyncio.current_task())
            entered.set()
            await resume.wait()
        return value

    async def observe_original_rpc(session, request):
        original_rpc_requests.append(request)
        return await original_revoke(session, request)

    monkeypatch.setattr(runtime.human, '_record', after_original_intent_commit)
    monkeypatch.setattr(extension, 'revoke', observe_original_rpc)
    try:
        revoke_task = asyncio.create_task(client.post(hp + '/revoke', headers=AUTH,
            json={'request_id': request_id, 'control_epoch': lease.request.control_epoch}))
        await asyncio.wait_for(entered.wait(), 5)
        assert not original_rpc_requests
        control_reply = await client.post(base + '/' + control, headers=AUTH, json={
            'request_id': 'genuine-later-' + control, 'session_id': sid,
            'expected_control_epoch': lease.request.control_epoch,
            'operation': 'pause' if control == 'stop' else 'takeover'})
        assert control_reply.status_code == 200, control_reply.text
        if control == 'stop':
            assert control_reply.json() == {'session_id': sid, 'stopped': True}
        status_reply = await client.get(base, headers=AUTH)
        assert status_reply.status_code == 200, status_reply.text
        current_session = status_reply.json()['session']
        assert current_session['control_epoch'] > lease.request.control_epoch
        assert current_session['controller'] == ('none' if control == 'stop' else 'human')
        serial = runtime._control_serials[key]
        assert key in runtime._fences
        terminal, ciphertext = await observed_original_human(repository, lease)
        assert terminal.lease.state == 'revoked' and terminal.revocation.request_id == request_id
        assert len(original_rpc_requests) == 1
        resume.set()
        reply = await asyncio.wait_for(asyncio.shield(revoke_task), 5)
        after, after_ciphertext = await observed_original_human(repository, lease)
        facts.update(stale_reply=reply.status_code, control_reply=control_reply.status_code,
            serial_after_control=serial, serial_after_stale=runtime._control_serials[key],
            genuine_control_fence_retained=key in runtime._fences,
            terminal_row_unchanged=after == terminal and after_ciphertext == ciphertext,
            original_rpc_count=len(original_rpc_requests), human_effects=extension.effects)
    finally:
        resume.set()
        originals = [t for t in (revoke_task, *original_tasks) if t is not None]
        await asyncio.gather(*originals, return_exceptions=True)
        monkeypatch.setattr(runtime.human, '_record', original_record)
        monkeypatch.setattr(extension, 'revoke', original_revoke)
        facts['all_original_Tasks_joined'] = all(t.done() for t in originals)
        cleanup = await client.post(base + '/release', headers=AUTH)
        facts['real_foreground_release_cleanup_status'] = cleanup.status_code
        facts['egress_empty_after_original_release'] = not runtime.egress._calls
        facts['original_result_holds_after_release'] = len(harness.db.backend._original_result_holds)
        (out / ('pending-human-revoke-' + control + '.json')).write_text(json.dumps(facts, indent=2) + '\n')

    assert cleanup.status_code == 200, cleanup.text
    assert facts['all_original_Tasks_joined'] and not runtime.egress._calls
    assert not harness.db.backend._original_result_holds
    assert reply is not None and reply.status_code == 409, facts
    assert facts['serial_after_stale'] == facts['serial_after_control'], facts
    assert facts['genuine_control_fence_retained'] and facts['terminal_row_unchanged'], facts
    assert facts['original_rpc_count'] == 1 and extension.effects == 0, facts


@pytest.mark.asyncio
@pytest.mark.parametrize('mismatch', ['epoch', 'accepted-request-id'])
async def test_invalid_original_human_target_refuses_before_fence_or_rpc(
    visual_host, harness, monkeypatch, tmp_path, mismatch,
):
    client, feature, extension, sid, base, body = await setup(visual_host, harness)
    lease = HumanControlLease.model_validate_json(json.dumps(body))
    runtime, repository = feature.runtime, feature.runtime.repository
    hp = base + '/human/' + lease.request.control_id
    key = runtime._key(lease.request.scope, sid)
    original_revoke = extension.revoke
    requests = []
    cleanup = refused = fresh = None
    facts = {'mismatch': mismatch, 'controlled_SDK_only': True, 'physical_browser_proof': False}
    out = Path(os.environ.get('CURRENT_ROW_PROOF_DIR', tmp_path))

    async def observe_original_rpc(session, request):
        requests.append(request)
        return await original_revoke(session, request)

    monkeypatch.setattr(extension, 'revoke', observe_original_rpc)
    try:
        if mismatch == 'accepted-request-id':
            accepted = await client.post(hp + '/revoke', headers=AUTH, json={
                'request_id': 'genuine-first-original-revoke', 'control_epoch': lease.request.control_epoch})
            assert accepted.status_code == 200, accepted.text
            assert len(requests) == 1
        else:
            view = await client.get(hp + '/observation', headers=AUTH)
            assert view.status_code == 200, view.text
            assert not requests
        original, ciphertext = await observed_original_human(repository, lease)
        serial, fenced = runtime._control_serials.get(key, 0), key in runtime._fences
        original_rpc_count = len(requests)
        refused = await client.post(hp + '/revoke', headers=AUTH, json={
            'request_id': 'different-original-revoke-id',
            'control_epoch': lease.request.control_epoch + (1 if mismatch == 'epoch' else 0)})
        after, after_ciphertext = await observed_original_human(repository, lease)
        facts.update(reply=refused.status_code, serial_before=serial,
            serial_after=runtime._control_serials.get(key, 0),
            original_fence_unchanged=(key in runtime._fences) == fenced,
            original_encrypted_row_unchanged=after == original and after_ciphertext == ciphertext,
            rpc_count_before=original_rpc_count, rpc_count_after=len(requests))
        if mismatch == 'epoch':
            fresh = await client.get(hp + '/observation', headers=AUTH)
            facts['fresh_original_human_observe'] = fresh.status_code
    finally:
        monkeypatch.setattr(extension, 'revoke', original_revoke)
        cleanup = await client.post(base + '/release', headers=AUTH)
        facts['real_foreground_release_cleanup_status'] = cleanup.status_code
        facts['egress_empty_after_original_release'] = not runtime.egress._calls
        facts['original_result_holds_after_release'] = len(harness.db.backend._original_result_holds)
        (out / ('invalid-human-revoke-' + mismatch + '.json')).write_text(json.dumps(facts, indent=2) + '\n')

    assert cleanup.status_code == 200, cleanup.text
    assert not runtime.egress._calls and not harness.db.backend._original_result_holds
    assert refused is not None and refused.status_code == 409, facts
    assert facts['serial_after'] == facts['serial_before'] and facts['original_fence_unchanged'], facts
    assert facts['original_encrypted_row_unchanged'], facts
    assert facts['rpc_count_after'] == facts['rpc_count_before'] and extension.effects == 0, facts
    if mismatch == 'epoch':
        assert fresh is not None and fresh.status_code == 200, facts


@pytest.mark.asyncio
async def test_new_preopen_revocation_id_after_handback_uses_original_rpc_without_agent_fence(
    visual_host, harness, monkeypatch, tmp_path,
):
    client, feature, extension, sid, base, body = await setup(visual_host, harness)
    lease = HumanControlLease.model_validate_json(json.dumps(body))
    runtime, repository = feature.runtime, feature.runtime.repository
    cid = 'genuine-never-opened-control'
    hp = base + '/human/' + cid
    key = runtime._key(lease.request.scope, sid)
    original_revoke = extension.revoke
    rpc_requests, rpc_results, rpc_tasks = [], [], []
    facts = {'controlled_SDK_only': True, 'physical_browser_proof': False}
    out = Path(os.environ.get('CURRENT_ROW_PROOF_DIR', tmp_path))
    response = fresh = cleanup = None

    async def observe_original_rpc(session, request):
        rpc_tasks.append(asyncio.current_task())
        value = await original_revoke(session, request)
        rpc_requests.append(request)
        rpc_results.append(value)
        return value

    monkeypatch.setattr(extension, 'revoke', observe_original_rpc)
    try:
        first = await client.post(hp + '/revoke', headers=AUTH, json={
            'request_id': 'genuine-preopen-fence-T', 'control_epoch': lease.request.control_epoch})
        assert first.status_code == 200, first.text
        original, _ = await observed_original_human(repository, lease, cid=cid)
        assert original.request is None and original.lease is None
        assert type(original.tombstone) is HumanControlRevocationTombstone
        assert original.tombstone == rpc_results[-1]
        resumed = await client.post(base + '/handback', headers=AUTH, json={
            'request_id': 'genuine-handback-after-preopen-T', 'session_id': sid,
            'expected_control_epoch': lease.request.control_epoch, 'operation': 'resume', 'share_screen': True})
        assert resumed.status_code == 200 and resumed.json()['controller'] == 'agent', resumed.text
        assert resumed.json()['control_epoch'] == lease.request.control_epoch + 1
        serial = runtime._control_serials[key]
        assert key not in runtime._fences
        count = len(rpc_requests)
        response = await client.post(hp + '/revoke', headers=AUTH, json={
            'request_id': 'genuine-new-id-confirm-T', 'control_epoch': lease.request.control_epoch})
        after, _ = await observed_original_human(repository, lease, cid=cid)
        fresh = await client.get(base + '/observation', headers=AUTH)
        facts.update(reply=response.status_code, fresh_agent_observe=fresh.status_code,
            serial_before=serial, serial_after=runtime._control_serials[key],
            agent_not_fenced=key not in runtime._fences,
            genuine_rpc_count_before=count, genuine_rpc_count_after=len(rpc_requests),
            actual_request_id=rpc_requests[-1].request_id, actual_rpc_epoch=rpc_requests[-1].control_epoch,
            stored_fact_matches_actual_original_rpc=after.tombstone == rpc_results[-1],
            original_fenced_at_preserved=after.tombstone.fenced_at == original.tombstone.fenced_at,
            original_request_and_lease_still_none=after.request is None and after.lease is None)
    finally:
        await asyncio.gather(*rpc_tasks, return_exceptions=True)
        monkeypatch.setattr(extension, 'revoke', original_revoke)
        facts['all_original_rpc_Tasks_joined'] = all(t.done() for t in rpc_tasks)
        cleanup = await client.post(base + '/release', headers=AUTH)
        facts['real_foreground_release_cleanup_status'] = cleanup.status_code
        facts['egress_empty_after_original_release'] = not runtime.egress._calls
        facts['original_result_holds_after_release'] = len(harness.db.backend._original_result_holds)
        (out / 'new-preopen-revoke-after-handback.json').write_text(json.dumps(facts, indent=2) + '\n')

    assert cleanup.status_code == 200, cleanup.text
    assert facts['all_original_rpc_Tasks_joined'] and not runtime.egress._calls
    assert not harness.db.backend._original_result_holds
    assert response is not None and response.status_code == 200, facts
    assert fresh is not None and fresh.status_code == 200, facts
    assert facts['serial_after'] == facts['serial_before'] and facts['agent_not_fenced'], facts
    assert facts['genuine_rpc_count_after'] == facts['genuine_rpc_count_before'] + 1, facts
    assert facts['actual_request_id'] == 'genuine-new-id-confirm-T', facts
    assert facts['actual_rpc_epoch'] == lease.request.control_epoch, facts
    assert facts['stored_fact_matches_actual_original_rpc'] and facts['original_fenced_at_preserved'], facts
    assert facts['original_request_and_lease_still_none'], facts
    assert extension.effects == 0 and harness.adapter.effects == 0


@pytest.mark.asyncio
async def test_external_stop_retires_consumer_before_panel_request_arrives(
    visual_host, harness, tmp_path,
):
    client, feature, extension, sid, base, body = await setup(visual_host, harness)
    lease = HumanControlLease.model_validate_json(json.dumps(body))
    runtime, repository = feature.runtime, feature.runtime.repository
    assert repository.db is harness.db and repository.db.backend_type == "sqlite"
    hp = base + "/human/" + lease.request.control_id
    key = runtime._key(lease.request.scope, sid)
    panel_request = {"request_id": "original-panel-close-R", "control_epoch": lease.request.control_epoch}
    facts = {"controlled_SDK_only": True, "not_browser_or_cross_tab_runtime_proof": True,
             "panel_request_preissued_only_not_sent_before_external_Stop": panel_request}
    stop = metadata = refused = cleanup = None
    retired = after = None
    try:
        original, _ = await observed_original_human(repository, lease)
        assert original.revocation is None and original.lease.state == "active"
        # A different HTTP caller uses the genuine current control API; no
        # runtime/row/permission/native checker or SDK method is replaced.
        stop = await client.post(base + "/stop", headers=AUTH, json={
            "request_id": "external-tab-original-stop", "session_id": sid,
            "expected_control_epoch": lease.request.control_epoch, "operation": "pause"})
        assert stop.status_code == 200, stop.text
        assert stop.json() == {"session_id": sid, "stopped": True}
        # Inspect the original channel through its ordinary metadata API. This
        # is a proposed concrete recovery fact; 409 alone is not release proof.
        metadata = await client.get(hp, headers=AUTH)
        retired, before_cipher = await observed_original_human(repository, lease)
        serial = runtime._control_serials[key]
        fenced = key in runtime._fences
        refused = await client.post(hp + "/revoke", headers=AUTH, json=panel_request)
        after, after_cipher = await observed_original_human(repository, lease)
        facts.update(external_stop_status=stop.status_code,
            original_channel_status=metadata.status_code,
            original_channel_body=metadata.json(),
            server_revocation_id=retired.revocation.request_id if retired.revocation else None,
            panel_reply_status=refused.status_code, panel_reply_body=refused.json(),
            serial_before=serial, serial_after=runtime._control_serials[key],
            control_fence_unchanged=(key in runtime._fences) == fenced,
            original_private_ciphertext_unchanged=before_cipher == after_cipher,
            original_private_model_unchanged=after == retired)
    finally:
        # Always use the real foreground Release; no fence clearing, row
        # repair, fake receipt or task.done-only owner retirement.
        cleanup = await client.post(base + "/release", headers=AUTH)
        facts.update(real_foreground_release_status=cleanup.status_code,
            original_egress_empty=not runtime.egress._calls,
            original_result_holds=len(harness.db.backend._original_result_holds),
            original_inflight=runtime._inflight,
            human_effects=extension.effects, agent_effects=harness.adapter.effects)
        (Path(os.environ.get("CURRENT_ROW_PROOF_DIR", tmp_path)) /
            "external-stop-panel-original-retirement.json").write_text(json.dumps(facts, indent=2) + "\n")
    # Behavioral oracles follow original cleanup. This server case is expected
    # to PASS on current source: it establishes the actual UI failure input.
    assert cleanup.status_code == 200, cleanup.text
    assert not runtime.egress._calls and not harness.db.backend._original_result_holds
    assert runtime._inflight == 0 and extension.effects == 0 and harness.adapter.effects == 0
    assert retired.request == lease.request and retired.lease.state == "revoked"
    assert retired.revocation is not None and retired.revocation.request_id != panel_request["request_id"]
    assert metadata.status_code == 200 and metadata.json()["state"] == "revoked", facts
    assert HumanControlLease.model_validate_json(metadata.content).request == lease.request, facts
    assert metadata.json()["request_sha256"] == body["request_sha256"], facts
    assert refused.status_code == 409 and refused.json() == {"error": {"code": "visual_conflict"}}, facts
    assert after == retired and facts["original_private_ciphertext_unchanged"], facts
    assert facts["serial_after"] == facts["serial_before"] and facts["control_fence_unchanged"], facts
