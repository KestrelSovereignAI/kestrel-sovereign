"""Host lifecycle regressions using the original authenticated local fixtures."""
import asyncio
import json
import os
from pathlib import Path
import sys

import httpx
import pytest

from tests.unit.visual_computer.test_host import AUTH, visual_host as visual_host
from tests.unit.visual_computer.test_human_http import setup
from tests.unit.visual_computer.test_human_delivery import OuterTransport


def save(tmp_path, name, value):
    root = Path(os.environ.get('CURRENT_ROW_PROOF_DIR', tmp_path))
    (root / name).write_text(json.dumps(value, indent=2) + '\n')


@pytest.mark.asyncio
async def test_original_operation_snapshot_cannot_admit_a_retired_observe_rpc(
    visual_host, harness, monkeypatch, tmp_path,
):
    client, feature, extension, _, base, _ = await setup(visual_host, harness)
    human = feature.runtime.human
    paused, resume = asyncio.Event(), asyncio.Event()
    original_record, original_observe = human._record, extension.observe
    operation_code = type(human)._operation.__wrapped__.__code__
    active = None
    calls = []
    facts = {'controlled_sdk_only': True}

    async def read(*args, **kwargs):
        caller = sys._getframe(1).f_code
        value = await original_record(*args, **kwargs)
        if caller is operation_code and not paused.is_set():
            assert value.lease.state == 'active' and value.revocation is None
            assert not harness.db.backend._connection.in_transaction
            facts['original_read_committed_before_resume'] = True
            paused.set()
            await resume.wait()
        return value

    async def observe(*args, **kwargs):
        calls.append(True)
        return await original_observe(*args, **kwargs)

    monkeypatch.setattr(human, '_record', read)
    monkeypatch.setattr(extension, 'observe', observe)
    active = asyncio.create_task(client.get(base + '/human/control/observation', headers=AUTH))
    try:
        await asyncio.wait_for(paused.wait(), 5)
        response = await client.post(base + '/human/control/revoke', headers=AUTH,
            json={'request_id': 'operation-read-retirement', 'control_epoch': 1})
        assert response.status_code == 200 and extension.lease.state == 'revoked'
        resume.set()
        observed = await active
        facts.update(revoke_http=response.status_code, response_http=observed.status_code,
                     SDK_observe_admissions=len(calls), worker_state=extension.lease.state)
        save(tmp_path, 'operation-admission-facts.json', facts)
        assert calls == [], 'retired original snapshot admitted an SDK observe RPC'
        assert observed.status_code == 409
    finally:
        resume.set()
        await asyncio.gather(active, return_exceptions=True)
        assert not feature.runtime.egress._calls


@pytest.mark.asyncio
async def test_prepared_status_cannot_report_active_after_original_revoke_ack(
    visual_host, harness, monkeypatch, tmp_path,
):
    client, feature, extension, _, base, _ = await setup(visual_host, harness)
    human = feature.runtime.human
    original = human.status
    paused, resume = asyncio.Event(), asyncio.Event()
    active = None

    async def status(*args, **kwargs):
        value = await original(*args, **kwargs)
        if not paused.is_set():
            assert value.state == 'active'
            paused.set()
            await resume.wait()
        return value

    monkeypatch.setattr(human, 'status', status)
    active = asyncio.create_task(client.get(base + '/human/control', headers=AUTH))
    try:
        await asyncio.wait_for(paused.wait(), 5)
        revoked = await client.post(base + '/human/control/revoke', headers=AUTH,
            json={'request_id': 'prepared-status-retirement', 'control_epoch': 1})
        assert revoked.status_code == 200 and extension.lease.state == 'revoked'
        resume.set()
        stale = await active
        fresh = await client.get(base + '/human/control', headers=AUTH)
        save(tmp_path, 'prepared-status-facts.json', {
            'revoke_http': revoked.status_code, 'prepared_http': stale.status_code,
            'prepared_body': stale.json(), 'fresh_http': fresh.status_code,
            'fresh_body': fresh.json(), 'controlled_sdk_only': True,
        })
        # Existing private delivery reports a prepared response as UNKNOWN.
        assert stale.status_code == 503, stale.text
        assert stale.json() == {'error': {'code': 'visual_outcome_unknown'}}
        assert fresh.status_code == 200 and fresh.json()['state'] == 'revoked'
    finally:
        resume.set()
        await asyncio.gather(active, return_exceptions=True)
        assert not feature.runtime.egress._calls


@pytest.mark.asyncio
async def test_publication_drain_joins_original_send_admitted_after_initial_drain(
    visual_host, harness, monkeypatch, tmp_path,
):
    client, feature, extension, _, base, _ = await setup(visual_host, harness)
    runtime = feature.runtime
    first_done, proceed = asyncio.Event(), asyncio.Event()
    original = runtime.egress.revoke_session
    drains = []
    transport = OuterTransport(visual_host[2], base + '/human/control/observation', suppress=True)
    active = revoke = cancelled = None

    async def drain(*args, **kwargs):
        drains.append(True)
        await original(*args, **kwargs)
        if len(drains) == 1:
            first_done.set()
            await proceed.wait()

    monkeypatch.setattr(runtime.egress, 'revoke_session', drain)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=transport), base_url='http://test') as delayed:
        revoke = asyncio.create_task(client.post(base + '/human/control/revoke', headers=AUTH,
            json={'request_id': 'publication-drain', 'control_epoch': 1}))
        try:
            await asyncio.wait_for(first_done.wait(), 5)
            assert not runtime.egress._calls
            active = asyncio.create_task(delayed.get(transport.path, headers=AUTH))
            await asyncio.wait_for(transport.entered.wait(), 5)
            assert any(transport.child in call.accepted_jobs for call in runtime.egress._calls.values())
            proceed.set()
            cancelled = asyncio.create_task(transport.cancelled.wait())
            done, _ = await asyncio.wait({revoke, cancelled}, timeout=5,
                                         return_when=asyncio.FIRST_COMPLETED)
            save(tmp_path, 'publication-drain-facts.json', {
                'initial_drain_completed': True, 'exact_original_send_registered': True,
                'drains_entered': len(drains), 'send_cancel_requested': transport.cancelled.is_set(),
                'revoke_done_before_send_release': revoke.done(),
                'revoke_http_if_done': revoke.result().status_code if revoke.done() else None,
                'controlled_sdk_only': True,
            })
            assert done and transport.cancelled.is_set(), 'publication did not retire original send'
            assert not revoke.done() and not transport.child.done()
            transport.release.set()
            response = await revoke
            await asyncio.gather(active, return_exceptions=True)
            assert response.status_code == 200 and extension.lease.state == 'revoked'
            assert transport.child.done() and transport.closed.is_set() and not transport.bodies
        finally:
            proceed.set()
            transport.release.set()
            if cancelled is not None and not cancelled.done():
                cancelled.cancel()
            await asyncio.gather(*(t for t in (active, revoke, cancelled) if t is not None),
                                 return_exceptions=True)
    assert not runtime.egress._calls
