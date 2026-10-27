"""Never-acquired tombstones use the same authenticated native request row."""
import asyncio
from datetime import UTC, datetime

import pytest
from kestrel_sdk.resources import ResourceConflict
from kestrel_sovereign.visual_computer.repository import VisualCustodyRepository
from tests.integration.test_visual_custody_backends import SCOPE, record
from tests.integration.test_visual_custody_backends import custody as custody


@pytest.mark.asyncio
async def test_native_retirement_migration_reopen_keeps_ids_without_quota_recharge(custody):
    store, reopen = custody
    store.max_requests = 1
    original = await store.admit(SCOPE, record(), lambda: None)
    stale = original.model_copy(deep=True)
    retired = await store.retire_unacquired(SCOPE, original, at=datetime.now(UTC))
    assert retired.phase == 'retired' and retired.lease is None and retired.retired_from == 'admitted'
    with pytest.raises(ResourceConflict):
        await store.save(SCOPE, retired.model_copy(update={'phase':'acquiring','retired_at':None,'retired_from':None}))
    assert await store.records(SCOPE) == []
    with pytest.raises(ResourceConflict):
        await store.save(SCOPE, stale.model_copy(update={'phase':'acquiring'}))
    with pytest.raises(ResourceConflict):
        await store.save(SCOPE, stale)
    await store.db.execute('DROP INDEX visual_computer_retired_requests')
    await store.db.execute('ALTER TABLE visual_computer_requests DROP COLUMN retired')
    db = await reopen()
    try:
        other = VisualCustodyRepository(db, max_requests=1)
        recovered = await other.get(SCOPE, request_id=original.request.request_id)
        assert recovered == retired and recovered.request == original.request
        assert await other.admit(SCOPE, stale, lambda:None) == retired
        assert await other.retire_unacquired(SCOPE, stale, at=datetime.now(UTC)) == retired
        assert (await db.fetchone('SELECT released, retired FROM visual_computer_requests')) == (0, 1)
        fresh = record();fresh.session_id='fresh';fresh.request=fresh.request.model_copy(update={'request_id':'fresh'})
        await other.admit(SCOPE, fresh, lambda:None)
        with pytest.raises(ResourceConflict, match='capacity'):
            another = record();another.session_id='over';another.request=another.request.model_copy(update={'request_id':'over'})
            await other.admit(SCOPE, another, lambda:None)
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('phase', ['acquiring', 'unknown', 'fenced'])
async def test_native_possible_effect_never_becomes_unacquired_from_absence_or_expiry(custody, phase):
    store, _ = custody
    store.max_requests=1
    original = await store.admit(SCOPE, record(), lambda:None)
    original.phase=phase
    await store.save(SCOPE, original)
    with pytest.raises(ResourceConflict, match='possible acquisition'):
        await store.retire_unacquired(SCOPE, original, at=datetime.now(UTC))
    stale=original.model_copy(update={'phase':'admitted'})
    with pytest.raises(ResourceConflict, match='cannot become unacquired'):
        await store.save(SCOPE, stale)
    fresh=record();fresh.session_id='fresh';fresh.request=fresh.request.model_copy(update={'request_id':'fresh'})
    with pytest.raises(ResourceConflict, match='capacity'):
        await store.admit(SCOPE, fresh, lambda:None)


@pytest.mark.asyncio
async def test_native_retirement_and_acquisition_cas_have_one_winner(custody):
    store, reopen = custody
    original = await store.admit(SCOPE, record(), lambda:None)
    db = await reopen()
    try:
        other=VisualCustodyRepository(db)
        await other.ensure_initialized()
        acquiring=original.model_copy(update={'phase':'acquiring'})
        results=await asyncio.gather(store.retire_unacquired(SCOPE, original, at=datetime.now(UTC)),
            other.save(SCOPE, acquiring), return_exceptions=True)
        assert sum(isinstance(value, ResourceConflict) for value in results)==1
        current=await store.get(SCOPE, request_id=original.request.request_id)
        assert current.phase in {'acquiring','retired'}
        if current.phase=='retired':assert current.lease is None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_native_last_slot_after_retirement_serializes_across_connections(custody, monkeypatch):
    store,reopen=custody;store.max_requests=1
    original=await store.admit(SCOPE,record(),lambda:None)
    await store.retire_unacquired(SCOPE,original,at=datetime.now(UTC))
    db=await reopen()
    try:
        other=VisualCustodyRepository(db,max_requests=1);await other.ensure_initialized()
        one=record();one.session_id='one';one.request=one.request.model_copy(update={'request_id':'one'})
        two=record();two.session_id='two';two.request=two.request.model_copy(update={'request_id':'two'})
        entered,resume=asyncio.Event(),asyncio.Event();fetchone=store.db.fetchone
        async def hold(sql,*args,**kwargs):
            value=await fetchone(sql,*args,**kwargs)
            if sql.startswith('SELECT COUNT(*) FROM visual_computer_requests'):
                entered.set();await resume.wait()
            return value
        monkeypatch.setattr(store.db,'fetchone',hold)
        first=asyncio.create_task(store.admit(SCOPE,one,lambda:None));second=None
        try:
            await asyncio.wait_for(entered.wait(),2)
            second=asyncio.create_task(other.admit(SCOPE,two,lambda:None))
            await asyncio.sleep(.03);resume.set()
            results=await asyncio.gather(first,second,return_exceptions=True)
            assert sum(isinstance(item,ResourceConflict) for item in results)==1
            assert (await db.fetchone('SELECT COUNT(*) FROM visual_computer_requests'))[0]==2
        finally:
            resume.set();await asyncio.gather(first,*([second] if second else []),return_exceptions=True)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_native_stale_preacquisition_revision_and_legacy_action_cannot_retire(custody):
    from datetime import timedelta
    from kestrel_sdk.computer_use import ActionRequest
    store,_=custody
    original=await store.admit(SCOPE,record(),lambda:None)
    stale=original.model_copy(deep=True)
    original.phase='quoted';await store.save(SCOPE,original)
    with pytest.raises(ResourceConflict):
        await store.retire_unacquired(SCOPE,stale,at=datetime.now(UTC))
    action=ActionRequest(action_id='legacy',session_id=original.session_id,observation_id='frame',control_epoch=0,
        not_after=datetime.now(UTC)+timedelta(seconds=10),action={'kind':'click','x':1,'y':1})
    await store.action(SCOPE,original.session_id,action.action_id,accept=action)
    with pytest.raises(ResourceConflict):
        await store.retire_unacquired(SCOPE,original,at=datetime.now(UTC))
