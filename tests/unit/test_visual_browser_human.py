"""Candidate SDK contract against the actual worker state engine; no isolation claim."""
import asyncio
import importlib.util
import json
from datetime import timedelta
from pathlib import Path
import sys

import pytest
from kestrel_sdk.computer_use.human_control import HumanControlRequest
from tests.unit.test_visual_browser_controller import controller, control, Refused, _module

spec=importlib.util.spec_from_file_location('visual_browser_human',Path(__file__).parents[2]/'docker/visual-browser/human.py')
module=importlib.util.module_from_spec(spec)
previous=sys.modules.get('controller')
sys.modules['controller']=_module
try:spec.loader.exec_module(module)
finally:
    if previous is None:sys.modules.pop('controller',None)
    else:sys.modules['controller']=previous


async def opened(tmp_path, *, accept_open=True, **options):
    c,page,session=controller(**options)
    async def focus(_):return ['about:blank',True,'BODY','',None]
    page.evaluate=focus
    human=module.HumanTransport(c,tmp_path,'a'*64)
    c.human=human
    session=await c.control(session,control(session,'takeover'))
    now=c.clock()
    request=HumanControlRequest(control_id='human',scope=session['scope'],session_id=session['session_id'],
        lease_id=session['lease_id'],driver_id=session['driver_id'],consumer_id='consumer',control_epoch=session['control_epoch'],
        issued_at=now,expires_at=now+timedelta(seconds=60),capabilities=frozenset({'screenshot','click','type','key'}),max_frame_age_seconds=15).model_dump(mode='json')
    lease=await human.open(session,request) if accept_open else None
    return c,page,session,human,request,lease


async def action(c,session,human,lease,**changes):
    observed=await human.observe(session,lease)
    request={'schema_version':1,'control_id':'human','consumer_id':'consumer','action':{'schema_version':1,
        'action_id':'manual-input','session_id':session['session_id'],'observation_id':observed['frame']['observation_id'],
        'control_epoch':session['control_epoch'],'not_after':(c.clock()+timedelta(seconds=5)).isoformat(),
        'action':{'schema_version':1,'kind':'type','text':'PRIVATE-MANUAL-SECRET'}}}
    request.update(changes)
    return request


@pytest.mark.asyncio
async def test_separate_channel_pixels_once_receipt_and_private_custody(tmp_path):
    c,page,s,h,request,lease=await opened(tmp_path)
    assert await h.open(s,request)==lease
    with pytest.raises(Refused,match='input_fenced'):await c.observe(s)
    req=await action(c,s,h,lease)
    assert req['action']['observation_id'].startswith('human-')
    result=await h.act(s,lease,req)
    assert result['receipt']['outcome']=='completed'
    assert await h.act(s,lease,req)==result
    assert h.action_status(s,lease,'manual-input')==result
    assert len(page.effects)==1
    with pytest.raises(Refused,match='action_unavailable'):c.action_status(s,'manual-input')
    journal=h.journal.read_text()
    assert 'PRIVATE-MANUAL-SECRET' not in journal and 'image_base64' not in journal
    assert h.frame is None
    assert h.journal.stat().st_mode & 0o777 == 0o600
    with pytest.raises(Refused,match='human_conflict'):module.HumanTransport(c,tmp_path,'a'*64)


@pytest.mark.asyncio
@pytest.mark.parametrize('field,value',[('consumer_id','foreign'),('control_id','foreign')])
async def test_foreign_binding_never_effects(tmp_path,field,value):
    c,page,s,h,_,lease=await opened(tmp_path)
    req=await action(c,s,h,lease,**{field:value})
    with pytest.raises(Refused):await h.act(s,lease,req)
    assert page.effects==[]


@pytest.mark.asyncio
async def test_display_or_focus_change_refuses_before_input(tmp_path):
    c,page,s,h,_,lease=await opened(tmp_path)
    req=await action(c,s,h,lease)
    async def changed(_):return ['about:blank',True,'INPUT','changed',None]
    page.evaluate=changed
    with pytest.raises(Refused,match='stale_observation'):await h.act(s,lease,req)
    assert not page.effects


@pytest.mark.asyncio
async def test_revoke_before_open_is_permanent_durable_fence(tmp_path):
    c,page,s,h,req,lease=await opened(tmp_path,accept_open=False)
    revoke={'schema_version':1,'request_id':'revoke','session_id':s['session_id'],
        'control_id':'late','consumer_id':'late-consumer','control_epoch':s['control_epoch']}
    result=await h.revoke(s,revoke)
    assert 'fenced_at' in result and await h.revoke(s,revoke)==result
    assert 'late' in json.loads(h.journal.read_text())['payload']['tombstones']
    with pytest.raises(Refused,match='human_conflict'):await h.open(s,{**req,'control_id':'late','consumer_id':'late-consumer'})
    with pytest.raises(Refused,match='human_conflict'):h.status(s,{**req,'control_id':'late'})
    with pytest.raises(Refused,match='human_conflict'):await h.revoke(s,{**revoke,'request_id':'changed','consumer_id':'foreign'})


@pytest.mark.asyncio
async def test_new_revocation_id_confirms_same_preopen_fence_without_renewing_it(tmp_path):
    c,page,s,h,_,_=await opened(tmp_path,accept_open=False)
    revoke={'schema_version':1,'request_id':'first-close','session_id':s['session_id'],
        'control_id':'late','consumer_id':'late-consumer','control_epoch':s['control_epoch']}
    original=await h.revoke(s,revoke)
    changed={**revoke,'request_id':'second-close'}
    confirmed=await h.revoke(s,changed)
    assert confirmed['request']==changed and confirmed['fenced_at']==original['fenced_at']
    assert await h.revoke(s,revoke)==original
    with pytest.raises(Refused,match='human_conflict'):
        await h.revoke(s,{**changed,'consumer_id':'foreign'})
    assert not page.effects


@pytest.mark.asyncio
async def test_priority_revoke_drains_input_before_ack_and_old_epoch_cannot_reopen(tmp_path):
    c,page,s,h,_,lease=await opened(tmp_path)
    req=await action(c,s,h,lease)
    page.gate=asyncio.Event()
    active=asyncio.create_task(h.act(s,lease,req))
    await page.entered.wait()
    revoke={'schema_version':1,'request_id':'revoke','session_id':s['session_id'],
        'control_id':'human','consumer_id':'consumer','control_epoch':s['control_epoch']}
    retiring=asyncio.create_task(h.revoke(s,revoke))
    await asyncio.sleep(0)
    assert not retiring.done()
    assert h.frame is None
    page.gate.set()
    assert (await active)['receipt']['outcome']=='completed'
    retired=await retiring
    assert retired['state']=='revoked'
    assert h.action_status(s,lease,'manual-input')['receipt']['outcome']=='completed'
    with pytest.raises(Refused):await h.observe(s,lease)
    with pytest.raises(Refused):await h.open(s,lease['request'])
    with pytest.raises(Refused):await h.open(s,{**lease['request'],'control_id':'replacement'})


@pytest.mark.asyncio
async def test_stop_and_handback_purge_human_frames(tmp_path):
    c,page,s,h,_,lease=await opened(tmp_path)
    await h.observe(s,lease)
    resumed=await c.control(s,control(s,'resume','handback'))
    assert h.frame is None and h.focus is None and h.leases['human']['state']=='revoked'
    with pytest.raises(Refused):await h.observe(resumed,lease)
    frame,_=await c.observe(resumed)
    assert not frame['observation_id'].startswith('human-')


@pytest.mark.asyncio
async def test_fixed_monotonic_lease_and_frame_expiry(tmp_path):
    from datetime import UTC,datetime
    wall=[datetime.now(UTC)];elapsed=[0]
    c,page,s,h,_,lease=await opened(tmp_path,clock=lambda:wall[0],monotonic=lambda:elapsed[0])
    req=await action(c,s,h,lease)
    elapsed[0]=16
    with pytest.raises(Refused,match='stale_observation'):await h.act(s,lease,req)
    assert page.effects==[]
    elapsed[0]=61
    await h.expire()
    assert h.leases['human']['state']=='revoked'
    with pytest.raises(Refused):await h.observe(s,lease)


@pytest.mark.asyncio
async def test_wire_capability_order_is_not_a_different_consumer_lease(tmp_path):
    c,page,s,h,_,lease=await opened(tmp_path)
    reordered=json.loads(json.dumps(lease))
    reordered['request']['capabilities'].reverse()
    assert reordered!=lease
    await h.observe(s,reordered)
    assert h.frame is not None


@pytest.mark.asyncio
async def test_failed_preopen_fence_write_cannot_ack_an_unpersisted_tombstone(tmp_path,monkeypatch):
    c,page,s,h,_,lease=await opened(tmp_path)
    original=h._persist;calls=[]
    def fail_once():
        calls.append(True)
        if len(calls)==1:raise OSError('owned simulated journal failure')
        original()
    monkeypatch.setattr(h,'_persist',fail_once)
    request={'schema_version':1,'request_id':'revoke-late','control_id':'late','session_id':s['session_id'],
        'consumer_id':'late-consumer','control_epoch':s['control_epoch']}
    with pytest.raises(OSError):await h.revoke(s,request)
    result=await h.revoke(s,request)
    assert 'fenced_at' in result
    assert 'late' in json.loads(h.journal.read_text())['payload']['tombstones']


@pytest.mark.asyncio
async def test_wrong_revocation_target_never_poison_original_accepted_binding(tmp_path):
    c,page,s,h,_,lease=await opened(tmp_path)
    request={'schema_version':1,'request_id':'revoke','control_id':'human','session_id':s['session_id'],
        'consumer_id':'consumer','control_epoch':s['control_epoch']}
    with pytest.raises(Refused):await h.revoke(s,{**request,'consumer_id':'foreign'})
    assert (await h.revoke(s,request))['state']=='revoked'


@pytest.mark.asyncio
@pytest.mark.parametrize('control_operation', ['resume', 'pause'])
async def test_original_human_input_fenced_before_worker_entry_has_negative_receipt(tmp_path, control_operation):
    c,page,session,h,_,lease=await opened(tmp_path)
    request=await action(c,session,h,lease)
    await c.control(session,control(session,control_operation,'control-before-human-input'))
    with pytest.raises(Refused):await h.act(session,lease,request)
    assert not page.effects
    receipt=h.action_status(session,lease,request['action']['action_id'])
    assert receipt['control_id']==request['control_id'] and receipt['consumer_id']==request['consumer_id']
    assert receipt['receipt']['outcome']=='denied'
    assert receipt['receipt']['control_epoch']==request['action']['control_epoch']
    assert request['action']['action_id'] in c._human_action_ids
    with pytest.raises(Refused):await h.act(session,lease,request)
    assert h.action_status(session,lease,request['action']['action_id'])==receipt
    assert not page.effects


@pytest.mark.asyncio
@pytest.mark.parametrize('wrong', ['consumer','control','epoch','session','unissued','accepted','digest','lease-request','agent-id'])
async def test_fenced_human_refusal_cannot_journal_foreign_or_unissued_identity(tmp_path,wrong):
    c,page,session,h,_,lease=await opened(tmp_path)
    request=await action(c,session,h,lease)
    await c.control(session,control(session,'resume','handback'))
    supplied=json.loads(json.dumps(lease))
    if wrong=='consumer':request['consumer_id']='foreign'
    elif wrong=='control':request['control_id']='foreign'
    elif wrong=='epoch':request['action']['control_epoch']+=1
    elif wrong=='session':session={**session,'session_id':'foreign'}
    elif wrong=='unissued':supplied['request']['control_id']='unissued'
    elif wrong=='accepted':
        from datetime import datetime
        supplied['accepted_at']=(datetime.fromisoformat(supplied['accepted_at'].replace('Z','+00:00'))+timedelta(microseconds=1)).isoformat()
    elif wrong=='digest':supplied['request_sha256']='0'*64
    elif wrong=='lease-request':supplied['request']['max_frame_age_seconds']=14
    elif wrong=='agent-id':
        c._actions[request['action']['action_id']]=('agent-digest',{'outcome':'unknown'})
    before=json.loads(h.journal.read_text())
    with pytest.raises((Refused,ValueError)):await h.act(session,supplied,request)
    assert not page.effects and not h.actions and not c._human_action_ids
    assert json.loads(h.journal.read_text())==before
