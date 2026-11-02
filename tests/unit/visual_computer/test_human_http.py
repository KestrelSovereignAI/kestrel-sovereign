"""Authenticated human HTTP routes; SDK doubles do not prove Docker isolation."""
from datetime import UTC,datetime,timedelta
import pytest
from kestrel_sovereign.visual_computer.human import HumanComputerRuntime
from kestrel_sovereign.visual_computer.runtime import VisualAction
from tests.unit.visual_computer.test_host import visual_host as visual_host,path,intent,AUTH,KEY
from tests.unit.visual_computer.test_human import Extension


async def setup(visual_host,harness,*,lease_seconds=120):
    client,agents,_=visual_host
    feature=agents['kite'].features['VisualComputerFeature']
    feature.runtime.human=HumanComputerRuntime(feature.runtime,lease_seconds=lease_seconds)
    extension=Extension(feature.runtime)
    harness.adapter.human_control=lambda:extension
    response=await client.post(path()+'/requests',headers=AUTH,json=intent())
    assert response.status_code==200,response.text
    sid=response.json()['session_id']
    base=path()+f'/sessions/{sid}'
    result=await client.post(base+'/takeover',headers=AUTH,json={'request_id':'takeover','session_id':sid,'expected_control_epoch':0,'operation':'takeover'})
    assert result.status_code==200,result.text
    opened=await client.post(base+'/human/control',headers=AUTH)
    assert opened.status_code==200,opened.text
    return client,feature,extension,sid,base,opened.json()


@pytest.mark.asyncio
async def test_human_private_http_auth_separate_grants_and_no_generic_frame_leak(visual_host,harness):
    client,feature,extension,sid,base,lease=await setup(visual_host,harness)
    hp=base+'/human/control'
    assert (await client.get(hp+'/observation')).status_code==401
    assert (await client.get(hp+'/observation',params={'api_key':KEY})).status_code==401
    assert (await client.get(path('other')+f'/sessions/{sid}/human/control',headers=AUTH)).status_code in {404,422}
    view=await client.get(hp+'/observation',headers=AUTH)
    assert view.status_code==200 and view.headers['cache-control'].startswith('private, no-store')
    assert view.json()['frame']['observation_id'].startswith('human-')
    payload={'control_id':'control','consumer_id':lease['request']['consumer_id'],'action':{
        'action_id':'manual-http','session_id':sid,'observation_id':view.json()['frame']['observation_id'],'control_epoch':1,
        'not_after':(datetime.now(UTC)+timedelta(seconds=5)).isoformat(),'action':{'kind':'type','text':'PRIVATE-HUMAN-HTTP'}}}
    feature._cfg['granted_actions'].remove(VisualAction.HUMAN_INPUT.value)
    refused=await client.post(hp+'/actions',headers=AUTH,json=payload)
    assert refused.status_code==403 and 'PRIVATE-HUMAN-HTTP' not in refused.text and extension.effects==0
    feature._cfg['granted_actions'].append(VisualAction.HUMAN_INPUT.value)
    receipt=await client.post(hp+'/actions',headers=AUTH,json=payload)
    assert receipt.status_code==200,receipt.text
    assert receipt.json()['receipt']['outcome']=='completed'
    assert (await client.get(base+'/actions/manual-http',headers=AUTH)).status_code==404
    assert (await client.get(base+'/observation',headers=AUTH)).status_code==409
    status=await client.get(base,headers=AUTH)
    assert 'human-' not in status.text and 'PRIVATE-HUMAN-HTTP' not in status.text
    malformed=await client.post(hp+'/actions',headers=AUTH,json={'action':{'action':{'kind':'type','text':'PRIVATE-MALFORMED'}}})
    assert malformed.status_code==400 and 'PRIVATE-MALFORMED' not in malformed.text


@pytest.mark.asyncio
async def test_human_handback_confirmation_and_revoke_are_explicit(visual_host,harness):
    client,feature,extension,sid,base,lease=await setup(visual_host,harness)
    request={'request_id':'handback','session_id':sid,'expected_control_epoch':1,'operation':'resume'}
    assert (await client.post(base+'/handback',headers=AUTH,json=request)).status_code==409
    response=await client.post(base+'/handback',headers=AUTH,json={**request,'share_screen':True})
    assert response.status_code==200,response.text
    assert response.json()['control_epoch']==2 and extension.lease.state=='revoked'
    assert (await client.get(base+'/human/control/observation',headers=AUTH)).status_code==409


@pytest.mark.asyncio
async def test_human_http_disabled_lane_refuses_without_candidate_shim(visual_host):
    client,agents,_=visual_host
    admitted=await client.post(path()+'/requests',headers=AUTH,json=intent())
    sid=admitted.json()['session_id']
    response=await client.post(path()+f'/sessions/{sid}/human/control',headers=AUTH)
    assert response.status_code==422 and response.json()['error']['code']=='visual_unsupported'
