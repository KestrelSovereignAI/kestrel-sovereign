"""Original pending-control host/auth/native SQLite contract; controlled SDK, not Docker."""
import asyncio
import json
import os
from pathlib import Path
import pytest
from kestrel_sdk.computer_use import ControlRequest
from kestrel_sdk.resources import ResourceOutcomeUnknown
from tests.unit.visual_computer.test_host import AUTH
from tests.unit.visual_computer.test_host import visual_host as visual_host
from tests.unit.visual_computer.test_human_http import setup

@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['takeover', 'handback'])
async def test_original_pending_control_refuses_other_stop_and_reconciles_same_id(visual_host, harness, monkeypatch, tmp_path, operation):
    client, feature, extension, sid, base, opened = await setup(visual_host, harness)
    runtime = feature.runtime
    epoch = opened['request']['control_epoch']
    original = {'request_id':'R-original-'+operation, 'session_id':sid,
        'expected_control_epoch':epoch, 'operation':'resume' if operation=='handback' else 'takeover'}
    body = {**original, **({'share_screen':True} if operation=='handback' else {})}
    stop = {'request_id':'S-independent-safety-stop', 'session_id':sid,
        'expected_control_epoch':epoch, 'operation':'pause'}
    original_control = harness.adapter.control
    calls, sdk_tasks = [], []
    facts = {'operation':operation, 'controlled_SDK_only':True, 'physical_browser_proof':False}
    first = refused = retried = None
    cleanup = None
    async def uncertain_once(session, request):
        task=asyncio.current_task()
        owners=[call for call in runtime.egress._calls.values() if call.session==session and task in call.accepted_jobs]
        assert len(owners)==1 and owners[0].origin is not task
        sdk_tasks.append(task);calls.append(request)
        if len(calls)==1:
            raise ResourceOutcomeUnknown('controlled-original-control-uncertainty')
        return await original_control(session, request)
    monkeypatch.setattr(harness.adapter,'control',uncertain_once)
    try:
        first=await client.post(base+'/'+operation,headers=AUTH,json=body)
        assert first.status_code==503 and first.json()['error']['code']=='visual_outcome_unknown',first.text
        pending=await client.get(base,headers=AUTH)
        assert pending.status_code==200,pending.text
        pending_request=ControlRequest.model_validate(pending.json()['pending_control'])
        assert pending_request==ControlRequest.model_validate(original)
        assert len(calls)==1
        refused=await client.post(base+'/stop',headers=AUTH,json=stop)
        assert refused.status_code==503 and refused.json()['error']['code']=='visual_outcome_unknown',refused.text
        unchanged=await client.get(base,headers=AUTH)
        assert unchanged.status_code==200,unchanged.text
        assert ControlRequest.model_validate(unchanged.json()['pending_control'])==pending_request
        assert len(calls)==1,'different Stop must not replace original driver control'
        retried=await client.post(base+'/'+operation,headers=AUTH,json=body)
        final=await client.get(base,headers=AUTH)
        facts.update(first_R_status=first.status_code,S_status=refused.status_code,R_retry_status=retried.status_code,
            original_R=original,S=stop,pending_before=pending.json(),pending_after_S=unchanged.json(),
            driver_requests=[x.model_dump(mode='json') for x in calls],final_status=final.status_code,
            final_state=final.json(),actual_original_sdk_tasks=len(sdk_tasks))
    finally:
        monkeypatch.setattr(harness.adapter,'control',original_control)
        await asyncio.gather(*sdk_tasks,return_exceptions=True)
        cleanup=await client.post(base+'/release',headers=AUTH)
        facts.update(all_original_sdk_tasks_joined=all(t.done() for t in sdk_tasks),
            foreground_release_status=cleanup.status_code,egress_empty=not runtime.egress._calls,
            result_holds=len(harness.db.backend._original_result_holds))
        out=Path(os.environ.get('CURRENT_ROW_PROOF_DIR',tmp_path))
        (out/('pending-control-'+operation+'.json')).write_text(json.dumps(facts,indent=2)+'\n')
    assert cleanup.status_code==200,cleanup.text
    assert not runtime.egress._calls and not harness.db.backend._original_result_holds
    assert all(t.done() for t in sdk_tasks)
    assert retried is not None and retried.status_code==200,facts
    assert len(calls)==2 and calls[0]==calls[1]==ControlRequest.model_validate(original),facts
    assert facts['final_state']['pending_control'] is None,facts
    assert facts['final_state']['session']['control_epoch']==epoch+1,facts
    assert facts['final_state']['session']['controller']==('agent' if operation=='handback' else 'human'),facts
    assert extension.effects==harness.adapter.effects==0
