import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { JSDOM } from 'jsdom';
import { createHash } from 'node:crypto';
import { parseResponseError } from '../../kestrel_sovereign/static/js/api_client.mjs';

let source = await readFile(new URL('../../kestrel_sovereign/features/visual_computer/static/visual-computer.js', import.meta.url), 'utf8');
source = source.replace("import API from '/js/api.js';", 'const API = {};')
    .replace("import { registerPanel } from '/js/ui-ext/panels.js';", 'const registerPanel = () => {};')
    .replace("import bus from '/js/ui-ext/bus.js';", 'const bus = { on() {} };');
const { mountVisualComputer } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const flush = () => new Promise(resolve => setTimeout(resolve, 0));

function setup() {
    const dom = new JSDOM('<main></main>');
    globalThis.document = dom.window.document;
    const calls = [];
    let agent = 'original';
    let changed;
    let screen;
    let controller = 'agent';
    let epoch = 0;
    const api = { getHostAgent: () => agent, onHostAgentChange: fn => { changed = fn; return () => {}; },
        async requestForAgent(path, options, owner) {
            calls.push({path, options, owner});
            if (path.endsWith('/sessions')) return [{session_id:'session', phase:'active'}];
            if (path.endsWith('/observation')) return new Promise(resolve => { screen = resolve; });
            if (path.endsWith('/takeover')) { controller = 'human'; epoch++; return {}; }
            if (path.endsWith('/handback')) { controller = 'agent'; epoch++; return {}; }
            if (path.endsWith('/stop')) { controller = 'none'; epoch++; return {}; }
            return {phase:controller==='agent'?'active':'fenced', session:{controller,control_epoch:epoch}, lease:{state:'ready'}};
        } };
    const body = document.querySelector('main');
    const view = mountVisualComputer(body, api);
    const click = text => [...body.querySelectorAll('button')].find(b => b.textContent === text).click();
    return {view, body, calls, click, resolveScreen: () => screen({frame:{media_type:'image/png'},image_base64:'eA=='}),
        switchAgent: () => {agent='other'; changed();}, dom};
}

test('Stop stays independently usable while a screen request is blocked', async () => {
    const h = setup();
    await h.view.refresh();
    h.click('Show current screen');
    await flush();
    h.click('Stop input');
    await flush();
    assert.ok(h.calls.some(call => call.path.endsWith('/stop')));
    h.resolveScreen();
    await flush();
    assert.equal(h.body.querySelector('img').hidden, true);
    assert.ok(h.calls.every(call => call.owner === 'original'));
    h.view.destroy();
});

test('takeover and agent changes discard late pixels and keep exact control request', async () => {
    const h = setup();
    await h.view.refresh();
    h.click('Show current screen');
    await flush();
    h.click('Take over');
    await flush();
    const call = h.calls.find(call => call.path.endsWith('/takeover'));
    const payload = JSON.parse(call.options.body);
    assert.equal(payload.session_id,'session');
    assert.equal(payload.expected_control_epoch,0);
    assert.equal(payload.operation,'takeover');
    assert.ok(payload.request_id);
    h.resolveScreen();
    await flush();
    assert.equal(h.body.querySelector('img').hasAttribute('src'),false);
    h.switchAgent();
    assert.equal(h.body.querySelector('select').children.length,0);
    h.view.destroy();
});

// Scripted UI transport; genuine authenticated terminal metadata/409 behavior
// is separately exercised by test_human_stale_revoke.py. These facts grant no IO.
function manualSetup({failInput=false, expires=120000, failRevoke=false, failControl=false,
    deferFirstRevoke=false, lostControlReplies=0, lostHandbackReplies=0, lostStopReplies=0,
    deferStopAck=false, externalRetirement409=false,
    metadataChange=value=>value, statusChange=value=>value, deferMetadata=false, loseOpenReply=false, sessions=['session']}={}) {
    const dom = new JSDOM('<main></main>', {url:'http://localhost/'});
    globalThis.document = dom.window.document;
    const calls=[];
    let binding;
    let lease;
    let releaseFirstRevoke;
    const firstRevoke=deferFirstRevoke ? new Promise(resolve=>{releaseFirstRevoke=resolve;}) : null;
    let releaseMetadata;
    const metadataWait=deferMetadata ? new Promise(resolve=>{releaseMetadata=resolve;}) : null;
    let releaseStopAck;
    const stopAck=deferStopAck ? new Promise(resolve=>{releaseStopAck=resolve;}) : null;
    let stopRepliesLost=0;
    let handbackRepliesLost=0;
    let revokeCount=0;
    let selectedOwner='owner';
    let ownerChanged;
    const lastControls=new Map();
    const sessionStates=new Map();
    const stateFor=(owner,sid)=>{
        const key=JSON.stringify([owner,sid]);
        if(!sessionStates.has(key))sessionStates.set(key,{controller:'human',epoch:1});
        return sessionStates.get(key);
    };
    const applyControl=async(original,owner,target,controller)=>{
        const key=JSON.stringify([owner,original.session_id]);
        const last=lastControls.get(key);
        if(last?.request_id===original.request_id){
            if(JSON.stringify(last)!==JSON.stringify(original))throw new Error('control ID differs');
            return;
        }
        if(original.expected_control_epoch!==target.epoch)throw await parseResponseError(new Response(JSON.stringify({error:{code:'visual_conflict'}}),{status:409,headers:{'Content-Type':'application/json'}}));
        target.controller=controller;target.epoch++;
        lastControls.set(key,structuredClone(original));
    };
    let controlRepliesLost=0;
    const retired=new Map();
    const key=(owner,path)=>JSON.stringify([owner,path]);
    const channelPath=request=>`/api/visual-computer/sessions/${request.session_id}/human/${request.control_id}`;
    const api={getHostAgent:()=>selectedOwner,onHostAgentChange:fn=>{ownerChanged=fn;return()=>{};},async requestForAgent(path,options,owner){
        calls.push({path,options,owner});
        const match=/\/sessions\/([^/]+)/.exec(path);
        const sid=match ? decodeURIComponent(match[1]) : sessions[0];
        const target=stateFor(owner,sid);
        if(path.endsWith('/sessions'))return sessions.map(session_id=>({session_id,phase:'fenced'}));
        if(/\/sessions\/[^/]+$/.test(path))return statusChange({phase:target.controller==='agent'?'active':'fenced',session:{session_id:sid,controller:target.controller,state:target.controller==='agent'?'ready':'paused',control_epoch:target.epoch},lease:{state:'ready'}});
        if(options?.method==='POST' && /\/human\/[^/]+$/.test(path)){
            binding={schema_version:1,control_id:decodeURIComponent(path.split('/').at(-1)),consumer_id:crypto.randomUUID(),
                session_id:decodeURIComponent(path.split('/').at(-3)),lease_id:'lease',driver_id:'driver',control_epoch:target.epoch,
                scope:{tenant_id:'tenant',principal_id:'principal',agent_id:owner},issued_at:new Date().toISOString(),
                expires_at:new Date(Date.now()+expires).toISOString(),capabilities:['screenshot','click','type'],max_frame_age_seconds:15};
            lease={schema_version:1,state:'active',request:binding,
                request_sha256:createHash('sha256').update(JSON.stringify(binding)).digest('hex')};
            if(loseOpenReply)throw new Error('accepted open reply lost');
            return structuredClone(lease);
        }
        if(path.endsWith('/stop')){
            const original=JSON.parse(options.body);
            await applyControl(original,owner,target,'none');
            if(stopRepliesLost++<lostStopReplies)throw new Error('original Stop ACK lost');
            if(stopAck)await stopAck;
            return {};
        }
        if(path.endsWith('/takeover')){
            const original=JSON.parse(options.body);
            await applyControl(original,owner,target,'human');
            if(controlRepliesLost++<lostControlReplies)throw new Error('original control ACK lost');
            return {};
        }
        if(path.endsWith('/handback')){
            if(failControl)throw new Error('private control failure');
            const original=JSON.parse(options.body);
            await applyControl(original,owner,target,'agent');
            if(handbackRepliesLost++<lostHandbackReplies)throw new Error('original handback ACK lost');
            return {};
        }
        if(/\/human\/[^/]+\/actions\/[^/]+$/.test(path))return {control_id:binding.control_id,consumer_id:binding.consumer_id,receipt:{action_id:decodeURIComponent(path.split('/').at(-1)),outcome:'completed'}};
        if(path.endsWith('/observation'))return {control_id:binding.control_id,consumer_id:binding.consumer_id,image_base64:'eA==',
            frame:{observation_id:'human-frame',session_id:binding.session_id,control_epoch:binding.control_epoch,media_type:'image/png',width:800,height:600,observed_at:new Date().toISOString()}};
        if(path.endsWith('/actions')) {
            if(failInput)throw new Error('private diagnostic must not be rendered');
            const id=JSON.parse(options.body).action.action_id;
            return {control_id:binding.control_id,consumer_id:binding.consumer_id,receipt:{action_id:id,outcome:'completed'}};
        }
        if(path.endsWith('/revoke')){
            revokeCount++;
            if(deferFirstRevoke && revokeCount===1){await firstRevoke;return {revoked:true};}
            if(externalRetirement409 && retired.has(key(owner,path.slice(0,-'/revoke'.length))))
                throw await parseResponseError(new Response(JSON.stringify({error:{code:'visual_conflict'}}),{status:409,headers:{'Content-Type':'application/json'}}));
            if(failRevoke)throw new Error('private revoke diagnostic');
            return {revoked:true};
        }
        if(retired.has(key(owner,path))){
            if(metadataWait)await metadataWait;
            return metadataChange(structuredClone(retired.get(key(owner,path))));
        }
        return structuredClone(lease);
    }};
    const body=document.querySelector('main');const view=mountVisualComputer(body,api);
    const click=label=>[...body.querySelectorAll('button')].find(node=>node.textContent===label).click();
    return {dom,calls,body,view,click,acceptedLease:()=>structuredClone(lease),releaseFirstRevoke:()=>releaseFirstRevoke?.(),releaseMetadata:()=>releaseMetadata?.(),releaseStopAck:()=>releaseStopAck?.(),
        externalStop:()=>{retired.set(key(selectedOwner,channelPath(binding)),{...structuredClone(lease),state:'revoked'});const target=stateFor(selectedOwner,binding.session_id);target.controller='none';target.epoch++;},
        switchSession:id=>{const node=body.querySelector('select');node.value=id;node.dispatchEvent(new dom.window.Event('change'));},
        switchAgent:(next='other')=>{selectedOwner=next;ownerChanged();}};
}

test('human view opens explicitly and input needs an independent arm and fresh frame', async()=>{
    const h=manualSetup();await h.view.refresh();
    assert.equal(h.calls.some(call=>call.path.includes('/human/')),false);
    h.click('Open human view');await flush();await flush();
    const image=h.body.querySelector('img');assert.equal(image.hidden,false);
    const text=h.body.querySelector('textarea');text.value='private operator text';
    h.click('Type private text');await flush();
    assert.equal(h.calls.filter(call=>call.path.endsWith('/actions')).length,0);
    h.click('Refresh human screen');await flush();
    text.value='private operator text';h.body.querySelector('[aria-label="Enable manual input"]').checked=true;
    h.click('Type private text');await flush();
    const sent=h.calls.filter(call=>call.path.endsWith('/actions'));assert.equal(sent.length,1);
    const payload=JSON.parse(sent[0].options.body);assert.equal(payload.action.action.text,'private operator text');
    assert.equal(payload.consumer_id,h.acceptedLease().request.consumer_id);assert.equal(payload.action.observation_id,'human-frame');
    assert.equal(text.value,'');assert.equal(h.body.querySelector('[aria-label="Enable manual input"]').checked,false);
    assert.equal(image.hasAttribute('src'),false);
    h.click('Type private text');await flush();assert.equal(h.calls.filter(call=>call.path.endsWith('/actions')).length,1);
    h.view.destroy();await flush();
});

test('manual keyboard needs explicit focus and consumes the fresh frame once', async()=>{
    const h=manualSetup();await h.view.refresh();h.click('Open human view');await flush();await flush();
    const image=h.body.querySelector('img');
    image.dispatchEvent(new h.dom.window.KeyboardEvent('keydown',{key:'End',bubbles:true}));
    await flush();assert.equal(h.calls.filter(call=>call.path.endsWith('/actions')).length,0);
    h.body.querySelector('[aria-label="Enable manual input"]').checked=true;
    h.click('Focus screen keyboard');await flush();assert.equal(document.activeElement,image);
    image.dispatchEvent(new h.dom.window.KeyboardEvent('keydown',{key:'End',bubbles:true}));
    image.dispatchEvent(new h.dom.window.KeyboardEvent('keydown',{key:'Enter',bubbles:true}));
    await flush();
    const inputs=h.calls.filter(call=>call.path.endsWith('/actions'));assert.equal(inputs.length,1);
    assert.deepEqual(JSON.parse(inputs[0].options.body).action.action,{kind:'key',key:'End'});
    assert.equal(image.hasAttribute('src'),false);
    h.view.destroy();await flush();
});

test('human error retains original input without retry and closing retires exact binding', async()=>{
    const h=manualSetup({failInput:true});await h.view.refresh();h.click('Open human view');await flush();await flush();
    h.body.querySelector('textarea').value='private';h.body.querySelector('[aria-label="Enable manual input"]').checked=true;
    h.click('Type private text');await flush();
    assert.equal(h.body.querySelector('img').hasAttribute('src'),false);
    assert.equal(h.body.textContent.includes('private diagnostic'),false);
    h.click('Type private text');await flush();assert.equal(h.calls.filter(call=>call.path.endsWith('/actions')).length,1);
    h.view.hide();await flush();
    const retire=h.calls.find(call=>call.path.endsWith('/revoke'));
    assert.equal(retire.owner,'owner');assert.equal(JSON.parse(retire.options.body).control_epoch,1);
    h.view.destroy();
});

test('human lease expiry clears pixels and closes without renewal or reopening', async()=>{
    const h=manualSetup({expires:30});await h.view.refresh();h.click('Open human view');await flush();await flush();
    await new Promise(resolve=>setTimeout(resolve,45));
    assert.equal(h.body.querySelector('img').hasAttribute('src'),false);
    assert.equal(h.calls.filter(call=>call.path.endsWith('/revoke')).length,1);
    assert.equal(h.calls.filter(call=>call.options?.method==='POST' && /\/human\/[^/]+$/.test(call.path)).length,1);
    h.view.destroy();
});

test('handback never sends sharing confirmation unless operator explicitly checks it', async()=>{
    const h=manualSetup();await h.view.refresh();h.click('Hand back');await flush();
    assert.equal(h.calls.some(call=>call.path.endsWith('/handback')),false);
    h.body.querySelector('[aria-label="Screen is ready to share with the agent"]').checked=true;
    h.click('Hand back');await flush();
    const call=h.calls.find(call=>call.path.endsWith('/handback'));
    assert.equal(JSON.parse(call.options.body).share_screen,true);
    h.view.destroy();
});

for (const [label, suffix] of [['Stop input','stop'], ['Take over','takeover']]) {
    test(`${label} preserves the original uncertain manual input for reconciliation`, async()=>{
        const h=manualSetup({failInput:true});
        try {
            await h.view.refresh();h.click('Open human view');await flush();await flush();
            h.body.querySelector('textarea').value='private operator text';
            h.body.querySelector('[aria-label="Enable manual input"]').checked=true;
            h.click('Type private text');await flush();
            const input=h.calls.find(call=>call.path.endsWith('/actions'));
            assert.ok(input);
            const original=JSON.parse(input.options.body);
            h.click(label);await flush();await flush();
            assert.equal(h.calls.filter(call=>call.path.endsWith('/'+suffix)).length,1);
            assert.equal(h.body.querySelector('img').hasAttribute('src'),false);
            assert.equal(h.body.querySelector('textarea').value,'');
            assert.equal(h.body.querySelector('[aria-label="Enable manual input"]').checked,false);
            h.click('Reconcile last manual input');await flush();
            const reconciliations=h.calls.filter(call=>/\/human\/[^/]+\/actions\/[^/]+$/.test(call.path));
            assert.equal(reconciliations.length,1,'the original action remains available after control changes');
            assert.equal(reconciliations[0].path,input.path+'/'+encodeURIComponent(original.action.action_id));
            assert.equal(reconciliations[0].options,undefined);
            assert.equal(reconciliations[0].owner,input.owner);
            assert.match(h.body.textContent,/Original manual input reconciled/);
            h.click('Reconcile last manual input');await flush();
            assert.equal(h.calls.filter(call=>/\/human\/[^/]+\/actions\/[^/]+$/.test(call.path)).length,1);
            assert.equal(h.calls.filter(call=>call.path.endsWith('/actions')).length,1,'input is never retried');
            assert.equal(h.body.querySelector('img').hasAttribute('src'),false);
            assert.equal(h.body.textContent.includes('private diagnostic'),false);
        } finally {
            h.view.destroy();await flush();h.dom.window.close();
        }
    });
}


for (const [label,suffix] of [['Stop input','stop'],['Take over','takeover'],['Hand back','handback']]) {
    test(`verified ${suffix} clears only its old Human retirement retry`,async()=>{
        const h=manualSetup({failRevoke:true});
        try {
            await h.view.refresh();h.click('Open human view');await flush();await flush();
            h.view.hide();await flush();
            const retires=()=>h.calls.filter(call=>call.path.endsWith('/revoke'));
            assert.equal(retires().length,1);
            const original=retires()[0];
            if(suffix==='handback')h.body.querySelector('[aria-label="Screen is ready to share with the agent"]').checked=true;
            h.click(label);await flush();await flush();
            assert.equal(h.calls.filter(call=>call.path.endsWith('/'+suffix)).length,1);
            const count=retires().length;
            h.view.hide();await flush();
            assert.equal(retires().length,count,'verified control ACK must retire pending channel retry');
            assert.equal(h.body.querySelector('img').hasAttribute('src'),false);
            assert.equal(h.body.querySelector('[aria-label="Enable manual input"]').checked,false);
            assert.equal(h.body.querySelector('textarea').value,'');
            if(suffix!=='takeover'){h.click('Take over');await flush();await flush();}
            h.click('Open human view');await flush();await flush();
            const opens=h.calls.filter(call=>call.options?.method==='POST' && /\/human\/[^/]+$/.test(call.path));
            assert.equal(opens.length,2,'new view must open after verified original retirement');
            assert.notEqual(opens[1].path,original.path.slice(0,-'/revoke'.length));
            assert.equal(h.calls.filter(call=>call.path.endsWith('/actions')).length,0);
            assert.ok(h.calls.every(call=>call.owner==='owner'));
            assert.equal(h.body.textContent.includes('private revoke diagnostic'),false);
        } finally {h.releaseFirstRevoke();h.view.destroy();await flush();h.dom.window.close();}
    });
}

test('failed control keeps the exact unresolved Human retirement retry',async()=>{
    const h=manualSetup({failRevoke:true,failControl:true});
    try {
        await h.view.refresh();h.click('Open human view');await flush();await flush();
        h.view.hide();await flush();
        const first=h.calls.find(call=>call.path.endsWith('/revoke'));
        h.body.querySelector('[aria-label="Screen is ready to share with the agent"]').checked=true;
        h.click('Hand back');await flush();await flush();
        h.view.hide();await flush();
        const retires=h.calls.filter(call=>call.path.endsWith('/revoke'));
        assert.equal(retires.length,2);
        assert.equal(retires[1].path,first.path);assert.equal(retires[1].options.body,first.options.body);
        assert.equal(retires[1].owner,first.owner);
        assert.equal(h.body.querySelector('img').hasAttribute('src'),false);
        assert.equal(h.calls.filter(call=>call.path.endsWith('/actions')).length,0);
        assert.equal(h.body.textContent.includes('private control failure'),false);
    } finally {h.releaseFirstRevoke();h.view.destroy();await flush();h.dom.window.close();}
});

test('a late original retirement reply cannot erase a newer channel retirement',async()=>{
    const h=manualSetup({failRevoke:true,deferFirstRevoke:true});
    try {
        await h.view.refresh();h.click('Open human view');await flush();await flush();
        h.view.hide();await flush();
        const first=h.calls.find(call=>call.path.endsWith('/revoke'));
        h.body.querySelector('[aria-label="Screen is ready to share with the agent"]').checked=true;
        h.click('Hand back');await flush();await flush();
        h.click('Take over');await flush();await flush();h.click('Open human view');await flush();await flush();
        const opens=h.calls.filter(call=>call.options?.method==='POST' && /\/human\/[^/]+$/.test(call.path));
        assert.equal(opens.length,2,'new view must open while only the old network reply remains delayed');
        h.view.hide();await flush();
        const second=h.calls.filter(call=>call.path.endsWith('/revoke')).at(-1);
        assert.notEqual(second.path,first.path);
        h.releaseFirstRevoke();await flush();await flush();
        h.view.hide();await flush();
        const retires=h.calls.filter(call=>call.path.endsWith('/revoke'));
        assert.equal(retires.length,3,'old completion must not clear the newer unresolved retirement');
        assert.equal(retires[2].path,second.path);assert.equal(retires[2].options.body,second.options.body);
        assert.equal(h.calls.filter(call=>call.path.endsWith('/actions')).length,0);
        assert.equal(h.body.querySelector('img').hasAttribute('src'),false);
    } finally {h.releaseFirstRevoke();await flush();await flush();h.view.destroy();await flush();h.dom.window.close();}
});

test('another agent control acknowledgement preserves the original retirement',async()=>{
    const h=manualSetup({failRevoke:true});
    try {
        await h.view.refresh();h.click('Open human view');await flush();await flush();
        h.view.hide();await flush();
        const first=h.calls.find(call=>call.path.endsWith('/revoke'));
        h.switchAgent();await flush();await h.view.refresh();
        h.click('Take over');await flush();await flush();
        const control=h.calls.find(call=>call.path.endsWith('/takeover'));
        assert.equal(control.owner,'other');
        const count=h.calls.filter(call=>call.path.endsWith('/revoke')).length;
        h.view.hide();await flush();
        const retires=h.calls.filter(call=>call.path.endsWith('/revoke'));
        assert.equal(retires.length,count+1,'another agent ACK cannot retire the original channel');
        assert.equal(retires.at(-1).owner,'owner');
        assert.equal(retires.at(-1).path,first.path);
        assert.equal(retires.at(-1).options.body,first.options.body);
        assert.equal(h.calls.filter(call=>call.path.endsWith('/actions')).length,0);
        assert.equal(h.body.querySelector('img').hasAttribute('src'),false);
    } finally {h.view.destroy();await flush();h.dom.window.close();}
});


test('refresh preserves the exact original control through repeated lost replies',async()=>{
    const h=manualSetup({lostControlReplies:2});
    try {
        await h.view.refresh();
        for(let attempt=0;attempt<3;attempt++){
            h.click('Take over');await flush();await flush();
            await h.view.refresh();
        }
        const controls=h.calls.filter(call=>call.path.endsWith('/takeover'));
        assert.equal(controls.length,3);
        for(const call of controls){
            assert.equal(call.options.body,controls[0].options.body,'refresh must not replace the original lost-ACK request');
            assert.equal(call.owner,controls[0].owner);
        }
        assert.equal(h.calls.filter(call=>call.path.endsWith('/actions')).length,0);
    } finally {h.view.destroy();await flush();h.dom.window.close();}
});


test('Release retains its original owner and session while Human close is delayed',async()=>{
    const h=manualSetup({deferFirstRevoke:true,sessions:['session','another']});
    try {
        await h.view.refresh();h.click('Open human view');await flush();await flush();
        h.click('Release environment');await flush();
        h.switchAgent();await h.view.refresh();h.switchSession('another');await flush();
        h.releaseFirstRevoke();await flush();await flush();
        const releases=h.calls.filter(call=>call.path.endsWith('/release'));
        assert.equal(releases.length,1);
        assert.equal(releases[0].owner,'owner','release must not borrow the newly selected agent');
        assert.equal(releases[0].path,'/api/visual-computer/sessions/session/release');
        assert.equal(h.calls.filter(call=>call.path.endsWith('/actions')).length,0);
    } finally {h.releaseFirstRevoke();h.view.destroy();await flush();h.dom.window.close();}
});


test('external retirement409 reconciles only the acknowledged original channel',async()=>{
    const h=manualSetup({externalRetirement409:true});
    try {
        await h.view.refresh();h.click('Open human view');await flush();await flush();
        const original=h.acceptedLease();
        h.externalStop();h.view.hide();await flush();await flush();
        const revoke=h.calls.find(call=>call.path.endsWith('/revoke'));
        const metadata=h.calls.find(call=>call.path===revoke.path.slice(0,-'/revoke'.length) && !call.options);
        assert.ok(metadata,'definitive409 must reconcile through authenticated original-channel metadata');
        assert.equal(metadata.owner,revoke.owner);
        assert.equal(JSON.parse(revoke.options.body).control_epoch,original.request.control_epoch);
        await h.view.refresh();h.click('Take over');await flush();await flush();
        h.click('Open human view');await flush();await flush();
        const opens=h.calls.filter(call=>call.options?.method==='POST' && /\/human\/[^/]+$/.test(call.path));
        assert.equal(opens.length,2,'verified fresh control must not remain blocked by an already retired old consumer');
        assert.notEqual(opens[0].path,opens[1].path);
        assert.equal(h.calls.filter(call=>call.path===revoke.path).length,1,'a definitively refused revoke is not resent');
        assert.equal(h.calls.filter(call=>call.path.endsWith('/actions')).length,0);
    } finally {h.view.destroy();await flush();h.dom.window.close();}
});

for(const [name,change] of [
    ['foreign digest',value=>({...value,request_sha256:'0'.repeat(64)})],
    ['foreign owner',value=>({...value,request:{...value.request,scope:{...value.request.scope,agent_id:'foreign'}}})],
    ['foreign channel',value=>({...value,request:{...value.request,consumer_id:'foreign'}})],
    ['unbound metadata',value=>({...value,request:undefined})],
    ['missing digest',value=>({...value,request_sha256:undefined})],
    ['nonterminal lease',value=>({...value,state:'active'})],
]) {
    test(`retirement409 keeps debt for ${name}`,async()=>{
        const h=manualSetup({externalRetirement409:true,metadataChange:change});
        try {
            await h.view.refresh();h.click('Open human view');await flush();await flush();
            h.externalStop();h.view.hide();await flush();await flush();
            const first=h.calls.find(call=>call.path.endsWith('/revoke'));
            await h.view.refresh();h.click('Take over');await flush();await flush();
            h.click('Open human view');await flush();await flush();
            h.view.hide();await flush();await flush();
            assert.equal(h.calls.filter(call=>call.options?.method==='POST' && /\/human\/[^/]+$/.test(call.path)).length,1);
            assert.equal(h.calls.filter(call=>call.path===first.path).length,1,'409 is a refusal, not permission to resend R');
            const metadata=h.calls.filter(call=>call.path===first.path.slice(0,-'/revoke'.length) && !call.options);
            assert.ok(metadata.length>=2,'retry only metadata after a definitive refusal');
            assert.ok(metadata.every(call=>call.owner===first.owner));
            assert.equal(h.calls.filter(call=>call.path.endsWith('/actions')).length,0);
        } finally {h.view.destroy();await flush();h.dom.window.close();}
    });
}

test('unacknowledged open cannot manufacture an accepted digest for409 recovery',async()=>{
    const h=manualSetup({externalRetirement409:true,loseOpenReply:true});
    try {
        await h.view.refresh();h.click('Open human view');await flush();await flush();
        h.externalStop();h.view.hide();await flush();await flush();
        await h.view.refresh();h.click('Take over');await flush();await flush();
        h.click('Open human view');await flush();await flush();
        assert.equal(h.calls.filter(call=>call.options?.method==='POST' && /\/human\/[^/]+$/.test(call.path)).length,1);
        assert.equal(h.body.querySelector('img').hasAttribute('src'),false);
    } finally {h.view.destroy();await flush();h.dom.window.close();}
});

for(const scope of ['agent','session']) {
    test(`unresolved original channel does not block a different ${scope}`,async()=>{
        const h=manualSetup({failRevoke:true,sessions:['session','another']});
        try {
            await h.view.refresh();h.click('Open human view');await flush();await flush();
            h.view.hide();await flush();const first=h.calls.find(call=>call.path.endsWith('/revoke'));
            if(scope==='agent'){h.switchAgent();await h.view.refresh();}
            else {h.switchSession('another');await flush();}
            h.click('Open human view');await flush();await flush();
            const opens=h.calls.filter(call=>call.options?.method==='POST' && /\/human\/[^/]+$/.test(call.path));
            assert.equal(opens.length,2,'unrelated channel retirement must not block this scope');
            assert.ok(scope==='agent' ? opens[0].owner!==opens[1].owner : opens[0].path!==opens[1].path);
            const retries=h.calls.filter(call=>call.path===first.path);
            assert.ok(retries.length>=2);
            assert.ok(retries.every(call=>call.owner===first.owner && call.options.body===first.options.body));
        } finally {h.view.destroy();await flush();h.dom.window.close();}
    });
}

test('late terminal metadata cannot clear a different current Human frame or input',async()=>{
    const h=manualSetup({externalRetirement409:true,deferMetadata:true});
    try {
        await h.view.refresh();h.click('Open human view');await flush();await flush();
        h.externalStop();h.view.hide();await flush();
        h.switchAgent();await h.view.refresh();h.click('Take over');await flush();await flush();
        h.click('Open human view');await flush();await flush();
        const image=h.body.querySelector('img');const src=image.getAttribute('src');assert.ok(src);
        h.body.querySelector('textarea').value='new private input';
        h.body.querySelector('[aria-label="Enable manual input"]').checked=true;
        h.releaseMetadata();await flush();await flush();
        assert.equal(image.getAttribute('src'),src);
        assert.equal(h.body.querySelector('textarea').value,'new private input');
        assert.equal(h.body.querySelector('[aria-label="Enable manual input"]').checked,true);
        assert.equal(h.calls.filter(call=>call.path.endsWith('/actions')).length,0);
    } finally {h.releaseMetadata();h.view.destroy();await flush();h.dom.window.close();}
});

test('lost control replies retain distinct original IDs across agent switches',async()=>{
    const h=manualSetup({lostControlReplies:2});
    try {
        await h.view.refresh();h.click('Take over');await flush();await flush();
        h.switchAgent();await h.view.refresh();h.click('Take over');await flush();await flush();
        h.switchAgent('owner');await h.view.refresh();h.click('Take over');await flush();await flush();
        h.switchAgent();await h.view.refresh();h.click('Take over');await flush();await flush();
        const controls=h.calls.filter(call=>call.path.endsWith('/takeover'));
        assert.equal(controls.length,4);
        assert.equal(controls[0].options.body,controls[2].options.body);
        assert.equal(controls[1].options.body,controls[3].options.body);
        assert.notEqual(controls[0].options.body,controls[1].options.body);
        assert.deepEqual(controls.map(call=>call.owner),['owner','other','owner','other']);
    } finally {h.view.destroy();await flush();h.dom.window.close();}
});


for(const [label,suffix,option] of [['Take over','takeover','lostControlReplies'],['Hand back','handback','lostHandbackReplies']]) {
    test(`dedicated Stop stays available after lost ${suffix} reply and refresh`,async()=>{
        const h=manualSetup({[option]:1});let original,stops,shares;
        try {
            await h.view.refresh();
            if(suffix==='handback')h.body.querySelector('[aria-label="Screen is ready to share with the agent"]').checked=true;
            h.click(label);await flush();await flush();
            await h.view.refresh();
            shares=h.body.querySelector('[aria-label="Screen is ready to share with the agent"]').checked;
            h.click('Stop input');await flush();await flush();
            original=h.calls.filter(call=>call.path.endsWith('/'+suffix));
            stops=h.calls.filter(call=>call.path.endsWith('/stop'));
        } finally {h.view.destroy();await flush();h.dom.window.close();}
        assert.equal(original.length,1,'Stop must not retry the unknown controller operation');
        assert.equal(stops.length,1,'dedicated Stop must remain independently usable');
        const control=JSON.parse(original[0].options.body);const pause=JSON.parse(stops[0].options.body);
        assert.notEqual(pause.request_id,control.request_id);
        assert.equal(pause.operation,'pause');assert.equal(pause.expected_control_epoch,2);
        assert.equal(stops[0].owner,original[0].owner);
        assert.equal(shares,false,'Stop does not borrow or require handback sharing consent');
        if(suffix==='handback')assert.equal(control.share_screen,true);
    });
}

test('lost Stop replies keep stable S then fresh takeover without replaying unresolved R',async()=>{
    const h=manualSetup({lostControlReplies:1,lostStopReplies:2});let controls,stops;
    try {
        await h.view.refresh();h.click('Take over');await flush();await flush();await h.view.refresh();
        for(let attempt=0;attempt<3;attempt++){
            h.click('Stop input');await flush();await flush();await h.view.refresh();
        }
        // The actual host retains only last_control. The old R cannot replay
        // after S: a fresh explicit intent must use the new status epoch.
        h.click('Take over');await flush();await flush();
        controls=h.calls.filter(call=>call.path.endsWith('/takeover'));
        stops=h.calls.filter(call=>call.path.endsWith('/stop'));
    } finally {h.view.destroy();await flush();h.dom.window.close();}
    assert.equal(stops.length,3);
    assert.ok(stops.every(call=>call.options.body===stops[0].options.body && call.owner===stops[0].owner),'Stop retries retain only their exact original S');
    assert.equal(controls.length,2);
    assert.match(h.body.textContent,/earlier control outcome unresolved/,'original R remains unclassified descriptive history');
    assert.notEqual(JSON.parse(controls[0].options.body).request_id,JSON.parse(controls[1].options.body).request_id);
    assert.equal(JSON.parse(controls[1].options.body).expected_control_epoch,3,'new explicit takeover uses the verified current epoch');
    assert.notEqual(JSON.parse(stops[0].options.body).request_id,JSON.parse(controls[0].options.body).request_id);
});

test('late original Stop acknowledgement cannot clear another owner current Human frame',async()=>{
    const h=manualSetup({lostControlReplies:1,deferStopAck:true});
    try {
        await h.view.refresh();h.click('Take over');await flush();await flush();await h.view.refresh();
        h.click('Stop input');await flush();
        h.switchAgent();await h.view.refresh();h.click('Take over');await flush();await flush();
        h.click('Open human view');await flush();await flush();
        const image=h.body.querySelector('img');const src=image.getAttribute('src');assert.ok(src);
        h.body.querySelector('textarea').value='current private text';
        h.body.querySelector('[aria-label="Enable manual input"]').checked=true;
        h.releaseStopAck();await flush();await flush();
        assert.equal(image.getAttribute('src'),src);
        assert.equal(h.body.querySelector('textarea').value,'current private text');
        assert.equal(h.body.querySelector('[aria-label="Enable manual input"]').checked,true);
        const stop=h.calls.find(call=>call.path.endsWith('/stop'));assert.equal(stop.owner,'owner');
        assert.equal(h.calls.filter(call=>call.path.endsWith('/actions')).length,0);
    } finally {h.releaseStopAck();h.view.destroy();await flush();h.dom.window.close();}
});


test('refused stale Stop requires fresh status before a new explicit safety intent',async()=>{
    const h=manualSetup({lostControlReplies:1});let controls,stops,beforeRefresh;
    try {
        await h.view.refresh();h.click('Take over');await flush();await flush();
        // No Refresh: the server-side takeover advanced the epoch, while the
        // panel still holds its original cached epoch. Canonical409, not loss.
        h.click('Stop input');await flush();await flush();
        h.click('Stop input');await flush();await flush();
        beforeRefresh=h.calls.filter(call=>call.path.endsWith('/stop')).length;
        await h.view.refresh();
        h.click('Stop input');await flush();await flush();
        controls=h.calls.filter(call=>call.path.endsWith('/takeover'));
        stops=h.calls.filter(call=>call.path.endsWith('/stop'));
    } finally {h.view.destroy();await flush();h.dom.window.close();}
    assert.equal(controls.length,1);
    assert.equal(beforeRefresh,1,'a refused S cannot be resent before fresh status');
    assert.equal(stops.length,2);
    assert.match(h.body.textContent,/earlier control outcome unresolved/);
    const first=JSON.parse(stops[0].options.body);const next=JSON.parse(stops[1].options.body);
    assert.equal(first.expected_control_epoch,1);assert.equal(next.expected_control_epoch,2);
    assert.notEqual(first.request_id,next.request_id,'a fresh explicit Stop must not resend definitively refused S');
});


for(const [name,change] of [
    ['unchanged epoch',value=>({...value,session:{...value.session,control_epoch:2}})],
    ['foreign session',value=>({...value,session:{...value.session,session_id:'foreign'}})],
    ['pending control',value=>({...value,pending_control:{request_id:'other',session_id:'session',expected_control_epoch:3,operation:'takeover'}})],
]) {
    test(`acknowledged Stop does not retire controller intent from ${name} status`,async()=>{
        let blocked=false;
        const h=manualSetup({lostControlReplies:1,statusChange:value=>blocked ? change(value) : value});
        try {
            await h.view.refresh();h.click('Take over');await flush();await flush();await h.view.refresh();
            blocked=true;h.click('Stop input');await flush();await flush();
            h.click('Take over');await flush();await flush();
            assert.equal(h.calls.filter(call=>call.path.endsWith('/takeover')).length,1,'S needs its own fresh current status before a new controller intent');
            blocked=false;await h.view.refresh();h.click('Take over');await flush();await flush();
            const controls=h.calls.filter(call=>call.path.endsWith('/takeover'));
            assert.equal(controls.length,2);
            assert.equal(JSON.parse(controls[1].options.body).expected_control_epoch,3);
            assert.notEqual(controls[0].options.body,controls[1].options.body);
            assert.match(h.body.textContent,/earlier control outcome unresolved/);
        } finally {h.view.destroy();await flush();h.dom.window.close();}
    });
}

// Real pending-control HTTP contract is pinned by test_ui_pending_control_contract.py.
// The following transports are scripted DOM data and confer no native authority.
const raceUnknown=()=>parseResponseError(new Response(JSON.stringify({error:{code:'visual_outcome_unknown'}}),{status:503,headers:{'Content-Type':'application/json'}}));
async function recordRace(name,facts){if(process.env.UI_BASELINE_OUTPUT)await writeFile(process.env.UI_BASELINE_OUTPUT+'/'+name+'.json',JSON.stringify(facts,null,2)+'\n');}
function raceSetup({pendingOperation=null,deferInput=false,rejectInput=false}={}){
 const dom=new JSDOM('<main></main>',{url:'http://localhost/'});globalThis.document=dom.window.document;
 const calls=[];let epoch=pendingOperation==='takeover'?0:1,controller=pendingOperation==='takeover'?'agent':'human';
 let pending=null,binding=null,lease=null,frame=0,resolveInput,rejectOriginalInput,request=null;
 const api={getHostAgent:()=> 'owner',onHostAgentChange:()=>()=>{},async requestForAgent(path,options,owner){
  calls.push({path,options,owner});
  if(path.endsWith('/sessions'))return [{session_id:'session',phase:controller==='agent'?'active':'fenced'}];
  if(/\/sessions\/[^/]+$/.test(path))return {phase:controller==='agent'?'active':'fenced',session:{session_id:'session',controller,control_epoch:epoch,state:controller==='agent'?'ready':'paused'},lease:{state:'ready'},pending_control:structuredClone(pending)};
  if(path.endsWith('/takeover')||path.endsWith('/handback')||path.endsWith('/stop')){
   const value=JSON.parse(options.body);
   const canonical={schema_version:1,request_id:value.request_id,session_id:value.session_id,expected_control_epoch:value.expected_control_epoch,operation:value.operation};
   if(pendingOperation){
    if(pending===null){pending=canonical;throw await raceUnknown();}
    if(pending.request_id!==value.request_id){throw await raceUnknown();}
    assert.deepEqual(canonical,pending);pending=null;
   }
   controller=value.operation==='resume'?'agent':value.operation==='pause'?'none':'human';epoch++;
   return {session_id:'session',controller,control_epoch:epoch};
  }
  if(options?.method==='POST'&&/\/human\/[^/]+$/.test(path)){
   binding={schema_version:1,control_id:path.split('/').at(-1),consumer_id:crypto.randomUUID(),session_id:'session',lease_id:'lease',driver_id:'driver',control_epoch:epoch,scope:{tenant_id:'tenant',principal_id:'principal',agent_id:owner},issued_at:new Date().toISOString(),expires_at:new Date(Date.now()+120000).toISOString(),capabilities:['screenshot','click','type','key'],max_frame_age_seconds:15};
   lease={schema_version:1,state:'active',request:binding,request_sha256:'a'.repeat(64)};return structuredClone(lease);
  }
  if(path.endsWith('/observation'))return {control_id:binding.control_id,consumer_id:binding.consumer_id,image_base64:'eA==',frame:{observation_id:'human-frame-'+(++frame),session_id:'session',control_epoch:epoch,media_type:'image/png',width:800,height:600,observed_at:new Date().toISOString()}};
  if(options?.method==='POST'&&path.endsWith('/actions')){request=JSON.parse(options.body);if(deferInput)return new Promise((resolve,reject)=>{resolveInput=resolve;rejectOriginalInput=reject;});}
  if(/\/actions\/[^/]+$/.test(path))return {control_id:binding.control_id,consumer_id:binding.consumer_id,receipt:{action_id:path.split('/').at(-1),outcome:'completed'}};
  if(path.endsWith('/revoke'))return {revoked:true};
  throw new Error('undeclared scripted endpoint '+path);
 }};
 const body=document.querySelector('main'),view=mountVisualComputer(body,api);
 return {dom,body,view,calls,pending:()=>structuredClone(pending),request:()=>structuredClone(request),
 click:label=>[...body.querySelectorAll('button')].find(x=>x.textContent===label).click(),
 releaseInput:()=>{if(rejectInput)rejectOriginalInput?.(new Error('held original input response failed'));else resolveInput?.({control_id:binding.control_id,consumer_id:binding.consumer_id,receipt:{action_id:request.action.action_id,outcome:'completed'}});}};
}
for(const operation of ['takeover','handback'])test('pending '+operation+' can reconcile original R after refused Stop S',async()=>{
 const h=raceSetup({pendingOperation:operation});let facts;
 try{
  await h.view.refresh();
  if(operation==='handback')h.body.querySelector('[aria-label="Screen is ready to share with the agent"]').checked=true;
  const label=operation==='takeover'?'Take over':'Hand back';h.click(label);await flush();await flush();
  const original=h.calls.find(c=>c.path.endsWith('/'+operation));assert.ok(original);
  const R=JSON.parse(original.options.body);assert.equal(h.pending().request_id,R.request_id);
  h.click('Stop input');await flush();await flush();
  const S=h.calls.find(c=>c.path.endsWith('/stop'));assert.ok(S);assert.notEqual(JSON.parse(S.options.body).request_id,R.request_id);
  h.click('Refresh sessions');await flush();await flush();assert.equal(h.pending().request_id,R.request_id);
  if(operation==='handback')h.body.querySelector('[aria-label="Screen is ready to share with the agent"]').checked=true;
  h.click(label);await flush();await flush();
  const retries=h.calls.filter(c=>c.path.endsWith('/'+operation));
  const reconciledState=h.body.textContent;const stopsBefore=h.calls.filter(c=>c.path.endsWith('/stop')).length;
  h.click('Stop input');await flush();await flush();
  const stopsAfter=h.calls.filter(c=>c.path.endsWith('/stop'));
  facts={operation,reconciledState,stopsBefore,stopsAfter:stopsAfter.map(c=>JSON.parse(c.options.body)),R,S:JSON.parse(S.options.body),R_attempts:retries.map(c=>JSON.parse(c.options.body)),pending:h.pending(),owners:h.calls.map(c=>c.owner),state:h.body.textContent};
 }finally{h.view.destroy();await flush();h.dom.window.close();}
 await recordRace('F1-'+operation,facts);
 assert.equal(facts.R_attempts.length,2,'original R must remain explicitly reconcilable while server pending_control is R');
 assert.deepEqual(facts.R_attempts[0],facts.R_attempts[1]);assert.equal(facts.stopsBefore,1,'R reconciliation must not resend S');assert.equal(facts.stopsAfter.length,2,'a NEXT explicit Stop remains usable');assert.notEqual(facts.stopsAfter[0].request_id,facts.stopsAfter[1].request_id,'unclassified S stays history, never reused as the next safety intent');assert.match(facts.reconciledState,/earlier control outcome unresolved/);assert.ok(facts.owners.every(x=>x==='owner'));
});
for(const input of ['type-button','keyboard-handler'])for(const rejected of [false,true])test('late '+input+(rejected?' error':' reply')+' cannot erase reconciled replacement frame',async()=>{
 const h=raceSetup({deferInput:true,rejectInput:rejected});let facts;
 try{
  await h.view.refresh();h.click('Open human view');await flush();await flush();
  const image=h.body.querySelector('img'),enable=h.body.querySelector('[aria-label="Enable manual input"]');assert.equal(image.hidden,false);
  enable.checked=true;
  if(input==='type-button'){h.body.querySelector('textarea').value='controlled-private-input';h.click('Type private text');}
  else{h.click('Focus screen keyboard');await flush();assert.equal(document.activeElement,image);image.dispatchEvent(new h.dom.window.KeyboardEvent('keydown',{key:'End',bubbles:true}));}
  await flush();assert.ok(h.request());assert.equal(image.hasAttribute('src'),false);
  const original=h.request();h.click('Reconcile last manual input');await flush();await flush();
  const status=h.calls.find(c=>/\/actions\/[^/]+$/.test(c.path));assert.equal(status.path.split('/').at(-1),original.action.action_id);
  h.click('Refresh human screen');await flush();await flush();enable.checked=true;
  assert.equal(image.hidden,false);assert.equal(image.hasAttribute('src'),true);assert.equal(enable.checked,true);
  const replacement=image.getAttribute('src');h.releaseInput();await flush();await flush();
  facts={input,original,status_action_id:status.path.split('/').at(-1),replacement_src:replacement,after_src:image.getAttribute('src'),after_hidden:image.hidden,after_armed:enable.checked,post_count:h.calls.filter(c=>c.options?.method==='POST'&&c.path.endsWith('/actions')).length,state:h.body.querySelector('fieldset').textContent};
 }finally{h.releaseInput();await flush();h.view.destroy();await flush();h.dom.window.close();}
 await recordRace('F2-'+input+(rejected?'-error':'-reply'),facts);
 assert.equal(facts.post_count,1);assert.equal(facts.status_action_id,facts.original.action.action_id);
 assert.equal(facts.after_src,facts.replacement_src,'late original POST must not clear replacement pixels after same-ID reconciliation');
 assert.equal(facts.after_hidden,false);assert.equal(facts.after_armed,true);assert.match(facts.state,/Private human view\./,'late completion must not replace the current view status');
});

function inputScopeSetup(){
 const dom=new JSDOM('<main></main>',{url:'http://localhost/'});globalThis.document=dom.window.document;
 const calls=[],leases=new Map(),inputs=new Map();let owner='owner-a',changed,frame=0;
 const key=(who,sid,cid)=>JSON.stringify([who,sid,cid]);
 const api={getHostAgent:()=>owner,onHostAgentChange:fn=>{changed=fn;return()=>{};},async requestForAgent(path,options,who){
  calls.push({path,options,owner:who});
  if(path.endsWith('/sessions'))return ['session-a','session-b'].map(session_id=>({session_id,phase:'fenced'}));
  const m=/\/sessions\/([^/]+)(?:\/human\/([^/]+))?/.exec(path);const sid=m?.[1],cid=m?.[2];
  if(/\/sessions\/[^/]+$/.test(path))return {phase:'fenced',session:{session_id:sid,controller:'human',control_epoch:1,state:'paused'},lease:{state:'ready'},pending_control:null};
  if(options?.method==='POST'&&/\/human\/[^/]+$/.test(path)){
   const request={schema_version:1,control_id:cid,consumer_id:crypto.randomUUID(),session_id:sid,lease_id:'lease-'+sid,driver_id:'driver',control_epoch:1,scope:{tenant_id:'tenant',principal_id:'principal',agent_id:who},issued_at:new Date().toISOString(),expires_at:new Date(Date.now()+120000).toISOString(),capabilities:['screenshot','click','type'],max_frame_age_seconds:15};
   const lease={state:'active',request,request_sha256:'a'.repeat(64)};leases.set(key(who,sid,cid),lease);return structuredClone(lease);
  }
  const lease=leases.get(key(who,sid,cid));
  if(path.endsWith('/observation')){assert.ok(lease);return {control_id:cid,consumer_id:lease.request.consumer_id,image_base64:'eA==',frame:{observation_id:'frame-'+(++frame),session_id:sid,control_epoch:1,media_type:'image/png',width:800,height:600,observed_at:new Date().toISOString()}};}
  if(options?.method==='POST'&&path.endsWith('/actions')){const value=JSON.parse(options.body);assert.equal(value.control_id,cid);assert.equal(value.consumer_id,lease.request.consumer_id);inputs.set(JSON.stringify([who,sid,cid,value.action.action_id]),value);throw await parseResponseError(new Response(JSON.stringify({error:{code:'visual_outcome_unknown'}}),{status:503,headers:{'Content-Type':'application/json'}}));}
  if(/\/actions\/[^/]+$/.test(path)){const aid=path.split('/').at(-1);assert.ok(inputs.has(JSON.stringify([who,sid,cid,aid])));return {control_id:cid,consumer_id:lease.request.consumer_id,receipt:{action_id:aid,outcome:'completed'}};}
  if(path.endsWith('/revoke')){assert.ok(lease);lease.state='revoked';return {revoked:true};}
  throw new Error('undeclared endpoint '+path);
 }};
 const body=document.querySelector('main'),view=mountVisualComputer(body,api);
 return {dom,body,view,calls,click:label=>[...body.querySelectorAll('button')].find(x=>x.textContent===label).click(),
 switchSession:sid=>{const s=body.querySelector('select');s.value=sid;s.dispatchEvent(new dom.window.Event('change'));},
 switchOwner:who=>{owner=who;changed();}};
}
for(const mode of ['same-owner-new-session','new-owner'])test('unresolved input is scoped: '+mode,async()=>{
 const h=inputScopeSetup();let facts;
 const posts=()=>h.calls.filter(c=>c.options?.method==='POST'&&c.path.endsWith('/actions'));
 const statuses=()=>h.calls.filter(c=>/\/actions\/[^/]+$/.test(c.path));
 try{
  await h.view.refresh();h.click('Open human view');await flush();await flush();
  h.body.querySelector('[aria-label="Enable manual input"]').checked=true;h.body.querySelector('textarea').value='old-scope-private';h.click('Type private text');await flush();await flush();
  assert.equal(posts().length,1);const original=structuredClone(posts()[0]);const oldRequest=JSON.parse(original.options.body);
  if(mode==='same-owner-new-session')h.switchSession('session-b');else{h.switchOwner('owner-b');await h.view.refresh();}
  await flush();await flush();h.click('Open human view');await flush();await flush();
  const newView=!h.body.querySelector('img').hidden&&h.body.querySelector('img').hasAttribute('src');
  h.body.querySelector('[aria-label="Enable manual input"]').checked=true;h.body.querySelector('textarea').value='new-scope-private';h.click('Type private text');await flush();await flush();
  const newPost=posts()[1];
  if(mode==='same-owner-new-session')h.switchSession('session-a');else{h.switchOwner('owner-a');await h.view.refresh();}
  await flush();await flush();h.click('Reconcile last manual input');await flush();await flush();
  const oldStatus=statuses().find(c=>c.owner===original.owner&&c.path.endsWith('/'+oldRequest.action.action_id));
  if(mode==='same-owner-new-session')h.switchSession('session-b');else{h.switchOwner('owner-b');await h.view.refresh();}
  await flush();await flush();h.click('Reconcile last manual input');await flush();await flush();
  facts={mode,old_post:original,new_post:newPost??null,new_scope_view_usable:newView,statuses:statuses(),original_status:oldStatus??null,post_count:posts().length,all_posts:posts(),old_action_id:oldRequest.action.action_id};
 }finally{h.view.destroy();await flush();h.dom.window.close();}
 if(process.env.UI_BASELINE_OUTPUT)await writeFile(process.env.UI_BASELINE_OUTPUT+'/'+mode+'.json',JSON.stringify(facts,null,2)+'\n');
 assert.ok(facts.original_status,'returning original scope must retain the exact original input ID');
 assert.equal(facts.original_status.owner,facts.old_post.owner);assert.equal(facts.all_posts.filter(c=>c.owner===facts.old_post.owner&&c.path===facts.old_post.path).length,1,'original input must never replay');
 assert.equal(facts.new_scope_view_usable,true,'unresolved old input must not block a different owner/session Human view');
 assert.equal(facts.post_count,2,'each scope can send only its one original input');assert.ok(facts.new_post);assert.notEqual(JSON.parse(facts.new_post.options.body).action.action_id,facts.old_action_id);
 const newID=JSON.parse(facts.new_post.options.body).action.action_id;assert.ok(facts.statuses.some(c=>c.owner===facts.new_post.owner&&c.path.endsWith('/'+newID)),'new scope keeps its own independent original ID');
});

// Shared presentation ownership regressions: original gates and actual disabled-button clicks.
function displaySetup(mode){
 const dom=new JSDOM('<main></main>',{url:'http://localhost/'});globalThis.document=dom.window.document;
 let owner='owner-a',changed,observations=0,settleObservation,settleInput,settleNewInput;const leases=new Map(),calls=[];
 const api={getHostAgent:()=>owner,onHostAgentChange:fn=>{changed=fn;return()=>{};},async requestForAgent(path,options,who){
  calls.push({path,options,owner:who});if(path.endsWith('/sessions'))return [{session_id:'session',phase:'fenced'}];
  if(/\/sessions\/[^/]+$/.test(path))return {phase:'fenced',session:{session_id:'session',controller:'human',control_epoch:1,state:'paused'},lease:{state:'ready'},pending_control:null};
  const cid=/\/human\/([^/]+)/.exec(path)?.[1],key=JSON.stringify([who,cid]);
  if(options?.method==='POST'&&/\/human\/[^/]+$/.test(path)){
   const request={schema_version:1,control_id:cid,consumer_id:crypto.randomUUID(),session_id:'session',lease_id:'lease',driver_id:'driver',control_epoch:1,scope:{tenant_id:'tenant',principal_id:'principal',agent_id:who},issued_at:new Date().toISOString(),expires_at:new Date(Date.now()+120000).toISOString(),capabilities:['screenshot','click','type'],max_frame_age_seconds:15};
   const lease={state:'active',request,request_sha256:'a'.repeat(64)};leases.set(key,lease);return structuredClone(lease);
  }
  const lease=leases.get(key);assert.ok(lease,'original scoped lease exists');
  if(path.endsWith('/observation')){
   const number=++observations;const value={control_id:cid,consumer_id:lease.request.consumer_id,image_base64:number===1?'eA==':'eQ==',frame:{observation_id:'O'+number,session_id:'session',control_epoch:1,media_type:'image/png',width:800,height:600,observed_at:new Date().toISOString()}};
   if((mode.startsWith('observation')&&number===1)||(mode==='current-observation-error'&&number===2))return new Promise((resolve,reject)=>{settleObservation=()=>mode!=='observation-success'?reject(new Error('held O1 failed')):resolve(value);});
   return value;
  }
  if(options?.method==='POST'&&path.endsWith('/actions')){
   const value=JSON.parse(options.body);const reply={control_id:cid,consumer_id:lease.request.consumer_id,receipt:{action_id:value.action.action_id,outcome:'completed'}};
   if(who==='owner-a')return new Promise(resolve=>{settleInput=()=>resolve(reply);});
   if(mode==='held-type-new-owner')return new Promise(resolve=>{settleNewInput=()=>resolve(reply);});
   return reply;
  }
  if(/\/actions\/[^/]+$/.test(path))return {control_id:cid,consumer_id:lease.request.consumer_id,receipt:{action_id:path.split('/').at(-1),outcome:'completed'}};
  if(path.endsWith('/revoke'))return {revoked:true};throw new Error('undeclared endpoint '+path);
 }};
 const body=document.querySelector('main'),view=mountVisualComputer(body,api);
 return {dom,body,view,calls,button:name=>[...body.querySelectorAll('button')].find(x=>x.textContent===name),click:name=>[...body.querySelectorAll('button')].find(x=>x.textContent===name).click(),settleObservation:()=>settleObservation?.(),settleInput:()=>settleInput?.(),settleNewInput:()=>settleNewInput?.(),switchOwner:who=>{owner=who;changed();}};
}
for(const mode of ['observation-error','observation-success'])test('manual display ownership '+mode,async()=>{
 const h=displaySetup(mode);let facts;
 try{
  await h.view.refresh();h.click('Open human view');await flush();await flush();
  assert.equal(h.button('Open human view').disabled,true);assert.equal(h.button('Refresh human screen').disabled,false,'O2 uses a distinct enabled real button');
  h.click('Refresh human screen');await flush();await flush();
  const img=h.body.querySelector('img'),enable=h.body.querySelector('[aria-label="Enable manual input"]');assert.equal(img.hidden,false);const src=img.getAttribute('src');assert.equal(src,'data:image/png;base64,eQ==');enable.checked=true;
  h.settleObservation();await flush();await flush();facts={mode,replacement:src,after:img.getAttribute('src'),armed:enable.checked,hidden:img.hidden,observation_calls:h.calls.filter(c=>c.path.endsWith('/observation')).length};
 }finally{h.settleObservation();await flush();h.view.destroy();await flush();h.dom.window.close();}
 if(process.env.UI_BASELINE_OUTPUT)await writeFile(process.env.UI_BASELINE_OUTPUT+'/'+mode+'.json',JSON.stringify(facts,null,2)+'\n');
 assert.equal(facts.observation_calls,2);assert.equal(facts.after,facts.replacement,'old O1 must not mutate O2 pixels');assert.equal(facts.hidden,false);assert.equal(facts.armed,true);
});
test('manual display ownership held-type-new-owner',async()=>{
 const h=displaySetup('held-type-new-owner');let facts;
 const posts=()=>h.calls.filter(c=>c.options?.method==='POST'&&c.path.endsWith('/actions'));
 try{
  await h.view.refresh();h.click('Open human view');await flush();await flush();h.body.querySelector('[aria-label="Enable manual input"]').checked=true;h.body.querySelector('textarea').value='old-private';h.click('Type private text');await flush();assert.equal(posts().length,1);assert.equal(h.button('Type private text').disabled,true);const old=posts()[0],oldID=JSON.parse(old.options.body).action.action_id;
  h.switchOwner('owner-b');await h.view.refresh();h.click('Open human view');await flush();await flush();const img=h.body.querySelector('img');assert.equal(img.hidden,false);h.body.querySelector('[aria-label="Enable manual input"]').checked=true;h.body.querySelector('textarea').value='new-private';const disabled=h.button('Type private text').disabled;h.click('Type private text');await flush();await flush();
  const count=posts().length;h.settleInput();await flush();await flush();const newBusyAfterOld=h.button('Type private text').disabled;h.switchOwner('owner-a');await h.view.refresh();h.click('Reconcile last manual input');await flush();await flush();const status=h.calls.find(c=>c.owner==='owner-a'&&c.path.endsWith('/actions/'+oldID));facts={mode:'held-type-new-owner',new_button_disabled:disabled,new_busy_after_old:newBusyAfterOld,posts_before_old_settled:count,oldID,old_status:status??null,all_posts:posts()};
 }finally{h.settleInput();h.settleNewInput();await flush();h.view.destroy();await flush();h.dom.window.close();}
 if(process.env.UI_BASELINE_OUTPUT)await writeFile(process.env.UI_BASELINE_OUTPUT+'/held-type-new-owner.json',JSON.stringify(facts,null,2)+'\n');
 assert.ok(facts.old_status,'old original input identity retained');assert.equal(facts.all_posts.filter(c=>c.owner==='owner-a').length,1,'old input never replayed');assert.equal(facts.posts_before_old_settled,2,'old-owner busy button must not disable new-owner input');assert.equal(facts.new_button_disabled,false);assert.equal(facts.new_busy_after_old,true,'old finally must not unlock the newer owner original input');
});

test('current owned observation failure still clears and disarms its view',async()=>{
 const h=displaySetup('current-observation-error');let facts;
 try{
  await h.view.refresh();h.click('Open human view');await flush();await flush();
  const img=h.body.querySelector('img'),enable=h.body.querySelector('[aria-label="Enable manual input"]');assert.equal(img.hidden,false);enable.checked=true;
  h.click('Refresh human screen');await flush();assert.equal(h.button('Refresh human screen').disabled,true);h.settleObservation();await flush();await flush();
  facts={src:img.getAttribute('src'),hidden:img.hidden,armed:enable.checked,text:h.body.textContent};
 }finally{h.settleObservation();await flush();h.view.destroy();await flush();h.dom.window.close();}
 assert.equal(facts.src,null,'current owned error must clear its pixels');assert.equal(facts.hidden,true);assert.equal(facts.armed,false);assert.ok(!facts.text.includes('held O1 failed'));
});
