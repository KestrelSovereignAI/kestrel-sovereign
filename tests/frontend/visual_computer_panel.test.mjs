import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
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
    deferFirstRevoke=false, lostControlReplies=0, externalRetirement409=false,
    metadataChange=value=>value, deferMetadata=false, loseOpenReply=false, sessions=['session']}={}) {
    const dom = new JSDOM('<main></main>', {url:'http://localhost/'});
    globalThis.document = dom.window.document;
    const calls=[];
    let binding;
    let lease;
    let controller='human';
    let epoch=1;
    let releaseFirstRevoke;
    const firstRevoke=deferFirstRevoke ? new Promise(resolve=>{releaseFirstRevoke=resolve;}) : null;
    let releaseMetadata;
    const metadataWait=deferMetadata ? new Promise(resolve=>{releaseMetadata=resolve;}) : null;
    let revokeCount=0;
    let selectedOwner='owner';
    let ownerChanged;
    const acceptedControls=new Map();
    let controlRepliesLost=0;
    const retired=new Map();
    const key=(owner,path)=>JSON.stringify([owner,path]);
    const channelPath=request=>`/api/visual-computer/sessions/${request.session_id}/human/${request.control_id}`;
    const api={getHostAgent:()=>selectedOwner,onHostAgentChange:fn=>{ownerChanged=fn;return()=>{};},async requestForAgent(path,options,owner){
        calls.push({path,options,owner});
        if(path.endsWith('/sessions'))return sessions.map(session_id=>({session_id,phase:'fenced'}));
        if(/\/sessions\/[^/]+$/.test(path))return {phase:controller==='agent'?'active':'fenced',session:{session_id:decodeURIComponent(path.split('/').at(-1)),controller,control_epoch:epoch},lease:{state:'ready'}};
        if(options?.method==='POST' && /\/human\/[^/]+$/.test(path)){
            binding={schema_version:1,control_id:decodeURIComponent(path.split('/').at(-1)),consumer_id:crypto.randomUUID(),
                session_id:decodeURIComponent(path.split('/').at(-3)),lease_id:'lease',driver_id:'driver',control_epoch:epoch,
                scope:{tenant_id:'tenant',principal_id:'principal',agent_id:owner},issued_at:new Date().toISOString(),
                expires_at:new Date(Date.now()+expires).toISOString(),capabilities:['screenshot','click','type'],max_frame_age_seconds:15};
            lease={schema_version:1,state:'active',request:binding,
                request_sha256:createHash('sha256').update(JSON.stringify(binding)).digest('hex')};
            if(loseOpenReply)throw new Error('accepted open reply lost');
            return structuredClone(lease);
        }
        if(path.endsWith('/stop')){controller='none';epoch++;return {};}
        if(path.endsWith('/takeover')){
            const original=JSON.parse(options.body);
            if(!acceptedControls.has(original.request_id)){controller='human';epoch++;acceptedControls.set(original.request_id,original);}
            if(controlRepliesLost++<lostControlReplies)throw new Error('original control ACK lost');
            return {};
        }
        if(path.endsWith('/handback')){if(failControl)throw new Error('private control failure');controller='agent';epoch++;return {};}
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
    return {dom,calls,body,view,click,acceptedLease:()=>structuredClone(lease),releaseFirstRevoke:()=>releaseFirstRevoke?.(),releaseMetadata:()=>releaseMetadata?.(),
        externalStop:()=>{retired.set(key(selectedOwner,channelPath(binding)),{...structuredClone(lease),state:'revoked'});controller='none';epoch++;},
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
