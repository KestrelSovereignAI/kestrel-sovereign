import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { JSDOM } from 'jsdom';

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

function manualSetup({failInput=false, expires=120000, failRevoke=false, failControl=false, deferFirstRevoke=false}={}) {
    const dom = new JSDOM('<main></main>', {url:'http://localhost/'});
    globalThis.document = dom.window.document;
    const calls=[];
    const binding={control_id:'control',consumer_id:'consumer',session_id:'session',control_epoch:1,
        expires_at:new Date(Date.now()+expires).toISOString()};
    const lease={state:'active',request:binding};
    let controller='human';
    let epoch=1;
    let releaseFirstRevoke;
    const firstRevoke=deferFirstRevoke ? new Promise(resolve=>{releaseFirstRevoke=resolve;}) : null;
    let revokeCount=0;
    let selectedOwner='owner';
    let ownerChanged;
    const api={getHostAgent:()=>selectedOwner,onHostAgentChange:fn=>{ownerChanged=fn;return()=>{};},async requestForAgent(path,options,owner){
        calls.push({path,options,owner});
        if(options?.method==='POST' && /\/human\/[^/]+$/.test(path)){binding.control_id=decodeURIComponent(path.split('/').at(-1));binding.control_epoch=epoch;}
        if(path.endsWith('/sessions'))return [{session_id:'session',phase:'fenced'}];
        if(path.endsWith('/sessions/session'))return {phase:controller==='agent'?'active':'fenced',session:{session_id:'session',controller,control_epoch:epoch},lease:{state:'ready'}};
        if(path.endsWith('/stop')){controller='none';epoch++;return {};}
        if(path.endsWith('/takeover')){controller='human';epoch++;return {};}
        if(path.endsWith('/handback')){if(failControl)throw new Error('private control failure');controller='agent';epoch++;return {};}
        if(/\/human\/[^/]+\/actions\/[^/]+$/.test(path))return {control_id:binding.control_id,consumer_id:binding.consumer_id,receipt:{action_id:decodeURIComponent(path.split('/').at(-1)),outcome:'completed'}};
        if(path.endsWith('/observation'))return {control_id:binding.control_id,consumer_id:'consumer',image_base64:'eA==',
            frame:{observation_id:'human-frame',session_id:'session',control_epoch:binding.control_epoch,media_type:'image/png',width:800,height:600,observed_at:new Date().toISOString()}};
        if(path.endsWith('/actions')) {
            if(failInput)throw new Error('private diagnostic must not be rendered');
            const id=JSON.parse(options.body).action.action_id;
            return {control_id:binding.control_id,consumer_id:binding.consumer_id,receipt:{action_id:id,outcome:'completed'}};
        }
        if(path.endsWith('/revoke')){revokeCount++;if(deferFirstRevoke && revokeCount===1){await firstRevoke;return {revoked:true};}if(failRevoke)throw new Error('private revoke diagnostic');return {revoked:true};}
        return lease;
    }};
    const body=document.querySelector('main');const view=mountVisualComputer(body,api);
    const click=label=>[...body.querySelectorAll('button')].find(node=>node.textContent===label).click();
    return {dom,calls,body,view,click,releaseFirstRevoke:()=>releaseFirstRevoke?.(),
        switchAgent:()=>{selectedOwner='other';ownerChanged();}};
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
    assert.equal(payload.consumer_id,'consumer');assert.equal(payload.action.observation_id,'human-frame');
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
