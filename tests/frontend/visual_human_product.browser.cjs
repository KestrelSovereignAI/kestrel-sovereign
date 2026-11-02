const fs=require('node:fs');
const {createHash}=require('node:crypto');
const {chromium}=require('playwright');
async function main(){
 const input=JSON.parse(fs.readFileSync(0,'utf8'));
 const browser=await chromium.launch({headless:true,channel:'chromium'});
 const context=await browser.newContext({viewport:{width:1100,height:1300}});
 const page=await context.newPage(); page.setDefaultTimeout(10000);
 await context.route('**/*',route=>{const u=new URL(route.request().url());return u.origin===input.origin || ['data:','blob:'].includes(u.protocol)?route.continue():route.abort();});
 const actions=[];const statuses=[];const leases=[];const errors=[];
 page.on('pageerror',()=>errors.push('pageerror'));
 page.on('request',request=>{
   if(request.url().includes('/human/') && request.url().endsWith('/actions')){
     const value=request.postDataJSON();actions.push({action_id:value.action.action_id,observation_id:value.action.observation_id,
       control_id:value.control_id,consumer_id:value.consumer_id,kind:value.action.action.kind});
   }
   if(request.url().includes('/human/') && /\/actions\/[^/]+$/.test(request.url()))statuses.push(request.url().split('/').at(-1));
 });
 page.on('response',async response=>{
   if(response.request().method()==='POST' && /\/human\/[^/]+$/.test(response.url()) && response.status()===200)leases.push(await response.json());
 });
 try{
   await page.addInitScript(({key})=>{globalThis.KESTREL_UI_CONFIG={bearerToken:{getToken:()=>key,headerName:'X-API-Key',tokenPrefix:''}};},{key:input.key});
   await page.route('**/human-component-proof',route=>route.fulfill({contentType:'text/html',body:`<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="stylesheet" href="/static/index.css" data-proof-production-style><main></main><script type=module>
      import API from '/js/api.js';
      import { mountVisualComputer } from '/features/visualcomputerfeature/static/visual-computer.js';
      await API.init(); window.proofView=mountVisualComputer(document.querySelector('main')); await window.proofView.refresh(); window.proofReady=true;
   </script>`}));
   await page.goto(input.origin+'/human-component-proof');await page.waitForFunction(()=>window.proofReady===true);
   await page.waitForFunction(()=>document.querySelector('link[data-proof-production-style]').sheet!==null);
   const sheet=await page.request.get(input.origin+'/static/index.css');
   const stylesheetSHA=createHash('sha256').update(await sheet.body()).digest('hex');
   if(stylesheetSHA!==input.stylesheet_sha256)throw Error('served production stylesheet differs');
   const style=await page.evaluate(()=>({background:getComputedStyle(document.body).backgroundColor,stylesheetLoaded:!!document.querySelector('link[data-proof-production-style]').sheet}));
   if(style.background==='rgba(0, 0, 0, 0)' || !style.stylesheetLoaded)throw Error('production theme not active');
   const status=page.locator('fieldset p');
   await page.getByRole('button',{name:'Open human view',exact:true}).click();
   await page.waitForFunction(()=>document.querySelector('img').hidden===false);
   await page.getByRole('checkbox',{name:'Enable manual input',exact:true}).check();
   await page.locator('img').click({position:{x:20,y:20}});
   await status.filter({hasText:'Manual input completed.'}).waitFor();
   await page.getByRole('button',{name:'Reconcile last manual input',exact:true}).click();
   await status.filter({hasText:'Original manual input reconciled.'}).waitFor();
   await page.getByRole('button',{name:'Refresh human screen',exact:true}).click();
   await status.filter({hasText:'Private human view.'}).waitFor();
   await page.getByRole('textbox',{name:'Private text to type',exact:true}).fill(input.secret);
   await page.getByRole('checkbox',{name:'Enable manual input',exact:true}).check();
   await page.getByRole('button',{name:'Type private text',exact:true}).click();
   await status.filter({hasText:'Manual input completed.'}).waitFor();
   await page.getByRole('button',{name:'Reconcile last manual input',exact:true}).click();
   await status.filter({hasText:'Original manual input reconciled.'}).waitFor();
   await page.getByRole('button',{name:'Refresh human screen',exact:true}).click();
   await status.filter({hasText:'Private human view.'}).waitFor();
   await page.getByRole('checkbox',{name:'Enable manual input',exact:true}).check();
   await page.getByRole('button',{name:'Focus screen keyboard',exact:true}).click();
   await page.locator('img').press('End');
   await status.filter({hasText:'Manual input completed.'}).waitFor();
   await page.getByRole('button',{name:'Reconcile last manual input',exact:true}).click();
   await status.filter({hasText:'Original manual input reconciled.'}).waitFor();
   await page.getByRole('button',{name:'Refresh human screen',exact:true}).click();
   await status.filter({hasText:'Private human view.'}).waitFor();
   const data=await page.locator('img').getAttribute('src');
   fs.writeFileSync(input.output+'/manual-screen.private.png',Buffer.from(data.split(',')[1],'base64'));
   await page.screenshot({path:input.output+'/console-component.private.png'});
   await page.getByRole('button',{name:'Close human view',exact:true}).click();
   await status.filter({hasText:'Human channel revoked.'}).waitFor();
   if(await page.locator('img').getAttribute('src')!==null)throw Error('closed channel retained pixels');
   if(actions.length!==3 || new Set(actions.map(x=>x.action_id)).size!==3 || statuses.length!==3)throw Error('UI submit-once/status contract failed');
   if(errors.length)throw Error('browser component failed');
   await page.evaluate(()=>window.proofView.destroy());
   process.stdout.write(JSON.stringify({actual_browser_component:true,production_stylesheet_sha256:stylesheetSHA,computed_style:style,actions,statuses,leases,pixels_purged_after_close:true,page_errors:errors.length}));
 }catch(error){
   fs.writeFileSync(input.output+'/browser-failure-state.private.json',JSON.stringify(await page.evaluate(()=>({
      html:document.body.innerHTML,controls:[...document.querySelectorAll('button')].map(e=>({text:e.textContent,disabled:e.disabled,
        rect:e.getBoundingClientRect().toJSON(),display:getComputedStyle(e).display,visibility:getComputedStyle(e).visibility,
        ancestors:[e.parentElement,e.parentElement?.parentElement].filter(Boolean).map(p=>({tag:p.tagName,hidden:p.hidden,display:getComputedStyle(p).display}))}))})),null,2));
   await page.screenshot({path:input.output+'/browser-failure.private.png'});
   throw error;
 }finally{await context.close();await browser.close();}
}
main().catch(error=>{process.stderr.write(error.stack);process.exitCode=1;});
