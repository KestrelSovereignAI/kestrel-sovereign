const fs=require('node:fs');
const {createHash}=require('node:crypto');
const {chromium}=require('playwright');
(async()=>{
 const input=JSON.parse(fs.readFileSync(0,'utf8'));
 const browser=await chromium.launch({headless:true,channel:'chromium'});
 try{
  const page=await browser.newPage({viewport:{width:1100,height:1000}}); page.setDefaultTimeout(10000);
 await page.route('**/*',route=>{const u=new URL(route.request().url());return u.origin===input.origin || ['data:','blob:'].includes(u.protocol)?route.continue():route.abort();});
  await page.addInitScript(({key,ref})=>{
   globalThis.KESTREL_UI_CONFIG={bearerToken:{getToken:()=>key,headerName:'X-API-Key',tokenPrefix:''}};
   globalThis.savedRef=ref;
  },{key:input.key,ref:input.ref});
  await page.route('**/artifact-after-destroy',route=>route.fulfill({contentType:'text/html',body:`<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="stylesheet" href="/static/index.css" data-proof-production-style><body><main>Worker physically destroyed</main><script type=module>
   import API from '/js/api.js';
   import {createArtifactClient} from '/js/artifact_client.mjs';
   import {openArtifactPanel} from '/js/artifact_panel.mjs';
   await API.init(); const owner=API.getHostAgent();
   window.panel=await openArtifactPanel({client:await createArtifactClient(API,owner),ref:window.savedRef,owner,root:document.body});
   window.artifactReady=true;
  </script>`}));
  await page.goto(input.origin+'/artifact-after-destroy');
  await page.waitForFunction(()=>window.artifactReady===true);
  await page.getByRole('status').filter({hasText:'Saved revision'}).waitFor();
  await page.waitForFunction(()=>document.querySelector('link[data-proof-production-style]').sheet!==null);
  const sheet=await page.request.get(input.origin+'/static/index.css');
  const stylesheetSHA=createHash('sha256').update(await sheet.body()).digest('hex');
  if(stylesheetSHA!==input.stylesheet_sha256)throw Error('served production stylesheet differs');
  async function geometry(){
   const value=await page.evaluate(()=>{
    const root=document.querySelector('.artifact-workspace'),header=root.querySelector('.artifact-workspace-header');
    const toolbar=root.querySelector('.artifact-toolbar'),editor=root.querySelector('textarea');
    const rect=node=>node.getBoundingClientRect().toJSON();
    const buttons=[...root.querySelectorAll('button')].filter(node=>node.getClientRects().length).map(rect);
    return {background:getComputedStyle(root).backgroundColor,header:rect(header),toolbar:rect(toolbar),editor:rect(editor),buttons,viewport:innerWidth,documentWidth:document.documentElement.scrollWidth};
   });
   if(value.background==='rgba(0, 0, 0, 0)' || value.editor.height<200 || value.editor.width<=0
     || value.toolbar.bottom>value.editor.y+1 || value.header.bottom>value.toolbar.y+1
     || value.documentWidth>value.viewport+1)throw Error('production editor/header geometry invalid');
   for(const b of value.buttons)if(b.x < -1 || b.right>value.viewport+1 || b.width<=0 || b.height<=0)throw Error('toolbar button outside viewport');
   for(let i=0;i<value.buttons.length;i++)for(let j=i+1;j<value.buttons.length;j++){
    const a=value.buttons[i],b=value.buttons[j];
    if(Math.min(a.right,b.right)-Math.max(a.x,b.x)>1 && Math.min(a.bottom,b.bottom)-Math.max(a.y,b.y)>1)throw Error('toolbar buttons overlap');
   }
   return value;
  }
  const desktopGeometry=await geometry();

  const text=await page.locator('.artifact-panel textarea').inputValue();
  if(!text.includes('Safe to share') || text.includes(input.secret))throw Error('saved artifact text invalid');
  const box=await page.locator('.artifact-workspace').boundingBox();
  if(!box || Math.abs(box.x+box.width-1100)>1 || box.width>600)throw Error('artifact not in right panel');
  await page.getByRole('button',{name:'Expand artifacts',exact:true}).click();
  const expanded=await page.locator('.artifact-workspace').boundingBox();
  if(expanded.width<=box.width)throw Error('artifact panel did not expand');
  await page.screenshot({path:input.output+'/artifact-after-destroy.png'});
  await page.setViewportSize({width:390,height:844});
  const mobileGeometry=await geometry();
  await page.screenshot({path:input.output+'/artifact-after-destroy-mobile.png'});

  process.stdout.write(JSON.stringify({opened_after_physical_destroy:true,original_ref:input.ref,right_panel:box,expanded_panel:expanded,safe_text_verified:true,production_stylesheet_sha256:stylesheetSHA,desktop_geometry:desktopGeometry,mobile_geometry:mobileGeometry}));
 } finally {await browser.close();}
})().catch(error=>{process.stderr.write(error.stack+'\n');process.exitCode=1;});
