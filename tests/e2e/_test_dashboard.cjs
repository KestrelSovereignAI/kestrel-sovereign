const { chromium } = require('playwright');

(async () => {
  const KEY = process.env.KESTREL_API_KEY;
  console.log('Using key:', KEY ? KEY.slice(0,10) + '...' : 'MISSING');

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

  // Load dashboard
  await page.goto('http://localhost:8888/static/falconer-dashboard.html');

  // Clear any stale localStorage
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await page.waitForLoadState('domcontentloaded');

  // Should see token bar
  const tokenVisible = await page.locator('#tokenBar').isVisible();
  console.log('Token bar visible:', tokenVisible);

  // Enter key and connect
  await page.fill('#apiKeyInput', KEY);
  await page.click('button:has-text("Connect")');

  // Wait for data to load
  await page.waitForTimeout(5000);

  // Check results
  const body = await page.textContent('body');
  console.log('Has agents online:', body.includes('/5'));
  console.log('Has Kestrel:', body.includes('Kestrel'));
  console.log('Has online badge:', body.includes('online'));
  console.log('Has skills:', body.includes('224'));

  // Check for error
  const liveDot = await page.locator('#liveDot').getAttribute('class');
  console.log('Live dot class:', liveDot);
  const liveText = await page.locator('#liveText').textContent();
  console.log('Live text:', liveText);

  // Screenshot
  await page.screenshot({ path: 'dashboard-browser-test.png', fullPage: true });
  console.log('Screenshot saved: dashboard-browser-test.png');

  await browser.close();
})();
