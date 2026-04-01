// Quick screenshot test for Falconer Dashboard
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

  // Navigate to dashboard
  await page.goto('http://localhost:8888/static/falconer-dashboard.html');
  await page.waitForLoadState('domcontentloaded');

  // Enter API key
  const tokenBar = page.locator('#tokenBar');
  if (await tokenBar.isVisible()) {
    const key = process.env.KESTREL_API_KEY;
    await page.fill('#apiKeyInput', key);
    await page.click('button:has-text("Connect")');
  }

  // Wait for data to load
  await page.waitForTimeout(3000);

  // Screenshot: Flock tab
  await page.screenshot({ path: 'tests/e2e/demo-output-falconer/dashboard-flock.png', fullPage: true });
  console.log('Screenshot 1: Flock Status saved');

  // Screenshot: Tasks tab
  await page.click('.tab:has-text("Tasks")');
  await page.waitForTimeout(1500);
  await page.screenshot({ path: 'tests/e2e/demo-output-falconer/dashboard-tasks.png', fullPage: true });
  console.log('Screenshot 2: Tasks saved');

  // Screenshot: Mesh tab
  await page.click('.tab:has-text("Mesh")');
  await page.waitForTimeout(1000);
  await page.screenshot({ path: 'tests/e2e/demo-output-falconer/dashboard-mesh.png', fullPage: true });
  console.log('Screenshot 3: Mesh saved');

  // Screenshot: Governance tab
  await page.click('.tab:has-text("Governance")');
  await page.waitForTimeout(2000);
  await page.screenshot({ path: 'tests/e2e/demo-output-falconer/dashboard-governance.png', fullPage: true });
  console.log('Screenshot 4: Governance saved');

  await browser.close();
  console.log('All dashboard screenshots captured!');
})();
