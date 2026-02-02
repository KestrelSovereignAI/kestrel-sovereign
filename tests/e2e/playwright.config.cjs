// @ts-check
const { defineConfig, devices } = require('@playwright/test');

/**
 * Kestrel Sovereign Console E2E Tests Configuration
 *
 * Tests the Sovereign Console UI at localhost:8888
 *
 * API Key: Set KESTREL_API_KEY env var or tests will fetch from /api/auth/key
 */
module.exports = defineConfig({
  testDir: './',
  testMatch: '**/*.spec.cjs',
  fullyParallel: false, // Run tests sequentially for agent state consistency
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: 1, // Single worker to avoid state conflicts
  reporter: [['html', { outputFolder: 'playwright-report' }], ['list']],
  timeout: 120000, // 2 minute timeout for LLM-based tests

  use: {
    baseURL: process.env.KESTREL_URL || 'http://localhost:8888',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    actionTimeout: 15000,
    // API key is fetched dynamically in tests - not hardcoded here
  },

  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],

  // Optionally start server before tests
  // webServer: {
  //   command: 'python server.py',
  //   url: 'http://127.0.0.1:8888/health',
  //   reuseExistingServer: !process.env.CI,
  //   timeout: 120 * 1000,
  // },
});
