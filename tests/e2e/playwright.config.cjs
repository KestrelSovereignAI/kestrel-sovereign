// @ts-check
const { defineConfig, devices } = require('@playwright/test');

/**
 * Kestrel Sovereign Console E2E Tests Configuration
 *
 * Tests the Sovereign Console UI at localhost:8888
 *
 * API Key: Set KESTREL_API_KEY env var or tests will fetch from /api/auth/key
 *
 * The `console-smoke` project is the CI subset. It runs only through
 * `uv run kestrel demo smoke`, which creates its isolated instance.
 */
const { MANIFEST_ENV } = require('./console_smoke_proof.cjs');

const SMOKE_SPEC = '**/test_console_smoke.spec.cjs';

// The CI smoke subset (#2682): `kestrel demo smoke` runs exactly this project
// against a fresh isolated instance. No LLM, so it is bounded: one minute per
// test, and no retries, because a retry would let a flaky boot pass the gate.
const CONSOLE_SMOKE_PROJECT = {
  name: 'console-smoke',
  testMatch: SMOKE_SPEC,
  retries: 0,
  timeout: 60000,
  expect: { timeout: 15000 },
  use: { ...devices['Desktop Chrome'] },
};

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
      // The smoke needs the instance `kestrel demo smoke` creates; it is not
      // part of the full suite run against a developer's server.
      testIgnore: SMOKE_SPEC,
      use: { ...devices['Desktop Chrome'] },
    },
    // Registered only when `kestrel demo smoke` launched this run and named
    // its instance manifest. Otherwise a plain `npx playwright test` would
    // select it too and fail on the missing manifest.
    ...(process.env[MANIFEST_ENV] ? [CONSOLE_SMOKE_PROJECT] : []),
  ],

  // Optionally start server before tests
  // webServer: {
  //   command: 'python -m kestrel_sovereign.server',
  //   url: 'http://127.0.0.1:8888/health',
  //   reuseExistingServer: !process.env.CI,
  //   timeout: 120 * 1000,
  // },
});
