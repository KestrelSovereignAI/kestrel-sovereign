// Resolve the actual locked browser registry; no browser download or app boot.
const assert = require('node:assert/strict');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { test } = require('node:test');

const cli = path.join(path.dirname(require.resolve('playwright/package.json')), 'cli.js');
for (const platform of ['ubuntu26.04-x64', 'ubuntu26.04-arm64']) {
  test(`locked Playwright resolves Chromium for ${platform}`, () => {
    const result = spawnSync(process.execPath, [cli, 'install', '--dry-run', 'chromium'], {
      env: { ...process.env, PLAYWRIGHT_HOST_PLATFORM_OVERRIDE: platform },
      encoding: 'utf8', timeout: 20000, maxBuffer: 1024 * 1024,
    });
    assert.ifError(result.error);
    assert.equal(result.status, 0, result.stdout + result.stderr);
    assert.match(result.stdout, /Download url:\s+https:\/\//i);
    assert.doesNotMatch(result.stdout + result.stderr, /does not support chromium|Cannot install dependencies/i);
  });
}
