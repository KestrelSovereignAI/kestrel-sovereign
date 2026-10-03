// The Playwright config's project set (issue #2682).
//
// `console-smoke` exists only for a run `kestrel demo smoke` launched, which
// names the instance manifest in KESTREL_CONSOLE_SMOKE_MANIFEST. Registered
// unconditionally, the documented `cd tests/e2e && npx playwright test` would
// select it as well and fail on the unset manifest against a healthy server.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { MANIFEST_ENV } = require('../e2e/console_smoke_proof.cjs');

const CONFIG = require.resolve('../e2e/playwright.config.cjs');
const SMOKE_SPEC = '**/test_console_smoke.spec.cjs';

function loadConfig(manifestPath) {
  const saved = process.env[MANIFEST_ENV];
  if (manifestPath === undefined) delete process.env[MANIFEST_ENV];
  else process.env[MANIFEST_ENV] = manifestPath;
  delete require.cache[CONFIG];
  try {
    return require(CONFIG);
  } finally {
    delete require.cache[CONFIG];
    if (saved === undefined) delete process.env[MANIFEST_ENV];
    else process.env[MANIFEST_ENV] = saved;
  }
}

const projectNames = (config) => config.projects.map((p) => p.name);

test('a default run registers no console-smoke project', () => {
  const config = loadConfig(undefined);
  assert.deepEqual(projectNames(config), ['chromium']);
  assert.equal(config.projects[0].testIgnore, SMOKE_SPEC);
});

test('an empty manifest variable does not register the smoke project', () => {
  assert.deepEqual(projectNames(loadConfig('')), ['chromium']);
});

test('a kestrel demo smoke run registers the bounded console-smoke project', () => {
  const config = loadConfig('/tmp/kestrel-console-smoke-x/console-smoke-instance.json');
  assert.deepEqual(projectNames(config), ['chromium', 'console-smoke']);
  const smoke = config.projects[1];
  assert.equal(smoke.testMatch, SMOKE_SPEC);
  assert.equal(smoke.retries, 0);
  assert.equal(smoke.timeout, 60000);
  // The full suite still never runs the smoke spec.
  assert.equal(config.projects[0].testIgnore, SMOKE_SPEC);
});
