// Guards for the demo database reset (issue #1973).
//
// resetDemoAgentDatabases deletes kestrel_prime.db files. These tests pin the
// hard guard that it can ONLY ever touch the isolated KESTREL_DB_PATH sandbox of
// a `kestrel demo run` (KESTREL_DEMO_SERVER=1) — never the live agent_data tree.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const require = createRequire(import.meta.url);
// Import the dependency-free safety module directly (no @kestrel/flight needed).
const {
  resetDemoAgentDatabases, requireDemoSandbox, isDemoServerEnv, assertOnlyDemoAgents,
} = require('../../demos/shared/demo_safety.cjs');

const NOOP_NARRATOR = { narrate() {}, act() {} };

function withEnv(env, fn) {
  const saved = {};
  for (const key of Object.keys(env)) {
    saved[key] = process.env[key];
    if (env[key] === undefined) delete process.env[key];
    else process.env[key] = env[key];
  }
  try {
    return fn();
  } finally {
    for (const key of Object.keys(saved)) {
      if (saved[key] === undefined) delete process.env[key];
      else process.env[key] = saved[key];
    }
  }
}

function mkPrimeDb(dir) {
  fs.mkdirSync(dir, { recursive: true });
  const db = path.join(dir, 'kestrel_prime.db');
  fs.writeFileSync(db, 'x');
  return db;
}

test('isDemoServerEnv reads the demo flag', () => {
  withEnv({ KESTREL_DEMO_SERVER: '1' }, () => assert.equal(isDemoServerEnv(), true));
  withEnv({ KESTREL_DEMO_SERVER: '' }, () => assert.equal(isDemoServerEnv(), false));
});

test('requireDemoSandbox refuses outside a demo run', () => {
  withEnv({ KESTREL_DEMO_SERVER: undefined, KESTREL_DB_PATH: '/tmp/x' }, () => {
    assert.throws(() => requireDemoSandbox(), /KESTREL_DEMO_SERVER is not set/);
  });
  withEnv({ KESTREL_DEMO_SERVER: '1', KESTREL_DB_PATH: undefined }, () => {
    assert.throws(() => requireDemoSandbox(), /KESTREL_DB_PATH is unset/);
  });
});

test('reset REFUSES a parent of the sandbox (the live-data wipe it must prevent)', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'demo-guard-'));
  const sandbox = path.join(root, 'demo'); // KESTREL_DB_PATH
  // Live agents live alongside the sandbox under the parent `root`.
  const liveDb = mkPrimeDb(path.join(root, 'meridian'));
  mkPrimeDb(sandbox);

  withEnv({ KESTREL_DEMO_SERVER: '1', KESTREL_DB_PATH: sandbox }, () => {
    // demo.cjs used to pass the parent (`agent_data/`) — this must now throw.
    assert.throws(() => resetDemoAgentDatabases(NOOP_NARRATOR, root), /outside the isolated demo sandbox/);
  });
  // The live agent's DB must be untouched.
  assert.equal(fs.existsSync(liveDb), true);
});

test('reset deletes prime DBs only inside the sandbox', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'demo-guard-'));
  const sandbox = path.join(root, 'demo');
  const sandboxDb = mkPrimeDb(sandbox);

  withEnv({ KESTREL_DEMO_SERVER: '1', KESTREL_DB_PATH: sandbox }, () => {
    resetDemoAgentDatabases(NOOP_NARRATOR, sandbox);
  });
  assert.equal(fs.existsSync(sandboxDb), false);
});

test('reset refuses entirely when not a demo run', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'demo-guard-'));
  const db = mkPrimeDb(path.join(root, 'meridian'));
  withEnv({ KESTREL_DEMO_SERVER: undefined, KESTREL_DB_PATH: undefined }, () => {
    assert.throws(() => resetDemoAgentDatabases(NOOP_NARRATOR, root), /KESTREL_DEMO_SERVER is not set/);
  });
  assert.equal(fs.existsSync(db), true);
});

// assertOnlyDemoAgents must FAIL CLOSED — every non-isolated/ambiguous response
// is a refusal, never a silent pass (the codex P2).
test('assertOnlyDemoAgents passes only for a non-empty all-demo roster', () => {
  assert.doesNotThrow(() => assertOnlyDemoAgents({ ok: true, status: 200, data: { agents: [{ name: 'demo', is_demo: true }] } }));
  assert.doesNotThrow(() => assertOnlyDemoAgents({ ok: true, status: 200, data: [{ name: 'demo', is_demo: true }] }));
});

test('assertOnlyDemoAgents refuses non-OK / error-shaped / empty / live responses', () => {
  // non-OK
  assert.throws(() => assertOnlyDemoAgents({ ok: false, status: 500, data: { detail: 'boom' } }), /HTTP 500/);
  // OK but error-shaped JSON with no agents array — must NOT pass as "no live agents"
  assert.throws(() => assertOnlyDemoAgents({ ok: true, status: 200, data: { detail: 'unauthorized' } }), /no agents array/);
  // null body (e.g. non-JSON)
  assert.throws(() => assertOnlyDemoAgents({ ok: true, status: 200, data: null }), /no agents array/);
  // zero agents
  assert.throws(() => assertOnlyDemoAgents({ ok: true, status: 200, data: { agents: [] } }), /zero agents/);
  // a live (non-demo) agent present
  assert.throws(
    () => assertOnlyDemoAgents({ ok: true, status: 200, data: { agents: [{ name: 'meridian', is_demo: false }] } }),
    /non-demo agent/,
  );
});
